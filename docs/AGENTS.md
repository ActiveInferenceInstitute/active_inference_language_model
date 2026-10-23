# docs/ — Agent Operating Rules

How to work inside `docs/`. The file index by category is
[`README.md`](README.md); this file is about the mechanics — what is gated, who
owns what, and how to check a change before leaving it behind.

## `docs/` is gated prose, not free text

Do not assume documentation is unenforced here. `tests/test_claims_discipline.py`
globs **every `.md` file under `docs/`** into its tag-validity set, alongside
all root Markdown, GNN/formal Markdown, and every `.py` under
`../src/`. Any `claim:` tag in any of those files must resolve to an entry in
[`../data/claims.json`](../data/claims.json) or the suite fails.

The same test also runs a conservative bare-quantitative scanner over current
root documentation, `docs/`, GNN/formal Markdown, and AST-identified Python
docstrings. It rejects untagged percentages, measured or population units,
resource units and rates, conventional statistical assignments, and clock
durations. Positive controls exercise every classified form, including Python
docstrings, and negative controls prove that executable strings and exclusions
do not disable the scan after a code fence.

That test exists because the opposite was true for a whole stage.
`ClaimsLedger.scan_prose` shipped in Stage 0 and was only ever pointed at
`tmp_path` fixtures. Every test passed while a phoneme-count tag sat unbacked in
the real README, the real AGENTS.md, and the real provenance doc. A gate that runs
only on synthetic input constrains nothing about the shipped artifact — so the
scanner was pointed at the real tree, with a positive control
(`test_scanner_detects_an_injected_unbacked_claim`) proving it can still fail.

Two consequences for anyone editing `docs/`:

- **A measured number in `docs/` prose is written `claim:<label>`.**
  Supporting docs hydrate backed claim tags and the registered source-reference
  and protocol presentation tokens through `ailm.docs_hydration`. Numerical
  manuscript tokens belong in `docs/manuscript/`; generic syntax examples stay
  literal. The claims scanner enforces the measured tag. Example, backed and live:
  the canonical ARPAbet inventory has claim:phoneme_count_39 phones.
- **A tag is not a syntax illustration.** The scanner cannot tell an example from
  an assertion. Writing a plausible-looking tag to demonstrate the notation —
  a label for an unbuilt stage's future measurement, or a fixture label borrowed
  from a test file — fails the gate exactly as a real unbacked claim does. If you
  need to show the notation, use a label that is genuinely recorded.

**Never fix this gate by deleting the tag.** The fix is to record the measurement
from the run that produced it (`../scripts/record_stage*_claims.py`). Deleting the
tag and typing the digit converts a caught problem into an uncaught one.

This is not a digit ban or a proof that every untagged number is harmless. The
bare scanner masks fenced and inline code, equations, versions, dates, stage and
level identifiers, source locators, citations, and links. It deliberately does
not classify naked scalars or spelled-out quantities. No metadata heading masks
an entire list item: numeric `Extent`, `Population`, `Exclusions`, `File`, and
`Source` fields are measurements and remain visible. The exact surface and excluded files are pinned
in `tests/test_claims_discipline.py`: append-only historical reports and decision
logs are disclosed migration debt, `research_queue.md` is exploratory scratch,
and `style_guide.md` contains intentional bad examples. Do not add an exclusion
merely to make a finding disappear; either connect a measured result to its
recorded label or remove unsupported numerical precision.

## Ownership

| Path | Owner |
|---|---|
| `README.md`, `AGENTS.md` | This directory's index and its own operating rules |
| `architecture.md`, `hierarchy.md`, `formal.md`, `gnn_composition.md`, `gate_design.md`, `testing_philosophy.md`, `syntax_guide.md`, `extension_protocols.md`, `troubleshooting.md`, `decisions.md`, `development_history.md`, `research_queue.md` | Governing references, kept at root because each is the file its topic's other members point back to. `hierarchy.md` wins on any disagreement about level structure and live score paths; `decisions.md` is append-only chronology; `research_queue.md` is exploratory and outside the bare-number scan |
| `guides/*.md` | Task-oriented reading: `quickstart.md`, `agent_instructions.md` (the canonical command reference), `forking_guide.md`, `faq.md`, `roadmap.md`. Change these when a rule, a convention, or the first-run sequence changes |
| `reference/*.md` | Consult-as-needed: `glossary.md`, `notation.md`, `formalism_map.md`, `output_conventions.md`, `output_inventory.md`, `data_statement.md`. Change these when the code layout, the token flow, the artifact set, or a corpus's generalizability changes. `glossary.md` carries no result numbers — link to the file that has them instead of retyping one |
| `method/*.md` | The mechanics: `style_guide.md`, `rendering_pipeline.md`, `release_checklist.md` |
| `design/*.md` | One decision per file. Change only when the decision changes, and say what the alternative was. The subdirectory's own [`design/README.md`](design/README.md) and [`design/AGENTS.md`](design/AGENTS.md) index it and state how a record is written |
| `manuscript/` | Publication source, governed by [`manuscript/AGENTS.md`](manuscript/AGENTS.md). Use backed variable tokens for measured quantities, preserve the layer contract, and regenerate through the documented rendering pipeline. Claim tags remain subject to the repository-wide validity scan |
| `provenance/sources.md` | Corpora, licenses, vendored artifacts, checksums, derivation biases. [`provenance/README.md`](provenance/README.md) is its section map and [`provenance/AGENTS.md`](provenance/AGENTS.md) the regeneration contract |

The split matters for one reason in particular: [`architecture.md`](architecture.md)
is the **code** architecture and [`design/architecture.md`](design/architecture.md)
is the **scientific** architecture. Moving a module edits the first. Changing what
a matrix means edits the second. Editing the wrong one produces two documents that
each look authoritative and disagree.

## Rules

1. **State the alternative.** A design doc that only describes what was built is a
   description, not a decision record. Every `design/` file names the rejected
   option and why.
2. **Cite code by path.** `src/ailm/levels/phoneme.py::build_g2p_emission`, not
   "the emission builder". A reader must be able to open the file and land on the
   function.
3. **Check repository links.** Use relative links for repository files and
   canonical URLs for external sources. Resolve repository links from the file
   they sit in. Verification command below.
4. **Do not depend on transient files in `output/`.** The stable publication
   allowlist is committed, but slides and render intermediates remain ignored
   (see [`output_conventions.md`](reference/output_conventions.md)). Prefer links to authored
   sources; name generated paths as inline code unless the generated artifact is
   itself the deliverable.
5. **Do not restate a measured result in two places.** If a number belongs to a
   result, it lives in the ledger and is referenced by tag. If a claim is
   qualitative, say it once and cross-link.
6. **Version identifiers are not measurements.** `pymdp 1.0.4` and `Python 3.12`
   are configuration identifiers that live in
   [`../pyproject.toml`](../pyproject.toml) and `.python-version`; quoting them is
   correct. Document the coverage floor with its recorded claim tag. A perplexity,
   an accuracy, or a corpus size is a measurement and needs a tag.

## Verifying a `docs/` change

Run from the project root, with the project's own interpreter (see
[`quickstart.md`](guides/quickstart.md) for why `uv run` is wrong here):

```bash
# 1. The claims-discipline gate — the one that actually reads docs/
.venv/bin/python -m pytest tests/test_claims_discipline.py -v

# 2. Which labels, if any, are unbacked right now, and in which file
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from ailm.claims import load_ledger
root = Path('.')
files = [root / n for n in ('README.md', 'AGENTS.md', 'CHANGELOG.md') if (root / n).exists()]
files += sorted((root / 'docs').rglob('*.md')) + sorted((root / 'src').rglob('*.py'))
print('UNBACKED:', sorted(set(load_ledger(root).scan_prose(*files))))
"

# 3. Canonical documentation checks resolve relative links, registered metadata
#    targets, fenced examples and the complete index. Rendered HTML anchors are
#    additionally checked by the portable-bundle validation below.
.venv/bin/python -m pytest tests/test_documentation.py -q --no-cov
TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT" .venv/bin/python scripts/z_render_docs.py --check --supporting-protocols
```

Select `AILM_TEMPLATE_ROOT` as described in the [quickstart](guides/quickstart.md),
or declare the checkout once in the repository's `.template-root` file, before
checking portable companions. This points to the existing clean template
provider; it is not an instruction to install or modify it.

Check 2 duplicates what check 1 asserts, but prints the offending label and file
instead of a bare failure, which is what you need while fixing it.

## Cross-references

- [`README.md`](README.md) — the categorized index of every document
- [`quickstart.md`](guides/quickstart.md) — interpreter rule, gate commands, render command
- [`architecture.md`](architecture.md) — module ownership and the import contract
- [`../AGENTS.md`](../AGENTS.md) — project-level invariants and stage gates
- [`../tests/test_claims_discipline.py`](../tests/test_claims_discipline.py) — the gate that reads this directory
