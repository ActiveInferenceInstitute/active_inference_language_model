# AGENTS.md — docs/design/

Operating rules for the design-decision records. The file-by-file index is
[`README.md`](README.md); the gating that applies to everything under `docs/` is
[`../AGENTS.md`](../AGENTS.md), which this file does not restate.

## Overview

One decision per file. Each records a choice that could reasonably have gone the
other way, names the rejected alternative, and says what the choice costs. These
are the **scientific** architecture; [`../architecture.md`](../architecture.md) is
the **code** architecture, and they are deliberately different documents.

## Ownership boundary

| Concern | Owner |
| --- | --- |
| Which score paths run live inference, and the governing level structure | [`../hierarchy.md`](../hierarchy.md) — it wins on any disagreement |
| The chronological record of when a decision was taken or restated | [`../decisions.md`](../decisions.md) |
| Every measured number cited here | [`../../data/claims.json`](../../data/claims.json), reached by `claim:` tag |
| Module ownership, import direction, forbidden patterns | [`../architecture.md`](../architecture.md) |
| Symbol-to-meaning | [`../notation.md`](../reference/notation.md) |
| Equation-to-code-to-test | [`../formalism_map.md`](../reference/formalism_map.md) |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **Every file names the rejected alternative** | A document that only describes what was built is a description. The directory's whole purpose is that the alternative is on the record |
| **Cite code by path and symbol** | `src/ailm/levels/phoneme.py::build_g2p_emission`, not "the emission builder". A reader must be able to open the file and land on the function |
| **A measured number carries a `claim:` tag** | These files are inside `tests/test_claims_discipline.py`'s scan, in both directions: an unbacked tag fails, and so does an untagged percentage, measured unit, resource unit, statistic, or clock duration |
| **Never fix a discipline failure by deleting the tag** | That converts a caught problem into an uncaught one. Record the measurement instead |
| **Do not restate a measured result in two files** | It lives in the ledger and is referenced by tag. A qualitative claim is said once and cross-linked |
| **`hierarchy.md` governs** | A design file that contradicts it is the file that is wrong |
| **Relative links only, and they must resolve** | A parameterized test checks every relative link in every document under `docs/` |
| **New files under `docs/` must be reachable from `../README.md`** | An unindexed document is invisible rather than broken, which is worse — and it fails `test_every_document_under_docs_is_reachable_from_the_index` |

## File organization

One decision per file, named for the thing decided. See the table in
[`README.md`](README.md) for the current set and what each records.

Adding a file here means adding a row to [`../README.md`](../README.md)'s "Science
and design decisions" table in the same change.

## Writing a decision record

1. State the decision in the first paragraph, in the past tense, as something that
   was chosen.
2. State the alternative, and why it was rejected — on evidence where evidence
   exists, and say so plainly where it was judgement.
3. State what the choice costs. A design record with no cost section is advocacy.
4. Link the code that implements it and the test that exercises it.
5. If the decision has a dated chronology, put the chronology in
   [`../decisions.md`](../decisions.md) and link to the entry rather than
   duplicating it.

## Verification

Run from the project root, with the project's own interpreter — never `uv run`
inside this sidecar (see [`../quickstart.md`](../guides/quickstart.md)).

```bash
# The gate that actually reads these files
.venv/bin/python -m pytest tests/test_claims_discipline.py -q --no-cov

# Index reachability, link resolution, empty labels
.venv/bin/python -m pytest tests/test_documentation.py -q --no-cov

# Which labels are unbacked right now, and where
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from ailm.claims import load_ledger
files = sorted(Path('docs/design').glob('*.md'))
print('UNBACKED:', sorted(set(load_ledger(Path('.')).scan_prose(*files))))
"
```

## See Also

- [`README.md`](README.md) — the decision index
- [`../AGENTS.md`](../AGENTS.md) — how `docs/` is gated, and the ownership split
- [`../README.md`](../README.md) — the full documentation index
- [`../hierarchy.md`](../hierarchy.md) — the governing level structure
- [`../decisions.md`](../decisions.md) — the chronological record
- [`../../AGENTS.md`](../../AGENTS.md) — project invariants and stage gates
