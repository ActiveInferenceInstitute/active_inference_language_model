# AGENTS.md — docs/provenance/

Operating rules for the provenance record. Quick reference: [`README.md`](README.md).
The gating that applies to everything under `docs/` is [`../AGENTS.md`](../AGENTS.md),
which this file does not restate.

## Overview

[`sources.md`](sources.md) is the single place a reader learns where a byte came
from: the pinned upstream revision, the license, the extent, the digests, the
generator that derived a vendored artifact from it, and the derivation choices that
bias what the artifact can be used for.

It is a reproducibility document, not a scope document. What a corpus *generalizes
to* is [`../data_statement.md`](../reference/data_statement.md).

## Ownership boundary

| Concern | Owner |
| --- | --- |
| The digests and the derivation prose | this directory |
| The artifacts themselves | [`../../vendor/`](../../vendor/AGENTS.md) — generator output or upstream download; nothing there is authored |
| Packaged vendor inventory | `ailm.resources.RUNTIME_VENDOR_ARTIFACTS`, which must equal the on-disk `vendor/*.json` set |
| Packaged canonical data inputs | `ailm.resources.RUNTIME_DATA_ARTIFACTS` and `pyproject.toml` force-includes; document their source paths and wheel paths when packaging changes |
| The machine-readable paper snapshot | [`../../data/reference_sources.json`](../../data/reference_sources.json), verified by `scripts/verify_reference_sources.py` |
| Which artifacts must be documented here | `DOCUMENTED_ARTIFACTS` in `tests/test_provenance_checksums.py` |
| Upstream license terms | [`../../LICENSE`](../../LICENSE) and [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **An artifact and its documented digest change in one commit** | Adding a field to a vendored artifact once moved its hash while this file kept the old value — prose asserting the wrong digest. A stale checksum tells a reader their correct file is corrupt |
| **Every derived artifact names its generator and its population** | A digest proves the bytes; only the derivation prose says what the bytes are of, and which fields were withheld |
| **Record identity and printed identity are both preserved** | Where a Zenodo record and its deposited PDF's title page disagree, neither is silently chosen. The shorthand is labeled as shorthand |
| **A derivation choice change means re-recording every dependent claim** | Primary-pronunciation-only, stress stripping, the length-matched alignment restriction, and each split rule all bias the data and are documented as such |
| **Never update the paper snapshot from a DOI alone** | A Zenodo title or concept DOI is not artifact identity. Inspect and hash the exact deposited file |
| **The normal suite stays network-free** | Revalidation against Zenodo is opt-in and explicit; no test may reach a URL |
| **Measured numbers carry a `claim:` tag** | This directory is inside `tests/test_claims_discipline.py`'s scan, in both directions |
| **New documents here are reachable from [`../README.md`](../README.md)** | An unindexed document is invisible rather than broken, and it fails the index-reachability test |

## File organization

| File | What it is |
| --- | --- |
| `sources.md` | The provenance record: papers, corpora, derived artifacts, pymdp, and design-decision sources |
| `README.md` | Quick reference and the section map |
| `AGENTS.md` | This file |

## Adding or regenerating an artifact

In one change:

1. Fix or add the generator under `scripts/`, and regenerate.
2. Add or update the artifact's section in `sources.md` — generator path,
   population, exclusions, byte extent, and SHA-256.
3. Add its `RUNTIME_VENDOR_ARTIFACTS` entry and its `DOCUMENTED_ARTIFACTS` entry.
4. Make the `.gitignore` decision explicitly rather than by default.
5. Re-record every claim whose value moved, and regenerate the token map and the
   figures in the same change.

## Verification

Run from the project root, with the project's own interpreter — never `uv run`
inside this sidecar (see [`../quickstart.md`](../guides/quickstart.md)).

```bash
# Digests, schemas, byte-identical regeneration where the inputs are committed
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q --no-cov

# The paper snapshot against the code that reads it
.venv/bin/python -m pytest tests/test_reference_source_fidelity.py -q --no-cov

# Documented digests vs the files on disk
shasum -a 256 vendor/*.json
rg -o '[0-9a-f]{64}' docs/provenance/sources.md
```

Never mask a generator failure with `2>/dev/null || true`. Suppressing stderr *and*
swallowing the exit code turns a broken regeneration into a silent no-op, and a
stale artifact then propagates into every recorded claim.

## See Also

- [`README.md`](README.md) — quick reference and section map
- [`../data_statement.md`](../reference/data_statement.md) — what each corpus generalizes to
- [`../../vendor/AGENTS.md`](../../vendor/AGENTS.md) — the artifacts and their regeneration contract
- [`../AGENTS.md`](../AGENTS.md) — how `docs/` is gated
- [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) — upstream terms that survive transformation
