# `docs/provenance/` — where every byte came from

One document, [`sources.md`](sources.md): for each corpus, paper, derived training
artifact and frozen evaluation input, the source identity, license, digests,
generator and derivation choices. Runtime packaging and evaluation evidence are
separate: selected canonical `data/` artifacts are packaged explicitly, while
raw GUM evaluation text and measured reports are not wheel resources.

It answers "can this number be rebuilt?". The companion question — "what population
can this number be *about*?" — is answered by
[`../data_statement.md`](../reference/data_statement.md).

## Quick start

```bash
# From the project root: the committed digests against the files on disk
shasum -a 256 vendor/*.json
rg -o '[0-9a-f]{64}' docs/provenance/sources.md
```

## What is in `sources.md`

| Section | Covers |
|---|---|
| Opt-in live revalidation | How to re-check the deposited PDFs against Zenodo without touching the network in a normal run |
| Primary Papers | The CEREBRUM and compositional-case records: concept DOI, exact record, deposited filename, extent, MD5, SHA-256, and the record-versus-printed identity conflict |
| Optional Companion Code | The CCD reference implementation |
| External Corpora | CMUdict, UD English EWT, the Brown corpus, the held-back EWT test partition, and every derived `vendor/*.json` artifact with its generator and digest |
| UD English GUM connected-evaluation input | Pinned raw test identity, local analysis protocol, withheld annotations, report/execution binding and non-commercial corpus terms |
| pymdp | The pinned generative-model dependency |
| Design Decision Sources | What each design record leans on |

Each derived-artifact section names its generator script, its evaluation
population, its exclusions, and its exact digest — so a regeneration that changes
any of those changes this file in the same commit.

## Key rules

- **A changed artifact and its documented digest move together, in one commit.** A
  stale checksum is worse than none: it tells a reader their correct file is
  corrupt.
- **Record identity is not printed identity.** Where a Zenodo record and its
  deposited PDF's title page disagree, both are preserved and the shorthand is
  labeled as shorthand.
- **The snapshot in [`../../data/reference_sources.json`](../../data/reference_sources.json)
  is machine-readable and network-free.** Update it only after inspecting and
  hashing the exact deposited file.
- **Measured numbers use a `claim:` tag.** This directory is inside the
  claims-discipline scan.

## Common commands

Run from the project root, with the project's own interpreter.

```bash
# Digests, schemas, and byte-identical regeneration where the inputs are committed
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q --no-cov

# Metadata-only revalidation against Zenodo (opt-in, network)
.venv/bin/python -I scripts/verify_reference_sources.py

# Plus a full byte-level check of each deposited PDF
.venv/bin/python -I scripts/verify_reference_sources.py --full-files --require-pdfinfo
```

## More information

See [`AGENTS.md`](AGENTS.md) for the rules in full,
[`../../vendor/AGENTS.md`](../../vendor/AGENTS.md) for the artifacts themselves,
and [`../README.md`](../README.md) for the full documentation index.
