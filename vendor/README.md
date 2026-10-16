# `vendor/` - reproducible corpus artifacts

This directory has two intentionally different tiers:

- committed derived JSON artifacts used by the package, tests, figures, and
  recorders;
- ignored upstream downloads and staging trees used only to regenerate those
  derived artifacts.

The authoritative committed-JSON inventory is
`ailm.resources.RUNTIME_VENDOR_ARTIFACTS`. Tests require that manifest to equal
the complete `vendor/*.json` set, that every file has a filename-bound checksum
in [`docs/provenance/sources.md`](../docs/provenance/sources.md), and that an
installed wheel contains byte-identical copies. This avoids maintaining another
filename table here that would drift.

## Committed versus ignored

All derived `vendor/*.json` files are tracked. They make the shipped models and
tests network-independent, and the wheel embeds the same files under
`ailm/_vendor/`.

Raw evaluation inputs are also tracked. `en_ewt-ud-test.conllu` is the UD English-EWT test
partition vendored under decision 79 as a *held-back* population. It was
committed so a successor registration could score against bytes that were fixed
before the prediction, and decision 111 is that registration: the four
`ud_ewt_test` family members have now been scored against it exactly once, under
the sentinel `confirmatory_run_completed_ud_ewt_test`. The population is
therefore spent, and no further result may be scored against this file. Its
pinned revision, URL, and digest are recorded in
[`docs/provenance/sources.md`](../docs/provenance/sources.md) under "UD EWT derived
raw artifact". Raw evaluation inputs are included in the source distribution,
not the runtime wheel. `en_gum-ud-test.conllu` is the separately pinned connected
evaluation input, governed by `data/connected_gum/protocol.json`; its earlier
inspected documents are excluded from the new analysis population.

The derived artifacts of that one run are deliberately **not** here. They live in
`data/confirmatory/`, because this directory's committed JSON is by definition
the wheel's packaged runtime surface (`ailm.resources.RUNTIME_VENDOR_ARTIFACTS`,
enforced by `tests/test_wheel_resources.py`), and one-run replication evidence
that no library code imports does not belong in a wheel. Their filename-bound
digests are recorded in [`docs/provenance/sources.md`](../docs/provenance/sources.md)
and enforced by `tests/test_confirmatory_replication.py`.

These upstream inputs remain ignored:

| Input | Why it is not committed | Acquisition or staging owner |
|---|---|---|
| `brown.zip` | Upstream corpus archive; canonical Brown-derived artifacts embed and validate its immutable revision, URL, byte extent, and SHA-256 | `ailm.corpus.brown.resolve_brown_source_zip()` via `--source-zip PATH` or `AILM_BROWN_SOURCE_ZIP` |
| `cmudict.dict` | Upstream dictionary; only its deterministic derived counts ship | `ailm.corpus.cmudict.load_cmudict(download=True)` using the immutable URL and raw digest in `ailm.corpus.cmudict` |
| `en_ewt-ud-dev.conllu` | Upstream treebank source; only pinned, raw-digest-verified bytes may feed the derived EWT artifacts | `ailm.corpus.ud.ensure_ud_ewt_dev_path(download=True)` |
| `staging_ud_czech/` | Large Czech-PDTC source shards under non-commercial ShareAlike terms | `scripts/generate_czech_case_counts.py` input |
| `cmu_arctic/` | Optional speech-boundary material, not a runtime model artifact | user-provided staging |

Upstream license terms do not disappear when the data are transformed. See
[`LICENSE`](../LICENSE) and
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md), especially Brown's
non-commercial-use notice and Czech-PDTC's CC BY-NC-SA terms.

## What the artifacts encode

The corpus-specific schemas, normalizations, split rules, exclusions, exact
checksums, and generator paths are documented once in
[`docs/provenance/sources.md`](../docs/provenance/sources.md). Highlights:

- `brown_counts.json` stores Level 1 running-text bigram counts over the AILM
  letter alphabet.
- `cmudict_counts.json` stores the canonical ARPAbet inventory, phone sequence
  counts, word-initial counts, and position-aligned G2P pairs. Its phone list is
  checked against `CANONICAL_ARPABET` before use. It also records the exact raw
  URL, Git revision, byte extent, and SHA-256, so a moving upstream branch cannot
  silently change what the same derived filename means.
- the word, case, clause, and narrative artifacts retain only the fields needed
  by their declared evaluation populations; especially, the English case
  artifact withholds the fields from which its labels are derived. Every English
  EWT artifact also retains the immutable raw revision, URL, SHA-256, and byte
  extent, which its runtime loader validates.
- `brown_heldout_stream.json` retains the complete unfiltered Stage 3 held-out
  denominator, while `brown_word_counts.json` contains the covered sublanguage
  used for scoring. Both bounded and full word artifacts record the immutable
  Brown raw-source identity, vocabulary bound, and lexicon source. The loader
  validates those fields plus schema, alphabet, positive counts, ordering,
  population cardinalities, and evaluation support before use; it also filters
  the separately committed unfiltered denominator and requires exact token and
  document-index reproduction of the word artifact's evaluation stream.
- the synthetic artifact is generated from a stipulated transition process,
  not from an external corpus.

## Regeneration contract

Never edit a file here by hand. Fix its generator, regenerate it, re-record every
dependent claim when values change, and update the filename-bound checksum in the
same change. A valid JSON hand edit can silently reindex matrices or change an
evaluation population; the digest alone cannot explain whether that change was
scientifically intended.

Generators must be deterministic under their documented inputs. Some require
ignored upstream files, so a normal test run verifies the committed bytes and
their schemas without downloading anything; generator-specific tests exercise
byte-identical reconstruction wherever the upstream sufficient statistics are
committed. Never turn a missing upstream input into a network call inside a test.
The full CMUdict raw-source replay is opt-in: set
`AILM_RUN_RAW_CORPUS_REGENERATION=1` only when the pinned ignored dictionary is
present, then run `tests/test_provenance_checksums.py`.

`generate_counts.py --force-fallback` is an explicitly noncanonical diagnostic,
not an admissible source for a runtime artifact or recorded claim. It must use a
noncanonical output filename, emits a schema the production loader rejects, and
cannot substitute for a missing pinned archive.

## Verification

```bash
.venv/bin/python -m pytest \
  tests/test_provenance_checksums.py tests/test_wheel_resources.py \
  tests/test_heldout_stream.py -q --no-cov

git check-ignore -v vendor/cmudict.dict
git ls-files 'vendor/*.json'
git status --porcelain --untracked-files=all vendor/
shasum -a 256 vendor/*.json
rg -o '[0-9a-f]{64}' docs/provenance/sources.md
```

The tracked JSON inventory must have no untracked entries. A generator failure
is a hard failure; never suppress its stderr or exit code.
