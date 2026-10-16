# AGENTS.md — vendor/

## Ownership boundary

Everything here is **generator output or upstream download**. Nothing in `vendor/` is
authored.

| Artifact | Owner |
| --- | --- |
| committed `vendor/*.json` | The generator and exact checksum named in `docs/provenance/sources.md`; the complete runtime set is `ailm.resources.RUNTIME_VENDOR_ARTIFACTS` |
| `cmudict.dict` | `ailm.corpus.cmudict.load_cmudict(download=True)` from its immutable revision URL and verified raw digest |
| `en_ewt-ud-dev.conllu` | `ailm.corpus.ud.ensure_ud_ewt_dev_path(download=True)` from its immutable revision URL, with raw SHA-256 and byte-extent validation before use or atomic write |
| `en_ewt-ud-test.conllu` | `ailm.corpus.ud.ensure_ud_ewt_test_path()` — tracked but held back: no recorder or gate reads it (decision 79) |
| `staging_ud_czech/*.conllu` | `scripts/generate_czech_case_counts.py` input; ignored, non-commercial ShareAlike terms |
| the recorded sha256 for each derived artifact | `docs/provenance/sources.md` |

Consumers (`ailm.levels.letter.load_brown_counts`,
`ailm.levels.phoneme.load_phoneme_counts`) read and never write.

## Committed vs gitignored, and why

`.gitignore` excludes the upstream downloads and staging trees, including
`brown.zip`, `cmudict.dict`, `en_ewt-ud-dev.conllu`, `staging_ud_czech/`, and
`cmu_arctic/`.
Every derived `vendor/*.json` artifact is tracked, and so is the one raw
exception, `en_ewt-ud-test.conllu` — committed precisely so a held-back population
is fixed in history before any successor prediction is scored against it. The runtime manifest must equal
that on-disk set exactly, and the wheel test requires byte-identical packaged
copies, so a newly added artifact cannot silently be absent from an install.

The rule: **derived statistics are committed so model and gate tests are offline;
upstream corpora are not.** Do not invert either half. The upstream licensing and
redistribution constraints remain documented in `LICENSE`,
`THIRD_PARTY_NOTICES.md`, and `docs/provenance/sources.md`.

## May change freely

- Regenerate an artifact from its generator, updating the sha256 in
  `docs/provenance/sources.md` in the same change.
- Add a new derived artifact — with a generator in `scripts/`, a documented checksum,
  a `RUNTIME_VENDOR_ARTIFACTS` entry, wheel force-inclusion, a
  `DOCUMENTED_ARTIFACTS` entry in `tests/test_provenance_checksums.py`, and a
  `.gitignore` decision that is stated rather than defaulted.

## Must not change without an explicit decision

**Never hand-edit an artifact.** Not to correct a count, not to add a key, not to make
a loader stop complaining. Fix the generator and regenerate. Beyond the checksum
breaking, `load_phoneme_counts` cross-checks the artifact's `phones` list against
`ailm.levels.phoneme.PHONES` precisely because a reordered or altered inventory would
silently misindex every matrix built from the file — a hand-edit can produce a valid
JSON document that yields wrong numbers everywhere and raises nothing.

**Do not break determinism.** Use stable ordering, fixed seeds where sampling is
intentional, and no wall-clock timestamps. Filename-bound checksums verify the
committed bytes; the relevant generator tests require byte-identical reconstruction
where the necessary upstream sufficient statistics are available.

**A changed artifact and its documented checksum move together, in one commit.**
Adding `phone_initials` to `cmudict_counts.json` once changed the hash while
`docs/provenance/sources.md` kept the old value — prose asserting the wrong digest. A
stale checksum is worse than none: it tells a reader their correct file is corrupt.

**Do not change a derivation choice without re-recording every dependent claim.**
Primary-pronunciation-only, stress stripping, and the length-matched
position-aligned letter-to-phone (G2P) restriction
each bias the data and are documented as such. The train/test split is defined by
`test_every` **position** in the already-sorted `g2p_pairs` list — that is why
`g2p_split` must not re-sort, and why the split is reproducible from the artifact with
no seed to record. Changing the stride reshuffles train and test and invalidates every
Stage 2 number.

**Do not remove the non-canonical phone guard.** `generate_phoneme_counts.py` raises
rather than vendoring an inventory containing a phone outside `CANONICAL_ARPABET`.

**Do not delete `phone_initials`.** `full_corpus_perplexity` raises without it rather
than falling back to the length-matched subset, which flatters the model by ~1.46
perplexity units. The exact sufficient statistics for the closed-form full-dictionary
perplexity are `phone_initials` plus `phone_bigrams`; losing either forces a pymdp walk over all
claim:stage2_phoneme_eval_positions positions, far too slow for a gate.

**Do not add a network call to make a test pass.** Every test reads from here, never
from a URL.

## Local gotchas

**A fresh clone has no `cmudict.dict`.** `generate_phoneme_counts.py` needs it, so on
a clean checkout the download path runs first. The *derived* artifact is already
present, so the Stage 2 suite passes without ever fetching anything — the download is
only needed to regenerate.

**`generate_counts.py --force-fallback` is noncanonical by construction.** It
requires a noncanonical output filename, emits a schema the production loader
rejects, and may never overwrite `brown_counts.json` or support a recorded
claim. Canonical generation fails closed when the pinned `brown.zip` is absent
or modified.

**Base-word count ≠ entry count.** `cmudict_counts.json` derives from
claim:cmudict_base_word_count base words after folding `WORD(2)` alternates onto
their base key, while it retains claim:cmudict_pronunciation_entry_count
pronunciation entries. Do not assert one against the other. The same artifact
records the immutable raw URL, revision, byte extent, and SHA-256 used to derive
those values.

**`vendor/` sizes drift in prose.** The core manuscript sizes are generated as
`META_*` tokens. Do not add hand-typed size inventories here; use the runtime
manifest and filesystem checks.

**Gitignored downloads are invisible to `git status`.** A corrupted or truncated
`cmudict.dict` will not show up as a change — it will show up as a wrong phoneme
inventory. `verify_cmudict` is the check.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# Checksums match, and the generators reproduce byte-identical output
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q --no-cov

# The committed/gitignored split is intact
git check-ignore -v vendor/cmudict.dict                        # must be ignored
git ls-files 'vendor/*.json'                                    # the derived JSON files
git status --porcelain --untracked-files=all vendor/            # nothing here should be '??'

# Documented digests vs the files on disk
shasum -a 256 vendor/*.json
rg -o '[0-9a-f]{64}' docs/provenance/sources.md

# The upstream dictionary, if present, still yields the canonical inventory
.venv/bin/python -c "
from ailm.corpus.cmudict import verify_cmudict
r = verify_cmudict('vendor/cmudict.dict')
print(r['status'], r.get('entry_count'), r.get('phoneme_count'), r.get('error',''))
"
```

Never mask a generator failure with `2>/dev/null || true`. Suppressing stderr *and*
swallowing the exit code turns a broken regenerate into a silent no-op, and a stale
artifact then propagates into every recorded claim.
