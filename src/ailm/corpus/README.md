# ailm/corpus/ — corpus loaders and the case oracle

Five source-contract modules. Only `cmudict.py` and `ud.py` touch the network, and
they do so only when a cached raw file is absent. A source checkout keeps raw downloads in
`vendor/`; an installed wheel uses the platform's writable per-user cache. Set the
absolute `AILM_CORPUS_CACHE_DIR` path to select another destination. Everything
downstream reads the derived artifacts packaged from `vendor/`, so ordinary tests
and model runs are offline by construction.

Both network paths refuse redirects, compression, transfer encoding, and absent,
duplicate, malformed, or inexact `Content-Length` declarations. They stream and
hash no more than the declared pinned extent plus one control byte, authenticate
both the exact extent and SHA-256, then fsync and atomically replace a sibling
temporary file. An HTTP error, truncated body, oversize body, or identity mismatch
therefore leaves no cache entry to be mistaken for a corpus.

| Module | Corpus | Feeds |
| --- | --- | --- |
| `brown.py` | exact pinned NLTK Brown archive (non-commercial-use metadata) | every `vendor/brown_*.json` artifact; direct verified-zip reader, no download |
| `cmudict.py` | CMU Pronouncing Dictionary (BSD-2-Clause) | the claim:phoneme_count_39 canonical ARPAbet phones, and every Stage 2 input via `vendor/cmudict_counts.json` |
| `ud.py` | Universal Dependencies English EWT (CC BY-SA 4.0) | the 8-case oracle for Stage 4 |
| `ud_czech.py` | Universal Dependencies Czech-PDTC (CC BY-NC-SA 4.0) | immutable identities for both shards behind the exploratory Czech morphology arm |
| `case_word.py` | loader for the derived case-word (L4→L3) bridge artifact over UD English EWT dev rows — repo-path evidence, no network path | the case-word bridge sender map and inlet (protocol P3) |

The connected GUM evaluation has a separate input boundary:
`connected_gum.py` authenticates the frozen local protocol and raw test file,
then exposes document IDs and raw text comments only. It performs no download
and does not pass annotation columns to the runtime.

## `brown.py`

`resolve_brown_source_zip`, `verify_brown_source_bytes`,
`load_brown_corpus`, and `validate_brown_artifact_provenance`, plus the exact
`BROWN_REVISION`, `BROWN_RAW_URL`, `BROWN_RAW_SHA256`, and
`BROWN_RAW_SIZE_BYTES` identity. The reader opens the authenticated archive
directly; it never delegates selection to NLTK's global lazy loader, which could
prefer an unverified extracted tree. There is no network path and no canonical
fallback. Supply `--source-zip PATH` to a writer or set
`AILM_BROWN_SOURCE_ZIP`.

## `ud_czech.py`

`UD_CZECH_PDTC_SHARDS` binds the artifact's `train` split to upstream
`train-ft` and its held-out `test` split to upstream `dev`, with an immutable
revision, URL, digest, byte extent, and filename for each. The Czech writer
reads and validates both exact byte strings before parsing either one; the
runtime artifact loader validates the same nested identity again.

## `cmudict.py`

`load_cmudict`, `build_phoneme_inventory`, `assert_full_inventory`, `word_to_phones`,
`verify_cmudict`, plus the constants `CANONICAL_ARPABET`, `EXPECTED_PHONEME_COUNT`,
`CMUDICT_REVISION`, `CMUDICT_URL`, `CMUDICT_RAW_SHA256`, and
`CMUDICT_RAW_SIZE_BYTES`. The URL contains the immutable revision rather than a
moving branch name; downloads through that URL are digest-checked before caching.

```python
from ailm.corpus.cmudict import assert_full_inventory, build_phoneme_inventory, load_cmudict

d = load_cmudict(download=False)               # checkout vendor/ or installed user cache
inventory = build_phoneme_inventory(d)         # stress digits stripped
assert_full_inventory(inventory)               # raises unless exactly the canonical set
```

Both raw loaders also accept a `url=` keyword for an internal mirror. An explicit
`path=` still wins over every default, and `download=False` fails rather than making
a network request when that path is absent. A CMUdict mirror must supply
`expected_sha256=` and supplies `expected_size_bytes=` when its authenticated body
is not the canonical pinned extent. UD mirrors always declare the full revision,
digest, and extent tuple. The same pre-cache checks apply to official and mirror
responses.

`build_phoneme_inventory` deliberately returns whatever it finds and does **not**
assert 39 — it is called on tiny test dicts too. `assert_full_inventory` is the
separate gate, and calling it on a small fixture is a misuse that will fail.

### The comment-stripping trap

Naive whitespace splitting of CMUdict lines yields
**claim:phoneme_count_naive_unstripped** apparent phone tokens, not
39. Some entries carry an inline `#` comment:

```
ABBÉ    AH0 B EY1  # french
```

Without stripping, `french` becomes a phoneme. `_strip_comment` runs before
`_parse_line` splits, and
`tests/test_cmudict.py::TestNegativeControl::test_assert_full_inventory_catches_comment_token`
is the negative control proving the claim:phoneme_count_39 phone gate can actually
fail. The failure mode is quiet: an over-large inventory still builds valid-looking
matrices, just with claim:phoneme_count_comment_phantoms phantom states.

Stress digits are a second, independent normalization: `AH0`/`AH1`/`AH2` collapse to
`AH` via `_strip_stress`. Skipping that inflates the inventory a different way.

## `ud.py`

`Token`, `Sentence`, `parse_conllu`, `ensure_ud_ewt_dev_path`,
`load_ud_ewt_dev`, `case_oracle_stats`, plus the immutable
`UD_EWT_REVISION`, `UD_EWT_DEV_URL`, `UD_EWT_RAW_SHA256`, and
`UD_EWT_RAW_SIZE_BYTES` source identity. The
parser handles `# sent_id` / `# text` metadata, ten-field token lines, blank-line
sentence boundaries, and skips multi-word tokens (`1-2`) and empty nodes (`1.1`).

Every loader call verifies the raw bytes, including an already-cached local
file. The default URL contains the pinned Git revision; mirrors must carry a
complete revision/digest/extent declaration, and moving branch URLs are refused.
Downloads are verified before an atomic cache write. The three derived English
EWT artifacts retain the same raw identity and their loaders validate it before
returning model inputs.

```python
from ailm.corpus.ud import case_oracle_stats, parse_conllu

sentences = parse_conllu(conllu_text)
print(case_oracle_stats(sentences)["derived_cases"])
```

### The 8-case oracle is a research claim, not ground truth

English marks only **Nominative, Accusative, and Genitive** morphologically. Those
three come from UD `feats["Case"]` and are gold. DAT / INS / LOC / ABL / VOC are
*derived* by this project from `deprel` labels and token-level adposition lemmas.
The dependency route (`UD_DEPREL_TO_CASE`) maps `iobj`→DAT, `obl`→LOC,
`obl:agent`→ABL, and `vocative`→VOC; no `obl:with` or `obl:from` variant exists.
The token-level adposition route (`ADPOSITION_TO_CASE`) labels the `case`-marked
ADP token from its own lemma — `with`/`by`→INS, `from`→ABL, `in`/`at`/`on`→LOC,
`to`/`for`→DAT, `of`/`'s`→GEN. Both mappings live in `ailm.cases`.

That mapping is a principled heuristic that has not been externally reviewed.
Agreement with it is **agreement**, never **accuracy**, and any Stage 4 number
resting on it must say so in the same sentence. Note also that the same UD EWT
treebank was once written into the *Stage 2* gate as a phoneme oracle — it has no
phonemic field at all, only `lemma`/`upos`/`feats`/`deprel`. Do not reach for EWT
for anything phonological.

`case_oracle_assignment` resolves in a fixed order — morphological feature, then
`deprel`, then the ADP token's own lemma — and returns no role/path when no rule
applies, so unassigned tokens are visibly unassigned rather than defaulted to
NOM. The last route labels that ADP token; it does not look up or relabel the
nominal head. `case_oracle_scope_audit` performs the separate ID-based head audit
and retains sufficient counts for the fidelity population, duplicated head
labels, and head/token role disagreement (decision 84).

## Verification

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -m pytest \
  tests/test_cmudict.py tests/test_cmudict_load.py tests/test_ud.py \
  tests/test_corpus_case_word.py -q --no-cov
```

## More information

[`AGENTS.md`](AGENTS.md) for boundaries. Licenses, URLs, checksums, and every
derivation choice that biases the data: `../../../docs/provenance/sources.md`.
