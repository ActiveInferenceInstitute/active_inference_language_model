# AGENTS.md — src/ailm/corpus/

## Ownership boundary

`corpus/` owns **parsing** upstream corpus formats and nothing else. It does not own:

- the derived artifacts → `vendor/`, written by `scripts/generate_counts.py` and
  `scripts/generate_phoneme_counts.py`
- the case ontology and the deprel/adposition tables → `ailm/cases.py`
- any model built from a corpus → `ailm/levels/`
- license and checksum provenance → `docs/provenance/sources.md`

`cmudict.py` and `ud.py` are the only modules in `src/` permitted to make a
network request, and only on a cache miss with `download=True`. `brown.py`
authenticates an already-present raw archive and `ud_czech.py` authenticates
already-staged shards; neither downloads. `case_word.py` has no network path at
all: it validates and loads the committed case-word (L4→L3) bridge artifact from
its canonical repository path, `data/case_word/`, or the identical `_data/`
resource embedded in an installed wheel for the connected runtime. That
artifact responsibility (identity, split, route declaration, and content
digests, exercised by `tests/test_corpus_case_word.py`) belongs to this
directory. No other module here may add a
network path. `_download.py` may validate an already-open response and publish
authenticated bytes, but it must never issue a request. Raw downloads are
redirect-free, identity-encoded streams with one exact `Content-Length`; they
are bounded and hashed before a fsync-plus-replace cache publication.

## May change freely

- Add a loader for a new corpus, with an offline fixture in `tests/` — a tiny inline
  dict or an inline CoNLL-U string, in the style of the `tiny_cmudict` fixture.
- Improve a parser's tolerance for a real upstream malformation, with the malformed
  line pinned in a test.
- Add a field to `Token` or `Sentence`.

## Must not change without an explicit decision

**Do not remove `_strip_comment`, and do not fold it into the split.** Comment
stripping must happen before tokenisation. Without it the inventory reads
claim:phoneme_count_naive_unstripped phones instead of claim:phoneme_count_39,
and every downstream matrix acquires claim:phoneme_count_comment_phantoms phantom
states while still validating.

**Do not make `build_phoneme_inventory` assert the count.** The split between it and
`assert_full_inventory` is deliberate: the builder runs on small test dicts, the
assertion only on the full dictionary. Merging them makes every fixture-based test
fail and invites someone to weaken the real gate.

**Do not silently widen `CANONICAL_ARPABET`.** It is asserted against
`EXPECTED_PHONEME_COUNT` at module load, `ailm.levels.phoneme.PHONES` is derived from
it, `load_phoneme_counts` cross-checks the vendored artifact's `phones` list against
it, and `tests/test_claims_discipline.py` binds the ledger's `phoneme_count_39` value
to `len(CANONICAL_ARPABET)` rather than to the literal 39. A change here therefore
has to propagate through the artifact, the ledger, and the manuscript.

**Do not present the derived cases as ground truth.** DAT/INS/LOC/ABL/VOC come from
`deprel` and token-level adposition lemmas, not from English morphology. Changing a mapping entry
is a revision of a research claim: version it, pin the new behavior in
`tests/test_ud.py`, and note it in `docs/design/cerebrum_mapping.md`. Never describe
oracle agreement as accuracy.

**Do not use UD EWT for anything phonological.** Its fields are
`lemma`/`upos`/`feats`/`deprel`; there is no phonemic annotation. Stage 2's original
gate named EWT as a letter-to-phone (G2P) oracle and was wrong on that basis alone.

**Do not add a network call to a test.** Every test in this project is offline. If a
test needs corpus data, use a fixture or the vendored artifact.

## Local gotchas

**`load_cmudict` reads latin-1, not utf-8.** Entries like `ABBÉ` are latin-1 in the
upstream file. `ud.py` reads utf-8. Do not unify them.

**Alternate pronunciations are folded onto the base key.** `WORD(2)` becomes another
entry in `result["WORD"]`, so `cmudict["CAT"]` is a *list of* pronunciations. Every
Stage 2 derivation uses only the first (primary) one — a scoping choice recorded in
`docs/provenance/sources.md`, not an accident.

**`word_to_phones` no longer takes `primary_only`, and the removal is pinned.** The
flag's two branches were identical (`pronunciations[0] if primary_only else
pronunciations[0]`), so `primary_only=False` silently returned the primary anyway. It
was deleted rather than implemented — a parameter advertising a capability no caller
wants is worse than its absence, because the next caller would trust it.
`TestWordToPhonesReturnsThePrimaryPronunciation::test_primary_only_is_rejected_rather_than_silently_ignored`
in `tests/test_cmudict.py` requires a `TypeError`, so re-adding the keyword as a no-op
fails the suite. Callers
needing alternates read `cmudict[word.upper()]`, which is already the full list.

**Base-word and pronunciation-entry extents are different quantities.** CMUdict's
claim:cmudict_base_word_count base words after folding alternates are distinct from
its claim:cmudict_pronunciation_entry_count pronunciation entries. Both are measured
into `vendor/cmudict_counts.json`; do not assert one against the other. Raw-source
identity is pinned separately by `CMUDICT_REVISION`, `CMUDICT_RAW_SHA256`, and
`CMUDICT_RAW_SIZE_BYTES`.

**`vendor/cmudict.dict` and `vendor/en_ewt-ud-dev.conllu` are gitignored.** The
loaders' `download=True` path is how a fresh checkout obtains them. A source checkout
retains `vendor/` as its default destination; an installed wheel must use the
per-user cache resolved by `ailm.resources`, never a path derived by walking upward
from `site-packages`. `AILM_CORPUS_CACHE_DIR` is the absolute-path override for both
modes. The *derived* JSON artifacts are committed instead — see
`../../../vendor/AGENTS.md`.

**The adposition rule labels the ADP token, not its head.** `_apply_case_oracle`
takes one token because one token is all its implementation has ever used. The
``adposition_token`` route maps that token's own lemma. Head lookup belongs only
to `case_oracle_scope_audit`, which resolves CoNLL-U IDs to measure duplication and
role disagreement without silently changing the labeled population. A future
head-transfer rule is a new research design, not a use for an unused parameter.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

.venv/bin/python -m pytest \
  tests/test_cmudict.py tests/test_cmudict_load.py tests/test_ud.py -q --no-cov

# The 39-phone invariant against the real vendored dictionary, if present
.venv/bin/python -c "
from ailm.corpus.cmudict import verify_cmudict
r = verify_cmudict('vendor/cmudict.dict')
print(r['status'], r.get('entry_count'), r.get('phoneme_count'), r.get('error',''))
"
```
