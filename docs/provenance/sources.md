# Source Provenance

## Opt-in live revalidation

Normal tests and manuscript rendering read `data/reference_sources.json` and do
not access the network. To compare that snapshot with the exact official Zenodo
version records, run:

```bash
.venv/bin/python -I scripts/verify_reference_sources.py
```

That mode validates the record identifier, version DOI, concept DOI, title,
version, publication date, complete file inventory, byte extent, Zenodo MD5,
and the canonical deposited-file URL. It refuses redirects, proxies, content
encoding, ambiguous framing, and any host or protocol other than the exact
`https://zenodo.org` API target.

To additionally stream each complete PDF through the bounded authenticator and
check its byte extent, MD5, SHA-256, and local Poppler page count, run:

```bash
.venv/bin/python -I scripts/verify_reference_sources.py \
  --full-files --require-pdfinfo
```

Omit `--require-pdfinfo` when Poppler is intentionally unavailable; the JSON
receipt then marks the page extent `unavailable` rather than claiming it was
checked. File bytes still have to pass both digests. The verifier never rewrites
the snapshot or retains downloaded source PDFs.

## Current project archival status

The project previously configured `10.5281/zenodo.22409419` as its archival
identifier. On 2026-10-05, `https://zenodo.org/api/records/22409419` returned
HTTP 404. That public-record observation did not establish whether a private
draft existed or an identifier had been reserved.

On 2026-10-11, an authenticated, read-only request to
`https://zenodo.org/api/deposit/depositions/22409419` returned HTTP 200. The
response identified the manuscript by its exact configured title, contained
`metadata.prereserve_doi.doi` equal to `10.5281/zenodo.22409419`, and reported
`submitted: false`. The retained response has SHA-256
`f6466482f0be2233199983bccd5a53db3a04606377ad2af79f73f3e139f0e67a`.
This verifies an existing reserved, unsubmitted draft; it does not establish
a published record or public DOI resolution. No draft was created or published
by this verification.

Following human manuscript acceptance, the title page, `CITATION.cff`, package
URLs and README identify the release archive through
`publication.doi` in `docs/manuscript/config.yaml`. The authenticated draft
identity was reconfirmed before release preparation. These earlier draft checks
remain historical evidence; completed publication requires the public record
and the deposited file checksums to match the final release artifacts.

### Published release record

The authenticated deposition `22409419` was published on 2026-10-30, and the
reserved identifier `10.5281/zenodo.22409419` became the public record DOI. The
deposited files are the tagged source archive for `v1.0.0` and the combined
release PDF, and the record's `related_identifiers` link it to the
ActiveInferenceInstitute release repository and to the `v1.0.0` release page,
which links back to the record. The title page, `CITATION.cff`, package URLs and
README name the same published archive.

## Primary Papers

### CEREBRUM: archival record and deposited artifact

- **Author:** Daniel Ari Friedman
- **Concept DOI:** https://doi.org/{{CEREBRUM_CONCEPT_DOI}}
- **Exact record:** https://doi.org/{{CEREBRUM_RECORD_DOI}}
- **Zenodo metadata identity:** *{{CEREBRUM_RECORD_TITLE}}*, version {{CEREBRUM_ZENODO_VERSION}}, published
  {{CEREBRUM_ZENODO_DATE}}
- **Deposited file:** `{{CEREBRUM_PDF_FILENAME}}`; {{CEREBRUM_PDF_SIZE_BYTES}} bytes;
  {{CEREBRUM_PDF_PAGE_COUNT}} PDF pages; MD5 `{{CEREBRUM_PDF_MD5}}`; SHA-256
  `{{CEREBRUM_PDF_SHA256}}`
- **Printed identity of those exact bytes:** *{{CEREBRUM_PRINTED_TITLE}}*, Version {{CEREBRUM_PRINTED_VERSION}}
  ({{CEREBRUM_PRINTED_DATE}})
- **Identity policy:** the Zenodo version/date and the deposited PDF's printed
  version/date conflict. AILM preserves both; “v{{CEREBRUM_ZENODO_VERSION}} PDF” is shorthand for the
  record identity, not a claim about what its title page prints.
- **Role in AILM:** core and novel model-role ontology; structurally distinct
  case free-energy formulations; [Supplement {{CEREBRUM_POMDP_SUPPLEMENT}}]({{CEREBRUM_POMDP_SUPPLEMENT_URL}})'s
  `(S, A, T, Ω, O, F, π)` tuple. [Figure {{CEREBRUM_ACTIVE_INFERENCE_FIGURE}}]({{CEREBRUM_ACTIVE_INFERENCE_FIGURE_URL}}) on physical page {{CEREBRUM_ACTIVE_INFERENCE_FIGURE_PAGES}} is caption-only
  and supplies no component crosswalk.
- **Truncated export warning:** the `full_text.md` export omits later
  supplements. Use the exact complete PDF, verified against the digest above,
  for source evaluation.

### Compositional-case work: archival record, deposited artifact, and companion

- **Author:** Daniel Ari Friedman
- **Concept DOI:** https://doi.org/{{COMPOSITIONAL_CASE_CONCEPT_DOI}}
- **Exact record:** https://doi.org/{{COMPOSITIONAL_CASE_RECORD_DOI}}
- **Zenodo metadata identity:** *{{COMPOSITIONAL_CASE_RECORD_TITLE}}*, {{COMPOSITIONAL_CASE_ZENODO_VERSION}}, published {{COMPOSITIONAL_CASE_ZENODO_DATE}}
- **Deposited file:** `{{COMPOSITIONAL_CASE_PDF_FILENAME}}`;
  {{COMPOSITIONAL_CASE_PDF_SIZE_BYTES}} bytes; {{COMPOSITIONAL_CASE_PDF_PAGE_COUNT}} PDF pages; MD5
  `{{COMPOSITIONAL_CASE_PDF_MD5}}`; SHA-256
  `{{COMPOSITIONAL_CASE_PDF_SHA256}}`
- **Printed identity of those exact bytes:** *{{COMPOSITIONAL_CASE_PRINTED_TITLE}}*, dated {{COMPOSITIONAL_CASE_PRINTED_DATE}}; the title page
  prints no edition number
- **Optional companion identity:** revision
  `{{COMPOSITIONAL_CASE_COMPANION_REVISION}}` of
  `docxology/cognitive_case_diagrams` declares the printed title, manuscript
  version {{COMPOSITIONAL_CASE_COMPANION_VERSION}}, and date {{COMPOSITIONAL_CASE_COMPANION_DATE}}
- **Identity policy:** the bibliography follows the Zenodo record title because
  that is what the DOI resolves to; the note and
  `data/reference_sources.json` preserve the deposited and companion identities.
- **Role in AILM:** enriched categorical formalism, belief update equation
  `q^(t+1)(c_k) ∝ q^(t)(c_k) · exp(w_k · o_k)`, and case-typed DisCoCat
  motivation in `FVect`/`FinHilb`. The paper does not define a `FinStoch`
  functor; AILM's earlier attribution of one was incorrect.

## Optional Companion Code (CCD Reference Implementation)

The companion repository is **not vendored by AILM**: `refs/` is ignored, and a
clean AILM checkout neither contains nor needs it. The exact archival paper above
is the source required to evaluate the manuscript claims. For optional code
inspection, clone `https://github.com/docxology/cognitive_case_diagrams` and
checkout revision `{{COMPOSITIONAL_CASE_COMPANION_REVISION}}`; that is the
revision inspected for this provenance record.

- **Optional local location:** `refs/cognitive_case_diagrams/src/`
- **Key modules:**
  - `case_systems/case_category.py` — `CaseRole` enum, `CaseCategory` with multiplicative weight composition
  - `cognitive/belief_updating.py` — `sequential_belief_update`
  - `enriched_cat/enriched.py` — `EnrichedCategory` with 8×8 proximity matrix
  - `daif/core.py` — distributional Bellman operator
  - `cognitive/free_energy.py` — `kl_divergence`, `variational_free_energy`

## External Corpora

### CMUdict

- **Source:** Carnegie Mellon University Pronouncing Dictionary
- **Official repository:** `https://github.com/cmusphinx/cmudict`
- **Pinned revision:** `74790861f652b15e4ac49015a90074ad62a27690`
- **Immutable raw URL:** `https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict`
- **Raw sha256:** `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`
- **Raw byte extent:** recorded as `source_size_bytes` in the committed derived
  artifact and checked against `CMUDICT_RAW_SIZE_BYTES` before regeneration
- **License:** BSD-2-Clause
- **Role in AILM:** Phoneme inventory (claim:phoneme_count_39 canonical ARPAbet), phone transitions `B` and the position-aligned `P(letter | phone)` emission `A`
- **Fail-closed acquisition:** both the official URL and an explicitly declared
  mirror are fetched without redirects or content encoding. One exact
  `Content-Length` is required; the body is bounded and incrementally hashed,
  then its extent and SHA-256 are authenticated before a fsynced sibling
  temporary is atomically published. A mirror still requires its own digest,
  and a noncanonical body also requires its exact extent.
- **TRAP:** Naive whitespace-split parsing yields claim:phoneme_count_naive_unstripped "phonemes" because comment
  tokens after `#` are captured. Must strip `#` comments first.
  Correct count: claim:phoneme_count_39 phonemes.

### Universal Dependencies English EWT

- **Source:** Universal Dependencies English Web Treebank (EWT)
- **Immutable revision:** `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`
- **URL:** `https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/en_ewt-ud-dev.conllu`
- **Raw sha256:** `39239e0a60db3ae68f4b7036189f11b6692741d10ff8240dd91f74f2760d90f8`
- **Raw byte extent:** 1,898,574 bytes
- **License:** CC BY-SA 4.0
- **Role in AILM:** UD-derived case-role data for Stages 4–6 and their measured
  inter-level seams. Stage 2 uses CMUdict and does not use EWT.
- **Fail-closed acquisition:** `ailm.corpus.ud.ensure_ud_ewt_dev_path` rejects
  moving `master`/`main` URLs and incomplete source identities, validates both
  existing cache files and downloaded mirror bytes, refuses redirects and
  ambiguous or encoded response framing, and retains at most the expected
  extent plus one rejecting control byte while hashing. It authenticates the
  exact extent and SHA-256 before publishing through a fsynced unique atomic
  temporary. A corrupt existing file is never silently replaced.
- **Research claim:** English EWT has NOM/ACC/GEN morphological marking only.
  DAT/INS/LOC/ABL are derived via deprel mapping (nsubj→NOM, obj→ACC, iobj→DAT,
  nmod:poss→GEN, obl→LOC, obl:to→DAT, obl:with→INS, obl:from→ABL, vocative→VOC).
  This derivation is a principled heuristic, not ground truth.

### UD EWT derived raw artifact — `vendor/en_ewt-ud-test.conllu` (decision 79 held-back partition)

- **Source:** the pinned revision's test split, acquired through the identical
  fail-closed path as the dev split
- **Immutable revision:** `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b` (same pin as the dev split)
- **URL:** `https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/en_ewt-ud-test.conllu`
- **Raw sha256:** `fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a`
- **Raw byte extent:** 1,900,101 bytes
- **License:** CC BY-SA 4.0
- **Historical status under decisions 79 and 91:** originally vendored as an
  unscored `ud_ewt_test` held-back partition. The decision-79 replication was
  retired after the paired `external_lexicon` population proved empty; decision
  91 preserved this partition for a successor registration using the identical
  generators (`src/ailm/corpus/ud.py`, `ensure_ud_ewt_test_path`). This was its
  status before decision 111, not a description of its current availability.
- **Historical falsifier (decision 91):** if any script reads
  `en_ewt-ud-test.conllu` before a successor registration vendors it through
  the identical generators, the population burns and any successor entry must
  say so.
- **Status under decision 111 (2026-09-05):** the successor registration landed
  and the four `ud_ewt_test` family members were scored against this file
  exactly once. The population is spent, marked by the ledger sentinel
  `confirmatory_run_completed_ud_ewt_test`. Decision 115 subsequently registered
  a separate document-clustered Stage 6 estimand on this already spent split; it
  does not restore held-back independence. No new result may claim an untouched
  population. Deterministic descriptive replay is permitted with the spent-split
  boundary intact. The three derived artifacts of the first run are recorded
  immediately below.

### Confirmatory replication artifacts — `data/confirmatory/` (decision 111)

- **Source:** the whole of `vendor/en_ewt-ud-test.conllu` above, at the same
  pinned revision, read through `ailm.corpus.ud.load_ud_ewt_test`
- **Generator:** `scripts/generate_confirmatory_ewt_counts.py --population ud_ewt_test`
- **License:** CC BY-SA 4.0, inherited from the source treebank
- **Role in AILM:** the evaluation population of the decision-111 confirmatory
  replication. The retained narrative artifact is also read indirectly by
  `scripts/record_confirmatory_stage6_clustered_claims.py` through
  `merged_narrative_artifact` for the separately registered decision-115
  estimand. No fresh split or new training population is implied. They are not
  part of
  `ailm.resources.RUNTIME_VENDOR_ARTIFACTS` and are not packaged into the wheel,
  which is why they live under `data/confirmatory/` rather than `vendor/`.
- **Population rule:** the entire corpus is the evaluation arm and **no stride is
  applied**, because nothing is fitted on it. Every model parameter is read
  unchanged from the existing development train split. The identical oracles,
  exclusion rules and minimum-length filters are imported from the development
  generators rather than retyped.
- **Census:** claim:confirmatory_census_sentences sentences,
  claim:confirmatory_census_tokens tokens and
  claim:confirmatory_census_documents authenticated `# newdoc id` documents,
  disjoint from every development document identity.
- **Artifact sha256 (`data/confirmatory/ud_ewt_test_case_counts.json`):**
  `68e68daed5503d9095d083124a52590846bccd672901b2b2a505b94ab3007573`
- **Artifact sha256 (`data/confirmatory/ud_ewt_test_clause_counts.json`):**
  `2277b46ccfda016c4530a5fcf786842d8566784780747563ecbce9619f7d830e`
- **Artifact sha256 (`data/confirmatory/ud_ewt_test_narrative_counts.json`):**
  `6d15a121ddd6ceae168f92d4b07040bc5964995ac749de770f20ee38aa96529a`
- **Determinism and enforcement:** verified byte-identical on regeneration; the
  three digests above are bound to their filenames and checked against the files
  on disk by `tests/test_confirmatory_replication.py`, which also re-derives
  every recorded p-value from its recorded counts and re-checks that the
  confirmatory documents are disjoint from the development ones.

### Penn Treebank — excluded historical target

- **Role:** NONE. Stage 3 was originally specified against "PTB-1k" and is not
  measured on PTB at all. The full corpus is LDC-licensed and NLTK ships only a
  small sample under research terms, so a number reported against "PTB" would have
  named a corpus this project does not have. Stage 3 is measured on Brown, which
  is already vendored, checksummed and byte-reproducible, and is the corpus
  Level 1 trains on. See `docs/decisions.md` entry 14.
- **Access:** not used

### Brown Corpus (in use — Stage 1)

- **Role:** Letter-level bigram training at Stage 1
- **Official repository:** `https://github.com/nltk/nltk_data`
- **Pinned revision:** `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`
- **Immutable raw URL:** `https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/corpora/brown.zip`
- **Raw sha256:** `9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6`
- **Raw byte extent:** `3314357`, enforced before any corpus member is read
- **Access terms:** NLTK's
  [`DATASET-LICENSES.md`](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md)
  groups `brown` under non-commercial use, and its
  [`index.xml`](https://github.com/nltk/nltk_data/blob/gh-pages/index.xml)
  says "May be used for non-commercial purposes." It is not public domain.
- **Extent used:** the deterministic normalized count artifact contains
  claim:stage1_brown_bigram_types distinct bigram types over
  claim:stage1_brown_bigram_tokens bigram transitions
- **Vendored artifact:** `vendor/brown_counts.json` — **committed**, unlike the
  upstream corpus, so the Stage 1 tests need no network access; its current byte
  extent is measured from disk during manuscript-variable generation
- **sha256:** `778e2b31740b417e2cc3829e4919a5e4d8f9a180a18695474ad05060d1f5f660`
- **Determinism:** verified byte-identical on regeneration via
  `scripts/generate_counts.py --source-zip PATH`; the writer authenticates the
  archive before constructing a reader directly over it, embeds the complete
  raw identity, and publishes atomically. The production loader validates the
  schema, source identity, alphabet, key/value domain, and count totals
- **Normalization applied:** lowercased; every non-ALPHABET character mapped to
  space; whitespace runs collapsed to a single space; spaces immediately before
  `.` and `'` stripped. The last two steps are deliberate — Brown's tokeniser
  emits punctuation as separate tokens, so without them `(space, space)` would
  dominate every other bigram and no letter would ever transition directly into
  a period, distorting end-of-word statistics
- **Fallback:** none for canonical generation. Explicit
  `--force-fallback --output noncanonical-brown-diagnostic.json` is diagnostic
  only: it has a noncanonical schema, cannot use the canonical filename, and is
  rejected by the production loader. Missing or modified raw bytes leave the
  canonical artifact untouched

### Stage 1 evaluation protocol fixture

- **Role:** the fixed letter sequence scored by the Stage 1 gate and its
  recorder, learning and complexity analyses, smoothing sweep, and gate audit.
  It is evaluation input, not Brown training data and not an independently
  sampled corpus.
- **Canonical owner:** `ailm.evaluation.STAGE1_EVALUATION`; benchmark code keeps
  `HELD_OUT_TEXT` only as a compatibility alias to the same object. Production
  code never imports the test suite.
- **Protocol:** `ailm-stage1-letter-evaluation-v1`
- **UTF-8 byte extent:** `1323`
- **SHA-256:**
  `513f73d02c2220185daf0a30865bed07a9af2167b40dcf5bb4341617efc6c20f`
- **Composition:** an AILM-curated normalized composite of an AILM-authored
  rhyming sentence, traditional tongue twisters and nursery verse, and a
  pangram. The base passage is repeated `3` times. That repetition makes the
  deterministic scoring stream longer but does not create independent
  linguistic material.
- **Historical references for incorporated phrases:** the Library of Congress
  documents early forms of [“she sells sea shells”](https://blogs.loc.gov/folklife/2017/07/she-sells-seashells-and-mary-anning-metafolklore-with-a-twist/)
  and [the woodchuck question](https://www.loc.gov/resource/sn84020645/1906-04-15/ed-1/?sp=20);
  Project Gutenberg preserves early print sources for [the pangram](https://www.gutenberg.org/ebooks/66541),
  [Peter Piper](https://www.gutenberg.org/ebooks/25027),
  [Mary's lamb](https://www.gutenberg.org/ebooks/67302), and
  [Humpty Dumpty](https://www.gutenberg.org/ebooks/24623).
- **Terms:** the AILM selection and normalization are CC BY-SA 4.0 under this
  repository's license. The cited constituent phrases are public domain in the
  United States. The fixture validator requires the protocol, source, license,
  references, digest, and extent and rejects a modified or incompletely sourced
  fixture before any consumer can score it.

### Brown derived artifact — `vendor/brown_word_counts.json` (Stage 3)

- **Role:** all Stage 3 inputs — the bounded word vocabulary with training
  frequencies, every training type's frequency for the letter-bigram bar, and
  the held-out in-vocabulary evaluation stream
- **Derived from:** the immutable NLTK Brown `brown.zip` revision and raw SHA-256
  pinned in `ailm.corpus.brown`, via `scripts/generate_word_counts.py` with
  `--source-zip PATH`. The artifact embeds the complete `raw_source` record and
  `WordArtifact` validates it before accepting any counts
- **Committed:** yes — so Stage 3 tests need no network, matching the
  `brown_counts.json` precedent
- **sha256:** `2d10d0c6e237207a53566e37ed6c66e95ae9258aa4e3c8fc953f3a2ce84a2268` (enforced against the file on disk by
  `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration. A revision-bound
  SHA-256 ordering selects the nearest target whole-document count separately
  within every Brown category, and frequency ties are sorted alphabetically, so
  there is no RNG and no timestamp.
  The writer verifies the pinned archive before reading and publishes atomically.
  The artifact records its raw-source identity, ordered document identities,
  category assignments and digest, exhaustive partition, per-document token counts, per-evaluation-token
  document indices, vocabulary bound, and lexicon source; the loader rejects
  missing, extra, moving-source, cross-arm, duplicate, or internally inconsistent
  provenance fields, then filters the independently committed unfiltered
  denominator and requires element-for-element token and document-index agreement
- **Normalization applied:** lowercased; a token is kept only if every character
  is an ASCII letter. That drops punctuation, numerals and hyphenated forms,
  which is what keeps the observation alphabet at 26 letters plus one
  end-of-word symbol — a **different** alphabet from Level 1's 30 running-text
  symbols, so Stage 3 and Stage 1 perplexities are not comparable
- **Split:** by authenticated source document, not by word type. The
  category-stratified document rule assigns
  claim:stage3_train_documents documents to training and
  claim:stage3_heldout_documents to evaluation, with
  claim:stage3_document_overlap overlap, with every Brown category represented
  in both arms. A type-level split would put every
  evaluation token outside a closed-vocabulary model's support and measure the
  closed-vocabulary assumption rather than the model
- **Scope:** claim:stage3_coverage_pct% of held-out tokens are in vocabulary.
  The model is **undefined** on the remainder, and every Stage 3 number
  describes that sublanguage
- **Fallback:** none. A lexicon assembled from diagnostic prose would be a
  different language and a different model

### Brown held-out stream — `vendor/brown_heldout_stream.json` (Stage 3 denominator)

- **Role:** the complete held-out alphabetic Brown token stream, in corpus
  order and before bounded-vocabulary filtering. It is the denominator needed
  to reproduce Stage 3 coverage and out-of-vocabulary analyses; the model's
  scored `eval_tokens` remain the in-vocabulary subsequence stored in
  `brown_word_counts.json`
- **Derived from:** the pinned, byte-verified NLTK Brown archive via
  `scripts/generate_heldout_stream.py --source-zip PATH`. Both Brown writers call
  `ailm.corpus.brown.build_brown_document_partition`, so normalization and split
  logic have one owner; the complete `raw_source` record is embedded
- **Committed:** yes — so coverage and out-of-vocabulary checks need no network
- **sha256:** `749b0ed11c00504247985a759eb466fe78408dffd9cd600f815ffb3ca1f39723`
  (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; the stream preserves
  corpus and document order, carries the same document identity/partition as the
  scoring artifact, and contains no RNG state or timestamp
- **Normalization applied:** lowercased; tokens are retained only when every
  character is an ASCII letter. Unlike the bounded word artifact's vocabulary,
  this stream applies no word-length cutoff and drops no held-out alphabetic
  token

### UD EWT derived artifact — `vendor/ud_ewt_narrative_counts.json` (Stage 6)

- **Role:** all Stage 6 inputs — one entry per document with at least three
  typed sentences: the ordered sentence-type sequence. All authenticated raw
  documents are partitioned by document-index stride before sentence typing or
  minimum-length filtering
- **Derived from:** `vendor/en_ewt-ud-dev.conllu` (CC BY-SA 4.0)
  via `scripts/generate_narrative_counts.py`
- **Raw identity retained:** immutable revision, URL, SHA-256, and byte extent
  above are embedded in `provenance` and validated on every artifact load
- **Committed:** yes — so Stage 6 tests need no network
- **sha256:** `11e7b49ce27d61a173de8ac296212fd801095612d312f694c875cca91c185133` (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; document-index
  stride over authenticated first-appearance order, no RNG, no timestamp
- **Identity retained:** ordered raw document IDs, their digest, the exhaustive
  train/test partition, and one document index per eligible sequence. The
  loader rejects duplicate IDs, duplicate sequence mappings, count drift,
  out-of-arm mappings, entry/index aliases, and train/test overlap
- **Phase definition:** document-position tercile — a deterministic function
  of position, so every parameter is a count; no EM, no fitted segmentation

### UD EWT derived artifact — `vendor/ud_ewt_clause_counts.json` (Stage 5)

- **Role:** all Stage 5 inputs — one entry per typed sentence: the sentence
  type (declarative/interrogative, from sentence-final punctuation) and its
  derived case-role sequence. All raw `# newdoc id` blocks are partitioned by
  document stride before sentence typing/filtering; Stage 4 inherits the same
  document partition
- **Derived from:** `vendor/en_ewt-ud-dev.conllu` (CC BY-SA 4.0)
  via `scripts/generate_clause_counts.py`
- **Raw identity retained:** immutable revision, URL, SHA-256, and byte extent
  above are embedded in `provenance` and validated on every artifact load
- **Committed:** yes (154400 bytes) — so Stage 5 tests need no network
- **sha256:** `e3fe327068149fbc64144d8fda4a8befe7ea4dc5a32393d0a79ab071f92a71d3` (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; document-index
  stride over authenticated first-appearance order, no RNG, no timestamp
- **Identity retained:** ordered raw document IDs, their digest, the complete
  train/test document-index partition, and one document index per retained
  sequence. The loader rejects inconsistent counts, out-of-arm mappings, and
  any train/test document overlap
- **Observation recorded:** **derived case roles only.** Token forms, UPOS and
  deprel are withheld; the type label's source is the sentence-final
  punctuation token, and PUNCT tokens carry no case role, so the label source
  cannot appear in the observation stream (the generator asserts this)
- **Exclusions, stated:** claim:stage5_excluded_exclamative exclamative sentences
  (`!` marks emphasis at least as often as mood) and
  claim:stage5_excluded_unpunctuated unpunctuated web fragments carry no label
  and are excluded; every Stage 5 number describes the typed sublanguage

### Widened clause arm — `data/widened/` (Stage 5w, decision 112)

- **Role:** all inputs to the widened-observation clause arm registered in
  `docs/decisions.md` entry 112 — one entry per typed sentence: the sentence
  type and its **widened** observation sequence, the derived case roles plus one
  distinguished symbol per `VERB` and per `AUX` token in surface order. It is a
  separately preserved artifact beside the case-only ablation. Decision 114
  designates the widened model as canonical Level 5 without deleting or
  relabeling the case-only evidence
- **Canonical path and wheel packaging:** the artifact retains its registered
  `data/widened/` source path. `ailm.levels.widened.load_widened_clause_artifact`
  loads it for library and connected execution. It is explicitly listed in
  `ailm.resources.RUNTIME_DATA_ARTIFACTS` and packaged by `pyproject.toml` at
  `ailm/_data/widened/ud_ewt_clause_widened_counts.json`. The separate
  `RUNTIME_VENDOR_ARTIFACTS` inventory still owns committed `vendor/*.json`;
  remaining evaluation-only data artifacts are not implicitly packaged
- **Derived from:** `vendor/en_ewt-ud-dev.conllu` (CC BY-SA 4.0)
  via `scripts/generate_clause_widened_counts.py`, which imports the partition,
  the stride and the type-label rule from `scripts/generate_clause_counts.py`
  rather than copying them, so both artifacts inherit one document partition
- **Raw identity retained:** immutable revision, URL, SHA-256, and byte extent
  above are embedded in `provenance` and validated on every artifact load
- **Committed:** yes — so the arm reproduces without network
- **Artifact sha256 (`data/widened/ud_ewt_clause_widened_counts.json`):**
  `28664a4c9efe690aa69997ea312f1597bce9723c12dc4a839d6a829f9603c4aa`
  (bound to its filename and checked against the file by
  `tests/test_stage5_widened.py`, which also regenerates it and requires the
  bytes to match)
- **Determinism:** verified byte-identical on regeneration; document-index
  stride over authenticated first-appearance order, no RNG, no timestamp
- **Identity retained:** ordered raw document IDs, their digest, the complete
  train/test document-index partition, and one document index per retained
  sequence. The loader reuses the case-only artifact's document validator, so
  the two cannot drift in how they check identity, partitioning, or alignment
- **Observation recorded:** **derived case roles plus UPOS for verbs and
  auxiliaries, and nothing else.** The type label's source is still the
  sentence-final punctuation token. `PUNCT` emits nothing, and a `VERB` or `AUX`
  contributes its UPOS only — never its lemma, `feats`, or surface form. Both
  properties are asserted per sentence at build time by
  `ailm.levels.widened.assert_widened_disjointness`, which recomputes rather
  than trusts and is exercised against a planted lemma leak in
  `tests/test_stage5_widened.py`
- **Exclusions, stated:** claim:stage5w_excluded_exclamative exclamative
  sentences and claim:stage5w_excluded_unpunctuated unpunctuated web fragments
  carry no label, and claim:stage5w_excluded_too_short sentences fall under the
  usability minimum. That last rule counts **widened symbols**, where the
  case-only artifact counts case-bearing tokens, so the two arms' populations
  are nested rather than equal and their per-position perplexities are not
  compared

### UD EWT derived artifact — `vendor/ud_ewt_case_counts.json` (Stage 4)

- **Role:** all Stage 4 inputs — the case-role sequences for both populations,
  inheriting one raw-document partition made before either population is
  filtered
- **Derived from:** `vendor/en_ewt-ud-dev.conllu` (not committed; CC BY-SA 4.0)
  via `scripts/generate_case_counts.py`
- **Raw identity retained:** immutable revision, URL, SHA-256, and byte extent
  above are embedded in `provenance` and validated on every artifact load
- **Committed:** yes (393194 bytes) — so Stage 4 tests need no network
- **sha256:** `3de812c40139decbb1272ed8e52e90b71942198d49dc62dabd2df0a8a0fb9afe` (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; the split is a
  document-index stride over authenticated first-appearance order, with no RNG
  or timestamp
- **Identity retained:** ordered raw document IDs, their digest, the complete
  train/test document-index partition, and per-population document indices for
  every retained sequence. The loader rejects inconsistent provenance,
  out-of-arm mappings, and train/test overlap
- **Observation recorded:** **UPOS tag only.** `deprel` and `feats[Case]` are
  deliberately withheld, because the evaluation labels are *derived from* those
  fields — a model observing them would be reading its own answer key and would
  score near-perfectly while measuring nothing. The artifact records this
  restriction in its own `provenance.observation` field. See
  `docs/decisions.md` entry 15
- **Two populations, never merged:** `gold` is UD `feats[Case]`, real English
  morphology, claim:stage4_gold_cases_present of 8 case roles; `derived` adds the
  `deprel` / token-level adposition heuristic to reach all 8. A score against
  `derived` is **agreement with a research claim**, never accuracy
- **Partial fidelity check, not a bound:** where both labelings exist
  (claim:stage4_heuristic_gold_tokens tokens), the syntactic heuristic agrees
  with real morphology claim:stage4_heuristic_gold_agreement_pct% of the time.
  The overlap consists of claim:stage4_heuristic_gold_deprel_tokens
  dependency-route tokens and
  claim:stage4_heuristic_gold_adposition_token_tokens token-level
  adposition-route tokens, so it cannot bound the full derived population
- **Token/head scope retained in the artifact:** the full parsed shard contains
  claim:stage4_oracle_full_path_adposition_token_tokens token-level adposition
  assignments. The rule maps the ADP token's own lemma; it does not consult its
  head. claim:stage4_adposition_token_head_also_labeled_tokens of those tokens
  have a separately labeled head, and
  claim:stage4_adposition_token_head_role_disagreements ADP/head roles disagree.
  These sufficient counts document the construct without silently relabeling it
- **What the corpus cannot validate:**
  claim:stage4_cases_without_morphology of the 8 roles — Dat, Ins, Loc, Abl, Voc
  — have no morphological ground truth in English EWT at all. A different corpus
  with richer case morphology would change the project's language scope

### UD Czech-PDTC derived artifact — `vendor/ud_czech_case_counts.json` (decision 77, exploratory)

- **Role:** the Czech arm of Level 4 — the only population where the case level
  is scored against morphological ground truth rather than a heuristic
- **Official repository:** `https://github.com/UniversalDependencies/UD_Czech-PDTC`
- **Pinned revision:** `6d206ec7d337a7f76f34ddfc82893389cabbd76d`
- **Exact fitting shard:** `cs_pdtc-ud-train-ft.conllu`, immutable
  [raw source](https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDTC/6d206ec7d337a7f76f34ddfc82893389cabbd76d/cs_pdtc-ud-train-ft.conllu),
  sha256 `ed5ce30de11ed9ac14ce1ca54dffb31e79211324850ec0badf5eb1e3a3e2dff6`,
  byte extent `3726243`
- **Exact held-out shard:** `cs_pdtc-ud-dev.conllu`, immutable
  [raw source](https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDTC/6d206ec7d337a7f76f34ddfc82893389cabbd76d/cs_pdtc-ud-dev.conllu),
  sha256 `73a19117ff39411b1134840c3a0383a37f5dd89a2e07b3f8c0a8821d8d37d4bc`,
  byte extent `50117913`
- **Derived from:** `vendor/staging_ud_czech/cs_pdtc-ud-{train-ft,dev}.conllu`
  (large local staging inputs, **not committed**; CC BY-NC-SA 4.0, from the official
  [UD Czech-PDTC record](https://universaldependencies.org/treebanks/cs_pdtc/index.html))
  via `scripts/generate_czech_case_counts.py`. Both required shards are checked
  against filename, byte extent, and digest before either is parsed. The
  upstream test shard is deliberately never read
- **Committed:** yes — so the Czech arm's tests need no network and no staging
  tree; tests measure the artifact directly rather than trusting a prose size
- **sha256:** `4abe72f960e0a11c5384454c8b6f131088bcd10b2487af700ffab246d1dc92f7` (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; corpus file order,
  sorted alphabets, no RNG, no timestamp, and atomic publication. The artifact
  embeds the revision and both shard identities; `CzechCaseArtifact` refuses a
  missing, modified, or moving-source declaration
- **Study-design provenance:** the artifact records a prospective
  development-session specification, not an independently timestamped
  preregistration. Its `immutable_prior_artifact` is null. The loader refuses
  `preregistered` terminology unless a timezone-stamped, content-addressed
  prior artifact is supplied, and also refuses any attempt to change this
  arm's recorded status under the existing schema
- **Token columns retained:** **UPOS and `feats[Case]` only**, encoded as one integer
  per token (`upos_index * 8 + case_index`). The authenticated `# newdoc id`
  alphabet and one document index per retained sequence preserve the outer
  sampling unit: claim:stage4_czech_test_sequences sequences nest within
  claim:stage4_czech_test_documents contributing documents. FORM, LEMMA, XPOS, HEAD, DEPREL
  and MISC never enter the file, so the decision 15 circularity is structurally
  impossible rather than merely guarded — and unlike English, the label here IS
  the morphology, not a function of the withheld fields
- **Population:** nominal, case-inflecting UPOS only (ADJ, DET, NOUN, NUM,
  PRON, PROPN). ADP is excluded although Czech PDT annotates `Case=` on
  prepositions: there the feature records government (the complement's case),
  not inflection — the analogue of the English token-level adposition path of decision 63
- **Label source:** all labels come directly from UD morphological `feats[Case]`;
  claim:stage4_czech_cases_present CEREBRUM cases are represented (Czech has no
  ablative), over claim:stage4_czech_test_tokens held-out tokens

### Brown full-dictionary artifact — `vendor/brown_word_counts_full.json` (Stage 3, unbounded)

- **Role:** the **full-dictionary** Stage 3 configuration — every eligible Brown
  training type, claim:stage3_full_vocab_size of them, reaching
  claim:stage3_full_coverage_pct% held-out token coverage against
  claim:stage3_coverage_pct% for the shipped bounded vocabulary
- **Generated by:** `scripts/generate_word_counts.py` with `--source-zip PATH`
  and `--vocab-size 0`; the same pinned `raw_source` identity as the bounded
  artifact is embedded and validated at load time
- **Committed:** yes; the checksum below, not a copied size literal, identifies
  its current bytes
- **sha256:** `9e9608b135d6ef5ec2438e077834eb0c918b4ef6e8351ad8e25c084ba635a41f` (enforced by `tests/test_provenance_checksums.py`)
- **What it demonstrates, and the boundary it makes explicit:** the *model*
  scales to the full dictionary but the *pymdp agent* does not. The word factor's
  dense transition would be claim:stage3_full_word_factor_gb GB at float32, over the
  budget in `ailm.levels.word.MAX_AGENT_VOCAB_BYTES`, which caps instantiation at
  roughly claim:stage3_max_agent_vocab types. `build_word_agent` raises with that
  arithmetic rather than failing on an allocation.
- **Why measurement survives past the boundary:** every factor except `D` is
  deterministic, so the path-probability identity makes the closed form **exact**
  at any vocabulary size — a word's score *is* its prior mass. Full-dictionary
  perplexity is claim:stage3_full_word_perplexity against a bar of
  claim:stage3_full_bigram_baseline over
  claim:stage3_full_eval_positions emissions, so the gate stays BRACKETED at full
  scale. What is lost above the cap is the ability to *run pymdp*, not the
  ability to measure the model, and the distinction is the point
- **Custom lexicons:** `scripts/generate_word_counts.py --lexicon PATH` accepts
  one `word[<TAB>frequency]` per line. A lexicon supplied *without* frequencies
  yields a uniform word prior, which `docs/decisions.md` entry 14 records as one
  of the mis-wirings that beats the accuracy bar while being wrong — so the
  generator warns rather than accepting it silently. Evaluation always uses
  held-out Brown restricted to what the lexicon covers, never the lexicon itself

### Brown derived artifact — `vendor/brown_clause_word_counts.json` (entry 27, typed oracle -> word prior)

- **Role:** the bridge corpus for the privileged punctuation-type-oracle → L3
  word-prior intervention (`docs/decisions.md` entries 27, 88, and 89). One entry
  per **typed** sentence whose words
  are restricted to the Stage 3 word vocabulary, pairing each word token with
  the sentence type (declarative / interrogative, from sentence-final
  punctuation) of the sentence it appears in, so `P(word | sentence type)` can
  be estimated on a train split and applied to held-out tokens. The receiver is
  the word level (Brown); the reference type is the same variable Stage 5's
  clause model reasons about, but the artifact does not carry an L5 posterior.
- **Derived from:** the pinned, byte-verified Brown archive via
  `scripts/generate_clause_word_counts.py --source-zip PATH`; the complete
  `raw_source` identity is embedded and checked before claims are recorded
- **Committed:** yes — so entry-27 tests need no network
- **sha256:** `3f061df8fb36103db6eb52e9232bd9efb6bb9de04db6bddee422e740fdae7e63` (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** schema `ailm-clause-word-counts-v3`, verified byte-identical
  on regeneration; reuses Stage 3's authenticated category-stratified
  whole-document split, with no cross-arm document and no RNG or timestamp
- **Type label source:** sentence-final punctuation token form: `.` declarative,
  `?` interrogative; `!` and unlabeled sentences excluded, with counts recorded
  in provenance. This is the same type variable the clause level's own posterior
  is about, but its value here is privileged punctuation-derived supervision.

### Brown genre-drift derived artifact — `vendor/brown_online_drift.json` (entry 28, online learning)

- **Role:** the non-stationary letter stream for the online-vs-frozen learning
  experiment (`docs/decisions.md` entry 28). Two Brown genres whose
  letter-bigram transition statistics genuinely differ — government (pre-drift)
  and romance (post-drift), total-variation distance measured by P1, not
  asserted — each contributing a learn stream and a held-out evaluate stream
  (split by sentence-index parity, disjoint).
- **Derived from:** the pinned, byte-verified Brown archive via
  `scripts/generate_online_genre_counts.py --source-zip PATH`; the complete
  `raw_source` identity is embedded and checked before claims are recorded
- **Committed:** yes — so entry-28 tests need no network
- **sha256:** `131e4653b229c732ff8f57bb3a091b6dc1f5d008f7e9bf63c2e43d3e8f7a5233` (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; fixed genre order,
  sentence-parity split, sorted keys, no RNG, no timestamp
- **Normalization applied:** each stream is a letter-alphabet string (the same
  30-symbol ALPHABET the Stage 1 level uses; see `ailm/levels/letter.py`),
  bounded to fixed character counts recorded in provenance

### CMUdict derived artifact — `vendor/cmudict_counts.json` (Stage 2)

- **Role:** all Stage 2 inputs — phoneme unigram/bigram counts and the G2P
  length-matched pairs
- **Derived from:** `vendor/cmudict.dict` (not committed; see the CMUdict entry
  above) via `scripts/generate_phoneme_counts.py`; the generator rejects raw
  bytes that do not match the pinned revision's digest and extent
- **Committed:** yes — so Stage 2 tests need no network. The manuscript reports
  its current metric-kB size from the file-backed `META_CMUDICT_COUNTS_KB` token
  instead of copying a size literal into this document
- **sha256:** `9f0f20746086e378b54f5cef82792c9d99f51c80a00b65a00bd783ef917d3def`
  (verified against the file on disk by
  `tests/test_provenance_checksums.py` — a documented checksum that has drifted
  from the artifact is worse than no checksum, so it is enforced rather than
  trusted)
- **Determinism:** verified byte-identical on regeneration; keys sorted, no RNG,
  no timestamps. The hash above is the reproduced value
- **Extent:** derived from claim:cmudict_base_word_count CMUdict base words and
  claim:cmudict_pronunciation_entry_count pronunciation entries → claim:phoneme_count_39
  phones, the complete distinct-bigram map stored under `phone_bigrams`, and
  claim:stage2_g2p_train_pairs train + claim:stage2_g2p_test_pairs test G2P pairs
- **Derivation choices, each of which biases the data:**
  - primary pronunciation only (first listed); alternates `WORD(2)` discarded
  - stress digits stripped, giving the canonical claim:phoneme_count_39 inventory
  - G2P pairs restricted to words whose letter count equals their phone count —
    **claim:stage2_g2p_length_matched_pct% of the dictionary.** This subset is
    biased toward short, transparent-orthography words and is *not* a random
    sample of English. See [`../design/stage2_g2p.md`](../design/stage2_g2p.md)
  - train/test split is by sorted-index stride stored as `test_every`, so it is
    reproducible from the artifact with no seed to record
- **Non-canonical phone guard:** the generator raises rather than vendoring an
  inventory containing any phone outside `CANONICAL_ARPABET`

### Synthetic derived artifact — `vendor/synthetic_counts.json` (stipulated-language exact-artifact-overlap control)

- **Role:** the training corpus for the stipulated-language letter probe. AILM's
  order-1 comparator is estimated from it; every registered external probe meets
  the same independently seeded evaluation trajectories with no separately
  labeled demonstration in the producer arm
- **Derived from:** *no corpus at all* — this is the one vendored artifact with no
  external source. The language is stipulated in `src/ailm/levels/synthetic.py` as
  an explicit column-stochastic transition (a vowel/consonant alternation, a
  designed dominant-successor map with no English counterpart, and a uniform
  floor), sampled by `scripts/generate_synthetic_counts.py`
- **Why it exists:** Brown and CMUdict are longstanding, widely distributed
  resources and may sit inside a pretrained model's training data, so the
  external benchmark on them cannot separate inference from recall. This
  exposure concern does not alter Brown's upstream non-commercial-use terms
  documented above. A locally generated stipulated artifact reduces exact
  overlap as an explanation; it does not establish any external model's
  pretraining contents or make the resulting system contrast causal
- **Committed:** yes — so the probe's tests need no network and no regeneration
  step; its byte extent is measured from disk where needed
- **sha256:** `32b00fa2fb3b818df7b509267ec70efb9418867eb7af7e4779069d25a9c78bb3`
  (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; fixed seed, one bulk
  uniform draw walked against precomputed cumulative columns rather than
  `Generator.choice`, keys sorted in ALPHABET order, no timestamp. The hash above
  is the reproduced value, not a one-off recording
- **Extent:** claim:bench_synth_corpus_chars sampled symbols; every possible
  bigram over the claim:bench_synth_alphabet_size symbol stipulated inventory is
  populated, with one fewer bigram token than sampled symbols
- **Shape:** a flat `{"<a>,<b>": count}` mapping matching the numerical
  `counts` object inside `brown_counts.json` over the same
  claim:stage0_letter_alphabet_size symbol ALPHABET,
  so `ailm.levels.letter.build_letter_transition` consumes either mapping with no
  numerical adapter. Brown additionally has a raw-source envelope because it
  derives from an external corpus. Only the claim:bench_synth_alphabet_size
  letters occur in the synthetic
  mapping; space, period, apostrophe and hyphen never appear
- **Evaluation trajectories:** not vendored. The v2 protocols generate one
  independently seeded trajectory per scored item through
  `ailm.levels.synthetic.independent_held_out_positions`; the recorded base seed,
  trajectory count, trajectory length, start-marginal TVD, and generator name
  make that population reconstructible without a bulk held-out stream
- **Regeneration caveat:** changing `--chars` or `--seed` changes the hash; update
  this entry in the same change

### Synthetic derived artifact — `vendor/synthetic_phone_counts.json` (stipulated-language exact-artifact-overlap control, phone arm)

- **Role:** the training corpus for the stipulated-language phone probe,
  vendored as `vendor/synthetic_phone_counts.json`. AILM's order-1 phone
  comparator is estimated from it; every registered external probe meets the
  same independently seeded evaluation trajectories with no separately
  labeled demonstration in the producer arm
- **Derived from:** *no corpus at all* — like the letter artifact above, this
  artifact has no external source. The language is stipulated in
  `src/ailm/levels/synthetic.py` as an explicit column-stochastic transition
  over the canonical `claim:phoneme_count_39` ARPAbet inventory (a stipulated
  vowel/consonant class alternation, a designed dominant-successor map with no
  English counterpart, and a uniform floor), sampled by
  `scripts/generate_synthetic_phone_counts.py`
- **Why it exists:** CMUdict is a longstanding, widely distributed resource
  and may sit inside a pretrained model's training data, so the external
  benchmark on it cannot separate inference from recall. A locally generated
  stipulated artifact reduces exact overlap as an explanation; it does not
  establish any external model's pretraining contents or make the resulting
  system contrast causal. The stipulated *transition* is the new object; the
  symbols are the canonical inventory a model may have met elsewhere, and
  this entry claims no absence of that exposure
- **Committed:** yes — so the probe's tests need no network and no
  regeneration step
- **sha256:** `319b3926e8a811ef8842602824c122905b11187d481891cdab42500a9116cc41`
  (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical on regeneration; fixed seed, one
  bulk uniform draw walked against precomputed cumulative columns rather than
  `Generator.choice`, keys sorted in inventory order, no timestamp. The hash
  above is the reproduced value, not a one-off recording
- **Shape:** a flat `{"<prev>,<next>": count}` mapping matching the key format
  of the `phone_bigrams` mapping inside `vendor/cmudict_counts.json`, so
  `ailm.levels.phoneme.build_phone_transition` consumes either with no
  numerical adapter (add-1 smoothing, identical default). Only canonical
  ARPAbet symbols occur in the mapping
- **Evaluation trajectories:** not vendored. The phone protocol generates one
  independently seeded trajectory per scored item through
  `ailm.levels.synthetic.independent_phone_held_out_positions`; the recorded
  base seed, trajectory count, trajectory length, start-marginal TVD, and
  generator name make that population reconstructible without a bulk held-out
  stream
- **Regeneration caveat:** changing `--symbols` or `--seed` changes the hash;
  update this entry in the same change


### Synthetic derived artifact — `vendor/synthetic_aligned_g2p_counts.json` (stipulated aligned spelling-to-pronunciation, protocol P2)

- **Role:** the pair-structured stipulated artifact for the aligned
  spelling-to-pronunciation arm: a seeded per-letter correspondence from the 26
  lowercase letters onto 13 of the canonical claim:phoneme_count_39 phones,
  and train/test splits of fixed-length (L = 6) spelling words whose
  pronunciations are the correspondence applied per position. Structurally
  distinct from the two order-one continuation chains above
- **Derived from:** *no corpus at all* — the same stipulated family as the two
  artifacts above. The letter trajectories come from the shipped
  `ailm.levels.synthetic.generate_sequence` chain (one word per trajectory,
  consecutive integer seeds, the fixed start symbol dropped); the
  correspondence is drawn by `scripts/generate_synthetic_aligned_g2p.py`
  through `ailm.benchmarks.aligned_g2p.stipulated_correspondence`; the
  arm's native model is the shipped G2P engine refit on the train pairs
- **Why it exists:** protocol P2 extends the stipulated design to aligned
  pronunciation (the TODO's second open item). Within-split duplicates and
  cross-split collisions are refused at generation (64-attempt resample cap,
  then a no-claims refusal); the disjointness is set-theoretic and re-asserted
  from the bytes on every load. Pretraining-exposure uncertainty remains the
  standing limitation of the stipulated family, and the artifact is never
  called contamination-free
- **Committed:** yes — so the arm's offline tests and evaluator need no
  network and no regeneration step
- **sha256:** `a6bdd76c4ced420661ab3dd1c2e772910604630e3227504e9dfd22dfb7d715c1`
  (enforced by `tests/test_provenance_checksums.py`)
- **Determinism:** verified byte-identical across consecutive runs; fixed
  seeds (train base 20261300, held-out base 20261540, correspondence seed
  20260912, broken-map seed 20260913), sorted canonical digests, no timestamp
- **Shape:** `schema` `ailm-synthetic-aligned-g2p-v1`; `provenance` (chain and
  correspondence versions, seeds, word length, resample cap, split sizes,
  per-split word-set digests, correspondence-table digest, the declared
  smoothing/unseen/decoder policies, and the exposure limitation);
  `correspondence` (complete over 26 letters, codomain 13 of 39);
  `splits.train` / `splits.test` as `[spelling, space-joined phones]` pairs
- **Recorded in:** `ailm.benchmarks.synthetic_protocol::SYNTHETIC_ALIGNED_G2P_PROTOCOL`
  (fourth registry member; distinct cache protocol)
- **Regeneration caveat:** changing any protocol constant (word length, seed
  domains, correspondence seed, codomain size) changes the hash; update this
  entry in the same change

### Case-word bridge artifact — `data/case_word/` (protocol P3, development-session)

- **Source:** the same authenticated `en_ewt-ud-dev.conllu` the shipped case
  artifact reads, at the same pinned revision, through
  `ailm.corpus.case_word.load_case_word_artifact`
- **Generator:** `scripts/generate_case_word_counts.py`
- **License:** CC BY-SA 4.0, inherited from the source treebank
- **Role:** the sender-side fitting artifact of the L4→L3 case-word bridge
  (protocol P3): one `(word_form, upos, case_label_or_null, label_source)` row
  per EWT dev token. The oracle case label is a sender-side fitting input
  recorded with the route that assigned it; it is structurally unavailable to
  any model observation, whose side reads UPOS only. Decision 121's influence
  sweep records `case_word_sweep_*` claims over the bridge's L3 word-prior
  inlet (`scripts/record_case_word_influence_claims.py`); the artifact itself
  remains the sender-side fitting input
- **Why it lives under `data/` and not `vendor/`:** a development-session
  bridge input that retains its canonical source path. The connected runtime
  now consumes its train-only sender map, so it is registered in
  `ailm.resources.RUNTIME_DATA_ARTIFACTS` and packaged into the wheel under
  `ailm/_data/case_word/ud_ewt_case_word_counts.json`. Its inclusion does not
  change the development-session status or the fitting/evaluation split; it
  remains distinct from `ailm.resources.RUNTIME_VENDOR_ARTIFACTS`.
- **Split:** the same document-index stride over the same authenticated
  `# newdoc id` values as `vendor/ud_ewt_case_counts.json`, cross-validated
  against its recorded split indices at build time and at every load
- **Shape:** `schema` `ailm-case-word-counts-v1`; `provenance` (source, split,
  license, label routes); `documents` (per-document token rows);
  `vocab` (word forms); `populations` (train/test token and route counts);
  `sender_map` (the train-only role-to-word message the bridge fits);
  `route_stats` (per-oracle-route coverage and agreement); `digests`
- **Artifact sha256 (`data/case_word/ud_ewt_case_word_counts.json`):**
  `64ddd48465745a23b9beda4733670897192b3d95efe1e613c7e66fd65d8b4c90`
- **Determinism:** verified byte-identical on regeneration; the artifact is
  committed, so the offline tests and the structural evaluation need no
  network and no regeneration step
- **Regeneration caveat:** changing the split stride, the label-route
  composition, or the source revision changes the hash; update this entry in
  the same change

## UD English GUM connected-evaluation input

- **Upstream revision:** `b58e74bc22d17220c9198864c253a50a897bf27f` (UD `r2.18`).
- **Source:** [pinned test file](https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/b58e74bc22d17220c9198864c253a50a897bf27f/en_gum-ud-test.conllu).
- **License:** CC BY-NC-SA 4.0, verified against the
  [pinned upstream notice](https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/b58e74bc22d17220c9198864c253a50a897bf27f/LICENSE.txt); source-text and annotator credit remain required.
- **Artifact sha256 (`vendor/en_gum-ud-test.conllu`):**
  `cd96a285e7339f401f4803dd0f4f61109c51692e3ef6f9186a1f977544080b69`.
- **Identity and analysis contract:** `data/connected_gum/input_manifest.json`
  preserves the frozen input identities; `data/connected_gum/protocol.json`
  preserves the locally specified analysis and the previously inspected document
  exclusion. These records do not constitute an external registration.
- **Observation:** raw `# text` comments joined in document order, normalized by
  the connected runtime; UPOS, dependency, case, phoneme and sentence-type labels
  do not enter inference. Source sentence boundaries are not separately supplied.
- **Runtime boundary:** this is an evaluation input, not a fitting input for the
  shipped bridge kernels and not a resource included in the wheel.
- **Measured-output binding:** `scripts/record_connected_gum_claims.py` writes
  `output/reports/connected_gum.json` and the `connected_gum_*` claims from a
  complete run over the authenticated retained documents. The report carries
  per-document paired losses and the registered bootstrap. The shared
  `ailm.verification.connected_provenance.canonical_execution_binding` authenticates
  the complete source/model-input inventory; the figure reader also checks the
  ordered document population against the frozen raw input. An updated hash is
  not a substitute for re-executing a changed measurement.
- **Interpretation:** the named-channel ablation keeps observed spelling and the
  train-fitted postword UPOS encoder. The [connected contract](../design/connected_runtime.md#recorded-connected-evaluation)
  and [results](../manuscript/04_results.md#sec:connected_gum) preserve the aggregate
  intervention, normalized character stream, vocabulary coverage and infinite
  full-stream closed-vocabulary word loss.

## pymdp

- **Package:** `inferactively-pymdp`
- **Version pinned:** 1.0.4
- **Repository:** https://github.com/infer-actively/pymdp
- **Role:** The ONLY generative model in AILM
- **License:** MIT

## Design Decision Sources

| Decision | Documented in |
|----------|--------------|
| CEREBRUM source tuple and AILM-owned pymdp heuristic | `docs/design/cerebrum_mapping.md` |
| Categorical boundary and no-embeddings invariant | `docs/design/no_embeddings.md` |
| Generation via policy selection | `docs/design/generation_problem.md` |
| Speech boundary (DSP/pymdp) | `docs/design/speech_boundary.md` |
| pymdp API surface required | `src/ailm/generative/README.md` |
| Hierarchical architecture | `docs/design/architecture.md` |
