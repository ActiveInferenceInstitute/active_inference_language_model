# Data statement

A data statement in the sense of Bender and Friedman: what each corpus is, who
produced it, what was done to it here, and — the part that matters most for this
project — **what population a result measured on it can be about**.

`provenance/sources.md` already records URLs, licenses, checksums and
regeneration commands. That answers *can this be rebuilt*. This file answers
*what does a number computed on it generalize to*, which is a different question
and the one the manuscript's scope claims depend on.

The English ladder uses different populations at different levels: Brown-derived
statistics, an authored evaluation fixture, a pronunciation lexicon, and a web
treebank. The exploratory Czech case arm and the stipulated letter and phone
controls have separate scope boundaries below. None is a representative sample
of language as a whole. The count artifacts do not retain producer demographic
profiles, so this statement does not infer demographic representativeness from
a corpus's language or genre label.

---

## Brown Corpus — Levels 1 and 3

| | |
|---|---|
| **Curation rationale** | Assembled as a balanced sample of published American English prose, sampled across fifteen genres |
| **Language variety** | American English, edited and published register |
| **Era** | Texts published in 1961 |
| **Producers** | Professional authors and editors, not speakers recorded in interaction |
| **Speech situation** | Written, planned, edited, non-interactive |
| **Text characteristics** | Prose across press, religion, skill and hobbies, popular lore, belles-lettres, government, learned, and fiction genres |
| **Access here** | NLTK `brown`; NLTK metadata says "May be used for non-commercial purposes" ([license summary](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md), [package index](https://github.com/nltk/nltk_data/blob/gh-pages/index.xml)) |
| **Extent used** | The deterministic normalized artifact records claim:stage1_brown_bigram_tokens bigram transitions spanning claim:stage1_brown_bigram_types distinct bigram types |

**What was done to it.** Lowercased; every character outside the declared
alphabet mapped to space; whitespace runs collapsed; spaces before `.` and `'`
stripped. Those last two are not cosmetic: Brown's tokenizer emits punctuation as
separate tokens, so without them the space-space bigram dominates every other and
no letter ever transitions into a period.

**What a result on it is about.** Character and word statistics of *edited
American prose published in 1961*. It is not about contemporary text, spoken
language, informal writing, or any other variety. A letter bigram fitted here
describes the orthographic habits of that register, and the manuscript's
perplexity figures should be read as statements about it rather than about
English.

**Who is absent.** Speakers of other varieties, unedited writing, and every genre
the 1961 sampling frame did not include. The corpus was assembled to be balanced
within its own frame, which is not the same as representative of English.

### Stage 1's evaluation fixture is not Brown

The Stage 1 perplexity gate scores a separately owned protocol fixture,
`ailm.evaluation.STAGE1_EVALUATION`, rather than a Brown split. It is a short
AILM-curated composite of normalized traditional phrases and an AILM-authored
rhyming sentence, with the base passage repeated `3` times. The canonical object
pins its UTF-8 digest, byte extent, version, source references, and terms, and all
recorders, audits, benchmarks, and tests import that one object.

This population is useful for deterministic regression and model-class-bound
checks. It is not independently sampled, and repeating it does not increase its
linguistic diversity or the number of independent passages. Stage 1's held-out
perplexity therefore describes this fixture only; it is not an estimate of
perplexity on Brown, contemporary English, or English in general. Exact
identity and constituent-source references are in
[`provenance/sources.md`](../provenance/sources.md#stage-1-evaluation-protocol-fixture).

**Data-use notice.** The repository and wheel include counts derived from Brown,
whose upstream NLTK metadata restricts use to non-commercial purposes, and a
Czech-PDTC-derived artifact under CC BY-NC-SA 4.0 according to its official
[UD record](https://universaldependencies.org/treebanks/cs_pdtc/index.html).
Consult the repository
[`LICENSE`](../../LICENSE) and [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)
for the authoritative component inventory and upstream notices; this statement
does not add a legal conclusion beyond those terms.

---

## CMUdict — Level 2

| | |
|---|---|
| **Curation rationale** | A pronunciation lexicon built for speech recognition |
| **Language variety** | General American English pronunciation |
| **Producers** | Compiled and hand-corrected by researchers, not sampled from speakers |
| **Speech situation** | None. Entries are dictionary citation forms, not utterances |
| **Text characteristics** | Orthography paired with ARPAbet phoneme strings; multiple pronunciations per spelling are common |
| **License** | BSD-2-Clause |

**What was done to it.** Comments after `#` are stripped before parsing. Skipping
that step yields claim:phoneme_count_naive_unstripped apparent phonemes rather than the
claim:phoneme_count_39 canonical ones, because comment tokens are captured as
symbols. Stress markers are removed.

**The restriction that matters.** Level 2B evaluates only entries whose spelling
and pronunciation have **equal length**, so that letters and phones can be
co-indexed without solving alignment. That subset is not a random sample of
English: length-matched words skew short, skew orthographically transparent, and
skew away from silent letters, digraphs and vowel clusters.
Only claim:stage2_g2p_length_matched_pct% of the lexicon is length-matched and
kept; the rest is discarded. `through` and `knight` are excluded; `cat`, `stop`
and `plant` are not.

**What a result on it is about.** Position-aligned letter-to-phone decoding on
the transparent sublanguage. It is not grapheme-to-phoneme conversion, which
treats the alignment as latent and variable-length, and the manuscript uses the
longer name throughout for that reason.

**Who is absent.** Every pronunciation variety other than General American, and
within that, most of the orthographically interesting vocabulary.

---

## Universal Dependencies English Web Treebank — Levels 4, 5 and 6

| | |
|---|---|
| **Curation rationale** | A dependency treebank of contemporary web English |
| **Language variety** | Contemporary American English, web registers |
| **Producers** | Authors of weblogs, reviews, emails, newsgroup posts and question-answer text |
| **Speech situation** | Written, largely unedited, sometimes interactive |
| **Annotators** | Trained annotators following Universal Dependencies guidelines |
| **License** | CC BY-SA 4.0 |
| **Splits used** | Document-disjoint fitting and evaluation partitions within upstream `dev`; upstream `test` separately used under decisions 111 and 115, with frozen development fits |

**What was done to it, and the construct boundary.** English EWT marks
morphological `Case` on **pronouns only**, and only for nominative, accusative
and genitive. The remaining roles this project uses are *derived* from dependency
relations and adpositions by a stated mapping. That derivation is a principled
heuristic and not ground truth, which is why the manuscript reports *agreement*
with the derived oracle rather than accuracy against gold, and why the level is
named for the role class rather than for grammatical case.

**The identifiability consequence.** On the morphologically marked subset every
observable part-of-speech label collapses to `PRON`, so the observation variable
carries no class-discriminating variation there. That is a property of the
corpus, not a failure of any particular model, and the manuscript reports it as
such.

**Partition history.** Development artifacts partition authenticated document
identities by a fixed stride before filtering retained sequences. Their internal
`train` and `test` keys therefore both refer to upstream `dev`; they must not be
confused with the separately vendored upstream test split. Decision 111 spent
that previously held-back split once on the eligible frozen-family comparisons.
The one-shot recorder refuses a second scoring run. Decision 115 then specified
the document-clustered Level 6 estimand on that already-spent population; its
deterministic replay uses frozen development fits and remains descriptive. The
canonical widened Level 5 arm has no held-back replication.

**What a result on it is about.** Development and held-back results concern the
retained documents of this treebank, not independent samples of all web English.
Level 6's outer sampling unit is the document; the development analysis includes
claim:cluster_stage6_sign_clusters contributing documents. Sentence observations
within each document are clustered rather than independent replications. Genre
and document length remain plausible confounders of document position. Moving to
the upstream test split does not remove those construct or sampling limits.

**Who is absent.** Every register the web treebank does not sample, every
language other than English, and any variety whose syntax the UD English
guidelines do not describe well.

### The case-word bridge artifact scores a dev-only vocabulary slice

`data/case_word/ud_ewt_case_word_counts.json` is the sender-side fitting and
evaluation artifact of the L4→L3 case-word bridge (protocol P3, scored by
decision 121's influence sweep): one `(word_form, upos, case_label_or_null,
label_source)` row per UD English EWT dev token, partitioned by the same
document-index stride as the shipped case artifact and cross-validated against
its recorded split indices at every load. The oracle case label is a
sender-side fitting input recorded with the route that assigned it; the model
side observes UPOS only.

**What a result on it is about.** The decision-121 estimand scores the L3
word level's per-token log-gain on the artifact's `test` documents, restricted
to word forms matching the Stage 3 receiver vocabulary by lowercasing
exact-string comparison. Its population is therefore the intersection of
retained EWT dev documents with that in-vocabulary slice, in document order
under a whole-document clustered bootstrap. It is not a statement about
English morphological case marking in general, and every out-of-vocabulary
dev token contributes nothing to the scored estimand.

**Who is absent.** The upstream EWT test file: its bytes were spent once under
decision 111 and the artifact's generator never reads them, so no case-word
number can be read as a confirmatory test-split result. The artifact remains a development-session bridge input. It is packaged for
the connected model’s default loader through the code-owned runtime resource
inventory; packaging does not broaden its evaluation population. Out-of-vocabulary
dev tokens remain absent from the historical bridge’s in-vocabulary estimand.

---

## Universal Dependencies English GUM — connected evaluation

GUM collects openly available texts across communicative settings through a
Georgetown University corpus-linguistics course. The pinned upstream treebank
contains diverse written and spoken-source genres; it is not a population sample
of English users. [The upstream overview](https://github.com/UniversalDependencies/UD_English-GUM/blob/b58e74bc22d17220c9198864c253a50a897bf27f/README.md)
provides the collection and annotation history.

The connected evaluation consumes only document-ordered raw text comments and
applies the runtime's lowercase ASCII lexical normalization. It withholds UD
annotation fields, retains out-of-vocabulary tokens, and distinguishes finite
in-vocabulary diagnostics from full-stream closed-vocabulary loss. The frozen
protocol excludes the documents inspected in the earlier bounded development
run. A locally specified fresh-population comparison is not an external
replication, and its scope is this normalized input and fixed engineering
configuration. Document means are paired before uncertainty estimation, so
long documents do not receive token-count weight in the primary contrast. The
reported character loss includes normalized token spaces; the finite
in-vocabulary word diagnostic excludes OOV tokens and therefore cannot replace
the full-stream word loss. The same raw lexical UPOS encoder remains active in
both connected and all-named-channels-disabled arms. Corpus identities and licensing are recorded in
[`sources.md`](../provenance/sources.md#ud-english-gum-connected-evaluation-input).

## Universal Dependencies Czech-PDTC — exploratory Level 4 case arm

The [official treebank description](https://universaldependencies.org/treebanks/cs_pdtc/index.html)
identifies Czech-PDTC as a conversion of the Prague Dependency Treebank —
Consolidated, bringing together original Czech writing, translated text, and
transcribed speech. That upstream breadth does not make AILM's selected fitting
shard representative of the collection. The exact pinned revision and shard
digests are recorded in [`provenance/sources.md`](../provenance/sources.md).

**What was selected.** The artifact fits from `cs_pdtc-ud-train-ft.conllu` and
evaluates on `cs_pdtc-ud-dev.conllu`; its generator never reads the upstream test
shard. It retains case-inflecting nominal parts of speech (`ADJ`, `DET`, `NOUN`,
`NUM`, `PRON`, `PROPN`). Prepositions are excluded because their annotated case
describes government rather than inflection. Only UPOS and morphological
`feats[Case]` enter the token encoding, with authenticated document identities
and a document index per retained sequence. Surface words, lemmas, dependency
heads and relations are absent from the derived artifact.

**What a result on it is about.** This arm evaluates prediction of annotated
morphological case on the retained Czech nominal sequences. It is distinct from
agreement with the English derived-role heuristic. Its
claim:stage4_czech_test_sequences evaluation sequences nest within
claim:stage4_czech_test_documents contributing documents; they are not that many
independent speakers. The annotation is a reference standard, not proof of a
language-universal role system. The arm was prospectively specified within a
development session under decision 77, without an independently timestamped
prior registration, and remains exploratory.

**Who and what remain outside scope.** Excluded parts of speech, unretained
sequences, unsampled varieties, and languages other than this Czech arm are not
evaluated. The artifact has no audio or speaker demographic profiles, even where
its upstream text originated in speech. It establishes no acoustic recognition
or spoken-interaction performance.

---

## Stipulated letter and phone trajectories — external-probe controls

These populations are generated by AILM, not collected from human speakers.
`vendor/synthetic_counts.json` and `vendor/synthetic_phone_counts.json` contain
training counts sampled under fixed seeds from explicitly stipulated transition
matrices. Scored trajectories use independent seeds; their protocol, generator,
and run provenance are recorded separately. The exact artifacts and generators
are inventoried in [`provenance/sources.md`](../provenance/sources.md).

**What a result on them is about.** Prediction under the stipulated transition
and declared prompt protocol. The phone arm uses the canonical ARPAbet inventory
but does not model natural English phonotactics or align pronunciations with
spellings. The letter in-context arm supplies bounded labeled demonstrations and
a deranged-demonstration control; those prompts do not give the external probes
the same fitting procedure as the count-trained AILM comparator.

**Exposure boundary.** A locally generated artifact limits exact-artifact overlap
as an explanation. It neither identifies a probe's pretraining corpus nor proves
that its symbols or component regularities were unseen. The recorded system
contrasts do not isolate a causal effect of architecture or fitting method.

---

## Scope across the populations

The English results concern different written registers and dictionary citation
forms, with a curated fixture for the Stage 1 gate. The Czech extension tests a
particular annotated morphology task; the stipulated controls test constructed
symbol processes. Those additions broaden the questions the repository can
test, but they do not turn the ladder into a representative evaluation of
languages, speakers, or conversation. Generalization must follow each arm's
population, filtering rule, fitting history, and sampling unit.
