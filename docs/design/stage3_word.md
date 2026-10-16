# Stage 3 — The Lexical Level

Why the word level is a *bounded* closed-vocabulary model over two factors rather
than the explicit trie-node state space considered in the historical design,
what that costs, and what pins it. This page describes the standalone Stage 3
score; [connected execution](connected_runtime.md) uses live L3 updates and
reports OOV continuation under a separate scoring contract.

Decision records: [`../decisions.md`](../decisions.md) entries 13 and 14.
Gate procedure: [`../gate_design.md`](../gate_design.md).

- [The trie does not fit](#the-trie-does-not-fit)
- [Word identity times position](#word-identity-times-position)
- [The trie is induced, not abandoned](#the-trie-is-induced-not-abandoned)
- [Why the closed vocabulary is a scoping decision](#why-the-closed-vocabulary-is-a-scoping-decision)
- [The gate, and why the bar is not enough](#the-gate-and-why-the-bar-is-not-enough)
- [Two comparisons never to make](#two-comparisons-never-to-make)

---

## The trie does not fit

The historical proposal represented words as paths through a letter–phoneme
trie, with explicit trie nodes as hidden states and trie-constrained transitions. Measured over the
length-matched position-aligned letter-to-phone (G2P) vocabulary:

| quantity | value |
|---|---|
| trie nodes | claim:stage3_trie_nodes_g2p |
| trie edges | claim:stage3_trie_edges_g2p |
| dense `B` if nodes are states | **claim:stage3_trie_dense_gb GB** float32 |
| density | **claim:stage3_trie_density_pct%** of dense |
| extrapolated to full CMUdict | ~claim:stage3_trie_dense_full_tb TB |

The shipped Agent builders use dense `jnp` arrays for `A` and `B`; this project
does not use a sparse Agent-construction path. Explicit trie-node states at the
reported extent therefore do not fit the adopted dense representation. This is
not a claim that pymdp's backend contains no sparse contraction machinery. The
distinction matters: the response to a performance problem is
tuning, and the response to a representation that does not fit is a design
decision recorded with its rejected alternatives.

Bounding the vocabulary *and* keeping trie nodes as states does fit, and was
measured rather than dismissed:

| vocabulary | trie nodes | dense `B` | word-factor `B` | held-out coverage |
|---|---|---|---|---|
| 1,000 | claim:stage3_trie_nodes_v1000 | claim:stage3_trie_dense_v1000_mb MB | claim:stage3_factored_v1000_mb MB | claim:stage3_scale_1000_coverage_pct% |
| 2,000 | claim:stage3_trie_nodes_v2000 | claim:stage3_trie_dense_v2000_mb MB | claim:stage3_factored_v2000_mb MB | claim:stage3_scale_2000_coverage_pct% |
| 4,000 | claim:stage3_trie_nodes_v4000 | claim:stage3_trie_dense_v4000_mb MB | claim:stage3_factored_v4000_mb MB | claim:stage3_scale_4000_coverage_pct% |
| 8,000 | claim:stage3_trie_nodes_v8000 | claim:stage3_trie_dense_v8000_mb MB | claim:stage3_factored_v8000_mb MB | claim:stage3_scale_8000_coverage_pct% |

This table was a second typed copy of the one in `decisions.md` entry 13, and it
had drifted the same way: none of its four node counts survived a rebuild
against the shipped artifact, and two of its four coverage figures did not round
from the ledger values they summarize. Both copies are now ledger references.

The trie encoding costs an order of magnitude more **under dense storage** for
the *same* vocabulary and the *same* coverage, because it spends states on
prefixes rather than on words. Under nonzero-count pricing the gap is much
smaller, which is decision 58's point and does not rescue the encoding: the trie
still hides word identity.

## Word identity times position

The shipped factorization:

| | factor 0 | factor 1 |
|---|---|---|
| meaning | word identity | position within the word |
| size | claim:stage3_vocab_size | configured length horizon plus end state (claim:complexity_l3_position_states states; observed longest retained word claim:stage3_observed_max_word_len letters) |
| `B` | the **identity** | the **+1 shift**, absorbing |
| `D` | **learned** word unigram frequencies | point mass on position 0 |

with one letter modality depending on both through `A_dependencies = [[0, 1]]`.
`A[k, w, p]` is 1 exactly when letter `p` of word `w` is symbol `k`, or when `k`
is the end-of-word symbol and `p` has passed the word's last letter.

**Where the learned content sits changes at this level.** At L1 and L2 the corpus
statistics parameterize the transition. Here both transitions are structural
constants and the single learned quantity is `D` over words. That is the level's
whole content — a lexicon with frequencies — and it relocates the mis-wiring risk
with it.

Because every factor except `D` is deterministic, the probability of a word's
full emission path collapses to `D[w]` exactly. That identity is what pins the
level, and `ailm.verification.oracle.word_path_logprob` derives it without
importing pymdp or jax.

## The trie is induced, not abandoned

After observing a prefix, the posterior over factor 0 is supported on exactly the
vocabulary words carrying that prefix — which *is* the trie node. The two
formulations are the same model in different coordinates.

This is checked rather than asserted: `test_posterior_support_is_the_trie_node`
compares the support against the set of words with each prefix, and
`test_the_trie_correspondence_is_not_vacuous` confirms the support genuinely
narrows as the prefix lengthens, so the equality cannot pass by returning
everything.

## Why the closed vocabulary is a scoping decision

The model assigns an out-of-vocabulary word **no** probability — not a small one.
claim:stage3_coverage_pct% of held-out alphabetic Brown tokens are in vocabulary,
and every Stage 3 number describes that sublanguage.

The standalone Stage 3 scorers **raise** on an out-of-vocabulary token rather than smoothing, because
a fallback would report a number for an input the model is undefined on. The
split is by **whole Brown source document, not by word type**: a type-level split
would put every evaluation token outside a closed-vocabulary model's support and
would measure the closed-vocabulary assumption rather than the model. A
revision-bound hash selects the nearest whole-document held-out count within
every Brown category, putting
claim:stage3_train_documents authenticated documents in training and
claim:stage3_heldout_documents in evaluation, with
claim:stage3_document_overlap overlap. All claim:stage3_source_categories source
categories contribute: claim:stage3_train_categories are represented in training
and claim:stage3_heldout_categories in evaluation.
Ordered file identities, category assignments, their framed digest, an
exhaustive partition, and per-token document indices are carried in both Stage 3
artifacts and validated at load time.

Stage 3 is measured on Brown, not the PTB the declared gate named. The full
treebank is LDC-licensed and `nltk` ships only a small sample under research terms,
so a number reported against "PTB" would have named a corpus this project does
not have.

## The gate, and why the bar is not enough

The declared "perplexity < 500 on PTB-1k" is **VACUOUS**: the letter-bigram bar
itself is claim:stage3_bigram_baseline, clearing 500 by
claim:stage3_declared_gate_slack times.

The restated gate compares against a letter-bigram model over the identical
alphabet and same whole-document training partition. The bar is fitted to every
training type while the word prior is fitted to its bounded lexicon; both are
scored on identical held-out in-vocabulary positions. This is the strongest
model in the ladder that does not know the lexicon, with broader training
support than the word prior.

| comparator | uses the lexicon? | per-emission perplexity |
|---|---|---|
| uniform over 27 symbols | no | claim:stage3_uniform_baseline |
| context-free letter unigram | no | claim:stage3_unigram_baseline |
| **letter bigram — the bar** | no | **claim:stage3_bigram_baseline** |
| **word × position (this level)** | yes | **claim:stage3_word_perplexity** |
| held-out-oracle word model — ceiling | yes | claim:stage3_ceiling_perplexity |

The level captures claim:stage3_signal_captured_pct% of the available signal. The
margin is wide by construction, so unlike Stage 2 no paired test is needed — a
measured conclusion, not an omission.

**Three of five mis-wirings clear the bar:**

| mis-wiring | per-emission perplexity | caught by the bar? |
|---|---|---|
| uniform `D` — lexicon kept, frequencies discarded | claim:stage3_miswire_uniform_prior | **no** |
| `D` from word length, not frequency | claim:stage3_miswire_length_prior | **no** |
| frequency ranking reversed | claim:stage3_miswire_reversed_prior | **no** |
| time-reversed emission | claim:stage3_miswire_time_reversed | yes |
| end-of-word emitted one position early | infinite — assigns zero to every true path | yes |

The last row is caught for free: a model that assigns probability zero to the
observed path has no finite perplexity, so it needs no bar to reject it. That
leaves the time-reversed emission as the only mis-wiring the bar catches on its
own merits.

The third is the striking one: inverting the ranking so the *rarest* words are
predicted most confidently still beats a letter model. A gate whose only guard is
`assert model < bar` admits a model whose word prior is upside down.

The general form: **when a level's mechanism is very informative, the baseline
comparison stops discriminating precisely because the level is good.** Necessary
conditions get weaker as models get stronger. What pins this level is the path
identity — all three survivors change `D`, so all three fail it.

## Two comparisons never to make

1. **Stage 3 against Stage 1 perplexity.** 27 symbols here (26 letters + END)
   versus 30 running-text symbols there, over different position sets. Different
   quantities. `test_stage3_perplexity_is_not_comparable_with_stage1` asserts the
   alphabets differ so the warning is enforced rather than remembered.
2. **Word-level against path-level perplexity.** These differ by orders of
   magnitude. Everything here is **per emission**, and any figure quoting a Stage
   3 perplexity must say so.
