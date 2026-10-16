# ailm/levels/ — the pymdp agents

One module per level of the linguistic hierarchy. Each is a real
`pymdp.agent.Agent` where the level is a generative model — no transformers, no
neural nets, no embeddings. Levels 1–6 are implemented and gated; Level 7 is an
interface boundary rather than a shipped generative model.

### The shipped level agents

| Module | Level | Hidden factor | Observation modality | Stage |
| --- | --- | --- | --- | --- |
| `letter.py` | 1 | current letter (30 states) | current letter (30 obs) | ✅ 1 |
| `phoneme.py` | 2 | current phone (39 states) | current phone (Part A) / current letter (Part B) | ✅ 2 |
| `word.py` | 3 | word identity × position in word | letter path | ✅ 3 |
| `case.py` | 4 | case role (8) | UPOS (derived oracle is an evaluation label) | ✅ 4, agreement reported |
| `clause.py` | 5, retained ablation | joint sentence type × case role | case role | non-passing case-only gate |
| `widened.py` | 5, canonical | joint sentence type × widened symbol | case role, verb, or auxiliary | descriptive development gate passes; no held-back replication |
| `narrative.py` | 6 | document-position tercile | sentence type | positive point estimate; development primary spans zero; held-back primary excludes zero only at uncorrected coverage |

Level 7 is an interface boundary, not an agent, and its contract module lives one
directory up as `../speech_boundary.py`.

### The channels, experiments, and shared pieces

| Module | What it is |
| --- | --- |
| `connected_evaluation.py` | Paired GUM document estimands and bootstrap under the frozen local protocol |
| `connected.py` | Causal multirate inference and live-policy conditional next-word generation; see the [runtime contract](../../../docs/design/connected_runtime.md) |
| `connected_builder.py` | GNN-built agent assembly and train-only bridge kernels |
| `connected_types.py` | Typed observations, channel interventions and inference/policy traces |
| `_stochastic.py` | The shared column-stochastic normalizer and validator every level builds `A` and `B` through |
| `coupling.py` | The L1→L2B message channel. Measured harmful and shipped gated off (`alpha` defaults to `0`); kept because a deleted negative result is not a result |
| `lexical_channel.py` | The L3→L2B lexical prefix message into the G2P prior, and the novel-information criterion of decision 18 |
| `clause_channel.py` | The privileged punctuation-type-oracle → L3 word-prior arm of decision 27 |
| `g2p_learning.py` | The latent-state G2P emission-learning experiment of decision 29, with its same-lane observed-state control |
| `online_learning.py` | Online parameter learning on non-stationary text (decision 28) |
| `widened.py` | The widened Level 5 observation alphabet — case roles plus verb and auxiliary — defined once and imported by every consumer |
| `case_czech.py` | The Czech arm of Level 4 (decision 77): the same model shape scored where every label is real UD morphology |
| `synthetic.py` | The stipulated synthetic letter language, an exact-artifact-overlap control for the external benchmarks |

## What each module gives you

### `letter.py`

`ALPHABET` (exactly 30 symbols: `a`–`z`, space, period, apostrophe, hyphen),
`ALPHA_LIST` / `ALPHA_TO_IDX` / `IDX_TO_ALPHA` for deterministic indexing,
`normalize_text`, `train_on_corpus`, `build_letter_transition`, `build_letter_agent`,
`compute_perplexity`, `load_brown_counts`.

### `phoneme.py`

Two agents in one module, because Stage 2 is where the levels stop being the same
shape:

- **Part A — the phone sequence model.** `build_phone_transition`,
  `build_phone_unigram`, `build_phoneme_agent`, `compute_phoneme_perplexity`, plus
  the closed-form full-dictionary forms `full_corpus_perplexity` and
  `full_corpus_unigram_perplexity`. `A` is the identity; `B` is the phone bigram.
- **Part B — the grapheme-to-phoneme model.** `g2p_split`, `build_g2p_emission`,
  `build_g2p_transition`, `build_g2p_start`, `build_g2p_agent`, `decode_phonemes`,
  `decode_positions`, `g2p_accuracy`, and the context-free comparator
  `per_letter_argmax_baseline` / `baseline_positions` / `baseline_accuracy`. Here `A`
  is a *real* emission — 30 letter observations over 39 hidden phones — so unlike
  Part A it cannot be mistaken for a transition by shape alone. The separate
  `g2p_learning.py` experiment tests Dirichlet emission learning with phone states
  unobserved; its qualified negative result and same-lane observed-state control are
  recorded under decision 29.

### `coupling.py`

The measured L1→L2B intervention: `couple_letter_to_phoneme`
mixes L1's predictive over letters with the observed glyph at weight `alpha` and
marginalizes it through the position-aligned letter-to-phone (G2P) emission. `sweep_alpha`, `validate_alpha_grid`,
`coupled_positions` / `uncoupled_positions`, and the paired `mcnemar` are the
measurement apparatus around it.

Its default is **uncoupled**: `alpha=0` reproduces the shipped decode, with
claim:coupling_alpha0_identity_diffs observed differences. The best nonzero arm
on the recorded grid has a margin of claim:coupling_margin_pts percentage points
against that default, with paired McNemar p = claim:coupling_mcnemar_p. This
outcome concerns the recorded protocol and grid. The other measured receiver
interventions are described in [the hierarchy guide](../../../docs/hierarchy.md);
they are separate experiments and do not establish an end-to-end cascade.

## Quick start

```python
from ailm.levels.letter import (
    build_letter_agent, build_letter_transition, compute_perplexity, load_brown_counts,
)

agent = build_letter_agent(build_letter_transition(load_brown_counts()))
print(compute_perplexity(agent, "the quick brown fox jumps over the lazy dog."))
```

```python
from ailm.levels.phoneme import (
    build_g2p_agent, build_g2p_emission, build_g2p_start, build_g2p_transition,
    decode_phonemes, g2p_split, load_phoneme_counts,
)

train, test = g2p_split(load_phoneme_counts())
agent = build_g2p_agent(
    build_g2p_emission(train), build_g2p_transition(train), build_g2p_start(train)
)
print(decode_phonemes(agent, "cat"))
```

## The three invariants a newcomer would otherwise violate

### 1. `B` carries the dynamics; `A` is the emission

This is the load-bearing modeling decision in the whole project, and Stage 1
shipped it backwards. Putting the learned bigram matrix in the emission `A` with an
identity `B` is *anti-informative*, not merely weak: observing symbol `c` yields a
posterior over the symbol that **preceded** `c`, and an identity `B` carries that
backward-looking belief forward as the prediction for the next symbol.

Measured, on held-out prose: claim:stage1_anti_informative_perplexity for the
mis-wiring against claim:stage1_letter_perplexity for correct wiring — worse than
the context-free claim:stage1_unigram_baseline, i.e. worse than using no context at
all. On a small in-sample corpus it scored worse than uniform and *still* slipped
under a naive threshold.

**pymdp cannot catch this.** A bigram matrix is column-stochastic, so
`Agent._validate()` accepts it in either slot. Orientation is pinned by numerical
agreement with `ailm.verification.oracle`, never by structure.

The G2P model offers the same trap through emission normalization. An array
normalized over phones and then renormalized to satisfy `pymdp` can look
structurally valid and outperform a comparator while representing the wrong
conditional distribution. The [Methods](../../../docs/manuscript/03_methods.md)
and `tests/test_stage2_gate.py` retain this control and its interpretation.

Beating the baseline is therefore necessary but not sufficient. What actually pins
the emission down is the column-sum invariant on `A` — `A[0, :, p]` must sum to 1
*before* any renormalization — plus numerical agreement with a closed-form oracle.

### 2. The prior resets at every sequence boundary

`compute_phoneme_perplexity` resets the empirical prior to `D` at the start of each
word, because a word boundary is not a phoneme transition. Position 0 has no
predecessor, so it is paid for under the unigram start prior. Treating the dictionary
as one continuous phone stream — each word's first phone predicted from the previous
word's last phone — measures *worse* than paying the unigram price, so the reset is
correct rather than merely convenient. Log probabilities accumulate across all
positions in all sequences and are divided by the total, giving a corpus-level
per-phone perplexity rather than a mean of per-word perplexities.

### 3. `limit=` is an evenly spaced stride, never a prefix

`_iter_pairs` chooses `pairs[i * len(pairs) // limit]` across the complete
list. It does not take a prefix or take a stride and truncate its tail. In the
recorded bias diagnostic, the alphabetical prefix reaches
claim:subsample_prefix_initials initials and reports a margin of
claim:subsample_prefix_margin_pts percentage points, against
claim:stage2_g2p_margin_pts on the full held-out set: an overstatement by a factor
of claim:subsample_prefix_overstatement_factor. The recorded
stride-five diagnostic spans claim:subsample_stride5_initials initials but its
paired p = claim:subsample_stride5_mcnemar_p does not cross the family threshold.
These diagnostics are distinct from the canonical evenly spaced index rule.

Use `limit=` for smoke checks. Compute the registered gate statistic on all
claim:stage2_g2p_eval_positions positions through the independent oracle after
verifying there are claim:stage2_decoder_disagreements agent–oracle disagreements. The full paired result is p = claim:stage2_g2p_mcnemar_p,
b = claim:stage2_g2p_mcnemar_b, c = claim:stage2_g2p_mcnemar_c. Reproducibility of
a subset does not make it representative or sufficiently powered.

The same bias recurred for Part A: on the length-matched G2P subset the phoneme
perplexity is claim:stage2_phoneme_subset_perplexity against
claim:stage2_phoneme_perplexity over the full dictionary, which is why
`full_corpus_perplexity` computes it in closed form over all
claim:stage2_phoneme_eval_positions positions instead of walking the subset.

## Verification

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -m pytest \
  tests/test_letter.py tests/test_phoneme.py tests/test_phoneme_full_corpus.py \
  tests/test_stage1_gate.py tests/test_stage2_gate.py -q --no-cov
```

## More information

[`AGENTS.md`](AGENTS.md) for boundaries and gotchas. Level hierarchy:
`../../../docs/design/architecture.md`. G2P scoping and its costs:
`../../../docs/design/stage2_g2p.md`.
