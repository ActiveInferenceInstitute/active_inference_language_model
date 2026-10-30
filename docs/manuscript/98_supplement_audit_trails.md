# Supplement A: Gate Audits, Mutation Controls, and Held-Back Replication {#sec:audit_trails}

[@sec:gate_design] states the typed gate taxonomy and how a threshold is placed
and checked; the verdicts that taxonomy produces are reported with each level's
results in [@sec:results]. This supplement is the trail behind those verdicts:
the re-runnable audit script that classifies every declared threshold, the
pre-build checklist the audit experience distilled, the defect findings worked
through stage by stage, the Level 3 mis-wiring catalog with the scaling ladder
behind it, and the per-member detail of the confirmatory replication on the
held-back split.

It sits here rather than in the main text because the reader who wants the
verdicts needs a paragraph, and the reader who wants to re-derive them needs the
trails. The verdicts stay in [@sec:gate_design] and [@sec:results]; the script's
exit conditions, the worked defects, the catalogs, and the replication detail
are here. Every number in both places resolves to the same ledger entry.

## Declared-threshold defects and gate audits {#sec:gate_audit}

### Stage 1: the declared threshold sat beyond the information-theoretic ceiling

The Stage 1 gate was declared as `>={{STAGE1_DECLARED_GATE_NATS}} nats above uniform`. The declaration is
information-theoretic, so it can be checked against the data before any model
exists, and doing so takes one pass over the training counts. The conditional
entropy $H_{\text{train}}(X_t \mid X_{t-1})$ of the Brown bigram distribution caps
the gain any order-1 letter model can realize over uniform *on that
distribution* at {{STAGE1_MAX_BIGRAM_GAIN_NATS}} nats. The declared target
exceeds that ceiling by a factor of {{STAGE1_CEILING_SHORTFALL_RATIO}}. By
[@eq:gate_verdict] the verdict is `UNREACHABLE`.

The population qualifier is not a hedge, and dropping it would contradict
[@sec:background]. What that sentence establishes is that the *fitting*
distribution contains barely more than half the first-order information the gate
demands, so no amount of better smoothing or better implementation supplies the
shortfall from the data the model was given. What it does not establish is anything about the held-out
passage: [@eq:conditional_entropy] bounds order-1 log loss under
$p_{\text{train}}$ and not under the evaluation sample, whose own first-order
optimum can sit either side of it.

The evaluation population is not a second corpus sample. It is the
{{STAGE1_EVALUATION_SOURCE}}: a short normalized passage repeated
{{STAGE1_EVALUATION_REPETITIONS}} times. The complete UTF-8 protocol string is
{{STAGE1_EVALUATION_SIZE_BYTES}} bytes, identified as
`{{STAGE1_EVALUATION_PROTOCOL}}` with SHA-256 prefix
`{{STAGE1_EVALUATION_SHA256_SHORT}}`; its terms are
{{STAGE1_EVALUATION_LICENSE}}. Repetition increases the scoring denominator but
does not create independent linguistic material, so this passage supports a
deterministic protocol check, not a representative-English claim.

The gap that leaves is worth closing rather than conceding, because the same
computation performed on the right population closes it exactly. On the
transition positions, cross-entropy under empirical bigram distribution $q$
decomposes as [@eq:transition_cross_entropy]:

$$
\mathbb{E}_q\!\left[-\log p(X_t \mid X_{t-1})\right]
\;=\; H_q(X_t \mid X_{t-1}) \;+\; \sum_{a} q(a)\,\mathrm{KL}\!\left(q(\cdot \mid a) \,\|\, p(\cdot \mid a)\right),
$$ {#eq:transition_cross_entropy}

and the second term is non-negative for every order-1 conditional $p$. The gate,
however, scores all $T$ characters: the first is predicted from the fixed
uniform $D$, and only the remaining $T-1$ are transitions. Its exact lower bound
on the same all-position mean as `compute_perplexity` is therefore

$$
H^{*}_{\mathrm{all}}
= \frac{\log |\mathcal{A}| + (T-1)H_q(X_t\mid X_{t-1})}{T}.
$$ {#eq:heldout_bound}

For this fixture, {{STAGE1_HELDOUT_CONDITIONAL_ENTROPY_NATS}} nats over
{{STAGE1_HELDOUT_TRANSITIONS}} transitions plus that initial prediction yields
{{STAGE1_HELDOUT_ALL_POSITION_ENTROPY_NATS}} nats over all
{{STAGE1_HELDOUT_POSITIONS}} scored positions. That all-position quantity — not
the transition-only entropy — is the exact lower bound on what any model with
this fixed uniform start and first-order conditional can score. It caps the gain
available over uniform at {{STAGE1_HELDOUT_MAX_GAIN_NATS}} nats, still short of
the {{STAGE1_DECLARED_GATE_NATS}}-nat target. The declared gate is unreachable
on the evaluation population as well as on the training one, and now as an
exact derived bound rather than as a hope.

The estimate's finite-sample bias runs the safe way, which is why so few
transitions can carry the argument. The plug-in estimator of an entropy is
biased *downward* in the undersampled regime, by an amount that grows with the
number of cells relative to the sample [@miller1955note; @paninski2003estimation];
here that is {{STAGE1_HELDOUT_TRANSITIONS}} transitions over an alphabet
admitting ${{STAGE0_LETTER_ALPHABET_SIZE}} \times
{{STAGE0_LETTER_ALPHABET_SIZE}}$ bigrams. A smaller entropy means a *larger* apparent
gain, so the bound above is generous to the gate. A gate that fails against a
generous bound fails, which is the one direction of bias an unreachability
argument can absorb. The same estimator would need a correction, not an appeal
to direction, if the argument ran the other way — if a *ceiling* were being
claimed as reachable — and no argument in this paper does that. `scripts/record_stage1_claims.py` refuses
to record at all if the measured gain ever reaches the declared target, because
that outcome would demand rewriting this section rather than re-rendering it,
and `tests/test_evaluation_population_bound.py` supplies the passage that makes
the refusal fire.

One derived quantity moves when the denominator is corrected, and the corrected
one is the honest figure for a same-population statement. Against the training
budget the agent captures {{STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT}}% of the
available first-order signal; against the bound on the passage it is actually
scored on, {{STAGE1_HELDOUT_SIGNAL_CAPTURED_PCT}}%. The first compares a
held-out numerator to a training denominator and is reported here only because
the literature's usual framing does the same.

The trained agent realizes {{STAGE1_GAIN_OVER_UNIFORM_NATS}} nats, which is
{{STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT}}% of what the order allows
([@tbl:letter_information]). Under the declared gate that near-optimal result reads
as a failure, and the natural response to a failure — build a better bigram model
— is guaranteed to be wasted. An unreachable gate does not merely fail to
discriminate; it actively points the work in a direction that cannot succeed. The
restated gate is the context-free unigram baseline, which the agent clears by
{{STAGE1_UNIGRAM_MARGIN}} perplexity units. That bar is reachable and, unlike
uniform, diagnostic — with one qualification the ledger carries. A model can
beat uniform while carrying no context at all, and beating the
training-estimated unigram on a finite held-out passage is *consistent with*
using the previous letter but does not by itself attribute the gain to it: a
different context-free distribution can land below the count-estimated bar
through distribution shift alone, and one measurably does — the trained
transition's own context-marginalized distribution scores
{{L1_CONTEXT_MARGINAL_PERPLEXITY}} against the bar's
{{STAGE1_UNIGRAM_BASELINE}} on this passage. The attribution is carried instead
by a context-destruction control: rotating the trained conditionals off their
own contexts — the same thirty distributions, mis-assigned — collapses the model
to {{L1_CONTEXT_DERANGEMENT_PERPLEXITY}}, far above the bar on the identical
positions, so the margin the agent realizes is attributable to
correctly-assigned previous-letter conditioning rather than to the letter
frequencies its transition happens to carry.

### Stage 2: the declared threshold sat below the baseline, against an absent oracle

The Stage 2 gate was declared as `letter->phoneme accuracy
>={{HISTORICAL_STAGE2_GATE_PCT}}% on the English EWT
oracle`. The first error is that the oracle does not exist. Universal Dependencies
English EWT is a syntactic treebank [@nivre2020ud; @silveira2014gold]; its CoNLL-U
fields are `lemma`, `upos`, `feats`, and `deprel`, with no phonemic annotation of
any kind. The line had been copied from the Stage 4 case gate, where UD is at least the
right corpus — though [@sec:case_level] shows the eight-way labeling that gate
scores against is a documented heuristic rather than a gold standard, so it was
defective too, for a different reason. A phoneme task cannot be scored on UD at
all, at any threshold.

The second error survives fixing the first. Measured on held-out CMUdict, a
context-free per-letter lookup table — no transition matrix, no pymdp, no active
inference — scores {{STAGE2_G2P_BASELINE_ACCURACY}}%. The declared threshold sits
below the trivial baseline, so a model doing nothing hierarchical passes. By
[@eq:gate_verdict] the verdict is `VACUOUS`, and it would have remained
vacuous on the correct corpus.

| Stage | Declared threshold | Trivial baseline $b_P$ | Best comparator $c_P$ | Comparator type | Verdict | Restated as |
|---|---|---|---|---|---|---|
| 1 letters | `>={{STAGE1_DECLARED_GATE_NATS}} nats above uniform` | unigram letter frequencies, {{STAGE1_UNIGRAM_BASELINE}} | empirical order-1 optimum on the evaluation passage, {{STAGE1_HELDOUT_FLOOR_PERPLEXITY}} | model-class bound | `UNREACHABLE` | beat the unigram bar |
| 2A phonemes | inherited {{STAGE1_DECLARED_GATE_NATS}}-nat wording | unigram phone frequencies, {{STAGE2_PHONEME_UNIGRAM_BASELINE}} | exact all-position order-1 optimum, {{STAGE2_PHONEME_ALL_POSITION_FLOOR_PERPLEXITY}} | model-class bound | `UNREACHABLE` | beat the unigram bar |
| 2B letter-to-phone | `>={{HISTORICAL_STAGE2_GATE_PCT}}% on the English EWT oracle` | per-letter lookup table, {{STAGE2_G2P_BASELINE_ACCURACY}}% | whole-word smoothing, {{STAGE2_G2P_SMOOTHING_CEILING}}% | reference model | `VACUOUS` | beat the lookup table, paired McNemar |

: Both shipped gate defects worked through [@eq:gate_verdict], with the measured baseline and comparator that classify them. Stage 1 and Stage 2B use held-out evaluation populations; Stage 2A is explicitly an in-sample full-dictionary protocol, and its baseline, model, and exact all-position bound share the same {{STAGE2_PHONEME_EVAL_POSITIONS}}-position denominator. The Stage 2A row is the same unreachable wording inherited one level down, caught before it shipped by applying the check that Stage 1 had already forced. The comparator-type column is load-bearing and is stated rather than left to the reader. A **model-class bound** is an upper bound on what any model of the class can reach, so a threshold beyond it cannot be met by better engineering. A **reference model** is a measured system with information the shipped one forgoes; it can be exceeded, and at Stage 4 it is, so the taxonomy no longer permits `UNREACHABLE` against a reference at all — a reference failing the bar classifies `ABOVE_REFERENCE`, a statement about that comparator and not a theorem. One residual asymmetry is recorded rather than silently absorbed: when the *stronger empirical comparator* (a reference model fitted on more information, such as [@sec:comparator_strength]'s lookup table) clears a bar the shipped model fails, `ABOVE_REFERENCE` still reads as a statement about the declared comparator. A future classifier revision could separate that case as a distinct verdict (`REFERENCE_NOT_CLEARED`: a stronger measured comparator cleared the bar and the shipped model did not) from the model-class-bound `UNREACHABLE`; that revision belongs to the gate code, not this manuscript, so the limitation is stated here. The two other types the project uses elsewhere are a **privileged oracle** (Stage 3's held-out word model, Stage 5's type oracle) and an **analytic bound** (generation's zero distance). Calling all four a ceiling would license claiming more from a bracketed verdict than the comparator supports, which is why the type is an argument to the classifier rather than a caption. {#tbl:gate_audit}

The middle row of [@tbl:gate_audit] is the payoff. Once Stage 1's defect had been
diagnosed as *unreachable* rather than as *a threshold that happened to be wrong*,
the same measurement was cheap to repeat at Level 2, where the available gain over
uniform is {{STAGE2_PHONEME_ALL_POSITION_MAX_GAIN_NATS}} nats on the exact
all-position protocol, below the inherited
{{STAGE1_DECLARED_GATE_NATS}}-nat threshold. The wording
would have shipped a second time. A taxonomy earns its keep by making the second
instance of a defect cheaper to find than the first.

### A reproducible subsample is not a representative one

The third finding is not about the threshold but about what the threshold is
evaluated on, and it is the one most likely to recur, because the reasoning that
produces it is locally correct.

The letter-to-phone evaluation originally took a deterministic prefix of the held-out pair
list, `pairs[:limit]`, on the argument that a prefix needs no seed and is therefore
perfectly reproducible. It is. But `g2p_pairs` is alphabetically sorted, so a prefix
is *only words from the start of the alphabet*. Reproducibility and
representativeness are independent properties, and satisfying the first is not
evidence about the second. Measured on that prefix the letter-to-phone (G2P)
margin is
{{SUBSAMPLE_PREFIX_MARGIN_PTS}} points against the full set's
{{SUBSAMPLE_FULL_MARGIN_PTS}}, an overstatement by a factor of
{{SUBSAMPLE_PREFIX_OVERSTATEMENT_FACTOR}}, which reads as a comfortable pass
rather than the thin one it is. The prefix spans
{{SUBSAMPLE_PREFIX_INITIALS}} word initials where the full held-out set spans
{{SUBSAMPLE_FULL_INITIALS}}. Evenly spaced indices cost exactly the same and
span far more of them, and `_iter_pairs` in `src/ailm/levels/phoneme.py` now
takes a stride instead.

The finding is not asserted from those numbers, it is re-derived from them:
`scripts/record_subsample_claims.py` computes the overstatement factor from the
two recorded margins and **refuses to record** if it is not greater than one, and
refuses again if the prefix ever spans as many initials as the full set. Either
refusal means the bias this section is built on has disappeared and the section
needs rewriting, which is the outcome a guard should force rather than absorb.

Fixing the bias exposes a second problem underneath it, and separating the two is
the part worth generalizing. The first contrast is about *selection* and holds
size fixed: the alphabetical prefix over {{SUBSAMPLE_PREFIX_POSITIONS}} positions
against an evenly spaced stride truncated to the same size,
{{SUBSAMPLE_STRIDE5_POSITIONS}} positions, whose margin is
{{SUBSAMPLE_STRIDE5_MARGIN_PTS}} points — much nearer the full-set figure than
the prefix's. The second contrast is about *size* and holds selection fixed:
that same honest stride returns *p* = {{SUBSAMPLE_STRIDE5_MCNEMAR_P}}, and a
denser stride, with a margin of {{SUBSAMPLE_STRIDE3_MARGIN_PTS}} points, is
closer but still short of the declared criterion, while the full
{{STAGE2_G2P_EVAL_POSITIONS}} positions reach *p* = {{STAGE2_G2P_MCNEMAR_P}}.
**An unbiased subsample of the same size is underpowered.** The biased prefix had
looked significant, and it looked significant for the wrong reason: not because
the effect was large but because the subset was easy. The bounded pymdp
subsample the stage tests still run — {{STAGE2_G2P_SUBSAMPLE_POSITIONS}}
positions, *p* = {{STAGE2_G2P_SUBSAMPLE_MCNEMAR_P}} — is kept for exactly that
reason: it is a live demonstration that this procedure returns "not significant"
on real data of an inadequate size.

At the Bonferroni-corrected per-test threshold of the declared family,
{{STAT_BONFERRONI_ALPHA}} ([@sec:multiplicity]), the full set has power
{{STAT_STAGE2_POWER_AT_REFERENCE}} against the fixed reference share
{{STAT_REFERENCE_SHARE}} over its {{STAT_STAGE2_DISCORDANT_N}} discordant
pairs: it could have detected the smallest effect this project treats as
interesting. Its effect size and exact interval are recorded with the remaining
primary paired effects in [@tbl:multiplicity_family].

![Prefix, stride[::{{SUBSAMPLE_MATCHED_STRIDE}}], stride[::{{SUBSAMPLE_LARGER_STRIDE}}], and full held-out rows are compared in two panels: the G2P margin and its paired-test p-value. The prefix overstates the effect; the same-sized stride[::{{SUBSAMPLE_MATCHED_STRIDE}}] lands near the full-set margin but cannot establish it; stride[::{{SUBSAMPLE_LARGER_STRIDE}}] is larger and closer but remains below the declared significance criterion; only the full set does both. The prefix-versus-stride[::{{SUBSAMPLE_MATCHED_STRIDE}}] pair holds size fixed and isolates selection; the stride-versus-full pair holds selection fixed and isolates size. A prefix that did not differ from the full estimate, a stride that was not representative, or a full-set bar that changed the recorded direction would falsify this panel.](../../output/figures/subsample_bias.png){#fig:subsample_bias width=85%}

[@fig:subsample_bias] shows why neither subsample is an acceptable evaluation. The
resolution was not to shrink the claim or inflate the sample, but to remove the
constraint that forced subsampling in the first place: test pymdp and the numpy
oracle against each other on a bounded subset — {{STAGE2_G2P_SUBSAMPLE_POSITIONS}} of the
{{STAGE2_G2P_EVAL_POSITIONS}} held-out positions — where they disagree on
{{STAGE2_DECODER_DISAGREEMENTS}} positions ([@eq:decoder_equivalence]), then
compute full-set statistics with the fast oracle. That is tested equivalence on
the checked positions and an extrapolation beyond them, not a proof that the two
decoders are the same function ([@sec:verification]). The same bias recurred in
Part A, where a length-matched-subset measurement gives
{{STAGE2_PHONEME_SUBSET_PERPLEXITY}} for a quantity whose full-dictionary value is
{{STAGE2_PHONEME_PERPLEXITY}}; the two are different measurements over different
position sets and must not be reported interchangeably, so Part A is now computed
in closed form over all {{STAGE2_PHONEME_EVAL_POSITIONS}} positions.

```{=latex}
\csname @ifclassloaded\endcsname{beamer}{}{\Needspace{32\baselineskip}}
```

### Level 3: the declared gate states a condition the bar itself already clears {#sec:word_gate}

Level 3's declared gate was "perplexity below {{STAGE3_DECLARED_GATE}} on
PTB-1k". Measured, the letter-bigram bar itself scores
{{STAGE3_BIGRAM_BASELINE}}, clearing {{STAGE3_DECLARED_GATE}} by a
factor of {{STAGE3_DECLARED_GATE_SLACK}}. The gate was vacuous by a wide margin
and named a corpus the project does not have — the third borrowed round constant
to turn out to constrain nothing.

The restated gate compares the level against a letter-bigram model over the
identical alphabet and the same whole-document training partition. The bar is
fitted to every training type while the word prior is fitted to its bounded
lexicon; both are scored on the identical held-out in-vocabulary positions.
This gives the non-lexical comparator strictly broader training support.
[@tbl:word_comparators] places it against the full comparator ladder.

| Comparator | Uses the lexicon? | Per-emission perplexity |
|---|---|---|
| Uniform over {{STAGE0_WORD_ALPHABET_SIZE}} symbols | no | {{STAGE3_UNIFORM_BASELINE}} |
| Context-free letter unigram | no | {{STAGE3_UNIGRAM_BASELINE}} |
| **Letter bigram — the bar** | no | **{{STAGE3_BIGRAM_BASELINE}}** |
| **Word times position model** | yes | **{{STAGE3_WORD_PERPLEXITY}}** |
| Held-out-oracle word model — privileged reference | yes | {{STAGE3_CEILING_PERPLEXITY}} |

: Stage 3 over {{STAGE3_EVAL_POSITIONS}} emissions from {{STAGE3_EVAL_TOKENS}} held-out in-vocabulary tokens. The gain over the bar is {{STAGE3_GAIN_NATS}} nats per emission and the remaining gap to the privileged reference is {{STAGE3_HEADROOM_NATS}}, so the level realizes {{STAGE3_SIGNAL_CAPTURED_PCT}}% of the measured bar-to-reference gap. That reference is not a bound on every admissible model. Unlike Stage 2 the margin is wide by construction, so no paired test is needed — a measured conclusion, not an omission. {#tbl:word_comparators}

The margin is comfortable. That comfort is exactly what makes the level the
sharpest illustration of the lesson [@tbl:miswiring_signatures] introduced one
level down: a lexicon is so much information that almost any word prior beats a
letter model. In all, {{STAGE3_MISWIRINGS_TOTAL_WORD}} mis-wirings were built against the
bar; {{STAGE3_MISWIRINGS_FINITE_WORD}} are scored and recorded, and tabulated
below. The remaining arm — end-of-word emitted one position
early — assigns zero probability to every true path, so its perplexity is
unbounded and it is not a comparable number. It is reasoned rather than recorded
(decision 14). Note that running it through the oracle would not return an
infinity either: the log floor of [@sec:verification] converts an impossible path
into a large finite penalty, so "infinite perplexity" is the analytic statement
and not a measurement anything here would print.

| Mis-wiring | Per-emission perplexity | Caught by the bar? |
|---|---|---|
| Uniform prior over words — lexicon kept, frequencies discarded | {{STAGE3_MISWIRE_UNIFORM_PRIOR}} | **no** |
| Prior derived from word length instead of frequency | {{STAGE3_MISWIRE_LENGTH_PRIOR}} | **no** |
| Frequency ranking reversed — the rarest words made the likeliest | {{STAGE3_MISWIRE_REVERSED_PRIOR}} | **no** |
| Time-reversed emission — letters emitted back to front | {{STAGE3_MISWIRE_TIME_REVERSED}} | yes |

: {{STAGE3_MISWIRINGS_CLEARING_BAR}} of the {{STAGE3_MISWIRINGS_TOTAL}} Level 3 mis-wirings clear the bar of {{STAGE3_BIGRAM_BASELINE}}; the {{STAGE3_MISWIRINGS_FINITE}} with finite perplexity are shown. The third is the striking one: inverting the frequency ranking, so that the rarest words are predicted most confidently, still beats a letter-bigram model. A gate whose only guard is an inequality against a baseline admits a model whose word prior is upside down. {#tbl:word_miswirings}

What pins the level is not the bar but a closed-form identity. Because every
factor except the word prior is deterministic, the probability the model assigns
to a word's whole emission path must equal that word's unigram probability. A
live `pymdp` walk and an independent derivation that imports neither `pymdp` nor
`jax` agree to {{STAGE3_PATH_IDENTITY_MAX_ABS_ERR}} nats over
{{STAGE3_PATH_IDENTITY_WORDS}} vocabulary words. All
{{STAGE3_MISWIRINGS_CLEARING_BAR_WORD}} surviving mis-wirings change the word
prior, so every one fails it.

The general form of the lesson: when a level's mechanism is highly informative, the
baseline comparison stops discriminating precisely because the level is good.
Necessary conditions get weaker as models get stronger, and a gate built only
from them gets correspondingly less able to fail.

### Level 4: one threshold failing in opposite directions on each population {#sec:case_gate}

Level 4's declared gate was "at least {{STAGE4_DECLARED_GATE}} percent against
the UD {{STAGE4_N_CASES}}-case
oracle". It fails in **both** directions simultaneously, and which failure is
observed depends only on which population the threshold is scored against.
[@tbl:case_both_defects] scores it against both.

| Population | $n$ held out | Cases | Majority | Context-free bar | **Model** | Smoothing | Viterbi | Verdict |
|---|---|---|---|---|---|---|---|---|
| Gold morphology | {{STAGE4_GOLD_TEST_TOKENS}} | {{STAGE4_GOLD_CASES_PRESENT}} of {{STAGE4_N_CASES}} | {{STAGE4_GOLD_MAJORITY}}% | {{STAGE4_GOLD_BASELINE}}% | **{{STAGE4_GOLD_AGREEMENT}}%** | {{STAGE4_GOLD_SMOOTHING}}% | {{STAGE4_GOLD_VITERBI}}% | **vacuous** |
| Derived oracle | {{STAGE4_DERIVED_TEST_TOKENS}} | {{STAGE4_DERIVED_CASES_PRESENT}} of {{STAGE4_N_CASES}} | {{STAGE4_DERIVED_MAJORITY}}% | {{STAGE4_DERIVED_BASELINE}}% | **{{STAGE4_DERIVED_AGREEMENT}}%** | {{STAGE4_DERIVED_SMOOTHING}}% | {{STAGE4_DERIVED_VITERBI}}% | **above-reference** |

: One declared threshold of {{STAGE4_DECLARED_GATE}}%, two opposite defects. On gold morphology a constant majority predictor already clears it. On the derived oracle it sits above the model, above both strictly more powerful decodes of that model, and above a context-free predictor fitted on the test labels themselves ({{STAGE4_DERIVED_CF_ORACLE}}%). Every one of those comparators is exceedable — a smoothing decode is a reference and a test-fitted lookup a privileged oracle, not bounds over the sequence-model class — so the licensed verdict is above-reference, not unreachable: the bar sits above everything measured here, and no impossibility theorem follows. An earlier version of this taxonomy called this row unreachable; that call is exactly what [@eq:reference_no_impossibility] forbids and `unreachable_requires_bound` machine-checks, hence the retype. {#tbl:case_both_defects}

This is the clearest evidence in the project for a claim the earlier stages only
gesture at: **a threshold without a named population is not a threshold.** The
same number, the same corpus and the same model yield opposite verdicts, and
nothing in the declared gate's wording distinguishes them.

The restated Stage 4 gate is beating the context-free
$\arg\max\, p(\text{case} \mid \text{UPOS})$ bar on the derived population,
defended by a paired test: {{STAGE4_DERIVED_AGREEMENT}}% against
{{STAGE4_DERIVED_BASELINE}}%, a margin of {{STAGE4_DERIVED_MARGIN_PTS}} points
over {{STAGE4_DERIVED_TEST_TOKENS}} positions, with
{{STAGE4_DERIVED_MCNEMAR_B}} against {{STAGE4_DERIVED_MCNEMAR_C}} discordant
positions at $p =$ {{STAGE4_DERIVED_MCNEMAR_P}}. No gate is stated on the gold
population, for the reason [@sec:case_level] measures: the committed split yields
identical predictions from the fitted sequence model and its context-free
comparator. That is an empirical paired null on this split, not an exact
consequence of the constant observation.
The model's favored share of those {{STAT_STAGE4_DISCORDANT_N}} discordant
positions is {{STAT_STAGE4_FAVORED_RATIO}}, marginal exact {{SHARE_CONFIDENCE_PCT}}% interval
[{{STAT_STAGE4_SHARE_CI_LOW}}, {{STAT_STAGE4_SHARE_CI_HIGH}}], with power
{{STAT_STAGE4_POWER_AT_REFERENCE}} against the fixed reference share
{{STAT_REFERENCE_SHARE}} at the family-corrected threshold
([@tbl:multiplicity_family]): unlike the Stage 2B comparison, this one could
not have detected the smallest effect the project treats as interesting.

### A context-free comparator is necessary but not sufficient {#sec:comparator_strength}

Requirement 1 of the staged-gate method ([@sec:introduction]) says the comparator is context-free and
never uniform, because a model that beats uniform has shown only that it knows
which symbols are frequent. That requirement is necessary. It is not sufficient,
and this section is the demonstration, on this project's own gates.

Nothing in those requirements or in the pre-build checklist says to look for the *strongest* comparator that is
still not the model. So each gate was stated against the first defensible
comparator rather than the best one. Searching that class instead of picking from
it changes two verdicts' meaning ([@tbl:comparator_strength]).

The search is austere by design. For each level, fit an argmax over counts of
(previous observed symbol, current observed symbol) to the label, on the training
split alone, backing off to the per-symbol argmax where a pair is unseen. There
is no latent state, no filtering, no inference at decode time, and strictly less
information than the filtering decode uses, which sees the whole left context
through its belief state. It is the obvious thing, which is the point: a
comparator that took ingenuity to construct would be a weaker criticism than one
that does not.

| | Stage 2B letter-to-phone | Level 4 case |
|---|---:|---:|
| declared context-free bar | {{COMPARATOR_STAGE2_UNIGRAM_ACCURACY}}% | {{COMPARATOR_STAGE4_UNIGRAM_ACCURACY}}% |
| shipped model | {{STAGE2_G2P_ACCURACY}}% | {{STAGE4_DERIVED_AGREEMENT}}% |
| left-context lookup | **{{COMPARATOR_STAGE2_LOOKUP_ACCURACY}}%** | **{{COMPARATOR_STAGE4_LOOKUP_ACCURACY}}%** |
| lookup parameters | {{COMPARATOR_STAGE2_LOOKUP_PARAMETERS}} | {{COMPARATOR_STAGE4_LOOKUP_PARAMETERS}} |
| lookup over model | {{COMPARATOR_STAGE2_OVER_MODEL_PTS}} pts | {{COMPARATOR_STAGE4_OVER_MODEL_PTS}} pts |
| model's own margin | {{STAGE2_G2P_MARGIN_PTS}} pts | {{STAGE4_DERIVED_MARGIN_PTS}} pts |

: The strongest member of the comparator class, searched rather than assumed. The declared-context-free-bar row is the shipped bar, recomputed by this probe and required to reproduce it; that agreement is what establishes the probe scores the gate's own population rather than a similar one. {#tbl:comparator_strength}

At Stage 2B the lookup exceeds the shipped model by
{{COMPARATOR_STAGE2_OVER_MODEL_PTS}} points, which is several times the model's
entire recorded margin over its declared bar. At Level 4 it exceeds it by
{{COMPARATOR_STAGE4_OVER_MODEL_PTS}}.

**No recorded number is withdrawn.** The model accuracies, the context-free bars,
the margins and every gate verdict stand exactly as recorded, and all of them
reproduce. What is withdrawn is an interpretation: that clearing the declared bar
showed the latent-state machinery doing the work. It did not, because the bar was
not the hardest thing in its own class.

**The controls, including the one that is weak.** A positive control requires the
bigram lookup not to score below the per-symbol argmax it backs off to, since it
uses strictly more context; it does not at either level. A negative control
deranges the same map, holding the key set, the label multiset and the parameter
count fixed and changing only which label sits at which key. At Stage 2B that
collapses to {{COMPARATOR_STAGE2_DERANGED_ACCURACY}}%, which is clean. At Level 4
it reaches {{COMPARATOR_STAGE4_DERANGED_ACCURACY}}%, above the bar and below the
lookup, so a substantial share of the Level 4 result is context-conditional label
frequency rather than the specific mapping. That is reported rather than
suppressed, and it means the Level 4 half of this finding should be discounted
relative to the Stage 2B half.

**Why this belongs in the paper rather than in a reviewer's report.** It is the
gate-audit defect class recurring inside the repair, found by applying the
project's own discipline to the project's own method. The taxonomy of
[@eq:gate_verdict] is unaffected: a verdict still classifies a threshold against a
baseline and a typed comparator, and those verdicts are still right. What this shows is
that the *check* is not yet a procedure. A procedure would say: enumerate the
comparator class, fit its strongest member on training data alone, and place the
threshold above that. Requirement 1 asks only that the comparator not be uniform,
and the distance between those two demands is the distance between a taxonomy and
a method.

This is the sharpest available argument that the working hypothesis of
[@sec:introduction] is worth testing outside this repository. A check that its own
author can defeat by fitting a two-symbol lookup table is a check whose failure
modes are not yet exhausted.

## Executable gate-audit report

Both defects were found by hand, and being found by hand is not a property that
survives contact with a schedule. `scripts/audit_gates.py` performs the
classification of [@eq:gate_verdict] as an executable artifact: for each
implemented stage it measures the trivial baseline, the shipped model, and the
typed comparator and prints them together. Its inventory comprises the rows
of [@tbl:gate_audit], the implemented later-stage rows of
[@tbl:unaudited_gates], and the generation gate, whose comparators belong with
the rest of that level and are tabulated in [@sec:generation]. The inventory is
fixed by `AUDITED_GATE_IDS` in `src/ailm/gates.py` and compared, in order,
against the rows the script actually built; a mismatch aborts the run with its
own exit code before a single verdict is printed. A gate that quietly stopped
being audited therefore fails the run rather than vanishing from it, which is
the failure mode an audit script is most prone to and least able to notice.

The script exits non-zero if any ordinary verdict is not bracketed (of either
strength) or if the model sits on the wrong side of its own bar, with one
exception required by the outcome of the retained case-only clause ablation
([@sec:clause_level]). That row must
remain `ABOVE_REFERENCE`, both the model and fitted privileged reference must
sit on the failing side, both whole-document-bootstrap privileged-label
intervals must span zero, and the planted-effect control must fire. A model
**pass**, a comparator that clears the bar, or a resolved interval
trips the exit code, because any of them means the finding changed rather than that
the status string may remain fixed. Stages with no
implementation are printed as `UNAUDITED` alongside their declared thresholds,
so an unchecked gate is visible rather than absent.

Three properties of the audit bound what a clean run establishes. The typed
comparators include a smoothing decode, a conditional entropy, a
per-item argmax, an oracle given the answer — and only some of them bound the
model class. The verdict carries that distinction rather than leaving it to
a footnote: each row declares its comparator's type, `BRACKETED_BOUND` and
`BRACKETED_REFERENCE` say which strength of comparator beat the baseline, and a
reference clearing a bar never reads as a bound doing so. Where the shipped
model scores above its own recorded smoothing reference, as on the Stage 4
derived row, the script prints the row with an explicit warning rather than
reclassifying it, because the row is still bracketed-by-reference. The audit classifies
placement only: it says nothing about whether a margin is large enough to be
believed, which is the separate question [@sec:multiplicity] answers and which
no threshold comparison can answer on its own.

The third property is the one this section's own history creates. Almost every
threshold the script audits is a *restated* threshold, restated
after its stage had been built and measured — the audit sets each row's bar to
the measured baseline of the population the model is then scored against. A
clean run therefore certifies that the restated rows are placed where the audit
convention places them; it does not certify that any of them was chosen without
knowledge of the answer. That is retrospective re-certification rather than
sequential validation, and [@sec:multiplicity] states what it costs every
significance verdict downstream. The counterweight is that a restated gate can
still fail: the retained case-only clause row is restated exactly like the
others and the shipped model lands on its failing side. Its fitted privileged
reference now does too, and its gain remains unresolved under whole-document
resampling ([@sec:clause_level]), which is why the
script treats a model pass or an interval that no longer spans zero as an alarm.

The canonical Stage 5 row is the one exception to the restatement, and it is
worth stating plainly rather than leaving a reader to infer it from a date. Its
bar was written down in the decision record before the artifact it scores was
regenerated, together with the alphabet, the population, the comparators, the
controls, the predictions and the falsifiers, and the entry said in as many
words that a mixture failing that bar would mean the widening was decorative.
The mixture cleared it. That is a stronger claim about placement than any
restated row here can make, and a weaker one than the confirmatory replication
of [@sec:confirmatory], because registration and outcome share one development
session and no held-back partition survives on which to repeat it
([@sec:clause_widened]).

The quoted speech threshold uses text-to-speech (TTS) synthesis.

| Stage | Declared threshold | Audit outcome |
|---|---|---|
| 3 words | `perplexity < {{STAGE3_DECLARED_GATE}} on PTB-1k` | VACUOUS — the letter-bigram bar itself clears it by {{STAGE3_DECLARED_GATE_SLACK}}× ([@sec:word_level]) |
| 4 cases | `>={{STAGE4_DECLARED_GATE}}% vs the UD {{STAGE4_N_CASES}}-case oracle` | VACUOUS on gold morphology and ABOVE_REFERENCE on the derived oracle, from one threshold ([@sec:case_level]) |
| 5 clauses | `free energy tracks clause length` | VACUOUS — cumulative free energy is monotone under ANY model; on the document-disjoint derived gate both the model and fitted privileged reference FAIL the bar (`ABOVE_REFERENCE`), and the paired privileged-label gain remains unresolved ([@sec:clause_level]); the canonical widened-observation gate, registered before its artifact was regenerated, is bracketed and passes ([@sec:clause_widened]) |
| 6 narrative | `narrative C vector constrains clause priors` | legacy `C` wording corrected to an L6→L5 `D`-prior intervention; its structural guards pass, and the registered recognition point estimate has a document-unit interval that spans zero on the development population and excludes zero on the held-back one ([@sec:narrative_level], [@sec:stage6_heldback_method]) |
| 7 speech | `Whisper -> phoneme -> TTS round trip` | out of scope as declared — it measures Whisper's and the TTS engine's models; re-scoped to the boundary interface contract, `UNAUDITED` as a model gate |

: Every declared later-stage gate, and what auditing it found once each stage was built. Every row failed its audit as declared through vacuity, a population-dependent double defect, or mis-scoping. That is the running argument of this section: a declared threshold is a conjecture about measurements nobody has made yet. {#tbl:unaudited_gates}

[@tbl:unaudited_gates] began as a liability list of unaudited declarations and
is now a completed record: every declared gate was audited when its stage was
built, and none survived unchanged. The Stage 3 constant was vacuous on
arrival, exactly as the unmeasured-baseline shape predicted. The Stage 4
percentage failed in both directions at once from a single threshold. The
Stage 5 declaration could not fail for any model. The pattern is uniform
enough to state as the section's conclusion: not one threshold declared before
its stage was built survived the measurement of its own baseline and typed comparator.

## Pre-build gate requirements

Applied before writing the level, not after measuring it. Steps 1 through 4 are
cheap — they need the corpus, not the model. No gate in this project was built
this way: each was declared cheaply, found defective once measured, and restated
against the measurement. The checklist is what that experience recommends, not
what it records.

1. **Name the metric and its direction.** Fix which side of the threshold is
   passing, in code, before any number exists. [@eq:gate_satisfies] is where a
   sign error becomes a silently inverted gate.
2. **Measure the trivial baseline $b_P$.** Build the dumbest model that produces a
   number on this task — context-free frequencies, a per-item lookup table, the
   majority class — and score it on the real evaluation set. If $b_P \succ t$, the
   gate is vacuous on that population; restate it against $b_P$ and stop.
3. **Measure and type the comparator $c_P$, on the evaluation population.**
   Use the conditional entropy of *that* population, an oracle with access the
   model does not have, or another explicit reference, and declare which
   comparator type the taxonomy assigns it. If $c_P \nsucc t$, a bound type makes
   the gate `UNREACHABLE`; an exceedable type makes it `ABOVE_REFERENCE` and
   licenses no impossibility claim. A training-distribution entropy is a
   legitimate calibration reference and is *not* $c_P$ for a gate scored
   held-out; if it is all that is available, say so in the row and do not read
   the verdict as a same-population comparison. Never inherit a threshold
   constant from another stage without redoing this step — that is precisely
   how the {{STAGE1_DECLARED_GATE_NATS}}-nat wording reached Level 2.
   `classify_gate` refuses an untyped comparator, and only a bound type can
   produce "no implementation could pass".
4. **Confirm $b_P$ and $c_P$ were measured on the same population $P$ the gate
   scores the model on.** A comparator from one subset and a baseline from another
   produce a margin that describes nothing, and [@eq:gate_verdict] is a function
   of the pair $(t, P)$ — a verdict quoted without its population is not a
   verdict.
5. **Check the evaluation set for structural bias.** Is it sorted? Is a prefix a
   proper subset of some category? Is it length-, frequency-, or
   alphabetically filtered? Reproducible is not representative.
6. **Check the evaluation set for power at the corrected threshold.** If the
   expected margin is small, compute whether the intended sample size can
   establish it at all — against the family-corrected per-test threshold of
   [@sec:multiplicity], not the nominal one, because adding a test tightens the
   bar for every test. With an exact test the question has a sharper form than
   "is power adequate": below a minimum discordant count there is no rejection
   region at any outcome, and `exact_two_sided_rejection_region` returns that
   emptiness rather than a small number. An honest effect on an underpowered
   sample fails, and that failure is uninformative.
7. **Write the negative control before the model.** A deliberately broken variant
   — transposed matrix, uniform emission, shuffled labels — that must land on the
   failing side. If it passes, `GateVacuousError` and the stage is blocked
   ([@sec:verification]).
8. **Check that the failure signature is sufficient, not merely necessary.**
   [@tbl:miswiring_signatures] is the counterexample: the wrong-normalization mis-wiring cleared the
   baseline bar while being wrong. Add an invariant that pins the structure
   directly.
9. **Record every threshold, baseline, and typed comparator in the ledger** and register the
   gate in `AUDITED_GATE_IDS` and `scripts/audit_gates.py`, so the classification
   is re-derived on every run rather than remembered and a gate that stops being
   audited fails the run.
10. **Declare any new inferential p-value into the multiplicity family.** A
    threshold-crossing claim that is not a member of `SIGNIFICANCE_FAMILY` is
    judged against an uncorrected threshold while tightening nobody else's,
    which is the one way to add a test and make every existing claim look
    stronger. A descriptive diagnostic or re-analysis may remain outside only
    through a named, machine-checked exclusion and prose that carries no
    threshold-crossing inference ([@sec:multiplicity]).
11. **Fix the threshold and the test before the population is scored, and hold
    a partition back for them.** A gate restated after its stage has been
    measured, or a p-value declared into the family once its value is known, is
    retrospective re-certification: the correction of [@sec:multiplicity] bounds
    a fixed family's Type-I rate and does nothing about a test chosen with the
    answer already in view. Step 2 and step 3 are the steps that consume the
    evaluation set, so the partition the final gate is scored on must be one
    they never touched.

## Level 3: the letter-bigram bar admits a wrong word prior

### The letter-bigram bar does not catch a wrong word prior

The level also relocates where the learned content sits. At the letter and phoneme levels the
corpus statistics parameterize the transition; here both transitions are
structural constants and the single learned quantity is the prior over word
identity. That is the level's whole content — a lexicon with frequencies — and it
moves the mis-wiring risk with it. A wrong prior at Level 3 is what a wrong
transition was at Level 1, with one difference that [@sec:word_gate] makes the
case for: the letter-bigram bar does not catch it. [@fig:word_miswirings] shows the
{{STAGE3_MISWIRINGS_FINITE_WORD}} finite-score mis-wirings beside the correct construction, and which of them the bar
catches.

![The {{STAGE3_MISWIRINGS_FINITE_WORD}} finite-perplexity Stage 3 mis-wirings against the letter-bigram bar, on a log axis because the time-reversed emission is {{STAGE3_MISWIRE_TIME_REVERSED_RATIO}} times the bar, or {{STAGE3_MISWIRE_TIME_REVERSED_ORDERS}} orders of magnitude worse. The analytic-infinite end-of-word arm is omitted because no log axis can carry it ([@tbl:word_miswirings]). The shipped model is drawn as the top bar for scale and is not one of the mis-wirings counted here. Every bar ending left of the dashed line clears it — {{STAGE3_MISWIRINGS_CLEARING_BAR}} of the {{STAGE3_MISWIRINGS_FINITE}} shown do, including a prior whose frequency ranking is reversed so the rarest words are predicted most confidently. The bar is a necessary condition, never a sufficient one. A changed number of wiring variants, a bar drawn at the wrong comparator, or a reversal of the time-reversed ordering would falsify this panel.](../../output/figures/word_miswirings.png){#fig:word_miswirings width=92%}

### The closed form scales to every length-eligible training type

The vocabulary bound is a property of the *agent*, not of the model, and the two
scale differently. Because every factor except the word prior is deterministic,
the closed form is exact at any vocabulary size — a word's score simply *is* its
prior mass — so the model can be measured wherever a lexicon can be built.

Measured over all {{STAGE3_FULL_VOCAB_SIZE}} Brown training types that meet the
word-length eligibility rule, coverage rises to {{STAGE3_FULL_COVERAGE_PCT}}%
of held-out tokens and the level scores {{STAGE3_FULL_WORD_PERPLEXITY}} per
emission against a letter-bigram bar of {{STAGE3_FULL_BIGRAM_BASELINE}} over
{{STAGE3_FULL_EVAL_POSITIONS}} emissions.

Numeric coverage and per-emission perplexity both rise, but their utility
directions oppose: more coverage is desirable and more perplexity is worse.
So the level is measured at {{STAGE3_SCALE_RUNGS}} nested frequency-ranked
lexicon sizes rather than at two endpoints ([@fig:scaling_ladder]).

Reading that ladder correctly requires two cautions, and both cut against the
easy version of the result.

The comparator is **not** re-fit at each rung. It is estimated once from the full
training frequencies and re-scored on whatever tokens each rung's lexicon
covers, so its matrices are bit-identical at every rung and all of its movement
is the changing evaluation population rather than a changing model. That
population is the second caution: a rung scores only the held-out tokens its
lexicon can represent, because the model assigns no mass outside its vocabulary,
so a smaller lexicon is scored on a smaller and more frequent — that is, easier —
subset of the same stream. Perplexity is an average over that subset, which makes
the per-emission gain incomparable across rungs by construction.

Both numbers are therefore recorded. Per emission the gain is nearly flat, from
{{STAGE3_SCALE_500_GAIN_NATS}} to {{STAGE3_FULL_GAIN_NATS}} nats, and
slightly non-monotone. Weighted so that every rung shares the top rung's
denominator — the same one-denominator rule the coverage axis already follows —
the gain widens strictly and by {{STAGE3_SCALE_GAIN_RISE_PCT}}%, from
{{STAGE3_SCALE_500_GAIN_SHARED}} to {{STAGE3_FULL_GAIN_SHARED}}. The
shared-denominator series is the one that compares rungs; the per-emission series
is what a reader of the table at a single operating point sees. Quoting only the
per-emission series would report a property of the denominator as a property of the model,
which is the error this level's own coverage rule exists to prevent.

The tempting reading, that the margin narrows because a larger lexicon is a
harder problem, is an artifact of comparing per-emission means over moving
populations. On the shared denominator, the gain widens across every rung.

![The word level across lexicon size, on nested frequency-ranked prefixes of the
Brown lexicon. Left: coverage of the held-out stream, measured against the
original held-out token count at every rung so the percentages share one
denominator. The final rung contains every training type that meets the
word-length eligibility rule. Right: per-emission perplexity for the level and for the
letter-bigram comparator, on log axes. The dashed line is
where the dense `pymdp` transition stops fitting in memory; the closed form is
exact beyond it, which is the distinction between what can be *run* and what can
be *measured*. The comparator is estimated once on the full
training frequencies and re-scored per rung, not re-fit, so its curve moves only
with the evaluation population. The panel is falsified if the shared-denominator
gain fails to rise monotonically, if any rung's recomputed gain disagrees with
its recorded pair, or if numeric coverage and perplexity fail to rise together.
Curve-crossing is deliberately *not* offered as a falsifier: on nested
frequency-ranked prefixes of a Zipfian corpus these curves cannot cross, so a
reader should treat their separation as arithmetic rather than as
evidence.](../../output/figures/scaling_ladder.png){#fig:scaling_ladder width=98%}

The gate stays bracketed at every rung: no rung has the comparator at or below
the model.

The `pymdp` agent does not follow it there. The word factor's dense transition
would be {{STAGE3_FULL_WORD_FACTOR_GB}} GB at float32, so instantiation caps at roughly
{{STAGE3_MAX_AGENT_VOCAB}} types, and the builder raises with that arithmetic
rather than failing on an allocation. Stating which of the two capabilities is
lost above the cap — the ability to *run* `pymdp`, not the ability to *measure*
the model — is the point of enforcing the bound explicitly. A custom lexicon can
be substituted for the frequency-ranked one, in which case evaluation still uses
held-out running text restricted to what that lexicon covers, never the lexicon
scored against itself.

A comparison the reader must not make: the perplexities in this section are
per emission over {{STAGE0_WORD_ALPHABET_SIZE}} symbols — {{STAGE0_WORD_LETTERS}} letters plus an end-of-word marker — while
Level 1's are over {{STAGE0_LETTER_ALPHABET_SIZE}} running-text symbols including space and punctuation. They
are different quantities over different alphabets and different position sets,
and placing them in one table would be a category error. `N_OBS` and `N_SYMBOLS`
are asserted to differ so that the warning is enforced rather than remembered.

## Beating the context-free baseline is necessary, not sufficient

Level 1 established that a mis-wired emission is detectable numerically but not
structurally. Level 2 sharpens the lesson, because its emission and transition are
both learned and expose separate placements for the error, and because one of the mis-wirings tested
against it *passed* the signature that caught the first one.

| Mis-wiring | Caught by "beats the context-free baseline"? | What does catch it |
|---|---|---|
| Time-reversed `B`, i.e. $P(\text{prev} \mid \text{cur})$ | yes — falls below the bar | the baseline comparison, and an index assertion on `B[0, next, current, 0]` |
| Uniform, information-free `A` | yes — collapses far below the bar | the baseline comparison |
| `A` normalized over phones instead of letters, then renormalized so pymdp still validates it | **no — clears the bar while wrong** | column sums of `A[0, :, p]` checked *before* any renormalization, and a value-for-value comparison of the shipped emission against an independent numpy re-derivation |

: The listed mis-wirings are evaluated against the same context-free baseline on the same deterministic stride subsample. The wrong-normalization row clears the baseline while being wrong, so the baseline signature is necessary but not sufficient. {#tbl:miswiring_signatures}

The third row in [@tbl:miswiring_signatures] is the useful one. An emission
normalized along the wrong axis and then renormalized is column-stochastic, so
`pymdp._validate()` accepts it; it is also close enough to the correct emission
that a filtering decode still beats a lookup table. Every check of the form "the
model beats the trivial comparator" would pass. Two invariants pin the emission,
and they catch the mis-wiring at different moments. The first is that
`A[0, :, p]` sums to one over *letters* for each phone $p$ — but only *before*
renormalization, because renormalizing is exactly what restores that sum, and
the guard is paired with a control asserting that the phone-normalized matrix
does break it. The second catches the renormalized variant, where the first
cannot: the shipped emission is compared value-for-value against an independent
numpy re-derivation of $P(\text{letter} \mid \text{phone})$, and that comparison
carries its own non-vacuity assertion — the mis-normalized matrix must differ
from the correct one by more than the comparison's tolerance, or value equality
would not be able to see it either. Structural validation and comparator-beating
are both necessary; neither is sufficient, and only a value-level comparison
against an independent derivation is sensitive to the difference between a right
answer and a nearly right one.

## Held-back replication of eligible family members {#sec:confirmatory}

The sixteen-member significance family was assembled retrospectively, on the
population every gate was revised against, and no correction factor turns that
into confirmation. The family is frozen and this section does not reopen it. Its
clause-level member is the case-only arm's paired classification test, which is
the arm that was scored when the family was frozen; promoting the widened arm to
canonical does not move that member, rename it, or add one. The members with a genuine held-back partition available were the four computed
on EWT populations: the UD English EWT test split, which no script in this repository
had scored. Under the protocol of [@sec:confirmatory_method], registered as
decision {{CONFIRMATORY_REGISTRATION_DECISION}} in the decisions log, those
four were scored against it exactly once. The population is the whole of that
split, {{CONFIRMATORY_CENSUS_SENTENCES}} sentences and
{{CONFIRMATORY_CENSUS_TOKENS}} tokens across
{{CONFIRMATORY_CENSUS_DOCUMENTS}} authenticated source documents, vendored
through the identical generators, oracles, exclusion rules and minimum-length
filters, with no stride applied because nothing is fitted on it. Every model
parameter was read unchanged from the existing development train fits, and the
recorder refused to run unless the arrays it built were equal, element for
element, to the ones the development artifacts produce. The document identities
are disjoint from every development document identity. Both controls were run
on this same population before any member was scored: the planted-effect
positive control fired decisively
({{CONFIRMATORY_CONTROL_POSITIVE_FAVORED}} against
{{CONFIRMATORY_CONTROL_POSITIVE_OPPOSED}} discordant pairs,
p = {{CONFIRMATORY_CONTROL_POSITIVE_P}}), and the label-shuffled negative
control did not ({{CONFIRMATORY_CONTROL_NEGATIVE_FAVORED}} against
{{CONFIRMATORY_CONTROL_NEGATIVE_OPPOSED}}, p =
{{CONFIRMATORY_CONTROL_NEGATIVE_P}}), so a null on this population is a
measurement rather than an instrument failure. Of the
{{CONFIRMATORY_MEMBERS_SCORED}} members scored, two cross the frozen per-test
threshold of {{CONFIRMATORY_FROZEN_ALPHA}} in the direction their development
result recorded, and two yield no discordant pair at all and are therefore not
shown rather than shown absent ([@tbl:confirmatory]). Every registered
direction and threshold prediction held. The remaining {{CONFIRMATORY_MEMBERS_UNRESCORED}} members of the
frozen family were not rescored and remain development-population inference.
The controls, paired counts, and document-clustered intervals are shown together
in [@fig:confirmatory_replication].
The run cannot be repeated: the sentinel claim marks the population spent, and
the recorder refuses a second pass.

| Member | Units | Favored | Opposed | Discordant | p | Crosses frozen threshold | Holm rejects |
|---|---:|---:|---:|---:|---:|---|---|
| Stage 4, derived population | {{CONFIRMATORY_STAGE4_DERIVED_POSITIONS}} positions | {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_FAVORED}} | {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_OPPOSED}} | {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_DISCORDANT_N}} | {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_P}} | {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_CROSSES_FROZEN_THRESHOLD}} | {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_HOLM_REJECTED}} |
| Stage 4, gold population | {{CONFIRMATORY_STAGE4_GOLD_POSITIONS}} positions | {{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_FAVORED}} | {{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_OPPOSED}} | {{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_DISCORDANT_N}} | {{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_P}} | {{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_CROSSES_FROZEN_THRESHOLD}} | {{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_HOLM_REJECTED}} |
| Stage 5, clause type | {{CONFIRMATORY_STAGE5_SENTENCES}} sentences | {{CONFIRMATORY_STAGE5_MCNEMAR_FAVORED}} | {{CONFIRMATORY_STAGE5_MCNEMAR_OPPOSED}} | {{CONFIRMATORY_STAGE5_MCNEMAR_DISCORDANT_N}} | {{CONFIRMATORY_STAGE5_MCNEMAR_P}} | {{CONFIRMATORY_STAGE5_MCNEMAR_CROSSES_FROZEN_THRESHOLD}} | {{CONFIRMATORY_STAGE5_MCNEMAR_HOLM_REJECTED}} |
| Stage 6, observed-position sign test | {{CONFIRMATORY_STAGE6_SENTENCES}} sentences | {{CONFIRMATORY_STAGE6_SIGN_TEST_FAVORED}} | {{CONFIRMATORY_STAGE6_SIGN_TEST_OPPOSED}} | {{CONFIRMATORY_STAGE6_SIGN_TEST_DISCORDANT_N}} | {{CONFIRMATORY_STAGE6_SIGN_TEST_P}} | {{CONFIRMATORY_STAGE6_SIGN_TEST_CROSSES_FROZEN_THRESHOLD}} | {{CONFIRMATORY_STAGE6_SIGN_TEST_HOLM_REJECTED}} |

: The four EWT members of the frozen significance family, each scored once against the whole held-back `en_ewt-ud-test.conllu` split under decision {{CONFIRMATORY_REGISTRATION_DECISION}}, with parameters read unchanged from the development fits. "Favored" counts units where only the model is right (Stages 4 and 5) or where the phase model has the lower loss (Stage 6); "Opposed" counts the reverse. The frozen per-test threshold is {{CONFIRMATORY_FROZEN_ALPHA}}; Holm's step-down is computed over the same {{CONFIRMATORY_FROZEN_FAMILY_SIZE}}-member family. A member with no discordant pair has p fixed at its maximum by construction and is not shown rather than shown absent. {#tbl:confirmatory}

![The held-back replication and its registered controls (decisions {{CONFIRMATORY_REGISTRATION_DECISION}} and {{CONFIRMATORY_STAGE6_CLUSTERED_DECISION}}). Top: exact two-sided p-values for the {{CONFIRMATORY_MEMBERS_SCORED}} scored members, development beside held-back, against the frozen threshold {{CONFIRMATORY_FROZEN_ALPHA}} and each member's Holm threshold over the {{CONFIRMATORY_FROZEN_FAMILY_SIZE}}-member family. The p-axis is truncated at a fixed floor rather than stretched to the smallest measured value: a p below the floor is drawn as a left-pointing triangle pinned at the floor, with its exact magnitude given in that row's own label, so the threshold, both Holm thresholds and the cluster of p-values near one keep real resolution instead of being compressed by a handful of near-machine-epsilon values. Open diamonds mark the registered positive and negative controls; rows with no discordant pair are labeled not computable. Middle: favored and opposed discordant pairs, with discordant/evaluated counts, units, and document counts. Bottom: separate rows for Level 6's held-back primary {{CONFIRMATORY_STAGE6_CLUSTERED_CONFIDENCE_PCT}}% interval, held-back family-corrected {{CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CONFIDENCE_PCT}}% interval, and development primary interval. Only the held-back primary excludes zero. The held-back minimum detectable effect, {{CONFIRMATORY_STAGE6_CLUSTERED_MIN_DETECTABLE_EFFECT_NATS}} nats, exceeds the development effect {{CLUSTER_STAGE6_LOGLOSS_EFFECT}}. The figure refuses to draw if recorded verdicts or computability flags contradict the plotted quantities, discordant counts disagree with their populations, controls fall on the wrong side of the threshold, or exclusion flags disagree with interval bounds.](../../output/figures/confirmatory_replication.png){#fig:confirmatory_replication width=95%}

### Stage 4 on the derived population, against the per-tag argmax baseline

On the derived case population the forward-filtering decoder was compared
against the context-free per-tag argmax baseline over
{{CONFIRMATORY_STAGE4_DERIVED_POSITIONS}} held-back token positions drawn from
{{CONFIRMATORY_STAGE4_DERIVED_DOCUMENTS}} documents. The decoder agrees with the
oracle labels on {{CONFIRMATORY_STAGE4_DERIVED_AGREEMENT_PCT}} percent of
positions against the comparator's
{{CONFIRMATORY_STAGE4_DERIVED_BASELINE_PCT}} percent. The paired exact McNemar
test finds {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_FAVORED}} positions where only
the decoder is right against
{{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_OPPOSED}} where only the baseline is,
giving p = {{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_P}} over
{{CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_DISCORDANT_N}} discordant pairs. The
direction agrees with the development result and the comparison crosses the
frozen threshold, and Holm's step-down correction over the same frozen family
rejects it as well. The limit this result does not escape is the one the derived
population always carried ([@sec:case_gate]): most of the case roles have no morphological ground truth in English, so this is agreement with a
documented syntactic heuristic and never accuracy against ground truth.

### Stage 4 on the gold population, where only marked case roles appear

On the gold population, where labels come from real morphological case marking
and cover only the minority of case roles that English marks, the same paired comparison over
{{CONFIRMATORY_STAGE4_GOLD_POSITIONS}} held-back positions from
{{CONFIRMATORY_STAGE4_GOLD_DOCUMENTS}} documents finds
{{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_DISCORDANT_N}} discordant pairs. The decoder
and the comparator are right on exactly the same positions, both scoring
{{CONFIRMATORY_STAGE4_GOLD_AGREEMENT_PCT}} percent, so there is no direction to
test and p is {{CONFIRMATORY_STAGE4_GOLD_MCNEMAR_P}} by construction. This
reproduces the development outcome of [@sec:gold_population], which was also a
zero-discordance tie, and it has the same character: the gold population's
observation alphabet contains a single symbol, so the emission carries no information, and on this split the fitted
transition did not move the decoder off its comparator. As in
[@sec:gold_population], this is an empirical null, not a structural guarantee. The registration named this
possibility in advance and required it to be recorded as the outcome rather
than designed around. It is a measured null with no resolving power, and it
must be read as not shown, never as evidence of no effect.

### Stage 5 clause type, against the always-majority baseline

The arm rescored here is the retained case-only clause ablation. It is the
member the frozen family names, and the canonical clause level of
[@sec:clause_widened] has no entry in this table, because the population was
already spent when that arm was specified. The case-only mixture classifier was
compared against the always-majority
baseline over {{CONFIRMATORY_STAGE5_SENTENCES}} typed held-back sentences from
{{CONFIRMATORY_STAGE5_DOCUMENTS}} documents. Both reach
{{CONFIRMATORY_STAGE5_CLASSIFICATION_PCT}} percent, and the paired test finds
{{CONFIRMATORY_STAGE5_MCNEMAR_DISCORDANT_N}} discordant pairs, so
p is {{CONFIRMATORY_STAGE5_MCNEMAR_P}}. The mixture assigns the majority type to
every sentence in the population. This is the development result reproduced in
a stronger form on several times the sentences: the development comparison had
a single discordant pair and it favored the baseline. The pre-registration
predicted that this arm would not cross the threshold, and it did not. The honest
reading is that the type signal is not detectable by this
classifier *on this observation alphabet* at this order on this corpus, and the
positive control above shows
that the instrument would have detected a planted effect on the same
population. The widened-observation arm of [@sec:clause_widened] is the
registered response to that reading, and it is now the canonical clause level.
The promotion does not reach back into this row. What the widening changed is
the alphabet, not this result: the case-only comparison stands exactly as
recorded, on the held-back population, with its zero discordant pairs, and it is
the only clause-level statement in this paper that a held-back partition ever
tested.

### Stage 6 document position, against the phase-blind unigram bar

The narrative level's registered observed-position score compares the
phase-conditional type distribution against the phase-blind unigram bar,
sentence by sentence, over {{CONFIRMATORY_STAGE6_SENTENCES}} held-back sentences
from {{CONFIRMATORY_STAGE6_DOCUMENTS}} eligible documents. Mean log loss is
{{CONFIRMATORY_STAGE6_PHASE_LOGLOSS}} nats for the phase model against
{{CONFIRMATORY_STAGE6_UNIGRAM_LOGLOSS}} for the bar, an improvement of
{{CONFIRMATORY_STAGE6_IMPROVEMENT_NATS}} nats per sentence. The exact two-sided
sign test finds {{CONFIRMATORY_STAGE6_SIGN_TEST_FAVORED}} sentences favoring the
phase model against {{CONFIRMATORY_STAGE6_SIGN_TEST_OPPOSED}} favoring the bar,
p = {{CONFIRMATORY_STAGE6_SIGN_TEST_P}}. The direction agrees with the
development result, and unlike the development run this crosses the frozen
threshold, with Holm's correction agreeing. Two limits travel with that. The
effect size is small in absolute terms, a fraction of a nat per sentence, and a
sign test is a statement about direction rather than magnitude. And this is the
sentence-level observed-position estimand, in which the true tercile is supplied
rather than inferred; the document-level clustered estimand is a different
quantity and this result does not speak to it. That quantity was registered
separately and carried to this same population, where its interval excludes zero
([@sec:stage6_heldback_clustered]); the two results agree here, but the
development population illustrates the distinction: its sentence-sign statistic
is nominally favorable but crosses neither corrected threshold, while its
document-clustered log-loss interval spans zero. Neither was allowed to stand in
for the other.
