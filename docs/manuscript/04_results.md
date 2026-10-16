# Connected and Component Results Under Their Registered Comparators {#sec:results}

The connected evaluation is reported first, followed by the historical stage,
generation and individual-channel experiments. Their observations, populations
and scoring routes differ; a component gate result is not an end-to-end
measurement of the connected runtime.

## Connected inference on the locally frozen GUM population {#sec:connected_gum}

[@tbl:connected_gum] reports the paired document result and full scoring population.
[@fig:connected_core] separates the implemented causal schedule from this paired
comparison; its arrows do not attribute the aggregate gain to individual channels.

The connected runtime was evaluated on the previously unexamined portion of a
frozen UD English-GUM test input [@zeldes2017gum]. The locally specified protocol excludes
{{CONNECTED_GUM_EXCLUDED_DOCUMENTS}} documents inspected in the preceding bounded
development run, leaving {{CONNECTED_GUM_DOCUMENTS}} paired documents. This is
local prospective specification within the development effort, not independent
external registration or replication. Model parameters and train-only bridges
remain fixed; the connection strength is {{CONNECTED_GUM_STRENGTH}}. The connected arm is called `integrated` in the recorded data. Its comparator
disables every named channel while retaining the same raw observation encoder,
so it is a channel ablation rather than a wholly independent representation.

Only source `# text` comments enter inference, joined in document order. Source
sentence boundaries are not separately supplied. The runtime lowercases lexical
runs restricted to the American Standard Code for Information Interchange
(ASCII), with punctuation defining its own sentence boundaries, and scores
letters plus a space after every retained token. This normalized stream is not
the original Unicode text or byte stream. UD UPOS, dependency, morphology, case,
phoneme and sentence-type labels are withheld from inference.

| Quantity | Recorded result |
|---|---:|
| Paired documents | {{CONNECTED_GUM_DOCUMENTS}} |
| Source sentences | {{CONNECTED_GUM_SOURCE_SENTENCES}} |
| Source characters | {{CONNECTED_GUM_SOURCE_CHARACTERS}} |
| Normalized lexical tokens | {{CONNECTED_GUM_NORMALIZED_TOKENS}} |
| Scored normalized characters, including token spaces | {{CONNECTED_GUM_SCORED_CHARACTERS}} |
| Integrated mean document-level character log loss | {{CONNECTED_GUM_DOCUMENT_INTEGRATED_LOG_LOSS}} |
| Named-channel-disabled mean document-level character log loss | {{CONNECTED_GUM_DOCUMENT_DISCONNECTED_LOG_LOSS}} |
| Mean paired gain, named-channel-disabled minus integrated | {{CONNECTED_GUM_DOCUMENT_MEAN_GAIN}} |
| Paired document-bootstrap interval | [{{CONNECTED_GUM_DOCUMENT_CI_LOW}}, {{CONNECTED_GUM_DOCUMENT_CI_HIGH}}] |
| In-vocabulary lexical tokens | {{CONNECTED_GUM_IN_VOCAB_TOKENS}} |
| Out-of-vocabulary lexical tokens | {{CONNECTED_GUM_OOV_TOKENS}} |
| Lexical vocabulary coverage | {{CONNECTED_GUM_VOCAB_COVERAGE_PCT}}% |

: Connected versus named-channel-disabled inference on the frozen raw-text population.
Losses and paired gains are in nats per scored normalized character, averaged
equally over documents. The paired percentile interval uses
{{CONNECTED_GUM_BOOTSTRAP_RESAMPLES}} document resamples at
{{CONNECTED_GUM_BOOTSTRAP_CONFIDENCE_PCT}}% coverage with seed
{{CONNECTED_GUM_BOOTSTRAP_SEED}}. Positive gain favors connection; the recorded
interval indicates {{CONNECTED_GUM_OUTCOME_TEXT}}. {#tbl:connected_gum}

The full-stream closed-vocabulary word loss is {{CONNECTED_GUM_WORD_LOSS_TEXT}}.
Vocabulary coverage accompanies the character result so a finite character loss
cannot conceal unsupported words. This comparison does not establish semantic
comprehension, a global exact posterior, or generalization beyond the specified
normalization, population and engineering setting. Historical stage and channel
metrics reported later remain their original independent estimands. Log loss
is appropriate for assessing the probability assigned to the observed outcome
[@gneiting2007strictly], but its propriety does not make differently normalized
streams comparable. Here both arms score the same normalized characters, and
the paired contrast concerns the fixed named-channel intervention as a whole;
it does not identify a contribution from each channel or validate the linguistic
meaning of its states.

The input corpus digest begins `{{CONNECTED_GUM_CORPUS_SHA256_SHORT}}`, the local
protocol digest begins `{{CONNECTED_GUM_PROTOCOL_SHA256_SHORT}}`, and the
execution-input digest begins `{{CONNECTED_GUM_EXECUTION_SHA256_SHORT}}`.
`scripts/record_connected_gum_claims.py` records the complete source/input binding,
per-document paired summaries, exclusions and environment alongside the ledger;
the report digest begins `{{CONNECTED_GUM_REPORT_SHA256_SHORT}}`. Reproduction
requires these exact inputs rather than a moving upstream treebank.


![**Connected conditional schedule and paired GUM result.** Panel A derives
named arrows from the runtime contract. Solid channels are disabled together;
spelling and the dashed train-fitted UPOS encoder remain in both arms. Raw UPOS
is encoded after the word; shared encoding need not give identical outputs.
Case evidence updates clauses after tokens; type evidence updates phase after
sentences. Within-level recursion is omitted. Panel B shows document gains and
their equal-document mean with the locally specified
{{CONNECTED_GUM_BOOTSTRAP_CONFIDENCE_PCT}}% document-bootstrap interval; positive
gain favors connection. Population, normalization, strength and out-of-vocabulary
(OOV) accounting
follow [@tbl:connected_gum]; full-stream word loss remains
{{CONNECTED_GUM_WORD_LOSS_TEXT}}. The figure authenticates typed claims, the
frozen report and execution inputs, then replays the registered statistic.
This is a consistency check, not an independent oracle or new measurement.
Neither a globally exact posterior nor individual channel effects are claimed.
A wrong channel direction, omitted shared observation, changed population, or
mean/interval inconsistent with the bound report falsifies this caption.](../../output/figures/connected_core.png){#fig:connected_core width=100%}

## Historical component gate outcomes and their scoring routes

The {{BUILT_LEVEL_COUNT_WORD}} levels are built and have recorded gate outcomes; Stage 5 passes on the widened observation alphabet
against a bar registered before its artifact existed — a development-session specification with no held-back replication
available, because that partition was already spent ([@sec:clause_widened]) — while the case-only clause model it
displaced is retained as a non-passing ablation with an unresolved privileged-label gain.
Stage 6's primary document-clustered analysis spans zero on the development population and excludes
zero on the held-back one, at the coverage the development analysis uses and not at the family-corrected
one. This section consolidates the sequence-model
measurements of the first two — the letter level of [@sec:letter_level] and the
phoneme level of [@sec:phoneme_level] — where the perplexity ladder and the
information budget live. Both are pymdp `Agent` instances whose
learned sequence statistics live in the transition matrix `B`; at Level 2 the
emission `A` is a learned letter-given-phone likelihood over co-indexed pairs rather
than the identity, so
that level performs inference rather than a state read-off. The later levels'
results, generation, the cross-level channels and learning are reported below as
well, and their instruments are described where they are built: the word and case
levels in [@sec:word_level] and [@sec:case_level], the clause and narrative
levels in [@sec:clause_level] and [@sec:narrative_level], and
the cross-level channels in [@sec:hierarchy], with the historical significance family judged
against its declared corrected threshold in [@sec:multiplicity]. Quantitative figures and these tables consume the same recorded claims, with
source and provenance checks intended to catch stale or inconsistent outputs.
That shared source does not substitute for checking the rendered artifacts.

Two conditions established in [@sec:multiplicity] govern the historical paired
comparisons summarized below,
and neither is repeated at each individual result. The first condition is
about the sampling unit. The historical paired tests are computed at a unit nested
inside the unit its corpus samples — and each has been recomputed at
that outer unit ([@sec:verification]). By construction, this recomputation
never changes the effect size itself; it only changes the interval around it,
and the intervals move in both directions ([@tbl:cluster]): clustering widens
some arms and tightens others. So the nesting is not a uniform penalty, and its
direction could not have been signed without measuring it. The second
condition is about the test family. That family was assembled retrospectively,
on the population every gate in this project was revised against. So a test
crossing the corrected threshold is a corrected-threshold survivor on the
development population, not a confirmed effect on fresh data, except for the
held-back rescoring of [@sec:confirmatory].
Neither condition moves an effect size; both instead bound what the
accompanying inference means.

The tables below report the per-level comparators
([@tbl:letter_comparators], [@tbl:phoneme_comparators], and [@tbl:g2p_comparators]),
restate each as a consolidated ladder, and add the information-budget accounting that
[@sec:gate_design] depends on. A reader following the argument rather than the
construction can read this section and [@sec:gate_design] alone.

The gate audit found that every declared threshold was defective as declared.
Stage 1's threshold sat beyond the information-theoretic ceiling of the
distribution the model was fitted to. Stage 2's sat below the trivial baseline,
against an oracle that does not exist. The later-stage declarations failed
through vacuity, a population-dependent double defect, or mis-scoping. Beyond
these per-stage defects, the audit found two general lessons. First, a
reproducible subsample is not a representative one. Second, a context-free
comparator, though necessary, is not sufficient. Stage 1's ceiling is stated under
the training distribution. The population qualifier is not a hedge: the fitting distribution contains only about half the first-order
information the declared target demands. The target exceeds that
ceiling by a factor of {{STAGE1_CEILING_SHORTFALL_RATIO}}, so no smoothing or
implementation can supply the shortfall from the data the model was given.
Nothing here, however, bounds the held-out passage's own first-order optimum.
The defects are worked through stage by stage below; the re-runnable audit
script, the pre-build checklist, and the Level 3 mis-wiring catalog are set out
in [@sec:audit_trails].

## Level 1: the letter bigram clears the restated unigram bar

### The bigram agent beats the unigram bar

The agent runs the filtering recursion of [@eq:filtering] over held-out prose,
accumulating [@eq:perplexity]. The uniform, unigram, mis-wired, and
training-distribution references bracket it; each is necessary to read the result.

| Comparator | Perplexity | Uses the previous letter? |
|---|---|---|
| Uniform over the alphabet | {{STAGE1_UNIFORM_BASELINE}} | no |
| Context-free unigram letter frequencies | {{STAGE1_UNIGRAM_BASELINE}} | no |
| Mis-wired variant: the bigram matrix in the emission `A` | {{STAGE1_ANTI_INFORMATIVE_PERPLEXITY}} | inverted — yields the *preceding* symbol |
| **Bigram agent: the bigram matrix in the transition `B`** | **{{STAGE1_LETTER_PERPLEXITY}}** | yes |
| Order-1 optimum on the training distribution, $\exp H_{\text{train}}(X_t \mid X_{t-1})$ | {{STAGE1_BIGRAM_FLOOR_PERPLEXITY}} | yes, optimally |

: Level 1 comparators. The uniform, unigram, mis-wired, and shipped-agent rows are measured on the same held-out passage; the conditional-entropy row is a property of the *training* distribution and is not a measurement on that passage. Uniform bounds the trivial end, the unigram row is the gate's actual comparator, the mis-wired row is the live negative control, and the conditional-entropy row is the best any first-order letter model does on the distribution the transition was fitted to. {#tbl:letter_comparators}

The agent beats the context-free unigram baseline by
{{STAGE1_UNIGRAM_MARGIN}} perplexity units, which is the gate. It does not beat
the order-1 reference: {{STAGE1_BIGRAM_FLOOR_PERPLEXITY}} is
[@eq:conditional_entropy] evaluated on the training counts, so it is where an
optimal first-order model lands on the distribution it was fitted to rather than
a bound on what any model scores on this passage ([@sec:background]). In nats
the agent sits at {{STAGE1_LETTER_NATS}} against that reference's
{{STAGE1_BIGRAM_FLOOR_NATS}} — above it, which is the ordinary direction for a
held-out score measured against a training-set optimum.

That is {{STAGE1_LETTER_BITS}} bits per character, a value in the range expected
for a deliberately low-order character model. It is not a replication of a
published figure and must not be read against one: @shannon1951 reports letter
entropy falling as the conditioning context lengthens and estimates the
long-context value through human prediction experiments, over a smaller alphabet
than the one used here, so the longer-context estimates in that literature are
measuring a different quantity. The letter ladder below states the alphabet,
normalization and corpus differences that each move the number.
[@fig:perplexity_ladder] arranges
these comparators and the phoneme level's on one axis; the gap from uniform to
unigram there is the part available without using any context at all, which is
why the gate is stated against the unigram rung.

`compute_perplexity` reports a number produced by `pymdp` `infer_states` and
`update_empirical_prior` calls, one pair per character, not by a shortcut around
them. The value is cross-checked against the closed-form numpy Markov chain in
`src/ailm/verification/oracle.py` to the tolerance pinned in
`tests/test_stage1_gate.py`, so the perplexity reflects `pymdp` actually
performing the inference — and a mis-wiring shows up there as a disagreement of
roughly the distance between the mis-wired and correct rows of
[@tbl:letter_comparators], which is a far louder signal than either number alone.

### Restated unigram gate and its controls

The level's gate is the context-free unigram comparison, which is both
reachable and diagnostic. A model can clear uniform while carrying no context
whatsoever. Clearing the unigram rung is consistent with conditioning on the
previous letter, but it does not prove that conditioning is happening: the bar
is estimated on training counts and scored on a finite passage, so another
context-free distribution can land below it through distribution shift alone.
What carries the attribution is a separate context-destruction control. It
takes the same trained conditionals and rotates them onto the wrong contexts;
that rotated model scores {{L1_CONTEXT_DERANGEMENT_PERPLEXITY}} against the
unigram bar's {{L1_CONTEXT_UNIGRAM_PERPLEXITY}} on the identical positions. So
the gate's margin does not survive destroying the context-to-conditional
assignment, even while every individual conditional distribution is kept
intact. The unigram rung and that control together test the thing the level
claims to do.

The alternative wording of the gate — a gain of at least
{{STAGE1_DECLARED_GATE_NATS}} nats over uniform — asks for more first-order
information than the training distribution holds. The argument for this takes
one pass over the counts. The gain a first-order model can realize on the
distribution it is fitted to is at most the difference between the uniform
entropy and the conditional entropy of [@eq:conditional_entropy]. On the Brown
counts, that difference is {{STAGE1_MAX_BIGRAM_GAIN_NATS}} nats, so the
{{STAGE1_DECLARED_GATE_NATS}}-nat target exceeds the level's entire order-1
information budget by a factor of {{STAGE1_CEILING_SHORTFALL_RATIO}}. The bar
was calibrated for a model class this level does not implement. The shortfall
is in the order of the model rather than in the code, so no implementation,
tuning choice, or smoothing constant can reach it *on that distribution*. The
held-out passage is a different population, though, and needs its own bound.
[@eq:heldout_bound] supplies that bound: it caps the order-1 gain available
across all {{STAGE1_HELDOUT_POSITIONS}} positions this level is scored on at
{{STAGE1_HELDOUT_MAX_GAIN_NATS}} nats, including the fixed uniform-start
prediction, and this bound is also short of the target. So the conclusion
survives on both populations rather than being carried from one to the other.
Against the training distribution's {{STAGE1_MAX_BIGRAM_GAIN_NATS}}-nat budget, the agent realizes {{STAGE1_GAIN_OVER_UNIFORM_NATS}} nats
on held-out prose, or {{STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT}}% of it — a failure
under the {{STAGE1_DECLARED_GATE_NATS}}-nat wording, and a result close to the
training distribution's first-order optimum. Measured against the bound on the
passage it is actually scored on, the same realized gain is
{{STAGE1_HELDOUT_SIGNAL_CAPTURED_PCT}}%. That is the same-population figure,
and the one to quote where a same-population statement is wanted. The first
percentage, by contrast, crosses populations — a held-out gain measured
against a training-set budget — and it is reported as a descriptive ratio for
exactly that reason, not as a bound the agent is proved to respect.
[@fig:information_ceiling] draws both readings on the same nat axis.
[@sec:gate_design] treats this as the worked example of a threshold beyond its
measured ceiling, and states why the wording must not be inherited by a later
stage.

### Letter perplexity from uniform to the empirical first-order optimum

This table restates Level 1's comparators as a single ladder, from a model
using no context at all up to the best any first-order model can do.

| Comparator | Perplexity | Uses the previous symbol? |
|---|---|---|
| Uniform over the letter alphabet | {{STAGE1_UNIFORM_BASELINE}} | no |
| Context-free unigram letter frequencies | {{STAGE1_UNIGRAM_BASELINE}} | no |
| Mis-wired variant: bigram matrix in the emission `A` | {{STAGE1_ANTI_INFORMATIVE_PERPLEXITY}} | inverted — yields the *preceding* symbol |
| **Bigram agent (`A` identity, `B` learned)** | **{{STAGE1_LETTER_PERPLEXITY}}** | yes |
| Empirical order-1 optimum | {{STAGE1_BIGRAM_FLOOR_PERPLEXITY}} | yes, optimally |

: Letter-level perplexity on held-out prose, Brown corpus bigram statistics ({{STAGE1_BROWN_BIGRAM_TYPES}} bigram types over {{STAGE1_BROWN_BIGRAM_TOKENS}} bigram tokens). The gate is beating the context-free unigram baseline, a margin of {{STAGE1_UNIGRAM_MARGIN}} perplexity units; uniform is reported only to show what a crippled comparator looks like, since a context-free model clears it. {#tbl:letter_ladder}

[@fig:perplexity_ladder] plots [@tbl:letter_ladder] alongside its Level 2
counterpart on a shared axis, which is where the structural similarity between the
two levels is easiest to read: in both cases the model sits close to the empirical
order-1 optimum and far from uniform, and the interval between unigram and
optimum — not the
interval between uniform and model — is the space in which a bigram model can
distinguish itself.

![Perplexity comparators for both built levels, each on its own level's evaluation stream: uniform, context-free unigram, the trained bigram agent, and the empirical order-1 optimum on that stream. The two panels are separate populations and the panel does not compare them to each other — the Stage 1 letter panel is measured on held-out prose, while the Stage 2 phoneme panel is measured in-sample over the full dictionary, on the same pronunciations that trained the transition matrix ([@sec:limitations]), so the two ladders are not comparable as generalization claims. Each panel therefore states the population it was scored over and its unit beneath its own title, because two panels on one shared axis otherwise invite a rung-for-rung reading their populations do not support. The order-1 conditional-entropy floor is named on the canvas with its value rather than only in this caption, and its rule stops short of the panel edge so the label sits clear of it. The model-to-optimum gap is the remaining headroom at this order; the uniform-to-unigram gap is available to a model with no context at all. A missing comparator, a non-monotone ladder where the recorded identities require ordering, a model-to-optimum gap inconsistent with the ledger, or a panel drawn over an empty population would falsify this panel.](../../output/figures/perplexity_ladder.png){#fig:perplexity_ladder width=85%}

{{STAGE1_LETTER_BITS}} bits per character sits in the range of published
estimates of English letter entropy at low context order. @shannon1951 measures
that entropy dropping as the conditioning context lengthens, and later bounded
estimates place fully contextual English well below any order-1 figure
[@cover1978gambling; @brown1992entropy]; the value here belongs to the shallow end
of that progression, which is where an order-1 model should land. This is a
consistency check, not a replication, and it is worth being precise about why it
cannot be one. The alphabet here adds space, period, apostrophe, and hyphen to the
{{STAGE0_WORD_LETTERS}} letters, so uniform over it is {{STAGE1_UNIFORM_BASELINE}} rather than Shannon's
27; the normalization collapses whitespace runs and strips spaces before `.` and
`'`; the training corpus is Brown [@francis1979brown; @bird2009nltk] and
the evaluation is held-out prose. Any of those choices moves the number, so the
agreement establishes that nothing is badly wrong rather than that a published
figure has been reproduced.

```{=latex}
\csname @ifclassloaded\endcsname{beamer}{}{\Needspace{20\baselineskip}}
```

## Level 2: filtering beats the context-free bar, by a thin decoding margin

### Part A: the phoneme sequence model beats its context-free bar

This table reports the phoneme sequence model's in-sample comparators against
uniform, unigram, and the order-1 optimum, on the full dictionary.

| Comparator | Perplexity | Uses the previous phone? |
|---|---|---|
| Uniform over the {{PHONEME_COUNT_39}} phones | {{STAGE2_PHONEME_UNIFORM_BASELINE}} | no |
| Context-free unigram phone frequencies | {{STAGE2_PHONEME_UNIGRAM_BASELINE}} | no |
| **Phoneme bigram agent** | **{{STAGE2_PHONEME_PERPLEXITY}}** | yes |
| Exact all-position order-1 optimum | {{STAGE2_PHONEME_ALL_POSITION_FLOOR_PERPLEXITY}} | yes, optimally |

: Part A perplexity over all {{STAGE2_PHONEME_EVAL_POSITIONS}} phone positions in the dictionary. This is a descriptive **in-sample dictionary protocol**: the same primary pronunciations supply the counts and the scored sufficient statistics, so it is not held-out generalization. The agent clears the context-free bar by {{STAGE2_PHONEME_UNIGRAM_MARGIN}} perplexity units. The exact optimum is the unsmoothed maximum-likelihood start distribution plus transition columns on this identical all-position denominator, including every independently reset word start; it is therefore the model-class bound used by the gate audit. The lower transitions-only diagnostic, {{STAGE2_PHONEME_FLOOR_PERPLEXITY}}, omits starts and is retained only as an optimistic reachability bound, never as the threshold for the all-position score. {#tbl:phoneme_comparators}

The same reachability argument as Level 1 applies with less room. The total gain
available over uniform at this level is
{{STAGE2_PHONEME_ALL_POSITION_MAX_GAIN_NATS}} nats on the exact all-position
population (and {{STAGE2_PHONEME_MAX_GAIN_NATS}} nats under the more optimistic
transitions-only relaxation), both below Level 1's inherited
{{STAGE1_DECLARED_GATE_NATS}}-nat threshold, so that wording is
unreachable here before a line of code is written. [@fig:perplexity_ladder]
shows both levels' rungs together.

### Family-disjoint evaluation and lexical overlap

The shared-prefix bound is also measured from the inside rather than only
stated. A family-disjoint arm assigns whole
{{STAGE2_G2P_FAMSPLIT_PREFIX_LENGTH}}-character prefix families to one side of
the split, using the same seedless stride construction lifted one level up:
every {{STAGE2_G2P_FAMSPLIT_FAMILY_STRIDE_ORDINAL}} sorted family is held out.
As a result, no held-out word of length {{STAGE2_G2P_FAMSPLIT_PREFIX_LENGTH}}
or more shares its {{STAGE2_G2P_FAMSPLIT_PREFIX_LENGTH}}-character prefix with any training word
({{STAGE2_G2P_FAMSPLIT_SHARED_PREFIX_PCT}}% shared, by construction, against
the interleave's {{STAGE2_G2P_SPLIT_SHARED_PREFIX_PCT}}%). Re-fitting the
identical decoder and the identical context-free lookup on the disjoint train
side gives {{STAGE2_G2P_FAMSPLIT_ACCURACY}}% against
{{STAGE2_G2P_FAMSPLIT_BASELINE_ACCURACY}}% over
{{STAGE2_G2P_FAMSPLIT_EVAL_POSITIONS}} positions — a margin of
{{STAGE2_G2P_FAMSPLIT_MARGIN_PTS}} points beside the interleave's
{{STAGE2_G2P_MARGIN_PTS}}, with discordant counts
{{STAGE2_G2P_FAMSPLIT_MCNEMAR_B}} against {{STAGE2_G2P_FAMSPLIT_MCNEMAR_C}} and
exact McNemar *p* = {{STAGE2_G2P_FAMSPLIT_MCNEMAR_P}}. This is an exploratory
robustness arm: its direction was prospectively specified within the
development session, but its p-value stays outside the frozen significance
family, so it is a bound rather than a second significance claim. Read as a
bound, it cuts both ways. Morphological-neighbor transfer accounts for part of
the interleave margin, but most of the margin survives with the neighbors
removed, so the left-context mechanism is not an artifact of `catalog`
teaching `catalogs`. The disjoint assignment has its own recorded price,
though: holding out whole alphabetical families moves the initial-letter total
variation to {{STAGE2_G2P_FAMSPLIT_INITIAL_TVD}}, well above the interleave's
{{STAGE2_G2P_SPLIT_INITIAL_TVD}}. That is why this arm bounds the shipped
result rather than replacing it.

The bound is itself partitioned. The same disjoint held-out set can be grouped
by morphological family — the identical four-character family key the split
assigns sides with — and each family can then be scored against the same
context-free bar with the same paired test. Doing so asks whether the disjoint
margin is a level-wide property or the average of unlike parts. The answer,
prospectively specified within the development session in the arm's decision
stub, is that it is not level-wide. Across the {{STAGE2_G2P_FAMGRP_FAMILIES}}
held-out families, {{STAGE2_G2P_FAMGRP_EVALUATED_FAMILIES}} carry at least
{{STAGE2_G2P_FAMGRP_MIN_POSITIONS}} scored positions, and only
{{STAGE2_G2P_FAMGRP_POSITIVE_MARGIN_FAMILIES}} of them show a positive margin
— a share of {{STAGE2_G2P_FAMGRP_POSITIVE_MARGIN_SHARE}}, against the stub's
registered bar of more than one half. The pooled disjoint margin of
{{STAGE2_G2P_FAMGRP_POOL_MARGIN_PTS}} points is therefore a majority-null
average, not a property most families carry. The largest evaluated family,
`{{STAGE2_G2P_FAMGRP_RANK1_FAMILY}}`, scores
{{STAGE2_G2P_FAMGRP_RANK1_ACCURACY}}% against
{{STAGE2_G2P_FAMGRP_RANK1_BASELINE_ACCURACY}}% over
{{STAGE2_G2P_FAMGRP_RANK1_POSITIONS}} positions (margin
{{STAGE2_G2P_FAMGRP_RANK1_MARGIN_PTS}} points, exact *p* =
{{STAGE2_G2P_FAMGRP_RANK1_MCNEMAR_P}}). By contrast, the
{{STAGE2_G2P_FAMGRP_RANK10_FAMILY}} family's margin is
{{STAGE2_G2P_FAMGRP_RANK10_MARGIN_PTS}} points over
{{STAGE2_G2P_FAMGRP_RANK10_POSITIONS}} positions. This measured heterogeneity
is real rather than an artifact of the scoring, for two reasons. First,
position-weighted, the per-family scores reproduce the disjoint arm exactly
({{STAGE2_G2P_FAMGRP_POOL_ACCURACY}}% over {{STAGE2_G2P_FAMGRP_POOL_POSITIONS}}
positions, the prospectively specified positive control). Second, a
deterministic reversal control — reversed pooled correctness vectors paired
with the same family sizes — reaches a largest per-family margin of only
{{STAGE2_G2P_FAMGRP_REVERSAL_MAX_ABS_MARGIN}} points, against the real
assignment's {{STAGE2_G2P_FAMGRP_REAL_MAX_ABS_MARGIN}}. Read together with the
disjoint bound above, the family partition sharpens it: the surviving margin
is concentrated in a minority of families, which is one more reason the
length-matched subset is not English and accuracy figures on it do not
transfer. Per-family p-values are diagnostics of a diagnostic arm and join no
significance family; this partition reports the shape of the margin across
the dictionary, not a new significance claim.

### Forward filtering beats the context-free baseline

This table compares the context-free lookup, the filtering decoder, and the
whole-word smoothing and Viterbi references on the same held-out positions.

```{=latex}
\csname @ifclassloaded\endcsname{beamer}{}{\Needspace{22\baselineskip}}
```

| Decoder | Position accuracy (%) | Context used |
|---|---|---|
| Context-free per-letter $\operatorname*{arg\,max}_p\, p(\text{phone} \mid \text{letter})$ | {{STAGE2_G2P_BASELINE_ACCURACY}} | none |
| **Full-set NumPy forward filtering, after a bounded `pymdp` check with {{STAGE2_DECODER_DISAGREEMENTS}} decision disagreements** | **{{STAGE2_G2P_ACCURACY}}** | left only |
| Forward-backward smoothing | {{STAGE2_G2P_SMOOTHING_CEILING}} | whole word |
| Viterbi maximum-a-posteriori path | {{STAGE2_G2P_VITERBI_CEILING}} | whole word |

: Position-aligned letter-to-phone accuracy over {{STAGE2_G2P_EVAL_POSITIONS}} held-out positions. Each position is scored against the phone the index pairing assigns to that letter, so the target inherits the alignment stipulation of [@sec:phoneme_level]. The gate compares the context-free lookup with forward filtering; smoothing and Viterbi are whole-word *references*, not ceilings. They use information the streaming architecture forgoes and they are not upper bounds on what a same-information model could reach — at Stage 4 a reference of the same kind is exceeded by the shipped model. They are recorded so the filtering agent's shortfall is visible rather than hidden. {#tbl:g2p_comparators}

On a length-matched subset the co-indexed letter alone is nearly sufficient, which
is precisely why the context-free baseline is already strong and why the headroom
for context is thin. Filtering captures part of that headroom and no more,
because `infer_states` conditions on observations up to the current position and
never revisits them. Smoothing and Viterbi see the whole word and gain several
points, as [@tbl:g2p_comparators] records. Reporting a smoothed figure would
flatter the gate while describing a different whole-word computation. The
canonical full-set filtering statistic is evaluated by the NumPy oracle only
after a bounded live-pymdp comparison finds {{STAGE2_DECODER_DISAGREEMENTS}} decision disagreements; no
shipped Level 3 path consumes Level 2's posterior. [@fig:g2p_accuracy] plots the baseline, the filtering decoder and the smoothing reference, and the
baseline and the filtering agent are visually indistinguishable at full scale —
which is the honest picture of a {{STAGE2_G2P_MARGIN_PTS}}-point margin and the
reason the next subsection exists.

### Decoding margin and the lookup comparator

Before the margin is read at all, the baseline comparison bounds what it can mean. A
left-context lookup over the observed symbols, with
{{COMPARATOR_STAGE2_LOOKUP_PARAMETERS}} parameters, no latent state and no
inference at decode time, scores {{COMPARATOR_STAGE2_LOOKUP_ACCURACY}}% on this
same population against the shipped model's {{STAGE2_G2P_ACCURACY}}%. It does not
merely clear the declared bar; it beats the model by
{{COMPARATOR_STAGE2_OVER_MODEL_PTS}} points, several times the margin this
section is about. Clearing a context-free comparator therefore does not show that
latent-state filtering is what produced the gain ([@sec:comparator_strength]).

McNemar's test conditions on the observed count of discordant pairs. The
discordant-pair interval constructed from it retains that conditioning; it is
not an unconditional interval for the difference in marginal accuracies
[@mcnemar1947; @newcombe1998paired]. The margin is
{{STAGE2_G2P_MARGIN_PTS}} accuracy points, and the gate does not
rest it on a comparison of aggregates. It compares the decoders in paired form
instead: `decode_positions` and `baseline_positions` line them up position for
position, and the test runs only on the positions where the two disagree.

That test is the exact conditional form of McNemar's test [@mcnemar1947]. Its
statistic is the two-sided exact binomial of [@eq:mcnemar], not the $\chi^2$
approximation to it. Over {{STAGE2_G2P_EVAL_POSITIONS}} held-out positions there
are {{STAGE2_MCNEMAR_TOTAL_DISCORDANT}} discordant pairs:
{{STAGE2_G2P_MCNEMAR_B}} where only filtering is correct against
{{STAGE2_G2P_MCNEMAR_C}} where only the baseline is,
*p* = {{STAGE2_G2P_MCNEMAR_P}}. The pairing is what carries that result: at
{{STAGE2_G2P_MARGIN_PTS}} points the difference is too small for an
independent-proportion approximation to bound, so the uncertainty has to be
evaluated on the paired predictions rather than on two aggregates.
[@sec:verification] gives the reason two aggregates cannot establish a margin
this thin, derives the exact form, and carries the identical-predictors control
that keeps the test itself falsifiable.

Significance here depends on using the whole held-out set. `_iter_pairs` in
`src/ailm/levels/phoneme.py` takes an evenly spaced stride and never a prefix
wherever the evaluation has to be subsampled at all. The pair list is sorted
alphabetically, so a prefix is only the words beginning with the first few
letters. Even the unbiased stride is not enough here: over
{{STAGE2_G2P_SUBSAMPLE_POSITIONS}} positions it reaches only
*p* = {{STAGE2_G2P_SUBSAMPLE_MCNEMAR_P}}, orders of magnitude weaker than the
full-set result and short of the gate's bar. [@fig:subsample_bias] plots both
failure modes against the full set and [@sec:gate_design] generalizes them.

Shrinking the evaluation is therefore not an option, and a full-set `pymdp` pass
is tens of thousands of `infer_states` calls.
`scripts/record_stage2_claims.py` computes the full-set statistics with the
implementation-independent NumPy reference decoder (the oracle) instead. Reading those figures as properties of the
`pymdp` agent is licensed by the {{STAGE2_DECODER_DISAGREEMENTS}}-disagreement result of
[@eq:decoder_equivalence], which [@sec:verification] states, measures, and
enforces at record time.

### Learning an unobserved emission does not improve decoding

Decision {{DECISION_UNOBSERVED_EMISSION_LEARNING}} tested a different question from the supervised gate above: whether
`pymdp`'s Dirichlet observation-likelihood learning could recover a useful
letter-to-phone emission when the phone states were deliberately **unobserved**.
The target is the same co-indexed relation the gate uses, so the experiment
inherits the alignment stipulation and cannot escape it by learning.
The lane trained on {{G2P_LEARN_A_TRAIN_PAIRS}} pairs and evaluated
{{G2P_LEARN_A_EVAL_POSITIONS}} positions, using an evenly spaced stride of
{{G2P_LEARN_A_LEARNING_STRIDE}} for the recorded experiment. The fixed comparator
is the recorded add-{{MODEL_DEFAULT_EMISSION_SMOOTHING}} count emission from the same lane, not a full-corpus count.

| Decoder or control | Held-out position accuracy (%) |
|---|---:|
| Latent-state emission learning | {{G2P_LEARN_A_LEARNED_ACCURACY}} |
| Same-lane fixed count emission | {{G2P_LEARN_A_FIXED_ACCURACY}} |
| Observed-state P4 gold-lane control | {{G2P_LEARN_A_ORACLE_GOLD_LANE_ACCURACY}} |

The learned emission moved by {{G2P_LEARN_A_MAX_MOVEMENT}} in its largest entry,
but its decoding accuracy did not merely fail to improve on the fixed control: it
collapsed, from {{G2P_LEARN_A_FIXED_ACCURACY}} to {{G2P_LEARN_A_LEARNED_ACCURACY}}.
The observed-state
control recovered the same-lane count emission to a maximum absolute difference of
{{G2P_LEARN_A_GOLD_MAX_ABS_DELTA}}, and the independent oracle and `pymdp` decoders
disagreed on {{G2P_LEARN_A_DECODER_DISAGREEMENTS}} positions. The result is therefore
**NO IMPROVEMENT / QUALIFIED**: this experiment demonstrates that the learning path
is active and that its observed-state control is correct, not that unobserved-phone
learning recovers the co-indexed emission — and certainly not
grapheme-to-phoneme conversion, which no lane at this level attempts. The result
is recorded under decision {{DECISION_UNOBSERVED_EMISSION_LEARNING}} and its limitations are carried into
[@sec:limitations].

### Either the emission or the transition can carry the placement error

Level 2 has a learned emission and a learned transition, so either slot can carry
the placement error stated as a proposition in [@sec:architecture]. The obvious
signature of a mis-wiring is that accuracy drops below the context-free bar: a
filtering decode that cannot beat a lookup table is broken, not merely weak. A
time-reversed transition and an information-free uniform emission were both
caught that way.

The third mis-wiring was not. An emission normalized over phones instead of
letters, then renormalized so that `pymdp` still validates it, cleared the
context-free bar while being wrong. Beating the baseline is therefore necessary
and not sufficient. The following guards actually pin the emission down: first, the
column-sum invariant on `A` before any renormalization; second, numerical
agreement with the closed-form oracle. Both are asserted in
`tests/test_stage2_gate.py` and `tests/test_phoneme.py`; [@sec:verification]
describes why the second one is the instrument the whole project now depends on.

### Phoneme and letter-to-phone comparison ladders

Part A is the phoneme sequence model, the Level 1 construction one level down over
the {{PHONEME_COUNT_39}} canonical ARPAbet phones.

| Comparator | Perplexity |
|---|---|
| Uniform over {{PHONEME_COUNT_39}} phones | {{STAGE2_PHONEME_UNIFORM_BASELINE}} |
| Context-free unigram phone frequencies | {{STAGE2_PHONEME_UNIGRAM_BASELINE}} |
| **Phoneme bigram agent** | **{{STAGE2_PHONEME_PERPLEXITY}}** |
| Exact all-position order-1 optimum | {{STAGE2_PHONEME_ALL_POSITION_FLOOR_PERPLEXITY}} |

: Phoneme sequence perplexity, measured **in-sample**: the evaluation positions are the pronunciations that trained the transition matrix, so this row is not a generalization claim and is not comparable to the held-out Level 1 figures ([@sec:limitations] states the defense and its limit). The margin over the context-free bar is {{STAGE2_PHONEME_UNIGRAM_MARGIN}} perplexity units — comfortable, unlike Part B. `scripts/record_stage2_claims.py` computes this in closed form over all {{STAGE2_PHONEME_EVAL_POSITIONS}} phone positions of the primary pronunciation of every CMUdict base word [@cmudict], with the prior reset at each word boundary. {#tbl:phoneme_sequence}

The exact all-position gain available over uniform at this level is
{{STAGE2_PHONEME_ALL_POSITION_MAX_GAIN_NATS}} nats, below the inherited Level 1 threshold of
{{STAGE1_DECLARED_GATE_NATS}} nats, so that wording is unreachable here too,
and for the same reason.
The per-word prior reset matters and is a modeling decision rather than a
convenience: position zero of a word has no predecessor and is therefore paid for
under the unigram prior `D`, which is the honest language-model accounting.

Part B is equal-length position-aligned letter-to-phone decoding — the first level
whose emission is not the identity, so the posterior is an inference rather than a
read-off. It is not grapheme-to-phoneme conversion, and a non-identity emission is
not a message between levels ([@sec:phoneme_level], [@sec:architecture]). Letters
are observed and phones inferred through a learned emission.

| Method | Position accuracy | Context used |
|---|---|---|
| Context-free per-letter $\arg\max_p P(\text{phone } p \mid \text{letter } l)$ | {{STAGE2_G2P_BASELINE_ACCURACY}}% | none |
| **Full-set NumPy forward filtering after a bounded pymdp check with {{STAGE2_DECODER_DISAGREEMENTS}} decision disagreements** | **{{STAGE2_G2P_ACCURACY}}%** | left context only |
| Forward–backward smoothing reference | {{STAGE2_G2P_SMOOTHING_CEILING}}% | whole word |
| Viterbi maximum a posteriori (MAP) path | {{STAGE2_G2P_VITERBI_CEILING}}% | whole word |

: Position-aligned letter-to-phone accuracy over {{STAGE2_G2P_EVAL_POSITIONS}} held-out positions from {{STAGE2_G2P_TEST_PAIRS}} test words, trained on {{STAGE2_G2P_TRAIN_PAIRS}} length-matched pairs. The gate compares the context-free lookup with forward filtering: a margin of {{STAGE2_G2P_MARGIN_PTS}} points, established by the paired test of [@eq:mcnemar] rather than by the difference of the aggregates. The smoothing and Viterbi rows are headroom the online architecture deliberately does not claim — see [@sec:limitations]. {#tbl:g2p_accuracy}

![**What to see: the letter-to-phone decoder beats a context-free per-letter lookup table by a thin but paired-significant margin, while the whole-word smoothing and Viterbi references shown alongside it are unreachable headroom the streaming architecture does not claim.** Position-aligned letter-to-phone accuracy on the Stage 2 held-out split — {{STAGE2_G2P_EVAL_POSITIONS}} aligned phone positions across {{STAGE2_G2P_TEST_PAIRS}} words. The context-free per-letter baseline, the fitted filtering model scored over the full set by the NumPy oracle after a bounded pymdp check with {{STAGE2_DECODER_DISAGREEMENTS}} decision disagreements, and the forward–backward smoothing reference; the Viterbi reference is tabulated in [@tbl:g2p_accuracy] rather than plotted. Panel A is drawn at full scale and panel B repeats the same three bars on a truncated axis so the margin is visible; the dotted rule marks the uniform-guess comparator and the box reports the paired McNemar counts. The gate margin is the small step from baseline to filtering; the larger step to the whole-word references requires access the streaming architecture does not have, which is why they are references rather than ceilings: they are not upper bounds on what a same-information model could reach. A missing decoder, a filtering value that disagreed with the live-Agent check, or a whole-word reference below the corresponding causal decoder would falsify this panel. Each decoder bar carries a descriptive {{BOUND_CONFIDENCE_PCT}}% Wilson interval [@wilson1927] derived at render time from its recorded accuracy and the {{STAGE2_G2P_EVAL_POSITIONS}}-position held-out denominator both decoders were scored on — the recorder pairs them position-for-position for the McNemar counts, so one denominator serves both bars — stated per the decision-101 convention on that population, in percentage points of position accuracy. The two marginal intervals can overlap even where the paired test separates the decoders, which is why the margin is read from the McNemar box rather than off the bars; the whole-word references draw bare, being references the streaming architecture does not claim rather than arms of the paired comparison. ](../../output/figures/g2p_accuracy.png){#fig:g2p_accuracy width=85%}

The margin in [@tbl:g2p_accuracy] is thin by construction, not by accident. The
length-matched training subset is biased toward transparent orthography, where the
letter alone is nearly sufficient, so the context-free baseline is already strong
and the headroom for context is small; [@sec:phoneme_level] prices that subset and
states what it costs in generality. Forward filtering captures part of that
headroom and nothing more. Both facts — the thin margin and its cause — are
load-bearing for [@sec:limitations].

## Level 3: the letter-bigram bar admits a wrong word prior

The level relocates where the learned content sits — both transitions are
structural constants and the single learned quantity is the prior over word
identity — so the mis-wiring risk moves with it, and the accuracy bar does not
catch a wrong prior; the recorded catalog makes that case in
[@sec:word_gate]. The closed form stays exact at any vocabulary size, the gate
stays bracketed at every rung of the lexicon-scaling ladder, and the dense
`pymdp` transition stops fitting in memory far below the closed form's reach.
The mis-wiring figure, the scaling ladder with its two cautions, and the
run-versus-measure boundary are set out in [@sec:audit_trails].

## Level 4: the declared gate fails against both populations at once

### The declared gate fails in opposite directions on each population

Level 4's declared gate fails against both populations at once, in opposite
directions. [@fig:case_both_defects] puts the two side by side;
[@sec:case_gate] works through why one threshold can be simultaneously vacuous
and above-reference.

![Stage 4's single declared threshold against both populations. On the committed gold-morphology split every measured predictor — majority class, context-free bar and fitted sequence model — lands on the *same* value; the threshold is therefore cleared by a constant predictor and is VACUOUS. That equality is an empirical result of this fitted transition and split, not a consequence of the one-tag observation alphabet. On the derived oracle the threshold sits above every measured comparator — all of them exceedable, so the verdict is ABOVE_REFERENCE rather than an impossibility claim. Only the population differs. If the two populations were merged, if the constant predictor did not match the gold bar, or if the derived-oracle verdict crossed the declared threshold, this panel would be falsified.](../../output/figures/case_both_defects.png){#fig:case_both_defects width=92%}

### The gold population cannot carry a gate {#sec:gold_population}

After the raw document split, the gold population holds
{{STAGE4_GOLD_TEST_TOKENS}} tokens in {{STAGE4_GOLD_TEST_SENTENCES}} sequences
from {{STAGE4_GOLD_TEST_DOCUMENTS}} contributing documents. On every one of
them the case-transition
model and the context-free baseline make **identical** predictions: the paired
test finds {{STAGE4_GOLD_MCNEMAR_B}} positions where only the model is right and
{{STAGE4_GOLD_MCNEMAR_C}} where only the baseline is. Model, baseline,
forward-backward smoothing, Viterbi and a majority-class constant all score
{{STAGE4_GOLD_AGREEMENT}}%.

Every gold-morphology token in the vendored EWT split is a pronoun, so the gold
observation alphabet has size {{STAGE4_GOLD_UPOS_ALPHABET_SIZE}}. A constant
observation carries no state information through the emission. It does not,
however, erase the learned transition: filtering still evolves the prior through
that transition, and a one-tag population with alternating case dynamics can
make the sequence decoder disagree with a context-free predictor. The identical
predictions in the committed split are therefore an empirical null of its fitted
transition, start prior, and held-out sequence labels, not a structural theorem
or an impossibility result. Reported rather than omitted, it shows that this
English gold arm does not support the shipped gate; an independent population
could produce a different paired outcome.

Stage 4's gate is therefore stated over the derived population only, and the
gold-population result is reported as the null it is. [@sec:case_gate] takes up
what happens to the declared threshold when it meets both populations at once.

### Derived case agreement by stratum

Stating the gate over the derived population buys all
{{STAGE4_DERIVED_CASES_PRESENT}} roles at the cost of a labeling this project has
not validated. That cost has a shape, and the shape is measurable, because the
oracle assigns each label by paths applied in strict precedence:
morphological `feats[Case]` where the corpus marks it, then a deterministic
dependency-relation map, then the ADP token's own lemma. The final route labels
that token and never transfers the role to its head. The paths are
recomputed from the treebank purely to partition the evaluation and never reach
the model, which observes the UPOS tag and nothing else.

Partitioned that way ([@tbl:case_oracle_paths]), the derived margin of
{{STAGE4_DERIVED_MARGIN_PTS}} points is not a property of the level. It is a
property of one stratum. A second bound sits beside that one: a
{{COMPARATOR_STAGE4_LOOKUP_PARAMETERS}}-parameter lookup over the observed tags
scores {{COMPARATOR_STAGE4_LOOKUP_ACCURACY}}% against this level's
{{STAGE4_DERIVED_AGREEMENT}}%, so the margin is bounded both by where it lives
and by what a table already achieves without any latent state
([@sec:comparator_strength]).

The agreement itself is computed by a NumPy forward filter, and the live
Agent's `infer_states` / `update_empirical_prior` recursion is compared with
that filter over the entire held-out split rather than a prefix: across all
{{STAGE4_PYMDP_SENTENCES}} test sentences ({{STAGE4_PYMDP_POSITIONS}} positions)
the two agree on the case-role posterior argmax with
{{STAGE4_PYMDP_ARGMAX_DISAGREEMENTS}} disagreements, and the largest absolute
deviation between the two posteriors at any position is
{{STAGE4_PYMDP_POSTERIOR_MAX_ABS_DEV}}. The recorder refuses to write if either
statistic moves, so the attribution of the reported agreement to the pymdp
Agent is exhaustive over the scored population while the headline path remains
the NumPy filter ([@sec:verification], [@fig:inference_evidence]).

| oracle path | tokens | baseline | model | margin | discordant | *p* |
|---|---:|---:|---:|---:|---:|---:|
| dependency relation | {{STAGE4_PATH_DEPREL_TOKENS}} | {{STAGE4_PATH_DEPREL_BASELINE}}% | {{STAGE4_PATH_DEPREL_AGREEMENT}}% | **{{STAGE4_PATH_DEPREL_MARGIN_PTS}}** | {{STAGE4_PATH_DEPREL_DISCORDANT_N}} | {{STAGE4_PATH_DEPREL_MCNEMAR_P}} |
| morphology (control) | {{STAGE4_PATH_MORPHOLOGY_TOKENS}} | {{STAGE4_PATH_MORPHOLOGY_BASELINE}}% | {{STAGE4_PATH_MORPHOLOGY_AGREEMENT}}% | {{STAGE4_PATH_MORPHOLOGY_MARGIN_PTS}} | {{STAGE4_PATH_MORPHOLOGY_DISCORDANT_N}} | {{STAGE4_PATH_MORPHOLOGY_MCNEMAR_P}} |
| adposition token | {{STAGE4_PATH_ADPOSITION_TOKENS}} | {{STAGE4_PATH_ADPOSITION_BASELINE}}% | {{STAGE4_PATH_ADPOSITION_AGREEMENT}}% | **{{STAGE4_PATH_ADPOSITION_MARGIN_PTS}}** | {{STAGE4_PATH_ADPOSITION_DISCORDANT_N}} | {{STAGE4_PATH_ADPOSITION_MCNEMAR_P}} |

: The derived gate partitioned by which oracle path assigned each label. The listed strata sum to {{STAGE4_DERIVED_TEST_TOKENS}} tokens and their pooled margin reproduces {{STAGE4_DERIVED_MARGIN_PTS}} exactly, which is what establishes that the partition is of the scored population rather than of a similar one. The three *p*-values partition a hypothesis already in the confirmatory family, whose pooled test (*p* = {{STAGE4_DERIVED_MCNEMAR_P}}) scores every position listed here, so they are uncorrected, sit outside the declared significance family by recorded exclusion, and are printed to show that the aggregate is a mixture rather than to make a threshold-crossing claim; the morphology and adposition rows are unresolved controls, not results. {#tbl:case_oracle_paths}

The margin's primary uncertainty analysis uses the source document, not the
sentence or token, as the independent outer unit. The held-out derived arm has
{{STAGE4_DERIVED_TEST_SENTENCES}} sequences from
{{STAGE4_DERIVED_TEST_DOCUMENTS}} contributing documents; whole-document
resampling preserves every nested sequence and position together. Its
token-weighted point effect is {{CLUSTER_STAGE4_DERIVED_EFFECT}}
{{CLUSTER_STAGE4_DERIVED_EFFECT_UNIT}}, with a
{{CLUSTER_CONFIDENCE}}-coverage percentile cluster-bootstrap interval of
[{{CLUSTER_STAGE4_DERIVED_CI_LOW}}, {{CLUSTER_STAGE4_DERIVED_CI_HIGH}}]. The
largest document contributes {{CLUSTER_STAGE4_DERIVED_LARGEST_CLUSTER}}
scored tokens, the estimated intracluster correlation is
{{CLUSTER_STAGE4_DERIVED_ICC}}, and the corresponding variance-inflation
factor is {{CLUSTER_STAGE4_DERIVED_VARIANCE_INFLATION}}. A whole-document
sign-flip sensitivity reaches the finite Monte Carlo floor
{{CLUSTER_STAGE4_DERIVED_P}} under joint central symmetry of document-level
effects; the bootstrap interval is the primary statement. The familiar
tokenwise McNemar result (*p* = {{STAGE4_DERIVED_MCNEMAR_P}}, in the caption of [@tbl:case_oracle_paths]) is secondary and conditional on treating nested
positions as independent.

The reading is not that the model cheated. It never observes the field the
label is derived from, and [@sec:verification] reports what a predictor that
*is* allowed that field scores. The correct reading is narrower: on the
dependency-relation stratum, "case" is a deterministic renaming of a syntactic
relation, and a first-order model over part-of-speech tags is good at
recovering syntactic relations. What that stratum demonstrates, in other
words, is a result about word order wearing case's name.

The morphology stratum is where the claim could be tested against real
English case marking, and it is registered as a **control rather than a
result**. Its {{STAGE4_PATH_MORPHOLOGY_DISCORDANT_N}} discordant pairs sit far
below the minimum this project requires to resolve an equivalence
([@sec:multiplicity]). So its outcome is printed in [@tbl:case_oracle_paths]
but not read as a verdict, and it must never be read as *absent*. The
token-level adposition stratum's negative margin is likewise unresolved, at
{{STAGE4_PATH_ADPOSITION_DISCORDANT_N}} pairs. What it does establish is that
the aggregate is pooling strata whose signs disagree.

One objection deserves answering rather than deflecting: the morphology stratum
starts from a much higher baseline, so it has less room to improve, and a small
margin there could be a ceiling effect rather than a provenance effect. It is
not, or not only. Expressed as the percentage of available headroom each stratum
captures, the dependency-relation path takes
{{STAGE4_PATH_DEPREL_HEADROOM_CAPTURED_PCT}}% and the morphology path
{{STAGE4_PATH_MORPHOLOGY_HEADROOM_CAPTURED_PCT}}%. The gap narrows under that
normalization and does not close.

There is a stronger statement available, and it is structural rather than
statistical. The observation is the UPOS tag, so a margin requires the tag to
carry information about the label. Within each stratum it either does or it
cannot ([@tbl:case_path_information]):

| oracle path | UPOS alphabet | $H(\text{case})$ | $I(\text{case};\text{UPOS})$ |
|---|---:|---:|---:|
| dependency relation | {{STAGE4_PATH_DEPREL_UPOS_ALPHABET_SIZE}} | {{STAGE4_PATH_DEPREL_LABEL_ENTROPY_NATS}} | {{STAGE4_PATH_DEPREL_MUTUAL_INFORMATION_NATS}} |
| morphology | {{STAGE4_PATH_MORPHOLOGY_UPOS_ALPHABET_SIZE}} | {{STAGE4_PATH_MORPHOLOGY_LABEL_ENTROPY_NATS}} | {{STAGE4_PATH_MORPHOLOGY_MUTUAL_INFORMATION_NATS}} |
| adposition token | {{STAGE4_PATH_ADPOSITION_UPOS_ALPHABET_SIZE}} | {{STAGE4_PATH_ADPOSITION_LABEL_ENTROPY_NATS}} | {{STAGE4_PATH_ADPOSITION_MUTUAL_INFORMATION_NATS}} |

: Observation content within each stratum, in nats. The mutual information is {{STAGE4_PATH_MORPHOLOGY_MUTUAL_INFORMATION_NATS}} on the morphology path and {{STAGE4_PATH_ADPOSITION_MUTUAL_INFORMATION_NATS}} on the token-level adposition path, not merely small, and the reason is in the first column: each path *defines* the tag it selects on, so the observation alphabet inside the stratum is a single symbol and the emission cannot separate distinct states. {#tbl:case_path_information}

This is the same degeneracy [@sec:gold_population] reports for the gold
population, where the alphabet is likewise
{{STAGE4_GOLD_UPOS_ALPHABET_SIZE}} and every predictor scores alike. It was not
known to recur *inside* the derived population, because nothing had partitioned
it. So the reading of [@tbl:case_oracle_paths] sharpens: Level 4's margin does
not merely happen to live on the dependency-relation path, it can only live
there. On the other two the emission is uninformative by construction, and
whatever those strata show is attributable to the case-to-case transition alone,
never to what the model observed.

None of this withdraws a recorded number. The aggregate is correct arithmetic and
[@tbl:gate_audit] keeps its verdict. What changes is what the number licenses: a
reader can now see where the effect lives rather than accept on trust that
agreement with an unvalidated heuristic is not accuracy.

### Scored against real case morphology, which English EWT cannot supply {#sec:case_czech}

Everything above shares one ceiling: English EWT cannot say whether the
sequence mechanism works on real case, because its gold morphology lives on a
one-symbol observation alphabet and its derived margin lives in the stratum a
dependency rule labeled. The question that remains — does the margin survive
when the labels are ground truth? — was answerable only on a language that
marks case morphologically on nominals. Its population, comparator, endpoint,
and refusal rules were prospectively specified within the same development
session before the sequence-model score, after feasibility and development-set
inspection (`docs/decisions.md` entry {{DECISION_CZECH_CASE_PROTOCOL}}). That
specification was not independently timestamped in an immutable prior artifact,
so it is not a preregistration. **The result is exploratory**: the project's
confirmatory family was frozen before this arm existed, neither of its p-values
joins that family, and nothing here rescues or retires an English number.

The corpus is the Universal Dependencies conversion of the Prague Dependency
Treebank -- Consolidated (UD Czech-PDTC). Scored tokens are the case-inflecting nominal
categories ({{STAGE4_CZECH_UPOS_ALPHABET_SIZE}} UPOS tags) carrying UD
`feats[Case]`; every label is morphology
({{STAGE4_CZECH_LABEL_SOURCE_MORPHOLOGY_PCT}}% of scored tokens, enforced by a
recorder refusal — the coverage English gold never had), spanning
{{STAGE4_CZECH_CASES_PRESENT}} of the {{STAGE4_N_CASES}} roles (Czech has no
ablative). ADP is excluded although Czech annotates `Case=` on prepositions:
there the feature records *government* — the case of the complement, not
inflection of the token — which is the analogue of the English token-level adposition path
whose margin reversed in [@tbl:case_oracle_paths]. The vendored artifact
retains UPOS and `Case` per token plus each retained sequence's authenticated
`# newdoc id` index; the fields the English trap turns on never enter the file.
The train-ft shard
({{STAGE4_CZECH_TRAIN_TOKENS}} tokens) fits the same builders the English arm
uses — the modules are shared, not forked — and the dev shard
({{STAGE4_CZECH_TEST_TOKENS}} tokens in
{{STAGE4_CZECH_TEST_SEQUENCES}} retained sequences nested within
{{STAGE4_CZECH_TEST_DOCUMENTS}} contributing documents) is scored held-out,
against the same
context-free `argmax P(case | UPOS)` comparator fitted on the same training
tokens. Because the labels are ground truth, the endpoint is *accuracy* — the
word every English number in this section is named to avoid.

The prospectively specified question resolves in the surviving direction. The
sequence model reaches {{STAGE4_CZECH_ACCURACY}}% against the comparator's
{{STAGE4_CZECH_BASELINE}}%. The **primary uncertainty analysis** preserves the
token-weighted effect but treats each authenticated source document as one
independent outer cluster: {{CLUSTER_STAGE4_CZECH_EFFECT}}
{{CLUSTER_STAGE4_CZECH_EFFECT_UNIT}} over
{{CLUSTER_STAGE4_CZECH_OBSERVATIONS}} tokens in
{{CLUSTER_STAGE4_CZECH_CLUSTERS}} {{CLUSTER_STAGE4_CZECH_CLUSTER_UNIT}}s. Its
{{CLUSTER_STAGE4_CZECH_CONFIDENCE_PCT}}% percentile cluster-bootstrap interval
is [{{CLUSTER_STAGE4_CZECH_CI_LOW}}, {{CLUSTER_STAGE4_CZECH_CI_HIGH}}]. A
whole-document sign-flip sensitivity analysis with
{{CLUSTER_STAGE4_CZECH_RESAMPLES}} draws and
deterministic seed {{CLUSTER_STAGE4_CZECH_SEED}} reaches its declared Monte
Carlo resolution floor at {{CLUSTER_STAGE4_CZECH_P}}; it is reported as
$p \leq {{CLUSTER_STAGE4_CZECH_P_FLOOR}}$ conditional on joint central symmetry
of whole-document differences, an assumption stronger than marginally fair
token discordances or zero expected document sum. The interval
therefore bounds the positive effect away from zero at the document sampling
unit; the finite randomization statement is a resolution bound, not a claim
that a probability is zero.

The token-level analysis is secondary and conditional on treating positions as
independent despite their nesting within sequences and documents. It reports a
{{STAGE4_CZECH_MARGIN_PTS}}-point margin over
{{STAGE4_CZECH_TEST_TOKENS}} positions, with McNemar
$b={{STAGE4_CZECH_MCNEMAR_B}}$ against $c={{STAGE4_CZECH_MCNEMAR_C}}$
({{STAGE4_CZECH_DISCORDANT_N}} discordant pairs, against the
{{STAGE4_CZECH_MIN_DISCORDANT_PAIRS}}-pair resolvability minimum). Its exact
two-sided p-value underflows a double and is therefore reported through the
finite log-tail, $\log_{10}p={{STAGE4_CZECH_MCNEMAR_LOG10_P}}$. This much
smaller conditional tail does not replace the cluster-primary interval or
randomization result.

The displayed comparisons locate the margin. Below: the majority class scores
{{STAGE4_CZECH_MAJORITY}}%, so the comparator is doing real work before the
model improves on it. Above: a context-free predictor fitted on the *test labels
themselves* — the best any member of the comparator class can do — reaches
{{STAGE4_CZECH_CF_ORACLE}}%, and the model sits above that too. The margin is
therefore not a better lookup; no lookup on this observation reaches it, and
what remains is the case-to-case transition. That is the mechanism this level
always claimed, shown against labels no heuristic generated. The mis-wiring
controls transfer: a cyclically deranged emission or transition falls below the
context-free baseline (`tests/test_czech_case.py`).

The honest boundary of the finding is the observation's poverty, and it is
measured rather than lamented. The mutual information is
$I(\text{case};\text{UPOS})$ = {{STAGE4_CZECH_MUTUAL_INFORMATION_NATS}} nats,
against a label entropy of {{STAGE4_CZECH_LABEL_ENTROPY_NATS}}. So
{{STAGE4_CZECH_ACCURACY}}% is nowhere near what a model observing word forms
would reach: Czech case is largely *written on the word*, and a UPOS tag
discards the suffix. The claim this arm supports is correspondingly narrow —
the transition structure of case sequences carries real information beyond
the tag, on ground truth — and it carries two costs the development-session
specification declared. First, Level 4 becomes the stack's only non-English
level, so no inter-level channel can cross into this arm. Second, the
inventory mapping (UD `Case=Ins` → INS) remains a research claim even though
the labels themselves are morphology.

## Level 5: case-only ablation and widened-alphabet development result

This level was measured twice, on two observation alphabets, and both results
are reported here. The **canonical** clause level observes the case role plus
one distinguished symbol each for `VERB` and `AUX`; its gate was registered
before its artifact existed and it clears that gate
([@sec:clause_widened]). The clause model that observes case roles alone is
**retained as an ablation**, and everything between here and
[@sec:clause_widened] describes it: its gate does not pass, its
privileged-label gain does not resolve, and the held-back rescoring of its
classification contrast yielded no discordant pair. Its failure is what motivated widening the alphabet in
the first place, which is why it is kept in full rather than replaced. Both
arms carry limits that are stated with their verdicts rather than deferred.

### The ablation's declared gate is vacuous, since free energy rises with length under any model

The declared Stage 5 gate — `free energy tracks clause length` — is VACUOUS as
stated, and the audit shows it executably: cumulative free energy is monotone
in length under *any* model, because every step adds non-negative surprise. A
uniform-random model — the trivial baseline that must fail a bracketed gate —
satisfies it exactly. The declared wording also names a quantity the level
does not have: what the model consumes is one sentence's case-role sequence,
so the only length available to it is that sentence's. This is another shipped
declared gate to fail its audit, and a distinct failure
mode in the gate taxonomy ([@sec:gate_design]).

### On the case-only ablation, neither the mixture nor the privileged label beats the type-free chain

Stated at the resolution this design supports: the fitted mixture does not beat
the type-free case-role chain on this document-disjoint EWT split. Supplying the
privileged punctuation-defined type label also gives a negative point gain,
but a paired whole-document bootstrap does not resolve its direction. No
equivalence margin was declared, so these data do not license calling the gain
absent or negligible.

The derived gate is the honest one: beat the **type-free** case bigram — the
strongest model that does not use sentence type, decision 2 applied to the
new variable — on the identical held-out positions. The gate is
ABOVE_REFERENCE: the bar fails the strict criterion definitionally, and the
measured fitted type-label reference also fails to clear it. The comparator is
privileged but exceedable, not a bound, so this verdict does not say that no
better model can pass. The case-only mixture fails as well. This arm is therefore
a non-passing above-reference gate whose privileged-label gain remains
unresolved, not a metric pass, an impossibility result, or an equivalence result.
Read together with [@sec:clause_widened], it is also the direct evidence that
the bar is reachable and the alphabet was the binding constraint.

Over {{STAGE5_TEST_POSITIONS}} held-out positions, the type-free bar scores
{{STAGE5_TYPEFREE_PERPLEXITY}}; the mixture scores
{{STAGE5_MIXTURE_PERPLEXITY}} — *worse*. With
{{STAGE5_TRAIN_INTERROGATIVES}} interrogative training sentences, that pattern
is consistent with limited-sample estimation variance, but this single fitted
split does not distinguish variance from smoothing, misspecification, or
distribution shift. The decisive row is the **privileged type-label reference**: scoring every
sentence under the fitted chain selected by its privileged reference label
scores {{STAGE5_TYPE_ORACLE_PERPLEXITY}}, a signed perplexity reduction of
{{STAGE5_PRIVILEGED_LABEL_GAIN_PCT}}%; the negative sign means the fitted privileged
reference is worse than the type-free bar. This is a
receiver-sensitivity reference, not a hard information bound: under
misspecification a mixture can assign more probability than the label-selected
component. On the paired log-evidence scale, the reference-minus-type-free point
estimate is {{STAGE5_ORACLE_GAIN_NATS_PER_SENTENCE}} nats per sentence, with a
{{STAGE5_ORACLE_GAIN_CI_CONFIDENCE_PCT}}% percentile interval of
[{{STAGE5_ORACLE_GAIN_SENTENCE_CI_LOW}}, {{STAGE5_ORACLE_GAIN_SENTENCE_CI_HIGH}}]
over {{STAGE5_ORACLE_GAIN_BOOTSTRAP_RESAMPLES}} whole-document resamples of
{{STAGE5_ORACLE_GAIN_BOOTSTRAP_DOCUMENTS}} contributing test documents at
fixed seed {{STAGE5_ORACLE_GAIN_BOOTSTRAP_SEED}};
that interval {{STAGE5_ORACLE_GAIN_SENTENCE_INTERVAL_STATUS}}. The companion
ratio of resampled gain to resampled positions is
{{STAGE5_ORACLE_GAIN_NATS_PER_POSITION}} nats per position, interval
[{{STAGE5_ORACLE_GAIN_POSITION_CI_LOW}}, {{STAGE5_ORACLE_GAIN_POSITION_CI_HIGH}}],
which {{STAGE5_ORACLE_GAIN_POSITION_INTERVAL_STATUS}}. The latter point estimate
is exactly the log-ratio corresponding to the position-weighted perplexity
comparison above. Both arms use the same sampled document indices; all
sentences and positions within a selected document move together. The gain is therefore
{{STAGE5_ORACLE_GAIN_RESOLUTION}} at this order and in this sample. MAP classification reaches
{{STAGE5_CLASSIFICATION_PCT}}% accuracy on the held-out sentences, below the
{{STAGE5_MAJORITY_BASELINE_PCT}}% majority baseline, and a length-only
classifier merely reproduces that baseline
({{STAGE5_LENGTH_CLASSIFIER_PCT}}%). Recall on the minority class says it more
sharply: of the held-out interrogatives, MAP classification recovers
{{STAGE5_INTERROGATIVE_RECALL_PCT}}%. The document-primary paired accuracy
difference is {{STAGE5_CLASSIFICATION_MARGIN_PTS}} percentage points, with a
{{STAGE5_CLASSIFICATION_DOCUMENT_CI_CONFIDENCE_PCT}}% whole-document-bootstrap
interval of [{{STAGE5_CLASSIFICATION_DOCUMENT_CI_LOW}},
{{STAGE5_CLASSIFICATION_DOCUMENT_CI_HIGH}}] over
{{STAGE5_CLASSIFICATION_DOCUMENT_CLUSTERS}} documents, computed from
{{STAGE5_CLASSIFICATION_DOCUMENT_BOOTSTRAP_RESAMPLES}} whole-document resamples
at fixed seed {{STAGE5_CLASSIFICATION_DOCUMENT_BOOTSTRAP_SEED}}; the largest of
those clusters carries {{STAGE5_CLASSIFICATION_DOCUMENT_LARGEST_CLUSTER}}
sentences. The interval is read at the same declared confidence level as the
paired-gain intervals above — one declared level that both intervals state by
construction — and its upper endpoint is zero, so it supplies no positive
classification claim: p = {{STAGE5_CLASSIFICATION_DOCUMENT_P}} against the
recorded display floor {{STAGE5_CLASSIFICATION_DOCUMENT_P_FLOOR}}. The historical
sentencewise McNemar contrast remains a conditional secondary analysis. The mixture
posterior's point area under the curve (AUC) of the receiver operating
characteristic (ROC) is {{STAGE5_AUC}}, with a
{{STAGE5_AUC_CI_CONFIDENCE_PCT}}% interval of
[{{STAGE5_AUC_CI99_LOW}}, {{STAGE5_AUC_CI99_HIGH}}] bootstrapped over test
documents ({{STAGE5_AUC_BOOTSTRAP_DOCUMENTS}} contributing clusters, with all
sentences moving together). It excludes chance on that rank scale, but it does not select a
decision threshold or establish calibrated probabilities.

A rank statistic of {{STAGE5_AUC}} sitting beside
{{STAGE5_INTERROGATIVE_RECALL_PCT}}% minority recall looks contradictory only if AUC is read as a claim about
decisions or about probabilities, and it is neither. Average precision and
Brier score, measured on the same posteriors, say what AUC leaves out; each is
reported against the reference that makes them interpretable. **Average
precision** is {{STAGE5_AVERAGE_PRECISION}} against a positive-class prevalence
of {{STAGE5_POSITIVE_PREVALENCE_PCT}}% — its chance level is that prevalence
rather than one half. **Brier score** [@brier1950], a strictly proper rule
[@gneiting2007strictly], is {{STAGE5_BRIER}} against
{{STAGE5_BRIER_REFERENCE}} for the constant predictor that always emits the
{{STAGE5_TRAIN_PREVALENCE_PCT}}% training prevalence: a skill of
{{STAGE5_BRIER_SKILL}}. The negative skill means the fitted posterior is worse
than that constant under this proper score. That is the recorded point skill; no action-utility
threshold or equivalence margin was specified. The reference prevalence is taken from
the training split so that the baseline never sees a test label.

Strict propriety is worth stating precisely, because it licenses less than it
appears to. It means a forecaster minimizes expected score by reporting its
true beliefs; it does not mean a score improvement is attributable to
calibration. The Brier score decomposes into reliability, resolution and
uncertainty, so a lower value can come from sharper discrimination as readily as
from better-calibrated probabilities, and this measurement does not separate the
two.

Together these give one consistent but qualified description: the model's AUC
and average-precision point estimates exceed their rank references, while its
Brier score and default maximum-a-posteriori accuracy are worse than their
references and the mixture does not clear its type-free gate. The paired
privileged-label gain is unresolved. That last qualification is not decorative.
An AUC of {{STAGE5_AUC}} is ranking information, and information a threshold
could in principle exploit — a cost-asymmetric rule trading precision for
minority recall is not ruled out by anything measured here. What is reported is
a non-passing result for the decision rule this project actually applies, not
for every rule that could be applied. The perplexity ladder and paired
log-evidence interval answer related but distinct questions and agree only on
the weaker conclusion: no gain is resolved on this split.

What makes this a finding rather than a shrug is the **detectability
control**. On a synthetic corpus whose two types genuinely differ, the
identical harness classifies at {{STAGE5_CONTROL_CLASSIFICATION_PCT}}% against
a {{STAGE5_CONTROL_MAJORITY_BASELINE_PCT}}% majority and drops mixture perplexity from
{{STAGE5_CONTROL_TYPEFREE_PERPLEXITY}} to
{{STAGE5_CONTROL_MIXTURE_PERPLEXITY}}. That is a *design* argument and not an
observed-power one: the control plants a known effect and shows the instrument
recovers it, whereas computing power from the very statistic that failed to
reject is a transformation of the p-value and says nothing new
[@hoenig2001abuse]. Read that way, the control rules out a generically dead
instrument; it does not turn the unresolved EWT result into a property of
English. The control's own operating
point differs from the measurement's — it is balanced and larger — so it bounds
the instrument's capability, not the sensitivity available at
{{STAGE5_TRAIN_INTERROGATIVES}} minority training sentences and
{{STAGE5_TEST_INTERROGATIVES}} test ones. At the operating point that produced
the failure, uncertainty is reported on the axes that were actually intervalled:
the AUC interval above, the paired oracle-gain intervals reported here, and the
exact conditional bound on the default MAP classification contrast in
[@tbl:derived_bounds]. The latter produced too few discordant pairs to support
equivalence, and the oracle-gain intervals were not evaluated against any
smallest effect of interest. The machinery ships — spec-built and bit-identical, with the closed form
agreeing with a manual walk over the Agent's stored arrays to
{{STAGE5_DECODER_MAX_DIVERGENCE}} — but the shipped mixture does not clear the
type-free bar and the default MAP rule does not improve on its majority
reference. What the synthetic control establishes is that the same machinery
recovers a much larger designed signal; it does not establish that the real
effect is exactly zero. This arm therefore reports a failed gate, paired
uncertainty on the privileged-label gain, and a bound on its classification
contrast—not absence, negligible value, or an equivalence result
([@sec:hierarchy] returns to what this outcome does to the inter-level criterion).
[@fig:clause_null] plots this arm's non-passing point result and interval beside
the detectability control.

![The retained case-only clause ablation, the project's first above-reference-and-failed gate. Left: the document-disjoint EWT measurement
over {{STAGE5_TEST_POSITIONS}} held-out positions — the type-free bar, the
mixture that sits above it, and the fitted privileged type-label reference that
also sits above it; the signed point perplexity reduction is
{{STAGE5_PRIVILEGED_LABEL_GAIN_PCT}}%. Its companion
position-weighted paired gain is {{STAGE5_ORACLE_GAIN_NATS_PER_POSITION}} nats
per position, {{STAGE5_ORACLE_GAIN_CI_CONFIDENCE_PCT}}% interval
[{{STAGE5_ORACLE_GAIN_POSITION_CI_LOW}}, {{STAGE5_ORACLE_GAIN_POSITION_CI_HIGH}}],
which {{STAGE5_ORACLE_GAIN_POSITION_INTERVAL_STATUS}} under a
{{STAGE5_ORACLE_GAIN_BOOTSTRAP_DOCUMENTS}}-document bootstrap and is drawn as
an interval against its zero line in the inset as well as quoted in the panel
title. Right:
the detectability control — the same harness on planted synthetic types,
where the mixture wins decisively and classification reaches
{{STAGE5_CONTROL_CLASSIFICATION_PCT}}%. A left-panel mixture or privileged
reference bar *below* the type-free bar, or a right panel without the decisive gap
would each contradict the recorded finding; so would either paired interval
ceasing to span zero. The data function refuses to draw in every such case.](../../output/figures/clause_null.png){#fig:clause_null width=97%}


![What each upper-level score is, and what kind of evidence backs it. Left: the Stage 4 case-role
agreement of {{STAGE4_DERIVED_AGREEMENT}}% over {{STAGE4_DERIVED_TEST_TOKENS}} tokens comes from the
numpy forward filter; live `infer_states` + `update_empirical_prior` agree with it at every one of
the {{STAGE4_PYMDP_POSITIONS}} held-out positions ({{STAGE4_PYMDP_ARGMAX_DISAGREEMENTS}} argmax
disagreements, largest posterior deviation {{STAGE4_PYMDP_POSTERIOR_MAX_ABS_DEV}}), so the
attribution is exhaustive over the scored population while the headline path stays NumPy. Middle:
the closed-form mixture path of the retained case-only Stage 5 ablation, drawn at its type-free bar of {{STAGE5_TYPEFREE_PERPLEXITY}}; the
live pymdp agent agrees with the NumPy joint filter on the type-posterior argmax at
{{STAGE5_PYMDP_TYPE_ARGMAX_DISAGREEMENTS}} disagreements over all {{STAGE5_PYMDP_PREFIX_SENTENCES}}
held-out sentences and on the full joint posterior within
{{STAGE5_PYMDP_POSTERIOR_MAX_ABS_DEV}} — posterior agreement on the whole held-out set, not score
equivalence for the headline mixture estimand. Right: the
registered observed-position score is a conditional table, not inference; the separately recorded
live-filter score ({{STAGE6_LIVE_FILTER_LOGLOSS}} nats, {{STAGE6_LIVE_FILTER_IMPROVEMENT_NATS}} nats
better than the unigram bar over {{STAGE6_TEST_SENTENCES}} sentences) is the only L6 quantity drawn
through pymdp's inference path. Any of the following would falsify this panel: a recorded L4 or L5
disagreement count other than the stated zero, an L4 comparison that stops short of the scored
population, a live-filter score inconsistent with its recorded
improvement, the two L6 estimand deltas collapsing onto one another, or the Stage 4 agreement
leaving the majority share its annotation requires.](../../output/figures/inference_evidence.png){#fig:inference_evidence width=95%}

### Canonical widened model: development-only gain over the type-free chain {#sec:clause_widened}

This is the canonical clause level. On {{STAGE5W_TEST_SENTENCES}} held-out
sentences from
{{STAGE5W_TEST_DOCUMENTS}} source documents, spanning
{{STAGE5W_TEST_POSITIONS}} positions, the widened mixture beats the type-free
bar refitted on the identical widened stream ([@tbl:stage5w]). That is the
registered gate and it is a measured outcome that could have failed.

Two limits belong with that verdict, and they are stated here once rather than
distributed through the section. First, this is a **development-session
specification**: the entry that registered the alphabet, the population, the
comparators, the controls, the predictions and the falsifiers was committed
before the artifact was regenerated, but its registration and its outcome sit
in one session, so every label the arm produces is recorded descriptive and
none may be read as confirmatory. Second, it has **no held-back replication**,
and it can never acquire one. The held-back `ud_ewt_test` partition was spent
by the confirmatory run of [@sec:confirmatory]; a sentinel refuses a second
pass from either side, and a second pass over a held-back set un-holds it (a
re-analysis of the spent population, as in [@sec:stage6_heldback_clustered], is
descriptive only). A
reader who wants the strongest available evidence about the clause level should
weigh a registered gate that could have failed against the absence of an
independent replication, and neither fact cancels the other.

| quantity | value |
|---|---|
| type-free bar refitted on the widened stream | {{STAGE5W_TYPEFREE_PERPLEXITY}} |
| mixture (the widened clause model) | {{STAGE5W_MIXTURE_PERPLEXITY}} |
| privileged type reference (true type given) | {{STAGE5W_TYPE_ORACLE_PERPLEXITY}} |
| privileged-label gain over the bar | {{STAGE5W_PRIVILEGED_LABEL_GAIN_PCT}}% |
| paired gain, nats per sentence | {{STAGE5W_ORACLE_GAIN_NATS_PER_SENTENCE}} |
| its {{STAGE5W_ORACLE_GAIN_CI_CONFIDENCE_PCT}}% document interval | [{{STAGE5W_ORACLE_GAIN_SENTENCE_CI_LOW}}, {{STAGE5W_ORACLE_GAIN_SENTENCE_CI_HIGH}}] |
| paired gain, nats per position | {{STAGE5W_ORACLE_GAIN_NATS_PER_POSITION}} |
| its interval | [{{STAGE5W_ORACLE_GAIN_POSITION_CI_LOW}}, {{STAGE5W_ORACLE_GAIN_POSITION_CI_HIGH}}] |
| entropy of the type variable | {{STAGE5W_TYPE_ENTROPY_NATS}} nats |
| share of that entropy realized | {{STAGE5W_TYPE_ENTROPY_CAPTURED_PCT}}% |
| MAP classification vs majority | {{STAGE5W_CLASSIFICATION_PCT}}% vs {{STAGE5W_MAJORITY_BASELINE_PCT}}% |
| interrogative recall | {{STAGE5W_INTERROGATIVE_RECALL_PCT}}% |
| rank signal (AUC) | {{STAGE5W_AUC}} |
| its {{STAGE5W_AUC_CI_CONFIDENCE_PCT}}% document interval | [{{STAGE5W_AUC_CI99_LOW}}, {{STAGE5W_AUC_CI99_HIGH}}] |
| comparator: sentence length only | {{STAGE5W_AUC_LENGTH_CONTROL}} |
| comparator: first symbol is an auxiliary | {{STAGE5W_AUC_FIRST_AUX_CONTROL}} |
| comparator: case-only arm, same sentences | {{STAGE5W_PAIRED_CASE_ONLY_AUC}} |
| widened arm on those same sentences | {{STAGE5W_PAIRED_WIDENED_AUC}} |

: The widened-observation clause arm on the document-disjoint partition.
Descriptive; the *p*-values below sit outside the frozen family, uncorrected.
{#tbl:stage5w}

The paired privileged-label gain is {{STAGE5W_ORACLE_GAIN_NATS_PER_SENTENCE}}
nats per sentence and its {{STAGE5W_ORACLE_GAIN_CI_CONFIDENCE_PCT}}%
whole-document interval excludes zero on both reported scales. This is the
quantity the retained case-only ablation leaves unresolved, and on the widened
stream its direction is resolved. It realizes
{{STAGE5W_TYPE_ENTROPY_CAPTURED_PCT}}% of the {{STAGE5W_TYPE_ENTROPY_NATS}}
nats the binary type variable makes available. The widening does not raise
that ceiling — the entropy is a property of the prior, not of the receiver —
so what moved is the share realized within it.

The rank signal clears every registered comparator. It exceeds the
length-only rule at {{STAGE5W_AUC_LENGTH_CONTROL}}, whose classifier never
beats the base rate at all, scoring {{STAGE5W_LENGTH_CLASSIFIER_PCT}}% against
the {{STAGE5W_MAJORITY_BASELINE_PCT}}% majority. It exceeds the single
first-symbol-is-an-auxiliary indicator at
{{STAGE5W_AUC_FIRST_AUX_CONTROL}}, so the effect is not one free parameter
wearing a sequence model's clothes. And on the
{{STAGE5W_PAIRED_SUBSET_SENTENCES}} sentences where both arms are defined, the
widened arm scores {{STAGE5W_PAIRED_WIDENED_AUC}} against the case-only arm's
{{STAGE5W_PAIRED_CASE_ONLY_AUC}}: the two added symbols, not the refit, carry
the difference.

Classification reaches {{STAGE5W_CLASSIFICATION_PCT}}% against the
{{STAGE5W_MAJORITY_BASELINE_PCT}}% majority, recovering
{{STAGE5W_INTERROGATIVE_RECALL_PCT}}% of held-out interrogatives where the
case-only ablation recovers none. Its discordant counts are {{STAGE5W_MCNEMAR_B}}
and {{STAGE5W_MCNEMAR_C}} at *p* = {{STAGE5W_MCNEMAR_P}}, uncorrected and
outside the frozen family. The limit belongs beside that number: documents are
the outer sampling unit, and the document-clustered margin of
{{STAGE5W_CLASSIFICATION_MARGIN_PTS}} points has interval
[{{STAGE5W_CLASSIFICATION_DOCUMENT_CI_LOW}},
{{STAGE5W_CLASSIFICATION_DOCUMENT_CI_HIGH}}] across
{{STAGE5W_CLASSIFICATION_DOCUMENT_CLUSTERS}} clusters, whose lower bound
touches zero. The classification contrast is therefore *not* resolved away
from zero at document level, even though the paired gain and the rank interval
both are. On the threshold-free scores the posteriors carry real information:
average precision {{STAGE5W_AVERAGE_PRECISION}} against the
{{STAGE5W_POSITIVE_PREVALENCE_PCT}}% prevalence, and Brier
{{STAGE5W_BRIER}} against the training-prevalence reference
{{STAGE5W_BRIER_REFERENCE}}, a skill score of {{STAGE5W_BRIER_SKILL}}.

The controls that stand behind those numbers each have the authority to stop
the recording. The closed-form mixture and oracle filtering agree to
{{STAGE5W_DECODER_MAX_DIVERGENCE}}, and that agreement is not vacuous: a
transition carrying mass across the type blocks drives the same comparison
apart by {{STAGE5W_LEAK_CONTROL_DIVERGENCE}}. Exchanging the two types' fitted
chains costs the classifier its margin, dropping it to
{{STAGE5W_SWAPPED_BLOCKS_CLASSIFICATION_PCT}}%, so the type assignment is
load-bearing. Finally, the identical harness on a synthetic corpus whose types
genuinely differ classifies at {{STAGE5W_CONTROL_CLASSIFICATION_PCT}}% against
its measured {{STAGE5W_CONTROL_MAJORITY_BASELINE_PCT}}% majority, with mixture
{{STAGE5W_CONTROL_MIXTURE_PERPLEXITY}} beating type-free
{{STAGE5W_CONTROL_TYPEFREE_PERPLEXITY}}. That control establishes a live
instrument for a large planted effect. It supplies no minimum detectable
effect, though, and it is not evidence about small ones.

The population is stated with the result. The arm fits on
{{STAGE5W_TRAIN_SENTENCES}} training sentences and scores {{STAGE5W_TEST_SENTENCES}} held-out
ones, drawn from {{STAGE5W_SOURCE_DOCUMENTS}} source documents
({{STAGE5W_TRAIN_DOCUMENTS}} training and {{STAGE5W_TEST_DOCUMENTS}} held out) and
carrying {{STAGE5W_TRAIN_INTERROGATIVES}} and {{STAGE5W_TEST_INTERROGATIVES}}
interrogatives respectively. {{STAGE5W_EXCLUDED_EXCLAMATIVE}} exclamative
sentences and {{STAGE5W_EXCLUDED_UNPUNCTUATED}} unpunctuated fragments carry no
label, and {{STAGE5W_EXCLUDED_TOO_SHORT}} fall under the usability minimum.

## Level 6: development and held-back document-position results

### Development estimate with a document-clustered interval spanning zero

The registered point estimate is positive. Its primary analysis spans zero on the
development population and excludes zero on the held-back one.

The train split shows interrogatives concentrating in document openings: in this
corpus, English web text opens with questions more often than it continues with
them. The association is stated over EWT and no further. The preregistered
direction was lower held-out mean log loss under the tercile-conditional table.
Its point estimate is positive. Over {{STAGE6_TEST_SENTENCES}} sentences in
{{STAGE6_TEST_DOCUMENTS}} held-out documents, the tercile-conditional model
scores {{STAGE6_PHASE_LOGLOSS}} nats per sentence against the type-unigram
bar's {{STAGE6_UNIGRAM_LOGLOSS}} — an improvement of
{{STAGE6_LOGLOSS_IMPROVEMENT_NATS}} nats, positive on
{{STAGE6_SIGN_TEST_POSITIVE}} sentences against
{{STAGE6_SIGN_TEST_NEGATIVE}} (sign test p = {{STAGE6_SIGN_TEST_P}}). This
registered observed-position score indexes the fitted table by the known
sentence position; it is not live pymdp filtering over the latent tercile. The
audit classifies the gate BRACKETED_REFERENCE: the bar fails it definitionally, and the
fitted-on-test reference clears it.

The document is the sampling unit. In the primary document-clustered analysis on
the development population, the log-loss interval
[{{CLUSTER_STAGE6_LOGLOSS_CI_LOW}}, {{CLUSTER_STAGE6_LOGLOSS_CI_HIGH}}] spans
zero and the p-value is {{CLUSTER_STAGE6_LOGLOSS_P}} under the
assumption-conditional whole-document sign-flip sensitivity analysis
(finite-sample exact only if the joint document-difference distribution is
centrally symmetric). That result on that population is imprecise, and it is
what the level's status rested on until the estimand was carried to a population
the models had never seen. The sentence sign count is secondary and
non-diagnostic: reversing the phase columns makes mean log loss worse yet still
yields a nominally favorable sentence sign test. A statistic that a worse control
passes cannot confirm the registered mean-log-loss prediction, which is why the
held-back sign test of decision {{CONFIRMATORY_REGISTRATION_DECISION}} was not treated as settling the question
below.

### Held-back document-clustered gain at uncorrected coverage {#sec:stage6_heldback_clustered}

Decision {{CONFIRMATORY_STAGE6_CLUSTERED_DECISION}} registered the primary estimand on the held-back UD EWT test split
before computing it, fixing the population, the bootstrap, the coverage, the seed
and the prediction in a commit that precedes the outcome. The models are the
frozen development fits, asserted equal array for array; nothing is refitted.
Its status is descriptive, not confirmatory: the population had already been
spent by the run of decision {{CONFIRMATORY_REGISTRATION_DECISION}}, so this estimand was registered by
someone who knew that population's other results ([@sec:stage6_heldback_method]).

Over {{CONFIRMATORY_STAGE6_CLUSTERED_SENTENCES}} sentences in
{{CONFIRMATORY_STAGE6_CLUSTERED_DOCUMENTS}} held-back documents the mean
improvement is {{CONFIRMATORY_STAGE6_CLUSTERED_MEAN_IMPROVEMENT_NATS}} nats, and
the document-clustered interval
[{{CONFIRMATORY_STAGE6_CLUSTERED_CI_LOW}},
{{CONFIRMATORY_STAGE6_CLUSTERED_CI_HIGH}}] excludes zero at
{{CONFIRMATORY_STAGE6_CLUSTERED_CONFIDENCE_PCT}} percent coverage, the coverage
the development analysis above uses. The improvement favors the tercile model in
{{CONFIRMATORY_STAGE6_CLUSTERED_POSITIVE_DOCUMENTS}} documents against
{{CONFIRMATORY_STAGE6_CLUSTERED_NEGATIVE_DOCUMENTS}}, and the sign-flip p-value
is {{CONFIRMATORY_STAGE6_CLUSTERED_P}}. Together with the held-back sign test of
decision {{CONFIRMATORY_REGISTRATION_DECISION}}, which crosses the frozen family threshold in the registered
direction, this is what the gate audit reads to compute Level 6's finding
status; the status is a function of those recorded flags rather than a sentence
anyone typed.

Three qualifications travel with that interval and none of them is optional.

First, the development-population interval still spans zero. The two populations
disagree, and the honest description is that the estimand resolved on the larger
held-back sample and did not on the smaller development one, not that the earlier
result was wrong.

Second, the same held-back bootstrap read at the family-corrected coverage of
{{CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CONFIDENCE_PCT}} percent gives
[{{CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CI_LOW}},
{{CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CI_HIGH}}], which spans zero. This arm
is a registered singleton and not a member of the frozen significance family, so
the corrected level is a declared sensitivity rather than its decision
threshold. A reader who prefers the conservative reading should take the effect
as unresolved at that level, and the recorded bounds are there to make that
reading available rather than to be argued away.

Third, the population's resolution is coarser than the effect the development
population produced. The smallest constant per-sentence shift this arm can
resolve is {{CONFIRMATORY_STAGE6_CLUSTERED_MIN_DETECTABLE_EFFECT_NATS}} nats,
measured by planting shifts on the centered differences and verified from both
sides. The development population's own clustered effect,
{{CONFIRMATORY_STAGE6_CLUSTERED_CONTROL_PLANTED_SHIFT_NATS}} nats, sits below
that, and planting it here does not resolve
([{{CONFIRMATORY_STAGE6_CLUSTERED_CONTROL_POSITIVE_CI_LOW}},
{{CONFIRMATORY_STAGE6_CLUSTERED_CONTROL_POSITIVE_CI_HIGH}}]). The held-back
effect is detected because it is larger than the development one, not because
this population is more sensitive. A null here would therefore have been
uninterpretable, and the detection should not be read as evidence that the
effect size generalizes.

The controls behave. Centering the differences to exactly zero mean gives
[{{CONFIRMATORY_STAGE6_CLUSTERED_CENTERED_BASELINE_CI_LOW}},
{{CONFIRMATORY_STAGE6_CLUSTERED_CENTERED_BASELINE_CI_HIGH}}], which does not
detect, so no plant above it is detecting the centering. Reversing the sign of
{{CONFIRMATORY_STAGE6_CLUSTERED_CONTROL_FLIPPED_DOCUMENTS}} whole documents
under a recorded seed gives
[{{CONFIRMATORY_STAGE6_CLUSTERED_CONTROL_NEGATIVE_CI_LOW}},
{{CONFIRMATORY_STAGE6_CLUSTERED_CONTROL_NEGATIVE_CI_HIGH}}], which also does not
detect, so the procedure does not find an effect in differences whose aggregate
direction has been destroyed.

A separate descriptive run does use the live Agent's filtering recursion. It
reports {{STAGE6_LIVE_FILTER_LOGLOSS}} nats, an improvement of
{{STAGE6_LIVE_FILTER_IMPROVEMENT_NATS}} nats, with sentence signs
{{STAGE6_LIVE_FILTER_SIGN_POSITIVE}} to {{STAGE6_LIVE_FILTER_SIGN_NEGATIVE}}
(*p* = {{STAGE6_LIVE_FILTER_SIGN_P}}). This is evidence that the built Agent can
run and produce a related positive descriptive score; it is not the registered
observed-position estimand and does not replace the registered
document-clustered primary analysis. As a post-registered execution diagnostic,
its p-value is outside the frozen significance family, is uncorrected, and
carries no threshold-crossing inference. The attribution gap between the two
estimands is measured rather than merely named: on the same bounded slice of
{{STAGE6_PYMDP_PREFIX_SENTENCES}} held-out sentences, the live Agent's
recursion scores {{STAGE6_PYMDP_LIVE_LOGLOSS}} nats against the registered
observed-position score's {{STAGE6_PYMDP_OBSERVED_LOGLOSS}} — a divergence of
{{STAGE6_PYMDP_LIVE_VS_OBSERVED_DELTA_NATS}} nats — and the two estimands
agree on the per-sentence sign of the improvement on a
{{STAGE6_PYMDP_LIVE_VS_OBSERVED_SIGN_AGREEMENT}} share of sentences. They index different
conditionals by construction (a `B`-carried latent-phase belief versus the
known position), so this is characterization, not equivalence.

Under the repository-wide multiplicity family of {{STAT_FAMILY_SIZE}}
declared tests ([@sec:multiplicity]), the secondary sentence-sign p-value is nominally positive at
$\alpha =$ {{STAT_FAMILYWISE_ALPHA}} and does not cross the fixed Bonferroni
per-test threshold of {{STAT_BONFERRONI_ALPHA}}; the recorded flag
`stat_stage6_familywise_confirmed` is {{STAT_STAGE6_FAMILYWISE_CONFIRMED}}.
Against its Holm step-down threshold of {{STAT_HOLM_STAGE6_THRESHOLD}} at the
same familywise level the recorded verdict `stat_holm_stage6_rejected` is
{{STAT_HOLM_STAGE6_REJECTED}}: the test {{STAGE6_SIGN_STEPDOWN_CLAUSE}}, and a
step-down verdict depends on where every other member of the family fell
([@sec:multiplicity]). The
threshold question remains secondary because neither multiplicity correction
repairs the wrong sampling unit or the mismatch between sentence signs and mean
log loss. It is not confirmation.

### The top-down prior intervention moves the case-only model's type prior

[@sec:generation] implements an L3→L1g `C`-preference seam. Stage 6 uses a
different mechanism in a separate experiment. The
position prediction $P(\text{type} \mid \text{tercile})$ blends into that
level's `D` type prior — $D'(\alpha) = (1-\alpha) \cdot P_{\text{L5}} + \alpha
\cdot P(\text{type} \mid \text{tercile})$, re-formed into the joint prior —
and the guards below make the demonstration falsifiable rather than decorative.
The $\alpha = 0$ anchor reproduces the sentence-type prior **byte-for-byte**
(the recorder refuses to run otherwise). At $\alpha = 1$ the opening-tercile
message moves P(interrogative) from the static
{{STAGE6_STATIC_INTERROGATIVE_MASS}} by
+{{STAGE6_OPENING_INTERROGATIVE_SHIFT}}, the middle tercile by
{{STAGE6_MIDDLE_INTERROGATIVE_SHIFT}}. And the tercile-permuted scramble — the
entropy-preserving control — lands elsewhere
({{STAGE6_SCRAMBLED_OPENING_SHIFT}} at the opening), so the effect depends on
which tercile says what. This intervention is not passed to L3 or L1g in the
same run.

The retained case-only clause ablation supplies a privileged-label
**sensitivity reference**, and it is the arm this intervention was measured
against. The intervention
demonstrably moves the sentence-type level's *type* prior, while selecting the
fitted case-only chain by its reference type changes case-role perplexity by
{{STAGE5_PRIVILEGED_LABEL_GAIN_PCT}}% (a signed reduction; here negative). That small,
unresolved value motivates expecting little
delivered effect, but it is not a hard bound under model misspecification.
Novelty and receiver sensitivity remain useful explanatory dimensions. The working hypothesis of
[@sec:hierarchy] therefore predicts a channel pays only when it is novel to
the receiver **and** consequential for the receiver's likelihood, and the
document-position→sentence-type intervention is evidence about that distinction,
not a proof of it. [@fig:narrative_channel] shows the intervention's measured effect.

![Stage 6's two preregistered experiments. Left: held-out observed-position type log-loss,
the tercile-conditional model — drawn as `phase-conditional` — against the
type-unigram bar over {{STAGE6_TEST_SENTENCES}}
sentences; the improvement of {{STAGE6_LOGLOSS_IMPROVEMENT_NATS}} nats
(secondary sentence sign p = {{STAGE6_SIGN_TEST_P}}) has a document-clustered
interval of [{{CLUSTER_STAGE6_LOGLOSS_CI_LOW}},
{{CLUSTER_STAGE6_LOGLOSS_CI_HIGH}}], so it is non-confirmatory. Right: the independent prior intervention — the sentence-type
level's P(interrogative) after the $\alpha = 1$ message, per tercile, against
the dashed static prior that the $\alpha = 0$ anchor reproduces exactly, with
the scrambled-message control (×) landing away from the real opening-tercile
message. Bars hugging the dashed line, or the × coinciding with the opening
bar, would each falsify the intervention demonstration, and the data function
refuses to draw in either case.](../../output/figures/narrative_channel.png){#fig:narrative_channel width=97%}

```{=latex}
\csname @ifclassloaded\endcsname{beamer}{}{\Needspace{24\baselineskip}}
```

## Generation: the optimum is interior, and the distance metric could not be the gate

### Interior optimum in the registered generation sweep

This table sweeps the generator's policy precision and shows that both
extremes generate badly, for opposite reasons, while an interior setting does
not.

| Configuration | Membership | Distinct words | Distance |
|---|---|---|---|
| Baseline — ancestral sampling from $\mathbf{B}$ | {{GENERATION_BASELINE_MEMBERSHIP_PCT}}% | {{GENERATION_BASELINE_DISTINCT}} | {{GENERATION_BASELINE_TV}} |
| $\gamma$ = {{GENERATION_LOW_GAMMA}} — goal too weak | {{GENERATION_LOW_GAMMA_MEMBERSHIP_PCT}}% | — | {{GENERATION_LOW_GAMMA_TV}} |
| **$\gamma$ = {{GENERATION_BEST_GAMMA}} — shipped** | **{{GENERATION_BEST_MEMBERSHIP_PCT}}%** | **{{GENERATION_BEST_DISTINCT}}** | **{{GENERATION_BEST_TV}}** |
| $\gamma$ = {{GENERATION_DEGENERATE_GAMMA}} — collapsed | {{GENERATION_DEGENERATE_MEMBERSHIP_PCT}}% | {{GENERATION_DEGENERATE_DISTINCT}} | {{GENERATION_DEGENERATE_TV}} |

: Every distance is computed exactly from a finite-lexicon path distribution plus an explicit out-of-lexicon remainder: Brown-transition paths for the ancestral baseline and policy-posterior paths for each of the {{GENERATION_EXACT_GRID_POINTS}} precision settings. Both distributions condition out an immediate empty attempt, matching the non-empty token unit scored by the sampled generator. Membership remains a sampled diagnostic and distinct-word count the sampled anti-collapse floor, over {{GENERATION_WORDS_PER_RUN}} words per run and {{GENERATION_SEEDS}} seeds, because both are properties of realized runs rather than of the induced distribution. The optimum is **interior**: both endpoints generate badly, for opposite reasons. {#tbl:generation_sweep}

[@tbl:generation_sweep] reports the sweep, and the shape is the result. As
$\gamma \to 0$ the policy posterior becomes the habit and the model *is* the
baseline. At high precision, the policy follows the locally most preferred trie
branch at each prefix, with the habit breaking ties. This greedy conditional
path is not generally the globally most probable word; on the shipped Brown
artifact its distinct-word count nevertheless collapses to
{{GENERATION_DEGENERATE_DISTINCT}}. Neither endpoint is
language.

#### Expected-free-energy contribution to letter generation

The generator selects actions by minimizing [@eq:efe], which is the one place in
this project where the distinctively active-inference machinery runs at all: in
the comprehension path every level has a single action and a flat $\mathbf{C}$,
so no policy is scored ([@sec:background]). It is therefore worth stating exactly
what that machinery buys, and the answer is narrower than the framing suggests.

\begin{proposition}[The information-gain term vanishes]
Let the transition be deterministic, so that taking action \(k\) from any state
yields \(q(s_\tau \mid \pi = k) = \delta_k\). Then for every policy the
information-gain term of expected free energy is exactly zero, and the expected
free energy reduces to its pragmatic part,
\(G(\pi = k) = -\sum_o \mathbf{A}[o, k]\, \mathbf{C}[o]\).
If in addition the emission is the identity, so that action \(k\) yields
observation \(k\) with certainty, this becomes \(G(\pi = k) = -\mathbf{C}[k]\).
\end{proposition}

The argument is one line. The posterior after an outcome is
$q(s_\tau \mid o_\tau, \pi) \propto p(o_\tau \mid s_\tau)\, q(s_\tau \mid \pi)$,
and a point-mass predictive is not sharpened by any likelihood, so the posterior
is $\delta_k$ as well and the divergence between them is zero. Nothing in *that*
half of the argument mentions $\mathbf{A}$: the vanishing follows from the
transition alone.

The second equality does depend on $\mathbf{A}$, and the distinction is easy to
lose because the first does not. The predictive over outcomes under policy $k$ is
$q(o_\tau \mid \pi = k) = \sum_s \mathbf{A}[o_\tau, s]\, \delta_k(s) =
\mathbf{A}[\cdot, k]$, so the pragmatic term is $-\sum_o \mathbf{A}[o,k]\,
\mathbf{C}[o]$ in general, and collapses to $-\mathbf{C}[k]$ only when
$\mathbf{A}$ maps state $k$ to observation $k$ with certainty. AILM's letter
generator does have an identity emission ([@sec:architecture]), so the reduction
below holds for the shipped model, and the condition is stated here because a
proposition that omits it would be false of a model with any other emission.
Substituting into the
policy posterior [@eq:policy_posterior] gives

$$
q(\pi = k) \;=\; \sigma\!\left(\gamma\, \mathbf{C}[k] + \log \mathbf{E}[k]\right),
$$ {#eq:policy_reduction}

which is what `efe_policy_posterior` computes and what
`tests/test_free_energy_decomposition.py` checks against the full decomposition
at representative current letters spanning the alphabet.

A proposition asserting that a quantity is zero reads as true whether or not
anyone computed it, so the term is measured against stochastic-transition and
flat-likelihood controls; these are the whole of the evidence. On the shipped construction the largest information gain over
all {{STAGE0_LETTER_ALPHABET_SIZE}} actions is
{{GENERATION_SHIPPED_INFORMATION_GAIN_NATS}} nats. Making the transition uniform
while keeping the identity likelihood raises it to
{{GENERATION_STOCHASTIC_CONTROL_INFORMATION_GAIN_NATS}} nats, which is exactly
$\log {{STAGE0_LETTER_ALPHABET_SIZE}}$ — the analytic ceiling, since a uniform
predictive resolved by a perfect observation yields precisely the prior entropy.
Flattening the likelihood as well returns it to
{{GENERATION_FLAT_LIKELIHOOD_CONTROL_INFORMATION_GAIN_NATS}}. The second control
is the one that identifies the cause: a zero alone is consistent with either
condition, and only exhibiting a spread predictive with an uninformative
observation shows that the shipped generator's maximally informative identity
$\mathbf{A}$ is not what produces the result. `scripts/record_generation_claims.py`
refuses to record if the first control fails to exceed the shipped value, because
a control that cannot fire turns the recorded zero into decoration.

This cuts two ways and both are worth having. It is a limitation: at this level
the expected free energy contributes no exploratory drive, and
[@eq:policy_reduction] is a preference-weighted softmax that a reader owing
nothing to active inference could have written down. It is also what makes the
rest of this section interpretable, because the generator's behavior is fully
explained by $\mathbf{C}$ and $\mathbf{E}$ — which is why the flat-preference and
scrambled-preference controls below isolate what they claim to isolate, and why
the $\gamma$ sweep is a clean one-parameter frontier rather than a trade among
three unobserved forces.

There is a third degeneracy, and it closes the escape the first two leave open.
Both conditions above drive the information gain itself to zero, which invites
an obvious repair: give the generator a hidden state that genuinely carries
uncertainty. This architecture already holds one — the unresolved word, whose
trie-node posterior at an ambiguous prefix is spread over many candidates. So
the construction was built and measured: hidden factor the word identity over
the {{STAGE3_VOCAB_SIZE}}-type lexicon, action the emitted letter, $\mathbf{A}$
the prefix-conditioned next-symbol map, belief the trie-node posterior, and
$\mathbf{B}$ the identity for *every* action, because a word does not become a
different word part-way through being spelled — the invariant that pins the
word level ([@sec:word_level]). Emitting a letter changes what is observed,
never which word is being spelled. On this construction the information gain is
measured and it is *not* zero: {{GENERATION_WORD_STATE_IG_EMPTY_NATS}} nats at
the empty prefix, and positive at the most-populated ambiguous trie node of
every probed depth — *{{GENERATION_WORD_STATE_PREFIX_DEPTH1}}*,
*{{GENERATION_WORD_STATE_PREFIX_DEPTH2}}*,
*{{GENERATION_WORD_STATE_PREFIX_DEPTH3}}*,
*{{GENERATION_WORD_STATE_PREFIX_DEPTH4}}*, selected by mass, not by hand. What
is zero is the term's per-action *spread*: across all
{{STAGE0_LETTER_ALPHABET_SIZE}} actions the information gain, the pragmatic
value, and $G$ each vary by exactly {{GENERATION_WORD_STATE_MAX_G_SPREAD_NATS}}
at every probed prefix. Transitions that do not depend on the action make the
predictive state — and therefore the predictive observation — identical across
actions, so both terms of [@eq:efe] are constants over the policy space, and
the softmax in [@eq:policy_posterior] cancels constants. The induced policy
posterior equals $\sigma(\ln \mathbf{E})$, the pure habit, to a measured
maximum deviation of {{GENERATION_WORD_STATE_POLICY_HABIT_MAX_ABS_DEV}}. The
recorder (`scripts/record_generation_word_state_claims.py`, prospectively
specified with its refusal conditions in the development session as decision
78) refuses to write if any spread is
nonzero, and its action-dependent control — the same node, the same belief,
transitions that differ by action — spreads $G$ by
{{GENERATION_WORD_STATE_CONTROL_G_SPREAD_NATS}} nats, so the zeros were
recorded by machinery that can report a nonzero.
`tests/test_generation_word_state.py` rejects the general class: for any
emission, belief, preference, and precision, an action-independent transition
stack leaves the policy posterior on the habit.

The three degeneracies are therefore of two kinds. A deterministic transition
or an uninformative likelihood makes the epistemic term zero; an
action-independent transition leaves it positive and makes it irrelevant —
nonzero information gain that differentiates nothing. The consequence is worth
stating plainly, because it is a fact about generation rather than about this
implementation: a hidden state with real uncertainty is not enough. The
epistemic term bears on action only where different actions induce different
predictive distributions over that state, and over a *fixed* word state no
emitted letter can do so — the environment would have to hold the true word
and answer differently depending on what is asked of it. That is
interrogation, not generation, and the regime it describes already exists in
this project under that description.

The obvious next question, then, is what happens where the information-gain
term is not zero *and* action-bearing, and this project has already answered
it rather than deferred it. Budgeted
attention at the letter-to-phone level ([@sec:limitations]) inverts this
section's regime exactly: there the preference term is absent and the
information-gain term is the *sole* selection criterion, with each candidate
position scored by the posterior entropy it is expected to leave. That arm was
measured against the exhaustively optimal fixed schedule and it loses, by
{{BUDGET_ADAPTIVITY_LOSS_PTS}} points at a {{BUDGET_FRACTION_PCT}}% budget.

The generation and budgeted-attention arms together are the decomposition, each term measured alone. Where the
information-gain term vanishes and preference carries the policy, generation
works: every sampled word is in the lexicon and the induced distribution sits
{{GENERATION_BEST_TV}} in total variation from the target. Where the
information-gain term is the whole policy and preference is absent, it is beaten
by a schedule that performs no inference at decode time. Neither result is an
argument that expected free energy is the wrong objective. Together they are the
narrower and more useful statement that in this architecture the two terms are
separable, and that on the evidence here the pragmatic term is the one carrying
the work.

#### Exact distributional distances

The policy generator's whole state is the prefix it has emitted so far:
`lexicon_preference` reads the prefix and nothing else, and the current letter is
reset to a space at every word boundary. Its distribution over the lexicon is
therefore the product of the action posteriors along each word's own path through
the trie. The ancestral comparator admits the same computation with each action
posterior replaced by the corresponding Brown-transition column. In both cases
the finite lexicon's complement is one explicit out-of-lexicon remainder. An
attempt that terminates immediately produces no token and is conditioned out on
both paths, exactly matching `GenerationResult.words`. Thus the model, baseline,
and their difference share one exact estimator rather than comparing an exact
model value with a pooled baseline sample.

That distinction changed a recorded result, which is why it is stated here
rather than in a methods note. The sampled sweep put the optimum at
$\gamma$ = {{GENERATION_SAMPLED_BEST_GAMMA}}, scoring
{{GENERATION_SAMPLED_BEST_TV}}; exactly, the optimum is
$\gamma$ = {{GENERATION_BEST_GAMMA}} at {{GENERATION_BEST_TV}}. The sampled and
exact values differ by up to {{GENERATION_MC_EXACT_MAX_ABS_GAP}} on the shared
grid, and the direction of that gap is not random. A run of
{{GENERATION_WORDS_PER_RUN}} words over {{GENERATION_SEEDS}} seeds is
{{GENERATION_SAMPLED_TOKENS}} tokens against a {{STAGE3_VOCAB_SIZE}}-type target,
so the empirical distribution is supported on at most
{{GENERATION_SAMPLED_TOKENS}} types and its distance to the truth is inflated by
construction — before any question of which $\gamma$ is best. Pooling the seeds
into one statistic with no spread reported left nothing to signal it.

The finding survives. The optimum is interior, both endpoints fail for opposite
reasons, and the exact curve is the same shape. What did not survive is the
operating point, and the ancestral comparator's old pooled distance is now
retained only as an explicitly sampled diagnostic: its pooled estimate is
{{GENERATION_SAMPLED_BASELINE_TV}}, whereas the compatible exact comparator is
{{GENERATION_BASELINE_TV}}. A sampled estimate of a distribution's distance to a
many-type target was never going to resolve the small gaps that separate
adjacent grid points near the minimum.

This gives the generation gate a property none of the recognition gates has: it
fails in **two opposite directions**, so it cannot be passed by turning a dial to
an extreme. Every gate in [@sec:gate_design] can be characterized by the single
question of whether a threshold sits between a baseline and a ceiling. This one
additionally requires that a free parameter be neither too small nor too large,
and the U-shape is what demonstrates the requirement rather than asserting it. [@fig:generation_samples] shows the sampled strings at each setting.

![Distance to the true word distribution against policy precision. Both extremes generate badly for opposite reasons: at low precision the policy posterior becomes the habit and the model *is* the sampling baseline, while on the shipped artifact high precision follows a greedy conditional-prefix path whose distinct-word count collapses to {{GENERATION_DEGENERATE_DISTINCT}}. That path need not be the globally most probable word in another lexicon. The optimum is interior, which is what gives this measured gate a failure mode in two directions. A boundary optimum, a monotonic curve, or a high-precision distribution that retained the full support would falsify this panel.](../../output/figures/generation_sweep.png){#fig:generation_sweep width=90%}

### What the distance metric does not catch, and how a degenerate generator beats it

The degenerate generator scores {{GENERATION_DEGENERATE_TV}}, which *beats* the
baseline's {{GENERATION_BASELINE_TV}} while holding its distinct-word count at
{{GENERATION_DEGENERATE_DISTINCT}} indefinitely. Distance
alone would admit it. The second gate conjunct requires more than
{{GENERATION_DIVERSITY_FRACTION_PCT}}% of the ancestral run's sampled diversity,
or more than {{GENERATION_DIVERSITY_FLOOR}} distinct words under the declared
run protocol; only that diversity floor rejects the collapsed endpoint.

That is recorded here for the same reason [@tbl:word_miswirings] records the
{{STAGE3_MISWIRINGS_CLEARING_BAR_WORD}} word-level mis-wirings that clear their
bar: a guard's blind spots are part
of its specification, and a project that reports only what its checks catch is
reporting half of what it knows.

![What the three regimes actually write. Real output regenerated at draw time
from a fixed seed through the shipped generators — never a mock-up — one
column per operating point of [@fig:generation_sweep], each word colored by
lexicon membership, with the column's membership rate and distinct-word count
beneath. The habit-only baseline drifts out of the lexicon; the shipped
operating point stays inside it while remaining diverse; the degenerate
endpoint has a distinct-word count of {{GENERATION_DEGENERATE_DISTINCT}} with a
perfect membership rate — the visible
form of the metric that could not be the gate. The panel refuses to draw if
the regenerated columns contradict the recorded sweep: a degenerate column
with more distinct words than the shipped one, or a shipped column below the
baseline's membership rate, would each falsify the sweep's
reading.](../../output/figures/generation_samples.png){#fig:generation_samples width=98%}

## Helpful and harmful cross-level interventions

### The lexical channel helps at every tested weight

This table sweeps the lexical message's mixing weight and shows it beating the
uncoupled decoder at every weight tested, unlike the harmful letter-coupling
sweep it is contrasted against.

| Mixing weight | Accuracy | vs anchor | Scrambled control |
|---|---|---|---|
{{LEXICAL_SWEEP_TABLE_ROWS}}

: The lexical channel over {{LEXICAL_EVAL_POSITIONS}} held-out positions, with a real prefix available at {{LEXICAL_POSITIONS_WITH_PREFIX_PCT}}% of them. McNemar at the optimum: {{LEXICAL_BEST_MCNEMAR_B}} positions where only the coupled decoder is right against {{LEXICAL_BEST_MCNEMAR_C}} where only the anchor is, $p =$ {{LEXICAL_BEST_MCNEMAR_P}}; the coupled decoder's share of the {{STAT_LEXICAL_DISCORDANT_N}} discordant pairs is {{STAT_LEXICAL_FAVORED_RATIO}}, exact {{SHARE_CONFIDENCE_PCT}}% interval [{{STAT_LEXICAL_SHARE_CI_LOW}}, {{STAT_LEXICAL_SHARE_CI_HIGH}}] ([@sec:multiplicity]). {#tbl:lexical_channel}

The first predicted signature appeared in [@tbl:lexical_channel]. The channel
helps by {{LEXICAL_BEST_MARGIN_PTS}} points — for scale, the entire
letter-to-phone margin over its own context-free baseline is
{{STAGE2_G2P_MARGIN_PTS}} points, so the channel is worth several times the
level's own headline result. The second did not survive the correction of
decision {{DECISION_LEXICAL_BACKOFF}}: the sample argmax is interior, at weight
{{LEXICAL_BEST_ALPHA}}, but the highest tested weight trails it by only
{{LEXICAL_PEAK_VS_HIGH_ALPHA_PTS}} points on
{{LEXICAL_PEAK_VS_HIGH_MCNEMAR_B}}/{{LEXICAL_PEAK_VS_HIGH_MCNEMAR_C}} discordant
positions (*p* = {{LEXICAL_PEAK_VS_HIGH_MCNEMAR_P}}), the effect helps at every
tested weight, and the wider grid of [@tbl:channel_factorial] places the same
message's best cell at full weight. The sweep therefore records an ordering,
not an optimum: it does not distinguish an interior peak from a plateau, and
the reversal the defective lookup had shown at weight {{LEXICAL_HIGH_ALPHA}}
was an artifact of matching the wrong context.

![The composition result, with the registered lexical prediction drawn on the evidence and the earlier coupling sweep shown as its motivating contrast. Both sweeps are measured on the same {{LEXICAL_EVAL_POSITIONS}} held-out receiver positions, and both are linear mixtures anchored at the uncoupled decode at weight zero. They do not hold insertion site fixed: coupling folds into the observation likelihood over phones, whereas the lexical message mixes into the phone predictive. The lexical message rises to a plateau whose sample argmax is interior and does not reverse at the highest tested weight; the letter predictive declines monotonically. Every lexical point now carries an exact conditional paired interval, derived at render time from that weight's own recorded discordant counts over the shared denominator, so the plateau's overlapping intervals show on the figure what this caption asserts about the argmax rather than asking the reader to take it on the point estimates alone. The bands show the registered lexical *interior optimum* class, which the corrected sweep's argmax satisfies without distinguishing it from a plateau, and the already-observed coupling class. The generator recomputes both shape verdicts from recorded sweep claims and refuses to draw a contradictory band. The coupling curve's intermediate segment is schematic between recorded endpoints; its full-mixing collapse is {{COUPLING_ALPHA1_ACCURACY}}% against the uncoupled {{COUPLING_ALPHA0_ACCURACY}}%. A harmful series extending past its last recorded anchor, an endpoint unequal to a recorded measurement, prose calling both curves prospective, or a drawn point whose estimate does not reproduce from its own counts over the recorded denominator would falsify this panel's account. The shape is evidence for a working hypothesis, not a universal law.](../../output/figures/channel_comparison.png){#fig:channel_comparison width=92%}

```{=latex}
\csname @ifclassloaded\endcsname{beamer}{}{\clearpage}
```

### Sweep-shape diagnostics and their scope

The part worth keeping is a diagnostic about the shape of a weight sweep, for
which these two sweeps are the evidence base:

> In the two measured sweeps, the message the receiver lacked helped at every
> tested weight, with a sample argmax strictly between zero and one that the
> sweep cannot distinguish from its highest weight, and the message it already
> observed had an optimal weight of zero. **The working hypothesis this supports: a monotone
> decline over the whole grid means no weight rescues the message** — a
> diagnostic every future channel either confirms or breaks.

That converts an expensive question into a cheap one in one direction only.
Whether to build an inter-level channel is otherwise answerable only by building
it; a monotone decline says no tuning will rescue this message into this
receiver at this insertion point, which is a verdict on the channel as built.

The converse does not hold, and the oracle intervention is why. An interior
optimum would say the mixing weight matters, and a weight can matter both when the message
carries something the receiver lacks and when it partially corrects a receiver
that is misspecified. The typed word-prior curve below improves through its high
tested endpoint and therefore locates no optimum. Its helpful direction from a
message predicted to be redundant still shows that sign alone does not separate
the two accounts.

The hypothesis also predicted where the remaining channels would and would not
pay, and one of those predictions has since been run — with an instructive
qualification. A document-position level passing a type distribution into the
clause joint prior
sends something no clause model can represent, and the channel demonstrably
moves the clause level's type prior ([@sec:narrative_level]). But the case-only
clause arm, which is the receiver this channel was measured against,
measured how its fitted predictive changes when a privileged reference type
selects the chain — {{STAGE5_PRIVILEGED_LABEL_GAIN_PCT}}% ([@sec:clause_level]). That
small reference suggests low sensitivity; it does not hard-bound an intervention
under misspecification. This is the sensitivity term of the
hypothesis doing the work that novelty cannot: the message moves a prior, and
the receiver's predictive is barely a function of the variable being moved.
Potential value prices the message, sensitivity prices the delivery. The
potential-value term rests on the interventions that vary it — the coupling,
the lexical channel, the generation channel, and the punctuation-type-oracle→word-prior arm below;
the sensitivity term on the case-only clause arm's privileged reference and the
L6→L5 prior intervention, the only intervention that has tested it. Whether the
widened receiver of [@sec:clause_widened] is more sensitive to the same message
is an open question this paper does not answer: no channel has been measured
into it. Five of the {{CHANNEL_COUNT}} registered interventions are separately
measured, and they bear on those two terms and nothing beyond them. Of the two
case-word seams that make up the rest, the bridge's influence is now measured as an exploratory
development-population sweep under a fresh registration
([@sec:case_word_sweep]), and the privileged-oracle diagnostic remains the
structural inlet whose evaluation records nothing.

### The predicted privileged-oracle harm did not reproduce {#sec:clause_word_channel}

The predicted harm did not reproduce.

The cheapest available test of the potential-value term was a privileged
punctuation-type-oracle→word intervention: a train-estimated
$P(\text{word}\mid\text{punctuation-defined sentence
type})$ mixed into the word level's prior for held-out typed sentences. The
message uses the same type variable the clause model represents, but it is a
reference label, not the clause Agent's posterior or an L5 output. The word
level already observes those words.
Under the biconditional form the prediction was
unambiguous — that channel harms, monotonically. It has since been built and
measured (decision {{DECISION_CLAUSE_WORD_HARM}}, corrected after the document-partition repair in decision {{DECISION_DOCUMENT_PARTITION_REPAIR}}): on the declared axis — the held-out perplexity of the
word prior, the quantity the pre-registration named as where harm would show —
the clause-typed word message produced a small *improvement* at every tested
positive weight (anchor {{CLAUSE_WORD_ANCHOR_PPL}} → best tested
{{CLAUSE_WORD_BEST_PPL}} at the high grid endpoint alpha =
{{CLAUSE_WORD_BEST_ALPHA}}, delta {{CLAUSE_WORD_BEST_DELTA_PPL}}). The sweep
therefore does not locate an optimum. The
token-direction tally agrees: {{CLAUSE_WORD_BEST_IMPROVED_TOKENS}} tokens improve
against {{CLAUSE_WORD_BEST_DEGRADED_TOKENS}} made less probable, while the
scrambled control is far worse ({{CLAUSE_WORD_SCRAMBLED_PPL_AT_BEST}}). No monotone harm
reproduced, so the predicted harm was **not** confirmed.

The population boundary is source-document disjoint: the typed statistic is
estimated on {{CLAUSE_WORD_TRAIN_DOCUMENTS}} Brown documents and evaluated on
{{CLAUSE_WORD_TEST_DOCUMENTS}} others, with
{{CLAUSE_WORD_DOCUMENT_OVERLAP}} overlap and every Brown category represented in
both arms. The type-free and swapped-type controls isolate the role of the
privileged type label. Replacing
both conditional rows by the type-free bridge marginal gives perplexity
{{CLAUSE_WORD_TYPEFREE_PPL_AT_BEST}} at the selected weight, while swapping the
declarative and interrogative rows gives
{{CLAUSE_WORD_SWAPPED_TYPE_PPL_AT_BEST}}; the real typed message is
{{CLAUSE_WORD_BEST_PPL}}. The vocabulary-axis reversal remains a wiring control,
not the evidence that sentence type itself matters.

The outcome is evidence against the biconditional and is what the
potential-value hypothesis expects to be possible. A coarse, type-conditioned
message is not cleanly redundant against a prior that is a marginal over words:
the receiver's prior carries no type structure at all, so the message can shift
mass the marginal had mis-assigned without telling the receiver anything it
could not have read off the observed word. Agreement between aggregate log-loss
and the majority token direction survives whole-document uncertainty. The mean
gain is {{CLAUSE_WORD_CLUSTER_LOGGAIN_NATS_PER_EMISSION}} nats per emission, with
a {{CLAUSE_WORD_CLUSTER_CONFIDENCE_PCT}}% whole-document interval of
[{{CLAUSE_WORD_CLUSTER_LOGGAIN_CI_LOW}},
{{CLAUSE_WORD_CLUSTER_LOGGAIN_CI_HIGH}}];
{{CLAUSE_WORD_CLUSTER_LOGGAIN_POSITIVE_DOCUMENTS}} documents favor the typed
message and {{CLAUSE_WORD_CLUSTER_LOGGAIN_NEGATIVE_DOCUMENTS}} favor the anchor.
The assumption-conditional document sign-flip sensitivity is reported only as
*p* ≤ {{CLAUSE_WORD_CLUSTER_LOGGAIN_P_FLOOR}}, its Monte Carlo resolution floor.
This resolves the descriptive sign on the reused population; it does not make
the correction confirmatory or identify whether the gain is conditional
information, mismatch correction, or regularization under the chosen mixing
rule. What neither metric supports is a large effect. The best point differs
only narrowly from the anchor ({{CLAUSE_WORD_ANCHOR_PPL}} to
{{CLAUSE_WORD_BEST_PPL}}), so the measured improvement
sits on a numerically narrow curve. That scale is reported by the injected
endpoints; no equivalence or near-null classification is inferred from it.

### Case-word influence sweep on the development population {#sec:case_word_sweep}

The case-word bridge's registered influence sweep
(`docs/decisions.md` entry {{DECISION_CASE_WORD_SWEEP}}; estimand and controls
in [@sec:hierarchy]) is the second intervention on the L3 word prior that was
specified before it ran, and the first in which the sender is the case
level's own posterior rather than a privileged reference. The receiver side
is scored over {{CASE_WORD_SWEEP_SCORED_TOKENS}} tokens of
{{CASE_WORD_SWEEP_EVAL_DOCUMENTS}} EWT-dev test documents —
{{CASE_WORD_SWEEP_SCORED_EMISSIONS}} word-level emissions — against the
zero-weight anchor, whose word prior has held-out perplexity
{{CASE_WORD_SWEEP_ANCHOR_PPL}} and which the recorder reproduces
bit-identically from the untouched receiver prior before any positive weight
is scored.

The mean per-token log-gain is positive across the tested weights and
peaks at an interior weight: the recorded best weight is alpha = {{CASE_WORD_SWEEP_BEST_ALPHA}},
at {{CASE_WORD_SWEEP_BEST_MEAN_LOGGAIN}} nats per scored token against the
anchor. The token-direction tally is not a uniform win:
{{CASE_WORD_SWEEP_ALPHA_0_3_IMPROVED}} tokens improve against the anchor,
{{CASE_WORD_SWEEP_ALPHA_0_3_DEGRADED}} are made less probable, and
{{CASE_WORD_SWEEP_ALPHA_0_3_TIED}} are exactly unchanged — the last the
no-message tokens the inlet never touches. Both controls behave as
registered: the scrambled-word negative control sits at
{{CASE_WORD_SWEEP_ALPHA_0_3_SCRAMBLED_MEAN_LOGGAIN}} nats per token at the
best weight, and the declared positive control leads the scrambled control
by {{CASE_WORD_SWEEP_ALPHA_0_3_POSITIVE_CONTROL_MARGIN}} nats, above the
pre-registered margin of {{CASE_WORD_SWEEP_POSITIVE_CONTROL_MARGIN_REGISTERED}}
nats and recorded as passing at every positive weight.

The whole-document clustered analysis expands each token's difference to its
word-level emissions ({{CASE_WORD_SWEEP_SCORED_EMISSIONS}} across the
{{CASE_WORD_SWEEP_CLUSTER_DOCUMENTS}} source documents), resamples whole
documents and sign-flips them: {{CASE_WORD_SWEEP_CLUSTER_RESAMPLES}} resamples at seed
{{CASE_WORD_SWEEP_CLUSTER_SEED}}. The clustered per-emission effect at the
best weight is {{CASE_WORD_SWEEP_ALPHA_0_3_CLUSTER_LOGGAIN_EFFECT}} nats per
emission, with a {{CASE_WORD_SWEEP_CLUSTER_CONFIDENCE_PCT}}% whole-document
interval of [{{CASE_WORD_SWEEP_ALPHA_0_3_CLUSTER_LOGGAIN_CI_LOW}},
{{CASE_WORD_SWEEP_ALPHA_0_3_CLUSTER_LOGGAIN_CI_HIGH}}];
{{CASE_WORD_SWEEP_ALPHA_0_3_CLUSTER_LOGGAIN_POSITIVE_DOCUMENTS}} documents
favor the message and
{{CASE_WORD_SWEEP_ALPHA_0_3_CLUSTER_LOGGAIN_NEGATIVE_DOCUMENTS}} favor the
anchor ([@fig:case_word_sweep]). The interval excludes zero on the
development population, which is a descriptive sign resolution, not a
confirmatory one, and the two units are not interchangeable: the clustered
estimand prices the bridge per word-level emission, the mean per scored
token.

![The case-word bridge's influence sweep across the shipped mixing-weight grid, on the EWT-dev development population. The solid series is the recorded mean per-token log-gain of the mixed L3 word prior against the bit-identical zero-weight anchor; the bracketed series is the whole-document clustered per-emission effect with its {{CASE_WORD_SWEEP_CLUSTER_CONFIDENCE_PCT}}% document-clustered interval at each weight; the dotted series is the scrambled-word negative control. The vertical rule marks the recorded best weight. Development evidence under a fresh registration, never confirmatory; the scored population is {{CASE_WORD_SWEEP_SCORED_TOKENS}} tokens in {{CASE_WORD_SWEEP_EVAL_DOCUMENTS}} documents, expanding to {{CASE_WORD_SWEEP_SCORED_EMISSIONS}} clustered emissions. The caption is falsified by a plotted value disagreeing with its recorded token, by an interval drawn around the per-token mean rather than the per-emission clustered effect, by the anchor point drawn away from exactly zero on any series, or by the recorded best weight failing to lie on the drawn grid.](../../output/figures/case_word_sweep.png){#fig:case_word_sweep width=75%}

The sweep's scope warnings are the registration's own and travel with the
result. The scored documents are the EWT dev population already scored for
Stage 4 case assignment: development evidence under a fresh
registration, never confirmatory, never "untouched", and the shared corpus's
test file remains spent (decision {{CONFIRMATORY_REGISTRATION_DECISION}}).
Cross-corpus transfer — an EWT-estimated message applied to the Stage 3
Brown-receiver vocabulary through a lowercased exact-string match — is part
of the estimand, named as such. The recorded receiver sensitivities of the
other arms bound nothing here. The bridge carries the exploratory prior-only
evidence reported above; its privileged-oracle diagnostic remains a
registration-only structure. These independently measured arms do not establish
an executed cascade.
Every derived label here remains *agreement, not accuracy* wherever quoted;
nothing in the sweep promotes a development-population result to
confirmatory, which would require a fresh population and a fresh
registration.

### Evidence and limits of the information-value hypothesis

The honest limit is the evidence sequence itself. The letter-predictive harm
motivated the account. The later registered lexical channel helped. The
L6→L5 prior intervention moved its receiver but has a non-confirmatory
recognition result. And the punctuation-type-oracle→L3 harm prediction failed,
with a directionally concordant helpful outcome under a corrected partition —
and that correction is not a fresh confirmatory population. Taken together,
this is a small, dependent set of experiments in one language, not
confirmation of an architectural principle.

The measurement that would test the hypothesis directly is not reported here.
Every channel in this paper is measured as an end-to-end intervention at a
fixed mixing rule, and this confounds the potential value of a message with the
insertion rule that delivers it. Separating the two requires estimating the
conditional information the message carries about the receiver's target given
what the receiver already has — $I(Y; M \mid X)$, or equivalently the held-out
log-loss reduction of a flexible predictor given $(X, M)$ against $X$ alone —
under the receiver's own proper loss and independently of any mixing weight.
That quantity is the Bayes-optimal potential-value term. Under a misspecified
receiver, though, it does not hard-bound realized improvement. What would turn
each channel from one number into a mechanistic decomposition is comparing
three things directly: the potential value, the mismatch correction, and the
realized effect.

### Message source and insertion point under proper loss

The letter-predictive and lexical sweeps above were each measured at exactly
one insertion point, which confounds which message was sent with where it
entered the receiver. A channel factorial crossing both factors, a
combination-rule control, a fitted variant, and a re-scoring of every cell on
the receiver's own proper predictive loss are reported in
[@sec:channel_factorial]: message source and insertion point both matter, but
the apparent site effect is largely an artifact of linear mixing, and every
decided sign agrees across the accuracy and loss axes.

## Learning outcomes beyond baseline comparisons

Beating the context-free baseline is necessary but not sufficient. Level 2's
emission and transition are both learned, and one of the mis-wirings tested
against them — an emission normalized over phones instead of letters and then
renormalized so `pymdp` still validates it — clears the baseline bar while
wrong. Only structural invariants checked before renormalization and a
value-for-value comparison against an independent re-derivation catch it
([@tbl:miswiring_signatures]); the catalog and the invariants that pin the
emission are set out in [@sec:audit_trails].

### Count-based learning and its registered prediction

Until this result, every matrix in the project was estimated by counting
outside `pymdp` and installed as a fixed array: the agents inferred, but had
never learned. The prediction in decision 22 was committed before the first
versioned learning recorder and outcome in decision 23. That chronology
establishes repository ordering; Git does not timestamp any earlier uncommitted
execution. The conjugacy checks retain descriptive ledger status, and this
protocol is outside the current machine-checked registration-provenance inventory.

The prediction is an identity, not a hope of approximate agreement. Add-$k$
counting *is* the posterior mean of a categorical–Dirichlet model, and under
the letter level's identity emission the filtered posterior at every step is a
one-hot on the observed letter, so `pymdp`'s Dirichlet accumulation over
posterior outer products is the accumulation of observed bigram indicators.
The learned transition and the counted transition are the same arithmetic in
different code, and were therefore required to agree to float32 rounding.
They did, to a maximum absolute deviation of
{{LEARNING_MATCHED_MAX_ABS_DELTA}} over {{LEARNING_STREAM_POSITIONS}}
positions, with identical perplexity on the passage the counts were
accumulated over ({{LEARNING_LEARNED_PERPLEXITY}} against {{LEARNING_COUNTED_PERPLEXITY}};
in-sample, so not comparable to the shipped agent's held-out score in
[@tbl:letter_comparators]) as the corollary. The agreement is evidence only because the controls show where
it fails: a Dirichlet prior mismatched to the comparator's pseudocount breaks
it by {{LEARNING_MISMATCHED_MAX_ABS_DELTA}}, re-matching restores it to
{{LEARNING_RESTORED_MAX_ABS_DELTA}}, and a learning rate other than
{{ONLINE_LEARNING_RATE}} breaks
the conjugate identity outright. `lr_pB` is therefore not a tuning knob here:
any value except one makes the learned matrix an estimator of nothing the
ledger describes. [@fig:learning_agreement] plots the agreement and its
mismatched-prior control.

![Every transition cell, learned against counted, log–log. Left: matched
priors ($k$ = {{LEARNING_PSEUDOCOUNT}}) over {{LEARNING_STREAM_POSITIONS}}
positions of the Stage 1 held-out text — a single point off the identity line
beyond a maximum absolute deviation of {{LEARNING_MATCHED_MAX_ABS_DELTA}}
would falsify the
conjugate-counting claim of decisions 22–23. Right: the control that proves
the comparison can fail — with the Dirichlet initialized at
$k$ = {{LEARNING_MISMATCHED_PSEUDOCOUNT}} while the comparator counts at
$k$ = {{LEARNING_PSEUDOCOUNT}}, the cells leave the line, most strongly where
counts are sparse and the prior dominates. A tight diagonal in the right panel
would mean the left one measured nothing. The matrices are the recorder's own
saved artifact, and the panel refuses to draw if their deviations disagree
with the recorded
claims.](../../output/figures/learning_agreement.png){#fig:learning_agreement width=95%}

Nothing above claims learning *helps*: agreement with counting is a
correctness anchor for the mechanism. The version that is a real
active-inference claim — a level that keeps learning beating a frozen one
on non-stationary text — has since been measured (decision 28). On a
drifted Brown genre stream (government then romance, letter-bigram
drift {{ONLINE_MAX_DRIFT}}, measured not asserted), the online agent that
keeps learning across the drift scores {{ONLINE_PPL_ONLINE_POST}} on
post-drift held-out text against the frozen control's
{{ONLINE_PPL_FROZEN_POST}}, a gap of
{{ONLINE_FROZEN_VS_POST_GAP_PPL}} perplexity. The frozen control is the arm
the prediction requires to lose, and its losing is informative because the
alternative is diagnostic rather than merely disappointing: a frozen model
fitted to the first genre only that did *not* lose on post-drift text would
mean the drift was not real or the online updates were not happening, and the
comparison would be measuring neither. The re-fit control
({{ONLINE_PPL_REFIT_POST}}) bounds the same comparison from the other side by
scoring at least as well as the online agent, and the pre-drift identity
confirms the online path is still counting at
`lr = {{ONLINE_LEARNING_RATE}}` (P3). The prediction was committed before its
measurement (decision {{DECISION_ONLINE_VS_FROZEN}}), but the entry is not in
the registration-provenance registry, so its ledger status is descriptive
rather than confirmatory, it sits outside the significance family, and no
interval is attached: the gap is a point comparison.

## Bits-per-symbol information budget

Perplexity throughout is [@eq:perplexity] — the exponentiated mean negative log
predictive probability under the model's own one-step-ahead prior — reported in bits
per symbol where that is the more legible unit,

$$
H_2 \;=\; \log_2 \mathrm{PPL},
$$ {#eq:bits_per_char}

and the fraction of the information available at a given model order that the
agent actually captures is

$$
\eta \;=\; \frac{\log \mathrm{PPL}_{\text{uniform}} - \log \mathrm{PPL}_{\text{model}}}{\log \mathrm{PPL}_{\text{uniform}} - \log \mathrm{PPL}_{\text{opt}}},
$$ {#eq:signal_captured}

with $\mathrm{PPL}_{\text{opt}}$ the empirical order-1 conditional-entropy
optimum of [@eq:conditional_entropy], computed on the sample the numerator is
scored over. [@eq:signal_captured] is the quantity the gate discussion in
[@sec:gate_design] turns on: it separates "the model is weak" from "the order is
exhausted", and those two situations call for opposite responses.

In information units the letter level reaches {{STAGE1_LETTER_NATS}} nats, which by
[@eq:bits_per_char] is {{STAGE1_LETTER_BITS}} bits per character.

| Quantity | Value (nats) |
|---|---|
| Uniform over the letter alphabet | $\log$ {{STAGE1_UNIFORM_BASELINE}} |
| Bigram agent | {{STAGE1_LETTER_NATS}} |
| Order-1 conditional-entropy floor | {{STAGE1_BIGRAM_FLOOR_NATS}} |
| Gain realized over uniform | {{STAGE1_GAIN_OVER_UNIFORM_NATS}} |
| Maximum gain any order-1 model can realize | {{STAGE1_MAX_BIGRAM_GAIN_NATS}} |

: The Level 1 information budget. The realized-gain and maximum-gain rows are the whole argument of [@sec:gate_design]: a target of {{STAGE1_DECLARED_GATE_NATS}} nats over uniform exceeds the maximum gain achievable on the training distribution by a factor of {{STAGE1_CEILING_SHORTFALL_RATIO}}, so the corpus the model was fitted to does not hold the first-order information the bar demands. The bound is under $p_{\text{train}}$ and not under the held-out evaluation sample ([@sec:background]); the finding is that the threshold was miscalibrated, not that a held-out score above the ceiling is impossible. {#tbl:letter_information}

Against the training distribution's order-1 budget, the agent's held-out gain is
{{STAGE1_BIGRAM_SIGNAL_CAPTURED_PCT}}% of the information an order-1 letter model
can access there; by [@eq:signal_captured], with the optimum taken on the scored
passage itself, it is {{STAGE1_HELDOUT_SIGNAL_CAPTURED_PCT}}%. [@fig:information_ceiling] draws that budget directly in nats, with the
declared threshold placed outside the achievable region.

![The Level 1 information budget in nats. The declared {{STAGE1_DECLARED_GATE_NATS}}-nat-over-uniform target lies beyond the conditional-entropy ceiling of the training distribution, so it is unreachable at this model order on that population regardless of implementation quality; [@eq:heldout_bound] shows it is unreachable on the held-out evaluation passage too, using that passage's transition entropy plus its fixed initial-prediction loss on the all-position denominator rather than extrapolating from training. A ceiling at or below the declared target, or a gain that did not equal the uniform cross-entropy minus the applicable all-position floor, would falsify this panel.](../../output/figures/information_ceiling.png){#fig:information_ceiling width=85%}

## Confirmatory replication on the held-back split

The members of the frozen significance family with a genuine held-back
partition available were the ones computed on EWT populations. Under the
registered protocol of [@sec:confirmatory_method] they were scored against the
whole held-back `en_ewt-ud-test.conllu` split exactly once, with every model
parameter read unchanged from the development fits and both controls run on the
same population first. Every registered direction and threshold prediction held:
the Stage 4 derived comparison and the Stage 6 sign test cross the frozen
per-test threshold of {{CONFIRMATORY_FROZEN_ALPHA}} in the direction their
development result recorded, the Stage 4 gold comparison and the case-only Stage 5 ablation's
classification yield no discordant pair and are therefore not shown rather than
shown absent, and the other members of the family were not rescored and remain
development-population inference. The run cannot be repeated — the sentinel
claim marks the population spent — and the per-member detail, the controls, and
the document-clustered intervals are set out in [@sec:confirmatory].

## What the historical component results establish

They establish that {{BUILT_LEVEL_COUNT_WORD}} levels have real pymdp Agent builders and recorded
figures traceable to runs. They do not establish that every headline score used
live Agent inference. L1 does; the exact or NumPy paths of L2A, L2B, L3, and L6 and their
live-Agent checks are inventoried in [@sec:verification]. L4 uses a NumPy
forward filter whose live-Agent recursion reproduces it at every one of the
{{STAGE4_PYMDP_POSITIONS}} held-out positions with
{{STAGE4_PYMDP_ARGMAX_DISAGREEMENTS}} argmax disagreements and a largest
posterior deviation of {{STAGE4_PYMDP_POSTERIOR_MAX_ABS_DEV}}. The retained case-only L5
ablation uses a closed-form mixture whose live-Agent recursion reproduces the NumPy joint
oracle with {{STAGE5_PYMDP_TYPE_ARGMAX_DISAGREEMENTS}} type-argmax
disagreements and a largest joint-posterior deviation of
{{STAGE5_PYMDP_POSTERIOR_MAX_ABS_DEV}} over all {{STAGE5_PYMDP_POSITIONS}}
held-out positions, which is posterior agreement rather than score equivalence
for the headline estimand; the canonical widened L5 mixture is checked against oracle
filtering ([@sec:clause_widened]). The clause level clears a gate
registered before its artifact existed, on the widened observation alphabet,
with its paired privileged-label interval resolved away from zero
([@sec:clause_widened]); that gate is a development-session specification and
has no held-back replication. The case-only clause model it displaced is
retained as an ablation, with a non-passing gate and a paired privileged-label
interval spanning zero, and its instrument recovers a planted effect in the
detectability control ([@sec:clause_level]). The system generates
text under policy selection with a measured interior optimum
([@sec:generation]); its {{CHANNEL_COUNT_WORD}} registered receiver
interventions — {{CHANNEL_SEPARATELY_MEASURED_COUNT}} measured, the case-word bridge measured as an exploratory
prior-only sweep ([@sec:case_word_sweep]), and the privileged-oracle
diagnostic the one registered seam that records no estimand — classify as
{{CHANNEL_OUTCOME_SUMMARY}}. By source kind,
{{CHANNEL_SOURCE_SUMMARY}}. The selected lexical channel, L3→L1g generation seam, and
corrected punctuation-type-oracle→L3 word-prior result help; letter coupling
harms; and L6→L5 moves a `D` prior. The oracle→L3 arm's predicted harm did not
reproduce, but it is not an L5-output seam. The L3→L1g and L6→L5 model-output
seams and the oracle intervention were not executed as a connected cascade, and
the privileged-label gain of the retained case-only clause ablation remains
unresolved ([@sec:hierarchy] states why neither is a hard bound). Both that arm
and the L6→L5 intervention are measured against the case-only clause model, not
against the widened one. A further
result, held at descriptive status in the ledger and outside the significance
family, comes from within-level parameter learning rather than from the
hierarchy. An agent that keeps learning across a distribution shift beats a
frozen control on the drifted stream — {{ONLINE_PPL_ONLINE_POST}} against
{{ONLINE_PPL_FROZEN_POST}} post-drift perplexity, a gap of
{{ONLINE_FROZEN_VS_POST_GAP_PPL}} — a point comparison with no interval; the
direction and the refusal clauses were committed before the run (decision
{{DECISION_ONLINE_VS_FROZEN}}, [@sec:verification]), and the ledger does not
promote the entry to confirmatory. What the numbers do *not* establish is language comprehension in any
behavioral sense, competitive capability on any benchmark, generality beyond
the corpora each level was scored on, or confirmation of any member other than those
that clear the frozen per-test bar on the held-back split ([@sec:confirmatory]); [@sec:limitations] draws the first three lines, and
[@sec:multiplicity] the fourth. [@sec:external_benchmarks] scores
{{BENCH_PROBE_COUNT_WORD}} zero-shot external models against an AILM comparator
on identical positions within each separate letter, phoneme, and G2P arm, using
benchmark-specific fixtures rather than the positions scored here. That comparison is a
reference frame: it is not contamination-controlled and settles no
competitiveness question in either direction.

```{=latex}
\clearpage
```

One further accounting spans the historical component and channel experiments
rather than a single level. Their declared significance tests form one family of size {{STAT_FAMILY_SIZE}} at familywise
$\alpha =$ {{STAT_FAMILYWISE_ALPHA}}, corrected two ways.
{{STAT_FAMILY_SURVIVOR_COUNT}} of the {{STAT_FAMILY_SIZE}} cross the fixed
Bonferroni per-test threshold of {{STAT_BONFERRONI_ALPHA}};
{{STAT_HOLM_SURVIVOR_COUNT}} cross their Holm step-down threshold at the same
familywise level, which controls the same error rate under the same dependence
assumptions. Of those, {{STAT_HOLM_ONLY_COUNT}} survive only the step-down. The Stage 6
sentence-sign test is nominally positive and {{STAGE6_SIGN_STEPDOWN_CLAUSE}}
(`stat_stage6_familywise_confirmed` = {{STAT_STAGE6_FAMILYWISE_CONFIRMED}}, a
ledger key that predates the vocabulary used here and records the fixed-threshold
crossing alone; `stat_holm_stage6_rejected` = {{STAT_HOLM_STAGE6_REJECTED}}, its
step-down bar being {{STAT_HOLM_STAGE6_THRESHOLD}}). It crossed that step-down
bar under the family ordering recorded before the lexical channel's backoff was
corrected (decision {{DECISION_LEXICAL_BACKOFF}}) and no longer does, which is
the instability of a step-down bar that [@sec:multiplicity] describes. This is
secondary evidence: a reversed-phase
model with worse mean log loss can still yield a nominally favorable sentence
sign, while the primary document-clustered log-loss interval spans zero on the
development population. That
survivor count is not a tally of endorsements. The biased prefix re-analysis of
[@sec:gate_design] is another survivor; it crosses the threshold precisely
because its subsample is unrepresentative. Within the family,
{{STAT_STRUCTURALLY_UNREJECTABLE_COUNT_WORD}} members cannot reject at the corrected
threshold for any outcome. And the family itself was declared after its members
were measured, so crossing a corrected threshold here means surviving a
retrospective procedure on the development population rather than confirming an
effect — the distinction [@sec:multiplicity] defines and this section's numbers
inherit. The historical family's development-population verdicts retain that status;
[@sec:confirmatory] separately reports the eligible held-back rescoring. The
benchmark family and locally specified GUM interval have their own stated
populations and inferential scope. The confirmatory run that could have
upgraded it was pre-registered, executed to its first data touch, and retired
by exhaustion for the `external_lexicon` members (decision
{{DECISION_REPLICATION_RETIRED}}): the registered population, upstream CMUdict
entries absent from the vendored pin, was measured empty because upstream is
byte-identical to the pin. The family, each member's endpoint and test, and
each substantive member's replication population remain frozen in
`CONFIRMATORY_REPLICATION`; the four split re-analyses are explicitly marked as
non-rescored diagnostics whose full-set member carries the replication
(`src/ailm/statistics.py`, decision {{DECISION_CONFIRMATORY_REPLICATION}}); the `ud_ewt_test` members were then scored exactly once against the held-back EWT
test split under decision {{CONFIRMATORY_REGISTRATION_DECISION}}
([@sec:confirmatory]): two clear the frozen per-test bar in their recorded direction, two yield no
discordant pair, and the other {{CONFIRMATORY_MEMBERS_UNRESCORED}} members of the family remain
development-population inference.
[@sec:multiplicity] states the family executably, gives all three
qualifications in full, and reports each primary effect with its exact {{SHARE_CONFIDENCE_PCT}}%
interval on the discordant win share ([@tbl:multiplicity_family]).

The external reference probes are scored on their own fixtures rather than on
the positions measured here, and [@sec:external_benchmarks] carries the protocol and the full
probe readings. The one result worth carrying into
the main text is that the comparator's raw letter-census margin is mostly
inventory compliance — whether a system will emit a space at all — and the
contrast that survives restriction to alphabetic targets, not the raw census
margin, is the finding.
