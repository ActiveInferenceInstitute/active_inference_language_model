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
8 documents inspected in the preceding bounded
development run, leaving 22 paired documents. This is
local prospective specification within the development effort, not independent
external registration or replication. Model parameters and train-only bridges
remain fixed; the connection strength is 0.5. The connected arm is called `integrated` in the recorded data. Its comparator
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
| Paired documents | 22 |
| Source sentences | 1040 |
| Source characters | 105,803 |
| Normalized lexical tokens | 18,188 |
| Scored normalized characters, including token spaces | 102,522 |
| Integrated mean document-level character log loss | 1.9345924139381503 |
| Named-channel-disabled mean document-level character log loss | 2.3766101701632905 |
| Mean paired gain, named-channel-disabled minus integrated | 0.4420177562251402 |
| Paired document-bootstrap interval | [0.42010385279024465, 0.46325671494396264] |
| In-vocabulary lexical tokens | 13,653 |
| Out-of-vocabulary lexical tokens | 4535 |
| Lexical vocabulary coverage | 75.065977567627% |

: Connected versus named-channel-disabled inference on the frozen raw-text population.
Losses and paired gains are in nats per scored normalized character, averaged
equally over documents. The paired percentile interval uses
10,000 document resamples at
95% coverage with seed
20,261,005. Positive gain favors connection; the recorded
interval indicates lower integrated loss. {#tbl:connected_gum}

The full-stream closed-vocabulary word loss is infinite because out-of-vocabulary tokens receive zero word probability.
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

The input corpus digest begins `cd96a285e733`, the local
protocol digest begins `c2ad1ee3b9aa`, and the
execution-input digest begins `0cd3b86fd81f`.
`scripts/record_connected_gum_claims.py` records the complete source/input binding,
per-document paired summaries, exclusions and environment alongside the ledger;
the report digest begins `045d953dd1d4`. Reproduction
requires these exact inputs rather than a moving upstream treebank.


![**Connected conditional schedule and paired GUM result.** Panel A derives
named arrows from the runtime contract. Solid channels are disabled together;
spelling and the dashed train-fitted UPOS encoder remain in both arms. Raw UPOS
is encoded after the word; shared encoding need not give identical outputs.
Case evidence updates clauses after tokens; type evidence updates phase after
sentences. Within-level recursion is omitted. Panel B shows document gains and
their equal-document mean with the locally specified
95% document-bootstrap interval; positive
gain favors connection. Population, normalization, strength and out-of-vocabulary
(OOV) accounting
follow [@tbl:connected_gum]; full-stream word loss remains
infinite because out-of-vocabulary tokens receive zero word probability. The figure authenticates typed claims, the
frozen report and execution inputs, then replays the registered statistic.
This is a consistency check, not an independent oracle or new measurement.
Neither a globally exact posterior nor individual channel effects are claimed.
A wrong channel direction, omitted shared observation, changed population, or
mean/interval inconsistent with the bound report falsifies this caption.](../../output/figures/connected_core.png){#fig:connected_core width=100%}

## Historical component gate outcomes and their scoring routes

The six levels are built and have recorded gate outcomes; Stage 5 passes on the widened observation alphabet
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
ceiling by a factor of 1.95, so no smoothing or
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
| Uniform over the alphabet | 30 | no |
| Context-free unigram letter frequencies | 19.9558 | no |
| Mis-wired variant: the bigram matrix in the emission `A` | 27.2499 | inverted — yields the *preceding* symbol |
| **Bigram agent: the bigram matrix in the transition `B`** | **11.9215** | yes |
| Order-1 optimum on the training distribution, $\exp H_{\text{train}}(X_t \mid X_{t-1})$ | 10.7506 | yes, optimally |

: Level 1 comparators. The uniform, unigram, mis-wired, and shipped-agent rows are measured on the same held-out passage; the conditional-entropy row is a property of the *training* distribution and is not a measurement on that passage. Uniform bounds the trivial end, the unigram row is the gate's actual comparator, the mis-wired row is the live negative control, and the conditional-entropy row is the best any first-order letter model does on the distribution the transition was fitted to. {#tbl:letter_comparators}

The agent beats the context-free unigram baseline by
8.0343 perplexity units, which is the gate. It does not beat
the order-1 reference: 10.7506 is
[@eq:conditional_entropy] evaluated on the training counts, so it is where an
optimal first-order model lands on the distribution it was fitted to rather than
a bound on what any model scores on this passage ([@sec:background]). In nats
the agent sits at 2.4783 against that reference's
2.375 — above it, which is the ordinary direction for a
held-out score measured against a training-set optimum.

That is 3.5755 bits per character, a value in the range expected
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
that rotated model scores 159.3667 against the
unigram bar's 19.9558 on the identical positions. So
the gate's margin does not survive destroying the context-to-conditional
assignment, even while every individual conditional distribution is kept
intact. The unigram rung and that control together test the thing the level
claims to do.

The alternative wording of the gate — a gain of at least
2 nats over uniform — asks for more first-order
information than the training distribution holds. The argument for this takes
one pass over the counts. The gain a first-order model can realize on the
distribution it is fitted to is at most the difference between the uniform
entropy and the conditional entropy of [@eq:conditional_entropy]. On the Brown
counts, that difference is 1.0262 nats, so the
2-nat target exceeds the level's entire order-1
information budget by a factor of 1.95. The bar
was calibrated for a model class this level does not implement. The shortfall
is in the order of the model rather than in the code, so no implementation,
tuning choice, or smoothing constant can reach it *on that distribution*. The
held-out passage is a different population, though, and needs its own bound.
[@eq:heldout_bound] supplies that bound: it caps the order-1 gain available
across all 1323 positions this level is scored on at
1.6106 nats, including the fixed uniform-start
prediction, and this bound is also short of the target. So the conclusion
survives on both populations rather than being carried from one to the other.
Against the training distribution's 1.0262-nat budget, the agent realizes 0.9229 nats
on held-out prose, or 89.9% of it — a failure
under the 2-nat wording, and a result close to the
training distribution's first-order optimum. Measured against the bound on the
passage it is actually scored on, the same realized gain is
57.3%. That is the same-population figure,
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
| Uniform over the letter alphabet | 30 | no |
| Context-free unigram letter frequencies | 19.9558 | no |
| Mis-wired variant: bigram matrix in the emission `A` | 27.2499 | inverted — yields the *preceding* symbol |
| **Bigram agent (`A` identity, `B` learned)** | **11.9215** | yes |
| Empirical order-1 optimum | 10.7506 | yes, optimally |

: Letter-level perplexity on held-out prose, Brown corpus bigram statistics (805 bigram types over 5,856,736 bigram tokens). The gate is beating the context-free unigram baseline, a margin of 8.0343 perplexity units; uniform is reported only to show what a crippled comparator looks like, since a context-free model clears it. {#tbl:letter_ladder}

[@fig:perplexity_ladder] plots [@tbl:letter_ladder] alongside its Level 2
counterpart on a shared axis, which is where the structural similarity between the
two levels is easiest to read: in both cases the model sits close to the empirical
order-1 optimum and far from uniform, and the interval between unigram and
optimum — not the
interval between uniform and model — is the space in which a bigram model can
distinguish itself.

![Perplexity comparators for both built levels, each on its own level's evaluation stream: uniform, context-free unigram, the trained bigram agent, and the empirical order-1 optimum on that stream. The two panels are separate populations and the panel does not compare them to each other — the Stage 1 letter panel is measured on held-out prose, while the Stage 2 phoneme panel is measured in-sample over the full dictionary, on the same pronunciations that trained the transition matrix ([@sec:limitations]), so the two ladders are not comparable as generalization claims. Each panel therefore states the population it was scored over and its unit beneath its own title, because two panels on one shared axis otherwise invite a rung-for-rung reading their populations do not support. The order-1 conditional-entropy floor is named on the canvas with its value rather than only in this caption, and its rule stops short of the panel edge so the label sits clear of it. The model-to-optimum gap is the remaining headroom at this order; the uniform-to-unigram gap is available to a model with no context at all. A missing comparator, a non-monotone ladder where the recorded identities require ordering, a model-to-optimum gap inconsistent with the ledger, or a panel drawn over an empty population would falsify this panel.](../../output/figures/perplexity_ladder.png){#fig:perplexity_ladder width=85%}

3.5755 bits per character sits in the range of published
estimates of English letter entropy at low context order. @shannon1951 measures
that entropy dropping as the conditioning context lengthens, and later bounded
estimates place fully contextual English well below any order-1 figure
[@cover1978gambling; @brown1992entropy]; the value here belongs to the shallow end
of that progression, which is where an order-1 model should land. This is a
consistency check, not a replication, and it is worth being precise about why it
cannot be one. The alphabet here adds space, period, apostrophe, and hyphen to the
26 letters, so uniform over it is 30 rather than Shannon's
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
| Uniform over the 39 phones | 39 | no |
| Context-free unigram phone frequencies | 27.5944 | no |
| **Phoneme bigram agent** | **16.7445** | yes |
| Exact all-position order-1 optimum | 15.8313 | yes, optimally |

: Part A perplexity over all 800,198 phone positions in the dictionary. This is a descriptive **in-sample dictionary protocol**: the same primary pronunciations supply the counts and the scored sufficient statistics, so it is not held-out generalization. The agent clears the context-free bar by 10.8499 perplexity units. The exact optimum is the unsmoothed maximum-likelihood start distribution plus transition columns on this identical all-position denominator, including every independently reset word start; it is therefore the model-class bound used by the gate audit. The lower transitions-only diagnostic, 14.7198, omits starts and is retained only as an optimistic reachability bound, never as the threshold for the all-position score. {#tbl:phoneme_comparators}

The same reachability argument as Level 1 applies with less room. The total gain
available over uniform at this level is
0.9016 nats on the exact all-position
population (and 0.9744 nats under the more optimistic
transitions-only relaxation), both below Level 1's inherited
2-nat threshold, so that wording is
unreachable here before a line of code is written. [@fig:perplexity_ladder]
shows both levels' rungs together.

### Family-disjoint evaluation and lexical overlap

The shared-prefix bound is also measured from the inside rather than only
stated. A family-disjoint arm assigns whole
4-character prefix families to one side of
the split, using the same seedless stride construction lifted one level up:
every fifth sorted family is held out.
As a result, no held-out word of length 4
or more shares its 4-character prefix with any training word
(0% shared, by construction, against
the interleave's 82.5%). Re-fitting the
identical decoder and the identical context-free lookup on the disjoint train
side gives 70.515% against
70.128% over
43,107 positions — a margin of
0.387 points beside the interleave's
0.546, with discordant counts
1387 against 1220 and
exact McNemar *p* = 0.001145. This is an exploratory
robustness arm: its direction was prospectively specified within the
development session, but its p-value stays outside the frozen significance
family, so it is a bound rather than a second significance claim. Read as a
bound, it cuts both ways. Morphological-neighbor transfer accounts for part of
the interleave margin, but most of the margin survives with the neighbors
removed, so the left-context mechanism is not an artifact of `catalog`
teaching `catalogs`. The disjoint assignment has its own recorded price,
though: holding out whole alphabetical families moves the initial-letter total
variation to 0.03458, well above the interleave's
0.00085. That is why this arm bounds the shipped
result rather than replacing it.

The bound is itself partitioned. The same disjoint held-out set can be grouped
by morphological family — the identical four-character family key the split
assigns sides with — and each family can then be scored against the same
context-free bar with the same paired test. Doing so asks whether the disjoint
margin is a level-wide property or the average of unlike parts. The answer,
prospectively specified within the development session in the arm's decision
stub, is that it is not level-wide. Across the 2422
held-out families, 241 carry at least
40 scored positions, and only
104 of them show a positive margin
— a share of 0.4315, against the stub's
registered bar of more than one half. The pooled disjoint margin of
0.387 points is therefore a majority-null
average, not a property most families carry. The largest evaluated family,
`cons`, scores
72.529% against
73.042% over
779 positions (margin
-0.513 points, exact *p* =
0.3438). By contrast, the
amer family's margin is
-16.201 points over
179 positions. This measured heterogeneity
is real rather than an artifact of the scoring, for two reasons. First,
position-weighted, the per-family scores reproduce the disjoint arm exactly
(70.515% over 43,107
positions, the prospectively specified positive control). Second, a
deterministic reversal control — reversed pooled correctness vectors paired
with the same family sizes — reaches a largest per-family margin of only
66.667 points, against the real
assignment's 100. Read together with the
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
| Context-free per-letter $\operatorname*{arg\,max}_p\, p(\text{phone} \mid \text{letter})$ | 70.255 | none |
| **Full-set NumPy forward filtering, after a bounded `pymdp` check with 0 decision disagreements** | **70.801** | left only |
| Forward-backward smoothing | 75.642 | whole word |
| Viterbi maximum-a-posteriori path | 75.306 | whole word |

: Position-aligned letter-to-phone accuracy over 42,820 held-out positions. Each position is scored against the phone the index pairing assigns to that letter, so the target inherits the alignment stipulation of [@sec:phoneme_level]. The gate compares the context-free lookup with forward filtering; smoothing and Viterbi are whole-word *references*, not ceilings. They use information the streaming architecture forgoes and they are not upper bounds on what a same-information model could reach — at Stage 4 a reference of the same kind is exceeded by the shipped model. They are recorded so the filtering agent's shortfall is visible rather than hidden. {#tbl:g2p_comparators}

On a length-matched subset the co-indexed letter alone is nearly sufficient, which
is precisely why the context-free baseline is already strong and why the headroom
for context is thin. Filtering captures part of that headroom and no more,
because `infer_states` conditions on observations up to the current position and
never revisits them. Smoothing and Viterbi see the whole word and gain several
points, as [@tbl:g2p_comparators] records. Reporting a smoothed figure would
flatter the gate while describing a different whole-word computation. The
canonical full-set filtering statistic is evaluated by the NumPy oracle only
after a bounded live-pymdp comparison finds 0 decision disagreements; no
shipped Level 3 path consumes Level 2's posterior. [@fig:g2p_accuracy] plots the baseline, the filtering decoder and the smoothing reference, and the
baseline and the filtering agent are visually indistinguishable at full scale —
which is the honest picture of a 0.546-point margin and the
reason the next subsection exists.

### Decoding margin and the lookup comparator

Before the margin is read at all, the baseline comparison bounds what it can mean. A
left-context lookup over the observed symbols, with
677 parameters, no latent state and no
inference at decode time, scores 72.8188% on this
same population against the shipped model's 70.801%. It does not
merely clear the declared bar; it beats the model by
2.0177 points, several times the margin this
section is about. Clearing a context-free comparator therefore does not show that
latent-state filtering is what produced the gain ([@sec:comparator_strength]).

McNemar's test conditions on the observed count of discordant pairs. The
discordant-pair interval constructed from it retains that conditioning; it is
not an unconditional interval for the difference in marginal accuracies
[@mcnemar1947; @newcombe1998paired]. The margin is
0.546 accuracy points, and the gate does not
rest it on a comparison of aggregates. It compares the decoders in paired form
instead: `decode_positions` and `baseline_positions` line them up position for
position, and the test runs only on the positions where the two disagree.

That test is the exact conditional form of McNemar's test [@mcnemar1947]. Its
statistic is the two-sided exact binomial of [@eq:mcnemar], not the $\chi^2$
approximation to it. Over 42,820 held-out positions there
are 2524 discordant pairs:
1379 where only filtering is correct against
1145 where only the baseline is,
*p* = 3.468e-06. The pairing is what carries that result: at
0.546 points the difference is too small for an
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
13,224 positions it reaches only
*p* = 0.02881, orders of magnitude weaker than the
full-set result and short of the gate's bar. [@fig:subsample_bias] plots both
failure modes against the full set and [@sec:gate_design] generalizes them.

Shrinking the evaluation is therefore not an option, and a full-set `pymdp` pass
is tens of thousands of `infer_states` calls.
`scripts/record_stage2_claims.py` computes the full-set statistics with the
implementation-independent NumPy reference decoder (the oracle) instead. Reading those figures as properties of the
`pymdp` agent is licensed by the 0-disagreement result of
[@eq:decoder_equivalence], which [@sec:verification] states, measures, and
enforces at record time.

### Learning an unobserved emission does not improve decoding

Decision 29 tested a different question from the supervised gate above: whether
`pymdp`'s Dirichlet observation-likelihood learning could recover a useful
letter-to-phone emission when the phone states were deliberately **unobserved**.
The target is the same co-indexed relation the gate uses, so the experiment
inherits the alignment stipulation and cannot escape it by learning.
The lane trained on 1621 pairs and evaluated
42,820 positions, using an evenly spaced stride of
16 for the recorded experiment. The fixed comparator
is the recorded add-1 count emission from the same lane, not a full-corpus count.

| Decoder or control | Held-out position accuracy (%) |
|---|---:|
| Latent-state emission learning | 12.487156 |
| Same-lane fixed count emission | 70.733302 |
| Observed-state P4 gold-lane control | 70.733302 |

The learned emission moved by 0.198777 in its largest entry,
but its decoding accuracy did not merely fail to improve on the fixed control: it
collapsed, from 70.733302 to 12.487156.
The observed-state
control recovered the same-lane count emission to a maximum absolute difference of
0, and the independent oracle and `pymdp` decoders
disagreed on 0 positions. The result is therefore
**NO IMPROVEMENT / QUALIFIED**: this experiment demonstrates that the learning path
is active and that its observed-state control is correct, not that unobserved-phone
learning recovers the co-indexed emission — and certainly not
grapheme-to-phoneme conversion, which no lane at this level attempts. The result
is recorded under decision 29 and its limitations are carried into
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
the 39 canonical ARPAbet phones.

| Comparator | Perplexity |
|---|---|
| Uniform over 39 phones | 39 |
| Context-free unigram phone frequencies | 27.5944 |
| **Phoneme bigram agent** | **16.7445** |
| Exact all-position order-1 optimum | 15.8313 |

: Phoneme sequence perplexity, measured **in-sample**: the evaluation positions are the pronunciations that trained the transition matrix, so this row is not a generalization claim and is not comparable to the held-out Level 1 figures ([@sec:limitations] states the defense and its limit). The margin over the context-free bar is 10.8499 perplexity units — comfortable, unlike Part B. `scripts/record_stage2_claims.py` computes this in closed form over all 800,198 phone positions of the primary pronunciation of every CMUdict base word [@cmudict], with the prior reset at each word boundary. {#tbl:phoneme_sequence}

The exact all-position gain available over uniform at this level is
0.9016 nats, below the inherited Level 1 threshold of
2 nats, so that wording is unreachable here too,
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
| Context-free per-letter $\arg\max_p P(\text{phone } p \mid \text{letter } l)$ | 70.255% | none |
| **Full-set NumPy forward filtering after a bounded pymdp check with 0 decision disagreements** | **70.801%** | left context only |
| Forward–backward smoothing reference | 75.642% | whole word |
| Viterbi maximum a posteriori (MAP) path | 75.306% | whole word |

: Position-aligned letter-to-phone accuracy over 42,820 held-out positions from 6484 test words, trained on 25,936 length-matched pairs. The gate compares the context-free lookup with forward filtering: a margin of 0.546 points, established by the paired test of [@eq:mcnemar] rather than by the difference of the aggregates. The smoothing and Viterbi rows are headroom the online architecture deliberately does not claim — see [@sec:limitations]. {#tbl:g2p_accuracy}

![**What to see: the letter-to-phone decoder beats a context-free per-letter lookup table by a thin but paired-significant margin, while the whole-word smoothing and Viterbi references shown alongside it are unreachable headroom the streaming architecture does not claim.** Position-aligned letter-to-phone accuracy on the Stage 2 held-out split — 42,820 aligned phone positions across 6484 words. The context-free per-letter baseline, the fitted filtering model scored over the full set by the NumPy oracle after a bounded pymdp check with 0 decision disagreements, and the forward–backward smoothing reference; the Viterbi reference is tabulated in [@tbl:g2p_accuracy] rather than plotted. Panel A is drawn at full scale and panel B repeats the same three bars on a truncated axis so the margin is visible; the dotted rule marks the uniform-guess comparator and the box reports the paired McNemar counts. The gate margin is the small step from baseline to filtering; the larger step to the whole-word references requires access the streaming architecture does not have, which is why they are references rather than ceilings: they are not upper bounds on what a same-information model could reach. A missing decoder, a filtering value that disagreed with the live-Agent check, or a whole-word reference below the corresponding causal decoder would falsify this panel. Each decoder bar carries a descriptive 95% Wilson interval [@wilson1927] derived at render time from its recorded accuracy and the 42,820-position held-out denominator both decoders were scored on — the recorder pairs them position-for-position for the McNemar counts, so one denominator serves both bars — stated per the decision-101 convention on that population, in percentage points of position accuracy. The two marginal intervals can overlap even where the paired test separates the decoders, which is why the margin is read from the McNemar box rather than off the bars; the whole-word references draw bare, being references the streaming architecture does not claim rather than arms of the paired comparison. ](../../output/figures/g2p_accuracy.png){#fig:g2p_accuracy width=85%}

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
221 tokens in 76 sequences
from 28 contributing documents. On every one of
them the case-transition
model and the context-free baseline make **identical** predictions: the paired
test finds 0 positions where only the model is right and
0 where only the baseline is. Model, baseline,
forward-backward smoothing, Viterbi and a majority-class constant all score
63.8%.

Every gold-morphology token in the vendored EWT split is a pronoun, so the gold
observation alphabet has size 1. A constant
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
8 roles at the cost of a labeling this project has
not validated. That cost has a shape, and the shape is measurable, because the
oracle assigns each label by paths applied in strict precedence:
morphological `feats[Case]` where the corpus marks it, then a deterministic
dependency-relation map, then the ADP token's own lemma. The final route labels
that token and never transfers the role to its head. The paths are
recomputed from the treebank purely to partition the evaluation and never reach
the model, which observes the UPOS tag and nothing else.

Partitioned that way ([@tbl:case_oracle_paths]), the derived margin of
7.81 points is not a property of the level. It is a
property of one stratum. A second bound sits beside that one: a
65-parameter lookup over the observed tags
scores 55.4208% against this level's
50.72%, so the margin is bounded both by where it lives
and by what a table already achieves without any latent state
([@sec:comparator_strength]).

The agreement itself is computed by a NumPy forward filter, and the live
Agent's `infer_states` / `update_empirical_prior` recursion is compared with
that filter over the entire held-out split rather than a prefix: across all
229 test sentences (1319 positions)
the two agree on the case-role posterior argmax with
0 disagreements, and the largest absolute
deviation between the two posteriors at any position is
1.25e-07. The recorder refuses to write if either
statistic moves, so the attribution of the reported agreement to the pymdp
Agent is exhaustive over the scored population while the headline path remains
the NumPy filter ([@sec:verification], [@fig:inference_evidence]).

| oracle path | tokens | baseline | model | margin | discordant | *p* |
|---|---:|---:|---:|---:|---:|---:|
| dependency relation | 704 | 36.65% | 52.84% | **16.19** | 192 | 3.688e-17 |
| morphology (control) | 296 | 64.53% | 65.54% | 1.01 | 35 | 0.7359 |
| adposition token | 319 | 36.68% | 32.29% | **-4.39** | 80 | 0.1456 |

: The derived gate partitioned by which oracle path assigned each label. The listed strata sum to 1319 tokens and their pooled margin reproduces 7.81 exactly, which is what establishes that the partition is of the scored population rather than of a similar one. The three *p*-values partition a hypothesis already in the confirmatory family, whose pooled test (*p* = 4.213e-09) scores every position listed here, so they are uncorrected, sit outside the declared significance family by recorded exclusion, and are printed to show that the aggregate is a mixture rather than to make a threshold-crossing claim; the morphology and adposition rows are unresolved controls, not results. {#tbl:case_oracle_paths}

The margin's primary uncertainty analysis uses the source document, not the
sentence or token, as the independent outer unit. The held-out derived arm has
229 sequences from
62 contributing documents; whole-document
resampling preserves every nested sequence and position together. Its
token-weighted point effect is 7.808946
percentage points, with a
0.95-coverage percentile cluster-bootstrap interval of
[4.889581, 11.180282]. The
largest document contributes 171
scored tokens, the estimated intracluster correlation is
0.028825, and the corresponding variance-inflation
factor is 1.4666. A whole-document
sign-flip sensitivity reaches the finite Monte Carlo floor
9.999e-05 under joint central symmetry of document-level
effects; the bootstrap interval is the primary statement. The familiar
tokenwise McNemar result (*p* = 4.213e-09, in the caption of [@tbl:case_oracle_paths]) is secondary and conditional on treating nested
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
result**. Its 35 discordant pairs sit far
below the minimum this project requires to resolve an equivalence
([@sec:multiplicity]). So its outcome is printed in [@tbl:case_oracle_paths]
but not read as a verdict, and it must never be read as *absent*. The
token-level adposition stratum's negative margin is likewise unresolved, at
80 pairs. What it does establish is that
the aggregate is pooling strata whose signs disagree.

One objection deserves answering rather than deflecting: the morphology stratum
starts from a much higher baseline, so it has less room to improve, and a small
margin there could be a ceiling effect rather than a provenance effect. It is
not, or not only. Expressed as the percentage of available headroom each stratum
captures, the dependency-relation path takes
25.56% and the morphology path
2.85%. The gap narrows under that
normalization and does not close.

There is a stronger statement available, and it is structural rather than
statistical. The observation is the UPOS tag, so a margin requires the tag to
carry information about the label. Within each stratum it either does or it
cannot ([@tbl:case_path_information]):

| oracle path | UPOS alphabet | $H(\text{case})$ | $I(\text{case};\text{UPOS})$ |
|---|---:|---:|---:|
| dependency relation | 9 | 1.443603 | 0.092571 |
| morphology | 1 | 0.892573 | 0 |
| adposition token | 1 | 1.441481 | 0 |

: Observation content within each stratum, in nats. The mutual information is 0 on the morphology path and 0 on the token-level adposition path, not merely small, and the reason is in the first column: each path *defines* the tag it selects on, so the observation alphabet inside the stratum is a single symbol and the emission cannot separate distinct states. {#tbl:case_path_information}

This is the same degeneracy [@sec:gold_population] reports for the gold
population, where the alphabet is likewise
1 and every predictor scores alike. It was not
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
inspection (`docs/decisions.md` entry 77). That
specification was not independently timestamped in an immutable prior artifact,
so it is not a preregistration. **The result is exploratory**: the project's
confirmatory family was frozen before this arm existed, neither of its p-values
joins that family, and nothing here rescues or retires an English number.

The corpus is the Universal Dependencies conversion of the Prague Dependency
Treebank -- Consolidated (UD Czech-PDTC). Scored tokens are the case-inflecting nominal
categories (6 UPOS tags) carrying UD
`feats[Case]`; every label is morphology
(100% of scored tokens, enforced by a
recorder refusal — the coverage English gold never had), spanning
7 of the 8 roles (Czech has no
ablative). ADP is excluded although Czech annotates `Case=` on prepositions:
there the feature records *government* — the case of the complement, not
inflection of the token — which is the analogue of the English token-level adposition path
whose margin reversed in [@tbl:case_oracle_paths]. The vendored artifact
retains UPOS and `Case` per token plus each retained sequence's authenticated
`# newdoc id` index; the fields the English trap turns on never enter the file.
The train-ft shard
(11,267 tokens) fits the same builders the English arm
uses — the modules are shared, not forked — and the dev shard
(169,331 tokens in
20,349 retained sequences nested within
1005 contributing documents) is scored held-out,
against the same
context-free `argmax P(case | UPOS)` comparator fitted on the same training
tokens. Because the labels are ground truth, the endpoint is *accuracy* — the
word every English number in this section is named to avoid.

The prospectively specified question resolves in the surviving direction. The
sequence model reaches 39.21% against the comparator's
32.36%. The **primary uncertainty analysis** preserves the
token-weighted effect but treats each authenticated source document as one
independent outer cluster: 6.847535
percentage points over
169,331 tokens in
1005 documents. Its
95% percentile cluster-bootstrap interval
is [6.430598, 7.260224]. A
whole-document sign-flip sensitivity analysis with
10,000 draws and
deterministic seed 20,260,811 reaches its declared Monte
Carlo resolution floor at 9.999e-05; it is reported as
$p \leq 9.999e-05$ conditional on joint central symmetry
of whole-document differences, an assumption stronger than marginally fair
token discordances or zero expected document sum. The interval
therefore bounds the positive effect away from zero at the document sampling
unit; the finite randomization statement is a resolution bound, not a claim
that a probability is zero.

The token-level analysis is secondary and conditional on treating positions as
independent despite their nesting within sequences and documents. It reports a
6.85-point margin over
169,331 positions, with McNemar
$b=23,692$ against $c=12,097$
(35,789 discordant pairs, against the
420-pair resolvability minimum). Its exact
two-sided p-value underflows a double and is therefore reported through the
finite log-tail, $\log_{10}p=-832.4$. This much
smaller conditional tail does not replace the cluster-primary interval or
randomization result.

The displayed comparisons locate the margin. Below: the majority class scores
29.05%, so the comparator is doing real work before the
model improves on it. Above: a context-free predictor fitted on the *test labels
themselves* — the best any member of the comparator class can do — reaches
35.97%, and the model sits above that too. The margin is
therefore not a better lookup; no lookup on this observation reaches it, and
what remains is the case-to-case transition. That is the mechanism this level
always claimed, shown against labels no heuristic generated. The mis-wiring
controls transfer: a cyclically deranged emission or transition falls below the
context-free baseline (`tests/test_czech_case.py`).

The honest boundary of the finding is the observation's poverty, and it is
measured rather than lamented. The mutual information is
$I(\text{case};\text{UPOS})$ = 0.0748 nats,
against a label entropy of 1.6145. So
39.21% is nowhere near what a model observing word forms
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

Over 1072 held-out positions, the type-free bar scores
4.3741; the mixture scores
4.3858 — *worse*. With
99 interrogative training sentences, that pattern
is consistent with limited-sample estimation variance, but this single fitted
split does not distinguish variance from smoothing, misspecification, or
distribution shift. The decisive row is the **privileged type-label reference**: scoring every
sentence under the fitted chain selected by its privileged reference label
scores 4.3819, a signed perplexity reduction of
-0.178%; the negative sign means the fitted privileged
reference is worse than the type-free bar. This is a
receiver-sensitivity reference, not a hard information bound: under
misspecification a mixture can assign more probability than the label-selected
component. On the paired log-evidence scale, the reference-minus-type-free point
estimate is -0.010698 nats per sentence, with a
99% percentile interval of
[-0.034233, 0.008941]
over 10,000 whole-document resamples of
53 contributing test documents at
fixed seed 20,260,804;
that interval spans zero. The companion
ratio of resampled gain to resampled positions is
-0.001776 nats per position, interval
[-0.006209, 0.001417],
which spans zero. The latter point estimate
is exactly the log-ratio corresponding to the position-weighted perplexity
comparison above. Both arms use the same sampled document indices; all
sentences and positions within a selected document move together. The gain is therefore
unresolved on both reported scales at this order and in this sample. MAP classification reaches
88.2% accuracy on the held-out sentences, below the
88.76% majority baseline, and a length-only
classifier merely reproduces that baseline
(88.76%). Recall on the minority class says it more
sharply: of the held-out interrogatives, MAP classification recovers
0%. The document-primary paired accuracy
difference is -0.5618 percentage points, with a
99% whole-document-bootstrap
interval of [-2.7779,
0] over
53 documents, computed from
10,000 whole-document resamples
at fixed seed 20,260,804; the largest of
those clusters carries 21
sentences. The interval is read at the same declared confidence level as the
paired-gain intervals above — one declared level that both intervals state by
construction — and its upper endpoint is zero, so it supplies no positive
classification claim: p = 1 against the
recorded display floor 9.999e-05. The historical
sentencewise McNemar contrast remains a conditional secondary analysis. The mixture
posterior's point area under the curve (AUC) of the receiver operating
characteristic (ROC) is 0.6483, with a
99% interval of
[0.5026, 0.7724] bootstrapped over test
documents (53 contributing clusters, with all
sentences moving together). It excludes chance on that rank scale, but it does not select a
decision threshold or establish calibrated probabilities.

A rank statistic of 0.6483 sitting beside
0% minority recall looks contradictory only if AUC is read as a claim about
decisions or about probabilities, and it is neither. Average precision and
Brier score, measured on the same posteriors, say what AUC leaves out; each is
reported against the reference that makes them interpretable. **Average
precision** is 0.1552 against a positive-class prevalence
of 11.24% — its chance level is that prevalence
rather than one half. **Brier score** [@brier1950], a strictly proper rule
[@gneiting2007strictly], is 0.1018 against
0.0998 for the constant predictor that always emits the
11.7% training prevalence: a skill of
-0.0201. The negative skill means the fitted posterior is worse
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
An AUC of 0.6483 is ranking information, and information a threshold
could in principle exploit — a cost-asymmetric rule trading precision for
minority recall is not ruled out by anything measured here. What is reported is
a non-passing result for the decision rule this project actually applies, not
for every rule that could be applied. The perplexity ladder and paired
log-evidence interval answer related but distinct questions and agree only on
the weaker conclusion: no gain is resolved on this split.

What makes this a finding rather than a shrug is the **detectability
control**. On a synthetic corpus whose two types genuinely differ, the
identical harness classifies at 99.5% against
a 50% majority and drops mixture perplexity from
4.0661 to
3.4556. That is a *design* argument and not an
observed-power one: the control plants a known effect and shows the instrument
recovers it, whereas computing power from the very statistic that failed to
reject is a transformation of the p-value and says nothing new
[@hoenig2001abuse]. Read that way, the control rules out a generically dead
instrument; it does not turn the unresolved EWT result into a property of
English. The control's own operating
point differs from the measurement's — it is balanced and larger — so it bounds
the instrument's capability, not the sensitivity available at
99 minority training sentences and
20 test ones. At the operating point that produced
the failure, uncertainty is reported on the axes that were actually intervalled:
the AUC interval above, the paired oracle-gain intervals reported here, and the
exact conditional bound on the default MAP classification contrast in
[@tbl:derived_bounds]. The latter produced too few discordant pairs to support
equivalence, and the oracle-gain intervals were not evaluated against any
smallest effect of interest. The machinery ships — spec-built and bit-identical, with the closed form
agreeing with a manual walk over the Agent's stored arrays to
1.0658141036401503e-14 — but the shipped mixture does not clear the
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
over 1072 held-out positions — the type-free bar, the
mixture that sits above it, and the fitted privileged type-label reference that
also sits above it; the signed point perplexity reduction is
-0.178%. Its companion
position-weighted paired gain is -0.001776 nats
per position, 99% interval
[-0.006209, 0.001417],
which spans zero under a
53-document bootstrap and is drawn as
an interval against its zero line in the inset as well as quoted in the panel
title. Right:
the detectability control — the same harness on planted synthetic types,
where the mixture wins decisively and classification reaches
99.5%. A left-panel mixture or privileged
reference bar *below* the type-free bar, or a right panel without the decisive gap
would each contradict the recorded finding; so would either paired interval
ceasing to span zero. The data function refuses to draw in every such case.](../../output/figures/clause_null.png){#fig:clause_null width=97%}


![What each upper-level score is, and what kind of evidence backs it. Left: the Stage 4 case-role
agreement of 50.72% over 1319 tokens comes from the
numpy forward filter; live `infer_states` + `update_empirical_prior` agree with it at every one of
the 1319 held-out positions (0 argmax
disagreements, largest posterior deviation 1.25e-07), so the
attribution is exhaustive over the scored population while the headline path stays NumPy. Middle:
the closed-form mixture path of the retained case-only Stage 5 ablation, drawn at its type-free bar of 4.3741; the
live pymdp agent agrees with the NumPy joint filter on the type-posterior argmax at
0 disagreements over all 178
held-out sentences and on the full joint posterior within
1.67e-07 — posterior agreement on the whole held-out set, not score
equivalence for the headline mixture estimand. Right: the
registered observed-position score is a conditional table, not inference; the separately recorded
live-filter score (0.3687 nats, 0.0043 nats
better than the unigram bar over 187 sentences) is the only L6 quantity drawn
through pymdp's inference path. Any of the following would falsify this panel: a recorded L4 or L5
disagreement count other than the stated zero, an L4 comparison that stops short of the scored
population, a live-filter score inconsistent with its recorded
improvement, the two L6 estimand deltas collapsing onto one another, or the Stage 4 agreement
leaving the majority share its annotation requires.](../../output/figures/inference_evidence.png){#fig:inference_evidence width=95%}

### Canonical widened model: development-only gain over the type-free chain {#sec:clause_widened}

This is the canonical clause level. On 205 held-out
sentences from
64 source documents, spanning
1679 positions, the widened mixture beats the type-free
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
| type-free bar refitted on the widened stream | 4.7708 |
| mixture (the widened clause model) | 4.7449 |
| privileged type reference (true type given) | 4.6564 |
| privileged-label gain over the bar | 2.396% |
| paired gain, nats per sentence | 0.198647 |
| its 99% document interval | [0.024598, 0.358617] |
| paired gain, nats per position | 0.024254 |
| its interval | [0.003053, 0.046058] |
| entropy of the type variable | 0.361478 nats |
| share of that entropy realized | 54.95% |
| MAP classification vs majority | 93.66% vs 88.29% |
| interrogative recall | 58.33% |
| rank signal (AUC) | 0.8908 |
| its 99% document interval | [0.7341, 0.9767] |
| comparator: sentence length only | 0.5828 |
| comparator: first symbol is an auxiliary | 0.7543 |
| comparator: case-only arm, same sentences | 0.6201 |
| widened arm on those same sentences | 0.8908 |

: The widened-observation clause arm on the document-disjoint partition.
Descriptive; the *p*-values below sit outside the frozen family, uncorrected.
{#tbl:stage5w}

The paired privileged-label gain is 0.198647
nats per sentence and its 99%
whole-document interval excludes zero on both reported scales. This is the
quantity the retained case-only ablation leaves unresolved, and on the widened
stream its direction is resolved. It realizes
54.95% of the 0.361478
nats the binary type variable makes available. The widening does not raise
that ceiling — the entropy is a property of the prior, not of the receiver —
so what moved is the share realized within it.

The rank signal clears every registered comparator. It exceeds the
length-only rule at 0.5828, whose classifier never
beats the base rate at all, scoring 88.29% against
the 88.29% majority. It exceeds the single
first-symbol-is-an-auxiliary indicator at
0.7543, so the effect is not one free parameter
wearing a sequence model's clothes. And on the
205 sentences where both arms are defined, the
widened arm scores 0.8908 against the case-only arm's
0.6201: the two added symbols, not the refit, carry
the difference.

Classification reaches 93.66% against the
88.29% majority, recovering
58.33% of held-out interrogatives where the
case-only ablation recovers none. Its discordant counts are 14
and 3 at *p* = 0.012725830078125, uncorrected and
outside the frozen family. The limit belongs beside that number: documents are
the outer sampling unit, and the document-clustered margin of
5.3659 points has interval
[0,
10.7695] across
55 clusters, whose lower bound
touches zero. The classification contrast is therefore *not* resolved away
from zero at document level, even though the paired gain and the rank interval
both are. On the threshold-free scores the posteriors carry real information:
average precision 0.7204 against the
11.71% prevalence, and Brier
0.0534 against the training-prevalence reference
0.1034, a skill score of 0.483.

The controls that stand behind those numbers each have the authority to stop
the recording. The closed-form mixture and oracle filtering agree to
7.105427357601002e-15, and that agreement is not vacuous: a
transition carrying mass across the type blocks drives the same comparison
apart by 4.348005. Exchanging the two types' fitted
chains costs the classifier its margin, dropping it to
57.56%, so the type assignment is
load-bearing. Finally, the identical harness on a synthetic corpus whose types
genuinely differ classifies at 99.5% against
its measured 50% majority, with mixture
4.6513 beating type-free
5.5835. That control establishes a live
instrument for a large planted effect. It supplies no minimum detectable
effect, though, and it is not evidence about small ones.

The population is stated with the result. The arm fits on
972 training sentences and scores 205 held-out
ones, drawn from 318 source documents
(254 training and 64 held out) and
carrying 114 and 24
interrogatives respectively. 130 exclamative
sentences and 571 unpunctuated fragments carry no
label, and 123 fall under the usability minimum.

## Level 6: development and held-back document-position results

### Development estimate with a document-clustered interval spanning zero

The registered point estimate is positive. Its primary analysis spans zero on the
development population and excludes zero on the held-back one.

The train split shows interrogatives concentrating in document openings: in this
corpus, English web text opens with questions more often than it continues with
them. The association is stated over EWT and no further. The preregistered
direction was lower held-out mean log loss under the tercile-conditional table.
Its point estimate is positive. Over 187 sentences in
27 held-out documents, the tercile-conditional model
scores 0.3678 nats per sentence against the type-unigram
bar's 0.3729 — an improvement of
0.0051 nats, positive on
116 sentences against
71 (sign test p = 0.001232362495942883). This
registered observed-position score indexes the fitted table by the known
sentence position; it is not live pymdp filtering over the latent tercile. The
audit classifies the gate BRACKETED_REFERENCE: the bar fails it definitionally, and the
fitted-on-test reference clears it.

The document is the sampling unit. In the primary document-clustered analysis on
the development population, the log-loss interval
[-0.009563, 0.024959] spans
zero and the p-value is 0.5469 under the
assumption-conditional whole-document sign-flip sensitivity analysis
(finite-sample exact only if the joint document-difference distribution is
centrally symmetric). That result on that population is imprecise, and it is
what the level's status rested on until the estimand was carried to a population
the models had never seen. The sentence sign count is secondary and
non-diagnostic: reversing the phase columns makes mean log loss worse yet still
yields a nominally favorable sentence sign test. A statistic that a worse control
passes cannot confirm the registered mean-log-loss prediction, which is why the
held-back sign test of decision 111 was not treated as settling the question
below.

### Held-back document-clustered gain at uncorrected coverage {#sec:stage6_heldback_clustered}

Decision 115 registered the primary estimand on the held-back UD EWT test split
before computing it, fixing the population, the bootstrap, the coverage, the seed
and the prediction in a commit that precedes the outcome. The models are the
frozen development fits, asserted equal array for array; nothing is refitted.
Its status is descriptive, not confirmatory: the population had already been
spent by the run of decision 111, so this estimand was registered by
someone who knew that population's other results ([@sec:stage6_heldback_method]).

Over 1090 sentences in
150 held-back documents the mean
improvement is 0.008288 nats, and
the document-clustered interval
[0.002562,
0.014556] excludes zero at
95 percent coverage, the coverage
the development analysis above uses. The improvement favors the tercile model in
98 documents against
52, and the sign-flip p-value
is 0.0045. Together with the held-back sign test of
decision 111, which crosses the frozen family threshold in the registered
direction, this is what the gate audit reads to compute Level 6's finding
status; the status is a function of those recorded flags rather than a sentence
anyone typed.

Three qualifications travel with that interval and none of them is optional.

First, the development-population interval still spans zero. The two populations
disagree, and the honest description is that the estimand resolved on the larger
held-back sample and did not on the smaller development one, not that the earlier
result was wrong.

Second, the same held-back bootstrap read at the family-corrected coverage of
99.9375 percent gives
[-0.001398,
0.019386], which spans zero. This arm
is a registered singleton and not a member of the frozen significance family, so
the corrected level is a declared sensitivity rather than its decision
threshold. A reader who prefers the conservative reading should take the effect
as unresolved at that level, and the recorded bounds are there to make that
reading available rather than to be argued away.

Third, the population's resolution is coarser than the effect the development
population produced. The smallest constant per-sentence shift this arm can
resolve is 0.005726 nats,
measured by planting shifts on the centered differences and verified from both
sides. The development population's own clustered effect,
0.0051 nats, sits below
that, and planting it here does not resolve
([-0.000626,
0.011369]). The held-back
effect is detected because it is larger than the development one, not because
this population is more sensitive. A null here would therefore have been
uninterpretable, and the detection should not be read as evidence that the
effect size generalizes.

The controls behave. Centering the differences to exactly zero mean gives
[-0.005726,
0.006269], which does not
detect, so no plant above it is detecting the centering. Reversing the sign of
80 whole documents
under a recorded seed gives
[-0.011465,
0.000634], which also does not
detect, so the procedure does not find an effect in differences whose aggregate
direction has been destroyed.

A separate descriptive run does use the live Agent's filtering recursion. It
reports 0.3687 nats, an improvement of
0.0043 nats, with sentence signs
105 to 82
(*p* = 0.10741108743413678). This is evidence that the built Agent can
run and produce a related positive descriptive score; it is not the registered
observed-position estimand and does not replace the registered
document-clustered primary analysis. As a post-registered execution diagnostic,
its p-value is outside the frozen significance family, is uncorrected, and
carries no threshold-crossing inference. The attribution gap between the two
estimands is measured rather than merely named: on the same bounded slice of
178 held-out sentences, the live Agent's
recursion scores 0.3798 nats against the registered
observed-position score's 0.3796 — a divergence of
0.0002 nats — and the two estimands
agree on the per-sentence sign of the improvement on a
0.7978 share of sentences. They index different
conditionals by construction (a `B`-carried latent-phase belief versus the
known position), so this is characterization, not equivalence.

Under the repository-wide multiplicity family of 16
declared tests ([@sec:multiplicity]), the secondary sentence-sign p-value is nominally positive at
$\alpha =$ 0.01 and does not cross the fixed Bonferroni
per-test threshold of 0.000625; the recorded flag
`stat_stage6_familywise_confirmed` is false.
Against its Holm step-down threshold of 0.00111111 at the
same familywise level the recorded verdict `stat_holm_stage6_rejected` is
false: the test crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold, and a
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
0.1179 by
+0.0521, the middle tercile by
-0.0189. And the tercile-permuted scramble — the
entropy-preserving control — lands elsewhere
(-0.01 at the opening), so the effect depends on
which tercile says what. This intervention is not passed to L3 or L1g in the
same run.

The retained case-only clause ablation supplies a privileged-label
**sensitivity reference**, and it is the arm this intervention was measured
against. The intervention
demonstrably moves the sentence-type level's *type* prior, while selecting the
fitted case-only chain by its reference type changes case-role perplexity by
-0.178% (a signed reduction; here negative). That small,
unresolved value motivates expecting little
delivered effect, but it is not a hard bound under model misspecification.
Novelty and receiver sensitivity remain useful explanatory dimensions. The working hypothesis of
[@sec:hierarchy] therefore predicts a channel pays only when it is novel to
the receiver **and** consequential for the receiver's likelihood, and the
document-position→sentence-type intervention is evidence about that distinction,
not a proof of it. [@fig:narrative_channel] shows the intervention's measured effect.

![Stage 6's two preregistered experiments. Left: held-out observed-position type log-loss,
the tercile-conditional model — drawn as `phase-conditional` — against the
type-unigram bar over 187
sentences; the improvement of 0.0051 nats
(secondary sentence sign p = 0.001232362495942883) has a document-clustered
interval of [-0.009563,
0.024959], so it is non-confirmatory. Right: the independent prior intervention — the sentence-type
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
| Baseline — ancestral sampling from $\mathbf{B}$ | 13.33% | 325 | 0.9092 |
| $\gamma$ = 0.1 — goal too weak | 54.75% | — | 0.6276 |
| **$\gamma$ = 0.55 — shipped** | **100%** | **180.7** | **0.3449** |
| $\gamma$ = 16 — collapsed | 100% | 1 | 0.9071 |

: Every distance is computed exactly from a finite-lexicon path distribution plus an explicit out-of-lexicon remainder: Brown-transition paths for the ancestral baseline and policy-posterior paths for each of the 21 precision settings. Both distributions condition out an immediate empty attempt, matching the non-empty token unit scored by the sampled generator. Membership remains a sampled diagnostic and distinct-word count the sampled anti-collapse floor, over 400 words per run and 3 seeds, because both are properties of realized runs rather than of the induced distribution. The optimum is **interior**: both endpoints generate badly, for opposite reasons. {#tbl:generation_sweep}

[@tbl:generation_sweep] reports the sweep, and the shape is the result. As
$\gamma \to 0$ the policy posterior becomes the habit and the model *is* the
baseline. At high precision, the policy follows the locally most preferred trie
branch at each prefix, with the habit breaking ties. This greedy conditional
path is not generally the globally most probable word; on the shipped Brown
artifact its distinct-word count nevertheless collapses to
1. Neither endpoint is
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
all 30 actions is
0 nats. Making the transition uniform
while keeping the identity likelihood raises it to
3.401197 nats, which is exactly
$\log 30$ — the analytic ceiling, since a uniform
predictive resolved by a perfect observation yields precisely the prior entropy.
Flattening the likelihood as well returns it to
0. The second control
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
the 2000-type lexicon, action the emitted letter, $\mathbf{A}$
the prefix-conditioned next-symbol map, belief the trie-node posterior, and
$\mathbf{B}$ the identity for *every* action, because a word does not become a
different word part-way through being spelled — the invariant that pins the
word level ([@sec:word_level]). Emitting a letter changes what is observed,
never which word is being spelled. On this construction the information gain is
measured and it is *not* zero: 2.716363 nats at
the empty prefix, and positive at the most-populated ambiguous trie node of
every probed depth — *t*,
*th*,
*the*,
*with*, selected by mass, not by hand. What
is zero is the term's per-action *spread*: across all
30 actions the information gain, the pragmatic
value, and $G$ each vary by exactly 0
at every probed prefix. Transitions that do not depend on the action make the
predictive state — and therefore the predictive observation — identical across
actions, so both terms of [@eq:efe] are constants over the policy space, and
the softmax in [@eq:policy_posterior] cancels constants. The induced policy
posterior equals $\sigma(\ln \mathbf{E})$, the pure habit, to a measured
maximum deviation of 1.11e-16. The
recorder (`scripts/record_generation_word_state_claims.py`, prospectively
specified with its refusal conditions in the development session as decision
78) refuses to write if any spread is
nonzero, and its action-dependent control — the same node, the same belief,
transitions that differ by action — spreads $G$ by
0.786256 nats, so the zeros were
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
1.0832 points at a 50% budget.

The generation and budgeted-attention arms together are the decomposition, each term measured alone. Where the
information-gain term vanishes and preference carries the policy, generation
works: every sampled word is in the lexicon and the induced distribution sits
0.3449 in total variation from the target. Where the
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
$\gamma$ = 1, scoring
0.4828; exactly, the optimum is
$\gamma$ = 0.55 at 0.3449. The sampled and
exact values differ by up to 0.1465 on the shared
grid, and the direction of that gap is not random. A run of
400 words over 3 seeds is
1200 tokens against a 2000-type target,
so the empirical distribution is supported on at most
1200 types and its distance to the truth is inflated by
construction — before any question of which $\gamma$ is best. Pooling the seeds
into one statistic with no spread reported left nothing to signal it.

The finding survives. The optimum is interior, both endpoints fail for opposite
reasons, and the exact curve is the same shape. What did not survive is the
operating point, and the ancestral comparator's old pooled distance is now
retained only as an explicitly sampled diagnostic: its pooled estimate is
0.9224, whereas the compatible exact comparator is
0.9092. A sampled estimate of a distribution's distance to a
many-type target was never going to resolve the small gaps that separate
adjacent grid points near the minimum.

This gives the generation gate a property none of the recognition gates has: it
fails in **two opposite directions**, so it cannot be passed by turning a dial to
an extreme. Every gate in [@sec:gate_design] can be characterized by the single
question of whether a threshold sits between a baseline and a ceiling. This one
additionally requires that a free parameter be neither too small nor too large,
and the U-shape is what demonstrates the requirement rather than asserting it. [@fig:generation_samples] shows the sampled strings at each setting.

![Distance to the true word distribution against policy precision. Both extremes generate badly for opposite reasons: at low precision the policy posterior becomes the habit and the model *is* the sampling baseline, while on the shipped artifact high precision follows a greedy conditional-prefix path whose distinct-word count collapses to 1. That path need not be the globally most probable word in another lexicon. The optimum is interior, which is what gives this measured gate a failure mode in two directions. A boundary optimum, a monotonic curve, or a high-precision distribution that retained the full support would falsify this panel.](../../output/figures/generation_sweep.png){#fig:generation_sweep width=90%}

### What the distance metric does not catch, and how a degenerate generator beats it

The degenerate generator scores 0.9071, which *beats* the
baseline's 0.9092 while holding its distinct-word count at
1 indefinitely. Distance
alone would admit it. The second gate conjunct requires more than
20% of the ancestral run's sampled diversity,
or more than 65 distinct words under the declared
run protocol; only that diversity floor rejects the collapsed endpoint.

That is recorded here for the same reason [@tbl:word_miswirings] records the
three word-level mis-wirings that clear their
bar: a guard's blind spots are part
of its specification, and a project that reports only what its checks catch is
reporting half of what it knows.

![What the three regimes actually write. Real output regenerated at draw time
from a fixed seed through the shipped generators — never a mock-up — one
column per operating point of [@fig:generation_sweep], each word colored by
lexicon membership, with the column's membership rate and distinct-word count
beneath. The habit-only baseline drifts out of the lexicon; the shipped
operating point stays inside it while remaining diverse; the degenerate
endpoint has a distinct-word count of 1 with a
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
| 0.0 — the anchor | 70.801% | — | identical by construction |
| 0.1 | 72.704% | 1.903 | 69.25% |
| 0.2 | 73.319% | 2.518 | 67.13% |
| 0.3 | 73.648% | 2.847 | 66.245% |
| 0.5 | 73.951% | 3.15 | 65.035% |
| **0.7 — best** | **74.134%** | **+3.333** | 62.55% |
| 0.9 | 74.059% | 3.258 | 58.309% |

: The lexical channel over 42,820 held-out positions, with a real prefix available at 99.9% of them. McNemar at the optimum: 2960 positions where only the coupled decoder is right against 1533 where only the anchor is, $p =$ 3.756e-102; the coupled decoder's share of the 4493 discordant pairs is 0.658803, exact 95% interval [0.644728, 0.67267] ([@sec:multiplicity]). {#tbl:lexical_channel}

The first predicted signature appeared in [@tbl:lexical_channel]. The channel
helps by 3.333 points — for scale, the entire
letter-to-phone margin over its own context-free baseline is
0.546 points, so the channel is worth several times the
level's own headline result. The second did not survive the correction of
decision 110: the sample argmax is interior, at weight
0.7, but the highest tested weight trails it by only
0.075 points on
302/270 discordant
positions (*p* = 0.1949), the effect helps at every
tested weight, and the wider grid of [@tbl:channel_factorial] places the same
message's best cell at full weight. The sweep therefore records an ordering,
not an optimum: it does not distinguish an interior peak from a plateau, and
the reversal the defective lookup had shown at weight 0.9
was an artifact of matching the wrong context.

![The composition result, with the registered lexical prediction drawn on the evidence and the earlier coupling sweep shown as its motivating contrast. Both sweeps are measured on the same 42,820 held-out receiver positions, and both are linear mixtures anchored at the uncoupled decode at weight zero. They do not hold insertion site fixed: coupling folds into the observation likelihood over phones, whereas the lexical message mixes into the phone predictive. The lexical message rises to a plateau whose sample argmax is interior and does not reverse at the highest tested weight; the letter predictive declines monotonically. Every lexical point now carries an exact conditional paired interval, derived at render time from that weight's own recorded discordant counts over the shared denominator, so the plateau's overlapping intervals show on the figure what this caption asserts about the argmax rather than asking the reader to take it on the point estimates alone. The bands show the registered lexical *interior optimum* class, which the corrected sweep's argmax satisfies without distinguishing it from a plateau, and the already-observed coupling class. The generator recomputes both shape verdicts from recorded sweep claims and refuses to draw a contradictory band. The coupling curve's intermediate segment is schematic between recorded endpoints; its full-mixing collapse is 16.922% against the uncoupled 70.801%. A harmful series extending past its last recorded anchor, an endpoint unequal to a recorded measurement, prose calling both curves prospective, or a drawn point whose estimate does not reproduce from its own counts over the recorded denominator would falsify this panel's account. The shape is evidence for a working hypothesis, not a universal law.](../../output/figures/channel_comparison.png){#fig:channel_comparison width=92%}

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
selects the chain — -0.178% ([@sec:clause_level]). That
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
into it. Five of the 7 registered interventions are separately
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
measured (decision 27, corrected after the document-partition repair in decision 88): on the declared axis — the held-out perplexity of the
word prior, the quantity the pre-registration named as where harm would show —
the clause-typed word message produced a small *improvement* at every tested
positive weight (anchor 3.268023 → best tested
3.263007 at the high grid endpoint alpha =
0.9, delta -0.005016). The sweep
therefore does not locate an optimum. The
token-direction tally agrees: 51,364 tokens improve
against 22,649 made less probable, while the
scrambled control is far worse (4.562879). No monotone harm
reproduced, so the predicted harm was **not** confirmed.

The population boundary is source-document disjoint: the typed statistic is
estimated on 448 Brown documents and evaluated on
52 others, with
0 overlap and every Brown category represented in
both arms. The type-free and swapped-type controls isolate the role of the
privileged type label. Replacing
both conditional rows by the type-free bridge marginal gives perplexity
3.267863 at the selected weight, while swapping the
declarative and interrogative rows gives
3.376872; the real typed message is
3.263007. The vocabulary-axis reversal remains a wiring control,
not the evidence that sentence type itself matters.

The outcome is evidence against the biconditional and is what the
potential-value hypothesis expects to be possible. A coarse, type-conditioned
message is not cleanly redundant against a prior that is a marginal over words:
the receiver's prior carries no type structure at all, so the message can shift
mass the marginal had mis-assigned without telling the receiver anything it
could not have read off the observed word. Agreement between aggregate log-loss
and the majority token direction survives whole-document uncertainty. The mean
gain is 0.00153609 nats per emission, with
a 95% whole-document interval of
[0.00088152,
0.00233064];
43 documents favor the typed
message and 9 favor the anchor.
The assumption-conditional document sign-flip sensitivity is reported only as
*p* ≤ 9.999e-05, its Monte Carlo resolution floor.
This resolves the descriptive sign on the reused population; it does not make
the correction confirmatory or identify whether the gain is conditional
information, mismatch correction, or regularization under the chosen mixing
rule. What neither metric supports is a large effect. The best point differs
only narrowly from the anchor (3.268023 to
3.263007), so the measured improvement
sits on a numerically narrow curve. That scale is reported by the injected
endpoints; no equivalence or near-null classification is inferred from it.

### Case-word influence sweep on the development population {#sec:case_word_sweep}

The case-word bridge's registered influence sweep
(`docs/decisions.md` entry 121; estimand and controls
in [@sec:hierarchy]) is the second intervention on the L3 word prior that was
specified before it ran, and the first in which the sender is the case
level's own posterior rather than a privileged reference. The receiver side
is scored over 3001 tokens of
64 EWT-dev test documents —
14,449 word-level emissions — against the
zero-weight anchor, whose word prior has held-out perplexity
3.439523 and which the recorder reproduces
bit-identically from the untouched receiver prior before any positive weight
is scored.

The mean per-token log-gain is positive across the tested weights and
peaks at an interior weight: the recorded best weight is alpha = 0.3,
at 0.171460512 nats per scored token against the
anchor. The token-direction tally is not a uniform win:
1015 tokens improve against the anchor,
1425 are made less probable, and
561 are exactly unchanged — the last the
no-message tokens the inlet never touches. Both controls behave as
registered: the scrambled-word negative control sits at
-0.229713297 nats per token at the
best weight, and the declared positive control leads the scrambled control
by 0.492584079 nats, above the
pre-registered margin of 0.177528463
nats and recorded as passing at every positive weight.

The whole-document clustered analysis expands each token's difference to its
word-level emissions (14,449 across the
64 source documents), resamples whole
documents and sign-flips them: 10,000 resamples at seed
20,260,811. The clustered per-emission effect at the
best weight is 0.035611668 nats per
emission, with a 95% whole-document
interval of [0.026024861,
0.045182061];
56 documents
favor the message and
8 favor the
anchor ([@fig:case_word_sweep]). The interval excludes zero on the
development population, which is a descriptive sign resolution, not a
confirmatory one, and the two units are not interchangeable: the clustered
estimand prices the bridge per word-level emission, the mean per scored
token.

![The case-word bridge's influence sweep across the shipped mixing-weight grid, on the EWT-dev development population. The solid series is the recorded mean per-token log-gain of the mixed L3 word prior against the bit-identical zero-weight anchor; the bracketed series is the whole-document clustered per-emission effect with its 95% document-clustered interval at each weight; the dotted series is the scrambled-word negative control. The vertical rule marks the recorded best weight. Development evidence under a fresh registration, never confirmatory; the scored population is 3001 tokens in 64 documents, expanding to 14,449 clustered emissions. The caption is falsified by a plotted value disagreeing with its recorded token, by an interval drawn around the per-token mean rather than the per-emission clustered effect, by the anchor point drawn away from exactly zero on any series, or by the recorded best weight failing to lie on the drawn grid.](../../output/figures/case_word_sweep.png){#fig:case_word_sweep width=75%}

The sweep's scope warnings are the registration's own and travel with the
result. The scored documents are the EWT dev population already scored for
Stage 4 case assignment: development evidence under a fresh
registration, never confirmatory, never "untouched", and the shared corpus's
test file remains spent (decision 111).
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
0 over 1323
positions, with identical perplexity on the passage the counts were
accumulated over (8.163892 against 8.163892;
in-sample, so not comparable to the shipped agent's held-out score in
[@tbl:letter_comparators]) as the corollary. The agreement is evidence only because the controls show where
it fails: a Dirichlet prior mismatched to the comparator's pseudocount breaks
it by 0.15343913435935974, re-matching restores it to
0, and a learning rate other than
1 breaks
the conjugate identity outright. `lr_pB` is therefore not a tuning knob here:
any value except one makes the learned matrix an estimator of nothing the
ledger describes. [@fig:learning_agreement] plots the agreement and its
mismatched-prior control.

![Every transition cell, learned against counted, log–log. Left: matched
priors ($k$ = 1) over 1323
positions of the Stage 1 held-out text — a single point off the identity line
beyond a maximum absolute deviation of 0
would falsify the
conjugate-counting claim of decisions 22–23. Right: the control that proves
the comparison can fail — with the Dirichlet initialized at
$k$ = 2 while the comparator counts at
$k$ = 1, the cells leave the line, most strongly where
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
drift 1.110006, measured not asserted), the online agent that
keeps learning across the drift scores 10.502499 on
post-drift held-out text against the frozen control's
11.611376, a gap of
1.108877 perplexity. The frozen control is the arm
the prediction requires to lose, and its losing is informative because the
alternative is diagnostic rather than merely disappointing: a frozen model
fitted to the first genre only that did *not* lose on post-drift text would
mean the drift was not real or the online updates were not happening, and the
comparison would be measuring neither. The re-fit control
(10.306888) bounds the same comparison from the other side by
scoring at least as well as the online agent, and the pre-drift identity
confirms the online path is still counting at
`lr = 1` (P3). The prediction was committed before its
measurement (decision 28), but the entry is not in
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

In information units the letter level reaches 2.4783 nats, which by
[@eq:bits_per_char] is 3.5755 bits per character.

| Quantity | Value (nats) |
|---|---|
| Uniform over the letter alphabet | $\log$ 30 |
| Bigram agent | 2.4783 |
| Order-1 conditional-entropy floor | 2.375 |
| Gain realized over uniform | 0.9229 |
| Maximum gain any order-1 model can realize | 1.0262 |

: The Level 1 information budget. The realized-gain and maximum-gain rows are the whole argument of [@sec:gate_design]: a target of 2 nats over uniform exceeds the maximum gain achievable on the training distribution by a factor of 1.95, so the corpus the model was fitted to does not hold the first-order information the bar demands. The bound is under $p_{\text{train}}$ and not under the held-out evaluation sample ([@sec:background]); the finding is that the threshold was miscalibrated, not that a held-out score above the ceiling is impossible. {#tbl:letter_information}

Against the training distribution's order-1 budget, the agent's held-out gain is
89.9% of the information an order-1 letter model
can access there; by [@eq:signal_captured], with the optimum taken on the scored
passage itself, it is 57.3%. [@fig:information_ceiling] draws that budget directly in nats, with the
declared threshold placed outside the achievable region.

![The Level 1 information budget in nats. The declared 2-nat-over-uniform target lies beyond the conditional-entropy ceiling of the training distribution, so it is unreachable at this model order on that population regardless of implementation quality; [@eq:heldout_bound] shows it is unreachable on the held-out evaluation passage too, using that passage's transition entropy plus its fixed initial-prediction loss on the all-position denominator rather than extrapolating from training. A ceiling at or below the declared target, or a gain that did not equal the uniform cross-entropy minus the applicable all-position floor, would falsify this panel.](../../output/figures/information_ceiling.png){#fig:information_ceiling width=85%}

## Confirmatory replication on the held-back split

The members of the frozen significance family with a genuine held-back
partition available were the ones computed on EWT populations. Under the
registered protocol of [@sec:confirmatory_method] they were scored against the
whole held-back `en_ewt-ud-test.conllu` split exactly once, with every model
parameter read unchanged from the development fits and both controls run on the
same population first. Every registered direction and threshold prediction held:
the Stage 4 derived comparison and the Stage 6 sign test cross the frozen
per-test threshold of 0.000625 in the direction their
development result recorded, the Stage 4 gold comparison and the case-only Stage 5 ablation's
classification yield no discordant pair and are therefore not shown rather than
shown absent, and the other members of the family were not rescored and remain
development-population inference. The run cannot be repeated — the sentinel
claim marks the population spent — and the per-member detail, the controls, and
the document-clustered intervals are set out in [@sec:confirmatory].

## What the historical component results establish

They establish that six levels have real pymdp Agent builders and recorded
figures traceable to runs. They do not establish that every headline score used
live Agent inference. L1 does; the exact or NumPy paths of L2A, L2B, L3, and L6 and their
live-Agent checks are inventoried in [@sec:verification]. L4 uses a NumPy
forward filter whose live-Agent recursion reproduces it at every one of the
1319 held-out positions with
0 argmax disagreements and a largest
posterior deviation of 1.25e-07. The retained case-only L5
ablation uses a closed-form mixture whose live-Agent recursion reproduces the NumPy joint
oracle with 0 type-argmax
disagreements and a largest joint-posterior deviation of
1.67e-07 over all 1072
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
([@sec:generation]); its seven registered receiver
interventions — 5 measured, the case-word bridge measured as an exploratory
prior-only sweep ([@sec:case_word_sweep]), and the privileged-oracle
diagnostic the one registered seam that records no estimand — classify as
three help on its declared result axis, one harms its receiver, three move a prior without a delivered end-to-end effect estimate, and none are mixed-direction. By source kind,
five have a model-output source and two have a privileged-oracle source. The selected lexical channel, L3→L1g generation seam, and
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
frozen control on the drifted stream — 10.502499 against
11.611376 post-drift perplexity, a gap of
1.108877 — a point comparison with no interval; the
direction and the refusal clauses were committed before the run (decision
28, [@sec:verification]), and the ledger does not
promote the entry to confirmatory. What the numbers do *not* establish is language comprehension in any
behavioral sense, competitive capability on any benchmark, generality beyond
the corpora each level was scored on, or confirmation of any member other than those
that clear the frozen per-test bar on the held-back split ([@sec:confirmatory]); [@sec:limitations] draws the first three lines, and
[@sec:multiplicity] the fourth. [@sec:external_benchmarks] scores
three zero-shot external models against an AILM comparator
on identical positions within each separate letter, phoneme, and G2P arm, using
benchmark-specific fixtures rather than the positions scored here. That comparison is a
reference frame: it is not contamination-controlled and settles no
competitiveness question in either direction.

```{=latex}
\clearpage
```

One further accounting spans the historical component and channel experiments
rather than a single level. Their declared significance tests form one family of size 16 at familywise
$\alpha =$ 0.01, corrected two ways.
7 of the 16 cross the fixed
Bonferroni per-test threshold of 0.000625;
7 cross their Holm step-down threshold at the same
familywise level, which controls the same error rate under the same dependence
assumptions. Of those, 0 survive only the step-down. The Stage 6
sentence-sign test is nominally positive and crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold
(`stat_stage6_familywise_confirmed` = false, a
ledger key that predates the vocabulary used here and records the fixed-threshold
crossing alone; `stat_holm_stage6_rejected` = false, its
step-down bar being 0.00111111). It crossed that step-down
bar under the family ordering recorded before the lexical channel's backoff was
corrected (decision 110) and no longer does, which is
the instability of a step-down bar that [@sec:multiplicity] describes. This is
secondary evidence: a reversed-phase
model with worse mean log loss can still yield a nominally favorable sentence
sign, while the primary document-clustered log-loss interval spans zero on the
development population. That
survivor count is not a tally of endorsements. The biased prefix re-analysis of
[@sec:gate_design] is another survivor; it crosses the threshold precisely
because its subsample is unrepresentative. Within the family,
two members cannot reject at the corrected
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
91): the registered population, upstream CMUdict
entries absent from the vendored pin, was measured empty because upstream is
byte-identical to the pin. The family, each member's endpoint and test, and
each substantive member's replication population remain frozen in
`CONFIRMATORY_REPLICATION`; the four split re-analyses are explicitly marked as
non-rescored diagnostics whose full-set member carries the replication
(`src/ailm/statistics.py`, decision 79); the `ud_ewt_test` members were then scored exactly once against the held-back EWT
test split under decision 111
([@sec:confirmatory]): two clear the frozen per-test bar in their recorded direction, two yield no
discordant pair, and the other 12 members of the family remain
development-population inference.
[@sec:multiplicity] states the family executably, gives all three
qualifications in full, and reports each primary effect with its exact 95%
interval on the discordant win share ([@tbl:multiplicity_family]).

The external reference probes are scored on their own fixtures rather than on
the positions measured here, and [@sec:external_benchmarks] carries the protocol and the full
probe readings. The one result worth carrying into
the main text is that the comparator's raw letter-census margin is mostly
inventory compliance — whether a system will emit a space at all — and the
contrast that survives restriction to alphabetic targets, not the raw census
margin, is the finding.
