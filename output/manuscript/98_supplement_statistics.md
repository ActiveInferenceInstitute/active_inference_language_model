# Supplement F: Paired Tests, Multiplicity, and Null Bounds {#sec:statistics}

[@sec:verification] states which instruments this project runs and what each was
built to catch. This supplement is the argument behind three of them: why the
paired test is the exact one rather than an aggregate comparison, how the
declared significance family is corrected and what each correction admits, and
what a bound on a null does and does not license.

It sits here rather than in the main text because the reader who wants the
verdicts needs a page, and the reader who wants to check them needs thirty. The
verdicts are in [@sec:verification] and [@sec:results]; the derivations, the
assumption audits, the fragility analyses and the exact-value tables are here.
Every number in both places resolves to the same ledger entry.

## Paired testing for thin margins

The letter-to-phone margin is 0.546 points
(70.801% against 70.255%). A margin
that small cannot be established by comparing two aggregate accuracies — it sits
inside the range a different tie-break rule or smoothing constant could produce.
The gate therefore pairs the two decoders position-for-position through
`decode_positions` and `baseline_positions` in `src/ailm/levels/phoneme.py` and
applies the exact conditional (binomial) form of McNemar's test [@mcnemar1947] to the discordant pairs. Let $b$ count
positions only the filtering agent gets right, $c$ positions only the context-free
baseline gets right, and $n = b + c$. Under the null the two decoders are equally
likely to win a discordant pair, so $b \sim \operatorname{Binomial}(n, \tfrac12)$.
The test is *conditional on the observed discordant count* $n$. The
discordant-pair interval constructed from it retains that conditioning rather
than estimating uncertainty in the marginal accuracy difference without fixing
the discordance rate. @newcombe1998paired explains this distinction and develops
paired-difference score intervals that this manuscript does not compute.
The reported significance is the two-sided exact binomial probability

$$
p \;=\; \min\!\left(1,\;\; 2 \sum_{k=0}^{\min(b,\,c)} \binom{n}{k} 2^{-n}\right),
$$ {#eq:mcnemar}

the doubling being exact because the null distribution is symmetric about $n/2$,
and the cap binding only in the degenerate case $b = c$. This is the statistic
the code computes: `exact_two_sided_binomial_p` in `src/ailm/statistics.py`
calls `scipy.stats.binomtest(b, b + c, 0.5)` for nonzero discordance and returns
unit probability for zero discordance. Both `scripts/record_stage2_claims.py`
and `mcnemar` in `src/ailm/levels/coupling.py` use that shared helper;
`tests/test_stage2_gate.py` also checks the direct SciPy calculation. None of
these paths uses the $\chi^2$ approximation.
Measured over the full held-out set: *b* = 1379,
*c* = 1145, 2524 discordant
pairs in total, and *p* = 3.468e-06.

The unit those counts accumulate over is not the unit the corpus samples, and
that is worth stating at the point the test is introduced rather than after its
results. The 42,820 G2P positions come from
6484 held-out words; the 1319
Stage 4 tokens come from 229 sentences nested in
62 source documents; the
178 Stage 5 sentences come from
53 documents; the 187 Stage 6 sentences come from
27 documents. Every p-value in the declared family of
[@sec:multiplicity] is computed at the inner unit. That is a fact about the
design and not about any measurement — it is fixed by how the corpora are
drawn, so it holds whatever the numbers turn out to be — and it conditions
every p-value, interval and power figure below. What follows states the
assumption it violates, and the direction of the resulting error.

### Exact-test choice and power cost

The $\chi^2$ form is the asymptotic approximation to this same conditional
binomial, and on the full held-out set the two would agree to more digits than
anything here reports. It is rejected anyway, for a reason about the family
rather than about this row. The identical procedure is run on subsamples where
$b + c$ is small, and on Stage 5 where it is very small, and there the
approximation is least trustworthy and would need a continuity correction that
the exact test has no use for. Choosing per row by sample size would mean the
procedure changes with the answer's size, which is the shape of a defensible
decision made indefensibly often. One test is used everywhere.

The price is paid in power, and it is real. The exact test on a discrete
statistic is conservative: the attainable tail masses are a finite set, so the
achieved size sits strictly below the nominal level almost everywhere, and at
small $n$ there is no rejection region at all. `exact_two_sided_rejection_region`
in `src/ailm/statistics.py` returns that region explicitly, and returns an empty
one when the corrected threshold cannot be reached at any outcome — which is how
[@sec:multiplicity] can say that two family members were never able to reject
rather than inferring it from their p-values.

### Paired-test assumptions and dependence

The null is conditional: given the discordant total, each discordant pair is an
independent fair coin. The conditioning matters for reading the result. The
interval and share reported for each effect in [@tbl:multiplicity_family] describe
the *win share among discordant positions*, not the accuracy difference, and the
two are different quantities on different scales — a large win share over few
discordant pairs is a small accuracy gain.

The independence half is the strained one, and it is strained the same way in
every paired comparison this manuscript reports. The G2P unit is a letter
position, and letter positions inside one word are not independent: they share a
word, a decoder trajectory, and an orthographic neighborhood. The Stage 4 unit
is a token nested inside a sentence and source document; the Stage 5 and Stage
6 units are sentences inside documents. In all three, the authenticated source
document is the outer sampling unit.

A cluster-level reanalysis at the outer sampling unit has been run. The design follows the clustered-errors
literature: resampling clusters rather than observations preserves the
within-cluster dependence that invalidates unit-level intervals
[@cameron2008bootstrap], and the small number of outer units —
27 documents at Stage 6 — sits squarely where that literature warns naive
cluster-robust inference overrejects, so the wild cluster bootstrap of
@cameron2008bootstrap Section 4 is computed as a sensitivity arm beside every
interval below, not left as a promised refinement. 6 arms are resampled at their
outer unit — words for Stage 2B, source documents for Stage 4's English gold,
English derived, and Czech arms, and documents for Stage 6's sign and log-loss arms,
each arm retaining its original cluster sizes —
with 10,000 cluster bootstrap replicates at
0.95 coverage under a recorded seed. The estimand is fixed
first and both analyses are routed through the same code path, with the
independence case expressed as *no clusters* rather than as a second
implementation, so the comparison isolates exactly what clustering changes: the
uncertainty, never the point estimate.

Primary uncertainty is carried by the cluster-bootstrap interval. Each `*_P`
token is a whole-cluster sign-flip sensitivity analysis, finite-sample exact
only under the strong assumption that independent whole-cluster difference
vectors (or their sums for this statistic) are jointly invariant under sign
reversal. Marginally fair discordance directions and a zero expected cluster
sum do not imply that central symmetry, so this is not McNemar's pair-level null
lifted automatically to clusters. The table's normal-tail column is a
self-normalized large-cluster approximation under a weaker mean-zero null; it
requires independent clusters, finite variance, no dominant cluster, and enough
clusters, and is not finite-sample exact.

: Cluster-robust reanalysis at the outer sampling unit. The variance inflation is the ratio of clustered to naive bootstrap-replicate variances, a Monte Carlo quantity computed from 10,000 replicates. A ratio clearly below one means the clustered analysis is *tighter*, which a negative intraclass correlation produces; a ratio within the bootstrap's resolution of one, as in the Stage 2B row, whose positive intraclass correlation implies a design effect only marginally above one, is not sign-interpretable. Every *p*-value here re-analyzes a member already in the declared family at a coarser unit and sits outside that family by recorded exclusion. {#tbl:cluster}

| Arm | Unit | Clusters | ICC | Var. inflation | Clustered interval | Naive normal-tail *p* | Clustered normal-tail *p* |
|---|---|---:|---:|---:|---|---:|---:|
| Stage 2B letter-to-phone | word | 6484 | 0.003821 | 0.9845 | [0.317905, 0.77891] pts | 3.198e-06 | 3.226e-06 |
| Stage 4 derived | document | 62 | 0.028825 | 1.4666 | [4.889581, 11.180282] pts | 4.139e-09 | 0.0002638 |
| Stage 6 log loss | document | 27 | 0.057971 | 1.5139 | [-0.009563, 0.024959] nats | 0.4728 | 0.5362 |
| Stage 6 sign | document | 27 | -0.084473 | 0.5679 | [0.14655, 0.359562] share | 0.0009993 | 0.0003587 |

Several things in that table and its companion recorders are worth stating
plainly, because two of them cut
against the expectation the argument above sets up.

**The Stage 2B margin survives.** It is the thinnest in the family and was the
comparison most exposed to clustering, but the measured intraclass correlation
between letter positions in a word is
0.003821 — small enough that the variance inflation is
0.9845, indistinguishable from one at the
bootstrap's Monte Carlo resolution, and the clustered interval is
essentially the naive one. Letter positions within a word are less alike, for
this decoder's purposes, than the sampling structure suggests.

**The Stage 4 derived margin survives at the document unit, with wider
uncertainty.** Its 62 contributing documents
give an interval of [4.889581,
11.180282] percentage points. The intracluster
correlation is 0.028825 and the variance inflation is
1.4666, so treating
1319 nested tokens as independent is
anti-conservative even though it does not change the sign of the result.

**The case-only clause arm is also document-primary, but its distinct
uncertainties live in its own recorder.** The paired privileged-label gain resamples
53 whole documents and spans zero on
both reported scales. The rank interval likewise resamples
53 documents. The default classification
difference is -0.5618 points with a
99% whole-document interval of [-2.7779,
0]; its sentencewise McNemar row in
[@tbl:derived_bounds] remains a conditional secondary bound. Both this interval
and the paired-gain interval above are read at the same declared confidence
level; the two intervals state one coverage by construction.

**Stage 6's primary log-loss arm is non-confirmatory.** At the document unit its interval
spans zero, -0.009563 to
0.024959 nats, and its p-value is
0.5469 under the assumption-conditional whole-document
sign-flip sensitivity analysis. The positive point estimate remains
descriptive, but treating 187 sentences as independent
would understate that they came from only 27
sampled documents.

**Stage 6's secondary sign arm tightens, but is not promoted.** Its variance inflation is
0.5679, below one, because the measured
intraclass correlation is negative
(-0.084473): within a document, sentences disagree with each
other more than random assignment would produce. Clustering is not uniformly
conservative. More importantly, reversed tercile columns can make mean log loss
worse while retaining a nominally favorable sentence sign. The sign statistic
therefore cannot rescue the registered mean-log-loss prediction or the
non-confirmatory primary analysis.

Within each fixed evaluation population, the effect sizes are unaffected by
changing the uncertainty estimator — $b$, $c$ and the accuracy
differences they map to are descriptive counts over the positions scored, and
the reanalysis leaves every one of them where it was. What clustering moved was
the inferential half, and it moved it in both directions.

### Wild cluster bootstrap under limited outer clusters

Because few outer units is exactly where the percentile cluster bootstrap
under-covers, each clustered arm above also carries a wild cluster bootstrap
sensitivity interval in the style of @cameron2008bootstrap Section 4: a
Rademacher weight (equal chance of $+1$ or $-1$) is drawn independently for
every cluster, and each replicate re-forms the observation-weighted mean from
the *demeaned* cluster contributions,
$\mathrm{effect} + \sum_k w_k (S_k - \mathrm{effect}\, n_k) / n$, where $S_k$
and $n_k$ are a cluster's difference sum and size. The demeaning makes the
perturbation orthogonal to the point estimate, and with Rademacher weights the
replicate variance equals the cluster-robust (CR0-style) variance of the mean
exactly, so the interval is a symmetric sensitivity band on the same estimand —
never a replacement: the cluster bootstrap keeps primary uncertainty. The wild
arm draws 10,000 replicates at 0.95 coverage under
the same recorded seed discipline as the primary arm, and its spans-zero
verdict is compared with the recorded cluster-bootstrap verdict, not with the
naive one. On this manuscript's data the prediction that every verdict agrees
holds: 6 of the 6 clustered arms,
including the Stage 4 English gold and Czech arms omitted from the table above,
agree; [@tbl:wild_cluster] sets the two intervals side by side, arm for arm.
Agreement on this sample does not establish nominal coverage. With few outer
units, both intervals remain sensitive to the sampled documents and the
resampling assumptions; their agreement does not make either interval a
calibrated uncertainty bound. The Stage 6 log-loss arm stays non-confirmatory
on both.

: Wild cluster bootstrap sensitivity intervals beside the primary
cluster-bootstrap intervals of [@tbl:cluster], same arms, replicates, seed
discipline, and coverage. Both intervals are Monte Carlo quantities computed
from 10,000 replicates. {#tbl:wild_cluster}

| Arm | Wild interval | Cluster-bootstrap interval | Verdict agrees |
|---|---|---|---|
| Stage 2B letter-to-phone | [0.319878, 0.773631] pts | [0.317905, 0.77891] pts | true |
| Stage 4 derived | [4.761375, 10.830519] pts | [4.889581, 11.180282] pts | true |
| Stage 6 log loss | [-0.011343, 0.02152] nats | [-0.009563, 0.024959] nats | true |
| Stage 6 sign | [0.137579, 0.342337] share | [0.14655, 0.359562] share | true |

### Document-boundary correction for sentence-stride generators

The original Stage 4 and Stage 5 generators applied the stride to sentence
indices. Reconstructing the raw `# newdoc id` blocks reveals
140 source documents in both
Stage 4 arms and 114 in both
Stage 5 arms. The corrected generators partition all
318 identities before population filtering; both
current overlap counts are exactly
0 and
0. The before arm below is
recomputed from the pinned raw corpus by
`scripts/record_ewt_document_correction_claims.py`; no historical measurement
is typed into this manuscript. The before/after comparison
[@tbl:ewt_document_correction] places the retired sentence-unit analysis beside
the document-disjoint replacement on their respective populations and
resampling units.

| Quantity | Retired sentence split/resampling | Document-disjoint correction |
|---|---:|---:|
| Stage 4 held-out sequences | 253 | 229 |
| Stage 4 held-out tokens | 1390 | 1319 |
| Stage 4 agreement / baseline | 49.14% / 42.45% | 50.72% / 42.91% |
| Stage 4 margin, points | 6.690647 | 7.808946 |
| Stage 4 primary interval, points | [4.483931, 8.868786] (sentence) | [4.889581, 11.180282] (document) |
| Stage 5 train / test sentences | 813 / 211 | 846 / 178 |
| Stage 5 held-out positions | 1182 | 1072 |
| Stage 5 type-free / mixture / fitted reference perplexity | 4.4298 / 4.4404 / 4.424 | 4.3741 / 4.3858 / 4.3819 |
| Stage 5 paired gain, nats/sentence | 0.007288 | -0.010698 |
| Stage 5 gain interval, nats/sentence | [-0.020032, 0.037204] (sentence bootstrap) | [-0.034233, 0.008941] (document bootstrap) |
| Stage 5 AUC and interval | 0.7035 [0.5526, 0.8316] (sentence bootstrap) | 0.6483 [0.5026, 0.7724] (document bootstrap) |

: Exact before/after consequences of making the authenticated source document
the split and resampling unit. The Stage 4 effect remains positive but changes
population and widens at the proper outer unit. The case-only clause arm's fitted-reference point
changes sign while both before and after intervals span zero; the gate placement
therefore changes from `BRACKETED_REFERENCE` to `ABOVE_REFERENCE`. {#tbl:ewt_document_correction}

### Null-effect controls

A pairing routine with an off-by-one in the discordance count would report
significance on everything, so the procedure is required to return "no effect"
on inputs that have none, three times over in `tests/test_stage2_gate.py`. The
first null is real underpowered data: the same procedure on the bounded pymdp
subsample must come back non-significant, which it does. The second is exact — a
cyclic rotation of the baseline hit vector against itself preserves the hit count,
so $b = c$ and the exact null p-value follows by construction, and the test additionally asserts that the
rotation produced discordant pairs at all, because a null with no discordant pairs
demonstrates nothing about the machinery. The third is a false-positive rate: a
seeded batch of accuracy-matched Bernoulli comparators is run against the real
baseline and the count of nominally significant results is held to a declared
budget, because one seeded null is a cherry-pick and a batch is a measurement.
Note that comparing a
decoder with *itself* would be the wrong control here: it yields no discordant pairs
at all, which the exact test cannot score. The rotation is the construction that
keeps the discordant count non-zero while forcing the null to be true.

## Declared families and their corrected thresholds {#sec:multiplicity}

### Historical component and external-reference families

The historical component experiments and external benchmark declare separate
multiplicity families at different familywise levels. A reader must identify
the relevant family before interpreting a corrected p-value. The locally
specified connected GUM document-bootstrap interval is reported separately in
[@tbl:connected_gum]; it is not appended retrospectively to either family and
is not a claim of family-corrected significance.

The historical component family holds 16 tests at familywise
$\alpha =$ 0.01; what it delivers is
**development-population inference** — the sense fixed two subsections below —
and not confirmation, and every appearance of this family in this manuscript is
to be read under that label. The external benchmark declares its own
family of 18 contrasts,
6 on each of the 3 served probes, at a
familywise level of 0.05 and therefore a per-test threshold of
0.002777777777777778. The benchmark's per-test bar is consequently
4.4 times looser than the repository's.

That difference is deliberate and it is defensible, but only because the two
families answer different questions. The historical component family is the set of
hypotheses the original stage and channel conclusions rest on: whether a level beat its
comparator, whether a channel helped, whether a phase effect is real. A false
positive there is a false claim about AILM, so the level is strict. The benchmark
family is an external *reference frame* — a descriptive comparison against
prompted probes that the paper explicitly declines to draw conclusions from, and
whose section states in its own summary that it does not certify competitiveness.
A false positive there still misstates a protocol-bound contrast, even when
that contrast is not used to claim general model competitiveness.

The stipulated synthetic and in-context experiments additionally declare their
own descriptive contrast families, recorded with each protocol in
[@sec:external_benchmarks]. Their p-values use their own thresholds, rather than
the original external-reference family's bar or the historical component bar.

Two things follow for the original families. No test belongs to both: the benchmark contrasts are excluded from
`SIGNIFICANCE_FAMILY` by name and with a recorded reason, so nothing is corrected
twice and nothing is corrected by neither. And no benchmark result may be
promoted into a claim about AILM without being re-judged at the repository level,
because the threshold it cleared is not the threshold this paper's claims are
held to.

### Retrospective family definition and selection limits

Bonferroni bounds the familywise Type-I rate of a family fixed *before* the data
are seen. This family was not fixed before the data were seen, and the way it
was built is on the record. Members enter `SIGNIFICANCE_FAMILY` as levels are
built, so the family that fixes the corrected threshold has grown across the
project's life rather than having been enumerated in advance. Of its
16 members, three exist *only*
because an earlier measurement was
inspected: the subsample re-analyses were chosen after the alphabetical prefix
was found to overstate the Stage 2B margin ([@sec:gate_audit]), so the
evaluation split had been read before those tests were defined on it. And every
declared gate in this project was restated or re-scoped after its stage had been
built and measured, without exception ([@tbl:gate_audit],
[@tbl:unaudited_gates]); `scripts/audit_gates.py` then sets each restated
threshold *to* the measured baseline of the population the model is scored on.
The method practiced here is retrospective re-certification, not sequential
validation. That is a property of the process rather than of any one number it
produced, which is why it belongs here rather than among the limitations of the
individual results.

Multiplicity correction does not address this, because it is not the failure
multiplicity correction is for. A correction bounds the rate at which a *fixed*
set of null hypotheses is rejected by chance; it says nothing about a procedure
that decides which hypothesis to ask, and against which threshold, after seeing
what the evaluation set answers. Repeated adaptive use of a held-out set can
overfit that set directly, and selection over a criterion produces optimism in
the estimate the selection was made on. Neither failure raises a p-value, so
neither is visible anywhere in the arithmetic of this section, and dividing the
level by 16 does not touch either.

The bias runs in the optimistic direction, and the mechanism is concrete rather
than theoretical: a threshold restated to a stage's own measured baseline is
chosen with that stage's numbers already in view, so clearing it cannot be
evidence the author had no opportunity to anticipate. What keeps this from
being a claim that every restated gate is a foregone conclusion is a
counterexample in the ledger — the retained case-only clause gate was restated
the same way and its model *fails* it. Its separate document-clustered,
paired-sentence privileged-label gain interval spans zero, so the result
remains unresolved rather than being promoted to an equivalence claim
([@sec:clause_level]).

What this costs the labels defined below is stated exactly rather than hedged.
A member that crosses the corrected threshold has survived a retrospective
multiplicity procedure on the population this project developed against. It has
not been confirmed in the sense that requires an evaluation population untouched
at the moment the test was fixed, and outside the `ud_ewt_test` members rescored
under decision 111 ([@sec:confirmatory]) no result in this manuscript meets that
condition. The pre-registered predictions come closest and show precisely where
the distinction bites. Several predictions here were committed to version
control before the measurements that scored them — decisions 22, 25, 27, 28 and
29 among them — and that is what keeps a hypothesis from being written to fit
its answer; one of them, the predicted clause-to-word harm, duly failed to
reproduce. What a
commit timestamp cannot fix is the population: the corpus, the level design and
the operationalization each prediction is stated over were settled earlier, with
related data already in view.

Closing the gap is not a matter of a further correction factor. It requires a
partition of Brown, CMUdict and EWT that nothing in this project has scored
against, or an independently sourced replication corpus, with the family frozen
as it now stands and run once against it. That is done here for the 4 members that had such a partition
([@sec:confirmatory]), and an honest census of the vendored artifacts says why
it cannot be done for the rest.
The CMUdict artifact holds no unscored material: the interleave's train side
fitted the model, its test side is scored in full, and the family-disjoint
re-partition of decision 74 has itself been scored, so the
CMUdict-population members can only be replicated against an independently
sourced lexicon. The UD EWT artifacts are built from the dev split alone, whose
test side is scored in full; the corpus's own `test` file is the registered replication population for the
EWT members, and no family member had been scored against it before the run
of decision 111. The subsample re-analyses are diagnostics of the shipped
split and have no analogue on fresh data. That census, the exact partition rule
for the EWT re-vendor, the frozen per-member endpoint and test, and the
one-run discipline were pre-registered as `CONFIRMATORY_REPLICATION` in
`src/ailm/statistics.py` and decision 79 of
the decision log. The run was then executed to its first data touch and
stopped by its own discipline: the registered `external_lexicon` population —
upstream CMUdict entries absent from the vendored pin — was measured empty,
because upstream is byte-identical to the pin, and the replication is retired
by exhaustion for those members (decision 91).
The `ud_ewt_test` members were then scored exactly once against the held-back
test split under decision 111
([@sec:confirmatory_method], [@sec:confirmatory]): two clear the frozen per-test bar in
their recorded direction and two yield no discordant pair. Every
other verdict in this section is a development-population verdict, and the
sections that report those verdicts — [@sec:results] and [@sec:gate_audit] —
are to be read under this heading rather than in spite of it.

### Declared family membership

Every p-value used for a threshold-crossing inference in this manuscript belongs
to a single declared multiplicity family, stated executably rather than
approximately. Descriptive diagnostics and re-analyses of an existing family
member are reported only when their exclusion is named and machine-checked:
`SIGNIFICANCE_FAMILY` in `src/ailm/statistics.py` enumerates the
16 p-value claims, and `scripts/record_statistical_claims.py`
re-derives every family verdict from the ledger on each run. The recorder
refuses to run if any declared member is absent from the ledger or if the
declared list contains a duplicate, so the family cannot shrink by a label
quietly failing to be produced.

The membership rule is one slot per *recorded label*, not one per scientific
hypothesis. The subsample re-analyses each hold their own slot: they are
deliberate re-measurements of the Stage 2 comparison over different
subsets of one test split. Those subsets are neither disjoint nor nested — an
alphabetical prefix and the registered strides overlap partially and
contain one another not at all — so they are dependent measurements rather
than independent ones, which is a reason to count them separately rather
than a reason to merge them. The bounded pymdp subsample and the subsample
study's strided arms enter as separate labels for the same reason: they are
separate recorded measurements over different position filters. The full-set
Stage 2 comparison holds one slot only — `subsample_full_mcnemar_p` is a second
label for the same paired comparison and must not consume a second slot. This
rule is conservative for every positive claim: a smaller, de-duplicated family
could only loosen the threshold. It does tighten the threshold that the Stage 6
sign test is judged against, and the verdict there is the same under each
family definition considered — see below.

### Bonferroni correction and its power cost

Bonferroni is the least powerful of the standard familywise corrections, and it
is the one used here. The reason is not conservatism for its own sake but the
shape of the family. Its members are strongly and unquantifiably dependent:
several are re-analyses of one test split, and the lexical and coupling channels
score the same G2P positions under different mixing weights. Bonferroni controls
the familywise error rate under *arbitrary* dependence, so nothing has to be
assumed or estimated about that dependence structure. Benjamini–Hochberg would
be more powerful but controls the false discovery rate, a weaker guarantee than
the one the claim-strength labels below are defined against, and its usual form
additionally assumes a dependence structure this family does not obviously have.

Holm's step-down procedure is the real alternative: it also controls the
familywise rate under arbitrary dependence and is uniformly at least as powerful,
so every member Bonferroni admits Holm admits too, and the converse can fail.
The arithmetic of that domination is machine-checked as
`holm_rejects_every_bonferroni_rejection` in `formal/AILM/Holm.lean`, which
proves the containment on a sorted family and exhibits a family on which it is
strict, and which proves nothing about familywise error control: that step is
Boole's inequality applied to the rejection events and stays assumed here,
exactly as it does for Bonferroni.
What Bonferroni buys for that loss is a threshold that is a constant. Under Holm,
the bar a given test is judged against depends on the whole sorted vector of the
family's p-values, so re-measuring an unrelated member — or adding one, which
this project does every time a level is built — can flip a claim's verdict
without anything about that claim having changed. A single fixed per-test
threshold keeps each claim's status a function of its own measurement, which is
what makes the ledger's per-claim verdicts stable across runs. The cost is
recorded honestly here: the survivor set reported below is the conservative one,
and a uniformly more powerful procedure at the same familywise level would admit
at least as much.

### Claim-strength labels and demotion rules

The manuscript uses the following threshold-defined labels for a p-value claim.
*Nominally
significant* means $p <$ 0.01, the uncorrected familywise
level, and is demoted by nothing except a re-measurement. *Corrected-threshold
survivor* means $p <$ 0.000625, the Bonferroni per-test
threshold for the declared family, and is demoted by a re-measurement *or* by
the family growing. Neither strength is asserted in prose: the survivor set is a
recorded list re-derived from the ledger on every recorder run, and the Stage 6
verdict is a recorded boolean, so a threshold that moved in code and not in
prose changes the rendered sentence rather than being contradicted by it.

Both labels name a threshold crossed on the development population and neither
means *confirmed*, for the reason given above: the family is retrospective, and
a corrected threshold bounds a fixed family's Type-I rate rather than restoring
independence to a reused split. The stronger word is reserved for a result
produced by running the frozen family once against data it has never scored, and
in this manuscript only the `ud_ewt_test` members rescored under decision
111 ([@sec:confirmatory]) were produced that way. That run was pre-registered rather
than merely wished for: `CONFIRMATORY_REPLICATION` in `src/ailm/statistics.py`
freezes every substantive member with its replication population and assigns
each split re-analysis the explicit non-rescored diagnostic disposition; decision
79 registers the partition rules and the
one-run discipline. It was executed to its first data touch and retired by
exhaustion for the `external_lexicon` members (decision
91); the `ud_ewt_test` members were scored once under decision
111 ([@sec:confirmatory]). One ledger label preserves the older
vocabulary — the recorded boolean is `stat_stage6_familywise_confirmed`, and it
keeps that name because renaming a recorded label would rewrite the ledger and
break the token that reads it. It records the threshold crossing described here
and nothing else; the name is a key, not a claim.

### Development-population outcomes after correction

Survival below is survival of the retrospective procedure, on the development
population, in the sense fixed two subsections above; it is reported as a
threshold crossing and not as a confirmation.

At familywise $\alpha =$ 0.01 the Bonferroni per-test
threshold is 0.000625, and 7 of
the 16 declared tests cross it, the recorded survivor set
being `coupling_mcnemar_p`, `fitted_paired_p`, `geometric_lexicon_paired_p`, `lexical_best_mcnemar_p`, `stage2_g2p_mcnemar_p`, `stage4_derived_mcnemar_p` and `subsample_prefix_mcnemar_p`. The prefix re-analysis among them is significant precisely
because the prefix subsample it measures is unrepresentative
([@sec:gate_audit]), so surviving the correction is not by itself an
endorsement — a corrected p-value inherits every bias in the population it was
computed over.

Holm's step-down correction over the same family at the same familywise level
admits 7: every fixed-threshold survivor above, which
it must, plus 0 more. Both corrections are reported
throughout, because they answer the same question with different properties. The
fixed threshold is a constant, so a regression test can assert against it and a
reader can check a single p-value against a single number. A Holm threshold
depends on the whole ordered family, so a member's verdict can change when an
unrelated measurement moves — informative in a paper, unstable in a gate.

Within the declared family, two
members could not have rejected under any outcome, and a
survivor count read without them overstates how discriminating the family is.
The gold case comparison has $b = c =$ 0: the two
predictors are identical on that population, so there are no discordant pairs to
test. The Stage 5 comparison has discordant pairs but too few for the exact
two-sided test to possess any critical region at the corrected threshold, which
is a property of $n$ and the threshold alone and is computed as such by
`exact_two_sided_rejection_region` rather than read off the p-value. Excluding
both leaves 7 survivors among the members that were
structurally able to reject.

### Stage 6 uncertainty and robustness limits

The registered P1 estimand is the mean held-out log-loss difference under the
observed-position conditional table, not the number of sentences on which that
table wins. The direction of that mean was registered before the held-out run,
and its point estimate is positive. Population inference must nevertheless use
the sampled document as the outer unit. The document-clustered mean-log-loss
interval spans zero and its randomization p-value is
0.5469, so the primary analysis is imprecise and
non-confirmatory. No target-effect power analysis was recorded.

The repository-wide family contains the *secondary* sentence-level sign test.
Its two-sided p-value is 0.001232362495942883: nominally significant at
$\alpha =$ 0.01, and it crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold,
the step-down bar being 0.00111111 and the fixed one
0.000625. The corresponding recorded booleans are
false for Holm and
false for the fixed threshold. The latter label is
a historical ledger key; as [@sec:multiplicity] states, neither boolean licenses
the word *confirmed* on this reused development population.

The sign statistic is not merely a noisier form of the registered mean. It drops
magnitude, thereby testing a different estimand. Reversing the fitted tercile
columns is a falsifier: it makes the mean log loss worse than the comparator yet
still yields a nominally favorable sentence sign. That control shows why a Holm
crossing, a one-sided p-value of 0.000616181, or the
document-clustered sign p-value 0.0003587 must stay
secondary. Selecting the sign formulation that crosses after the mean analysis
does not would exchange estimands in the model's favor.

Family-size and correction sensitivities remain useful arithmetic checks. If the
subsample re-analyses were collapsed, the two-sided sign p-value would still
not cross the resulting fixed threshold. Applied to all 16
declared members, Holm admits 7 tests, with Stage 6 at
rank 8; Bonferroni admits the fixed survivor set reported
above. These facts describe the secondary statistic. The paper's Stage 6 verdict
comes from the registered positive point estimate together with the
document-clustered mean-log-loss uncertainty: promising, imprecise, and
non-confirmatory.

A separate descriptive score runs sequential prediction through the live pymdp
Agent and is reported in [@sec:narrative_level]. It establishes live execution
and transition dependence, but it is neither the registered observed-position
estimand nor a substitute for the document-clustered primary analysis.

### Effect sizes, intervals, and agreement interpretation

One warning goes before the numbers, because the shape of the display invites
the wrong reading. **None of the intervals in [@tbl:multiplicity_family] or
[@fig:multiplicity_forest] is a confidence interval on an accuracy difference.**
They are intervals on the win share among discordant pairs, a conditional
quantity on a different scale whose null is one half rather than zero. A forest
plot of them has the visual grammar of a treatment-effect plot and does not
carry its meaning: mapping a row onto the accuracy scale multiplies it by that
row's discordance rate, which for the Stage 2B row is
2524 over 42,820, so a share far
from one half sits alongside an accuracy difference of
0.546 points. [@eq:paired_difference] is that mapping and
[@tbl:derived_bounds] is where the accuracy-scale intervals are reported.

Effect sizes carry exact uncertainty, not only p-values. Each primary paired
effect reports the favored share of its discordant pairs with an exact
binomial Clopper–Pearson 95% interval and the power of the exact two-sided
test at the corrected threshold against a fixed reference share of
0.55 (`clopper_pearson_interval` and
`exact_binomial_power` in `src/ailm/statistics.py`).

The interval is Clopper–Pearson rather than Wilson or Wald, for the same reason
the test is exact rather than asymptotic and at the same cost. Clopper–Pearson
guarantees at-least-nominal coverage for every $n$ and every true share; Wilson
is shorter and better centered on average but its coverage oscillates and dips
below nominal for some parameter values, and Wald's coverage collapses near the
boundary. In a document whose thesis is that a check must be able to fail, an
interval that is sometimes narrower than its stated confidence is the wrong
trade, so the reported intervals are wider than they need to be and never
narrower than they claim. What the interval covers is the *conditional* win
share among discordant pairs, not an accuracy difference; the two are not
interchangeable, and intervals on the paired accuracy difference are reported
separately, under the mapping and at the coverage stated in [@sec:bounds].

The reference share is a declared convention, not a derived quantity: one
constant in `src/ailm/statistics.py`, applied unchanged to all registered primary comparisons,
fixed as the smallest paired effect the project would treat as scientifically
interesting. Its arbitrariness is bounded by being *fixed* — a reference share
chosen per comparison would make every power figure a statement about the choice.
Power at the *observed* share is deliberately not tabulated: it is a monotone
transform of the p-value and adds no independent evidence. Power at the fixed
reference share is genuinely informative and separates the comparisons — the two
large-$n$ comparisons could have detected a small effect, while the coupling and
Stage 4 comparisons, at their corpus-determined sizes, largely could not.

Two limits on reading those power figures. They are computed at the *observed*
discordant total, treated as fixed, although $n$ is itself random in a paired
design — so each is a conditional detectability statement about the sample that
was obtained, not about the design that would obtain it again. And all of them
are retrospective diagnostics, not prospective sample-size justifications: no
sample here was sized to reach a power target, because every one of them is the
whole of an available corpus.

```{=latex}
\csname @ifclassloaded\endcsname{beamer}{}{\Needspace{36\baselineskip}}
```

### Forest-plot geometry and exact-value provenance

[@fig:multiplicity_forest] draws this family's geometry on the share scale —
the Stage 2B, Stage 4, lexical-channel, and coupling-harm intervals against the
structural null of one half, and the Stage 6 one- and two-sided p values against
the familywise, Bonferroni, Holm, and collapsed-family thresholds — while
[@tbl:multiplicity_family] below remains the exact-value record.

![**What to see: each declared significance test is drawn with both its uncorrected and its multiplicity-corrected interval side by side, so a reader can see directly which results survive correction for testing many hypotheses at once and which do not.** The multiplicity family at a glance. Top: the primary paired effects,
each drawn as the favored share of its discordant pairs with nested
whiskers: the exact *marginal* Clopper–Pearson 95% interval, and behind it
the family-corrected interval at one minus the Bonferroni per-test level, whose
exclusion of one half is the verdict the badge states
([@tbl:multiplicity_family] is the exact-value record). These are share intervals, not treatment-effect intervals: the axis is
the conditional win share among discordant pairs, its null is one half rather
than zero, and converting a row to the accuracy scale requires the discordance
rate through [@eq:paired_difference]. Two reference lines are drawn: the
McNemar null at the structural
constant 1/2 (dashed) and the fixed power reference share
0.55 (dotted). A statistics column beside the forest gives,
per row, the discordant $n$ and the evaluated population it is drawn from with
its unit; whether a cluster-robust re-analysis covers that same population, and
if so its clustering unit, cluster count and level, or else that the drawn
interval is not cluster-adjusted; the exact power at the reference share
(matching the table); the corrected-level badge; the Holm verdict; and which
held-back population the confirmatory protocol registers to re-score the row,
with whether that run has happened. Every verdict word is read directly from a
recorded boolean rather than inferred from the geometry beside it, and the
panel footer states how many of the family survive each correction. Power at
the observed share is a monotone transform of the p-value and is not drawn. Bottom: the Stage 6
sign test's one- and two-sided p values, drawn against the familywise
0.01, the fixed Bonferroni per-test
0.000625, this member's Holm step-down threshold
0.00111111, and the relaxed threshold for the collapsed
family, derived in code as the familywise level over the size that
16 members become when the subsample re-analyses are
counted once. The shaded band runs from the fixed threshold to the Holm one:
that is the interval in which the verdict depends on which of two equally valid
familywise corrections is applied. Where the two-sided test lands relative to
that band is what the recorded booleans state: the fixed-threshold flag is
false and the Holm verdict is
false. Of the two disclosed relaxations only
the one-sided test crosses; the family-size relaxation does not rescue the
result, and it is drawn to show that it does not. The figure refuses to draw if
the corrected Clopper–Pearson verdict recomputed from the recorded raw discordant
counts contradicts any recorded exclusion boolean; if a drawn marginal whisker
straddles 1/2 while its corrected verdict claims exclusion; if the drawn Holm
geometry contradicts the recorded Holm verdict; if Holm's threshold is drawn
below Bonferroni's; or if the drawn p-threshold geometry contradicts the
recorded survival flag or either disclosed fragility direction. It also refuses
to draw if a recorded corrected interval disagrees with the recorded survivor
set it is the dual of, if the Bonferroni survivors are not a subset of Holm's,
if a survivor list disagrees with its recorded count or names a label outside
the declared family, if a discordant count exceeds the population it is drawn
from, or if the protocol maps a different number of members to a completed
confirmatory population than the ledger counts as scored.](../../output/figures/multiplicity_forest.png){#fig:multiplicity_forest width=95%}

| Paired effect | Discordant pairs | Favored share | Exact 95% interval | Power at share 0.55 |
|---|---|---|---|---|
| Stage 2B letter-to-phone against the lookup table ([@eq:mcnemar]) | 2524 | 0.546355 | [0.52669, 0.565912] | 0.943796 |
| Stage 4 derived against the UPOS bar ([@sec:case_gate]) | 307 | 0.667752 | [0.612006, 0.720224] | 0.04599 |
| Lexical channel against the anchor ([@sec:hierarchy]) | 4493 | 0.658803 | [0.644728, 0.67267] | 0.999488 |
| Coupling harm, uncoupled decoder favored ([@sec:limitations]) | 547 | 0.603291 | [0.560912, 0.644548] | 0.138409 |

: The primary paired effects. Power is reported only at the fixed reference share 0.55 and never at the observed share, because power computed at the share the data happened to produce is a monotone transform of that arm's *p*-value and adds no evidence the *p*-value did not already carry [@hoenig2001abuse]. On that scale Stage 2B and Stage 4 separate sharply rather than reading alike: Stage 2B reaches 0.943796 and could have detected the smallest effect this project treats as interesting, while Stage 4 reaches 0.04599 and could not, so a null at Stage 4 would have bounded nothing — which is what [@sec:gate_audit] says of it independently. Every row describes the win share among *discordant* pairs, not an accuracy difference. The favored share is the favored count over the discordant total, in the direction the claim names — $b/(b+c)$ for the Stage 2B, Stage 4, and lexical-channel effects, and $c/(b+c)$ for the coupling row, where the measured effect is harm and the *uncoupled* decoder is favored. Intervals are exact Clopper–Pearson at the marginal 95% level and are not simultaneous across the table, so reading them jointly overstates the joint confidence; the separately recorded intervals at the Bonferroni-corrected level still exclude one half for every effect, and those are the ones the corrected verdicts are read from (`scripts/record_statistical_claims.py`). Power is the exact two-sided test's power at the corrected threshold 0.000625 against the fixed reference share, computed at the observed discordant total from ledger counts. {#tbl:multiplicity_family}

## Null bounds and limits on absence claims {#sec:bounds}

Several results in this project are null or near-null, and "we found no effect"
is weaker than what the data support. What the data support is a *bound*: an
interval saying how large the effect can be. That convention is stated once,
executably, in the module docstring of `src/ailm/statistics.py`, rather than
restated approximately at each null where it could drift between sections.

### Share and difference scales with distinct nulls

Every paired comparison here is exact and conditional, and it admits two scales
that answer different questions. The **discordant share** $b/(b+c)$ is the
probability that a discordant pair falls the probe's way; its null value is one
half, and [@tbl:multiplicity_family] reports it. The **paired difference**
$(b-c)/n$ is the probe's accuracy minus the comparator's over the same
positions; its null value is zero, and [@tbl:derived_bounds] reports it. The
second is the first mapped back through

$$
\delta \;=\; (2s - 1)\,\frac{b + c}{n},
$$ {#eq:paired_difference}

so a large win share over few discordant pairs is a small accuracy difference,
and the two numbers are never interchangeable. `paired_difference_interval` in
`src/ailm/benchmarks/` is the single implementation of that mapping.

Difference bounds are drawn at 95% two-sided coverage, and
at 99.9375% — one minus the Bonferroni per-test
threshold — wherever the bound has to pair with a family-corrected decision. An
interval is read at the level of the decision it supports, so the two levels are
not substitutes for one another. The share intervals of
[@tbl:multiplicity_family] sit at the same coverage as the difference bounds:
`SHARE_CONFIDENCE` and `DIFFERENCE_CONFIDENCE` are one constant, and both print
95%. That is a property of one line of source rather than
of a convention, which is the point of writing it once: a future divergence
would have to be declared there deliberately instead of accumulating between two
recorders. The Stage 5 bootstrap intervals — the paired privileged-label gain,
the rank interval, and the classification difference — are the declared
exception: they are read at 99% under
their recorder's own contract (`STAGE5_BOOTSTRAP_CONFIDENCE`), because they
are percentile intervals rather than the share interval and were kept out of
the harmonization on that ground (decision 66).

### Derivation and scope of null bounds

No bound in [@tbl:derived_bounds] is a new measurement. The recorder wrote
`*_delta_low_*` and `*_delta_high_*` for the external-benchmark arms and for
nothing else, so every other paired result in this project could state its
contrast but not how large the effect can be. `paired_difference_interval` is a
pure function of $b$, $c$ and $n$, and all three are already in the ledger for
each arm, so the missing bounds need no run — only the same call at render time.
`paired_difference_bound_from_counts` in `src/ailm/manuscript_variables/__init__.py`
makes that call.

It wraps the recorder's own function rather than reimplementing it, and that
distinction carries the argument. A reimplementation at a different confidence
level, or with a Wald interval in place of the exact conditional one, or over
the unconditional difference, would each produce a plausible pair of numbers,
render without error, and place two different kinds of interval in one table
under one name. `tests/test_statistics.py::TestDerivedPairedBound` forecloses
that: it runs the derivation on every arm that does have a recorded bound and
requires it to reproduce each recorded endpoint. Those arms are the only place
the recorder and render-time derivation can be compared, and a failure there would not mean the tolerance
is too tight — it would mean the recorder builds its interval from something
other than $b$, $c$ and $n$, and that no row derived from those three is
admissible.

The denominator is the one input the counts cannot check about themselves. Both
$b$ and $c$ are read from labels naming the arm, so reading the wrong one is
loud; $n$ is read from a different label, and a plausible wrong choice — test
positions instead of test sentences, the full pool instead of the subsample —
rescales every bound in the block while leaving each number looking reasonable.
Each arm's $(b-c)/n$ is therefore checked against a point estimate recorded by a
different code path, and the arms whose comparison admits no ties are
additionally required to have their counts exhaust the denominator exactly. The
denominator-doubling and fixed-width controls stop those checks from being satisfied by a stub: doubling $n$
must halve the bound, because the share is mapped back through the discordance
rate, and a derivation hard-wired to a fixed wide interval must fail the
requirements that the bound move with its inputs and narrow as pairs accumulate.

### Interpretation limits of null bounds

A bound says that values outside it are inconsistent with the data at the stated
coverage. That is the whole of what it says. An interval that contains the null
value bounds the effect; it does **not** establish that the effect is absent.
"The difference is between -0.561798 and
0.533708 points" is a claim about how large the
effect can be. "There is no difference" is a different claim, and no interval
containing the null has ever supported it.

Coverage here is marginal, not simultaneous, and it carries no multiplicity
correction below 99.9375%. A
95% interval that excludes the null is therefore not
equivalent to clearing the corrected threshold 0.000625, and
this project holds the case that makes the gap concrete. The Stage 6 sign
test's bound of [0.093249, 0.380275]
excludes zero while its recorded corrected-threshold survival flag is
false. Reading "excludes zero" as "significant"
would contradict [@sec:multiplicity]. The lexical channel at mixing weight
0.9 was a second such case, with a bound excluding zero
beside a p-value above even the uncorrected level, until its backoff defect was
corrected (decision 110); its corrected p-value is
3.903e-87, and it is now a decided effect on both
readings.

Conditioning on the discordant set is what makes the construction exact, and the
price is conservatism. With few discordant pairs the interval is wide, and a
wide interval is a weak bound rather than a strong null. The case-only clause
arm is the limit case: 1 discordant pairs, and a bound nearly
as wide as the quantity it bounds. Its lower endpoint coincides exactly with its
point estimate -0.561798, which is a property of the
construction rather than a coincidence: every discordant pair there fell one
way, so the exact lower limit on the share is zero and [@eq:paired_difference]
cannot map it past the observed difference. A reader who treats that endpoint as
independent information is counting one number twice.

### Conditions for an equivalence claim

Turning a bound into an equivalence claim — "the effect is negligible" rather
than "the effect is at most this large" — needs a smallest effect of interest
declared before the data are seen. This project declares one, on the share
scale: 0.55, already in service as the fixed
reference share for power. `equivalence_region` mirrors it about the null,
giving [0.45, 0.55], so the region
is symmetric and covers effects in either direction.
`share_interval_supports_equivalence` is then the two-one-sided-tests decision
read straight off the interval: the claim holds only when *both* ends fall
inside the region. Checking only that the interval contains one half would pass
every arm with too few pairs to say anything, which is exactly the defect that
makes an underpowered null look like a positive finding.

The estimand is the win share among discordant pairs, so an equivalence claim
made here is never a claim that two systems are equivalent in accuracy, and no
sentence in this manuscript converts one into the other. The two are different
quantities: a split close to the exact-test null over many discordant pairs and
the same split over a handful bound the marginal disagreement by amounts that
differ by the ratio of their discordance rates.

The region is declared on the share scale and not on the difference scale, and
the asymmetry is deliberate. Converting it through [@eq:paired_difference]
requires the *observed* discordance rate, so the threshold would depend on the
data it judges, and the dependence runs the wrong way: the fewer discordant
pairs an arm produced, the smaller the converted threshold would become, so
equivalence would be easiest to claim precisely where the evidence is thinnest.
No difference-scale margin is declared here, so no difference-scale equivalence
claim is made anywhere in this manuscript.

How many discordant pairs an equivalence claim needs at all is analytic in the
sense that matters: it consumes no measurement.
`minimum_discordant_pairs_for_equivalence` asks, of the most favorable outcome
available at each pair count — the split closest to even, where the interval is
narrowest — how many pairs are needed before that interval fits inside the
region. The answer is 420 pairs at
95% and 1204 at
99.9375%. The search requires the predicate to keep
holding over a window of further counts, because it is not monotone in the pair
count: an odd count cannot split evenly, so its most balanced split is
off-center and its interval is wider than that of the even count below it.
Returning the first count that qualifies would report a parity artifact as a
sample-size requirement.

That threshold is what separates "this null is only a bound" from "this null
could have been an equivalence and was not". The case-only clause arm at
1 discordant pairs and Stage 6 at
187 sit below it, so each of those results is a bound and
nothing more. That is a statement about what that many pairs can support, not a
complaint about the measurement: every one of these corpora is used whole, and
none was sized to a target.

The registered context-window arms were listed here, and the census removed them from
the list twice over. They now carry
771 and 515
discordant pairs, both above the threshold, and both contrasts clear the
corrected family threshold, so they are established effects rather than nulls of
any kind. They remain in [@tbl:derived_bounds] because the table is the inventory
of arms whose bounds are *derived* rather than recorded, which is a statement
about provenance and not about verdict; they are absent from [@fig:null_bounds],
which is the inventory of nulls. Keeping a resolved contrast in a bounds panel
would report it as unresolved, so the figure refuses to draw such a contrast
rather than relabel it. The separation is enforced by the drawing code, not by
the caption.

No arm in the current ledger passes the equivalence test. The lexical channel at mixing weight
0.9 did, under the backoff defect corrected in decision
110: with 5029 discordant
pairs it had the precision, and its corrected-level share interval then sat
inside the region. Corrected, the same arm is a decided effect;
`share_interval_supports_equivalence` reports 0
for it at 99.9375% coverage, the indicator being
nonzero only when both ends of the interval fall inside the region, and its
interval sits far outside it. Every other near-null drawn in
[@fig:null_bounds] is below the pair count an equivalence claim needs, so what
the roster now shows is the general case: equivalence is paid for in
precision, and no near-null in this project has produced the pairs to pay. The
arm's two readings before and after the correction are set out together
below, because together they say more about the convention than either does
alone.

| Paired arm | Direction and scale | Discordant pairs | Point estimate | Bound at 95% |
|---|---|---|---|---|
| Stage 5 clause-type classification ([@sec:clause_level]) | agent minus majority baseline, points | 1 | -0.561798 | [-0.561798, 0.533708] |
| Stage 6 narrative-phase sign test ([@sec:narrative_level]) | phase model minus comparator, net directional share | 187 | 0.240642 | [0.093249, 0.380275] |
| Stage 2 G2P versus the per-letter bar ([@sec:phoneme_level]) | model minus bar among discordant positions, net directional share | 2524 | 0.09271 | [0.053381, 0.131824] |
| Lexical channel at mixing weight 0.9 ([@sec:hierarchy]) | mixed minus anchor, points | 5029 | 3.258 | [2.941811, 3.57005] |
| Context window, Gemma 3 4B ([@sec:external_benchmarks]) | long minus short window, accuracy difference (proportion) | 771 | -0.01245 | [-0.019304, -0.0055] |
| Context window, Qwen 2.5 3B ([@sec:external_benchmarks]) | long minus short window, accuracy difference (proportion) | 515 | -0.013707 | [-0.019207, -0.008049] |

: The derived paired bounds. Every row is computed at render time from ledger counts by the same function that produced the recorded external-benchmark bounds; none of them is a new measurement. Directions and scales differ by row and are named in the second column, so the numbers are not comparable across rows. Recomputing a row from its recorded $b$, $c$ and $n$ and landing outside the printed interval falsifies that row; the convention behind every row is falsified if the same derivation fails to reproduce the recorded endpoints on the external-benchmark arms, which are the only arms where both paths exist. Every Stage 6 sentence is directional; the Stage 2 row instead conditions explicitly on its recorded discordant subset. A pair count below 420 cannot support an equivalence claim at 95% under any outcome, so those rows bound their effect and say nothing about its absence. The Stage 2, lexical high-weight and context-window contrasts are above that count and established, so their intervals locate measured effects rather than bound unresolved ones; they appear here because their bounds are derived rather than recorded, and this table's membership rule is provenance. {#tbl:derived_bounds}

The punctuation-type-oracle→word-prior intervention is intentionally absent from
[@tbl:derived_bounds]. Its 74,013 token outcomes are nested
inside 52 Brown documents, so an exact interval
that treats tokens as independent pairs would be pseudoreplication. Its declared
uncertainty is instead the whole-document bootstrap over per-emission log gain:
[0.00088152,
0.00233064] nats per emission, reported in
[@sec:clause_word_channel].

[@fig:null_bounds] draws [@tbl:derived_bounds] and adds the question the table
cannot answer by itself: not how wide each bound is, but whether the arm that
produced it could ever have supported an equivalence claim. Most could not, at
any outcome, and that is a property of the arm rather than of the result.

![**What to see: no arm in this project is reported as "no effect" — every null and near-null is drawn instead as a bounded interval, so a reader can tell a tight bound from an inconclusive one at a glance.** Every null and near-null in the project, drawn as a bounded effect. Top: the arms measured on an accuracy difference, each with its exact conditional 95% paired interval, its discordant pair count over the population that produced it with that population's unit, and whether a cluster-robust re-analysis covers the same arm or none was recorded; the dashed line is no difference, and the established lexical high-weight contrast is drawn there for scale as the Stage 2 contrast is in the middle panel. Middle: the directional-share arms, with the familywise-established Stage 2 G2P contrast drawn beside the Stage 6 near-null as the live positive control. Bottom: the same arms against the pairs each produced, on a log axis, with the shaded region marking arms that cannot support an equivalence claim at any outcome because they never produced enough discordant pairs — the point being that a bounded null and an unresolvable one look identical in a table and different here. Every value is derived at render time from recorded counts by the function that produced the recorded external-benchmark bounds. The nested-token punctuation-oracle intervention is excluded because its valid interval resamples whole documents. A drawn interval that disagrees with its recorded endpoints, an arm placed inside the shaded region while its interval is narrow enough to exclude the declared smallest interesting effect, an established contrast drawn as though it were a null, or a cluster count that does not partition the population it claims to cover would each falsify this panel. A bound stated over an unnamed unit cannot be judged wide or narrow, which is why each row names its own.](../../output/figures/null_bounds.png){#fig:null_bounds width=95%}

### Nominal equivalence and the effect of correction

The Stage 3 lexical channel at mixing weight 0.9 carried,
until decision 110, two readings a reader is likely to
take as incompatible: a 95% difference bound that
excluded zero beside an exact p-value that missed even the uncorrected level,
and a corrected-level share interval that fit inside the equivalence region.
The arm is kept in this section because the argument it prompted is about the
convention and survives the arm's own correction, and because the corrected arm
shows the convention's other face on the same counts. On its
5029 discordant positions the mixed decoder now
wins 3212 and the anchor wins
1817; the 95% difference
bound is [2.941811,
3.57005] points and the exact test of
[@eq:mcnemar] returns 3.903e-87, far below the
corrected threshold 0.000625. Both readings now agree, and
why they must is the point.

The exclusion is not a second opinion about the p-value; it is the p-value. A
Clopper–Pearson interval at coverage $1 - \alpha$ is the inversion of the same
exact two-sided procedure the p-value comes from, so it excludes one half
exactly when that procedure rejects at $\alpha$ — the endpoint condition and
the tail condition are one inequality on one binomial tail.
[@eq:paired_difference] then rescales by a strictly positive factor, so the
difference interval excludes zero exactly when the share interval excludes one
half. Composing the two, "the 95% bound excludes zero"
carries exactly one piece of information — that $p$ falls below the level
complementary to that coverage — and carries nothing beyond it. That level is
not one this project ever declared: its weakest declared bar is
0.01 and its operative bar is 0.000625.
Under the defective lookup the two readings disagreed for the ordinary reason
that the arm's p-value then lay between those levels, not because two
instruments disagreed about the arm; corrected, its p-value lies below both,
and the readings agree for the same ordinary reason.

Read at the coverage that matches the decision, as this section's convention
requires, the corrected arm's interval is the one at
99.9375% — one minus 0.000625 —
and that interval excludes one half, agreeing with the p-value. It is also the
interval the recorded equivalence indicator is computed from, and
0 is what it reports: a decided effect is not
a bounded null, and one interval says so in both directions.

What the correction removed from the ledger is the one arm with the precision
to support an equivalence claim. 5029 discordant
pairs clears 1204, the analytic
minimum at that coverage, by a wide margin, but the arm is no longer near the
null; every arm in this project whose corrected-level interval covers the null
sits below 420 and so could not have supported
an equivalence claim under any outcome, the nearest being the Stage 6 sign test
at 187 pairs. An equivalence claim needs a near-null
*and* the pairs to resolve it, and no arm in the ledger now has both.

What is claimed here stops at the share scale. The equivalence region is
declared on shares only, for the reason given above — a difference-scale margin
would depend on the observed discordance rate and so would be easiest to meet
where the evidence is thinnest — so the 95% difference
bound in [@tbl:derived_bounds] stays a bound and is not converted into a claim
of any other kind. Two recomputations from
3212 and
1817 would falsify this passage without any
new run: a corrected-level interval that contained one half, or a recorded
0 of one. The absence of an equivalence arm is
a statement about the ledger as it stands rather than about the method —
another arm accumulating pairs while staying near the null would supply one,
and that would be new evidence rather than a changed convention.

## Statistical choices and rejected alternatives

The apparatus that decides whether a difference in this manuscript is real is
standard, but it is assembled out of four separate literatures, and every choice
in it has a named alternative that was declined rather than overlooked.

The paired test is the exact conditional form of McNemar's test [@mcnemar1947].
Conditioning on the number of discordant pairs is what makes it exact and also
what makes it conservative, and the long argument over conditional versus
unconditional exact procedures for matched pairs is surveyed by
@agresti2001exact; this manuscript takes the conditional side and says where the
resulting interval is conditional rather than marginal
([@sec:external_benchmarks]). Interval estimates appear in two forms, distinguished by the role each plays
rather than by the population each covers: a verdict-bearing form and a
descriptive form. The primary paired family uses Clopper–Pearson
[@clopper1934], whose coverage is never below nominal, because a verdict is
read off it ([@sec:verification]). The benchmark proportions, and the
per-decoder Wilson bars drawn on [@fig:g2p_accuracy] in the main text, use
Wilson [@wilson1927] instead, which is shorter and better centered on average
but whose coverage oscillates; wherever it appears, it is reported as a
description of one group's accuracy rather than as the interval a verdict is
read from.

The assumption that test makes about *independence between pairs* has its own
literature, and this manuscript uses it to bound its results rather than to
repair them. @eliasziw1991mcnemar state the assumption explicitly and derive the
variant that applies when matched pairs arrive in clusters;
@obuchowski1998clustered shows that ignoring positive intracluster correlation
inflates the size of the test, so nominal p-values come out smaller than a
cluster-aware analysis would give. Every paired unit reported here is nested
below what its corpus samples — positions within a word, and tokens or sentences
within a document — and the cluster-robust variant
is computed for 6 of them at
10,000 replicates ([@tbl:cluster]). What these citations add is
that the direction of the error under positive intracluster correlation is a
property of the estimator rather than an intuition about this particular data,
which is what makes the measured result worth reporting: on this corpus the
correlation is near zero at the word unit, positive for the Stage 4 document
arm, and *negative* for one Stage 6 document arm. Thus clustering leaves the
thinnest margin in the family intact, widens Stage 4, and makes one Stage 6 arm
tighter rather than wider. The literature supplies the
direction under an assumption; the reanalysis supplies the sign and size that
this data actually has. @durkalski2003clustered give the practical variant for
unequal cluster sizes, which is the case here — the largest word contributes
19 positions — and @gonen2004sample sets
out why the design question is the number of clusters rather than the number of
observations, which is exactly what disqualifies the sentence-level reading of
Stage 6's log-loss arm and the sentence- or token-level primary readings of the
English Stage 4 and Stage 5 arms.

Correction at a constant threshold is the Bonferroni inequality as applied to
simultaneous inference by @dunn1961multiple. Holm's step-down procedure
[@holm1979] controls the same familywise rate and is uniformly at least as
powerful, and the Benjamini–Hochberg procedure [@benjamini1995controlling] is
more powerful still but controls the false discovery rate, which is a weaker
guarantee. This project reports the first two and not the third. Bonferroni is primary
because its constant per-test threshold is auditable against a single recorded
number; Holm's step-down is reported alongside it as a sensitivity analysis,
computed over the same family by `holm_step_down` (`src/ailm/statistics.py`),
and every Holm crossing is stated wherever a familywise verdict appears.
[@sec:multiplicity] records where the two verdicts differ.

Multiplicity is nonetheless the smaller of the two threats to a confirmatory
reading, and the larger one is a distinct failure with its own name.
@dwork2015adaptive show that repeatedly using a holdout adaptively — inspecting
it, revising a hypothesis on what it showed, testing again — overfits the
holdout itself, and that no correction applied to a fixed family addresses this,
because the family was selected after looking. @cawley2010overfitting establish
the adjacent result for model selection: selecting over a criterion biases the
estimate that criterion reports, independently of any parameter overfitting.
@simmons2011false catalog how researcher degrees of freedom exercised after
seeing preliminary results inflate the false-positive rate even when each
individual decision looks defensible in isolation, and @gelman2013garden name
the same failure a garden of forking paths: no single choice need resemble
p-hacking for the path-dependent multiplicity to be real. This project revised gates after seeing stage outcomes and assembled its test
family retrospectively, on the same population every gate was tuned against,
which [@sec:verification] and [@sec:results] both state in those terms before
reporting a number. The consequence is that the corrected threshold of
[@sec:multiplicity] controls exactly what it claims and nothing else: crossing
it makes an effect a corrected-threshold survivor on the development population,
not a confirmed one. Only a partition nothing in this manuscript had touched could change that
label, and one did, for those of the 4 members it scored that
cleared the frozen bar ([@sec:confirmatory]). No such partition
remains among the vendored artifacts; the one registered outside them for the
CMUdict members was measured empty and that replication is retired by
exhaustion (decision 91), while the EWT members'
registered population is spent by the one run of decision
111; split
diagnostics are explicitly not re-scored, and their full-set member carries
the replication ([@sec:multiplicity], decision
79).

Power is reported the way it is because retrospective power computed at the
*observed* effect is a monotone transform of the p-value and adds nothing to it
[@hoenig2001abuse]. What is informative is detectability against an effect size
fixed in advance — the design-analysis position of @gelman2014beyond — and that
is what the declared reference share of 0.55 implements
here. That comparisons in this field are routinely underpowered, and that their
test sets are often too small to resolve the differences reported on them, is
the finding of @card2020little; @dror2018hitchhiker surveys significance-testing
practice for the same field. The verdict *unresolved rather than absent*, which
recurs throughout [@sec:verification] and [@sec:external_benchmarks], is that
recommendation applied to this project's own nulls.

That verdict is deliberately weaker than the one a reader might want, and the
literature says what the stronger one costs. Establishing that an effect is
absent rather than undetected requires an equivalence test — the two one-sided
tests procedure [@schuirmann1987comparison] against a bound declared before the
data are seen [@lakens2017equivalence]. [@sec:verification] declares such a
bound, on the win-share scale, and reads the decision off the same interval the
p-value is read from. The price it names is precision rather than
reinterpretation: an equivalence claim needs both a near-null and enough
discordant pairs to resolve one, and no near-null in the plotted
paired-bound roster now has both.
Because the bound is declared on shares and not on differences, no null in this
manuscript is a negligibility claim about an accuracy difference.

Calibration has its own lineage. That a forecaster's stated probabilities should
match realized frequencies, and that this is separable from how sharp those
probabilities are, is @degroot1983comparison. The binned estimator that
[@sec:external_benchmarks] reports as expected calibration error is the one
introduced by @naeini2015obtaining and made standard for neural classifiers by
@guo2017calibration, and it is a diagnostic rather than a score: it can be
driven down by flattening a distribution instead of by improving it, which is
exactly the failure mode that makes the supplement withhold a calibration figure
for a system whose calibration numbers move beyond tolerance between the
completion rules. The Brier
and negative-log-likelihood statistics reported beside it are strictly proper in the sense of
@gneiting2007strictly — the multiclass Brier score [@brier1950] and the negative
log likelihood — which is the property that a system cannot improve its number
by reporting something other than what it believes, a property that for an
external probe attaches only to the reconstructed distribution
([@sec:external_benchmarks]).
