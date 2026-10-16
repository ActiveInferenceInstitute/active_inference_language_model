# Power Analysis and Confidence Intervals

**Power analysis (W2-2).** Corpus size determines the available sample. The
registered stage/intervention significance family contains claim:stat_family_size
members at familywise claim:stat_familywise_alpha, giving the fixed Bonferroni
per-test threshold claim:stat_bonferroni_alpha. These values come from the
ledger; this document does not freeze a second copy of the family arithmetic.

Power below is evaluated at the fixed discordant-pair share
claim:stat_reference_share, not at each observed share. Power at an observed
share is a monotone restatement of the p-value and adds no evidence. The exact
two-sided rejection region is enumerated under the null and its binomial mass is
summed under the fixed reference alternative.

| Test | Discordant pairs *n* | Observed favored share | Power at the fixed reference share and corrected threshold |
|---|---:|---:|---:|
| Stage 2B position-aligned letter-to-phone | claim:stat_stage2_discordant_n | claim:stat_stage2_favored_ratio | claim:stat_stage2_power_at_reference |
| Stage 4 derived agreement | claim:stat_stage4_discordant_n | claim:stat_stage4_favored_ratio | claim:stat_stage4_power_at_reference |
| L3→L2B lexical channel | claim:stat_lexical_discordant_n | claim:stat_lexical_favored_ratio | claim:stat_lexical_power_at_reference |
| L1→L2B coupling harm | claim:stat_coupling_harm_discordant_n | claim:stat_coupling_harm_favored_ratio | claim:stat_coupling_harm_power_at_reference |

The last column is a design-sensitivity statement at a common reference effect,
not an endorsement of the observed result and not a prospective sample-size
justification.

**Confidence intervals (W2-3).** The paired-effect shares use exact
Clopper–Pearson intervals at the repository's recorded marginal coverage:

| Test | Favored share | Exact interval |
|---|---:|---:|
| Stage 2B position-aligned letter-to-phone | claim:stat_stage2_favored_ratio | [claim:stat_stage2_share_ci_low, claim:stat_stage2_share_ci_high] |
| Stage 4 derived agreement | claim:stat_stage4_favored_ratio | [claim:stat_stage4_share_ci_low, claim:stat_stage4_share_ci_high] |
| L3→L2B lexical channel | claim:stat_lexical_favored_ratio | [claim:stat_lexical_share_ci_low, claim:stat_lexical_share_ci_high] |
| L1→L2B coupling harm | claim:stat_coupling_harm_favored_ratio | [claim:stat_coupling_harm_share_ci_low, claim:stat_coupling_harm_share_ci_high] |

The intervals are marginal, not simultaneous. Corrected-threshold verdicts use
the separately recorded Bonferroni-level intervals and the current family
threshold. `scripts/record_statistical_claims.py` owns both calculations; the
manuscript's multiplicity supplement gives the full family and assumptions.

The [connected GUM evaluation](connected_runtime.md#recorded-connected-evaluation)
has its own frozen equal-document paired bootstrap estimand. Its descriptive
interval is not a member of this historical discordant-pair family, and these
power calculations do not establish sensitivity for the connected comparison.

**Cluster-aware uncertainty.** For observations nested in words, sentences, or
documents, `ailm.clustering.summarize_clustered_difference` resamples the outer
unit and retains the sample variance of the same bootstrap-replicate vector used
to form its percentile interval. The corresponding independence-assuming
summary uses singleton clusters. `variance_inflation` reports the first empirical
replicate variance divided by the second, after verifying identical ordered data,
confidence, resample count, and seed. It is therefore an actual bootstrap
variance ratio—not the squared ratio of percentile-interval widths, which is not
a variance estimator when a bootstrap distribution is skewed. The recorder and
its planted unequal-cluster control enforce this distinction.
