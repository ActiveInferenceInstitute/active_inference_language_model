"""Cluster-aware inference for paired comparisons whose observations nest.

Why this module exists
----------------------

``statistics.py`` treats every matched pair as one independent unit. That is
the assumption the exact McNemar test and the exact sign test are built on, and
in this project it is unjustified wherever the scored observations nest inside a
larger sampling unit. The G2P comparison scores phone positions drawn from
held-out *words*: positions inside one word share a lexical item, one filtering
trajectory, and one orthographic context. Stage 4 scores tokens drawn from
sentences. Stage 6 scores sentences drawn from documents, and there are an order
of magnitude fewer documents than sentences.

Conditional on independent matched pairs the exact tests are exactly right.
Under positive intracluster correlation they are anti-conservative: the variance
of the paired mean is understated by roughly the design effect, so the p-value is
too small and the interval too narrow (Eliasziw and Donner 1991; Obuchowski
1998). Under negative intracluster correlation they are conservative instead.

Which of those holds is not deducible from the nesting. What has to be
independent is the *paired difference*, not either system's outcome, and two
decoders can be strongly correlated within a cluster while the positions at
which they disagree are scattered near-independently through it. So this module
does not assume the direction of the correction: it computes the same effect at
the same scale, attaches an interval and a p-value that resample the cluster,
and reports :func:`intracluster_correlation` and :func:`variance_inflation` so
the size and the sign of the correction are measured rather than asserted.

What is computed, and why these two procedures
----------------------------------------------

The estimand is unchanged: the observation-weighted mean paired difference,
``sum(d) / n``. For 0/1 hit vectors that is exactly ``(b - c) / n``, the
quantity McNemar's test is about, so the headline effect a cluster analysis
reports is numerically the effect the position-level analysis reported. Only
the uncertainty changes. Effect sizes survive a cluster reanalysis; small
p-values need not.

- **Cluster bootstrap.** Resample whole clusters with replacement and recompute
  the ratio ``sum of resampled cluster sums / total resampled observations``.
  The percentile interval of that distribution is the reported interval. The
  ratio form keeps the estimand fixed under the unequal cluster sizes this
  project actually has (words differ in length, sentences in token count).
- **Cluster sign-flip randomization test.** Multiply every difference in a
  cluster by a common random sign and recompute the statistic. Its finite-sample
  exactness requires independent cluster sums whose *joint distribution is
  centrally symmetric*: each whole cluster difference vector (equivalently its
  sum for this statistic) must be invariant under sign reversal. That is
  stronger than zero expected cluster sum, and stronger than every position's
  discordance direction being marginally fair. It is therefore not simply
  McNemar's pair-level null "lifted" to clusters. With singleton 0/1 clusters
  the stronger assumption reduces to McNemar and the two p-values agree up to
  Monte Carlo resolution; with multi-observation clusters the randomization
  p-value is reported explicitly as assumption-conditional sensitivity.

A resampling correction is used rather than the closed-form clustered McNemar
adjustment because the closed form needs a variance model for the intracluster
correlation, and the cluster sizes here are neither equal nor small. The price
is Monte Carlo resolution: a randomization p-value from ``R`` draws cannot fall
below ``1 / (R + 1)``, so a p-value at that floor means "below the resolution of
this procedure", not "equal to that number". :attr:`ClusteredPairedSummary.
p_floor` carries the floor so a caller can say so.

The sign-flip reference distribution also has closed-form conditional moments:
given the observed cluster sums, its variance is ``sum(S_k**2)``. The reported
``z = sum(S_k) / sqrt(sum(S_k**2))`` therefore has a normal-tail approximation.
For a weaker mean-zero null that approximation needs independent clusters, a
sufficient cluster count, finite variance, and no dominant cluster; it is not a
finite-sample exact weak-null test. On singleton clusters over a 0/1 hit vector
the same z reduces algebraically to ``(b - c) / sqrt(b + c)``.

The naive analysis is the special case
--------------------------------------

Passing ``clusters=None`` puts every observation in its own cluster, which is
the independence-assuming analysis. It is the same code path, so a comparison
between the two isolates the clustering and nothing else: no second
implementation, no second interval convention. :func:`variance_inflation`
reports the ratio of the empirical variances of the exact bootstrap-replicate
vectors that produced the two intervals.

What this module does not fix
-----------------------------

Clustering is one of the ways this project's inference can be optimistic, not
the only one. A cluster-aware interval is still marginal, not simultaneous —
the multiplicity discipline in ``statistics.py`` applies unchanged — and it says
nothing about holdout reuse: an arm whose gate, comparator, or operationalisation
was revised after seeing the evaluation sample does not become confirmatory
because its interval was widened. With few clusters the percentile bootstrap
under-covers, so a recorded interval from a small cluster count is a *floor* on
the width the data warrant, not a conservative bound.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.stats import norm

from ailm.statistics import DIFFERENCE_CONFIDENCE

#: Resamples for both procedures. Fixed rather than a caller default that drifts
#: between arms: the Monte Carlo p-value floor is ``1 / (RESAMPLES + 1)``, and two
#: arms recorded at different resample counts would carry different floors under
#: label names that look comparable.
DEFAULT_RESAMPLES = 10_000

#: Seed for both procedures. Every recorded resampling number in this project is
#: reproducible from a fixed seed; the recorder replay test re-runs the recorders
#: and requires the ledger to be unchanged, which a wall-clock or entropy seed
#: would break on every run.
DEFAULT_SEED = 20260811

#: Assumption under which the whole-cluster sign-flip p-value is finite-sample
#: exact. Kept as a literal public contract so recorders and prose cannot
#: silently weaken it to marginal fairness or a zero-mean condition.
CLUSTER_SIGN_FLIP_EXACTNESS_ASSUMPTION = (
    "independent cluster sums with a joint distribution invariant under whole-cluster "
    "sign reversal (central symmetry)"
)

#: Resamples generated per block. The resample matrices are ``block x clusters``,
#: and the naive analysis of the G2P arm has one cluster per position, so drawing
#: all resamples at once would allocate gigabytes. Blocking bounds the peak
#: allocation without changing the draw sequence.
_RESAMPLE_BLOCK = 256


@dataclass(frozen=True, slots=True)
class ClusteredPairedSummary:
    """One paired effect with its cluster-aware interval and p-value.

    Attributes:
        n_observations: Scored observations (positions, tokens, sentences).
        n_clusters: Independent units the resampling treats as exchangeable.
        largest_cluster: Observations in the largest cluster. A single huge
            cluster is the case where a cluster analysis matters most, so it is
            reported rather than left to be inferred from the two counts above.
        positive_clusters: Clusters whose total difference favors the probe.
        negative_clusters: Clusters whose total difference favors the
            comparator. Clusters summing to exactly zero are in neither count.
        effect: Observation-weighted mean difference, in the caller's units.
        ci_low: Lower percentile bound of the cluster bootstrap.
        ci_high: Upper percentile bound of the cluster bootstrap.
        bootstrap_variance: Sample variance of the exact bootstrap replicates
            used for ``ci_low`` and ``ci_high``. Retaining the variance from that
            same vector prevents a later design-effect calculation from
            reconstructing a different resampling run or treating percentile
            width as if it were a standard error.
        confidence: Two-sided coverage the interval was computed at.
        p_value: Two-sided cluster sign-flip randomization p-value, exact only
            under :data:`CLUSTER_SIGN_FLIP_EXACTNESS_ASSUMPTION` and otherwise
            an assumption-conditional sensitivity result.
        p_floor: Smallest p-value the randomization procedure can return.
        z_statistic: The same statistic standardised by the exact sign-flip
            null variance, ``sum(S_k) / sqrt(sum(S_k ** 2))``. Zero when no
            cluster carries any difference.
        normal_p_value: Two-sided normal-tail approximation for ``z_statistic``.
            Under a weaker mean-zero null it additionally requires enough
            independent clusters and no dominant cluster; it is not exact.
        resamples: Draws behind both the interval and the p-value.
        seed: Seed both draw sequences derive from.
        data_digest: SHA-256 fingerprint of the ordered finite difference
            vector. It lets :func:`variance_inflation` prove that its two
            summaries came from the same observations, not merely two vectors
            with the same length or mean.
    """

    n_observations: int
    n_clusters: int
    largest_cluster: int
    positive_clusters: int
    negative_clusters: int
    effect: float
    ci_low: float
    ci_high: float
    bootstrap_variance: float
    confidence: float
    p_value: float
    p_floor: float
    z_statistic: float
    normal_p_value: float
    resamples: int
    seed: int
    data_digest: str

    @property
    def interval_width(self) -> float:
        """Width of the reported interval, on the effect's scale."""
        return self.ci_high - self.ci_low

    @property
    def p_at_resolution_floor(self) -> bool:
        """Whether the p-value is the floor rather than a measured value.

        No resample exceeded the observed statistic, so the procedure resolved
        the p-value only as "at most ``p_floor``". Reporting the floor as the
        p-value would assert a precision the draw count does not have.
        """
        return self.p_value <= self.p_floor

    @property
    def interval_excludes_zero(self) -> bool:
        """Whether the interval bounds the effect away from no difference."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _prepare(
    differences: Sequence[float],
    clusters: Sequence[Hashable] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the paired input and reduce it to per-cluster sufficient stats.

    Both procedures depend on the differences only through each cluster's sum
    and size, so the reduction happens once here and the resampling loops touch
    arrays of length ``n_clusters`` instead of ``n_observations``.

    Args:
        differences: Per-observation paired differences, probe minus comparator.
        clusters: Cluster label per observation, or ``None`` for one cluster per
            observation — the independence-assuming analysis.

    Returns:
        ``(values, sizes, sums)``: the differences as an array, and each
        cluster's observation count and total. Clusters are numbered in order of
        first appearance, so the reduction never depends on labels being
        sortable — the G2P cluster key is a word string and Stage 6's is an
        integer index.

    Raises:
        ValueError: If the input is empty, non-finite, or unpaired. An unpaired
            difference vector cannot support the claim in either direction, and
            a non-finite difference would propagate silently into a percentile.
    """
    values = np.asarray(list(differences), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional sequence")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(
            "differences must all be finite; a non-finite difference makes every "
            "resampled statistic non-finite and the percentiles meaningless"
        )

    if clusters is None:
        codes = np.arange(values.size, dtype=np.int64)
    else:
        labels = list(clusters)
        if len(labels) != values.size:
            raise ValueError(
                f"unpaired input: {values.size} differences but {len(labels)} cluster labels"
            )
        index: dict[Hashable, int] = {}
        assigned = np.empty(values.size, dtype=np.int64)
        for position, label in enumerate(labels):
            try:
                assigned[position] = index.setdefault(label, len(index))
            except TypeError as exc:
                raise ValueError(
                    f"cluster label at position {position} must be hashable, got {label!r}"
                ) from exc
        codes = assigned

    n_clusters = int(codes.max()) + 1
    sizes = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    sums = np.bincount(codes, weights=values, minlength=n_clusters)
    return values, sizes, sums


def _bootstrap_ratios(
    sizes: np.ndarray,
    sums: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cluster-bootstrap replicates of the observation-weighted mean.

    Each replicate draws ``n_clusters`` clusters with replacement and forms
    ``sum of drawn cluster sums / sum of drawn cluster sizes``. The denominator
    varies across replicates because cluster sizes vary, and that variation is
    part of the sampling variability of the estimator — holding it fixed at the
    observed ``n`` would understate the interval for exactly the unequal-size
    case this project has.
    """
    n_clusters = sums.size
    replicates = np.empty(resamples, dtype=np.float64)
    filled = 0
    while filled < resamples:
        block = min(_RESAMPLE_BLOCK, resamples - filled)
        draw = rng.integers(0, n_clusters, size=(block, n_clusters))
        replicates[filled : filled + block] = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
        filled += block
    return replicates


def _sign_flip_statistics(
    sums: np.ndarray,
    n_observations: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null replicates of the mean difference under cluster-level sign flips.

    A flip negates a whole cluster at once. Finite-sample exactness requires the
    distribution of each independent whole-cluster vector/sum to be invariant
    under that reversal; marginally fair observations or a zero mean do not
    suffice. The denominator is fixed at ``n_observations`` because a sign flip
    changes no cluster's size.
    """
    n_clusters = sums.size
    statistics = np.empty(resamples, dtype=np.float64)
    filled = 0
    while filled < resamples:
        block = min(_RESAMPLE_BLOCK, resamples - filled)
        signs = np.where(rng.random((block, n_clusters)) < 0.5, -1.0, 1.0)
        statistics[filled : filled + block] = signs @ sums / n_observations
        filled += block
    return statistics


def summarize_clustered_difference(
    differences: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    confidence: float = DIFFERENCE_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> ClusteredPairedSummary:
    """Effect, cluster bootstrap interval, and conditional sign-flip p-value.

    The two procedures draw from generators spawned from one seed sequence, so
    each consumes its own stream: changing the resample count of one does not
    shift the other's draws, and the whole summary is reproducible from ``seed``.

    Args:
        differences: Per-observation paired differences, probe minus comparator,
            in whatever units the caller wants the effect reported in. Scale
            them before calling (0/1 hits times 100 gives percentage points) so
            the effect is directly comparable to the recorded margin.
        clusters: Cluster label per observation. ``None`` puts every observation
            in its own cluster, which is the independence-assuming analysis.
        confidence: Two-sided coverage of the percentile interval.
        resamples: Draws for each procedure.
        seed: Seed for both draw sequences.

    Returns:
        The summary, with the effect on the caller's scale. The randomization
        p-value is finite-sample exact only under
        :data:`CLUSTER_SIGN_FLIP_EXACTNESS_ASSUMPTION`.

    Raises:
        ValueError: If the input is unpaired, empty, or non-finite, or if fewer
            than two ``resamples`` are requested, or ``confidence`` is not in
            ``(0, 1)``.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 2:
        raise ValueError(f"resamples must be an integer of at least two, got {resamples!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError(f"confidence must be a finite number in (0, 1), got {confidence!r}")
    coverage = float(confidence)
    if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    values, sizes, sums = _prepare(differences, clusters)
    if sums.size < 2:
        raise ValueError(
            "cluster-aware inference requires at least two independent clusters; "
            "one cluster cannot identify sampling uncertainty"
        )
    n_observations = int(values.size)
    effect = float(values.sum() / n_observations)
    digest_values = values.astype("<f8", copy=True)
    digest_values[digest_values == 0.0] = 0.0  # canonicalize negative zero
    data_digest = hashlib.sha256(digest_values.tobytes(order="C")).hexdigest()

    bootstrap_rng, sign_rng = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
    )

    replicates = _bootstrap_ratios(sizes, sums, resamples, bootstrap_rng)
    tail = (1.0 - coverage) / 2.0
    ci_low = float(np.percentile(replicates, 100.0 * tail))
    ci_high = float(np.percentile(replicates, 100.0 * (1.0 - tail)))
    bootstrap_variance = float(np.var(replicates, ddof=1))

    null = _sign_flip_statistics(sums, n_observations, resamples, sign_rng)
    # Ties count as exceedances. Without that, a difference vector of all zeros
    # — Stage 4's gold population is exactly that — would return 1/(R+1) and
    # read as a significant result computed from no discordance at all. The
    # tolerance absorbs the float error between the observed statistic and the
    # replicate that happens to draw all-positive signs; it is scaled to the
    # total absolute mass so it means the same thing in points and in nats.
    observed = abs(effect)
    tolerance = 1e-12 * float(np.abs(sums).sum()) / n_observations
    exceedances = int(np.count_nonzero(np.abs(null) >= observed - tolerance))
    p_value = (1.0 + exceedances) / (resamples + 1.0)

    # Closed-form standardisation of the sign-flip reference distribution: the
    # artificial flips are independent and centered, so conditional variance is
    # sum S_k ** 2. This does not establish weak-null finite-sample exactness. A cluster set
    # whose sums are all zero has no null variation at all, and dividing by that
    # zero would manufacture an infinite z out of no evidence.
    dispersion = float(np.sum(sums**2))
    z_statistic = float(sums.sum() / np.sqrt(dispersion)) if dispersion > 0.0 else 0.0
    normal_p_value = float(2.0 * norm.sf(abs(z_statistic))) if dispersion > 0.0 else 1.0

    return ClusteredPairedSummary(
        n_observations=n_observations,
        n_clusters=int(sums.size),
        largest_cluster=int(sizes.max()),
        positive_clusters=int(np.count_nonzero(sums > 0.0)),
        negative_clusters=int(np.count_nonzero(sums < 0.0)),
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_variance=bootstrap_variance,
        confidence=coverage,
        p_value=p_value,
        p_floor=1.0 / (resamples + 1.0),
        z_statistic=z_statistic,
        normal_p_value=normal_p_value,
        resamples=resamples,
        seed=seed,
        data_digest=data_digest,
    )


def variance_inflation(
    clustered: ClusteredPairedSummary,
    independent: ClusteredPairedSummary,
) -> float:
    """Factor by which the independence assumption understates the variance.

    This is the design effect: the empirical variance of the cluster-bootstrap
    replicates divided by the empirical variance of the independence-assuming
    bootstrap replicates. Both variances come from the exact replicate vectors
    used for their summaries' percentile intervals. A value near one says the
    clustering carried no dependence; a large value says the naive analysis was
    reading the same information many times.

    Args:
        clustered: Summary computed with the real cluster labels.
        independent: Summary computed with ``clusters=None`` over the same
            differences, same confidence, same resamples.

    Returns:
        The ratio of the two bootstrap-replicate variances.

    Raises:
        ValueError: If the two summaries are not comparable, or the independent
            bootstrap variance is zero. A zero-variance naive bootstrap means a
            ratio against it would be an artifact of division rather than a
            measurement of dependence.
    """
    if clustered.n_observations != independent.n_observations:
        raise ValueError(
            f"summaries describe different data: {clustered.n_observations} vs "
            f"{independent.n_observations} observations"
        )
    if clustered.data_digest != independent.data_digest:
        raise ValueError(
            "summaries describe different ordered difference vectors; their "
            "bootstrap variances are not comparable"
        )
    if independent.n_clusters != independent.n_observations or independent.largest_cluster != 1:
        raise ValueError(
            "the reference summary is not independence-assuming: it must have "
            "one singleton cluster per observation"
        )
    if clustered.confidence != independent.confidence:
        raise ValueError(
            f"summaries use different coverage: {clustered.confidence} vs "
            f"{independent.confidence}; their bootstrap summaries are not comparable"
        )
    if clustered.resamples != independent.resamples:
        raise ValueError(
            f"summaries use different resample counts: {clustered.resamples} vs "
            f"{independent.resamples}"
        )
    if clustered.seed != independent.seed:
        raise ValueError(f"summaries use different seeds: {clustered.seed} vs {independent.seed}")
    if not math.isfinite(clustered.bootstrap_variance) or clustered.bootstrap_variance < 0.0:
        raise ValueError(
            f"the cluster-aware bootstrap variance is invalid: {clustered.bootstrap_variance!r}"
        )
    if not math.isfinite(independent.bootstrap_variance) or independent.bootstrap_variance <= 0.0:
        raise ValueError(
            "the independence-assuming bootstrap has zero or invalid variance, so "
            "no inflation factor is defined; report the two intervals instead"
        )
    return clustered.bootstrap_variance / independent.bootstrap_variance


def intracluster_correlation(
    differences: Sequence[float],
    clusters: Sequence[Hashable],
) -> float | None:
    """One-way ANOVA estimate of the intracluster correlation of the differences.

    This is the mechanism behind the correction, reported directly: it is the
    share of the variance of the paired difference that sits between clusters
    rather than within them. The estimator handles unequal cluster sizes through
    the usual ``n_0`` adjustment, and it can go negative when clusters are less
    alike than chance, which is reported rather than clipped to zero.

    Args:
        differences: Per-observation paired differences.
        clusters: Cluster label per observation.

    Returns:
        The estimate, or ``None`` when it is not estimable: every cluster a
        singleton leaves no within-cluster variance to compare against, and a
        constant difference vector leaves no variance at all. ``None`` means the
        data cannot say, which is a different statement from an estimate of zero
        and must not be recorded as one.

    Raises:
        ValueError: If the input is unpaired, empty, or non-finite.
    """
    values, sizes, sums = _prepare(differences, clusters)
    n_observations = float(values.size)
    n_clusters = float(sums.size)
    if n_clusters < 2.0 or n_observations <= n_clusters:
        return None

    cluster_means = sums / sizes
    grand_mean = float(values.sum()) / n_observations
    between = float(np.sum(sizes * (cluster_means - grand_mean) ** 2)) / (n_clusters - 1.0)

    # Pooled within-cluster sum of squares, from the sufficient statistics alone:
    # the total sum of squares minus each cluster's own correction term
    # ``sum_k ** 2 / size_k``. Recovering the per-observation grouping to compute
    # it directly would give the same number at more cost.
    total_squares = float(np.sum(values**2))
    within = (total_squares - float(np.sum(sums**2 / sizes))) / (n_observations - n_clusters)

    adjusted_size = (n_observations - float(np.sum(sizes**2)) / n_observations) / (n_clusters - 1.0)
    denominator = between + (adjusted_size - 1.0) * within
    if denominator == 0.0:
        return None
    return float((between - within) / denominator)


#: Weight draw per replicate for the wild cluster bootstrap. Rademacher: each
#: cluster's perturbation is multiplied by ``+1`` or ``-1`` with equal
#: probability, independently across clusters and replicates. This is the
#: degenerate two-point member of the wild-bootstrap weight family of
#: Cameron, Gelbach and Miller (2008, Section 4); under it the replicate
#: variance of the mean equals the cluster-robust (CR0-style) variance
#: exactly, which is what the positive control checks against.
WILD_CLUSTER_WEIGHTS = "rademacher"


@dataclass(frozen=True, slots=True)
class WildClusterSummary:
    """The wild cluster bootstrap interval beside the effect it perturbs.

    The procedure resamples the *demeaned* cluster contributions: each
    replicate forms ``effect + sum_k w_k (S_k - effect * n_k) / n`` where
    ``S_k`` and ``n_k`` are a cluster's difference sum and size and ``w_k``
    is its Rademacher weight. The demeaning is what makes the perturbation
    orthogonal to the point estimate: the expected replicate is the observed
    effect, and with Rademacher weights the replicate variance is exactly the
    cluster-robust variance of the observation-weighted mean,

    ``sum_k (S_k - effect * n_k) ** 2 / n ** 2``,

    so this is a sensitivity interval on the same estimand, not a second
    estimator. It is recorded beside the cluster-bootstrap interval, never in
    place of it: with few clusters the Rademacher-t rademacher percentile
    interval inherits the same under-coverage the percentile bootstrap does,
    and the two intervals bracketing the effect in different directions is a
    finding, not a bug to be averaged away.

    Attributes:
        n_observations: Scored observations, as in the cluster summary.
        n_clusters: Independent units whose demeaned sums are perturbed.
        effect: Observation-weighted mean difference (unchanged by the
            procedure; carried so the summary is self-contained).
        ci_low: Lower percentile bound of the wild replicates.
        ci_high: Upper percentile bound of the wild replicates.
        replicate_variance: Sample variance of the wild replicates. With
            Rademacher weights this equals the CR0 cluster-robust variance of
            the mean up to Monte Carlo error, which the positive control
            uses as its ground truth.
        confidence: Two-sided coverage the interval was computed at.
        resamples: Wild replicates behind the interval.
        seed: Seed the weight draws derive from.
        data_digest: SHA-256 fingerprint of the ordered finite difference
            vector, matching :attr:`ClusteredPairedSummary.data_digest` so a
            caller can prove the two summaries describe the same data.
    """

    n_observations: int
    n_clusters: int
    effect: float
    ci_low: float
    ci_high: float
    replicate_variance: float
    confidence: float
    resamples: int
    seed: int
    data_digest: str

    @property
    def interval_width(self) -> float:
        """Width of the wild interval, on the effect's scale."""
        return self.ci_high - self.ci_low

    @property
    def interval_excludes_zero(self) -> bool:
        """Whether the wild interval bounds the effect away from no difference."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def verdict_agrees_with(self) -> str:
        """What a verdict comparison is against, stated so it cannot be misread.

        The recorded comparison is against the *cluster bootstrap* verdict,
        because that is the arm whose uncertainty this sensitivity interval
        sits beside. The naive verdict is a different comparison and is not
        what ``wild_*_verdict_agrees`` labels assert.
        """
        return "cluster bootstrap"


def _wild_replicates(
    sizes: np.ndarray,
    sums: np.ndarray,
    effect: float,
    n_observations: int,
    resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Wild replicates of the mean from Rademacher-perturbed demeaned sums.

    Each replicate draws one weight per cluster and re-forms the
    observation-weighted mean of the perturbed cluster contributions. The
    denominator is the fixed observed ``n`` because a weight changes no
    cluster's size.
    """
    n_clusters = sums.size
    demeaned = (sums - effect * sizes) / n_observations
    replicates = np.empty(resamples, dtype=np.float64)
    filled = 0
    while filled < resamples:
        block = min(_RESAMPLE_BLOCK, resamples - filled)
        signs = np.where(rng.random((block, n_clusters)) < 0.5, -1.0, 1.0)
        replicates[filled : filled + block] = effect + signs @ demeaned
        filled += block
    return replicates


def summarize_wild_cluster_difference(
    differences: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    *,
    confidence: float = DIFFERENCE_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> WildClusterSummary:
    """Wild cluster bootstrap interval (Rademacher weights) for a paired mean.

    The sensitivity arm of the clustered reanalysis: it perturbs each
    cluster's demeaned contribution by a common random sign and reads the
    percentile interval of the resulting replicate distribution. Argument
    validation, the estimand, the seed discipline, and the digest are the
    same machinery :func:`summarize_clustered_difference` uses, so the two
    summaries of one arm are comparable by construction.

    Args:
        differences: Per-observation paired differences, probe minus
            comparator, in the caller's units.
        clusters: Cluster label per observation. ``None`` puts every
            observation in its own cluster, the independence-assuming case —
            which with Rademacher weights is exactly the paired-difference
            analogue the positive control checks.
        confidence: Two-sided coverage of the percentile interval.
        resamples: Wild replicates.
        seed: Seed for the weight draws.

    Returns:
        The summary. The estimand is unchanged; only the interval is new.

    Raises:
        ValueError: If the input is unpaired, empty, or non-finite, or if
            fewer than two ``resamples`` are requested, or ``confidence`` is
            not in ``(0, 1)``, or fewer than two independent clusters remain.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 2:
        raise ValueError(f"resamples must be an integer of at least two, got {resamples!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise ValueError(f"confidence must be a finite number in (0, 1), got {confidence!r}")
    coverage = float(confidence)
    if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    values, sizes, sums = _prepare(differences, clusters)
    if sums.size < 2:
        raise ValueError(
            "cluster-aware inference requires at least two independent clusters; "
            "one cluster cannot identify sampling uncertainty"
        )
    n_observations = int(values.size)
    effect = float(values.sum() / n_observations)
    digest_values = values.astype("<f8", copy=True)
    digest_values[digest_values == 0.0] = 0.0  # canonicalize negative zero
    data_digest = hashlib.sha256(digest_values.tobytes(order="C")).hexdigest()

    (wild_rng,) = np.random.SeedSequence(seed).spawn(1)
    replicates = _wild_replicates(
        sizes, sums, effect, n_observations, resamples, np.random.default_rng(wild_rng)
    )
    tail = (1.0 - coverage) / 2.0
    ci_low = float(np.percentile(replicates, 100.0 * tail))
    ci_high = float(np.percentile(replicates, 100.0 * (1.0 - tail)))
    return WildClusterSummary(
        n_observations=n_observations,
        n_clusters=int(sums.size),
        effect=effect,
        ci_low=ci_low,
        ci_high=ci_high,
        replicate_variance=float(np.var(replicates, ddof=1)),
        confidence=coverage,
        resamples=resamples,
        seed=seed,
        data_digest=data_digest,
    )


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "WILD_CLUSTER_WEIGHTS",
    "ClusteredPairedSummary",
    "intracluster_correlation",
    "summarize_clustered_difference",
    "summarize_wild_cluster_difference",
    "variance_inflation",
]
