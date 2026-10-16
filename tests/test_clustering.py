"""Contract for cluster-aware paired inference.

What this file has to establish
-------------------------------

``src/ailm/clustering.py`` exists because a correction that never changes any
number is indistinguishable from no correction. So the controls come in pairs:

- **Positive control** — paired data built with strong intracluster correlation,
  where the cluster-aware interval MUST be materially wider than the
  independence-assuming one and the p-value materially larger. A bug that
  ignored the cluster labels would pass every other assertion in this file.
- **Negative control** — independent data with cluster labels that carry no
  dependence, where the two analyzes MUST agree closely. Without it, a procedure
  that simply inflated every interval would look like a working correction.

Beyond the pair: the sign-flip test is checked against exact McNemar only in its
singleton-cluster special case. A separate asymmetric mean-zero construction
proves why that identity cannot be generalized to multi-observation clusters
without the stronger joint central-symmetry assumption.

All data here is constructed in-file from a seeded generator. These are
properties of the estimator, not measurements of the corpus, so a fixture that
depended on vendored artifacts would make the contract hostage to a corpus
regeneration.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

from ailm.clustering import (
    CLUSTER_SIGN_FLIP_EXACTNESS_ASSUMPTION,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    ClusteredPairedSummary,
    intracluster_correlation,
    summarize_clustered_difference,
    summarize_wild_cluster_difference,
    variance_inflation,
)

#: Resamples for the tests. Below the recorder's DEFAULT_RESAMPLES so the suite
#: stays fast, high enough that the Monte Carlo error on a p-value near 0.05 is
#: about half a percentage point.
TEST_RESAMPLES = 2_000


def _clustered_differences(
    *,
    n_clusters: int,
    cluster_size: int,
    cluster_sd: float,
    noise_sd: float,
    shift: float,
    seed: int,
) -> tuple[list[float], list[int]]:
    """Paired differences with a tunable amount of between-cluster variance.

    Every observation in a cluster shares one cluster-level draw, so
    ``cluster_sd`` controls the intracluster correlation directly: at
    ``cluster_sd = 0`` the observations are independent given the shift, and as
    it grows relative to ``noise_sd`` the cluster becomes a single repeated
    measurement wearing ``cluster_size`` disguises.
    """
    rng = np.random.default_rng(seed)
    values: list[float] = []
    labels: list[int] = []
    for cluster in range(n_clusters):
        level = shift + cluster_sd * float(rng.normal())
        for _ in range(cluster_size):
            values.append(level + noise_sd * float(rng.normal()))
            labels.append(cluster)
    return values, labels


# ── The positive control: clustering must change the answer ──────────────────


def test_strong_intracluster_correlation_widens_the_interval():
    """POSITIVE CONTROL. Dependence inside clusters must widen the interval.

    Each cluster here is nearly one repeated observation, so the honest sample
    size is the cluster count, not the observation count. A procedure that
    ignored the labels would report the narrow interval and pass every
    assertion that does not compare the two.
    """
    values, labels = _clustered_differences(
        n_clusters=40,
        cluster_size=50,
        cluster_sd=1.0,
        noise_sd=0.05,
        shift=0.3,
        seed=11,
    )
    clustered = summarize_clustered_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    naive = summarize_clustered_difference(
        values, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )

    assert clustered.n_clusters == 40
    assert naive.n_clusters == 2000
    assert clustered.interval_width > 3.0 * naive.interval_width, (
        f"cluster-aware width {clustered.interval_width:.4f} is not materially "
        f"wider than the naive {naive.interval_width:.4f}; the labels are being "
        f"ignored"
    )
    assert variance_inflation(clustered, naive) > 9.0


def test_strong_intracluster_correlation_raises_the_p_value():
    """POSITIVE CONTROL, second half: the p-value must move the same way.

    A wider interval with an unchanged p-value would mean the two procedures
    disagree about the same data, which is the failure mode that matters most:
    the manuscript reads significance off the p-value.
    """
    values, labels = _clustered_differences(
        n_clusters=20,
        cluster_size=60,
        cluster_sd=1.0,
        noise_sd=0.05,
        shift=0.25,
        seed=3,
    )
    clustered = summarize_clustered_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    naive = summarize_clustered_difference(
        values, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )

    assert naive.p_at_resolution_floor, (
        "the naive analysis was expected to be far below its resolution floor on "
        "this construction; the fixture no longer exercises the contrast"
    )
    assert clustered.p_value > 20.0 * naive.p_value


def test_an_effect_that_lives_entirely_between_clusters_loses_significance():
    """The end-to-end version of the review's charge, in miniature.

    Ten clusters of two hundred near-identical observations, seven leaning one
    way and three the other. Read at the observation level the evidence is
    overwhelming: two thousand observations and a mean four tenths from zero.
    Read at the cluster level it is a coin landing seven times out of ten, and
    the sign-flip null puts about a third of its mass at or beyond that.

    The levels are written out rather than drawn so the cluster-level verdict is
    a property of the construction and not of a lucky seed.
    """
    levels = [1.0] * 7 + [-1.0] * 3
    rng = np.random.default_rng(7)
    values = [level + 0.01 * float(rng.normal()) for level in levels for _ in range(200)]
    labels = [cluster for cluster in range(len(levels)) for _ in range(200)]
    clustered = summarize_clustered_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    naive = summarize_clustered_difference(
        values, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )

    assert naive.interval_excludes_zero
    assert not clustered.interval_excludes_zero
    assert clustered.p_value > 0.05
    # The effect itself is untouched by the reanalysis: that is the point of
    # keeping the estimand fixed at the observation-weighted mean.
    assert clustered.effect == pytest.approx(naive.effect, abs=1e-12)


# ── The negative control: clustering must NOT change an independent answer ────


def test_independent_data_agrees_with_the_naive_analysis():
    """NEGATIVE CONTROL. With no dependence the two analyzes must agree.

    Cluster labels are still present and still resampled, so this is not the
    ``clusters=None`` path in disguise. A correction that widened every interval
    unconditionally would fail here, and a procedure that fails here is
    inflating uncertainty rather than measuring it.
    """
    values, labels = _clustered_differences(
        n_clusters=200,
        cluster_size=10,
        cluster_sd=0.0,
        noise_sd=1.0,
        shift=0.1,
        seed=5,
    )
    clustered = summarize_clustered_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    naive = summarize_clustered_difference(
        values, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )

    width_ratio = clustered.interval_width / naive.interval_width
    assert 0.85 < width_ratio < 1.15, (
        f"widths differ by {width_ratio:.3f}x on data with no intracluster "
        f"correlation; the procedure is not measuring dependence"
    )
    assert variance_inflation(clustered, naive) == pytest.approx(
        clustered.bootstrap_variance / naive.bootstrap_variance, rel=1e-12
    )


def test_skewed_unequal_clusters_reject_the_squared_interval_width_proxy():
    """NEGATIVE CONTROL for the former design-effect implementation.

    Percentile width is not a standard error on a skewed bootstrap distribution.
    One large cluster and several tiny extreme clusters make the distinction
    material and deterministic: the old squared-width proxy cannot satisfy the
    actual replicate-variance contract.
    """
    sizes = [100, 1, 1, 1, 1, 1]
    levels = [0.0, 5.0, 5.0, 5.0, 5.0, -5.0]
    values = [level for size, level in zip(sizes, levels, strict=True) for _ in range(size)]
    labels = [cluster for cluster, size in enumerate(sizes) for _ in range(size)]
    clustered = summarize_clustered_difference(values, labels, resamples=10_000, seed=DEFAULT_SEED)
    naive = summarize_clustered_difference(values, None, resamples=10_000, seed=DEFAULT_SEED)

    actual = variance_inflation(clustered, naive)
    squared_width_proxy = (clustered.interval_width / naive.interval_width) ** 2
    assert actual == pytest.approx(
        clustered.bootstrap_variance / naive.bootstrap_variance, rel=1e-12
    )
    assert abs(actual - squared_width_proxy) / actual > 0.2, (
        "the planted skew no longer distinguishes empirical replicate variance "
        "from squared percentile-interval width"
    )


def test_singleton_labels_reproduce_the_none_path_exactly():
    """Explicit one-observation-per-cluster labels are the naive analysis.

    ``clusters=None`` is documented as that special case. If the two paths
    diverged, every comparison in this file would be against a different
    procedure rather than against the same one with different labels.
    """
    values, _ = _clustered_differences(
        n_clusters=30, cluster_size=4, cluster_sd=0.5, noise_sd=1.0, shift=0.2, seed=9
    )
    labeled = summarize_clustered_difference(
        values, list(range(len(values))), resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    unlabeled = summarize_clustered_difference(
        values, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    assert labeled == unlabeled


# ── The tie to the test the manuscript already reports ───────────────────────


def test_singleton_sign_flip_reproduces_the_exact_mcnemar_p_value():
    """On 0/1 hits with singleton clusters the randomization test is McNemar.

    Flipping the sign of a zero does nothing, so the randomization distribution
    is carried entirely by the discordant pairs — which is exactly the exact
    binomial McNemar construction. This is what licenses reading a cluster
    p-value beside the recorded exact one only for singleton clusters.
    """
    rng = np.random.default_rng(23)
    model = rng.random(4000) < 0.62
    baseline = rng.random(4000) < 0.58
    differences = [float(m) - float(s) for m, s in zip(model, baseline, strict=True)]
    favored = sum(1 for d in differences if d > 0)
    opposed = sum(1 for d in differences if d < 0)

    exact = binomtest(favored, favored + opposed, 0.5).pvalue
    summary = summarize_clustered_difference(differences, None, resamples=20_000, seed=DEFAULT_SEED)
    # Monte Carlo error on a p-value near ``exact`` from R draws is about
    # sqrt(p(1-p)/R); four of those plus the resolution floor is a bound the
    # procedure must meet, and is tight enough to fail a wrong statistic.
    slack = 4.0 * (exact * (1.0 - exact) / 20_000) ** 0.5 + summary.p_floor
    assert summary.p_value == pytest.approx(exact, abs=slack)


def _exact_conditional_sign_flip_p(cluster_sums: list[float]) -> float:
    observed = abs(sum(cluster_sums))
    reference = [
        abs(sum(sign * value for sign, value in zip(signs, cluster_sums, strict=True)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(cluster_sums))
    ]
    return sum(value >= observed for value in reference) / len(reference)


def test_mean_zero_marginal_fairness_does_not_make_cluster_sign_flips_exact():
    """NEGATIVE CONTROL: weak-null validity requires more than fair coordinates.

    Each cluster has three paired differences. With probability 1/4 all three
    are +1; otherwise one uniformly chosen coordinate is +1 and the other two
    are -1. Every coordinate is marginally a fair sign and the cluster sum has
    expectation zero, but the sum is +3 with probability 1/4 and -1 with
    probability 3/4, so its distribution is not centrally symmetric. Under six
    independent clusters the nominal 5% conditional sign-flip test rejects far
    too often. This is an exact enumeration over the number of +3 clusters,
    not a Monte Carlo anecdote.
    """

    clusters = 6
    rejection_rate = 0.0
    for positive_three in range(clusters + 1):
        sums = [3.0] * positive_three + [-1.0] * (clusters - positive_three)
        probability = (
            math.comb(clusters, positive_three)
            * 0.25**positive_three
            * 0.75 ** (clusters - positive_three)
        )
        if _exact_conditional_sign_flip_p(sums) <= 0.05:
            rejection_rate += probability

    assert rejection_rate == pytest.approx(0.17822265625, abs=1e-15)
    assert rejection_rate > 3.0 * 0.05
    assert "central symmetry" in CLUSTER_SIGN_FLIP_EXACTNESS_ASSUMPTION
    assert "independent cluster sums" in CLUSTER_SIGN_FLIP_EXACTNESS_ASSUMPTION


def test_cluster_sign_flip_documentation_carries_the_strong_assumption() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/ailm/clustering.py").read_text(
        encoding="utf-8"
    )
    assert "stronger than zero expected cluster sum" in source
    assert "not simply\n  McNemar's pair-level null" in source

    # NEGATIVE CONTROL: the former weak-null equivalence must not satisfy this guard.
    planted = source.replace(
        "stronger than zero expected cluster sum",
        "equivalent to zero expected cluster sum",
    )
    assert "stronger than zero expected cluster sum" not in planted


def test_the_singleton_z_statistic_is_the_mcnemar_statistic():
    """``z`` must reduce to ``(b - c) / sqrt(b + c)`` on singleton 0/1 pairs.

    This is the algebraic identity that lets the closed-form p-value be read as
    the same test at a different unit rather than as a new one. It is checked
    exactly, not approximately: any other standardisation would fail here.
    """
    rng = np.random.default_rng(31)
    model = rng.random(3000) < 0.55
    baseline = rng.random(3000) < 0.50
    differences = [float(m) - float(s) for m, s in zip(model, baseline, strict=True)]
    favored = sum(1 for d in differences if d > 0)
    opposed = sum(1 for d in differences if d < 0)

    summary = summarize_clustered_difference(differences, None, resamples=64)
    expected = (favored - opposed) / (favored + opposed) ** 0.5
    assert summary.z_statistic == pytest.approx(expected, rel=1e-12)


def test_the_closed_form_p_value_resolves_below_the_monte_carlo_floor():
    """NON-VACUITY for the closed form: it must say something the draws cannot.

    The whole reason for carrying a second p-value is that the project's
    strongest paired effects sit orders of magnitude below any affordable
    resolution floor. If the closed form merely agreed with the floor, it would
    be redundant machinery.
    """
    rng = np.random.default_rng(41)
    model = rng.random(6000) < 0.70
    baseline = rng.random(6000) < 0.60
    differences = [float(m) - float(s) for m, s in zip(model, baseline, strict=True)]

    summary = summarize_clustered_difference(differences, None, resamples=2_000)
    assert summary.p_at_resolution_floor
    assert summary.normal_p_value < 1e-8


def test_clustering_moves_the_closed_form_p_value_too():
    """POSITIVE CONTROL for the closed form under real dependence."""
    values, labels = _clustered_differences(
        n_clusters=30, cluster_size=60, cluster_sd=1.0, noise_sd=0.05, shift=0.3, seed=17
    )
    clustered = summarize_clustered_difference(values, labels, resamples=256)
    naive = summarize_clustered_difference(values, None, resamples=256)
    assert abs(clustered.z_statistic) < 0.3 * abs(naive.z_statistic)
    assert clustered.normal_p_value > 1000.0 * naive.normal_p_value


def test_all_zero_differences_return_a_p_value_of_one():
    """No discordance anywhere is not evidence of anything.

    Stage 4's gold population is literally this vector. Counting ties as
    exceedances is what keeps the procedure from returning its resolution floor
    — a number that would read as overwhelming significance — for data in which
    the two systems never once disagreed.
    """
    summary = summarize_clustered_difference(
        [0.0] * 500, [i // 10 for i in range(500)], resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    assert summary.effect == 0.0
    assert summary.p_value == 1.0
    assert summary.z_statistic == 0.0 and summary.normal_p_value == 1.0
    assert summary.ci_low == 0.0 and summary.ci_high == 0.0
    assert summary.positive_clusters == 0 and summary.negative_clusters == 0
    assert not summary.interval_excludes_zero


# ── Determinism, because the recorder replay depends on it ───────────────────


def test_the_same_seed_gives_the_identical_summary():
    """Bit-identical, not approximately equal: the ledger stores the digits."""
    values, labels = _clustered_differences(
        n_clusters=25, cluster_size=8, cluster_sd=0.6, noise_sd=0.9, shift=0.15, seed=2
    )
    first = summarize_clustered_difference(values, labels, resamples=TEST_RESAMPLES, seed=4242)
    second = summarize_clustered_difference(values, labels, resamples=TEST_RESAMPLES, seed=4242)
    assert first == second


def test_a_different_seed_moves_the_resampled_quantities_only():
    """NON-VACUITY for the determinism check above.

    Two runs agreeing under a fixed seed proves nothing if the procedure ignores
    the seed entirely. The effect and the counts are seed-free by construction;
    the interval is not.
    """
    values, labels = _clustered_differences(
        n_clusters=25, cluster_size=8, cluster_sd=0.6, noise_sd=0.9, shift=0.15, seed=2
    )
    first = summarize_clustered_difference(values, labels, resamples=TEST_RESAMPLES, seed=1)
    second = summarize_clustered_difference(values, labels, resamples=TEST_RESAMPLES, seed=2)

    assert first.effect == second.effect
    assert first.n_clusters == second.n_clusters
    assert first.positive_clusters == second.positive_clusters
    assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high)


def test_block_boundaries_do_not_change_the_result():
    """The blocking in the resampling loops is an allocation detail.

    Resample counts either side of the internal block size must not change what
    the procedure means, only how much memory it touches at once.
    """
    values, labels = _clustered_differences(
        n_clusters=15, cluster_size=6, cluster_sd=0.4, noise_sd=1.0, shift=0.2, seed=8
    )
    small = summarize_clustered_difference(values, labels, resamples=300, seed=77)
    large = summarize_clustered_difference(values, labels, resamples=3000, seed=77)
    assert small.effect == large.effect
    assert small.ci_low == pytest.approx(large.ci_low, abs=0.15 * large.interval_width)
    assert small.ci_high == pytest.approx(large.ci_high, abs=0.15 * large.interval_width)


# ── The reported diagnostics ─────────────────────────────────────────────────


def test_intracluster_correlation_tracks_the_construction():
    """The ICC estimate must order the two constructions correctly."""
    dependent_values, labels = _clustered_differences(
        n_clusters=40, cluster_size=20, cluster_sd=1.0, noise_sd=0.1, shift=0.0, seed=13
    )
    independent_values, _ = _clustered_differences(
        n_clusters=40, cluster_size=20, cluster_sd=0.0, noise_sd=1.0, shift=0.0, seed=13
    )
    dependent = intracluster_correlation(dependent_values, labels)
    independent = intracluster_correlation(independent_values, labels)
    assert dependent is not None and independent is not None
    assert dependent > 0.95
    assert abs(independent) < 0.1


def test_intracluster_correlation_is_none_when_it_is_not_estimable():
    """``None`` is not zero, and must not be recorded as zero.

    Singleton clusters leave no within-cluster variance, and a constant
    difference vector leaves no variance at all. Both are cases where the data
    cannot separate between-cluster from within-cluster variation.
    """
    values = [0.3, -0.1, 0.9, 0.2]
    assert intracluster_correlation(values, [0, 1, 2, 3]) is None
    assert intracluster_correlation([1.0] * 20, [i // 5 for i in range(20)]) is None


def test_summary_counts_describe_the_clusters_not_the_observations():
    """Cluster-level counts are what a reader needs to judge the sample size."""
    values = [1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 2.0]
    labels = ["a", "a", "a", "b", "b", "c", "c"]
    summary = summarize_clustered_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    assert summary.n_observations == 7
    assert summary.n_clusters == 3
    assert summary.largest_cluster == 3
    assert summary.positive_clusters == 2  # "a" sums to +3, "c" to +2
    assert summary.negative_clusters == 1  # "b" sums to -2
    assert summary.effect == pytest.approx(3.0 / 7.0)


def test_string_cluster_labels_group_by_identity():
    """Word strings are the G2P cluster key, so label identity has to group."""
    values = [1.0, 0.0, 1.0, 0.0]
    grouped = summarize_clustered_difference(
        values, ["cat", "dog", "cat", "dog"], resamples=64, seed=DEFAULT_SEED
    )
    assert grouped.n_clusters == 2
    assert grouped.largest_cluster == 2


# ── Input validation ─────────────────────────────────────────────────────────


def test_unpaired_input_is_refused():
    """An unpaired comparison cannot establish the claim in either direction."""
    with pytest.raises(ValueError, match="unpaired"):
        summarize_clustered_difference([0.1, 0.2, 0.3], [0, 1])


def test_empty_and_non_finite_input_are_refused():
    with pytest.raises(ValueError, match="non-empty"):
        summarize_clustered_difference([], [])
    with pytest.raises(ValueError, match="finite"):
        summarize_clustered_difference([0.1, float("nan")], [0, 1])


def test_one_cluster_is_refused_instead_of_manufacturing_certainty():
    """NEGATIVE CONTROL: one sampling unit cannot identify uncertainty."""
    with pytest.raises(ValueError, match="at least two independent clusters"):
        summarize_clustered_difference([0.1, 0.2, 0.3], ["only"] * 3, resamples=64)


def test_unhashable_cluster_labels_are_reported_as_invalid_input():
    with pytest.raises(ValueError, match="must be hashable"):
        summarize_clustered_difference(
            [0.1, 0.2],
            [["left"], ["right"]],
            resamples=64,  # type: ignore[list-item]
        )


def test_degenerate_resample_and_confidence_arguments_are_refused():
    values = [0.1, -0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="resamples"):
        summarize_clustered_difference(values, [0, 0, 1, 1], resamples=0)
    with pytest.raises(ValueError, match="at least two"):
        summarize_clustered_difference(values, [0, 0, 1, 1], resamples=1)
    with pytest.raises(ValueError, match="confidence"):
        summarize_clustered_difference(values, [0, 0, 1, 1], confidence=1.0)
    for bad_confidence in (True, "0.95", float("nan")):
        with pytest.raises(ValueError, match="confidence"):
            summarize_clustered_difference(
                values,
                [0, 0, 1, 1],
                confidence=bad_confidence,  # type: ignore[arg-type]
            )
    for bad_resamples in (True, 1.5, "64"):
        with pytest.raises(ValueError, match="resamples"):
            summarize_clustered_difference(
                values,
                [0, 0, 1, 1],
                resamples=bad_resamples,  # type: ignore[arg-type]
            )
    for bad_seed in (True, -1, "2026"):
        with pytest.raises(ValueError, match="seed"):
            summarize_clustered_difference(
                values,
                [0, 0, 1, 1],
                seed=bad_seed,  # type: ignore[arg-type]
            )


def test_variance_inflation_refuses_incomparable_summaries():
    """A ratio between two different populations is not a design effect."""
    short = summarize_clustered_difference([0.1, -0.2, 0.4], None, resamples=64)
    long = summarize_clustered_difference([0.1, -0.2, 0.4, 0.5], None, resamples=64)
    with pytest.raises(ValueError, match="different data"):
        variance_inflation(short, long)

    wide = summarize_clustered_difference([0.1, -0.2, 0.4], None, resamples=64, confidence=0.99)
    with pytest.raises(ValueError, match="different coverage"):
        variance_inflation(wide, short)

    different_data = summarize_clustered_difference([0.4, -0.2, 0.1], None, resamples=64)
    with pytest.raises(ValueError, match="different ordered difference vectors"):
        variance_inflation(short, different_data)

    different_resamples = summarize_clustered_difference([0.1, -0.2, 0.4], None, resamples=65)
    with pytest.raises(ValueError, match="different resample counts"):
        variance_inflation(short, different_resamples)

    different_seed = summarize_clustered_difference(
        [0.1, -0.2, 0.4], None, resamples=64, seed=DEFAULT_SEED + 1
    )
    with pytest.raises(ValueError, match="different seeds"):
        variance_inflation(short, different_seed)

    clustered_reference = summarize_clustered_difference([0.1, -0.2, 0.4], [0, 0, 1], resamples=64)
    with pytest.raises(ValueError, match="not independence-assuming"):
        variance_inflation(short, clustered_reference)

    flat = summarize_clustered_difference([0.0, 0.0, 0.0], None, resamples=64)
    with pytest.raises(ValueError, match="zero or invalid variance"):
        variance_inflation(flat, flat)


def test_data_digest_proves_the_compared_observation_vector_is_identical():
    first = summarize_clustered_difference([0.1, -0.2, 0.4], None, resamples=64)
    replay = summarize_clustered_difference([0.1, -0.2, 0.4], None, resamples=64)
    reordered = summarize_clustered_difference([0.4, -0.2, 0.1], None, resamples=64)
    assert first.data_digest == replay.data_digest
    assert first.data_digest != reordered.data_digest


def test_module_defaults_are_the_ones_the_recorder_relies_on():
    """The recorder does not restate these; a silent change would move the ledger."""
    assert DEFAULT_RESAMPLES == 10_000
    assert DEFAULT_SEED == 20260811
    assert isinstance(
        summarize_clustered_difference([0.1, -0.2], None, resamples=64),
        ClusteredPairedSummary,
    )


# ── The wild cluster bootstrap sensitivity arm ────────────────────────────────


def test_wild_singleton_clusters_reproduce_the_naive_interval():
    """POSITIVE CONTROL. Singleton clusters under degenerate weights = naive.

    With one observation per cluster and Rademacher weights the wild
    perturbation is the independence-assuming resampling analogue, so the
    interval must land on the naive percentile interval within a tolerance
    set by the Monte Carlo resolution of the two replicate vectors. A wild
    implementation that resampled observations instead of perturbing them, or
    that forgot the demeaning, would shift the center of mass or the spread
    and fail here.
    """
    rng = np.random.default_rng(61)
    n = 4000
    model = rng.random(n) < 0.62
    baseline = rng.random(n) < 0.58
    differences = [float(m) - float(s) for m, s in zip(model, baseline, strict=True)]
    wild = summarize_wild_cluster_difference(
        differences, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    naive = summarize_clustered_difference(
        differences, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )

    tolerance = 0.25 * naive.interval_width
    assert wild.ci_low == pytest.approx(naive.ci_low, abs=tolerance)
    assert wild.ci_high == pytest.approx(naive.ci_high, abs=tolerance)
    assert 0.8 < wild.interval_width / naive.interval_width < 1.25


def test_wild_replicate_variance_matches_the_cluster_robust_closed_form():
    """The Rademacher replicate variance is the CR0 variance, not a second one.

    This is the identity that makes the interval interpretable as a
    sensitivity band rather than a new estimator: the closed form
    ``sum_k (S_k - effect * n_k) ** 2 / n ** 2`` must be reproduced within
    Monte Carlo error.
    """
    values, labels = _clustered_differences(
        n_clusters=40, cluster_size=50, cluster_sd=1.0, noise_sd=0.05, shift=0.3, seed=61
    )
    wild = summarize_wild_cluster_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    expected_variance = (
        float(
            np.sum((np.bincount(labels, weights=values) - wild.effect * np.bincount(labels)) ** 2)
        )
        / len(values) ** 2
    )
    # Monte Carlo relative error on a variance from R replicates is about
    # ``sqrt(2 / (R - 1))``; three of those bound the procedure.
    assert wild.replicate_variance == pytest.approx(
        expected_variance, rel=3.0 * (2.0 / (TEST_RESAMPLES - 1)) ** 0.5
    )


def test_planted_strong_cluster_effect_widens_the_wild_interval():
    """NEGATIVE CONTROL (per the brief): the wild arm must see the clustering.

    The same construction that drives the clustered bootstrap's positive
    control must widen the wild interval relative to the naive one. A wild
    implementation that silently dropped the cluster labels would return the
    narrow independence interval here.
    """
    values, labels = _clustered_differences(
        n_clusters=40, cluster_size=50, cluster_sd=1.0, noise_sd=0.05, shift=0.3, seed=11
    )
    wild = summarize_wild_cluster_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    naive = summarize_clustered_difference(
        values, None, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )

    assert wild.interval_width > 3.0 * naive.interval_width


def test_wild_verdict_tracks_the_cluster_bootstrap_verdict():
    """The sensitivity arm is recorded beside the cluster bootstrap, not the naive one.

    On the strong-cluster construction both cluster-aware intervals exclude
    zero while the naive one does not, so a verdict recorded against the
    wrong comparator would read as disagreement.
    """
    values, labels = _clustered_differences(
        n_clusters=40, cluster_size=50, cluster_sd=1.0, noise_sd=0.05, shift=0.3, seed=11
    )
    wild = summarize_wild_cluster_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    clustered = summarize_clustered_difference(
        values, labels, resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )

    assert wild.interval_excludes_zero == clustered.interval_excludes_zero
    assert wild.verdict_agrees_with == "cluster bootstrap"


def test_wild_summary_is_deterministic_and_digest_pinned():
    """Same seed, bit-identical summary; same digest contract as the clustered path."""
    values, labels = _clustered_differences(
        n_clusters=25, cluster_size=8, cluster_sd=0.6, noise_sd=0.9, shift=0.15, seed=2
    )
    first = summarize_wild_cluster_difference(values, labels, resamples=512, seed=4242)
    second = summarize_wild_cluster_difference(values, labels, resamples=512, seed=4242)
    clustered = summarize_clustered_difference(values, labels, resamples=512, seed=4242)
    assert first == second
    assert first.data_digest == clustered.data_digest
    assert first.effect == clustered.effect


def test_wild_input_validation_refuses_degenerate_arguments():
    """The wild arm inherits the clustered path's refusals, not weaker ones."""
    values = [0.1, -0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="resamples"):
        summarize_wild_cluster_difference(values, [0, 0, 1, 1], resamples=1)
    with pytest.raises(ValueError, match="confidence"):
        summarize_wild_cluster_difference(values, [0, 0, 1, 1], confidence=1.0)
    with pytest.raises(ValueError, match="seed"):
        summarize_wild_cluster_difference(values, [0, 0, 1, 1], seed=-1)
    with pytest.raises(ValueError, match="unpaired"):
        summarize_wild_cluster_difference([0.1, 0.2, 0.3], [0, 1])
    with pytest.raises(ValueError, match="at least two independent clusters"):
        summarize_wild_cluster_difference([0.1, 0.2, 0.3], ["only"] * 3, resamples=64)
    with pytest.raises(ValueError, match="finite"):
        summarize_wild_cluster_difference([0.1, float("nan")], [0, 1])


def test_all_zero_differences_give_the_degenerate_wild_interval():
    """Stage 4's gold population is this vector; zero mass must stay zero."""
    summary = summarize_wild_cluster_difference(
        [0.0] * 500, [i // 10 for i in range(500)], resamples=TEST_RESAMPLES, seed=DEFAULT_SEED
    )
    assert summary.ci_low == 0.0 and summary.ci_high == 0.0
    assert summary.replicate_variance == 0.0
    assert not summary.interval_excludes_zero
