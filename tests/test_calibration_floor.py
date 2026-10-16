"""Guards for the offline experiment that locates the captured-mass floor.

What this file protects
-----------------------
``ailm.benchmarks.CALIBRATION_MASS_FLOOR`` withholds calibration below half
observed mass. ``scripts/record_calibration_floor_claims.py`` locates where that
threshold belongs by truncating AILM's own letter columns across a grid of
target captured masses and reconstructing each one under both completion rules
the project has available. The experiment is only worth anything if three things
hold, and each has a test below.

1. **The sweep is reproducible.** It makes no model call and draws no random
   number, so two runs on the same inputs must agree bit for bit. A sweep that
   drifted would put a different floor in the manuscript on every render.

2. **Truncation at full mass is the identity, and the two rules then agree.**
   This is the positive control for the divergence metric itself: if the rules
   disagreed where nothing was truncated, every divergence on the grid would be
   measuring the harness rather than the completion. It is paired with the
   opposite control — the metric must report a large divergence where the rules
   really do disagree — because a divergence function stuck at zero would pass
   the first control alone and make the located floor meaningless.

3. **The locator survives a transient crossing.** The two rules bias a reported
   metric in opposite directions, so their difference passes through zero in the
   middle of the grid while the reconstructions are still nearly disjoint. A
   first-crossing rule reads that coincidence as convergence. The locator is
   therefore fed a planted series with exactly that shape and required to ignore
   the dip.

The recorded finding on non-monotonicity
----------------------------------------
The rule-versus-rule total variation is monotone non-increasing in captured
mass, as a floor argument implicitly assumes. The two *reported metrics* are
not: their divergences rise again after the crossing. That is a property of the
measurement, not a bug to assert away, so the tests here record it — by fixing
the consequence (the sustained crossing lands strictly above the first one)
rather than by demanding a monotonicity the data does not have.
"""

from __future__ import annotations

import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

import pytest

from ailm.benchmarks import (
    CALIBRATION_FLOOR_EPSILON,
    CALIBRATION_FLOOR_GRID_HIGH,
    CALIBRATION_FLOOR_GRID_LOW,
    CALIBRATION_FLOOR_IDENTITY_TOLERANCE,
    CALIBRATION_MASS_FLOOR,
    DEFAULT_SEED,
    LETTER_TEXT,
    _distribution_from_candidates,
    brier_score,
    calibration_floor_grid,
    complete_min_bound,
    complete_observed_only,
    completion_rule_sweep,
    divergence_increases,
    divergence_sign_changes,
    expected_calibration_error,
    fidelity_crossover,
    letter_posterior,
    locate_divergence_floor,
    subsample_positions,
    total_variation,
    truncate_to_mass,
    unique_letter_positions,
)
from ailm.levels.letter import ALPHA_LIST, load_brown_counts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"
RECORDER = "record_calibration_floor_claims.py"

#: Coarse grid for the tests that only need the sweep's shape. The recorder runs
#: the full 0.01 grid; repeating that in every test buys nothing but seconds.
COARSE_GRID = calibration_floor_grid(0.1, 1.0, 0.1)

#: Position count for the sweep-level tests. Small enough to stay well inside
#: the suite's per-test budget, large enough that the letter pool contributes
#: many distinct conditioning symbols rather than one column repeated.
TEST_POSITIONS = 60

#: Every field one sweep row carries. The recorder reads rows by name, so this
#: is pinned rather than inferred.
ROW_KEYS = (
    "target_mass",
    "captured_mass",
    "mean_k",
    "over_cap_fraction",
    "min_bound_ece",
    "observed_only_ece",
    "ece_signed",
    "ece_divergence",
    "min_bound_brier",
    "observed_only_brier",
    "brier_signed",
    "brier_divergence",
    "joint_divergence",
    "rule_tv",
    "min_bound_tv_truth",
    "observed_only_tv_truth",
)


@lru_cache(maxsize=1)
def _brown() -> dict:
    return load_brown_counts(PROJECT_ROOT / "vendor")


@lru_cache(maxsize=8)
def _positions(count: int) -> tuple[tuple[str, str], ...]:
    pool, _unused = unique_letter_positions(LETTER_TEXT)
    return tuple(subsample_positions(pool, count, DEFAULT_SEED))


@lru_cache(maxsize=4)
def _sweep(count: int, grid: tuple[float, ...]) -> tuple[dict, ...]:
    return tuple(completion_rule_sweep(_brown(), list(_positions(count)), grid=grid))


def _claims() -> dict:
    if not LEDGER.exists():
        return {}
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


def _floor_claim(label: str) -> object:
    claims = _claims()
    if label not in claims:
        pytest.skip(f"{label} not recorded in this checkout")
    return claims[label]["value"]


# ── The grid ─────────────────────────────────────────────────────────────────


def test_the_grid_spans_the_region_no_live_system_measured():
    """The grid must cover the gap the floor sits in, or it locates nothing.

    The three measured probes captured roughly 0.07, 0.86 and 1.00 of their
    answer-slot mass. The declared floor at 0.5 lies strictly between the first
    two, where no system was ever observed, so a grid that did not straddle it
    would leave the floor exactly as unlocated as before.
    """
    grid = calibration_floor_grid()
    assert grid[0] == CALIBRATION_FLOOR_GRID_LOW
    assert grid[-1] == CALIBRATION_FLOOR_GRID_HIGH
    assert grid[0] < CALIBRATION_MASS_FLOOR < grid[-1]
    assert len(grid) > 10, "too few grid points to locate a crossing to any precision"


def test_the_grid_is_built_from_exact_decimals():
    """Accumulated floating-point addition would put 0.44999999999999996 on the axis.

    A grid point off by its last bit changes which point a crossing is
    attributed to, so the located floor would depend on the order the additions
    happened in rather than on the measurement.
    """
    grid = calibration_floor_grid(0.05, 1.0, 0.01)
    assert 0.45 in grid
    assert 0.99 in grid
    assert all(value == round(value, 2) for value in grid)


def test_the_grid_rejects_a_step_that_does_not_divide_the_span():
    """A step that misses the top of the grid would silently drop full capture.

    Full capture is the identity point and the positive control, so a grid that
    stops short of it removes the only place the two rules are known to agree.
    """
    with pytest.raises(ValueError, match="does not divide"):
        calibration_floor_grid(0.05, 1.0, 0.03)


# ── Truncation ───────────────────────────────────────────────────────────────


def test_truncation_at_full_mass_is_the_identity():
    """At target 1.0 nothing is removed, on every column of the real matrix.

    The floating-point sum of a normalized column lands a few units in the last
    place below 1.0, so an exact comparison would drop the final symbol and make
    full capture indistinguishable from truncation at k = 29. That would quietly
    delete the experiment's control point.
    """
    for symbol in ALPHA_LIST:
        column = letter_posterior(_brown(), symbol)
        truncated, captured = truncate_to_mass(column, 1.0)
        assert truncated == column, f"column {symbol!r} lost symbols at full mass"
        # Against the column's own mass rather than against 1.0: the transition
        # matrix is built in float32, so a normalized column sums to 1 within
        # about 1e-9 and comparing to a hard 1.0 would test the matrix's dtype
        # rather than the truncation.
        assert captured == pytest.approx(sum(column.values()), abs=1e-12)
        assert captured == pytest.approx(1.0, abs=1e-6)


def test_truncation_returns_the_smallest_prefix_reaching_the_target():
    """The top-k boundary is where ``top_logprobs`` puts it, not one symbol away."""
    distribution = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    assert truncate_to_mass(distribution, 0.5)[0] == {"a": 0.5}
    assert truncate_to_mass(distribution, 0.51)[0] == {"a": 0.5, "b": 0.3}
    assert truncate_to_mass(distribution, 0.8)[0] == {"a": 0.5, "b": 0.3}
    assert truncate_to_mass(distribution, 0.81)[0] == {"a": 0.5, "b": 0.3, "c": 0.15}
    assert truncate_to_mass(distribution, 0.9)[1] == pytest.approx(0.95)


def test_truncation_breaks_ties_by_symbol_rather_than_by_insertion_order():
    """A tie broken by dict order makes the sweep depend on how a column was built."""
    ascending = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    descending = {"d": 0.25, "c": 0.25, "b": 0.25, "a": 0.25}
    assert truncate_to_mass(ascending, 0.3)[0] == truncate_to_mass(descending, 0.3)[0]
    assert set(truncate_to_mass(descending, 0.3)[0]) == {"a", "b"}


@pytest.mark.parametrize("target", [0.0, -0.1, 1.5])
def test_truncation_rejects_a_target_outside_the_unit_interval(target: float):
    """A target of zero would return an empty report and a divide-by-zero downstream."""
    with pytest.raises(ValueError, match="target_mass"):
        truncate_to_mass({"a": 1.0}, target)


# ── The completion rules, and the control that the divergence metric works ───


def test_the_two_rules_agree_when_nothing_is_truncated():
    """POSITIVE CONTROL for the divergence metric.

    Where the report is complete, both rules must reduce to the same
    distribution: min-bound has no absent symbol to bound and observed-only has
    no survivor to drop. If they disagreed here, every divergence on the grid
    would be measuring the harness rather than the completion rule, and the
    located floor would be an artifact.

    Agreement is checked to ``CALIBRATION_FLOOR_IDENTITY_TOLERANCE`` rather than
    bit-exactly: ``complete_min_bound`` routes through the candidate payload the
    server would have sent, so every probability makes a log/exp round trip the
    other rule does not. Measuring the *shipped* rule is worth more than
    bit-exactness against a rule that skips it.
    """
    for symbol in ALPHA_LIST:
        column = letter_posterior(_brown(), symbol)
        truncated, _captured = truncate_to_mass(column, 1.0)
        divergence = total_variation(
            complete_min_bound(truncated), complete_observed_only(truncated)
        )
        assert divergence < CALIBRATION_FLOOR_IDENTITY_TOLERANCE, (
            f"the two rules disagree by {divergence} on column {symbol!r} where "
            f"nothing was truncated; the divergence metric is measuring the harness"
        )


def test_the_divergence_metric_reports_a_real_disagreement():
    """NON-VACUITY for the control above.

    A divergence function that always returned zero would pass
    ``test_the_two_rules_agree_when_nothing_is_truncated`` and make the whole
    sweep read as "the rules never disagree". At a single reported candidate the
    rules are maximally far apart by construction — min-bound assigns every
    absent symbol the one observed probability and lands on the uniform
    distribution, observed-only lands on a one-hot — so the divergence must be
    the largest it can be over a thirty-symbol inventory.
    """
    column = letter_posterior(_brown(), "t")
    truncated, _captured = truncate_to_mass(column, 0.01)
    assert len(truncated) == 1
    min_bound = complete_min_bound(truncated)
    observed_only = complete_observed_only(truncated)
    assert min_bound == pytest.approx({s: 1.0 / len(ALPHA_LIST) for s in ALPHA_LIST})
    assert observed_only[next(iter(truncated))] == pytest.approx(1.0)
    assert total_variation(min_bound, observed_only) == pytest.approx(1.0 - 1.0 / len(ALPHA_LIST))


def test_observed_only_gives_an_absent_symbol_nothing():
    """The rule's defining property, and the reason it is the comparison point."""
    reconstructed = complete_observed_only({"a": 0.4, "b": 0.2})
    assert reconstructed["a"] == pytest.approx(2 / 3)
    assert reconstructed["b"] == pytest.approx(1 / 3)
    assert all(reconstructed[s] == 0.0 for s in ALPHA_LIST if s not in {"a", "b"})
    assert sum(reconstructed.values()) == pytest.approx(1.0)


def test_min_bound_refuses_a_zero_probability_candidate():
    """A zero would become the bound and collapse min-bound into observed-only.

    The two rules would then agree for a reason that has nothing to do with how
    much mass was captured, and the sweep would report a floor of zero.
    """
    with pytest.raises(ValueError, match="non-positive probability"):
        complete_min_bound({"a": 0.5, "b": 0.0})


def test_min_bound_is_the_shipped_rule_rather_than_a_copy_of_it():
    """The reconstruction must match what the live probe path produces.

    ``complete_min_bound`` exists so the sweep measures the rule the published
    calibration numbers came from. Reimplementing the arithmetic would let the
    two drift apart and the experiment would then describe a rule nothing ships.
    """
    observed = {"a": 0.5, "b": 0.25, "e": 0.125}
    direct = _distribution_from_candidates(
        [{"token": s, "logprob": math.log(p)} for s, p in observed.items()]
    )
    assert complete_min_bound(observed) == direct


# ── The sweep ────────────────────────────────────────────────────────────────


def test_the_sweep_is_reproducible():
    """No model call, no random draw: two runs on the same inputs must be identical.

    A sweep that drifted would put a different located floor in the manuscript
    on every render, and the drift would be invisible because each individual
    run is internally consistent.
    """
    positions = list(_positions(TEST_POSITIONS))
    first = completion_rule_sweep(_brown(), positions, grid=COARSE_GRID)
    second = completion_rule_sweep(_brown(), positions, grid=COARSE_GRID)
    assert first == second


def test_the_sweep_identity_row_reproduces_the_untruncated_calibration():
    """At full capture the sweep must score exactly the untruncated column.

    Computed here from ``letter_posterior`` directly, without going through the
    truncation or either completion rule, so this is an independent derivation
    rather than the sweep checking itself.
    """
    positions = list(_positions(TEST_POSITIONS))
    rows = _sweep(TEST_POSITIONS, COARSE_GRID)
    confidences: list[float] = []
    correct: list[bool] = []
    distributions: list[dict[str, float]] = []
    golds: list[str] = []
    for prefix, gold in positions:
        column = letter_posterior(_brown(), prefix[-1] if prefix else "")
        best = max(column, key=lambda s: (column[s], s))
        confidences.append(column[best])
        correct.append(best == gold)
        distributions.append(column)
        golds.append(gold)
    # Tolerance, not equality: the shipped completion rule round-trips every
    # probability through log and exp, and the columns arrive from a float32
    # matrix, so the two derivations agree to about nine significant figures
    # rather than bit for bit.
    assert rows[-1]["min_bound_ece"] == pytest.approx(
        expected_calibration_error(confidences, correct), abs=1e-7
    )
    assert rows[-1]["min_bound_brier"] == pytest.approx(brier_score(distributions, golds), abs=1e-7)


def test_the_sweep_rejects_an_empty_position_list():
    """An empty sweep would post zero divergence everywhere and locate the grid floor."""
    with pytest.raises(ValueError, match="at least one scored position"):
        completion_rule_sweep(_brown(), [], grid=COARSE_GRID)


def test_the_sweep_row_carries_exactly_the_documented_fields():
    """The recorder indexes rows by name, so a renamed field must fail here first.

    A missing key would surface as a ``KeyError`` deep inside the recorder, and
    a *silently added* one would go unrecorded. Pinning the schema keeps the
    ledger's coverage of the sweep a deliberate choice rather than an accident.
    """
    assert set(_sweep(TEST_POSITIONS, COARSE_GRID)[0]) == set(ROW_KEYS)


def test_the_rule_divergence_is_monotone_in_captured_mass():
    """Total variation between the rules never rises as more mass is captured.

    This is the shape a floor argument assumes: capture more, disagree less. It
    holds for the rule-versus-rule distance, which is why the located floor is
    reported against a divergence that behaves and the reported metrics are
    reported beside it rather than in place of it.
    """
    rows = list(_sweep(TEST_POSITIONS, calibration_floor_grid(0.05, 1.0, 0.01)))
    assert divergence_increases(rows, "rule_tv") == 0
    assert rows[0]["rule_tv"] > rows[-1]["rule_tv"], "the sweep found no divergence at all"


def test_the_reported_metrics_cross_before_they_converge():
    """RECORDED FINDING, not a desired property.

    The disagreement in expected calibration error and in the Brier score is not
    monotone in captured mass. Min-bound flattens the distribution and understates
    confidence; observed-only sharpens it and overstates confidence. The two
    biases cancel somewhere in the middle of the grid, the signed difference
    changes sign there, and the absolute divergence dips to near zero while the
    two reconstructions are still nearly disjoint.

    The test fixes the consequence rather than the shape: a first-crossing floor
    lands strictly below a sustained-crossing one, which is why the recorder uses
    the latter. If this ever stops holding, the sweep has changed character and
    the manuscript's account of the crossing needs rewriting — it should not be
    silently satisfied by a laxer assertion.
    """
    rows = list(_sweep(TEST_POSITIONS, calibration_floor_grid(0.05, 1.0, 0.01)))
    for key, signed in (("ece_divergence", "ece_signed"), ("brier_divergence", "brier_signed")):
        located = locate_divergence_floor(rows, key, CALIBRATION_FLOOR_EPSILON)
        assert divergence_sign_changes(rows, signed) >= 1, (
            f"{signed} never changes sign, so the two rules never cross and the "
            f"sustained-crossing definition is unnecessary for {key}"
        )
        assert located["naive_target"] < located["located_target"], (
            f"{key} reaches its first crossing at the same point it settles, so "
            f"the transient dip this experiment reports has disappeared"
        )
        assert located["increases"] > 0, f"{key} is monotone; the finding has changed"


def test_the_shipped_rule_is_not_uniformly_the_closer_reconstruction():
    """RECORDED FINDING about the min-bound rule's fidelity to the truth.

    The shipped rule is conservative in the direction it was designed for — an
    absent symbol is bounded above, never below — and that is a statement about
    the bound rather than about accuracy. Measured against the column it is
    reconstructing, it is the closer of the two rules only below a crossover in
    captured mass; above it the plain renormalization of the survivors is closer.
    """
    rows = list(_sweep(TEST_POSITIONS, calibration_floor_grid(0.05, 1.0, 0.01)))
    crossover = fidelity_crossover(rows)
    assert crossover is not None, "min-bound now dominates the whole grid; the finding changed"
    assert rows[0]["min_bound_tv_truth"] < rows[0]["observed_only_tv_truth"]
    assert rows[-2]["observed_only_tv_truth"] < rows[-2]["min_bound_tv_truth"]


# ── The locator ──────────────────────────────────────────────────────────────


def _planted(values: list[float]) -> list[dict[str, float]]:
    """A synthetic sweep carrying one divergence series and its grid positions."""
    return [
        {"target_mass": round(0.1 * (index + 1), 2), "captured_mass": 0.1 * (index + 1), "d": v}
        for index, v in enumerate(values)
    ]


def test_the_locator_ignores_a_transient_crossing():
    """POSITIVE CONTROL for the sustained-crossing definition.

    The planted series dips below the threshold at index 1 and rises back above
    it at index 2. A first-crossing rule returns index 1; the locator must return
    index 4, the first point after which the series never exceeds the threshold
    again. Without this control, a locator that simply returned the first
    crossing would pass every other test in this file.
    """
    rows = _planted([0.5, 0.001, 0.4, 0.02, 0.005, 0.0])
    result = locate_divergence_floor(rows, "d", 0.01)
    assert result["naive_index"] == 1
    assert result["located_index"] == 4
    assert result["naive_target"] < result["located_target"]
    assert result["max_divergence"] == 0.5
    assert result["increases"] == 1


def test_the_locator_returns_the_first_grid_point_when_nothing_ever_exceeds():
    """A series already below the threshold everywhere has its floor at the bottom.

    The recorder refuses in that case rather than publishing it, because a grid
    that never saw the transition has located nothing.
    """
    result = locate_divergence_floor(_planted([0.001, 0.002, 0.0]), "d", 0.01)
    assert result["located_index"] == 0


def test_the_locator_refuses_when_the_top_of_the_grid_still_diverges():
    """Full capture is the identity; a divergence there means the harness is broken."""
    with pytest.raises(ValueError, match="no floor is locatable"):
        locate_divergence_floor(_planted([0.5, 0.2, 0.3]), "d", 0.01)


def test_the_locator_rejects_a_non_positive_threshold():
    """A zero threshold makes every floating-point residual a divergence."""
    with pytest.raises(ValueError, match="epsilon must be positive"):
        locate_divergence_floor(_planted([0.5, 0.0]), "d", 0.0)


def test_sign_changes_counts_crossings_and_ignores_exact_zeros():
    """A signed series that touches zero has not necessarily crossed it."""
    assert divergence_sign_changes(_planted([1.0, 0.5, -0.2, -0.4]), "d") == 1
    assert divergence_sign_changes(_planted([1.0, 0.0, 0.5]), "d") == 0
    assert divergence_sign_changes(_planted([1.0, -1.0, 1.0]), "d") == 2


def test_divergence_increases_counts_only_rises():
    """NON-VACUITY: the monotonicity check must be able to report a non-zero count."""
    assert divergence_increases(_planted([0.5, 0.4, 0.3]), "d") == 0
    assert divergence_increases(_planted([0.5, 0.4, 0.6, 0.1]), "d") == 1


# ── The recorder's refusals ──────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _recorder():
    """Import the recorder as a module so its refusals can be exercised directly."""
    spec = importlib.util.spec_from_file_location(
        "_calibration_floor_recorder", PROJECT_ROOT / "scripts" / RECORDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(target: float, divergence: float, rule_tv: float) -> dict[str, float]:
    """One synthetic sweep row carrying a chosen divergence at a chosen mass."""
    row = dict.fromkeys(ROW_KEYS, 0.0)
    row["target_mass"] = target
    row["captured_mass"] = target
    row["ece_divergence"] = divergence
    row["brier_divergence"] = divergence
    row["joint_divergence"] = divergence
    row["rule_tv"] = rule_tv
    return row


def _validate_planted_sweep(rows: list[dict[str, float]]) -> None:
    """Submit planted measurements to the recorder's real pre-write boundary."""
    _recorder()._validate_primary_sweep(rows)


def test_the_recorder_refuses_when_the_identity_point_diverges():
    """POSITIVE CONTROL for refusal 1: a broken identity must stop the run.

    Truncation at full mass removes nothing, so a divergence there is a bug in
    the truncation or in a completion rule. Every number the recorder would
    write then describes the bug. Without this control the refusal could be
    unreachable and nothing would say so.
    """
    rows = [_row(0.05, 0.5, 0.9), _row(1.0, 0.2, 0.3)]
    with pytest.raises(SystemExit, match="above the identity tolerance"):
        _validate_planted_sweep(rows)


def test_the_recorder_refuses_a_sweep_that_never_diverges():
    """POSITIVE CONTROL for refusal 2: a vacuous sweep must not publish a floor.

    If no grid point exceeds the threshold, the located floor is the bottom of
    the grid for want of any signal — a number that looks like a measurement and
    is not one.
    """
    rows = [_row(0.05, 0.001, 0.001), _row(0.5, 0.002, 0.001), _row(1.0, 0.0, 0.0)]
    with pytest.raises(SystemExit, match="largest rule divergence"):
        _validate_planted_sweep(rows)


def test_the_recorder_refuses_a_grid_that_starts_above_the_transition():
    """POSITIVE CONTROL for refusal 3, and proof it is reachable at all.

    The obvious form of this guard — "the located index must not be zero" — is
    dead: under the sustained-crossing definition a located index of zero means
    nothing exceeded the threshold, which refusal 2 has already caught. The
    reachable condition is a grid whose *lowest* point is already negligible,
    and the planted series here has exactly that shape while still diverging in
    the middle, so it clears refusal 2 and must be stopped by refusal 3.
    """
    rows = [_row(0.05, 0.001, 0.001), _row(0.5, 0.4, 0.8), _row(1.0, 0.0, 0.0)]
    with pytest.raises(SystemExit, match="begins above the transition"):
        _validate_planted_sweep(rows)


# ── The recorded claims ──────────────────────────────────────────────────────


@pytest.mark.skipif(not LEDGER.exists(), reason="data/claims.json absent; run the recorders")
def test_every_recorded_floor_label_comes_from_this_experiment_s_recorder():
    """Recorders write, tests verify — and one recorder owns one namespace.

    ``bench_floor_*`` shares the ``bench_`` prefix with two Ollama-backed
    recorders whose stale-label sweeps are scoped by source. A label here
    sourced anywhere else means two producers are writing one namespace, which
    is the defect ``tests/test_claims_single_writer.py`` exists for.
    """
    claims = _claims()
    labels = [label for label in claims if label.startswith("bench_floor_")]
    if not labels:
        pytest.skip("no bench_floor_* claims recorded in this checkout")
    offenders = {
        label: claims[label]["source"]
        for label in labels
        if Path(claims[label]["source"]).name != RECORDER
    }
    assert offenders == {}, f"bench_floor_* labels written by another producer: {offenders}"


@pytest.mark.skipif(not LEDGER.exists(), reason="data/claims.json absent; run the recorders")
def test_the_recorded_floor_sits_above_the_declared_one():
    """The experiment's result, held against the value it was run to check.

    The declared floor is one half. If the located floor ever fell to or below
    it, the manuscript's account of this experiment would be wrong in its
    direction, not merely in its digits, so the assertion is on the comparison
    rather than on the number.
    """
    declared = float(_floor_claim("bench_floor_declared_floor"))  # type: ignore[arg-type]
    located = float(_floor_claim("bench_floor_located_captured"))  # type: ignore[arg-type]
    assert declared == CALIBRATION_MASS_FLOOR
    assert located > declared, (
        f"the located floor {located} is at or below the declared {declared}; the "
        f"supplement's claim that the declared floor is too permissive no longer holds"
    )


@pytest.mark.skipif(not LEDGER.exists(), reason="data/claims.json absent; run the recorders")
def test_the_recorded_first_crossing_is_reported_beside_a_large_rule_divergence():
    """The trap must be recorded with the evidence that it is a trap.

    A first crossing is only worth reporting if the two reconstructions are still
    far apart there. If the recorded total variation at that point were small,
    the crossing would be genuine convergence and the sustained-crossing
    definition would be over-engineering.
    """
    naive_tv = float(_floor_claim("bench_floor_naive_ece_rule_tv"))  # type: ignore[arg-type]
    located_tv = float(_floor_claim("bench_floor_located_rule_tv"))  # type: ignore[arg-type]
    assert naive_tv > 0.5, (
        f"the two rules differ by only {naive_tv} in total variation at the first "
        f"crossing, so that crossing is convergence rather than coincidence"
    )
    assert located_tv < naive_tv


@pytest.mark.skipif(not LEDGER.exists(), reason="data/claims.json absent; run the recorders")
def test_the_recorded_identity_point_agrees_with_the_live_calibration_arm():
    """Cross-recorder check: the offline sweep scores what the benchmark scores.

    At full capture the sweep reduces to AILM's untruncated letter posterior,
    which is exactly the comparator ``calibration_metrics`` reports. The two
    numbers are produced by different recorders, one of them against a live
    Ollama server, so agreement is evidence that this experiment is set on the
    same positions and the same metric as the arm whose floor it is locating.

    Agreement is to six significant figures rather than bit-exact: the shipped
    completion rule round-trips every probability through log and exp, and the
    transition matrix is built in float32. Six figures is far tighter than any
    difference in population or metric could survive — a sweep set on different
    positions, or scoring the reconstruction's argmax instead of the decode,
    moves these numbers in the second decimal place.
    """
    claims = _claims()
    for label in ("bench_calib_positions", "bench_floor_positions"):
        if label not in claims:
            pytest.skip(f"{label} not recorded in this checkout")
    if claims["bench_calib_positions"]["value"] != claims["bench_floor_positions"]["value"]:
        pytest.skip("the calibration arm and the floor sweep ran at different sample sizes")
    for floor_label, calib_label in (
        ("bench_floor_identity_ece", "bench_calib_ece_ailm"),
        ("bench_floor_identity_brier", "bench_calib_brier_ailm"),
    ):
        if calib_label not in claims:
            pytest.skip(f"{calib_label} not recorded in this checkout")
        assert float(claims[floor_label]["value"]) == pytest.approx(
            float(claims[calib_label]["value"]), rel=1e-6
        ), f"{floor_label} disagrees with {calib_label}"


# ── The figure ───────────────────────────────────────────────────────────────
#
# ``fig:calibration_floor`` draws the per-grid-point series, which the ledger
# does not carry: only summaries of it are recorded. The figure therefore
# recomputes the sweep from the functions above and refuses to draw if any
# re-derived summary contradicts a recorded claim. These tests exist because a
# refusal that cannot fire is decoration, and a recomputation nobody compares
# against the ledger is a second, unaudited measurement.

VARIABLES = PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"

_NEEDS_TOKENS = pytest.mark.skipif(
    not VARIABLES.exists(),
    reason="output/data/manuscript_variables.json absent; run z_generate_manuscript_variables.py",
)


@lru_cache(maxsize=1)
def _token_map() -> dict[str, str]:
    payload = json.loads(VARIABLES.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in payload.items()}


@_NEEDS_TOKENS
def test_the_figure_recomputes_the_recorded_sweep_rather_than_a_different_one():
    """The drawn series must be the swept grid the recorder recorded.

    The data function raises rather than returning on any disagreement, so
    reaching this assertion at all is most of the check. What is asserted here is
    the part a refusal cannot cover: that the rows handed to the plotting code
    are one per recorded grid point, so the panels draw the sweep at its measured
    resolution instead of a subsample or an interpolation of it.
    """
    from ailm.figures.calibration_floor import calibration_floor_data

    data = calibration_floor_data(_token_map())
    recorded_points = int(float(_token_map()["BENCH_FLOOR_GRID_POINTS"]))
    assert len(data["rows"]) == recorded_points
    assert len(data["captured"]) == recorded_points
    assert float(data["located"]["captured_mass"]) == pytest.approx(
        float(_token_map()["BENCH_FLOOR_LOCATED_CAPTURED"]), rel=1e-5
    )
    assert float(data["at_declared"]["ece_divergence"]) == pytest.approx(
        float(_token_map()["BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE"]), rel=1e-5
    )


@_NEEDS_TOKENS
@pytest.mark.parametrize(
    "token,value",
    [
        ("BENCH_FLOOR_LOCATED_CAPTURED", "0.5"),
        ("BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE", "0.0001"),
        ("BENCH_FLOOR_NAIVE_ECE_CAPTURED", "0.9"),
        ("BENCH_FLOOR_IDENTITY_ECE", "0.5"),
        ("BENCH_FLOOR_LOCATED_K", "2.0"),
        ("BENCH_FLOOR_ECE_SIGN_CHANGES", "3"),
        ("BENCH_FLOOR_TV_INCREASES", "7"),
        ("BENCH_FLOOR_GRID_POINTS", "12"),
    ],
)
def test_the_figure_refuses_to_draw_against_a_contradicted_claim(token: str, value: str):
    """POSITIVE CONTROL — the refusal fires, on every class of claim it guards.

    One doctored token per class: where the floor is, how far the rules disagree
    at the gate, where the first crossing falls, the identity control, what
    reaching the floor costs, the sign-change and monotonicity counts, and the
    shape of the grid. A refusal that only caught one of these would leave the
    rest of the panel free to contradict the prose beside it.
    """
    from ailm.figures.calibration_floor import calibration_floor_data

    drifted = dict(_token_map())
    drifted[token] = value
    with pytest.raises(ValueError, match="contradicts the ledger"):
        calibration_floor_data(drifted)


@_NEEDS_TOKENS
def test_the_figure_draws_the_ledger_s_zero_crossings_and_no_others():
    """The trap is the finding, so the marks that point at it are pinned.

    Each metric's signed difference passes through zero exactly once. A figure
    that drew two crossings, or none, would be making a claim about the locator
    that the recorded sign-change counts deny.
    """
    from ailm.figures.calibration_floor import calibration_floor_data

    data = calibration_floor_data(_token_map())
    assert len(data["ece_crossings"]) == int(float(_token_map()["BENCH_FLOOR_ECE_SIGN_CHANGES"]))
    assert len(data["brier_crossings"]) == int(
        float(_token_map()["BENCH_FLOOR_BRIER_SIGN_CHANGES"])
    )
    # The crossing the trap is named for sits below the located floor, which is
    # the whole reason a first-crossing locator is unsound.
    first = data["captured"][data["ece_crossings"][0]]
    assert first < float(data["located"]["captured_mass"])


@_NEEDS_TOKENS
def test_the_figure_plots_every_swept_point_rather_than_a_smooth_curve():
    """Non-monotonicity between grid points is data, not noise to be smoothed.

    The panels draw a marker per grid point precisely so the rises the ledger
    counts are visible. Rendering the same series as an interpolated line would
    hide the property that makes the sustained-crossing definition necessary, so
    the drawn vertex count is held to the swept point count.
    """
    import matplotlib.pyplot as plt

    from ailm.figures.calibration_floor import build_calibration_floor_figure

    figure = build_calibration_floor_figure(_token_map())
    try:
        expected = int(float(_token_map()["BENCH_FLOOR_GRID_POINTS"]))
        series = [
            line
            for axes in figure.axes
            for line in axes.get_lines()
            if len(line.get_xdata()) == expected
        ]
        assert len(series) >= 5, (
            f"only {len(series)} drawn series carry all {expected} swept points; the "
            f"panels are showing a smoothed or subsampled sweep"
        )
        assert all(line.get_marker() not in ("", "None", None) for line in series), (
            "a swept series is drawn without per-point markers, which hides the "
            "non-monotonicity the recorded increase counts report"
        )
    finally:
        plt.close(figure)
