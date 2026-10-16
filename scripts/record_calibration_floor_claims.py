#!/usr/bin/env python3
"""Locate the external benchmark's captured-mass floor empirically, offline.

Why this recorder exists
------------------------
``ailm.benchmarks.CALIBRATION_MASS_FLOOR`` withholds calibration for any probe
whose answer-slot distribution is less than half *observed*. The reasoning is
sound — below some captured mass the completion rule rather than the model
decides the reported number — but one half was argued, never located. The three
measured probes sit at roughly 0.07, 0.86 and 1.00 captured mass, so the live
protocol measures nothing between about 12% and 98% and cannot say where the
transition happens. ``docs/decisions.md`` entry 30 and the calibration
subsection of ``docs/manuscript/98_supplement_external_benchmarks.md`` both state the
floor as a principled choice.

What this measures
------------------
The experiment needs distributions whose truncation behavior can be swept, and
it needs them without a model call. AILM's own letter transition columns are
exactly that: real next-symbol distributions over the same thirty-symbol
inventory the benchmark scores, deterministic, and already the comparator the
live calibration arm reports. Each scored position contributes the column
:func:`ailm.benchmarks.letter_posterior` returns.

For each target captured mass on the grid, the column is truncated to the
smallest top-k reaching that mass — the truncation ``top_logprobs`` imposes —
and reconstructed under both completion rules the project has available: the
shipped min-bound rule, read out of ``_distribution_from_candidates`` rather
than restated, and the observed-only rule that renormalizes the survivors and
gives an absent symbol nothing. Both reconstructions are scored with the two
metrics the benchmark reports, and the divergence between the two rules is the
measured quantity. Where that divergence is negligible the reported number is a
property of the distribution; where it is not, it is a property of the
completion rule, which is what the floor exists to prevent.

Refusals — no claim is written unless every one of these holds:

* truncation at full mass is the identity, so both rules agree exactly there;
* the divergence exceeds the declared threshold somewhere on the grid, so the
  located floor is a measurement rather than a vacuously satisfied bound;
* the located floor is not at the bottom of the grid, which would mean the grid
  starts above the transition and never saw it.

Namespace ownership
-------------------
This shares the ``bench_`` prefix with the two Ollama-backed recorders.
Ownership is by recorded ``source``, not by prefix: each recorder retires only
the labels ``ClaimsLedger.labels_owned_by`` attributes to its own file. All
three take the ledger's single-writer lock, so they cannot run at the same
time.

Usage
-----
    .venv/bin/python scripts/record_calibration_floor_claims.py

    # isolated validation, never the tracked ledger:
    AILM_CLAIMS_PATH=/tmp/floor_claims.json \
        .venv/bin/python scripts/record_calibration_floor_claims.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.benchmarks import (  # noqa: E402  (path setup must precede the import)
    CALIBRATION_BINS,
    CALIBRATION_FLOOR_EPSILON,
    CALIBRATION_FLOOR_GRID_HIGH,
    CALIBRATION_FLOOR_GRID_LOW,
    CALIBRATION_FLOOR_GRID_STEP,
    CALIBRATION_FLOOR_IDENTITY_TOLERANCE,
    CALIBRATION_FLOOR_PROTOCOL,
    CALIBRATION_MASS_FLOOR,
    DEFAULT_CALIBRATION_N,
    DEFAULT_SEED,
    LETTER_TEXT,
    TOP_LOGPROBS,
    calibration_floor_grid,
    completion_rule_sweep,
    divergence_increases,
    divergence_sign_changes,
    fidelity_crossover,
    locate_divergence_floor,
    subsample_positions,
    unique_letter_positions,
)
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.letter import ALPHA_LIST, load_brown_counts  # noqa: E402

#: Recorded as each entry's ``source``. A repo-relative literal, never
#: ``__file__``: an absolute path pins the ledger to one machine's checkout and
#: ``manuscript_variables`` emits a ``*_SOURCE`` token per label, so a machine
#: path here is one citation away from being printed into the PDF.
SOURCE = "scripts/record_calibration_floor_claims.py"

RUN_ID = "calibration_floor_v1"

#: Positions the primary sweep scores. Matched to ``DEFAULT_CALIBRATION_N`` so
#: the located floor describes the estimator at the sample size the live
#: calibration arm actually runs at, and so the grid's top point reproduces the
#: recorded AILM calibration numbers exactly rather than approximately.
DEFAULT_POSITIONS = DEFAULT_CALIBRATION_N

#: A second sweep at four times the sample, recorded as a sensitivity check.
#: Expected calibration error is a binned statistic and binned statistics move
#: with n; if the located floor moved with n it would be a property of the
#: sample rather than of the completion rules.
DEFAULT_CHECK_POSITIONS = 4 * DEFAULT_CALIBRATION_N


def _sweep(counts: dict, positions_n: int, bins: int) -> list[dict[str, Any]]:
    """Run the offline sweep on a deterministic subsample of the letter pool."""
    pool, _unused = unique_letter_positions(LETTER_TEXT)
    positions = subsample_positions(pool, positions_n, DEFAULT_SEED)
    return completion_rule_sweep(counts, positions, bins=bins)


def _nearest_to_declared(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The grid point whose achieved captured mass sits closest to the declared floor.

    The gate compares a *measured* mean captured mass against the declared
    value, so the row that stands in for "a system sitting exactly at the floor"
    is the one whose achieved mass is nearest it — not the one whose requested
    target is 0.5, which lands somewhere above the target because k is an
    integer.
    """
    return min(rows, key=lambda row: abs(float(row["captured_mass"]) - CALIBRATION_MASS_FLOOR))


def _validate_primary_sweep(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Refuse an invalid primary sweep before any claim can be assembled.

    Keeping these checks on the measured rows makes the recorder's validation
    boundary directly reusable by offline callers.  In particular, tests can
    submit planted measurement tables to the real refusal logic without
    replacing the experiment or the claims ledger.
    """
    if not rows:
        raise SystemExit(
            "calibration-floor recorder refused: the sweep is empty. No claims written."
        )

    grid = calibration_floor_grid()
    identity = rows[-1]

    # REFUSAL 1. Truncating at full mass removes nothing, so both rules must
    # reduce to the untruncated column and agree there. Any divergence beyond
    # the documented log/exp round-trip residual means the truncation or a
    # completion rule is broken, and every number below would then describe the
    # bug rather than the floor.
    for field in ("ece_divergence", "brier_divergence", "rule_tv"):
        if abs(float(identity[field])) > CALIBRATION_FLOOR_IDENTITY_TOLERANCE:
            raise SystemExit(
                f"calibration-floor recorder refused: at target mass "
                f"{identity['target_mass']} the two completion rules differ on "
                f"{field} by {identity[field]!r}, above the identity tolerance "
                f"{CALIBRATION_FLOOR_IDENTITY_TOLERANCE}. Truncation at full mass "
                f"removes nothing, so they must agree there. No claims written."
            )

    joint = locate_divergence_floor(rows, "joint_divergence", CALIBRATION_FLOOR_EPSILON)
    ece = locate_divergence_floor(rows, "ece_divergence", CALIBRATION_FLOOR_EPSILON)
    brier = locate_divergence_floor(rows, "brier_divergence", CALIBRATION_FLOOR_EPSILON)

    # REFUSAL 2. If nothing on the grid exceeds the threshold, the located floor
    # is the grid's first point for want of any signal, and the experiment has
    # measured nothing. This is the guard against a vacuous sweep.
    if float(joint["max_divergence"]) <= CALIBRATION_FLOOR_EPSILON:
        raise SystemExit(
            f"calibration-floor recorder refused: the largest rule divergence on the "
            f"grid is {joint['max_divergence']:.6g}, at or below the declared "
            f"threshold {CALIBRATION_FLOOR_EPSILON}. Nothing was located. No claims "
            f"written."
        )

    # REFUSAL 3. The transition has to happen inside the grid. Checking that the
    # *located* index is non-zero would be a dead guard — under the sustained
    # definition a located index of zero means nothing on the grid exceeded the
    # threshold, which REFUSAL 2 has already caught. What is reachable, and what
    # actually matters, is the grid's lowest point already being negligible: the
    # sweep would then start above the crossing and report its own lower bound.
    if float(rows[0]["joint_divergence"]) <= CALIBRATION_FLOOR_EPSILON:
        raise SystemExit(
            f"calibration-floor recorder refused: at the grid's lowest target mass "
            f"({grid[0]}) the two rules already agree to within "
            f"{CALIBRATION_FLOOR_EPSILON}, so the sweep begins above the transition "
            f"and would report its own lower bound. Lower CALIBRATION_FLOOR_GRID_LOW. "
            f"No claims written."
        )

    return joint, ece, brier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Locate the captured-mass floor offline.")
    parser.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--check-positions", type=int, default=DEFAULT_CHECK_POSITIONS)
    parser.add_argument("--bins", type=int, default=CALIBRATION_BINS)
    args = parser.parse_args(argv)
    # The ledger is single-writer, and this recorder shares the bench_ namespace
    # with two others. Protecting one of three contending writers is not mutual
    # exclusion.
    with single_writer_lock(ledger_lock_path(REPO_ROOT)):
        return _measure_and_record(args)


def _measure_and_record(args: argparse.Namespace) -> int:
    counts = load_brown_counts(REPO_ROOT / "vendor")
    rows = _sweep(counts, args.positions, args.bins)
    grid = calibration_floor_grid()
    identity = rows[-1]
    joint, ece, brier = _validate_primary_sweep(rows)

    at_declared = _nearest_to_declared(rows)
    located = rows[int(joint["located_index"])]
    crossover = fidelity_crossover(rows)
    check_rows = _sweep(counts, args.check_positions, args.bins)
    check = locate_divergence_floor(check_rows, "joint_divergence", CALIBRATION_FLOOR_EPSILON)

    values: dict[str, Any] = {
        # Protocol.
        "bench_floor_protocol_version": CALIBRATION_FLOOR_PROTOCOL,
        "bench_floor_positions": args.positions,
        "bench_floor_seed": DEFAULT_SEED,
        "bench_floor_bins": args.bins,
        "bench_floor_inventory_size": len(ALPHA_LIST),
        "bench_floor_cap": TOP_LOGPROBS,
        "bench_floor_grid_low": CALIBRATION_FLOOR_GRID_LOW,
        "bench_floor_grid_high": CALIBRATION_FLOOR_GRID_HIGH,
        "bench_floor_grid_step": CALIBRATION_FLOOR_GRID_STEP,
        "bench_floor_grid_points": len(grid),
        "bench_floor_epsilon": CALIBRATION_FLOOR_EPSILON,
        "bench_floor_identity_tolerance": CALIBRATION_FLOOR_IDENTITY_TOLERANCE,
        "bench_floor_declared_floor": CALIBRATION_MASS_FLOOR,
        # The identity point. Both rules reduce to the untruncated column here,
        # and the resulting figures are the live benchmark's recorded AILM
        # calibration numbers on the same positions — an external check that the
        # offline experiment scores the same thing the benchmark scores.
        "bench_floor_identity_ece": identity["min_bound_ece"],
        "bench_floor_identity_brier": identity["min_bound_brier"],
        "bench_floor_identity_rule_tv": identity["rule_tv"],
        "bench_floor_max_rule_tv": max(float(row["rule_tv"]) for row in rows),
        "bench_floor_max_joint_divergence": joint["max_divergence"],
        # Where the located floor lands, and how it compares to the declared one.
        "bench_floor_located_target": located["target_mass"],
        "bench_floor_located_captured": located["captured_mass"],
        "bench_floor_located_k": located["mean_k"],
        "bench_floor_located_over_cap_fraction": located["over_cap_fraction"],
        "bench_floor_located_rule_tv": located["rule_tv"],
        "bench_floor_located_ratio": (float(located["captured_mass"]) / CALIBRATION_MASS_FLOOR),
        "bench_floor_located_ece_target": ece["located_target"],
        "bench_floor_located_ece_captured": ece["located_captured"],
        "bench_floor_located_brier_target": brier["located_target"],
        "bench_floor_located_brier_captured": brier["located_captured"],
        # What the reported figures do at the declared floor itself.
        "bench_floor_atdeclared_target": at_declared["target_mass"],
        "bench_floor_atdeclared_captured": at_declared["captured_mass"],
        "bench_floor_atdeclared_k": at_declared["mean_k"],
        "bench_floor_atdeclared_ece_divergence": at_declared["ece_divergence"],
        "bench_floor_atdeclared_brier_divergence": at_declared["brier_divergence"],
        "bench_floor_atdeclared_rule_tv": at_declared["rule_tv"],
        "bench_floor_atdeclared_minbound_tv_truth": at_declared["min_bound_tv_truth"],
        "bench_floor_atdeclared_observed_tv_truth": at_declared["observed_only_tv_truth"],
        # The trap: the first crossing, which a naive rule would report.
        "bench_floor_naive_ece_target": ece["naive_target"],
        "bench_floor_naive_ece_captured": ece["naive_captured"],
        "bench_floor_naive_ece_rule_tv": rows[int(ece["naive_index"])]["rule_tv"],
        "bench_floor_naive_brier_target": brier["naive_target"],
        "bench_floor_naive_brier_captured": brier["naive_captured"],
        "bench_floor_naive_brier_rule_tv": rows[int(brier["naive_index"])]["rule_tv"],
        # Shape of each divergence in captured mass.
        "bench_floor_tv_increases": divergence_increases(rows, "rule_tv"),
        "bench_floor_ece_increases": divergence_increases(rows, "ece_divergence"),
        "bench_floor_brier_increases": divergence_increases(rows, "brier_divergence"),
        "bench_floor_ece_sign_changes": divergence_sign_changes(rows, "ece_signed"),
        "bench_floor_brier_sign_changes": divergence_sign_changes(rows, "brier_signed"),
        # Sample-size sensitivity of the located floor.
        "bench_floor_check_positions": args.check_positions,
        "bench_floor_check_located_target": check["located_target"],
        "bench_floor_check_located_captured": check["located_captured"],
    }

    # Which rule reconstructs the truth better is not fixed across the grid, and
    # a crossover is a statement about the shipped rule rather than about the
    # floor. Absence of a crossover is equally a result, so the label records
    # the grid's top point in that case and a companion flag says which happened.
    values["bench_floor_fidelity_crossover_exists"] = int(crossover is not None)
    values["bench_floor_fidelity_crossover_target"] = (
        crossover["target_mass"] if crossover is not None else grid[-1]
    )
    values["bench_floor_fidelity_crossover_captured"] = (
        crossover["captured_mass"] if crossover is not None else rows[-1]["captured_mass"]
    )

    ledger = load_ledger(REPO_ROOT)
    stale = sorted(label for label in ledger.labels_owned_by(SOURCE) if label not in values)
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
    )

    print(
        f"recorded {len(values)} calibration-floor claims over {args.positions} positions "
        f"and {len(grid)} grid points"
    )
    print(
        f"  declared floor {CALIBRATION_MASS_FLOOR}: rules differ by "
        f"{float(at_declared['ece_divergence']):.4f} ECE, "
        f"{float(at_declared['brier_divergence']):.4f} Brier, "
        f"{float(at_declared['rule_tv']):.4f} total variation"
    )
    print(
        f"  located floor {float(located['captured_mass']):.4f} captured mass "
        f"(target {located['target_mass']}), needing {float(located['mean_k']):.1f} of "
        f"{len(ALPHA_LIST)} candidates against a server cap of {TOP_LOGPROBS}"
    )
    print(
        f"  first crossing at {float(ece['naive_captured']):.4f} captured mass, where the "
        f"two rules still differ by "
        f"{float(values['bench_floor_naive_ece_rule_tv']):.4f} in total variation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
