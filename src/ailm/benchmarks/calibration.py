"""Reconstructing a distribution from truncated top-k mass, and the floor that creates."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any, Callable

from ailm.benchmarks.constants import (
    CALIBRATION_BINS,
    CALIBRATION_FLOOR_EPSILON,
    CALIBRATION_FLOOR_GRID_HIGH,
    CALIBRATION_FLOOR_GRID_LOW,
    CALIBRATION_FLOOR_GRID_STEP,
    CALIBRATION_MASS_FLOOR,
    DEFAULT_SEED,
    TOP_LOGPROBS,
)
from ailm.benchmarks.corpora import letter_posterior
from ailm.benchmarks.ollama import (
    _distribution_from_candidates,
    ask_ollama_answer_candidates,
    captured_mass,
    observed_symbol_mass,
)
from ailm.benchmarks.prompts import build_prompt, parse_letter_answer
from ailm.benchmarks.scoring import brier_score, expected_calibration_error, negative_log_likelihood
from ailm.levels.letter import ALPHA_LIST


def reconstructed_calibration_metrics(
    model: str,
    brown: dict,
    positions: list[tuple[str, str]],
    *,
    seed: int = DEFAULT_SEED,
    bins: int = CALIBRATION_BINS,
    top_logprobs: int = TOP_LOGPROBS,
    answer_candidates: Callable[[str, str, str, int, int, int], tuple[str, list[dict[str, Any]]]]
    | None = None,
) -> dict[str, Any]:
    """Calibration scores for the probe and the comparator on identical positions.

    The name carries the caveat because the caveat decides how the numbers may
    be read. Both systems are scored on a distribution over the same declared
    thirty-symbol inventory, but the two distributions do not have the same
    standing. The comparator's is exact: :func:`letter_posterior` returns the
    predictive column it decodes from, with only a float64 unit-mass
    normalization that preserves its argmax. The probe's is a
    reconstruction: the server returns at most ``top_logprobs`` raw next-token
    candidates for an inventory of thirty symbols, most of that mass sits on
    tokens the response grammar forbids and is discarded, every inventory symbol
    missing from the truncated list is assigned the smallest reported
    probability as a bound, and the result is renormalized
    (:func:`_distribution_from_candidates`). Whatever comes out is not the
    probe's constrained output distribution, and no request to this endpoint
    returns that distribution.

    So the scores below are proper scores of a reconstructed inventory
    distribution. That is a real quantity and a comparable one — the same
    reconstruction rule is applied at every position of every probe — but it is
    not the probe's own predictive distribution, and calling this the axis on
    which the systems are *directly* commensurable would overstate it: the
    comparator is scored on what it computes and the probe on what could be
    recovered of what it computed. Two things are recorded so a reader can size
    the gap rather than take it on assurance: ``captured_mass``, the share of
    each reported distribution that was observed rather than assumed, and
    ``rule_check``, the direct measurement of how far the reported metrics move
    when the completion rule is swapped for the other defensible one.

    The probe's predicted class is the answer it actually returned, not the
    argmax of the reconstructed distribution. The two are not the same thing.
    ``top_logprobs`` reports the *raw* next-token distribution while generation
    is grammar-constrained, so the constrained decoder can be pushed off its raw
    argmax, and it can be pushed onto a token outside the reported top-k
    entirely. Scoring the reconstruction's argmax would make the calibration
    accuracy disagree with the accuracy the benchmark recorded — a calibration
    number for a decision that was never made. The rate at which the two agree
    is measured and recorded rather than assumed, because it bounds how far the
    reported distribution describes the decode it came from.
    """
    candidate_call = (
        ask_ollama_answer_candidates if answer_candidates is None else answer_candidates
    )
    model_confidences: list[float] = []
    model_correct: list[bool] = []
    model_distributions: list[dict[str, float]] = []
    comparator_confidences: list[float] = []
    comparator_correct: list[bool] = []
    comparator_distributions: list[dict[str, float]] = []
    golds: list[str] = []
    captured: list[float] = []
    observations: list[dict[str, float]] = []
    predictions: list[str] = []
    unlocated = 0
    argmax_agrees = 0
    unparsable = 0
    for prefix, gold in positions:
        # ``ask_ollama_distribution`` is unrolled into its two steps rather than
        # called: the direct rule check needs this position's own candidate list,
        # and asking the server a second time for it would compare two decodes.
        # The distribution and the captured mass below are that function's body,
        # so nothing about the reported numbers changes.
        text, candidates = candidate_call(
            model, build_prompt("letter", prefix), "letter", 32, seed, top_logprobs
        )
        distribution = _distribution_from_candidates(candidates)
        observed_mass = captured_mass(candidates)
        observations.append(observed_symbol_mass(candidates))
        captured.append(observed_mass)
        unlocated += int(observed_mass == 0.0)
        top = max(distribution, key=lambda symbol: (distribution[symbol], symbol))
        answer, bad = parse_letter_answer(text)
        unparsable += int(bad)
        # An unparsable reply has no scored class, so the reconstruction's
        # argmax is the only thing left to score; that case is counted.
        predicted = top if bad or answer is None else answer
        argmax_agrees += int((not bad) and top == answer)
        predictions.append(predicted)
        model_distributions.append(distribution)
        model_confidences.append(distribution.get(predicted, 0.0))
        model_correct.append(predicted == gold)
        posterior = letter_posterior(brown, prefix)
        best = max(posterior, key=lambda symbol: (posterior[symbol], symbol))
        comparator_distributions.append(posterior)
        comparator_confidences.append(posterior[best])
        comparator_correct.append(best == gold)
        golds.append(gold)
    mean_captured = sum(captured) / len(captured) if captured else 0.0
    ordered = sorted(captured)
    median_captured = (
        0.0
        if not ordered
        else (
            ordered[len(ordered) // 2]
            if len(ordered) % 2
            else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
        )
    )
    # The captured share is bimodal in practice: most positions are near-fully
    # observed and a minority are near-empty. A mean over those two populations
    # describes neither, so the median and the starved share are recorded too —
    # the starved share is what tells a reader how much of a published
    # calibration number rests on positions that were mostly reconstructed.
    starved = sum(1 for value in captured if value < CALIBRATION_MASS_FLOOR)
    # The measurability decision is the direct dual-rule check (decision 93),
    # not the captured-mass floor: the sweep measured the floor's proxy as not
    # conservative, and the per-model divergence is the quantity the proxy
    # stood in for. The check is computed for every probe regardless of its
    # verdict, so a withheld calibration is still an audited one.
    rule_check = dual_rule_calibration(
        model_distributions, observations, predictions, model_correct, golds, bins=bins
    )
    return {
        "n": len(positions),
        "bins": bins,
        "top_logprobs": top_logprobs,
        "captured_mass": mean_captured,
        "captured_median": median_captured,
        "starved_fraction": starved / len(captured) if captured else 0.0,
        "mass_floor": CALIBRATION_MASS_FLOOR,
        # Retired from gating (decision 93): the floor is recorded as
        # provenance, and the gate is the rule check below.
        "measurable": rule_check["agrees"],
        # The direct check on this model's own reported distributions. Computed
        # for every probe, admitted or withheld.
        "rule_check": rule_check,
        "unlocated": unlocated,
        "unparsable": unparsable,
        "argmax_agreement": (
            argmax_agrees / (len(positions) - unparsable) if len(positions) - unparsable else 0.0
        ),
        "model_ece": expected_calibration_error(model_confidences, model_correct, bins),
        "model_brier": brier_score(model_distributions, golds),
        "model_nll": negative_log_likelihood(model_distributions, golds),
        "model_confidence": (
            sum(model_confidences) / len(model_confidences) if model_confidences else 0.0
        ),
        "model_top1_accuracy": (sum(model_correct) / len(model_correct) if model_correct else 0.0),
        "comparator_ece": expected_calibration_error(
            comparator_confidences, comparator_correct, bins
        ),
        "comparator_brier": brier_score(comparator_distributions, golds),
        "comparator_nll": negative_log_likelihood(comparator_distributions, golds),
        "comparator_confidence": (
            sum(comparator_confidences) / len(comparator_confidences)
            if comparator_confidences
            else 0.0
        ),
        "comparator_top1_accuracy": (
            sum(comparator_correct) / len(comparator_correct) if comparator_correct else 0.0
        ),
    }


# ── Locating the captured-mass floor offline ─────────────────────────────────
#
# ``CALIBRATION_MASS_FLOOR`` is principled but was never located. Everything
# below measures where it belongs, using no model call and no network: AILM's
# own letter transition columns are real next-symbol distributions over the same
# thirty-symbol inventory, so they can be truncated the way ``top_logprobs``
# truncates a probe and reconstructed under each completion rule the project has
# available. The quantity of interest is not how well either rule recovers the
# truth — it is how far apart the two rules are, because that gap is exactly
# "the completion rule rather than the model decides the reported number".


def _argmax_symbol(distribution: dict[str, float]) -> str:
    """Deterministic argmax over a symbol distribution, ties broken by symbol.

    Dict insertion order would otherwise decide a tie, which makes a sweep
    result depend on how the column happened to be built rather than on the
    column.
    """
    return max(distribution, key=lambda symbol: (distribution[symbol], symbol))


def calibration_floor_grid(
    low: float = CALIBRATION_FLOOR_GRID_LOW,
    high: float = CALIBRATION_FLOOR_GRID_HIGH,
    step: float = CALIBRATION_FLOOR_GRID_STEP,
) -> tuple[float, ...]:
    """Target captured masses for the floor sweep, inclusive of both ends.

    Built from integer index arithmetic and rounded. Accumulating ``low + step``
    in floating point puts values like 0.44999999999999996 on the axis, and a
    grid point that differs in its last bit changes which point a threshold
    crossing is attributed to — the located floor would then depend on the order
    the additions happened in.
    """
    if not 0.0 < low <= high <= 1.0:
        raise ValueError(f"need 0 < low <= high <= 1, got low={low}, high={high}")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    count = int(round((high - low) / step))
    if abs(low + count * step - high) > 1e-9:
        raise ValueError(f"step {step} does not divide the span {low} to {high}")
    return tuple(round(low + index * step, 10) for index in range(count + 1))


def truncate_to_mass(
    distribution: dict[str, float], target_mass: float
) -> tuple[dict[str, float], float]:
    """The smallest top-k prefix of ``distribution`` whose mass reaches ``target_mass``.

    This is the truncation ``top_logprobs`` imposes, expressed in the units the
    floor is stated in. The server reports the k most probable candidates and
    nothing else, so the tail is not small-but-observed, it is absent. Indexing
    the sweep by target mass rather than by k is what makes it an experiment
    about captured mass rather than about one server's candidate limit.

    Returns the truncated (unnormalized) distribution and the mass it captures.
    That mass is at least ``target_mass`` and usually overshoots, because k is an
    integer and the last symbol admitted carries whatever it carries; the
    overshoot is returned rather than hidden so the sweep can report the mass it
    actually reached.
    """
    if not 0.0 < target_mass <= 1.0:
        raise ValueError(f"target_mass must be in (0, 1], got {target_mass}")
    if not distribution:
        raise ValueError("cannot truncate an empty distribution")
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    truncated: dict[str, float] = {}
    captured = 0.0
    for symbol, probability in ordered:
        truncated[symbol] = probability
        captured += probability
        # A normalized column does not sum to exactly 1.0 — the transition
        # matrix is built in float32, so the residual is around 1e-9 — and an
        # exact comparison would let a rounding difference decide whether the
        # last symbol survives at a target the column can only just reach. The
        # slack is well below any real probability in the inventory, so it can
        # never admit a symbol that the target did not already reach.
        if captured >= target_mass - 1e-12:
            break
    return truncated, captured


def complete_min_bound(observed: dict[str, float]) -> dict[str, float]:
    """Complete a truncated report under the rule this benchmark ships.

    The arithmetic is not restated here. The observed symbols are packed back
    into the candidate payload the server would have sent and passed through
    :func:`_distribution_from_candidates`, so the sweep measures the shipped rule
    itself rather than a second copy of it that could drift away from the one
    the published calibration numbers were produced by.
    """
    for symbol, probability in observed.items():
        if probability <= 0.0:
            raise ValueError(
                f"symbol {symbol!r} has non-positive probability {probability}. The "
                "min-bound rule takes the smallest reported probability as its bound "
                "for every absent symbol, so a zero would silently collapse it into "
                "the observed-only rule and the two rules would agree for the wrong "
                "reason."
            )
    candidates: list[dict[str, Any]] = [
        {"token": symbol, "logprob": math.log(probability)}
        for symbol, probability in observed.items()
    ]
    return _distribution_from_candidates(candidates)


def complete_observed_only(observed: dict[str, float]) -> dict[str, float]:
    """Complete a truncated report by renormalizing the survivors and nothing else.

    The alternative rule, and the strictest reading of "report only what was
    observed": a symbol absent from the truncated list is assigned no
    probability at all. It is not a straw man. It is what any reader who is told
    "these are the reported probabilities, renormalized" would assume, and the
    distance between it and the shipped rule is the quantity the captured-mass
    floor exists to bound.
    """
    mass = sum(observed.values())
    if mass <= 0.0:
        raise ValueError("cannot renormalize a truncated report with no mass")
    return {symbol: observed.get(symbol, 0.0) / mass for symbol in ALPHA_LIST}


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    """Total variation distance between two distributions over the inventory.

    Half the L1 gap, so it lands on [0, 1] and reads as the largest probability
    the two distributions assign differently to any one event. It is the sweep's
    rule-versus-rule divergence because it needs no gold labels: two
    reconstructions can post the same expected calibration error while being
    nearly disjoint, and a comparison made only in the reported metrics cannot
    see that.
    """
    return 0.5 * sum(abs(left.get(s, 0.0) - right.get(s, 0.0)) for s in ALPHA_LIST)


def completion_rule_sweep(
    counts: dict,
    positions: list[tuple[str, str]],
    *,
    grid: tuple[float, ...] | None = None,
    bins: int = CALIBRATION_BINS,
    cap: int = TOP_LOGPROBS,
) -> list[dict[str, Any]]:
    """Score both completion rules across a grid of target captured masses.

    Ground truth is AILM's own order-1 letter posterior at each position — the
    exact column :func:`letter_posterior` returns, which the live benchmark
    already scores as the comparator. Those columns are real next-symbol
    distributions over the scored inventory, they are deterministic, and they
    are available with no model call, which is what lets this experiment cover
    the captured-mass region no live system landed in.

    At each grid point the column is truncated to the smallest top-k reaching the
    target mass and reconstructed twice, once under each rule. Both
    reconstructions are then scored against the same gold symbols with the same
    two metrics the benchmark reports.

    The predicted class is the argmax of the **untruncated** column and is held
    fixed across the whole grid. Truncation is a reporting artifact, not a
    decision artifact: the system decoded from its full distribution and only the
    report was capped. Letting the prediction follow the reconstruction would
    make the sweep measure a decision no system made, which is the same defect
    :func:`reconstructed_calibration_metrics` avoids by scoring the parsed
    answer rather than the reconstruction's argmax.

    Returns one row per grid point. ``rule_tv`` is the divergence between the two
    rules; ``ece_divergence`` and ``brier_divergence`` are their disagreement in
    the reported metrics; the ``*_tv_truth`` columns say how far each rule is
    from the column it is trying to reconstruct.
    """
    grid_values = calibration_floor_grid() if grid is None else grid
    if not grid_values:
        raise ValueError("the sweep grid must be non-empty")
    if not positions:
        raise ValueError("the sweep needs at least one scored position")

    # One posterior per distinct conditioning symbol. ``letter_posterior``
    # rebuilds the whole transition matrix per call, so calling it per position
    # would rebuild it hundreds of times for at most thirty distinct answers.
    cache: dict[str, dict[str, float]] = {}
    truths: list[dict[str, float]] = []
    predictions: list[str] = []
    golds: list[str] = []
    for prefix, gold in positions:
        key = prefix[-1] if prefix else ""
        if key not in cache:
            cache[key] = letter_posterior(counts, key)
        column = cache[key]
        truths.append(column)
        predictions.append(_argmax_symbol(column))
        golds.append(gold)
    # Correctness is a property of the decode and the gold symbol, so it is the
    # same at every grid point. Only the stated confidence moves — which is
    # precisely the channel the floor exists to control.
    correct = [p == g for p, g in zip(predictions, golds, strict=True)]

    rows: list[dict[str, Any]] = []
    for target in grid_values:
        captured: list[float] = []
        sizes: list[int] = []
        min_bound: list[dict[str, float]] = []
        observed_only: list[dict[str, float]] = []
        for column in truths:
            truncated, mass = truncate_to_mass(column, target)
            captured.append(mass)
            sizes.append(len(truncated))
            min_bound.append(complete_min_bound(truncated))
            observed_only.append(complete_observed_only(truncated))
        total = len(truths)
        mb_confidence = [d.get(p, 0.0) for d, p in zip(min_bound, predictions, strict=True)]
        oo_confidence = [d.get(p, 0.0) for d, p in zip(observed_only, predictions, strict=True)]
        mb_ece = expected_calibration_error(mb_confidence, correct, bins)
        oo_ece = expected_calibration_error(oo_confidence, correct, bins)
        mb_brier = brier_score(min_bound, golds)
        oo_brier = brier_score(observed_only, golds)
        rows.append(
            {
                "target_mass": target,
                "captured_mass": sum(captured) / total,
                "mean_k": sum(sizes) / total,
                "over_cap_fraction": sum(1 for k in sizes if k > cap) / total,
                "min_bound_ece": mb_ece,
                "observed_only_ece": oo_ece,
                "ece_signed": mb_ece - oo_ece,
                "ece_divergence": abs(mb_ece - oo_ece),
                "min_bound_brier": mb_brier,
                "observed_only_brier": oo_brier,
                "brier_signed": mb_brier - oo_brier,
                "brier_divergence": abs(mb_brier - oo_brier),
                "joint_divergence": max(abs(mb_ece - oo_ece), abs(mb_brier - oo_brier)),
                "rule_tv": sum(
                    total_variation(a, b) for a, b in zip(min_bound, observed_only, strict=True)
                )
                / total,
                "min_bound_tv_truth": sum(
                    total_variation(a, t) for a, t in zip(min_bound, truths, strict=True)
                )
                / total,
                "observed_only_tv_truth": sum(
                    total_variation(b, t) for b, t in zip(observed_only, truths, strict=True)
                )
                / total,
            }
        )
    return rows


def locate_divergence_floor(rows: list[dict[str, Any]], key: str, epsilon: float) -> dict[str, Any]:
    """Smallest grid point above which ``key`` never again exceeds ``epsilon``.

    The floor is defined by a *sustained* crossing rather than a first crossing,
    and that distinction is the finding rather than a technicality. The two
    completion rules bias a reported metric in opposite directions — min-bound
    flattens the distribution and understates confidence, observed-only sharpens
    it and overstates confidence — so their difference passes through zero
    somewhere in the middle of the grid while the two reconstructions are still
    nearly disjoint. A first-crossing rule reads that coincidence as convergence
    and locates a floor far below the point where the completion stops deciding
    the number.

    Both are returned. ``naive_*`` is the first crossing and is reported so the
    size of the trap is visible next to the answer, not so it can be used.

    ``increases`` counts grid steps on which the divergence *rises* as more mass
    is captured. A well-behaved divergence has none; a non-zero count is a
    finding about the metric, and it is recorded rather than asserted away.
    """
    if not rows:
        raise ValueError("cannot locate a floor in an empty sweep")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    values = [float(row[key]) for row in rows]

    located = 0
    for index in range(len(values) - 1, -1, -1):
        if values[index] > epsilon:
            located = index + 1
            break
    if located >= len(rows):
        raise ValueError(
            f"{key} exceeds epsilon {epsilon} at the top of the grid "
            f"(target mass {rows[-1]['target_mass']}), so no floor is locatable. "
            "Truncation at full mass must be the identity; a non-zero divergence "
            "there means the truncation or a completion rule is broken."
        )

    naive = next(index for index, value in enumerate(values) if value <= epsilon)
    return {
        "key": key,
        "epsilon": epsilon,
        "located_index": located,
        "located_target": rows[located]["target_mass"],
        "located_captured": rows[located]["captured_mass"],
        "naive_index": naive,
        "naive_target": rows[naive]["target_mass"],
        "naive_captured": rows[naive]["captured_mass"],
        "max_divergence": max(values),
        "increases": divergence_increases(rows, key),
    }


def divergence_increases(rows: list[dict[str, Any]], key: str) -> int:
    """Grid steps on which ``key`` rises as more probability mass is captured.

    Zero means the divergence is monotone non-increasing in captured mass, which
    is the shape a floor argument assumes without saying so. A non-zero count is
    a finding about that metric — reported, not asserted away.
    """
    values = [float(row[key]) for row in rows]
    # ``pairwise``, never ``zip(values, values[1:], strict=True)``: a self-offset
    # zip has arguments of unequal length by construction and raises on every
    # non-empty input (tests/test_no_self_offset_strict_zip.py).
    return sum(1 for a, b in pairwise(values) if b > a)


def divergence_sign_changes(rows: list[dict[str, Any]], key: str) -> int:
    """How often a signed rule difference changes sign across the grid.

    A sign change is a point where the two completion rules cross. It is the
    mechanism that makes a first-crossing floor unsound, so it is counted and
    recorded rather than described.
    """
    signs = [1 if float(row[key]) > 0 else (-1 if float(row[key]) < 0 else 0) for row in rows]
    nonzero = [s for s in signs if s != 0]
    return sum(1 for a, b in pairwise(nonzero) if a != b)


def fidelity_crossover(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First grid point where observed-only reconstructs the truth better than min-bound.

    The shipped rule is conservative in the direction it was designed for — it
    can overstate an unreported symbol, never understate it — and that is a
    statement about the bound, not about accuracy. Whether the bound is also the
    closer reconstruction is a separate question with a measurable answer, and
    the answer turns out to depend on how much mass was captured.

    Returns ``None`` when one rule dominates across the whole grid, which is
    itself the result in that case.
    """
    for index, row in enumerate(rows):
        if row["observed_only_tv_truth"] < row["min_bound_tv_truth"]:
            return {
                "index": index,
                "target_mass": row["target_mass"],
                "captured_mass": row["captured_mass"],
            }
    return None


# ── The same rule comparison, on a live probe's own reported candidates ──────
#
# The sweep above measures rule-dependence on AILM's columns and indexes it by
# captured mass, which makes captured mass a *proxy* for rule-dependence — and
# the sweep found the proxy is not conservative at the declared floor. What
# licenses a reported calibration number is not the proxy but the thing the
# proxy stands in for, measured on the model whose number it is: apply both
# completion rules to that model's own answer-slot candidates and see whether
# the reported metrics move. Below is that check. It makes no model call; the
# candidates were already obtained by the calibration arm.


def dual_rule_calibration(
    reconstructions: list[dict[str, float]],
    observations: list[dict[str, float]],
    predictions: list[str],
    correct: list[bool],
    golds: list[str],
    *,
    bins: int = CALIBRATION_BINS,
    epsilon: float = CALIBRATION_FLOOR_EPSILON,
) -> dict[str, Any]:
    """Score one probe's positions under both completion rules and compare.

    Guards against a reported calibration number that is a property of the
    reconstruction rather than of the model. Truncation at ``top_logprobs``
    leaves every reported distribution partly assumed, and the two available
    completions bias the result in opposite directions — min-bound flattens the
    column and understates confidence, observed-only sharpens it and overstates
    confidence. If the published metrics move materially between them, the
    number describes the completion rule; if they do not, it describes the
    model. Nothing else in the benchmark can tell those apart, and the
    captured-mass gate that stands in for this check was measured by
    :func:`completion_rule_sweep` and found not to be conservative.

    ``reconstructions`` are the shipped min-bound distributions the calibration
    arm already scored, passed in rather than rebuilt from ``observations``:
    rebuilding would send every probability through a second ``log``/``exp``
    round trip, and the min-bound figures below would then differ in their last
    bits from the ones the ledger publishes. Held identical, they *are* the
    published figures, so the divergence is stated against the number it is
    about.

    The predicted class is held fixed across both rules, exactly as
    :func:`completion_rule_sweep` holds it fixed across the grid and as
    :func:`reconstructed_calibration_metrics` scores the parsed answer rather
    than the reconstruction's argmax. The decode happened once, off the model's
    full distribution; only the report was truncated, so only the stated
    confidence may move. Letting the class follow the reconstruction would
    compare two decisions no system made.

    A position whose candidates put no mass on the inventory carries no
    information for either rule, so both fall back to the same uniform column
    and it contributes exactly zero divergence. Those positions are counted and
    returned as ``degenerate`` rather than left to flatter the mean: a probe
    whose slot is never located would otherwise report perfect rule agreement.

    Returns the metrics under each rule, their signed and absolute divergences,
    the label-free total variation between the two reconstructions, and whether
    the larger metric divergence stays within ``epsilon``. Keys match the rows
    of :func:`completion_rule_sweep` so a probe and the sweep can be read on one
    scale.

    Raises:
        ValueError: If there are no positions. Every metric here is defined as
            zero on an empty input, so an empty check would report a probe as
            perfectly rule-insensitive on no evidence at all — the exact reading
            this measurement exists to make impossible.
    """
    if not reconstructions:
        raise ValueError(
            "the direct rule check needs at least one scored position; on none of "
            "them every divergence is zero by definition and the probe would be "
            "reported rule-insensitive without a single measurement"
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    observed_only: list[dict[str, float]] = []
    degenerate = 0
    for reconstruction, observed in zip(reconstructions, observations, strict=True):
        # ``complete_observed_only`` cannot renormalize nothing, and there is
        # nothing to renormalize: no candidate landed on the inventory. The
        # shipped rule answers the uniform column there, so the alternative rule
        # answers it too and the two agree for the honest reason.
        if sum(observed.values()) <= 0.0:
            degenerate += 1
            observed_only.append(reconstruction)
            continue
        observed_only.append(complete_observed_only(observed))

    min_bound_confidence = [
        distribution.get(predicted, 0.0)
        for distribution, predicted in zip(reconstructions, predictions, strict=True)
    ]
    observed_only_confidence = [
        distribution.get(predicted, 0.0)
        for distribution, predicted in zip(observed_only, predictions, strict=True)
    ]
    min_bound_ece = expected_calibration_error(min_bound_confidence, correct, bins)
    observed_only_ece = expected_calibration_error(observed_only_confidence, correct, bins)
    min_bound_brier = brier_score(reconstructions, golds)
    observed_only_brier = brier_score(observed_only, golds)
    ece_divergence = abs(min_bound_ece - observed_only_ece)
    brier_divergence = abs(min_bound_brier - observed_only_brier)
    total = len(reconstructions)
    return {
        "n": total,
        "degenerate": degenerate,
        "epsilon": epsilon,
        "bins": bins,
        "min_bound_ece": min_bound_ece,
        "observed_only_ece": observed_only_ece,
        "ece_signed": min_bound_ece - observed_only_ece,
        "ece_divergence": ece_divergence,
        "min_bound_brier": min_bound_brier,
        "observed_only_brier": observed_only_brier,
        "brier_signed": min_bound_brier - observed_only_brier,
        "brier_divergence": brier_divergence,
        "joint_divergence": max(ece_divergence, brier_divergence),
        # Label-free, so it sees what the metrics cannot: two reconstructions can
        # post the same expected calibration error while being nearly disjoint,
        # and a small metric divergence on a large total variation is a
        # coincidence of opposite biases rather than agreement.
        "rule_tv": sum(
            total_variation(left, right)
            for left, right in zip(reconstructions, observed_only, strict=True)
        )
        / total,
        "agrees": max(ece_divergence, brier_divergence) <= epsilon,
    }
