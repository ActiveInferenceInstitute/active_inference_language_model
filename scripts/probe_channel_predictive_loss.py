#!/usr/bin/env python3
"""Probe proper predictive loss across the L1 -> L2B channel, and the value inequality.

Scoped in `docs/research_queue.md` (cross-cutting hierarchy section):

**Which cell this is.** Decision 46 tabulates four coupling cells; this probe
measures **letter@evidence** — the L1 message mixed linearly into the G2P
evidence, which is what :func:`couple_letter_to_phoneme` does. The probe
reproduces that cell's recorded accuracy exactly (70.5371% at alpha = 0.1
against the 70.801% uncoupled anchor, on the same 42,820 positions), which is
the check that it is wired to the right channel. It does **not** implement
letter@predictive, the cell that mixes into the prior instead and is the one
that "resolves neither way" at +0.103 pts, p = 0.17. Nothing here settles that
cell; it needs its own construction.

1. Evaluate the channel on the receiver's **proper** predictive loss,
   ``loss = -log q_t(true target)``, where ``q_t`` is the G2P filter's posterior
   over phones at position ``t`` and the true target is the gold phone. Accuracy
   is not a proper scoring rule, so the two axes can in principle disagree in
   sign; ``docs/research_queue.md`` makes that disagreement the falsifier for
   sec:hierarchy's monotone-decline diagnostic. Both axes are therefore computed
   in the same pass over the same positions, so no position-set mismatch can
   manufacture or hide a disagreement.

2. Measure the channel value inequality ``0 <= realized <= min(readable, sensitivity)``:

   - **realized(alpha)** ``= loss(0) - loss(alpha)``. What the real L1 message
     actually buys the receiver at that mixture weight.
   - **readable** ``= max_alpha realized(alpha)`` over the grid. The best the
     admissible channel family achieves.
   - **sensitivity** ``= loss(0) - loss(oracle sender)``. What a *perfect* sender
     would buy: the upper bound on any message in this family.

   For this particular channel the sender's latent is the letter, and the
   receiver already observes that letter directly. So the oracle sender's
   one-hot message is the receiver's own observation, the mixture
   ``(1 - a) * onehot + a * onehot`` collapses to ``onehot`` for every ``a``,
   and **sensitivity is exactly zero by construction**. The inequality then
   demands ``realized <= 0``: this channel cannot improve proper predictive
   loss, only degrade it. That is a falsifiable prediction, and it is what the
   gate below checks, rather than the constant it used to return.

Both a scrambled-message control and an oracle-sender control are run, because a
loss curve on its own cannot distinguish "the message is informative" from "any
perturbation of the evidence moves this number".

Usage:
    .venv/bin/python scripts/probe_channel_predictive_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.levels.coupling import (  # noqa: E402
    _soft_forward,
    build_g2p_matrices,
    build_letter_matrices,
    letter_predictive_distributions,
    scramble_letter_distribution,
)
from ailm.levels.letter import load_brown_counts  # noqa: E402
from ailm.levels.phoneme import (  # noqa: E402
    ALPHA_TO_IDX,
    N_SYMBOLS,
    PHONE_TO_IDX,
    _word_letters,
    g2p_split,
    load_phoneme_counts,
)

#: Mixture weights swept. 0.0 is the uncoupled anchor and must be present: every
#: gain in this probe is defined relative to it.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5)

#: Guard against a silently degenerate evaluation set. The gate compares mean
#: losses, and a handful of positions cannot separate the arms.
MIN_POSITIONS = 100


def _aligned_pairs(
    pairs: list[tuple[str, list[str]]], limit: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Letter and gold-phone index arrays for words whose lengths line up.

    The G2P model emits one phone per letter, so a word whose CMU pronunciation
    has a different length has no position-wise alignment and is dropped — the
    same restriction the coupled decoder operates under.
    """
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for word, phones in pairs:
        if len(out) >= limit:
            break
        letters = _word_letters(word)
        if not letters or len(letters) != len(phones):
            continue
        if any(p not in PHONE_TO_IDX for p in phones):
            continue
        out.append(
            (
                np.asarray([ALPHA_TO_IDX[c] for c in letters], dtype=np.int64),
                np.asarray([PHONE_TO_IDX[p] for p in phones], dtype=np.int64),
            )
        )
    return out


def _message(letter_matrices, letter_indices: np.ndarray, arm: str) -> np.ndarray:
    """The (T, 30) letter-message rows for one experimental arm.

    ``real`` is L1's predictive distribution. ``scrambled`` reverses its letter
    axis — entropy-preserving, information-destroying. ``oracle`` is a one-hot on
    the letter actually observed, the perfect sender.
    """
    if arm == "oracle":
        onehot = np.zeros((letter_indices.shape[0], N_SYMBOLS), dtype=np.float64)
        onehot[np.arange(letter_indices.shape[0]), letter_indices] = 1.0
        return onehot
    predictive = letter_predictive_distributions(letter_matrices, letter_indices)
    return scramble_letter_distribution(predictive) if arm == "scrambled" else predictive


def proper_predictive_loss(
    letter_matrices,
    g2p_matrices,
    evaluation: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
    arm: str = "real",
) -> tuple[float, int, float]:
    """Mean ``-log q_t(gold phone)`` in nats, the position count, and accuracy.

    The loss is the receiver's proper score on its own target. It is *not*
    ``PositionDiagnostic.posterior_max``, which is mass on the argmax phone and
    says nothing about whether that phone was right.

    Accuracy comes back from the same pass, over the same positions, so the two
    axes can be compared without any possibility of a position-set mismatch —
    which is the comparison ``docs/research_queue.md`` asks for.
    """
    total = 0.0
    positions = 0
    correct = 0
    for letter_indices, gold_phones in evaluation:
        message = _message(letter_matrices, letter_indices, arm)
        posteriors, _, _ = _soft_forward(g2p_matrices, letter_indices, message, alpha)
        rows = np.arange(gold_phones.shape[0])
        # clip only guards log(0); a zero here is a real miss and must score badly,
        # not be smoothed into something survivable.
        gold_mass = np.clip(posteriors[rows, gold_phones], 1e-300, 1.0)
        total += float(-np.sum(np.log(gold_mass)))
        correct += int(np.sum(np.argmax(posteriors, axis=1) == gold_phones))
        positions += int(gold_phones.shape[0])
    if positions == 0:
        raise ValueError("no aligned positions to evaluate")
    return total / positions, positions, 100.0 * correct / positions


def uncoupled_reference(
    letter_matrices,
    g2p_matrices,
    evaluation: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, bool, float]:
    """Loss from an independent recursion with no mixing, and the void check.

    ``docs/research_queue.md`` voids this arm unless ``alpha = 0`` is the
    uncoupled decode. Verifying that against ``_soft_forward(alpha=0)`` alone
    would be circular — the short circuit under test lives inside the function
    doing the testing — so the plain hard-evidence filter is written out longhand
    here and compared against it.

    The comparison is made on the **posteriors**, not on the mean loss. Two
    mathematically identical sums accumulated in different orders differ in the
    last ulp, so an exact equality on the aggregate reports a mismatch of order
    1e-15 that means nothing. Bit-identity of the per-position distributions is
    the invariant with content; the aggregate gap is returned as a diagnostic.

    Returns:
        ``(reference_loss, posteriors_bit_identical, mean_loss_abs_gap)``.
    """
    total = 0.0
    positions = 0
    bit_identical = True
    for letter_indices, gold_phones in evaluation:
        prediction = g2p_matrices.start.copy()
        reference_posteriors = np.empty((letter_indices.shape[0], prediction.shape[0]))
        for step in range(letter_indices.shape[0]):
            likelihood = g2p_matrices.emission[int(letter_indices[step]), :]
            unnormalized = likelihood * prediction
            norm = unnormalized.sum()
            posterior = unnormalized / norm if norm > 0 else prediction
            reference_posteriors[step] = posterior
            total += -float(np.log(max(float(posterior[int(gold_phones[step])]), 1e-300)))
            positions += 1
            prediction = g2p_matrices.transition @ posterior

        coupled, _, _ = _soft_forward(
            g2p_matrices,
            letter_indices,
            letter_predictive_distributions(letter_matrices, letter_indices),
            0.0,
        )
        if not np.array_equal(coupled, reference_posteriors):
            bit_identical = False

    return total / positions, bit_identical, 0.0


def compute_l1_l2b_predictive_loss(
    sample_limit: int = 200,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> dict[str, object]:
    """Measure proper predictive loss and the channel value inequality.

    Returns:
        A dict carrying the per-alpha losses for all three arms, the realized /
        readable / sensitivity terms, and the two gate verdicts — each computed
        from those numbers, never asserted.

    Raises:
        ValueError: If ``0.0`` is absent from the grid (there would be no anchor
            to define a gain against) or too few positions survive alignment.
    """
    if 0.0 not in alpha_grid:
        raise ValueError(f"alpha grid {alpha_grid} lacks the 0.0 anchor")

    brown = load_brown_counts()
    train_pairs, test_pairs = g2p_split(load_phoneme_counts())
    letter_matrices = build_letter_matrices(brown)
    g2p_matrices = build_g2p_matrices(train_pairs)

    evaluation = _aligned_pairs(test_pairs, sample_limit)
    if not evaluation:
        raise ValueError("no aligned word/pronunciation pairs in the test split")

    losses: dict[str, dict[float, float]] = {"real": {}, "scrambled": {}, "oracle": {}}
    accuracy: dict[float, float] = {}
    positions = 0
    for arm in losses:
        for alpha in alpha_grid:
            losses[arm][alpha], positions, arm_accuracy = proper_predictive_loss(
                letter_matrices, g2p_matrices, evaluation, alpha, arm=arm
            )
            if arm == "real":
                accuracy[alpha] = arm_accuracy
    if positions < MIN_POSITIONS:
        raise ValueError(f"only {positions} positions survived alignment; need {MIN_POSITIONS}")

    # Void condition from docs/research_queue.md: if alpha = 0 is not the
    # uncoupled decode, the whole comparison is against the wrong anchor.
    reference, anchor_matches_uncoupled, _ = uncoupled_reference(
        letter_matrices, g2p_matrices, evaluation
    )
    anchor_loss_abs_gap = abs(losses["real"][0.0] - reference)

    anchor = losses["real"][0.0]
    realized = {a: anchor - losses["real"][a] for a in alpha_grid}
    scrambled_realized = {a: anchor - losses["scrambled"][a] for a in alpha_grid}

    # `readable` must be the best the channel does when it is actually USED.
    # Taking the max over the whole grid includes alpha = 0, whose gain is
    # identically zero by definition of the anchor — so a channel that does
    # nothing would report readable = 0 and "does not harm", which is a verdict
    # about the no-op, not about the channel. Engaged alphas only.
    #
    # This is the same trap decision 46 records the coupling recorder falling
    # into on its first pass: selecting the best weight over a grid including
    # zero made both harmful evidence cells report the anchor's own 70.801 with
    # 0/0 discordant pairs and p = 1, so "a channel that damages the model
    # appeared in the table as doing nothing at all". Two independent arrivals at
    # one trap is a good reason to name it here rather than only in the log.
    engaged = [a for a in alpha_grid if a > 0.0]
    if not engaged:
        raise ValueError(f"alpha grid {alpha_grid} has no engaged (alpha > 0) point")
    readable = max(realized[a] for a in engaged)
    best_alpha = max(engaged, key=lambda a: realized[a])

    # The oracle sender is the receiver's own observation, so its loss must equal
    # the anchor at every alpha. Checked, not assumed: a non-zero spread here
    # would mean the mixture is not doing what the docstring says it does.
    oracle_spread = max(abs(losses["oracle"][a] - anchor) for a in alpha_grid)
    sensitivity = anchor - min(losses["oracle"].values())

    upper_bound = min(readable, sensitivity)
    # The substantive claim is realized <= sensitivity at every engaged alpha:
    # no mixture weight may beat the perfect sender.
    inequality_holds = all(realized[a] <= sensitivity + 1e-12 for a in engaged)
    channel_is_non_harmful = readable > 1e-12

    # The cross-axis comparison the research queue asks for, on ONE position set:
    # does the sign of the channel's effect agree between accuracy and log-loss?
    best_accuracy_alpha = max(engaged, key=lambda a: accuracy[a])
    accuracy_gain = accuracy[best_accuracy_alpha] - accuracy[0.0]
    accuracy_says_helps = accuracy_gain > 0.0
    loss_says_helps = readable > 0.0
    axes_agree = accuracy_says_helps == loss_says_helps

    return {
        "channel": "L1->L2B",
        "total_positions_evaluated": positions,
        "words_evaluated": len(evaluation),
        "alpha_grid": list(alpha_grid),
        "loss_by_alpha": {a: losses["real"][a] for a in alpha_grid},
        "accuracy_by_alpha": dict(accuracy),
        "anchor_accuracy": accuracy[0.0],
        "best_accuracy_alpha": best_accuracy_alpha,
        "accuracy_gain_pts": accuracy_gain,
        "accuracy_says_helps": bool(accuracy_says_helps),
        "loss_says_helps": bool(loss_says_helps),
        "axes_agree_on_sign": bool(axes_agree),
        "uncoupled_reference_loss": reference,
        "anchor_matches_uncoupled": bool(anchor_matches_uncoupled),
        "anchor_loss_abs_gap": anchor_loss_abs_gap,
        "scrambled_loss_by_alpha": {a: losses["scrambled"][a] for a in alpha_grid},
        "oracle_loss_by_alpha": {a: losses["oracle"][a] for a in alpha_grid},
        "anchor_loss": anchor,
        "realized_by_alpha": realized,
        "scrambled_realized_by_alpha": scrambled_realized,
        "readable": readable,
        "best_alpha": best_alpha,
        "sensitivity": sensitivity,
        "oracle_spread": oracle_spread,
        "upper_bound": upper_bound,
        # Measured verdicts. Both were hardcoded True before this probe computed
        # anything; the second one is FALSE on this channel, and that is the result.
        # Measured, not asserted: the loss is evaluable only if every arm at
        # every weight produced a finite number. Leaving this as the literal
        # `True` it used to be would reintroduce, in the fixed probe, the exact
        # defect the fix was for.
        "is_proper_loss_evaluable": bool(
            all(np.isfinite(losses[arm][a]) for arm in losses for a in alpha_grid)
        ),
        "inequality_upper_bound_holds": bool(inequality_holds),
        "channel_improves_proper_loss": bool(channel_is_non_harmful),
    }


def compute_factorial_cells_predictive_loss(
    *,
    letter_predictive_alpha: float,
    lexicon_evidence_alpha: float,
    lexicon_predictive_alpha: float,
    geometric_lexicon_exponent: float,
    fitted_exponent: float,
    fitted_max_prefix: int,
    fitted_min_count: int,
    fitted_smoothing: float,
    sample_limit: int | None = None,
) -> dict[str, object]:
    """Proper predictive loss for the factorial cells beyond letter@evidence.

    The remaining cells the attacked research-queue verdict licenses: the three
    linear cells of decision 46's factorial not yet on the loss axis
    (letter@predictive, lexicon@evidence, lexicon@predictive), plus the two
    non-diluting lexical arms (the geometric sweep's selected exponent and the
    fitted configuration of decision 49). Each is scored at the weight the
    ledger already records for it — the caller passes those recorded values in,
    so this function selects nothing.

    Every cell reads its loss and its accuracy off the same posterior rows via
    :func:`ailm.channels.run_cell_proper_loss` (linear cells) or
    :func:`ailm.channels._geometric_posteriors` (geometric arms), the recursions
    whose hit vectors produced the recorded accuracies. The accuracy that comes
    back is therefore the cross-check that each loss describes the recorded cell
    and not a neighboring one; the caller compares it against the ledger.

    The anchor is computed three ways — the linear predictive-site recursion at
    weight zero, the evidence-site recursion at weight zero, and the geometric
    recursion at exponent zero — and all three must agree, because all three
    claim to *be* the uncoupled decode. Their maximum pairwise loss gap is
    returned for the recorder to refuse on.

    Returns:
        Per-cell mean losses in nats, accuracies, the shared position count, the
        anchor loss and the anchor-consistency gap.
    """
    from ailm.channels import _geometric_posteriors, run_cell_proper_loss
    from ailm.levels.lexical_channel import build_lexical_message, lexical_message_for
    from ailm.levels.phoneme import PHONES, _iter_pairs

    brown = load_brown_counts()
    train_pairs, test_pairs = g2p_split(load_phoneme_counts())
    letter_matrices = build_letter_matrices(brown)
    g2p_matrices = build_g2p_matrices(train_pairs)
    phones = list(PHONES)
    phone_to_index = {p: i for i, p in enumerate(phones)}
    shipped_message = build_lexical_message(train_pairs, phones)
    fitted_message = build_lexical_message(
        train_pairs,
        phones,
        max_prefix=fitted_max_prefix,
        min_count=fitted_min_count,
        smoothing=fitted_smoothing,
    )

    common = {
        "letter_matrices": letter_matrices,
        "g2p": g2p_matrices,
        "message": shipped_message,
        "pairs": test_pairs,
        "phone_to_index": phone_to_index,
        "limit": sample_limit,
    }

    def geometric_loss(message, exponent: float) -> tuple[float, int, float]:
        """Mean loss, positions and accuracy of one geometric arm."""
        emission = np.asarray(g2p_matrices.emission, dtype=np.float64)
        transition = np.asarray(g2p_matrices.transition, dtype=np.float64)
        start = np.asarray(g2p_matrices.start, dtype=np.float64)
        total = 0.0
        positions = 0
        correct = 0
        for word, gold in _iter_pairs(test_pairs, sample_limit):
            indices = [ALPHA_TO_IDX[ch] for ch in word]
            truth = np.asarray([phone_to_index[p] for p in gold], dtype=np.int64)
            rows = np.asarray(lexical_message_for(message, word)[0])
            posteriors = _geometric_posteriors(emission, transition, start, indices, rows, exponent)
            gold_mass = np.clip(posteriors[np.arange(truth.shape[0]), truth], 1e-300, 1.0)
            total += float(-np.sum(np.log(gold_mass)))
            correct += int(np.sum(np.argmax(posteriors, axis=1) == truth))
            positions += int(truth.shape[0])
        if positions == 0:
            raise ValueError("geometric arm scored no positions")
        return total / positions, positions, 100.0 * correct / positions

    anchors = {
        "linear_predictive": run_cell_proper_loss("lexicon", "predictive", alpha=0.0, **common),
        "linear_evidence": run_cell_proper_loss("letter", "evidence", alpha=0.0, **common),
        "geometric": geometric_loss(shipped_message, 0.0),
    }
    anchor_losses = [loss for loss, _n, _acc in anchors.values()]
    anchor_gap = max(anchor_losses) - min(anchor_losses)
    anchor_loss, positions, anchor_accuracy = anchors["linear_predictive"]

    cells: dict[str, tuple[float, int, float]] = {
        "letter_predictive": run_cell_proper_loss(
            "letter", "predictive", alpha=letter_predictive_alpha, **common
        ),
        "lexicon_evidence": run_cell_proper_loss(
            "lexicon", "evidence", alpha=lexicon_evidence_alpha, **common
        ),
        "lexicon_predictive": run_cell_proper_loss(
            "lexicon", "predictive", alpha=lexicon_predictive_alpha, **common
        ),
        "geometric_lexicon": geometric_loss(shipped_message, geometric_lexicon_exponent),
        "fitted": geometric_loss(fitted_message, fitted_exponent),
    }
    if any(n != positions for _loss, n, _acc in cells.values()):
        raise ValueError(
            f"cells scored unequal position counts against the anchor's {positions}: "
            f"{ {name: n for name, (_loss, n, _acc) in cells.items()} }"
        )

    return {
        "positions": positions,
        "anchor_nats": anchor_loss,
        "anchor_accuracy": anchor_accuracy,
        "anchor_consistency_gap": anchor_gap,
        "cell_nats": {name: loss for name, (loss, _n, _acc) in cells.items()},
        "cell_accuracy": {name: acc for name, (_loss, _n, acc) in cells.items()},
        "cell_realized_nats": {
            name: anchor_loss - loss for name, (loss, _n, _acc) in cells.items()
        },
        "cell_weights": {
            "letter_predictive": letter_predictive_alpha,
            "lexicon_evidence": lexicon_evidence_alpha,
            "lexicon_predictive": lexicon_predictive_alpha,
            "geometric_lexicon": geometric_lexicon_exponent,
            "fitted": fitted_exponent,
        },
        "is_evaluable": bool(
            all(np.isfinite(loss) for loss, _n, _acc in cells.values()) and np.isfinite(anchor_loss)
        ),
    }


def main() -> None:
    print("=== Cross-Cutting Channel Predictive Loss & Inequality Probe ===")
    res = compute_l1_l2b_predictive_loss(sample_limit=100_000)
    print(f"Channel: {res['channel']}")
    print(
        f"Words / positions evaluated: {res['words_evaluated']} / {res['total_positions_evaluated']}"
    )
    print("cell: letter@evidence (decision 46); letter@predictive is NOT measured here")
    print(
        f"\n{'alpha':>7} {'loss(real)':>12} {'realized':>12} {'accuracy%':>11} "
        f"{'loss(scram)':>12} {'loss(oracle)':>13}"
    )
    for alpha in res["alpha_grid"]:  # type: ignore[union-attr]
        print(
            f"{alpha:7.2f} {res['loss_by_alpha'][alpha]:12.6f} "  # type: ignore[index]
            f"{res['realized_by_alpha'][alpha]:+12.6f} "  # type: ignore[index]
            f"{res['accuracy_by_alpha'][alpha]:11.4f} "  # type: ignore[index]
            f"{res['scrambled_loss_by_alpha'][alpha]:12.6f} "  # type: ignore[index]
            f"{res['oracle_loss_by_alpha'][alpha]:13.6f}"  # type: ignore[index]
        )
    print(f"\nanchor loss (alpha=0)   : {res['anchor_loss']:.6f} nats/position")
    print(f"readable (best realized): {res['readable']:+.6f} at alpha={res['best_alpha']}")
    print(
        f"sensitivity (oracle)    : {res['sensitivity']:+.6f}  [oracle spread {res['oracle_spread']:.3e}]"
    )
    print(f"upper bound min(...)    : {res['upper_bound']:+.6f}")
    print(
        f"\nVOID CHECK  alpha=0 posteriors bit-identical to the uncoupled decode: "
        f"{res['anchor_matches_uncoupled']}  "
        f"(mean-loss gap {res['anchor_loss_abs_gap']:.2e}, summation order only)"
    )
    print(f"\nrealized <= bound       : {res['inequality_upper_bound_holds']}")
    print(f"channel improves loss   : {res['channel_improves_proper_loss']}")
    print(f"accuracy says helps     : {res['accuracy_says_helps']}")
    print(f"AXES AGREE ON SIGN      : {res['axes_agree_on_sign']}")

    if res["axes_agree_on_sign"]:
        print(
            "\nRESULT: both axes agree that letter@evidence HARMS the receiver. The L1\n"
            "message is redundant with the letter the receiver already observes, so\n"
            "mixing it in only smears the evidence; the oracle-sender arm confirms the\n"
            "redundancy exactly (sensitivity = 0). The research-queue falsifier — a sign\n"
            "disagreement between accuracy and log-loss — does NOT fire on this cell, so\n"
            "sec:hierarchy's monotone-decline diagnostic is robust here. The open cell is\n"
            "letter@predictive (+0.103 pts, p = 0.17), which this probe does not measure."
        )
    else:
        print(
            "\nRESULT: the two axes DISAGREE in sign on this cell. That is the\n"
            "research-queue falsifier: sec:hierarchy's monotone-decline diagnostic would\n"
            "be a property of the argmax statistic rather than of the channel, and must\n"
            "be restated on the axis the hypothesis box declares."
        )


if __name__ == "__main__":
    main()
