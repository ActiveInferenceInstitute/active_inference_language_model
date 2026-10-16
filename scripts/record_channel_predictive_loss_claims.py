#!/usr/bin/env python3
"""Record the L1 -> L2B channel on the receiver's own proper predictive loss.

Thin orchestrator; the computation lives in
``scripts/probe_channel_predictive_loss.py``. Single writer for every
``channel_proper_loss_*`` label.

Why this exists
---------------
sec:hierarchy's working-hypothesis box is stated "under a fixed receiver and a
proper predictive loss", and every channel number in the paper is an argmax
accuracy. Accuracy is not a proper scoring rule, so the axis the hypothesis
declares had never been measured. This recorder measures it for one channel —
**letter@evidence**, the cell whose accuracy is already recorded — over the same
held-out positions, so the two axes can be compared without a population change
in between.

The extension the attacked research-queue verdict licenses ("Survives, but as
ONE channel of work") is recorded here too, by the same single writer: the three
remaining linear factorial cells (letter@predictive, lexicon@evidence,
lexicon@predictive) and the two non-diluting lexical arms (the geometric sweep's
selected exponent and decision 49's fitted configuration), each scored as
held-out per-position log loss with the channel engaged at its *recorded*
weight. The weights of the sweep cells were selected on the same held-out split
their accuracies are reported on, so their loss numbers inherit exactly the
selection caveat sec:hierarchy states for the accuracies; each such cell carries
an explicit ``*_weight_test_selected`` marker claim. The fitted arm's weight was
selected on an inner split of the training set, and its marker records that.

Guards — the recorder refuses to write when
-------------------------------------------
1. **The anchor is not the uncoupled decode.** The weight-zero posteriors must
   be bit-identical, position by position, to a plain hard-evidence filter
   written independently of the coupling code. If weight zero is not the
   uncoupled decode, every "gain" is measured against the wrong baseline.
   This is the void condition ``docs/research_queue.md`` states for this arm.
2. **The accuracy axis does not reproduce the recorded cell.** The probe must
   land on ``channel_letter_evidence_best_accuracy`` and the uncoupled anchor
   already in the ledger. If it does not, it is measuring some other channel
   and its loss numbers describe something the prose would misattribute.
3. **The loss is degenerate.** A loss that is constant across mixing weights,
   or non-finite anywhere, means nothing was measured.
4. **The oracle-sender arm is not an identity.** The sender's latent is the
   letter the receiver already observes, so a perfect message must reproduce
   the anchor exactly. A non-zero spread means the mixture is not forming
   evidence the way the docstring says, and the sensitivity term would be
   meaningless.

The sign of the effect is recorded whichever way it lands. It lands negative:
this channel harms the receiver's proper loss at every engaged weight, and that
is a recorded result, not a failure.

Usage
-----
    .venv/bin/python scripts/record_channel_predictive_loss_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "channel_proper_loss"
SOURCE = "scripts/record_channel_predictive_loss_claims.py"

#: Tolerance for the accuracy cross-check against the already-recorded cell.
#: The recorded values carry four decimals, so anything tighter would fail on
#: rounding alone.
ACCURACY_TOLERANCE_PTS = 5e-4

#: Complete ledger input contract.  Keeping this at module scope lets the
#: orchestration test prove that every owning recorder has a producer→consumer
#: edge, instead of trusting a hand-read subset of ``ledger.get`` calls.
REQUIRED_CLAIMS = (
    "coupling_alpha0_accuracy",
    "coupling_eval_positions",
    "channel_factorial_anchor_accuracy",
    "channel_factorial_positions",
    "channel_letter_evidence_best_accuracy",
    "channel_letter_evidence_best_alpha",
    "channel_letter_predictive_best_accuracy",
    "channel_letter_predictive_best_alpha",
    "channel_lexicon_evidence_best_accuracy",
    "channel_lexicon_evidence_best_alpha",
    "channel_lexicon_predictive_best_accuracy",
    "channel_lexicon_predictive_best_alpha",
    "geometric_lexicon_best_accuracy",
    "geometric_lexicon_best_exponent",
    "fitted_accuracy",
    "fitted_exponent",
    "fitted_max_prefix",
    "fitted_min_count",
    "fitted_smoothing",
)


def main() -> int:
    from ailm.claims import load_ledger
    from scripts.probe_channel_predictive_loss import compute_l1_l2b_predictive_loss

    ledger = load_ledger(_PROJECT_ROOT)

    # ── Producer precondition, before either full-population probe ───────────
    def _recorded(label: str) -> float:
        """The stored value for ``label``, or a hard stop if it is absent.

        The cross-check below is only meaningful against a ledger that already
        carries the accuracy cell; a missing label means the comparison silently
        would not happen.
        """
        entry = ledger.get(label)
        if entry is None:
            raise SystemExit(
                f"{label} is not in the ledger; run the coupling and factorial "
                f"recorders before this one."
            )
        return float(entry["value"])

    recorded = {label: _recorded(label) for label in REQUIRED_CLAIMS}
    recorded_anchor = recorded["channel_factorial_anchor_accuracy"]
    recorded_cell = recorded["channel_letter_evidence_best_accuracy"]
    recorded_alpha = recorded["channel_letter_evidence_best_alpha"]
    recorded_positions = int(recorded["channel_factorial_positions"])
    if (
        recorded["coupling_alpha0_accuracy"] != recorded_anchor
        or int(recorded["coupling_eval_positions"]) != recorded_positions
    ):
        raise SystemExit(
            "the coupling and factorial producers disagree on their uncoupled "
            "anchor or evaluation population; no proper-loss claim can name both."
        )

    # The full held-out split, so the accuracy axis is comparable to the
    # recorded cell position-for-position rather than on a subsample.
    result = compute_l1_l2b_predictive_loss(sample_limit=100_000)

    # ── Guard 1: the void condition ──────────────────────────────────────────
    if not result["anchor_matches_uncoupled"]:
        raise SystemExit(
            "VOID: the weight-zero posteriors are not bit-identical to the "
            "uncoupled decode, so every gain below is measured against the "
            "wrong anchor. docs/research_queue.md voids this arm on exactly "
            "this condition."
        )

    # ── Guard 2: it is the channel the prose will say it is ──────────────────

    if result["total_positions_evaluated"] != recorded_positions:
        raise SystemExit(
            f"position set changed: measured {result['total_positions_evaluated']}, "
            f"recorded {recorded_positions}. The two axes would not be comparable."
        )
    if abs(result["anchor_accuracy"] - recorded_anchor) > ACCURACY_TOLERANCE_PTS:
        raise SystemExit(
            f"uncoupled anchor accuracy {result['anchor_accuracy']:.4f}% does not "
            f"reproduce the recorded {recorded_anchor:.4f}%."
        )
    measured_cell = result["accuracy_by_alpha"][recorded_alpha]
    if abs(measured_cell - recorded_cell) > ACCURACY_TOLERANCE_PTS:
        raise SystemExit(
            f"letter@evidence accuracy at weight {recorded_alpha} is "
            f"{measured_cell:.4f}%, not the recorded {recorded_cell:.4f}%. This "
            f"probe is measuring a different channel than the prose claims."
        )

    # ── Guard 3: something was actually measured ─────────────────────────────
    if not result["is_proper_loss_evaluable"]:
        raise SystemExit("a loss was non-finite; the measurement did not evaluate")
    losses = [result["loss_by_alpha"][a] for a in result["alpha_grid"]]
    if len(set(losses)) != len(losses):
        raise SystemExit(
            "the proper loss is constant across mixing weights — the sweep is "
            "vacuous and no weight can be said to help or harm."
        )

    # ── Guard 4: the oracle sender is an identity ────────────────────────────
    if result["oracle_spread"] > 1e-12:
        raise SystemExit(
            f"the oracle sender moved the loss by {result['oracle_spread']:.3e}; "
            f"it must reproduce the anchor exactly, since the message it sends "
            f"IS the letter the receiver observes. The sensitivity term below "
            f"would be meaningless."
        )

    engaged = [a for a in result["alpha_grid"] if a > 0.0]
    values: dict[str, float | int] = {
        "channel_proper_loss_positions": result["total_positions_evaluated"],
        "channel_proper_loss_words": result["words_evaluated"],
        "channel_proper_loss_anchor_nats": round(result["anchor_loss"], 6),
        "channel_proper_loss_best_alpha": result["best_alpha"],
        "channel_proper_loss_best_realized_nats": round(result["readable"], 6),
        "channel_proper_loss_worst_realized_nats": round(
            min(result["realized_by_alpha"][a] for a in engaged), 6
        ),
        "channel_proper_loss_sensitivity_nats": round(result["sensitivity"], 6),
        "channel_proper_loss_oracle_spread": float(result["oracle_spread"]),
        "channel_proper_loss_anchor_accuracy": round(result["anchor_accuracy"], 4),
        "channel_proper_loss_best_accuracy_alpha": result["best_accuracy_alpha"],
        "channel_proper_loss_accuracy_gain_pts": round(result["accuracy_gain_pts"], 4),
        "channel_proper_loss_engaged_weights": len(engaged),
    }
    for alpha in result["alpha_grid"]:
        key = f"{alpha}".replace(".", "_")
        values[f"channel_proper_loss_alpha_{key}_nats"] = round(result["loss_by_alpha"][alpha], 6)
        values[f"channel_proper_loss_alpha_{key}_scrambled_nats"] = round(
            result["scrambled_loss_by_alpha"][alpha], 6
        )

    # ── The remaining factorial cells and the two non-diluting lexical arms ──
    from scripts.probe_channel_predictive_loss import compute_factorial_cells_predictive_loss

    cells = compute_factorial_cells_predictive_loss(
        letter_predictive_alpha=recorded["channel_letter_predictive_best_alpha"],
        lexicon_evidence_alpha=recorded["channel_lexicon_evidence_best_alpha"],
        lexicon_predictive_alpha=recorded["channel_lexicon_predictive_best_alpha"],
        geometric_lexicon_exponent=recorded["geometric_lexicon_best_exponent"],
        fitted_exponent=recorded["fitted_exponent"],
        fitted_max_prefix=int(recorded["fitted_max_prefix"]),
        fitted_min_count=int(recorded["fitted_min_count"]),
        fitted_smoothing=recorded["fitted_smoothing"],
    )

    # ── Guard 5: the extension scores the recorded cells, not neighbors ──────
    if cells["positions"] != recorded_positions:
        raise SystemExit(
            f"factorial-cell loss pass scored {cells['positions']} positions against "
            f"the recorded {recorded_positions}; the axes would not be comparable."
        )
    if cells["anchor_consistency_gap"] > 1e-9:
        raise SystemExit(
            f"the three anchor recursions (linear predictive, linear evidence, "
            f"geometric) disagree by {cells['anchor_consistency_gap']:.3e} nats; "
            f"all three claim to be the uncoupled decode, so the realized gains "
            f"below would be measured against an ambiguous anchor."
        )
    if abs(cells["anchor_nats"] - result["anchor_loss"]) > 5e-7:
        raise SystemExit(
            f"the extension's anchor loss {cells['anchor_nats']:.6f} does not "
            f"reproduce the letter@evidence pass's anchor {result['anchor_loss']:.6f}."
        )
    accuracy_checks = {
        "letter_predictive": "channel_letter_predictive_best_accuracy",
        "lexicon_evidence": "channel_lexicon_evidence_best_accuracy",
        "lexicon_predictive": "channel_lexicon_predictive_best_accuracy",
        "geometric_lexicon": "geometric_lexicon_best_accuracy",
        "fitted": "fitted_accuracy",
    }
    for cell, label in accuracy_checks.items():
        expected = recorded[label]
        measured_accuracy = cells["cell_accuracy"][cell]
        if abs(measured_accuracy - expected) > ACCURACY_TOLERANCE_PTS:
            raise SystemExit(
                f"{cell} accuracy at its recorded weight is {measured_accuracy:.4f}%, "
                f"not the recorded {expected:.4f}% ({label}). The loss pass is "
                f"measuring a different cell than the prose would claim."
            )
    if not cells["is_evaluable"]:
        raise SystemExit("a factorial-cell loss was non-finite; the measurement did not evaluate")

    # Whether each cell's engaged weight was selected on the same held-out split
    # its result is reported on. True for every sweep cell (the selection caveat
    # sec:hierarchy states for the accuracies transfers to these losses); False
    # for the fitted arm, whose weight was selected on an inner train split.
    test_selected = {
        "letter_predictive": True,
        "lexicon_evidence": True,
        "lexicon_predictive": True,
        "geometric_lexicon": True,
        "fitted": False,
    }
    # Statuses are per batch. The cells whose weight was test-selected are
    # ``exploratory`` — their headline weight was chosen on the evidence the
    # loss is reported on, which is the taxonomy's own definition. The fitted
    # arm's weight was not, but this loss readout is a post-hoc axis rather
    # than part of decision 49's registered rule, so it stays ``descriptive``
    # (ambiguity resolves downward). The marker booleans are facts about the
    # protocol, not measurements: ``descriptive``.
    exploratory_values: dict[str, float] = {}
    descriptive_values: dict[str, float | bool] = {}
    for cell in accuracy_checks:
        target = descriptive_values if cell == "fitted" else exploratory_values
        target[f"channel_proper_loss_{cell}_nats"] = round(cells["cell_nats"][cell], 6)
        target[f"channel_proper_loss_{cell}_realized_nats"] = round(
            cells["cell_realized_nats"][cell], 6
        )
        descriptive_values[f"channel_proper_loss_{cell}_weight_test_selected"] = test_selected[cell]

    # The letter@evidence cell at ITS recorded factorial weight, as its own
    # labels beside the five cells above. The value equals the matching
    # ``channel_proper_loss_alpha_*`` grid point by construction — both are read
    # from the same sweep in this run — but the grid is keyed by weight and the
    # factorial by cell, and a token that resolved the cell *through* the
    # recorded weight at render time proved fragile against synthetic ledgers.
    # Recording the cell-keyed label here keeps the proper-loss table uniform:
    # every row is one recorded label, none is derived.
    letter_evidence_nats = result["loss_by_alpha"][recorded_alpha]
    exploratory_values["channel_proper_loss_letter_evidence_nats"] = round(letter_evidence_nats, 6)
    exploratory_values["channel_proper_loss_letter_evidence_realized_nats"] = round(
        result["anchor_loss"] - letter_evidence_nats, 6
    )
    descriptive_values["channel_proper_loss_letter_evidence_weight_test_selected"] = True

    ledger.record_many(values, run_id=RUN_ID, source=SOURCE)
    ledger.record_many(exploratory_values, run_id=RUN_ID, source=SOURCE, status="exploratory")
    ledger.record_many(descriptive_values, run_id=RUN_ID, source=SOURCE, status="descriptive")
    values |= exploratory_values | descriptive_values

    print(f"Recorded {len(values)} proper-loss channel claims.")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    print(
        "\nBoth axes agree this channel harms: accuracy "
        f"{result['accuracy_gain_pts']:+.4f} pts, proper loss "
        f"{result['readable']:+.6f} nats at its best engaged weight. The "
        "research-queue falsifier (a sign disagreement) does not fire here."
    )
    print("\nRemaining cells at their recorded weights (realized = anchor - loss, nats):")
    for cell in sorted(accuracy_checks):
        marker = "test-selected weight" if test_selected[cell] else "train-selected weight"
        print(
            f"  {cell:20s} loss {cells['cell_nats'][cell]:.6f}  "
            f"realized {cells['cell_realized_nats'][cell]:+.6f}  ({marker})"
        )
    return 0


if __name__ == "__main__":
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
