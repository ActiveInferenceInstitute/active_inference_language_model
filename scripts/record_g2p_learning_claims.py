#!/usr/bin/env python3
"""Record the pre-registered latent-state G2P ``learn_A`` experiment (decision 29)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

from ailm.claims import load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.g2p_learning import run_g2p_emission_learning  # noqa: E402
from ailm.levels.phoneme import g2p_split, load_phoneme_counts  # noqa: E402

RUN_ID = "g2p_learn_A_latent_states"
SOURCE = "scripts/record_g2p_learning_claims.py"


def main() -> int:
    train, test = g2p_split(load_phoneme_counts())
    result = run_g2p_emission_learning(train, test, learning_stride=16)
    if result.max_movement <= 1e-6:
        raise SystemExit("REFUSING TO RECORD: learned A did not move from symmetric initialization")
    if result.gold_max_abs_delta > 1e-5:
        raise SystemExit(
            "REFUSING TO RECORD: observed-state P4 control does not recover "
            f"same-lane counting (max delta {result.gold_max_abs_delta})"
        )
    if result.decoder_disagreements:
        raise SystemExit(
            "REFUSING TO RECORD: pymdp and independent oracle decoders disagree "
            f"at {result.decoder_disagreements} positions"
        )
    values = {
        "g2p_learn_A_train_pairs": result.train_pairs,
        "g2p_learn_A_eval_positions": result.eval_positions,
        "g2p_learn_A_max_movement": round(result.max_movement, 6),
        "g2p_learn_A_max_count_deviation": round(result.max_count_deviation, 6),
        "g2p_learn_A_learned_accuracy": round(result.learned_accuracy * 100, 6),
        "g2p_learn_A_fixed_accuracy": round(result.fixed_accuracy * 100, 6),
        "g2p_learn_A_oracle_learned_accuracy": round(result.oracle_learned_accuracy * 100, 6),
        "g2p_learn_A_oracle_fixed_accuracy": round(result.oracle_fixed_accuracy * 100, 6),
        "g2p_learn_A_oracle_gold_lane_accuracy": round(result.oracle_gold_lane_accuracy * 100, 6),
        "g2p_learn_A_gold_max_abs_delta": round(result.gold_max_abs_delta, 8),
        "g2p_learn_A_mcnemar_learned_only": result.mcnemar_learned_only,
        "g2p_learn_A_mcnemar_fixed_only": result.mcnemar_fixed_only,
        "g2p_learn_A_decoder_disagreements": result.decoder_disagreements,
        "g2p_learn_A_learning_stride": 16,
        "g2p_learn_A_confirmed": int(result.learned_accuracy > result.fixed_accuracy),
    }
    load_ledger(_PROJECT_ROOT).record_many(values, run_id=RUN_ID, source=SOURCE)
    for label, value in values.items():
        print(f"{label} = {value}")
    print(
        "verdict =", "IMPROVED" if values["g2p_learn_A_confirmed"] else "NO IMPROVEMENT / QUALIFIED"
    )
    return 0


if __name__ == "__main__":
    # The ledger is read whole and written whole, so two overlapping recorder
    # runs lose one side's entire batch. Every writer takes the same lock.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
