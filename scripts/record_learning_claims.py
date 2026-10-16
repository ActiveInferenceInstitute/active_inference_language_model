#!/usr/bin/env python3
"""Record the W2 learning-equals-counting result, from the run that measured it.

Thin orchestrator; computation lives in ``ailm.learning``. Single writer for
every ``learning_*`` label, and the writer of ``data/learning_agreement.npz``
— the matrices the agreement figure draws, saved here so the figure never
re-runs pymdp and can be checked against the ledger instead.

The prediction this measures was pre-registered in ``docs/decisions.md``
entry 22 and committed BEFORE this script existed; entry 23 records the
outcome. The recorder refuses to write unless every pre-registered clause
holds:

1. **P1** — matched priors agree to the pre-registered float32 tolerance.
2. **P2** — a mismatched Dirichlet prior breaks the agreement, and matching
   the comparator's pseudocount restores it. Without this the comparison
   would be vacuous.
3. **P3** — the learned matrix moved from its initialization.
4. **Corollary** — the learned-B and counted-B agents score identical
   perplexity on the Stage 1 evaluation text (in-sample: the lane learns on
   and scores the same text).

Usage
-----
    .venv/bin/python scripts/record_learning_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.evaluation import STAGE1_EVALUATION_TEXT  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "learning_conjugacy"
SOURCE = "scripts/record_learning_claims.py"

#: Pre-registered float32 agreement tolerance (decision 22, P1).
TOLERANCE = 1e-5

#: The mismatched Dirichlet concentration for P2.
MISMATCHED_PSEUDOCOUNT = 2.0

ARTIFACT = _PROJECT_ROOT / "data" / "learning_agreement.npz"


def main() -> int:
    import jax.numpy as jnp

    from ailm.claims import load_ledger
    from ailm.learning import learn_letter_transition
    from ailm.levels.letter import build_letter_agent, compute_perplexity

    matched = learn_letter_transition(STAGE1_EVALUATION_TEXT, pseudocount=1.0)
    mismatched = learn_letter_transition(
        STAGE1_EVALUATION_TEXT,
        pseudocount=MISMATCHED_PSEUDOCOUNT,
        comparator_pseudocount=1.0,
    )
    restored = learn_letter_transition(STAGE1_EVALUATION_TEXT, pseudocount=MISMATCHED_PSEUDOCOUNT)

    if not matched.max_abs_delta <= TOLERANCE:
        raise SystemExit(
            f"P1 FAILED: matched-prior deviation {matched.max_abs_delta:.3e} "
            f"exceeds the pre-registered tolerance {TOLERANCE:.0e}. Decision 22's "
            f"withdrawal condition applies — do not record; trace which "
            f"implementation departs from the conjugate arithmetic."
        )
    if not mismatched.max_abs_delta > TOLERANCE:
        raise SystemExit(
            "P2 FAILED (vacuity): a mismatched Dirichlet prior still agrees, so "
            "the comparison cannot fail and nothing was tested."
        )
    if not restored.max_abs_delta <= TOLERANCE:
        raise SystemExit(
            f"P2 FAILED (restore): matching the comparator at k = "
            f"{MISMATCHED_PSEUDOCOUNT} leaves deviation "
            f"{restored.max_abs_delta:.3e}, so the prior is not the only "
            f"difference between the two estimators."
        )
    if not matched.moved_from_initial > 0.1:
        raise SystemExit(
            f"P3 FAILED: learning moved only {matched.moved_from_initial:.4f} "
            f"from the uniform initialization — the run may not have learned "
            f"at all."
        )

    learned_ppl = compute_perplexity(
        build_letter_agent(jnp.array(matched.learned, dtype=jnp.float32)[jnp.newaxis]),
        STAGE1_EVALUATION_TEXT,
    )
    counted_ppl = compute_perplexity(
        build_letter_agent(jnp.array(matched.counted, dtype=jnp.float32)[jnp.newaxis]),
        STAGE1_EVALUATION_TEXT,
    )
    ppl_delta = abs(learned_ppl - counted_ppl)
    if not ppl_delta <= 1e-6:
        raise SystemExit(
            f"COROLLARY FAILED: learned-B and counted-B agents differ by "
            f"{ppl_delta:.3e} perplexity on the identical stream, which "
            f"contradicts P1's matrix agreement."
        )

    np.savez_compressed(
        ARTIFACT,
        learned=matched.learned,
        counted=matched.counted,
        mismatched_learned=mismatched.learned,
        mismatched_counted=mismatched.counted,
    )

    values: dict[str, float | int] = {
        "learning_stream_positions": matched.n_positions,
        "learning_pseudocount": matched.pseudocount,
        "learning_tolerance": TOLERANCE,
        "learning_matched_max_abs_delta": matched.max_abs_delta,
        "learning_mismatched_pseudocount": MISMATCHED_PSEUDOCOUNT,
        "learning_mismatched_max_abs_delta": mismatched.max_abs_delta,
        "learning_restored_max_abs_delta": restored.max_abs_delta,
        "learning_moved_from_initial": round(matched.moved_from_initial, 4),
        "learning_learned_perplexity": round(learned_ppl, 6),
        "learning_counted_perplexity": round(counted_ppl, 6),
        "learning_perplexity_abs_delta": ppl_delta,
    }

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(values)} learning claims; matrices -> {ARTIFACT.name}")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
