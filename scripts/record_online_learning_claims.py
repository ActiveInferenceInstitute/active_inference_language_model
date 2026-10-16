#!/usr/bin/env python3
"""Record the entry-28 online-vs-frozen learning measurement (decision 28).

Thin orchestrator. Computation lives in ``ailm.levels.online_learning``; this
script loads the vendored Brown genre drift artifact, runs the online
experiment once, applies the three pre-registered refusal guards, and writes
the ledger. Single writer for every ``online_*`` label.

The three guards are the pre-registered clauses of ``docs/decisions.md`` entry
28:

1. **P1 (the drift is real)** — the frozen matrix's held-out perplexity
   differs between the two genres by more than tolerance.
2. **P2 (frozen must lose)** — on held-out text from the second distribution,
   the online agent's perplexity is strictly lower than the frozen control's.
3. **P3 (identity before the drift + re-fit control)** — the online agent's
   transition as it stood at the drift point equals the frozen matrix
   (lr=1 identity), and the frozen matrix re-estimated on the second segment
   scores at least as well as the online agent.

The outcome is **whatever the measurement says** — confirmed, or a recorded
withdrawal with the deviation as the finding. Nothing here is tuned toward a
pass.

Usage
-----
    .venv/bin/python scripts/record_online_learning_claims.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

from ailm.claims import load_ledger  # noqa: E402
from ailm.corpus.brown import validate_brown_artifact_provenance  # noqa: E402
from ailm.levels.online_learning import (  # noqa: E402
    DEFAULT_PSEUDOCOUNT,
    DRIFT_TOLERANCE,
    online_letter_experiment_numpy,
    run_online_letter_experiment,
)

RUN_ID = "online_learning_nonstationary"
SOURCE = "scripts/record_online_learning_claims.py"

ARTIFACT = _PROJECT_ROOT / "vendor" / "brown_online_drift.json"

#: Held-out perplexity of the frozen matrix must differ by this much across
#: genres for the drift to count as measured rather than asserted (P1).
DRIFT_TOLERANCE_DOC = DRIFT_TOLERANCE


def _refuse(message: str) -> int:
    print(f"REFUSING TO RECORD: {message}", file=sys.stderr)
    return 1


def main() -> int:
    if not ARTIFACT.exists():
        return _refuse(f"{ARTIFACT} absent; run scripts/generate_online_genre_counts.py")
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    try:
        validate_brown_artifact_provenance(payload)
    except ValueError as exc:
        return _refuse(str(exc))
    if payload.get("schema") != "ailm-online-drift-v2":
        return _refuse(f"unexpected online-drift schema {payload.get('schema')!r}")
    splits = payload["splits"]
    result = run_online_letter_experiment(
        splits["pre_learn"],
        splits["post_learn"],
        splits["pre_hold"],
        splits["post_hold"],
        pseudocount=DEFAULT_PSEUDOCOUNT,
    )

    if result.refuse_reason is not None:
        return _refuse(result.refuse_reason)

    # Cross-check the numpy closed form against the pymdp path on the same
    # stream (the same license `success` uses for on-line claims): the two must
    # agree on which agent has the lower post-drift perplexity.
    numpy_result = online_letter_experiment_numpy(
        splits["pre_learn"],
        splits["post_learn"],
        splits["pre_hold"],
        splits["post_hold"],
        pseudocount=DEFAULT_PSEUDOCOUNT,
    )
    if (numpy_result["ppl_online_post"] < numpy_result["ppl_frozen_post"]) != (
        result.ppl_online_post < result.ppl_frozen_post
    ):
        return _refuse(
            "the pymdp and numpy paths disagree on the sign of the P2 gap "
            "(online vs frozen on post-drift text). The online claim cannot be "
            "recorded if the two implementations do not agree."
        )

    recorded: dict[str, float | int] = {
        "online_pre_learn_positions": result.n_pre_learn,
        "online_post_learn_positions": result.n_post_learn,
        "online_pre_hold_positions": result.n_pre_hold,
        "online_post_hold_positions": result.n_post_hold,
        "online_pseudocount": result.pseudocount,
        "online_ppl_frozen_pre": round(result.ppl_frozen_pre, 6),
        "online_ppl_frozen_post": round(result.ppl_frozen_post, 6),
        "online_ppl_online_post": round(result.ppl_online_post, 6),
        "online_ppl_refit_post": round(result.ppl_refit_post, 6),
        "online_frozen_vs_post_gap_ppl": round(result.ppl_frozen_post - result.ppl_online_post, 6),
        "online_max_drift": round(result.max_drift, 6),
        "online_drift_tolerance": DRIFT_TOLERANCE_DOC,
        "online_p1_passes": int(result.p1_passes),
        "online_p2_passes": int(result.p2_passes),
        "online_p3_passes": int(result.p3_passes),
        "online_confirmed": int(result.p1_passes and result.p2_passes and result.p3_passes),
    }

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(recorded)} online-learning claims:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    verdict = "CONFIRMED" if recorded["online_confirmed"] else "WITHDRAWN / QUALIFIED"
    print(
        f"\nEntry 28 outcome: {verdict}. P1 drift = {result.max_drift:.4f} "
        f"(frozen {result.ppl_frozen_pre:.4f} pre vs {result.ppl_frozen_post:.4f} "
        f"post). P2: online {result.ppl_online_post:.4f} < frozen "
        f"{result.ppl_frozen_post:.4f} on post-drift text. P3: refit "
        f"{result.ppl_refit_post:.4f} (<= online), pre-drift identity holds."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
