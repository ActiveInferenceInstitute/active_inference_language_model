"""Measure geometric (log-linear) pooling against the shipped linear mixing rule.

Decision 47 proves that under geometric pooling the evidence and predictive
insertion sites are the same computation, so the site factor decision 46 measured
is an artifact of the linear rule. It leaves open whether the geometric rule is
*better*, and makes no prediction. This recorder answers that.

The uniform-message arm is the reason the whole exercise is worth running. Under
linear mixing a uniform message materially moves posteriors, so a measured gain
can always be regularisation rather than information and the two are separable
only by argument. Under geometric pooling a uniform message is provably inert — a
constant raised to any power cancels in normalization — so the arm is a control
that must return the anchor exactly. When it does, every remaining effect is
content.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.channels import (  # noqa: E402
    _geometric_positions,
    geometric_site_equivalence,
    letters_to_phones,
)
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.coupling import (  # noqa: E402
    build_g2p_matrices,
    build_letter_matrices,
    letter_predictive_distributions,
    mcnemar,
)
from ailm.levels.letter import ALPHA_TO_IDX, load_brown_counts  # noqa: E402
from ailm.levels.lexical_channel import (  # noqa: E402
    build_lexical_message,
    lexical_message_for,
    scramble_phone_distribution,
)
from ailm.levels.phoneme import PHONES, _iter_pairs, g2p_split, load_phoneme_counts  # noqa: E402

SOURCE = "scripts/record_geometric_pooling_claims.py"
RUN_ID = "geometric-pooling"
GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
ARMS = ("lexicon", "letter", "scrambled", "uniform")


#: Exponent the site-equivalence magnitude is measured at. Zero would make
#: the two sites identical trivially (anything to the zeroth power is one),
#: so the measurement has to run at an engaged weight to mean anything.
GEOMETRIC_SITE_EXPONENT = 0.5


def main() -> int:
    vendor = PROJECT_ROOT / "vendor"
    train, test = g2p_split(load_phoneme_counts(vendor))
    g2p = build_g2p_matrices(train)
    emission = np.asarray(g2p.emission, dtype=np.float64)
    transition = np.asarray(g2p.transition, dtype=np.float64)
    start = np.asarray(g2p.start, dtype=np.float64)
    letters_model = build_letter_matrices(load_brown_counts(vendor))
    phones = list(PHONES)
    phone_to_index = {p: i for i, p in enumerate(phones)}
    message = build_lexical_message(train, phones)

    def sweep(arm: str, exponent: float) -> list[bool]:
        hits: list[bool] = []
        for word, gold in _iter_pairs(test, None):
            indices = [ALPHA_TO_IDX[ch] for ch in word]
            truth = [phone_to_index[p] for p in gold]
            if arm == "lexicon":
                rows = np.asarray(lexical_message_for(message, word)[0])
            elif arm == "scrambled":
                rows = scramble_phone_distribution(
                    np.asarray(lexical_message_for(message, word)[0])
                )
            elif arm == "uniform":
                rows = np.full((len(indices), len(phones)), 1.0 / len(phones))
            else:
                predictive = letter_predictive_distributions(
                    letters_model, np.array(indices, dtype=np.int64)
                )
                rows = letters_to_phones(np.asarray(predictive), emission, start)
            hits.extend(
                _geometric_positions(emission, transition, start, indices, truth, rows, exponent)
            )
        return hits

    def accuracy(hits: list[bool]) -> float:
        return 100.0 * sum(hits) / len(hits)

    def site_equivalence(exponent: float) -> tuple[float, int]:
        """Largest posterior gap between the two insertion sites, over the split.

        Decision 47's proof says this is zero in exact arithmetic. The paper
        reports a magnitude, so the magnitude is measured on the shipped
        matrices rather than typed — it was a hand-written 1.7e-16 in the
        manuscript and in `ailm.channels`'s own docstring until this ran.
        """
        worst = 0.0
        disagreements = 0
        for word, _gold in _iter_pairs(test, None):
            indices = [ALPHA_TO_IDX[ch] for ch in word]
            rows = np.asarray(lexical_message_for(message, word)[0])
            gap, disagree = geometric_site_equivalence(
                emission, transition, start, indices, rows, exponent
            )
            worst = max(worst, gap)
            disagreements += disagree
        return worst, disagreements

    anchor = sweep("lexicon", 0.0)
    equivalence_gap, equivalence_disagreements = site_equivalence(GEOMETRIC_SITE_EXPONENT)
    values: dict[str, object] = {
        "geometric_positions": len(anchor),
        "geometric_anchor_accuracy": round(accuracy(anchor), 4),
        "geometric_grid_points": len(GRID),
        "geometric_site_exponent": GEOMETRIC_SITE_EXPONENT,
        "geometric_site_max_abs_gap": float(f"{equivalence_gap:.3g}"),
        "geometric_site_argmax_disagreements": equivalence_disagreements,
    }

    for arm in ARMS:
        engaged = {e: sweep(arm, e) for e in GRID if e > 0.0}
        top = max(engaged, key=lambda e: accuracy(engaged[e]))
        hits = engaged[top]
        b, c, p_value = mcnemar(hits, anchor)
        values |= {
            f"geometric_{arm}_best_exponent": top,
            f"geometric_{arm}_best_accuracy": round(accuracy(hits), 4),
            f"geometric_{arm}_best_margin_pts": round(accuracy(hits) - accuracy(anchor), 4),
            f"geometric_{arm}_paired_b": b,
            f"geometric_{arm}_paired_c": c,
            f"geometric_{arm}_paired_p": float(f"{p_value:.4g}"),
            f"geometric_{arm}_full_exponent_margin_pts": round(
                accuracy(engaged[1.0]) - accuracy(anchor), 4
            ),
        }

    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        ledger = load_ledger(PROJECT_ROOT)
        ledger.replace_claims(
            remove=[label for label in ledger.labels_owned_by(SOURCE) if label not in values],
            batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
        )
    for label, value in sorted(values.items()):
        print(f"  {label:48s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
