"""Measure all four cells of the channel factorial and record them atomically.

[@sec:hierarchy] states that "the two cells that would separate source from
insertion point are not built". This recorder builds them. Decision 46 records
the prospective development-session design, the three confounds the factorial
does not remove, and the alphabet transport the two off-diagonal cells require.
The decision and recorder first appear in the same commit, so this is not an
independently timestamped preregistration.

The executor is gated before it is used: ``ailm.channels.run_cell`` must reproduce
the shipped per-position hit vectors exactly on both diagonal cells, which is
asserted here as a precondition rather than left to the test suite, because a
recorder that writes claims from an unverified instrument is the failure this
project keeps finding in other forms.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.channels import CELLS, run_cell  # noqa: E402
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.coupling import (  # noqa: E402
    build_g2p_matrices,
    build_letter_matrices,
    coupled_positions,
    mcnemar,
)
from ailm.levels.letter import load_brown_counts  # noqa: E402
from ailm.levels.lexical_channel import build_lexical_message  # noqa: E402
from ailm.levels.phoneme import PHONES, g2p_split, load_phoneme_counts  # noqa: E402

SOURCE = "scripts/record_channel_factorial_claims.py"
RUN_ID = "channel-factorial"
GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def main() -> int:
    vendor = PROJECT_ROOT / "vendor"
    train, test = g2p_split(load_phoneme_counts(vendor))
    g2p = build_g2p_matrices(train)
    letters = build_letter_matrices(load_brown_counts(vendor))
    phones = list(PHONES)
    phone_to_index = {p: i for i, p in enumerate(phones)}
    message = build_lexical_message(train, phones)

    common = {
        "letter_matrices": letters,
        "g2p": g2p,
        "message": message,
        "pairs": test,
        "phone_to_index": phone_to_index,
    }

    # ── PRECONDITION: the executor is a dispatcher, not a reimplementation ────
    # Hit vectors, not accuracies: two decoders can agree to four decimals on an
    # aggregate while disagreeing on thousands of positions in both directions.
    for alpha in (0.0, 0.1):
        mine = run_cell("letter", "evidence", alpha=alpha, **common)
        shipped = coupled_positions(letters, g2p, test, alpha=alpha)
        if mine != shipped:
            disagreements = sum(a != b for a, b in zip(mine, shipped, strict=True))
            raise SystemExit(
                f"executor refused: letter@evidence at alpha={alpha} disagrees with the "
                f"shipped coupled decoder on {disagreements} of {len(mine)} positions. "
                f"The factorial would be measuring the executor, not the channels."
            )

    anchor = run_cell("lexicon", "predictive", alpha=0.0, **common)
    positions = len(anchor)

    def accuracy(hits: list[bool]) -> float:
        return 100.0 * sum(hits) / len(hits)

    values: dict[str, object] = {
        "channel_factorial_positions": positions,
        "channel_factorial_anchor_accuracy": round(accuracy(anchor), 4),
        "channel_factorial_grid_points": len(GRID),
    }

    best: dict[tuple[str, str], float] = {}
    for source, site in CELLS:
        key = f"{source}_{site}"
        sweep = {alpha: run_cell(source, site, alpha=alpha, **common) for alpha in GRID}
        # Select over POSITIVE weights only, which is the convention the shipped
        # recorders already use and the only one that can report a harmful cell.
        # Maximizing over the full grid picks weight zero whenever a channel
        # harms, and weight zero IS the anchor: the cell would then report the
        # uncoupled accuracy, its paired test would compare the anchor with
        # itself on 0/0 discordant pairs, and a channel that damages the model
        # would appear in the table as doing nothing at all. The shipped coupling
        # recorder selects 0.1 at 70.5371, below the 70.801 anchor, for exactly
        # this reason.
        engaged = {a: hits for a, hits in sweep.items() if a > 0.0}
        top_alpha = max(engaged, key=lambda a: accuracy(engaged[a]))
        top = engaged[top_alpha]
        b, c, p_value = mcnemar(top, anchor)
        best[(source, site)] = accuracy(top)
        values |= {
            f"channel_{key}_best_alpha": top_alpha,
            f"channel_{key}_best_accuracy": round(accuracy(top), 4),
            f"channel_{key}_best_margin_pts": round(accuracy(top) - accuracy(anchor), 4),
            f"channel_{key}_paired_b": b,
            f"channel_{key}_paired_c": c,
            f"channel_{key}_paired_p": float(f"{p_value:.4g}"),
        }

    # The decomposition the factorial exists to produce. Reported as main effects
    # AND as the interaction, because the interaction is the finding: neither
    # factor alone moves the result.
    le = best[("letter", "evidence")]
    lp = best[("letter", "predictive")]
    xe = best[("lexicon", "evidence")]
    xp = best[("lexicon", "predictive")]
    values |= {
        "channel_source_effect_at_evidence_pts": round(xe - le, 4),
        "channel_source_effect_at_predictive_pts": round(xp - lp, 4),
        "channel_site_effect_for_letter_pts": round(lp - le, 4),
        "channel_site_effect_for_lexicon_pts": round(xp - xe, 4),
        "channel_interaction_pts": round((xp - xe) - (lp - le), 4),
        "channel_diagonal_gap_pts": round(xp - le, 4),
    }

    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        ledger = load_ledger(PROJECT_ROOT)
        own = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]
        ledger.replace_claims(
            remove=own,
            batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
        )
    for label, value in sorted(values.items()):
        print(f"  {label:48s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
