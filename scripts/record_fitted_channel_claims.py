"""Fit the lexical channel's hyperparameters instead of choosing them.

Three numbers in `build_lexical_message` were set by hand and never fitted:
``max_prefix``, ``min_count`` and ``smoothing``. The pooling exponent was
likewise read off a declared grid. This recorder selects all four on a split
taken from TRAIN and evaluates once on the held-out TEST set.

The leak guard is the whole design. Selection runs on an inner split of TRAIN,
and the matrices used during selection are built from the inner fit half only, so
no scored position and no statistic derived from one is visible while the
hyperparameters are chosen. TEST is scored exactly once, after selection closes.
Choosing a value on the positions the headline is reported on is the single most
likely way this arm produces a fake gain, and it is the reason the protocol is
recorded rather than described.

The inner-dev gain and the test gain are both recorded. Two similar numbers are
evidence the fit generalized; a large dev gain beside a small test gain would be
evidence it did not, and reporting only the second would hide which happened.

The protocol and its scoring rule are prospectively specified in
docs/decisions.md entry 49. That entry and this recorder first appear in the
same commit, so the held-out score is exploratory rather than independently
preregistered or confirmatory (see scripts/assign_claim_status.py and
data/registration_provenance.json).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.channels import _geometric_positions  # noqa: E402
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.coupling import build_g2p_matrices, mcnemar  # noqa: E402
from ailm.levels.letter import ALPHA_TO_IDX  # noqa: E402
from ailm.levels.lexical_channel import build_lexical_message, lexical_message_for  # noqa: E402
from ailm.levels.phoneme import PHONES, _iter_pairs, g2p_split, load_phoneme_counts  # noqa: E402

SOURCE = "scripts/record_fitted_channel_claims.py"
RUN_ID = "fitted-channel"
INNER_STRIDE = 5
PREFIXES = (4, 6, 8)
MIN_COUNTS = (2, 3, 5)
SMOOTHINGS = (0.5, 1.0, 2.0)
EXPONENTS = (0.3, 0.5, 0.75)


def main() -> int:
    vendor = PROJECT_ROOT / "vendor"
    train, test = g2p_split(load_phoneme_counts(vendor))
    phones = list(PHONES)
    phone_to_index = {p: i for i, p in enumerate(phones)}

    inner_dev = [pair for i, pair in enumerate(train) if i % INNER_STRIDE == 0]
    inner_fit = [pair for i, pair in enumerate(train) if i % INNER_STRIDE != 0]
    leaked = {w for w, _ in inner_dev} & {w for w, _ in test}
    if leaked:
        raise SystemExit(
            f"refused: {len(leaked)} selection words also appear in the held-out set, so a "
            f"hyperparameter chosen here would be chosen on scored positions"
        )

    def scored(pairs, matrices, message, exponent) -> list[bool]:
        emission = np.asarray(matrices.emission, dtype=np.float64)
        transition = np.asarray(matrices.transition, dtype=np.float64)
        start = np.asarray(matrices.start, dtype=np.float64)
        hits: list[bool] = []
        for word, gold in _iter_pairs(pairs, None):
            indices = [ALPHA_TO_IDX[ch] for ch in word]
            truth = [phone_to_index[p] for p in gold]
            rows = np.asarray(lexical_message_for(message, word)[0])
            hits.extend(
                _geometric_positions(emission, transition, start, indices, truth, rows, exponent)
            )
        return hits

    def accuracy(hits: list[bool]) -> float:
        return 100.0 * sum(hits) / len(hits)

    # ── SELECTION, on TRAIN only ─────────────────────────────────────────────
    inner_matrices = build_g2p_matrices(inner_fit)
    best = None
    for prefix, minimum, smoothing in itertools.product(PREFIXES, MIN_COUNTS, SMOOTHINGS):
        message = build_lexical_message(
            inner_fit, phones, max_prefix=prefix, min_count=minimum, smoothing=smoothing
        )
        for exponent in EXPONENTS:
            value = accuracy(scored(inner_dev, inner_matrices, message, exponent))
            if best is None or value > best[0]:
                best = (value, prefix, minimum, smoothing, exponent)
    if best is None:
        raise RuntimeError("the fitted-channel hyperparameter grid is empty")
    dev_accuracy, prefix, minimum, smoothing, exponent = best
    default_dev = accuracy(
        scored(inner_dev, inner_matrices, build_lexical_message(inner_fit, phones), 0.5)
    )

    # ── EVALUATION, once, on the untouched held-out set ──────────────────────
    matrices = build_g2p_matrices(train)
    shipped = build_lexical_message(train, phones)
    fitted = build_lexical_message(
        train, phones, max_prefix=prefix, min_count=minimum, smoothing=smoothing
    )
    anchor = scored(test, matrices, shipped, 0.0)
    shipped_hits = scored(test, matrices, shipped, 0.5)
    fitted_hits = scored(test, matrices, fitted, exponent)
    b, c, p_value = mcnemar(fitted_hits, shipped_hits)

    values: dict[str, object] = {
        "fitted_inner_fit_pairs": len(inner_fit),
        "fitted_inner_dev_pairs": len(inner_dev),
        "fitted_grid_points": len(PREFIXES) * len(MIN_COUNTS) * len(SMOOTHINGS) * len(EXPONENTS),
        "fitted_max_prefix": prefix,
        "fitted_min_count": minimum,
        "fitted_smoothing": smoothing,
        "fitted_exponent": exponent,
        "fitted_dev_accuracy": round(dev_accuracy, 4),
        "fitted_dev_gain_pts": round(dev_accuracy - default_dev, 4),
        "fitted_positions": len(anchor),
        "fitted_anchor_accuracy": round(accuracy(anchor), 4),
        "fitted_default_accuracy": round(accuracy(shipped_hits), 4),
        "fitted_accuracy": round(accuracy(fitted_hits), 4),
        "fitted_gain_pts": round(accuracy(fitted_hits) - accuracy(shipped_hits), 4),
        "fitted_margin_over_anchor_pts": round(accuracy(fitted_hits) - accuracy(anchor), 4),
        "fitted_paired_b": b,
        "fitted_paired_c": c,
        "fitted_paired_p": float(f"{p_value:.4g}"),
    }

    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        ledger = load_ledger(PROJECT_ROOT)
        ledger.replace_claims(
            remove=[label for label in ledger.labels_owned_by(SOURCE) if label not in values],
            batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
        )
    for label, value in sorted(values.items()):
        print(f"  {label:38s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
