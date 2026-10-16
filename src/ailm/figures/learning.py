"""The W2 learning-equals-counting figure (decisions 22–23).

One figure, two scatter panels over the same 900 transition cells:

* **matched priors** — pymdp's learned posterior-mean transition against the
  shipped add-1 counted matrix. Every cell must sit on the identity line to
  float32 rounding; a single point off it falsifies the conjugate-counting
  claim (P1).
* **mismatched priors** — the P2 control: with the Dirichlet initialized at
  k = 2 while the comparator counts at k = 1, the cells *must* leave the
  identity line, most strongly in sparse columns where the prior dominates.
  A tight diagonal here would mean the comparison could not fail.

The matrices are read from ``data/learning_agreement.npz``, written by the
single recorder — the figure never re-runs pymdp — and the panel refuses to
draw if the artifact's deviations disagree with the recorded claims.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ailm.figures._common import (
    project_root,
    token_float,
)
from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = ["generate_learning_agreement", "learning_agreement_data"]


def learning_agreement_data(
    variables: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load the recorder's matrices and hold them against the ledger.

    Raises:
        FileNotFoundError: If the recorder has not produced the artifact.
        ValueError: If the artifact's deviations disagree with the recorded
            claims beyond token rounding, or if the mismatched control fails
            to exceed the tolerance — either way the figure would illustrate
            something the ledger does not say.
    """
    v = _v(variables)
    artifact = project_root() / "data" / "learning_agreement.npz"
    if not artifact.exists():
        raise FileNotFoundError(f"{artifact} is absent; run scripts/record_learning_claims.py")
    payload = np.load(artifact)
    matched_delta = float(np.max(np.abs(payload["learned"] - payload["counted"])))
    mismatched_delta = float(
        np.max(np.abs(payload["mismatched_learned"] - payload["mismatched_counted"]))
    )
    recorded_matched = token_float(v, "LEARNING_MATCHED_MAX_ABS_DELTA")
    recorded_mismatched = token_float(v, "LEARNING_MISMATCHED_MAX_ABS_DELTA")
    tolerance = token_float(v, "LEARNING_TOLERANCE")

    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= 0.02 * max(abs(a), abs(b), 1e-300)

    if not _close(matched_delta, recorded_matched) or not _close(
        mismatched_delta, recorded_mismatched
    ):
        raise ValueError(
            f"artifact deviations ({matched_delta:.3e}, {mismatched_delta:.3e}) "
            f"disagree with the ledger ({recorded_matched:.3e}, "
            f"{recorded_mismatched:.3e}); re-run the recorder."
        )
    if not (matched_delta <= tolerance < mismatched_delta):
        raise ValueError(
            "the artifact does not exhibit the pre-registered partition "
            "(matched within tolerance, mismatched beyond it); the finding "
            "changed and decision 23 must change with it."
        )
    return {
        "learned": payload["learned"],
        "counted": payload["counted"],
        "mismatched_learned": payload["mismatched_learned"],
        "mismatched_counted": payload["mismatched_counted"],
        "matched_delta": matched_delta,
        "mismatched_delta": mismatched_delta,
        "tolerance": tolerance,
    }


def generate_learning_agreement(
    variables: dict[str, str] | None = None,
) -> Figure:
    """Learned vs counted transition cells, matched and mismatched priors."""
    apply_style()
    data = learning_agreement_data(variables)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 5.2))
    floor = 1e-4

    # The clamp and the axis limit were one number, so both panels spanned four
    # decades while every cell in the artifact lives in the top two: roughly
    # three fifths of each panel was empty, and the diagonal a reader is asked to
    # judge departures from ran mostly through that emptiness. The limit is now
    # taken from the data, rounded down to a decade and shared by both panels so
    # the control stays comparable to the measurement, and it can never rise
    # above a point the artifact actually contains.
    smallest = min(
        float(np.min(np.maximum(values.ravel(), floor)))
        for values in (
            data["learned"],
            data["counted"],
            data["mismatched_learned"],
            data["mismatched_counted"],
        )
    )
    lower = max(floor, 10.0 ** np.floor(np.log10(smallest)))

    for axis, learned, counted, title, delta in (
        (
            left,
            data["learned"],
            data["counted"],
            "matched priors (k = 1)",
            data["matched_delta"],
        ),
        (
            right,
            data["mismatched_learned"],
            data["mismatched_counted"],
            "mismatched priors (k = 2 vs k = 1) — the control",
            data["mismatched_delta"],
        ),
    ):
        x = np.maximum(counted.ravel(), floor)
        y = np.maximum(learned.ravel(), floor)
        axis.plot([lower, 1.0], [lower, 1.0], color=PALETTE["OUTLINE"], lw=1.2, linestyle="--")
        axis.scatter(
            x,
            y,
            s=14,
            alpha=0.55,
            color=PALETTE["MODEL"],
            edgecolors="none",
            zorder=3,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, 1.0)
        axis.set_ylim(lower, 1.0)
        axis.set_xlabel("counted  (add-k on the identical stream)")
        axis.set_title(title, fontsize=FONTS["subtitle"])
        axis.text(
            lower * 1.6,
            0.4,
            f"max |Δ| = {delta:.2e}",
            fontsize=FONTS["footnote"],
            color=PALETTE["OUTLINE"],
        )
        axis.spines[["top", "right"]].set_visible(False)
    left.set_ylabel("learned  (pymdp infer_parameters)")
    right.set_ylabel("learned  (pymdp infer_parameters)")

    fig.tight_layout()
    return fig
