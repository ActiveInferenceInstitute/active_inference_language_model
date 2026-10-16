"""Direction words about the cluster-robust reanalysis, re-derived from the ledger.

What this file protects
-----------------------
The variance-inflation column of ``tbl:cluster`` is a ratio of two bootstrap
replicate variances: a Monte Carlo quantity, recomputed on every recorder run.
The prose around it is English — "below one, because the measured intraclass
correlation is negative", "anti-conservative", "widens some arms and tightens
others" — and English does not re-render. ``04_results.md`` once stated a
per-arm direction list that was false at HEAD while every token in it printed
a correct digit. The Stage 2B ratio in particular sits within the bootstrap's
resolution of one, so its sign flips with the seed; the supplement now says it
is not sign-interpretable rather than asserting a direction.

Each test re-derives one direction sentence the manuscript still states, and
locks only relations the ledger supports at HEAD. The Monte Carlo resolution
is derived from the recorded replicate count rather than pinned.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"
SUPPLEMENT = PROJECT_ROOT / "docs" / "manuscript" / "98_supplement_statistics.md"
RESULTS = PROJECT_ROOT / "docs" / "manuscript" / "04_results.md"

_NEEDS_LEDGER = pytest.mark.skipif(
    not LEDGER.exists(), reason="data/claims.json absent; run the recorders"
)

#: The arms ``tbl:cluster`` tabulates, in table order.
TABLE_ARMS = ("stage2_g2p", "stage4_derived", "stage6_logloss", "stage6_sign")

#: Sentence fragments whose truth the tests below re-derive. Whitespace is
#: collapsed before matching so a fragment may span the source's hard wrap.
GUARDED = {
    SUPPLEMENT: (
        "below one, because the measured intraclass correlation is negative",
        "anti-conservative even though it does not change the sign of the result",
        "indistinguishable from one at the bootstrap's Monte Carlo resolution",
        "is not sign-interpretable",
    ),
    RESULTS: ("clustering widens some arms and tightens others",),
}


@lru_cache(maxsize=1)
def _claims() -> dict[str, dict]:
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


def _value(claims: dict[str, dict], label: str) -> float:
    assert label in claims, f"{label} is not in the ledger; this guard would be vacuous"
    return float(claims[label]["value"])


def monte_carlo_resolution(resamples: int) -> float:
    """Approximate relative standard error of a ratio of two sample variances.

    Each variance is estimated from ``resamples`` bootstrap replicates, so its
    relative standard error is about ``sqrt(2 / (R - 1))``; the ratio of two
    independent ones carries twice that variance. The bound is used at two
    standard errors, so a ratio inside it is not sign-interpretable.
    """
    if resamples < 2:
        raise ValueError("resolution needs at least two replicates")
    return math.sqrt(4.0 / (resamples - 1))


def direction_defects(claims: dict[str, dict]) -> list[str]:
    """Every direction sentence that the recorded values no longer support."""
    defects: list[str] = []
    inflation = {arm: _value(claims, f"cluster_{arm}_variance_inflation") for arm in TABLE_ARMS}
    icc = {arm: _value(claims, f"cluster_{arm}_icc") for arm in TABLE_ARMS}

    # "Stage 6's secondary sign arm tightens ... below one, because the measured
    # intraclass correlation is negative"
    if not (inflation["stage6_sign"] < 1.0 and icc["stage6_sign"] < 0.0):
        defects.append(
            f"stage6_sign: inflation {inflation['stage6_sign']} / ICC {icc['stage6_sign']} no "
            "longer read 'below one, because the intraclass correlation is negative'"
        )

    # "treating ... nested tokens as independent is anti-conservative"
    if not (inflation["stage4_derived"] > 1.0 and icc["stage4_derived"] > 0.0):
        defects.append(
            f"stage4_derived: inflation {inflation['stage4_derived']} / ICC "
            f"{icc['stage4_derived']} no longer read 'anti-conservative'"
        )

    # "clustering widens some arms and tightens others"
    if not (any(v > 1.0 for v in inflation.values()) and any(v < 1.0 for v in inflation.values())):
        defects.append(
            f"no longer both directions among {inflation}; 'widens some arms and tightens "
            "others' is false"
        )

    # Stage 2B: "indistinguishable from one at the bootstrap's Monte Carlo
    # resolution", with a positive ICC whose analytic design effect is only
    # marginally above one.
    resolution = monte_carlo_resolution(int(_value(claims, "cluster_resamples")))
    if abs(inflation["stage2_g2p"] - 1.0) > 2.0 * resolution:
        defects.append(
            f"stage2_g2p: inflation {inflation['stage2_g2p']} is more than two Monte Carlo "
            f"standard errors ({2.0 * resolution:.4f}) from one; it is sign-interpretable "
            "and the caption's 'not sign-interpretable' reading is wrong"
        )
    mean_cluster = _value(claims, "cluster_stage2_g2p_observations") / _value(
        claims, "cluster_stage2_g2p_clusters"
    )
    design_effect = 1.0 + (mean_cluster - 1.0) * icc["stage2_g2p"]
    if not (icc["stage2_g2p"] > 0.0 and 1.0 < design_effect < 1.1):
        defects.append(
            f"stage2_g2p: ICC {icc['stage2_g2p']} gives an analytic design effect of "
            f"{design_effect}; the caption says a positive intraclass correlation implies a "
            "design effect only marginally above one"
        )
    return defects


@_NEEDS_LEDGER
def test_the_cluster_direction_sentences_hold_at_head() -> None:
    assert direction_defects(_claims()) == []


@_NEEDS_LEDGER
def test_the_guarded_sentences_are_still_in_the_manuscript() -> None:
    """POSITIVE CONTROL: a reworded sentence must re-derive its guard, not orphan it."""
    for path, fragments in GUARDED.items():
        text = " ".join(path.read_text(encoding="utf-8").split())
        missing = [fragment for fragment in fragments if fragment not in text]
        assert missing == [], f"{path.name} no longer contains guarded fragments {missing}"


@_NEEDS_LEDGER
def test_each_direction_check_can_fail() -> None:
    """NEGATIVE CONTROL: every relation is checked, not merely computed."""
    base = _claims()

    def mutated(**changes: float) -> dict[str, dict]:
        changed = deepcopy(base)
        for label, value in changes.items():
            changed[label]["value"] = value
        return changed

    assert any(
        "stage6_sign" in defect
        for defect in direction_defects(mutated(cluster_stage6_sign_icc=0.05))
    )
    assert any(
        "stage4_derived" in defect
        for defect in direction_defects(mutated(cluster_stage4_derived_variance_inflation=0.9))
    )
    everything_widens = mutated(
        cluster_stage6_sign_variance_inflation=1.2,
        cluster_stage6_sign_icc=0.05,
        cluster_stage2_g2p_variance_inflation=1.2,
    )
    assert any("both directions" in defect for defect in direction_defects(everything_widens))
    assert any(
        "sign-interpretable" in defect
        for defect in direction_defects(mutated(cluster_stage2_g2p_variance_inflation=0.8))
    )
    assert any(
        "design effect" in defect
        for defect in direction_defects(mutated(cluster_stage2_g2p_icc=-0.01))
    )


def test_the_resolution_shrinks_with_replicates_and_rejects_a_degenerate_count() -> None:
    assert monte_carlo_resolution(10_000) < monte_carlo_resolution(1_000)
    assert monte_carlo_resolution(10_000) == pytest.approx(math.sqrt(4.0 / 9_999))
    with pytest.raises(ValueError):
        monte_carlo_resolution(1)
