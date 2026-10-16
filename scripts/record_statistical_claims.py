#!/usr/bin/env python3
"""Record the declared multiplicity family and exact uncertainty summaries.

This recorder derives every value from existing ledger counts and p-values.  It
does not re-run a model and it does not treat achieved power as a prospective
sample-size justification; power is retained only as a retrospective
detectability diagnostic for the corpus-determined paired comparisons.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "repository_significance_family_v1"
SOURCE = "scripts/record_statistical_claims.py"


def _exact_count(entries: Mapping[str, Mapping[str, object]], label: str) -> int:
    entry = entries.get(label)
    if entry is None:
        raise SystemExit(f"missing sufficient-count claim {label!r}")
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(
            f"sufficient-count claim {label!r} must be a non-negative integer, got {value!r}"
        )
    return value


def _stored_probability(entries: Mapping[str, Mapping[str, object]], label: str) -> float:
    entry = entries.get(label)
    if entry is None:
        raise SystemExit(f"missing p-value claim {label!r}")
    value = entry.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"p-value claim {label!r} must be numeric, got {value!r}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise SystemExit(f"p-value claim {label!r} must lie in [0, 1], got {value!r}")
    return probability


def reconstruct_family_p_values(
    entries: Mapping[str, Mapping[str, object]],
) -> dict[str, float]:
    """Reconstruct every family p-value from its integer sufficient counts.

    The stored scalar is retained for prose compatibility but is only accepted
    when it equals the exact value or the recorders' canonical four-significant-
    digit representation. Decisions always use the exact reconstruction, so a
    rounded ledger value cannot cross a threshold merely because it rounded.
    """
    from ailm.statistics import SIGNIFICANCE_COUNT_LABELS, exact_two_sided_binomial_p

    reconstructed: dict[str, float] = {}
    for p_label, favored_label, opposed_label in SIGNIFICANCE_COUNT_LABELS:
        favored = _exact_count(entries, favored_label)
        opposed = _exact_count(entries, opposed_label)
        exact = exact_two_sided_binomial_p(favored, opposed)
        stored = _stored_probability(entries, p_label)
        canonical = float(f"{exact:.4g}")
        if not (math.isclose(stored, exact, rel_tol=1e-15, abs_tol=0.0) or stored == canonical):
            raise SystemExit(
                f"p-value claim {p_label!r}={stored!r} disagrees with exact "
                f"reconstruction {exact!r} from {favored_label}={favored} and "
                f"{opposed_label}={opposed} (canonical stored form {canonical!r})"
            )
        reconstructed[p_label] = exact
    return reconstructed


def main() -> int:
    from scipy.stats import binomtest

    from ailm.claims import ClaimBatch, load_ledger
    from ailm.statistics import (
        BONFERRONI_ALPHA,
        FAMILYWISE_ALPHA,
        REFERENCE_SHARE,
        SIGNIFICANCE_COUNT_LABELS,
        SIGNIFICANCE_FAMILY,
        clopper_pearson_interval,
        holm_step_down,
        summarize_paired_effect,
    )

    ledger = load_ledger(_PROJECT_ROOT)

    family_inputs = {label for triple in SIGNIFICANCE_COUNT_LABELS for label in triple}
    missing = sorted(label for label in family_inputs if ledger.get(label) is None)
    if missing:
        raise SystemExit(
            "Cannot record the significance family; run the owning recorders for "
            f"these missing p-values or sufficient counts first: {missing}"
        )
    if len(SIGNIFICANCE_FAMILY) != len(set(SIGNIFICANCE_FAMILY)):
        raise SystemExit("SIGNIFICANCE_FAMILY contains duplicate labels")

    entries = {label: ledger.get(label) for label in family_inputs}
    p_values = reconstruct_family_p_values(entries)  # type: ignore[arg-type]
    survivors = sorted(label for label, p_value in p_values.items() if p_value < BONFERRONI_ALPHA)

    values: dict[str, object] = {
        "stat_familywise_alpha": FAMILYWISE_ALPHA,
        "stat_family_size": len(SIGNIFICANCE_FAMILY),
        "stat_bonferroni_alpha": BONFERRONI_ALPHA,
        "stat_family_survivor_count": len(survivors),
        "stat_family_survivors": survivors,
        "stat_stage6_familywise_confirmed": (p_values["stage6_sign_test_p"] < BONFERRONI_ALPHA),
        "stat_reference_share": REFERENCE_SHARE,
    }

    # Holm's step-down correction over the same family and the same alpha.
    # Holm controls the familywise rate under the same arbitrary dependence
    # Bonferroni assumes and rejects a superset of what Bonferroni rejects, so
    # the two are not alternatives to choose between: the fixed threshold stays
    # the automated regression gate because it is a constant, and Holm is the
    # sharper scientific reading of the same sixteen tests.
    holm = holm_step_down(p_values, FAMILYWISE_ALPHA)
    holm_survivors = sorted(decision.label for decision in holm if decision.rejected)
    holm_only = sorted(set(holm_survivors) - set(survivors))
    stage6 = next(decision for decision in holm if decision.label == "stage6_sign_test_p")
    values.update(
        {
            "stat_holm_survivor_count": len(holm_survivors),
            "stat_holm_survivors": holm_survivors,
            "stat_holm_only_survivors": holm_only,
            "stat_holm_only_count": len(holm_only),
            "stat_holm_stage6_rank": stage6.rank,
            "stat_holm_stage6_threshold": float(f"{stage6.threshold:.6g}"),
            "stat_holm_stage6_rejected": stage6.rejected,
        }
    )
    if not set(survivors) <= set(holm_survivors):
        raise SystemExit(
            "Holm rejected less than Bonferroni, which is impossible; refusing "
            f"to record: Bonferroni={survivors} Holm={holm_survivors}"
        )

    # Sidedness sensitivity for the one directional pre-registration in the
    # family. The recorded family test is the conservative two-sided form; the
    # one-sided p appropriate to the pre-registered direction is recorded so
    # the manuscript can state the sensitivity rather than hide it.
    positive = _exact_count(entries, "stage6_sign_test_positive")  # type: ignore[arg-type]
    negative = _exact_count(entries, "stage6_sign_test_negative")  # type: ignore[arg-type]
    one_sided = binomtest(positive, positive + negative, 0.5, alternative="greater").pvalue
    values["stat_stage6_sign_test_one_sided_p"] = float(f"{one_sided:.6g}")

    effects = {
        "stage2": ("stage2_g2p_mcnemar_b", "stage2_g2p_mcnemar_c"),
        "stage4": ("stage4_derived_mcnemar_b", "stage4_derived_mcnemar_c"),
        "lexical": ("lexical_best_mcnemar_b", "lexical_best_mcnemar_c"),
        # Here "favored" means the uncoupled decoder is correct: the measured
        # effect is harm from coupling, so c is the scientifically favored side.
        "coupling_harm": ("coupling_mcnemar_c", "coupling_mcnemar_b"),
    }
    for prefix, (favored_label, opposed_label) in effects.items():
        favored_entry = ledger.get(favored_label)
        opposed_entry = ledger.get(opposed_label)
        if favored_entry is None or opposed_entry is None:
            raise SystemExit(
                f"Cannot summarize {prefix}: missing {favored_label!r} or {opposed_label!r}"
            )
        summary = summarize_paired_effect(
            _exact_count({favored_label: favored_entry}, favored_label),
            _exact_count({opposed_label: opposed_entry}, opposed_label),
        )
        # Bonferroni-adjusted marginal interval: the decision threshold is
        # BONFERRONI_ALPHA, so the matching interval level is 1 - that alpha.
        corrected_low, _corrected_high = clopper_pearson_interval(
            summary.favored, summary.n, confidence=1.0 - BONFERRONI_ALPHA
        )
        corrected_excludes_half = corrected_low > 0.5
        values.update(
            {
                f"stat_{prefix}_discordant_n": summary.n,
                f"stat_{prefix}_favored_ratio": round(summary.ratio, 6),
                # Level-neutral names. `ci99` baked a coverage into a label, so
                # the name went false the moment decision 37 moved the level and
                # nothing in the ledger could notice. The coverage now lives in
                # SHARE_CONFIDENCE alone, which is the only place it can drift.
                f"stat_{prefix}_share_ci_low": round(summary.ci_low, 6),
                f"stat_{prefix}_share_ci_high": round(summary.ci_high, 6),
                f"stat_{prefix}_power_corrected": round(summary.power, 6),
                f"stat_{prefix}_power_at_reference": round(summary.power_at_reference, 6),
                f"stat_{prefix}_corrected_ci_excludes_half": corrected_excludes_half,
            }
        )

    # Retire what this recorder used to write and no longer does. Ownership is
    # by recorded ``source``, not by prefix: other recorders also write ``stat_``
    # labels, and a prefix sweep here would delete theirs. Without this, a
    # renamed measurement leaves its old value on disk looking current — which
    # is not hypothetical: the British-to-American spelling change would
    # have left four stale ratios behind, each still resolving into prose that
    # no longer cites it.
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
    )
    if stale:
        print(f"  retired {len(stale)} label(s) this recorder no longer produces: {sorted(stale)}")
    print(
        f"Recorded {len(values)} statistical claims: "
        f"{len(survivors)}/{len(SIGNIFICANCE_FAMILY)} tests survive "
        f"Bonferroni alpha={BONFERRONI_ALPHA:.9f}; "
        f"{len(holm_survivors)}/{len(SIGNIFICANCE_FAMILY)} survive Holm at "
        f"alpha={FAMILYWISE_ALPHA} (gained: {holm_only or 'none'})."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
