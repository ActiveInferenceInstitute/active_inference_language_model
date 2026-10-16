#!/usr/bin/env python3
"""Record the family-disjoint G2P arm (decision 74) and its family-grouped arm (decision 102).

Single writer for every ``stage2_g2p_famsplit_*`` and ``stage2_g2p_famgrp_*``
label.

Why this exists. The shipped Stage 2B split interleaves at the word level, and
the ledger records what the interleave does not remove:
``stage2_g2p_split_shared_prefix_pct``% of held-out words share a four-character
prefix with some training word. The shipped margin could therefore be carried in
unknown part by morphological neighbors. This arm assigns whole prefix families
to one side (``ailm.levels.phoneme.g2p_family_split``), re-fits the identical
decoder and the identical context-free comparator on the disjoint train side,
and records the margin, the discordant counts, and the exact McNemar p on the
disjoint test side.

The family-grouped arm (decision 102) keeps the identical split, decoder, and
comparator and changes only the reporting unit: the held-out set is grouped by
morphological family (``ailm.levels.phoneme.g2p_family_group``) and each
family is scored against the per-letter argmax bar with the paired McNemar
test the level already uses. Aggregate labels report the number of families,
how many carry a positive margin, and the position-weighted pool the positive
control checks. Selected per-family rows are recorded so the heterogeneity is
visible rather than summarized away; the full per-family table is written to
``scratchpad/fleet_2026-09-02/ailm-g2p-famsplit/family_table.json`` when that
directory exists.

EXPLORATORY / DIAGNOSTIC, by its prospective development-session specification
(decision 74; decision 102 inherits the same status). Neither the famsplit nor
the famgrp p-values join ``SIGNIFICANCE_FAMILY``: that family is frozen, and
these arms re-test or re-partition the same Stage 2B hypothesis rather than
posing new ones. They are excluded from the repository-wide family with a
documented reason in ``tests/test_statistics.py``, and the manuscript reports
them as bounds on the leakage story, not as significance claims.

Why the numpy oracle decodes here. The full disjoint test side is tens of
thousands of positions; a pymdp pass is ~one ``infer_states`` call per position.
``scripts/record_stage2_claims.py`` proves the closed-form forward filter is
position-identical to the pymdp agent at this model order on the canonical
split, so these diagnostic arms inherit that equivalence rather than re-running
it, exactly as decisions 74 and 102 register.

Pre-registered predictions. Decision 74 (famsplit): the margin SHRINKS relative
to the shipped ``stage2_g2p_margin_pts``. Decision 102 (famgrp): the margin is
HETEROGENEOUS across families — more than half of the families with at least
40 evaluation positions carry a positive margin; the pooled famsplit result
reproduces exactly (positive control); a deterministic reversal control —
reversed pooled correctness vectors paired with the same family sizes — shows
no between-family structure (negative control; amended from a vacuous
family-rotation draft before any ledger write). Growth, nulls,
or reversals are all admissible outcomes and are recorded as measured.

Usage:
    .venv/bin/python scripts/record_g2p_famsplit_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from record_stage2_claims import (  # noqa: E402
    _initial_letter_tvd,
    _oracle_matrices,
    _shared_prefix_pct,
    oracle_decode,
)

from ailm.claims import load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.letter import ALPHA_TO_IDX  # noqa: E402
from ailm.levels.phoneme import (  # noqa: E402
    G2P_FAMILY_PREFIX_LENGTH,
    G2P_FAMILY_STRIDE,
    g2p_family_group,
    g2p_family_margin_record,
    g2p_family_margin_summary,
    g2p_family_split,
    load_phoneme_counts,
    per_letter_argmax_baseline,
)
from ailm.statistics import exact_two_sided_binomial_p  # noqa: E402

RUN_ID = "stage2_g2p_famsplit"
RUN_ID_FAMGRP = "stage2_g2p_famgrp"
SOURCE = "scripts/record_g2p_famsplit_claims.py"

#: Public aliases retained for the recorder's protocol report. The values have
#: one owner beside the split implementation rather than being redeclared here.
PREFIX_LENGTH = G2P_FAMILY_PREFIX_LENGTH
FAMILY_STRIDE = G2P_FAMILY_STRIDE

#: Refusal bar on the disjoint split's own shared-prefix percentage. The
#: construction makes it exactly zero for words of length >= PREFIX_LENGTH, so
#: anything near the shipped 82.5 means the split function is broken.
MAX_SHARED_PREFIX_PCT = 5.0

#: Decision 102's minimum evaluation positions for a family to enter the
#: sign-share statistic. Below this a family's margin sign is dominated by
#: single-position noise.
FAMGRP_MIN_POSITIONS = 40

#: Positive-control tolerance (decision 102). The per-family pool is the same
#: position set as the famsplit arm partitioned, so the reproduction is exact
#: arithmetic; the tolerance covers only float summation-order rounding.
FAMGRP_POOL_TOLERANCE_PTS = 0.01

#: Negative-control bar (decision 102, amended before any ledger write): the
#: largest absolute per-family margin under the reversal assignment must be
#: STRICTLY SMALLER than the real assignment's. Reaching or exceeding it means
#: the structure is an artifact of the scoring, and the arm refuses to record.
FAMGRP_REVERSAL_STRICTLY_SMALLER = True

#: Where the full per-family table lands when the scratchpad exists. The
#: ledger records aggregates and selected rows; this table is the audit trail.
FAMGRP_TABLE = (
    PROJECT_ROOT / "scratchpad" / "fleet_2026-09-02" / "ailm-g2p-famsplit" / "family_table.json"
)


def _score_pairs(
    pairs: list[tuple[str, list[str]]],
    emission,
    transition,
    start,
    mapping: dict[str, str],
) -> tuple[list[bool], list[bool]]:
    """Paired model/baseline correctness vectors over ``pairs``.

    The identical equal-length filter, forward-filter decode, and per-letter
    lookup the famsplit arm and the shipped pipeline use.
    """
    model_hits: list[bool] = []
    base_hits: list[bool] = []
    for word, phones in pairs:
        decoded = oracle_decode(emission, transition, start, word)
        if len(decoded) != len(phones):
            continue  # the same equal-length filter as the shipped pipeline
        letters = [c for c in word if c in ALPHA_TO_IDX]
        model_hits.extend(d == p for d, p in zip(decoded, phones, strict=True))
        base_hits.extend(mapping.get(c) == p for c, p in zip(letters, phones, strict=True))
    return model_hits, base_hits


def _reversal_family_assignment(
    model_hits: list[bool],
    baseline_hits: list[bool],
    family_position_counts: dict[str, int],
) -> dict[str, tuple[list[bool], list[bool]]]:
    """Pair reversed pooled correctness vectors with the same family sizes.

    The negative control's deterministic, documented assignment (decision
    102, amended before any ledger write): the pooled model and baseline
    vectors are reversed in position order, then cut into families by the
    real assignment's position counts in family-key order. Every marginal
    statistic is preserved exactly — same totals, same family sizes — and
    only the family-to-result pairing is destroyed. No RNG, no seed to
    record.

    Amendment note: the first draft rotated whole correctness vectors one
    family over. That statistic is invariant under any permutation of
    families (it preserves the multiset of per-family margins), so it could
    never fail and was vacuous; this control is discriminating.
    """
    reversed_model = list(reversed(model_hits))
    reversed_base = list(reversed(baseline_hits))
    assignment: dict[str, tuple[list[bool], list[bool]]] = {}
    cursor = 0
    for key in sorted(family_position_counts):
        n = family_position_counts[key]
        assignment[key] = (
            reversed_model[cursor : cursor + n],
            reversed_base[cursor : cursor + n],
        )
        cursor += n
    if cursor != len(model_hits):
        raise SystemExit(
            f"famgrp refused: the reversal control consumed {cursor} of "
            f"{len(model_hits)} positions. The family position counts do not "
            f"partition the pooled vectors. No claims written."
        )
    return assignment


def measure() -> dict[str, float | int]:
    counts = load_phoneme_counts(PROJECT_ROOT / "vendor")
    train, test = g2p_family_split(counts, prefix_length=PREFIX_LENGTH, family_stride=FAMILY_STRIDE)
    families = sorted({word[:PREFIX_LENGTH] for word, _phones in train + test})
    test_families = {word[:PREFIX_LENGTH] for word, _phones in test}

    # ── The leakage guard the arm exists for ─────────────────────────────────
    shared_pct = _shared_prefix_pct(train, test, PREFIX_LENGTH)
    if shared_pct >= MAX_SHARED_PREFIX_PCT:
        raise SystemExit(
            f"famsplit refused: the family-disjoint split still shares {shared_pct:.1f}% "
            f"of {PREFIX_LENGTH}-char prefixes across the split (bar: "
            f"{MAX_SHARED_PREFIX_PCT}). The split function does not do what decision 74 "
            f"registers. No claims written."
        )

    # ── The identical downstream pipeline, re-fitted on the disjoint train ───
    emission, transition, start = _oracle_matrices(train)
    mapping = per_letter_argmax_baseline(train)

    model_hits, base_hits = _score_pairs(test, emission, transition, start, mapping)

    if len(model_hits) != len(base_hits) or not model_hits:
        raise SystemExit(
            f"famsplit refused: model scored {len(model_hits)} positions against the "
            f"comparator's {len(base_hits)}. An unpaired or empty comparison cannot "
            f"feed McNemar. No claims written."
        )

    acc_model = 100.0 * sum(model_hits) / len(model_hits)
    acc_base = 100.0 * sum(base_hits) / len(base_hits)
    b = sum(1 for m, s in zip(model_hits, base_hits, strict=True) if m and not s)
    c = sum(1 for m, s in zip(model_hits, base_hits, strict=True) if s and not m)
    p_value = exact_two_sided_binomial_p(b, c)

    # ── Decision 102: the family-grouped arm ─────────────────────────────────
    # Same positions, same decoder, same comparator — grouped per family.
    groups = g2p_family_group(test, prefix_length=PREFIX_LENGTH)
    model_hits_by_family: dict[str, list[bool]] = {}
    baseline_hits_by_family: dict[str, list[bool]] = {}
    cursor = 0
    for key, family_pairs in groups.items():
        # Recompute per family so the vectors are produced, not sliced — the
        # equal-length filter runs inside _score_pairs, so slicing the pooled
        # vector would misalign whenever any word is filtered out.
        f_model, f_base = _score_pairs(family_pairs, emission, transition, start, mapping)
        if len(f_model) != len(f_base):
            raise SystemExit(
                f"famgrp refused: family {key!r} scored {len(f_model)} model positions "
                f"against {len(f_base)} comparator positions. An unpaired comparison "
                f"cannot feed McNemar. No claims written."
            )
        model_hits_by_family[key] = f_model
        baseline_hits_by_family[key] = f_base
        cursor += len(family_pairs)

    family_results = [
        g2p_family_margin_record(
            key, groups[key], model_hits_by_family[key], baseline_hits_by_family[key]
        )
        for key in sorted(groups)
    ]
    summary = g2p_family_margin_summary(family_results, min_positions=FAMGRP_MIN_POSITIONS)

    # ── Positive control (decision 102): the pool IS the famsplit population ─
    pooled_margin_delta = abs(summary["pool_margin_pts"] - round(acc_model - acc_base, 3))
    pool_accuracy_delta = abs(summary["pool_accuracy"] - round(acc_model, 3))
    if (
        pooled_margin_delta > FAMGRP_POOL_TOLERANCE_PTS
        or pool_accuracy_delta > FAMGRP_POOL_TOLERANCE_PTS
        or summary["pool_positions"] != len(model_hits)
        or summary["pool_model_correct"] != sum(model_hits)
        or summary["pool_baseline_correct"] != sum(base_hits)
    ):
        raise SystemExit(
            f"famgrp refused: the per-family pool (accuracy {summary['pool_accuracy']}, "
            f"margin {summary['pool_margin_pts']}, {summary['pool_positions']} positions) "
            f"does not reproduce the famsplit arm (accuracy {acc_model:.3f}, margin "
            f"{acc_model - acc_base:.3f}, {len(model_hits)} positions) within "
            f"{FAMGRP_POOL_TOLERANCE_PTS} pts. The partition does not cover the same "
            f"population. No claims written."
        )

    # ── Negative control (decision 102): the reversal assignment ─────────────
    family_position_counts = {
        key: int(row["eval_positions"])
        for key, row in zip(sorted(groups), family_results, strict=True)
    }
    reversal = _reversal_family_assignment(model_hits, base_hits, family_position_counts)
    reversed_results = [
        g2p_family_margin_record(key, groups[key], m, s) for key, (m, s) in reversal.items()
    ]
    reversed_summary = g2p_family_margin_summary(
        reversed_results, min_positions=FAMGRP_MIN_POSITIONS
    )
    real_max_abs_margin = max((abs(float(r["margin_pts"])) for r in family_results), default=0.0)
    reversed_max_abs_margin = max(
        (abs(float(r["margin_pts"])) for r in reversed_results), default=0.0
    )
    if reversed_max_abs_margin >= real_max_abs_margin:
        raise SystemExit(
            f"famgrp refused: the reversal control shows the same between-family "
            f"structure as the real assignment (max |margin| {reversed_max_abs_margin} "
            f"vs real {real_max_abs_margin}). The structure is an artifact of the "
            f"scoring, not of the families. No claims written."
        )

    # Selected per-family rows for the ledger: the largest evaluated families,
    # bounded so the ledger records the shape of the heterogeneity, not a dump.
    evaluated = [r for r in family_results if int(r["eval_positions"]) >= FAMGRP_MIN_POSITIONS]
    evaluated_by_size = sorted(evaluated, key=lambda r: int(r["eval_positions"]), reverse=True)
    top_families = evaluated_by_size[:10]

    recorded = {
        "stage2_g2p_famsplit_prefix_length": PREFIX_LENGTH,
        "stage2_g2p_famsplit_family_stride": FAMILY_STRIDE,
        "stage2_g2p_famsplit_families": len(families),
        "stage2_g2p_famsplit_test_families": len(test_families),
        "stage2_g2p_famsplit_train_pairs": len(train),
        "stage2_g2p_famsplit_test_pairs": len(test),
        "stage2_g2p_famsplit_shared_prefix_pct": round(shared_pct, 1),
        "stage2_g2p_famsplit_initial_tvd": round(_initial_letter_tvd(train, test), 5),
        "stage2_g2p_famsplit_eval_positions": len(model_hits),
        "stage2_g2p_famsplit_accuracy": round(acc_model, 3),
        "stage2_g2p_famsplit_baseline_accuracy": round(acc_base, 3),
        "stage2_g2p_famsplit_margin_pts": round(acc_model - acc_base, 3),
        "stage2_g2p_famsplit_mcnemar_b": b,
        "stage2_g2p_famsplit_mcnemar_c": c,
        "stage2_g2p_famsplit_mcnemar_p": float(f"{p_value:.4g}"),
        # Family-grouped arm (decision 102)
        "stage2_g2p_famgrp_min_positions": FAMGRP_MIN_POSITIONS,
        "stage2_g2p_famgrp_families": summary["families"],
        "stage2_g2p_famgrp_evaluated_families": summary["evaluated_families"],
        "stage2_g2p_famgrp_positive_margin_families": summary["positive_margin_families"],
        "stage2_g2p_famgrp_positive_margin_share": summary["positive_margin_share"],
        "stage2_g2p_famgrp_pool_positions": summary["pool_positions"],
        "stage2_g2p_famgrp_pool_accuracy": summary["pool_accuracy"],
        "stage2_g2p_famgrp_pool_baseline_accuracy": summary["pool_baseline_accuracy"],
        "stage2_g2p_famgrp_pool_margin_pts": summary["pool_margin_pts"],
        "stage2_g2p_famgrp_pool_mcnemar_b": b,
        "stage2_g2p_famgrp_pool_mcnemar_c": c,
        "stage2_g2p_famgrp_reversal_max_abs_margin": round(reversed_max_abs_margin, 3),
        "stage2_g2p_famgrp_real_max_abs_margin": round(real_max_abs_margin, 3),
        "stage2_g2p_famgrp_control_confirmed": 1,
    }
    # Ten bounded per-family rows: the heterogeneity's shape, in the ledger.
    for rank, row in enumerate(top_families, start=1):
        recorded[f"stage2_g2p_famgrp_rank{rank}_family"] = str(row["family"])
        recorded[f"stage2_g2p_famgrp_rank{rank}_words"] = int(row["words"])
        recorded[f"stage2_g2p_famgrp_rank{rank}_positions"] = int(row["eval_positions"])
        recorded[f"stage2_g2p_famgrp_rank{rank}_accuracy"] = float(row["accuracy"])
        recorded[f"stage2_g2p_famgrp_rank{rank}_baseline_accuracy"] = float(
            row["baseline_accuracy"]
        )
        recorded[f"stage2_g2p_famgrp_rank{rank}_margin_pts"] = float(row["margin_pts"])
        recorded[f"stage2_g2p_famgrp_rank{rank}_mcnemar_b"] = int(row["mcnemar_b"])
        recorded[f"stage2_g2p_famgrp_rank{rank}_mcnemar_c"] = int(row["mcnemar_c"])
        recorded[f"stage2_g2p_famgrp_rank{rank}_mcnemar_p"] = float(row["mcnemar_p"])

    _write_family_table(family_results, reversed_results, summary, reversed_summary)
    return recorded


def _write_family_table(
    family_results: list[dict[str, float | int | str]],
    reversed_results: list[dict[str, float | int | str]],
    summary: dict[str, float | int],
    reversed_summary: dict[str, float | int],
) -> None:
    """Persist the full per-family table beside the lane's scratchpad."""
    if not FAMGRP_TABLE.parent.is_dir():
        return
    import json

    FAMGRP_TABLE.write_text(
        json.dumps(
            {
                "min_positions": FAMGRP_MIN_POSITIONS,
                "summary": summary,
                "reversal_summary": reversed_summary,
                "families": family_results,
                "reversal": reversed_results,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ledger = load_ledger(PROJECT_ROOT)
    shipped = ledger.get("stage2_g2p_margin_pts")
    if shipped is None:
        raise SystemExit(
            "family-disjoint G2P recorder refused: claim:stage2_g2p_margin_pts "
            "is absent; run scripts/record_stage2_claims.py first. No claims written."
        )

    recorded = measure()
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    width = max(len(label) for label in recorded)
    print(f"Recorded {len(recorded)} family-disjoint G2P claims:\n")
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    delta = recorded["stage2_g2p_famsplit_margin_pts"] - float(shipped["value"])
    verdict = "SHRANK (P1 held)" if delta < 0 else "did NOT shrink (P1 refuted)"
    print(
        f"\nFamsplit margin {recorded['stage2_g2p_famsplit_margin_pts']} pts vs the "
        f"shipped {shipped['value']} pts: the margin {verdict}. Exploratory arm; "
        f"the p-value stays outside the frozen significance family (decision 74)."
    )
    share = recorded["stage2_g2p_famgrp_positive_margin_share"]
    print(
        f"\nFamily-grouped arm (decision 102): "
        f"{recorded['stage2_g2p_famgrp_positive_margin_families']} of "
        f"{recorded['stage2_g2p_famgrp_evaluated_families']} evaluated families carry a "
        f"positive margin (share {share}); reversal-control max |margin| "
        f"{recorded['stage2_g2p_famgrp_reversal_max_abs_margin']} vs real "
        f"{recorded['stage2_g2p_famgrp_real_max_abs_margin']}; pooled reproduction of "
        f"the famsplit arm confirmed within {FAMGRP_POOL_TOLERANCE_PTS} pts. "
        "Decision-102 P1 "
        + ("HELD (more than half positive)" if share > 0.5 else "REFUTED (half or fewer positive)")
        + "."
    )
    return 0


if __name__ == "__main__":
    # The single-writer lock, taken exactly as every other recorder takes it:
    # two recorders writing the ledger at once is how a label acquires a value
    # from one run and a provenance from another.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
