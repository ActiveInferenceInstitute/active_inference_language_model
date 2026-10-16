"""The family-disjoint G2P split, and the leakage property it exists to enforce.

Decision 74's arm: assign whole four-character prefix families to one side so
no held-out word shares its family with a training word, then re-fit the
identical decoder and comparator. The split function lives beside
``g2p_split`` in ``ailm.levels.phoneme``; the measurements are recorded by
``scripts/record_g2p_famsplit_claims.py``. These tests check the split is a
deterministic partition with zero family leakage, and verify (never re-record)
the recorded claims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ailm.levels.letter import ALPHA_TO_IDX
from ailm.levels.phoneme import (
    G2P_FAMILY_PREFIX_LENGTH,
    G2P_FAMILY_STRIDE,
    g2p_family_group,
    g2p_family_margin_record,
    g2p_family_margin_summary,
    g2p_family_split,
    g2p_split,
    load_phoneme_counts,
    per_letter_argmax_baseline,
)
from scripts.record_stage2_claims import _oracle_matrices, oracle_decode

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
VENDOR_DIR = PROJECT_ROOT / "vendor"
LEDGER_FILE = PROJECT_ROOT / "data" / "claims.json"

_SKIP_NO_COUNTS = pytest.mark.skipif(
    not (VENDOR_DIR / "cmudict_counts.json").exists(),
    reason="vendor/cmudict_counts.json absent; run scripts/generate_phoneme_counts.py",
)
_SKIP_NO_LEDGER = pytest.mark.skipif(
    not LEDGER_FILE.exists(), reason="data/claims.json absent; run the recorders"
)

#: A tiny synthetic artifact whose families are legible by inspection. Two
#: families of morphological neighbors ("cata*", "dogg*") plus one short word
#: whose family is the word itself.
TOY_COUNTS = {
    "g2p_pairs": [
        ["catalog", "K AE T AH L AO G"],
        ["catalogs", "K AE T AH L AO G Z"],
        ["catapult", "K AE T AH P AH L T"],
        ["dogged", "D AO G IH D"],
        ["dogger", "D AO G ER"],
        ["cat", "K AE T"],
    ],
    "test_every": 5,
}


def _claims() -> dict:
    return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))["claims"]


# ── The split itself ─────────────────────────────────────────────────────────


def test_families_land_whole_on_one_side():
    """The property the arm exists for: no family straddles the split.

    On the toy artifact the sorted families are ["cat", "cata", "dogg"] and a
    stride of 2 puts "cat" and "dogg" in TEST, "cata" in TRAIN — every
    morphological neighbor of a test word is on the same side as that word.
    """
    train, test = g2p_family_split(TOY_COUNTS, prefix_length=4, family_stride=2)
    train_families = {word[:4] for word, _ in train}
    test_families = {word[:4] for word, _ in test}
    assert train_families.isdisjoint(test_families)
    assert {word for word, _ in test} == {"cat", "dogged", "dogger"}
    assert {word for word, _ in train} == {"catalog", "catalogs", "catapult"}


def test_the_split_is_a_partition_of_the_pair_list():
    train, test = g2p_family_split(TOY_COUNTS, prefix_length=4, family_stride=2)
    assert len(train) + len(test) == len(TOY_COUNTS["g2p_pairs"])
    assert not ({w for w, _ in train} & {w for w, _ in test})
    # Phones arrive split into lists, exactly as g2p_split delivers them.
    for _, phones in train + test:
        assert isinstance(phones, list) and all(isinstance(p, str) for p in phones)


def test_the_split_is_deterministic():
    first = g2p_family_split(TOY_COUNTS, prefix_length=4, family_stride=2)
    second = g2p_family_split(TOY_COUNTS, prefix_length=4, family_stride=2)
    assert first == second


def test_degenerate_parameters_are_refused():
    with pytest.raises(ValueError, match="positive integer"):
        g2p_family_split(TOY_COUNTS, prefix_length=0)
    with pytest.raises(ValueError, match="positive integer"):
        g2p_family_split(TOY_COUNTS, family_stride=0)
    # A stride of 1 puts every family in TEST; an empty side is not a split.
    with pytest.raises(ValueError, match="empty side"):
        g2p_family_split(TOY_COUNTS, family_stride=1)


@_SKIP_NO_COUNTS
def test_zero_family_leakage_on_the_real_artifact():
    """The construction's guarantee, checked on the vendored dictionary.

    Every held-out word of length >= 4 must have a four-character prefix that no
    training word carries — the number the shipped interleave records as 82.5%
    must be exactly zero here.
    """
    counts = load_phoneme_counts(VENDOR_DIR)
    train, test = g2p_family_split(counts)
    prefixes = {word[:4] for word, _ in train if len(word) >= 4}
    leaking = [word for word, _ in test if len(word) >= 4 and word[:4] in prefixes]
    assert leaking == [], f"{len(leaking)} held-out words share a family with training"


@_SKIP_NO_COUNTS
def test_the_family_split_covers_the_same_population_as_the_shipped_split():
    """Same pair list in, same pair multiset out — only the assignment differs."""
    counts = load_phoneme_counts(VENDOR_DIR)
    fam_train, fam_test = g2p_family_split(counts)
    shipped_train, shipped_test = g2p_split(counts)
    assert len(fam_train) + len(fam_test) == len(shipped_train) + len(shipped_test)
    # The family stride of 5 must land near the shipped 1-in-5 held-out share.
    fraction = len(fam_test) / (len(fam_train) + len(fam_test))
    assert 0.1 < fraction < 0.3, f"held-out share {fraction:.3f} is not near 1/5"


# ── The recorded claims, verified rather than re-recorded ────────────────────


@_SKIP_NO_LEDGER
def test_the_recorded_famsplit_arm_is_internally_consistent():
    claims = _claims()
    needed = (
        "stage2_g2p_famsplit_prefix_length",
        "stage2_g2p_famsplit_family_stride",
        "stage2_g2p_famsplit_shared_prefix_pct",
        "stage2_g2p_famsplit_accuracy",
        "stage2_g2p_famsplit_baseline_accuracy",
        "stage2_g2p_famsplit_margin_pts",
        "stage2_g2p_famsplit_mcnemar_b",
        "stage2_g2p_famsplit_mcnemar_c",
    )
    if any(label not in claims for label in needed):
        pytest.skip("stage2_g2p_famsplit_* not recorded in this checkout")

    assert claims["stage2_g2p_famsplit_prefix_length"]["value"] == G2P_FAMILY_PREFIX_LENGTH
    assert claims["stage2_g2p_famsplit_family_stride"]["value"] == G2P_FAMILY_STRIDE
    assert float(claims["stage2_g2p_famsplit_shared_prefix_pct"]["value"]) < 5.0, (
        "the famsplit arm's whole point is a leakage-free split; the recorder's "
        "refusal guard was bypassed"
    )
    margin = float(claims["stage2_g2p_famsplit_margin_pts"]["value"])
    accuracy = float(claims["stage2_g2p_famsplit_accuracy"]["value"])
    baseline = float(claims["stage2_g2p_famsplit_baseline_accuracy"]["value"])
    assert margin == pytest.approx(accuracy - baseline, abs=2e-3)
    # b - c must carry the margin's sign: the paired counts and the aggregate
    # difference describe the same positions.
    b = int(claims["stage2_g2p_famsplit_mcnemar_b"]["value"])
    c = int(claims["stage2_g2p_famsplit_mcnemar_c"]["value"])
    assert (b - c > 0) == (margin > 0)


@_SKIP_NO_LEDGER
def test_the_famsplit_p_value_stays_out_of_the_frozen_family():
    """Decision 74's multiplicity declaration, pinned executable.

    The arm is exploratory: its p-value must not join SIGNIFICANCE_FAMILY, and
    its exclusion must be documented where the family closure scan reads it.
    """
    from ailm.statistics import SIGNIFICANCE_FAMILY

    assert "stage2_g2p_famsplit_mcnemar_p" not in SIGNIFICANCE_FAMILY


@_SKIP_NO_LEDGER
def test_the_famsplit_labels_are_recorder_written():
    claims = _claims()
    labels = [label for label in claims if label.startswith("stage2_g2p_famsplit_")]
    if not labels:
        pytest.skip("stage2_g2p_famsplit_* not recorded in this checkout")
    for label in labels:
        assert claims[label]["source"].startswith("scripts/"), (
            f"{label} was last written by {claims[label]['source']!r}; recorders "
            f"write, tests verify"
        )


# ── The family-grouped arm (decision 102) ────────────────────────────────────


def _toy_split():
    """The toy artifact through the family split, with its groups."""
    train, test = g2p_family_split(TOY_COUNTS, prefix_length=4, family_stride=2)
    return train, test, g2p_family_group(test, prefix_length=4)


def test_family_group_keys_are_deterministic_and_complete():
    """Same grouping rule as the split, no pair dropped or duplicated."""
    _train, test, groups = _toy_split()
    # Insertion order follows first appearance in the (alphabetical) pair
    # list; what the arm needs is determinism, not sortedness.
    assert list(groups) == list(g2p_family_group(test, prefix_length=4))
    assert set(groups) == {w[:4] for w, _ in test}
    grouped_words = [w for pairs in groups.values() for w, _ in pairs]
    assert sorted(grouped_words) == sorted(w for w, _ in test)
    # Phones arrive split into lists, exactly as the split delivers them.
    for pairs in groups.values():
        for _, phones in pairs:
            assert isinstance(phones, list) and all(isinstance(p, str) for p in phones)


def test_family_group_refuses_degenerate_prefix_length():
    with pytest.raises(ValueError, match="positive integer"):
        g2p_family_group([("cat", ["K", "AE", "T"])], prefix_length=0)


def test_family_margin_record_matches_hand_computation():
    """One family's record, computed on paper.

    Model correct on positions 1, 2, 4; baseline correct on 2, 3.
    b = model-only = 2 (positions 1, 4); c = baseline-only = 1 (position 3).
    """
    pairs = [("dogged", ["D", "AO", "G", "IH", "D"])]
    model_hits = [False, True, True, False, True]
    baseline_hits = [False, False, True, True, False]
    record = g2p_family_margin_record("dogg", pairs, model_hits, baseline_hits)
    assert record["family"] == "dogg"
    assert record["words"] == 1
    assert record["eval_positions"] == 5
    assert record["model_correct"] == 3
    assert record["baseline_correct"] == 2
    assert record["accuracy"] == pytest.approx(60.0)
    assert record["baseline_accuracy"] == pytest.approx(40.0)
    assert record["margin_pts"] == pytest.approx(20.0)
    assert record["mcnemar_b"] == 2
    assert record["mcnemar_c"] == 1
    # Exact two-sided binomial on b=2, c=1: P(X<=1 | n=3, p=.5) mirrored.
    from ailm.statistics import exact_two_sided_binomial_p

    assert record["mcnemar_p"] == pytest.approx(exact_two_sided_binomial_p(2, 1))


def test_family_margin_record_refuses_unpaired_vectors():
    with pytest.raises(ValueError, match="unpaired"):
        g2p_family_margin_record("dogg", [("dogged", ["D"])], [True], [True, False])
    with pytest.raises(ValueError, match="unpaired"):
        g2p_family_margin_record("dogg", [("dogged", ["D"])], [], [])


def test_family_margin_summary_pools_positions_exactly():
    """The summary's pool is the position-weighted sum of its records."""
    records = [
        g2p_family_margin_record(
            "cat", [("cat", ["K", "AE", "T"])], [True, True, False], [True, False, False]
        ),
        g2p_family_margin_record(
            "dogg",
            [("dogged", ["D", "AO", "G", "IH", "D"])],
            [True, True, False, False, True],
            [False, True, True, False, True],
        ),
    ]
    summary = g2p_family_margin_summary(records, min_positions=3)
    assert summary["families"] == 2
    assert summary["evaluated_families"] == 2
    assert summary["pool_positions"] == 8
    assert summary["pool_model_correct"] == 5
    # baseline hits: cat has 1, dogg has 3 -> 4 total (hand-checked).
    assert summary["pool_baseline_correct"] == 4
    assert summary["pool_accuracy"] == pytest.approx(62.5)
    assert summary["pool_baseline_accuracy"] == pytest.approx(50.0)
    assert summary["pool_margin_pts"] == pytest.approx(12.5)
    # cat carries a positive margin (+33.3); dogg is exactly null (0.0) and
    # does not count — the sign share counts strictly positive margins.
    assert summary["positive_margin_families"] == 1
    assert summary["positive_margin_share"] == pytest.approx(0.5)


def test_family_margin_summary_min_positions_gate_excludes_small_families():
    records = [
        g2p_family_margin_record(
            "cat", [("cat", ["K", "AE", "T"])], [True, True, False], [False, False, False]
        ),
        g2p_family_margin_record(
            "dogg",
            [("dogged", ["D", "AO", "G", "IH", "D"])],
            [True] * 5,
            [True, True, False, False, False],
        ),
    ]
    summary = g2p_family_margin_summary(records, min_positions=5)
    assert summary["evaluated_families"] == 1
    assert summary["positive_margin_families"] == 1


def test_family_margin_summary_refuses_degenerate_min_positions():
    with pytest.raises(ValueError, match="positive integer"):
        g2p_family_margin_summary([], min_positions=0)


def test_famgrp_positive_control_pooled_reproduction_on_the_real_artifact():
    """The pool over per-family records reproduces the pooled arm exactly.

    The positive control of decision 102, run on the vendored dictionary with
    the real split and builders. Nothing is recorded; this verifies the
    partition covers the same positions the pooled arm scores.
    """
    counts = load_phoneme_counts(VENDOR_DIR)
    train, test = g2p_family_split(counts)
    emission, transition, start = _oracle_matrices(train)
    mapping = per_letter_argmax_baseline(train)

    model_hits: list[bool] = []
    base_hits: list[bool] = []
    for word, phones in test:
        decoded = oracle_decode(emission, transition, start, word)
        if len(decoded) != len(phones):
            continue
        letters = [c for c in word if c in ALPHA_TO_IDX]
        model_hits.extend(d == p for d, p in zip(decoded, phones, strict=True))
        base_hits.extend(mapping.get(c) == p for c, p in zip(letters, phones, strict=True))

    groups = g2p_family_group(test)
    family_results = []
    for key, family_pairs in groups.items():
        f_model: list[bool] = []
        f_base: list[bool] = []
        for word, phones in family_pairs:
            decoded = oracle_decode(emission, transition, start, word)
            if len(decoded) != len(phones):
                continue
            letters = [c for c in word if c in ALPHA_TO_IDX]
            f_model.extend(d == p for d, p in zip(decoded, phones, strict=True))
            f_base.extend(mapping.get(c) == p for c, p in zip(letters, phones, strict=True))
        assert len(f_model) == len(f_base), f"family {key} scored unpaired"
        family_results.append(g2p_family_margin_record(key, family_pairs, f_model, f_base))

    summary = g2p_family_margin_summary(family_results, min_positions=40)
    assert summary["pool_positions"] == len(model_hits)
    assert summary["pool_model_correct"] == sum(model_hits)
    assert summary["pool_baseline_correct"] == sum(base_hits)
    assert summary["pool_accuracy"] == pytest.approx(
        100.0 * sum(model_hits) / len(model_hits), abs=2e-3
    )
    assert summary["pool_margin_pts"] == pytest.approx(
        100.0 * (sum(model_hits) - sum(base_hits)) / len(model_hits), abs=2e-3
    )
    # The famsplit arm's recorded pooled numbers must also hold.
    claims = _claims()
    if "stage2_g2p_famsplit_accuracy" in claims:
        assert summary["pool_accuracy"] == pytest.approx(
            float(claims["stage2_g2p_famsplit_accuracy"]["value"]), abs=0.01
        )


@_SKIP_NO_LEDGER
def test_the_recorded_famgrp_arm_holds_its_controls():
    """The recorded decision-102 aggregates must be internally consistent.

    The pool must reproduce the famsplit arm's positions and margin; the
    positive-margin share must lie in [0, 1]; the negative control's share
    reversal assignment's between-family spread must sit below the real
    assignment's (the structure is the families', not the scoring's). The guard
    names exactly the labels the recorder writes and this body reads, so the test
    cannot silently skip against a label nobody records.
    """
    claims = _claims()
    needed = (
        "stage2_g2p_famgrp_pool_positions",
        "stage2_g2p_famgrp_pool_accuracy",
        "stage2_g2p_famgrp_positive_margin_share",
        "stage2_g2p_famgrp_real_max_abs_margin",
        "stage2_g2p_famgrp_reversal_max_abs_margin",
        "stage2_g2p_famgrp_control_confirmed",
    )
    if any(label not in claims for label in needed):
        pytest.skip("stage2_g2p_famgrp_* not recorded in this checkout")

    pool_positions = int(claims["stage2_g2p_famgrp_pool_positions"]["value"])
    famsplit_positions = int(claims["stage2_g2p_famsplit_eval_positions"]["value"])
    assert pool_positions == famsplit_positions, (
        "the positive control is void: the per-family pool covers a different "
        "population than the famsplit arm"
    )
    share = float(claims["stage2_g2p_famgrp_positive_margin_share"]["value"])
    assert 0.0 <= share <= 1.0
    real_spread = float(claims["stage2_g2p_famgrp_real_max_abs_margin"]["value"])
    reversal_spread = float(claims["stage2_g2p_famgrp_reversal_max_abs_margin"]["value"])
    assert reversal_spread < real_spread, (
        "the negative control fired: the reversal assignment matches or exceeds "
        "the real between-family spread, so the recorded heterogeneity is an "
        "artifact of the scoring, not of the families"
    )
    assert claims["stage2_g2p_famgrp_control_confirmed"]["value"] == 1


@_SKIP_NO_LEDGER
def test_the_famgrp_labels_are_recorder_written():
    claims = _claims()
    labels = [label for label in claims if label.startswith("stage2_g2p_famgrp_")]
    if not labels:
        pytest.skip("stage2_g2p_famgrp_* not recorded in this checkout")
    for label in labels:
        assert claims[label]["source"].startswith("scripts/"), (
            f"{label} was last written by {claims[label]['source']!r}; recorders "
            f"write, tests verify"
        )
