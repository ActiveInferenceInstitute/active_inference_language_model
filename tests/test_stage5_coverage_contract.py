"""Stage 5 interval-coverage contract: computation and metadata must agree.

Methods finding S2: the Stage 5 classification document-CI was serialized with
no coverage metadata of its own while the manuscript qualified it with the
oracle-gain arm's coverage token; the two arms shared
``STAGE5_BOOTSTRAP_CONFIDENCE`` only because both happen to read one constant.
If either arm's level moved, the prose stated a coverage the interval did not
have and nothing failed.

The closed contract, per arm (``scripts/record_stage5_claims.py`` and
``scripts/record_stage5_widened_claims.py``):

* the document-CI comes from ``ailm.clustering.summarize_clustered_difference``
  called with the arm's declared level, and the returned summary carries
  ``confidence``/``resamples``/``seed`` from the actual computation — the
  metadata has one source, not a parallel constant;
* the recorders serialize those summary fields under
  ``*_classification_document_ci_confidence_pct`` / ``*_bootstrap_resamples`` /
  ``*_bootstrap_seed``, beside the interval they describe;
* both arms deliberately declare the same level (the clause.py constant), and
  the shared declaration is pinned here — so a future divergence shows up as
  visible per-arm ledger fields, never as a silent prose mismatch.

``ClusteredPairedSummary`` already carried the three fields before this
contract: the helper needed no change, only the serialization and its
enforcement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ailm.clustering import ClusteredPairedSummary, summarize_clustered_difference
from ailm.levels.clause import (
    STAGE5_BOOTSTRAP_CONFIDENCE,
    STAGE5_BOOTSTRAP_RESAMPLES,
    STAGE5_BOOTSTRAP_SEED,
)
from ailm.value_formatting import format_ledger_value

#: Deterministic paired differences and their document clusters. Three
#: clusters with real spread, so the bootstrap interval is non-degenerate and
#: strictly narrows as the declared coverage drops.
DIFFERENCES = [0.4, -0.1, 0.2, 0.5, -0.3, 0.6, 0.1, 0.3]
CLUSTERS = [0, 0, 0, 1, 1, 2, 2, 2]

#: A deliberately different coverage for the coupling control. At this level
#: the same data and draws must produce a strictly narrower interval — proof
#: that the bounds move with the level rather than reporting a frozen pair.
COVERAGE_CONTRAST = 0.50

_RECORDERS = (
    Path(__file__).resolve().parents[1] / "scripts" / "record_stage5_claims.py",
    Path(__file__).resolve().parents[1] / "scripts" / "record_stage5_widened_claims.py",
)


def _arm_summary(confidence: float) -> ClusteredPairedSummary:
    """The document-CI summary exactly as each Stage 5 recorder computes it."""
    return summarize_clustered_difference(
        DIFFERENCES,
        CLUSTERS,
        confidence=confidence,
        resamples=STAGE5_BOOTSTRAP_RESAMPLES,
        seed=STAGE5_BOOTSTRAP_SEED,
    )


def test_the_document_interval_moves_with_a_deliberately_different_level():
    """The computed bounds track the declared coverage; metadata follows.

    POSITIVE: at the arm's level the summary reports that level, with the
    resampling constants it was computed from. NEGATIVE CONTROL: at a
    deliberately different level the same data and draws produce a strictly
    narrower interval, so a metadata path that kept claiming the arm's
    percentage while the computation moved cannot pass this.
    """
    shipped = _arm_summary(STAGE5_BOOTSTRAP_CONFIDENCE)
    contrast = _arm_summary(COVERAGE_CONTRAST)

    assert shipped.confidence == STAGE5_BOOTSTRAP_CONFIDENCE
    assert contrast.confidence == COVERAGE_CONTRAST
    assert shipped.resamples == STAGE5_BOOTSTRAP_RESAMPLES
    assert shipped.seed == STAGE5_BOOTSTRAP_SEED
    assert contrast.resamples == STAGE5_BOOTSTRAP_RESAMPLES
    assert contrast.seed == STAGE5_BOOTSTRAP_SEED

    assert contrast.ci_low > shipped.ci_low
    assert contrast.ci_high < shipped.ci_high


def test_the_serialized_metadata_replays_the_recorded_interval_exactly():
    """Re-running at any declared level reproduces the interval bit for bit.

    The recorder serializes ``100.0 * summary.confidence`` beside bounds from
    the same object, so the serialization is a pure function of the declared
    level, resamples and seed. A ledger refresh therefore cannot silently
    disagree with its own recorded coverage — at the shipped integral level or
    at a non-integer one.
    """
    for level in (STAGE5_BOOTSTRAP_CONFIDENCE, 0.975, 0.995):
        first = _arm_summary(level)
        second = _arm_summary(level)

        assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
        assert (first.confidence, first.resamples, first.seed) == (
            second.confidence,
            second.resamples,
            second.seed,
        )

    shipped = _arm_summary(STAGE5_BOOTSTRAP_CONFIDENCE)
    assert 100.0 * shipped.confidence == 99.0


def test_the_serialized_percent_carries_full_information():
    """``*_ci_confidence_pct`` is the exact percent, not a truncated int.

    X1a: the recorders serialized ``int(100.0 * confidence)``, which wrote 97
    for a 0.975 declared level (true 97.5) and 99 for 0.995. The percent is
    now the float product itself, at all four percent sites of both recorders:
    integral levels render unchanged, non-integer levels keep their fraction,
    and no arbitrary rounding replaces the old truncation.
    """
    for level, low, high in ((0.975, 97.0, 98.0), (0.995, 99.0, 100.0)):
        summary = _arm_summary(level)
        serialized = 100.0 * summary.confidence  # the recorders' expression
        assert serialized == 100.0 * level, "full numeric percent information"
        assert low < serialized < high, "a truncated int would fail this band"
        assert format_ledger_value(serialized) == repr(serialized)

    shipped = _arm_summary(STAGE5_BOOTSTRAP_CONFIDENCE)
    assert 100.0 * shipped.confidence == 99.0
    # The shipped integral level renders exactly as the frozen prose has it.
    assert format_ledger_value(100.0 * shipped.confidence) == "99"


@pytest.mark.parametrize("bad_level", [0.0, 1.5, float("inf"), -0.99])
def test_unsupported_confidence_levels_are_refused_on_the_recorder_path(bad_level):
    """The refusal surface the two recorders rely on stays closed.

    ``tests/test_clustering.py`` pins non-numeric and NaN inputs; these are the
    remaining out-of-range levels. An interval outside (0, 1) coverage must
    refuse, never silently compute a degenerate percentile bound.
    """
    with pytest.raises(ValueError, match="confidence"):
        summarize_clustered_difference(
            DIFFERENCES,
            CLUSTERS,
            confidence=bad_level,
            resamples=STAGE5_BOOTSTRAP_RESAMPLES,
            seed=STAGE5_BOOTSTRAP_SEED,
        )


@pytest.mark.parametrize("recorder", _RECORDERS, ids=lambda path: path.name)
def test_each_arm_declares_one_shared_level_and_serializes_its_own_summary(recorder):
    """Both arms compute at the shared constant and serialize their own fields.

    Structural pin of the wiring the behavioral tests above cannot see without
    running the recorders (a full-model run, deferred to a later grant): the
    document-CI call passes ``confidence=STAGE5_BOOTSTRAP_CONFIDENCE``
    explicitly — not the helper's 0.95 ``DIFFERENCE_CONFIDENCE`` default — and
    the ``*_classification_document_*`` metadata labels are derived from
    ``classification_clustered.confidence``/``.resamples``/``.seed``, the same
    object the serialized interval bounds come from.
    """
    source = recorder.read_text(encoding="utf-8")

    assert "STAGE5_BOOTSTRAP_CONFIDENCE" in source
    assert re.search(
        r"summarize_clustered_difference\(.*confidence=STAGE5_BOOTSTRAP_CONFIDENCE",
        source,
        re.S,
    ), "the document-CI is not computed at the arm's declared level"
    arm = "stage5w" if "widened" in recorder.name else "stage5"
    assert re.search(
        rf'"{arm}_classification_document_ci_confidence_pct": \(\s*'
        rf"100\.0 \* classification_clustered\.confidence\s*\),",
        source,
    ), "the serialized coverage does not come from the computed summary"
    # X1a: no ``int(100.0 * ...)`` truncation may remain at any percent site —
    # the four *_ci_confidence_pct labels share the one float-percent convention.
    assert re.search(r"int\(100\.0 ?\*", source) is None, (
        "a percent site still truncates: int(100.0 * confidence) wrote 97 for a "
        "0.975 level instead of 97.5"
    )
    assert (
        f'"{arm}_classification_document_bootstrap_resamples": classification_clustered.resamples'
    ) in source
    assert (
        f'"{arm}_classification_document_bootstrap_seed": classification_clustered.seed'
    ) in source


def test_the_recorded_metadata_binds_to_the_computed_summary_through_a_real_ledger(
    tmp_path,
):
    """An isolated writer run records exactly what the computation produced.

    Drives the real ``ailm.claims`` serialization machinery on a temporary
    ledger — the only write this suite performs — with the values dict built
    from the computed summaries the way the recorders build theirs, then
    reloaded and compared. The float percent round-trips JSON exactly, and an
    int-truncated coverage (the X1a defect: 97 for a 0.975 level) can neither
    equal the computed summary nor carry its type.
    """
    from ailm.claims import ClaimBatch, ClaimsLedger

    shipped = _arm_summary(STAGE5_BOOTSTRAP_CONFIDENCE)
    fractional = _arm_summary(0.975)
    values = {
        "stage5_classification_document_ci_low": round(shipped.ci_low, 4),
        "stage5_classification_document_ci_high": round(shipped.ci_high, 4),
        "stage5_classification_document_ci_confidence_pct": (100.0 * shipped.confidence),
        "stage5_classification_document_bootstrap_resamples": shipped.resamples,
        "stage5_classification_document_bootstrap_seed": shipped.seed,
        "stage5w_classification_document_ci_confidence_pct": (100.0 * fractional.confidence),
    }
    ledger_path = tmp_path / "claims.json"
    ledger = ClaimsLedger(ledger_path)
    ledger.replace_claims(
        remove=[],
        batches=[
            ClaimBatch(
                values=values,
                run_id="stage5_coverage_contract",
                source="tests/test_stage5_coverage_contract.py",
                status="descriptive",
            )
        ],
    )

    reloaded = ClaimsLedger(ledger_path)
    recorded_pct = reloaded.get("stage5_classification_document_ci_confidence_pct")["value"]
    widened_pct = reloaded.get("stage5w_classification_document_ci_confidence_pct")["value"]
    assert recorded_pct == 100.0 * shipped.confidence == 99.0
    assert widened_pct == 100.0 * fractional.confidence == 97.5
    assert isinstance(widened_pct, float) and not widened_pct.is_integer(), (
        "an int-truncated serialization (97) would fail this check"
    )
    assert (
        reloaded.get("stage5_classification_document_bootstrap_resamples")["value"]
        == STAGE5_BOOTSTRAP_RESAMPLES
    )
    assert (
        reloaded.get("stage5_classification_document_bootstrap_seed")["value"]
        == STAGE5_BOOTSTRAP_SEED
    )
