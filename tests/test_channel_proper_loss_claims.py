"""The proper-loss channel arm: recorded values must match a fresh measurement.

``scripts/record_channel_predictive_loss_claims.py`` is the single writer for
every ``channel_proper_loss_*`` label. This file verifies; it never records.

What makes this arm worth guarding beyond "the recorder ran": it is the only
place in the paper where a channel is scored on the axis sec:hierarchy's
hypothesis box actually declares, and the number it reports is a **negative**
result. A negative result is exactly the kind that decays quietly — nobody
notices when a harm stops reproducing, because a passing suite looks the same
either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.probe_channel_predictive_loss import (  # noqa: E402
    compute_l1_l2b_predictive_loss,
)

LEDGER = PROJECT_ROOT / "data" / "claims.json"

pytestmark = pytest.mark.skipif(
    not LEDGER.exists(), reason="data/claims.json absent; run the recorders"
)


@pytest.fixture(scope="module")
def claims() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


@pytest.fixture(scope="module")
def measured() -> dict:
    """A fresh full-split measurement, independent of what is on disk."""
    return compute_l1_l2b_predictive_loss(sample_limit=100_000)


def _value(claims: dict, label: str) -> float:
    assert label in claims, f"{label} is not recorded; run the recorder"
    return float(claims[label]["value"])


def test_every_proper_loss_label_is_written_by_the_one_recorder(claims: dict):
    """Single-writer discipline, checked on the stored source field.

    Written against the label PREFIX rather than a list of the twenty-two
    labels: a twenty-third added later must be caught by the same check, and a
    hand-maintained list would not catch it.
    """
    owned = {k: v for k, v in claims.items() if k.startswith("channel_proper_loss_")}
    assert owned, "no channel_proper_loss_* labels recorded at all"
    sources = {entry["source"] for entry in owned.values()}
    assert sources == {"scripts/record_channel_predictive_loss_claims.py"}, (
        f"channel_proper_loss_* written by more than one producer: {sorted(sources)}"
    )


def test_recorded_values_reproduce_a_fresh_measurement(claims: dict, measured: dict):
    """The stored numbers must still be what the code produces.

    Six decimals, so this is a real equality and not a loose band: the
    computation is deterministic (no sampling anywhere in the loss path), so
    any drift is a change in the model or the split, not noise.
    """
    assert _value(claims, "channel_proper_loss_positions") == measured["total_positions_evaluated"]
    assert _value(claims, "channel_proper_loss_words") == measured["words_evaluated"]
    assert _value(claims, "channel_proper_loss_anchor_nats") == pytest.approx(
        measured["anchor_loss"], abs=5e-7
    )
    assert _value(claims, "channel_proper_loss_best_realized_nats") == pytest.approx(
        measured["readable"], abs=5e-7
    )
    assert _value(claims, "channel_proper_loss_sensitivity_nats") == pytest.approx(
        measured["sensitivity"], abs=5e-7
    )


def test_the_recorded_result_is_still_a_harm(claims: dict):
    """The negative result must not decay into a silent positive.

    NEGATIVE CONTROL for the arm's own framing: if the best engaged weight ever
    became an improvement, the manuscript paragraph built on "no engaged weight
    improves on it" would be false while every structural test still passed.
    """
    best = _value(claims, "channel_proper_loss_best_realized_nats")
    worst = _value(claims, "channel_proper_loss_worst_realized_nats")
    assert best < 0.0, f"the channel now helps ({best:+.6f} nats); the prose says it cannot"
    assert worst <= best, "worst realized gain is better than the best; the ordering is wrong"


def test_the_sensitivity_term_is_exactly_zero(claims: dict):
    """The redundancy identity that bounds this channel at zero.

    The manuscript paragraph rests on it: the receiver already observes the
    letter, so a perfect message about the letter is worth exactly nothing, and
    the realized effect is bounded above by that. If the oracle spread ever
    became non-zero, the bound argument would be unsupported.
    """
    assert _value(claims, "channel_proper_loss_sensitivity_nats") == 0.0
    assert _value(claims, "channel_proper_loss_oracle_spread") == 0.0


def test_the_two_axes_are_measured_on_the_same_positions(claims: dict):
    """The comparison is void if the accuracy axis came from other positions."""
    assert _value(claims, "channel_proper_loss_positions") == _value(
        claims, "channel_factorial_positions"
    )
    assert _value(claims, "channel_proper_loss_anchor_accuracy") == pytest.approx(
        _value(claims, "channel_factorial_anchor_accuracy"), abs=5e-4
    )


def test_both_axes_still_agree_in_sign(claims: dict):
    """The research-queue falsifier: a sign disagreement would break the diagnostic.

    Recorded as an agreement. If accuracy ever turned positive here while the
    loss stayed negative, sec:hierarchy's monotone-decline diagnostic would be a
    property of the argmax statistic and the section would need restating — so
    this is the assertion that would tell us.
    """
    assert _value(claims, "channel_proper_loss_accuracy_gain_pts") < 0.0
    assert _value(claims, "channel_proper_loss_best_realized_nats") < 0.0


def test_the_loss_declines_monotonically_in_the_weight(claims: dict):
    """The shape class the manuscript reports, read straight off the ledger."""
    weights = ["0_0", "0_05", "0_1", "0_25", "0_5"]
    losses = [_value(claims, f"channel_proper_loss_alpha_{w}_nats") for w in weights]
    assert losses == sorted(losses), f"loss is not monotone in the weight: {losses}"

    # NEGATIVE CONTROL: a constant sweep would also be "sorted". It must vary.
    assert len(set(losses)) == len(losses), "the sweep is constant; nothing is measured"


def test_the_scrambled_control_matches_at_the_anchor_and_differs_after(claims: dict):
    """At weight zero no message is mixed, so scrambling it cannot be visible."""
    assert _value(claims, "channel_proper_loss_alpha_0_0_scrambled_nats") == _value(
        claims, "channel_proper_loss_alpha_0_0_nats"
    )
    for weight in ["0_05", "0_1", "0_25", "0_5"]:
        real = _value(claims, f"channel_proper_loss_alpha_{weight}_nats")
        control = _value(claims, f"channel_proper_loss_alpha_{weight}_scrambled_nats")
        assert real != control, f"the scramble is invisible at weight {weight}"


def test_the_manuscript_paragraph_cites_this_arm():
    """The recorded arm must actually reach the paper.

    A recorded measurement no prose cites is a number nobody reads; the reverse
    direction (prose citing an unbacked token) is covered by the claims
    discipline suite.
    """
    section = "\n".join(
        (PROJECT_ROOT / "docs" / "manuscript" / name).read_text(encoding="utf-8")
        for name in ("03_methods.md", "04_results.md", "98_supplement_channel_factorial.md")
    )
    for token in (
        "{{CHANNEL_PROPER_LOSS_ANCHOR_NATS}}",
        "{{CHANNEL_PROPER_LOSS_BEST_REALIZED_NATS}}",
        "{{CHANNEL_PROPER_LOSS_SENSITIVITY_NATS}}",
        "{{CHANNEL_PROPER_LOSS_POSITIONS}}",
        # The extension: every remaining factorial cell and both non-diluting
        # lexical arms, loss primary and accuracy secondary.
        "{{CHANNEL_PROPER_LOSS_LETTER_PREDICTIVE_NATS}}",
        "{{CHANNEL_PROPER_LOSS_LEXICON_EVIDENCE_NATS}}",
        "{{CHANNEL_PROPER_LOSS_LEXICON_PREDICTIVE_NATS}}",
        "{{CHANNEL_PROPER_LOSS_GEOMETRIC_LEXICON_NATS}}",
        "{{CHANNEL_PROPER_LOSS_FITTED_NATS}}",
    ):
        assert token in section, f"{token} is recorded but never cited"

    # The scope limits must survive editing: the extension is a point
    # re-scoring at recorded weights, it attaches no inferential apparatus,
    # and the axis still does not reach the generation or narrative channels.
    assert "tbl:channel_proper_loss" in section
    assert "re-scoring, not a sweep" in section
    assert "descriptive re-scorings" in section


def test_the_extension_rows_are_arithmetically_consistent(claims: dict):
    """Each cell's recorded gain must be the anchor minus its recorded loss.

    Both sides are recorded to six decimals from unrounded intermediates, so the
    recomputation can disagree with the stored difference by at most one unit in
    the last place of each operand.
    """
    anchor = _value(claims, "channel_proper_loss_anchor_nats")
    for cell in (
        "letter_evidence",
        "letter_predictive",
        "lexicon_evidence",
        "lexicon_predictive",
        "geometric_lexicon",
        "fitted",
    ):
        loss = _value(claims, f"channel_proper_loss_{cell}_nats")
        realized = _value(claims, f"channel_proper_loss_{cell}_realized_nats")
        assert realized == pytest.approx(anchor - loss, abs=2.5e-6), cell


def test_the_extension_rows_carry_their_selection_markers(claims: dict):
    """Weight provenance is recorded per cell, and the statuses match it.

    The four sweep cells' weights were chosen on the same held-out split their
    loss is reported on, so their loss claims must be marked exploratory; the
    fitted arm's weight came from an inner train split and its marker must say
    so. A marker that drifted would let a test-selected number read as a clean
    one.
    """
    for cell in (
        "letter_evidence",
        "letter_predictive",
        "lexicon_evidence",
        "lexicon_predictive",
        "geometric_lexicon",
    ):
        assert claims[f"channel_proper_loss_{cell}_weight_test_selected"]["value"] is True, cell
        for kind in ("nats", "realized_nats"):
            label = f"channel_proper_loss_{cell}_{kind}"
            assert claims[label].get("status") == "exploratory", label
    assert claims["channel_proper_loss_fitted_weight_test_selected"]["value"] is False


#: Level at which a cell's accuracy margin counts as having a sign at all.
#: The repository familywise alpha; see docs/manuscript/98_supplement_statistics.md.
SIGN_RESOLVED_ALPHA = 0.01

#: Paired p-value label carrying each cell's accuracy-axis uncertainty.
ACCURACY_PAIRED_P = {
    "letter_evidence": "channel_letter_evidence_paired_p",
    "letter_predictive": "channel_letter_predictive_paired_p",
    "lexicon_evidence": "channel_lexicon_evidence_paired_p",
    "lexicon_predictive": "channel_lexicon_predictive_paired_p",
    "geometric_lexicon": "geometric_lexicon_paired_p",
    "fitted": "fitted_paired_p",
}


def test_the_extension_signs_agree_with_the_accuracy_cells(claims: dict):
    """The cross-axis agreement the section reports, read off the ledger.

    NEGATIVE-CONTROL framing: if a cell's loss sign ever flipped against its
    recorded accuracy margin, the research-queue falsifier fires and the
    manuscript's "every sign agrees" paragraph is false while every structural
    test still passes — this is the assertion that would tell us.

    Scope, and it is load-bearing: the comparison runs only on cells whose
    accuracy margin is **resolved** by its own paired test at
    ``SIGN_RESOLVED_ALPHA``. An unresolved margin has no sign to agree with, and
    asserting against one reads a direction off noise. ``lexicon_evidence`` is
    the live case: b and c differ by a single position at p = 1.0, so its
    +0.002-point margin is a coin flip while its loss is a small negative. That
    is not a cross-axis contradiction, and reporting it as one would falsify a
    paragraph on the strength of one discordant pair.

    NON-VACUITY: a majority of the cells must be resolved, or this test would
    pass by having nothing to check.
    """
    margins = {
        "letter_evidence": _value(claims, "channel_letter_evidence_best_accuracy")
        - _value(claims, "channel_factorial_anchor_accuracy"),
        "letter_predictive": _value(claims, "channel_letter_predictive_best_accuracy")
        - _value(claims, "channel_factorial_anchor_accuracy"),
        "lexicon_evidence": _value(claims, "channel_lexicon_evidence_best_accuracy")
        - _value(claims, "channel_factorial_anchor_accuracy"),
        "lexicon_predictive": _value(claims, "channel_lexicon_predictive_best_accuracy")
        - _value(claims, "channel_factorial_anchor_accuracy"),
        "geometric_lexicon": _value(claims, "geometric_lexicon_best_margin_pts"),
        "fitted": _value(claims, "fitted_margin_over_anchor_pts"),
    }
    assert set(margins) == set(ACCURACY_PAIRED_P), "the two cell rosters disagree"

    resolved, unresolved = [], []
    for cell, accuracy_margin in margins.items():
        p_value = _value(claims, ACCURACY_PAIRED_P[cell])
        if p_value >= SIGN_RESOLVED_ALPHA:
            unresolved.append((cell, accuracy_margin, p_value))
            continue
        resolved.append(cell)
        realized = _value(claims, f"channel_proper_loss_{cell}_realized_nats")
        assert (realized > 0) == (accuracy_margin > 0), (
            f"{cell}: loss realized {realized:+.6f} nats against accuracy margin "
            f"{accuracy_margin:+.4f} pts at p = {p_value:.3g} — the sign "
            f"disagreement the manuscript says does not occur"
        )

    assert len(resolved) * 2 > len(margins), (
        f"only {len(resolved)} of {len(margins)} cells have a resolved accuracy "
        f"sign, so this comparison checks almost nothing: {unresolved}"
    )
