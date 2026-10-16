"""Evidence-status controls for the Stage 5 and Stage 6 summary figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Text

from ailm.figures.levels56 import (
    clause_null_data,
    generate_clause_null,
    generate_narrative_channel,
    narrative_channel_data,
)
from ailm.manuscript_variables import generate_manuscript_variables

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    # Exercise the current ledger-to-token path directly. The tracked output
    # map is a generated publication artifact and may legitimately be stale
    # until the repository-wide renderer runs after concurrent source edits.
    return generate_manuscript_variables(PROJECT_ROOT)


def _with(variables: dict[str, str], token: str, value: str) -> dict[str, str]:
    changed = dict(variables)
    changed[token] = value
    return changed


def _figure_text(figure) -> str:
    return "\n".join(text.get_text() for text in figure.findobj(match=Text))


def test_stage5_figure_names_a_reference_not_a_ceiling_or_accuracy_oracle(variables) -> None:
    data = clause_null_data(variables)
    assert data["type_free"] <= data["privileged_reference"] <= data["mixture"]
    assert data["privileged_label_gain_pct"] < 0
    assert data["gain_documents"] > 1
    assert data["gain_sentence_ci_low"] < 0 < data["gain_sentence_ci_high"]
    assert data["gain_position_ci_low"] < 0 < data["gain_position_ci_high"]
    figure = generate_clause_null(variables)
    try:
        text = _figure_text(figure).casefold()
        assert "privileged" in text
        assert "gain unresolved" in text
        assert "whole-document bootstrap" in text
        assert "ceiling" not in text
        assert "oracle" not in text
    finally:
        plt.close(figure)


def test_stage5_figure_draws_the_paired_interval_from_the_tokens(variables) -> None:
    """The interval is a mark on the canvas, not only title text.

    The inset's error bar must reach exactly the recorded interval bounds and
    the zero line must be drawn, so a reader sees the interval span zero rather
    than being told so.
    """
    data = clause_null_data(variables)
    figure = generate_clause_null(variables)
    try:
        # An inset is a child axes, not a member of figure.axes.
        insets = [child for ax in figure.axes for child in ax.child_axes if child.containers]
        assert len(insets) == 1, "expected exactly one inset carrying the interval"
        inset = insets[0]
        (container,) = inset.containers
        (_lower_caps, _upper_caps), (bars,) = container.lines[1], container.lines[2]
        segment = bars.get_segments()[0]
        drawn_low, drawn_high = sorted(float(point[1]) for point in segment)
        assert drawn_low == pytest.approx(data["gain_position_ci_low"], abs=1e-12)
        assert drawn_high == pytest.approx(data["gain_position_ci_high"], abs=1e-12)
        assert drawn_low < 0.0 < drawn_high
        zero_lines = [line for line in inset.get_lines() if list(line.get_ydata()) == [0.0, 0.0]]
        assert zero_lines, "the inset draws no zero line"
    finally:
        plt.close(figure)


def test_stage5_figure_refuses_a_planted_resolved_gain(variables) -> None:
    """POSITIVE CONTROL: the interval-status guard must be able to fail."""
    with pytest.raises(ValueError, match="sentence-level.*no longer unresolved"):
        clause_null_data(_with(variables, "STAGE5_ORACLE_GAIN_SENTENCE_CI_LOW", "0.001"))
    with pytest.raises(ValueError, match="position-weighted.*no longer unresolved"):
        clause_null_data(_with(variables, "STAGE5_ORACLE_GAIN_POSITION_CI_HIGH", "-0.001"))


def test_stage5_figure_refuses_the_retired_bracketed_ordering(variables) -> None:
    """POSITIVE CONTROL: a reference below the bar changes the gate taxonomy."""
    with pytest.raises(ValueError, match="now clears the bar"):
        clause_null_data(_with(variables, "STAGE5_TYPE_ORACLE_PERPLEXITY", "4.0"))


def test_stage6_figure_exposes_nonconfirmatory_primary_and_distinct_live_score(variables) -> None:
    data = narrative_channel_data(variables)
    assert data["cluster_ci_low"] < 0 < data["cluster_ci_high"]
    assert data["cluster_p"] > 0.05
    assert data["live_improvement"] > 0
    figure = generate_narrative_channel(variables)
    try:
        text = _figure_text(figure)
        assert "non-confirmatory" in text
        assert "reversed-position control" in text
        assert "live Agent" in text
        assert "L6 -> L5 D" in text
    finally:
        plt.close(figure)


def test_stage6_cluster_verdict_guard_rejects_a_planted_positive_interval(variables) -> None:
    with pytest.raises(ValueError, match="no longer non-confirmatory"):
        narrative_channel_data(_with(variables, "CLUSTER_STAGE6_LOGLOSS_CI_LOW", "0.001"))


def test_stage6_live_score_guard_rejects_a_planted_inconsistent_delta(variables) -> None:
    with pytest.raises(ValueError, match="live-Agent score"):
        narrative_channel_data(_with(variables, "STAGE6_LIVE_FILTER_IMPROVEMENT_NATS", "0.5"))


def test_stage6_correct_ledger_does_not_trigger_refusal(variables) -> None:
    """Negative control: guards that always raise cannot satisfy the figure gate."""
    assert narrative_channel_data(variables)["cluster_documents"] > 1
