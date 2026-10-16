"""The multiplicity forest figure: ledger-bound data and refuse-to-draw controls.

``ailm.figures.multiplicity`` draws the manuscript's central inferential claim
(all four corrected intervals exclude one half) and the Stage 6 fragility
geometry. Its falsifier is structural: the data function re-derives the
corrected verdict from the recorded raw discordant counts and refuses to draw
on any disagreement with the ledger. A refuse-to-draw check that cannot fire is
worse than none, so every check here carries a positive control that plants a
contradiction and requires the ValueError.

No mocks: the controls copy the real generated variable map and tamper one
token, exactly the drift the checks exist to catch.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from ailm.figures._common import (
    MissingVariableError,
    load_manuscript_variables,
    token_float,
    token_int,
)
from ailm.figures.multiplicity import (
    _token_bool,
    generate_multiplicity_forest,
    multiplicity_forest_data,
)
from ailm.statistics import (
    CONFIRMATORY_POPULATIONS,
    CONFIRMATORY_REPLICATION,
    FAMILYWISE_ALPHA,
    SIGNIFICANCE_FAMILY,
)


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """The real generated token map — the same one the manuscript renders from."""
    return load_manuscript_variables()


# ── The data is the ledger's, not the module's ───────────────────────────────


def test_rows_come_from_the_ledger(variables):
    data = multiplicity_forest_data(variables)
    assert len(data["rows"]) == 4
    for row in data["rows"]:
        assert row["ci_low"] < row["ratio"] < row["ci_high"]
        assert row["n"] > 0
        assert isinstance(row["corrected_excludes"], bool)
    # Spot-bind one row to the token map so a hardcoded copy cannot pass.
    stage2 = data["rows"][0]
    assert stage2["n"] == token_int(variables, "STAT_STAGE2_DISCORDANT_N")
    assert stage2["ratio"] == token_float(variables, "STAT_STAGE2_FAVORED_RATIO")


def test_family_metadata_matches_the_executable_declaration(variables):
    data = multiplicity_forest_data(variables)
    assert data["family_size"] == len(SIGNIFICANCE_FAMILY)
    assert data["familywise_alpha"] == FAMILYWISE_ALPHA
    # The second disclosed fragility direction is derived, not recorded.
    # Derived from the declaration, never from a literal. This assertion hardcoded
    # 9 and broke when the family grew to 13, which is the right failure for the
    # wrong reason: the figure had followed the change and the test had not.
    assert data["collapsed_family_alpha"] == FAMILYWISE_ALPHA / data["collapsed_family_size"]


def test_strip_geometry_is_the_disclosed_one(variables):
    stage6 = multiplicity_forest_data(variables)["stage6"]
    # The drawn geometry must carry the manuscript's verdict: nominally
    # significant, not familywise-confirmed, fragile in both disclosed ways.
    assert stage6["familywise_confirmed"] is False
    assert stage6["one_sided_p"] < stage6["two_sided_p"]


def test_every_row_states_its_population_its_unit_and_its_clustering(variables):
    """A discordant count is not a sample size until it says what it counts."""
    data = multiplicity_forest_data(variables)
    for row in data["rows"]:
        assert row["unit"]
        assert row["population"] >= row["n"] > 0
        assert row["family_label"] in SIGNIFICANCE_FAMILY
        # Either a cluster arm covering the same population, or the honest
        # absence. A row that reports neither would leave the interval's
        # dependence structure unstated, which is the defect this carries.
        if row["cluster"] is not None:
            assert row["cluster"]["unit"]
            assert 0 < row["cluster"]["count"] <= row["population"]
            assert 0.0 < row["cluster"]["confidence"] < 1.0
    # Bind one row's population to the token map so a hardcoded copy cannot pass.
    assert data["rows"][0]["population"] == token_int(variables, "STAGE2_G2P_EVAL_POSITIONS")


def test_the_survivor_set_is_the_recorded_one(variables):
    data = multiplicity_forest_data(variables)
    assert data["bonferroni_survivors"] == sorted(json.loads(variables["STAT_FAMILY_SURVIVORS"]))
    assert set(data["bonferroni_survivors"]) <= set(data["holm_survivors"])
    assert len(data["holm_only"]) == token_int(variables, "STAT_HOLM_ONLY_COUNT")
    # Each drawn badge is the row's membership in the recorded survivor set.
    for row in data["rows"]:
        assert row["corrected_excludes"] == row["bonferroni_survivor"]


def test_the_replication_marks_match_the_frozen_protocol(variables):
    data = multiplicity_forest_data(variables)
    dispositions = dict(CONFIRMATORY_REPLICATION)
    for row in data["rows"]:
        assert row["disposition"] == dispositions[row["family_label"]]
        assert row["disposition_rescorable"] == (row["disposition"] in CONFIRMATORY_POPULATIONS)
        # An unscorable disposition can never be marked as scored.
        assert not (row["disposition_scored"] and not row["disposition_rescorable"])
    assert len(data["scored_members"]) == token_int(variables, "CONFIRMATORY_MEMBERS_SCORED")


def test_generate_returns_a_figure_with_text(variables):
    fig = generate_multiplicity_forest(variables)
    try:
        assert isinstance(fig, Figure)
        # Forest, the statistics column that reads off it, and the Stage 6 strip.
        assert len(fig.axes) == 3
        drawn = " ".join(text.get_text() for text in fig.findobj(Text))
        # The statistics column is the reason this figure grew a third axes; a
        # silently empty one would look like a wide margin.
        assert "discordant pairs of" in drawn
        assert "Holm at familywise" in drawn
        assert "held back:" in drawn
        assert "survive correction:" in drawn
    finally:
        plt.close(fig)


# ── POSITIVE CONTROLS: every refuse-to-draw check must be able to fire ───────


def test_a_flipped_corrected_verdict_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAT_STAGE2_CORRECTED_CI_EXCLUDES_HALF"] = "false"
    with pytest.raises(ValueError, match="[Rr]efusing to draw"):
        multiplicity_forest_data(tampered)


def test_a_tampered_whisker_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAT_LEXICAL_SHARE_CI_LOW"] = "0.42"
    with pytest.raises(ValueError, match="share CI low|straddles"):
        multiplicity_forest_data(tampered)


def test_a_tampered_raw_count_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAGE2_G2P_MCNEMAR_B"] = str(token_int(variables, "STAGE2_G2P_MCNEMAR_B") + 50)
    with pytest.raises(ValueError):
        multiplicity_forest_data(tampered)


def test_a_flipped_familywise_flag_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAT_STAGE6_FAMILYWISE_CONFIRMED"] = "true"
    with pytest.raises(ValueError, match="familywise flag"):
        multiplicity_forest_data(tampered)


def test_a_vanished_sidedness_fragility_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAT_STAGE6_SIGN_TEST_ONE_SIDED_P"] = "0.005"
    with pytest.raises(ValueError, match="sidedness fragility"):
        multiplicity_forest_data(tampered)


def test_a_drifted_family_size_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAT_FAMILY_SIZE"] = str(len(SIGNIFICANCE_FAMILY) + 1)
    with pytest.raises(ValueError, match="SIGNIFICANCE_FAMILY"):
        multiplicity_forest_data(tampered)


def test_a_discordant_count_over_its_population_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAGE2_G2P_EVAL_POSITIONS"] = str(
        token_int(variables, "STAT_STAGE2_DISCORDANT_N") - 1
    )
    with pytest.raises(ValueError, match="describe different populations"):
        multiplicity_forest_data(tampered)


def test_a_cluster_arm_over_a_different_population_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CLUSTER_STAGE2_G2P_OBSERVATIONS"] = str(
        token_int(variables, "CLUSTER_STAGE2_G2P_OBSERVATIONS") + 1
    )
    with pytest.raises(ValueError, match="is not this row's clustering"):
        multiplicity_forest_data(tampered)


def test_a_survivor_set_disagreeing_with_a_badge_refuses_to_draw(variables):
    """Drop a row's member from the Bonferroni survivors, keep its badge true."""
    survivors = json.loads(variables["STAT_FAMILY_SURVIVORS"])
    survivors.remove("stage2_g2p_mcnemar_p")
    tampered = dict(variables)
    tampered["STAT_FAMILY_SURVIVORS"] = json.dumps(survivors)
    tampered["STAT_FAMILY_SURVIVOR_COUNT"] = str(len(survivors))
    tampered["STAT_HOLM_ONLY_COUNT"] = "1"
    with pytest.raises(ValueError, match="duals"):
        multiplicity_forest_data(tampered)


def test_a_bonferroni_only_survivor_refuses_to_draw(variables):
    """Holm can only ever retain more than Bonferroni, never fewer."""
    holm = json.loads(variables["STAT_HOLM_SURVIVORS"])
    holm.remove("stage2_g2p_mcnemar_p")
    tampered = dict(variables)
    tampered["STAT_HOLM_SURVIVORS"] = json.dumps(holm)
    tampered["STAT_HOLM_SURVIVOR_COUNT"] = str(len(holm))
    with pytest.raises(ValueError, match="not a subset"):
        multiplicity_forest_data(tampered)


def test_a_survivor_list_disagreeing_with_its_count_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAT_HOLM_SURVIVOR_COUNT"] = str(token_int(variables, "STAT_HOLM_SURVIVOR_COUNT") + 1)
    with pytest.raises(ValueError, match="came from different"):
        multiplicity_forest_data(tampered)


def test_a_survivor_outside_the_family_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["STAT_HOLM_SURVIVORS"] = json.dumps(["not_a_family_member_p"])
    tampered["STAT_HOLM_SURVIVOR_COUNT"] = "1"
    with pytest.raises(ValueError, match="not members of SIGNIFICANCE_FAMILY"):
        multiplicity_forest_data(tampered)


def test_an_uncounted_replication_refuses_to_draw(variables):
    """A completion flag the recorder never counted must not become a mark."""
    tampered = dict(variables)
    tampered["CONFIRMATORY_MEMBERS_SCORED"] = str(
        token_int(variables, "CONFIRMATORY_MEMBERS_SCORED") + 1
    )
    with pytest.raises(ValueError, match="a replication the ledger does not count"):
        multiplicity_forest_data(tampered)


def test_stage6_counts_disagreeing_with_their_total_refuse_to_draw(variables):
    tampered = dict(variables)
    tampered["STAGE6_SIGN_DISCORDANT"] = str(token_int(variables, "STAGE6_SIGN_DISCORDANT") + 3)
    with pytest.raises(ValueError, match="came from different runs"):
        multiplicity_forest_data(tampered)


# ── The boolean token reader fails loudly ────────────────────────────────────


def test_token_bool_rejects_non_boolean_values(variables):
    tampered = dict(variables)
    tampered["STAT_STAGE6_FAMILYWISE_CONFIRMED"] = "maybe"
    with pytest.raises(MissingVariableError, match="non-boolean"):
        _token_bool(tampered, "STAT_STAGE6_FAMILYWISE_CONFIRMED")


def test_token_bool_rejects_a_missing_token():
    with pytest.raises(MissingVariableError):
        _token_bool({}, "STAT_STAGE6_FAMILYWISE_CONFIRMED")


def _assert_statistics_columns_have_gutter(figure):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    stats = figure.axes[1]
    anchors = sorted({artist.get_position()[0] for artist in stats.texts})
    assert len(anchors) == 2
    left = [artist for artist in stats.texts if artist.get_position()[0] == anchors[0]]
    right = [artist for artist in stats.texts if artist.get_position()[0] == anchors[1]]
    for column in (left, right):
        ordered = sorted(column, key=lambda artist: artist.get_position()[1])
        for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
            assert lower.get_window_extent(renderer).y1 < upper.get_window_extent(renderer).y0, (
                "Statistics rows collide vertically"
            )
    left_edge = max(artist.get_window_extent(renderer).x1 for artist in left)
    right_edge = min(artist.get_window_extent(renderer).x0 for artist in right)
    # A subpixel print gap can blend glyphs without intersecting their boxes.
    # Require one real inter-word space in the actual rendered annotation font.
    minimum_gap = max(
        renderer.get_text_width_height_descent(" ", artist.get_fontproperties(), False)[0]
        for artist in (*left, *right)
    )
    assert right_edge - left_edge >= minimum_gap, (
        f"Statistics columns lack gutter: gap {right_edge - left_edge}, required {minimum_gap}"
    )


def test_population_precision_text_stays_inside_its_statistics_column(variables):
    figure = generate_multiplicity_forest(variables)
    try:
        _assert_statistics_columns_have_gutter(figure)
    finally:
        plt.close(figure)


def test_unwrapped_no_cluster_annotations_trigger_real_gutter_control(variables):
    figure = generate_multiplicity_forest(variables)
    try:
        annotations = [
            artist
            for artist in figure.axes[1].texts
            if "no cluster arm recorded" in artist.get_text()
        ]
        assert len(annotations) == 2
        for artist in annotations:
            wrapped = artist.get_text()
            original = wrapped.replace("conditional test\nover", "conditional\ntest over")
            assert original != wrapped, "Negative control must restore the original line break"
            artist.set_text(original)
        with pytest.raises(AssertionError, match="Statistics columns lack gutter"):
            _assert_statistics_columns_have_gutter(figure)
    finally:
        plt.close(figure)
