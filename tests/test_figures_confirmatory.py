"""The confirmatory replication figure: ledger-bound rows and refuse-to-draw controls.

``ailm.figures.confirmatory`` draws the held-back replication registered by
decision 111 and the Level 6 clustered estimand registered by decision 115.
Decision 116 fixes the rule the module exists to satisfy: every verdict word is
read from a recorded boolean, never from which side of a drawn rule a marker
lands on, and the two must agree or the figure refuses to draw.

A refuse-to-draw check nobody has seen fail is indistinguishable from one that
cannot fail, so every check carries a positive control that plants a
contradiction and requires the ValueError. No mocks: the controls copy the real
generated variable map and tamper one token.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from ailm.figures._common import MissingVariableError, load_manuscript_variables, token_int
from ailm.figures.confirmatory import (
    _MEMBERS,
    _token_bool,
    _token_direction,
    confirmatory_replication_data,
    generate_confirmatory_replication,
)
from ailm.statistics import (
    CONFIRMATORY_REPLICATION,
    FAMILYWISE_ALPHA,
    SIGNIFICANCE_FAMILY,
)


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """The real generated token map — the same one the manuscript renders from."""
    return load_manuscript_variables()


# ── The rows are the ledger's, not the module's ──────────────────────────────


def test_the_drawn_members_are_the_registered_ones(variables):
    """Exactly the family members the protocol maps to the held-back split."""
    data = confirmatory_replication_data(variables)
    registered = {
        member for member, disposition in CONFIRMATORY_REPLICATION if disposition == "ud_ewt_test"
    }
    assert {row["member"].family_label for row in data["rows"]} == registered
    assert len(data["rows"]) == token_int(variables, "CONFIRMATORY_MEMBERS_SCORED")
    for row in data["rows"]:
        assert row["member"].family_label in SIGNIFICANCE_FAMILY


def test_the_frozen_threshold_is_the_familywise_level_over_the_frozen_size(variables):
    data = confirmatory_replication_data(variables)
    assert data["frozen_family_size"] == len(SIGNIFICANCE_FAMILY)
    assert data["frozen_alpha"] == pytest.approx(
        FAMILYWISE_ALPHA / data["frozen_family_size"], rel=1e-2
    )


def test_every_row_states_its_population_its_unit_and_its_counts(variables):
    """A p-value with no sample size beside it overstates what was decided."""
    data = confirmatory_replication_data(variables)
    for row in data["rows"]:
        for arm in (row["held"], row["development"]):
            assert arm["population"] > 0
            assert arm["discordant"] <= arm["population"]
            assert arm["favored"] + arm["opposed"] == arm["discordant"]
            assert arm["documents"] > 0
        assert row["member"].unit
    # Bind one row to the token map so a hardcoded copy cannot pass.
    derived = next(r for r in data["rows"] if r["member"].family_label.startswith("stage4_derived"))
    assert derived["held"]["discordant"] == token_int(
        variables, "CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_DISCORDANT_N"
    )


def test_a_member_with_no_discordant_pair_is_marked_not_computable(variables):
    """Two members produced no discordant pair; the figure must not hide that."""
    data = confirmatory_replication_data(variables)
    incomputable = [r for r in data["rows"] if not r["held"]["computable"]]
    assert incomputable, "the not-computable branch is never exercised on the live ledger"
    for row in incomputable:
        assert row["held"]["discordant"] == 0
        assert row["held"]["direction"] is None
        assert not row["held"]["crosses"]


def test_the_controls_land_on_the_sides_they_were_planted_for(variables):
    data = confirmatory_replication_data(variables)
    positive, negative = data["controls"]
    assert positive["p"] < data["frozen_alpha"]
    assert negative["p"] >= data["frozen_alpha"]


def test_the_level_six_panel_states_both_populations_and_both_coverages(variables):
    """The three things that would make this panel misleading, all present."""
    clustered = confirmatory_replication_data(variables)["clustered"]
    # The populations disagree, and the development interval is not retracted.
    assert clustered["dev_low"] < 0.0 < clustered["dev_high"]
    assert clustered["held_excludes_zero"] is True
    # The corrected coverage is wider and does not exclude zero.
    assert clustered["corrected_low"] <= clustered["held_low"]
    assert clustered["corrected_high"] >= clustered["held_high"]
    assert clustered["corrected_excludes_zero"] is False
    # The arm's own resolution is coarser than the development effect beside it.
    assert clustered["development_effect_resolved"] is False
    assert clustered["min_detectable"] > clustered["dev_effect"]


def test_generate_returns_a_figure_carrying_its_verdicts(variables):
    fig = generate_confirmatory_replication(variables)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3
        drawn = " ".join(text.get_text() for text in fig.findobj(Text))
        assert "not computable: no discordant pair" in drawn
        assert "replicated: crosses the frozen threshold and Holm's" in drawn
        assert "positive control" in drawn
        assert "negative control" in drawn
        assert "minimum detectable effect" in drawn
        assert "documents" in drawn
        assert "type-unigram minus phase-model log loss" in fig.axes[2].get_xlabel()
        labels = " ".join(text.get_text() for text in fig.axes[2].get_yticklabels())
        assert (
            f"primary {float(variables['CONFIRMATORY_STAGE6_CLUSTERED_CONFIDENCE']) * 100:g}%"
            in labels
        )
        assert (
            f"corrected {float(variables['CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CONFIDENCE']) * 100:g}%"
            in labels
        )
        assert "L5 case-only clauses" in drawn
    finally:
        plt.close(fig)


# ── POSITIVE CONTROLS: every refuse-to-draw check must be able to fire ───────


def test_a_flipped_crossing_flag_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE5_MCNEMAR_CROSSES_FROZEN_THRESHOLD"] = "true"
    with pytest.raises(ValueError, match="contradicts the recorded crossing flag"):
        confirmatory_replication_data(tampered)


def test_a_flipped_holm_verdict_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE6_SIGN_TEST_HOLM_REJECTED"] = "false"
    with pytest.raises(ValueError, match="contradicts the recorded Holm"):
        confirmatory_replication_data(tampered)


def test_a_wrong_computability_flag_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE5_MCNEMAR_COMPUTABLE"] = "true"
    with pytest.raises(ValueError, match="computable exactly when a discordant pair exists"):
        confirmatory_replication_data(tampered)


def test_counts_that_do_not_sum_refuse_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_FAVORED"] = str(
        token_int(variables, "CONFIRMATORY_STAGE4_DERIVED_MCNEMAR_FAVORED") + 7
    )
    with pytest.raises(ValueError, match="do not sum to the recorded discordant"):
        confirmatory_replication_data(tampered)


def test_a_discordant_count_over_its_population_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE4_DERIVED_POSITIONS"] = "10"
    with pytest.raises(ValueError, match="is impossible"):
        confirmatory_replication_data(tampered)


def test_a_drifted_scored_count_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_MEMBERS_SCORED"] = str(len(_MEMBERS) + 1)
    with pytest.raises(ValueError, match="disagree about what was replicated"):
        confirmatory_replication_data(tampered)


def test_a_dead_positive_control_refuses_to_draw(variables):
    """If the apparatus cannot find a planted effect, nothing else is trustworthy."""
    tampered = dict(variables)
    tampered["CONFIRMATORY_CONTROL_POSITIVE_P"] = "0.4"
    with pytest.raises(ValueError, match="cannot detect an effect"):
        confirmatory_replication_data(tampered)


def test_a_firing_negative_control_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_CONTROL_NEGATIVE_P"] = "1e-9"
    with pytest.raises(ValueError, match="reports an effect where none was planted"):
        confirmatory_replication_data(tampered)


def test_a_drifted_frozen_family_size_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_FROZEN_FAMILY_SIZE"] = str(len(SIGNIFICANCE_FAMILY) + 1)
    with pytest.raises(ValueError, match="SIGNIFICANCE_FAMILY declares"):
        confirmatory_replication_data(tampered)


def test_an_incomplete_run_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_RUN_COMPLETED_UD_EWT_TEST"] = "false"
    with pytest.raises(ValueError, match="a replication that did not happen"):
        confirmatory_replication_data(tampered)


def test_a_flipped_exclusion_boolean_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_EXCLUDES_ZERO"] = "true"
    with pytest.raises(ValueError, match="an exclusion the interval does not show"):
        confirmatory_replication_data(tampered)


def test_a_corrected_interval_narrower_than_the_primary_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CI_LOW"] = "0.004"
    tampered["CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_EXCLUDES_ZERO"] = "true"
    with pytest.raises(ValueError, match="wider coverage makes impossible"):
        confirmatory_replication_data(tampered)


def test_a_drifted_resolution_flag_refuses_to_draw(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE6_CLUSTERED_DEVELOPMENT_EFFECT_RESOLVED"] = "true"
    with pytest.raises(ValueError, match="but the ledger records"):
        confirmatory_replication_data(tampered)


@pytest.mark.parametrize(
    ("bound", "value"),
    [("LOW", "0.001"), ("HIGH", "-0.001")],
)
def test_a_development_interval_that_stopped_spanning_zero_refuses_to_draw(variables, bound, value):
    """The panel's subject is that the two populations disagree."""
    tampered = dict(variables)
    tampered[f"CLUSTER_STAGE6_LOGLOSS_CI_{bound}"] = value
    with pytest.raises(ValueError, match="no longer disagree"):
        confirmatory_replication_data(tampered)


# ── The token readers fail loudly ────────────────────────────────────────────


def test_token_bool_rejects_a_non_boolean(variables):
    tampered = dict(variables)
    tampered["CONFIRMATORY_STAGE5_MCNEMAR_COMPUTABLE"] = "maybe"
    with pytest.raises(MissingVariableError, match="non-boolean"):
        _token_bool(tampered, "CONFIRMATORY_STAGE5_MCNEMAR_COMPUTABLE")


def test_token_direction_keeps_undefined_distinct_from_false(variables):
    """An arm with no discordant pair has no direction, and must not read False."""
    assert _token_direction({"T": "undefined"}, "T") is None
    assert _token_direction({"T": "true"}, "T") == "true"
    assert _token_direction({"T": "false"}, "T") == "false"
    with pytest.raises(MissingVariableError):
        _token_direction({"T": "maybe"}, "T")
