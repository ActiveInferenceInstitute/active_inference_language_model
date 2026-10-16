"""Positive controls for every refuse-to-draw guard in ``fig:null_bounds``.

The figure's whole claim is that its numbers cannot drift from the ledger: it
re-derives point estimates from recorded counts, holds convention tokens against
the executable constants in :mod:`ailm.statistics`, and refuses to draw on a
disagreement. A guard nobody has ever seen fail is indistinguishable from a
guard that cannot fail, and it would license exactly the belief the guard exists
to earn. Each test below perturbs one token and asserts the refusal.

Nothing here writes ``data/claims.json`` or runs a recorder. Every case is a
mutated *copy* of the rendered variable map.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Text

from ailm.figures import null_bounds as module
from ailm.figures._common import MissingVariableError, load_manuscript_variables
from ailm.figures.null_bounds import generate_null_bounds, null_bounds_data


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """The live token map, loaded once and never mutated in place."""
    return load_manuscript_variables()


def _with(variables: dict[str, str], token: str, value: str) -> dict[str, str]:
    """A copy of the map with one token replaced."""
    mutated = dict(variables)
    mutated[token] = value
    return mutated


def test_the_live_ledger_draws(variables):
    """NEGATIVE CONTROL — the guards must not reject the correct ledger.

    Without this every test below could be satisfied by a function that raises
    unconditionally, which would pass the whole file and ship no figure.
    """
    data = null_bounds_data(variables)
    assert len(data["rows"]) == len(module._ROWS)
    assert any(row["established"] for row in data["rows"]), (
        "the established contrast row is what exercises the status check's "
        "positive branch; without it the check proves nothing"
    )
    assert any(not row["established"] for row in data["rows"])
    equivalent = [row["label"] for row in data["rows"] if row["equivalent"]]
    # The lexical high-weight arm was the one equivalent row until its backoff
    # lookup was corrected (decision 110); the corrected arm is a decided effect
    # and is drawn as the accuracy panel's established contrast instead.
    assert equivalent == []
    assert [row["label"] for row in data["rows"] if row["established"]] == [
        "Stage 3 lexical channel, mixing weight 0.9 (established)",
        "Stage 2 G2P vs per-letter bar (established)",
    ]


def test_every_row_names_its_unit_and_its_dependence_structure(variables):
    """A bound over an unnamed unit cannot be judged wide or narrow.

    The figure's subject is how much each null still permits, which is a
    statement about a population; and the drawn interval assumes independence,
    so whether a cluster-robust re-analysis exists belongs beside it either way.
    """
    data = null_bounds_data(variables)
    for row in data["rows"]:
        assert row["unit"]
        assert row["n"] > 0
        assert row["discordant"] <= row["n"]
        if row["every_item_discordant"]:
            assert row["discordant"] == row["n"]
        if row["cluster"] is not None:
            assert row["cluster"]["unit"]
            assert 0 < row["cluster"]["count"] <= row["cluster"]["observations"]
            assert 0.0 < row["cluster"]["confidence"] < 1.0
    # At least one row of each kind, so the drawn wording exercises both branches
    # rather than only ever printing one of them.
    assert any(row["cluster"] is not None for row in data["rows"])
    assert any(row["cluster"] is None for row in data["rows"])


def test_a_cluster_count_over_its_observations_refuses_to_draw(variables):
    """Positive control for the clustering guard."""
    tampered = _with(
        variables,
        "CLUSTER_STAGE6_SIGN_CLUSTERS",
        str(int(float(variables["CLUSTER_STAGE6_SIGN_OBSERVATIONS"].replace(",", "")) + 1)),
    )
    with pytest.raises(ValueError, match="not a partition"):
        null_bounds_data(tampered)


def test_the_drawn_figure_carries_the_unit_and_the_clustering(variables):
    figure = generate_null_bounds(variables)
    try:
        drawn = " ".join(text.get_text() for text in figure.findobj(Text))
        assert "discordant" in drawn
        assert "clustered:" in drawn
        assert "no cluster-robust arm recorded" in drawn
    finally:
        plt.close(figure)


#: One mutation per guard: token, replacement, and a fragment of the refusal.
#: The replacements are chosen to trip exactly one guard, so a test that starts
#: failing names the guard that changed rather than the first one in the chain.
_MUTATIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "coverage drift",
        "BOUND_CONFIDENCE_PCT",
        "90",
        "not the declared",
    ),
    (
        "corrected coverage drift",
        "BOUND_CORRECTED_CONFIDENCE_PCT",
        "99",
        "not the declared",
    ),
    (
        "reference share drift",
        "EQUIVALENCE_REFERENCE_SHARE",
        "0.6",
        "wrong region",
    ),
    (
        "equivalence region drift",
        "EQUIVALENCE_REGION_HIGH",
        "0.6",
        "equivalence region",
    ),
    (
        "reachability threshold drift",
        "EQUIVALENCE_MIN_DISCORDANT_PAIRS",
        "300",
        "reachability thresholds",
    ),
    (
        "inverted interval",
        "STAGE6_SIGN_NET_SHARE_LOW",
        "0.9",
        "inverted",
    ),
    (
        "point estimate outside its own interval",
        "LEXICAL_HIGH_ALPHA_MARGIN_PTS",
        "-0.9",
        "falls outside its own interval",
    ),
    (
        # Re-pointed from a context-window token when the census resolved those
        # contrasts and their rows left this figure. A mutation aimed at a token
        # the figure no longer draws stops testing anything and keeps passing,
        # which is the quietest way for a positive control to die.
        "an arm's estimate paired with its neighbor's",
        "STAGE6_SIGN_NET_SHARE",
        # Inside the drawn interval on purpose: a value outside it trips the
        # bounds check first and would test that instead of the recomputation.
        "0.26",
        "recomputed from the recorded counts",
    ),
    (
        "denominator from another population",
        "STAGE5_TEST_SENTENCES",
        "999999",
        "recomputed from the recorded counts",
    ),
    (
        "recorded accuracies contradict the drawn difference",
        "STAGE5_CLASSIFICATION_PCT",
        "88.10",
        "Two recorded routes to one difference disagree",
    ),
    (
        "discordant pairs exceed the paired items",
        "STAGE6_TEST_SENTENCES",
        "100",
        "exceed the",
    ),
    (
        # Re-pointed from the lexical high-weight arm when decision 110 made it
        # a decided effect; the Stage 6 sign test is the null row this guard
        # now has to catch.
        "a null becomes familywise-significant",
        "STAGE6_SIGN_TEST_P",
        "0.0001",
        "wrong heading",
    ),
    (
        "the established share-panel contrast loses significance",
        "STAGE2_G2P_MCNEMAR_P",
        "0.4",
        "wrong heading",
    ),
    (
        "the established accuracy-panel contrast loses significance",
        "LEXICAL_SWEEP_ALPHA_0_9_MCNEMAR_P",
        "0.4",
        "wrong heading",
    ),
    (
        "recorded equivalence verdict contradicts the recomputation",
        "LEXICAL_HIGH_ALPHA_EQUIVALENT",
        "1",
        "LEXICAL_HIGH_ALPHA_EQUIVALENT",
    ),
)


@pytest.mark.parametrize("name,token,value,fragment", _MUTATIONS, ids=[m[0] for m in _MUTATIONS])
def test_a_contradicted_value_refuses_to_draw(variables, name, token, value, fragment):
    """Each guard must reject the defect it was written for."""
    with pytest.raises(ValueError, match=fragment):
        null_bounds_data(_with(variables, token, value))


def test_a_missing_token_refuses_to_draw(variables):
    """A token the figure needs must fail loudly, not default to a plausible number."""
    mutated = dict(variables)
    del mutated["STAGE6_SIGN_NET_SHARE_HIGH"]
    with pytest.raises(MissingVariableError):
        null_bounds_data(mutated)


def test_an_established_row_may_not_also_be_equivalent(variables):
    """A self-consistent planted row reaches the exclusivity refusal.

    The recorded significance decision and the equivalence computation travel
    through independent inputs: the former is the p-value token and the latter
    is the exact interval over the raw directional counts.  Balancing a large
    real row therefore creates the prohibited state without replacing either
    implementation.  Its point and bounds are changed with the counts so every
    earlier integrity check still has to pass before the exclusivity guard is
    reached.
    """
    mutated = dict(variables)
    total = int(mutated["STAT_STAGE2_DISCORDANT_N"].replace(",", ""))
    assert total % 2 == 0
    half = total // 2
    mutated.update(
        {
            "STAGE2_G2P_MCNEMAR_B": str(half),
            "STAGE2_G2P_MCNEMAR_C": str(half),
            "STAGE2_G2P_DISCORDANT_NET_SHARE": "0",
            "STAGE2_G2P_DISCORDANT_NET_SHARE_LOW": "-0.01",
            "STAGE2_G2P_DISCORDANT_NET_SHARE_HIGH": "0.01",
        }
    )
    with pytest.raises(ValueError, match="both established and equivalent"):
        null_bounds_data(mutated)


def test_no_row_is_drawn_from_a_number_typed_into_the_figure(variables):
    """Every drawn quantity must trace to a token, not to figure code.

    The check is indirect but decisive: shifting a token must move the drawn
    value. A literal left in the module would keep drawing the same number while
    the ledger said something else.
    """
    baseline = {row["label"]: row for row in null_bounds_data(variables)["rows"]}
    shifted = null_bounds_data(_with(variables, "STAGE6_SIGN_NET_SHARE_HIGH", "0.4"))
    moved = {row["label"]: row for row in shifted["rows"]}
    label = "Stage 6 document position, sign test"
    assert moved[label]["high_pts"] != baseline[label]["high_pts"], (
        "the Stage 6 upper bound did not follow its token, so the figure is "
        "drawing a number of its own"
    )
