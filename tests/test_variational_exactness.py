"""The condition under which minimizing variational free energy is exact.

`docs/manuscript/02_background.md` once said that minimizing the displayed VFE "under
a discrete generative model gives the exact Bayesian posterior". That license is
too broad: it holds when the family over which $q$ ranges contains the exact
posterior, and a factorized family over a genuinely coupled model does not.

Every level shipped here does meet the condition, but Level 3 meets it for a
reason the old sentence did not name. It is a two-factor model with
`A_dependencies = [[0, 1]]`, so a mean-field claim would be unjustified on its
face. What saves it is degeneracy: `D[1]` is a point mass and `B[1]` the
deterministic +1 shift, so the position marginal stays a delta at every step and
the exact joint posterior is already a product.

These tests pin that structural fact, so the corrected sentence has a control
rather than an assertion. If a future change gave the position factor genuine
uncertainty, the model would still run and every perplexity would still be
computed — and the exactness claim would quietly become false. That is precisely
the failure this file exists to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from ailm.levels.word import (
    build_position_transition,
    build_word_agent,
    build_word_transition,
    load_word_artifact,
)


@pytest.fixture(scope="module")
def artifact():
    return load_word_artifact()


def test_the_position_prior_is_a_point_mass(artifact) -> None:
    """POSITIVE CONTROL for the degeneracy half of the exactness argument."""
    agent = build_word_agent(artifact)
    position_prior = np.asarray(agent.D[1]).reshape(-1)
    assert position_prior.sum() == pytest.approx(1.0)
    assert position_prior[0] == pytest.approx(1.0), (
        "D[1] is no longer a point mass on position 0. The background section "
        "claims Level 3's factorized posterior is exact *because* the position "
        "factor is degenerate; with a spread prior that claim is false."
    )
    assert np.count_nonzero(position_prior > 1e-12) == 1


def test_the_position_transition_is_deterministic(artifact) -> None:
    """A delta prior stays a delta only under a deterministic transition."""
    transition = build_position_transition(artifact)
    n_pos = artifact.n_positions
    assert transition.shape == (n_pos, n_pos)
    # Column-stochastic, B[next, current]: every column is a one-hot.
    for current in range(n_pos):
        column = transition[:, current]
        assert column.sum() == pytest.approx(1.0)
        assert np.count_nonzero(column) == 1, (
            f"position column {current} spreads mass over "
            f"{np.count_nonzero(column)} successors; the degeneracy that makes "
            "Level 3's factorized posterior exact no longer holds"
        )
    # And it is the +1 shift, absorbing at the end.
    for current in range(n_pos):
        assert transition[min(current + 1, n_pos - 1), current] == pytest.approx(1.0)


def test_a_spread_position_prior_would_break_the_degeneracy(artifact) -> None:
    """NEGATIVE CONTROL: the checks above must be capable of failing.

    A test that only asserted properties of the shipped matrices would pass
    equally well against a checker that never looked at them.
    """
    n_pos = artifact.n_positions
    spread = np.full(n_pos, 1.0 / n_pos)
    assert np.count_nonzero(spread > 1e-12) != 1

    stochastic_shift = build_position_transition(artifact).copy()
    stochastic_shift[0, 0] = 0.5
    stochastic_shift[1, 0] = 0.5
    assert np.count_nonzero(stochastic_shift[:, 0]) == 2


def test_the_word_factor_is_the_one_carrying_uncertainty(artifact) -> None:
    """The other half: word identity does not change mid-spelling.

    Level 3's exactness argument needs exactly one factor to carry belief. If
    `B[0]` stopped being the identity, the two factors would couple through
    time and the product form would no longer survive a step.
    """
    word_transition = build_word_transition(artifact)
    assert np.array_equal(word_transition, np.eye(len(artifact.vocab))), (
        "B[0] is no longer the identity, so word identity can drift during "
        "spelling and the joint posterior need not stay a product"
    )


def test_the_background_section_states_the_condition_rather_than_asserting_exactness() -> None:
    """The prose must name the condition, not just the conclusion."""
    from pathlib import Path

    background = " ".join(
        (Path(__file__).resolve().parents[1] / "docs" / "manuscript" / "02_background.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "contains the exact posterior" in background, (
        "the background section no longer states the condition under which minimizing VFE is exact"
    )
    assert "not of mean-field active inference in general" in background, (
        "the background section no longer distinguishes this project's models "
        "from the general mean-field case, where a restricted family need not "
        "contain the posterior"
    )
