"""W2: pymdp parameter learning against the shipped counting path.

Decision 22 pre-registered the prediction — committed before any update ran —
and decision 23 records the outcome. This file re-verifies both on bounded
real streams, carries the controls that make the agreement falsifiable, and
binds the recorded ``learning_*`` labels to fresh executions.

The load-bearing structure:

* a **toy anchor** whose Dirichlet increments are hand-checkable integers, so
  the conjugate mechanism is verified at the level of exact counts, not just
  a small norm;
* **P1** re-run on a real stream slice, against the pre-registered tolerance;
* **P2** in both directions — a mismatched prior must break the agreement,
  and matching the comparator must restore it;
* harness controls: a comparator counted on a *different* stream must not
  agree, and a learning rate other than 1 must break the conjugate identity —
  each proving a knob is load-bearing rather than decorative.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.evaluation import STAGE1_EVALUATION_TEXT
from ailm.learning import (
    LearningResult,
    learn_letter_transition,
    letter_stream_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The pre-registered float32 tolerance (decision 22, P1).
TOLERANCE = 1e-5


#: A bounded slice of the Stage 1 held-out text — long enough to touch many
#: bigram cells, short enough for four pymdp runs to stay inside the timeout.
def _stream() -> str:
    return STAGE1_EVALUATION_TEXT[:300]


@pytest.fixture(scope="module")
def matched() -> LearningResult:
    return learn_letter_transition(_stream(), pseudocount=1.0)


# ── THE TOY ANCHOR: exact integer counts ─────────────────────────────────────


def test_dirichlet_increments_are_exactly_the_observed_transition_counts():
    """The conjugate mechanism, verified at integer resolution.

    A stream short enough to count by hand: the posterior Dirichlet must be
    the prior plus exactly one unit per observed transition, and the learned
    column for a letter must be the add-1 estimate of its observed successors.
    """
    import jax.numpy as jnp
    from pymdp.agent import Agent

    n = 3
    agent = Agent(
        A=[jnp.eye(n)[jnp.newaxis]],
        B=[jnp.full((1, n, n, 1), 1.0 / n)],
        C=[jnp.zeros((1, n))],
        D=[jnp.ones((1, n)) / n],
        pB=[jnp.ones((1, n, n, 1))],
        learn_B=True,
        num_controls=[1],
        batch_size=1,
    )
    stream = [0, 1, 2, 1, 0, 1]  # transitions: 0>1, 1>2, 2>1, 1>0, 0>1
    beliefs, prior = [], [jnp.ones((1, n)) / n]
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    for obs in stream:
        qs = agent.infer_states([jnp.array([obs])], prior)
        beliefs.append(qs[0].reshape(1, 1, n))
        prior = agent.update_empirical_prior(action, qs)
    updated = agent.infer_parameters(
        beliefs_A=[jnp.concatenate(beliefs, axis=1)],
        observations=[jnp.asarray(stream)[None, :]],
        actions=jnp.zeros((1, len(stream) - 1, 1), dtype=jnp.int32),
    )
    posterior_concentration = np.asarray(updated.pB[0])[0, :, :, 0]
    expected = np.ones((n, n))
    expected[1, 0] += 2  # 0>1 twice
    expected[2, 1] += 1
    expected[0, 1] += 1
    expected[1, 2] += 1
    np.testing.assert_array_equal(posterior_concentration, expected)
    learned = np.asarray(updated.B[0])[0, :, :, 0]
    np.testing.assert_allclose(learned[:, 0], [1 / 5, 3 / 5, 1 / 5], atol=1e-6)


# ── P1 / P2 / P3 on a real stream ────────────────────────────────────────────


def test_p1_matched_priors_agree_to_the_preregistered_tolerance(matched):
    assert matched.max_abs_delta <= TOLERANCE, (
        f"learned vs counted deviate by {matched.max_abs_delta:.3e} at matched "
        f"priors — decision 22's withdrawal condition applies."
    )


def test_p2_a_mismatched_prior_breaks_the_agreement_and_matching_restores_it():
    broken = learn_letter_transition(_stream(), pseudocount=2.0, comparator_pseudocount=1.0)
    assert broken.max_abs_delta > TOLERANCE, (
        "a mismatched Dirichlet prior still agrees — the comparison is vacuous"
    )
    healed = learn_letter_transition(_stream(), pseudocount=2.0)
    assert healed.max_abs_delta <= TOLERANCE, (
        f"matching the comparator's pseudocount leaves {healed.max_abs_delta:.3e} "
        f"— the prior is not the only difference between the estimators"
    )


def test_p3_learning_moved_from_the_initialization(matched):
    assert matched.moved_from_initial > 0.1
    assert not np.allclose(matched.learned, matched.initial, atol=1e-3)


# ── HARNESS CONTROLS: every knob is load-bearing ─────────────────────────────


def test_a_comparator_counted_on_a_different_stream_does_not_agree(matched):
    """Same estimator, different data — agreement here would mean the check
    could not tell two corpora apart, let alone two estimators."""
    from ailm.config import LevelConfig
    from ailm.levels.letter import build_letter_transition, train_on_corpus

    other = np.asarray(
        build_letter_transition(
            train_on_corpus([_stream()[150:] + " entirely different tail"]),
            config=LevelConfig(smoothing=1.0),
        ),
        dtype=np.float64,
    )[0]
    assert float(np.max(np.abs(matched.learned - other))) > TOLERANCE


def test_a_learning_rate_other_than_one_breaks_the_conjugate_identity():
    scaled = learn_letter_transition(_stream(), pseudocount=1.0, lr=0.5)
    assert scaled.max_abs_delta > TOLERANCE, (
        "lr_pB=0.5 still reproduces add-1 counting, so the learning rate is "
        "not entering the update and the matched runs proved nothing about it"
    )


def test_degenerate_inputs_are_refused():
    with pytest.raises(ValueError, match="transition"):
        letter_stream_indices("a")
    with pytest.raises(ValueError, match="positive"):
        learn_letter_transition("abcdef", pseudocount=0.0)


# ── LEDGER: the recorded run is the one this file re-verifies ────────────────


LEARNING_LABELS = (
    "learning_stream_positions",
    "learning_pseudocount",
    "learning_tolerance",
    "learning_matched_max_abs_delta",
    "learning_mismatched_pseudocount",
    "learning_mismatched_max_abs_delta",
    "learning_restored_max_abs_delta",
    "learning_moved_from_initial",
    "learning_learned_perplexity",
    "learning_counted_perplexity",
    "learning_perplexity_abs_delta",
)


@pytest.fixture(scope="module")
def ledger():
    return load_ledger(PROJECT_ROOT)


def test_every_learning_label_is_recorded(ledger):
    missing = [label for label in LEARNING_LABELS if not ledger.verify_label(label)]
    assert missing == [], (
        f"unrecorded learning labels {missing}; run scripts/record_learning_claims.py"
    )


def test_recorded_deltas_respect_the_preregistered_partition(ledger):
    tolerance = float(ledger.get("learning_tolerance")["value"])
    assert tolerance == TOLERANCE
    assert float(ledger.get("learning_matched_max_abs_delta")["value"]) <= tolerance
    assert float(ledger.get("learning_mismatched_max_abs_delta")["value"]) > tolerance
    assert float(ledger.get("learning_restored_max_abs_delta")["value"]) <= tolerance
    assert float(ledger.get("learning_perplexity_abs_delta")["value"]) <= 1e-6
    assert (
        ledger.get("learning_learned_perplexity")["value"]
        == ledger.get("learning_counted_perplexity")["value"]
    )


def test_the_saved_matrices_match_the_recorded_deviation(ledger):
    """The figure draws from the npz; the npz must agree with the ledger."""
    artifact = PROJECT_ROOT / "data" / "learning_agreement.npz"
    assert artifact.exists(), "run scripts/record_learning_claims.py"
    payload = np.load(artifact)
    delta = float(np.max(np.abs(payload["learned"] - payload["counted"])))
    assert delta == float(ledger.get("learning_matched_max_abs_delta")["value"])
    mismatched_delta = float(
        np.max(np.abs(payload["mismatched_learned"] - payload["mismatched_counted"]))
    )
    assert mismatched_delta == float(ledger.get("learning_mismatched_max_abs_delta")["value"])
