"""The Stage 1 unreachability argument, proved on the population it is about.

`bigram_information_bounds` measures the conditional entropy of the *Brown
training* distribution. That quantity is the order-1 optimum under `p_train`; it
is not a lower bound on held-out cross-entropy, which `docs/manuscript/02_background.md`
states explicitly. Six places in the repository nevertheless used it to argue
that no bigram model could pass a gate scored on held-out prose — proving the
right conclusion with the wrong quantity.

`evaluation_population_bound` proves it with the right one. Transition
cross-entropy decomposes as the passage's conditional entropy plus a non-negative
KL term; the scorer also includes the first character under a fixed uniform
prior. Adding that start loss and dividing by all positions yields the exact
lower bound on the same metric the gate scores.

The controls below are the ones the recorder's refusal gate needs to be
meaningful: a bound that could never fall short of the target would make the
refusal unreachable, which is the same defect as a gate a baseline cannot fail.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ailm.evaluation import STAGE1_EVALUATION_TEXT, evaluation_population_bound
from ailm.levels.letter import ALPHABET, N_SYMBOLS

_ROOT = Path(__file__).resolve().parents[1]

#: The gain the original Stage 1 gate demanded over uniform.
_DECLARED_GATE_NATS = 2.0


def test_an_unpredictable_passage_admits_almost_no_gain() -> None:
    """POSITIVE CONTROL: on near-uniform text the bound is near the uniform entropy.

    A deterministic cycle through the alphabet has one successor per symbol, so
    it is maximally *predictable*; the opposite construction is needed here. A
    seeded uniform draw over the alphabet gives a conditional entropy close to
    log|A| and therefore a maximum gain close to zero, which is the regime in
    which the bound says the gate is achievable by nobody at all.
    """
    import random

    rng = random.Random(20260817)
    passage = "".join(rng.choice(sorted(ALPHABET)) for _ in range(40_000))
    bound = evaluation_population_bound(passage)
    # Finite-sample bias pulls the estimate below log|A|; the tolerance is loose
    # for that reason and the direction of the inequality is what matters.
    assert bound["max_gain_nats"] < 0.5, bound
    assert bound["max_gain_nats"] < _DECLARED_GATE_NATS


def test_a_perfectly_predictable_passage_admits_more_than_the_declared_gain() -> None:
    """NEGATIVE CONTROL: the refusal gate must be reachable.

    On text whose next symbol is determined by the current one, the conditional
    entropy is zero and the achievable gain is the full log|A| — well past the
    two nats the gate demanded. If this passage did *not* trip the recorder's
    refusal, the refusal could never fire and would be decoration.
    """
    bound = evaluation_population_bound("ab" * 5_000)
    assert bound["conditional_entropy_nats"] == pytest.approx(0.0, abs=1e-12)
    expected = math.log(N_SYMBOLS) * (bound["transitions"] / bound["positions"])
    assert bound["max_gain_nats"] == pytest.approx(expected, abs=1e-12)
    assert bound["max_gain_nats"] >= _DECLARED_GATE_NATS, (
        "the recorder refuses to record when this holds; if it stopped holding "
        "here, the refusal branch would be unreachable"
    )


def test_the_bound_is_below_the_declared_gate_on_the_real_passage() -> None:
    """The finding itself, on the passage the gate scores."""
    bound = evaluation_population_bound(STAGE1_EVALUATION_TEXT)
    assert bound["max_gain_nats"] < _DECLARED_GATE_NATS, (
        f"the held-out passage admits {bound['max_gain_nats']:.4f} nats of order-1 "
        f"gain over uniform, reaching the declared {_DECLARED_GATE_NATS}. The "
        "gate-design argument would need rewriting, not re-rendering."
    )


def test_the_bound_really_bounds_the_shipped_model() -> None:
    """A lower bound that the shipped model beat would be no bound at all.

    Checked against the ledger rather than recomputed, so a re-measurement that
    broke the inequality fails here instead of quietly contradicting the section.
    """
    from ailm.claims import load_ledger

    ledger = load_ledger(_ROOT)

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"{label} is not in the ledger"
        return float(entry["value"])

    floor = value("stage1_heldout_floor_perplexity")
    model = value("stage1_letter_perplexity")
    assert model > floor, (
        f"the agent scored {model} on the held-out passage, below the exact order-1 "
        f"floor {floor} for that passage. One of the two is wrong: the floor is a "
        "theorem, so suspect the perplexity denominator or the normalization."
    )


def test_the_two_bounds_are_not_confused_with_each_other() -> None:
    """They differ, and each is reported under its own name.

    The training bound is tighter here, so a copy-paste that swapped them would
    change every derived ratio without changing the qualitative conclusion —
    exactly the kind of substitution the ledger's naming exists to prevent.
    """
    from ailm.claims import load_ledger

    ledger = load_ledger(_ROOT)

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"{label} is not in the ledger"
        return float(entry["value"])

    training = value("stage1_max_bigram_gain_nats")
    heldout = value("stage1_heldout_max_gain_nats")
    assert training != heldout
    assert max(training, heldout) < _DECLARED_GATE_NATS, (
        "the declared gate must exceed BOTH bounds for the section's argument to "
        "hold on both populations"
    )
    # The captured-signal percentages are ratios against different denominators
    # and must not be equal either.
    assert value("stage1_bigram_signal_captured_pct") != value("stage1_heldout_signal_captured_pct")


def test_all_position_bound_includes_the_uniform_initial_prediction() -> None:
    """The transition-only entropy cannot silently stand in for the gate metric."""
    bound = evaluation_population_bound("abab")
    assert bound["positions"] == 4
    assert bound["transitions"] == 3
    assert bound["conditional_entropy_nats"] == pytest.approx(0.0, abs=1e-12)
    assert bound["all_position_entropy_nats"] == pytest.approx(math.log(N_SYMBOLS) / 4, abs=1e-12)
    assert bound["perplexity"] == pytest.approx(N_SYMBOLS ** (1 / 4), abs=1e-12)


def test_dropping_the_initial_prediction_changes_the_reported_bound() -> None:
    """NEGATIVE CONTROL: the historical T-1 denominator must be distinguishable."""
    bound = evaluation_population_bound(STAGE1_EVALUATION_TEXT)
    wrong_transitions_only = math.exp(bound["conditional_entropy_nats"])
    assert bound["perplexity"] != pytest.approx(wrong_transitions_only, abs=1e-12)


@pytest.mark.parametrize("degenerate", ["", "a", " "])
def test_a_passage_too_short_to_have_a_transition_fails_closed(degenerate: str) -> None:
    with pytest.raises(ValueError):
        evaluation_population_bound(degenerate)


def test_the_stage_one_audit_compares_on_one_population() -> None:
    """The audited row's baseline, model and comparator must share a population.

    the gate-audit checklist in `docs/manuscript/03_methods.md` requires it, and by
    `bracketed_at_baseline_bar_iff` a baseline-relative verdict contains exactly
    one measured comparison — so a comparator from a different population makes
    that single claim a cross-population one. The Stage 1 row used the Brown
    *training* conditional entropy against a held-out baseline and model.

    Checked against the audit's own output rather than against the recorded
    claim, because the defect lives in what the audit compares.
    """
    import sys as _sys

    _sys.path.insert(0, str(_ROOT / "scripts"))
    from audit_gates import audit_stage1

    from ailm.claims import load_ledger

    row = audit_stage1()
    ledger = load_ledger(_ROOT)

    def value(label: str) -> float:
        entry = ledger.get(label)
        assert entry is not None, f"{label} is not in the ledger"
        return float(entry["value"])

    # The ledger stores four decimals, so the tolerance is that rounding and not
    # a slack allowance: at 1e-6 this compares a rounded number to an unrounded
    # one and fails on arithmetic rather than on the property being checked.
    assert row["ceiling"] == pytest.approx(value("stage1_heldout_floor_perplexity"), abs=5e-5), (
        "the Stage 1 audit's comparator is not the evaluation passage's own "
        "order-1 floor. If it reverted to the training-distribution entropy, the "
        "row's one measured comparison straddles two populations again."
    )
    assert row["ceiling"] != pytest.approx(value("stage1_bigram_floor_perplexity"), abs=5e-5), (
        "the comparator equals the TRAINING floor; the two are different "
        "quantities and only one of them matches this row's baseline"
    )
    # The comparison the verdict actually rests on, on one population.
    assert row["ceiling"] < row["baseline"], (
        "the same-population floor no longer beats the same-population baseline, "
        "so the Stage 1 row's BRACKETED verdict is no longer supported"
    )
    assert row["ceiling"] < row["model"] < row["baseline"], (
        "the shipped model no longer sits between the floor and the bar"
    )
