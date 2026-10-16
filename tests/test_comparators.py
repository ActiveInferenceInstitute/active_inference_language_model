"""The comparator search, and the properties that make its result mean anything.

`ailm.comparators` exists to answer a question the staged-gate method does not ask:
what is the *strongest* comparator that is still not the model? Its answer beats
two shipped gates, so these tests check the answer is real rather than an artifact
of how the lookup was fitted or scored.
"""

from __future__ import annotations

import pytest

from ailm.comparators import (
    START_CONTEXT,
    deranged_lookup,
    fit_left_context_lookup,
    score_left_context_lookup,
    score_unigram_lookup,
)

#: A tiny corpus where the right answer is obvious by inspection: the label
#: depends on the PAIR, never on the current symbol alone, so a unigram argmax
#: must do badly and a bigram lookup must be perfect.
PAIR_DEPENDENT = [
    [("a", "X"), ("b", "P")],
    [("a", "X"), ("b", "P")],
    [("c", "Y"), ("b", "Q")],
    [("c", "Y"), ("b", "Q")],
]


def test_the_bigram_lookup_learns_a_dependency_the_unigram_cannot():
    """The positive control, on data built so the gap is unambiguous.

    ``b`` is followed by ``P`` after ``a`` and by ``Q`` after ``c``, in equal
    numbers. A per-symbol argmax on ``b`` must pick one and be wrong half the
    time; the pair lookup has no such excuse.
    """
    bigram, unigram = fit_left_context_lookup(PAIR_DEPENDENT)
    rich = score_left_context_lookup(PAIR_DEPENDENT, bigram, unigram)
    poor = score_unigram_lookup(PAIR_DEPENDENT, unigram)
    assert rich.accuracy_pct == 100.0
    assert poor.accuracy_pct < 100.0
    assert rich.accuracy_pct > poor.accuracy_pct


def test_more_context_never_scores_worse_on_its_own_training_data():
    """The invariant the recorder refuses on.

    The bigram map backs off to the unigram map, so on the data both were fitted
    to it can only match or improve. A violation means the fit or the scoring
    alignment is broken, not that context is unhelpful.
    """
    bigram, unigram = fit_left_context_lookup(PAIR_DEPENDENT)
    rich = score_left_context_lookup(PAIR_DEPENDENT, bigram, unigram)
    poor = score_unigram_lookup(PAIR_DEPENDENT, unigram)
    assert rich.accuracy_pct >= poor.accuracy_pct


def test_the_derangement_preserves_everything_except_the_mapping():
    """The negative control must differ in one respect only.

    If a derangement changed the key set, the label multiset or the parameter
    count, a drop in its score would be explained by the change rather than by
    the mapping, and the control would be measuring itself.
    """
    bigram, _unigram = fit_left_context_lookup(PAIR_DEPENDENT)
    shuffled = deranged_lookup(bigram)
    assert set(shuffled) == set(bigram)
    assert sorted(shuffled.values()) == sorted(bigram.values())
    assert len(shuffled) == len(bigram)


def test_the_derangement_actually_moves_labels_off_their_keys():
    """A control that returned its input would guarantee a null.

    This is the failure the in-context arm's derangement guard was written for,
    in a different part of the project, and it is worth checking here too rather
    than trusting a rotation to be fixed-point-free.
    """
    bigram, _unigram = fit_left_context_lookup(PAIR_DEPENDENT)
    if len(set(bigram.values())) < 2:
        pytest.skip("a single-label map admits no derangement")
    shuffled = deranged_lookup(bigram)
    assert shuffled != bigram


def test_sequence_start_uses_a_context_outside_every_alphabet():
    """The start marker must not collide with a real symbol.

    If it did, position 0 would share statistics with genuine occurrences of that
    symbol and the first position of every sequence would be scored under the
    wrong distribution.
    """
    bigram, _unigram = fit_left_context_lookup(PAIR_DEPENDENT)
    starts = {key for key in bigram if key[0] == START_CONTEXT}
    assert starts, "no sequence-start keys were fitted"
    observed = {symbol for sequence in PAIR_DEPENDENT for symbol, _label in sequence}
    assert START_CONTEXT not in observed


def test_an_empty_fit_is_refused_rather_than_returned():
    """A comparator fitted on nothing scores as a constant predictor.

    Returning it would put a constant in a table headed "comparator", which is
    the shape of every defect this project's gate audit exists to catch.
    """
    with pytest.raises(ValueError, match="constant"):
        fit_left_context_lookup([])


def test_scoring_an_empty_evaluation_set_is_refused():
    """An accuracy over zero positions is not a measurement."""
    bigram, unigram = fit_left_context_lookup(PAIR_DEPENDENT)
    with pytest.raises(ValueError, match="not a measurement"):
        score_left_context_lookup([], bigram, unigram)


def test_backoff_positions_are_reported_not_hidden():
    """A lookup that backs off everywhere is a unigram wearing a bigram's name.

    The reader needs the count to tell the two apart, so the score carries it.
    """
    bigram, unigram = fit_left_context_lookup(PAIR_DEPENDENT)
    unseen = [[("z", "X"), ("z", "X")]]
    score = score_left_context_lookup(unseen, bigram, unigram)
    assert score.backoff_positions == score.positions


def test_the_shipped_probe_reproduces_both_declared_context_free_bars():
    """The claim that makes the whole finding admissible.

    The comparator probe's unigram backoff and the gates' own context-free
    baselines are two implementations of one predictor. If they disagree, the
    probe is scoring a different population and the recorded comparator numbers
    are about something else. The recorder refuses on this; the assertion is
    repeated here so the property is checked without running a recorder.
    """
    from pathlib import Path

    from ailm.claims import load_ledger

    root = Path(__file__).resolve().parents[1]
    ledger = load_ledger(root)
    for probe, shipped in (
        ("comparator_stage2_unigram_accuracy", "stage2_g2p_baseline_accuracy"),
        ("comparator_stage4_unigram_accuracy", "stage4_derived_baseline"),
    ):
        mine = ledger.get(probe)
        bar = ledger.get(shipped)
        assert mine is not None and bar is not None, f"{probe} or {shipped} missing"
        assert abs(float(mine["value"]) - float(bar["value"])) <= 0.01, (
            f"{probe} = {mine['value']} against {shipped} = {bar['value']}; the probe is "
            f"not scoring the gate's population"
        )


def test_the_lookup_beats_the_model_at_both_levels_or_the_finding_is_gone():
    """The finding itself, pinned so it cannot quietly evaporate.

    Written as an assertion rather than left in prose because the manuscript now
    states it. If a future change makes the model win, this fails and the
    manuscript section must be rewritten rather than left standing.
    """
    from pathlib import Path

    from ailm.claims import load_ledger

    ledger = load_ledger(Path(__file__).resolve().parents[1])
    for label in ("comparator_stage2_over_model_pts", "comparator_stage4_over_model_pts"):
        entry = ledger.get(label)
        assert entry is not None, f"{label} missing"
        assert float(entry["value"]) > 0.0, (
            f"{label} = {entry['value']}; the lookup no longer beats the model, so "
            f"[@sec:comparator_strength] no longer describes the tree"
        )
