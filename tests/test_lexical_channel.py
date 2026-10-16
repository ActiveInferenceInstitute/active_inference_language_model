"""L3 → L2B lexical channel — the test of the novel-information criterion.

What this file is for
---------------------
``docs/decisions.md`` entry 18 states a criterion for when an inter-level message
helps — it must carry information the receiver does not already have — and
derives a prediction *before* measuring. Entry 19 records that the prediction
held. These tests keep both the result and the controls that make it evidence.

The comparison is controlled: entry 9's channel sent L1's letter predictive into
L2B and measured harmful; this one sends a lexical prefix statistic into the
**same** receiving level and helps. Architecture, direction and receiver are
matched; novelty is the variable.

The two signatures that carry the claim
---------------------------------------
1. The effect is positive and survives a paired test.
2. The sweep does **not** fall monotonically — every engaged weight helps,
   including the largest — which is exactly what entry 9's channel did not do.

What the shape does and does not show
-------------------------------------
The sample argmax is interior, and that ordering is recorded. It is **not**
evidence that over-weighting costs accuracy: the peak and the highest weight are
separated by claim:lexical_peak_vs_high_alpha_pts points on
claim:lexical_eval_positions paired positions, at
claim:lexical_peak_vs_high_mcnemar_p, so the two arms are not distinguishable.
The claim the sweep does carry is the contrast with entry 9: that channel fell
monotonically into harm, and this one is above the anchor at every weight on the
grid. The distinguishing signature is the *direction* over the grid, not a
demonstrated interior cost.

An earlier revision of this file asserted "the effect reverses at high weight".
That statement was produced by a backoff defect in ``LexicalMessage.lookup``
(the reader truncated the joined key string from the right, dropping the newest
letters), which depressed the high-weight arm below the anchor. Under the
documented longest-suffix rule the reversal is absent and the claim is withdrawn
rather than qualified.

Claims discipline
-----------------
This file **verifies**; ``scripts/record_lexical_channel_claims.py`` writes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

from ailm.claims import load_ledger
from ailm.levels.letter import ALPHA_TO_IDX
from ailm.levels.lexical_channel import (
    LEXICAL_SWEEP_ALPHAS,
    build_lexical_message,
    coupled_g2p_positions,
    lexical_message_for,
    lexical_sweep_claim_suffix,
    scramble_phone_distribution,
    sweep_lexical_alpha,
)
from ailm.levels.phoneme import (
    PHONES,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    g2p_split,
    load_phoneme_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNTS = PROJECT_ROOT / "vendor" / "cmudict_counts.json"
GRID = LEXICAL_SWEEP_ALPHAS


def test_sweep_claim_suffix_preserves_the_decimal_anchor_and_rejects_extra_rungs():
    assert [lexical_sweep_claim_suffix(alpha) for alpha in GRID] == [
        "0_0",
        "0_1",
        "0_2",
        "0_3",
        "0_5",
        "0_7",
        "0_9",
    ]
    with pytest.raises(ValueError, match="registered lexical sweep grid"):
        lexical_sweep_claim_suffix(0.4)


BEST_ALPHA = 0.7
ALPHA_LEVEL = 0.01

pytestmark = pytest.mark.skipif(not COUNTS.exists(), reason="vendor/cmudict_counts.json absent")


@pytest.fixture(scope="module")
def model():
    counts = load_phoneme_counts(PROJECT_ROOT / "vendor")
    train, test = g2p_split(counts)
    return {
        "train": train,
        "test": test,
        "emission": np.asarray(build_g2p_emission(train))[0],
        "transition": np.asarray(build_g2p_transition(train))[0, :, :, 0],
        "start": np.asarray(build_g2p_start(train)).ravel(),
        "message": build_lexical_message(train, list(PHONES)),
    }


@pytest.fixture(scope="module")
def sweep(model) -> dict[float, list[bool]]:
    return sweep_lexical_alpha(
        model["test"],
        model["emission"],
        model["transition"],
        model["start"],
        model["message"],
        list(PHONES),
        ALPHA_TO_IDX,
        GRID,
    )


def _accuracy(hits: list[bool]) -> float:
    return 100.0 * sum(hits) / len(hits)


# ══ THE RESULT ════════════════════════════════════════════════════════════════


def test_the_lexical_channel_helps(sweep):
    """(1) The predicted sign. Entry 18 recorded 'helps' before measuring."""
    anchor = _accuracy(sweep[0.0])
    best = _accuracy(sweep[BEST_ALPHA])
    assert best > anchor, (
        f"The lexical channel scores {best:.3f}% against an anchor of "
        f"{anchor:.3f}%. Entry 18 predicted it would help; if it no longer does, "
        f"the novel-information criterion is falsified and entry 19 must be "
        f"withdrawn rather than qualified."
    )


def test_the_margin_survives_a_paired_test(sweep):
    """A margin without a paired statistic is not evidence (decision 5)."""
    anchor, best = sweep[0.0], sweep[BEST_ALPHA]
    b = sum(1 for x, y in zip(best, anchor, strict=True) if x and not y)
    c = sum(1 for x, y in zip(best, anchor, strict=True) if y and not x)
    assert b > c, f"Discordance favors the anchor: b={b}, c={c}."
    assert binomtest(b, b + c, 0.5).pvalue < ALPHA_LEVEL


def test_the_sweep_does_not_fall_monotonically_and_the_argmax_is_interior(sweep):
    """(2) The SHAPE that distinguishes novel information from redundant.

    Entry 9's channel fell monotonically into harm because its message was
    redundant. This one is above the anchor at every engaged weight, and the
    sample argmax sits strictly inside the grid. Both halves are asserted, and
    the second is asserted *as an ordering only*: the companion test below shows
    the peak and the highest weight are not distinguishable, so nothing here may
    be read as over-weighting having a measured cost.
    """
    anchor = _accuracy(sweep[0.0])
    deltas = {a: _accuracy(sweep[a]) - anchor for a in GRID}

    engaged = [a for a in GRID if a > 0.0]
    harmed = {a: round(deltas[a], 3) for a in engaged if deltas[a] <= 0.0}
    assert not harmed, (
        f"The channel is at or below the anchor at {harmed}. Entry 9's redundant "
        f"channel harmed the same receiver; a lexical message that also harms it "
        f"removes the contrast the criterion rests on."
    )
    assert deltas[max(GRID)] > 0.0, (
        f"The largest weight is {deltas[max(GRID)]:+.3f} pts. The sweep no longer "
        f"contrasts with the monotone fall into harm of entry 9's channel."
    )

    best_alpha = max(deltas, key=lambda a: deltas[a])
    assert 0.0 < best_alpha < max(GRID), (
        f"The best alpha is {best_alpha}, at an endpoint. The recorded "
        f"lexical_optimum_is_interior flag asserts it is inside the range."
    )


def test_the_interior_argmax_is_not_evidence_that_over_weighting_costs_accuracy(sweep):
    """The limit that must travel with the interior-optimum ordering.

    An argmax is a comparison of two sample accuracies on one split. Reporting it
    as "over-weighting the message costs accuracy" asserts a difference that has
    to survive a paired test on the same positions, and this one does not: the
    peak and the highest weight differ by a fraction of a point across a few
    hundred discordant positions. Stated as a test so the limit cannot be dropped
    while the ordering is kept.

    PLANTED-FAILURE CONTROL: the same predicate, applied to a sweep whose high
    arm really is worse, must fire. Without it "not distinguishable" could be
    produced by a predicate that never distinguishes anything.
    """
    peak, high = sweep[BEST_ALPHA], sweep[max(GRID)]
    b = sum(1 for x, y in zip(peak, high, strict=True) if x and not y)
    c = sum(1 for x, y in zip(peak, high, strict=True) if y and not x)
    assert binomtest(b, b + c, 0.5).pvalue > ALPHA_LEVEL, (
        f"The peak now beats the highest weight at b={b}, c={c}, "
        f"p={binomtest(b, b + c, 0.5).pvalue:.3g}. Over-weighting has become a "
        f"measured cost, so the manuscript may state it — restate the limit "
        f"rather than deleting this test."
    )

    # PLANTED FAILURE: flip a decisive block of the high arm to misses.
    planted = list(high)
    flipped = 0
    for index, (p_hit, h_hit) in enumerate(zip(peak, high, strict=True)):
        if p_hit and h_hit and flipped < 500:
            planted[index] = False
            flipped += 1
    assert flipped == 500, "the split is too small to plant a detectable effect"
    pb = sum(1 for x, y in zip(peak, planted, strict=True) if x and not y)
    pc = sum(1 for x, y in zip(peak, planted, strict=True) if y and not x)
    assert binomtest(pb, pb + pc, 0.5).pvalue < ALPHA_LEVEL, (
        "the planted degradation was not detected, so this test cannot tell a "
        "real cost from an indistinguishable one"
    )


def test_the_effect_is_larger_than_the_levels_own_headline_margin(sweep):
    """Scale check: the channel is worth several times Stage 2's own result.

    Recorded because a statistically significant effect can still be negligible,
    and 'significant' is not a synonym for 'matters'.
    """
    ledger = load_ledger(PROJECT_ROOT)
    stage2 = ledger.get("stage2_g2p_margin_pts")
    if stage2 is None:
        pytest.skip("stage2_g2p_margin_pts not recorded")
    margin = _accuracy(sweep[BEST_ALPHA]) - _accuracy(sweep[0.0])
    assert margin > float(stage2["value"]), (
        f"The lexical channel adds {margin:.3f} pts against the level's own "
        f"headline margin of {stage2['value']} pts."
    )


# ══ CONTROLS ══════════════════════════════════════════════════════════════════


def test_the_anchor_is_exact(model, sweep):
    """At alpha = 0 the mixture term vanishes IDENTICALLY, not approximately."""
    scrambled = sweep_lexical_alpha(
        model["test"],
        model["emission"],
        model["transition"],
        model["start"],
        model["message"],
        list(PHONES),
        ALPHA_TO_IDX,
        GRID,
        scrambled=True,
    )
    assert sweep[0.0] == scrambled[0.0], (
        "The real and scrambled sweeps disagree at alpha = 0, where the message "
        "cannot influence anything. Something other than the message is driving "
        "the decode."
    )


def test_the_anchor_is_not_vacuous(sweep):
    """A positive alpha must change the output, or the anchor proves nothing."""
    assert sweep[BEST_ALPHA] != sweep[0.0]


def test_the_scrambled_message_does_worse(model, sweep):
    """The entropy-preserving control.

    Reversing the phone axis is a fixed involutive permutation, so the scrambled
    message keeps its entropy and loses only its meaning — strictly stronger than
    a uniform replacement, which would confound content with entropy.
    """
    scrambled = sweep_lexical_alpha(
        model["test"],
        model["emission"],
        model["transition"],
        model["start"],
        model["message"],
        list(PHONES),
        ALPHA_TO_IDX,
        GRID,
        scrambled=True,
    )
    for alpha in (a for a in GRID if a > 0.0):
        assert _accuracy(scrambled[alpha]) < _accuracy(sweep[alpha]), (
            f"At alpha={alpha} the scrambled message scores as well as the real "
            f"one, so the channel's content is doing no work."
        )


def test_the_channel_actually_conditions_on_prefixes(model):
    """NON-VACUITY: a channel that always backs off is not lexical.

    Without this, a null result would be uninterpretable — it could mean the
    criterion is wrong, or merely that the message never had anything to say.
    """
    depths: Counter = Counter()
    for word, _ in model["test"][:2000]:
        _, per_position = lexical_message_for(model["message"], word)
        depths.update(per_position)
    lexical = sum(v for k, v in depths.items() if k > 0) / sum(depths.values())
    assert lexical > 0.5, (
        f"Only {100 * lexical:.1f}% of positions had a real prefix to condition "
        f"on; the rest fell back to the context-free distribution."
    )


def test_scrambling_preserves_entropy(model):
    """The control must differ from the real message ONLY in meaning."""
    rows, _ = lexical_message_for(model["message"], "cat")
    flipped = scramble_phone_distribution(rows)
    np.testing.assert_allclose(np.sort(rows, axis=1), np.sort(flipped, axis=1))
    assert not np.allclose(rows, flipped), "Reversal left the message unchanged"


# ══ THE SWEEP REFUSES AN UNSUPPORTABLE CLAIM ═════════════════════════════════


def test_sweep_requires_an_anchor_and_a_real_grid(model):
    """ "No alpha helped" must not be producible by a collapsed sweep."""
    args = (
        model["test"][:50],
        model["emission"],
        model["transition"],
        model["start"],
        model["message"],
        list(PHONES),
        ALPHA_TO_IDX,
    )
    with pytest.raises(ValueError, match="collapsed sweep"):
        sweep_lexical_alpha(*args, (0.1, 0.2, 0.3))  # no anchor
    with pytest.raises(ValueError, match="collapsed sweep"):
        sweep_lexical_alpha(*args, (0.0, 0.3))  # too few positive alphas


def test_alpha_outside_the_unit_interval_raises(model):
    with pytest.raises(ValueError, match="alpha must lie"):
        coupled_g2p_positions(
            model["emission"],
            model["transition"],
            model["start"],
            np.array([0, 1], dtype=np.int64),
            [0, 1],
            None,
            1.5,
        )


def test_a_positive_alpha_without_a_message_raises(model):
    """Silently treating a missing message as zero would fake the anchor."""
    with pytest.raises(ValueError, match="requires a message"):
        coupled_g2p_positions(
            model["emission"],
            model["transition"],
            model["start"],
            np.array([0, 1], dtype=np.int64),
            [0, 1],
            None,
            0.5,
        )


def test_memorising_prefixes_is_refused(model):
    """A prefix seen once yields a point mass — memorisation, not generalization."""
    with pytest.raises(ValueError, match="memorises"):
        build_lexical_message(model["train"][:100], list(PHONES), min_count=0)


# ══ CLAIMS LEDGER — verify, never write ══════════════════════════════════════


def test_lexical_claims_verified(sweep):
    ledger = load_ledger(PROJECT_ROOT)
    anchor = _accuracy(sweep[0.0])
    _peak, _high = sweep[BEST_ALPHA], sweep[max(GRID)]
    _peak_b = sum(1 for x, y in zip(_peak, _high, strict=True) if x and not y)
    _peak_c = sum(1 for x, y in zip(_peak, _high, strict=True) if y and not x)
    expected: dict[str, object] = {
        "lexical_peak_vs_high_mcnemar_b": _peak_b,
        "lexical_peak_vs_high_mcnemar_c": _peak_c,
        "lexical_peak_vs_high_mcnemar_p": float(
            f"{binomtest(_peak_b, _peak_b + _peak_c, 0.5).pvalue:.4g}"
        ),
        "lexical_anchor_accuracy": round(anchor, 3),
        "lexical_best_alpha": BEST_ALPHA,
        "lexical_best_accuracy": round(_accuracy(sweep[BEST_ALPHA]), 3),
        "lexical_best_margin_pts": round(_accuracy(sweep[BEST_ALPHA]) - anchor, 3),
        "lexical_optimum_is_interior": 1,
        "lexical_peak_vs_high_alpha_pts": round(
            _accuracy(sweep[BEST_ALPHA]) - _accuracy(sweep[max(GRID)]), 3
        ),
        # The full per-alpha sweep rows the recorder writes, re-derived here
        # from the same paired hits the recorder counts. lexical_high_alpha_*
        # names alpha = max(GRID); the per-alpha labels name their own weight.
        "lexical_high_alpha_accuracy": round(_accuracy(sweep[max(GRID)]), 3),
    }
    for swept_alpha in GRID:
        hits = sweep[swept_alpha]
        _b = sum(1 for x, y in zip(hits, sweep[0.0], strict=True) if x and not y)
        _c = sum(1 for x, y in zip(hits, sweep[0.0], strict=True) if y and not x)
        _label = lexical_sweep_claim_suffix(swept_alpha)
        expected[f"lexical_sweep_alpha_{_label}_mcnemar_b"] = _b
        expected[f"lexical_sweep_alpha_{_label}_mcnemar_c"] = _c
        expected[f"lexical_sweep_alpha_{_label}_mcnemar_p"] = float(
            f"{binomtest(_b, _b + _c, 0.5).pvalue if _b + _c else 1.0:.4g}"
        )
    for label, value in expected.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent. Run scripts/record_lexical_channel_claims.py — "
            f"this test may not write it."
        )
        assert entry["source"].startswith("scripts/"), (
            f"{label} was last written by {entry['source']!r}."
        )
        assert entry["value"] == pytest.approx(value, rel=1e-3), (
            f"Ledger {label}={entry['value']!r} != measurement {value!r}."
        )


# ══ VALIDATION / ERROR-PATH BRANCHES ═══════════════════════════════════════════


def test_build_lexical_message_rejects_empty_train_pairs():
    """An empty training set is nothing to estimate a generalization from."""
    with pytest.raises(ValueError, match="train_pairs is empty"):
        build_lexical_message([], PHONES)


def test_build_lexical_message_rejects_min_count_below_one():
    """min_count=0 would let a single observation memorise a prefix."""
    with pytest.raises(ValueError, match="min_count must be at least 1"):
        build_lexical_message([("cat", ["K", "AE", "T"])], PHONES, min_count=0)


def test_build_lexical_message_skips_length_mismatched_pairs():
    """A pair whose letter/phone counts differ is not length-matched; skip it.

    Stated as an exact key set rather than a count. ``through`` has seven letters
    against three phones, so every one of its positions is unalignable; if the
    skip were dropped it would contribute ``0|t`` and five more keys, and the
    equality below is what notices.
    """
    msg = build_lexical_message(
        [("cat", ["K", "AE", "T"]), ("through", ["TH", "R", "UW"])], PHONES, min_count=1
    )
    assert set(msg.by_prefix) == {"0|c", "1|a", "1|ca", "2|t", "2|at", "2|cat"}
    assert "0|t" not in msg.by_prefix, "the length-mismatched pair contributed a key"


def test_build_lexical_message_ignores_out_of_inventory_phones():
    """A phone not in the given inventory is skipped, not mis-indexed."""
    msg = build_lexical_message([("cat", ["K", "AE", "T"])], PHONES, min_count=1)
    # The model does not crash on inventory membership; fallback sums to ~1.
    assert abs(float(msg.fallback.sum()) - 1.0) < 1e-6


# ══ BACKOFF SEMANTICS — the rule the tables were built under ═════════════════
#
# The message keys are ``position|context`` where the context is the run of
# letters ENDING at that position, so backoff must shorten the context from the
# LEFT. Shortening the joined key string from the right instead drops the newest
# letters and matches a word whose current letter is not the observed one; the
# tests below are the ones that would have caught it, and each names the value
# the defective reader produced.

TOY_PHONES = ["Z", "AA", "B", "K", "T", "W"]


def _toy_message(**kwargs):
    """Three copies of one word, so every key clears ``min_count=3`` unsmoothed."""
    train = [("zabc", ["Z", "AA", "B", "K"])] * 3
    return build_lexical_message(train, TOY_PHONES, min_count=3, smoothing=0.0, **kwargs)


def test_the_table_keys_are_contexts_that_end_at_their_position():
    """The premise every other backoff test rests on, stated exactly once."""
    assert set(_toy_message().by_prefix) == {
        "0|z",
        "1|a",
        "1|za",
        "2|b",
        "2|ab",
        "2|zab",
        "3|c",
        "3|bc",
        "3|abc",
        "3|zabc",
    }


def test_an_unseen_context_falls_back_rather_than_matching_a_different_letter():
    """'abcq' shares a prefix with the trained 'zabc' but shares no suffix.

    At position 3 the observed letter is 'q'. None of ``3|abcq``, ``3|bcq``,
    ``3|cq``, ``3|q`` was ever seen, so the context-free distribution is the only
    admissible answer. Truncating the key string from the right instead resolved
    this position to ``3|abc`` — a context whose current letter is 'c' — and
    reported depth 5; positions 1 and 2 landed on ``1|a`` and ``2|abc`` the same
    way, for depths [0, 3, 5, 5] where the rule gives [0, 0, 0, 0].
    """
    msg = _toy_message()
    rows, depths = lexical_message_for(msg, "abcq")
    assert depths == [0, 0, 0, 0], (
        f"depths {depths} — a non-zero depth here means the message matched a "
        f"context that does not end at the letter being decoded."
    )
    np.testing.assert_array_equal(rows[3], msg.fallback)


def test_a_short_suffix_beats_a_long_non_suffix():
    """Depth is not a proxy for relevance: ``3|q`` must win over ``3|abc``.

    Both keys exist here. The rule takes the one-letter suffix, whose phone is T;
    right-truncation took the three-letter non-suffix, whose phone is K.
    """
    train = [("zabc", ["Z", "AA", "B", "K"])] * 3 + [("wxyq", ["W", "AA", "B", "T"])] * 3
    msg = build_lexical_message(train, TOY_PHONES, min_count=3, smoothing=0.0)
    assert "3|abc" in msg.by_prefix and "3|q" in msg.by_prefix
    rows, depths = lexical_message_for(msg, "abcq")
    assert depths[3] == 1
    assert TOY_PHONES[int(np.argmax(rows[3]))] == "T"


def test_depth_counts_context_letters_and_not_key_characters():
    """The reported depth was ``len('3|zabc')`` — the stem inflated every count.

    On the trained word itself the four positions match contexts of 1, 2, 3 and 4
    letters; the defective reader returned 3, 4, 5 and 6.
    """
    _, depths = lexical_message_for(_toy_message(), "zabc")
    assert depths == [1, 2, 3, 4]


def test_the_lookup_bound_comes_from_the_message_not_from_the_caller():
    """A table built to length 8 must be read to length 8.

    ``scripts/record_fitted_channel_claims.py`` fits ``max_prefix`` over
    (4, 6, 8), so a reader pinned at 6 would ignore the longest keys of every 8
    arm and the selection would range over a grid it never actually ran. Here the
    eight-letter context is a point mass and the six-letter one is a 50/50 tie, so
    the bound is visible in the row rather than only in the depth.
    """
    train = [("abcdefgh", ["Z", "AA", "B", "K", "T", "Z", "AA", "B"])] * 3 + [
        ("zycdefgh", ["Z", "AA", "B", "K", "T", "Z", "AA", "T"])
    ] * 3
    msg = build_lexical_message(train, TOY_PHONES, max_prefix=8, min_count=3, smoothing=0.0)
    assert msg.max_prefix == 8
    assert "7|abcdefgh" in msg.by_prefix and "7|cdefgh" in msg.by_prefix
    rows, depths = lexical_message_for(msg, "abcdefgh")
    assert depths[7] == 8
    assert float(rows[7].max()) == pytest.approx(1.0), (
        "position 7 resolved to the ambiguous six-letter context, so the reader "
        "is not using the bound its own tables were built under"
    )


def test_every_matched_context_ends_at_the_current_letter(model):
    """The same invariant over the real held-out split, at every position.

    A depth of L asserts the key ``f"{t}|{word[t + 1 - L : t + 1]}"`` was found.
    That key must exist and its context must end with the letter at t. On the
    shipped artifact the defective reader violated this at 38% of the 42,820
    held-out positions, which is the scale at which it rewrote the message.
    """
    message = model["message"]
    checked = 0
    for word, _ in model["test"]:
        _, depths = lexical_message_for(message, word)
        for position, depth in enumerate(depths):
            if depth == 0:
                continue
            context = word[position + 1 - depth : position + 1]
            assert len(context) == depth
            assert context.endswith(word[position]), (
                f"{word!r} position {position}: matched context {context!r} ends "
                f"at {context[-1]!r}, not at the decoded letter {word[position]!r}"
            )
            assert f"{position}|{context}" in message.by_prefix
            checked += 1
    assert checked > 0, "no position matched a context; the invariant is vacuous here"
