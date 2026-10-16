"""The word level's coverage/perplexity sweep, and why it reports both.

Numeric coverage and per-emission perplexity both rise with lexicon size, while
their utility directions oppose: more coverage is desirable and more
perplexity is worse. The shared-denominator gain widens across the same ladder.
A single operating point can be quoted to support either reading, so the sweep
exists to make the trade visible rather than selectable.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from ailm.levels.word import SCALING_RUNGS, WordArtifact, load_word_artifact, scaling_rungs

FULL_ARTIFACT = Path(__file__).resolve().parents[1] / "vendor" / "brown_word_counts_full.json"

pytestmark = pytest.mark.skipif(
    not FULL_ARTIFACT.exists(), reason="full-vocabulary artifact absent; run the generators"
)


@pytest.fixture(scope="module")
def rungs() -> list[dict[str, float | int]]:
    return scaling_rungs(load_word_artifact(FULL_ARTIFACT))


def test_the_sweep_ends_at_every_length_eligible_training_type(rungs) -> None:
    """The top rung contains every length-eligible training type, not a literal.

    Hard-coding the largest size would go stale the moment the corpus changes,
    and it would go stale silently: the sweep would still run, still look like a
    ladder, and quietly stop short of the lexicon it claims to reach.
    """
    full = load_word_artifact(FULL_ARTIFACT)
    assert full.provenance["vocab_bound"] == "unbounded"
    assert len(full.vocab) == (
        int(full.provenance["train_types"]) - int(full.provenance["types_dropped_for_length"])
    )
    assert rungs[-1]["vocab_size"] == len(full.vocab)
    assert rungs[-1]["coverage_pct"] == pytest.approx(round(full.coverage_pct, 2))


def test_coverage_rises_and_perplexity_worsens_together(rungs) -> None:
    """The finding, stated as the thing that would falsify it.

    The numeric values move the same way, but their utility does not: a larger
    lexicon covers more of the held-out stream and is a harder prediction
    problem. Reporting either number alone would hide that trade.
    """
    coverage = [r["coverage_pct"] for r in rungs]
    perplexity = [r["perplexity"] for r in rungs]
    assert coverage == sorted(coverage), f"coverage is not monotone in lexicon size: {coverage}"
    assert perplexity == sorted(perplexity), (
        f"perplexity is not monotone in lexicon size: {perplexity}"
    )
    assert coverage[-1] > coverage[0] and perplexity[-1] > perplexity[0], (
        "coverage and numeric perplexity did not both rise; the recorded trade has changed"
    )


def test_shared_denominator_gain_widens_across_the_ladder(rungs) -> None:
    """One denominator must turn added covered positions into a wider gain."""
    shared = [float(rung["gain_nats_shared_denominator"]) for rung in rungs]
    assert all(current > previous for previous, current in pairwise(shared)), (
        f"shared-denominator gain did not widen strictly: {shared}"
    )


def test_the_bar_is_beaten_at_every_rung(rungs) -> None:
    """The gate must hold across the whole ladder, not at one operating point.

    A level that clears its comparator only at the shipped vocabulary size has a
    result about that size. The claim in the manuscript is about the model.
    """
    for rung in rungs:
        assert rung["bigram_baseline"] > rung["perplexity"], (
            f"at {rung['vocab_size']} types the letter-bigram bar "
            f"({rung['bigram_baseline']}) does not exceed the model "
            f"({rung['perplexity']}), so the gate does not hold at that scale"
        )


def test_every_rung_scores_only_tokens_its_lexicon_covers(rungs) -> None:
    """Positions must grow with coverage, or evaluation drifted off the rule.

    The model assigns no mass outside its vocabulary, so a rung scoring more
    positions than its lexicon covers would be scoring a fallback. Positions
    rising in step with coverage is the observable form of that rule holding.
    """
    positions = [r["positions"] for r in rungs]
    assert positions == sorted(positions)
    coverage = [r["coverage_pct"] for r in rungs]
    # ``pairwise`` rather than zipping a sequence against its own one-step
    # offset: those are never the same length, so the strict form raises on
    # every input and the check would never run. The repository forbids that
    # idiom by name, and the guard caught this line when it was written the
    # other way — which is the guard doing exactly its job.
    assert all(
        (p2 > p1) == (c2 > c1)
        for (p1, p2), (c1, c2) in zip(pairwise(positions), pairwise(coverage), strict=True)
    ), "positions and coverage disagreed on direction; the restriction rule drifted"


def test_coverage_is_measured_against_the_original_held_out_count(rungs) -> None:
    """Each rung's percentage must share one denominator.

    Recomputing coverage against a rung's own surviving tokens would make every
    rung read 100% — a sweep of perfect coverage at every size, which is the
    most confidently wrong version of this figure.
    """
    assert rungs[0]["coverage_pct"] < 100.0
    assert all(r["coverage_pct"] < 100.0 for r in rungs), (
        "a rung reported full coverage, which means the denominator became its "
        "own survivors rather than the held-out stream"
    )


def test_rungs_at_or_above_the_full_size_collapse_rather_than_duplicate() -> None:
    """Asking for more types than exist must not emit a second identical rung."""
    full = load_word_artifact(FULL_ARTIFACT)
    n = len(full.vocab)
    swept = scaling_rungs(full, sizes=(n, n + 1000, n * 2))
    assert [r["vocab_size"] for r in swept] == [n]


def test_a_non_positive_rung_raises_rather_than_reporting_an_empty_lexicon() -> None:
    """Zero types has no lexicon; a coverage of zero beside a perplexity over
    nothing is two numbers that cannot both be about the same thing."""
    full = load_word_artifact(FULL_ARTIFACT)
    with pytest.raises(ValueError, match="positive vocabulary sizes"):
        scaling_rungs(full, sizes=(0, 500))


def test_each_rung_is_validated_by_the_real_artifact_contract() -> None:
    """A truncated vocabulary is exactly how an out-of-vocabulary eval token
    gets in, so the rung is rebuilt through the validator rather than patched
    around it. This is the positive control: skipping the restriction must fail.
    """
    raw = json.loads(FULL_ARTIFACT.read_text())
    ranked = sorted(raw["word_freq"].items(), key=lambda item: (-item[1], item[0]))
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        WordArtifact({**raw, "word_freq": dict(ranked[:500])})


def test_the_declared_rung_ladder_is_nested_and_ordered() -> None:
    """Frequency-ranked prefixes, so each rung contains the one below it.

    Unrelated lexicons of increasing size would also produce a rising curve, and
    the curve would mean nothing: a difference between two rungs would be a
    difference between two vocabularies rather than an effect of size.
    """
    assert list(SCALING_RUNGS) == sorted(SCALING_RUNGS)
    assert len(set(SCALING_RUNGS)) == len(SCALING_RUNGS)
    assert SCALING_RUNGS[0] > 0
