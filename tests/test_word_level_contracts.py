"""Coverage for word-level paths the existing gates do not exercise.

Three families, each chosen because the uncovered lines carry real behavior:

1. ``word_perplexity`` and the oracle's closed-vocabulary refusal - the OOV
   raise is the closed-vocabulary *finding*; an untested refusal is an
   unenforced contract.
2. ``build_letter_bigram_bar`` orientation - column-stochasticity and the
   end-of-word symbol are the axis mistakes Stage 2 taught this repo to check.
3. ``scaling_rungs`` over the real artifact - the gain arithmetic and the
   collapse-above-full-size rule, exercised end to end.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ailm.levels.word import (
    DEFAULT_LEVEL_CONFIG,
    LevelConfig,
    WordArtifact,
    build_letter_bigram_bar,
    build_word_prior,
    load_word_artifact,
    scaling_rungs,
    word_perplexity,
)

FULL_ARTIFACT = Path(__file__).resolve().parents[1] / "vendor" / "brown_word_counts_full.json"


@pytest.fixture(scope="module")
def artifact() -> WordArtifact:
    return load_word_artifact()


# -- word_perplexity: closed-vocabulary contract -----------------------------


def test_word_perplexity_rejects_an_out_of_vocabulary_token(artifact) -> None:
    """The closed-vocabulary bound is the finding; a smoothed fallback would
    report a number for a model that is undefined on that token."""
    oov = "qqzzqqx"  # not a plausible Brown type
    assert oov not in artifact.word_to_index
    with pytest.raises(ValueError, match="out of vocabulary"):
        word_perplexity(artifact, tokens=[oov])


def test_word_perplexity_rejects_an_empty_stream(artifact) -> None:
    with pytest.raises(ValueError, match="tokens is empty"):
        word_perplexity(artifact, tokens=[])


def test_word_perplexity_per_emission_denominator_counts_end_of_word(artifact) -> None:
    """A token of length L contributes L+1 emissions; the count must show it."""
    tokens = [t for t in artifact.eval_tokens if len(t) == 3][:8]
    assert tokens, "artifact has no 3-letter held-out tokens"
    _ppl, n_emissions = word_perplexity(artifact, tokens=tokens)
    assert n_emissions == len(tokens) * 4  # 3 letters + 1 end-of-word each


def test_word_perplexity_equals_prior_mass_identity(artifact) -> None:
    """Per-emission perplexity is the geometric mean of the level's own prior
    mass over the emission path. Rebuild the prior the level builds and check
    the arithmetic, rather than assuming the raw counts are the prior."""
    tokens = artifact.eval_tokens[:12]
    ppl, n_emissions = word_perplexity(artifact, tokens=tokens)

    prior = build_word_prior(artifact, config=DEFAULT_LEVEL_CONFIG)
    index = artifact.word_to_index
    total_logp = 0.0
    counted = 0
    for token in tokens:
        mass = float(prior[index[token]])
        total_logp += math.log(mass)
        counted += len(token) + 1
    assert counted == n_emissions
    expected = math.exp(-total_logp / counted)
    assert abs(ppl - expected) / expected < 1e-6, (
        "word_perplexity is not the geometric-mean prior mass over emissions"
    )


def test_word_perplexity_is_monotone_in_prior_mass(artifact) -> None:
    """A prior that boosts the held-out tokens' mass must lower their
    per-emission perplexity - the direction the negative controls rely on.
    Built by direct array override (the documented negative-control path) so
    the artifact's ordering contract is not disturbed."""
    tokens = artifact.eval_tokens[:16]
    base_ppl, _ = word_perplexity(artifact, tokens=tokens)

    prior = build_word_prior(artifact, config=DEFAULT_LEVEL_CONFIG)
    boosted = np.asarray(prior, dtype=np.float64).copy()
    index = artifact.word_to_index
    for token in tokens:
        boosted[index[token]] = boosted[index[token]] * 2.0 + 1e-12
    boosted /= boosted.sum()
    boosted_ppl, _ = word_perplexity(artifact, word_prior=boosted, tokens=tokens)
    assert boosted_ppl < base_ppl, (
        "boosting the held-out tokens' prior mass did not lower their perplexity"
    )


# -- build_letter_bigram_bar: orientation and smoothing -----------------------


def test_letter_bigram_bar_is_column_stochastic(artifact) -> None:
    transition, start = build_letter_bigram_bar(artifact)
    np.testing.assert_allclose(transition.sum(axis=0), 1.0, atol=1e-9)
    assert abs(float(start.sum()) - 1.0) < 1e-9


def test_letter_bigram_bar_shape_and_no_dead_columns(artifact) -> None:
    transition, start = build_letter_bigram_bar(artifact)
    assert transition.shape == (27, 27)
    assert start.shape == (27,)
    assert float(transition.min()) > 0.0, "add-k smoothing left a dead column"


def test_letter_bigram_bar_changes_with_the_smoothing_pseudocount(artifact) -> None:
    k1_transition, _ = build_letter_bigram_bar(artifact, config=DEFAULT_LEVEL_CONFIG)
    k9_transition, _ = build_letter_bigram_bar(artifact, config=LevelConfig(smoothing=9.0))
    assert not np.allclose(k1_transition, k9_transition), (
        "the pseudocount has no effect on the bar; smoothing is decorative"
    )


# -- scaling_rungs: gain arithmetic over the real artifact --------------------


def test_scaling_rungs_gain_arithmetic_matches_its_definition(artifact) -> None:
    """gain_nats = log(baseline) - log(perplexity), per emission."""
    rungs = scaling_rungs(artifact, sizes=(500, 1000))
    for rung in rungs:
        expected = math.log(float(rung["bigram_baseline"])) - math.log(float(rung["perplexity"]))
        assert abs(rung["gain_nats"] - round(expected, 4)) < 1e-9


def test_scaling_rungs_shared_denominator_normalizes_to_the_top_rung(
    artifact,
) -> None:
    rungs = scaling_rungs(artifact, sizes=(500, 1000, 2000))
    top_positions = rungs[-1]["positions"]
    for rung in rungs:
        expected = rung["gain_nats"] * rung["positions"] / top_positions
        assert abs(rung["gain_nats_shared_denominator"] - round(expected, 4)) < 2e-3, (
            "shared-denominator gain is not gain scaled to the top rung"
        )


def test_scaling_rungs_coverage_is_monotone_in_vocabulary_size(artifact) -> None:
    rungs = scaling_rungs(artifact, sizes=(500, 1000, 2000))
    coverages = [float(rung["coverage_pct"]) for rung in rungs]
    assert coverages == sorted(coverages), (
        "a frequency-ranked nested sweep must have monotone coverage"
    )


# -- clause_channel: uncovered guards -----------------------------------------


def test_build_clause_message_rejects_nonpositive_smoothing() -> None:
    """Zero smoothing would emit zero-probability slots for unseen words —
    the closed-vocabulary refusal in another costume."""
    from ailm.levels.clause_channel import build_clause_message

    sentences = [
        {"type": "declarative", "words": ["the", "cat"], "document_index": 0, "sentence_index": 0}
    ]
    word_to_index = {"the": 0, "cat": 1}
    with pytest.raises(ValueError, match="smoothing must be positive"):
        build_clause_message(sentences, word_to_index, smoothing=0.0)


def test_build_clause_message_is_a_distribution_per_type() -> None:
    from ailm.levels.clause_channel import build_clause_message

    word_to_index = {"the": 0, "cat": 1, "dog": 2}
    sentences = [
        {"type": "declarative", "words": ["the", "cat"], "document_index": 0, "sentence_index": 0},
        {
            "type": "interrogative",
            "words": ["the", "dog"],
            "document_index": 0,
            "sentence_index": 1,
        },
    ]
    message = build_clause_message(sentences, word_to_index)
    for row in (message.lookup("declarative"), message.lookup("interrogative")):
        assert abs(float(row.sum()) - 1.0) < 1e-9
        assert float(row.min()) > 0.0, "unsmoothed message left a dead slot"


def test_mix_prior_at_the_endpoints_is_exact() -> None:
    """alpha=0 must be bit-identical to the prior (the identity anchor) and
    alpha=1 exactly the message — the sweep's anchor checks depend on both."""
    from ailm.levels.clause_channel import mix_prior

    rng = np.random.default_rng(7)
    prior = rng.dirichlet(np.ones(6))
    message = rng.dirichlet(np.ones(6))
    np.testing.assert_array_equal(mix_prior(prior, message, 0.0), prior)
    np.testing.assert_array_equal(mix_prior(prior, message, 1.0), message)
    mixed = mix_prior(prior, message, 0.37)
    np.testing.assert_allclose(mixed, 0.63 * prior + 0.37 * message, atol=1e-12)


def test_mix_prior_rejects_an_out_of_range_alpha() -> None:
    from ailm.levels.clause_channel import mix_prior

    prior = np.ones(4) / 4
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        mix_prior(prior, prior, -0.1)
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        mix_prior(prior, prior, 1.1)
