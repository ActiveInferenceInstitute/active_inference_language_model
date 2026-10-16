"""Punctuation-type oracle -> L3 intervention — tests of decision entry 27.

What this file is for
---------------------
``docs/decisions.md`` entry 27 pre-registered what it called the **clause ->
word** channel as
a falsification test of the first clause of the novel-information criterion: a
message helps iff it is novel to the receiver and the receiver's likelihood is
sensitive to it. Decision 89 corrects the implemented sender: held-out
punctuation supplies a privileged reference type; no L5 posterior is consumed.
The pre-registered prediction was that mixing the resulting type-conditioned
word statistic into the word prior ``D[0]`` would **harm** the held-out
predictive with a monotone decline (no interior optimum).

The corrected document-disjoint measurement (recorded by
``scripts/record_clause_channel_claims.py``) does **not** confirm harm: every
positive grid point improves the declared perplexity, the selected point is the
high endpoint at ``alpha = 0.9``, the per-token direction agrees, and the
scrambled control is far worse. These tests keep both the result and the
controls that make it evidence, and they re-derive every recorded number from
the same paired stream so a drift is caught.

Claims discipline
-----------------
This file **verifies**; ``scripts/record_clause_channel_claims.py`` writes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import binom

from ailm.claims import load_ledger
from ailm.levels.clause_channel import (
    CLAUSE_TYPES,
    ClauseMessage,
    ClauseWordArtifact,
    build_clause_message,
    clustered_clause_word_summaries,
    load_clause_word_artifact,
    mix_prior,
    sweep_clause_alpha,
)
from ailm.levels.word import build_word_prior, load_word_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIDGE = PROJECT_ROOT / "vendor" / "brown_clause_word_counts.json"
WORD_ARTIFACT = PROJECT_ROOT / "vendor" / "brown_word_counts.json"
GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
BEST_ALPHA = 0.9

pytestmark = pytest.mark.skipif(
    not BRIDGE.exists() or not WORD_ARTIFACT.exists(),
    reason="vendored word / clause-word artifacts absent",
)


def _scramble(dist: np.ndarray) -> np.ndarray:
    """Reverse the word axis — the entropy-preserving control, inlined."""
    return dist[::-1].copy()


@pytest.fixture(scope="module")
def model():
    bridge = load_clause_word_artifact(BRIDGE, word_path=WORD_ARTIFACT)
    word_artifact = load_word_artifact(WORD_ARTIFACT)
    prior = build_word_prior(word_artifact)
    message = build_clause_message(bridge.train, word_artifact.word_to_index)
    return {
        "bridge": bridge,
        "train": bridge.train,
        "test": bridge.test,
        "prior": prior,
        "message": message,
        "word_to_index": word_artifact.word_to_index,
    }


def _tokens(test) -> list[str]:
    return [w for s in test for w in s["words"]]


def _ppl(log_probs: list[float], n_emissions: int) -> float:
    return float(np.exp(-sum(log_probs) / n_emissions))


def _log10_p(improved: int, degraded: int) -> float:
    discordant = improved + degraded
    if not discordant:
        return 0.0
    tail = min(improved, degraded)
    log_lower = float(logsumexp(binom.logpmf(np.arange(tail + 1), discordant, 0.5)))
    return min(0.0, (np.log(2.0) + log_lower) / np.log(10.0))


@pytest.fixture(scope="module")
def sweep(model):
    return sweep_clause_alpha(
        model["test"], model["prior"], model["message"], model["word_to_index"], GRID
    )


@pytest.fixture(scope="module")
def clustered(sweep, model):
    return clustered_clause_word_summaries(model["test"], sweep[BEST_ALPHA], sweep[0.0])


# ══ THE RESULT ═══════════════════════════════════════════════════════════════


def test_harm_was_not_confirmed(sweep, model):
    """The pre-registered prediction (monotone harm) did not reproduce.

    Entry 27 predicted harm with a monotone decline. The measured outcome on
    the declared perplexity axis is a small improvement with an interior
    improvement, i.e. harm is NOT confirmed. This is the honest recorded
    finding; the grid does not identify an interior optimum.
    """
    n_emissions = sum(len(w) + 1 for w in _tokens(model["test"]))
    anchor = _ppl(sweep[0.0], n_emissions)
    best = min((_ppl(sweep[a], n_emissions) for a in GRID), key=lambda p: p)
    assert best < anchor, (
        "The recorded finding is that the clause->word message does NOT "
        "confirm the predicted harm. If the measure has drifted so the best "
        "weight no longer improves the predictive, the outcome entry in "
        "docs/decisions.md entry 27 and the manuscript must be updated to "
        "match the new measurement — not the other way around."
    )


def test_selected_point_is_the_high_grid_endpoint(sweep, model):
    """The corrected population does not support an interior-optimum claim."""
    n_emissions = sum(len(w) + 1 for w in _tokens(model["test"]))
    ppls = {a: _ppl(sweep[a], n_emissions) for a in GRID}
    best_alpha = min(ppls, key=lambda a: ppls[a])
    assert best_alpha == BEST_ALPHA == max(GRID)
    assert all(ppls[right] < ppls[left] for left, right in zip(GRID[:-1], GRID[1:], strict=True))


def test_corrected_token_direction_agrees_with_the_predictive(sweep):
    """The corrected result is concordant help, not the retired mixed sign."""
    improved = sum(
        1 for value, anchor in zip(sweep[BEST_ALPHA], sweep[0.0], strict=True) if value > anchor
    )
    degraded = sum(
        1 for value, anchor in zip(sweep[BEST_ALPHA], sweep[0.0], strict=True) if value < anchor
    )
    assert improved > degraded
    assert improved + degraded == len(sweep[BEST_ALPHA])


# ══ CONTROLS ═════════════════════════════════════════════════════════════════


def test_the_anchor_is_exact(model, sweep):
    """At alpha=0 the mixed prior is bit-identical to the uncoupled prior."""
    expected = [
        float(np.log(max(model["prior"][model["word_to_index"][word]], 1e-300)))
        for word in _tokens(model["test"])
    ]
    assert sweep[0.0] == expected
    again = sweep_clause_alpha(
        model["test"], model["prior"], model["message"], model["word_to_index"], GRID
    )
    assert again[0.0] == sweep[0.0]


def test_the_anchor_is_not_vacuous(sweep):
    """A positive alpha must change the output, or the anchor proves nothing."""
    assert sweep[BEST_ALPHA] != sweep[0.0]


def test_the_scrambled_message_does_worse(sweep, model):
    """The entropy-preserving control.

    Reversing the word axis keeps the message's entropy, peakedness and
    normalization while destroying its correspondence to the vocabulary — so
    a worse shaken score proves the *content* of the message (which words are
    linked to which sentence type) is doing the work, not merely that extra
    mass was added to the prior.
    """
    scrambled = sweep_clause_alpha(
        model["test"],
        model["prior"],
        model["message"],
        model["word_to_index"],
        GRID,
        scrambled=True,
    )
    n_emissions = sum(len(w) + 1 for w in _tokens(model["test"]))
    for a in (al for al in GRID if al > 0.0):
        assert _ppl(scrambled[a], n_emissions) >= _ppl(sweep[a], n_emissions), (
            f"At alpha={a} the reversed-axis control scores better than the "
            f"real message, so the channel's content is doing no work."
        )


def test_the_message_is_type_conditioned(model):
    """NON-VACUITY: the message must differ from the marginal prior."""
    differe = [
        float(np.max(np.abs(model["message"].by_type[t] - model["prior"])))
        for t in model["message"].by_type
    ]
    assert max(differe) > 0.0, (
        "The clause word message is identical to the marginal word prior, so "
        "it carries no clause structure and the sweep has nothing to measure."
    )


def test_scrambling_preserves_entropy(model):
    """The control must differ from the real message ONLY in meaning."""
    for t in model["message"].by_type:
        dist = model["message"].by_type[t]
        flipped = _scramble(dist)
        np.testing.assert_allclose(np.sort(dist), np.sort(flipped))
        assert not np.allclose(dist, flipped), "Reversal left the message unchanged"


def test_typefree_and_swapped_label_controls_are_worse(model):
    """The signal is type conditioning, not merely correct vocabulary wiring."""
    typefree_message = ClauseMessage(
        {sentence_type: model["message"].fallback.copy() for sentence_type in CLAUSE_TYPES},
        list(CLAUSE_TYPES),
        model["message"].fallback.copy(),
        model["message"].n_vocab,
    )
    swapped_message = ClauseMessage(
        {
            CLAUSE_TYPES[0]: model["message"].by_type[CLAUSE_TYPES[1]].copy(),
            CLAUSE_TYPES[1]: model["message"].by_type[CLAUSE_TYPES[0]].copy(),
        },
        list(CLAUSE_TYPES),
        model["message"].fallback.copy(),
        model["message"].n_vocab,
    )
    real = sweep_clause_alpha(
        model["test"], model["prior"], model["message"], model["word_to_index"], GRID
    )
    typefree = sweep_clause_alpha(
        model["test"], model["prior"], typefree_message, model["word_to_index"], GRID
    )
    swapped = sweep_clause_alpha(
        model["test"], model["prior"], swapped_message, model["word_to_index"], GRID
    )
    emissions = sum(len(word) + 1 for word in _tokens(model["test"]))
    assert _ppl(real[BEST_ALPHA], emissions) < _ppl(typefree[BEST_ALPHA], emissions)
    assert _ppl(real[BEST_ALPHA], emissions) < _ppl(swapped[BEST_ALPHA], emissions)


def test_document_clustering_control_detects_pseudoreplication():
    """Repeating within-document signs cannot create independent replication."""
    from ailm.clustering import summarize_clustered_difference

    differences = [1.0] * 80 + [-1.0] * 40 + [1.0] * 30 + [-1.0] * 70
    documents = ["a"] * 80 + ["b"] * 40 + ["c"] * 30 + ["d"] * 70
    clustered = summarize_clustered_difference(differences, documents, resamples=2_000, seed=17)
    naive = summarize_clustered_difference(differences, None, resamples=2_000, seed=17)
    assert clustered.n_clusters == 4
    assert naive.n_clusters == len(differences)
    assert clustered.interval_width > naive.interval_width


# ══ THE SWEEP REFUSES AN UNSUPPORTABLE CLAIM ═══════════════════════════════════


def test_sweep_requires_an_anchor_and_a_real_grid(model):
    args = (model["test"], model["prior"], model["message"], model["word_to_index"])
    with pytest.raises(ValueError, match="collapsed sweep"):
        sweep_clause_alpha(*args, (0.1, 0.2, 0.3))  # no anchor
    with pytest.raises(ValueError, match="collapsed sweep"):
        sweep_clause_alpha(*args, (0.0, 0.3))  # too few positive alphas


def test_mix_prior_alpha_outside_unit_interval_raises(model):
    with pytest.raises(ValueError, match="alpha must lie"):
        mix_prior(model["prior"], model["prior"], 1.5)


def test_mix_prior_shape_mismatch_raises(model):
    with pytest.raises(ValueError, match="must share shape"):
        mix_prior(model["prior"], model["prior"][:-1], 0.5)


# ══ CLAIMS LEDGER — verify, never write ════════════════════════════════════════


def test_clause_word_claims_verified(sweep, model, clustered):
    """Re-derive the recorded claims from the same paired stream."""
    ledger = load_ledger(PROJECT_ROOT)
    n_emissions = sum(len(w) + 1 for w in _tokens(model["test"]))
    anchor = _ppl(sweep[0.0], n_emissions)
    ppls = {a: _ppl(sweep[a], n_emissions) for a in GRID}
    best_alpha = min(ppls, key=lambda a: ppls[a])
    best_improved = sum(
        1 for value, base in zip(sweep[best_alpha], sweep[0.0], strict=True) if value > base
    )
    best_degraded = sum(
        1 for value, base in zip(sweep[best_alpha], sweep[0.0], strict=True) if value < base
    )
    token_clustered, loggain_clustered = clustered
    typefree_message = ClauseMessage(
        {sentence_type: model["message"].fallback.copy() for sentence_type in CLAUSE_TYPES},
        list(CLAUSE_TYPES),
        model["message"].fallback.copy(),
        model["message"].n_vocab,
    )
    swapped_message = ClauseMessage(
        {
            CLAUSE_TYPES[0]: model["message"].by_type[CLAUSE_TYPES[1]].copy(),
            CLAUSE_TYPES[1]: model["message"].by_type[CLAUSE_TYPES[0]].copy(),
        },
        list(CLAUSE_TYPES),
        model["message"].fallback.copy(),
        model["message"].n_vocab,
    )
    typefree = sweep_clause_alpha(
        model["test"], model["prior"], typefree_message, model["word_to_index"], GRID
    )
    swapped_type = sweep_clause_alpha(
        model["test"], model["prior"], swapped_message, model["word_to_index"], GRID
    )

    expected: dict[str, object] = {
        "clause_word_source_documents": 500,
        "clause_word_train_documents": 448,
        "clause_word_test_documents": 52,
        "clause_word_document_overlap": 0,
        "clause_word_source_categories": 15,
        "clause_word_train_categories": 15,
        "clause_word_test_categories": 15,
        "clause_word_anchor_ppl": round(anchor, 6),
        "clause_word_best_alpha": best_alpha,
        "clause_word_best_ppl": round(ppls[best_alpha], 6),
        "clause_word_best_delta_ppl": round(ppls[best_alpha] - anchor, 6),
        "clause_word_best_pair_log10_p": round(_log10_p(best_improved, best_degraded), 6),
        "clause_word_optimum_is_interior": int(0.0 < best_alpha < max(GRID)),
        "clause_word_confirms_harm": int(ppls[best_alpha] >= anchor),
        "clause_word_typefree_ppl_at_best": round(_ppl(typefree[best_alpha], n_emissions), 6),
        "clause_word_swapped_type_ppl_at_best": round(
            _ppl(swapped_type[best_alpha], n_emissions), 6
        ),
        "clause_word_real_beats_typefree": int(
            ppls[best_alpha] < _ppl(typefree[best_alpha], n_emissions)
        ),
        "clause_word_real_beats_swapped_type": int(
            ppls[best_alpha] < _ppl(swapped_type[best_alpha], n_emissions)
        ),
        "clause_word_cluster_documents": token_clustered.n_clusters,
        "clause_word_cluster_token_largest_document": token_clustered.largest_cluster,
        "clause_word_cluster_token_positive_documents": token_clustered.positive_clusters,
        "clause_word_cluster_token_negative_documents": token_clustered.negative_clusters,
        "clause_word_cluster_token_net_share": round(token_clustered.effect, 6),
        "clause_word_cluster_token_ci_low": round(token_clustered.ci_low, 6),
        "clause_word_cluster_token_ci_high": round(token_clustered.ci_high, 6),
        "clause_word_cluster_loggain_largest_document": loggain_clustered.largest_cluster,
        "clause_word_cluster_loggain_positive_documents": loggain_clustered.positive_clusters,
        "clause_word_cluster_loggain_negative_documents": loggain_clustered.negative_clusters,
        "clause_word_cluster_loggain_nats_per_emission": round(loggain_clustered.effect, 8),
        "clause_word_cluster_loggain_ci_low": round(loggain_clustered.ci_low, 8),
        "clause_word_cluster_loggain_ci_high": round(loggain_clustered.ci_high, 8),
        "clause_word_cluster_loggain_p": float(f"{loggain_clustered.p_value:.4g}"),
        "clause_word_cluster_loggain_p_floor": float(f"{loggain_clustered.p_floor:.4g}"),
    }
    # Per-alpha rows, re-derived.
    for swept_alpha in GRID:
        _label = str(swept_alpha).replace(".", "_")
        expected[f"clause_word_sweep_alpha_{_label}_ppl"] = round(ppls[swept_alpha], 6)
        expected[f"clause_word_sweep_alpha_{_label}_delta"] = round(ppls[swept_alpha] - anchor, 6)
        imp = sum(1 for x, a in zip(sweep[swept_alpha], sweep[0.0], strict=True) if x > a)
        deg = sum(1 for x, a in zip(sweep[swept_alpha], sweep[0.0], strict=True) if x < a)
        expected[f"clause_word_sweep_alpha_{_label}_improved"] = imp
        expected[f"clause_word_sweep_alpha_{_label}_degraded"] = deg
        expected[f"clause_word_sweep_alpha_{_label}_pair_log10_p"] = round(_log10_p(imp, deg), 6)

    for label, value in expected.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent. Run scripts/record_clause_channel_claims.py — "
            f"this test may not write it."
        )
        assert entry["source"].startswith("scripts/"), (
            f"{label} was last written by {entry['source']!r}."
        )
        assert entry["value"] == pytest.approx(value, rel=1e-3), (
            f"Ledger {label}={entry['value']!r} != measurement {value!r}."
        )


# ══ VALIDATION / ERROR-PATH BRANCHES ════════════════════════════════════════════


def test_build_clause_message_rejects_empty_train():
    with pytest.raises(ValueError, match="train_sentences is empty"):
        build_clause_message([], {})


# == BRIDGE ARTIFACT VALIDATION ===============================================


@pytest.fixture(scope="module")
def bridge_payload():
    return json.loads(BRIDGE.read_text(encoding="utf-8"))


def _validate_payload(payload: dict) -> ClauseWordArtifact:
    word_artifact = load_word_artifact(WORD_ARTIFACT)
    return ClauseWordArtifact(
        payload,
        expected_vocab=word_artifact.vocab,
        expected_vocab_sha256=hashlib.sha256(WORD_ARTIFACT.read_bytes()).hexdigest(),
        expected_document_ids=word_artifact.document_ids,
        expected_categories=word_artifact.document_categories,
        expected_train_indices=word_artifact.train_document_indices,
        expected_heldout_indices=word_artifact.heldout_document_indices,
    )


def test_bridge_is_the_exact_stage3_document_partition(model):
    bridge = model["bridge"]
    assert len(bridge.document_ids) == 500
    assert len(bridge.train_document_indices) == 448
    assert len(bridge.heldout_document_indices) == 52
    assert not set(bridge.train_document_indices) & set(bridge.heldout_document_indices)
    assert {bridge.document_categories[index] for index in bridge.train_document_indices} == set(
        bridge.document_categories
    )
    assert {bridge.document_categories[index] for index in bridge.heldout_document_indices} == set(
        bridge.document_categories
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["provenance"].__setitem__(
                "target_train_document_fraction", 0.8
            ),
            "target_train_document_fraction",
        ),
        (
            lambda payload: payload["provenance"].__setitem__("corpus", "mutable Brown"),
            "provenance constants",
        ),
        (
            lambda payload: payload.__setitem__("unexpected", 1),
            "top-level keys",
        ),
        (
            lambda payload: payload["provenance"].__setitem__("vocab_source_sha256", "0" * 64),
            "vocabulary source digest",
        ),
        (
            lambda payload: payload["vocab"].__setitem__(0, payload["vocab"][1]),
            "vocab",
        ),
        (
            lambda payload: payload["splits"]["train"][0]["words"].__setitem__(
                0, "not_in_the_stage3_vocabulary"
            ),
            "outside Stage 3 vocab",
        ),
        (
            lambda payload: payload["documents"]["typed_sentence_counts"].__setitem__(
                0, payload["documents"]["typed_sentence_counts"][0] + 1
            ),
            "per-document counts",
        ),
        (
            lambda payload: payload["documents"]["in_vocab_word_counts"].__setitem__(0, True),
            "non-negative integer",
        ),
    ],
    ids=(
        "target-fraction",
        "corpus-name",
        "extra-top-key",
        "word-artifact-digest",
        "vocab-permutation",
        "oov-row",
        "per-document-count",
        "boolean-count",
    ),
)
def test_bridge_rejects_planted_schema_and_population_mutations(bridge_payload, mutate, match):
    planted = copy.deepcopy(bridge_payload)
    mutate(planted)
    with pytest.raises(ValueError, match=match):
        _validate_payload(planted)


def test_bridge_rejects_sentence_moved_across_document_arms(bridge_payload):
    planted = copy.deepcopy(bridge_payload)
    planted["splits"]["train"][0]["document_index"] = planted["documents"]["split_indices"][
        "heldout"
    ][0]
    with pytest.raises(ValueError, match="outside its document arm"):
        _validate_payload(planted)


def test_bridge_rejects_duplicate_sentence_coordinates(bridge_payload):
    planted = copy.deepcopy(bridge_payload)
    first = planted["splits"]["train"][0]
    second = planted["splits"]["train"][1]
    second["document_index"] = first["document_index"]
    second["sentence_index"] = first["sentence_index"]
    with pytest.raises(ValueError, match="unique and ordered"):
        _validate_payload(planted)


def test_bridge_rejects_coherent_but_noncanonical_split_relabel(bridge_payload):
    planted = copy.deepcopy(bridge_payload)
    train = planted["documents"]["split_indices"]["train"]
    heldout = planted["documents"]["split_indices"]["heldout"]
    train[0], heldout[0] = heldout[0], train[0]
    train.sort()
    heldout.sort()
    with pytest.raises(ValueError, match="Stage 3 receiver"):
        _validate_payload(planted)
