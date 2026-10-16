"""The case-only clause arm: a non-passing gate with unresolved privileged-label gain.

Decision 114 made the widened-observation clause model the canonical Level 5 and
this arm the retained ablation. Nothing in this file changed with it, which is
the point: the promotion is a naming decision, and the guards that make this
arm's record honest have to keep holding or the ablation stops being evidence of
anything. The arm's audit lives in ``scripts/audit_gates.py::audit_stage5_case_only``
and its widened counterpart is covered by ``tests/test_stage5_widened.py``.

The result (decisions 24 and 83): on the document-disjoint UD EWT split at
bigram order, the fitted privileged-label reference has a negative point gain
whose document-clustered interval spans zero. This file
enforces everything that makes the non-passing result interpretable:

* the declared gate ("free energy tracks clause length") is VACUOUS, shown
  executably — a uniform model's cumulative free energy also grows with length;
* the point result and document-clustered oracle-gain interval, re-measured fresh
  and bound to the ledger;
* the **detectability control** — the identical harness on a synthetic corpus
  with genuinely different types must fire decisively (failure-to-reject needs
  a positive control);
* the negative controls — swapped type blocks and cross-type leakage, injected
  through the shipped builder, must degrade the synthetic result;
* decoder equivalence — closed-form mixture, oracle filter, and the live pymdp
  agent must agree on the evidence of every test sentence;
* the structural disjointness inherited from decision 16 — the observation
  alphabet is exactly the eight case roles, and the artifact records that the
  label source (punctuation) cannot appear in it;
* GNN conformance for ``L5_clause`` and rejection of its mutations.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ailm.cases import CaseRole
from ailm.claims import load_ledger
from ailm.levels.clause import (
    CLAUSE_ARTIFACT_SCHEMA,
    CLAUSE_TYPES,
    N_CASES,
    N_JOINT,
    STAGE5_BOOTSTRAP_CONFIDENCE,
    STAGE5_BOOTSTRAP_RESAMPLES,
    STAGE5_BOOTSTRAP_SEED,
    ClauseArtifact,
    binary_auc,
    build_clause_agent,
    build_joint_emission,
    build_joint_start,
    build_joint_transition,
    build_type_free_chain,
    build_type_models,
    classify_sentences,
    clustered_binary_auc_interval,
    load_clause_artifact,
    sequence_log_evidence,
    summarize_oracle_gain,
    synthetic_control_corpus,
    type_conditional_perplexities,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _retired_stage5_wording(source: str) -> list[str]:
    normalized = " ".join(source.lower().split())
    retired = (
        "recorded " + "bounded null",
        "almost no " + "information",
        "negligible " + "information",
        "negligible " + "gain",
    )
    return [phrase for phrase in retired if phrase in normalized]


def test_current_surfaces_do_not_reinstate_a_power_or_negligibility_claim() -> None:
    surfaces = (
        PROJECT_ROOT / "ISA.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "decisions.md",
        PROJECT_ROOT / "docs" / "manuscript" / "03_methods.md",
        PROJECT_ROOT / "docs" / "manuscript" / "04_results.md",
        Path(__file__),
    )
    for surface in surfaces:
        assert _retired_stage5_wording(surface.read_text(encoding="utf-8")) == [], surface

    identity = surfaces[0].read_text(encoding="utf-8")
    assert "Stage 5 is non-passing" in identity
    assert "unresolved privileged-label gain" in identity

    # NEGATIVE CONTROLS: each retired strength claim must fail the same scanner.
    planted = [
        "Stage 5 is a recorded " + "bounded null.",
        "Sentence type carries almost no " + "information.",
        "The model has negligible " + "information.",
        "The paired contrast shows negligible " + "gain.",
    ]
    assert all(_retired_stage5_wording(text) for text in planted)


@pytest.fixture(scope="module")
def artifact():
    return load_clause_artifact()


@pytest.fixture(scope="module")
def models(artifact):
    return build_type_models(artifact.train)


@pytest.fixture(scope="module")
def comparator(artifact):
    return build_type_free_chain(artifact.train)


@pytest.fixture(scope="module")
def synthetic():
    train, test = synthetic_control_corpus()
    return {"train": train, "test": test, "models": build_type_models(train)}


@pytest.fixture(scope="module")
def ledger():
    return load_ledger(PROJECT_ROOT)


# ── THE DECLARED GATE IS VACUOUS, executably ─────────────────────────────────


def test_the_declared_gate_passes_under_a_uniform_model(artifact):
    """ "Free energy tracks clause length" cannot fail: cumulative free energy
    is monotone in length under ANY model, because every step adds
    non-negative surprise. A uniform-random model — the trivial baseline that
    must fail a sound gate — satisfies it exactly, so the gate is VACUOUS.
    """
    uniform_step_surprise = float(np.log(N_CASES))
    lengths = sorted({len(seq) for _t, seq in artifact.test})
    cumulative = [length * uniform_step_surprise for length in lengths]
    assert all(b > a for a, b in pairwise(cumulative)), (
        "the uniform model's cumulative free energy failed to grow with "
        "length, which would be an arithmetic impossibility"
    )


# ── THE NON-PASSING RESULT, re-measured and bound to the ledger ──────────────


def test_the_privileged_label_reference_point_does_not_clear_the_bar(artifact, models, comparator):
    """The document-disjoint split's fitted-reference point is negative."""
    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    assert perplexities["type_oracle"] > perplexities["type_free"], (
        "the privileged-label reference now clears the type-free bar; "
        "the recorded point direction changed and decision 83 needs revision"
    )


def test_the_mixture_does_not_beat_the_type_free_bar(artifact, models, comparator):
    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    assert perplexities["mixture"] >= perplexities["type_free"], (
        "the mixture now beats the type-free bar — the recorded Stage 5 outcome changed"
    )


def _independent_oracle_gains(sentences, models, comparator):
    """Re-derive paired gains without calling the production scorer."""
    free_transition, free_start = comparator
    gains = []
    lengths = []
    for sentence_type, sequence in sentences:
        type_index = CLAUSE_TYPES.index(sentence_type)
        oracle = float(np.log(models["starts"][type_index, sequence[0]]))
        type_free = float(np.log(free_start[sequence[0]]))
        for previous, current in pairwise(sequence):
            oracle += float(np.log(models["transitions"][type_index, current, previous]))
            type_free += float(np.log(free_transition[current, previous]))
        gains.append(oracle - type_free)
        lengths.append(len(sequence))
    return np.asarray(gains, dtype=np.float64), np.asarray(lengths, dtype=np.float64)


def test_oracle_gain_bootstrap_matches_an_independent_fixed_seed_rederivation(
    artifact, models, comparator
):
    """The primary bound resamples documents, not sentences or positions."""
    summary = summarize_oracle_gain(
        artifact.test,
        models,
        comparator,
        clusters=artifact.test_sequence_document_indices,
    )
    gains, lengths = _independent_oracle_gains(artifact.test, models, comparator)

    document_keys = list(dict.fromkeys(artifact.test_sequence_document_indices))
    document_to_index = {key: index for index, key in enumerate(document_keys)}
    inverse = np.asarray(
        [document_to_index[key] for key in artifact.test_sequence_document_indices],
        dtype=np.int64,
    )
    document_gain_sums = np.bincount(inverse, weights=gains)
    document_sentence_counts = np.bincount(inverse)
    document_position_counts = np.bincount(inverse, weights=lengths)
    rng = np.random.default_rng(STAGE5_BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(document_keys),
        size=(STAGE5_BOOTSTRAP_RESAMPLES, len(document_keys)),
    )
    sampled_gains = document_gain_sums[indices].sum(axis=1)
    sentence_replicates = sampled_gains / document_sentence_counts[indices].sum(axis=1)
    position_replicates = sampled_gains / document_position_counts[indices].sum(axis=1)
    tail = (1.0 - STAGE5_BOOTSTRAP_CONFIDENCE) * 50.0
    expected_sentence = np.percentile(sentence_replicates, [tail, 100.0 - tail], method="linear")
    expected_position = np.percentile(position_replicates, [tail, 100.0 - tail], method="linear")

    assert summary.n_sentences == len(artifact.test)
    assert summary.n_documents == len(document_keys) == artifact.n_test_documents
    assert summary.n_positions == int(lengths.sum())
    assert summary.mean_nats_per_sentence == pytest.approx(float(gains.mean()), abs=1e-15)
    assert summary.pooled_nats_per_position == pytest.approx(
        float(gains.sum() / lengths.sum()), abs=2e-15
    )
    assert (summary.sentence_ci_low, summary.sentence_ci_high) == pytest.approx(
        expected_sentence, abs=1e-15
    )
    assert (summary.position_ci_low, summary.position_ci_high) == pytest.approx(
        expected_position, abs=1e-15
    )
    assert summary.confidence == STAGE5_BOOTSTRAP_CONFIDENCE
    assert summary.resamples == STAGE5_BOOTSTRAP_RESAMPLES
    assert summary.seed == STAGE5_BOOTSTRAP_SEED

    # Both supported readings remain unresolved on this split. This is the
    # load-bearing negative result: a future interval that moves off zero must
    # force decision 83 and every public status line to be revised.
    assert summary.sentence_ci_low < 0.0 < summary.sentence_ci_high
    assert summary.position_ci_low < 0.0 < summary.position_ci_high

    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    assert summary.pooled_nats_per_position == pytest.approx(
        np.log(perplexities["type_free"] / perplexities["type_oracle"]), abs=2e-15
    )


def test_oracle_gain_interval_detects_planted_positive_and_negative_effects(synthetic):
    """Controls: the same bound resolves strong gains and inverted labels."""
    comparator = build_type_free_chain(synthetic["train"])
    positive = summarize_oracle_gain(synthetic["test"], synthetic["models"], comparator)
    assert positive.sentence_ci_low > 0.0
    assert positive.position_ci_low > 0.0

    swapped = {
        "transitions": synthetic["models"]["transitions"][::-1].copy(),
        "starts": synthetic["models"]["starts"][::-1].copy(),
        "type_prior": synthetic["models"]["type_prior"][::-1].copy(),
    }
    negative = summarize_oracle_gain(synthetic["test"], swapped, comparator)
    assert negative.sentence_ci_high < 0.0
    assert negative.position_ci_high < 0.0


def test_sentence_resampling_can_false_resolve_repeated_document_evidence(
    artifact, models, comparator
):
    """NEGATIVE CONTROL: duplicated sentences are not independent evidence.

    The retired sentence bootstrap sees many repeats and excludes zero. Whole
    source documents move together under the corrected method, whose interval
    spans zero on the identical paired gains.
    """
    gains, _lengths = _independent_oracle_gains(artifact.test, models, comparator)
    positive_index = int(np.argmax(gains))
    negative_index = max(
        (index for index, gain in enumerate(gains) if gain < 0.0),
        key=lambda index: gains[index],
    )
    repeated = (
        [artifact.test[positive_index]] * 25
        + [artifact.test[positive_index]] * 25
        + [artifact.test[negative_index]] * 25
        + [artifact.test[negative_index]] * 25
    )
    document_ids = [0] * 25 + [1] * 25 + [2] * 25 + [3] * 25

    retired = summarize_oracle_gain(repeated, models, comparator)
    corrected = summarize_oracle_gain(repeated, models, comparator, clusters=document_ids)
    assert retired.sentence_ci_low > 0.0, (
        "negative-control sentence replication no longer produces a false resolution"
    )
    assert corrected.sentence_ci_low < 0.0 < corrected.sentence_ci_high


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"confidence": float("nan")},
        {"confidence": True},
        {"resamples": 0},
        {"resamples": -1},
        {"resamples": 2.5},
        {"resamples": True},
        {"seed": -1},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_oracle_gain_bootstrap_rejects_invalid_resampling_contracts(
    artifact, models, comparator, overrides
):
    with pytest.raises(ValueError):
        summarize_oracle_gain(artifact.test[:1], models, comparator, **overrides)


def test_oracle_gain_bootstrap_rejects_empty_or_invalid_paired_evidence(models, comparator):
    with pytest.raises(ValueError, match="at least two evaluation sentences"):
        summarize_oracle_gain([], models, comparator)
    with pytest.raises(ValueError, match="at least two evaluation sentences"):
        summarize_oracle_gain([("declarative", [0, 1])], models, comparator)
    with pytest.raises(ValueError, match="unknown sentence type"):
        summarize_oracle_gain(
            [("exclamative", [0, 1]), ("declarative", [0, 1])], models, comparator
        )
    with pytest.raises(ValueError, match="at least one case-role position"):
        summarize_oracle_gain([("declarative", []), ("interrogative", [0, 1])], models, comparator)
    with pytest.raises(ValueError, match="invalid case-role index"):
        summarize_oracle_gain(
            [("declarative", [-1, 0]), ("interrogative", [0, 1])], models, comparator
        )
    with pytest.raises(ValueError, match="numeric starts and transitions"):
        summarize_oracle_gain([("declarative", [0, 1]), ("interrogative", [1, 0])], {}, comparator)
    with pytest.raises(ValueError, match="models must provide starts and transitions"):
        summarize_oracle_gain(
            [("declarative", [0, 1]), ("interrogative", [1, 0])],
            {"starts": np.ones((1, N_CASES)), "transitions": np.ones((1, N_CASES, N_CASES))},
            comparator,
        )
    with pytest.raises(ValueError, match="type-free comparator must provide start"):
        summarize_oracle_gain(
            [("declarative", [0, 1]), ("interrogative", [1, 0])],
            models,
            (np.eye(N_CASES - 1), np.ones(N_CASES - 1)),
        )

    free_transition, free_start = comparator
    invalid_start = free_start.copy()
    invalid_start[0] = 0.0
    invalid_start[1] += free_start[0]
    with pytest.raises(ValueError, match="non-positive probability"):
        summarize_oracle_gain(
            [("declarative", [0, 1]), ("interrogative", [1, 0])],
            models,
            (free_transition, invalid_start),
        )


@pytest.mark.parametrize("mutation", ["scaled", "negative", "nan", "row-oriented"])
def test_oracle_gain_rejects_malformed_comparator_probability_simplexes(
    models, comparator, mutation
):
    transition, start = (array.copy() for array in comparator)
    if mutation == "scaled":
        transition *= 10.0
        start *= 10.0
    elif mutation == "negative":
        start[0] = -0.1
    elif mutation == "nan":
        transition[0, 0] = np.nan
    else:
        transition = np.zeros((N_CASES, N_CASES), dtype=np.float64)
        transition[:, 0] = np.arange(1, N_CASES + 1, dtype=np.float64)
        transition[:, 0] /= transition[:, 0].sum()
        transition[1:, 1:] = np.eye(N_CASES - 1)
        transition = transition.T
        assert np.allclose(transition.sum(axis=1), 1.0)
        assert not np.allclose(transition.sum(axis=0), 1.0)

    evidence = [("declarative", [0, 1]), ("interrogative", [1, 0])]
    with pytest.raises(ValueError, match="finite|non-negative|probability simplex"):
        summarize_oracle_gain(evidence, models, (transition, start))


def test_oracle_gain_validates_every_privileged_type_model(models, comparator):
    malformed = {
        "starts": models["starts"].copy(),
        "transitions": models["transitions"].copy(),
        "type_prior": models["type_prior"].copy(),
    }
    malformed["transitions"][1] *= 10.0
    # Only declaratives are scored: validation must still reject the malformed
    # interrogative chain rather than let unused invalid model state ship.
    evidence = [("declarative", [0, 1]), ("declarative", [1, 0])]
    with pytest.raises(ValueError, match="probability simplex"):
        summarize_oracle_gain(evidence, malformed, comparator)


def test_classification_does_not_beat_majority(artifact, models):
    classification = classify_sentences(artifact.test, models)
    assert classification["accuracy"] <= classification["majority_accuracy"]
    assert classification["recalls"]["interrogative"] == 0.0, (
        "the model now recovers held-out interrogatives; update decision 24"
    )
    # Rank signal exists but cannot overcome the class prior — both facts are
    # part of the finding.
    assert 0.5 < classification["auc"] < 0.9


def _ordinal_auc_negative_control(scores: list[float], labels: list[int]) -> float:
    """The retired implementation: tied rows receive arbitrary ordinal ranks."""
    # Stable sorting makes the defect deterministic: each tied row's rank is
    # assigned from arrival order, so reversing the same observations reverses
    # which class receives the favorable rank.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    binary_labels = np.asarray(labels, dtype=np.int64)
    n_pos = int(binary_labels.sum())
    n_neg = len(binary_labels) - n_pos
    return float((ranks[binary_labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def test_auc_gives_tied_positive_negative_pairs_half_credit():
    """A tied cross-class pair contributes one half, never arrival-order credit."""
    scores = [0.1, 0.5, 0.5, 0.9]
    labels = [0, 1, 0, 1]
    assert binary_auc(scores, labels) == pytest.approx(0.875)


def test_auc_is_invariant_to_permutation_and_reversal_with_ties():
    """NEGATIVE CONTROL: the retired ordinal rank changes under reversal."""
    scores = [0.1, 0.5, 0.5, 0.9]
    labels = [0, 1, 0, 1]
    reversed_scores = list(reversed(scores))
    reversed_labels = list(reversed(labels))

    ordinal = _ordinal_auc_negative_control(scores, labels)
    ordinal_reversed = _ordinal_auc_negative_control(reversed_scores, reversed_labels)
    assert ordinal != ordinal_reversed, (
        "the negative-control fixture no longer exposes ordinal tie breaking"
    )

    expected = binary_auc(scores, labels)
    assert binary_auc(reversed_scores, reversed_labels) == pytest.approx(expected)
    permutation = [2, 0, 3, 1]
    assert binary_auc(
        [scores[index] for index in permutation],
        [labels[index] for index in permutation],
    ) == pytest.approx(expected)


def test_auc_is_chance_when_every_score_is_tied():
    """An uninformative constant score has AUC one-half in every row order."""
    scores = [0.5] * 6
    labels = [1, 0, 1, 0, 1, 0]
    assert binary_auc(scores, labels) == pytest.approx(0.5)
    assert binary_auc(list(reversed(scores)), list(reversed(labels))) == pytest.approx(0.5)


def test_auc_interval_resamples_whole_documents_not_repeated_sentences():
    """NEGATIVE CONTROL: pseudo-replicated sentences make the old CI too narrow."""
    scores = [0.9] * 20 + [0.6] * 20 + [0.7] * 20 + [0.4] * 20
    labels = [1] * 40 + [0] * 40
    documents = [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20
    retired = clustered_binary_auc_interval(
        scores,
        labels,
        list(range(len(scores))),
        resamples=5_000,
        seed=17,
    )
    corrected = clustered_binary_auc_interval(
        scores,
        labels,
        documents,
        resamples=5_000,
        seed=17,
    )
    assert retired.ci_low > 0.5, "sentence-level negative control no longer false-resolves AUC"
    assert corrected.ci_low <= 0.5 <= corrected.ci_high


def test_clustered_auc_rejects_alignment_and_single_document_class_failures():
    with pytest.raises(ValueError, match="align"):
        clustered_binary_auc_interval([0.1, 0.9], [0, 1], [0])
    with pytest.raises(ValueError, match="at least two independent documents"):
        clustered_binary_auc_interval([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], [0, 1, 2, 2])


@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        ([], []),
        ([0.5], [0.5, 0.6]),
        ([[0.5, 0.6]], [[0, 1]]),
        ([0.5, float("nan")], [0, 1]),
        ([0.5, 0.6], [0, 2]),
        ([0.5, 0.6], [1, 1]),
        ([0.5, 0.6], [0, 0]),
    ],
)
def test_auc_rejects_malformed_or_single_class_inputs(scores, labels):
    with pytest.raises(ValueError):
        binary_auc(scores, labels)


def test_recorded_claims_match_a_fresh_measurement(artifact, models, comparator, ledger):
    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    classification = classify_sentences(artifact.test, models)
    oracle_gain = summarize_oracle_gain(
        artifact.test,
        models,
        comparator,
        clusters=artifact.test_sequence_document_indices,
    )
    auc_summary = clustered_binary_auc_interval(
        classification["scores"],
        classification["labels"],
        artifact.test_sequence_document_indices,
    )
    assert ledger.get("stage5_mixture_perplexity")["value"] == pytest.approx(
        perplexities["mixture"], abs=1e-4
    )
    assert ledger.get("stage5_typefree_perplexity")["value"] == pytest.approx(
        perplexities["type_free"], abs=1e-4
    )
    assert ledger.get("stage5_type_oracle_perplexity")["value"] == pytest.approx(
        perplexities["type_oracle"], abs=1e-4
    )
    assert ledger.get("stage5_auc")["value"] == pytest.approx(classification["auc"], abs=1e-4)
    assert ledger.get("stage5_auc_ci99_low")["value"] == pytest.approx(auc_summary.ci_low, abs=1e-4)
    assert ledger.get("stage5_auc_ci99_high")["value"] == pytest.approx(
        auc_summary.ci_high, abs=1e-4
    )
    assert ledger.get("stage5_auc_bootstrap_documents")["value"] == auc_summary.n_documents
    assert ledger.get("stage5_test_positions")["value"] == perplexities["positions"]
    assert ledger.get("stage5_majority_baseline_pct")["value"] == pytest.approx(
        100.0 * classification["majority_accuracy"], abs=1e-2
    )
    assert ledger.get("stage5_oracle_gain_nats_per_sentence")["value"] == pytest.approx(
        oracle_gain.mean_nats_per_sentence, abs=1e-6
    )
    assert ledger.get("stage5_oracle_gain_sentence_ci_low")["value"] == pytest.approx(
        oracle_gain.sentence_ci_low, abs=1e-6
    )
    assert ledger.get("stage5_oracle_gain_sentence_ci_high")["value"] == pytest.approx(
        oracle_gain.sentence_ci_high, abs=1e-6
    )
    assert ledger.get("stage5_oracle_gain_nats_per_position")["value"] == pytest.approx(
        oracle_gain.pooled_nats_per_position, abs=1e-6
    )
    assert ledger.get("stage5_oracle_gain_position_ci_low")["value"] == pytest.approx(
        oracle_gain.position_ci_low, abs=1e-6
    )
    assert ledger.get("stage5_oracle_gain_position_ci_high")["value"] == pytest.approx(
        oracle_gain.position_ci_high, abs=1e-6
    )
    assert ledger.get("stage5_oracle_gain_ci_confidence_pct")["value"] == int(
        100 * oracle_gain.confidence
    )
    assert ledger.get("stage5_oracle_gain_bootstrap_resamples")["value"] == oracle_gain.resamples
    assert ledger.get("stage5_oracle_gain_bootstrap_seed")["value"] == oracle_gain.seed
    assert ledger.get("stage5_oracle_gain_bootstrap_documents")["value"] == (
        oracle_gain.n_documents
    )


# ── DETECTABILITY: a non-passing result needs a positive control ─────────────


def test_the_instrument_fires_on_a_planted_effect(synthetic):
    """Instrument interpretation needs a planted effect: on types that genuinely
    differ, the identical harness must classify far above majority AND the
    mixture must beat type-free decisively."""
    classification = classify_sentences(synthetic["test"], synthetic["models"])
    assert classification["accuracy"] > 0.9
    assert classification["majority_accuracy"] == 0.5
    perplexities = type_conditional_perplexities(
        synthetic["test"],
        synthetic["models"],
        build_type_free_chain(synthetic["train"]),
    )
    assert perplexities["mixture"] < perplexities["type_free"] * 0.95


def test_swapped_type_blocks_destroy_the_synthetic_classification(synthetic):
    """NEGATIVE CONTROL: exchanging the two type chains must invert the
    decision — accuracy collapses to near zero, not near chance."""
    swapped = {
        "transitions": synthetic["models"]["transitions"][::-1].copy(),
        "starts": synthetic["models"]["starts"][::-1].copy(),
        "type_prior": synthetic["models"]["type_prior"][::-1].copy(),
    }
    classification = classify_sentences(synthetic["test"], swapped)
    assert classification["accuracy"] < 0.1, (
        f"swapping the type blocks left accuracy at "
        f"{classification['accuracy']:.3f}; the classifier is not reading the "
        f"blocks it claims to read"
    )


def test_cross_type_leakage_collapses_the_type_posterior(synthetic):
    """NEGATIVE CONTROL: averaging the chains removes the evidence."""
    mean_transition = synthetic["models"]["transitions"].mean(axis=0)
    mean_start = synthetic["models"]["starts"].mean(axis=0)
    leaked = {
        "transitions": np.stack([mean_transition, mean_transition]),
        "starts": np.stack([mean_start, mean_start]),
        "type_prior": synthetic["models"]["type_prior"].copy(),
    }
    for _sentence_type, sequence in synthetic["test"][:20]:
        _, posterior = sequence_log_evidence(sequence, leaked)
        np.testing.assert_allclose(posterior, leaked["type_prior"], atol=1e-9)


# ── DECODER EQUIVALENCE: three derivations of the same evidence ──────────────


def test_closed_form_oracle_and_live_agent_agree(artifact, models):
    from ailm.verification.oracle import forward_filter

    emission = build_joint_emission()
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)
    agent = build_clause_agent(artifact)
    agent_a = np.asarray(agent.A[0], dtype=np.float64)[0]
    agent_b = np.asarray(agent.B[0], dtype=np.float64)[0, :, :, 0]
    agent_d = np.asarray(agent.D[0], dtype=np.float64)[0]

    for _sentence_type, sequence in artifact.test[:40]:
        closed, _ = sequence_log_evidence(sequence, models)
        _, oracle = forward_filter(emission, joint_b, joint_d, sequence)
        assert abs(closed - oracle) < 1e-9
        # The live agent's float32 matrices walk to the same evidence.
        predictive = agent_d
        live = 0.0
        for obs in sequence:
            like = agent_a[obs, :] * predictive
            normalizer = like.sum()
            live += float(np.log(normalizer))
            predictive = agent_b @ (like / normalizer)
        assert abs(live - closed) < 1e-4  # float32 storage, float64 walk


# ── DISJOINTNESS: the structural guard inherited from decision 16 ────────────


def test_the_observation_alphabet_is_exactly_the_case_roles(artifact):
    role_names = {role.name for role in CaseRole}
    observed = set()
    for _t, seq in artifact.train + artifact.test:
        observed.update(seq)
    assert observed <= set(range(N_CASES))
    assert len(role_names) == N_CASES
    provenance = artifact.provenance
    assert "PUNCT tokens carry no case role" in provenance["type_label_source"]
    assert provenance["excluded_exclamative"] > 0
    assert provenance["excluded_unpunctuated"] > 0


def test_clause_split_is_document_disjoint_and_matches_stage4(artifact):
    from ailm.levels.case import load_case_artifact

    case_artifact = load_case_artifact("derived")
    assert artifact.document_ids == case_artifact.document_ids
    assert artifact.train_split_document_indices == case_artifact.train_split_document_indices
    assert artifact.test_split_document_indices == case_artifact.test_split_document_indices
    assert set(artifact.train_document_ids).isdisjoint(artifact.test_document_ids)
    assert len(artifact.train_sequence_document_indices) == len(artifact.train)
    assert len(artifact.test_sequence_document_indices) == len(artifact.test)
    assert artifact.n_test_documents < len(artifact.test), (
        "the held-out clause fixture no longer contains multiple sentences per document"
    )
    stride = artifact.provenance["test_every"]
    assert artifact.test_split_document_indices == list(
        range(0, len(artifact.document_ids), stride)
    )


def test_clause_loader_rejects_cross_split_document_aliasing():
    import copy
    import json

    payload = json.loads(
        (PROJECT_ROOT / "vendor" / "ud_ewt_clause_counts.json").read_text(encoding="utf-8")
    )
    broken = copy.deepcopy(payload)
    broken["documents"]["sequence_document_indices"]["train"][0] = broken["documents"][
        "split_indices"
    ]["test"][0]
    with pytest.raises(ValueError, match="sequence/document alignment"):
        ClauseArtifact(broken)


def test_degenerate_artifacts_are_refused(tmp_path):
    import json

    bad = {
        "schema": CLAUSE_ARTIFACT_SCHEMA,
        "provenance": {},
        "splits": {
            "train": [{"type": "declarative", "cases": ["NOM", "PUNCTLEAK"]}],
            "test": [{"type": "declarative", "cases": ["NOM", "ACC"]}],
        },
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="outside the case-role alphabet"):
        load_clause_artifact(path)


# ── GNN CONFORMANCE ──────────────────────────────────────────────────────────


def test_live_clause_agent_conforms_to_its_spec(artifact):
    from ailm.gnn import assert_conforms, describe_shapes, load_spec

    assert_conforms(load_spec("L5_clause"), describe_shapes(build_clause_agent(artifact)))


@pytest.mark.parametrize("mutation", ["identity_joint_transition", "single_type"])
def test_clause_spec_mutations_are_rejected(artifact, mutation):
    from ailm.gnn import check_conformance, describe_shapes, load_spec

    violations = check_conformance(
        load_spec(f"L5_clause@{mutation}"),
        describe_shapes(build_clause_agent(artifact)),
    )
    assert violations, (
        f"L5_clause@{mutation} passed conformance against the correct live "
        f"agent, so that check is vacuous for the property it targets."
    )


# ── LEDGER COMPLETENESS ──────────────────────────────────────────────────────

STAGE5_LABELS = (
    "stage5_train_sentences",
    "stage5_test_sentences",
    "stage5_source_documents",
    "stage5_document_split_train",
    "stage5_document_split_test",
    "stage5_train_documents",
    "stage5_test_documents",
    "stage5_train_interrogatives",
    "stage5_test_interrogatives",
    "stage5_test_positions",
    "stage5_excluded_exclamative",
    "stage5_excluded_unpunctuated",
    "stage5_mixture_perplexity",
    "stage5_typefree_perplexity",
    "stage5_type_oracle_perplexity",
    "stage5_privileged_label_gain_pct",
    "stage5_oracle_gain_nats_per_sentence",
    "stage5_oracle_gain_sentence_ci_low",
    "stage5_oracle_gain_sentence_ci_high",
    "stage5_oracle_gain_nats_per_position",
    "stage5_oracle_gain_position_ci_low",
    "stage5_oracle_gain_position_ci_high",
    "stage5_oracle_gain_ci_confidence_pct",
    "stage5_oracle_gain_bootstrap_resamples",
    "stage5_oracle_gain_bootstrap_seed",
    "stage5_oracle_gain_bootstrap_documents",
    "stage5_classification_pct",
    "stage5_majority_baseline_pct",
    "stage5_interrogative_recall_pct",
    "stage5_auc",
    "stage5_auc_ci99_low",
    "stage5_auc_ci99_high",
    "stage5_auc_ci_confidence_pct",
    "stage5_auc_bootstrap_documents",
    "stage5_classification_margin_pts",
    "stage5_classification_document_ci_low",
    "stage5_classification_document_ci_high",
    "stage5_classification_document_clusters",
    "stage5_classification_document_largest_cluster",
    "stage5_classification_document_p",
    "stage5_classification_document_p_floor",
    "stage5_mcnemar_b",
    "stage5_mcnemar_c",
    "stage5_mcnemar_p",
    "stage5_length_classifier_pct",
    "stage5_decoder_max_divergence",
    "stage5_control_classification_pct",
    "stage5_control_mixture_perplexity",
    "stage5_control_typefree_perplexity",
)


def test_every_stage5_label_is_recorded(ledger):
    missing = [label for label in STAGE5_LABELS if not ledger.verify_label(label)]
    assert missing == [], f"unrecorded stage5 labels {missing}; run scripts/record_stage5_claims.py"


def test_the_joint_dimensions_are_what_the_spec_declares():
    assert N_JOINT == len(CLAUSE_TYPES) * N_CASES == 16
    emission = build_joint_emission()
    assert emission.shape == (N_CASES, N_JOINT)
    np.testing.assert_allclose(emission.sum(axis=0), 1.0)
