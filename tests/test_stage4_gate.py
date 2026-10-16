"""Stage 4 gate — the case level, and the two ways its declared gate fails.

The finding
-----------
The declared ">=60% vs the UD 8-case oracle" fails in **both** directions at
once, and which failure you get depends only on which population you score it
against:

* **gold morphology** — a constant majority predictor already scores
  claim:stage4_gold_majority%, so the threshold is VACUOUS;
* **derived oracle** — the model, both strictly-more-powerful decodes of it, and
  a context-free predictor fitted on the test labels all sit below 60%, so the
  threshold is ABOVE_REFERENCE; none of those comparators is a class bound.

This is the first gate in the project carrying both defects simultaneously, and
it is why a threshold without a named population is not a threshold.
``docs/decisions.md`` entry 16 records it; the tests below re-derive both
verdicts rather than trusting the prose.

The shipped gold arm is an empirical null
-----------------------------------------
Every gold-morphology token in English EWT is a ``PRON``, so the gold
observation alphabet has size claim:stage4_gold_upos_alphabet_size. A constant
observation carries no state information through the emission, but the filter
still evolves through the learned transition. The committed artifact happens
to yield zero discordant pairs *in both directions*; that is a measured result,
not a theorem about every one-tag population. An alternating one-tag control
below proves that transition dynamics can separate the decoder from the
context-free baseline.

The circularity trap
--------------------
The model observes the UPOS tag and nothing else. ``deprel`` and ``feats[Case]``
are withheld because the evaluation labels are derived from them — a model
reading those fields would score near-perfectly while measuring nothing. Unlike
every mis-wiring below this level, that one makes the model look *excellent*, so
there is no numeric signature and the guard must be structural
(``test_the_observation_never_carries_the_label_source``).

Claims discipline
-----------------
This file **verifies**; ``scripts/record_stage4_claims.py`` writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest

from ailm.cases import CaseRole
from ailm.claims import load_ledger
from ailm.control import ControlHarness
from ailm.gnn import assert_conforms, check_conformance, describe_shapes, load_spec
from ailm.levels.case import (
    CASE_ARTIFACT_SCHEMA,
    CASE_TO_IDX,
    FORBIDDEN_OBSERVATIONS,
    N_CASES,
    CaseArtifact,
    baseline_case_positions,
    build_case_agent,
    build_case_emission,
    build_case_start,
    build_case_transition,
    case_agreement,
    context_free_baseline,
    decode_case_positions,
    load_case_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = PROJECT_ROOT / "vendor" / "ud_ewt_case_counts.json"
DECLARED_GATE_PCT = 60.0

#: Significance level for the paired McNemar test. Part of the 11-test
#: family (see docs/decisions.md, tests/test_stage2_gate.py for the
#: Bonferroni analysis). The token-level McNemar result is secondary to the
#: source-document cluster interval recorded by ``record_cluster_claims.py``;
#: the gold-morphology arm has zero discordance.
ALPHA = 0.01

_SKIP_NO_ARTIFACT = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="vendor/ud_ewt_case_counts.json absent; run scripts/generate_case_counts.py",
)

pytestmark = _SKIP_NO_ARTIFACT


def _measure(population: str) -> dict:
    artifact = load_case_artifact(population)
    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)
    model = decode_case_positions(artifact, emission, transition, start)
    baseline = baseline_case_positions(artifact)
    b = sum(1 for m, s in zip(model, baseline, strict=True) if m and not s)
    c = sum(1 for m, s in zip(model, baseline, strict=True) if s and not m)
    return {
        "artifact": artifact,
        "emission": emission,
        "transition": transition,
        "start": start,
        "model_hits": model,
        "baseline_hits": baseline,
        "model": 100.0 * case_agreement(model),
        "baseline": 100.0 * case_agreement(baseline),
        "b": b,
        "c": c,
        "p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0,
    }


@pytest.fixture(scope="module")
def derived() -> dict:
    return _measure("derived")


@pytest.fixture(scope="module")
def gold() -> dict:
    return _measure("gold")


# ══ THE GATE, ON THE POPULATION THAT CAN CARRY ONE ════════════════════════════


def test_stage4_beats_the_context_free_bar_on_the_derived_population(derived):
    """(1) The sequence mechanism must buy something over a context-free lookup."""
    assert derived["model"] > derived["baseline"], (
        f"Case model {derived['model']:.2f}% does not beat the context-free bar "
        f"{derived['baseline']:.2f}%."
    )


def test_the_derived_margin_survives_a_paired_test(derived):
    """A margin without a paired statistic is not evidence (decision 5)."""
    assert derived["p"] < ALPHA, (
        f"McNemar p={derived['p']:.3g} over b={derived['b']} c={derived['c']} "
        f"does not reach alpha={ALPHA}."
    )
    assert derived["b"] > derived["c"], (
        f"Discordance favors the baseline (b={derived['b']}, c={derived['c']}), "
        f"so the margin is in the wrong direction."
    )


def test_the_paired_test_reports_no_effect_on_two_identical_predictors(derived):
    """NO-EFFECT CONTROL: a test that always says 'significant' reads as a pass."""
    hits = derived["model_hits"]
    b = sum(1 for m, s in zip(hits, hits, strict=True) if m and not s)
    c = sum(1 for m, s in zip(hits, hits, strict=True) if s and not m)
    assert (b, c) == (0, 0)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    assert p >= ALPHA, "Comparing a predictor with itself reported an effect."


def test_the_gate_is_live_under_the_control_harness(derived):
    """The gate must discriminate: a mis-wired model lands on the wrong side."""
    artifact = derived["artifact"]
    flat = np.full_like(derived["emission"], 1.0 / derived["emission"].shape[0])
    broken = decode_case_positions(artifact, flat, derived["transition"], derived["start"])
    ControlHarness("stage4_case").assert_gate_is_live(
        good_value=derived["model"],
        bad_value=100.0 * case_agreement(broken),
        direction="higher_is_better",
    )


# ══ THE DECLARED GATE HAS TWO POPULATION-DEPENDENT DEFECTS ════════════════════


def test_the_declared_gate_is_vacuous_on_gold(gold):
    """A constant majority predictor already clears 60% on gold morphology."""
    truth = [case for seq in gold["artifact"].test for _, case in seq]
    majority = 100.0 * max(truth.count(c) for c in set(truth)) / len(truth)
    assert majority >= DECLARED_GATE_PCT, (
        f"The gold majority class scores {majority:.2f}%, below the declared "
        f"{DECLARED_GATE_PCT}%, so the VACUOUS finding in decisions.md entry 16 "
        f"no longer holds."
    )


def test_the_declared_gate_is_above_measured_references_on_derived(derived):
    """Every measured, exceedable reference falls below 60% on derived data.

    Checked against the model AND against strictly more powerful decodes of the
    same model. These values demonstrate an above-reference threshold, not an
    unreachable one: none is a bound over every admissible model (decision 75).
    """
    artifact = derived["artifact"]
    emission, transition, start = (
        derived["emission"],
        derived["transition"],
        derived["start"],
    )
    smoothed_hits = total = 0
    for sequence in artifact.test:
        obs = [artifact.upos_to_idx[u] for u, _ in sequence]
        truth = [CASE_TO_IDX[CaseRole[c]] for _, c in sequence]
        length = len(obs)
        alpha = np.zeros((length, N_CASES))
        beta = np.ones((length, N_CASES))
        alpha[0] = start * emission[obs[0], :]
        alpha[0] /= alpha[0].sum()
        for t in range(1, length):
            alpha[t] = (transition @ alpha[t - 1]) * emission[obs[t], :]
            alpha[t] /= alpha[t].sum()
        for t in range(length - 2, -1, -1):
            beta[t] = transition.T @ (beta[t + 1] * emission[obs[t + 1], :])
            beta[t] /= beta[t].sum()
        smoothed_hits += sum(int(np.argmax(alpha[t] * beta[t]) == truth[t]) for t in range(length))
        total += length
    smoothing = 100.0 * smoothed_hits / total

    assert derived["model"] < DECLARED_GATE_PCT
    assert smoothing < DECLARED_GATE_PCT, (
        f"Forward-backward smoothing reaches {smoothing:.2f}%, clearing the "
        f"declared {DECLARED_GATE_PCT}%. The UNREACHABLE finding needs revisiting."
    )


def test_the_two_verdicts_come_from_one_threshold(gold, derived):
    """The whole Stage 4 finding, in one assertion.

    One declared number; opposite defects; the only difference is the population.
    """
    gold_truth = [case for seq in gold["artifact"].test for _, case in seq]
    gold_majority = 100.0 * max(gold_truth.count(c) for c in set(gold_truth)) / len(gold_truth)
    assert gold_majority >= DECLARED_GATE_PCT > derived["model"], (
        f"Expected the declared {DECLARED_GATE_PCT}% to be cleared by the gold "
        f"majority ({gold_majority:.2f}%) and unreached by the derived model "
        f"({derived['model']:.2f}%)."
    )


# ══ THE SHIPPED GOLD NULL IS EMPIRICAL ════════════════════════════════════════


def test_the_gold_observation_alphabet_is_a_single_tag(gold):
    """Every gold-morphology token is a PRON, so the observation is constant."""
    assert gold["artifact"].upos_tags == ["PRON"], (
        f"The gold population's UPOS alphabet is "
        f"{gold['artifact'].upos_tags}, not the single tag the null rests on."
    )


def test_the_shipped_gold_arm_has_zero_discordance_in_both_directions(gold):
    """The committed artifact reproduces its recorded empirical null."""
    assert (gold["b"], gold["c"]) == (0, 0), (
        f"Gold discordance is b={gold['b']}, c={gold['c']}; the committed "
        "artifact no longer reproduces the recorded null."
    )
    assert gold["model"] == pytest.approx(gold["baseline"]), (
        "Model and baseline scores differ on a population where every prediction is identical."
    )


def test_a_single_tag_does_not_force_decoder_baseline_agreement():
    """NEGATIVE CONTROL: transition dynamics survive a constant emission.

    Both cases emit the same tag, yet the alternating transition makes the
    filtering decoder time-dependent while the context-free baseline remains
    constant. This counterexample prevents the shipped empirical null from
    being promoted to a structural impossibility claim.
    """
    train = [("NOUN", "NOM"), ("NOUN", "ACC")] * (N_CASES * N_CASES)
    test = [("NOUN", "NOM"), ("NOUN", "ACC"), ("NOUN", "NOM"), ("NOUN", "ACC")]
    from ailm.corpus.ud import document_identity_sha256

    document_ids = ["fixture-train", "fixture-test"]
    payload = {
        "schema": CASE_ARTIFACT_SCHEMA,
        "provenance": {
            "documents": 2,
            "train_documents": 1,
            "test_documents": 1,
            "document_identity_sha256": document_identity_sha256(document_ids),
        },
        "heuristic_vs_gold": {
            "tokens_with_both": len(train),
            "agreements": len(train),
            "source_path_tokens": {"deprel": len(train), "adposition_token": 0},
        },
        "oracle_scope_audit": {
            "full_derived_tokens": len(train) + len(test),
            "derived_path_tokens": {
                "morphology": len(train) + len(test),
                "deprel": 0,
                "adposition_token": 0,
            },
            "heuristic_vs_gold": {
                "tokens_with_both": len(train),
                "agreements": len(train),
                "source_path_tokens": {"deprel": len(train), "adposition_token": 0},
            },
            "adposition_token": {
                "tokens": 0,
                "morphological_case_tokens": 0,
                "head_resolved_tokens": 0,
                "head_unresolved_tokens": 0,
                "head_also_labeled_tokens": 0,
                "head_role_agreements": 0,
                "head_role_disagreements": 0,
            },
        },
        "populations": {"gold": {"train": [train], "test": [test]}},
        "documents": {
            "ids": document_ids,
            "split_indices": {"train": [0], "test": [1]},
            "sequence_document_indices": {"gold": {"train": [0], "test": [1]}},
        },
    }
    artifact = CaseArtifact(payload, "gold")
    emission = build_case_emission(artifact)
    assert artifact.n_obs == 1
    assert np.array_equal(emission, np.ones_like(emission))

    model_hits = decode_case_positions(
        artifact,
        emission,
        build_case_transition(artifact),
        build_case_start(artifact),
    )
    baseline_hits = baseline_case_positions(artifact)
    assert model_hits == [True] * len(test)
    # The tag ties NOM against ACC exactly, and the comparator breaks ties by
    # sorted role name (decision 110, F5), so the constant prediction is ACC.
    # Before that change the tie fell to insertion order and the pattern was
    # inverted; either way the baseline is constant while the decoder is not,
    # which is the property this control exists to show.
    assert baseline_hits == [False, True, False, True]
    assert len(set(baseline_hits[::2])) == 1 and len(set(baseline_hits[1::2])) == 1
    assert model_hits != baseline_hits


def test_the_zero_discordance_check_is_not_vacuous(derived):
    """POSITIVE CONTROL: the same comparison finds discordance where it exists.

    Without this, a bug that always reported (0, 0) would read as a clean null.
    """
    assert (derived["b"], derived["c"]) != (0, 0), (
        "The derived population shows no discordance either, so the gold result "
        "is a property of the comparison rather than of the data."
    )


# ══ THE CIRCULARITY GUARD ═════════════════════════════════════════════════════


def test_the_observation_never_carries_the_label_source():
    """The structural guard against reading the answer key (decision 15).

    The oracle labels are derived from ``deprel`` and ``feats[Case]``. A model
    observing either would score near-perfectly while measuring nothing, and —
    unlike every other mis-wiring here — it would look *excellent* rather than
    broken. There is no numeric signature to catch, so the vendored artifact is
    checked for the fields themselves.
    """
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    observed = {
        upos
        for splits in payload["populations"].values()
        for split in splits.values()
        for sequence in split
        for upos, _ in sequence
    }
    from ailm.corpus.ud import UD_UPOS_TAGS
    from ailm.levels.case import CaseArtifact

    offenders = observed - UD_UPOS_TAGS
    assert offenders == set(), f"The observation alphabet contains non-UPOS values {offenders}."
    assert payload["provenance"]["observation"] == (
        "UPOS tag only; deprel and feats[Case] withheld"
    ), "The artifact no longer records the observation restriction it was built under."
    assert frozenset({"deprel", "feats", "case"}) == FORBIDDEN_OBSERVATIONS

    # NEGATIVE CONTROLS: uppercase answer keys and uppercased feature strings
    # defeated the old isupper()/colon guard. The production loader must reject
    # them even when the self-declared provenance remains untouched.
    for planted_observation in ("NOM", "CASE=NOM"):
        planted = json.loads(json.dumps(payload))
        for splits in planted["populations"].values():
            for split in splits.values():
                for sequence in split:
                    for pair in sequence:
                        pair[0] = planted_observation
        with pytest.raises(ValueError, match="canonical UD UPOS"):
            CaseArtifact(planted, "derived")


def test_train_and_test_are_disjoint_at_the_authenticated_document_boundary(derived, gold):
    """Every retained sequence stays inside the pre-filter document partition."""
    for measured in (derived, gold):
        artifact = measured["artifact"]
        assert set(artifact.train_split_document_indices).isdisjoint(
            artifact.test_split_document_indices
        )
        assert set(artifact.train_document_ids).isdisjoint(artifact.test_document_ids)
        assert len(artifact.train_sequence_document_indices) == len(artifact.train)
        assert len(artifact.test_sequence_document_indices) == len(artifact.test)
        assert artifact.n_test_documents < len(artifact.test), (
            "the held-out fixture no longer contains multiple sentences per document; "
            "document clustering would be vacuous"
        )

        stride = artifact.provenance["test_every"]
        assert artifact.test_split_document_indices == list(
            range(0, len(artifact.document_ids), stride)
        ), "the split was not fixed over document identities before filtering"


def test_retired_sentence_stride_leaks_real_ewt_documents():
    """NEGATIVE CONTROL: the old green split puts source documents in both arms."""
    from ailm.corpus.ud import load_ud_ewt_dev

    sentences = load_ud_ewt_dev()
    old_test = {sentence.document_id for index, sentence in enumerate(sentences) if index % 5 == 0}
    old_train = {sentence.document_id for index, sentence in enumerate(sentences) if index % 5 != 0}
    overlap = old_train & old_test
    assert overlap, "negative control no longer exposes sentence-stride document leakage"

    artifact = load_case_artifact("derived")
    new_train = {artifact.document_ids[index] for index in artifact.train_split_document_indices}
    new_test = {artifact.document_ids[index] for index in artifact.test_split_document_indices}
    assert new_train.isdisjoint(new_test)


def test_document_mapping_rejects_cross_split_sequence_aliasing():
    """POSITIVE CONTROL: aligned sequence counts alone cannot certify disjointness."""
    import copy

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    broken = copy.deepcopy(payload)
    broken["documents"]["sequence_document_indices"]["derived"]["train"][0] = broken["documents"][
        "split_indices"
    ]["test"][0]
    with pytest.raises(ValueError, match="sequence/document alignment"):
        CaseArtifact(broken, "derived")


def test_a_deprel_observation_would_be_near_perfect(derived):
    """WHY the guard exists, demonstrated rather than asserted.

    This builds the circular predictor the guard forbids and shows it scores far
    above the honest model. Recording that number is what makes the prohibition
    legible: the danger is not that circularity looks broken, it is that it looks
    like success.
    """
    from ailm.cases import UD_DEPREL_TO_CASE
    from ailm.corpus.ud import load_ud_ewt_dev
    from ailm.levels.case import load_case_artifact, oracle_path_strata

    # BUILD the circular predictor, which is what this test's name promises and
    # what it did not previously do: the old body asserted
    # len(set(UD_DEPREL_TO_CASE.values())) >= 5 and model < 60, so it counted
    # dictionary values and never constructed a predictor at all. A guard whose
    # docstring describes an experiment it does not run is worse than no guard,
    # because the suite reports it as covered.
    artifact = load_case_artifact("derived")
    gold_labels = [case for sequence in artifact.test for _upos, case in sequence]
    strata = oracle_path_strata()
    assert len(strata) == len(gold_labels), "stratification lost its alignment"

    # The forbidden observation, used as an observation exactly once, here.
    from ailm.cases import CaseRole
    from ailm.corpus.ud import _apply_case_oracle

    gold_map = {"Nom": CaseRole.NOM, "Acc": CaseRole.ACC, "Gen": CaseRole.GEN}
    deprels: list[str] = []
    heldout_documents = set(artifact.test_document_ids)
    for sentence in load_ud_ewt_dev():
        if sentence.document_id not in heldout_documents:
            continue
        sequence = []
        for token in sentence.tokens:
            case = gold_map.get(token.case_morphological)
            if case is None:
                case = _apply_case_oracle(token)
            if case is not None:
                sequence.append(token.deprel)
        if len(sequence) >= 2:
            deprels.extend(sequence)
    assert len(deprels) == len(gold_labels), "deprel reconstruction lost its alignment"

    circular = [
        UD_DEPREL_TO_CASE[deprel].name if deprel in UD_DEPREL_TO_CASE else None
        for deprel in deprels
    ]
    circular_pct = (
        100.0
        * sum(1 for predicted, gold in zip(circular, gold_labels, strict=True) if predicted == gold)
        / len(gold_labels)
    )

    # On the deprel stratum the map is the label's own generator, so a predictor
    # observing deprel must be perfect there. That is the hazard, stated as a
    # number rather than as a worry.
    deprel_index = [i for i, path in enumerate(strata) if path == "deprel"]
    deprel_exact = sum(1 for i in deprel_index if circular[i] == gold_labels[i])
    assert deprel_exact == len(deprel_index), (
        f"the deprel rule recovered {deprel_exact} of {len(deprel_index)} labels on the "
        f"stratum it generates; if this is not exact, the oracle and UD_DEPREL_TO_CASE "
        f"have diverged and decision 15 needs revisiting"
    )
    assert circular_pct > derived["model"] + 10.0, (
        f"the circular predictor scores {circular_pct:.2f}% against the honest model's "
        f"{derived['model']}%. The prohibition is legible only while the forbidden "
        f"observation is visibly better; if the gap has closed, the guard no longer "
        f"demonstrates why the observation is withheld."
    )


# ══ STRUCTURAL INVARIANTS ═════════════════════════════════════════════════════


def test_emission_is_normalized_over_observations_not_over_cases(derived):
    """The transpose is a different model, and pymdp accepts it (decision 1)."""
    emission = derived["emission"]
    assert emission.shape == (derived["artifact"].n_obs, N_CASES)
    np.testing.assert_allclose(
        emission.sum(axis=0),
        np.ones(N_CASES),
        atol=1e-9,
        err_msg="A[:, case] must be a distribution over UPOS tags",
    )


def test_transition_is_column_stochastic_over_the_next_case(derived):
    transition = derived["transition"]
    assert transition.shape == (N_CASES, N_CASES)
    np.testing.assert_allclose(
        transition.sum(axis=0),
        np.ones(N_CASES),
        atol=1e-9,
        err_msg="B[:, prev] must be a distribution over the next case",
    )


def test_the_transition_carries_real_dynamics(derived):
    """A transition indistinguishable from uniform carries no sequence claim."""
    transition = derived["transition"]
    assert not np.allclose(transition, 1.0 / N_CASES, atol=0.02), (
        "The case transition is indistinguishable from uniform — the vendored "
        "counts were dropped during parsing, and the level's whole claim is empty."
    )


def test_the_baseline_is_context_free_by_construction(derived):
    """The comparator must not use sequence context, or it is not the bar."""
    table = context_free_baseline(derived["artifact"])
    assert set(table) == set(derived["artifact"].upos_tags)
    assert all(isinstance(v, str) for v in table.values()), (
        "The baseline must map each UPOS tag to a single case, independent of "
        "position — a position-dependent table would already use context."
    )


def test_scoring_over_no_positions_raises(derived):
    """An agreement over zero positions is undefined, not zero."""
    with pytest.raises(ValueError, match="undefined"):
        case_agreement([])


# ══ NEGATIVE CONTROLS ═════════════════════════════════════════════════════════


def test_uniform_emission_collapses_agreement(derived):
    """An information-free emission must fail the bar."""
    artifact = derived["artifact"]
    flat = np.full_like(derived["emission"], 1.0 / derived["emission"].shape[0])
    hits = decode_case_positions(artifact, flat, derived["transition"], derived["start"])
    assert 100.0 * case_agreement(hits) < derived["baseline"], (
        "A uniform emission still beats the context-free bar, so the bar does "
        "not require the emission to carry information."
    )


def test_transposed_emission_is_rejected_by_the_builder(derived):
    """A transposed emission must not be silently accepted."""
    with pytest.raises(ValueError, match="emission must be"):
        build_case_agent(derived["artifact"], emission=derived["emission"].T)


def test_time_reversed_transition_is_caught(derived):
    """P(prev | cur) in the transition slot is the anti-informative wiring."""
    artifact = derived["artifact"]
    reversed_transition = derived["transition"].T
    reversed_transition = reversed_transition / reversed_transition.sum(axis=0, keepdims=True)
    hits = decode_case_positions(
        artifact, derived["emission"], reversed_transition, derived["start"]
    )
    assert 100.0 * case_agreement(hits) < derived["model"], (
        "The time-reversed transition scores at least as well as the shipped "
        "one, so the transition's orientation is not doing any work."
    )


# ══ GNN CONFORMANCE ═══════════════════════════════════════════════════════════


def test_live_agent_conforms_to_its_spec(derived):
    assert_conforms(load_spec("L4_case"), describe_shapes(build_case_agent(derived["artifact"])))


#: The intended semantic reason each on-disk mutation must be rejected for,
#: pinned from the violation text the conformance checker emits against the
#: live agent — same contract as the Stage 3 gate's mutation pins.
L4_MUTATION_EXPECTATIONS = {
    "identity_emission": "cannot be a 8x8 identity",
    "frozen_transition": (
        "B: declared Identity(8) in InitialParameterization but the live matrix is NOT the identity"
    ),
    "uninformed_prior": "D: declared Uniform(8) but the live array is not uniform",
    "swapped_alphabets": ("but ModelParameters num_obs=12, num_hidden_states=8 require (12, 8)"),
}


@pytest.mark.parametrize("mutation", sorted(L4_MUTATION_EXPECTATIONS))
def test_spec_mutations_are_rejected_for_their_intended_reason(derived, mutation):
    """NEGATIVE CONTROLS on conformance itself, pinned to the intended reason."""
    violations = check_conformance(
        load_spec(f"L4_case@{mutation}"),
        describe_shapes(build_case_agent(derived["artifact"])),
    )
    joined = "\n".join(violations)
    assert violations, (
        f"L4_case@{mutation} passed conformance against the correct live agent, "
        f"so that check is vacuous for the property the mutation targets."
    )
    assert L4_MUTATION_EXPECTATIONS[mutation] in joined, (
        f"L4_case@{mutation} was rejected, but not for its intended reason, so "
        f"the rejection no longer measures the defect the mutation carries: {joined}"
    )


_L4_DISTINCT_PAIRS = [
    (intended, other)
    for intended in sorted(L4_MUTATION_EXPECTATIONS)
    for other in sorted(L4_MUTATION_EXPECTATIONS)
    if intended != other
]


@pytest.mark.parametrize("intended,other", _L4_DISTINCT_PAIRS)
def test_a_different_l4_mutation_cannot_satisfy_the_intended_expectation(derived, intended, other):
    """FALSIFIER over the distinct shipped mutations, not every possible fault.

    For every ordered pair of DIFFERENT on-disk mutations, the OTHER
    mutation's violations must not contain THIS mutation's expected reason;
    only the pairs that exist are collected, so the run carries no skips. The
    identical pair is pinned directly by the rejection test.
    """
    violations = check_conformance(
        load_spec(f"L4_case@{other}"),
        describe_shapes(build_case_agent(derived["artifact"])),
    )
    joined = "\n".join(violations)
    assert violations, "the unrelated mutation must still be rejected, or the pair is vacuous"
    assert L4_MUTATION_EXPECTATIONS[intended] not in joined, (
        f"L4_case@{other}'s violations satisfy L4_case@{intended}'s expected "
        f"reason, so the fragment does not pin the intended semantic defect: {joined}"
    )


# ══ SCOPE, STATED RATHER THAN HIDDEN ══════════════════════════════════════════


def test_five_of_eight_cases_have_no_morphological_ground_truth(gold, derived):
    """The limit is the corpus, not the modeling effort."""
    assert len(gold["artifact"].cases_present) == 3
    assert len(derived["artifact"].cases_present) == 8
    missing = N_CASES - len(gold["artifact"].cases_present)
    assert missing == 5, (
        f"{missing} cases lack morphological ground truth, not the 5 recorded in "
        f"decisions.md entry 16."
    )


def test_the_heuristic_fidelity_is_measured_on_its_named_overlap(derived):
    """The covered overlap is imperfect and records its path composition."""
    agreement = derived["artifact"].heuristic_gold_agreement_pct
    assert 50.0 < agreement < 100.0, (
        f"Heuristic-vs-gold agreement is {agreement}%. At exactly 100% the "
        f"comparison is green by construction — check that it is not consulting "
        f"feats[Case] on the very tokens it is scoring."
    )
    scope = derived["artifact"].oracle_scope_audit
    fidelity = scope["heuristic_vs_gold"]
    paths = fidelity["source_path_tokens"]
    assert sum(paths.values()) == fidelity["tokens_with_both"]
    assert paths["adposition_token"] == 0, (
        "The heuristic-vs-gold result now contains token-level adposition examples; "
        "its scope and interpretation must be re-audited before publication."
    )


def test_oracle_scope_sufficient_counts_are_fail_closed():
    """POSITIVE CONTROL: an inconsistent head partition must be rejected."""
    import copy

    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    broken = copy.deepcopy(payload)
    broken["oracle_scope_audit"]["adposition_token"]["head_role_disagreements"] += 1
    with pytest.raises(ValueError, match="partition the labeled heads"):
        CaseArtifact(broken, "derived")


# ══ CLAIMS LEDGER — verify, never write ══════════════════════════════════════


def test_stage4_claims_verified(gold, derived):
    ledger = load_ledger(PROJECT_ROOT)
    scope = derived["artifact"].oracle_scope_audit
    paths = scope["derived_path_tokens"]
    fidelity = scope["heuristic_vs_gold"]
    fidelity_paths = fidelity["source_path_tokens"]
    adposition = scope["adposition_token"]
    expected = {
        "stage4_derived_agreement": round(derived["model"], 2),
        "stage4_derived_baseline": round(derived["baseline"], 2),
        "stage4_derived_mcnemar_b": derived["b"],
        "stage4_derived_mcnemar_c": derived["c"],
        "stage4_gold_agreement": round(gold["model"], 2),
        "stage4_gold_mcnemar_b": gold["b"],
        "stage4_gold_mcnemar_c": gold["c"],
        "stage4_gold_upos_alphabet_size": gold["artifact"].n_obs,
        "stage4_cases_without_morphology": 5,
        "stage4_heuristic_gold_tokens": fidelity["tokens_with_both"],
        "stage4_heuristic_gold_adposition_token_tokens": fidelity_paths["adposition_token"],
        "stage4_oracle_full_derived_tokens": scope["full_derived_tokens"],
        "stage4_oracle_full_path_morphology_tokens": paths["morphology"],
        "stage4_oracle_full_path_deprel_tokens": paths["deprel"],
        "stage4_oracle_full_path_adposition_token_tokens": paths["adposition_token"],
        "stage4_adposition_token_morphology_tokens": adposition["morphological_case_tokens"],
        "stage4_adposition_token_head_also_labeled_tokens": adposition["head_also_labeled_tokens"],
        "stage4_adposition_token_head_role_disagreements": adposition["head_role_disagreements"],
    }
    for label, value in expected.items():
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent. Run scripts/record_stage4_claims.py — this test may not write it."
        )
        assert entry["source"].startswith("scripts/"), (
            f"{label} was last written by {entry['source']!r}. Recorders write; tests verify."
        )
        assert entry["value"] == pytest.approx(value, rel=1e-4), (
            f"Ledger {label}={entry['value']!r} != fresh measurement {value!r}."
        )


# ── THE COMPARATOR'S TIE-BREAK MUST BE A FUNCTION OF THE COUNTS ──────────────


def _one_tag_payload(train: list[tuple[str, str]], test: list[tuple[str, str]]) -> dict:
    """Minimal single-population artifact payload around one train/test sequence."""
    from ailm.corpus.ud import document_identity_sha256

    document_ids = ["fixture-train", "fixture-test"]
    scope = {
        "tokens_with_both": len(train),
        "agreements": len(train),
        "source_path_tokens": {"deprel": len(train), "adposition_token": 0},
    }
    return {
        "schema": CASE_ARTIFACT_SCHEMA,
        "provenance": {
            "documents": 2,
            "train_documents": 1,
            "test_documents": 1,
            "document_identity_sha256": document_identity_sha256(document_ids),
        },
        "heuristic_vs_gold": dict(scope),
        "oracle_scope_audit": {
            "full_derived_tokens": len(train) + len(test),
            "derived_path_tokens": {
                "morphology": len(train) + len(test),
                "deprel": 0,
                "adposition_token": 0,
            },
            "heuristic_vs_gold": dict(scope),
            "adposition_token": {
                "tokens": 0,
                "morphological_case_tokens": 0,
                "head_resolved_tokens": 0,
                "head_unresolved_tokens": 0,
                "head_also_labeled_tokens": 0,
                "head_role_agreements": 0,
                "head_role_disagreements": 0,
            },
        },
        "populations": {"gold": {"train": [train], "test": [test]}},
        "documents": {
            "ids": document_ids,
            "split_indices": {"train": [0], "test": [1]},
            "sequence_document_indices": {"gold": {"train": [0], "test": [1]}},
        },
    }


def test_the_context_free_comparator_breaks_ties_by_name_not_by_row_order():
    """A tied argmax must not depend on which row the generator emitted first.

    ``Counter.most_common`` returns first-seen insertion order on a tie, so the
    same counts in a different sentence order gave a different prediction, and a
    regenerated artifact could move the Stage 4 comparator without any count
    changing. The rule here is the one
    ``phoneme.per_letter_argmax_baseline`` already documents: sort by name, then
    take the maximum. Both orders below must therefore give 'ACC'.
    """
    tied_nom_first = [("NOUN", "NOM"), ("NOUN", "ACC")]
    tied_acc_first = [("NOUN", "ACC"), ("NOUN", "NOM")]
    held_out = [("NOUN", "NOM")]

    first = context_free_baseline(CaseArtifact(_one_tag_payload(tied_nom_first, held_out), "gold"))
    second = context_free_baseline(CaseArtifact(_one_tag_payload(tied_acc_first, held_out), "gold"))

    assert first == second, (
        f"identical counts in a different row order gave {first} and {second}; "
        f"the comparator is reading the artifact's ordering, not its data"
    )
    assert first == {"NOUN": "ACC"}, (
        f"expected the name-sorted winner 'ACC', got {first}. Ties must resolve "
        f"the same way phoneme.per_letter_argmax_baseline resolves them."
    )
