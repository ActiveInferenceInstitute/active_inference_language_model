"""The case-word channel: model-sender, message math, and the L3 inlet.

Controls, per protocol P3 §4.3 and the BUILD dispatch:

- the runtime posterior is the real case agent's (built through the shipped
  ``build_case_agent`` on a small real fixture) and each row is numerically
  identical to its Bayes expression — the helper itself refuses otherwise;
- reset semantics: every token is scored from the shipped start prior ``D``,
  never chained through ``B`` (no unregistered sequential dependence);
- ``alpha = 0`` and the no-message case are bit-identical anchors;
- an actual posterior intervention changes receiver priors and outputs when
  the message has support, with the exact ``(1 − α)`` shrink at
  out-of-intersection words;
- zero-support roles are excluded and ``q`` renormalized; the no-message case
  fires only when nothing is supported;
- the withheld evaluation oracle fields cannot influence the model sender:
  the fitting and message functions take no label input, and a test-only word
  never acquires message mass;
- malformed inputs (unnormalized posteriors, bad counts, out-of-vocabulary
  sender words, unnormalized messages) fail closed with real errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from ailm.case_word_channel import (
    apply_case_word_inlet,
    build_message,
    fit_sender,
    role_posterior_from_agent,
)
from ailm.cases import CaseRole
from ailm.channel_registry import (
    ChannelEvidenceRule,
    ChannelSourceKind,
    channel_by_key,
)
from ailm.corpus.ud import document_identity_sha256
from ailm.levels.case import (
    CASE_ARTIFACT_SCHEMA,
    N_CASES,
    CaseArtifact,
    build_case_agent,
    build_case_emission,
    build_case_start,
)

VOCAB = ["the", "dog", "cat", "sed"]
_CASE_ROLE_ORDER = [role.name for role in CaseRole]


def _case_fixture():
    """A small real case artifact and the agent built from it by the shipper."""
    train = [("NOUN", "NOM"), ("NOUN", "ACC"), ("VERB", "NOM"), ("ADJ", "GEN"), ("PRON", "DAT")] * 8
    test = [("NOUN", "NOM"), ("VERB", "NOM")]
    document_ids = ["case-word-channel-train", "case-word-channel-test"]
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
    return artifact, build_case_agent(artifact)


@pytest.fixture(scope="module")
def posterior():
    artifact, agent = _case_fixture()
    return role_posterior_from_agent(agent, n_upos=artifact.n_obs, n_cases=N_CASES), artifact


@pytest.fixture(scope="module")
def sender():
    return fit_sender({"NOM": {"dog": 3, "cat": 1}, "GEN": {"sed": 2}, "ACC": {}}, VOCAB)


def test_the_posterior_rows_are_the_model_s_own_bayes_inversion(posterior):
    """The helper refuses any row that drifts from its closed form; this test
    additionally pins one row against an independent recomputation from the
    shipped builders. The tolerance admits the agent's float32 storage."""
    values, artifact = posterior
    emission = build_case_emission(artifact)
    start = build_case_start(artifact)
    noun = artifact.upos_to_idx["NOUN"]
    closed = start * emission[noun, :]
    closed = closed / closed.sum()
    np.testing.assert_allclose(values[noun], closed, rtol=1e-6, atol=1e-8)


def test_the_posterior_is_per_token_reset_not_sequentially_chained(posterior):
    """Reset semantics: re-scoring the same UPOS is deterministic and does not
    depend on any previous token, and the matrix rows are all that the inlet
    ever sees."""
    values, _ = posterior
    artifact, agent = _case_fixture()
    again = role_posterior_from_agent(agent, n_upos=artifact.n_obs, n_cases=N_CASES)
    np.testing.assert_array_equal(values, again)


def test_fit_sender_returns_supported_roles_as_empirical_distributions(sender):
    """The zero-support role (empty counts) is excluded outright — that is the
    declared policy, so it must not appear as a supported sender."""
    assert set(sender) == {"NOM", "GEN"}
    for vector in sender.values():
        assert abs(vector.sum() - 1.0) < 1e-12
        assert np.all(vector >= 0.0)
    assert sender["NOM"][VOCAB.index("dog")] == pytest.approx(3 / 4)
    assert sender["GEN"][VOCAB.index("sed")] == pytest.approx(1.0)


def test_fit_sender_refuses_malformed_counts():
    with pytest.raises(ValueError, match="positive integer"):
        fit_sender({"NOM": {"dog": 0}}, VOCAB)
    with pytest.raises(ValueError, match="positive integer"):
        fit_sender({"NOM": {"dog": -1}}, VOCAB)
    with pytest.raises(ValueError, match="positive integer"):
        fit_sender({"NOM": {"dog": True}}, VOCAB)


def test_fit_sender_refuses_an_out_of_vocabulary_word():
    with pytest.raises(ValueError, match="pre-intersected"):
        fit_sender({"NOM": {"bear": 1}}, VOCAB)


def test_zero_support_roles_are_excluded_and_the_posterior_renormalized(sender):
    message = build_message(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _CASE_ROLE_ORDER,
        sender,
    )
    assert message is not None
    assert abs(message.sum() - 1.0) < 1e-12


def test_a_message_for_an_out_of_intersection_word_is_exactly_zero(sender):
    """Word-label/split integrity: "the" occurs in no training row, so no
    route and no role can give it message mass — a test-population word never
    acquires support the sender never fitted."""
    message = build_message(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        _CASE_ROLE_ORDER,
        sender,
    )
    assert message is not None
    assert message[VOCAB.index("the")] == 0.0


def test_an_unnormalized_posterior_row_is_refused(sender):
    row = np.zeros(N_CASES)
    row[0] = 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        build_message(row, _CASE_ROLE_ORDER, sender)


def test_a_negative_posterior_row_is_refused(sender):
    row = np.zeros(N_CASES)
    row[0] = 1.2
    row[1] = -0.2
    with pytest.raises(ValueError, match="nonnegative"):
        build_message(row, _CASE_ROLE_ORDER, sender)


def test_misaligned_role_names_are_refused(sender):
    with pytest.raises(ValueError, match="supported role"):
        build_message(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ["NOT_A_ROLE"] * N_CASES,
            sender,
        )


def test_the_no_message_case_returns_the_prior_unchanged():
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    np.testing.assert_array_equal(apply_case_word_inlet(prior, None, 0.7), prior)


def test_alpha_zero_is_a_bit_identical_anchor(sender, posterior):
    values, _ = posterior
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    message = build_message(values[0], _CASE_ROLE_ORDER, sender)
    np.testing.assert_array_equal(apply_case_word_inlet(prior, message, 0.0), prior)


def test_the_intervention_changes_the_receiver_prior_and_its_outputs(sender, posterior):
    """Causal influence, separately from any performance reading: with support
    and alpha > 0 the mixed prior differs from the anchor, and the per-token
    log-probability of an in-intersection word under the mixed prior differs
    from its anchor value; at alpha = 0 it does not."""
    values, _ = posterior
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    message = build_message(values[0], _CASE_ROLE_ORDER, sender)
    mixed = apply_case_word_inlet(prior, message, 0.5)
    assert not np.array_equal(mixed, prior)
    word = VOCAB.index("dog")
    anchor_logp = np.log(prior[word])
    mixed_logp = np.log(mixed[word])
    assert mixed_logp != anchor_logp
    zero_logp = np.log(apply_case_word_inlet(prior, message, 0.0)[word])
    assert zero_logp == anchor_logp


def test_out_of_intersection_words_shrink_by_exactly_one_minus_alpha(sender, posterior):
    values, _ = posterior
    prior = np.array([0.28, 0.24, 0.24, 0.24])
    message = build_message(values[0], _CASE_ROLE_ORDER, sender)
    alpha = 0.2
    mixed = apply_case_word_inlet(prior, message, alpha)
    assert mixed[VOCAB.index("the")] == pytest.approx((1.0 - alpha) * 0.28, rel=1e-15)
    assert abs(mixed.sum() - 1.0) < 1e-12


def test_an_unnormalized_message_is_refused_at_the_seam():
    prior = np.ones(4) / 4.0
    with pytest.raises(ValueError, match="sum to 1"):
        apply_case_word_inlet(prior, np.array([0.9, 0.05, 0.04, 0.0]), 0.5)


def test_a_negative_message_is_refused_at_the_seam():
    prior = np.ones(4) / 4.0
    with pytest.raises(ValueError, match="nonnegative"):
        apply_case_word_inlet(prior, np.array([1.3, -0.1, -0.1, -0.1]), 0.5)


def test_the_registry_declares_both_case_word_identities_with_their_authority():
    keyed = channel_by_key()
    bridge = keyed["case_word_bridge"]
    assert bridge.sender == "L4" and bridge.receiver == "L3"
    assert bridge.source_kind is ChannelSourceKind.MODEL_OUTPUT
    assert bridge.evidence_rule is ChannelEvidenceRule.CASE_WORD_BRIDGE
    assert bridge.comprehension_only is True
    diagnostic = keyed["case_word_oracle_diagnostic"]
    assert diagnostic.source_kind is ChannelSourceKind.PRIVILEGED_ORACLE
    assert diagnostic.sender.startswith("case-oracle")
    assert diagnostic.evidence_rule is ChannelEvidenceRule.CASE_WORD_ORACLE_DIAGNOSTIC


def test_the_registry_composes_the_registered_cascade_segments():
    """Structural reachability, registration-only by protocol P3 §4.3: the
    registered level-to-level hops connect L6→L5, L4→L3, and L3→L1g, so the
    descriptive cascade segments exist as typed identities. This asserts
    registry structure — never an executed cascade, and no performance."""
    keyed = channel_by_key()
    hops = {
        (channel.sender, channel.receiver)
        for channel in keyed.values()
        if channel.sender.startswith("L") and channel.receiver.startswith("L")
    }
    assert ("L6", "L5") in hops
    assert ("L4", "L3") in hops
    assert ("L3", "L1g") in hops
