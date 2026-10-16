"""Executable causal, exact-conditional and adversarial connected-runtime controls."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ailm.levels.connected import ConnectedRuntime
from ailm.levels.connected_builder import load_connected_model
from ailm.levels.connected_types import (
    Channel,
    ConnectedInputError,
    Connections,
    Sentence,
    Token,
    parse_text,
)
from ailm.levels.letter import ALPHA_TO_IDX


@pytest.fixture(scope="module")
def model():
    return load_connected_model()


@pytest.fixture(scope="module")
def result(model):
    return ConnectedRuntime(model).infer_text("The man. A man?")


def _exact_marginals(event, agent):
    """Enumerate the actual conditional joint, independently of the runtime's update."""
    emission = np.asarray(agent.A[0], dtype=np.float64)[0]
    joint_prior = event.prior[0]
    for prior in event.prior[1:]:
        joint_prior = np.multiply.outer(joint_prior, prior)
    log_likelihood = np.sum(
        event.observation.reshape((-1,) + (1,) * len(event.prior))
        * np.log(np.maximum(emission, np.finfo(np.float64).eps)),
        axis=0,
    )
    joint = joint_prior * np.exp(log_likelihood - log_likelihood.max())
    joint /= joint.sum()
    marginals = []
    for factor in range(joint.ndim):
        axes = tuple(i for i in range(joint.ndim) if i != factor)
        marginals.append(joint.sum(axis=axes))
    predictive = np.sum(emission * joint_prior[None, ...], axis=tuple(range(1, emission.ndim)))
    return marginals, predictive


def test_every_live_update_matches_independently_enumerated_conditional_joint(model, result):
    # Given: the real GNN-built agents, and a document exercising all clocks.
    expected_levels = {"letter", "phone", "word", "case", "clause", "phase"}
    assert {event.level for event in result.events} == expected_levels
    # When: each event is reconstructed from its pre-observation prior and A.
    for event in result.events:
        expected, predictive = _exact_marginals(event, getattr(model, event.level))
        # Then: both posterior marginals and genuinely predictive scores agree.
        for actual, exact in zip(event.posterior, expected, strict=True):
            np.testing.assert_allclose(actual, exact, atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(event.predictive, predictive, atol=2e-7, rtol=2e-6)


def test_exact_oracle_detects_an_anti_informative_posterior(model, result):
    # Given: an actual informative posterior and its label-reversed corruption.
    event = next(e for e in result.events if e.level == "phone")
    expected, _ = _exact_marginals(event, model.phone)
    corrupted = event.posterior[0][::-1]
    # When/Then: the same agreement criterion fails on the planted defect.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(corrupted, expected[0], atol=3e-6, rtol=2e-5)


def test_word_score_is_the_prior_before_any_word_letters(model, result):
    for token in result.tokens:
        index = model.lexicon.word_to_index[token.form]
        assert token.word_probability == token.word_prior[index]
    assert np.isfinite(result.word_log_loss)
    assert result.tokens[0].word_probability < 1


def test_raw_current_word_and_future_suffix_cannot_change_earlier_predictions(model):
    runtime = ConnectedRuntime(model)
    left = runtime.infer_text("the man.")
    right = runtime.infer_text("a woman? arbitrary future.")
    np.testing.assert_array_equal(left.tokens[0].word_prior, right.tokens[0].word_prior)
    first_left = next(e for e in left.events if e.level == "letter")
    first_right = next(e for e in right.events if e.level == "letter")
    np.testing.assert_array_equal(first_left.predictive, first_right.predictive)
    assert not np.array_equal(first_left.posterior[0], first_right.posterior[0])


def test_raw_punctuation_type_is_not_an_oracle_input(model):
    runtime = ConnectedRuntime(model)
    left = runtime.infer_text("the man. a man.")
    right = runtime.infer_text("the man? a man!")
    for a, b in zip(left.events, right.events, strict=True):
        for qa, qb in zip(a.posterior, b.posterior, strict=True):
            np.testing.assert_array_equal(qa, qb)


def test_structured_upos_changes_preword_prediction_but_raw_tag_is_delayed(model):
    runtime = ConnectedRuntime(model)
    noun = runtime.infer_document((Sentence((Token("the", "NOUN"),)),))
    pronoun = runtime.infer_document((Sentence((Token("the", "PRON"),)),))
    assert np.max(np.abs(noun.tokens[0].word_prior - pronoun.tokens[0].word_prior)) > 1e-5
    assert noun.events[0].level == "case"
    raw = runtime.infer_text("the")
    assert raw.events[0].level == "letter"
    assert next(e.position for e in raw.events if e.level == "case") == len("the")


def test_case_word_disconnect_removes_the_sender_intervention(model):
    runtime = ConnectedRuntime(model, Connections(disabled=frozenset({Channel.CASE_WORD})))
    left = runtime.infer_document((Sentence((Token("the", "NOUN"),)),))
    right = runtime.infer_document((Sentence((Token("the", "PRON"),)),))
    np.testing.assert_array_equal(left.tokens[0].word_prior, right.tokens[0].word_prior)
    assert not np.allclose(left.tokens[0].case_posterior, right.tokens[0].case_posterior)


@pytest.mark.parametrize("channel", tuple(Channel))
def test_each_shuffled_channel_changes_a_live_receiver(model, result, channel):
    shuffled = ConnectedRuntime(model, Connections(shuffled=frozenset({channel})))
    changed = shuffled.infer_text("The man. A man?")
    differences = [
        max(
            np.max(np.abs(a.posterior[0] - b.posterior[0])),
            np.max(np.abs(a.predictive - b.predictive)),
        )
        for a, b in zip(result.events, changed.events, strict=True)
    ]
    assert max(differences) > 1e-6, channel


def test_disconnected_prior_channels_restore_each_receivers_own_dynamics(model):
    connections = Connections(disabled=frozenset(Channel))
    result = ConnectedRuntime(model, connections).infer_text("the the. the")
    for level in ("letter", "phone", "word", "case", "clause"):
        agent = getattr(model, level)
        events = [e for e in result.events if e.level == level]
        previous = None
        for event in events:
            reset = previous is None
            if level in {"phone", "word"} and previous is not None:
                reset = (event.sentence, event.token) != (previous.sentence, previous.token)
            if level in {"case", "clause"} and previous is not None:
                reset = event.sentence != previous.sentence
            for factor, prior in enumerate(event.prior):
                expected = (
                    np.asarray(agent.D[factor]).ravel()
                    if reset
                    else np.asarray(agent.B[factor])[0, :, :, 0] @ previous.posterior[factor]
                )
                np.testing.assert_allclose(prior, expected, atol=2e-7)
            previous = event


def test_reusing_runtime_resets_all_document_state(model, result):
    runtime = ConnectedRuntime(model)
    runtime.infer_text("a different context with many words.")
    repeated = runtime.infer_text("The man. A man?")
    np.testing.assert_array_equal(repeated.phase_posterior, result.phase_posterior)
    for left, right in zip(result.tokens, repeated.tokens, strict=True):
        np.testing.assert_array_equal(left.word_prior, right.word_prior)


def test_disabled_bottom_up_evidence_leaves_receiver_prior_unchanged(model):
    runtime = ConnectedRuntime(
        model,
        Connections(
            disabled=frozenset(
                {
                    Channel.CASE_CLAUSE,
                    Channel.CLAUSE_PHASE,
                }
            )
        ),
    )
    result = runtime.infer_text("the man. a man?")
    for event in result.events:
        if event.level in {"clause", "phase"}:
            assert not event.observation.any()
            np.testing.assert_array_equal(event.posterior[0], event.prior[0])
    # A uniform categorical pseudo-observation is a real negative control: it
    # changes the nondegenerate phase prior under expected-log likelihood.
    event = [e for e in result.events if e.level == "phase"][-1]
    corrupted = dataclasses.replace(event, observation=np.full(2, 0.5))
    marginals, _ = _exact_marginals(corrupted, model.phase)
    assert np.max(np.abs(marginals[0] - event.prior[0])) > 1e-5


def test_zero_strength_is_the_all_channel_disconnected_arm(model):
    zero = ConnectedRuntime(model, Connections(strength=0)).infer_text("the man. a")
    disabled = ConnectedRuntime(model, Connections(disabled=frozenset(Channel))).infer_text(
        "the man. a"
    )
    for left, right in zip(zero.events, disabled.events, strict=True):
        np.testing.assert_array_equal(left.posterior[0], right.posterior[0])


def test_oov_is_explicit_and_does_not_stop_character_or_phone_inference(model):
    result = ConnectedRuntime(model).infer_text("zzzzzzzzzz")
    assert result.tokens[0].oov
    assert result.tokens[0].word_probability == 0
    assert np.isinf(result.word_log_loss)
    assert np.isfinite(result.letter_log_loss)
    assert len(result.tokens[0].phone_posteriors) == len("zzzzzzzzzz")
    np.testing.assert_array_equal(result.tokens[0].upos_observation, model.unknown_upos)


def test_generation_runs_real_policy_inference_and_matches_exact_action_law(model, result):
    runtime = ConnectedRuntime(model)
    generated = runtime.generate_next_word(result)
    assert generated.terminated
    assert generated.word in model.lexicon.vocab
    np.testing.assert_array_equal(generated.word_prior, result.next_word_prior)
    gamma = float(np.asarray(model.generator.gamma).ravel()[0])
    for event in generated.policies:
        log_weight = np.log(event.habit) + gamma * event.preferences
        expected = np.exp(log_weight - log_weight.max())
        expected /= expected.sum()
        np.testing.assert_allclose(event.posterior, expected, atol=3e-6, rtol=2e-5)
        assert event.action == int(np.argmax(expected))
    assert generated.policies[-1].action == ALPHA_TO_IDX[" "]


def test_generation_uses_the_integrated_context_prior(model, result):
    words = ("a", "the")
    outputs = []
    for word in words:
        prior = np.zeros(len(model.lexicon.vocab))
        prior[model.lexicon.word_to_index[word]] = 1
        context = dataclasses.replace(result, next_word_prior=prior)
        outputs.append(ConnectedRuntime(model).generate_next_word(context).word)
    assert tuple(outputs) == words


def test_generation_reports_length_limit_without_claiming_termination(model, result):
    prior = np.zeros(len(model.lexicon.vocab))
    prior[model.lexicon.word_to_index["the"]] = 1
    context = dataclasses.replace(result, next_word_prior=prior)
    generated = ConnectedRuntime(model).generate_next_word(context, max_letters=1)
    assert generated.word == "t"
    assert not generated.terminated


@pytest.mark.parametrize("value", [-1.0, 1.0, np.nan, np.inf])
def test_channel_strength_refuses_unsupported_values(value):
    with pytest.raises(ConnectedInputError):
        Connections(strength=value)


@pytest.mark.parametrize("text", ["", "?!", "123", "東京"])
def test_empty_normalized_text_fails_closed(text):
    with pytest.raises(ConnectedInputError):
        parse_text(text)


def test_token_and_observation_boundary_refuses_invalid_inputs(model):
    with pytest.raises(ConnectedInputError):
        Token("The")
    with pytest.raises(TypeError):
        Token("the", case="NOM")
    with pytest.raises(ConnectedInputError):
        Sentence(())
    with pytest.raises(ConnectedInputError):
        ConnectedRuntime(model).infer_document(())
    with pytest.raises(ConnectedInputError):
        ConnectedRuntime(model).infer_document((Sentence((Token("the", "invalid"),)),))


def test_invalid_kernels_are_rejected_before_inference(model):
    corrupted = model.word_given_case.copy()
    corrupted[:, 0] = 0
    with pytest.raises(ConnectedInputError):
        dataclasses.replace(model, word_given_case=corrupted)


def test_cli_raw_and_structured_observations_produce_strict_json(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_connected_ailm.py"
    raw = subprocess.run(
        [sys.executable, str(script), "--text", "the zzzzz", "--generate"],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(raw.stdout, parse_constant=lambda value: pytest.fail(value))
    assert payload["word_log_loss"] is None
    assert payload["word_loss_status"] == "infinite_oov"
    assert payload["generation"]["terminated"]
    structured = tmp_path / "document.json"
    structured.write_text(json.dumps([[{"form": "the", "upos": "DET"}]]))
    completed = subprocess.run(
        [sys.executable, str(script), "--structured", str(structured)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout)["tokens"][0]["form"] == "the"
    structured.write_text(json.dumps([[{"form": "the", "deprel": "nsubj"}]]))
    invalid = subprocess.run(
        [sys.executable, str(script), "--structured", str(structured)],
        text=True,
        capture_output=True,
    )
    assert invalid.returncode != 0
    assert "deprel" in invalid.stderr


@pytest.mark.parametrize(
    "connections",
    [
        Connections(),
        Connections(disabled=frozenset(Channel)),
        Connections(shuffled=frozenset(Channel)),
    ],
    ids=["connected", "disconnected", "shuffled"],
)
@pytest.mark.parametrize("structured", [False, True], ids=["raw", "structured"])
def test_compiled_live_agents_match_eager_whole_trace_and_generation(
    model, connections, structured
):
    # Given: identical real agents and inputs, including OOV and multiple clocks.
    document = (
        (
            Sentence((Token("the", "DET"), Token("man", "NOUN"))),
            Sentence((Token("zzzzzz", "NOUN"), Token("a", "DET"))),
        )
        if structured
        else parse_text("The man. Zzzzzz a?")
    )
    eager = ConnectedRuntime(model, connections, compile_agent_calls=False)
    compiled = ConnectedRuntime(model, connections, compile_agent_calls=True)
    # When: the same observation schedule calls pymdp eagerly or through JIT.
    reference = eager.infer_document(document)
    actual = compiled.infer_document(document)
    # Then: every exposed belief and pre-observation prediction agrees.
    assert len(reference.events) == len(actual.events)
    for left, right in zip(reference.events, actual.events, strict=True):
        assert (left.level, left.sentence, left.token, left.position) == (
            right.level,
            right.sentence,
            right.token,
            right.position,
        )
        for field in ("prior", "posterior"):
            for a, b in zip(getattr(left, field), getattr(right, field), strict=True):
                np.testing.assert_allclose(a, b, atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(left.observation, right.observation, atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(left.predictive, right.predictive, atol=3e-6, rtol=2e-5)
    for left, right in zip(reference.tokens, actual.tokens, strict=True):
        assert (left.form, left.oov) == (right.form, right.oov)
        assert left.word_probability == pytest.approx(right.word_probability, rel=2e-5, abs=3e-6)
        for field in ("word_prior", "case_posterior", "upos_observation", "phone_posteriors"):
            np.testing.assert_allclose(
                getattr(left, field), getattr(right, field), atol=3e-6, rtol=2e-5
            )
    for field in ("sentence_types", "phase_posterior", "next_word_prior"):
        np.testing.assert_allclose(
            getattr(reference, field), getattr(actual, field), atol=3e-6, rtol=2e-5
        )
    assert reference.letter_log_loss == pytest.approx(actual.letter_log_loss, rel=2e-5, abs=3e-6)
    assert reference.word_log_loss == actual.word_log_loss == float("inf")
    reference_word = eager.generate_next_word(reference)
    actual_word = compiled.generate_next_word(actual)
    assert (reference_word.word, reference_word.terminated) == (
        actual_word.word,
        actual_word.terminated,
    )
    for left, right in zip(reference_word.policies, actual_word.policies, strict=True):
        assert (left.prefix, left.action) == (right.prefix, right.action)
        for field in ("preferences", "habit", "posterior"):
            np.testing.assert_allclose(
                getattr(left, field), getattr(right, field), atol=3e-6, rtol=2e-5
            )
    # Reset behavior also crosses the executor boundary after a different document.
    compiled.infer_text("a different preceding document")
    reset = compiled.infer_document(document)
    for left, right in zip(actual.events, reset.events, strict=True):
        for a, b in zip(left.posterior, right.posterior, strict=True):
            np.testing.assert_array_equal(a, b)


def test_compiled_oracle_check_detects_corrupt_predictions(model, result):
    # Given: the default compiled path and an independent exact conditional oracle.
    event = next(e for e in result.events if e.level == "phone")
    _marginals, exact = _exact_marginals(event, model.phone)
    corrupted = event.predictive[::-1]
    # When/Then: a semantic axis reversal cannot satisfy the unchanged criterion.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(corrupted, exact, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_real_agent_nonfinite_emission_fails_closed_in_both_executors(model, compiled):
    # Given: a real Agent whose emission values have been deliberately corrupted.
    corrupt_phone = eqx.tree_at(
        lambda agent: agent.A,
        model.phone,
        [jnp.full_like(model.phone.A[0], jnp.nan)],
    )
    corrupted = dataclasses.replace(model, phone=corrupt_phone)
    runtime = ConnectedRuntime(corrupted, compile_agent_calls=compiled)
    # When/Then: compiled execution must not hide or sanitize the numerical failure.
    with pytest.raises(ConnectedInputError, match="nonfinite"):
        runtime.infer_text("the")
