"""Contracts for the case-word influence-sweep recorder (decision entry 121).

Every test drives real functions — the shipped builders, the shipped mixing
and clustering machinery, and the recorder's own gate chain — over small
fixture channels or the frozen artifacts, into a temporary ledger. No mocks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ailm.case_word_channel import apply_case_word_inlet, build_message, fit_sender
from ailm.cases import CaseRole
from ailm.levels.case import N_CASES
from ailm.levels.clause_channel import scramble_word_distribution

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOCAB = ["the", "dog", "cat", "sed"]
_CASE_ROLE_ORDER = [role.name for role in CaseRole]

#: Labels the happy path must write, spot-checked against the full inventory
#: declared in decision entry 121 (the per-alpha stems are checked at one
#: representative weight; the recorder emits them for every grid weight).
EXPECTED_LABELS = [
    "case_word_sweep_scored_tokens",
    "case_word_sweep_scored_emissions",
    "case_word_sweep_eval_documents",
    "case_word_sweep_vocab_types",
    "case_word_sweep_sender_roles",
    "case_word_sweep_no_message_tokens",
    "case_word_sweep_anchor_ppl",
    "case_word_sweep_anchor_bit_identical",
    "case_word_sweep_anchor_live",
    "case_word_sweep_positive_control_margin_registered",
    "case_word_sweep_best_alpha",
    "case_word_sweep_best_mean_loggain",
    "case_word_sweep_cluster_documents",
    "case_word_sweep_alpha_0_1_mean_loggain",
    "case_word_sweep_alpha_0_1_improved",
    "case_word_sweep_alpha_0_1_degraded",
    "case_word_sweep_alpha_0_1_tied",
    "case_word_sweep_alpha_0_1_scrambled_mean_loggain",
    "case_word_sweep_alpha_0_1_positive_control_mean_loggain",
    "case_word_sweep_alpha_0_1_positive_control_margin",
    "case_word_sweep_alpha_0_1_margin_ok",
    "case_word_sweep_alpha_0_1_cluster_loggain_effect",
    "case_word_sweep_alpha_0_1_cluster_loggain_ci_low",
    "case_word_sweep_alpha_0_1_cluster_loggain_ci_high",
]


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_{name}", PROJECT_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The recorder defines NamedTuples rather than dataclasses exactly so
    # spec-based loaders can execute it without a ``sys.modules`` registration
    # (a dependency replacement the executable-test guard forbids).
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def recorder():
    return _module("record_case_word_influence_claims")


def _posterior_row(weights: dict[str, float]) -> np.ndarray:
    """One real per-UPOS role posterior row over the shipped role inventory."""
    row = np.zeros(N_CASES, dtype=np.float64)
    for role, weight in weights.items():
        row[_CASE_ROLE_ORDER.index(role)] = weight
    return row / row.sum()


@pytest.fixture(scope="module")
def channel(recorder):
    """A small real channel: fitted sender, marginalized message, no-message
    UPOS, two evaluation documents, and the pooled positive control.

    Fixture arithmetic (the assertions below bind it): prior
    ``(0.4, 0.3, 0.2, 0.1)`` over ``VOCAB``; the ``NOUN`` message is
    ``(0, 0.675, 0.075, 0.25)``; ``VERB`` sends nothing. The scored tokens
    are ``dog, dog, the, sed``, so the real arm's mean log-gain is
    ``0.225 * alpha``, the scrambled arm's ``-0.1375 * alpha``, and the
    positive control's ``0.1375 * alpha`` per token.
    """
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    sender = fit_sender({"NOM": {"dog": 9, "cat": 1}, "GEN": {"sed": 2}}, VOCAB)
    messages = {
        "NOUN": build_message(_posterior_row({"NOM": 0.75, "GEN": 0.25}), _CASE_ROLE_ORDER, sender),
        "VERB": None,
    }
    tokens = [("dog", "NOUN"), ("dog", "NOUN"), ("the", "VERB"), ("sed", "NOUN")]
    rows = [
        {"document_index": 0, "words": ["dog", "dog"]},
        {"document_index": 1, "words": ["the", "sed"]},
    ]
    counts = np.zeros(len(VOCAB), dtype=np.float64)
    for word, _upos in tokens:
        counts[VOCAB.index(word)] += 1.0
    return recorder.CaseWordChannel(
        vocab=list(VOCAB),
        word_to_index={word: index for index, word in enumerate(VOCAB)},
        prior=prior,
        messages=messages,
        rows=rows,
        tokens=tokens,
        positive_control=counts / counts.sum(),
        sender_roles=len(sender),
    )


def test_a_collapsed_grid_is_refused(recorder):
    with pytest.raises(SystemExit, match="collapsed"):
        recorder._grid_guard((0.0, 0.5))


def test_the_anchor_is_bit_identical_at_alpha_zero(recorder, channel):
    """At alpha = 0 the sweep's value is the receiver prior's own log-probability,
    bit for bit, at every scored position — and `mix_prior` itself returns the
    prior unchanged."""
    sweep = recorder._sweep(channel)
    for position, (word, _upos) in enumerate(channel.tokens):
        expected = float(np.log(channel.prior[channel.word_to_index[word]]))
        assert sweep[0.0][position] == expected
    noun_message = channel.messages["NOUN"]
    assert noun_message is not None
    np.testing.assert_array_equal(
        apply_case_word_inlet(channel.prior, noun_message, 0.0), channel.prior
    )


def test_the_positive_control_beats_the_scrambled_control_on_fixtures(recorder, channel):
    """The declared positive control — the evaluation stream's own pooled
    unigram — yields a strictly higher mean per-token log-gain than the
    reversed-axis scrambled control at every positive alpha."""
    scrambled_messages = {
        upos: None if message is None else scramble_word_distribution(message)
        for upos, message in channel.messages.items()
    }
    real = recorder._sweep(channel)
    scrambled = recorder._sweep(channel, scrambled_messages)
    positive = recorder._sweep(channel, recorder._positive_control_sender(channel))
    anchor = real[0.0]
    for alpha in recorder.GRID:
        if alpha <= 0.0:
            continue
        positive_gain = recorder._mean_log_gain(positive[alpha], anchor)
        scrambled_gain = recorder._mean_log_gain(scrambled[alpha], anchor)
        assert positive_gain > scrambled_gain


def test_the_positive_control_sender_respects_the_no_message_policy(recorder, channel):
    message_of = recorder._positive_control_sender(channel)
    assert message_of("NOUN") is channel.positive_control
    assert message_of("VERB") is None


def test_the_no_message_policy_fires_d_alpha_equals_d_zero(recorder, channel):
    """A `None` message returns the receiver prior unchanged — no arithmetic at
    all — and the messageless token's sweep value equals the anchor's at every
    alpha, bit for bit."""
    np.testing.assert_array_equal(apply_case_word_inlet(channel.prior, None, 0.7), channel.prior)
    sweep = recorder._sweep(channel)
    anchor = sweep[0.0]
    position = channel.tokens.index(("the", "VERB"))
    for alpha in recorder.GRID:
        assert sweep[alpha][position] == anchor[position]


def test_the_refusing_run_writes_no_ledger(recorder, channel, tmp_path, monkeypatch):
    """The fixture channel's positive-control gap (`0.275 * alpha`, below the
    pre-registered margin at every positive alpha) must trip the margin gate:
    `SystemExit`, and the temporary ledger byte-identical to its skeleton."""
    ledger_path = tmp_path / "claims.json"
    skeleton = json.dumps({"_schema": "ailm-claims-v2", "claims": {}})
    ledger_path.write_text(skeleton)
    monkeypatch.setenv("AILM_CLAIMS_PATH", str(ledger_path))
    with pytest.raises(SystemExit, match="pre-registered margin"):
        recorder.main([], channel=channel)
    assert ledger_path.read_text() == skeleton


def test_the_happy_path_records_the_expected_labels(recorder, tmp_path, monkeypatch):
    """A real run over the frozen artifacts passes every gate and writes the
    `case_word_sweep_*` inventory into the temporary ledger."""
    ledger_path = tmp_path / "claims.json"
    ledger_path.write_text(json.dumps({"_schema": "ailm-claims-v2", "claims": {}}))
    monkeypatch.setenv("AILM_CLAIMS_PATH", str(ledger_path))
    assert recorder.main([]) == 0
    written = json.loads(ledger_path.read_text())["claims"]
    assert written, "the recorder produced no claims"
    assert all(label.startswith("case_word_sweep_") for label in written)
    missing = [label for label in EXPECTED_LABELS if label not in written]
    assert not missing, f"the recorder omitted {missing}"
    assert written["case_word_sweep_anchor_bit_identical"]["value"] == 1
    assert written["case_word_sweep_anchor_live"]["value"] == 1
    assert (
        written["case_word_sweep_positive_control_margin_registered"]["value"]
        == recorder.POSITIVE_CONTROL_MARGIN
    )
    assert written["case_word_sweep_cluster_documents"]["value"] >= 2
