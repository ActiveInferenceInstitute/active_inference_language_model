"""Real tiny-corpus controls for the frozen GUM recorder, without scoring GUM."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ailm.claims import ClaimsLedger
from ailm.corpus.connected_gum import (
    GumDocument,
    GumPopulation,
    GumProtocolError,
    authenticated_bytes,
    read_raw_documents,
    select_unexamined_documents,
    validate_registration,
)
from ailm.levels.connected import ConnectedRuntime
from ailm.levels.connected_builder import load_connected_model
from ailm.levels.connected_evaluation import (
    document_uncertainty,
    evaluate_documents,
    validate_frozen_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def _conllu(documents):
    chunks = []
    for identity, sentences in documents:
        chunks.append(f"# newdoc id = {identity}\n")
        for text in sentences:
            chunks.append(
                f"# text = {text}\n1\tsecret\tlemma\tNOUN\t_\tCase=Nom\t0\tnsubj\t_\t_\n\n"
            )
    return "".join(chunks)


def _read_fixture(tmp_path, text):
    path = tmp_path / "fixture.conllu"
    path.write_text(text)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return read_raw_documents(path, expected_sha256=digest), digest


@pytest.fixture(scope="module")
def real_model():
    return load_connected_model()


@pytest.fixture(scope="module")
def measured_fixture(real_model):
    documents = (
        GumDocument("kept-a", ("the man.", "a man?")),
        GumDocument("kept-b", ("the zzzzz.",)),
    )
    return documents, tuple(evaluate_documents(real_model, documents))


def test_raw_comment_parser_preserves_boundaries_only_as_joined_spaces(tmp_path):
    # Given: token labels and forms deliberately disagree with the raw text.
    documents, _ = _read_fixture(
        tmp_path,
        _conllu(
            [("seen", ["the."]), ("kept-a", ["the man", "a man?"]), ("kept-b", ["the zzzzz."])]
        ),
    )
    # When: the already examined source document is removed.
    selected = select_unexamined_documents(documents, ("seen",))
    # Then: only raw text reaches the retained population in source order.
    assert tuple(doc.document_id for doc in selected) == ("kept-a", "kept-b")
    assert selected[0].text == "the man a man?"
    assert "secret" not in selected[0].text


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "# newdoc id = empty\n",
        "# newdoc id = a\n1\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_\n",
        "# text = the\n1\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_\n",
        "# newdoc id = a\n# text = the\n",
        "# newdoc id = a\n# text = the\n# text = another\n",
        _conllu([("duplicate", ["the"]), ("duplicate", ["a"])]),
    ],
)
def test_empty_missing_text_or_ambiguous_corpus_is_rejected(tmp_path, bad):
    with pytest.raises(GumProtocolError):
        _read_fixture(tmp_path, bad)


def test_input_hash_and_extent_controls_are_live(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"registered bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert authenticated_bytes(path, digest, len(b"registered bytes")) == b"registered bytes"
    with pytest.raises(GumProtocolError, match="extent"):
        authenticated_bytes(path, digest, 1)
    path.write_bytes(b"changed bytes")
    with pytest.raises(GumProtocolError, match="SHA-256"):
        read_raw_documents(path, expected_sha256=digest)


def test_exclusion_requires_exact_present_population_identities():
    docs = (GumDocument("a", ("the",)), GumDocument("b", ("a",)), GumDocument("c", ("man",)))
    for excluded in (("missing",), ("a", "a"), ("a", "b", "c"), ("a", "b")):
        with pytest.raises(GumProtocolError):
            select_unexamined_documents(docs, excluded)
    with pytest.raises(GumProtocolError):
        select_unexamined_documents((docs[0], docs[0]), ())


def test_canonical_registration_rejects_changed_bytes_without_scoring(tmp_path):
    directory = tmp_path / "data" / "connected_gum"
    directory.mkdir(parents=True)
    for name in ("protocol.json", "input_manifest.json"):
        shutil.copyfile(ROOT / "data" / "connected_gum" / name, directory / name)
    payload = json.loads((directory / "protocol.json").read_text())
    payload["strength"] = 0.9
    (directory / "protocol.json").write_text(json.dumps(payload))
    with pytest.raises(GumProtocolError, match="SHA-256"):
        validate_registration(tmp_path)


def test_real_live_document_summary_retains_every_token_and_oov(real_model, measured_fixture):
    documents, pairs = measured_fixture
    assert [p.document_id for p in pairs] == [d.document_id for d in documents]
    for document, pair in zip(documents, pairs, strict=True):
        independent = ConnectedRuntime(real_model).infer_text(document.text)
        letter_events = [event for event in independent.events if event.level == "letter"]
        loss = np.mean(
            [-np.log(event.predictive[np.argmax(event.observation)]) for event in letter_events]
        )
        assert pair.integrated.letter_log_loss == pytest.approx(loss, abs=1e-12)
        assert pair.integrated.normalized_tokens == len(independent.tokens)
        assert pair.integrated.scored_characters == len(letter_events)
        assert pair.integrated.normalized_tokens == pair.disconnected.normalized_tokens
        assert not hasattr(pair.integrated, "events")
    assert pairs[1].integrated.oov_tokens == 1
    assert pairs[1].integrated.word_log_loss is None
    assert pairs[1].integrated.word_loss_status == "infinite_oov"


def test_failed_document_does_not_get_silently_skipped(real_model):
    stream = evaluate_documents(
        real_model, (GumDocument("valid", ("the",)), GumDocument("empty", ("123",)))
    )
    assert next(stream).document_id == "valid"
    with pytest.raises(ValueError, match="no ASCII"):
        next(stream)


@pytest.mark.parametrize(
    "value,outcome",
    [
        (1.0, "integrated_lower_loss"),
        (-1.0, "integrated_higher_loss"),
        (0.0, "unresolved"),
    ],
)
def test_bootstrap_positive_harmful_and_null_controls(value, outcome):
    summary = document_uncertainty([value, value, value])
    assert (summary.document_mean_gain, summary.ci_low, summary.ci_high) == (value, value, value)
    assert summary.outcome == outcome


def test_bootstrap_matches_independent_document_draws_and_is_not_token_weighted():
    values = np.array([-0.1, 0.2, 0.4, 1.0])
    summary = document_uncertainty(values)
    rng = np.random.default_rng(20261005)
    draws = [
        float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(10000)
    ]
    low, high = np.percentile(draws, [2.5, 97.5])
    assert summary.document_mean_gain == pytest.approx(float(values.mean()))
    assert summary.ci_low == pytest.approx(low, abs=1e-14)
    assert summary.ci_high == pytest.approx(high, abs=1e-14)


@pytest.mark.parametrize("values", [[], [1.0], [0, float("nan")], [0, float("inf")]])
def test_bootstrap_rejects_incomplete_or_nonfinite_population(values):
    with pytest.raises(GumProtocolError):
        document_uncertainty(values)


def test_retuned_settings_are_rejected():
    validate_frozen_settings(0.5, 20261005, 10000, 0.95)
    for changed in [
        (0.7, 20261005, 10000, 0.95),
        (0.5, 3, 10000, 0.95),
        (0.5, 20261005, 5, 0.95),
        (0.5, 20261005, 10000, 0.90),
    ]:
        with pytest.raises(GumProtocolError):
            validate_frozen_settings(*changed)


def test_real_tiny_measurement_records_atomically_to_only_its_isolated_ledger(
    tmp_path, measured_fixture
):
    # Given: actual live scores on explicitly identified fixture bytes, no canonical GUM score.
    documents, pairs = measured_fixture
    fixture_bytes = _conllu([(doc.document_id, doc.sentences) for doc in documents]).encode()
    population = GumPopulation(
        hashlib.sha256(fixture_bytes).hexdigest(),
        "fixture-protocol",
        "fixture-manifest",
        (),
        tuple(doc.document_id for doc in documents),
    )
    sys.path.insert(0, str(ROOT / "scripts"))
    recorder = importlib.import_module("record_connected_gum_claims")
    binding = recorder.execution_binding(ROOT)
    ledger = ClaimsLedger(tmp_path / "claims.json")
    report = tmp_path / "report.json"
    canonical_before = (ROOT / "data" / "claims.json").read_bytes()
    # When: the same publication function records those measured fixture summaries.
    values = recorder.publish_measurement(pairs, binding, ledger, report, population)
    # Then: provenance, namespace, report digest and measured values agree, with no canonical write.
    assert (ROOT / "data" / "claims.json").read_bytes() == canonical_before
    assert values["connected_gum_corpus_sha256"] == population.corpus_sha256
    assert values["connected_gum_report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert values["connected_gum_document_mean_gain"] == np.mean([p.letter_gain for p in pairs])
    assert values["connected_gum_word_loss_is_infinite"] is True
    assert ledger.get("connected_gum_documents")["status"] == "descriptive"
    before = (tmp_path / "claims.json").read_bytes()
    recorder.publish_measurement(pairs, binding, ledger, report, population)
    assert (tmp_path / "claims.json").read_bytes() == before
    with pytest.raises(GumProtocolError, match="identities"):
        recorder.publish_measurement(pairs[:-1], binding, ledger, report, population)
    assert (tmp_path / "claims.json").read_bytes() == before
    altered = (dataclasses.replace(pairs[0], document_id="wrong"), pairs[1])
    with pytest.raises(GumProtocolError):
        recorder.publish_measurement(altered, binding, ledger, report, population)


def test_recorder_cli_does_not_offer_a_population_limit():
    script = ROOT / "scripts" / "record_connected_gum_claims.py"
    result = subprocess.run(
        [sys.executable, str(script), "--limit", "1"], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr
