"""Connected-result prose is derived from recorded real runtime fixtures."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

from ailm.claims import ClaimsLedger
from ailm.corpus.connected_gum import (
    EXCLUDED_DOCUMENTS,
    MANIFEST_SHA256,
    PROTOCOL_SHA256,
    TEST_SHA256,
    GumDocument,
    GumPopulation,
)
from ailm.levels.connected_builder import load_connected_model
from ailm.levels.connected_evaluation import evaluate_documents
from ailm.manuscript_variables._claims import _ClaimReaders, _validate_required_claim_values
from ailm.manuscript_variables._exceptions import MissingClaimError
from ailm.manuscript_variables._io import _StrictVariableMap
from ailm.manuscript_variables.connected import _produce_connected_gum
from ailm.required_claims import CONNECTED_GUM_REQUIRED_CLAIMS
from ailm.value_formatting import ManuscriptVariableError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recorded_fixture(tmp_path_factory):
    directory = tmp_path_factory.mktemp("connected-manuscript")
    sys.path.insert(0, str(ROOT / "scripts"))
    recorder = importlib.import_module("record_connected_gum_claims")
    model = load_connected_model()

    def record(*, oov: bool, swap: bool = False, mixed: bool = False):
        text = "the man bzzzz." if oov else "the man."
        documents = (GumDocument("fixture-a", (text,)), GumDocument("fixture-b", (text,)))
        pairs = tuple(evaluate_documents(model, documents))
        if swap or mixed:
            pairs = tuple(
                dataclasses.replace(
                    pair, integrated=pair.disconnected, disconnected=pair.integrated
                )
                if swap or index == 1
                else pair
                for index, pair in enumerate(pairs)
            )
        population = GumPopulation(
            TEST_SHA256,
            PROTOCOL_SHA256,
            MANIFEST_SHA256,
            EXCLUDED_DOCUMENTS,
            tuple(document.document_id for document in documents),
        )
        ledger_path = directory / f"fixture-{oov}-{swap}-{mixed}.json"
        recorder.publish_measurement(
            pairs,
            {
                "src/ailm/levels/connected.py": hashlib.sha256(
                    (ROOT / "src/ailm/levels/connected.py").read_bytes()
                ).hexdigest()
            },
            ClaimsLedger(ledger_path),
            directory / f"report-{oov}-{swap}-{mixed}.json",
            population,
        )
        return json.loads(ledger_path.read_text())["claims"]

    return record


@pytest.mark.parametrize("swap,mixed", [(False, False), (True, False), (False, True)])
def test_connected_prose_accepts_either_direction_and_unresolved_intervals(
    recorded_fixture, swap, mixed
):
    claims = recorded_fixture(oov=True, swap=swap, mixed=mixed)
    assert set(claims) == set(CONNECTED_GUM_REQUIRED_CLAIMS)
    _validate_required_claim_values(claims)
    variables = _StrictVariableMap()
    _produce_connected_gum(variables, _ClaimReaders(claims))
    assert variables["CONNECTED_GUM_WORD_LOSS_TEXT"].startswith("infinite")
    if mixed:
        assert variables["CONNECTED_GUM_OUTCOME_TEXT"] == "an interval spanning zero"
    else:
        assert variables["CONNECTED_GUM_OUTCOME_TEXT"] in {
            "lower integrated loss",
            "higher integrated loss",
        }


def test_finite_word_population_and_each_missing_claim_are_distinguished(recorded_fixture):
    claims = recorded_fixture(oov=False)
    variables = _StrictVariableMap()
    _produce_connected_gum(variables, _ClaimReaders(claims))
    assert variables["CONNECTED_GUM_WORD_LOSS_TEXT"].startswith("finite")
    for label in CONNECTED_GUM_REQUIRED_CLAIMS:
        incomplete = dict(claims)
        incomplete.pop(label)
        with pytest.raises(MissingClaimError):
            _produce_connected_gum(_StrictVariableMap(), _ClaimReaders(incomplete))


@pytest.mark.parametrize(
    "suffix,value",
    [
        ("documents", 0),
        ("oov_tokens", -1),
        ("normalized_tokens", 2.5),
        ("in_vocab_tokens", 999),
        ("scored_characters", 0),
        ("vocab_coverage_pct", 0),
        ("word_loss_is_infinite", False),
        ("strength", 0.7),
        ("document_ci_low", 999),
        ("document_integrated_log_loss", -1),
        ("document_mean_gain", 999),
        ("outcome", "unsupported"),
        ("runtime_sha256", "not-a-digest"),
    ],
)
def test_incoherent_connected_claims_refuse_prose(recorded_fixture, suffix, value):
    claims = copy.deepcopy(recorded_fixture(oov=True))
    claims["connected_gum_" + suffix]["value"] = value
    with pytest.raises(ManuscriptVariableError):
        _produce_connected_gum(_StrictVariableMap(), _ClaimReaders(claims))


def test_mixed_run_connected_claims_refuse_prose(recorded_fixture):
    claims = recorded_fixture(oov=True)
    claims["connected_gum_outcome"]["run_id"] = "different-run"
    with pytest.raises(ManuscriptVariableError, match="not all bound"):
        _produce_connected_gum(_StrictVariableMap(), _ClaimReaders(claims))
