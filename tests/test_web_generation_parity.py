"""Standalone-web generation parity, including a prior-drift control.

The page is intentionally a JavaScript implementation of the shipped arithmetic.
These tests run that implementation under Node, without a browser or DOM, and
compare its fixed-prefix policy posteriors with probes produced by Python.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ailm.levels.word import build_word_prior, load_word_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_MODEL = PROJECT_ROOT / "output" / "web" / "ailm_model.json"
WEB_PAGE = PROJECT_ROOT / "output" / "web" / "ailm.html"
WORD_ARTIFACT = PROJECT_ROOT / "vendor" / "brown_word_counts.json"
FIXED_PREFIXES = {"", "t", "th", "the", "co", "s"}
HIGH_GAMMA = 16.0

NODE_RUNNER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const runtimePath = process.argv[2];
const modelPath = process.argv[3];
globalThis.AILM_MODEL_JSON = fs.readFileSync(modelPath, "utf8");
vm.runInThisContext(fs.readFileSync(runtimePath, "utf8"), { filename: runtimePath });
const api = globalThis.AILMWebModel;
const diagnostics = api.modelDiagnostics();
process.stdout.write(JSON.stringify({
  diagnostics,
  highGammaWord: api.greedyWord(16),
  highGammaProbability: api.wordPathProbability(api.greedyWord(16), 16),
}));
"""


def _web_model() -> dict[str, object]:
    assert WEB_MODEL.is_file(), "run scripts/export_web_model.py before the web parity tests"
    value = json.loads(WEB_MODEL.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_script() -> str:
    assert WEB_PAGE.is_file(), "run scripts/build_web_ailm.py before the web parity tests"
    scripts = re.findall(
        r"<script(?:\s[^>]*)?>(.*?)</script>",
        WEB_PAGE.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    assert len(scripts) == 2, "the standalone page no longer has one model and one runtime script"
    return scripts[-1]


def _javascript_process(
    model: dict[str, object], tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    assert node is not None, "Node is required to execute the browser-independent JS parity gate"

    runtime_path = tmp_path / "ailm-runtime.js"
    model_path = tmp_path / "ailm-model.json"
    runner_path = tmp_path / "run-ailm.js"
    runtime_path.write_text(_runtime_script(), encoding="utf-8")
    model_path.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    runner_path.write_text(NODE_RUNNER, encoding="utf-8")
    return subprocess.run(
        [node, runner_path, runtime_path, model_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_javascript_model(model: dict[str, object], tmp_path: Path) -> dict[str, object]:
    result = _javascript_process(model, tmp_path)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


def test_exported_word_prior_is_exactly_aligned_and_digest_bound() -> None:
    model = _web_model()
    artifact = load_word_artifact(WORD_ARTIFACT)
    shipped_prior = build_word_prior(artifact)

    assert model["schema"] == "ailm-web-model-v2"
    assert model["lexicon"] == artifact.vocab
    exported_prior = np.asarray(model["word_prior"], dtype=np.float64)
    np.testing.assert_array_equal(exported_prior, shipped_prior)

    provenance = model["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["word_artifact"] == "vendor/brown_word_counts.json"
    assert (
        provenance["word_artifact_sha256"] == hashlib.sha256(WORD_ARTIFACT.read_bytes()).hexdigest()
    )
    bound_value = {"lexicon": model["lexicon"], "word_prior": model["word_prior"]}
    assert provenance["lexicon_word_prior_sha256"] == _canonical_sha256(bound_value)


def test_fixed_prefix_probes_cover_the_operating_and_high_gamma_endpoints() -> None:
    model = _web_model()
    claims = model["claims"]
    probes = model["policy_parity_probes"]
    alphabet = model["alphabet"]
    assert isinstance(claims, dict)
    assert isinstance(probes, list)
    assert isinstance(alphabet, list)

    observed = {(probe["prefix"], float(probe["gamma"])) for probe in probes}
    expected_gammas = {float(claims["generation_best_gamma"]), HIGH_GAMMA}
    assert observed == {(prefix, gamma) for prefix in FIXED_PREFIXES for gamma in expected_gammas}
    assert all(probe["current_letter"] == (probe["prefix"][-1:] or " ") for probe in probes)
    assert all(len(probe["posterior"]) == len(alphabet) for probe in probes)


def test_javascript_policy_posteriors_match_python_and_show_high_gamma_collapse(
    tmp_path: Path,
) -> None:
    result = _run_javascript_model(_web_model(), tmp_path)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    parity = diagnostics["parity"]
    assert isinstance(parity, dict)
    assert diagnostics["ok"] is True
    assert parity["ok"] is True
    assert float(parity["maxAbs"]) <= 2e-6
    assert float(diagnostics["gammaEffect"]) > 1e-3
    assert diagnostics["highGammaCollapse"] is True
    assert result["highGammaWord"] == "the"
    high_gamma_probability = result["highGammaProbability"]
    assert isinstance(high_gamma_probability, (int, float))
    assert float(high_gamma_probability) > 0.99


def test_a_planted_prior_alignment_error_fails_the_javascript_parity_gate(tmp_path: Path) -> None:
    model = json.loads(json.dumps(_web_model()))
    lexicon = model["lexicon"]
    prior = model["word_prior"]
    assert isinstance(lexicon, list)
    assert isinstance(prior, list)

    first = 0
    second = next(
        index
        for index, word in enumerate(lexicon[1:], start=1)
        if word[0] != lexicon[first][0] and prior[index] != prior[first]
    )
    prior[first], prior[second] = prior[second], prior[first]

    result = _run_javascript_model(model, tmp_path)
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics, dict)
    parity = diagnostics["parity"]
    assert isinstance(parity, dict)
    assert parity["ok"] is False, "the planted prior misalignment escaped the JS parity gate"
    assert diagnostics["ok"] is False
    assert float(parity["maxAbs"]) > 2e-6


def _runtime_claim_references() -> list[str]:
    """Exercise every direct claim read, including future additions to the page."""
    references = sorted(set(re.findall(r"\bC\.([a-z][a-z0-9_]*)", _runtime_script())))
    assert references, "the runtime claim inventory must not be empty"
    return references


@pytest.mark.parametrize("label", _runtime_claim_references())
def test_missing_displayed_claim_refuses_initialization(label: str, tmp_path: Path) -> None:
    model = _web_model()
    claims = model["claims"]
    assert isinstance(claims, dict)
    del claims[label]
    result = _javascript_process(model, tmp_path)
    assert result.returncode != 0, f"missing displayed claim {label} passed initialization"
    assert f"claim {label} must be a finite number" in result.stderr


@pytest.mark.parametrize("value", [None, True, "0"])
def test_invalid_displayed_claim_refuses_initialization(value: object, tmp_path: Path) -> None:
    model = _web_model()
    claims = model["claims"]
    assert isinstance(claims, dict)
    claims["generation_best_tv"] = value
    result = _javascript_process(model, tmp_path)
    assert result.returncode != 0, f"invalid displayed claim {value!r} passed initialization"
    assert "claim generation_best_tv must be a finite number" in result.stderr


def test_unscorable_evaluation_fixture_refuses_initialization(tmp_path: Path) -> None:
    model = _web_model()
    model["held_out_text"] = "a"
    result = _javascript_process(model, tmp_path)
    assert result.returncode != 0
    assert "evaluation fixture must contain at least two supported symbols" in result.stderr
