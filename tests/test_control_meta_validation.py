"""Meta-validated negative controls.

Each negative control in the test suite must fail for the RIGHT structural reason,
not because its own construction has a bug. This file verifies, for each negative
control, that the control's failure mechanism is the expected one — and logs a
warning rather than an error when only the failure outcome can be checked.

The registry covers every gate in :data:`ailm.gates.AUDITED_GATE_IDS` exactly
once. This makes a newly audited stage fail until it names a representative
load-bearing negative control.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from ailm.gates import AUDITED_GATE_IDS, GateId

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 1. NEGATIVE CONTROL REGISTRY ──────────────────────────────────────────────

# Each entry names both the scientific failure reason and structural symbols
# that prove the target test constructs and evaluates that failure.  Names and
# prose alone are not validation: a ``pass`` function with the old name used to
# satisfy this entire file.

NEGATIVE_CONTROLS: tuple[dict[str, object], ...] = (
    {
        "gate_id": GateId.STAGE1_LETTER,
        "name": "test_stage1_negative_control_uniform_model",
        "file": "test_stage1_gate.py",
        "reason": "uniform letter model cannot beat unigram context-free baseline",
        "required_symbols": ("uniform_perplexity", "GATE_THRESHOLD", "UNIFORM_BASELINE"),
    },
    {
        "gate_id": GateId.STAGE2_PHONEME,
        "name": "test_stage2_phoneme_negative_control_uniform_transition",
        "file": "test_stage2_gate.py",
        "reason": "uniform B destroys sequential phone structure and reaches uniform perplexity",
        "required_symbols": (
            "pymdp_uniform_sample",
            "unigram_sample",
            "verify_negative_control_fails",
        ),
    },
    {
        "gate_id": GateId.STAGE2_G2P,
        "name": "test_stage2_g2p_negative_control_uniform_emission",
        "file": "test_stage2_gate.py",
        "reason": "uniform A carries no letter-given-phone information and collapses accuracy",
        "required_symbols": (
            "uniform_emission",
            "build_g2p_agent",
            "verify_negative_control_fails",
        ),
    },
    {
        "gate_id": GateId.STAGE3_WORD,
        "name": "test_time_reversed_emission_is_caught_by_the_bar",
        "file": "test_stage3_gate.py",
        "reason": "time-reversed spelling emissions destroy the word-path likelihood",
        "required_symbols": ("reversed_paths", "letter_bigram_perplexity", "reversed_ppl"),
    },
    {
        "gate_id": GateId.STAGE4_CASE,
        "name": "test_uniform_emission_collapses_agreement",
        "file": "test_stage4_gate.py",
        "reason": "uniform UPOS-given-case emission removes the case-recognition signal",
        "required_symbols": ("full_like", "decode_case_positions", "case_agreement"),
    },
    {
        "gate_id": GateId.GENERATION,
        "name": "test_degenerate_precision_fails_the_composite_generation_gate",
        "file": "test_generation.py",
        "reason": "high precision beats the TV bar while collapsing realized diversity",
        "required_symbols": ("evaluate_generation_gate", "tv_passes", "diversity_passes"),
    },
    {
        "gate_id": GateId.STAGE5_CLAUSE,
        "name": "test_swapped_type_blocks_destroy_the_synthetic_classification",
        "file": "test_stage5_gate.py",
        "reason": "swapped sentence-type transition blocks destroy a planted detectable effect",
        "required_symbols": ("transitions", "classify_sentences", "accuracy"),
    },
    {
        "gate_id": GateId.STAGE6_NARRATIVE,
        "name": "test_phase_blind_negative_control_cannot_clear_the_recognition_gate",
        "file": "test_stage6_gate.py",
        "reason": "identical phase columns remove document-position information exactly",
        "required_symbols": ("repeat", "heldout_type_logloss", "verify_negative_control_fails"),
    },
)


def test_all_negative_controls_have_meta_validation():
    """Every known negative control must have a documented failure reason."""
    names_seen: dict[str, dict[str, object]] = {}
    for ctrl in NEGATIVE_CONTROLS:
        name = str(ctrl["name"])
        assert name not in names_seen, f"Duplicate negative control: {name}"
        names_seen[name] = ctrl
        assert len(str(ctrl["reason"])) > 10, f"{name}: reason too short"
        required = ctrl["required_symbols"]
        assert isinstance(required, tuple) and len(required) >= 3
        assert all(isinstance(symbol, str) and symbol for symbol in required)
    assert len(names_seen) >= len(NEGATIVE_CONTROLS), "All controls must have unique names"


def test_negative_control_registry_covers_every_audited_gate_exactly_once():
    observed = tuple(ctrl["gate_id"] for ctrl in NEGATIVE_CONTROLS)
    assert observed == AUDITED_GATE_IDS
    assert len(observed) == len(set(observed))


def test_negative_control_file_paths_are_plausible():
    """Each negative control's file must exist in the test directory."""
    for ctrl in NEGATIVE_CONTROLS:
        file = str(ctrl["file"])
        path = PROJECT_ROOT / "tests" / file
        assert path.exists(), f"Negative control file not found: {path}"


def _function_symbols(node: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Return identifiers, attributes, keyword names, and string keys used."""
    symbols: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            symbols.add(child.id)
        elif isinstance(child, ast.Attribute):
            symbols.add(child.attr)
        elif isinstance(child, ast.keyword) and child.arg is not None:
            symbols.add(child.arg)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            symbols.add(child.value)
    return frozenset(symbols)


def _control_structure_errors(
    source: str, name: str, required_symbols: tuple[str, ...]
) -> tuple[str, ...]:
    """Reject absent, no-op, non-asserting, or structurally unrelated controls."""
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        return (f"expected exactly one function named {name}, found {len(matches)}",)

    node = matches[0]
    errors: list[str] = []
    if any(isinstance(child, ast.Pass) for child in ast.walk(node)):
        errors.append("contains pass")
    meaningful_asserts = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Assert)
        and not (isinstance(child.test, ast.Constant) and child.test.value is True)
    ]
    if not meaningful_asserts:
        errors.append("contains no non-constant assertion")
    if not any(isinstance(child, ast.Call) for child in ast.walk(node)):
        errors.append("never calls a metric, constructor, or harness")

    symbols = _function_symbols(node)
    missing = sorted(set(required_symbols) - symbols)
    if missing:
        errors.append(f"missing load-bearing symbols {missing}")
    return tuple(errors)


def test_negative_control_functions_are_structurally_load_bearing():
    """Every registered function must construct, score, and assert its defect."""
    for ctrl in NEGATIVE_CONTROLS:
        file = str(ctrl["file"])
        name = str(ctrl["name"])
        target_path = PROJECT_ROOT / "tests" / file
        errors = _control_structure_errors(
            target_path.read_text(encoding="utf-8"),
            name,
            tuple(str(symbol) for symbol in ctrl["required_symbols"]),
        )
        assert errors == (), f"negative control {file}::{name} is vacuous: {errors}"


def test_structural_validator_rejects_named_noop_and_assert_true_controls():
    """NEGATIVE CONTROL: names and token mentions cannot make a no-op valid."""
    required = ("broken_input", "production_metric", "gate_threshold")
    planted = """
def test_planted():
    broken_input = production_metric = gate_threshold = None
    assert True
    pass
"""
    errors = _control_structure_errors(planted, "test_planted", required)
    assert "contains pass" in errors
    assert "contains no non-constant assertion" in errors


@pytest.mark.timeout(1920)
def test_registered_negative_controls_execute_and_do_not_skip(tmp_path):
    """Run every declared control through pytest and inspect its exact receipt."""
    nodeids = [f"tests/{ctrl['file']}::{ctrl['name']}" for ctrl in NEGATIVE_CONTROLS]
    junit = tmp_path / "negative-controls.xml"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *nodeids,
            "-q",
            "--tb=short",
            f"--junitxml={junit}",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=960,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert junit.is_file(), f"pytest emitted no JUnit receipt:\n{output}"

    cases = ET.parse(junit).findall(".//testcase")
    assert len(cases) == len(NEGATIVE_CONTROLS), output
    nonpasses = [
        case.attrib.get("name", "unknown")
        for case in cases
        if case.find("failure") is not None
        or case.find("error") is not None
        or case.find("skipped") is not None
    ]
    assert nonpasses == [], f"registered controls did not pass cleanly: {nonpasses}\n{output}"
