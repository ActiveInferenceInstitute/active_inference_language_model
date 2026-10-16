"""Integrity controls for the canonical Stage 1 evaluation population."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from ailm.benchmarks.constants import HELD_OUT_TEXT
from ailm.evaluation import (
    STAGE1_EVALUATION,
    STAGE1_EVALUATION_TEXT,
    EvaluationFixtureError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_stage1_fixture_has_the_pinned_protocol_identity() -> None:
    assert STAGE1_EVALUATION.protocol == "ailm-stage1-letter-evaluation-v1"
    assert (
        STAGE1_EVALUATION.sha256
        == "513f73d02c2220185daf0a30865bed07a9af2167b40dcf5bb4341617efc6c20f"
    )
    encoded = STAGE1_EVALUATION_TEXT.encode("utf-8")
    assert len(encoded) == STAGE1_EVALUATION.size_bytes
    assert hashlib.sha256(encoded).hexdigest() == STAGE1_EVALUATION.sha256


def test_stage1_fixture_declares_source_and_license() -> None:
    metadata = STAGE1_EVALUATION.metadata()
    assert metadata["source"]
    assert "CC BY-SA 4.0" in str(metadata["license"])
    assert "public domain" in str(metadata["license"]).lower()
    assert len(metadata["source_references"]) >= 5


def test_stage1_fixture_repetition_is_explicit_and_exact() -> None:
    repetitions = STAGE1_EVALUATION.repetitions
    assert repetitions == 3
    assert len(STAGE1_EVALUATION_TEXT) % repetitions == 0
    period = len(STAGE1_EVALUATION_TEXT) // repetitions
    chunks = {
        STAGE1_EVALUATION_TEXT[index : index + period]
        for index in range(0, len(STAGE1_EVALUATION_TEXT), period)
    }
    assert len(chunks) == 1


@pytest.mark.parametrize(
    "broken",
    (
        replace(STAGE1_EVALUATION, text=STAGE1_EVALUATION.text + "x"),
        replace(STAGE1_EVALUATION, sha256="0" * 64),
        replace(STAGE1_EVALUATION, source=""),
        replace(STAGE1_EVALUATION, license=""),
        replace(STAGE1_EVALUATION, source_references=()),
    ),
)
def test_modified_or_incompletely_sourced_fixture_fails_closed(broken) -> None:
    with pytest.raises(EvaluationFixtureError):
        broken.validate()


def test_benchmark_compatibility_name_is_the_canonical_fixture() -> None:
    assert HELD_OUT_TEXT is STAGE1_EVALUATION_TEXT


def test_canonical_module_is_the_only_owner_of_the_fixture_prose() -> None:
    sentinel = "how much wood would " + "a woodchuck chuck if " + "a woodchuck could chuck wood"
    owners = []
    for directory in ("src", "scripts", "tests"):
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            if sentinel in path.read_text(encoding="utf-8"):
                owners.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert owners == ["src/ailm/evaluation.py"]


def test_production_consumers_do_not_import_the_gate_test() -> None:
    offenders = []
    for path in (PROJECT_ROOT / "scripts").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "test_stage1_gate import HELD_OUT_TEXT" in text:
            offenders.append(path.name)
    assert offenders == []
