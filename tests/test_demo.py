"""Real-execution tests for the offline newcomer demo (:mod:`ailm.demo`).

No mocks: the gate-table tests read the actual committed
``output/reports/gate_audit.json``, and the decode tests build a real
``pymdp`` Agent from the actual vendored CMUdict artifact (marked
``slow_demo`` since that pays JAX's cold-compile cost).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ailm.demo import (
    decode_word_with_posteriors,
    format_decode_table,
    format_gate_table,
    load_gate_audit,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_EXPECTED_GATE_IDS = {
    "stage1_letter",
    "stage2_phoneme",
    "stage2_g2p",
    "stage3_word",
    "stage4_case",
    "generation",
    "stage5_clause",
    "stage6_narrative",
}


def test_load_gate_audit_reads_the_committed_report() -> None:
    rows = load_gate_audit(PROJECT_ROOT)
    assert {row["gate_id"] for row in rows} == _EXPECTED_GATE_IDS
    for row in rows:
        for key in ("model", "baseline", "ceiling", "ceiling_type", "verdict", "outcome"):
            assert key in row, f"gate row {row.get('gate_id')!r} is missing {key!r}"
        assert isinstance(row["model"], int | float)
        assert row["outcome"] in {"PASS", "FAIL"}


def test_load_gate_audit_missing_report_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Regenerate"):
        load_gate_audit(tmp_path)


def test_load_gate_audit_rejects_non_list_json(tmp_path: Path) -> None:
    report = tmp_path / "output" / "reports" / "gate_audit.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON list"):
        load_gate_audit(tmp_path)


def test_load_gate_audit_rejects_empty_list(tmp_path: Path) -> None:
    report = tmp_path / "output" / "reports" / "gate_audit.json"
    report.parent.mkdir(parents=True)
    report.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON list"):
        load_gate_audit(tmp_path)


def test_format_gate_table_renders_one_aligned_row_per_gate() -> None:
    rows = [
        {
            "gate_id": "synthetic_gate",
            "stage": "synthetic stage",
            "model": 1.5,
            "baseline": 2.0,
            "ceiling": 1.0,
            "ceiling_type": "model_class_bound",
            "verdict": "BRACKETED_BOUND",
            "outcome": "PASS",
        }
    ]
    table = format_gate_table(rows)
    lines = table.splitlines()
    assert len(lines) == 3  # header, rule, one data row
    for header in ("gate", "stage", "model", "baseline", "ceiling", "verdict", "outcome"):
        assert header in lines[0]
    assert "synthetic_gate" in lines[2]
    assert "synthetic stage" in lines[2]
    assert "model_class_bound" in lines[2]
    assert "BRACKETED_BOUND" in lines[2]
    assert "PASS" in lines[2]
    assert "1.5000" in lines[2]  # model value, rendered rather than typed literally


def test_decode_word_with_posteriors_rejects_empty_word() -> None:
    # ailm.levels.letter.normalize_text maps every out-of-alphabet character to
    # a space, and space is itself an ALPHABET symbol, so the empty string is
    # the only input that normalizes to nothing.
    with pytest.raises(ValueError, match="no ALPHABET-recognized letters"):
        decode_word_with_posteriors("")


@pytest.mark.parametrize("bad_top_k", [0, -1, True, 1.5])
def test_decode_word_with_posteriors_rejects_bad_top_k(bad_top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        decode_word_with_posteriors("cat", top_k=bad_top_k)  # type: ignore[arg-type]


@pytest.mark.slow_demo
def test_decode_word_with_posteriors_returns_well_formed_positions() -> None:
    from ailm.levels.phoneme import PHONES

    positions = decode_word_with_posteriors("cat", top_k=3)
    assert len(positions) == 3  # "cat" is three ALPHABET letters

    for index, (entry, letter) in enumerate(zip(positions, "cat", strict=True)):
        assert entry["position"] == index
        assert entry["letter"] == letter
        top_phones = entry["top_phones"]
        assert len(top_phones) == 3
        probabilities = [probability for _phone, probability in top_phones]
        # Ranked most-likely first.
        assert probabilities == sorted(probabilities, reverse=True)
        for phone, probability in top_phones:
            assert phone in PHONES
            assert 0.0 <= probability <= 1.0 + 1e-6


@pytest.mark.slow_demo
def test_decode_word_with_posteriors_honors_top_k() -> None:
    positions = decode_word_with_posteriors("a", top_k=5)
    assert len(positions) == 1
    assert len(positions[0]["top_phones"]) == 5


def test_format_decode_table_includes_every_position() -> None:
    positions = [
        {"position": 0, "letter": "c", "top_phones": [("K", 0.9), ("CH", 0.05)]},
        {"position": 1, "letter": "a", "top_phones": [("AE", 0.8), ("AH", 0.1)]},
    ]
    table = format_decode_table("ca", positions)
    assert "'ca'" in table
    assert "'c'" in table and "'a'" in table
    assert "K 0.900" in table
    assert "AE 0.800" in table


@pytest.mark.slow_demo
def test_main_runs_end_to_end_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["cat", "--top-k", "2"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "gate-verdict table" in captured.out
    assert "PASS" in captured.out or "FAIL" in captured.out
    assert "G2P decode of 'cat'" in captured.out


def test_main_reports_empty_word_with_exit_code_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([""])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "no ALPHABET-recognized letters" in captured.err
