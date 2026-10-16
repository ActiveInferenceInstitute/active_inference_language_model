"""Edge contracts of the CMUdict reader that the inventory tests do not reach.

Covers the line parser's short-line refusal, the naive comment audit's
single-field lines, the pinned reader's path handling without any network,
the mirror-identity refusal, and the status probe's success path on a file
that carries every canonical phone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ailm.corpus.cmudict import (
    CANONICAL_ARPABET,
    CMUDICT_RAW_SHA256,
    PinnedCMUDictSource,
    _parse_line,
    assert_full_inventory,
    audit_comment_stripping_bytes,
    load_cmudict,
    load_pinned_cmudict_source,
    verify_cmudict,
)
from ailm.resources import resolve_raw_corpus_path


def test_a_line_with_no_pronunciation_is_skipped() -> None:
    assert _parse_line("HELLO") is None
    assert _parse_line("HELLO  # comment only") is None


def test_the_naive_audit_ignores_single_field_lines() -> None:
    audit = audit_comment_stripping_bytes(b"HEADER\nCAT K AE1 T\n")
    assert "K" in audit.naive_inventory
    assert "HEADER" not in audit.naive_inventory


def test_the_full_inventory_check_accepts_the_canonical_set() -> None:
    assert assert_full_inventory(CANONICAL_ARPABET) is None


def test_pinned_reader_defaults_to_the_raw_corpus_path_without_downloading() -> None:
    default = resolve_raw_corpus_path("cmudict.dict")
    pinned = (
        default.is_file() and hashlib.sha256(default.read_bytes()).hexdigest() == CMUDICT_RAW_SHA256
    )
    if pinned:
        source = load_pinned_cmudict_source(download=False)
        assert isinstance(source, PinnedCMUDictSource)
        assert source.content == default.read_bytes()
    else:
        with pytest.raises(FileNotFoundError, match="Run with download=True"):
            load_pinned_cmudict_source(download=False)


def test_pinned_reader_refuses_a_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not a file"):
        load_pinned_cmudict_source(tmp_path, download=False)


def test_loader_refuses_a_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not a file"):
        load_cmudict(tmp_path, download=False)


def test_loader_refuses_a_mirror_file_whose_identity_disagrees(tmp_path) -> None:
    mirror = tmp_path / "mirror.dict"
    mirror.write_text("cat K AE1 T\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the declared identity"):
        load_cmudict(
            mirror,
            download=False,
            expected_sha256="0" * 64,
            expected_size_bytes=mirror.stat().st_size,
        )


def _write_full_inventory_file(path: Path) -> None:
    lines = [f"w{index} {phone}" for index, phone in enumerate(sorted(CANONICAL_ARPABET))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_status_probe_reports_ok_with_samples_for_a_complete_inventory(tmp_path) -> None:
    complete = tmp_path / "complete.dict"
    _write_full_inventory_file(complete)
    status = verify_cmudict(complete)
    assert status["status"] == "ok"
    assert status["entry_count"] == len(CANONICAL_ARPABET)
    assert len(status["sample_entries"]) == 5
    assert set(status["phoneme_inventory"]) == CANONICAL_ARPABET
