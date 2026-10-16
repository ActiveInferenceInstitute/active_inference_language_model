"""Rejection contracts of the Brown corpus identity and source-resolution helpers.

The digests here are what bind every Brown-derived artifact to an ordered,
categorized set of source documents. Each rejection below is a way a relabeled
or malformed identity list could otherwise produce a valid-looking digest.
The source-resolution tests use the pinned archive only when it is present on
this machine and byte-identical to the pin; they skip otherwise rather than
fetch anything.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ailm.corpus.brown import (
    BROWN_RAW_SHA256,
    BROWN_SOURCE_ENV,
    BrownSourceIntegrityError,
    _default_source_candidates,
    _read_verified_brown_source,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    category_stratified_document_indices,
    validate_brown_artifact_provenance,
    verify_brown_source_zip,
)


def _pinned_archive() -> Path | None:
    """The local pinned archive, or None when it is absent or not the pin."""
    for candidate in _default_source_candidates():
        if candidate.is_file():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest == BROWN_RAW_SHA256:
                return candidate.resolve()
    return None


# -- identity digests --------------------------------------------------------


def test_identity_digest_refuses_an_empty_id_list() -> None:
    with pytest.raises(ValueError, match="needs at least one id"):
        brown_document_identity_sha256([])


@pytest.mark.parametrize("ids", [["ca01", ""], ["ca01", 3], ["ca01", "ca\x0002"]])
def test_identity_digest_refuses_an_invalid_id(ids) -> None:
    with pytest.raises(ValueError, match="invalid Brown document id"):
        brown_document_identity_sha256(ids)


def test_partition_digest_refuses_misaligned_categories() -> None:
    with pytest.raises(ValueError, match="categories must align one-for-one"):
        brown_partition_input_sha256(["ca01", "ca02"], ["news"])


@pytest.mark.parametrize("category", ["", "ne\x00ws", 7])
def test_partition_digest_refuses_an_invalid_category(category) -> None:
    with pytest.raises(ValueError, match="invalid Brown partition input"):
        brown_partition_input_sha256(["ca01"], [category])


# -- the stratified split ----------------------------------------------------


def test_split_refuses_misaligned_categories() -> None:
    with pytest.raises(ValueError, match="categories must align one-for-one"):
        category_stratified_document_indices(["ca01", "ca02"], ["news"], 0.9)


@pytest.mark.parametrize("category", ["", "ne\x00ws", 7])
def test_split_refuses_an_invalid_category(category) -> None:
    with pytest.raises(ValueError, match="invalid Brown category at index 1"):
        category_stratified_document_indices(["ca01", "ca02"], ["news", category], 0.9)


# -- artifact provenance -----------------------------------------------------


@pytest.mark.parametrize("payload", [{}, {"provenance": None}, {"provenance": []}])
def test_artifact_provenance_must_be_an_object(payload) -> None:
    with pytest.raises(BrownSourceIntegrityError, match="lacks a provenance object"):
        validate_brown_artifact_provenance(payload)


# -- source resolution -------------------------------------------------------


def test_source_zip_check_refuses_a_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="absent or not a file"):
        verify_brown_source_zip(tmp_path / "brown.zip")


def test_verified_reader_refuses_an_explicit_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="absent or not a file"):
        _read_verified_brown_source(tmp_path / "brown.zip")


def test_source_zip_check_returns_the_resolved_pinned_archive() -> None:
    archive = _pinned_archive()
    if archive is None:
        pytest.skip("the pinned Brown archive is not present on this machine")
    assert verify_brown_source_zip(archive) == archive


def test_default_resolution_skips_an_absent_configured_candidate(monkeypatch, tmp_path) -> None:
    """A configured path that does not exist is passed over, not reported as corrupt."""
    monkeypatch.setenv(BROWN_SOURCE_ENV, str(tmp_path / "absent.zip"))
    assert _default_source_candidates()[0] == tmp_path / "absent.zip"
    archive = _pinned_archive()
    if archive is None:
        with pytest.raises(FileNotFoundError, match="brown.zip is unavailable") as excinfo:
            _read_verified_brown_source()
        assert "Invalid candidates" not in str(excinfo.value)
    else:
        resolved, content = _read_verified_brown_source()
        assert resolved == archive
        assert hashlib.sha256(content).hexdigest() == BROWN_RAW_SHA256


def test_default_resolution_records_a_configured_candidate_that_is_not_a_file(
    monkeypatch, tmp_path
) -> None:
    """A directory at the configured path is diagnosed, then the search continues."""
    monkeypatch.setenv(BROWN_SOURCE_ENV, str(tmp_path))
    archive = _pinned_archive()
    if archive is None:
        with pytest.raises(FileNotFoundError, match="Invalid candidates") as excinfo:
            _read_verified_brown_source()
        assert str(tmp_path) in str(excinfo.value)
    else:
        resolved, _content = _read_verified_brown_source()
        assert resolved == archive
