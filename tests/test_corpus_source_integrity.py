"""Fail-closed raw-source controls for Brown and Czech-PDTC writers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ailm.corpus.brown import (
    BROWN_NONCANONICAL_ARTIFACT_SCHEMA,
    BROWN_RAW_SHA256,
    BROWN_RAW_SIZE_BYTES,
    BROWN_RAW_URL,
    BROWN_REVISION,
    BrownSourceIntegrityError,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    brown_source_metadata,
    build_brown_document_partition,
    category_stratified_document_indices,
    load_brown_corpus,
    resolve_brown_source_zip,
    validate_brown_artifact_provenance,
    validate_brown_source_metadata,
    verify_brown_source_bytes,
)
from ailm.corpus.ud_czech import (
    UD_CZECH_PDTC_REVISION,
    UD_CZECH_PDTC_SHARDS,
    UDCzechSourceIntegrityError,
    read_verified_ud_czech_shard,
    ud_czech_source_metadata,
    validate_ud_czech_source_metadata,
)
from ailm.levels.letter import load_brown_counts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_brown_source_declaration_is_complete_and_immutable():
    metadata = brown_source_metadata()
    validate_brown_source_metadata(metadata)
    assert metadata["source_revision"] == BROWN_REVISION
    assert metadata["source_url"] == BROWN_RAW_URL
    assert metadata["source_sha256"] == BROWN_RAW_SHA256
    assert metadata["source_size_bytes"] == BROWN_RAW_SIZE_BYTES
    assert BROWN_REVISION in BROWN_RAW_URL
    assert not {"main", "master", "gh-pages"} & set(BROWN_RAW_URL.split("/"))


def test_brown_source_validator_rejects_wrong_extent_and_same_extent_corruption():
    with pytest.raises(BrownSourceIntegrityError, match="does not match"):
        verify_brown_source_bytes(b"truncated")

    same_extent_wrong_bytes = bytes(BROWN_RAW_SIZE_BYTES)
    with pytest.raises(BrownSourceIntegrityError, match="does not match"):
        verify_brown_source_bytes(same_extent_wrong_bytes)


@pytest.mark.raw_corpus_regeneration
def test_brown_reader_keeps_authenticated_bytes_after_path_replacement(tmp_path):
    """NEGATIVE CONTROL: verification must not precede a lazy path reopen."""
    try:
        canonical = resolve_brown_source_zip()
    except FileNotFoundError:
        pytest.skip("pinned Brown archive absent; raw-source control cannot run offline")

    attack_target = tmp_path / "brown.zip"
    attack_target.write_bytes(canonical.read_bytes())
    reader = load_brown_corpus(attack_target)

    # A path is mutable even after a successful digest check.  The reader must
    # already own the authenticated snapshot when a later lazy member is read.
    attack_target.write_bytes(b"atomically replaced after verification")
    assert reader.words(fileids="ca01")[:3] == ["The", "Fulton", "County"]
    assert reader.sents(categories="news")[0][:3] == ["The", "Fulton", "County"]


def test_brown_metadata_rejects_moving_source_and_missing_revision():
    moving = brown_source_metadata()
    moving["source_url"] = (
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
    )
    with pytest.raises(BrownSourceIntegrityError, match="disagrees with the pin"):
        validate_brown_source_metadata(moving)

    incomplete = brown_source_metadata()
    incomplete.pop("source_revision")
    with pytest.raises(BrownSourceIntegrityError, match="lacks pinned"):
        validate_brown_source_metadata(incomplete)

    ambiguous = brown_source_metadata()
    ambiguous["effective_source_url"] = "https://example.invalid/moving.zip"
    with pytest.raises(BrownSourceIntegrityError, match="unexpected raw-source metadata"):
        validate_brown_source_metadata(ambiguous)


def test_brown_artifact_validator_requires_nested_raw_source():
    with pytest.raises(BrownSourceIntegrityError, match="raw_source"):
        validate_brown_artifact_provenance({"provenance": {"corpus": "NLTK Brown"}})


def test_brown_document_split_is_stratified_deterministic_and_nonvacuous():
    ids = ["news-a", "news-b", "fiction-a", "fiction-b"]
    categories = ["news", "news", "fiction", "fiction"]
    first = category_stratified_document_indices(ids, categories, 0.9)
    assert first == category_stratified_document_indices(ids, categories, 0.9)
    train, heldout = first
    assert set(train).isdisjoint(heldout)
    assert set(train) | set(heldout) == set(range(len(ids)))
    for category in set(categories):
        indices = {index for index, value in enumerate(categories) if value == category}
        assert indices & set(train)
        assert indices & set(heldout)
    with pytest.raises(ValueError, match="at least two documents"):
        category_stratified_document_indices(["only"], ["news"], 0.9)
    for fraction in (True, 0.0, 1.0, float("nan"), "0.9"):
        with pytest.raises(ValueError, match="train_fraction"):
            category_stratified_document_indices(ids, categories, fraction)


def test_brown_document_split_uses_nearest_count_with_declared_half_up_tie():
    """A one-heldout-per-category stub must not satisfy the split contract."""
    ids = [f"news-{index:02d}" for index in range(15)]
    categories = ["news"] * len(ids)

    train_90, heldout_90 = category_stratified_document_indices(ids, categories, 0.9)
    assert len(train_90) == 13
    assert len(heldout_90) == 2, "1.5 held-out documents rounds half-up to two"

    train_80, heldout_80 = category_stratified_document_indices(ids, categories, 0.8)
    assert len(train_80) == 12
    assert len(heldout_80) == 3
    assert set(heldout_90) < set(heldout_80), (
        "changing the target must extend the same revision-bound ordering"
    )


def test_brown_document_identity_digest_binds_order_and_rejects_aliases():
    forward = brown_document_identity_sha256(["ca01", "ca02"])
    assert forward != brown_document_identity_sha256(["ca02", "ca01"])
    with pytest.raises(ValueError, match="duplicate"):
        brown_document_identity_sha256(["ca01", "ca01"])
    partition = brown_partition_input_sha256(["ca01", "ca02"], ["news", "news"])
    assert partition != brown_partition_input_sha256(["ca01", "ca02"], ["editorial", "editorial"])


@pytest.mark.raw_corpus_regeneration
def test_brown_partition_never_splits_a_source_document():
    try:
        source_zip = resolve_brown_source_zip()
    except FileNotFoundError:
        pytest.skip("pinned Brown archive absent; raw-source partition cannot run offline")
    partition = build_brown_document_partition(0.9, source_zip)
    assert set(partition.train_indices).isdisjoint(partition.heldout_indices)
    assert set(partition.train_indices) | set(partition.heldout_indices) == set(
        range(len(partition.document_ids))
    )
    assert len(partition.heldout_tokens) == len(partition.heldout_token_document_indices)
    assert set(partition.heldout_token_document_indices) == set(partition.heldout_indices)
    for category in set(partition.categories):
        category_indices = {
            index for index, value in enumerate(partition.categories) if value == category
        }
        assert category_indices & set(partition.train_indices)
        assert category_indices & set(partition.heldout_indices)
    assert len(partition.train_tokens) == sum(
        partition.alphabetic_token_counts[index] for index in partition.train_indices
    )
    assert len(partition.heldout_tokens) == sum(
        partition.alphabetic_token_counts[index] for index in partition.heldout_indices
    )


def test_canonical_brown_writer_rejects_corrupt_raw_bytes_without_output(tmp_path):
    raw = tmp_path / "brown.zip"
    raw.write_bytes(b"not the pinned archive")
    output = tmp_path / "candidate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_counts.py"),
            "--source-zip",
            str(raw),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "REFUSING TO WRITE" in result.stderr
    assert not output.exists()


def test_fallback_cannot_masquerade_as_canonical_brown_artifact(tmp_path):
    canonical_name = tmp_path / "brown_counts.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_counts.py"),
            "--force-fallback",
            "--output",
            str(canonical_name),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "noncanonical" in result.stderr
    assert not canonical_name.exists()


def test_explicit_fallback_is_labeled_and_rejected_by_production_loader(tmp_path):
    diagnostic = tmp_path / "noncanonical-brown-diagnostic.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_counts.py"),
            "--force-fallback",
            "--output",
            str(diagnostic),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert payload["schema"] == BROWN_NONCANONICAL_ARTIFACT_SCHEMA
    assert payload["provenance"]["canonical"] is False

    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "brown_counts.json").write_bytes(diagnostic.read_bytes())
    with pytest.raises(ValueError, match="schema"):
        load_brown_counts(vendor)


def test_czech_source_declaration_binds_both_exact_shards():
    metadata = ud_czech_source_metadata()
    validate_ud_czech_source_metadata(metadata)
    assert metadata["source_revision"] == UD_CZECH_PDTC_REVISION
    assert set(metadata["shards"]) == set(UD_CZECH_PDTC_SHARDS)
    for split, shard in UD_CZECH_PDTC_SHARDS.items():
        embedded = metadata["shards"][split]
        assert embedded == shard.metadata()
        assert UD_CZECH_PDTC_REVISION in embedded["source_url"]


def test_czech_metadata_rejects_moving_url_and_missing_shard():
    moving = ud_czech_source_metadata()
    moving["shards"]["test"]["source_url"] = (
        "https://raw.githubusercontent.com/UniversalDependencies/"
        "UD_Czech-PDTC/master/cs_pdtc-ud-dev.conllu"
    )
    with pytest.raises(UDCzechSourceIntegrityError, match="disagrees with the pin"):
        validate_ud_czech_source_metadata(moving)

    incomplete = ud_czech_source_metadata()
    incomplete["shards"].pop("train")
    with pytest.raises(UDCzechSourceIntegrityError, match="disagrees with the pin"):
        validate_ud_czech_source_metadata(incomplete)


def test_czech_shard_reader_rejects_missing_modified_and_wrong_filename(tmp_path):
    expected = UD_CZECH_PDTC_SHARDS["train"]
    missing = tmp_path / expected.filename
    with pytest.raises(FileNotFoundError, match="absent"):
        read_verified_ud_czech_shard(missing, "train")

    missing.write_bytes(b"modified")
    with pytest.raises(UDCzechSourceIntegrityError, match="does not match revision"):
        read_verified_ud_czech_shard(missing, "train")

    wrong_name = tmp_path / "moving-branch-download.conllu"
    wrong_name.write_bytes(b"modified")
    with pytest.raises(UDCzechSourceIntegrityError, match="wrong filename"):
        read_verified_ud_czech_shard(wrong_name, "train")


def test_czech_generator_failure_preserves_existing_output(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    for shard in UD_CZECH_PDTC_SHARDS.values():
        (staging / shard.filename).write_bytes(b"modified")
    output = tmp_path / "ud_czech_case_counts.json"
    sentinel = b"existing artifact remains intact\n"
    output.write_bytes(sentinel)

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_czech_case_counts.py"),
            "--staging-dir",
            str(staging),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert output.read_bytes() == sentinel
    assert not list(tmp_path.glob(".ud_czech_case_counts.json.*.tmp"))
