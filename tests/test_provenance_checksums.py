"""Every checksum documented in prose must match the artifact on disk.

``docs/provenance/sources.md`` records a sha256 for each vendored corpus artifact
so a reader can confirm they hold the same bytes. That guarantee is only worth
anything if the recorded value tracks the file — and it does not do so by itself.
This project already shipped a drifted one: adding the ``phone_initials`` key to
``vendor/cmudict_counts.json`` changed its hash while the documented value stayed
behind, leaving prose that confidently asserted the wrong digest.

A stale checksum is worse than an absent one. An absent checksum tells a reader to
verify some other way; a wrong checksum tells them their correct file is corrupt.

Determinism is checked here too: the generators must reproduce byte-identical
artifacts, since a checksum on a non-deterministic artifact is meaningless.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ailm.corpus.brown import (
    BROWN_RAW_SHA256,
    BROWN_RAW_SIZE_BYTES,
    BROWN_RAW_URL,
    BROWN_REVISION,
    resolve_brown_source_zip,
    validate_brown_artifact_provenance,
)
from ailm.corpus.cmudict import (
    CMUDICT_RAW_SHA256,
    CMUDICT_REVISION,
    CMUDICT_URL,
    validate_cmudict_source_metadata,
)
from ailm.corpus.ud import (
    UD_EWT_RAW_SHA256,
    UD_EWT_RAW_SIZE_BYTES,
    UD_EWT_REVISION,
    UD_EWT_SOURCE_METADATA,
    validate_ud_ewt_source_metadata,
)
from ailm.corpus.ud_czech import (
    UD_CZECH_PDTC_REVISION,
    UD_CZECH_PDTC_SHARDS,
    ud_czech_source_metadata,
    validate_ud_czech_source_metadata,
)
from ailm.resources import RUNTIME_VENDOR_ARTIFACTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DOC = PROJECT_ROOT / "docs" / "provenance" / "sources.md"
VENDOR_DIR = PROJECT_ROOT / "vendor"

#: Artifacts whose checksums are documented, mapped to the generator that rebuilds
#: them. Keep in step with docs/provenance/sources.md.
DOCUMENTED_ARTIFACTS = {
    "brown_counts.json": "scripts/generate_counts.py",
    "brown_heldout_stream.json": "scripts/generate_heldout_stream.py",
    "cmudict_counts.json": "scripts/generate_phoneme_counts.py",
    "brown_word_counts.json": "scripts/generate_word_counts.py",
    "ud_ewt_case_counts.json": "scripts/generate_case_counts.py",
    "ud_czech_case_counts.json": "scripts/generate_czech_case_counts.py",
    "ud_ewt_clause_counts.json": "scripts/generate_clause_counts.py",
    "ud_ewt_narrative_counts.json": "scripts/generate_narrative_counts.py",
    "brown_word_counts_full.json": "scripts/generate_word_counts.py",
    "brown_clause_word_counts.json": "scripts/generate_clause_word_counts.py",
    "brown_online_drift.json": "scripts/generate_online_genre_counts.py",
    "synthetic_counts.json": "scripts/generate_synthetic_counts.py",
    "synthetic_phone_counts.json": "scripts/generate_synthetic_phone_counts.py",
    "synthetic_aligned_g2p_counts.json": "scripts/generate_synthetic_aligned_g2p.py",
}

_SHA256_RE = re.compile(r"\b([0-9a-f]{64})\b")
_DOCUMENTED_SHA256_RE = re.compile(r"\*\*sha256:\*\*\s*`([0-9a-f]{64})`")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _documented_hashes(text: str | None = None) -> dict[str, str]:
    """Map each documented artifact filename to its adjacent digest.

    A bare set of digests loses the assertion the document makes: which digest
    belongs to which file. The nearest explicit ``vendor/<filename>`` marker
    before each ``sha256`` field owns that digest.
    """
    if text is None:
        if not SOURCES_DOC.exists():
            return {}
        text = SOURCES_DOC.read_text(encoding="utf-8")

    documented: dict[str, str] = {}
    cursor = 0
    for digest_match in _DOCUMENTED_SHA256_RE.finditer(text):
        block = text[cursor : digest_match.start()]
        markers: list[tuple[int, str]] = []
        for filename in DOCUMENTED_ARTIFACTS:
            marker = f"vendor/{filename}"
            position = block.rfind(marker)
            if position >= 0:
                markers.append((position, filename))
        if not markers:
            raise AssertionError(
                f"documented sha256 {digest_match.group(1)} is not bound to an artifact filename"
            )
        _, filename = max(markers)
        if filename in documented:
            raise AssertionError(f"{filename} has more than one documented sha256")
        documented[filename] = digest_match.group(1)
        cursor = digest_match.end()
    return documented


def _checksum_mismatches(
    documented: dict[str, str], vendor_dir: Path = VENDOR_DIR
) -> dict[str, tuple[str, str]]:
    """Return ``filename: (documented, actual)`` for present mismatches."""
    mismatches: dict[str, tuple[str, str]] = {}
    for filename, expected in documented.items():
        path = vendor_dir / filename
        if path.exists():
            actual = _sha256(path)
            if actual != expected:
                mismatches[filename] = (expected, actual)
    return mismatches


# ── The scan set must be non-empty ───────────────────────────────────────────


def test_provenance_document_exists():
    assert SOURCES_DOC.exists(), f"{SOURCES_DOC} is missing"


def test_provenance_document_records_at_least_one_checksum():
    """Guard against vacuity: with no documented hashes the checks below pass free."""
    assert _documented_hashes(), (
        f"{SOURCES_DOC} records no sha256 values, so the drift checks below would "
        "be vacuously true."
    )


def test_documented_inventory_equals_the_complete_runtime_manifest():
    """A packaged derived artifact may not escape filename-bound provenance."""
    assert set(DOCUMENTED_ARTIFACTS) == set(RUNTIME_VENDOR_ARTIFACTS), (
        "runtime/package and documented-checksum inventories disagree: "
        f"undocumented={sorted(set(RUNTIME_VENDOR_ARTIFACTS) - set(DOCUMENTED_ARTIFACTS))}, "
        f"not_packaged={sorted(set(DOCUMENTED_ARTIFACTS) - set(RUNTIME_VENDOR_ARTIFACTS))}"
    )


# ── THE ACTUAL GATE ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("filename", sorted(DOCUMENTED_ARTIFACTS))
def test_documented_checksum_matches_the_artifact(filename: str):
    """The prose digest must equal the digest of the committed file."""
    path = VENDOR_DIR / filename
    if not path.exists():
        pytest.skip(
            f"{path} not vendored in this checkout (rebuild with {DOCUMENTED_ARTIFACTS[filename]})"
        )
    actual = _sha256(path)
    documented = _documented_hashes()
    assert filename in documented, (
        f"{filename} has no filename-bound sha256 in {SOURCES_DOC.relative_to(PROJECT_ROOT)}"
    )
    assert actual == documented[filename], (
        f"{filename} hashes to {actual}, but its documented sha256 is "
        f"{documented[filename]} in "
        f"{SOURCES_DOC.relative_to(PROJECT_ROOT)}. Update the documented sha256 — "
        "do not silently leave prose asserting a digest the file does not have."
    )


def test_no_orphan_checksums_in_prose():
    """Every documented digest must belong to an artifact that exists.

    An orphan digest is a leftover from a regenerated artifact — exactly the drift
    this file exists to catch, seen from the other direction.
    """
    present = {
        name: _sha256(VENDOR_DIR / name)
        for name in DOCUMENTED_ARTIFACTS
        if (VENDOR_DIR / name).exists()
    }
    if not present:
        pytest.skip("no vendored artifacts in this checkout")

    documented = _documented_hashes()
    assert set(documented) == set(DOCUMENTED_ARTIFACTS), (
        "the provenance document and checksum inventory disagree: "
        f"undocumented={sorted(set(DOCUMENTED_ARTIFACTS) - set(documented))}, "
        f"unexpected={sorted(set(documented) - set(DOCUMENTED_ARTIFACTS))}"
    )
    orphans = {
        filename: digest
        for filename, digest in documented.items()
        if digest not in present.values()
    }
    assert not orphans, (
        f"{len(orphans)} documented sha256 value(s) match no vendored artifact: "
        f"{orphans}. These are stale digests left behind by a regenerated "
        f"artifact — remove or update them."
    )


def test_swapped_documented_hashes_are_rejected():
    """POSITIVE CONTROL: correct digests on the wrong filenames must fail."""
    source = SOURCES_DOC.read_text(encoding="utf-8")
    documented = _documented_hashes(source)
    first = "brown_counts.json"
    second = "cmudict_counts.json"
    first_digest = documented[first]
    second_digest = documented[second]
    sentinel = "__PLANTED_FIRST_DIGEST__"
    swapped = source.replace(f"`{first_digest}`", sentinel, 1)
    swapped = swapped.replace(f"`{second_digest}`", f"`{first_digest}`", 1)
    swapped = swapped.replace(sentinel, f"`{second_digest}`", 1)

    mismatches = _checksum_mismatches(_documented_hashes(swapped))
    assert {first, second} <= set(mismatches), (
        "the filename-bound checksum gate accepted two valid digests after their "
        "artifact associations were swapped"
    )


# ── Determinism: a checksum on a variable artifact means nothing ─────────────


@pytest.mark.parametrize("filename", sorted(DOCUMENTED_ARTIFACTS))
def test_artifact_is_stable_on_disk(filename: str):
    """Reading twice must give the same digest.

    Cheap sanity check that the file is not being rewritten underneath the suite.
    Full regeneration determinism is verified by the generator scripts (which are
    slow and need the source corpora), not here.
    """
    path = VENDOR_DIR / filename
    if not path.exists():
        pytest.skip(f"{path} not vendored")
    assert _sha256(path) == _sha256(path)


def test_checksum_helper_detects_a_difference(tmp_path):
    """POSITIVE CONTROL: the comparison must be able to fail.

    Without this, a bug making ``_sha256`` return a constant would let every check
    above pass on mismatched files.
    """
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert _sha256(first) != _sha256(second)


@pytest.mark.raw_corpus_regeneration
def test_brown_word_generator_reproduces_committed_artifacts_byte_identically(tmp_path):
    """The source writer, not a hand edit, owns both Stage 3 word artifacts.

    Brown is NLTK data rather than package code and is intentionally not fetched
    during a test. A checkout without the local corpus skips this raw-source
    replay while retaining the always-on schema and committed-checksum gates.
    """
    try:
        source_zip = resolve_brown_source_zip()
    except FileNotFoundError:
        pytest.skip("pinned NLTK brown.zip absent; raw-source replay cannot run offline")

    for filename, vocab_size in (
        ("brown_word_counts.json", 2000),
        ("brown_word_counts_full.json", 0),
    ):
        generated = tmp_path / filename
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "generate_word_counts.py"),
                "--source-zip",
                str(source_zip),
                "--output",
                str(generated),
                "--vocab-size",
                str(vocab_size),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert generated.read_bytes() == (VENDOR_DIR / filename).read_bytes(), (
            f"{filename} is not the byte-identical output of generate_word_counts.py"
        )


@pytest.mark.parametrize(
    "filename",
    (
        "brown_counts.json",
        "brown_word_counts.json",
        "brown_word_counts_full.json",
        "brown_heldout_stream.json",
        "brown_clause_word_counts.json",
        "brown_online_drift.json",
    ),
)
def test_every_brown_artifact_names_the_pinned_raw_source(filename: str):
    """Always-on source identity does not depend on a local raw archive."""
    artifact = json.loads((VENDOR_DIR / filename).read_text(encoding="utf-8"))

    validate_brown_artifact_provenance(artifact)
    raw_source = artifact["provenance"]["raw_source"]
    assert raw_source["source_revision"] == BROWN_REVISION
    assert raw_source["source_url"] == BROWN_RAW_URL
    assert raw_source["source_sha256"] == BROWN_RAW_SHA256
    assert raw_source["source_size_bytes"] == BROWN_RAW_SIZE_BYTES


def test_brown_word_source_gate_rejects_branch_and_missing_revision():
    """NEGATIVE CONTROL: partial or moving provenance is not immutable provenance."""
    artifact = json.loads((VENDOR_DIR / "brown_word_counts.json").read_text(encoding="utf-8"))
    for mutation in ("branch", "missing"):
        planted = json.loads(json.dumps(artifact))
        raw_source = planted["provenance"]["raw_source"]
        if mutation == "branch":
            raw_source["source_url"] = (
                "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
                "packages/corpora/brown.zip"
            )
        else:
            raw_source.pop("source_revision")
        with pytest.raises(ValueError, match="pinned raw-source metadata|disagrees with the pin"):
            validate_brown_artifact_provenance(planted)


@pytest.mark.parametrize(
    ("script", "artifact", "extra_args"),
    (
        ("generate_counts.py", "brown_counts.json", ()),
        ("generate_heldout_stream.py", "brown_heldout_stream.json", ()),
        ("generate_clause_word_counts.py", "brown_clause_word_counts.json", ()),
        ("generate_online_genre_counts.py", "brown_online_drift.json", ()),
    ),
)
@pytest.mark.raw_corpus_regeneration
def test_other_brown_writers_reproduce_committed_artifacts_byte_identically(
    tmp_path, script: str, artifact: str, extra_args: tuple[str, ...]
):
    """Every direct Brown writer authenticates one source and reproduces bytes."""
    try:
        source_zip = resolve_brown_source_zip()
    except FileNotFoundError:
        pytest.skip("pinned NLTK brown.zip absent; raw-source replay cannot run offline")
    generated = tmp_path / artifact
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / script),
            "--source-zip",
            str(source_zip),
            "--output",
            str(generated),
            *extra_args,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert generated.read_bytes() == (VENDOR_DIR / artifact).read_bytes()


def test_czech_artifact_names_both_pinned_official_shards():
    artifact = json.loads((VENDOR_DIR / "ud_czech_case_counts.json").read_text(encoding="utf-8"))
    raw_source = artifact["provenance"]["raw_source"]
    validate_ud_czech_source_metadata(raw_source)
    assert raw_source == ud_czech_source_metadata()
    assert raw_source["source_revision"] == UD_CZECH_PDTC_REVISION


def test_czech_artifact_source_gate_rejects_a_planted_moving_shard():
    planted = ud_czech_source_metadata()
    planted["shards"]["test"]["source_url"] = (
        "https://raw.githubusercontent.com/UniversalDependencies/"
        "UD_Czech-PDTC/master/cs_pdtc-ud-dev.conllu"
    )
    with pytest.raises(ValueError, match="disagrees with the pin"):
        validate_ud_czech_source_metadata(planted)


@pytest.mark.raw_corpus_regeneration
def test_czech_writer_reproduces_committed_artifact_byte_identically(tmp_path):
    staging = VENDOR_DIR / "staging_ud_czech"
    required = [staging / shard.filename for shard in UD_CZECH_PDTC_SHARDS.values()]
    if not all(path.is_file() for path in required):
        pytest.skip("pinned Czech-PDTC staging shards are absent")
    generated = tmp_path / "ud_czech_case_counts.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_czech_case_counts.py"),
            "--staging-dir",
            str(staging),
            "--output",
            str(generated),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert generated.read_bytes() == (VENDOR_DIR / "ud_czech_case_counts.json").read_bytes()


def test_cmudict_artifact_names_the_pinned_official_source():
    """Committed sufficient statistics retain their exact raw-source identity."""
    artifact = json.loads((VENDOR_DIR / "cmudict_counts.json").read_text(encoding="utf-8"))

    validate_cmudict_source_metadata(artifact)
    assert artifact["source_revision"] == CMUDICT_REVISION
    assert artifact["source_url"] == CMUDICT_URL
    assert artifact["source_sha256"] == CMUDICT_RAW_SHA256


def test_cmudict_source_metadata_gate_rejects_a_planted_branch_url():
    """NEGATIVE CONTROL: a moving branch URL cannot masquerade as the raw source."""
    artifact = json.loads((VENDOR_DIR / "cmudict_counts.json").read_text(encoding="utf-8"))
    planted = dict(artifact)
    planted["source_url"] = (
        "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    )

    with pytest.raises(ValueError, match="disagrees with the pin"):
        validate_cmudict_source_metadata(planted)


def test_cmudict_source_metadata_gate_rejects_a_missing_revision():
    """NEGATIVE CONTROL: partial provenance is not pinned provenance."""
    artifact = json.loads((VENDOR_DIR / "cmudict_counts.json").read_text(encoding="utf-8"))
    artifact.pop("source_revision")

    with pytest.raises(ValueError, match="lacks pinned source metadata"):
        validate_cmudict_source_metadata(artifact)


def test_cmudict_generator_rejects_an_unpinned_raw_file(tmp_path):
    """NEGATIVE CONTROL: canonical output cannot be relabeled from other bytes."""
    raw = tmp_path / "cmudict.dict"
    generated = tmp_path / "cmudict_counts.json"
    raw.write_text("cat  K AE1 T\n", encoding="latin-1")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_phoneme_counts.py"),
            "--cmudict",
            str(raw),
            "--output",
            str(generated),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "pinned upstream revision" in result.stderr
    assert not generated.exists()


@pytest.mark.parametrize(
    "filename",
    (
        "ud_ewt_case_counts.json",
        "ud_ewt_clause_counts.json",
        "ud_ewt_narrative_counts.json",
    ),
)
def test_every_english_ewt_artifact_names_the_same_pinned_raw_source(filename: str):
    artifact = json.loads((VENDOR_DIR / filename).read_text(encoding="utf-8"))

    validate_ud_ewt_source_metadata(artifact["provenance"])
    assert artifact["provenance"]["source_revision"] == UD_EWT_REVISION
    assert artifact["provenance"]["source_sha256"] == UD_EWT_RAW_SHA256
    assert artifact["provenance"]["source_size_bytes"] == UD_EWT_RAW_SIZE_BYTES


def test_english_ewt_artifact_source_gate_rejects_branch_and_missing_revision():
    for mutation in ("branch", "missing"):
        planted = dict(UD_EWT_SOURCE_METADATA)
        if mutation == "branch":
            planted["source_url"] = (
                "https://raw.githubusercontent.com/UniversalDependencies/"
                "UD_English-EWT/master/en_ewt-ud-dev.conllu"
            )
        else:
            planted.pop("source_revision")
        with pytest.raises(ValueError, match="pinned source metadata|disagrees with the pin"):
            validate_ud_ewt_source_metadata(planted)


def test_english_ewt_generator_rejects_modified_cached_source(tmp_path):
    raw_cache = tmp_path / "raw"
    raw_cache.mkdir()
    (raw_cache / "en_ewt-ud-dev.conllu").write_bytes(b"modified source")
    generated = tmp_path / "case.json"
    environment = {**os.environ, "AILM_CORPUS_CACHE_DIR": str(raw_cache)}

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_case_counts.py"),
            "--output",
            str(generated),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "UD EWT source does not match" in result.stderr
    assert not generated.exists()


@pytest.mark.raw_corpus_regeneration
@pytest.mark.skipif(
    os.environ.get("AILM_RUN_RAW_CORPUS_REGENERATION") != "1",
    reason="set AILM_RUN_RAW_CORPUS_REGENERATION=1 to replay ignored raw corpora",
)
def test_cmudict_raw_source_regenerates_committed_artifact_byte_identically(tmp_path):
    """Opt-in raw replay; ordinary tests remain independent of ignored downloads."""
    raw = VENDOR_DIR / "cmudict.dict"
    assert raw.is_file(), (
        "AILM_RUN_RAW_CORPUS_REGENERATION=1 was set, but vendor/cmudict.dict is absent. "
        "Fetch the pinned source with scripts/generate_phoneme_counts.py --download."
    )
    generated = tmp_path / "cmudict_counts.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_phoneme_counts.py"),
            "--cmudict",
            str(raw),
            "--output",
            str(generated),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert generated.read_bytes() == (VENDOR_DIR / "cmudict_counts.json").read_bytes()


@pytest.mark.raw_corpus_regeneration
@pytest.mark.skipif(
    os.environ.get("AILM_RUN_RAW_CORPUS_REGENERATION") != "1",
    reason="set AILM_RUN_RAW_CORPUS_REGENERATION=1 to replay ignored raw corpora",
)
@pytest.mark.parametrize(
    ("script", "artifact"),
    (
        ("generate_case_counts.py", "ud_ewt_case_counts.json"),
        ("generate_clause_counts.py", "ud_ewt_clause_counts.json"),
        ("generate_narrative_counts.py", "ud_ewt_narrative_counts.json"),
    ),
)
def test_ud_ewt_raw_source_regenerates_stage4_through_6_byte_identically(
    tmp_path, script: str, artifact: str
):
    raw = VENDOR_DIR / "en_ewt-ud-dev.conllu"
    assert raw.is_file(), (
        "AILM_RUN_RAW_CORPUS_REGENERATION=1 was set, but the pinned UD EWT raw file is absent"
    )
    generated = tmp_path / artifact
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--output", str(generated)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert generated.read_bytes() == (VENDOR_DIR / artifact).read_bytes()


def test_regex_extracts_a_planted_digest(tmp_path):
    """POSITIVE CONTROL on the extraction, not just the comparison."""
    digest = "a" * 64
    assert digest in set(_SHA256_RE.findall(f"- **sha256:** `{digest}`"))
    # And must not match a too-short hex run.
    assert not _SHA256_RE.findall("deadbeef")
