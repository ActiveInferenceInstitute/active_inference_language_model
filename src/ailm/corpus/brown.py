"""Immutable-source access and provenance validation for NLTK Brown.

AILM's Brown-derived artifacts are canonical only when they were produced from
the exact ``brown.zip`` archive pinned here.  NLTK's ordinary lazy corpus
loader searches mutable, user-wide data directories and can prefer an
unverified extracted tree over the archive beside it.  :func:`load_brown_corpus`
instead verifies the archive bytes and constructs a reader directly over that
verified zip path.

This module intentionally does not download.  Tests stay offline, and a
generator cannot silently replace a missing scientific source with whichever
revision a package index serves that day.  The immutable URL in
:data:`BROWN_RAW_URL` is the acquisition instruction.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

BROWN_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
BROWN_REPOSITORY_URL = "https://github.com/nltk/nltk_data"
BROWN_RAW_URL = (
    f"https://raw.githubusercontent.com/nltk/nltk_data/{BROWN_REVISION}/packages/corpora/brown.zip"
)
BROWN_RAW_SHA256 = "9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6"
BROWN_RAW_SIZE_BYTES = 3_314_357
#: Ordered file-identity and file/category projections derived from the pinned
#: archive.  Artifact-embedded digests prove internal consistency; these source
#: pins additionally prevent a coordinated metadata rewrite from authenticating
#: itself merely by recomputing its own digest.
BROWN_DOCUMENT_IDENTITY_SHA256 = "2ae4ce2db7f343a69aa6943092d2486c504d8337c788c419cb4cb8b6537316a2"
BROWN_PARTITION_INPUT_SHA256 = "944a2d200e6234117e06932f1baaea0758c3783503a39926ec519005ee9f26fb"
BROWN_TRAIN_DOCUMENT_FRACTION = 0.9
BROWN_SOURCE_ENV = "AILM_BROWN_SOURCE_ZIP"
BROWN_BIGRAM_ARTIFACT_SCHEMA = "ailm-brown-bigram-counts-v2"
BROWN_NONCANONICAL_ARTIFACT_SCHEMA = "ailm-noncanonical-character-bigram-counts-v1"
BROWN_WORD_ARTIFACT_SCHEMA = "ailm-word-counts-v2"
BROWN_HELDOUT_ARTIFACT_SCHEMA = "ailm-heldout-stream-v3"

_ASCII_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")

BROWN_SOURCE_METADATA: Mapping[str, str | int] = MappingProxyType(
    {
        "repository_url": BROWN_REPOSITORY_URL,
        "source_revision": BROWN_REVISION,
        "source_url": BROWN_RAW_URL,
        "source_sha256": BROWN_RAW_SHA256,
        "source_size_bytes": BROWN_RAW_SIZE_BYTES,
    }
)


class BrownSourceIntegrityError(ValueError):
    """Brown bytes or embedded source identity do not match the canonical pin."""


@dataclass(frozen=True, slots=True)
class BrownDocumentPartition:
    """One authenticated, document-disjoint partition of normalized Brown.

    Token streams preserve NLTK's corpus order.  The per-token document indices
    make the held-out population clusterable without reopening the raw corpus,
    while the ordered identity digest prevents an artifact from silently
    relabeling those clusters.
    """

    document_ids: tuple[str, ...]
    categories: tuple[str, ...]
    alphabetic_token_counts: tuple[int, ...]
    train_indices: tuple[int, ...]
    heldout_indices: tuple[int, ...]
    train_tokens: tuple[str, ...]
    heldout_tokens: tuple[str, ...]
    heldout_token_document_indices: tuple[int, ...]


def brown_document_identity_sha256(document_ids: Sequence[str]) -> str:
    """Digest ordered Brown file identities without concatenation ambiguity."""
    if not document_ids:
        raise ValueError("Brown document identity digest needs at least one id")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Brown document identity digest refuses duplicate ids")
    encoded = bytearray()
    for document_id in document_ids:
        if not isinstance(document_id, str) or not document_id or "\0" in document_id:
            raise ValueError(f"invalid Brown document id {document_id!r}")
        payload = document_id.encode("utf-8")
        encoded.extend(len(payload).to_bytes(8, "big"))
        encoded.extend(payload)
    return hashlib.sha256(encoded).hexdigest()


def brown_partition_input_sha256(document_ids: Sequence[str], categories: Sequence[str]) -> str:
    """Digest every immutable identity/category input that determines the split."""
    ids = tuple(document_ids)
    genre_labels = tuple(categories)
    brown_document_identity_sha256(ids)
    if len(genre_labels) != len(ids):
        raise ValueError("Brown partition categories must align one-for-one with ids")
    fields = [BROWN_REVISION]
    for document_id, category in zip(ids, genre_labels, strict=True):
        fields.extend((document_id, category))
    encoded = bytearray()
    for field in fields:
        if not isinstance(field, str) or not field or "\0" in field:
            raise ValueError(f"invalid Brown partition input {field!r}")
        payload = field.encode("utf-8")
        encoded.extend(len(payload).to_bytes(8, "big"))
        encoded.extend(payload)
    return hashlib.sha256(encoded).hexdigest()


def validate_canonical_brown_document_metadata(
    document_ids: Sequence[str], categories: Sequence[str]
) -> None:
    """Bind artifact document/category metadata to the immutable source pin."""
    identity = brown_document_identity_sha256(document_ids)
    partition_input = brown_partition_input_sha256(document_ids, categories)
    mismatches: dict[str, tuple[str, str]] = {}
    if identity != BROWN_DOCUMENT_IDENTITY_SHA256:
        mismatches["document_identity_sha256"] = (
            BROWN_DOCUMENT_IDENTITY_SHA256,
            identity,
        )
    if partition_input != BROWN_PARTITION_INPUT_SHA256:
        mismatches["document_partition_input_sha256"] = (
            BROWN_PARTITION_INPUT_SHA256,
            partition_input,
        )
    if mismatches:
        raise BrownSourceIntegrityError(
            "Brown document/category metadata disagrees with the pinned source "
            f"projection: {mismatches}"
        )


def validate_canonical_brown_partition_target(value: object) -> None:
    """Require the declared split target to equal the canonical protocol."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or float(value) != BROWN_TRAIN_DOCUMENT_FRACTION
    ):
        raise BrownSourceIntegrityError(
            "Brown target_train_document_fraction disagrees with the canonical "
            f"protocol: expected {BROWN_TRAIN_DOCUMENT_FRACTION}, got {value!r}"
        )


def category_stratified_document_indices(
    document_ids: Sequence[str],
    categories: Sequence[str],
    train_fraction: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition documents by a revision-bound hash within every Brown genre.

    Each category contributes the nearest whole-document count to the target
    held-out fraction, with at least one document in each arm. Selection uses
    only immutable source identity, never text length, labels, or model output.
    Returned indices preserve corpus order even though selection is hash-based.
    """
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, Real)
        or not math.isfinite(float(train_fraction))
        or not 0.0 < float(train_fraction) < 1.0
    ):
        raise ValueError(f"train_fraction must be finite and in (0, 1), got {train_fraction!r}")
    ids = tuple(document_ids)
    genre_labels = tuple(categories)
    brown_document_identity_sha256(ids)
    if len(genre_labels) != len(ids):
        raise ValueError("Brown document categories must align one-for-one with ids")

    by_category: dict[str, list[int]] = {}
    for index, category in enumerate(genre_labels):
        if not isinstance(category, str) or not category or "\0" in category:
            raise ValueError(f"invalid Brown category at index {index}: {category!r}")
        by_category.setdefault(category, []).append(index)

    heldout: set[int] = set()
    heldout_fraction = Fraction(1) - Fraction(str(float(train_fraction)))
    for category, indices in sorted(by_category.items()):
        if len(indices) < 2:
            raise ValueError(
                f"Brown category {category!r} needs at least two documents to "
                "leave both train and held-out representatives"
            )
        heldout_count = math.floor(heldout_fraction * len(indices) + Fraction(1, 2))
        heldout_count = max(1, min(len(indices) - 1, heldout_count))

        def selection_key(index: int, category_name: str = category) -> tuple[bytes, str]:
            framed = f"{BROWN_REVISION}\0{category_name}\0{ids[index]}".encode()
            return hashlib.sha256(framed).digest(), ids[index]

        heldout.update(sorted(indices, key=selection_key)[:heldout_count])

    train_indices = tuple(index for index in range(len(ids)) if index not in heldout)
    heldout_indices = tuple(sorted(heldout))
    if not train_indices or not heldout_indices:
        raise ValueError("category-stratified Brown split must leave both arms")
    return train_indices, heldout_indices


def brown_source_metadata() -> dict[str, str | int]:
    """Return a fresh JSON-serialisable copy of the canonical source identity."""
    return dict(BROWN_SOURCE_METADATA)


def validate_brown_source_metadata(payload: Mapping[str, object]) -> None:
    """Require an embedded Brown source declaration to equal the full pin."""
    missing = sorted(set(BROWN_SOURCE_METADATA) - set(payload))
    if missing:
        raise BrownSourceIntegrityError(
            f"Brown artifact lacks pinned raw-source metadata: {missing}"
        )
    unexpected = sorted(set(payload) - set(BROWN_SOURCE_METADATA))
    if unexpected:
        raise BrownSourceIntegrityError(
            "Brown artifact has unexpected raw-source metadata that could create "
            f"an ambiguous source declaration: {unexpected}"
        )
    mismatches = {
        key: (expected, payload[key])
        for key, expected in BROWN_SOURCE_METADATA.items()
        if payload[key] != expected
    }
    if mismatches:
        raise BrownSourceIntegrityError(
            f"Brown artifact raw-source metadata disagrees with the pin: {mismatches}"
        )


def validate_brown_artifact_provenance(payload: Mapping[str, object]) -> None:
    """Validate the nested raw-source identity of a Brown-derived artifact."""
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BrownSourceIntegrityError("Brown artifact lacks a provenance object")
    raw_source = provenance.get("raw_source")
    if not isinstance(raw_source, Mapping):
        raise BrownSourceIntegrityError("Brown artifact provenance lacks a raw_source object")
    validate_brown_source_metadata(raw_source)


def verify_brown_source_bytes(content: bytes) -> str:
    """Return the digest only when ``content`` is the pinned readable archive."""
    measured_size = len(content)
    measured_sha256 = hashlib.sha256(content).hexdigest()
    if measured_size != BROWN_RAW_SIZE_BYTES or measured_sha256 != BROWN_RAW_SHA256:
        raise BrownSourceIntegrityError(
            "Brown source does not match the pinned archive: "
            f"bytes={measured_size} (expected {BROWN_RAW_SIZE_BYTES}), "
            f"sha256={measured_sha256} (expected {BROWN_RAW_SHA256})"
        )

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            bad_member = archive.testzip()
            names = frozenset(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise BrownSourceIntegrityError("pinned Brown bytes are not a readable zip") from exc
    if bad_member is not None:
        raise BrownSourceIntegrityError(
            f"pinned Brown archive member failed its CRC check: {bad_member}"
        )
    required = {"brown/cats.txt", "brown/README", "brown/ca01", "brown/cr09"}
    missing = sorted(required - names)
    if missing:
        raise BrownSourceIntegrityError(
            f"pinned Brown archive lacks required corpus members: {missing}"
        )
    return measured_sha256


def verify_brown_source_zip(path: Path | str) -> Path:
    """Validate an on-disk Brown archive and return its resolved file path.

    This is an instantaneous path check, not a capability that makes later
    reopens safe.  Scientific readers use :func:`_read_verified_brown_source`
    and retain the exact authenticated bytes instead.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Brown source archive is absent or not a file: {resolved}")
    verify_brown_source_bytes(resolved.read_bytes())
    return resolved


def _read_verified_brown_source(path: Path | str | None = None) -> tuple[Path, bytes]:
    """Read and authenticate one immutable Brown snapshot.

    The returned bytes are the bytes that passed the digest, extent, ZIP CRC,
    and member checks.  Callers must parse this snapshot rather than reopen the
    path: a path can be atomically replaced after any successful verification.
    """
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Brown source archive is absent or not a file: {resolved}")
        content = resolved.read_bytes()
        verify_brown_source_bytes(content)
        return resolved, content

    failures: list[str] = []
    for candidate in _default_source_candidates():
        if not candidate.exists():
            continue
        try:
            resolved = candidate.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Brown source archive is absent or not a file: {resolved}")
            content = resolved.read_bytes()
            verify_brown_source_bytes(content)
            return resolved, content
        except (FileNotFoundError, OSError, BrownSourceIntegrityError) as exc:
            failures.append(f"{candidate}: {exc}")

    detail = f" Invalid candidates: {'; '.join(failures)}" if failures else ""
    raise FileNotFoundError(
        "The pinned NLTK Brown brown.zip is unavailable. Fetch the immutable "
        f"source {BROWN_RAW_URL} and pass --source-zip PATH or set "
        f"{BROWN_SOURCE_ENV}.{detail}"
    )


def _default_source_candidates() -> list[Path]:
    """Return NLTK search-path candidates without loading any corpus data."""
    try:
        import nltk.data
    except ImportError as exc:  # pragma: no cover - dependency gate covers this
        raise RuntimeError("nltk is required to locate and read the Brown corpus") from exc

    candidates: list[Path] = []
    configured = os.environ.get(BROWN_SOURCE_ENV)
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(Path(root).expanduser() / "corpora" / "brown.zip" for root in nltk.data.path)
    # Preserve search order while avoiding duplicate diagnostics.
    return list(dict.fromkeys(candidates))


def resolve_brown_source_zip(path: Path | str | None = None) -> Path:
    """Resolve and verify the exact Brown archive used by canonical writers.

    An explicit ``path`` is authoritative and is never replaced by another
    candidate.  Without one, ``AILM_BROWN_SOURCE_ZIP`` and then NLTK's standard
    data roots are searched for a byte-identical archive.  Extracted directories
    are deliberately ignored because they do not carry a verifiable raw extent.
    """
    resolved, _content = _read_verified_brown_source(path)
    return resolved


def load_brown_corpus(path: Path | str | None = None) -> Any:
    """Construct an NLTK Brown reader over the exact authenticated bytes.

    NLTK's ordinary on-demand ZIP reader reopens its path for every member.
    That creates a check/use race if the file is replaced after verification.
    Keeping an in-memory :class:`zipfile.ZipFile` binds every lazy member read
    to the same snapshot whose digest and CRCs passed above.
    """
    source_path, content = _read_verified_brown_source(path)
    try:
        from nltk.corpus.reader.tagged import CategorizedTaggedCorpusReader
        from nltk.data import ZipFilePathPointer
    except ImportError as exc:  # pragma: no cover - dependency gate covers this
        raise RuntimeError("nltk is required to read the Brown corpus") from exc

    authenticated_archive = zipfile.ZipFile(io.BytesIO(content))
    # NLTK uses ``zipfile.filename`` only to resolve its category sidecar name.
    # A BytesIO-backed stdlib ZipFile reports None, so retain the authenticated
    # source name as metadata while all member reads continue to use ``content``.
    authenticated_archive.filename = str(source_path)
    root = ZipFilePathPointer(authenticated_archive, "brown/")
    return CategorizedTaggedCorpusReader(
        root,
        r"c[a-z]\d\d",
        cat_file="cats.txt",
        tagset="brown",
        encoding="ascii",
    )


def build_brown_document_partition(
    train_fraction: float,
    path: Path | str | None = None,
) -> BrownDocumentPartition:
    """Normalize Brown by source file and split whole documents within genre.

    A revision-bound identity hash selects the nearest target document count
    separately inside every Brown category. This prevents the corpus's
    genre-blocked file order from turning a prefix split into a genre shift.
    """
    reader = load_brown_corpus(path)
    document_ids = tuple(reader.fileids())
    categories: list[str] = []
    documents: list[tuple[str, ...]] = []
    for document_id in document_ids:
        document_categories = reader.categories(fileids=[document_id])
        if len(document_categories) != 1:
            raise ValueError(
                f"Brown document {document_id!r} must have exactly one category, "
                f"got {document_categories!r}"
            )
        categories.append(document_categories[0])
        normalized = tuple(
            lowered
            for token in reader.words(fileids=[document_id])
            if (lowered := token.lower()) and _ASCII_LETTERS.issuperset(lowered)
        )
        documents.append(normalized)

    token_counts = tuple(len(tokens) for tokens in documents)
    validate_canonical_brown_document_metadata(document_ids, categories)
    train_indices, heldout_indices = category_stratified_document_indices(
        document_ids, categories, train_fraction
    )
    train_tokens = tuple(token for index in train_indices for token in documents[index])
    heldout_tokens = tuple(token for index in heldout_indices for token in documents[index])
    heldout_token_document_indices = tuple(
        index for index in heldout_indices for _token in documents[index]
    )
    if len(heldout_tokens) != len(heldout_token_document_indices):
        raise RuntimeError("Brown held-out token/document alignment failed during construction")
    return BrownDocumentPartition(
        document_ids=document_ids,
        categories=tuple(categories),
        alphabetic_token_counts=token_counts,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        train_tokens=train_tokens,
        heldout_tokens=heldout_tokens,
        heldout_token_document_indices=heldout_token_document_indices,
    )
