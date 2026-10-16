"""Pinned raw-source contract for the Czech-PDTC morphology artifact."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

UD_CZECH_PDTC_REVISION = "6d206ec7d337a7f76f34ddfc82893389cabbd76d"
UD_CZECH_PDTC_REPOSITORY_URL = "https://github.com/UniversalDependencies/UD_Czech-PDTC"
UD_CZECH_PDTC_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Czech-PDTC/"
    f"{UD_CZECH_PDTC_REVISION}"
)


@dataclass(frozen=True)
class UDCzechShard:
    """Immutable identity of one source shard and its artifact split role."""

    artifact_split: str
    upstream_split: str
    filename: str
    sha256: str
    size_bytes: int

    @property
    def source_url(self) -> str:
        """Immutable raw URL for this shard."""
        return f"{UD_CZECH_PDTC_RAW_BASE_URL}/{self.filename}"

    def metadata(self) -> dict[str, str | int]:
        """Return the exact JSON representation embedded in the artifact."""
        return {
            "artifact_split": self.artifact_split,
            "upstream_split": self.upstream_split,
            "filename": self.filename,
            "source_url": self.source_url,
            "source_sha256": self.sha256,
            "source_size_bytes": self.size_bytes,
        }


UD_CZECH_PDTC_SHARDS: Mapping[str, UDCzechShard] = MappingProxyType(
    {
        "train": UDCzechShard(
            artifact_split="train",
            upstream_split="train-ft",
            filename="cs_pdtc-ud-train-ft.conllu",
            sha256="ed5ce30de11ed9ac14ce1ca54dffb31e79211324850ec0badf5eb1e3a3e2dff6",
            size_bytes=3_726_243,
        ),
        "test": UDCzechShard(
            artifact_split="test",
            upstream_split="dev",
            filename="cs_pdtc-ud-dev.conllu",
            sha256="73a19117ff39411b1134840c3a0383a37f5dd89a2e07b3f8c0a8821d8d37d4bc",
            size_bytes=50_117_913,
        ),
    }
)


class UDCzechSourceIntegrityError(ValueError):
    """A Czech-PDTC shard or embedded source identity does not match its pin."""


def ud_czech_source_metadata() -> dict[str, object]:
    """Return the complete JSON-serialisable source declaration."""
    return {
        "repository_url": UD_CZECH_PDTC_REPOSITORY_URL,
        "source_revision": UD_CZECH_PDTC_REVISION,
        "shards": {split: shard.metadata() for split, shard in UD_CZECH_PDTC_SHARDS.items()},
    }


def _expected_shard(split: str) -> UDCzechShard:
    """Resolve a declared split without accepting caller-supplied identities."""
    try:
        return UD_CZECH_PDTC_SHARDS[split]
    except KeyError as exc:
        raise UDCzechSourceIntegrityError(
            f"unknown Czech artifact split {split!r}; expected {sorted(UD_CZECH_PDTC_SHARDS)}"
        ) from exc


def verify_ud_czech_shard_bytes(content: bytes, split: str, *, filename: str) -> str:
    """Return the digest only when one shard's bytes and filename match its pin."""
    expected = _expected_shard(split)
    if filename != expected.filename:
        raise UDCzechSourceIntegrityError(
            f"Czech {split} source has wrong filename {filename!r}; expected {expected.filename!r}"
        )
    measured_size = len(content)
    measured_sha256 = hashlib.sha256(content).hexdigest()
    if measured_size != expected.size_bytes or measured_sha256 != expected.sha256:
        raise UDCzechSourceIntegrityError(
            f"Czech {split} source does not match revision {UD_CZECH_PDTC_REVISION}: "
            f"bytes={measured_size} (expected {expected.size_bytes}), "
            f"sha256={measured_sha256} (expected {expected.sha256})"
        )
    return measured_sha256


def read_verified_ud_czech_shard(path: Path | str, split: str) -> bytes:
    """Read one shard once, validate those exact bytes, and return them for parsing."""
    expected = _expected_shard(split)

    resolved = Path(path).resolve()
    if resolved.name != expected.filename:
        raise UDCzechSourceIntegrityError(
            f"Czech {split} source has wrong filename {resolved.name!r}; "
            f"expected {expected.filename!r}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Czech {split} source shard is absent: {resolved}")
    content = resolved.read_bytes()
    verify_ud_czech_shard_bytes(content, split, filename=resolved.name)
    return content


def verify_ud_czech_shard(path: Path | str, split: str) -> Path:
    """Validate one staged source shard before parsing and return its path."""
    resolved = Path(path).resolve()
    read_verified_ud_czech_shard(resolved, split)
    return resolved


def validate_ud_czech_source_metadata(payload: Mapping[str, object]) -> None:
    """Require an artifact's complete source declaration to equal the pin."""
    expected = ud_czech_source_metadata()
    missing = sorted(set(expected) - set(payload))
    if missing:
        raise UDCzechSourceIntegrityError(f"Czech artifact lacks pinned source metadata: {missing}")
    mismatches = {
        key: (value, payload[key]) for key, value in expected.items() if payload[key] != value
    }
    if mismatches:
        raise UDCzechSourceIntegrityError(
            f"Czech artifact source metadata disagrees with the pin: {mismatches}"
        )


def verified_ud_czech_sources(staging_dir: Path | str) -> dict[str, Path]:
    """Resolve and verify every required shard; no partial mapping is returned."""
    root = Path(staging_dir).resolve()
    return {
        split: verify_ud_czech_shard(root / shard.filename, split)
        for split, shard in UD_CZECH_PDTC_SHARDS.items()
    }
