"""CMU Pronouncing Dictionary loader for AILM.

Provides the letter-to-phoneme bridge between Level 2 (grapheme) and Level 1 (phoneme).

CRITICAL TRAP — verified from actual file analysis:
  Naive whitespace splitting yields claim:phoneme_count_naive_unstripped apparent
  "phoneme" tokens because some lines have comment tokens after '#'. Only
  claim:phoneme_count_39 are canonical ARPAbet; the difference is recorded as
  claim:phoneme_count_comment_phantoms.
  You MUST strip trailing '# ...' comments before building the phoneme inventory.
  TestNegativeControl::test_assert_full_inventory_catches_comment_token pins this
  behavior.

THE CANONICAL ARPABET PHONEMES (stress digits stripped):
  AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K
  L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH

Source: the immutable official GitHub revision named by CMUDICT_REVISION.
License: BSD-style, Carnegie Mellon University 1993-2015. Redistributable.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import requests

from ailm.corpus._download import atomic_publish_bytes, read_authenticated_response
from ailm.resources import resolve_raw_corpus_path

logger = logging.getLogger(__name__)

CMUDICT_REVISION = "74790861f652b15e4ac49015a90074ad62a27690"
CMUDICT_URL = f"https://raw.githubusercontent.com/cmusphinx/cmudict/{CMUDICT_REVISION}/cmudict.dict"
CMUDICT_RAW_SHA256 = "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"
CMUDICT_RAW_SIZE_BYTES = 3_618_488
EXPECTED_PHONEME_COUNT = 39  # canonical ARPAbet; stress digits stripped
_SHA256 = re.compile(r"[0-9a-f]{64}")

CMUDICT_SOURCE_METADATA: Mapping[str, str | int] = MappingProxyType(
    {
        "source_url": CMUDICT_URL,
        "source_revision": CMUDICT_REVISION,
        "source_sha256": CMUDICT_RAW_SHA256,
        "source_size_bytes": CMUDICT_RAW_SIZE_BYTES,
    }
)

CANONICAL_ARPABET: frozenset[str] = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    }
)
if len(CANONICAL_ARPABET) != EXPECTED_PHONEME_COUNT:  # module-load sanity check
    raise AssertionError(
        f"CANONICAL_ARPABET holds {len(CANONICAL_ARPABET)} phones, "
        f"expected {EXPECTED_PHONEME_COUNT}"
    )


@dataclass(frozen=True, slots=True)
class CommentStrippingAudit:
    """Inventories produced with and without CMUdict comment stripping."""

    canonical_inventory: frozenset[str]
    naive_inventory: frozenset[str]
    leaked_comment_tokens: frozenset[str]


@dataclass(frozen=True, slots=True)
class PinnedCMUDictSource:
    """One authenticated CMUdict snapshot and every derivation from it.

    Keeping the raw bytes, parsed dictionary, and comment-stripping control in
    one value prevents a writer from authenticating one path state and deriving
    claims from a later, atomically replaced state.
    """

    content: bytes
    dictionary: dict[str, list[list[str]]]
    comment_audit: CommentStrippingAudit


def verify_pinned_cmudict_bytes(content: bytes) -> str:
    """Verify bytes against the immutable CMUdict revision used by AILM.

    Returns:
        The verified hexadecimal SHA-256 digest.

    Raises:
        ValueError: If either the byte extent or digest differs from the pinned
            upstream source.
    """
    measured_size = len(content)
    measured_digest = hashlib.sha256(content).hexdigest()
    if measured_size != CMUDICT_RAW_SIZE_BYTES or measured_digest != CMUDICT_RAW_SHA256:
        raise ValueError(
            "CMUdict source does not match the pinned upstream revision: "
            f"bytes={measured_size} (expected {CMUDICT_RAW_SIZE_BYTES}), "
            f"sha256={measured_digest} (expected {CMUDICT_RAW_SHA256})"
        )
    return measured_digest


def _validate_cmudict_expectations(*, expected_sha256: str, expected_size_bytes: int) -> None:
    """Reject incomplete or malformed mirror identities before network I/O."""
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("CMUdict expected SHA-256 must be a lowercase digest")
    if (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes <= 0
    ):
        raise ValueError("CMUdict expected byte extent must be a positive integer")


def _download_cmudict_bytes(*, url: str, expected_sha256: str, expected_size_bytes: int) -> bytes:
    """Acquire one bounded, redirect-free, authenticated CMUdict body."""
    _validate_cmudict_expectations(
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    with requests.get(
        url,
        timeout=(10, 30),
        stream=True,
        allow_redirects=False,
        headers={"Accept-Encoding": "identity"},
    ) as response:
        return read_authenticated_response(
            response,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            source_name="CMUdict",
        )


def validate_cmudict_source_metadata(payload: Mapping[str, object]) -> None:
    """Require an artifact to identify the exact pinned raw CMUdict source.

    Raises:
        ValueError: If a required source field is absent or disagrees with the
            immutable source constants above.
    """
    missing = sorted(set(CMUDICT_SOURCE_METADATA) - set(payload))
    if missing:
        raise ValueError(f"CMUdict artifact lacks pinned source metadata: {missing}")
    mismatches = {
        key: (expected, payload[key])
        for key, expected in CMUDICT_SOURCE_METADATA.items()
        if payload[key] != expected
    }
    if mismatches:
        raise ValueError(f"CMUdict artifact source metadata disagrees with the pin: {mismatches}")


def audit_comment_stripping(path: str | Path) -> CommentStrippingAudit:
    """Measure the exact inventory corruption caused by retaining comments.

    The deliberately naive arm still skips CMUdict's full-line ``;;;`` headers,
    then treats every field after the word as a phone. The canonical arm uses the
    production parser. Their set difference is therefore the concrete set of
    inline-comment tokens that would become hidden states if ``# ...`` stripping
    regressed.

    Args:
        path: Local CMUdict file to inspect. This diagnostic never downloads.

    Returns:
        Both inventories and the exact leaked-token set.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CMUdict not found at {source}")

    return audit_comment_stripping_bytes(source.read_bytes())


def _strip_comment(line: str) -> str:
    """Strip trailing '#' comment from a CMUdict line.

    CMUdict uses '#' for comments. Some entries have inline comments like:
        ABBÉ    AH0 B EY1  # french
    Without stripping, 'french' appears as a phoneme token.

    Args:
        line: Raw line from cmudict.dict.

    Returns:
        Line with '#' and everything after it removed, then rstripped.
    """
    idx = line.find("#")
    if idx != -1:
        line = line[:idx]
    return line.rstrip()


def _parse_line(line: str) -> tuple[str, list[str]] | None:
    """Parse one CMUdict line into (word, phones).

    Strips '#' trailing comments, then skips:
      - empty lines
      - lines that start with ';' (CMUdict header lines like ';;; CMU...')

    Args:
        line: Raw line from cmudict.dict.

    Returns:
        (word_uppercase, [phones]) or None if line is empty/comment/unparseable.
    """
    # Strip trailing '#' comment first
    clean = _strip_comment(line)
    clean_stripped = clean.strip()
    # Skip empty lines and ';' comment lines (CMUdict header convention: ';;;')
    if not clean_stripped or clean_stripped.startswith(";"):
        return None
    parts = clean_stripped.split()
    if len(parts) < 2:
        return None
    word = parts[0].upper()
    phones = parts[1:]
    return word, phones


def _strip_stress(phone: str) -> str:
    """Strip trailing stress digit (0, 1, 2) from an ARPAbet phone symbol.

    CMUdict vowels carry stress: AH0 (unstressed), AH1 (primary), AH2 (secondary).
    Consonants are unaffected. Stripping yields the canonical 39-phone inventory.

    Args:
        phone: ARPAbet phone, possibly with trailing digit.

    Returns:
        Canonical phone without stress digit.
    """
    if phone and phone[-1].isdigit():
        return phone[:-1]
    return phone


def _parse_cmudict_bytes(content: bytes) -> dict[str, list[list[str]]]:
    """Parse one immutable Latin-1 CMUdict byte snapshot."""
    result: dict[str, list[list[str]]] = {}
    for line in content.decode("latin-1").splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        word, phones = parsed
        base_word = word.split("(")[0] if "(" in word else word
        result.setdefault(base_word, []).append(phones)
    return result


def audit_comment_stripping_bytes(content: bytes) -> CommentStrippingAudit:
    """Run canonical and deliberately naive inventories on the same bytes."""
    naive: set[str] = set()
    for line in content.decode("latin-1").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        fields = stripped.split()
        if len(fields) >= 2:
            naive.update(_strip_stress(field) for field in fields[1:])

    dictionary = _parse_cmudict_bytes(content)
    canonical = build_phoneme_inventory(dictionary)
    naive_inventory = frozenset(naive)
    return CommentStrippingAudit(
        canonical_inventory=canonical,
        naive_inventory=naive_inventory,
        leaked_comment_tokens=naive_inventory - canonical,
    )


def load_pinned_cmudict_source(
    path: str | Path | None = None,
    *,
    download: bool = True,
) -> PinnedCMUDictSource:
    """Read, authenticate, and derive from exactly one official snapshot.

    Unlike the fixture-friendly :func:`load_cmudict`, this scientific-writer
    entry point always requires AILM's exact immutable revision.  All returned
    derivations are computed from the same in-memory bytes that passed the pin.
    """
    if path is None:
        path = resolve_raw_corpus_path("cmudict.dict")
    source = Path(path)
    if not source.exists():
        if not download:
            raise FileNotFoundError(
                f"CMUdict not found at {source}. Run with download=True or provide the path."
            )
        logger.info("Downloading pinned CMUdict from %s -> %s", CMUDICT_URL, source)
        content = _download_cmudict_bytes(
            url=CMUDICT_URL,
            expected_sha256=CMUDICT_RAW_SHA256,
            expected_size_bytes=CMUDICT_RAW_SIZE_BYTES,
        )
        verify_pinned_cmudict_bytes(content)
        atomic_publish_bytes(source, content)
    else:
        if not source.is_file():
            raise FileNotFoundError(f"CMUdict source is not a file: {source}")
        content = source.read_bytes()
        verify_pinned_cmudict_bytes(content)

    dictionary = _parse_cmudict_bytes(content)
    return PinnedCMUDictSource(
        content=content,
        dictionary=dictionary,
        comment_audit=audit_comment_stripping_bytes(content),
    )


def load_cmudict(
    path: str | Path | None = None,
    *,
    download: bool = True,
    url: str = CMUDICT_URL,
    expected_sha256: str | None = None,
    expected_size_bytes: int = CMUDICT_RAW_SIZE_BYTES,
) -> dict[str, list[list[str]]]:
    """Load CMU Pronouncing Dictionary.

    Args:
        path: Path to cmudict.dict. If omitted, a source checkout uses
            ``vendor/cmudict.dict`` and an installed package uses its writable
            user cache (overridable with ``AILM_CORPUS_CACHE_DIR``).
        download: If True and the file is missing, download it.
        url: Download endpoint. Defaults to :data:`CMUDICT_URL`; exposed so an
            offline caller can use an internal mirror.
        expected_sha256: Optional digest for a custom mirror. The official
            :data:`CMUDICT_URL` is always checked against
            :data:`CMUDICT_RAW_SHA256` before its bytes are cached.
        expected_size_bytes: Exact expected raw-body extent. It defaults to the
            official pinned extent; fixture or non-identical mirror bodies must
            declare their own extent alongside ``expected_sha256``.

    Returns:
        dict mapping WORD → list of pronunciations.
        Each pronunciation is a list of ARPAbet phones WITH stress digits.
        Multiple pronunciations for a word appear as WORD(2), WORD(3) keys.

    Raises:
        FileNotFoundError: If file is missing and download=False.
        requests.RequestException: If download fails.
        ValueError: If source identity is incomplete or downloaded bytes
            disagree with the required extent or SHA-256.
    """
    require_pinned_existing = path is None and expected_sha256 is None
    if path is None:
        path = resolve_raw_corpus_path("cmudict.dict")

    path = Path(path)

    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"CMUdict not found at {path}. Run with download=True or provide the path."
            )
        if url != CMUDICT_URL and expected_sha256 is None:
            raise ValueError("a custom CMUdict mirror requires expected_sha256")
        logger.info("Downloading CMUdict from %s -> %s", url, path)
        if url == CMUDICT_URL:
            download_sha256 = CMUDICT_RAW_SHA256
            download_size_bytes = CMUDICT_RAW_SIZE_BYTES
        else:
            if expected_sha256 is None:  # pragma: no cover - guarded above
                raise ValueError("a custom CMUdict mirror requires expected_sha256")
            download_sha256 = expected_sha256
            download_size_bytes = expected_size_bytes
        content = _download_cmudict_bytes(
            url=url,
            expected_sha256=download_sha256,
            expected_size_bytes=download_size_bytes,
        )
        if url == CMUDICT_URL:
            verify_pinned_cmudict_bytes(content)
        atomic_publish_bytes(path, content)
        logger.info("Downloaded %d bytes", len(content))
    else:
        if not path.is_file():
            raise FileNotFoundError(f"CMUdict source is not a file: {path}")
        content = path.read_bytes()
        if require_pinned_existing:
            verify_pinned_cmudict_bytes(content)
        elif expected_sha256 is not None:
            _validate_cmudict_expectations(
                expected_sha256=expected_sha256,
                expected_size_bytes=expected_size_bytes,
            )
            measured_digest = hashlib.sha256(content).hexdigest()
            if len(content) != expected_size_bytes or measured_digest != expected_sha256:
                raise ValueError(
                    "CMUdict mirror source does not match the declared identity: "
                    f"bytes={len(content)} (expected {expected_size_bytes}), "
                    f"sha256={measured_digest} (expected {expected_sha256})"
                )

    result = _parse_cmudict_bytes(content)

    logger.info("Loaded CMUdict: %d base words", len(result))
    return result


def build_phoneme_inventory(
    cmudict: dict[str, list[list[str]]],
) -> frozenset[str]:
    """Extract the ARPAbet phoneme inventory seen in a CMUdict (stress digits stripped).

    Returns whatever phonemes appear in the provided dict — does NOT assert that
    the count equals 39. To enforce the canonical 39-phoneme count (e.g., after
    loading the full CMUdict), call assert_full_inventory() on the result.

    Args:
        cmudict: Output of load_cmudict(), or any word→pronunciations dict.

    Returns:
        frozenset of canonical ARPAbet phoneme symbols (stress digits stripped).
    """
    inventory: set[str] = set()
    for pronunciations in cmudict.values():
        for phones in pronunciations:
            for phone in phones:
                inventory.add(_strip_stress(phone))
    return frozenset(inventory)


def assert_full_inventory(inventory: frozenset[str]) -> None:
    """Assert that a phoneme inventory has exactly 39 canonical ARPAbet phonemes.

    Call this after build_phoneme_inventory(load_cmudict(...)) to enforce the
    canonical 39-phone invariant.  Do NOT call on small/test dicts.

    Args:
        inventory: Return value of build_phoneme_inventory().

    Raises:
        AssertionError: If count != 39, likely due to comment-stripping failure.
    """
    count = len(inventory)
    # Explicit raise: `python -O` strips `assert`, and this is the invariant that
    # catches a comment-stripping failure before a malformed 39-phoneme inventory
    # reaches Level 2. AssertionError is kept because the docstring declares it.
    if count != EXPECTED_PHONEME_COUNT:
        raise AssertionError(
            f"Phoneme inventory has {count} phones, expected {EXPECTED_PHONEME_COUNT}. "
            f"This almost certainly means '#' comment stripping was skipped during parsing. "
            f"Extra tokens found: {sorted(inventory - CANONICAL_ARPABET)}"
        )


def word_to_phones(
    word: str,
    cmudict: dict[str, list[list[str]]],
    *,
    strip_stress: bool = True,
) -> list[str] | None:
    """Look up a word's PRIMARY phoneme sequence in CMUdict.

    Returns the first pronunciation only. CMUdict lists alternates for many words
    (``READ`` has two), and every AILM level that consumes this — the Stage 2
    phoneme bigram model and the G2P coupling — is defined over one pronunciation
    per word, so the primary is the whole contract.

    This used to take a ``primary_only`` flag whose two branches were identical
    (``pronunciations[0] if primary_only else pronunciations[0]``), so passing
    ``primary_only=False`` silently returned the primary anyway. The flag is gone
    rather than implemented: a parameter that advertises a capability the callers
    never wanted is worse than its absence, because a future caller would trust it.
    Callers needing alternates should read ``cmudict[word.upper()]`` directly,
    which is already a list of every pronunciation.

    Args:
        word: Word to look up (case-insensitive).
        cmudict: Output of load_cmudict().
        strip_stress: If True, strip stress digits from phones.

    Returns:
        The primary pronunciation's phones, or None if the word is OOV.
    """
    pronunciations = cmudict.get(word.upper())
    if not pronunciations:
        return None
    phones = pronunciations[0]
    if strip_stress:
        phones = [_strip_stress(p) for p in phones]
    return phones


def verify_cmudict(path: str | Path) -> dict:
    """Load and validate a CMUdict file. Returns a verification report.

    Args:
        path: Path to cmudict.dict file.

    Returns:
        dict with keys:
          - entry_count: int (number of base words)
          - phoneme_count: int (should be 39)
          - sample_entries: dict of 5 example words with pronunciations
          - phoneme_inventory: sorted list of phoneme symbols
          - status: "ok" or "error"
          - error: error message if status is "error"
    """
    try:
        d = load_cmudict(path, download=False)
        inventory = build_phoneme_inventory(d)
        assert_full_inventory(inventory)
        samples = {w: d[w] for w in list(d.keys())[:5]}
        return {
            "entry_count": len(d),
            "phoneme_count": len(inventory),
            "phoneme_inventory": sorted(inventory),
            "sample_entries": samples,
            "status": "ok",
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
