"""Universal Dependencies English EWT loader + case oracle for AILM.

Provides the case-labeled corpus for Level 4 (Clause/Case) validation.

CASE ORACLE — RESEARCH CLAIM:
  English marks only Nominative, Accusative, and Genitive morphologically.
  CEREBRUM's fuller inventory (DAT, INS, LOC, ABL, VOC) is NOT directly annotated
  in UD. This module derives them via a documented, deterministic mapping from
  UD dependency relations and adposition lemmas.
  THE MAPPING IS A RESEARCH CLAIM, NOT GROUND TRUTH.
  Version it, test it, and have it reviewed before any accuracy number derived
  from it is treated as meaningful. See docs/design/cerebrum_mapping.md.

Source: Universal Dependencies English Web Treebank (EWT)
License: CC BY-SA 4.0
URL: https://github.com/UniversalDependencies/UD_English-EWT
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import requests

from ailm.cases import ADPOSITION_TO_CASE, UD_DEPREL_TO_CASE, CaseRole
from ailm.corpus._download import atomic_publish_bytes, read_authenticated_response
from ailm.resources import resolve_raw_corpus_path

logger = logging.getLogger(__name__)

UD_EWT_REVISION = "4a4d77f599ea53cc405f85d0cec4b2f14f81d42b"
UD_EWT_DEV_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    f"UD_English-EWT/{UD_EWT_REVISION}/en_ewt-ud-dev.conllu"
)
UD_EWT_RAW_SHA256 = "39239e0a60db3ae68f4b7036189f11b6692741d10ff8240dd91f74f2760d90f8"
UD_EWT_RAW_SIZE_BYTES = 1_898_574
UD_EWT_TEST_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    f"UD_English-EWT/{UD_EWT_REVISION}/en_ewt-ud-test.conllu"
)
#: Measured once on first contact with the pinned revision's test split
#: (2026-08-26, bounded redirect-free streaming read); never guessed.
UD_EWT_TEST_RAW_SHA256 = "fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a"
UD_EWT_TEST_RAW_SIZE_BYTES = 1_900_101
UD_EWT_TEST_SOURCE_METADATA: Mapping[str, str | int] = MappingProxyType(
    {
        "source_url": UD_EWT_TEST_URL,
        "source_revision": UD_EWT_REVISION,
        "source_sha256": UD_EWT_TEST_RAW_SHA256,
        "source_size_bytes": UD_EWT_TEST_RAW_SIZE_BYTES,
    }
)
UD_EWT_SOURCE_METADATA: Mapping[str, str | int] = MappingProxyType(
    {
        "source_url": UD_EWT_DEV_URL,
        "source_revision": UD_EWT_REVISION,
        "source_sha256": UD_EWT_RAW_SHA256,
        "source_size_bytes": UD_EWT_RAW_SIZE_BYTES,
    }
)
_GIT_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

#: Stable provenance names for the mutually exclusive routes by which the
#: English oracle can label a token.  ``adposition_token`` is deliberately
#: explicit: the mapped lemma belongs to the ADP token itself.  The current
#: implementation does not transfer that role to the token's nominal head.
CASE_ORACLE_PATHS: tuple[str, ...] = ("morphology", "deprel", "adposition_token")

# Universal Dependencies v2 closed UPOS inventory. Artifact loaders use this
# external alphabet rather than accepting arbitrary uppercase strings: case
# labels themselves are uppercase and would otherwise pass that weak guard.
UD_UPOS_TAGS: frozenset[str] = frozenset(
    {
        "ADJ",
        "ADP",
        "ADV",
        "AUX",
        "CCONJ",
        "DET",
        "INTJ",
        "NOUN",
        "NUM",
        "PART",
        "PRON",
        "PROPN",
        "PUNCT",
        "SCONJ",
        "SYM",
        "VERB",
        "X",
    }
)

_MORPHOLOGICAL_CASE_TO_ROLE: Mapping[str, CaseRole] = MappingProxyType(
    {
        "Nom": CaseRole.NOM,
        "Acc": CaseRole.ACC,
        "Gen": CaseRole.GEN,
    }
)


class UDSourceIntegrityError(ValueError):
    """UD EWT bytes or source identity do not match the declared pin."""


def validate_ud_ewt_source_identity(
    *,
    url: str,
    source_revision: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> None:
    """Validate a complete immutable source declaration before any I/O."""
    if _GIT_REVISION.fullmatch(source_revision) is None:
        raise UDSourceIntegrityError("UD EWT source revision must be a full Git commit digest")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise UDSourceIntegrityError("UD EWT source SHA-256 must be a lowercase digest")
    if (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes <= 0
    ):
        raise UDSourceIntegrityError("UD EWT source byte extent must be positive")
    lowered_segments = {segment.lower() for segment in url.split("/")}
    if {"master", "main"} & lowered_segments:
        raise UDSourceIntegrityError("UD EWT source URL must not name a moving branch")
    for canonical_url, canonical_metadata in (
        (UD_EWT_DEV_URL, UD_EWT_SOURCE_METADATA),
        (UD_EWT_TEST_URL, UD_EWT_TEST_SOURCE_METADATA),
    ):
        if url == canonical_url:
            declared = {
                "source_revision": source_revision,
                "source_sha256": expected_sha256,
                "source_size_bytes": expected_size_bytes,
            }
            expected = {key: canonical_metadata[key] for key in declared}
            if declared != expected:
                raise UDSourceIntegrityError(
                    f"official UD EWT URL source identity disagrees with the pin: {declared}"
                )


def verify_ud_ewt_bytes(
    content: bytes,
    *,
    expected_sha256: str = UD_EWT_RAW_SHA256,
    expected_size_bytes: int = UD_EWT_RAW_SIZE_BYTES,
) -> str:
    """Return the digest only when raw bytes match the declared identity."""
    measured_sha256 = hashlib.sha256(content).hexdigest()
    measured_size = len(content)
    if measured_sha256 != expected_sha256 or measured_size != expected_size_bytes:
        raise UDSourceIntegrityError(
            "UD EWT source does not match the declared revision: "
            f"bytes={measured_size} (expected {expected_size_bytes}), "
            f"sha256={measured_sha256} (expected {expected_sha256})"
        )
    return measured_sha256


def validate_ud_ewt_source_metadata(payload: Mapping[str, object]) -> None:
    """Require a derived artifact to retain the canonical raw source identity."""
    missing = sorted(set(UD_EWT_SOURCE_METADATA) - set(payload))
    if missing:
        raise UDSourceIntegrityError(f"UD EWT artifact lacks pinned source metadata: {missing}")
    mismatches = {
        key: (expected, payload[key])
        for key, expected in UD_EWT_SOURCE_METADATA.items()
        if payload[key] != expected
    }
    if mismatches:
        raise UDSourceIntegrityError(
            f"UD EWT artifact source metadata disagrees with the pin: {mismatches}"
        )


def _download_ud_ewt_bytes(*, url: str, expected_sha256: str, expected_size_bytes: int) -> bytes:
    """Acquire one bounded, redirect-free, authenticated UD EWT body."""
    with requests.get(
        url,
        timeout=(10, 60),
        stream=True,
        allow_redirects=False,
        headers={"Accept-Encoding": "identity"},
    ) as response:
        return read_authenticated_response(
            response,
            expected_size_bytes=expected_size_bytes,
            expected_sha256=expected_sha256,
            source_name="UD EWT",
            integrity_error=UDSourceIntegrityError,
        )


@dataclass
class Token:
    """One token from a CoNLL-U sentence."""

    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: dict[str, str] = field(default_factory=dict)
    head: int = 0
    deprel: str = "_"
    misc: dict[str, str] = field(default_factory=dict)
    case_morphological: str | None = None  # from UD feats["Case"] if present
    case_derived: CaseRole | None = None  # from our oracle mapping


@dataclass
class Sentence:
    """One sentence from a CoNLL-U file."""

    sent_id: str
    text: str
    tokens: list[Token] = field(default_factory=list)
    document_id: str = ""

    def tokens_with_case(self) -> list[Token]:
        """Return only tokens that have a derived case assignment."""
        return [t for t in self.tokens if t.case_derived is not None]


def document_stride_partition(
    sentences: Sequence[Sentence], *, test_every: int
) -> tuple[list[str], dict[str, str]]:
    """Return ordered document identities and a deterministic disjoint split.

    CoNLL-U ``# newdoc id`` metadata is the outer sampling identity for EWT.
    The held-out rule is applied to that ordered identity list *before* any
    task-specific sentence or token filtering: document ``i`` is test exactly
    when ``i % test_every == 0``.  Reappearing document blocks are rejected,
    since they make an order-based partition ambiguous and can silently put one
    source document in both arms.

    Args:
        sentences: Parsed sentences carrying authenticated document metadata.
        test_every: Document-index stride; must leave both train and test arms.

    Returns:
        ``(ordered_ids, split_by_id)`` where every value is ``"train"`` or
        ``"test"``.

    Raises:
        ValueError: On missing/NUL/repeated-block identities, invalid stride, or
            a population too small to leave both arms non-empty.
    """
    if isinstance(test_every, bool) or not isinstance(test_every, int) or test_every < 2:
        raise ValueError(f"test_every must be an integer of at least two, got {test_every!r}")
    if not sentences:
        raise ValueError("document partition needs at least one parsed sentence")

    ordered_ids: list[str] = []
    closed: set[str] = set()
    previous: str | None = None
    for index, sentence in enumerate(sentences):
        document_id = sentence.document_id
        if not isinstance(document_id, str) or not document_id or "\0" in document_id:
            raise ValueError(f"sentence {sentence.sent_id or index!r} lacks a usable # newdoc id")
        if document_id == previous:
            continue
        if previous is not None:
            closed.add(previous)
        if document_id in closed:
            raise ValueError(f"document id {document_id!r} reappears in a later block")
        ordered_ids.append(document_id)
        previous = document_id

    split_by_id = {
        document_id: ("test" if index % test_every == 0 else "train")
        for index, document_id in enumerate(ordered_ids)
    }
    if set(split_by_id.values()) != {"train", "test"}:
        raise ValueError(
            f"document stride {test_every} over {len(ordered_ids)} documents does not leave "
            "both train and test arms"
        )
    return ordered_ids, split_by_id


def document_identity_sha256(document_ids: Sequence[str]) -> str:
    """Digest one ordered document-identity alphabet without concatenation ambiguity."""
    if not document_ids:
        raise ValueError("document identity digest needs at least one id")
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("document identity digest refuses duplicate ids")
    encoded = bytearray()
    for document_id in document_ids:
        if not isinstance(document_id, str) or not document_id or "\0" in document_id:
            raise ValueError(f"invalid document id {document_id!r}")
        payload = document_id.encode("utf-8")
        encoded.extend(len(payload).to_bytes(8, "big"))
        encoded.extend(payload)
    return hashlib.sha256(encoded).hexdigest()


def _parse_feats(feat_str: str) -> dict[str, str]:
    """Parse a CoNLL-U feature string into a dict.

    "Case=Nom|Number=Sing" → {"Case": "Nom", "Number": "Sing"}
    "_" → {}
    """
    if feat_str == "_":
        return {}
    result: dict[str, str] = {}
    for item in feat_str.split("|"):
        if "=" in item:
            key, val = item.split("=", 1)
            result[key] = val
    return result


def _parse_misc(misc_str: str) -> dict[str, str]:
    """Parse a CoNLL-U misc string into a dict."""
    if misc_str == "_":
        return {}
    result: dict[str, str] = {}
    for item in misc_str.split("|"):
        if "=" in item:
            key, val = item.split("=", 1)
            result[key] = val
    return result


def _syntactic_case_assignment(token: Token) -> tuple[CaseRole | None, str | None]:
    """Apply only the non-morphological oracle routes to one token.

    This is the comparison needed for the heuristic-versus-gold audit: calling
    the full oracle on a token carrying ``feats[Case]`` would simply return the
    gold label and manufacture perfect agreement.
    """
    if token.deprel in UD_DEPREL_TO_CASE:
        return UD_DEPREL_TO_CASE[token.deprel], "deprel"
    if token.upos == "ADP" and token.deprel == "case":
        role = ADPOSITION_TO_CASE.get(token.lemma.lower())
        if role is not None:
            return role, "adposition_token"
    return None, None


def case_oracle_assignment(token: Token) -> tuple[CaseRole | None, str | None]:
    """Return the oracle role and the route that assigned it.

    The adposition route is token-level.  For an ADP such as ``to``, the ADP's
    own lemma supplies the ADP token's role; the nominal named by ``token.head``
    is neither consulted nor relabeled.  Head relationships are audited
    separately by :func:`case_oracle_scope_audit` so that the published labels
    remain unchanged while their construct scope is visible.
    """
    morphological = _MORPHOLOGICAL_CASE_TO_ROLE.get(token.case_morphological or "")
    if morphological is not None:
        return morphological, "morphology"
    return _syntactic_case_assignment(token)


def _apply_case_oracle(token: Token) -> CaseRole | None:
    """Apply the derived case oracle to assign a CaseRole to one token.

    RESEARCH CLAIM — see module docstring. This mapping is approximate and
    English-specific. It must be reviewed before accuracy numbers are reported.

    Strategy:
    1. Check UD morphological feats["Case"] first (most reliable).
    2. Fall back to UD_DEPREL_TO_CASE for the token's deprel.
    3. If the token is an ADP with deprel ``case``, map that token's own lemma.

    Args:
        token: The token to assign a case role to. No sentence or head lookup is
            performed; the return always labels this token.

    Returns:
        CaseRole or None if no assignment is possible.
    """
    return case_oracle_assignment(token)[0]


def case_oracle_scope_audit(sentences: Sequence[Sentence]) -> dict[str, object]:
    """Measure which tokens the oracle validates, duplicates, and disagrees on.

    The heuristic-versus-gold percentage covers only tokens that carry real
    morphology *and* receive a non-morphological syntactic label.  It cannot
    validate a route whose tokens carry no gold morphology.  This audit records
    that route composition explicitly and, for the token-level adposition route,
    resolves each UD head by token ID to measure whether the nominal head also
    receives a role and whether the two roles agree.

    Returns:
        A JSON-serialisable sufficient-count record. Every percentage used in
        prose can be recomputed from its numerator and denominator.

    Raises:
        ValueError: If a sentence contains duplicate integer token IDs, because
            a head lookup would then be ambiguous.
    """
    derived_paths: Counter[str] = Counter({path: 0 for path in CASE_ORACLE_PATHS})
    fidelity_paths: Counter[str] = Counter({"deprel": 0, "adposition_token": 0})
    fidelity_tokens = 0
    fidelity_agreements = 0
    adposition = Counter(
        {
            "tokens": 0,
            "morphological_case_tokens": 0,
            "head_resolved_tokens": 0,
            "head_unresolved_tokens": 0,
            "head_also_labeled_tokens": 0,
            "head_role_agreements": 0,
            "head_role_disagreements": 0,
        }
    )

    for sentence in sentences:
        tokens_by_id = {token.id: token for token in sentence.tokens}
        if len(tokens_by_id) != len(sentence.tokens):
            raise ValueError(f"sentence {sentence.sent_id!r} contains duplicate token IDs")

        for token in sentence.tokens:
            role, path = case_oracle_assignment(token)
            if role is not None:
                if path is None or path not in CASE_ORACLE_PATHS:  # pragma: no cover
                    raise ValueError(f"unknown case-oracle path {path!r}")
                derived_paths[path] += 1

            gold = _MORPHOLOGICAL_CASE_TO_ROLE.get(token.case_morphological or "")
            syntactic, syntactic_path = _syntactic_case_assignment(token)
            if gold is not None and syntactic is not None:
                if (
                    syntactic_path is None or syntactic_path not in fidelity_paths
                ):  # pragma: no cover - closed helper output
                    raise ValueError(f"unknown syntactic path {syntactic_path!r}")
                fidelity_tokens += 1
                fidelity_paths[syntactic_path] += 1
                fidelity_agreements += int(syntactic is gold)

            if path != "adposition_token":
                continue
            adposition["tokens"] += 1
            adposition["morphological_case_tokens"] += int(token.case_morphological is not None)
            head = tokens_by_id.get(token.head)
            if head is None:
                adposition["head_unresolved_tokens"] += 1
                continue
            adposition["head_resolved_tokens"] += 1
            head_role, _head_path = case_oracle_assignment(head)
            if head_role is None:
                continue
            adposition["head_also_labeled_tokens"] += 1
            if head_role is role:
                adposition["head_role_agreements"] += 1
            else:
                adposition["head_role_disagreements"] += 1

    return {
        "full_derived_tokens": sum(derived_paths.values()),
        "derived_path_tokens": dict(derived_paths),
        "heuristic_vs_gold": {
            "tokens_with_both": fidelity_tokens,
            "agreements": fidelity_agreements,
            "source_path_tokens": dict(fidelity_paths),
        },
        "adposition_token": dict(adposition),
    }


def parse_conllu(text: str) -> list[Sentence]:
    """Parse CoNLL-U format text into Sentence objects.

    Handles:
    - stateful "# newdoc id = ..." metadata lines
    - "# sent_id = ..." metadata lines
    - "# text = ..." metadata lines
    - Token lines (10 tab-separated fields)
    - Empty lines as sentence boundaries
    - Multi-word tokens (IDs like "1-2") are skipped

    Args:
        text: Raw CoNLL-U file content.

    Returns:
        List of Sentence objects with case oracle applied.
    """
    sentences: list[Sentence] = []
    current_sent_id = ""
    current_text = ""
    current_tokens: list[Token] = []
    current_document_id = ""

    def _flush():
        nonlocal current_sent_id, current_text, current_tokens
        if current_tokens:
            sent = Sentence(
                sent_id=current_sent_id,
                text=current_text,
                tokens=current_tokens,
                document_id=current_document_id,
            )
            # Apply the token-level oracle after parsing. No head lookup is part
            # of label assignment; head relations are a separate scope audit.
            for tok in sent.tokens:
                tok.case_derived = _apply_case_oracle(tok)
            sentences.append(sent)
        current_sent_id = ""
        current_text = ""
        current_tokens = []

    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            _flush()
            continue
        if line.startswith("# newdoc id"):
            current_document_id = line.split("=", 1)[-1].strip()
        elif line.startswith("# sent_id"):
            current_sent_id = line.split("=", 1)[-1].strip()
        elif line.startswith("# text"):
            current_text = line.split("=", 1)[-1].strip()
        elif line.startswith("#"):
            continue  # other comment
        else:
            parts = line.split("\t")
            if len(parts) < 10:
                continue
            tok_id_str = parts[0]
            # Skip multi-word tokens (e.g. "1-2")
            if "-" in tok_id_str or "." in tok_id_str:
                continue
            feats = _parse_feats(parts[5])
            token = Token(
                id=int(tok_id_str),
                form=parts[1],
                lemma=parts[2],
                upos=parts[3],
                xpos=parts[4],
                feats=feats,
                head=int(parts[6]) if parts[6] != "_" else 0,
                deprel=parts[7],
                misc=_parse_misc(parts[9]),
                case_morphological=feats.get("Case"),
            )
            current_tokens.append(token)

    _flush()  # final sentence
    logger.info("Parsed %d sentences from CoNLL-U text", len(sentences))
    return sentences


def ensure_ud_ewt_dev_path(
    path: str | Path | None = None,
    *,
    download: bool = True,
    url: str = UD_EWT_DEV_URL,
    source_revision: str = UD_EWT_REVISION,
    expected_sha256: str = UD_EWT_RAW_SHA256,
    expected_size_bytes: int = UD_EWT_RAW_SIZE_BYTES,
) -> Path:
    """Resolve, fetch if necessary, and verify the raw development split.

    An existing file is verified on every call. A bad cache entry is never
    silently replaced: the caller must inspect and remove or relocate it. A
    downloaded body is streamed through an exact byte bound and incremental
    hash before a durable atomic sibling write.

    Args:
        path: Path to en_ewt-ud-dev.conllu. If omitted, a source checkout uses
            ``vendor/`` and an installed package uses its writable user cache
            (overridable with ``AILM_CORPUS_CACHE_DIR``).
        download: Download the corpus if the local file is missing.
        url: Download endpoint. Defaults to :data:`UD_EWT_DEV_URL`; exposed so
            an offline caller can use an internal mirror.
        source_revision: Full immutable Git revision represented by the bytes.
        expected_sha256: Required raw-body digest.
        expected_size_bytes: Required raw-body extent.

    Returns:
        Verified local path.
    """
    validate_ud_ewt_source_identity(
        url=url,
        source_revision=source_revision,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    if path is None:
        path = resolve_raw_corpus_path("en_ewt-ud-dev.conllu")
    path = Path(path)

    if path.exists():
        if not path.is_file():
            raise UDSourceIntegrityError(f"UD EWT source path is not a file: {path}")
        verify_ud_ewt_bytes(
            path.read_bytes(),
            expected_sha256=expected_sha256,
            expected_size_bytes=expected_size_bytes,
        )
        return path

    if not download:
        raise FileNotFoundError(
            f"UD EWT dev set not found at {path}. Run with download=True or provide the path."
        )
    logger.info("Downloading UD English EWT dev from %s", url)
    content = _download_ud_ewt_bytes(
        url=url,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    verify_ud_ewt_bytes(
        content,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    atomic_publish_bytes(path, content)
    logger.info("Downloaded and verified %d bytes", len(content))
    return path


def load_ud_ewt_dev(
    path: str | Path | None = None,
    *,
    download: bool = True,
    url: str = UD_EWT_DEV_URL,
    source_revision: str = UD_EWT_REVISION,
    expected_sha256: str = UD_EWT_RAW_SHA256,
    expected_size_bytes: int = UD_EWT_RAW_SIZE_BYTES,
) -> list[Sentence]:
    """Load UD EWT from the same immutable bytes that pass authentication."""
    content = read_ud_ewt_dev_bytes(
        path,
        download=download,
        url=url,
        source_revision=source_revision,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UDSourceIntegrityError("verified UD EWT bytes are not valid UTF-8") from exc
    return parse_conllu(text)


def read_ud_ewt_dev_bytes(
    path: str | Path | None = None,
    *,
    download: bool = True,
    url: str = UD_EWT_DEV_URL,
    source_revision: str = UD_EWT_REVISION,
    expected_sha256: str = UD_EWT_RAW_SHA256,
    expected_size_bytes: int = UD_EWT_RAW_SIZE_BYTES,
) -> bytes:
    """Acquire and authenticate the exact snapshot a parser may consume.

    ``ensure_ud_ewt_dev_path`` verifies the path at acquisition time, but no
    path stays immutable merely because a check succeeded.  This helper reads
    once more, authenticates those exact bytes, and returns the immutable byte
    value.  Callers parse the return value and never reopen the path, so an
    atomic replacement after this read cannot change the scientific input.
    """
    verified_path = ensure_ud_ewt_dev_path(
        path,
        download=download,
        url=url,
        source_revision=source_revision,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    content = verified_path.read_bytes()
    verify_ud_ewt_bytes(
        content,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    return content


def _default_test_path() -> str | Path:
    """Default local location for en_ewt-ud-test.conllu (source checkout: vendor/)."""
    return resolve_raw_corpus_path("en_ewt-ud-test.conllu")


def ensure_ud_ewt_test_path(
    path: str | Path | None = None,
    *,
    download: bool = True,
    url: str = UD_EWT_TEST_URL,
    source_revision: str = UD_EWT_REVISION,
    expected_sha256: str = UD_EWT_TEST_RAW_SHA256,
    expected_size_bytes: int = UD_EWT_TEST_RAW_SIZE_BYTES,
) -> Path:
    """Resolve, fetch if necessary, and verify the raw test split.

    Mechanical twin of :func:`ensure_ud_ewt_dev_path` for the held-out test
    split of the same pinned UD_English-EWT revision; only the filename and the
    measured byte identity differ.
    """
    return ensure_ud_ewt_dev_path(
        path if path is not None else _default_test_path(),
        download=download,
        url=url,
        source_revision=source_revision,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )


def load_ud_ewt_test(
    path: str | Path | None = None,
    *,
    download: bool = True,
    url: str = UD_EWT_TEST_URL,
    source_revision: str = UD_EWT_REVISION,
    expected_sha256: str = UD_EWT_TEST_RAW_SHA256,
    expected_size_bytes: int = UD_EWT_TEST_RAW_SIZE_BYTES,
) -> list[Sentence]:
    """Load the UD EWT test split from the same immutable authenticated bytes."""
    content = read_ud_ewt_dev_bytes(
        path if path is not None else _default_test_path(),
        download=download,
        url=url,
        source_revision=source_revision,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
    )
    try:
        text_body = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UDSourceIntegrityError("verified UD EWT bytes are not valid UTF-8") from exc
    return parse_conllu(text_body)


def case_oracle_stats(sentences: list[Sentence]) -> dict:
    """Count token-level case assignments from the oracle.

    Args:
        sentences: Output of load_ud_ewt_dev() or parse_conllu().

    Returns:
        dict with:
          total_tokens: int
          morphological_cases: dict mapping UD Case feat value to count
          derived_cases: dict mapping CaseRole name to count
    """
    total = 0
    morph: dict[str, int] = {}
    derived: dict[str, int] = {}

    for sent in sentences:
        for tok in sent.tokens:
            total += 1
            if tok.case_morphological:
                morph[tok.case_morphological] = morph.get(tok.case_morphological, 0) + 1
            if tok.case_derived is not None:
                name = tok.case_derived.name
                derived[name] = derived.get(name, 0) + 1

    return {
        "total_tokens": total,
        "morphological_cases": morph,
        "derived_cases": derived,
    }


def build_upos_lexicon(sentences: list[Sentence] | None = None) -> dict[str, str]:
    """Most-frequent UPOS tag per lowercased word form, from UD EWT.

    The only part-of-speech source available at inference time on arbitrary
    text: this project has no tagger, and a tagger would be a learned model of
    exactly the kind the architecture excludes. A word the lexicon does not
    carry has no tag here, and the caller decides what to do about that — which
    is the point of returning the mapping rather than a defaulted lookup. The
    dashboard, for instance, reports a defaulted tag as `fallback` so a guess is
    never displayed with the authority of a lookup.

    Maximum-likelihood over the corpus, so an ambiguous form takes its most
    common tag and the ambiguity is discarded. That is a real limitation and it
    is the reason the interface marks provenance rather than hiding it.

    Args:
        sentences: Parsed sentences. Defaults to the vendored UD EWT dev split.

    Returns:
        ``{word: upos}``, lowercased keys.
    """
    corpus = load_ud_ewt_dev() if sentences is None else sentences
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sentence in corpus:
        for token in sentence.tokens:
            counts[token.form.lower()][token.upos] += 1
    return {form: max(tags.items(), key=lambda pair: pair[1])[0] for form, tags in counts.items()}
