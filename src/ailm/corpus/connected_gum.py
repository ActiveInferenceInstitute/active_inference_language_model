"""Authenticated raw-text documents for the frozen connected GUM evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROTOCOL_SHA256: Final = "c2ad1ee3b9aa285b5ef80ff29895665e959748bb9a90e94d1890ff18c7b2578d"
MANIFEST_SHA256: Final = "ca5f0e69d07c3328ec23f9dec6647b71a05cd3ba2e56483fd812cae4e80cf2e0"
TEST_SHA256: Final = "cd96a285e7339f401f4803dd0f4f61109c51692e3ef6f9186a1f977544080b69"
TEST_BYTES: Final = 2950684
EXCLUDED_DOCUMENTS: Final = (
    "GUM_academic_discrimination",
    "GUM_academic_eegimaa",
    "GUM_bio_dvorak",
    "GUM_bio_jespersen",
    "GUM_conversation_lambada",
    "GUM_conversation_retirement",
    "GUM_court_insanity",
    "GUM_court_mitigation",
)


class GumProtocolError(ValueError):
    """Frozen input, population or raw-observation integrity was violated."""


@dataclass(frozen=True, slots=True)
class GumDocument:
    document_id: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        """The registered raw input; source sentence boundaries add only a space."""
        return " ".join(self.sentences)


@dataclass(frozen=True, slots=True)
class GumPopulation:
    """Authenticated population identity carried into every recorded measurement."""

    corpus_sha256: str
    protocol_sha256: str
    input_manifest_sha256: str
    excluded_documents: tuple[str, ...]
    document_ids: tuple[str, ...]


def authenticated_bytes(
    path: Path, expected_sha256: str, expected_size: int | None = None
) -> bytes:
    """Authenticate exact bytes before any parsing or inference."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise GumProtocolError(f"SHA-256 mismatch: {path.name}")
    if expected_size is not None and len(content) != expected_size:
        raise GumProtocolError(f"byte extent mismatch: {path.name}")
    return content


def validate_registration(root: Path) -> Path:
    """Bind the original local registration and input manifest, without scoring."""
    directory = root / "data" / "connected_gum"
    protocol = json.loads(authenticated_bytes(directory / "protocol.json", PROTOCOL_SHA256))
    manifest = json.loads(authenticated_bytes(directory / "input_manifest.json", MANIFEST_SHA256))
    if tuple(protocol["excluded_documents"]) != EXCLUDED_DOCUMENTS:
        raise GumProtocolError("excluded document identities differ from the registration")
    if protocol["strength"] != 0.5 or protocol["input_manifest_sha256"] != MANIFEST_SHA256:
        raise GumProtocolError("registered model or manifest binding differs")
    test = [item for item in manifest["files"] if item["split"] == "test"]
    if len(test) != 1 or test[0]["sha256"] != TEST_SHA256 or test[0]["bytes"] != TEST_BYTES:
        raise GumProtocolError("test source identity differs")
    corpus = root / "vendor" / "en_gum-ud-test.conllu"
    authenticated_bytes(corpus, TEST_SHA256, TEST_BYTES)
    return corpus


def read_raw_documents(
    path: Path, *, expected_sha256: str, expected_size: int | None = None
) -> tuple[GumDocument, ...]:
    """Read only document IDs and sentence text; reject unrepresented sentences.

    Token rows establish that a sentence exists but their fields are never
    decoded. A sentence without its own nonempty text comment is an error,
    never a silently dropped observation or a reconstruction from gold tokens.
    """
    content = authenticated_bytes(path, expected_sha256, expected_size).decode("utf-8")
    documents: list[GumDocument] = []
    document_id: str | None = None
    seen: set[str] = set()
    sentences: list[str] = []
    text: str | None = None
    has_rows = False

    def finish_sentence() -> None:
        nonlocal text, has_rows
        if has_rows or text is not None:
            if document_id is None or not text or not has_rows:
                raise GumProtocolError("sentence lacks document identity, text or token rows")
            sentences.append(text)
        text, has_rows = None, False

    def finish_document() -> None:
        if document_id is not None:
            if not sentences:
                raise GumProtocolError(f"empty document: {document_id}")
            documents.append(GumDocument(document_id, tuple(sentences)))

    for line in content.splitlines():
        if line.startswith("# newdoc id = "):
            finish_sentence()
            finish_document()
            document_id = line.removeprefix("# newdoc id = ").strip()
            if not document_id or document_id in seen or "\0" in document_id:
                raise GumProtocolError("missing or repeated document identity")
            seen.add(document_id)
            sentences = []
        elif line.startswith("# text = "):
            if text is not None:
                raise GumProtocolError("multiple text comments for one sentence")
            text = line.removeprefix("# text = ").strip()
        elif not line.strip():
            finish_sentence()
        elif not line.startswith("#"):
            has_rows = True
    finish_sentence()
    finish_document()
    if not documents:
        raise GumProtocolError("empty corpus")
    return tuple(documents)


def select_unexamined_documents(
    documents: tuple[GumDocument, ...], excluded: tuple[str, ...]
) -> tuple[GumDocument, ...]:
    """Exclude exactly the registered identities, refusing absent or vacuous populations."""
    identities = [document.document_id for document in documents]
    if len(set(identities)) != len(identities) or len(set(excluded)) != len(excluded):
        raise GumProtocolError("duplicate population identity")
    if not set(excluded) <= set(identities):
        raise GumProtocolError("registered excluded documents are absent")
    selected = tuple(document for document in documents if document.document_id not in excluded)
    if len(selected) < 2:
        raise GumProtocolError("document uncertainty requires a nonempty paired population")
    return selected


def load_frozen_population(root: Path) -> tuple[GumPopulation, tuple[GumDocument, ...]]:
    """Authenticate the registered canonical inputs and complete unexamined population."""
    corpus = validate_registration(root)
    documents = read_raw_documents(corpus, expected_sha256=TEST_SHA256, expected_size=TEST_BYTES)
    selected = select_unexamined_documents(documents, EXCLUDED_DOCUMENTS)
    population = GumPopulation(
        TEST_SHA256,
        PROTOCOL_SHA256,
        MANIFEST_SHA256,
        EXCLUDED_DOCUMENTS,
        tuple(document.document_id for document in selected),
    )
    return population, selected
