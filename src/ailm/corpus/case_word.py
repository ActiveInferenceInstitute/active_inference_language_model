"""The case-word (L4→L3) bridge artifact: word forms typed by case role.

Protocol P3 (`output/reports/omp-dispatch-20260908/evidence/
extensions-E1R-frozen/docs/extension_protocols.md` §4): the TODO-required
label-construction / split / observation account, built before the artifact.
This module owns the artifact's schema, its loader, and the per-token
label-construction helpers. The sender model, the runtime posterior, and the
L3 inlet live in :mod:`ailm.case_word_channel`.

## What this artifact is

One row per EWT dev token: ``(word_form, upos, case_label_or_null,
label_source)``. The oracle case label is a *sender-side fitting input*,
recorded per token with the route that assigned it
(:data:`ailm.corpus.ud.CASE_ORACLE_PATHS`), and is structurally unavailable to
any model observation — the model side observes UPOS only
(:data:`ailm.levels.case.FORBIDDEN_OBSERVATIONS`).

## Population status

The split is the same document-index stride (``test_every = 5``) over the same
authenticated ``# newdoc id`` values as the shipped case artifact
(``vendor/ud_ewt_case_counts.json``), cross-validated against its recorded
``split_indices`` at build and load time. The artifact is therefore the same
development population in a new representation: development evidence under a
fresh registration — never confirmatory, never "untouched". The EWT *test*
file's bytes are spent (decision 111) and are not read by anything here.

## Vocabulary normalization (declared, not post-hoc)

EWT word forms are lowercased and matched by exact string against the Stage 3
receiver vocabulary (``vendor/brown_word_counts.json`` ``word_freq`` — 2,000
lowercase types). No other folding or stemming. ``sender_map`` counts only
in-vocabulary lowercase forms; the message's support is the intersection, and
out-of-intersection receiver mass shrinks by ``(1 − α)`` under the mixing at
:func:`ailm.levels.clause_channel.mix_prior`.

## Genitive namespace

Role names are the :class:`ailm.cases.CaseRole` names exactly — ``GEN`` is the
linguistic Genitive role, and this artifact does not introduce or alias any
other generative reading of that name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ailm.cases import ADPOSITION_TO_CASE, UD_DEPREL_TO_CASE, CaseRole
from ailm.corpus.ud import (
    CASE_ORACLE_PATHS,
    UD_EWT_RAW_SHA256,
    UD_EWT_REVISION,
    UD_UPOS_TAGS,
    Token,
)
from ailm.resources import resolve_data_artifact

__all__ = [
    "CASE_WORD_ARTIFACT_SCHEMA",
    "CASE_WORD_OBSERVATION",
    "CASE_WORD_TEST_EVERY",
    "CaseWordArtifact",
    "load_case_word_artifact",
    "route_roles",
]

#: The schema name proposed by protocol P3 §4.4.
CASE_WORD_ARTIFACT_SCHEMA = "ailm-case-word-counts-v1"

#: The observation contract, enforced verbatim by the loader: the artifact
#: refuses to load if this string is missing or altered, because a sender-side
#: label that leaked into a model-observable field would be the Stage 4 trap.
CASE_WORD_OBSERVATION = (
    "UPOS only for any model observation; oracle case labels are sender-side "
    "fitting inputs, recorded per token with label_source"
)

#: The same document stride as the shipped case artifact.
CASE_WORD_TEST_EVERY = 5

#: Source provenance remains under ``data/``; installed runtimes read identical bytes.
DEFAULT_CASE_WORD_ARTIFACT = resolve_data_artifact("case_word/ud_ewt_case_word_counts.json")


def route_roles(token: Token) -> dict[str, CaseRole | None]:
    """Every oracle route's role for one token, computed independently.

    :func:`ailm.corpus.ud.case_oracle_assignment` short-circuits on the first
    matching route, so the artifact's per-token ``label_source`` names the
    *selected* route only. Pooling routes into one sender map is a declared,
    inspectable choice, and inspection needs the counterfactual: which other
    routes fire on the same token and whether they agree with the selected
    label. This helper computes each route directly from the token's own
    fields — morphology from ``feats[Case]``, deprel from the published map,
    adposition from the token's own lemma — without transferring any role to a
    head (the token-level adposition rule).
    """
    from ailm.corpus.ud import _MORPHOLOGICAL_CASE_TO_ROLE  # single definition reuse

    return {
        "morphology": _MORPHOLOGICAL_CASE_TO_ROLE.get(token.case_morphological or ""),
        "deprel": UD_DEPREL_TO_CASE.get(token.deprel),
        "adposition_token": (
            ADPOSITION_TO_CASE.get(token.lemma.lower())
            if token.upos == "ADP" and token.deprel == "case"
            else None
        ),
    }


@dataclass(frozen=True, slots=True)
class CaseWordRow:
    """One token row: ``(word_form, upos, case_label_or_null, label_source)``."""

    word_form: str
    upos: str
    case_label: str | None
    label_source: str | None


class CaseWordArtifact:
    """The case-word bridge artifact, validated on load.

    Attributes:
        provenance: The generator's recorded parameters and source identity.
        vocab: The Stage 3 receiver vocabulary in receiver index order; the
            message vectors are aligned to this order.
        train_documents: Per-document rows for the train split.
        test_documents: Per-document rows for the test split.
        sender_map: ``role -> {lowercase in-vocabulary word: count}``, fitted
            from train rows only.
        route_stats: Per-route coverage and agreement with the selected label.
        sender_support: Roles with at least one in-vocabulary training word.
    """

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != CASE_WORD_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unexpected artifact schema {payload.get('schema')!r}; "
                f"regenerate with scripts/generate_case_word_counts.py"
            )
        provenance = payload["provenance"]
        if provenance.get("observation") != CASE_WORD_OBSERVATION:
            raise ValueError(
                "case-word artifact provenance does not carry the exact "
                "UPOS-only observation contract; a sender-side label that "
                "reached a model-observable field would be the Stage 4 trap"
            )
        if provenance.get("source_sha256") != UD_EWT_RAW_SHA256:
            raise ValueError(
                f"case-word artifact cites source_sha256 "
                f"{provenance.get('source_sha256')!r}, not the authenticated "
                f"EWT dev revision {UD_EWT_REVISION}"
            )
        if provenance.get("test_every") != CASE_WORD_TEST_EVERY:
            raise ValueError(
                f"case-word artifact declares test_every "
                f"{provenance.get('test_every')!r}; this loader reads only the "
                f"case artifact's stride {CASE_WORD_TEST_EVERY}"
            )
        documents = payload["documents"]
        self.document_ids: list[str] = list(documents["ids"])
        self.split_indices: dict[str, list[int]] = {
            name: list(indices) for name, indices in documents["split_indices"].items()
        }
        self.provenance: dict = provenance
        self._validate_documents(provenance)

        self.vocab: list[str] = self._validate_vocab(payload["vocab"])
        self.word_to_index: dict[str, int] = {word: index for index, word in enumerate(self.vocab)}

        self.train_documents = self._validate_documents_rows(
            payload["populations"]["train"], "train"
        )
        self.test_documents = self._validate_documents_rows(payload["populations"]["test"], "test")

        self.sender_map: dict[str, dict[str, int]] = self._validate_sender_map(
            payload["sender_map"]
        )
        self.sender_support: frozenset[str] = frozenset(
            role for role, counts in self.sender_map.items() if counts
        )
        self.route_stats: dict[str, dict[str, int]] = payload["route_stats"]
        self._validate_route_stats()

        self._validate_digests(payload["digests"], payload)

    # ── validation pieces ────────────────────────────────────────────────────

    def _validate_documents(self, provenance: dict) -> None:
        """Document identity digest, stride partition, and split coverage.

        The cross-validation of ``split_indices`` against the shipped case
        artifact's recorded split is the generator's refusal
        (``scripts/generate_case_word_counts.py``) and the canonical
        artifact test's assertion: the loader validates this artifact's own
        internal consistency so small fixtures can exercise it.
        """
        from ailm.corpus.ud import document_identity_sha256

        if provenance.get("documents") != len(self.document_ids):
            raise ValueError("case-word artifact provenance document count is inconsistent")
        if provenance.get("document_identity_sha256") != document_identity_sha256(
            self.document_ids
        ):
            raise ValueError(
                "case-word artifact document identity digest disagrees with its ordered ids"
            )
        split_indices = self.split_indices
        expected_test = {i for i in range(len(self.document_ids)) if i % CASE_WORD_TEST_EVERY == 0}
        listed = sorted(split_indices["train"] + split_indices["test"])
        if listed != list(range(len(self.document_ids))):
            raise ValueError("case-word artifact split_indices do not partition the documents")
        if set(split_indices["train"]) & set(split_indices["test"]):
            raise ValueError("case-word artifact splits overlap")
        if set(split_indices["test"]) != expected_test:
            raise ValueError(
                "case-word artifact split_indices disagree with the stride rule "
                f"(document i is test iff i % {CASE_WORD_TEST_EVERY} == 0)"
            )

    def _validate_vocab(self, vocab: list) -> list[str]:
        if not vocab:
            raise ValueError("case-word artifact vocabulary is empty")
        if any(not isinstance(word, str) or not word or word != word.lower() for word in vocab):
            raise ValueError("case-word artifact vocabulary must be non-empty lowercase strings")
        if len(set(vocab)) != len(vocab):
            raise ValueError("case-word artifact vocabulary contains duplicates")
        declared = self.provenance.get("vocab_size")
        if declared is not None and declared != len(vocab):
            raise ValueError(
                f"case-word artifact declares vocab_size {declared!r} but ships {len(vocab)}"
            )
        return list(vocab)

    def _validate_documents_rows(self, entries: list, name: str) -> list[dict[str, Any]]:
        if not entries:
            raise ValueError(f"{name} split is empty")
        role_names = {role.name for role in CaseRole}
        seen_documents: set[int] = set()
        out: list[dict[str, Any]] = []
        for entry in entries:
            document_index = entry["document_index"]
            if document_index in seen_documents:
                raise ValueError(f"{name} split carries document {document_index} twice")
            seen_documents.add(document_index)
            if not 0 <= document_index < len(self.document_ids):
                raise ValueError(f"{name} split carries unknown document {document_index}")
            if self.document_ids[document_index] != entry["document_id"]:
                raise ValueError(
                    f"{name} split document index {document_index} disagrees with its id"
                )
            expected_split = "test" if document_index % CASE_WORD_TEST_EVERY == 0 else "train"
            if expected_split != name:
                raise ValueError(
                    f"{name} split carries document {document_index}, which the "
                    f"stride rule assigns to {expected_split}"
                )
            rows: list[CaseWordRow] = []
            for row in entry["rows"]:
                if len(row) != 4:
                    raise ValueError(f"{name} row is not a 4-field record: {row!r}")
                word_form, upos, case_label, label_source = row
                if not isinstance(word_form, str) or not word_form:
                    raise ValueError(f"{name} row carries an empty word form")
                if upos not in UD_UPOS_TAGS:
                    raise ValueError(
                        f"{name} row carries non-canonical UPOS {upos!r}; the "
                        f"observation space is the UD inventory"
                    )
                if (case_label is None) != (label_source is None):
                    raise ValueError(
                        f"{name} row pairs a case label with a missing route or "
                        f"the reverse: {row!r}"
                    )
                if case_label is not None:
                    if case_label not in role_names:
                        raise ValueError(f"{name} row carries unknown case role {case_label!r}")
                    if label_source not in CASE_ORACLE_PATHS:
                        raise ValueError(f"{name} row cites unknown oracle route {label_source!r}")
                rows.append(CaseWordRow(word_form, upos, case_label, label_source))
            out.append(
                {
                    "document_index": document_index,
                    "document_id": entry["document_id"],
                    "rows": rows,
                }
            )
        return out

    def _validate_sender_map(self, sender_map: dict) -> dict[str, dict[str, int]]:
        role_names = {role.name for role in CaseRole}
        vocab_set = set(self.vocab)
        cleaned: dict[str, dict[str, int]] = {}
        for role, counts in sender_map.items():
            if role not in role_names:
                raise ValueError(f"sender_map carries unknown role {role!r}")
            if not isinstance(counts, dict):
                raise ValueError(f"sender_map[{role!r}] is not a word-count mapping")
            for word, count in counts.items():
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                    raise ValueError(f"sender_map[{role!r}][{word!r}] must be a positive integer")
                if word not in vocab_set:
                    raise ValueError(
                        f"sender_map[{role!r}] counts out-of-vocabulary word {word!r}; "
                        f"the message's support is the declared intersection"
                    )
            cleaned[role] = dict(counts)
        if not cleaned:
            raise ValueError("sender_map is empty; the sender has no declared routes")
        return cleaned

    def _validate_route_stats(self) -> None:
        for route in CASE_ORACLE_PATHS:
            stats = self.route_stats.get(route)
            if not isinstance(stats, dict):
                raise ValueError(f"route_stats lacks the {route!r} route")
            for field in (
                "selected_tokens",
                "fired_tokens",
                "agreements_with_selected",
                "disagreements_with_selected",
            ):
                value = stats.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"route_stats[{route!r}][{field!r}] must be a count")
        train_tokens = sum(len(doc["rows"]) for doc in self.train_documents) + sum(
            len(doc["rows"]) for doc in self.test_documents
        )
        selected_total = sum(stats["selected_tokens"] for stats in self.route_stats.values())
        if selected_total > train_tokens:
            raise ValueError("route_stats selects more tokens than the artifact carries rows")

    def _validate_digests(self, digests: dict, payload: dict) -> None:
        def digest_of(fragment: object) -> str:
            canonical = json.dumps(fragment, sort_keys=True, ensure_ascii=True).encode("utf-8")
            return hashlib.sha256(canonical).hexdigest()

        expected = {
            "populations": digest_of(payload["populations"]),
            "sender_map": digest_of(payload["sender_map"]),
            "vocab": digest_of(payload["vocab"]),
        }
        for name, expected_digest in expected.items():
            recorded = digests.get(name)
            if recorded != expected_digest:
                raise ValueError(
                    f"case-word artifact {name} digest {recorded!r} does not match "
                    f"its shipped content ({expected_digest})"
                )

    def rows(self, split: str) -> list[CaseWordRow]:
        """All rows of one split, in document then token order."""
        if split not in ("train", "test"):
            raise ValueError(f"unknown split {split!r}")
        documents = self.train_documents if split == "train" else self.test_documents
        return [row for document in documents for row in document["rows"]]


def load_case_word_artifact(path: Path | str | None = None) -> CaseWordArtifact:
    """Load and validate the case-word bridge artifact.

    Raises:
        FileNotFoundError: If absent — this arm has no fallback corpus.
        ValueError: On any schema, provenance, label-route, or digest defect.
    """
    resolved = Path(path) if path is not None else DEFAULT_CASE_WORD_ARTIFACT
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} is absent; run scripts/generate_case_word_counts.py")
    return CaseWordArtifact(json.loads(resolved.read_text(encoding="utf-8")))
