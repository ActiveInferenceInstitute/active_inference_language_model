#!/usr/bin/env python3
"""Generate the vendored Stage 6 narrative artifact from UD English EWT.

Writes ``vendor/ud_ewt_narrative_counts.json``: one entry per eligible document
— the ordered sequence of its typed sentences' types — after all authenticated
raw document IDs have been partitioned by a document-index stride.

What the narrative level is, in this corpus
-------------------------------------------
One level up from clauses: the document. The observation stream is the
sentence-type sequence (Level 5's latent variable, one level down), and the
hidden state is the **narrative phase** — operationalised deterministically as
the document-position tercile (opening / middle / closing). No EM, no fitted
segmentation: the phase is a function of position, so every count is a count.

Only documents with at least ``MIN_TYPED_SENTENCES`` typed sentences are
included (a two-sentence "document" has no arc to model); the excluded count
is recorded. Sentence typing follows the Stage 5 rules exactly — declarative
``.``, interrogative ``?``, with exclamative and unpunctuated sentences
dropped — so the two levels describe the same sublanguage.

Determinism
-----------
Document-index stride split over authenticated first-appearance order *before*
sentence typing or minimum-length filtering, no RNG, no timestamp:
byte-identical on regeneration.

Usage
-----
    python scripts/generate_narrative_counts.py [--output PATH] [--test-every N]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "ud_ewt_narrative_counts.json"
CONLLU = REPO_ROOT / "vendor" / "en_ewt-ud-dev.conllu"

TEST_EVERY = 5
MIN_TYPED_SENTENCES = 3
TYPE_BY_FINAL_PUNCT = {".": "declarative", "?": "interrogative"}


def build_artifact(test_every: int = TEST_EVERY) -> dict:
    from ailm.corpus.ud import (
        UD_EWT_SOURCE_METADATA,
        document_identity_sha256,
        document_stride_partition,
        ensure_ud_ewt_dev_path,
        load_ud_ewt_dev,
    )

    source_path = ensure_ud_ewt_dev_path(CONLLU)
    sentences = load_ud_ewt_dev(source_path, download=False)
    document_ids, split_by_document = document_stride_partition(sentences, test_every=test_every)
    document_to_index = {document_id: index for index, document_id in enumerate(document_ids)}

    # Start with every authenticated raw document, including documents that
    # contribute no retained type. Otherwise the exclusion count and split
    # identity silently omit the zero-observation documents.
    documents: dict[str, list[str]] = {document_id: [] for document_id in document_ids}
    for sentence in sentences:
        final = next(
            (
                form
                for form in reversed([token.form for token in sentence.tokens])
                if form in {".", "?", "!"}
            ),
            None,
        )
        sentence_type = TYPE_BY_FINAL_PUNCT.get(final) if final else None
        if sentence_type is None:
            continue
        documents[sentence.document_id].append(sentence_type)

    kept = [
        (doc_id, types) for doc_id, types in documents.items() if len(types) >= MIN_TYPED_SENTENCES
    ]
    excluded_short = len(document_ids) - len(kept)

    splits: dict[str, list] = {"train": [], "test": []}
    sequence_document_indices: dict[str, list[int]] = {"train": [], "test": []}
    for doc_id, types in kept:
        split = split_by_document[doc_id]
        splits[split].append({"doc": doc_id, "types": types})
        sequence_document_indices[split].append(document_to_index[doc_id])

    return {
        "schema": "ailm-narrative-counts-v2",
        "provenance": {
            "corpus": "UD English EWT dev (en_ewt-ud-dev.conllu)",
            "license": "CC BY-SA 4.0",
            **UD_EWT_SOURCE_METADATA,
            "observation": (
                "sentence-type sequence per document (Level 5's latent "
                "variable, one level down); typing rules identical to the "
                "Stage 5 artifact"
            ),
            "phase_definition": (
                "document-position tercile (opening/middle/closing) — a "
                "deterministic function of position, so every parameter is a "
                "count; no EM, no fitted segmentation"
            ),
            "split": (
                "document-index stride over authenticated # newdoc id values, "
                "before sentence typing and minimum-length filtering"
            ),
            "test_every": test_every,
            "min_typed_sentences": MIN_TYPED_SENTENCES,
            "documents_total": len(document_ids),
            "documents_with_typed_sentences": sum(bool(types) for types in documents.values()),
            "documents_excluded_short": excluded_short,
            "train_documents": sum(split == "train" for split in split_by_document.values()),
            "test_documents": sum(split == "test" for split in split_by_document.values()),
            "train_eligible_documents": len(splits["train"]),
            "test_eligible_documents": len(splits["test"]),
            "document_identity_source": (
                "# newdoc id comments from the authenticated CoNLL-U source"
            ),
            "document_identity_sha256": document_identity_sha256(document_ids),
        },
        "splits": splits,
        "documents": {
            "ids": document_ids,
            "split_indices": {
                split: [
                    index
                    for index, document_id in enumerate(document_ids)
                    if split_by_document[document_id] == split
                ]
                for split in ("train", "test")
            },
            "sequence_document_indices": sequence_document_indices,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-every", type=int, default=TEST_EVERY)
    args = parser.parse_args(argv)

    artifact = build_artifact(args.test_every)
    atomic_write_json(args.output, artifact, ensure_ascii=True, sort_keys=False, indent=1)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    try:
        displayed_output = args.output.relative_to(REPO_ROOT)
    except ValueError:
        displayed_output = args.output
    print(f"Wrote {displayed_output}")
    for split_name, entries in artifact["splits"].items():
        n_sentences = sum(len(entry["types"]) for entry in entries)
        print(f"  {split_name:<6} {len(entries):>4} documents, {n_sentences:>5} sentences")
    print(
        f"  excluded (fewer than {MIN_TYPED_SENTENCES} typed sentences): "
        f"{artifact['provenance']['documents_excluded_short']}"
    )
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
