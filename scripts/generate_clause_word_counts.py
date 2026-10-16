#!/usr/bin/env python3
"""Generate the vendored punctuation-type-oracle -> word-prior artifact.

Writes ``vendor/brown_clause_word_counts.json``: one entry per **typed**
sentence whose words are restricted to the Stage 3 word vocabulary. It pairs
each word token with the sentence type (declarative / interrogative, from the
sentence-final punctuation) of the sentence it appears in, so a clause-derived
word statistic — ``P(word | sentence type)`` — can be estimated on a train
split and applied to held-out tokens. This is the bridge corpus the clause ->
word intervention (``docs/decisions.md`` entries 27 and 89) is measured on: the receiver is
the word level (Stage 3, Brown vocabulary), the message is a clause-typed word
distribution.

Why Brown and not the UD EWT clause artifact
---------------------------------------------
The clause level (Stage 5) is measured on UD EWT *case-role* sequences; the
word level (Stage 3) is measured on Brown *words*. There is no existing corpus
placing words next to a clause label. This artifact is the honest bridge: take
the same Brown sentences the word vocabulary was built from, derive the
sentence type from the sentence-final punctuation exactly as the clause
generator does (``.`` declarative, ``?`` interrogative, ``!`` and unlabeled
excluded, with counts recorded), and keep only the in-vocabulary words. The
words and the label live on the same lines, so the channel message and the
held-out evaluation describe the same population.

Whats measured
--------------
- ``P(word | sentence type)`` over the Stage 3 vocabulary, estimated from the
  training sentences and applied to held-out tokens.
- Sentence type by final punctuation has the same alphabet as the clause
  level's type factor, but it is a privileged reference label. This artifact
  does not contain or approximate an L5 posterior.

Determinism and population boundary
-----------------------------------
The bridge reuses Stage 3's authenticated category-stratified whole-document
partition. Message fitting sees only Stage 3 training documents; evaluation sees
only Stage 3 held-out documents. Keys are ordered by the pinned Brown file and
sentence order, with no RNG or timestamp, so the artifact regenerates byte for
byte without leaking a source document across arms.

Usage
-----
    python scripts/generate_clause_word_counts.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.corpus.brown import (  # noqa: E402
    BROWN_TRAIN_DOCUMENT_FRACTION,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    brown_source_metadata,
    build_brown_document_partition,
    load_brown_corpus,
)
from ailm.levels.clause_channel import (  # noqa: E402
    CLAUSE_TYPES,
    CLAUSE_WORD_ARTIFACT_SCHEMA,
    CLAUSE_WORD_CORPUS,
    CLAUSE_WORD_SPLIT,
    CLAUSE_WORD_TYPE_LABEL_SOURCE,
    CLAUSE_WORD_VOCAB_SOURCE,
    ClauseWordArtifact,
)

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "brown_clause_word_counts.json"

#: Sentence-final punctuation -> type label. ``!`` deliberately maps to none.
TYPE_BY_FINAL_PUNCT = {".": "declarative", "?": "interrogative"}


def _sentence_type(sentence: list[str]) -> str | None:
    """Type from the final ``.``/``?``/``!`` token; None when unlabeled."""
    for token in reversed(sentence):
        if token in (".", "?", "!"):
            return TYPE_BY_FINAL_PUNCT.get(token)
    return None


def build_artifact(source_zip: Path | str | None = None) -> dict:
    """Assemble the bridge artifact from the verified Brown archive."""
    brown = load_brown_corpus(source_zip)
    partition = build_brown_document_partition(BROWN_TRAIN_DOCUMENT_FRACTION, source_zip)
    if tuple(brown.fileids()) != partition.document_ids:
        raise RuntimeError("Brown reader identity order changed between authenticated reads")

    # The Stage 3 vocabulary and index, loaded from the shipped word artifact
    # so this bridge describes the exact receiving level's alphabet. A word
    # outside the vocabulary is dropped (the word model is closed-vocabulary).
    from ailm.levels.word import load_word_artifact

    word_path = REPO_ROOT / "vendor" / "brown_word_counts.json"
    word_artifact = load_word_artifact(word_path)
    word_digest = hashlib.sha256(word_path.read_bytes()).hexdigest()
    vocab = word_artifact.vocab
    word_to_index = word_artifact.word_to_index
    if (
        tuple(word_artifact.document_ids) != partition.document_ids
        or tuple(word_artifact.document_categories) != partition.categories
        or tuple(word_artifact.train_document_indices) != partition.train_indices
        or tuple(word_artifact.heldout_document_indices) != partition.heldout_indices
    ):
        raise RuntimeError("Stage 3 word artifact disagrees with the canonical Brown partition")

    splits: dict[str, list] = {"train": [], "test": []}
    n_exclamative = n_unpunctuated = 0
    vocab_drop = 0
    typed_sentence_counts = [0] * len(partition.document_ids)
    in_vocab_word_counts = [0] * len(partition.document_ids)
    train_set = set(partition.train_indices)

    for document_index, document_id in enumerate(partition.document_ids):
        split = "train" if document_index in train_set else "test"
        for sentence_index, sentence in enumerate(brown.sents(fileids=[document_id])):
            sentence_type = _sentence_type(sentence)
            if sentence_type is None:
                # Either ended in ``!`` (excluded) or carried no final punct.
                if sentence and sentence[-1] == "!":
                    n_exclamative += 1
                else:
                    n_unpunctuated += 1
                continue
            words: list[str] = []
            for token in sentence:
                low = token.lower()
                if low in word_to_index:
                    words.append(low)
                else:
                    vocab_drop += 1
            splits[split].append(
                {
                    "document_index": document_index,
                    "sentence_index": sentence_index,
                    "type": sentence_type,
                    "words": words,
                }
            )
            typed_sentence_counts[document_index] += 1
            in_vocab_word_counts[document_index] += len(words)

    typed_sentences = sum(typed_sentence_counts)
    train_typed_sentences = len(splits["train"])
    test_typed_sentences = len(splits["test"])

    artifact = {
        "schema": CLAUSE_WORD_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": CLAUSE_WORD_CORPUS,
            "raw_source": brown_source_metadata(),
            "type_label_source": CLAUSE_WORD_TYPE_LABEL_SOURCE,
            "excluded_exclamative": n_exclamative,
            "excluded_unpunctuated": n_unpunctuated,
            "vocab_source": CLAUSE_WORD_VOCAB_SOURCE,
            "vocab_source_schema": "ailm-word-counts-v2",
            "vocab_source_sha256": word_digest,
            "vocab_size": len(vocab),
            "words_dropped_out_of_vocabulary": vocab_drop,
            "split": CLAUSE_WORD_SPLIT,
            "target_train_document_fraction": BROWN_TRAIN_DOCUMENT_FRACTION,
            "train_document_fraction": len(partition.train_indices) / len(partition.document_ids),
            "documents": len(partition.document_ids),
            "train_documents": len(partition.train_indices),
            "heldout_documents": len(partition.heldout_indices),
            "document_overlap": 0,
            "categories": len(set(partition.categories)),
            "typed_sentences": typed_sentences,
            "train_typed_sentences": train_typed_sentences,
            "test_typed_sentences": test_typed_sentences,
            "sentence_types": list(CLAUSE_TYPES),
        },
        "documents": {
            "ids": list(partition.document_ids),
            "categories": list(partition.categories),
            "identity_sha256": brown_document_identity_sha256(partition.document_ids),
            "partition_input_sha256": brown_partition_input_sha256(
                partition.document_ids, partition.categories
            ),
            "split_indices": {
                "train": list(partition.train_indices),
                "heldout": list(partition.heldout_indices),
            },
            "typed_sentence_counts": typed_sentence_counts,
            "in_vocab_word_counts": in_vocab_word_counts,
        },
        "vocab": vocab,
        "splits": splits,
    }
    ClauseWordArtifact(
        artifact,
        expected_vocab=word_artifact.vocab,
        expected_vocab_sha256=word_digest,
        expected_document_ids=word_artifact.document_ids,
        expected_categories=word_artifact.document_categories,
        expected_train_indices=word_artifact.train_document_indices,
        expected_heldout_indices=word_artifact.heldout_document_indices,
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-zip", type=Path, default=None, help="exact pinned NLTK brown.zip")
    args = parser.parse_args(argv)

    artifact = build_artifact(args.source_zip)
    atomic_write_json(args.output, artifact, ensure_ascii=True, sort_keys=False, indent=1)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    prov = artifact["provenance"]
    try:
        shown = args.output.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"Wrote {shown}")
    for split_name, entries in artifact["splits"].items():
        by_type = Counter(entry["type"] for entry in entries)
        n_words = sum(len(entry["words"]) for entry in entries)
        print(
            f"  {split_name:<6} {len(entries):>5} sentences "
            f"({dict(sorted(by_type.items()))}), {n_words:,} in-vocab words"
        )
    print(
        f"  excluded: {prov['excluded_exclamative']} exclamative, "
        f"{prov['excluded_unpunctuated']} unpunctuated; "
        f"{prov['words_dropped_out_of_vocabulary']:,} words dropped OOV"
    )
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
