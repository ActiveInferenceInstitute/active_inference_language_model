#!/usr/bin/env python3
"""Vendor the complete held-out alphabetic Brown stream.

Why this artifact exists
------------------------
Every Stage 3 number describes a *sublanguage*. ``vendor/brown_word_counts.json``
carries ``eval_tokens``, which is the held-out stream already filtered to the
bounded lexicon, so the tokens the model assigns no probability to are not in the
file at all. That is correct for scoring — the level refuses to score an
out-of-vocabulary token rather than smoothing it — but it means the repository
has never held the denominator those numbers are a fraction of. The
out-of-vocabulary comparison was consequently *unrunnable*, not merely unshown.

This file is that denominator: every held-out alphabetic Brown token, in order,
unfiltered. What it makes possible is the first Stage 3 statement not
conditioned on the coverage sublanguage.

Why it is a separate artifact rather than a field
-------------------------------------------------
Keeping the unfiltered stream separate prevents the scoring artifact from
carrying every excluded token. Both artifacts still bind the same document
identity and partition, and each remains byte-identically regenerable.

Agreement, not duplication
--------------------------
The split is not reimplemented here. ``BROWN_TRAIN_DOCUMENT_FRACTION`` is owned
by ``ailm.corpus.brown`` and both writers call the same authenticated
``build_brown_document_partition`` source function. A copied split would agree
on the day it was written and drift silently afterwards, and the drift would
show up as a coverage figure nobody could reproduce.

Usage
-----
    .venv/bin/python scripts/generate_heldout_stream.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _artifact_io import atomic_write_json  # noqa: E402
from generate_word_counts import MAX_WORD_LEN  # noqa: E402

from ailm.corpus.brown import (  # noqa: E402
    BROWN_HELDOUT_ARTIFACT_SCHEMA,
    BROWN_TRAIN_DOCUMENT_FRACTION,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    brown_source_metadata,
    build_brown_document_partition,
)

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "brown_heldout_stream.json"
SCHEMA = BROWN_HELDOUT_ARTIFACT_SCHEMA


def build_artifact(source_zip: Path | str | None = None) -> dict:
    """The held-out half of the same split ``brown_word_counts.json`` uses."""
    partition = build_brown_document_partition(BROWN_TRAIN_DOCUMENT_FRACTION, source_zip)
    train = partition.train_tokens
    held = partition.heldout_tokens
    return {
        "schema": SCHEMA,
        "provenance": {
            "corpus": "NLTK Brown",
            "raw_source": brown_source_metadata(),
            "normalization": "lowercased; tokens of ASCII letters only",
            "target_train_document_fraction": BROWN_TRAIN_DOCUMENT_FRACTION,
            "train_document_fraction": len(partition.train_indices) / len(partition.document_ids),
            "train_token_fraction": len(train) / (len(train) + len(held)),
            "split": "category-stratified revision-bound SHA-256 document split",
            "total_alphabetic_tokens": len(train) + len(held),
            "train_tokens": len(train),
            "held_tokens": len(held),
            "held_types": len(set(held)),
            "documents": len(partition.document_ids),
            "train_documents": len(partition.train_indices),
            "heldout_documents": len(partition.heldout_indices),
            "categories": len(set(partition.categories)),
            "document_identity_sha256": brown_document_identity_sha256(partition.document_ids),
            "document_partition_input_sha256": brown_partition_input_sha256(
                partition.document_ids, partition.categories
            ),
            # Recorded because the word artifact drops long types from its
            # *vocabulary*; this stream drops nothing, so a reader comparing the
            # two needs to know the bound exists and was not applied here.
            "max_word_len_applied": False,
            "max_word_len_in_word_artifact": MAX_WORD_LEN,
        },
        "documents": {
            "ids": list(partition.document_ids),
            "categories": list(partition.categories),
            "alphabetic_token_counts": list(partition.alphabetic_token_counts),
            "split_indices": {
                "train": list(partition.train_indices),
                "heldout": list(partition.heldout_indices),
            },
            "held_token_document_indices": list(partition.heldout_token_document_indices),
        },
        # Order is the text's order and is load-bearing: the agreement test
        # filters this list and compares element-for-element against
        # ``eval_tokens``, which a set or a sorted list could not support.
        "held_tokens": list(held),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vendor the held-out Brown stream.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-zip", type=Path, default=None, help="exact pinned NLTK brown.zip")
    args = parser.parse_args(argv)

    artifact = build_artifact(args.source_zip)
    payload = atomic_write_json(args.output, artifact, ensure_ascii=True, sort_keys=False, indent=1)

    digest = hashlib.sha256(payload).hexdigest()
    prov = artifact["provenance"]
    try:
        shown = args.output.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"wrote {shown}")
    print(f"  held tokens  {prov['held_tokens']:,}")
    print(f"  held types   {prov['held_types']:,}")
    print(f"  sha256       {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
