#!/usr/bin/env python3
"""Generate the vendored Stage 5 clause artifact from UD English EWT.

Writes ``vendor/ud_ewt_clause_counts.json``: one entry per **typed** sentence —
its sentence type and its derived case-role sequence — split train/test by the
same document-index stride as the Stage 4 artifact.

What the clause level is, in this corpus
----------------------------------------
A clause is to case roles what a word is to letters: a latent discrete variable
(the sentence type) that conditions a sequence over the level below's alphabet.
The observation stream is the **derived case-role sequence** — Level 4's hidden
states, exactly one level down — and the latent type label comes from the
sentence-final punctuation token:

    ``.`` → declarative        ``?`` → interrogative

Two exclusions, stated rather than smoothed over. Sentences ending ``!`` are
excluded (n is small and English ``!`` marks emphasis at least as often as
imperative mood); sentences with no final ``.``/``?``/``!`` (headings, list
items, web fragments) carry no label at all. Both counts are recorded in
provenance; every Stage 5 number describes the typed sublanguage only.

The disjointness that makes the label usable
--------------------------------------------
Stage 4's lesson (decision 16) applies with full force: the model must never
observe the fields its labels are derived from. Here the label source is the
punctuation token's ``form``; the observation is the case-role sequence derived
from ``deprel``/adposition structure — and **PUNCT tokens carry no case role**,
so the label-bearing token cannot appear in the observation stream at all. The
generator asserts this rather than assuming it, and
``tests/test_stage5_gate.py`` re-checks it against the shipped artifact.

Document boundary and determinism
---------------------------------
The authenticated ``# newdoc id`` partition is fixed before type and length
filtering. The artifact retains the complete ordered identity alphabet and one
document index per retained sentence, so document-clustered inference never
reconstructs membership from a different source. Sorted keys, no RNG, no
timestamp: the artifact regenerates byte for byte.

Usage
-----
    python scripts/generate_clause_counts.py [--output PATH] [--test-every N]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "ud_ewt_clause_counts.json"

#: Held-out stride over the ordered document identities, matching the Stage 4
#: artifact so the two levels' populations nest rather than crosscut.
TEST_EVERY = 5

#: Sequences shorter than this carry no transition to learn from.
MIN_SEQUENCE_LEN = 2

#: Sentence-final punctuation → type label. ``!`` is deliberately absent.
TYPE_BY_FINAL_PUNCT = {".": "declarative", "?": "interrogative"}


def _sentence_type(sentence) -> str | None:
    """Type from the last ``.``/``?``/``!`` token form; None when unlabeled."""
    for form in reversed([token.form for token in sentence.tokens]):
        if form in {".", "?", "!"}:
            return TYPE_BY_FINAL_PUNCT.get(form)  # '!' maps to None on purpose
    return None


def build_artifact(test_every: int = TEST_EVERY) -> dict:
    """Assemble the vendored Stage 5 artifact from UD English EWT dev."""
    from ailm.corpus.ud import (
        UD_EWT_SOURCE_METADATA,
        document_identity_sha256,
        document_stride_partition,
        load_ud_ewt_dev,
    )

    sentences = load_ud_ewt_dev()
    document_ids, split_by_document = document_stride_partition(sentences, test_every=test_every)
    document_to_index = {document_id: index for index, document_id in enumerate(document_ids)}
    splits: dict[str, list] = {"train": [], "test": []}
    sequence_document_indices: dict[str, list[int]] = {"train": [], "test": []}
    n_exclamative = n_unpunctuated = 0

    for sentence in sentences:
        case_tokens = sentence.tokens_with_case()
        if len(case_tokens) < MIN_SEQUENCE_LEN:
            continue
        forms = [token.form for token in sentence.tokens]
        final = next((f for f in reversed(forms) if f in {".", "?", "!"}), None)
        if final == "!":
            n_exclamative += 1
            continue
        if final is None:
            n_unpunctuated += 1
            continue
        sentence_type = TYPE_BY_FINAL_PUNCT[final]

        sequence = [token.case_derived.name for token in case_tokens]
        # The disjointness invariant, asserted at build time: no label-bearing
        # punctuation token may contribute an observation.
        if any(token.upos == "PUNCT" for token in case_tokens):
            raise AssertionError(
                f"sentence {sentence.sent_id}: a PUNCT token carries a case "
                f"role, so the type label's source would leak into the "
                f"observation stream. The clause level cannot be built on this."
            )
        split = split_by_document[sentence.document_id]
        splits[split].append({"type": sentence_type, "cases": sequence})
        sequence_document_indices[split].append(document_to_index[sentence.document_id])

    return {
        "schema": "ailm-clause-counts-v2",
        "provenance": {
            "corpus": "UD English EWT dev (en_ewt-ud-dev.conllu)",
            "license": "CC BY-SA 4.0",
            **UD_EWT_SOURCE_METADATA,
            "observation": (
                "derived case-role sequence (Level 4's states, one level "
                "down); UPOS, deprel and all token forms withheld"
            ),
            "type_label_source": (
                "sentence-final punctuation token form: '.' declarative, "
                "'?' interrogative. PUNCT tokens carry no case role, so the "
                "label source cannot appear in the observation stream."
            ),
            "excluded_exclamative": n_exclamative,
            "excluded_unpunctuated": n_unpunctuated,
            "split": (
                "document-index stride over authenticated # newdoc id values, "
                "before type and length filtering; matching the case artifact"
            ),
            "test_every": test_every,
            "min_sequence_len": MIN_SEQUENCE_LEN,
            "sentences": len(sentences),
            "documents": len(document_ids),
            "train_documents": sum(split == "train" for split in split_by_document.values()),
            "test_documents": sum(split == "test" for split in split_by_document.values()),
            "document_identity_source": (
                "# newdoc id comments from the authenticated CoNLL-U source"
            ),
            "document_identity_sha256": document_identity_sha256(document_ids),
            "case_label_source": (
                "ailm.cases derived heuristic — a research claim, not ground "
                "truth (decisions 16 and 84); Stage 5 inherits its label "
                "uncertainty, while the English morphological overlap validates "
                "only its explicitly measured subset"
            ),
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
        from collections import Counter

        by_type = Counter(entry["type"] for entry in entries)
        n_positions = sum(len(entry["cases"]) for entry in entries)
        print(
            f"  {split_name:<6} {len(entries):>5} sentences "
            f"({dict(sorted(by_type.items()))}), {n_positions:>6,} positions"
        )
    prov = artifact["provenance"]
    print(
        f"  excluded: {prov['excluded_exclamative']} exclamative, "
        f"{prov['excluded_unpunctuated']} unpunctuated"
    )
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
