#!/usr/bin/env python3
"""Generate the Stage 4 case artifact from the UD English EWT *test* split.

Mechanical twin of ``scripts/generate_case_counts.py``: identical generator and
oracle, identical stride and filters, only the corpus split (and therefore the
provenance string, loader, and default output location) differ. Per decision 79's
confirmatory protocol this run is measurement-only: nothing downstream scores,
and no output lands under ``vendor/``.

Usage:
    python scripts/generate_case_counts_test.py [--output PATH] [--test-every N]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

SCRATCH_DIR = REPO_ROOT / "scratchpad" / "review_2026-08-26" / "ewttest_vendor"
DEFAULT_OUTPUT = SCRATCH_DIR / "ud_ewt_test_case_counts.stage.json"

#: Held-out stride. Document ``i`` is TEST iff ``i % TEST_EVERY == 0``. The rule
#: is applied before sentence filtering.
TEST_EVERY = 5

#: Sequences shorter than this carry no transition to learn from.
MIN_SEQUENCE_LEN = 2


def build_artifact(test_every: int = TEST_EVERY) -> dict:
    """Assemble the stage artifact from UD English EWT test."""
    from ailm.cases import CaseRole
    from ailm.corpus.ud import (
        UD_EWT_TEST_SOURCE_METADATA,
        UD_UPOS_TAGS,
        case_oracle_assignment,
        case_oracle_scope_audit,
        document_identity_sha256,
        document_stride_partition,
        load_ud_ewt_test,
    )

    gold_map = {"Nom": CaseRole.NOM, "Acc": CaseRole.ACC, "Gen": CaseRole.GEN}
    sentences = load_ud_ewt_test()
    document_ids, split_by_document = document_stride_partition(sentences, test_every=test_every)
    document_to_index = {document_id: index for index, document_id in enumerate(document_ids)}

    populations: dict[str, dict[str, list]] = {
        "gold": {"train": [], "test": []},
        "derived": {"train": [], "test": []},
    }
    sequence_document_indices: dict[str, dict[str, list[int]]] = {
        "gold": {"train": [], "test": []},
        "derived": {"train": [], "test": []},
    }
    n_tokens = 0
    for sentence in sentences:
        n_tokens += len(sentence.tokens)
        split = split_by_document[sentence.document_id]
        for population in ("gold", "derived"):
            sequence = []
            for token in sentence.tokens:
                if token.upos not in UD_UPOS_TAGS:
                    raise ValueError(
                        f"UD token {token.id} has non-canonical UPOS observation "
                        f"{token.upos!r}; refusing to serialize a label-source lookalike"
                    )
                case = gold_map.get(token.case_morphological)
                if population == "derived" and case is None:
                    case = case_oracle_assignment(token)[0]
                if case is not None:
                    # UPOS only. The label's source fields are withheld.
                    sequence.append([token.upos, case.name])
            if len(sequence) >= MIN_SEQUENCE_LEN:
                populations[population][split].append(sequence)
                sequence_document_indices[population][split].append(
                    document_to_index[sentence.document_id]
                )

    scope_audit = case_oracle_scope_audit(sentences)

    return {
        "schema": "ailm-case-counts-v2",
        "provenance": {
            "corpus": "UD English EWT test (en_ewt-ud-test.conllu)",
            "license": "CC BY-SA 4.0",
            **UD_EWT_TEST_SOURCE_METADATA,
            "observation": "UPOS tag only; deprel and feats[Case] withheld",
            "split": (
                "document-index stride over authenticated # newdoc id values, "
                "before population filtering"
            ),
            "test_every": test_every,
            "min_sequence_len": MIN_SEQUENCE_LEN,
            "sentences": len(sentences),
            "tokens": n_tokens,
            "documents": len(document_ids),
            "train_documents": sum(split == "train" for split in split_by_document.values()),
            "test_documents": sum(split == "test" for split in split_by_document.values()),
            "document_identity_source": (
                "# newdoc id comments from the authenticated CoNLL-U source"
            ),
            "document_identity_sha256": document_identity_sha256(document_ids),
            "gold_label_source": "UD feats[Case]; Nom/Acc/Gen only",
            "derived_label_source": (
                "ailm.cases.UD_DEPREL_TO_CASE plus token-level "
                "ADPOSITION_TO_CASE — a research claim, not ground truth"
            ),
        },
        "heuristic_vs_gold": scope_audit["heuristic_vs_gold"],
        "oracle_scope_audit": scope_audit,
        "populations": populations,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, artifact, ensure_ascii=True, sort_keys=False, indent=1)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    try:
        displayed_output = args.output.relative_to(REPO_ROOT)
    except ValueError:
        displayed_output = args.output
    print(f"Wrote {displayed_output}")
    print(f"  sentences           {artifact['provenance']['sentences']:,}")
    print(f"  tokens              {artifact['provenance']['tokens']:,}")
    print(f"  documents           {artifact['provenance']['documents']:,}")
    print(
        f"  stride partition    train={artifact['provenance']['train_documents']:,} documents, "
        f"test={artifact['provenance']['test_documents']:,} documents (test_every={args.test_every})"
    )
    for population, splits in artifact["populations"].items():
        n_train = sum(len(s) for s in splits["train"])
        n_test = sum(len(s) for s in splits["test"])
        print(
            f"  {population:<8} {len(splits['train']):>5} train sequences "
            f"({n_train:>6,} tokens), {len(splits['test']):>4} test "
            f"({n_test:>5,} tokens)"
        )
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
