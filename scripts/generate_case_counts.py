#!/usr/bin/env python3
"""Generate the vendored Stage 4 case artifact from UD English EWT.

Writes ``vendor/ud_ewt_case_counts.json``. The EWT ``.conllu`` file is
downloaded on demand and not committed (it is large and CC BY-SA); this derived
artifact is committed and checksummed, matching the ``brown_counts.json`` and
``cmudict_counts.json`` precedent so Stage 4 tests need no network.

The two populations, kept apart on purpose
------------------------------------------
``gold``
    Tokens carrying UD ``feats[Case]``. Real English morphology, and only
    Nom/Acc/Gen — five of the eight case roles have **no** morphological ground
    truth in this corpus at all.
``derived``
    Tokens labeled by the ``deprel`` / token-level adposition heuristic in
    ``ailm.cases``. This is a **research claim, not ground truth**. A score
    against it is *agreement*, never accuracy.

Both are emitted. Nothing merges them, because a merged number would be part
measurement and part heuristic with no way to tell which.

What the observation deliberately excludes
------------------------------------------
Each token is recorded as its **UPOS tag only**. Not ``deprel``, and not
``feats[Case]`` — those are the fields the oracle *derives its labels from*, so a
model observing them would be reading the answer. The resulting agreement would
be near-perfect and would measure nothing. See ``docs/decisions.md`` entry 16.

Document boundary and determinism
---------------------------------
The split is a stride over authenticated ``# newdoc id`` values in first-seen
corpus order, applied before either population's minimum-sequence filter.  A
document can therefore contribute to at most one arm.  The complete identity
alphabet, its digest, and each retained sequence's document index are shipped
with the artifact.  There is no RNG or timestamp, so it regenerates byte for
byte.

Usage
-----
    python scripts/generate_case_counts.py [--output PATH] [--test-every N]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "ud_ewt_case_counts.json"

#: Held-out stride. Document ``i`` is TEST iff ``i % TEST_EVERY == 0``. A stride
#: rather than a prefix: EWT is ordered by web domain, so a prefix would be a
#: single genre (decision 4). The rule is applied before sentence filtering.
TEST_EVERY = 5

#: Sequences shorter than this carry no transition to learn from.
MIN_SEQUENCE_LEN = 2


def build_artifact(test_every: int = TEST_EVERY) -> dict:
    """Assemble the vendored artifact from UD English EWT dev."""
    from ailm.cases import CaseRole
    from ailm.corpus.ud import (
        UD_EWT_SOURCE_METADATA,
        UD_UPOS_TAGS,
        case_oracle_assignment,
        case_oracle_scope_audit,
        document_identity_sha256,
        document_stride_partition,
        load_ud_ewt_dev,
    )

    gold_map = {"Nom": CaseRole.NOM, "Acc": CaseRole.ACC, "Gen": CaseRole.GEN}
    sentences = load_ud_ewt_dev()
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

    # The fidelity comparison and its population boundary are recorded together.
    # The aggregate only covers gold-morphology tokens that also receive a
    # syntactic label; it does not validate the token-level adposition route.
    # Head duplication/disagreement is retained as sufficient counts rather than
    # silently choosing which half of a prepositional phrase should own the role.
    scope_audit = case_oracle_scope_audit(sentences)

    return {
        "schema": "ailm-case-counts-v2",
        "provenance": {
            "corpus": "UD English EWT dev (en_ewt-ud-dev.conllu)",
            "license": "CC BY-SA 4.0",
            **UD_EWT_SOURCE_METADATA,
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
    atomic_write_json(args.output, artifact, ensure_ascii=True, sort_keys=False, indent=1)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    try:
        displayed_output = args.output.relative_to(REPO_ROOT)
    except ValueError:
        displayed_output = args.output
    print(f"Wrote {displayed_output}")
    for population, splits in artifact["populations"].items():
        n_train = sum(len(s) for s in splits["train"])
        n_test = sum(len(s) for s in splits["test"])
        print(
            f"  {population:<8} {len(splits['train']):>5} train sentences "
            f"({n_train:>6,} tokens), {len(splits['test']):>4} test "
            f"({n_test:>5,} tokens)"
        )
    both = artifact["heuristic_vs_gold"]
    print(
        f"  heuristic agrees with gold morphology on "
        f"{100.0 * both['agreements'] / both['tokens_with_both']:.1f}% of the "
        f"{both['tokens_with_both']:,} tokens carrying both"
    )
    scope = artifact["oracle_scope_audit"]
    adposition = scope["adposition_token"]
    print(
        "  token-level adposition route: "
        f"{adposition['tokens']:,} tokens; "
        f"{adposition['head_also_labeled_tokens']:,} also have a labeled head; "
        f"{adposition['head_role_disagreements']:,} of those roles disagree"
    )
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
