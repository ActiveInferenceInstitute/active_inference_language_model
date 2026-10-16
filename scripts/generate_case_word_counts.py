"""Generate the case-word (L4→L3) bridge artifact from UD EWT dev.

Protocol P3 (`output/reports/omp-dispatch-20260908/evidence/
extensions-E1R-frozen/docs/extension_protocols.md` §4): one row per token,
``(word_form, upos, case_label_or_null, label_source)``, split by the same
document-index stride as the shipped case artifact and cross-validated against
its recorded ``split_indices``. The sender map ``K(word | role)`` is fitted
from train rows only, pooled over the declared oracle routes, counted over the
lowercase exact-string intersection with the Stage 3 receiver vocabulary
(``vendor/brown_word_counts.json`` ``word_freq`` — 2,000 types), with per-route
coverage and per-route agreement fields so the pooling is inspectable.

What the observation deliberately excludes: the oracle case label is a
sender-side fitting input recorded per token with its route; the model side
observes UPOS only. The artifact's ``observation`` provenance string states
this and the loader refuses an artifact that drops it.

The EWT *test* file's bytes are spent (decision 111) and are not read here.

Usage:
    .venv/bin/python scripts/generate_case_word_counts.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from _artifact_io import atomic_write_json  # noqa: E402

from ailm.corpus.case_word import (  # noqa: E402
    CASE_WORD_ARTIFACT_SCHEMA,
    CASE_WORD_OBSERVATION,
    CASE_WORD_TEST_EVERY,
    route_roles,
)
from ailm.corpus.ud import (  # noqa: E402
    UD_EWT_SOURCE_METADATA,
    UD_UPOS_TAGS,
    case_oracle_assignment,
    document_identity_sha256,
    document_stride_partition,
    load_ud_ewt_dev,
)
from ailm.levels.word import load_word_artifact  # noqa: E402

#: Output home: ``data/``, not ``vendor/`` — the wheel's runtime manifest owns
#: every committed ``vendor/*.json``, and this development-session artifact is
#: consumed by nothing packaged (the widened-arm precedent, decision 112).
DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "case_word" / "ud_ewt_case_word_counts.json"

BROWN_WORD_COUNTS = _PROJECT_ROOT / "vendor" / "brown_word_counts.json"


def build_artifact(test_every: int = CASE_WORD_TEST_EVERY) -> dict:
    """Assemble the case-word artifact from authenticated UD EWT dev."""
    sentences = load_ud_ewt_dev()
    document_ids, split_by_document = document_stride_partition(sentences, test_every=test_every)
    document_to_index = {document_id: index for index, document_id in enumerate(document_ids)}
    word_artifact = load_word_artifact()
    vocabulary: list[str] = list(word_artifact.vocab)
    vocab_set = set(vocabulary)

    populations: dict[str, list] = {"train": [], "test": []}
    document_rows: dict[int, list] = {}
    route_stats = {
        route: {
            "selected_tokens": 0,
            "fired_tokens": 0,
            "agreements_with_selected": 0,
            "disagreements_with_selected": 0,
        }
        for route in ("morphology", "deprel", "adposition_token")
    }
    sender_counts: dict[str, dict[str, int]] = {}
    dropped_oov_train_words = 0
    labeled_tokens = 0
    total_tokens = 0

    for sentence in sentences:
        split = split_by_document[sentence.document_id]
        document_index = document_to_index[sentence.document_id]
        rows: list[list] = []
        for token in sentence.tokens:
            if token.upos not in UD_UPOS_TAGS:
                raise ValueError(
                    f"UD token {token.id} has non-canonical UPOS observation "
                    f"{token.upos!r}; refusing to serialize a label-source lookalike"
                )
            total_tokens += 1
            role, route = case_oracle_assignment(token)
            if role is not None:
                labeled_tokens += 1
            rows.append(
                [
                    token.form,
                    token.upos,
                    role.name if role is not None else None,
                    route,
                ]
            )
            if split == "train":
                # Per-route coverage and agreement, over the fitting population.
                fired = route_roles(token)
                for route_name, fired_role in fired.items():
                    stats = route_stats[route_name]
                    if fired_role is not None:
                        stats["fired_tokens"] += 1
                        if role is not None:
                            if fired_role.name == role.name:
                                stats["agreements_with_selected"] += 1
                            else:
                                stats["disagreements_with_selected"] += 1
                    if route is not None and route_name == route:
                        stats["selected_tokens"] += 1
                # K pools the declared routes on train rows, counted over the
                # lowercase exact-string intersection with the receiver
                # vocabulary only: the message's support is the intersection.
                if role is not None:
                    word = token.form.lower()
                    if word in vocab_set:
                        sender_counts.setdefault(role.name, {})
                        sender_counts[role.name][word] = sender_counts[role.name].get(word, 0) + 1
                    else:
                        dropped_oov_train_words += 1
        document_rows.setdefault(document_index, []).extend(rows)

    for document_index, rows in document_rows.items():
        split = split_by_document[document_ids[document_index]]
        populations[split].append(
            {
                "document_index": document_index,
                "document_id": document_ids[document_index],
                "rows": rows,
            }
        )

    split_indices = {
        name: [
            index
            for index, document_id in enumerate(document_ids)
            if split_by_document[document_id] == name
        ]
        for name in ("train", "test")
    }

    # Cross-validation against the shipped case artifact's recorded split: the
    # two artifacts must describe the same development documents.
    case_payload = json.loads(_CASE_COUNTS_BYTES())
    case_splits = case_payload["documents"]["split_indices"]
    if (
        case_splits["train"] != split_indices["train"]
        or case_splits["test"] != split_indices["test"]
    ):
        raise SystemExit(
            "REFUSING TO WRITE: the case-word split disagrees with "
            "vendor/ud_ewt_case_counts.json; the two artifacts must describe "
            "the same development documents under the same stride rule."
        )

    digest_fragment = {
        "populations": populations,
        "sender_map": sender_counts,
        "vocab": vocabulary,
    }
    digests = {
        name: hashlib.sha256(
            json.dumps(fragment, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        for name, fragment in digest_fragment.items()
    }

    return {
        "schema": CASE_WORD_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "UD English EWT dev (en_ewt-ud-dev.conllu)",
            "license": "CC BY-SA 4.0",
            **UD_EWT_SOURCE_METADATA,
            "observation": CASE_WORD_OBSERVATION,
            "split": (
                "document-index stride over authenticated # newdoc id values "
                "(dev), cross-validated against vendor/ud_ewt_case_counts.json"
            ),
            "test_every": test_every,
            "sentences": len(sentences),
            "tokens": total_tokens,
            "labeled_tokens": labeled_tokens,
            "documents": len(document_ids),
            "train_documents": len(split_indices["train"]),
            "test_documents": len(split_indices["test"]),
            "document_identity_source": (
                "# newdoc id comments from the authenticated CoNLL-U source"
            ),
            "document_identity_sha256": document_identity_sha256(document_ids),
            "label_source": (
                "ailm.corpus.ud.case_oracle_assignment unchanged: feats[Case] "
                "morphology first, then deprel, then the token-level "
                "adposition route; derived labels are agreement, not accuracy"
            ),
            "sender_fit_population": "train rows only",
            "normalization": (
                "EWT word forms lowercased, matched by exact string against "
                "the Stage 3 vocabulary's lowercase type keys; no other "
                "folding or stemming; message support is the intersection"
            ),
            "vocab_source": "vendor/brown_word_counts.json word_freq (receiver index order)",
            "vocab_source_sha256": hashlib.sha256(BROWN_WORD_COUNTS.read_bytes()).hexdigest(),
            "vocab_size": len(vocabulary),
            "dropped_oov_train_words": dropped_oov_train_words,
        },
        "documents": {"ids": document_ids, "split_indices": split_indices},
        "vocab": vocabulary,
        "populations": populations,
        "sender_map": sender_counts,
        "route_stats": route_stats,
        "digests": digests,
    }


def _CASE_COUNTS_BYTES() -> bytes:
    return (_PROJECT_ROOT / "vendor" / "ud_ewt_case_counts.json").read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-every", type=int, default=CASE_WORD_TEST_EVERY)
    args = parser.parse_args(argv)

    artifact = build_artifact(args.test_every)
    atomic_write_json(args.output, artifact, ensure_ascii=True, sort_keys=False, indent=1)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    try:
        displayed_output = args.output.relative_to(_PROJECT_ROOT)
    except ValueError:
        displayed_output = args.output
    print(f"Wrote {displayed_output}")
    for split in ("train", "test"):
        rows = sum(len(doc["rows"]) for doc in artifact["populations"][split])
        print(f"  {split:<5} {len(artifact['populations'][split]):>4} documents, {rows:>6,} rows")
    labeled = sum(
        1
        for split in ("train", "test")
        for doc in artifact["populations"][split]
        for row in doc["rows"]
        if row[2] is not None
    )
    print(f"  labeled tokens: {labeled:,}")
    for route, stats in artifact["route_stats"].items():
        print(
            f"  route {route:<16} selected {stats['selected_tokens']:>6,} "
            f"fired {stats['fired_tokens']:>6,} agree {stats['agreements_with_selected']:>6,} "
            f"disagree {stats['disagreements_with_selected']:>6,}"
        )
    print(f"  sender roles with in-vocabulary support: {sorted(artifact['sender_map'])}")
    print(f"  dropped OOV train words: {artifact['provenance']['dropped_oov_train_words']:,}")
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
