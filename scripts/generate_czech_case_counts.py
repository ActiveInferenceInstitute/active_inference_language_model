#!/usr/bin/env python3
"""Generate the vendored Czech case artifact from UD Czech-PDTC (decision 77).

Writes ``vendor/ud_czech_case_counts.json``. The raw ``.conllu`` files (~90 MB,
CC BY-NC-SA 4.0) are staged under ``vendor/staging_ud_czech/`` and never
committed; this derived artifact is the committed surface, matching the
``brown_counts.json`` / ``ud_ewt_case_counts.json`` precedent so the Czech arm's
tests need no network and no staging tree.

One population, on purpose
--------------------------
Unlike the English artifact there is no ``derived`` population here — every
label is UD morphological ``feats[Case]``, which is the entire point of the arm
(decision 77). Tokens are retained only if their UPOS is a nominal category
that *inflects* for case. ADP is excluded although Czech PDT annotates
``Case=`` on prepositions: there the feature records government — the case of
the complement, not inflection of the token — the structural analogue of the
English token-level adposition path decision 63 found reversed.

What the artifact retains, and what it cannot leak
--------------------------------------------------
Exactly two columns per token: UPOS (the observation) and ``feats[Case]``
mapped to the 8-role inventory (the label), plus the authenticated ``# newdoc
id`` alphabet and one document index per retained sequence. FORM, LEMMA, XPOS,
HEAD, DEPREL and MISC are absent from the artifact entirely, so no consumer can
spring the decision 15 trap with fields the file does not carry. Retaining the
outer document identity prevents sentence-level sequences from masquerading as
independent primary units.

Encoding
--------
Each token is one integer ``upos_index * 8 + case_index`` against the two
alphabets recorded beside the sequences. That keeps 180K tokens under a
megabyte where the English artifact's ``["PRON", "NOM"]`` pairs would not.

Determinism
-----------
Sorted alphabets, the corpus's own file order, no RNG, no timestamp: the
artifact regenerates byte for byte from the same staging tree.

Usage
-----
    .venv/bin/python scripts/generate_czech_case_counts.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.corpus.ud_czech import (  # noqa: E402
    UD_CZECH_PDTC_SHARDS,
    read_verified_ud_czech_shard,
    ud_czech_source_metadata,
)
from ailm.levels.case_czech import (  # noqa: E402
    CZECH_ADP_EXCLUDED_REASON,
    CZECH_ARTIFACT_SCHEMA,
    CZECH_CLUSTER_HIERARCHY,
    CZECH_CORPUS_NAME,
    CZECH_DOCUMENT_IDENTITY_SOURCE,
    CZECH_ENCODING,
    CZECH_LABEL_SOURCE,
    CZECH_LICENSE,
    CZECH_MIN_SEQUENCE_LEN,
    CZECH_NOMINAL_UPOS,
    CZECH_OBSERVATION,
    CZECH_SPLIT_SCOPE,
    CZECH_STUDY_DESIGN_PROVENANCE,
    CZECH_UD_CASE_VALUES,
)

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "ud_czech_case_counts.json"
STAGING_DIR = REPO_ROOT / "vendor" / "staging_ud_czech"

#: UPOS categories that inflect for case in Czech. ADP is excluded: its
#: ``Case=`` feature records government, not inflection (decision 77).
NOMINAL_UPOS = CZECH_NOMINAL_UPOS

#: The 8 UD ``Case=`` values of the CEREBRUM inventory, in ``CaseRole`` order.
#: Czech carries 7 (no ablative); the mapping still declares all 8 so the
#: case index is the ``CaseRole`` index, not a corpus-dependent renumbering.
UD_CASE_VALUES = CZECH_UD_CASE_VALUES

#: Sequences shorter than this carry no transition to learn from.
MIN_SEQUENCE_LEN = CZECH_MIN_SEQUENCE_LEN


def _sentence_document_ids(text: str, *, split: str) -> list[str]:
    """Recover one authenticated ``# newdoc id`` for every parsed sentence.

    The raw bytes are authenticated before this helper is called.  Document
    identity is stateful CoNLL-U metadata: a ``# newdoc id`` applies until the
    next declaration, while ``# sent_id`` marks each sentence the shared parser
    will emit.  Refusing missing or repeated declarations prevents a sentence
    index from being substituted for the outer sampling unit.
    """
    current: str | None = None
    declared: set[str] = set()
    document_ids: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("# newdoc id"):
            if "=" not in line:
                raise ValueError(
                    f"Czech {split} line {line_number} has malformed # newdoc id metadata"
                )
            current = line.split("=", 1)[1].strip()
            if not current:
                raise ValueError(f"Czech {split} line {line_number} has an empty document id")
            if current in declared:
                raise ValueError(
                    f"Czech {split} document id {current!r} is declared more than once"
                )
            declared.add(current)
        elif line.startswith("# sent_id"):
            if current is None:
                raise ValueError(
                    f"Czech {split} line {line_number} has a sentence before any # newdoc id"
                )
            document_ids.append(current)
    if not document_ids:
        raise ValueError(f"Czech {split} source contains no sentence document identities")
    return document_ids


def build_artifact(staging_dir: Path | str = STAGING_DIR) -> dict:
    """Assemble the vendored artifact from the staged Czech-PDTC shards."""
    from ailm.cases import CaseRole
    from ailm.corpus.ud import parse_conllu

    if [v.upper() for v in UD_CASE_VALUES] != [c.name for c in CaseRole]:
        raise SystemExit(
            "UD_CASE_VALUES is out of order with CaseRole; the encoded case index "
            "would silently denote a different role in every consumer."
        )
    case_to_idx = {value: i for i, value in enumerate(UD_CASE_VALUES)}

    staging_root = Path(staging_dir)
    # Read and authenticate every required shard before parsing either one.
    # This prevents a partial artifact from being assembled when only one
    # staged source happens to be valid.
    source_bytes = {
        split: read_verified_ud_czech_shard(staging_root / shard.filename, split)
        for split, shard in UD_CZECH_PDTC_SHARDS.items()
    }

    raw: dict[str, list[list[tuple[str, int]]]] = {}
    retained_document_ids: dict[str, list[str]] = {}
    provenance_counts: dict[str, dict[str, int | str]] = {}
    upos_seen: set[str] = set()
    for split, shard in UD_CZECH_PDTC_SHARDS.items():
        source_text = source_bytes[split].decode("utf-8")
        sentences = parse_conllu(source_text)
        sentence_document_ids = _sentence_document_ids(source_text, split=split)
        if len(sentences) != len(sentence_document_ids):
            raise ValueError(
                f"Czech {split} parser emitted {len(sentences)} sentences but "
                f"authenticated metadata names {len(sentence_document_ids)}"
            )
        sequences: list[list[tuple[str, int]]] = []
        sequence_documents: list[str] = []
        n_tokens = n_case_tokens = 0
        for sentence, document_id in zip(sentences, sentence_document_ids, strict=True):
            n_tokens += len(sentence.tokens)
            sequence: list[tuple[str, int]] = []
            for token in sentence.tokens:
                case_value = token.feats.get("Case")
                if case_value is None:
                    continue
                n_case_tokens += 1
                if token.upos in NOMINAL_UPOS and case_value in case_to_idx:
                    sequence.append((token.upos, case_to_idx[case_value]))
            if len(sequence) >= MIN_SEQUENCE_LEN:
                sequences.append(sequence)
                sequence_documents.append(document_id)
                upos_seen.update(u for u, _ in sequence)
        raw[split] = sequences
        retained_document_ids[split] = sequence_documents
        provenance_counts[split] = {
            "file": shard.filename,
            "sentences": len(sentences),
            "documents": len(set(sentence_document_ids)),
            "tokens": n_tokens,
            "case_marked_tokens": n_case_tokens,
            "retained_tokens": sum(len(s) for s in sequences),
            "retained_sequences": len(sequences),
            "retained_documents": len(set(sequence_documents)),
        }

    upos_tags = sorted(upos_seen)
    upos_to_idx = {u: i for i, u in enumerate(upos_tags)}
    n_cases = len(UD_CASE_VALUES)
    populations = {
        split: [[upos_to_idx[u] * n_cases + c for u, c in seq] for seq in sequences]
        for split, sequences in raw.items()
    }
    documents: dict[str, dict[str, list[str] | list[int]]] = {}
    for split, sequence_documents in retained_document_ids.items():
        ids = list(dict.fromkeys(sequence_documents))
        index_by_id = {document_id: index for index, document_id in enumerate(ids)}
        documents[split] = {
            "ids": ids,
            "sequence_document_indices": [
                index_by_id[document_id] for document_id in sequence_documents
            ],
        }

    return {
        "schema": CZECH_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": CZECH_CORPUS_NAME,
            "license": CZECH_LICENSE,
            "raw_source": ud_czech_source_metadata(),
            "split_scope": CZECH_SPLIT_SCOPE,
            "observation": CZECH_OBSERVATION,
            "label_source": CZECH_LABEL_SOURCE,
            "label_source_morphology_pct": 100.0,
            "nominal_upos": sorted(NOMINAL_UPOS),
            "adp_excluded_reason": CZECH_ADP_EXCLUDED_REASON,
            "min_sequence_len": MIN_SEQUENCE_LEN,
            "encoding": CZECH_ENCODING,
            "document_identity_source": CZECH_DOCUMENT_IDENTITY_SOURCE,
            "cluster_hierarchy": CZECH_CLUSTER_HIERARCHY,
            "splits": provenance_counts,
            "study_design": dict(CZECH_STUDY_DESIGN_PROVENANCE),
        },
        "upos_tags": upos_tags,
        "ud_case_values": list(UD_CASE_VALUES),
        "populations": populations,
        "documents": documents,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=STAGING_DIR,
        help="directory holding the two pinned Czech-PDTC source shards",
    )
    args = parser.parse_args(argv)

    artifact = build_artifact(args.staging_dir)
    atomic_write_json(
        args.output,
        artifact,
        ensure_ascii=True,
        sort_keys=False,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    try:
        shown = args.output.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"Wrote {shown}")
    for split, counts in artifact["provenance"]["splits"].items():
        print(
            f"  {split:<6} {counts['retained_sequences']:>6,} sequences, "
            f"{counts['retained_documents']:>5,} contributing documents, "
            f"{counts['retained_tokens']:>7,} retained tokens "
            f"(of {counts['case_marked_tokens']:,} case-marked in {counts['file']})"
        )
    print(f"  upos alphabet: {artifact['upos_tags']}")
    print(f"  size     {args.output.stat().st_size / 1024:.0f} KB")
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
