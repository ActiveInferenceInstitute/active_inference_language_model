#!/usr/bin/env python3
"""Generate the vendored widened Stage 5 artifact from UD English EWT.

Writes ``data/widened/ud_ewt_clause_widened_counts.json``: one entry per **typed**
sentence — its sentence type and its widened observation sequence — split
train/test by the same authenticated document partition the shipped Stage 5
artifact uses.

What the widened arm is
-----------------------
The shipped Level 5 observes the derived case-role sequence: Level 4's hidden
states, exactly one level down. The *widened* stream adds the two parts of
speech that occupy a position in surface order and carry no case, ``VERB`` and
``AUX``, which is what lets the level see subject-auxiliary inversion — the
strongest interrogative cue in English, and one the case-only projection cannot
represent at all. Decision 112 registers the arm; this script builds its
artifact.

The case-only clause arm is untouched. Decision 114 made it the retained
ablation rather than Level 5, and it keeps its own artifact, its own recorder,
and a consistency guard (``scripts/audit_gates.py::audit_stage5_case_only``)
that trips if its finding changes.

What is reused rather than copied
---------------------------------
The partition and the exclusion rule are imported from
``scripts/generate_clause_counts.py``, not transcribed. The two artifacts must
inherit the *same* authenticated ``# newdoc id`` partition, fixed before any
filtering (decision 85); a second copy of that logic is a second thing to drift.
The alphabet, the extraction and the disjointness assertion are imported from
``ailm.levels.widened`` for the same reason.

The one rule that genuinely differs, stated rather than inherited silently
-------------------------------------------------------------------------
Sentence usability. The shipped artifact keeps a sentence contributing at least
two **case-bearing** tokens; this one keeps a sentence contributing at least two
**widened symbols**. A sentence of one case role and one verb is usable here and
not there. The two populations are therefore nested, not equal — which is why
decision 112 retains the shipped case-only arm as a separate baseline instead of
treating the two as the same measurement, and why the recorded census is part of
the result rather than a footnote.

The disjointness that makes the label usable, strengthened
----------------------------------------------------------
``PUNCT`` tokens emit nothing and remain the label source, and a ``VERB`` or
``AUX`` token contributes its UPOS only — never lemma, ``feats`` or surface
form. Both are asserted per sentence at build time by
``ailm.levels.widened.assert_widened_disjointness``, which recomputes rather
than trusts, and ``tests/test_stage5_widened.py`` plants a lemma leak and
requires the assertion to fire.

Determinism
-----------
Sorted keys, no RNG, no timestamp: the artifact regenerates byte for byte, which
is what makes the recorded checksum worth anything.

Usage
-----
    .venv/bin/python scripts/generate_clause_widened_counts.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _artifact_io import atomic_write_json  # noqa: E402

# The partition, the type-label rule and the stride are the SHIPPED ones,
# imported from the Stage 5 generator. Both artifacts must inherit one
# authenticated document partition; two copies of that rule is two things to
# drift, and the drift would be invisible until an arm leaked across a split.
from generate_clause_counts import (  # noqa: E402
    TEST_EVERY,
    TYPE_BY_FINAL_PUNCT,
    _sentence_type,
)

from ailm.levels.widened import (  # noqa: E402
    MIN_SEQUENCE_LEN,
    WIDENED_ARTIFACT_SCHEMA,
    WIDENED_SYMBOL_LABELS,
    assert_widened_disjointness,
    widened_symbols,
)

DEFAULT_OUTPUT = REPO_ROOT / "data" / "widened" / "ud_ewt_clause_widened_counts.json"


def build_artifact(test_every: int = TEST_EVERY) -> dict:
    """Assemble the vendored widened Stage 5 artifact from UD English EWT dev."""
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
    n_exclamative = n_unpunctuated = n_too_short = 0

    for sentence in sentences:
        forms = [token.form for token in sentence.tokens]
        final = next((f for f in reversed(forms) if f in {".", "?", "!"}), None)
        if final == "!":
            n_exclamative += 1
            continue
        if final is None:
            n_unpunctuated += 1
            continue
        sentence_type = _sentence_type(sentence)
        if sentence_type is None:  # pragma: no cover - '!' and None handled above
            raise AssertionError(
                f"sentence {sentence.sent_id}: final punctuation {final!r} is in the "
                f"label alphabet but yields no type; the two rules have drifted"
            )

        sequence = widened_symbols(sentence.tokens)
        # The usability rule of this arm, applied AFTER the partition and after
        # the type exclusions, so the census below decomposes the corpus exactly.
        if len(sequence) < MIN_SEQUENCE_LEN:
            n_too_short += 1
            continue

        assert_widened_disjointness(sentence.tokens, sequence, sent_id=sentence.sent_id)

        split = split_by_document[sentence.document_id]
        splits[split].append(
            {"type": sentence_type, "symbols": [WIDENED_SYMBOL_LABELS[s] for s in sequence]}
        )
        sequence_document_indices[split].append(document_to_index[sentence.document_id])

    return {
        "schema": WIDENED_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "UD English EWT dev (en_ewt-ud-dev.conllu)",
            "license": "CC BY-SA 4.0",
            **UD_EWT_SOURCE_METADATA,
            "alphabet": "widened",
            "alphabet_symbols": list(WIDENED_SYMBOL_LABELS),
            "observation": (
                "derived case-role sequence widened by one distinguished symbol "
                "per VERB and AUX token in surface order; lemma, feats and all "
                "token forms withheld"
            ),
            "type_label_source": (
                "sentence-final punctuation token form: '.' declarative, "
                "'?' interrogative. PUNCT tokens emit nothing and a VERB or AUX "
                "contributes its UPOS only, so the label source cannot appear in "
                "the observation stream; both are asserted per sentence at build "
                "time by ailm.levels.widened.assert_widened_disjointness"
            ),
            "usability_rule": (
                "a sentence is usable when its final punctuation names one of the "
                "two modeled types and it contributes at least "
                f"{MIN_SEQUENCE_LEN} widened symbols. The shipped case-only "
                "artifact instead requires that many case-bearing tokens, so the "
                "two populations are nested rather than equal (decision 112)"
            ),
            "excluded_exclamative": n_exclamative,
            "excluded_unpunctuated": n_unpunctuated,
            "excluded_too_short": n_too_short,
            "split": (
                "document-index stride over authenticated # newdoc id values, "
                "before type and length filtering; the same partition the shipped "
                "case artifact inherits (decision 85)"
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
                "truth (decisions 16 and 84); the widened arm inherits its label "
                "uncertainty on the eight case symbols, while the two added "
                "symbols are UD UPOS tags and carry the corpus's own annotation "
                "uncertainty instead"
            ),
            "registered_in": (
                "docs/decisions.md entry 112 — a development-session "
                "specification, not a confirmatory registration"
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
    print(f"  alphabet {len(WIDENED_SYMBOL_LABELS)} symbols: {', '.join(WIDENED_SYMBOL_LABELS)}")
    for split_name, entries in artifact["splits"].items():
        from collections import Counter

        by_type = Counter(entry["type"] for entry in entries)
        n_positions = sum(len(entry["symbols"]) for entry in entries)
        print(
            f"  {split_name:<6} {len(entries):>5} sentences "
            f"({dict(sorted(by_type.items()))}), {n_positions:>6,} positions"
        )
    prov = artifact["provenance"]
    print(
        f"  excluded: {prov['excluded_exclamative']} exclamative, "
        f"{prov['excluded_unpunctuated']} unpunctuated, "
        f"{prov['excluded_too_short']} under the usability minimum"
    )
    print(f"  types    {sorted(TYPE_BY_FINAL_PUNCT.items())}")
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
