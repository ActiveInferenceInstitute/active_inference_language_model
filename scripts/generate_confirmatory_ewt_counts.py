#!/usr/bin/env python3
"""Vendor the decision-111 confirmatory evaluation population from UD English EWT test.

Writes three artifacts under ``data/confirmatory/``: the case, clause and
narrative counts for the whole of ``en_ewt-ud-test.conllu`` at the pinned
UD_English-EWT revision. This script measures nothing and scores nothing; it is
the vendor step decision 111 registers, and
``scripts/record_confirmatory_replication.py`` is the only consumer.

The one deliberate difference from the development generators
-------------------------------------------------------------
**No stride is applied.** Decision 79 registered the ``ud_ewt_test`` population
as the *entire* re-vendored corpus serving as evaluation, with the model
parameters read unchanged from the existing dev-train fits. Nothing here is
fitted, so there is no train arm to hold out and no split to make. The ordered
document identities are still extracted by the development generators' own
helper (``document_stride_partition``), which is what authenticates the
``# newdoc id`` blocks and rejects a reappearing one; only its split is
discarded.

Everything else is identical, and identity is imported rather than retyped
------------------------------------------------------------------------------
The minimum-sequence and minimum-typed-sentence filters, the sentence-type
punctuation map, and the observation/label rules are read from the development
generator modules themselves (``generate_case_counts``,
``generate_clause_counts``, ``generate_narrative_counts``) and from
``ailm.cases`` / ``ailm.corpus.ud``, so a change to a development filter moves
this generator too rather than silently diverging from it.
``tests/test_confirmatory_replication.py`` proves the identity the hard way: it
runs this module's population builders over the development corpus and requires
them to reproduce the committed development artifact's sequences exactly.

Why not ``vendor/``
-------------------
``tests/test_wheel_resources.py::test_runtime_manifest_owns_every_committed_vendor_json``
holds every committed ``vendor/*.json`` equal to
``ailm.resources.RUNTIME_VENDOR_ARTIFACTS``, which is the wheel's packaged
surface. These artifacts are one-run replication evidence that no library code
imports, so packaging them would be wrong. Their digests are bound to their
filenames and enforced by ``tests/test_confirmatory_replication.py`` and
documented in ``docs/provenance/sources.md``.

Determinism
-----------
No RNG, no timestamp, no wall-clock: sorted keys where order is not semantic and
first-seen corpus order where it is. Regenerating reproduces the bytes, which is
what makes the recorded digests worth anything.

Usage
-----
    .venv/bin/python scripts/generate_confirmatory_ewt_counts.py --population ud_ewt_test
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_case_counts as _dev_case  # noqa: E402
import generate_clause_counts as _dev_clause  # noqa: E402
import generate_narrative_counts as _dev_narrative  # noqa: E402
from _artifact_io import atomic_write_json  # noqa: E402

#: The only population this script will vendor. Decision 111 registers exactly
#: one, and an unrecognized value is a protocol violation rather than a typo to
#: be helpfully corrected.
REGISTERED_POPULATION = "ud_ewt_test"

DEFAULT_OUTPUT_DIRECTORY = REPO_ROOT / "data" / "confirmatory"

CASE_SCHEMA = "ailm-confirmatory-case-counts-v1"
CLAUSE_SCHEMA = "ailm-confirmatory-clause-counts-v1"
NARRATIVE_SCHEMA = "ailm-confirmatory-narrative-counts-v1"

ARTIFACT_FILENAMES = {
    "case": "ud_ewt_test_case_counts.json",
    "clause": "ud_ewt_test_clause_counts.json",
    "narrative": "ud_ewt_test_narrative_counts.json",
}

#: Read from the development generators so a filter cannot drift between them.
MIN_SEQUENCE_LEN = _dev_case.MIN_SEQUENCE_LEN
CLAUSE_MIN_SEQUENCE_LEN = _dev_clause.MIN_SEQUENCE_LEN
TYPE_BY_FINAL_PUNCT = _dev_clause.TYPE_BY_FINAL_PUNCT
MIN_TYPED_SENTENCES = _dev_narrative.MIN_TYPED_SENTENCES

#: The development case and clause generators share this minimum; the
#: confirmatory artifacts inherit whichever value each of them carries, and this
#: assertion fails loudly if they ever stop agreeing rather than letting one
#: population be filtered differently from the other.
if MIN_SEQUENCE_LEN != CLAUSE_MIN_SEQUENCE_LEN:
    raise RuntimeError(
        "the development case and clause generators no longer share a minimum "
        f"sequence length ({MIN_SEQUENCE_LEN} vs {CLAUSE_MIN_SEQUENCE_LEN}); the "
        "confirmatory vendor cannot claim identical filters until that is resolved"
    )


def ordered_document_identities(sentences: list) -> list[str]:
    """Ordered authenticated ``# newdoc id`` values, with no split applied.

    Delegates to the development generators' own
    :func:`ailm.corpus.ud.document_stride_partition` so the identity extraction,
    the NUL/empty rejection and the reappearing-block rejection are literally the
    same code. Its stride argument must be at least two, so the smallest legal
    value is passed and the returned split is discarded unread: this population
    is undivided by registration.
    """
    from ailm.corpus.ud import document_stride_partition

    document_ids, _discarded_split = document_stride_partition(sentences, test_every=2)
    return document_ids


def build_case_population(sentences: list) -> tuple[dict[str, list], dict[str, list[int]], int]:
    """The gold and derived case sequences for every sentence, in corpus order.

    The observation is the UPOS tag alone. ``deprel`` and ``feats[Case]`` are the
    fields the labels are derived from, so a model observing them would be
    reading its own answer key (decision 16); the tag alphabet is checked against
    UD's canonical set so a label-source lookalike cannot be serialized.
    """
    from ailm.cases import CaseRole
    from ailm.corpus.ud import UD_UPOS_TAGS, case_oracle_assignment

    gold_map = {"Nom": CaseRole.NOM, "Acc": CaseRole.ACC, "Gen": CaseRole.GEN}
    document_ids = ordered_document_identities(sentences)
    document_to_index = {document_id: index for index, document_id in enumerate(document_ids)}

    populations: dict[str, list] = {"gold": [], "derived": []}
    sequence_document_indices: dict[str, list[int]] = {"gold": [], "derived": []}
    n_tokens = 0
    for sentence in sentences:
        n_tokens += len(sentence.tokens)
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
                    sequence.append([token.upos, case.name])
            if len(sequence) >= MIN_SEQUENCE_LEN:
                populations[population].append(sequence)
                sequence_document_indices[population].append(
                    document_to_index[sentence.document_id]
                )
    return populations, sequence_document_indices, n_tokens


def build_clause_population(sentences: list) -> tuple[list[dict], list[int], int, int]:
    """Typed sentences and their derived case-role sequences, in corpus order.

    The type label comes from the sentence-final punctuation token's form and the
    observation is the case-role sequence, and PUNCT tokens carry no case role,
    so the label source cannot enter the observation stream. That disjointness is
    asserted here rather than assumed, exactly as the development generator
    asserts it.
    """
    document_ids = ordered_document_identities(sentences)
    document_to_index = {document_id: index for index, document_id in enumerate(document_ids)}

    entries: list[dict] = []
    sequence_document_indices: list[int] = []
    n_exclamative = n_unpunctuated = 0
    for sentence in sentences:
        case_tokens = sentence.tokens_with_case()
        if len(case_tokens) < CLAUSE_MIN_SEQUENCE_LEN:
            continue
        forms = [token.form for token in sentence.tokens]
        final = next((form for form in reversed(forms) if form in {".", "?", "!"}), None)
        if final == "!":
            n_exclamative += 1
            continue
        if final is None:
            n_unpunctuated += 1
            continue
        if any(token.upos == "PUNCT" for token in case_tokens):
            raise AssertionError(
                f"sentence {sentence.sent_id}: a PUNCT token carries a case role, so "
                f"the type label's source would leak into the observation stream. "
                f"The clause level cannot be built on this."
            )
        entries.append(
            {
                "type": TYPE_BY_FINAL_PUNCT[final],
                "cases": [token.case_derived.name for token in case_tokens],
            }
        )
        sequence_document_indices.append(document_to_index[sentence.document_id])
    return entries, sequence_document_indices, n_exclamative, n_unpunctuated


def build_narrative_population(sentences: list) -> tuple[list[dict], int, int]:
    """Per-document ordered sentence-type sequences, in first-seen document order.

    Typing follows the clause rules exactly, so the two levels describe the same
    sublanguage. Documents with fewer than ``MIN_TYPED_SENTENCES`` typed
    sentences have no arc to model and are excluded; the exclusion count is
    returned so it can be recorded rather than absorbed.
    """
    document_ids = ordered_document_identities(sentences)
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

    entries = [
        {"doc": document_id, "types": types}
        for document_id, types in documents.items()
        if len(types) >= MIN_TYPED_SENTENCES
    ]
    with_typed = sum(1 for types in documents.values() if types)
    return entries, with_typed, len(document_ids) - len(entries)


def _shared_provenance(document_ids: list[str]) -> dict[str, Any]:
    """The source identity, population declaration and document alphabet digest."""
    from ailm.corpus.ud import UD_EWT_TEST_SOURCE_METADATA, document_identity_sha256

    return {
        "corpus": "UD English EWT test (en_ewt-ud-test.conllu)",
        "license": "CC BY-SA 4.0",
        **UD_EWT_TEST_SOURCE_METADATA,
        "population": REGISTERED_POPULATION,
        "registration": "docs/decisions.md entry 111",
        "stride_applied": False,
        "split": (
            "none: the whole authenticated corpus is the evaluation population, "
            "because nothing is fitted on it (decision 79's registered rule, "
            "re-registered by decision 111)"
        ),
        "documents": len(document_ids),
        "document_identity_source": "# newdoc id comments from the authenticated CoNLL-U source",
        "document_identity_sha256": document_identity_sha256(document_ids),
    }


def build_artifacts() -> dict[str, dict]:
    """Assemble all three confirmatory artifacts from one authenticated read."""
    from ailm.corpus.ud import case_oracle_scope_audit, load_ud_ewt_test

    sentences = load_ud_ewt_test()
    document_ids = ordered_document_identities(sentences)
    shared = _shared_provenance(document_ids)

    populations, case_sequence_documents, n_tokens = build_case_population(sentences)
    scope_audit = case_oracle_scope_audit(sentences)
    case_artifact = {
        "schema": CASE_SCHEMA,
        "provenance": {
            **shared,
            "observation": "UPOS tag only; deprel and feats[Case] withheld",
            "min_sequence_len": MIN_SEQUENCE_LEN,
            "sentences": len(sentences),
            "tokens": n_tokens,
            "gold_label_source": "UD feats[Case]; Nom/Acc/Gen only",
            "derived_label_source": (
                "ailm.cases.UD_DEPREL_TO_CASE plus token-level ADPOSITION_TO_CASE "
                "- a research claim, not ground truth"
            ),
        },
        "heuristic_vs_gold": scope_audit["heuristic_vs_gold"],
        "oracle_scope_audit": scope_audit,
        "populations": populations,
        "documents": {
            "ids": document_ids,
            "sequence_document_indices": case_sequence_documents,
        },
    }

    entries, clause_sequence_documents, n_exclamative, n_unpunctuated = build_clause_population(
        sentences
    )
    clause_artifact = {
        "schema": CLAUSE_SCHEMA,
        "provenance": {
            **shared,
            "observation": (
                "derived case-role sequence (Level 4's states, one level down); "
                "UPOS, deprel and all token forms withheld"
            ),
            "type_label_source": (
                "sentence-final punctuation token form: '.' declarative, "
                "'?' interrogative. PUNCT tokens carry no case role, so the label "
                "source cannot appear in the observation stream."
            ),
            "min_sequence_len": CLAUSE_MIN_SEQUENCE_LEN,
            "excluded_exclamative": n_exclamative,
            "excluded_unpunctuated": n_unpunctuated,
            "sentences": len(sentences),
            "typed_sentences": len(entries),
        },
        "entries": entries,
        "documents": {
            "ids": document_ids,
            "sequence_document_indices": clause_sequence_documents,
        },
    }

    narrative_entries, with_typed, excluded_short = build_narrative_population(sentences)
    narrative_artifact = {
        "schema": NARRATIVE_SCHEMA,
        "provenance": {
            **shared,
            "observation": (
                "sentence-type sequence per document (Level 5's latent variable, "
                "one level down); typing rules identical to the clause artifact"
            ),
            "phase_definition": (
                "document-position tercile (opening/middle/closing) - a "
                "deterministic function of position, so every parameter is a "
                "count; no EM, no fitted segmentation"
            ),
            "min_typed_sentences": MIN_TYPED_SENTENCES,
            "documents_with_typed_sentences": with_typed,
            "documents_excluded_short": excluded_short,
            "eligible_documents": len(narrative_entries),
        },
        "entries": narrative_entries,
        "documents": {"ids": document_ids},
    }

    return {"case": case_artifact, "clause": clause_artifact, "narrative": narrative_artifact}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vendor the decision-111 confirmatory population.")
    parser.add_argument(
        "--population",
        required=True,
        help=f"the registered population to vendor (only {REGISTERED_POPULATION!r} exists)",
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    args = parser.parse_args(argv)

    if args.population != REGISTERED_POPULATION:
        raise SystemExit(
            f"refusing to vendor population {args.population!r}: decision 111 registers "
            f"exactly one, {REGISTERED_POPULATION!r}. Vendoring anything else would be a "
            f"new experiment, not this replication."
        )

    artifacts = build_artifacts()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for level, artifact in artifacts.items():
        output = args.output_directory / ARTIFACT_FILENAMES[level]
        atomic_write_json(output, artifact, ensure_ascii=True, sort_keys=False, indent=1)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        try:
            displayed = output.relative_to(REPO_ROOT)
        except ValueError:
            displayed = output
        print(f"Wrote {displayed}")
        print(f"  sha256   {digest}")

    case = artifacts["case"]["provenance"]
    print(
        f"  population census: {case['sentences']:,} sentences, {case['tokens']:,} tokens, "
        f"{case['documents']:,} documents; no stride applied"
    )
    for population, sequences in artifacts["case"]["populations"].items():
        print(
            f"  case {population:<8} {len(sequences):>5} sequences "
            f"({sum(len(s) for s in sequences):>6,} labeled tokens)"
        )
    clause = artifacts["clause"]["provenance"]
    print(
        f"  clause    {clause['typed_sentences']:,} typed sentences "
        f"(excluded {clause['excluded_exclamative']:,} exclamative, "
        f"{clause['excluded_unpunctuated']:,} unpunctuated)"
    )
    narrative = artifacts["narrative"]["provenance"]
    print(
        f"  narrative {narrative['eligible_documents']:,} eligible documents "
        f"(excluded {narrative['documents_excluded_short']:,} short)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
