#!/usr/bin/env python3
"""Score the four ud_ewt_test family members once, under decision 111.

Decision 79 froze a sixteen-member ``SIGNIFICANCE_FAMILY`` and registered a
confirmatory replication of it. Decision 91 retired the eight
``external_lexicon`` members by exhaustion and left the four ``ud_ewt_test``
members rescoring-eligible under a future successor registration. Decision 111
is that successor: the principal ruled that the case-count twin generator's read
of ``en_ewt-ud-test.conllu`` is the sanctioned identical-generator re-vendor,
not a scoring read, so the population is unburned, and this recorder scores the
four members exactly once.

Why this script is not named ``record_*_claims.py``
--------------------------------------------------
``scripts/orchestrate_claim_recorders.py`` discovers recorders by globbing
``scripts/record_*_claims.py`` and runs every one of them on every orchestrated
pass. A recorder that must refuse a second run would break that orchestration
permanently the moment it succeeded once. A one-run-ever recorder belongs
outside the re-runnable set, and the name is what keeps it there.

The discipline, in the order this script applies it
---------------------------------------------------
1. **Refuse on the sentinel.** ``confirmatory_run_completed_ud_ewt_test`` in the
   ledger means the population has already been used. A second pass over a
   held-back set un-holds it, so the run refuses rather than repeating.
2. **Prove everything deterministic before spending anything.** Under the
   ledger lock and before the sentinel write, the confirmatory artifacts are
   loaded and their populations checked, the document-identity disjointness
   is enforced against the WHOLE development population (falsifier 4, train
   and held-out arms alike), and every model array is built from the
   development train split and asserted equal, element for element
   (falsifier 3). A refusal here burns nothing; recovery never requires
   deleting a sentinel decision 111 forbids deleting.
3. **Write the sentinel and its filesystem marker, once, then score.** The
   population is spent the moment it is scored, whether or not the scoring
   finishes, so the sentinel claim -- and, beside the ledger the run writes,
   a ``confirmatory_run_completed_*`` marker file -- land immediately before
   the first statistic-computing step. The marker is the second channel the
   precheck's filesystem scan reads.
4. **Guard where the run may write.** Production mode (no environment
   overrides) reads the production corpus and ledger. The sanctioned
   isolated shakedown is fixture mode: ``AILM_CONFIRMATORY_ROOT`` naming an
   isolated fixture root with its own genuinely synthetic
   ``data/confirmatory``, ``AILM_CLAIMS_PATH`` naming a ledger outside the
   production tree, and ``AILM_CONFIRMATORY_ALTERNATE_LEDGER=1``. Any
   partial override, any override that resolves to or inside the production
   tree, and any fixture corpus byte-identical to the spent production
   corpus refuse before anything is read -- an alternate ledger over the
   production corpus is a rescore of held-back data, and no opt-in
   authorizes that.
5. **Run both controls on this population.** A planted-effect positive control
   must fire and a label-shuffled negative control must not. A non-firing
   positive control means a null here would be instrument-ambiguous; a firing
   negative control means the comparison would accept noise.
6. **Score, and record whatever comes out.** A member that fails to replicate is
   recorded as having failed. A member that cannot be computed at all (the gold
   comparison again yielding b = c = 0) is recorded as uncomputable. Neither is
   re-designed around.

Usage
-----
    .venv/bin/python scripts/record_confirmatory_replication.py
    .venv/bin/python scripts/record_confirmatory_replication.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: The production project root, resolved from this file's location. The
#: fixture overrides below are compared against THIS tree by resolved
#: identity: an override that resolves here — or inside it, through any
#: symlink chain — is not a fixture, and neither is a copy of the spent
#: production corpus. The executing code always comes from this real src
#: tree; a fixture root supplies data/, never code.
PRODUCTION_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PRODUCTION_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.statistics import (  # noqa: E402
    BONFERRONI_ALPHA,
    CONFIRMATORY_REPLICATION,
    FAMILYWISE_ALPHA,
    SIGNIFICANCE_COUNT_LABELS,
    SIGNIFICANCE_FAMILY,
    exact_two_sided_binomial_p,
    holm_step_down,
)

RUN_ID = "confirmatory_replication_ud_ewt_test_v1"
SOURCE = "scripts/record_confirmatory_replication.py"
POPULATION = "ud_ewt_test"
SENTINEL = f"confirmatory_run_completed_{POPULATION}"
LEDGER_ENV = "AILM_CLAIMS_PATH"
CORPUS_ROOT_ENV = "AILM_CONFIRMATORY_ROOT"
OPT_IN_ENV = "AILM_CONFIRMATORY_ALTERNATE_LEDGER"
MARKER_NAME = SENTINEL


def _corpus_root() -> Path:
    """Where this run looks for ``data/``: the fixture root, or production.

    Only the confirmatory corpus redirects. The development artifacts stay the
    committed vendor fits under every mode, because falsifier 3 requires
    exactly those parameters; a fixture changes what is SCORED, never what
    the models are.
    """
    override = os.environ.get(CORPUS_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    return PRODUCTION_PROJECT_ROOT


def _confirmatory_directory() -> Path:
    """The confirmatory corpus directory this call reads, resolved now.

    Read at call time, not at import: the clustered recorder imports this
    module for its merger helpers, and an environment captured at import
    would let a run that sets the fixture overrides afterwards silently
    score the production corpus. The executing code always comes from the
    real src tree; only this data path moves.
    """
    return _corpus_root() / "data" / "confirmatory"


#: Fixed permutation seed for the label-shuffled negative control. A control
#: whose seed moved between runs would be a different control each time, and its
#: verdict could then be shopped for.
NEGATIVE_CONTROL_SEED = 20260905

#: The four members this run is registered to score, in ledger order.
REGISTERED_MEMBERS: tuple[str, ...] = tuple(
    label for label, disposition in CONFIRMATORY_REPLICATION if disposition == POPULATION
)


class ProtocolRefusal(SystemExit):
    """The run stops without recording an outcome it is not entitled to record."""


# ── Loading and merging ──────────────────────────────────────────────────────


def _load_confirmatory(level: str) -> dict:
    """One vendored confirmatory artifact, with its registered population checked."""
    path = _confirmatory_directory() / f"ud_ewt_test_{level}_counts.json"
    if not path.is_file():
        raise ProtocolRefusal(
            f"{path} is absent. Run scripts/generate_confirmatory_ewt_counts.py "
            f"--population {POPULATION} first; this recorder never vendors."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance", {})
    if provenance.get("population") != POPULATION:
        raise ProtocolRefusal(
            f"{path} declares population {provenance.get('population')!r}, not {POPULATION!r}"
        )
    if provenance.get("stride_applied") is not False:
        raise ProtocolRefusal(
            f"{path} was built with a stride; decision 79 registers the whole "
            f"corpus as the evaluation population"
        )
    return payload


# ── Execution mode: the production boundary for alternate ledgers ───────────


PRODUCTION = "production"
FIXTURE = "fixture"


def _execution_mode() -> tuple[str, Path, str | None]:
    """Resolve the execution mode and the ledger this run would write.

    Production mode — no overrides — reads the production confirmatory corpus
    and writes the production ledger under the one-run sentinel.

    Fixture mode — ``AILM_CONFIRMATORY_ROOT`` naming an isolated fixture root that
    carries its own ``data/confirmatory``, ``AILM_CLAIMS_PATH`` naming a
    ledger outside the production tree, and the explicit opt-in
    ``AILM_CONFIRMATORY_ALTERNATE_LEDGER=1`` — is the inspectable shakedown
    seam decision 111's execution note sanctions. Every write lands inside
    the fixture root.

    Every partial or identity-violating combination refuses before anything
    is read, because each one re-scores or corrupts real state invisibly to
    the sentinel check: an alternate ledger over the production corpus
    re-scores the spent held-back population into a side ledger no precheck
    can see, and no opt-in authorizes that; a fixture corpus against the
    production ledger would write spent markers derived from fixture data
    into the real tree; an override that resolves to or inside the production
    tree is the production tree under another name, and a name change is not
    isolation.
    """
    root_override = os.environ.get(CORPUS_ROOT_ENV, "").strip()
    ledger_override = os.environ.get(LEDGER_ENV, "").strip()
    opt_in = os.environ.get(OPT_IN_ENV, "").strip() == "1"
    production_ledger = PRODUCTION_PROJECT_ROOT / "data" / "claims.json"
    if root_override and not Path(root_override).is_absolute():
        return (
            PRODUCTION,
            production_ledger,
            (
                f"{CORPUS_ROOT_ENV} must be absolute so fixture isolation cannot depend "
                f"on a subprocess working directory, got {root_override!r}"
            ),
        )
    if ledger_override and not Path(ledger_override).is_absolute():
        return (
            PRODUCTION,
            production_ledger,
            (
                f"{LEDGER_ENV} must be absolute so recorder isolation cannot depend on "
                f"a subprocess working directory, got {ledger_override!r}"
            ),
        )
    if not root_override and not ledger_override:
        return PRODUCTION, production_ledger, None

    # The partial-override refusals must come BEFORE any identity comparison:
    # with one override absent, ``Path("")`` would resolve to the working
    # directory and misreport a missing fixture root as a production alias.
    if not root_override:
        return (
            PRODUCTION,
            Path(ledger_override),
            (
                f"REFUSED: {LEDGER_ENV} is set ({ledger_override!r}) but {CORPUS_ROOT_ENV} "
                "is not, so this run would re-score the production confirmatory corpus "
                "into a side ledger no precheck can see. The spent-population rule has "
                f"no opt-in for that. Shakedowns isolate the corpus too: set {CORPUS_ROOT_ENV} "
                "to a fixture root carrying its own genuinely synthetic data/confirmatory."
            ),
        )
    if not ledger_override:
        return (
            PRODUCTION,
            production_ledger,
            (
                f"REFUSED: {CORPUS_ROOT_ENV} is set ({root_override!r}) but {LEDGER_ENV} is "
                "not, so this fixture-corpus run would write its sentinel and spent marker "
                "into the production ledger. Isolate the ledger as well."
            ),
        )
    fixture_root = Path(root_override)
    claims_path = Path(ledger_override)
    identity_problem = _fixture_identity_problem(fixture_root, claims_path)
    if identity_problem is not None:
        return PRODUCTION, production_ledger, identity_problem
    if not opt_in:
        return (
            FIXTURE,
            claims_path,
            (
                f"REFUSED: fixture mode ({CORPUS_ROOT_ENV}={root_override!r}, {LEDGER_ENV}="
                f"{ledger_override!r}) requires the explicit opt-in {OPT_IN_ENV}=1, so an "
                "isolated shakedown can never start by accident."
            ),
        )
    return FIXTURE, claims_path, None


def _fixture_identity_problem(fixture_root: Path, claims_path: Path) -> str | None:
    """Refuse overrides that resolve to the production tree, before anything is read.

    Paths are compared RESOLVED, so a symlink alias of the production root or
    ledger cannot pass as a fixture, and a fixture root or ledger placed
    inside the production tree is refused for the same reason: its writes
    would land in the real tree the sentinel is supposed to protect.
    """
    production_root = PRODUCTION_PROJECT_ROOT.resolve()
    resolved_root = fixture_root.resolve()
    resolved_ledger = claims_path.resolve()
    production_ledger = (production_root / "data" / "claims.json").resolve()
    if resolved_root == production_root or production_root in resolved_root.parents:
        return (
            f"REFUSED: {CORPUS_ROOT_ENV}={str(fixture_root)!r} resolves to or inside "
            "the production tree; a fixture root must live outside it, because its "
            "writes would land where the sentinel is checked."
        )
    if resolved_ledger == production_ledger or production_root in resolved_ledger.parents:
        return (
            f"REFUSED: {LEDGER_ENV}={str(claims_path)!r} resolves to or inside the "
            "production tree; the fixture ledger must live outside it."
        )
    return None


def _corpus_identity_problem() -> str | None:
    """Whether the corpus about to be scored is a renamed copy of production.

    A fixture corpus must be genuinely synthetic: a byte-copy of the spent
    production confirmatory artifacts under a fixture root is still the
    held-back data, and renaming is not isolation. Compared by content digest
    against the production tree's own artifacts, so path aliasing does not
    matter, and any one-byte difference counts as a different corpus. Never
    called in production mode, where the corpus is the production corpus by
    definition and the sentinel guards it.
    """
    production_dir = PRODUCTION_PROJECT_ROOT / "data" / "confirmatory"
    for artifact in sorted(_confirmatory_directory().glob("ud_ewt_test_*_counts.json")):
        counterpart = production_dir / artifact.name
        if not counterpart.is_file():
            continue
        fixture_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        production_digest = hashlib.sha256(counterpart.read_bytes()).hexdigest()
        if fixture_digest == production_digest:
            return (
                f"REFUSED: fixture corpus artifact {artifact.name} is byte-identical to "
                "the spent production confirmatory corpus. A copy of the held-back "
                "data is not a safe shakedown fixture; a fixture corpus must be "
                "genuinely synthetic. No claims written."
            )
    return None


def _write_spent_marker(marker_path: Path, claims_path: Path) -> None:
    """Publish the filesystem half of the one-run sentinel, atomically.

    The ledger claim is the authority; this marker is the second channel the
    precheck's filesystem scan already treats as evidence, so a bypass ledger
    that shows no sentinel still leaves proof beside wherever the run wrote.
    In fixture mode it lands beside the isolated ledger; a production run
    would place it beside data/claims.json. The claim is written first, so a
    crash between the two leaves the ledger spent with no marker — the
    fail-closed direction. The destination is resolved first: the atomic
    writer refuses symlinked path components (macOS /tmp and /var are
    symlinks), and a fixture ledger under such a prefix is still the right
    destination by content of its realpath.
    """
    from _artifact_io import atomic_write_text

    atomic_write_text(
        marker_path.resolve(),
        json.dumps(
            {
                "sentinel": SENTINEL,
                "population": POPULATION,
                "decision": 111,
                "run_id": RUN_ID,
                "ledger": str(claims_path),
            },
            indent=1,
        )
        + "\n",
    )


def _merged_documents(
    development_ids: list[str],
    development_train_indices: list[int],
    confirmatory_ids: list[str],
) -> tuple[list[str], dict[str, int], int]:
    """Merged ordered identities: development train documents, then confirmatory.

    Only the development *train* documents are carried over, because only their
    sequences are read, and a merged identity list containing the development
    test documents would have to place them in an arm they do not belong to.

    The DISJOINTNESS check, though, is against the whole development
    population -- train and held-out development arms alike. Decision 111
    falsifier 4 refuses on any intersection between the confirmatory
    population and *the development population*, and a collision confined to
    the development held-out arm would otherwise slip past a train-only
    merge and spend the population on a frame whose "held back" claim is
    false. The test suite compares the full development id set; the runtime
    guard now does the same.
    """
    train_ids = [development_ids[index] for index in sorted(development_train_indices)]
    overlap = set(development_ids) & set(confirmatory_ids)
    if overlap:
        raise ProtocolRefusal(
            f"decision 111 falsifier 4: {len(overlap)} document identity/identities appear "
            f"in both the confirmatory population and the development population (train "
            f"and held-out arms alike), so the confirmatory population is not held back. "
            f"No claims written. Example: {sorted(overlap)[:3]}"
        )
    merged = [*train_ids, *confirmatory_ids]
    return merged, {document_id: index for index, document_id in enumerate(merged)}, len(train_ids)


def _merged_provenance(
    development: dict, confirmatory: dict, merged_ids: list[str], n_train_documents: int
) -> dict[str, Any]:
    """Provenance for a merged artifact, honest about being two populations."""
    from ailm.corpus.ud import document_identity_sha256

    return {
        **{key: value for key, value in confirmatory["provenance"].items() if key != "documents"},
        "corpus": (
            "merged scoring frame: UD English EWT dev train split supplies the "
            "frozen parameters, UD English EWT test supplies the whole evaluation "
            "population (decision 111)"
        ),
        "parameter_source": development["provenance"]["corpus"],
        "evaluation_source": confirmatory["provenance"]["corpus"],
        "documents": len(merged_ids),
        "train_documents": n_train_documents,
        "test_documents": len(merged_ids) - n_train_documents,
        "document_identity_sha256": document_identity_sha256(merged_ids),
    }


def merged_case_payload() -> dict:
    """The merged scoring frame as a payload: development train, confirmatory test."""
    from ailm.levels.case import DEFAULT_ARTIFACT

    development = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    confirmatory = _load_confirmatory("case")

    merged_ids, index_of, n_train = _merged_documents(
        development["documents"]["ids"],
        development["documents"]["split_indices"]["train"],
        confirmatory["documents"]["ids"],
    )
    development_ids = development["documents"]["ids"]
    confirmatory_ids = confirmatory["documents"]["ids"]

    populations: dict[str, dict[str, list]] = {}
    sequence_documents: dict[str, dict[str, list[int]]] = {}
    for name in development["populations"]:
        train_sequences = development["populations"][name]["train"]
        train_documents = [
            index_of[development_ids[index]]
            for index in development["documents"]["sequence_document_indices"][name]["train"]
        ]
        test_sequences = confirmatory["populations"][name]
        test_documents = [
            index_of[confirmatory_ids[index]]
            for index in confirmatory["documents"]["sequence_document_indices"][name]
        ]
        populations[name] = {"train": train_sequences, "test": test_sequences}
        sequence_documents[name] = {"train": train_documents, "test": test_documents}

    payload = {
        "schema": development["schema"],
        "provenance": _merged_provenance(development, confirmatory, merged_ids, n_train),
        "heuristic_vs_gold": confirmatory["heuristic_vs_gold"],
        "oracle_scope_audit": confirmatory["oracle_scope_audit"],
        "populations": populations,
        "documents": {
            "ids": merged_ids,
            "split_indices": {
                "train": list(range(n_train)),
                "test": list(range(n_train, len(merged_ids))),
            },
            "sequence_document_indices": sequence_documents,
        },
    }
    return payload


def merged_case_artifact(population: str):
    """A ``CaseArtifact`` whose train arm is development and whose test arm is the whole test file."""
    from ailm.levels.case import CaseArtifact

    return CaseArtifact(merged_case_payload(), population)


def _with_swapped_arms(payload: dict) -> dict:
    """The same frame with its two arms exchanged, for the planted-effect control.

    Fitting on the confirmatory arm is the plant. Keeping the development arm as
    the (unused) held-out side rather than dropping it is what preserves the
    observation alphabet: ``CaseArtifact`` derives the alphabet from
    ``train + test``, so an artifact carrying only the confirmatory sequences
    would build a narrower emission whose row indices no longer line up with the
    frame the scoring runs in.
    """
    swapped = json.loads(json.dumps(payload))
    documents = swapped["documents"]
    documents["split_indices"] = {
        "train": documents["split_indices"]["test"],
        "test": documents["split_indices"]["train"],
    }
    for name, splits in swapped["populations"].items():
        splits["train"], splits["test"] = splits["test"], splits["train"]
        mapping = documents["sequence_document_indices"][name]
        mapping["train"], mapping["test"] = mapping["test"], mapping["train"]
    provenance = swapped["provenance"]
    provenance["train_documents"], provenance["test_documents"] = (
        provenance["test_documents"],
        provenance["train_documents"],
    )
    provenance["corpus"] = "planted-effect positive control: fitted on the confirmatory arm"
    return swapped


def _with_shuffled_test_labels(payload: dict, seed: int) -> dict:
    """The same frame with the confirmatory arm's labels deterministically permuted.

    Observations, sequence lengths, document alignment and the whole train arm are
    untouched, so the frozen parameters and the frozen comparator table are the
    ones that meet the permuted labels. Only the association between observation
    and label is destroyed, which is exactly what the negative control is for.
    """
    shuffled = json.loads(json.dumps(payload))
    for splits in shuffled["populations"].values():
        labels = [case for sequence in splits["test"] for _upos, case in sequence]
        permuted = _shuffled(labels, seed)
        cursor = 0
        rebuilt = []
        for sequence in splits["test"]:
            rebuilt.append(
                [[upos, permuted[cursor + offset]] for offset, (upos, _case) in enumerate(sequence)]
            )
            cursor += len(sequence)
        splits["test"] = rebuilt
    shuffled["provenance"]["corpus"] = (
        f"label-shuffled negative control over the confirmatory arm (seed {seed})"
    )
    return shuffled


def merged_clause_artifact():
    """A ``ClauseArtifact`` with development train sentences and the confirmatory test arm."""
    from ailm.levels.clause import DEFAULT_ARTIFACT, ClauseArtifact

    development = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    confirmatory = _load_confirmatory("clause")

    merged_ids, index_of, n_train = _merged_documents(
        development["documents"]["ids"],
        development["documents"]["split_indices"]["train"],
        confirmatory["documents"]["ids"],
    )
    development_ids = development["documents"]["ids"]
    confirmatory_ids = confirmatory["documents"]["ids"]

    payload = {
        "schema": development["schema"],
        "provenance": _merged_provenance(development, confirmatory, merged_ids, n_train),
        "splits": {
            "train": development["splits"]["train"],
            "test": confirmatory["entries"],
        },
        "documents": {
            "ids": merged_ids,
            "split_indices": {
                "train": list(range(n_train)),
                "test": list(range(n_train, len(merged_ids))),
            },
            "sequence_document_indices": {
                "train": [
                    index_of[development_ids[index]]
                    for index in development["documents"]["sequence_document_indices"]["train"]
                ],
                "test": [
                    index_of[confirmatory_ids[index]]
                    for index in confirmatory["documents"]["sequence_document_indices"]
                ],
            },
        },
    }
    return ClauseArtifact(payload)


def merged_narrative_artifact():
    """A ``NarrativeArtifact`` with development train documents and the confirmatory test arm."""
    from ailm.levels.narrative import DEFAULT_ARTIFACT, NarrativeArtifact

    development = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    confirmatory = _load_confirmatory("narrative")

    merged_ids, index_of, n_train = _merged_documents(
        development["documents"]["ids"],
        development["documents"]["split_indices"]["train"],
        confirmatory["documents"]["ids"],
    )
    train_entries = development["splits"]["train"]
    test_entries = confirmatory["entries"]
    eligible = len(train_entries) + len(test_entries)

    provenance = _merged_provenance(development, confirmatory, merged_ids, n_train)
    provenance.update(
        {
            "documents_total": len(merged_ids),
            "train_eligible_documents": len(train_entries),
            "test_eligible_documents": len(test_entries),
            "documents_excluded_short": len(merged_ids) - eligible,
            "documents_with_typed_sentences": len(merged_ids),
        }
    )
    payload = {
        "schema": development["schema"],
        "provenance": provenance,
        "splits": {"train": train_entries, "test": test_entries},
        "documents": {
            "ids": merged_ids,
            "split_indices": {
                "train": list(range(n_train)),
                "test": list(range(n_train, len(merged_ids))),
            },
            "sequence_document_indices": {
                "train": [index_of[entry["doc"]] for entry in train_entries],
                "test": [index_of[entry["doc"]] for entry in test_entries],
            },
        },
    }
    return NarrativeArtifact(payload)


# ── The parameters-are-unchanged guard ───────────────────────────────────────


def assert_case_parameters_are_unchanged(merged, population: str) -> None:
    """The merged frame must build exactly the development model, array for array.

    ``build_case_emission`` and its siblings read ``artifact.train`` only, but the
    observation alphabet is derived from ``train + test``, so a confirmatory tag
    the development split never saw would re-index every row. Comparing the built
    arrays against the development artifact's is the check that catches that, and
    anything else that would silently change the model.
    """
    from ailm.levels.case import (
        build_case_emission,
        build_case_start,
        build_case_transition,
        context_free_baseline,
        load_case_artifact,
    )

    development = load_case_artifact(population)
    if merged.upos_tags != development.upos_tags:
        raise ProtocolRefusal(
            f"decision 111 falsifier 3: the merged {population} observation alphabet "
            f"{merged.upos_tags} differs from the development one {development.upos_tags}, "
            f"so the emission would be re-indexed and the model would not be the frozen "
            f"one. No claims written."
        )
    for name, builder in (
        ("emission", build_case_emission),
        ("transition", build_case_transition),
        ("start", build_case_start),
    ):
        if not np.array_equal(builder(merged), builder(development)):
            raise ProtocolRefusal(
                f"decision 111 falsifier 3: the {population} {name} built from the merged "
                f"frame differs from the development one; a parameter was re-fitted. "
                f"No claims written."
            )
    if context_free_baseline(merged) != context_free_baseline(development):
        raise ProtocolRefusal(
            f"decision 111 falsifier 3: the {population} context-free comparator table "
            f"differs from the development one. No claims written."
        )


def _preflight() -> dict[str, Any]:
    """Deterministic non-scoring validation, before the sentinel spends anything.

    Every refusal decision 111 can make without computing a statistic happens
    here, under the ledger lock and before the sentinel write: absent or
    mislabeled confirmatory artifacts, a stride, a document identity that
    intersects the development population (falsifier 4, train and held-out
    arms alike), and the parameters-are-unchanged guards (falsifier 3). All
    of it used to run after the sentinel write, so any such failure burned
    the population with zero member claims recorded, and recovery required
    hand-deleting a sentinel the registration forbids deleting. The corpus
    identity check also lives here: a fixture corpus that is a byte-copy of
    the spent production corpus is not a fixture. Returns the prebuilt
    frames and frozen models so the post-sentinel phase computes statistics
    and nothing else.
    """
    from ailm.levels.case import CaseArtifact
    from ailm.levels.clause import build_type_models, load_clause_artifact
    from ailm.levels.narrative import build_phase_models, load_narrative_artifact

    if _corpus_root().resolve() != PRODUCTION_PROJECT_ROOT.resolve():
        corpus_problem = _corpus_identity_problem()
        if corpus_problem is not None:
            raise ProtocolRefusal(corpus_problem)

    case_payload = merged_case_payload()
    case = {
        population: CaseArtifact(case_payload, population) for population in ("derived", "gold")
    }
    for population, merged in case.items():
        assert_case_parameters_are_unchanged(merged, population)

    clause_merged = merged_clause_artifact()
    clause_models = build_type_models(load_clause_artifact().train)
    for key, built in build_type_models(clause_merged.train).items():
        if not np.array_equal(built, clause_models[key]):
            raise ProtocolRefusal(
                f"decision 111 falsifier 3: the clause models[{key}] built from the "
                f"merged frame differs from the development one. No claims written."
            )

    narrative_merged = merged_narrative_artifact()
    narrative_models = build_phase_models(load_narrative_artifact().train)
    for key, built in build_phase_models(narrative_merged.train).items():
        if not np.array_equal(built, narrative_models[key]):
            raise ProtocolRefusal(
                f"decision 111 falsifier 3: the narrative {key} built from the merged "
                f"frame differs from the development one. No claims written."
            )

    return {
        "case_payload": case_payload,
        "case": case,
        "clause_merged": clause_merged,
        "clause_models": clause_models,
        "narrative_merged": narrative_merged,
        "narrative_models": narrative_models,
    }


# ── Scoring one member at a time ─────────────────────────────────────────────


def _mcnemar(favored: list[bool], opposed: list[bool]) -> tuple[int, int, float]:
    """Exact paired McNemar over aligned outcome vectors, as the stage recorders do."""
    if len(favored) != len(opposed):
        raise ProtocolRefusal(f"unpaired vectors: {len(favored)} vs {len(opposed)}")
    b = sum(1 for m, s in zip(favored, opposed, strict=True) if m and not s)
    c = sum(1 for m, s in zip(favored, opposed, strict=True) if s and not m)
    return b, c, exact_two_sided_binomial_p(b, c)


def score_stage4(population: str, merged) -> dict[str, Any]:
    """The Stage 4 paired comparison on the whole confirmatory population.

    The merged artifact and the parameters-unchanged assertion are preflight
    work; this function computes the statistic and nothing else.
    """
    from ailm.levels.case import (
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        case_agreement,
        decode_case_positions,
    )

    emission = build_case_emission(merged)
    transition = build_case_transition(merged)
    start = build_case_start(merged)
    model_hits = decode_case_positions(merged, emission, transition, start)
    baseline_hits = baseline_case_positions(merged)
    b, c, p = _mcnemar(model_hits, baseline_hits)
    return {
        "artifact": merged,
        "b": b,
        "c": c,
        "p": p,
        "n_positions": len(model_hits),
        "n_sequences": len(merged.test),
        "n_documents": len(merged.test_document_ids),
        "model_pct": 100.0 * case_agreement(model_hits),
        "baseline_pct": 100.0 * case_agreement(baseline_hits),
    }


def score_stage5(models, merged) -> dict[str, Any]:
    """The Stage 5 paired classification comparison on the whole confirmatory population.

    The merged artifact, the development models and the parameters-unchanged
    assertion are preflight work; this function computes the statistic and
    nothing else.
    """
    from ailm.levels.clause import classify_sentences

    classification = classify_sentences(merged.test, models)
    b, c, p = _mcnemar(classification["model_hits"], classification["majority_hits"])
    return {
        "artifact": merged,
        "classification": classification,
        "models": models,
        "b": b,
        "c": c,
        "p": p,
        "n_sentences": len(merged.test),
        "n_documents": len(merged.test_document_ids),
        "model_pct": 100.0 * classification["accuracy"],
        "majority_pct": 100.0 * classification["majority_accuracy"],
    }


def score_stage6(models, merged) -> dict[str, Any]:
    """The Stage 6 sign test on the whole confirmatory population.

    The merged artifact, the development models and the parameters-unchanged
    assertion are preflight work; this function computes the statistic and
    nothing else.
    """
    from ailm.levels.narrative import heldout_type_logloss

    recognition = heldout_type_logloss(merged.test, models)
    positive = sum(1 for delta in recognition["deltas"] if delta > 0)
    negative = sum(1 for delta in recognition["deltas"] if delta < 0)
    return {
        "artifact": merged,
        "models": models,
        "positive": positive,
        "negative": negative,
        "p": exact_two_sided_binomial_p(positive, negative),
        "n_sentences": recognition["n_sentences"],
        "n_documents": len(merged.test),
        "phase_logloss": recognition["phase_mean"],
        "unigram_logloss": recognition["unigram_mean"],
        "improvement_nats": recognition["mean_delta"],
    }


# ── Controls ─────────────────────────────────────────────────────────────────


def _shuffled(values: list, seed: int) -> list:
    """A deterministic permutation of a list, so the control is one fixed control."""
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(values))
    return [values[index] for index in order]


def run_controls(preflight: dict[str, Any]) -> dict[str, Any]:
    """A planted-effect positive control and a label-shuffled negative control.

    Both run on the confirmatory population itself, in the same scoring frame the
    members are scored in, so neither control is a different experiment wearing
    the same name. The positive control fits the model on that population's own
    labels: a privileged model that must beat the frozen comparator decisively,
    or the instrument cannot detect an effect here and a null would be
    instrument-ambiguous. The negative control permutes the labels and keeps the
    frozen parameters: the comparison must not then produce a model-favored
    crossing, or it would credit noise.
    """
    from ailm.control import ControlHarness
    from ailm.levels.case import (
        CaseArtifact,
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
    )

    harness = ControlHarness("confirmatory_ud_ewt_test")
    results: dict[str, Any] = {}

    payload = preflight["case_payload"]
    merged = CaseArtifact(payload, "derived")
    emission = build_case_emission(merged)
    transition = build_case_transition(merged)
    start = build_case_start(merged)
    baseline_hits = baseline_case_positions(merged)

    # POSITIVE: privileged parameters fitted on the confirmatory labels, decoded
    # over the confirmatory arm and compared against the same frozen comparator.
    planted = CaseArtifact(_with_swapped_arms(payload), "derived")
    planted_hits = decode_case_positions(
        merged,
        build_case_emission(planted),
        build_case_transition(planted),
        build_case_start(planted),
    )
    positive_b, positive_c, positive_p = _mcnemar(planted_hits, baseline_hits)
    results.update({"positive_b": positive_b, "positive_c": positive_c, "positive_p": positive_p})
    if not (positive_b > positive_c and positive_p < BONFERRONI_ALPHA):
        raise ProtocolRefusal(
            f"decision 111 falsifier 5: the planted-effect positive control did NOT fire on "
            f"this population (b={positive_b}, c={positive_c}, p={positive_p:.4g} against "
            f"{BONFERRONI_ALPHA:.6g}). The instrument cannot detect an effect here, so a "
            f"null would be instrument-ambiguous. No member is scored."
        )
    harness.assert_gate_is_live(
        good_value=float(positive_b), bad_value=float(positive_c), direction="higher_is_better"
    )

    # NEGATIVE: the frozen parameters and the frozen comparator against permuted
    # labels. Both sides of the pairing degrade together, which is the point.
    shuffled = CaseArtifact(_with_shuffled_test_labels(payload, NEGATIVE_CONTROL_SEED), "derived")
    negative_b, negative_c, negative_p = _mcnemar(
        decode_case_positions(shuffled, emission, transition, start),
        baseline_case_positions(shuffled),
    )
    results.update(
        {
            "negative_b": negative_b,
            "negative_c": negative_c,
            "negative_p": negative_p,
            "negative_seed": NEGATIVE_CONTROL_SEED,
        }
    )
    if negative_b > negative_c and negative_p < BONFERRONI_ALPHA:
        raise ProtocolRefusal(
            f"decision 111 falsifier 5: the label-shuffled negative control FIRED "
            f"(b={negative_b}, c={negative_c}, p={negative_p:.4g} against "
            f"{BONFERRONI_ALPHA:.6g}). The comparison credits the model on permuted "
            f"labels, so it would accept noise. No member is scored."
        )
    return results


# ── Recording ────────────────────────────────────────────────────────────────


def _direction_agrees(confirmatory: tuple[int, int], development: tuple[int, int]) -> bool | None:
    """Whether the confirmatory discordance points the same way as the development one.

    ``None`` when either comparison has no direction to agree with: a tie, or the
    b = c = 0 case decision 79 names explicitly. A tie recorded as ``False``
    would read as a reversal, which it is not.
    """
    for favored, opposed in (confirmatory, development):
        if favored == opposed:
            return None
    return (confirmatory[0] > confirmatory[1]) == (development[0] > development[1])


def _development_counts(ledger, favored_label: str, opposed_label: str) -> tuple[int, int]:
    counts = []
    for label in (favored_label, opposed_label):
        entry = ledger.get(label)
        if entry is None:
            raise ProtocolRefusal(f"missing development sufficient count {label!r}")
        counts.append(int(entry["value"]))
    return counts[0], counts[1]


def build_values(
    ledger, stage4: dict, stage5: dict, stage6: dict, controls: dict
) -> tuple[dict, dict]:
    """Return the (confirmatory, descriptive) value maps this run records."""
    count_labels = {p: (favored, opposed) for p, favored, opposed in SIGNIFICANCE_COUNT_LABELS}

    scored = {
        "stage4_derived_mcnemar_p": (
            stage4["derived"]["b"],
            stage4["derived"]["c"],
            stage4["derived"]["p"],
        ),
        "stage4_gold_mcnemar_p": (stage4["gold"]["b"], stage4["gold"]["c"], stage4["gold"]["p"]),
        "stage5_mcnemar_p": (stage5["b"], stage5["c"], stage5["p"]),
        "stage6_sign_test_p": (stage6["positive"], stage6["negative"], stage6["p"]),
    }
    if set(scored) != set(REGISTERED_MEMBERS):
        raise ProtocolRefusal(
            f"scored members {sorted(scored)} are not the registered four "
            f"{sorted(REGISTERED_MEMBERS)}; decision 111 forbids re-selection"
        )

    # Holm over the frozen family, with these four substituted for their
    # development counterparts and the other twelve left exactly as recorded.
    substituted: dict[str, float] = {}
    for member in SIGNIFICANCE_FAMILY:
        if member in scored:
            substituted[member] = scored[member][2]
            continue
        favored_label, opposed_label = count_labels[member]
        favored, opposed = _development_counts(ledger, favored_label, opposed_label)
        substituted[member] = exact_two_sided_binomial_p(favored, opposed)
    holm = {decision.label: decision for decision in holm_step_down(substituted, FAMILYWISE_ALPHA)}

    confirmatory: dict[str, Any] = {SENTINEL: True}
    descriptive: dict[str, Any] = {}
    for member, (favored, opposed, p_value) in scored.items():
        prefix = f"confirmatory_{member.removesuffix('_p')}"
        development_counts = _development_counts(ledger, *count_labels[member])
        crosses = p_value < BONFERRONI_ALPHA
        confirmatory.update(
            {
                f"{prefix}_favored": favored,
                f"{prefix}_opposed": opposed,
                f"{prefix}_p": float(f"{p_value:.4g}"),
                f"{prefix}_discordant_n": favored + opposed,
                f"{prefix}_crosses_frozen_threshold": crosses,
                f"{prefix}_holm_rejected": holm[member].rejected,
                f"{prefix}_holm_rank": holm[member].rank,
                f"{prefix}_holm_threshold": float(f"{holm[member].threshold:.6g}"),
                f"{prefix}_computable": (favored + opposed) > 0,
            }
        )
        agreement = _direction_agrees((favored, opposed), development_counts)
        confirmatory[f"{prefix}_direction_agrees"] = (
            "undefined" if agreement is None else str(bool(agreement)).lower()
        )

    descriptive.update(
        {
            "confirmatory_population": POPULATION,
            "confirmatory_registration_decision": "111",
            "confirmatory_frozen_alpha": BONFERRONI_ALPHA,
            "confirmatory_frozen_family_size": len(SIGNIFICANCE_FAMILY),
            "confirmatory_members_scored": len(scored),
            "confirmatory_census_sentences": stage4["census"]["sentences"],
            "confirmatory_census_tokens": stage4["census"]["tokens"],
            "confirmatory_census_documents": stage4["census"]["documents"],
            "confirmatory_stage4_derived_positions": stage4["derived"]["n_positions"],
            "confirmatory_stage4_derived_documents": stage4["derived"]["n_documents"],
            "confirmatory_stage4_derived_agreement_pct": round(stage4["derived"]["model_pct"], 2),
            "confirmatory_stage4_derived_baseline_pct": round(stage4["derived"]["baseline_pct"], 2),
            "confirmatory_stage4_gold_positions": stage4["gold"]["n_positions"],
            "confirmatory_stage4_gold_documents": stage4["gold"]["n_documents"],
            "confirmatory_stage4_gold_agreement_pct": round(stage4["gold"]["model_pct"], 2),
            "confirmatory_stage4_gold_baseline_pct": round(stage4["gold"]["baseline_pct"], 2),
            "confirmatory_stage5_sentences": stage5["n_sentences"],
            "confirmatory_stage5_documents": stage5["n_documents"],
            "confirmatory_stage5_classification_pct": round(stage5["model_pct"], 2),
            "confirmatory_stage5_majority_pct": round(stage5["majority_pct"], 2),
            "confirmatory_stage6_sentences": stage6["n_sentences"],
            "confirmatory_stage6_documents": stage6["n_documents"],
            "confirmatory_stage6_phase_logloss": round(stage6["phase_logloss"], 4),
            "confirmatory_stage6_unigram_logloss": round(stage6["unigram_logloss"], 4),
            "confirmatory_stage6_improvement_nats": round(stage6["improvement_nats"], 4),
            "confirmatory_control_positive_favored": controls["positive_b"],
            "confirmatory_control_positive_opposed": controls["positive_c"],
            "confirmatory_control_positive_p": float(f"{controls['positive_p']:.4g}"),
            "confirmatory_control_negative_favored": controls["negative_b"],
            "confirmatory_control_negative_opposed": controls["negative_c"],
            "confirmatory_control_negative_p": float(f"{controls['negative_p']:.4g}"),
            "confirmatory_control_negative_seed": controls["negative_seed"],
        }
    )
    return confirmatory, descriptive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the four ud_ewt_test members once.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-derive the recorded p-values from the recorded counts; write nothing",
    )
    args = parser.parse_args(argv)

    from ailm.claims import ClaimBatch, ClaimsLedger

    mode, claims_path, mode_problem = _execution_mode()
    if mode_problem is not None:
        raise ProtocolRefusal(mode_problem)
    ledger = ClaimsLedger(claims_path)

    if args.verify:
        return _verify(ledger)

    if ledger.get(SENTINEL) is not None:
        raise ProtocolRefusal(
            f"REFUSED: {SENTINEL} already exists in the ledger. Decision 111 registers one "
            f"run per member, ever; a second pass over a held-back population un-holds it. "
            f"Nothing was read, computed or written."
        )

    marker_path = claims_path.parent / MARKER_NAME
    if marker_path.exists():
        raise ProtocolRefusal(
            f"REFUSED: {marker_path} already exists. The filesystem half of the one-run "
            "sentinel is present even though this ledger shows no sentinel claim, so a "
            "second run would still repeat a spent population. Nothing was read, "
            "computed or written."
        )

    preflight = _preflight()

    # The population is spent the moment it is scored, whether or not the
    # scoring finishes, so the sentinel -- and, beside the ledger the run
    # writes, its filesystem marker -- land after the deterministic
    # non-scoring preflight and immediately before the first
    # statistic-computing step.
    ledger.record_many({SENTINEL: True}, run_id=RUN_ID, source=SOURCE, status="confirmatory")
    _write_spent_marker(marker_path, claims_path)
    print(f"sentinel written: {SENTINEL}; spent marker at {marker_path}; scoring begins.")

    controls = run_controls(preflight)
    print(
        f"controls passed: positive b/c = {controls['positive_b']}/{controls['positive_c']} "
        f"(p={controls['positive_p']:.4g}); negative b/c = "
        f"{controls['negative_b']}/{controls['negative_c']} (p={controls['negative_p']:.4g})"
    )

    derived = score_stage4("derived", preflight["case"]["derived"])
    gold = score_stage4("gold", preflight["case"]["gold"])
    census = derived["artifact"].provenance
    stage4 = {
        "derived": derived,
        "gold": gold,
        "census": {
            "sentences": census["sentences"],
            "tokens": census["tokens"],
            "documents": census["test_documents"],
        },
    }
    stage5 = score_stage5(preflight["clause_models"], preflight["clause_merged"])
    stage6 = score_stage6(preflight["narrative_models"], preflight["narrative_merged"])

    confirmatory, descriptive = build_values(ledger, stage4, stage5, stage6, controls)
    ledger.replace_claims(
        batches=[
            ClaimBatch(values=confirmatory, run_id=RUN_ID, source=SOURCE, status="confirmatory"),
            ClaimBatch(values=descriptive, run_id=RUN_ID, source=SOURCE, status="descriptive"),
        ]
    )

    print(
        f"\nRecorded {len(confirmatory)} confirmatory and {len(descriptive)} descriptive claims "
        f"at frozen alpha {BONFERRONI_ALPHA:.6g}:\n"
    )
    for member in REGISTERED_MEMBERS:
        prefix = f"confirmatory_{member.removesuffix('_p')}"
        print(
            f"  {member:<28} b/c = {confirmatory[f'{prefix}_favored']}/"
            f"{confirmatory[f'{prefix}_opposed']}  p = {confirmatory[f'{prefix}_p']}  "
            f"crosses = {confirmatory[f'{prefix}_crosses_frozen_threshold']}  "
            f"holm = {confirmatory[f'{prefix}_holm_rejected']}  "
            f"direction agrees = {confirmatory[f'{prefix}_direction_agrees']}"
        )
    return 0


def _verify(ledger) -> int:
    """Re-derive every recorded p-value from its recorded counts and compare."""
    if ledger.get(SENTINEL) is None:
        print(f"nothing to verify: {SENTINEL} is absent, so no run has been recorded.")
        return 1
    defects: list[str] = []
    for member in REGISTERED_MEMBERS:
        prefix = f"confirmatory_{member.removesuffix('_p')}"
        favored = ledger.get(f"{prefix}_favored")
        opposed = ledger.get(f"{prefix}_opposed")
        stored = ledger.get(f"{prefix}_p")
        if favored is None or opposed is None or stored is None:
            defects.append(f"{member}: recorded counts or p-value missing")
            continue
        exact = exact_two_sided_binomial_p(int(favored["value"]), int(opposed["value"]))
        if float(stored["value"]) != float(f"{exact:.4g}"):
            defects.append(
                f"{member}: stored p {stored['value']!r} does not reproduce from "
                f"b={favored['value']} c={opposed['value']} (exact {exact!r})"
            )
    for defect in defects:
        print(f"  DEFECT: {defect}")
    if defects:
        return 1
    print(f"verified: all {len(REGISTERED_MEMBERS)} recorded p-values reproduce from their counts.")
    return 0


if __name__ == "__main__":
    # Write map, in the order a run touches anything: (1) the outer CLI lock,
    # acquired here BEFORE main() validates the execution mode;
    # ledger_lock_path resolves ``AILM_CLAIMS_PATH`` itself, so the lock is
    # keyed to the ledger the run would actually write -- beside the fixture
    # ledger in fixture mode, beside data/claims.json in production mode --
    # and never inferred from the production-root argument alone; (2) the
    # sentinel claim inside the ledger, after the mode guard, the sentinel
    # refusal and the preflight; (3) the spent marker beside that same ledger;
    # (4) the outcome claims. In fixture mode every one of these lands inside
    # the fixture root; a mode-guard refusal happens under the lock but writes
    # nothing else.
    #
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(PRODUCTION_PROJECT_ROOT)):
        raise SystemExit(main())
