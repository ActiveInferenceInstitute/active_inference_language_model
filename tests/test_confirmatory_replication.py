"""Guards for the decision-111 confirmatory replication of the four EWT members.

A confirmatory result is only worth more than an exploratory one because of the
discipline around it, so the discipline is what this file tests, not the numbers.
Five properties have to hold, and each of them is a way the run could have been
worthless without looking it:

1. **The run cannot repeat.** A second pass over a held-back population un-holds
   it. The sentinel makes the recorder refuse, and that refusal is checked by
   running the real recorder in a real subprocess against an isolated ledger, not
   by asserting that a branch exists.
2. **The recorded p-values are re-derivable.** Every verdict reproduces from its
   own recorded integer counts through the same exact test, so a stored scalar
   cannot drift away from the counts that justify it.
3. **The population really was held back.** Every confirmatory document identity
   is compared against every development one. An intersection would mean the
   model had seen its own evaluation data.
4. **The frozen family did not move.** The threshold is the familywise alpha
   divided by the family size, so a family edited between registration and run
   would silently move every verdict.
5. **Nothing recorded is unreachable.** Every label the recorder writes is in
   ``REQUIRED_CLAIMS``, so a render cannot describe the replication without it.

The vendored artifacts' digests are bound to their filenames here rather than in
``tests/test_provenance_checksums.py``, because that file's inventory is held
equal to the wheel's packaged surface and these artifacts are deliberately not
packaged. The binding and its positive control are the same shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ailm.claims import ClaimsLedger
from ailm.required_claims import CONFIRMATORY_REPLICATION_CLAIMS, REQUIRED_CLAIMS
from ailm.statistics import (
    BONFERRONI_ALPHA,
    CONFIRMATORY_REPLICATION,
    FAMILYWISE_ALPHA,
    SIGNIFICANCE_FAMILY,
    exact_two_sided_binomial_p,
    holm_step_down,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RECORDER = PROJECT_ROOT / "scripts" / "record_confirmatory_replication.py"
GENERATOR = PROJECT_ROOT / "scripts" / "generate_confirmatory_ewt_counts.py"
CONFIRMATORY_DIR = PROJECT_ROOT / "data" / "confirmatory"
DEVELOPMENT_CASE_ARTIFACT = PROJECT_ROOT / "vendor" / "ud_ewt_case_counts.json"

SENTINEL = "confirmatory_run_completed_ud_ewt_test"

#: Filename-bound digests, transcribed from docs/provenance/sources.md. Two
#: independent records of the same fact is the whole mechanism: a regenerated
#: artifact moves the file, and one of the two records then disagrees.
DOCUMENTED_DIGESTS = {
    "ud_ewt_test_case_counts.json": (
        "68e68daed5503d9095d083124a52590846bccd672901b2b2a505b94ab3007573"
    ),
    "ud_ewt_test_clause_counts.json": (
        "2277b46ccfda016c4530a5fcf786842d8566784780747563ecbce9619f7d830e"
    ),
    "ud_ewt_test_narrative_counts.json": (
        "6d15a121ddd6ceae168f92d4b07040bc5964995ac749de770f20ee38aa96529a"
    ),
}

#: The four members decision 79 assigned to this population and decision 111
#: scored. Written out rather than derived so a silent re-mapping in
#: ``CONFIRMATORY_REPLICATION`` is caught rather than followed.
REGISTERED_MEMBERS = (
    "stage4_derived_mcnemar_p",
    "stage4_gold_mcnemar_p",
    "stage5_mcnemar_p",
    "stage6_sign_test_p",
)

_SKIP_NO_LEDGER = pytest.mark.skipif(
    not LEDGER.exists(), reason="data/claims.json absent; run the recorders"
)
_SKIP_NO_RUN = pytest.mark.skipif(
    not LEDGER.exists()
    or json.loads(LEDGER.read_text(encoding="utf-8"))["claims"].get(SENTINEL) is None,
    reason="the decision-111 run has not been recorded in this checkout",
)


def _claims() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8"))["claims"]


def _value(label: str):
    entry = _claims().get(label)
    assert entry is not None, f"{label} is absent from the ledger"
    return entry["value"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix(member: str) -> str:
    return f"confirmatory_{member.removesuffix('_p')}"


# ── 1. The run cannot repeat ─────────────────────────────────────────────────


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project interpreter absent")
@pytest.mark.timeout(900)
def _synthetic_miniature_corpus(confirmatory_dir: Path, *, variant: str = "standard") -> None:
    """Write a genuinely synthetic miniature confirmatory corpus into a fixture root.

    Parses a hand-written miniature CoNLL-U with the shipped reader and runs
    the vendored generator's own population builders over it, so the artifacts
    carry the registered schema and provenance conventions without the test
    hand-rolling JSON. No production byte can match it — the recorder's
    corpus-identity check verifies that at run time — and no held-back token
    is present anywhere in it.
    """
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import generate_confirmatory_ewt_counts as generator

    from ailm.corpus.ud import case_oracle_scope_audit, document_identity_sha256, parse_conllu

    def sentence_lines(subject, subject_case, verb, obj, obj_case, final):
        return [
            f"# text = {subject} {verb} {obj.lower()} {final}",
            f"1\t{subject}\t{subject.lower()}\tPRON\t_\tCase={subject_case}\t2\tnsubj\t_\t_",
            f"2\t{verb}\t{verb.lower()}\tVERB\t_\t_\t0\troot\t_\t_",
            f"3\t{obj}\t{obj.lower()}\tPRON\t_\tCase={obj_case}\t2\tobj\t_\t_",
            f"4\t{final}\t{final}\tPUNCT\t_\t_\t2\tpunct\t_\t_",
        ]

    if variant == "standard":
        documents = {
            "fixture_doc_alpha": [
                ("She", "Nom", "sees", "Them", "Acc", "."),
                ("They", "Nom", "gave", "Her", "Gen", "."),
                ("We", "Nom", "ask", "Them", "Acc", "?"),
                ("He", "Nom", "tells", "Her", "Gen", "."),
                ("It", "Nom", "surprises", "Everyone", "Acc", "."),
                ("She", "Nom", "finds", "It", "Gen", "."),
            ],
            "fixture_doc_beta": [
                ("She", "Nom", "knows", "It", "Acc", "."),
                ("They", "Nom", "see", "Her", "Acc", "?"),
                ("He", "Nom", "tells", "Them", "Gen", "."),
                ("We", "Nom", "ask", "Her", "Acc", "."),
                ("It", "Nom", "bothers", "Them", "Gen", "?"),
                ("They", "Nom", "recall", "Everything", "Acc", "."),
            ],
            "fixture_doc_gamma": [
                ("He", "Nom", "reads", "Them", "Acc", "."),
                ("She", "Nom", "shows", "Her", "Gen", "?"),
                ("They", "Nom", "hear", "Us", "Acc", "."),
                ("We", "Nom", "describe", "It", "Gen", "?"),
                ("She", "Nom", "waits", "Nobody", "Acc", "."),
                ("They", "Nom", "recognize", "Him", "Gen", "."),
            ],
            "fixture_doc_delta": [
                ("You", "Nom", "convince", "Them", "Acc", "?"),
                ("She", "Nom", "diagnoses", "It", "Gen", "."),
                ("They", "Nom", "praise", "Her", "Acc", "?"),
                ("We", "Nom", "remember", "Them", "Gen", "."),
                ("It", "Nom", "distracts", "Everyone", "Acc", "?"),
                ("He", "Nom", "values", "Her", "Gen", "."),
            ],
        }
    elif variant == "underpowered":
        # Deliberately too small for the planted-effect control to reach the
        # frozen threshold: the privileged fit wins every discordance (b>0,
        # c=0) but the exact p stays above BONFERRONI_ALPHA, so the run
        # consumes the sentinel and then refuses at falsifier 5 — the
        # deterministic interrupted-after-consumption state.
        documents = {
            "fixture_doc_alpha": [
                ("She", "Nom", "sees", "Them", "Acc", "."),
                ("They", "Nom", "gave", "Her", "Gen", "."),
                ("We", "Nom", "ask", "Them", "Acc", "?"),
                ("He", "Nom", "tells", "Her", "Gen", "."),
            ],
            "fixture_doc_beta": [
                ("She", "Nom", "knows", "It", "Acc", "."),
                ("They", "Nom", "see", "Her", "Acc", "?"),
                ("He", "Nom", "tells", "Them", "Gen", "."),
                ("We", "Nom", "ask", "Her", "Acc", "."),
            ],
        }
    else:
        raise ValueError(f"unknown fixture corpus variant {variant!r}")
    blocks = []
    for document_id, sentences in documents.items():
        block = [f"# newdoc id = {document_id}"]
        for sentence in sentences:
            block.append("")
            block.extend(sentence_lines(*sentence))
        blocks.append("\n".join(block))

    parsed = parse_conllu("\n\n".join(blocks) + "\n")
    ids = generator.ordered_document_identities(parsed)
    shared = {
        "corpus": (
            "synthetic miniature fixture corpus (hand-written CoNLL-U; not the "
            "held-back EWT test file)"
        ),
        "population": "ud_ewt_test",
        "registration": (
            "hermetic fixture built by tests/test_confirmatory_replication.py; "
            "not a production vendor"
        ),
        "stride_applied": False,
        "split": "none: the synthetic miniature corpus is this fixture's whole evaluation population",
        "documents": len(ids),
        "document_identity_source": "# newdoc id comments from the synthetic miniature CoNLL-U",
        "document_identity_sha256": document_identity_sha256(ids),
    }
    audit = case_oracle_scope_audit(parsed)
    populations, case_indices, n_tokens = generator.build_case_population(parsed)
    clause_entries, clause_indices, n_exclamative, n_unpunctuated = (
        generator.build_clause_population(parsed)
    )
    narrative_entries, with_typed, excluded_short = generator.build_narrative_population(parsed)
    artifacts = {
        "case": {
            "schema": generator.CASE_SCHEMA,
            "provenance": {
                **shared,
                "observation": "UPOS tag only; deprel and feats[Case] withheld",
                "min_sequence_len": generator.MIN_SEQUENCE_LEN,
                "sentences": len(parsed),
                "tokens": n_tokens,
                "gold_label_source": "UD feats[Case]; Nom/Acc/Gen only",
                "derived_label_source": (
                    "ailm.cases.UD_DEPREL_TO_CASE plus token-level ADPOSITION_TO_CASE "
                    "- a research claim, not ground truth"
                ),
            },
            "heuristic_vs_gold": audit["heuristic_vs_gold"],
            "oracle_scope_audit": audit,
            "populations": populations,
            "documents": {"ids": ids, "sequence_document_indices": case_indices},
        },
        "clause": {
            "schema": generator.CLAUSE_SCHEMA,
            "provenance": {
                **shared,
                "observation": "derived case-role sequence; UPOS, deprel and all token forms withheld",
                "min_sequence_len": generator.CLAUSE_MIN_SEQUENCE_LEN,
                "excluded_exclamative": n_exclamative,
                "excluded_unpunctuated": n_unpunctuated,
                "sentences": len(parsed),
                "typed_sentences": len(clause_entries),
            },
            "entries": clause_entries,
            "documents": {"ids": ids, "sequence_document_indices": clause_indices},
        },
        "narrative": {
            "schema": generator.NARRATIVE_SCHEMA,
            "provenance": {
                **shared,
                "observation": "sentence-type sequence per document; typing rules identical to the clause artifact",
                "min_typed_sentences": generator.MIN_TYPED_SENTENCES,
                "documents_with_typed_sentences": with_typed,
                "documents_excluded_short": excluded_short,
                "eligible_documents": len(narrative_entries),
            },
            "entries": narrative_entries,
            "documents": {"ids": ids},
        },
    }
    confirmatory_dir.mkdir(parents=True, exist_ok=True)
    for level, artifact in artifacts.items():
        (confirmatory_dir / f"ud_ewt_test_{level}_counts.json").write_text(
            json.dumps(artifact, indent=1), encoding="utf-8"
        )


def _fixture_environment(
    tmp_path: Path, *, corpus: str, seed_development_rows: bool
) -> tuple[Path, dict[str, str]]:
    """A fully isolated fixture root, ledger and environment for one recorder run.

    The corpus is genuinely synthetic (never a production copy), the ledger is
    a fixture file whose development rows are copied from the published
    ledger — decision 111's substitution rule needs those recorded counts, and
    they are development-population results, not held-back data — and every
    write of the run lands inside ``tmp_path``.
    """
    root = tmp_path / "fixture_root"
    _synthetic_miniature_corpus(root / "data" / "confirmatory")
    ledger = root / "data" / "claims.json"
    if seed_development_rows:
        payload = json.loads(LEDGER.read_text(encoding="utf-8"))
        payload["claims"] = {
            label: entry
            for label, entry in payload["claims"].items()
            if not label.startswith("confirmatory_")
        }
        ledger.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    environment = {
        **os.environ,
        "AILM_CLAIMS_PATH": str(ledger.resolve()),
        "AILM_PROJECT_ROOT": str(root.resolve()),
        "AILM_CONFIRMATORY_ALTERNATE_LEDGER": "1",
    }
    return ledger, environment


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project interpreter absent")
@pytest.mark.timeout(900)
def test_the_sentinel_makes_a_second_run_refuse(tmp_path: Path) -> None:
    """The real recorder, in a real subprocess, against a ledger that already ran.

    A branch that reads ``if sentinel: refuse`` proves nothing on its own; what
    matters is that the shipped entry point, invoked the way a person would
    invoke it, declines and writes nothing. The run is a full fixture-mode
    shakedown — synthetic corpus, isolated ledger, explicit opt-in — and the
    seeded sentinel alone makes it decline before the corpus is read.
    """
    ledger, environment = _fixture_environment(
        tmp_path, corpus="synthetic", seed_development_rows=True
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["claims"][SENTINEL] = {
        "value": True,
        "run_id": "seeded_by_test",
        "source": "tests/test_confirmatory_replication.py",
        "status": "confirmatory",
    }
    ledger.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    before = ledger.read_bytes()

    result = subprocess.run(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert result.returncode != 0, (
        "the recorder ran a second time against a spent population; the one-run "
        f"discipline of decision 111 is not enforced.\n{result.stdout[-1500:]}"
    )
    assert "REFUSED" in result.stdout + result.stderr
    assert ledger.read_bytes() == before, (
        "a refused run still modified the ledger; a refusal that writes is not a refusal"
    )


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project interpreter absent")
@pytest.mark.timeout(900)
def test_the_refusal_is_not_vacuous(tmp_path: Path) -> None:
    """POSITIVE CONTROL: without the sentinel the same invocation gets past the gate.

    A recorder that refused unconditionally would pass the test above while being
    useless. This one removes the sentinel and requires the run to progress far
    enough to write it, which is the first thing a permitted run does. The run
    scores the genuinely synthetic miniature corpus inside the fixture root —
    the real held-back population is never read — and its writes land beside
    the isolated ledger, including the filesystem half of the sentinel.
    """
    ledger, environment = _fixture_environment(
        tmp_path, corpus="synthetic", seed_development_rows=True
    )
    production_before = LEDGER.read_bytes()

    result = subprocess.run(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert "REFUSED" not in result.stdout + result.stderr, (
        "the recorder refuses even without the sentinel, so the refusal test above "
        f"would pass on a recorder that never runs.\n{result.stdout[-1500:]}"
    )
    assert result.returncode == 0, (
        "the recorder did not complete without the sentinel. A run that crashes after "
        "writing the sentinel would satisfy a weaker version of this control while "
        f"proving nothing.\n{result.stdout[-800:]}\n{result.stderr[-1500:]}"
    )
    recorded = json.loads(ledger.read_text(encoding="utf-8"))["claims"]
    assert recorded.get(SENTINEL) is not None
    assert set(CONFIRMATORY_REPLICATION_CLAIMS) <= set(recorded), (
        "the permitted run did not record the declared label set, so the refusal "
        "control above is not comparing against a working recorder"
    )
    assert (ledger.parent / SENTINEL).is_file(), (
        "the permitted run did not write the filesystem half of the sentinel beside "
        "its ledger; the precheck's second channel would have nothing to find"
    )
    assert LEDGER.read_bytes() == production_before, (
        "a fixture-mode run touched the production ledger"
    )


# ── 2. The recorded p-values are re-derivable ────────────────────────────────


@_SKIP_NO_RUN
@pytest.mark.parametrize("member", REGISTERED_MEMBERS)
def test_every_recorded_p_value_reproduces_from_its_recorded_counts(member: str) -> None:
    """The stored scalar must equal the exact test applied to the stored counts."""
    prefix = _prefix(member)
    favored = _value(f"{prefix}_favored")
    opposed = _value(f"{prefix}_opposed")
    stored = float(_value(f"{prefix}_p"))
    exact = exact_two_sided_binomial_p(int(favored), int(opposed))
    assert stored == float(f"{exact:.4g}"), (
        f"{member}: stored p {stored!r} does not reproduce from b={favored}, "
        f"c={opposed} (exact {exact!r})"
    )


@_SKIP_NO_RUN
@pytest.mark.parametrize("member", REGISTERED_MEMBERS)
def test_every_recorded_verdict_follows_from_its_p_value(member: str) -> None:
    """The crossing flag is the comparison it claims to be, at the frozen alpha."""
    prefix = _prefix(member)
    p_value = float(_value(f"{prefix}_p"))
    assert _value(f"{prefix}_crosses_frozen_threshold") is (p_value < BONFERRONI_ALPHA)
    assert _value(f"{prefix}_discordant_n") == int(_value(f"{prefix}_favored")) + int(
        _value(f"{prefix}_opposed")
    )
    assert _value(f"{prefix}_computable") is (_value(f"{prefix}_discordant_n") > 0)


@_SKIP_NO_RUN
def test_the_recorded_holm_verdicts_reproduce_over_the_frozen_family() -> None:
    """Holm is recomputed over the frozen sixteen with the four substituted.

    Holm's threshold for one hypothesis depends on where every other p-value in
    the family fell, so a recorded Holm verdict is only checkable by rebuilding
    the whole family. Rebuilding it here is also what would catch a substitution
    that quietly dropped or duplicated a member.
    """
    from ailm.statistics import SIGNIFICANCE_COUNT_LABELS

    claims = _claims()
    substituted: dict[str, float] = {}
    for p_label, favored_label, opposed_label in SIGNIFICANCE_COUNT_LABELS:
        if p_label in REGISTERED_MEMBERS:
            prefix = _prefix(p_label)
            substituted[p_label] = exact_two_sided_binomial_p(
                int(claims[f"{prefix}_favored"]["value"]),
                int(claims[f"{prefix}_opposed"]["value"]),
            )
            continue
        substituted[p_label] = exact_two_sided_binomial_p(
            int(claims[favored_label]["value"]), int(claims[opposed_label]["value"])
        )
    assert set(substituted) == set(SIGNIFICANCE_FAMILY)

    decisions = {
        decision.label: decision for decision in holm_step_down(substituted, FAMILYWISE_ALPHA)
    }
    for member in REGISTERED_MEMBERS:
        prefix = _prefix(member)
        assert _value(f"{prefix}_holm_rejected") is decisions[member].rejected, member
        assert _value(f"{prefix}_holm_rank") == decisions[member].rank, member


# ── 3. The population really was held back ───────────────────────────────────


@pytest.mark.skipif(
    not (CONFIRMATORY_DIR / "ud_ewt_test_case_counts.json").exists(),
    reason="confirmatory artifacts absent in this checkout",
)
def test_the_confirmatory_documents_are_disjoint_from_every_development_document() -> None:
    """No document identity may appear in both populations.

    This is decision 111's fourth falsifier, and it is the one that decides
    whether the word "held-back" is true. Identities are compared directly rather
    than through counts, because two populations of the same size can still
    overlap entirely.
    """
    development = json.loads(DEVELOPMENT_CASE_ARTIFACT.read_text(encoding="utf-8"))
    confirmatory = json.loads(
        (CONFIRMATORY_DIR / "ud_ewt_test_case_counts.json").read_text(encoding="utf-8")
    )
    development_ids = set(development["documents"]["ids"])
    confirmatory_ids = set(confirmatory["documents"]["ids"])
    assert development_ids and confirmatory_ids, "an empty identity set would pass vacuously"
    overlap = sorted(development_ids & confirmatory_ids)
    assert overlap == [], (
        f"{len(overlap)} document identity/identities appear in both the development "
        f"and confirmatory populations, so the confirmatory population is not held "
        f"back: {overlap[:5]}"
    )


@pytest.mark.skipif(
    not (CONFIRMATORY_DIR / "ud_ewt_test_case_counts.json").exists(),
    reason="confirmatory artifacts absent in this checkout",
)
def test_the_disjointness_check_would_fail_on_a_planted_overlap() -> None:
    """POSITIVE CONTROL: the comparison above must be able to fail."""
    development = json.loads(DEVELOPMENT_CASE_ARTIFACT.read_text(encoding="utf-8"))
    confirmatory = json.loads(
        (CONFIRMATORY_DIR / "ud_ewt_test_case_counts.json").read_text(encoding="utf-8")
    )
    planted = set(confirmatory["documents"]["ids"]) | {development["documents"]["ids"][0]}
    assert planted & set(development["documents"]["ids"])


@pytest.mark.parametrize("level", ("case", "clause", "narrative"))
def test_each_artifact_declares_the_registered_population_and_no_stride(level: str) -> None:
    """The population rule is carried by the artifact, not asserted about it."""
    path = CONFIRMATORY_DIR / f"ud_ewt_test_{level}_counts.json"
    if not path.exists():
        pytest.skip(f"{path} absent in this checkout")
    provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
    assert provenance["population"] == "ud_ewt_test"
    assert provenance["stride_applied"] is False
    assert provenance["registration"] == "docs/decisions.md entry 111"
    assert provenance["source_url"].endswith("en_ewt-ud-test.conllu")


@pytest.mark.parametrize("filename", sorted(DOCUMENTED_DIGESTS))
def test_documented_digest_matches_the_artifact(filename: str) -> None:
    """Prose and file must agree, or the prose is telling a reader the wrong thing."""
    path = CONFIRMATORY_DIR / filename
    if not path.exists():
        pytest.skip(f"{path} absent (rebuild with {GENERATOR.name} --population ud_ewt_test)")
    assert _sha256(path) == DOCUMENTED_DIGESTS[filename], (
        f"{filename} hashes to {_sha256(path)} but its documented digest is "
        f"{DOCUMENTED_DIGESTS[filename]}. Update docs/provenance/sources.md and this "
        f"table together; do not leave prose asserting a digest the file does not have."
    )


def test_every_documented_digest_is_also_recorded_in_the_provenance_document() -> None:
    """The two records of each digest must be the same record.

    A table in a test file that nobody cross-checks against the prose is a second
    source of truth, which is the failure mode this repository has already shipped
    once for a vendored artifact.
    """
    sources = (PROJECT_ROOT / "docs" / "provenance" / "sources.md").read_text(encoding="utf-8")
    for filename, digest in DOCUMENTED_DIGESTS.items():
        assert filename in sources, f"{filename} is undocumented in docs/provenance/sources.md"
        assert digest in sources, f"{filename}'s digest is not the one the provenance doc records"


def test_swapped_digests_would_be_caught() -> None:
    """POSITIVE CONTROL: correct digests attached to the wrong filenames must fail."""
    names = sorted(DOCUMENTED_DIGESTS)
    swapped = dict(DOCUMENTED_DIGESTS)
    swapped[names[0]], swapped[names[1]] = swapped[names[1]], swapped[names[0]]
    present = [name for name in names if (CONFIRMATORY_DIR / name).exists()]
    if len(present) < 2:
        pytest.skip("fewer than two confirmatory artifacts present")
    mismatched = [name for name in present if _sha256(CONFIRMATORY_DIR / name) != swapped[name]]
    assert set(mismatched) >= {names[0], names[1]} & set(present)


# ── 4. The frozen family did not move ────────────────────────────────────────


def test_the_frozen_family_and_threshold_are_unchanged() -> None:
    """The registration is worthless if the family it names can be re-shaped."""
    assert len(SIGNIFICANCE_FAMILY) == 16
    assert len(set(SIGNIFICANCE_FAMILY)) == len(SIGNIFICANCE_FAMILY)
    assert BONFERRONI_ALPHA == FAMILYWISE_ALPHA / len(SIGNIFICANCE_FAMILY)
    scored = sorted(
        label for label, disposition in CONFIRMATORY_REPLICATION if disposition == "ud_ewt_test"
    )
    assert scored == sorted(REGISTERED_MEMBERS), (
        "the ud_ewt_test disposition no longer names the four members decision 111 "
        f"scored: {scored}"
    )


@_SKIP_NO_RUN
def test_the_recorded_protocol_constants_match_the_frozen_ones() -> None:
    """A recorded constant that drifted from the code would date the whole run."""
    assert _value("confirmatory_frozen_alpha") == BONFERRONI_ALPHA
    assert _value("confirmatory_frozen_family_size") == len(SIGNIFICANCE_FAMILY)
    assert _value("confirmatory_members_scored") == len(REGISTERED_MEMBERS)
    assert _value("confirmatory_population") == "ud_ewt_test"
    assert _value("confirmatory_registration_decision") == "111"


@_SKIP_NO_RUN
def test_no_development_population_label_was_overwritten() -> None:
    """The development results stay where they are; the confirmatory rows sit beside them.

    Decision 79 is explicit that no development-population label may be
    overwritten. The check is that every development member still exists, still
    carries its own recorder as its source, and is not owned by the confirmatory
    recorder.
    """
    claims = _claims()
    for member in REGISTERED_MEMBERS:
        entry = claims.get(member)
        assert entry is not None, f"{member} disappeared from the ledger"
        assert entry["source"] != "scripts/record_confirmatory_replication.py", (
            f"{member} is now owned by the confirmatory recorder, so a "
            f"development-population result was overwritten"
        )


# ── 5. Nothing recorded is unreachable ───────────────────────────────────────


def test_every_confirmatory_label_is_render_blocking() -> None:
    """A ledger label the render does not require can go missing unnoticed."""
    missing = sorted(set(CONFIRMATORY_REPLICATION_CLAIMS) - set(REQUIRED_CLAIMS))
    assert missing == [], f"confirmatory labels absent from REQUIRED_CLAIMS: {missing}"


@_SKIP_NO_RUN
def test_every_non_numeric_confirmatory_claim_is_type_registered() -> None:
    """A boolean or string in the ledger must be declared as one, or the render refuses.

    ``ailm.manuscript_variables._validate_required_claim_values`` walks the whole
    ledger and requires every value to be quantitative unless its label is in an
    explicit textual or boolean inventory. That is what stops a string from being
    rendered into scientific prose as though it were a measurement. This test is
    the other half: it fails if this lane records a non-numeric value that is not
    in one of those inventories, rather than letting the render discover it.
    """
    from ailm.manuscript_variables import (
        _REQUIRED_BOOLEAN_CLAIMS,
        _REQUIRED_STRING_CLAIMS,
        _REQUIRED_STRING_LIST_CLAIMS,
    )

    registered = _REQUIRED_STRING_CLAIMS | _REQUIRED_BOOLEAN_CLAIMS | _REQUIRED_STRING_LIST_CLAIMS
    claims = _claims()
    unregistered = sorted(
        label
        for label in CONFIRMATORY_REPLICATION_CLAIMS
        if not isinstance(claims[label]["value"], (int, float))
        or isinstance(claims[label]["value"], bool)
        if label not in registered
    )
    assert unregistered == [], (
        f"these confirmatory claims are not numeric and are not type-registered in "
        f"ailm.manuscript_variables: {unregistered}"
    )


@_SKIP_NO_RUN
def test_every_recorded_verdict_flag_re_derives_from_recorded_numbers() -> None:
    """NEGATIVE CONTROL on the flags: none of them may be free-standing assertions.

    Each recorded verdict is a deterministic function of counts that are also in
    the ledger. If a flag could disagree with the numbers beside it, the ledger
    would carry two versions of the same fact and prose could quote the wrong one.
    """
    claims = _claims()
    for member in REGISTERED_MEMBERS:
        prefix = _prefix(member)
        favored = int(claims[f"{prefix}_favored"]["value"])
        opposed = int(claims[f"{prefix}_opposed"]["value"])
        p_value = float(claims[f"{prefix}_p"]["value"])
        threshold = float(claims[f"{prefix}_holm_threshold"]["value"])
        assert claims[f"{prefix}_computable"]["value"] is (favored + opposed > 0), member
        assert claims[f"{prefix}_crosses_frozen_threshold"]["value"] is (
            p_value < BONFERRONI_ALPHA
        ), member
        assert claims[f"{prefix}_holm_rejected"]["value"] is (p_value < threshold), member

        development = (
            int(claims[member.replace("_p", "_b")]["value"])
            if member != "stage6_sign_test_p"
            else int(claims["stage6_sign_test_positive"]["value"]),
            int(claims[member.replace("_p", "_c")]["value"])
            if member != "stage6_sign_test_p"
            else int(claims["stage6_sign_test_negative"]["value"]),
        )
        recorded = claims[f"{prefix}_direction_agrees"]["value"]
        if favored == opposed or development[0] == development[1]:
            assert recorded == "undefined", member
        else:
            expected = (favored > opposed) == (development[0] > development[1])
            assert recorded == str(expected).lower(), member


@_SKIP_NO_RUN
def test_the_recorder_owns_exactly_the_declared_label_set() -> None:
    """What the recorder wrote and what the registry declares must be one set."""
    ledger = ClaimsLedger(LEDGER)
    owned = set(ledger.labels_owned_by("scripts/record_confirmatory_replication.py"))
    declared = set(CONFIRMATORY_REPLICATION_CLAIMS)
    assert owned == declared, (
        f"recorded but undeclared: {sorted(owned - declared)}; "
        f"declared but unrecorded: {sorted(declared - owned)}"
    )


@_SKIP_NO_RUN
def test_the_four_verdicts_carry_confirmatory_status_and_the_census_does_not() -> None:
    """Status is the audit's whole point: a census is not a prediction."""
    claims = _claims()
    for member in REGISTERED_MEMBERS:
        prefix = _prefix(member)
        for suffix in ("favored", "opposed", "p", "crosses_frozen_threshold", "holm_rejected"):
            assert claims[f"{prefix}_{suffix}"]["status"] == "confirmatory", f"{prefix}_{suffix}"
    for label in ("confirmatory_census_sentences", "confirmatory_control_positive_p"):
        assert claims[label]["status"] == "descriptive", label
    assert claims[SENTINEL]["status"] == "confirmatory"


# ── The generators really are the development ones ───────────────────────────


@pytest.mark.timeout(900)
def test_the_confirmatory_builders_reproduce_the_development_artifact() -> None:
    """Run this vendor's own builders over the development corpus and compare.

    Decision 79 requires the confirmatory population to pass through the
    *identical* generators, oracles, exclusion rules and minimum-length filters.
    Reading the two scripts side by side is not evidence of that; running the
    confirmatory builders over the development corpus and requiring them to
    reproduce the committed development artifact's sequences exactly is. The
    development artifact splits its sequences by stride and this vendor does not,
    so the comparison is against the union of the two development arms, which is
    the set the same filters admit.
    """
    pytest.importorskip("requests")
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from generate_confirmatory_ewt_counts import build_case_population

    from ailm.corpus.ud import ensure_ud_ewt_dev_path, load_ud_ewt_dev

    try:
        source = ensure_ud_ewt_dev_path(download=False)
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - checkout dependent
        pytest.skip(f"development EWT source absent, cannot replay the generator: {exc}")

    sentences = load_ud_ewt_dev(source, download=False)
    populations, _documents, _tokens = build_case_population(sentences)

    committed = json.loads(DEVELOPMENT_CASE_ARTIFACT.read_text(encoding="utf-8"))
    for name in ("gold", "derived"):
        expected = sorted(
            tuple(tuple(pair) for pair in sequence)
            for arm in ("train", "test")
            for sequence in committed["populations"][name][arm]
        )
        produced = sorted(tuple(tuple(pair) for pair in seq) for seq in populations[name])
        assert produced, f"the {name} builder produced nothing; the comparison would be vacuous"
        assert expected, f"the committed {name} population is empty; nothing is being checked"
        assert produced == expected, (
            f"the confirmatory {name} builder does not reproduce the committed "
            f"development artifact over the development corpus, so the two "
            f"populations were not built by identical rules "
            f"({len(produced)} sequences produced against {len(expected)} committed)"
        )


# ── The precheck's sentinel guard must be able to fire ───────────────────────


@pytest.mark.timeout(600)
def test_the_precheck_finds_a_sentinel_in_a_dict_shaped_ledger(tmp_path: Path) -> None:
    """REGRESSION: the guard read an empty name for every claim and could not fire.

    The ledger maps label to entry, so its labels are its keys. The precheck
    iterated it expecting a list of entry objects carrying a ``name`` field;
    iterating a dict yields keys, every key failed the ``isinstance(entry, dict)``
    check, and the guard read ``""`` for every claim. It reported "no sentinel
    found" while the sentinel sat in the ledger it had just read. A guard that
    cannot fail is worse than an absent one, because it is reported as a pass.
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import precheck_confirmatory_replication as precheck

    ledger = {"schema": "ailm-claims-v2", "claims": {SENTINEL: {"value": True}}}
    labels = list(ledger["claims"])
    assert [label for label in labels if label.startswith(precheck.SENTINEL_CLAIM_PREFIX)] == [
        SENTINEL
    ]

    result = subprocess.run(
        [str(VENV_PYTHON), str(PROJECT_ROOT / "scripts" / "precheck_confirmatory_replication.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    recorded = json.loads(LEDGER.read_text(encoding="utf-8"))["claims"].get(SENTINEL)
    if recorded is None:
        pytest.skip("the run has not been recorded, so there is no sentinel to detect")
    assert result.returncode == 1, (
        "the precheck did not refuse although the sentinel is in the ledger; its "
        f"sentinel guard is hollow again.\n{result.stdout[-1200:]}"
    )
    assert "SENTINEL PRESENT" in result.stdout


# ── The precheck's filesystem sentinel scan and its ledger guard ─────────────
# The filesystem scan must stay cheap enough for every precheck invocation, must
# never mistake a dependency or VCS marker for a scientific one, and must refuse
# readiness whenever the ledger that would prove a population unspent cannot be
# read. These controls run against real directory trees under tmp_path; none of
# them scores anything.


def _load_precheck():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import precheck_confirmatory_replication as precheck

    return precheck


def test_the_precheck_file_scan_reports_genuine_sentinels_across_project_content(
    tmp_path: Path,
) -> None:
    """A marker beside real project content is evidence wherever it sits.

    ``output/``, ``cache/``, ``handoff/`` and ordinary nested subdirectories all
    stay in scope; only the dependency and VCS trees are pruned. A scan that
    dropped any of these locations could miss exactly the marker someone leaves
    beside the artifacts it describes.
    """
    precheck = _load_precheck()
    planted = (
        "output/confirmatory_run_completed_pop_a",
        "cache/deep/confirmatory_run_completed_pop_b",
        "handoff/confirmatory_run_completed_pop_c",
        "docs/notes/confirmatory_run_completed_pop_d",
    )
    for rel in planted:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("ledger snapshot", encoding="utf-8")

    hits, unreadable = precheck._find_file_sentinels(tmp_path)

    assert unreadable == []
    assert sorted(hits) == sorted(planted)


def test_the_precheck_file_scan_ignores_dependency_and_vcs_markers(tmp_path: Path) -> None:
    """A marker under ``.venv`` or ``.git``, at any depth, is not a sentinel.

    The old ``REPO_ROOT.glob("**/...")`` walked both trees wholesale and then
    filtered ``.git`` paths afterwards -- seconds of traversal per invocation,
    with ``.venv`` contents left free to masquerade. Pruning before descending
    removes both problems at once.
    """
    precheck = _load_precheck()
    for rel in (
        ".venv/release-cache/x/confirmatory_run_completed_decoy1",
        ".git/objects/ab/confirmatory_run_completed_decoy2",
        "nested/pkg/.venv/confirmatory_run_completed_decoy3",
        "nested/repo/.git/confirmatory_run_completed_decoy4",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("", encoding="utf-8")

    hits, unreadable = precheck._find_file_sentinels(tmp_path)

    assert unreadable == []
    assert hits == []


def test_the_precheck_fails_closed_on_a_missing_ledger(tmp_path: Path, capsys) -> None:
    """Without the ledger the guard cannot certify readiness, so it must refuse.

    A guard that reports "no sentinel found" over a ledger it never read would
    certify exactly the thing it exists to deny.
    """
    precheck = _load_precheck()

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "LEDGER GUARD INCONCLUSIVE" in out
    assert "PROTOCOL FROZEN" not in out


def test_the_precheck_fails_closed_on_a_malformed_ledger(tmp_path: Path, capsys) -> None:
    """Unparseable ledger bytes are a refusal, not a silent pass."""
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text("{not json", encoding="utf-8")

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "LEDGER GUARD INCONCLUSIVE" in out
    assert "could not parse" in out
    assert "PROTOCOL FROZEN" not in out


def test_the_precheck_fails_closed_on_an_unexpectedly_shaped_ledger(tmp_path: Path, capsys) -> None:
    """A claims section that is neither mapping nor accepted list export refuses.

    The older list-of-entries shape stays accepted; anything else means the
    guard does not know what it is reading, and an unreadable guard may not
    certify readiness.
    """
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps({"schema": "ailm-claims-v2", "claims": "a string"}), encoding="utf-8"
    )

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "unexpected shape str" in out
    assert "PROTOCOL FROZEN" not in out


def test_the_precheck_certifies_readiness_on_a_clean_ledger_and_tree(
    tmp_path: Path, capsys
) -> None:
    """The one shape that may print PROTOCOL FROZEN: readable sentinel-free ledger.

    The census gaps against an empty tmp tree must not flip the exit code --
    the documented distinction between missing optional population artifacts
    and an unreadable sentinel ledger.
    """
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps({"schema": "ailm-claims-v2", "claims": {}}), encoding="utf-8"
    )

    assert precheck.main(repo_root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "PROTOCOL FROZEN" in out
    assert "SENTINEL PRESENT" not in out
    assert "LEDGER GUARD INCONCLUSIVE" not in out
    assert "DATA GAPS REMAIN" in out


def test_the_precheck_refuses_on_a_genuine_file_sentinel_alone(tmp_path: Path, capsys) -> None:
    """A marker on disk blocks the run even beside a readable sentinel-free ledger.

    The ledger records what the recorder wrote; the filesystem scan catches
    markers left by any other hand. Either evidence alone must refuse.
    """
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps({"schema": "ailm-claims-v2", "claims": {}}), encoding="utf-8"
    )
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "confirmatory_run_completed_ud_ewt_test").write_text(
        "", encoding="utf-8"
    )

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "SENTINEL PRESENT: output/confirmatory_run_completed_ud_ewt_test" in out
    assert "PROTOCOL FROZEN" not in out


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="an unreadable-directory control needs an unprivileged POSIX user",
)
def test_the_precheck_file_scan_fails_closed_on_an_unreadable_directory(
    tmp_path: Path,
) -> None:
    """A directory the scan cannot list makes the scan incomplete; incomplete fails.

    A scan that silently treated an unscannable directory as empty would let a
    sentinel hide behind a filesystem permission, and then certify readiness
    over the hole.
    """
    precheck = _load_precheck()
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "confirmatory_run_completed_hidden").write_text("", encoding="utf-8")
    vault.chmod(0o000)
    try:
        hits, unreadable = precheck._find_file_sentinels(tmp_path)
    finally:
        vault.chmod(0o755)

    assert unreadable, "an unscannable directory was silently treated as empty"
    assert hits == []


# ── Ledger list exports must be well-formed to count as readable ─────────────
# An older ledger export carries a list of entry objects. Entries that are not
# objects, or that carry no usable name/label, mean the guard does not know
# what it is reading; a list of plain strings used to return no labels and no
# problem -- a silent pass over an unreadable guard.


def test_the_precheck_rejects_a_list_export_of_plain_strings(tmp_path: Path, capsys) -> None:
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps(
            {
                "schema": "ailm-claims-v2",
                "claims": ["confirmatory_run_completed_ud_ewt_test", "another_claim"],
            }
        ),
        encoding="utf-8",
    )

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "LEDGER GUARD INCONCLUSIVE" in out
    assert "not an entry object" in out
    assert "PROTOCOL FROZEN" not in out


def test_the_precheck_accepts_list_exports_with_usable_name_or_label(
    tmp_path: Path, capsys
) -> None:
    """The supported list-export shapes still work -- name and label fields alike."""
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps(
            {
                "claims": [
                    {"name": "some_development_claim", "value": 1},
                    {"label": "confirmatory_run_completed_ud_ewt_test"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "LEDGER GUARD INCONCLUSIVE" not in out
    assert "SENTINEL PRESENT: confirmatory_run_completed_ud_ewt_test" in out


def test_the_precheck_rejects_malformed_list_entry_fields(tmp_path: Path, capsys) -> None:
    """A non-string or absent name/label field is a problem, not a silent skip."""
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps({"claims": [{"name": 123}, {"value": True}]}),
        encoding="utf-8",
    )

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "carries no usable string name or label" in out
    assert "PROTOCOL FROZEN" not in out


def test_the_precheck_reports_mapping_drift_instead_of_crashing() -> None:
    """A label enumerated by decision 79 but absent from the code mapping renders.

    The check-4 loop indexed ``POPULATION_SOURCES["?"]`` and crashed with a
    KeyError, burying the collected MISMATCH verdict in a traceback. The row
    helper reports the drift; legitimate dispositions render their registered
    population source unchanged.
    """
    precheck = _load_precheck()

    drifted = precheck._member_mapping_row(
        "stage4_derived_mcnemar_p", {}, precheck.POPULATION_SOURCES, precheck.FROZEN_TESTS
    )
    assert any("disposition: ?" in line for line in drifted)
    assert any("mapping has drifted" in line for line in drifted)

    code_map = dict(CONFIRMATORY_REPLICATION)
    for label, disposition in code_map.items():
        row = precheck._member_mapping_row(
            label, code_map, precheck.POPULATION_SOURCES, precheck.FROZEN_TESTS
        )
        rendered = "\n".join(row)
        assert f"disposition: {disposition}" in rendered
        assert precheck.POPULATION_SOURCES[disposition] in rendered
        assert not any("mapping has drifted" in line for line in row)


def test_the_alpha_drift_diagnostic_enforces_the_exact_frozen_contract() -> None:
    """POSITIVE/NEGATIVE CONTROL -- the single contract check, both directions.

    The frozen contract is exact equality with the familywise alpha over the
    frozen family size, so the old magnitude sanity checks are subsumed: a
    half-corrected ``FAMILYWISE_ALPHA / 2`` passed both of them, and the
    algebraically-``>= 0.05`` artifact could never fire at all. Every drifted
    candidate here must read as a problem; the real frozen constants must not.
    """
    precheck = _load_precheck()

    assert (
        precheck._alpha_drift_problem(BONFERRONI_ALPHA, FAMILYWISE_ALPHA, len(SIGNIFICANCE_FAMILY))
        is None
    )
    for drifted_alpha in (
        0.003125,  # the auditor's stale 0.05/16 misreading of the contract
        FAMILYWISE_ALPHA,  # an uncorrected familywise alpha
        FAMILYWISE_ALPHA / 2,  # a half-corrected divisor both old checks accepted
        0.05 / 16,  # the other historical misreading of the contract
        BONFERRONI_ALPHA * 10,  # a silently shrunk family size
    ):
        problem = precheck._alpha_drift_problem(
            drifted_alpha, FAMILYWISE_ALPHA, len(SIGNIFICANCE_FAMILY)
        )
        assert problem is not None, f"{drifted_alpha!r} passed the frozen-contract check"
        assert "familywise alpha over the frozen family size" in problem


# ── Dual-alias list entries must not be silently resolved by precedence ──────
# An entry carrying both ``name`` and ``label`` used to be read by name
# precedence alone, so a sentinel recorded only in ``label`` was dropped and
# the guard certified readiness over it. Identical aliases still name one
# claim; conflicting aliases make the read undecidable, so the entry is
# refused -- fail-closed, not a documented miss.


def test_the_precheck_refuses_conflicting_dual_aliases(tmp_path: Path, capsys) -> None:
    """The exact reproduced miss: sentinel only in label, name present and divergent.

    Name precedence used to yield ``['stage1_x']`` with no problem and a
    PROTOCOL FROZEN exit 0 while the sentinel sat in the label field. The
    refusal must name the undecidable entry, including the alias that carries
    the sentinel, so nothing is silently dropped.
    """
    precheck = _load_precheck()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "claims.json").write_text(
        json.dumps(
            {"claims": [{"name": "stage1_x", "label": "confirmatory_run_completed_ud_ewt_test"}]}
        ),
        encoding="utf-8",
    )

    assert precheck.main(repo_root=tmp_path) == 1
    out = capsys.readouterr().out
    assert "LEDGER GUARD INCONCLUSIVE" in out
    assert "conflicting aliases" in out
    assert "confirmatory_run_completed_ud_ewt_test" in out
    assert "PROTOCOL FROZEN" not in out


def test_the_precheck_accepts_compatible_dual_aliases(tmp_path: Path, capsys) -> None:
    """POSITIVE CONTROL -- identical name/label aliases still name one claim.

    The refusal above must not evict the supported compatible form: an entry
    whose two fields agree is read normally, with or without a sentinel.
    """
    precheck = _load_precheck()
    clean_dir = tmp_path / "clean"
    spent_dir = tmp_path / "spent"
    for directory, label_value in ((clean_dir, "ordinary_claim"), (spent_dir, SENTINEL)):
        (directory / "data").mkdir(parents=True)
        (directory / "data" / "claims.json").write_text(
            json.dumps({"claims": [{"name": label_value, "label": label_value}]}),
            encoding="utf-8",
        )

    assert precheck.main(repo_root=clean_dir) == 0
    clean_out = capsys.readouterr().out
    assert "PROTOCOL FROZEN" in clean_out
    assert "LEDGER GUARD INCONCLUSIVE" not in clean_out

    assert precheck.main(repo_root=spent_dir) == 1
    spent_out = capsys.readouterr().out
    assert f"SENTINEL PRESENT: {SENTINEL}" in spent_out
    assert "PROTOCOL FROZEN" not in spent_out


# ── The production boundary: where a confirmatory run may write ─────────────
# ``AILM_CLAIMS_PATH`` alone was an accidental bypass: a fresh side ledger
# holds no sentinel, so a run over the production corpus could rescore the
# spent population without the precheck's second channel ever seeing it. The
# boundary is structural: a sanctioned shakedown isolates the corpus (fixture
# root with genuinely synthetic data), the ledger, and the spent marker
# together, requires an explicit opt-in, and refuses every partial or
# identity-violating combination before anything is read.


def _load_recorder():
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import record_confirmatory_replication as recorder

    return recorder


def _run_recorder_main(monkeypatch, recorder, root, ledger, *, opt_in="1"):
    import contextlib
    import io

    production_before = LEDGER.read_bytes()
    if root is None:
        monkeypatch.delenv(recorder.CORPUS_ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(recorder.CORPUS_ROOT_ENV, str(root))
    if ledger is None:
        monkeypatch.delenv(recorder.LEDGER_ENV, raising=False)
    else:
        monkeypatch.setenv(recorder.LEDGER_ENV, str(ledger))
    if opt_in is None:
        monkeypatch.delenv(recorder.OPT_IN_ENV, raising=False)
    else:
        monkeypatch.setenv(recorder.OPT_IN_ENV, opt_in)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            # Empty argv: in-process argparse would otherwise parse pytest's
            # own command line and exit with its usage code.
            code = recorder.main([])
        return code, out.getvalue(), None
    except SystemExit as exit_signal:
        return 1, out.getvalue(), str(exit_signal)
    finally:
        assert LEDGER.read_bytes() == production_before, (
            "a refused recorder run modified the production ledger"
        )


def test_an_alternate_ledger_over_the_production_corpus_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """A side ledger with no fixture root would rescore the spent population.

    This is the audited silent-bypass shape: the alternate ledger shows no
    sentinel, so only the production corpus boundary stops it, and no opt-in
    authorizes that combination.
    """
    recorder = _load_recorder()
    code, out, refusal = _run_recorder_main(
        monkeypatch, recorder, root=None, ledger=tmp_path / "side.json", opt_in="1"
    )
    assert code == 1 and refusal is not None
    assert "re-score the production confirmatory corpus" in refusal
    assert "no opt-in" in refusal
    assert not (LEDGER.parent / SENTINEL).is_file(), "a marker appeared in the production tree"


def test_a_fixture_corpus_without_an_isolated_ledger_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """A fixture corpus against the production ledger writes spent markers into it."""
    recorder = _load_recorder()
    fixture_root = tmp_path / "fixture_root"
    _synthetic_miniature_corpus(fixture_root / "data" / "confirmatory")
    code, out, refusal = _run_recorder_main(
        monkeypatch, recorder, root=fixture_root, ledger=None, opt_in="1"
    )
    assert code == 1 and refusal is not None
    assert "into the production ledger" in refusal


def test_fixture_mode_requires_the_explicit_opt_in(tmp_path: Path, monkeypatch) -> None:
    """Both overrides without the opt-in refuse, so isolation never starts by accident."""
    recorder = _load_recorder()
    fixture_root = tmp_path / "fixture_root"
    _synthetic_miniature_corpus(fixture_root / "data" / "confirmatory")
    code, out, refusal = _run_recorder_main(
        monkeypatch,
        recorder,
        root=fixture_root,
        ledger=tmp_path / "claims.json",
        opt_in=None,
    )
    assert code == 1 and refusal is not None
    assert "explicit opt-in" in refusal


def test_overrides_resolving_to_the_production_tree_are_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """Names are not isolation: resolved identities are what the guard compares.

    An override pointing at the production root itself, at a directory inside
    it, or a ledger placed beside the production one, all resolve into the
    tree the sentinel protects and must refuse before anything is read.
    """
    recorder = _load_recorder()
    code, out, refusal = _run_recorder_main(
        monkeypatch,
        recorder,
        root=PROJECT_ROOT,
        ledger=tmp_path / "claims.json",
        opt_in="1",
    )
    assert code == 1 and refusal is not None
    assert "resolves to or inside the production tree" in refusal

    code, out, refusal = _run_recorder_main(
        monkeypatch,
        recorder,
        root=PROJECT_ROOT / "output",
        ledger=tmp_path / "claims.json",
        opt_in="1",
    )
    assert code == 1 and refusal is not None
    assert "resolves to or inside the production tree" in refusal

    fixture_root = tmp_path / "fixture_root"
    _synthetic_miniature_corpus(fixture_root / "data" / "confirmatory")
    production_ledger_sibling = PROJECT_ROOT / "data" / "side_ledger.json"
    code, out, refusal = _run_recorder_main(
        monkeypatch,
        recorder,
        root=fixture_root,
        ledger=production_ledger_sibling,
        opt_in="1",
    )
    assert code == 1 and refusal is not None
    assert "fixture ledger must live outside it" in refusal
    assert not production_ledger_sibling.exists()


def test_a_copied_production_corpus_is_not_a_safe_fixture(tmp_path: Path, monkeypatch) -> None:
    """A byte-copy of the spent confirmatory artifacts is still the held-back data.

    Renaming is not isolation: the recorder compares content digests against
    the production corpus and refuses the copy before the sentinel is written.
    """
    recorder = _load_recorder()

    fixture_root = tmp_path / "fixture_root"
    (fixture_root / "data").mkdir(parents=True)
    shutil.copytree(CONFIRMATORY_DIR, fixture_root / "data" / "confirmatory")
    code, out, refusal = _run_recorder_main(
        monkeypatch,
        recorder,
        root=fixture_root,
        ledger=tmp_path / "claims.json",
        opt_in="1",
    )
    assert code == 1 and refusal is not None
    assert "byte-identical" in refusal
    claims_path = tmp_path / "claims.json"
    claims = json.loads(claims_path.read_text())["claims"] if claims_path.exists() else {}
    assert SENTINEL not in claims
    assert not (claims_path.parent / SENTINEL).is_file()


def test_preflight_failure_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    """Deterministic non-scoring validation runs before the sentinel consumes.

    An absent confirmatory artifact refuses in the preflight, so the ledger
    shows no sentinel and no marker exists: the population is not spent by a
    run that never scored, and recovery does not require deleting a sentinel
    decision 111 forbids deleting.
    """
    recorder = _load_recorder()
    fixture_root = tmp_path / "fixture_root"
    (fixture_root / "data" / "confirmatory").mkdir(parents=True)
    ledger = tmp_path / "claims.json"
    code, out, refusal = _run_recorder_main(monkeypatch, recorder, root=fixture_root, ledger=ledger)
    assert code == 1 and refusal is not None
    assert "is absent" in refusal
    claims = json.loads(ledger.read_text())["claims"] if ledger.exists() else {}
    assert SENTINEL not in claims
    assert not (ledger.parent / SENTINEL).is_file()


def test_an_existing_spent_marker_refuses_before_preflight(tmp_path: Path, monkeypatch) -> None:
    """The filesystem half of the sentinel stops a run by itself.

    A ledger that shows no sentinel — rebuilt, cleared, or never written —
    must still refuse when the marker beside it says the population was
    consumed; the two channels are independent evidence.
    """
    recorder = _load_recorder()
    fixture_root = tmp_path / "fixture_root"
    (fixture_root / "data" / "confirmatory").mkdir(parents=True)
    ledger = tmp_path / "claims.json"
    (tmp_path / SENTINEL).write_text("{}\n", encoding="utf-8")
    code, out, refusal = _run_recorder_main(monkeypatch, recorder, root=fixture_root, ledger=ledger)
    assert code == 1 and refusal is not None
    assert "filesystem half" in refusal and "already exists" in refusal


def test_the_runtime_disjointness_guard_covers_the_development_heldout_arm() -> None:
    """Falsifier 4 refuses on any development intersection, train or held-out.

    The runtime guard intersected the confirmatory ids against the development
    TRAIN ids only, so a collision confined to the development held-out arm
    would spend the population on a frame whose held-back claim is false. The
    test suite already compared the full development id set; the guard now
    does the same, and a train collision refuses exactly as before.
    """
    recorder = _load_recorder()
    development_ids = ["dev_a", "dev_b", "dev_c", "dev_d"]
    train_indices = [0, 1]

    merged, index_of, n_train = recorder._merged_documents(
        development_ids, train_indices, ["conf_1", "conf_2"]
    )
    assert merged == ["dev_a", "dev_b", "conf_1", "conf_2"]
    assert n_train == 2 and index_of["conf_1"] == 2

    for planted, _arm in ((["dev_b"], "train"), (["conf_1", "dev_d"], "held-out")):
        with pytest.raises(SystemExit, match="falsifier 4"):
            recorder._merged_documents(development_ids, train_indices, planted)


def test_the_status_writer_reclassifies_direction_agreement_markers() -> None:
    """S7: the direction-agrees markers are comparisons, not pre-registered outcomes.

    They rode the confirmatory batch while decision 111 registers only the
    four verdicts and their sufficient counts as confirmatory; the census and
    the controls are descriptive, and a post-hoc direction comparison belongs
    with them. The authoritative writer now classifies them descriptive; the
    production ledger keeps its recorded status until the final executor's
    regeneration (tests/test_claims_status.py carries the explicit pending
    map for that interim state).
    """
    scripts = str(PROJECT_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import assign_claim_status as status_writer

    for label in (
        "confirmatory_stage4_derived_mcnemar_direction_agrees",
        "confirmatory_stage4_gold_mcnemar_direction_agrees",
        "confirmatory_stage5_mcnemar_direction_agrees",
        "confirmatory_stage6_sign_test_direction_agrees",
    ):
        assert (
            status_writer.status_for(label, "scripts/record_confirmatory_replication.py")
            == "descriptive"
        )
    assert not status_writer.DESCRIPTIVE_LABELS & status_writer.ANALYTIC_LABELS
    assert not status_writer.DESCRIPTIVE_LABELS & status_writer.EXPLORATORY_LABELS


# ── Exactly-once consumption under a real process race and a real crash ──────
# An existing marker or a source-order argument is not proof that a job
# completed. These controls run the shipped entry point as real OS processes
# against isolated fixtures, and inspect actual process exits, pids and
# timings.


def _fixture_ledger_and_environment(
    tmp_path: Path, *, variant: str
) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / f"fixture_root_{variant}"
    _synthetic_miniature_corpus(root / "data" / "confirmatory", variant=variant)
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload["claims"] = {
        label: entry
        for label, entry in payload["claims"].items()
        if not label.startswith("confirmatory_")
    }
    ledger = root / "data" / "claims.json"
    ledger.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    environment = {
        **os.environ,
        "AILM_CLAIMS_PATH": str(ledger.resolve()),
        "AILM_PROJECT_ROOT": str(root.resolve()),
        "AILM_CONFIRMATORY_ALTERNATE_LEDGER": "1",
    }
    return ledger, environment, ledger.parent / SENTINEL


def _outcome_recorded(ledger: Path) -> bool:
    claims = json.loads(ledger.read_text(encoding="utf-8"))["claims"]
    return "confirmatory_stage4_derived_mcnemar_favored" in claims


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project interpreter absent")
@pytest.mark.timeout(900)
def test_simultaneous_attempts_consume_exactly_once(tmp_path: Path) -> None:
    """Two real simultaneous attempts: one consumes, one refuses, at real exits.

    Both processes are started back to back against the same fixture ledger,
    corpus and opt-in. The outer CLI lock is acquired before mode validation,
    so the loser either contends on that lock — its refusal names the
    winner's pid, which is the cross-check that the exclusion mechanism is
    the single-writer lock — or, if it starts after the winner released,
    refuses on the sentinel. Either way exactly one process may consume, and
    the assertion inspects real process exits rather than marker presence.
    """
    import time

    ledger, environment, marker = _fixture_ledger_and_environment(tmp_path, variant="standard")
    started = time.monotonic()
    first = subprocess.Popen(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    second = subprocess.Popen(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    first_out, first_err = first.communicate()
    second_out, second_err = second.communicate()
    elapsed = time.monotonic() - started

    outcomes = sorted(
        [
            (first.pid, first.returncode, first_out, first_err),
            (second.pid, second.returncode, second_out, second_err),
        ],
        key=lambda item: item[1],
    )
    # Ascending returncode: the completing process sorts first.
    (
        (winner_pid, winner_code, winner_out, winner_err),
        (loser_pid, loser_code, loser_out, loser_err),
    ) = outcomes
    assert winner_code == 0, (
        f"no process completed a permitted fixture run; exits "
        f"{first.returncode}/{second.returncode}.\n{first_out[-600:]}\n{second_out[-600:]}"
    )
    assert loser_code != 0, "both processes completed: the population was consumed twice"
    loser_text = loser_out + loser_err
    assert "REFUSED" in loser_text or "recorder refused" in loser_text, loser_text[-800:]
    if "another recorder run (pid" in loser_text:
        # The exclusion mechanism was the outer CLI lock; its refusal names
        # the holder's pid, which must be the process that completed.
        named_pid = int(loser_text.split("another recorder run (pid ")[1].split(")")[0])
        assert named_pid == winner_pid, (named_pid, winner_pid)
        mechanism = "single-writer lock"
    else:
        assert "already exists in the ledger" in loser_text
        mechanism = "ledger sentinel"
    assert "sentinel written" in winner_out
    assert _outcome_recorded(ledger), "the winning process did not record the outcomes"
    assert marker.is_file()
    print(
        f"race: winner pid {winner_pid} exit 0, loser pid {loser_pid} exit "
        f"{loser_code}, mechanism={mechanism}, wall={elapsed:.2f}s"
    )


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project interpreter absent")
@pytest.mark.timeout(900)
def test_an_interrupted_run_leaves_reruns_refused(tmp_path: Path) -> None:
    """Consumption followed by a mid-run refusal leaves reruns refused.

    The underpowered corpus passes the preflight and consumes the sentinel
    and marker, then refuses at falsifier 5 (the planted-effect control
    cannot reach the frozen threshold on this few sequences) — the exact
    state a crash after consumption leaves. A second run must refuse on the
    sentinel; the population stays spent and no outcome rows exist.
    """
    ledger, environment, marker = _fixture_ledger_and_environment(tmp_path, variant="underpowered")
    first = subprocess.run(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert first.returncode != 0, "the underpowered run should have refused at falsifier 5"
    assert "falsifier 5" in first.stdout + first.stderr
    assert json.loads(ledger.read_text(encoding="utf-8"))["claims"].get(SENTINEL) is not None
    assert marker.is_file(), "consumption happened but the marker is missing"
    assert not _outcome_recorded(ledger), "a refused run recorded outcome rows"

    second = subprocess.run(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert second.returncode != 0
    assert "already exists in the ledger" in second.stdout + second.stderr
    assert not _outcome_recorded(ledger)


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project interpreter absent")
@pytest.mark.timeout(900)
def test_a_killed_run_leaves_reruns_refused(tmp_path: Path) -> None:
    """A SIGKILL after consumption leaves reruns refused, and the lock self-heals.

    The child is killed the moment the spent marker appears — after the
    sentinel claim, mid-scoring. The killed holder's pid stays in the lock
    file; the rerun must reclaim that stale lock (dead holder), then refuse
    on the sentinel. Process identities and exits are inspected, not assumed.
    """
    import time

    from ailm.ledger_lock import LOCK_FILENAME

    ledger, environment, marker = _fixture_ledger_and_environment(tmp_path, variant="standard")
    lock_path = ledger.parent / LOCK_FILENAME
    killed = subprocess.Popen(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    deadline = time.monotonic() + 120
    while not marker.exists() and time.monotonic() < deadline and killed.poll() is None:
        time.sleep(0.002)
    assert marker.exists(), "the run never reached consumption"
    killed.kill()
    killed_out, killed_err = killed.communicate()
    assert killed.returncode != 0 and killed.poll() is not None, (
        f"the killed process exited cleanly ({killed.returncode}); the crash "
        "did not land after consumption"
    )
    assert json.loads(ledger.read_text(encoding="utf-8"))["claims"].get(SENTINEL) is not None
    assert not _outcome_recorded(ledger), "a killed run recorded outcome rows"
    if lock_path.exists():
        holder = int(lock_path.read_text().split()[0])
        assert holder == killed.pid, (holder, killed.pid)

    rerun = subprocess.run(
        [str(VENV_PYTHON), str(RECORDER)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert rerun.returncode != 0, "a rerun after a killed run completed: double consumption"
    rerun_text = rerun.stdout + rerun.stderr
    assert "already exists in the ledger" in rerun_text, rerun_text[-800:]
    assert not _outcome_recorded(ledger)


def test_the_cli_lock_is_keyed_to_the_ledger_actually_used(monkeypatch) -> None:
    """The lock identity comes from the ledger resolution, not the root argument.

    ``ledger_lock_path`` receives the production root, but it resolves
    ``AILM_CLAIMS_PATH`` itself: in fixture mode the lock is the fixture
    ledger's neighbor, in production mode it is the production ledger's
    neighbor. Pinning that here keeps the race test's mechanism claim honest.
    """
    from ailm.ledger_lock import LOCK_FILENAME, ledger_lock_path

    fixture_ledger = Path("/tmp/fixture_root/data/claims.json")
    monkeypatch.setenv("AILM_CLAIMS_PATH", str(fixture_ledger))
    assert ledger_lock_path(PROJECT_ROOT) == fixture_ledger.parent / LOCK_FILENAME, (
        "fixture-mode lock is not keyed to the fixture ledger"
    )
    monkeypatch.delenv("AILM_CLAIMS_PATH", raising=False)
    assert ledger_lock_path(PROJECT_ROOT) == PROJECT_ROOT / "data" / LOCK_FILENAME, (
        "production-mode lock is not keyed to the production ledger"
    )
