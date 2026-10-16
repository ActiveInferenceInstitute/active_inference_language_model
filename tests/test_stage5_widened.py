"""The widened-observation clause arm: artifact, alphabet, controls, ledger.

What this file proves
---------------------
Decision 112 registers a second Level 5 arm over an alphabet widened by one
distinguished symbol per ``VERB`` and per ``AUX`` token. This suite checks the
five properties that arm's numbers rest on, each by recomputation rather than by
assertion in prose:

1. **The artifact reproduces.** ``scripts/generate_clause_widened_counts.py``
   rebuilds the committed bytes exactly. A checksum on a non-deterministic
   artifact means nothing.
2. **One alphabet, not two.** The artifact's symbol inventory is
   ``ailm.levels.widened``'s, and the vendored sequences are what the shipped
   extractor produces from the corpus. The alphabet was written down three
   times before that module collected it; this is what keeps it collected.
3. **The disjointness assertion can fail.** A planted lemma leak and a planted
   punctuation emitter must both be rejected. An assertion that cannot reject
   is a comment, and this one guards the arm's whole claim to be measuring
   something other than its own labels.
4. **The negative controls fire.** Swapping the two types' fitted chains must
   cost accuracy, and a cross-type-leak transition must break the
   decoder-equivalence comparison. The recorder refuses to write if either
   fails; these tests confirm the refusals are reachable.
5. **The ledger reproduces from its own sufficient statistics.** The recorded
   exact p must be recomputable from the recorded discordant counts.

Scope discipline
----------------
Nothing here re-runs the bootstrap or re-derives an interval. The recorder owns
those; this file owns the properties that make them worth reading. And nothing
here touches the shipped Level 5 — ``tests/test_stage5_gate.py`` is unchanged,
and the shipped arm's recorded verdict stands whatever this arm found.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.gates import (
    STAGE5W_REGISTERED_GAIN_INTERVAL_NATS,
    GateVerdict,
    stage5_widened_auc_above_registered_comparators,
    stage5_widened_passing_resolved_is_consistent,
    stage5_widened_registered_gain_in_interval,
)
from ailm.levels.clause import (
    CLAUSE_TYPES,
    build_joint_emission,
    build_joint_start,
    build_joint_transition,
    build_type_free_chain,
    build_type_models,
    classify_sentences,
    sequence_log_evidence,
    type_conditional_perplexities,
)
from ailm.levels.widened import (
    AUX_SYMBOL,
    MIN_SEQUENCE_LEN,
    N_WIDENED_JOINT,
    N_WIDENED_SYMBOLS,
    VERB_SYMBOL,
    WIDENED_ARTIFACT_SCHEMA,
    WIDENED_DECODER_TOLERANCE,
    WIDENED_LEAK_MASS,
    WIDENED_SYMBOL_LABELS,
    WIDENED_SYMBOL_TO_IDX,
    WidenedClauseArtifact,
    assert_widened_disjointness,
    build_widened_agent,
    load_widened_clause_artifact,
    widened_symbols,
)
from ailm.statistics import exact_two_sided_binomial_p

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = PROJECT_ROOT / "data" / "widened" / "ud_ewt_clause_widened_counts.json"
GENERATOR = PROJECT_ROOT / "scripts" / "generate_clause_widened_counts.py"


def _load_generator():
    """Import the generator by path — ``scripts/`` is not a package."""
    import sys

    for extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location("ailm_generate_clause_widened", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.is_file():
        pytest.skip(f"{ARTIFACT_PATH} absent; run {GENERATOR.name}")
    return load_widened_clause_artifact()


@pytest.fixture(scope="module")
def models(artifact):
    return build_type_models(artifact.train, n_symbols=N_WIDENED_SYMBOLS)


@pytest.fixture(scope="module")
def ledger():
    return load_ledger(PROJECT_ROOT)


def _claim(ledger, label):
    entry = ledger.get(label)
    assert entry is not None, f"{label} is absent from the ledger; run the widened recorder"
    return entry["value"] if isinstance(entry, dict) else entry


# ── 1. THE ARTIFACT REPRODUCES ───────────────────────────────────────────────


def test_the_committed_artifact_regenerates_byte_identically(tmp_path):
    """Determinism, checked against the committed bytes rather than asserted."""
    if not ARTIFACT_PATH.is_file():
        pytest.skip(f"{ARTIFACT_PATH} absent; run {GENERATOR.name}")
    module = _load_generator()
    try:
        rebuilt = module.build_artifact()
    except FileNotFoundError:  # pragma: no cover - raw corpus not vendored here
        pytest.skip("the pinned UD EWT raw source is not present in this checkout")

    destination = Path(tmp_path).resolve() / "regenerated.json"
    from _artifact_io import atomic_write_json  # noqa: PLC0415 - loaded with the generator

    atomic_write_json(destination, rebuilt, ensure_ascii=True, sort_keys=False, indent=1)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == (
        hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest()
    ), "the generator no longer reproduces the committed artifact byte for byte"


def test_the_documented_digest_matches_the_committed_artifact():
    """The digest in the provenance document must be this file's digest.

    ``tests/test_provenance_checksums.py`` binds each digest it checks to a
    ``vendor/<filename>`` marker, and this artifact deliberately does not live
    in ``vendor/`` — that directory is the wheel's packaged surface, which this
    arm is not part of. The guarantee is not dropped for that reason, it is
    enforced here instead: a stale digest is worse than an absent one, because
    it tells a reader their correct file is corrupt.
    """
    if not ARTIFACT_PATH.is_file():
        pytest.skip(f"{ARTIFACT_PATH} absent; run {GENERATOR.name}")
    doc = (PROJECT_ROOT / "docs" / "provenance" / "sources.md").read_text(encoding="utf-8")
    marker = "**Artifact sha256 (`data/widened/ud_ewt_clause_widened_counts.json`):**"
    assert marker in doc, f"{marker} is absent from the provenance document"
    tail = doc[doc.index(marker) + len(marker) :]
    documented = re.search(r"`([0-9a-f]{64})`", tail)
    assert documented is not None, "no sha256 follows the widened artifact's marker"
    assert documented.group(1) == hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest(), (
        "the documented digest and the committed artifact disagree; regenerate "
        "the artifact or correct docs/provenance/sources.md"
    )


def test_the_artifact_is_not_in_the_packaged_vendor_surface():
    """NEGATIVE CONTROL on where this arm lives.

    ``vendor/`` is the wheel's packaged runtime surface and
    ``tests/test_wheel_resources.py`` holds ``RUNTIME_VENDOR_ARTIFACTS`` equal
    to every committed ``vendor/*.json``. Putting this arm there turned that
    invariant red, which is exactly what it is for. This guard states the
    general class rather than naming the one file that was moved: no artifact
    of this arm may reappear under ``vendor/``.
    """
    strays = sorted(p.name for p in (PROJECT_ROOT / "vendor").glob("*widened*"))
    assert strays == [], (
        f"widened-arm artifacts found in the packaged vendor surface: {strays}. "
        f"They belong under data/widened/, which no library code imports."
    )


def test_the_artifact_declares_the_widened_schema_and_partition(artifact):
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == WIDENED_ARTIFACT_SCHEMA
    assert artifact.provenance["alphabet"] == "widened"
    assert artifact.provenance["min_sequence_len"] == MIN_SEQUENCE_LEN
    # Decision 85: the arms are document-disjoint, not merely sentence-disjoint.
    assert not set(artifact.train_split_document_indices) & set(
        artifact.test_split_document_indices
    )
    assert set(artifact.train_sequence_document_indices) <= set(
        artifact.train_split_document_indices
    )
    assert set(artifact.test_sequence_document_indices) <= set(artifact.test_split_document_indices)


# ── 2. ONE ALPHABET, NOT TWO ─────────────────────────────────────────────────


def test_the_artifact_alphabet_is_the_modules_alphabet(artifact):
    """The vendored symbol names and the module's inventory are one list."""
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert tuple(payload["provenance"]["alphabet_symbols"]) == WIDENED_SYMBOL_LABELS
    assert len(WIDENED_SYMBOL_LABELS) == N_WIDENED_SYMBOLS
    assert WIDENED_SYMBOL_LABELS[VERB_SYMBOL] == "VERB"
    assert WIDENED_SYMBOL_LABELS[AUX_SYMBOL] == "AUX"
    assert WIDENED_SYMBOL_TO_IDX["VERB"] == VERB_SYMBOL
    assert WIDENED_SYMBOL_TO_IDX["AUX"] == AUX_SYMBOL

    observed = {s for _t, sequence in artifact.train + artifact.test for s in sequence}
    assert observed <= set(range(N_WIDENED_SYMBOLS))
    # The widening is not decorative in the data: both added symbols occur.
    assert {VERB_SYMBOL, AUX_SYMBOL} <= observed


def test_a_symbol_outside_the_alphabet_is_refused():
    """Decision 16's structural guard, restated at the widened alphabet."""
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    payload["splits"]["train"][0]["symbols"] = ["NOM", "PUNCT"]
    with pytest.raises(ValueError, match="outside the widened alphabet"):
        WidenedClauseArtifact(payload)


def test_a_case_only_artifact_is_refused_by_the_widened_loader():
    """The two schemas may not be read by each other's loader."""
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    payload["schema"] = "ailm-clause-counts-v2"
    with pytest.raises(ValueError, match="Unexpected artifact schema"):
        WidenedClauseArtifact(payload)


# ── 3. THE DISJOINTNESS ASSERTION CAN FAIL ───────────────────────────────────


class _Token:
    """A UD token carrying only the fields this alphabet may read, plus the
    three it may not, so a leak has something real to leak."""

    def __init__(self, upos, case_derived=None, form="x", lemma="x", feats="_"):
        self.upos = upos
        self.case_derived = case_derived
        self.form = form
        self.lemma = lemma
        self.feats = feats


def _sentence_tokens():
    from ailm.cases import CaseRole

    return [
        _Token("AUX", form="Is", lemma="be"),
        _Token("PRON", case_derived=list(CaseRole)[0], form="it", lemma="it"),
        _Token("VERB", form="working", lemma="work"),
        _Token("PUNCT", form="?", lemma="?"),
    ]


def test_the_shipped_extractor_passes_the_disjointness_assertion():
    """POSITIVE anchor: the assertion must accept the real extractor, or the
    two negative controls below would pass for the wrong reason."""
    tokens = _sentence_tokens()
    assert_widened_disjointness(tokens, widened_symbols(tokens), sent_id="anchor")


def test_a_planted_lemma_leak_is_rejected():
    """NEGATIVE CONTROL: an extractor reading the lemma must be refused.

    A verb's lemma and form are the richest correlates of interrogative mood in
    this corpus. If the assertion cannot see an extractor consulting them, the
    arm's disjointness claim is decoration.
    """

    def leaking_extractor(tokens):
        sequence = []
        for token in tokens:
            if token.upos == "PUNCT":
                continue
            if token.upos in {"VERB", "AUX"}:
                # The leak: the symbol depends on the lemma, not on UPOS alone.
                sequence.append(VERB_SYMBOL if token.lemma.startswith("w") else AUX_SYMBOL)
            elif token.case_derived is not None:
                sequence.append(0)
        return sequence

    tokens = _sentence_tokens()
    with pytest.raises(AssertionError, match="lemma, feats"):
        assert_widened_disjointness(
            tokens, leaking_extractor(tokens), extract=leaking_extractor, sent_id="planted"
        )


def test_a_planted_punctuation_emitter_is_rejected():
    """NEGATIVE CONTROL: PUNCT is the label source and must emit nothing."""

    def punct_emitting_extractor(tokens):
        sequence = []
        for token in tokens:
            if token.upos == "PUNCT":
                sequence.append(AUX_SYMBOL if token.form == "?" else VERB_SYMBOL)
            elif token.upos in {"VERB", "AUX"}:
                sequence.append(VERB_SYMBOL if token.upos == "VERB" else AUX_SYMBOL)
            elif token.case_derived is not None:
                sequence.append(0)
        return sequence

    tokens = _sentence_tokens()
    with pytest.raises(AssertionError, match="PUNCT token contributes an observation"):
        assert_widened_disjointness(
            tokens,
            punct_emitting_extractor(tokens),
            extract=punct_emitting_extractor,
            sent_id="planted",
        )


# ── 4. THE NEGATIVE CONTROLS FIRE ────────────────────────────────────────────


def test_swapping_the_two_types_chains_costs_accuracy(artifact, models):
    """NEGATIVE CONTROL: if the type assignment is free, nothing is measured."""
    correct = classify_sentences(artifact.test, models)
    swapped = classify_sentences(
        artifact.test,
        {
            "transitions": models["transitions"][::-1].copy(),
            "starts": models["starts"][::-1].copy(),
            "type_prior": models["type_prior"],
        },
    )
    assert swapped["accuracy"] < correct["accuracy"], (
        "exchanging the two fitted chains cost nothing, so the widened arm is "
        "not measuring the sentence-type signal it reports"
    )


def test_the_decoder_equivalence_guard_holds_and_is_not_vacuous(artifact, models):
    """Closed form equals oracle filtering, AND a planted leak breaks that."""
    from ailm.verification.oracle import forward_filter

    emission = build_joint_emission(N_WIDENED_SYMBOLS)
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)

    worst = 0.0
    for _stype, sequence in artifact.test[:25]:
        closed, _ = sequence_log_evidence(sequence, models)
        _, oracle = forward_filter(emission, joint_b, joint_d, sequence)
        worst = max(worst, abs(closed - oracle))
    assert worst <= WIDENED_DECODER_TOLERANCE, (
        f"closed form and oracle filtering disagree by {worst:.3e}"
    )

    leaked = joint_b.copy()
    for source_type in range(len(CLAUSE_TYPES)):
        target_type = 1 - source_type
        source = slice(source_type * N_WIDENED_SYMBOLS, (source_type + 1) * N_WIDENED_SYMBOLS)
        target = slice(target_type * N_WIDENED_SYMBOLS, (target_type + 1) * N_WIDENED_SYMBOLS)
        leaked[target, source] += WIDENED_LEAK_MASS
    leaked /= leaked.sum(axis=0, keepdims=True)

    divergence = 0.0
    for _stype, sequence in artifact.test[:25]:
        closed, _ = sequence_log_evidence(sequence, models)
        _, leaked_evidence = forward_filter(emission, leaked, joint_d, sequence)
        divergence = max(divergence, abs(closed - leaked_evidence))
    assert divergence > WIDENED_DECODER_TOLERANCE, (
        "a transition with mass across the type blocks still agrees with the "
        "closed-form mixture, so the equivalence check above is vacuous"
    )


def test_the_joint_transition_has_exactly_zero_cross_type_mass(models):
    """The level's structural claim, checked on the built matrix."""
    joint = build_joint_transition(models)
    assert joint.shape == (N_WIDENED_JOINT, N_WIDENED_JOINT)
    for source_type in range(len(CLAUSE_TYPES)):
        target_type = 1 - source_type
        source = slice(source_type * N_WIDENED_SYMBOLS, (source_type + 1) * N_WIDENED_SYMBOLS)
        target = slice(target_type * N_WIDENED_SYMBOLS, (target_type + 1) * N_WIDENED_SYMBOLS)
        assert not joint[target, source].any(), "a sentence may not change type mid-way"
    assert np.allclose(joint.sum(axis=0), 1.0)


# ── THE LIVE AGENT AND ITS SPEC ──────────────────────────────────────────────


def test_live_widened_agent_conforms_to_its_spec(models):
    from ailm.gnn import assert_conforms, describe_shapes, load_spec

    assert_conforms(load_spec("L5w_clause_widened"), describe_shapes(build_widened_agent(models)))


@pytest.mark.parametrize("mutation", ["identity_joint_transition", "single_type"])
def test_widened_spec_mutations_are_rejected(models, mutation):
    from ailm.gnn import check_conformance, describe_shapes, load_spec

    violations = check_conformance(
        load_spec(f"L5w_clause_widened@{mutation}"),
        describe_shapes(build_widened_agent(models)),
    )
    assert violations, (
        f"L5w_clause_widened@{mutation} passed conformance against the correct "
        f"live agent, so that check is vacuous for the property it targets."
    )


# ── 5. THE LEDGER REPRODUCES FROM ITS OWN SUFFICIENT STATISTICS ──────────────


def test_the_recorded_exact_p_reproduces_from_its_discordant_counts(ledger):
    b = int(_claim(ledger, "stage5w_mcnemar_b"))
    c = int(_claim(ledger, "stage5w_mcnemar_c"))
    recorded = float(_claim(ledger, "stage5w_mcnemar_p"))
    assert recorded == pytest.approx(float(exact_two_sided_binomial_p(b, c)), rel=1e-12)


def test_the_recorded_perplexities_reproduce_from_the_artifact(artifact, models, ledger):
    """The gate's three numbers, recomputed from the vendored statistics."""
    comparator = build_type_free_chain(artifact.train, n_symbols=N_WIDENED_SYMBOLS)
    measured = type_conditional_perplexities(artifact.test, models, comparator)
    assert float(_claim(ledger, "stage5w_mixture_perplexity")) == pytest.approx(
        measured["mixture"], abs=5e-5
    )
    assert float(_claim(ledger, "stage5w_typefree_perplexity")) == pytest.approx(
        measured["type_free"], abs=5e-5
    )
    assert float(_claim(ledger, "stage5w_type_oracle_perplexity")) == pytest.approx(
        measured["type_oracle"], abs=5e-5
    )
    assert int(_claim(ledger, "stage5w_test_positions")) == measured["positions"]


def test_the_recorded_census_matches_the_artifact(artifact, ledger):
    assert int(_claim(ledger, "stage5w_train_sentences")) == len(artifact.train)
    assert int(_claim(ledger, "stage5w_test_sentences")) == len(artifact.test)
    assert int(_claim(ledger, "stage5w_source_documents")) == len(artifact.document_ids)
    assert int(_claim(ledger, "stage5w_test_documents")) == len(
        artifact.test_split_document_indices
    )
    assert int(_claim(ledger, "stage5w_alphabet_symbols")) == N_WIDENED_SYMBOLS
    assert int(_claim(ledger, "stage5w_joint_states")) == N_WIDENED_JOINT


def test_every_widened_claim_is_recorded_descriptive(ledger):
    """Decision 112 is a development-session specification, not a registration.

    Ambiguity resolves downward. A label from this arm carrying a stronger
    status would be claiming a timestamped prior artifact that does not exist.
    """
    labels = [label for label in ledger.all_labels() if label.startswith("stage5w_")]
    assert labels, "no stage5w_* labels in the ledger; run the widened recorder"
    for label in labels:
        entry = ledger.get(label)
        assert entry.get("status") == "descriptive", (
            f"{label} is recorded {entry.get('status')!r}; every widened-arm label "
            f"must be descriptive (decision 112)"
        )


def test_the_shipped_case_only_arm_is_untouched(ledger):
    """The widened arm may not move the case-only arm's recorded verdict.

    Whatever this arm found, decision 83's finding is the case-only arm's, is
    owned by a different recorder, and is enforced by
    ``scripts/audit_gates.py::audit_stage5_case_only``. This is the guard against
    the one failure mode a second arm introduces: quietly relabeling the first.
    Decision 114 makes it load-bearing rather than precautionary. The widened arm
    is now the canonical Level 5, so the temptation this guards against is no
    longer hypothetical: the case-only result must survive the promotion exactly
    as recorded, or the retained ablation is decoration.
    """
    shipped_mixture = float(_claim(ledger, "stage5_mixture_perplexity"))
    shipped_typefree = float(_claim(ledger, "stage5_typefree_perplexity"))
    assert shipped_mixture > shipped_typefree, (
        "the case-only mixture no longer sits on the failing side of its "
        "own bar; that is a change to the ablation, not to the canonical arm"
    )
    owners = {
        label: ledger.get(label).get("source")
        for label in ledger.all_labels()
        if label.startswith("stage5_") or label.startswith("stage5w_")
    }
    shipped_sources = {source for label, source in owners.items() if label.startswith("stage5_")}
    widened_sources = {source for label, source in owners.items() if label.startswith("stage5w_")}
    assert widened_sources == {"scripts/record_stage5_widened_claims.py"}
    assert "scripts/record_stage5_widened_claims.py" not in shipped_sources


# ── 6. THE REGISTERED PREDICTIONS AND THE WITHDRAWAL GUARD ───────────────────


def test_the_encoded_interval_is_the_registered_interval():
    """The constant encodes decision 112's band exactly as registered.

    The registration quotes the probe's 99% bootstrap CI as ``[+0.039, +0.321]``
    nats/sentence at the widened item of ``docs/research_queue.md``, and
    decision 112 registers that interval by reference ("the interval the probe
    registered"). The same entry's original evidence line (commit 3162b61)
    recorded the endpoints to four decimals as ``[+0.0394, +0.3209]``, computed
    by uncommitted scratchpad scope scripts on the retired sentence split; the
    committed rounding widens the band outward, so this pin holds the encoded
    band to the registered-as-written quoting. The probe's resampling unit was
    never committed and is not invented here.
    """
    assert STAGE5W_REGISTERED_GAIN_INTERVAL_NATS == (0.039, 0.321)


def test_the_recorded_gain_sits_inside_the_registered_interval(ledger):
    """Prediction 1, on the recorded numbers: positive and inside the band."""
    gain = float(_claim(ledger, "stage5w_oracle_gain_nats_per_sentence"))
    assert stage5_widened_registered_gain_in_interval(gain), (
        f"recorded gain {gain} is not positive and inside the registered "
        f"interval {STAGE5W_REGISTERED_GAIN_INTERVAL_NATS} (decision 112): "
        "the pre-computation was contaminated and the proposal is withdrawn"
    )


def test_the_recorded_auc_sits_above_both_registered_comparators(ledger):
    """Registered prediction 2 holds on the recorded numbers.

    This is a REPORTABLE registered prediction, not a withdrawal condition:
    decision 112 records its failure as an outcome ("the signal is length or
    a single fronting indicator") and decision 114's withdrawal list does not
    name it, so the standing audit reports a violated AUC relation through
    ``canonical_stage5_prediction2_diagnostic`` while the exit status follows
    the enumerated withdrawal conditions. This test pins the recorded
    numbers' relation; the audit's planted diagnostic cases in
    test_gate_audit_export separate the reportable failure from a real
    withdrawal failure.
    """
    auc = float(_claim(ledger, "stage5w_auc"))
    length_control = float(_claim(ledger, "stage5w_auc_length_control"))
    first_aux = float(_claim(ledger, "stage5w_auc_first_aux_control"))
    assert stage5_widened_auc_above_registered_comparators(
        auc=auc, auc_length_control=length_control, auc_first_aux_control=first_aux
    ), (
        f"recorded AUC {auc} does not sit strictly above both the length-only "
        f"comparator {length_control} and the first-is-AUX comparator "
        f"{first_aux} (decision 112, prediction 2)"
    )


def test_the_recorded_canonical_row_satisfies_the_withdrawal_guard(ledger):
    """Decision 114's three withdrawal conditions, on the recorded numbers.

    The recorded control and classification fields are percentages on a 0-100
    scale (``*_pct``); the predicate takes fractions on the 0-1 scale, so the
    conversion is explicit here rather than silent. The verdict is derived
    from the recorded numbers (the oracle must beat the bar for the placement
    to stay bracketed), not typed in.
    """
    mixture = float(_claim(ledger, "stage5w_mixture_perplexity"))
    typefree = float(_claim(ledger, "stage5w_typefree_perplexity"))
    oracle = float(_claim(ledger, "stage5w_type_oracle_perplexity"))
    consistent = stage5_widened_passing_resolved_is_consistent(
        verdict=GateVerdict.BRACKETED_REFERENCE
        if oracle < typefree
        else GateVerdict.ABOVE_REFERENCE,
        point_passes=mixture < typefree,
        oracle_sentence_ci_low=float(_claim(ledger, "stage5w_oracle_gain_sentence_ci_low")),
        oracle_sentence_ci_high=float(_claim(ledger, "stage5w_oracle_gain_sentence_ci_high")),
        oracle_position_ci_low=float(_claim(ledger, "stage5w_oracle_gain_position_ci_low")),
        oracle_position_ci_high=float(_claim(ledger, "stage5w_oracle_gain_position_ci_high")),
        control_accuracy=float(_claim(ledger, "stage5w_control_classification_pct")) / 100.0,
        control_mixture_perplexity=float(_claim(ledger, "stage5w_control_mixture_perplexity")),
        control_typefree_perplexity=float(_claim(ledger, "stage5w_control_typefree_perplexity")),
        swapped_blocks_accuracy=float(_claim(ledger, "stage5w_swapped_blocks_classification_pct"))
        / 100.0,
        model_classification_accuracy=float(_claim(ledger, "stage5w_classification_pct")) / 100.0,
        leak_control_divergence=float(_claim(ledger, "stage5w_leak_control_divergence")),
        decoder_tolerance=WIDENED_DECODER_TOLERANCE,
    )
    assert consistent, (
        "the recorded canonical row no longer satisfies decision 114's "
        "withdrawal conditions; the promotion must be reviewed"
    )


def test_the_registered_gain_predicate_fails_closed():
    """Planted bands: below, above, nonpositive, and schema violations."""
    assert not stage5_widened_registered_gain_in_interval(0.01)
    assert not stage5_widened_registered_gain_in_interval(0.5)
    assert not stage5_widened_registered_gain_in_interval(-0.1)
    for bad in (None, math.nan, math.inf, True, "0.2"):
        with pytest.raises(ValueError):
            stage5_widened_registered_gain_in_interval(bad)


def test_the_registered_auc_predicate_fails_closed():
    """Planted comparators: one-sided, tie, percentage scale, nonfinite."""

    def call(auc, length, first_aux):
        return stage5_widened_auc_above_registered_comparators(
            auc=auc, auc_length_control=length, auc_first_aux_control=first_aux
        )

    assert not call(0.8908, 0.9, 0.5), "above (b) but not (c) is a failure, not a partial success"
    assert not call(0.8908, 0.5, 0.9), "above (c) but not (b) is a failure, not a partial success"
    assert not call(0.8908, 0.5828, 0.8908), "a tie with a comparator is strict-fail"
    with pytest.raises(ValueError):
        call(89.08, 58.28, 75.43), "a 0-100 percentage field is not an AUC"
    with pytest.raises(ValueError):
        call(math.nan, 0.5, 0.5)


def test_the_withdrawal_guard_fails_closed():
    """Each planted dead state fails decision 114's guard for its own reason."""
    live = dict(
        verdict=GateVerdict.BRACKETED_REFERENCE,
        point_passes=True,
        oracle_sentence_ci_low=0.024598,
        oracle_sentence_ci_high=0.358617,
        oracle_position_ci_low=0.003053,
        oracle_position_ci_high=0.046058,
        control_accuracy=0.995,
        control_mixture_perplexity=4.6513,
        control_typefree_perplexity=5.5835,
        swapped_blocks_accuracy=0.5756,
        model_classification_accuracy=0.9366,
        leak_control_divergence=4.348005,
        decoder_tolerance=WIDENED_DECODER_TOLERANCE,
    )
    assert stage5_widened_passing_resolved_is_consistent(**live)
    # The gate and its placement.
    assert not stage5_widened_passing_resolved_is_consistent(**{**live, "point_passes": False})
    assert not stage5_widened_passing_resolved_is_consistent(
        **{**live, "verdict": GateVerdict.ABOVE_REFERENCE}
    )
    # Either gain interval resolving onto zero withdraws the promotion.
    assert not stage5_widened_passing_resolved_is_consistent(
        **{**live, "oracle_sentence_ci_low": 0.0}
    )
    assert not stage5_widened_passing_resolved_is_consistent(
        **{**live, "oracle_position_ci_low": 0.0}
    )
    # The detectability control dead — or arriving on the wrong scale.
    assert not stage5_widened_passing_resolved_is_consistent(**{**live, "control_accuracy": 0.5})
    assert not stage5_widened_passing_resolved_is_consistent(
        **{**live, "control_mixture_perplexity": live["control_typefree_perplexity"]}
    )
    with pytest.raises(ValueError):
        stage5_widened_passing_resolved_is_consistent(**{**live, "control_accuracy": 99.5})
    # Either negative control dead.
    assert not stage5_widened_passing_resolved_is_consistent(
        **{**live, "swapped_blocks_accuracy": live["model_classification_accuracy"]}
    )
    assert not stage5_widened_passing_resolved_is_consistent(
        **{**live, "leak_control_divergence": WIDENED_DECODER_TOLERANCE}
    )
