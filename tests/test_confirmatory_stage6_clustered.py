"""Guards for the decision-115 held-back Stage 6 document-clustered interval.

The arm this file guards is one number and its interval, so what is worth
testing is not the number but the six ways the arm could have been worthless
without looking it:

1. **The interval is re-derivable.** Every recorded bound recomputes from the
   vendored artifacts through the same public functions, so a stored scalar
   cannot drift away from the data that justifies it.
2. **The models were not re-fitted.** The held-back scoring uses the development
   phase emission and unigram, array for array. A re-fit would make this a
   development result wearing a held-back name.
3. **Every control is real.** A control that cannot fail and a control that
   cannot fire both prove nothing, so the family is checked from both sides: the
   centered baseline must not detect, a plant above the recorded resolution must,
   a plant below it must not, and the whole-document sign flip must not. The
   development-sized plant is pinned to its measured value rather than to the
   comfortable direction, because decision 115's amendment made it a recorded
   fact and not a gate.
4. **The coverage matches the development arm.** The finding status compares a
   development bound against a held-back one. Two intervals at two coverages are
   two different quantities, and the comparison would then be meaningless.
5. **The recorder may re-run without changing anything.** Unlike the one-run
   replication recorder, this one is idempotent by design, and idempotency
   asserted in a docstring is not idempotency.
6. **Nothing recorded is unreachable.** Every label is render-blocking and every
   boolean is type-registered, so a render cannot describe the arm without it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ailm.claims import ClaimsLedger
from ailm.clustering import DEFAULT_RESAMPLES, DEFAULT_SEED, summarize_clustered_difference
from ailm.manuscript_variables import _REQUIRED_BOOLEAN_CLAIMS
from ailm.required_claims import REQUIRED_CLAIMS
from ailm.statistics import CORRECTED_CONFIDENCE, DIFFERENCE_CONFIDENCE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "claims.json"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RECORDER = PROJECT_ROOT / "scripts" / "record_confirmatory_stage6_clustered_claims.py"
SOURCE = "scripts/record_confirmatory_stage6_clustered_claims.py"
PREFIX = "confirmatory_stage6_clustered_"
DECISION = 115

#: The tolerance every reproduction comparison uses. The recorder rounds to six
#: decimal places on the nats scale, so a recomputation that agrees to within
#: half a unit in the last recorded place is the same number.
PLACES = 6
TOLERANCE = 0.5 * 10.0**-PLACES


def _ledger() -> ClaimsLedger:
    return ClaimsLedger(LEDGER)


def _value(label: str):
    entry = _ledger().get(label)
    assert entry is not None, f"required claim {label!r} is absent from the ledger"
    return entry["value"]


_HAS_ARM = LEDGER.is_file() and _ledger().get(f"{PREFIX}ci_low") is not None
_SKIP_NO_ARM = pytest.mark.skipif(
    not _HAS_ARM, reason="the decision-115 held-back clustered arm has not been recorded"
)


def _recompute():
    """The held-back deltas and their document keys, through the shipped recorder."""
    import sys

    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from record_confirmatory_stage6_clustered_claims import held_back_differences

    return held_back_differences()


# ── 1. The interval is re-derivable ──────────────────────────────────────────


@_SKIP_NO_ARM
def test_the_primary_interval_reproduces_from_the_artifacts() -> None:
    """Every recorded primary bound recomputes from the vendored counts."""
    deltas, documents, n_documents = _recompute()
    summary = summarize_clustered_difference(deltas, documents, confidence=DIFFERENCE_CONFIDENCE)

    assert n_documents == _value(f"{PREFIX}documents")
    assert len(deltas) == _value(f"{PREFIX}sentences")
    assert summary.ci_low == pytest.approx(_value(f"{PREFIX}ci_low"), abs=TOLERANCE)
    assert summary.ci_high == pytest.approx(_value(f"{PREFIX}ci_high"), abs=TOLERANCE)
    assert summary.effect == pytest.approx(_value(f"{PREFIX}mean_improvement_nats"), abs=TOLERANCE)
    assert summary.interval_excludes_zero is _value(f"{PREFIX}excludes_zero")
    assert summary.largest_cluster == _value(f"{PREFIX}largest_document")
    assert summary.positive_clusters == _value(f"{PREFIX}positive_documents")
    assert summary.negative_clusters == _value(f"{PREFIX}negative_documents")


@_SKIP_NO_ARM
def test_the_recorded_boolean_is_the_bounds_own_verdict() -> None:
    """POSITIVE CONTROL on the flag: it is not free to disagree with its bounds.

    ``excludes_zero`` is what the audit's finding status reads. A flag that could
    drift from the two bounds beside it would let the status say one thing while
    the interval printed in the manuscript said another.
    """
    low = _value(f"{PREFIX}ci_low")
    high = _value(f"{PREFIX}ci_high")
    assert _value(f"{PREFIX}excludes_zero") is (low > 0.0 or high < 0.0)

    corrected_low = _value(f"{PREFIX}corrected_ci_low")
    corrected_high = _value(f"{PREFIX}corrected_ci_high")
    assert _value(f"{PREFIX}corrected_excludes_zero") is (
        corrected_low > 0.0 or corrected_high < 0.0
    )


@_SKIP_NO_ARM
def test_the_corrected_interval_contains_the_primary_one() -> None:
    """The stricter coverage cannot produce the narrower interval.

    Both are percentile bounds on one bootstrap replicate vector, so the
    family-corrected bounds must bracket the ``DIFFERENCE_CONFIDENCE`` ones. If
    they did not, the two rows came from different resampling runs and the
    sensitivity is not a sensitivity of this result.
    """
    assert _value(f"{PREFIX}corrected_confidence") > _value(f"{PREFIX}confidence")
    assert _value(f"{PREFIX}corrected_ci_low") <= _value(f"{PREFIX}ci_low")
    assert _value(f"{PREFIX}corrected_ci_high") >= _value(f"{PREFIX}ci_high")
    assert not (
        _value(f"{PREFIX}corrected_excludes_zero") and not _value(f"{PREFIX}excludes_zero")
    ), "the stricter interval excludes zero while the looser one does not; impossible"


# ── 2. The models were not re-fitted ─────────────────────────────────────────


@_SKIP_NO_ARM
def test_the_held_back_scoring_uses_the_development_fits() -> None:
    """The merged frame's narrative model equals the development one, array for array.

    ``held_back_differences`` raises if it does not, so this test asserts the
    guard is live rather than trusting the recorder's own report of it.
    """
    import sys

    import numpy as np

    from ailm.levels.narrative import build_phase_models, load_narrative_artifact

    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from record_confirmatory_replication import merged_narrative_artifact

    development = load_narrative_artifact()
    merged = merged_narrative_artifact()
    expected = build_phase_models(development.train)
    built = build_phase_models(merged.train)
    assert set(built) == set(expected)
    for key, array in built.items():
        assert np.array_equal(array, expected[key]), (
            f"the narrative {key} built from the merged frame differs from the "
            "development one; the held-back arm re-fitted a parameter"
        )


@_SKIP_NO_ARM
def test_the_held_back_documents_are_disjoint_from_the_development_ones() -> None:
    """The population really was held back on this level's own artifact."""
    import sys

    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from record_confirmatory_replication import merged_narrative_artifact

    from ailm.levels.narrative import load_narrative_artifact

    development = load_narrative_artifact()
    merged = merged_narrative_artifact()
    overlap = set(merged.test_document_ids) & set(development.document_ids)
    assert not overlap, (
        f"{len(overlap)} held-back document identities also appear in the "
        f"development artifact; the model saw its own evaluation data"
    )


@_SKIP_NO_ARM
def test_the_arm_agrees_with_the_decision_111_census() -> None:
    """This arm re-analyzes the same population decision 111 scored, not another."""
    assert _value(f"{PREFIX}documents") == _value("confirmatory_stage6_documents")
    assert _value(f"{PREFIX}sentences") == _value("confirmatory_stage6_sentences")
    assert _value(f"{PREFIX}mean_improvement_nats") == pytest.approx(
        _value("confirmatory_stage6_improvement_nats"), abs=5e-5
    )
    assert _value(f"{PREFIX}positive_documents") + _value(f"{PREFIX}negative_documents") <= _value(
        f"{PREFIX}documents"
    )


# ── 3. Every control is real ─────────────────────────────────────────────────


def _centered(deltas: list[float]) -> list[float]:
    import numpy as np

    mean = float(np.mean(deltas))
    return [delta - mean for delta in deltas]


@_SKIP_NO_ARM
def test_the_centered_baseline_does_not_detect() -> None:
    """NEGATIVE CONTROL on the whole control family: planting nothing finds nothing.

    Every plant is measured against the centered differences, whose mean is
    exactly zero. If that baseline already excluded zero, every plant above it
    would too, and the family would pass on a dead instrument.
    """
    deltas, documents, _ = _recompute()
    baseline = summarize_clustered_difference(
        _centered(deltas), documents, confidence=DIFFERENCE_CONFIDENCE
    )
    assert not baseline.interval_excludes_zero, (
        "the centered differences, with no planted effect at all, still produce "
        "an interval that excludes zero; every control below is meaningless"
    )
    assert _value(f"{PREFIX}centered_baseline_excludes_zero") is False
    assert baseline.ci_low == pytest.approx(
        _value(f"{PREFIX}centered_baseline_ci_low"), abs=TOLERANCE
    )
    assert baseline.ci_high == pytest.approx(
        _value(f"{PREFIX}centered_baseline_ci_high"), abs=TOLERANCE
    )


@_SKIP_NO_ARM
def test_the_minimum_detectable_effect_is_the_centered_lower_bound() -> None:
    """The recorded resolution is the closed form its derivation claims.

    A constant plant shifts every bootstrap replicate by the same constant, so
    the smallest detectable shift is the negated lower bound of the centered
    differences. Recomputing it here is what stops the recorded value from
    becoming a number nobody can re-derive.
    """
    deltas, documents, _ = _recompute()
    baseline = summarize_clustered_difference(
        _centered(deltas), documents, confidence=DIFFERENCE_CONFIDENCE
    )
    assert -baseline.ci_low == pytest.approx(
        _value(f"{PREFIX}min_detectable_effect_nats"), abs=TOLERANCE
    )
    assert _value(f"{PREFIX}min_detectable_effect_nats") > 0.0


@_SKIP_NO_ARM
def test_the_two_sided_probe_pins_the_resolution_as_a_boundary() -> None:
    """POSITIVE CONTROL, both sides: above the resolution detects, below does not.

    This is the test of shift equivariance the closed form above rests on. It
    cannot pass on a dead instrument, which would fail the upper probe, nor on a
    hair-trigger one, which would fail the lower. Both recorded flags are
    reproduced rather than trusted.
    """
    deltas, documents, _ = _recompute()
    centered = _centered(deltas)
    resolution = _value(f"{PREFIX}min_detectable_effect_nats")
    tolerance = _value(f"{PREFIX}min_detectable_tolerance_nats")

    above = summarize_clustered_difference(
        [value + resolution + tolerance for value in centered],
        documents,
        confidence=DIFFERENCE_CONFIDENCE,
    )
    below = summarize_clustered_difference(
        [value + resolution - tolerance for value in centered],
        documents,
        confidence=DIFFERENCE_CONFIDENCE,
    )

    assert above.interval_excludes_zero, (
        "a plant one tolerance above the recorded resolution is not detected; the "
        "bootstrap is not equivariant under a constant plant and the recorded "
        "resolution is not a boundary"
    )
    assert not below.interval_excludes_zero, (
        "a plant one tolerance below the recorded resolution is detected; the "
        "procedure is a hair trigger and its detections cannot be trusted"
    )
    assert _value(f"{PREFIX}probe_above_detects") is True
    assert _value(f"{PREFIX}probe_below_detects") is False


@_SKIP_NO_ARM
def test_the_development_sized_probe_is_recorded_as_measured_not_as_a_gate() -> None:
    """The development-sized plant is a measurement, and it did not resolve.

    Decision 115's amendment demoted this from a gating positive control to a
    recorded fact, because it decides whether a NULL would be interpretable and
    says nothing about a detection. On this population it comes out FALSE: the
    development effect is smaller than the arm's own resolution. That is a real
    limitation of the held-back population and the manuscript states it, so this
    test pins the value rather than asserting the comfortable direction.
    """
    deltas, documents, _ = _recompute()
    shift = _value(f"{PREFIX}control_planted_shift_nats")
    planted = summarize_clustered_difference(
        [value + shift for value in _centered(deltas)],
        documents,
        confidence=DIFFERENCE_CONFIDENCE,
    )

    assert planted.interval_excludes_zero is _value(f"{PREFIX}development_effect_resolved")
    assert planted.ci_low == pytest.approx(
        _value(f"{PREFIX}control_positive_ci_low"), abs=TOLERANCE
    )
    assert planted.ci_high == pytest.approx(
        _value(f"{PREFIX}control_positive_ci_high"), abs=TOLERANCE
    )
    # The flag and the resolution must tell one story: a plant below the
    # resolution cannot resolve, and a plant above it must.
    resolved = _value(f"{PREFIX}development_effect_resolved")
    assert resolved is (shift > _value(f"{PREFIX}min_detectable_effect_nats"))


@_SKIP_NO_ARM
def test_a_detection_does_not_depend_on_the_development_sized_probe() -> None:
    """The promotion rests on the observed effect clearing the resolution.

    The arm reports a detection while the development-sized probe is unresolved,
    which is only coherent because the observed effect is larger than the
    development one. If that ever stops holding, the two facts contradict and
    the finding status is not interpretable.
    """
    if not _value(f"{PREFIX}excludes_zero"):
        pytest.skip("the primary interval spans zero; the conjunction below is vacuous")
    assert _value(f"{PREFIX}mean_improvement_nats") > _value(
        f"{PREFIX}min_detectable_effect_nats"
    ), (
        "the primary interval excludes zero while the observed effect sits below "
        "the arm's own resolution; those cannot both be true"
    )


@_SKIP_NO_ARM
def test_the_negative_control_does_not_fire() -> None:
    """Sign-flipping whole documents destroys the effect, and the bounds reproduce."""
    import sys

    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from record_confirmatory_stage6_clustered_claims import _flip_whole_documents

    deltas, documents, _ = _recompute()
    seed = _value(f"{PREFIX}control_flip_seed")
    flipped, n_flipped = _flip_whole_documents(deltas, documents, seed)
    summary = summarize_clustered_difference(flipped, documents, confidence=DIFFERENCE_CONFIDENCE)

    assert n_flipped == _value(f"{PREFIX}control_flipped_documents")
    assert not summary.interval_excludes_zero, (
        "the procedure finds an effect in differences whose aggregate direction has been destroyed"
    )
    assert _value(f"{PREFIX}control_negative_excludes_zero") is False
    assert summary.ci_low == pytest.approx(
        _value(f"{PREFIX}control_negative_ci_low"), abs=TOLERANCE
    )
    assert summary.ci_high == pytest.approx(
        _value(f"{PREFIX}control_negative_ci_high"), abs=TOLERANCE
    )


@_SKIP_NO_ARM
def test_the_negative_control_would_fire_on_a_real_effect() -> None:
    """POSITIVE CONTROL on the negative control: it is capable of firing.

    A negative control whose procedure never excludes zero would pass the test
    above on a broken instrument. Planting a large effect on the flipped vector
    must make the same call fire.
    """
    import sys

    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from record_confirmatory_stage6_clustered_claims import _flip_whole_documents

    deltas, documents, _ = _recompute()
    flipped, _ = _flip_whole_documents(deltas, documents, _value(f"{PREFIX}control_flip_seed"))
    width = abs(_value(f"{PREFIX}ci_high") - _value(f"{PREFIX}ci_low"))
    loud = summarize_clustered_difference(
        [value + 10.0 * width for value in flipped],
        documents,
        confidence=DIFFERENCE_CONFIDENCE,
    )
    assert loud.interval_excludes_zero, (
        "an effect ten interval widths wide is not detected even after the flip; "
        "the negative control could not fire and so cannot exonerate anything"
    )


@_SKIP_NO_ARM
def test_the_flip_touches_whole_documents_only() -> None:
    """Every sentence in a flipped document flips; none in an unflipped one does."""
    import sys

    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from record_confirmatory_stage6_clustered_claims import _flip_whole_documents

    deltas, documents, _ = _recompute()
    flipped, n_flipped = _flip_whole_documents(
        deltas, documents, _value(f"{PREFIX}control_flip_seed")
    )
    per_document: dict[str, set[bool]] = {}
    for original, changed, document_id in zip(deltas, flipped, documents, strict=True):
        if original == 0.0:
            continue
        per_document.setdefault(document_id, set()).add(changed == -original)
    mixed = [document_id for document_id, seen in per_document.items() if len(seen) > 1]
    assert not mixed, (
        f"{len(mixed)} documents were flipped sentence-by-sentence rather than "
        "whole; the control would then break the cluster structure it exists to keep"
    )
    assert 0 < n_flipped < len(set(documents)), (
        "the flip selected every document or none; either makes the control a "
        "relabeling of the primary analysis"
    )


# ── 4. The coverage matches the development arm ──────────────────────────────


@_SKIP_NO_ARM
def test_the_coverage_and_resampling_match_the_development_arm() -> None:
    """The two sides of the finding-status comparison are the same quantity.

    The audit's Stage 6 status reads the development interval and this one
    together. If they were computed at different coverages, or from different
    resample counts, the comparison between them would not be a comparison.
    """
    assert _value(f"{PREFIX}confidence") == DIFFERENCE_CONFIDENCE
    assert _value(f"{PREFIX}confidence") == _value("cluster_confidence")
    assert _value(f"{PREFIX}resamples") == DEFAULT_RESAMPLES == _value("cluster_resamples")
    assert _value(f"{PREFIX}seed") == DEFAULT_SEED == _value("cluster_seed")
    assert _value(f"{PREFIX}corrected_confidence") == CORRECTED_CONFIDENCE
    assert _value(f"{PREFIX}cluster_unit") == _value("cluster_stage6_logloss_cluster_unit")
    assert _value(f"{PREFIX}effect_unit") == _value("cluster_stage6_logloss_effect_unit")


@_SKIP_NO_ARM
def test_the_planted_shift_is_the_development_effect() -> None:
    """The positive control plants the development effect, not an invented size."""
    assert _value(f"{PREFIX}control_planted_shift_nats") == pytest.approx(
        _value("cluster_stage6_logloss_effect"), abs=TOLERANCE
    )


# ── 5. The recorder may re-run without changing anything ─────────────────────


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="project interpreter absent")
@pytest.mark.timeout(1800)
def test_the_recorder_is_idempotent(tmp_path: Path) -> None:
    """Two consecutive runs against one ledger leave identical claims.

    This is the property that separates this recorder from the one-run
    replication recorder, and it is the reason it is safe inside the
    ``record_*_claims.py`` glob the orchestrator runs on every pass.
    """
    isolated = tmp_path / "claims.json"
    shutil.copy2(LEDGER, isolated)
    environment = {**os.environ, "AILM_CLAIMS_PATH": str(isolated.resolve())}

    def run() -> dict:
        result = subprocess.run(
            [str(VENV_PYTHON), str(RECORDER)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, (
            f"the recorder did not complete.\n{result.stdout[-1200:]}\n{result.stderr[-1500:]}"
        )
        claims = json.loads(isolated.read_text(encoding="utf-8"))["claims"]
        return {label: entry for label, entry in claims.items() if label.startswith(PREFIX)}

    first = run()
    second = run()
    assert first, "the recorder wrote no claims under its own prefix"
    assert first == second, (
        "a second run changed the recorded arm. An idempotent recorder inside the "
        "orchestrated core phase must be a function of its inputs alone."
    )


# ── 6. Nothing recorded is unreachable ───────────────────────────────────────


@_SKIP_NO_ARM
def test_the_recorder_owns_exactly_the_declared_label_set() -> None:
    """Every label under the prefix is written by this recorder and no other."""
    ledger = _ledger()
    owned = {label for label in ledger.labels_owned_by(SOURCE)}
    prefixed = {label for label in ledger.all_labels() if label.startswith(PREFIX)}
    assert owned == prefixed, (
        f"labels under {PREFIX!r} not owned by this recorder: "
        f"{sorted(prefixed - owned)}; labels owned but outside the prefix: "
        f"{sorted(owned - prefixed)}"
    )


@_SKIP_NO_ARM
def test_every_label_is_render_blocking() -> None:
    """A render cannot describe this arm without the measurement behind it."""
    prefixed = {label for label in _ledger().all_labels() if label.startswith(PREFIX)}
    missing = sorted(prefixed - set(REQUIRED_CLAIMS))
    assert not missing, (
        f"these recorded labels are not in REQUIRED_CLAIMS, so a render could "
        f"proceed without them: {missing}"
    )


@_SKIP_NO_ARM
def test_every_recorded_boolean_is_type_registered() -> None:
    """The validator walks the whole ledger; an unregistered boolean blocks it."""
    ledger = _ledger()
    booleans = {
        label
        for label in ledger.all_labels()
        if label.startswith(PREFIX) and isinstance(ledger.get(label)["value"], bool)
    }
    assert booleans, "the arm records no boolean at all; excludes_zero is missing"
    missing = sorted(booleans - _REQUIRED_BOOLEAN_CLAIMS)
    assert not missing, f"these recorded booleans are not in the manuscript boolean set: {missing}"


@_SKIP_NO_ARM
def test_the_arm_is_descriptive_and_cites_its_decision() -> None:
    """Decision 115 registers the estimand and assigns it the descriptive status.

    Registration preceding the outcome does not promote this arm: the population
    was already spent by decision 111, so the status resolves downward, and the
    decision number recorded beside the bounds is what points a reader at why.
    """
    ledger = _ledger()
    for label in (label for label in ledger.all_labels() if label.startswith(PREFIX)):
        assert ledger.get(label)["status"] == "descriptive", (
            f"{label} carries a status other than descriptive; decision "
            f"{DECISION} assigns this arm the downward default"
        )
    assert _value(f"{PREFIX}decision") == DECISION

    decisions = (PROJECT_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    assert f"\n## {DECISION} —" in decisions, (
        f"decision {DECISION} has no entry in the decision log, so the arm's "
        "registration is not on the record"
    )
