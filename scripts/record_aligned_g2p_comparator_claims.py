#!/usr/bin/env python3
"""Record the P2 comparator arm's offline claims (protocol §3.4).

The stipulated aligned spelling-to-pronunciation artifact
(``vendor/synthetic_aligned_g2p_counts.json``, schema
``ailm-synthetic-aligned-g2p-v1``) freezes a seeded per-letter correspondence
and the train/test pronunciation splits generated under it
(``ailm.benchmarks.aligned_g2p``).  This recorder fits the shipped G2P engine
— ``ailm.levels.phoneme.build_g2p_emission`` (add-k over letters, its own
emission smoothing dial) + ``build_g2p_transition`` (train-only pronunciation
bigrams through ``build_phone_transition``, add-k) + ``build_g2p_start``
(first-phone unigram, add-k), decoded by the shipped
``ailm.benchmarks.comparators._ailm_g2p_decode`` readout with default add-1
Laplace throughout (``ailm.config.DEFAULT_LEVEL_CONFIG``, whole-word
conditioning, no window truncation) — on the artifact's ``splits.train``
pairs and scores every arm on the identical ``splits.test`` positions:

* the native stipulated engine;
* the separately named ``T``-lookup baseline
  (``ailm.benchmarks.aligned_g2p.t_lookup_baseline``): the trivial
  closed-form map is exactly right on ``T``-generated data by construction,
  so the native comparison against it is reported, never gated, and the
  lookup is never presented as the shipped engine;
* the context-free pronunciation-bigram comparator fitted on the same train
  pairs (``ailm.benchmarks.corpora.phone_statistics`` over ``splits.train``,
  read out through the shipped ``_phoneme_comparator``, prefix context reset
  per pair, an empty prefix falling back to the unigram argmax, an unseen
  bigram landing at ``k / (n + k * 39)`` through ``build_phone_transition``).
  It conditions on the held-out pronunciation prefix per the phoneme task's
  ``phoneme_positions`` convention, so it is the pronunciation-only bar, not
  a spelling arm, and it is recorded descriptively and never gated;
* the analytic uniform floor (``1 / 39`` per position against the
  ``claim:phoneme_count_39`` inventory, its word-length power for the exact
  reading).

Refusal gates — each raises ``SystemExit`` before any ledger write, the
``_phone_comparator_refusals`` pattern of
``scripts/record_synthetic_benchmark_claims.py``.  Every threshold is derived
offline from the frozen artifact's own arithmetic at recording time; no
factor is carried over from the phone arm:

* A1, English-transfer refusal: a comparator fitted on the English CMUdict
  stride-split train side (``ailm.levels.phoneme.g2p_split`` over
  ``vendor/cmudict_counts.json`` ``g2p_pairs``) must not meet or clear the
  native bar on the identical stipulated test positions.  The engine's own
  skip of non-1:1 English pairs is that shipped policy applied to English
  data, disclosed wherever the control is quoted (the
  ``a1_english_skip_disclosure`` policy of ``evaluate_aligned_g2p``).  A
  fired A1 means the stipulated signal is not separable from English
  transfer — an attribution ambiguity and a legitimate finding, reported by
  refusal, never silently retried.
* A2, broken-correspondence control: the shape-valid broken map
  ``T' = sigma . T`` (``broken_correspondence``, a complete derangement of
  ``T``'s phone codomain) defines a refit whose comparator must not meet or
  clear the native bar on ``T``-generated data.  A fired A2 is the same
  legitimate attribution finding.
* A3, uniform floor: the native engine must beat the uniform floor on both
  the position and the exact-word reading.

The bar is the native engine's own held-out position accuracy — the
``pre_registration`` bar declared by
``ailm.benchmarks.aligned_g2p.evaluate_aligned_g2p`` — so a control that
meets or clears it on the identical positions is exactly the condition that
evaluation's ``pre_registration`` block names as refusing publication.

Ledger discipline: wholesale namespace replacement through
``ailm.claims.ClaimsLedger.replace_claims`` (stale labels owned by this
source removed and the new batch written in one atomic transition) under
``ailm.ledger_lock.single_writer_lock`` — the
``scripts/record_synthetic_benchmark_claims.py`` pattern.  Every label
carries the ``bench_synthg2p_comparator_`` prefix; the recorder maps to
``exploratory`` in ``scripts/assign_claim_status.py``, with the two
uniform-floor labels that file's analytic closed-form exception.

Usage
-----
    .venv/bin/python scripts/record_aligned_g2p_comparator_claims.py

    # isolated validation, never the tracked ledger:
    AILM_CLAIMS_PATH=/tmp/aligned_g2p_claims.json .venv/bin/python \
        scripts/record_aligned_g2p_comparator_claims.py

    # against a fixture artifact instead of the vendored one:
    .venv/bin/python scripts/record_aligned_g2p_comparator_claims.py \
        --vendor-dir /tmp/aligned_g2p_vendor
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.benchmarks.aligned_g2p import (  # noqa: E402  (path setup precedes imports)
    ALIGNED_G2P_ARTIFACT_NAME,
    artifact_pairs,
    broken_correspondence,
    load_aligned_g2p,
    t_lookup_baseline,
)
from ailm.benchmarks.comparators import (  # noqa: E402  (path setup precedes imports)
    _ailm_g2p_metrics,
    _phoneme_comparator,
)
from ailm.benchmarks.corpora import phone_statistics  # noqa: E402  (path setup precedes imports)
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402  (path setup precedes imports)
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.phoneme import PHONES, g2p_split, load_phoneme_counts  # noqa: E402
from ailm.resources import resolve_vendored_artifact  # noqa: E402  (path setup precedes imports)

#: One run id for the one measured run this recorder performs; the source is
#: the label-ownership key the ledger's wholesale replacement clears by.
RUN_ID = "aligned-g2p-comparator-v1"
SOURCE = "scripts/record_aligned_g2p_comparator_claims.py"

#: The gate inputs ``_comparator_refusals`` consumes, in extraction order.
GATE_KEYS = (
    "native_position",
    "native_exact",
    "english_position",
    "broken_position",
    "position_floor",
    "exact_floor",
)


def _contextfree_bigram_metrics(
    train: list[tuple[str, list[str]]],
    test: list[tuple[str, list[str]]],
) -> tuple[float, float]:
    """Score the context-free pronunciation-bigram comparator on the test pairs.

    Fitted on ``train`` only (``phone_statistics`` resets per pair), read out
    through the shipped ``_phoneme_comparator``: position ``i`` of each test
    pronunciation conditions on the gold prefix ``phones[:i]`` — the phoneme
    task's ``phoneme_positions`` convention, the prefix built from the word's
    start and reset per pair — so position accuracy keeps the exact
    denominator the other arms score, and an empty prefix (the word-initial
    position of each pair) falls back to the unigram argmax.  This is the
    pronunciation-only bar: it never sees a spelling, is never a gate, and is
    quoted only beside that disclosure.
    """

    unigrams, bigrams = phone_statistics(train)
    positions: list[tuple[str, str]] = []
    lengths: list[int] = []
    for _word, gold in test:
        lengths.append(len(gold))
        positions.extend((" ".join(gold[:i]), gold[i]) for i in range(len(gold)))
    flat = _phoneme_comparator(unigrams, bigrams, positions)
    predictions: list[list[str]] = []
    offset = 0
    for length in lengths:
        predictions.append(flat[offset : offset + length])
        offset += length
    total = sum(lengths)
    position = (
        sum(
            predicted == gold_phone
            for prediction, (_word, gold) in zip(predictions, test, strict=True)
            for predicted, gold_phone in zip(prediction, gold, strict=True)
        )
        / total
        if total
        else 0.0
    )
    exact = (
        sum(prediction == gold for prediction, (_word, gold) in zip(predictions, test, strict=True))
        / len(test)
        if test
        else 0.0
    )
    return position, exact


def _gate_materials(vendor_dir: Path | None = None) -> dict[str, object]:
    """Every offline quantity the refusals and the claim batch consume.

    All arithmetic runs against the frozen artifact's own pairs before any
    gate is consulted: the native engine refit on ``splits.train``, the
    named baselines, both controls, and the analytic floors.  Nothing here
    touches the ledger or the network.
    """

    payload = load_aligned_g2p(vendor_dir)
    correspondence, train, test = artifact_pairs(payload)

    native_position, native_exact, native_predictions = _ailm_g2p_metrics(train, test)
    lookup_position, lookup_exact, lookup_predictions = t_lookup_baseline(correspondence, test)
    contextfree_position, contextfree_exact = _contextfree_bigram_metrics(train, test)

    # A1: the same engine fitted on the English CMUdict stride-split train
    # side (the shipped split; its non-1:1 pairs are skipped by that engine's
    # own logged policy — the a1_english_skip_disclosure of the evaluator).
    english_train, _english_test = g2p_split(load_phoneme_counts())
    english_position, english_exact, _english_predictions = _ailm_g2p_metrics(english_train, test)

    # A2: the engine refit on T' = sigma . T pairs, scored against T gold.
    broken = broken_correspondence(correspondence)
    broken_train = [(word, [broken[letter] for letter in word]) for word, _ in train]
    broken_position, broken_exact, _broken_predictions = _ailm_g2p_metrics(broken_train, test)

    position_floor = 1.0 / len(PHONES)
    word_length = len(test[0][0])
    exact_floor = position_floor**word_length

    artifact_path = resolve_vendored_artifact(ALIGNED_G2P_ARTIFACT_NAME, directory=vendor_dir)
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    return {
        "native_position": native_position,
        "native_exact": native_exact,
        "native_predictions": native_predictions,
        "lookup_position": lookup_position,
        "lookup_exact": lookup_exact,
        "lookup_predictions": lookup_predictions,
        "contextfree_position": contextfree_position,
        "contextfree_exact": contextfree_exact,
        "english_position": english_position,
        "english_exact": english_exact,
        "broken_position": broken_position,
        "broken_exact": broken_exact,
        "position_floor": position_floor,
        "exact_floor": exact_floor,
        "word_length": word_length,
        "test_word_count": len(test),
        "test_position_count": sum(len(gold) for _word, gold in test),
        "artifact_sha256": artifact_sha256,
    }


def _comparator_refusals(
    *,
    native_position: float,
    native_exact: float,
    english_position: float,
    broken_position: float,
    position_floor: float,
    exact_floor: float,
) -> None:
    """The pre-registered comparator refusals, before any ledger write.

    The bar is the native stipulated engine's own held-out position accuracy
    (the ``pre_registration`` bar of ``evaluate_aligned_g2p``), and every
    margin is recomputed from the frozen artifact's own arithmetic by
    ``_gate_materials`` — the ``_phone_comparator_refusals`` derivation
    pattern, with no factor carried over from the phone arm.  A control that
    meets or clears the bar is an attribution ambiguity and a legitimate
    finding: the recorder refuses, writes nothing, and the finding is
    reported by the refusal itself.
    """

    if english_position >= native_position:
        raise SystemExit(
            "aligned-g2p comparator refused (A1): an English CMUdict-fitted "
            "comparator meets or clears the native bar on the identical "
            f"stipulated test positions ({english_position:.4f} >= "
            f"{native_position:.4f}); the stipulated signal is not separable "
            "from English transfer — an attribution ambiguity, reported by "
            "refusal. No claims written."
        )
    if broken_position >= native_position:
        raise SystemExit(
            "aligned-g2p comparator refused (A2): the broken-correspondence "
            "(sigma . T) comparator meets or clears the native bar on "
            f"T-generated data ({broken_position:.4f} >= "
            f"{native_position:.4f}); the win would not come from the "
            "stipulated correspondence — an attribution ambiguity, reported "
            "by refusal. No claims written."
        )
    if native_position <= position_floor or native_exact <= exact_floor:
        raise SystemExit(
            "aligned-g2p comparator refused (A3): the native stipulated "
            "engine does not beat the uniform floor on the identical test "
            f"positions (position {native_position:.4f} vs floor "
            f"{position_floor:.4f}, exact {native_exact:.6g} vs floor "
            f"{exact_floor:.6g}); no claims written."
        )


def _claim_values(materials: dict[str, object]) -> dict[str, object]:
    """Map the measured materials onto the recorder's claim-label namespace."""

    return {
        "bench_synthg2p_comparator_native_position_accuracy": round(
            materials["native_position"], 8
        ),
        "bench_synthg2p_comparator_native_exact_accuracy": round(materials["native_exact"], 8),
        "bench_synthg2p_comparator_t_lookup_position_accuracy": round(
            materials["lookup_position"], 8
        ),
        "bench_synthg2p_comparator_t_lookup_exact_accuracy": round(materials["lookup_exact"], 8),
        "bench_synthg2p_comparator_context_free_position_accuracy": round(
            materials["contextfree_position"], 8
        ),
        "bench_synthg2p_comparator_context_free_exact_accuracy": round(
            materials["contextfree_exact"], 8
        ),
        # Closed-form analytic quantities, recorded unrounded: the exact-word
        # floor sits far below any float rounding grid that survives.
        "bench_synthg2p_comparator_uniform_position_floor": float(materials["position_floor"]),
        "bench_synthg2p_comparator_uniform_exact_floor": float(materials["exact_floor"]),
        "bench_synthg2p_comparator_a1_english_position_accuracy": round(
            materials["english_position"], 8
        ),
        "bench_synthg2p_comparator_a1_english_exact_accuracy": round(materials["english_exact"], 8),
        "bench_synthg2p_comparator_a2_broken_position_accuracy": round(
            materials["broken_position"], 8
        ),
        "bench_synthg2p_comparator_a2_broken_exact_accuracy": round(materials["broken_exact"], 8),
        "bench_synthg2p_comparator_native_ties_t_lookup": (
            materials["native_predictions"] == materials["lookup_predictions"]
        ),
        "bench_synthg2p_comparator_test_word_count": materials["test_word_count"],
        "bench_synthg2p_comparator_test_position_count": materials["test_position_count"],
        "bench_synthg2p_comparator_artifact_sha256": materials["artifact_sha256"],
    }


def _measure(vendor_dir: Path | None = None) -> dict[str, object]:
    """Derive every gate input, refuse any gate that is not live, map claims.

    The refusals run strictly before any ledger object is constructed, so a
    fired gate cannot leave a partial write behind.
    """

    materials = _gate_materials(vendor_dir)
    _comparator_refusals(**{key: materials[key] for key in GATE_KEYS})
    return _claim_values(materials)


def _record(values: dict[str, object]) -> None:
    """Wholesale namespace replacement: stale labels out, this run's batch in.

    Removal and recording are one atomic ``replace_claims`` transition under
    the ledger's single-writer lock, exactly the
    ``scripts/record_synthetic_benchmark_claims.py`` discipline.
    """

    ledger = load_ledger(REPO_ROOT)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record the P2 comparator arm's offline bench_synthg2p_comparator claims."
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=None,
        help=(
            "directory holding "
            f"{ALIGNED_G2P_ARTIFACT_NAME} (default: the canonical vendored artifact)"
        ),
    )
    args = parser.parse_args(argv)
    with single_writer_lock(ledger_lock_path(REPO_ROOT)):
        values = _measure(args.vendor_dir)
        _record(values)
    print(f"recorded {len(values)} bench_synthg2p_comparator claims (run_id={RUN_ID}):")
    for label in sorted(values):
        print(f"  {label} = {values[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
