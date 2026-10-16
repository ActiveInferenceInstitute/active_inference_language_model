#!/usr/bin/env python3
"""Measure and record the first Stage 3 statement not conditioned on coverage.

The gap this closes
-------------------
Every Stage 3 number describes a sublanguage. The level assigns an
out-of-vocabulary token **no** probability rather than a small one, so the
held-out stream it is scored on has already had those tokens removed;
``vendor/brown_word_counts.json`` carries the filtered stream and never carried
the unfiltered one. The out-of-vocabulary comparison was therefore unrunnable,
not merely unshown, and the section's coverage percentage was a number with no
measurement behind it beyond its own definition.

``vendor/brown_heldout_stream.json`` supplies the denominator, and this recorder
asks the one question it makes answerable: **is the covered sublanguage an
easier population than the stream it was drawn from?**

Why the comparator answers it and the word level cannot
-------------------------------------------------------
The word level is undefined off its lexicon, so it cannot be scored on the part
of the stream it excludes; asking it to would require the smoothing the level
refuses. The letter-bigram bar has no such limit — it is a model over letters
and is defined on any alphabetic string — so scoring the *same comparator* on
the covered subset, the excluded remainder, and the whole stream isolates the
population effect from any difference between systems. If the bar finds the
covered subset easier, then Stage 3's margin, though it is a paired comparison
on identical tokens and is not thereby wrong, is reported on a favorable slice
of the corpus, and the size of that favor is a disclosure the section owes.

What this deliberately does not do
----------------------------------
It does not score the word level on out-of-vocabulary tokens under any smoothing
rule. A smoothed number would report a result for an input the model is
undefined on, which is the thing [@sec:word_level] refuses on the record. The
selection effect is measurable without it.

Usage
-----
    .venv/bin/python scripts/record_heldout_stream_claims.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_heldout_stream import SCHEMA, build_artifact  # noqa: E402

from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.word import (  # noqa: E402
    _letter_bigram_perplexity,
    build_letter_bigram_bar,
    load_word_artifact,
    validate_heldout_stream_artifact,
)

STREAM_PATH = REPO_ROOT / "vendor" / "brown_heldout_stream.json"
#: Smallest bucket that may contribute to the length-matched comparison. A
#: bucket of three tokens would put a whole length's weight on an estimate
#: made from three words.
_MIN_BUCKET = 30

RUN_ID = "heldout_stream_document_disjoint_emission_weighted_v3"
SOURCE = "scripts/record_heldout_stream_claims.py"


def main() -> int:
    with single_writer_lock(ledger_lock_path(REPO_ROOT)):
        return _measure_and_record()


def _measure_and_record() -> int:
    if not STREAM_PATH.exists():
        raise SystemExit(
            f"refused: {STREAM_PATH.relative_to(REPO_ROOT)} is absent. Run "
            f"scripts/generate_heldout_stream.py first. No claims written."
        )
    payload = json.loads(STREAM_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise SystemExit(
            f"refused: stream schema is {payload.get('schema')!r}, expected {SCHEMA!r}"
        )
    # Reproduction, not trust. The artifact is derived from the same corpus and
    # the same split constant as the word artifact; if it no longer regenerates,
    # it is describing text this run cannot vouch for.
    if payload != build_artifact():
        raise SystemExit(
            "refused: vendor/brown_heldout_stream.json does not reproduce from "
            "scripts/generate_heldout_stream.py; the artifact is stale or "
            "hand-edited. No claims written."
        )

    artifact = load_word_artifact()
    try:
        held = validate_heldout_stream_artifact(payload, artifact)
    except ValueError as exc:
        raise SystemExit(f"refused: {exc}. No claims written.") from exc
    # The shared validator proves elementwise token and document-cluster
    # alignment before anything is measured.
    vocabulary = set(artifact.word_freq)
    covered = [token for token in held if token in vocabulary]
    excluded = [token for token in held if token not in vocabulary]

    transition, start = build_letter_bigram_bar(artifact)
    full_pp, full_n = _letter_bigram_perplexity(held, transition, start)
    covered_pp, covered_n = _letter_bigram_perplexity(covered, transition, start)
    excluded_pp, excluded_n = _letter_bigram_perplexity(excluded, transition, start)

    values: dict[str, object] = {
        "stage3_oov_stream_schema": SCHEMA,
        "stage3_oov_held_tokens": len(held),
        "stage3_oov_held_types": payload["provenance"]["held_types"],
        "stage3_oov_covered_tokens": len(covered),
        "stage3_oov_excluded_tokens": len(excluded),
        "stage3_oov_excluded_pct": 100.0 * len(excluded) / len(held),
        "stage3_oov_covered_emissions": covered_n,
        "stage3_oov_excluded_emissions": excluded_n,
        "stage3_oov_full_emissions": full_n,
        "stage3_oov_bar_full_perplexity": full_pp,
        "stage3_oov_bar_covered_perplexity": covered_pp,
        "stage3_oov_bar_excluded_perplexity": excluded_pp,
        # The selection effect, as a ratio, so the prose does not have to divide
        # two other tokens to state the section's own qualification.
        "stage3_oov_selection_ratio": full_pp / covered_pp,
        "stage3_oov_excluded_hardness_ratio": excluded_pp / covered_pp,
        "stage3_oov_covered_mean_length": sum(map(len, covered)) / len(covered),
        "stage3_oov_excluded_mean_length": sum(map(len, excluded)) / len(excluded),
    }
    # Length is the obvious mechanism and it is NOT the mechanism. Recording the
    # two marginal means and calling the difference "length" was a causal claim
    # resting on no control, and the control refutes it: the excluded tokens are
    # harder at every shared length, and matching the two populations on length
    # makes the gap LARGER. The stratified numbers are recorded so the prose
    # cites a measurement rather than an intuition.
    values.update(_length_stratified(covered, excluded, transition, start))

    ledger = load_ledger(REPO_ROOT)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
    )
    print(f"recorded {len(values)} held-out stream claims over {len(held)} tokens")
    print(f"  bar on the covered sublanguage {covered_pp:.4f} vs {full_pp:.4f} on the whole stream")
    return 0


def _length_stratified(
    covered: list[str],
    excluded: list[str],
    transition: np.ndarray,
    start: np.ndarray,
) -> dict[str, object]:
    """Does the coverage effect survive matching the two populations on length?

    The naive reading of the raw gap is that a frequency-ranked lexicon selects
    short words and short words are easy for a letter model. That reading is
    testable and false. Each length bucket with enough tokens on both sides is
    scored separately, and the excluded side is then reweighted to the covered
    side's length distribution, which is the standard way to ask "what would the
    gap be if the two populations had the same lengths". The reported score is
    per emission, not per token, so a length-``L`` bucket receives emission
    weight ``n_L * (L + 1)``. Token weights would average bucket scores under a
    different estimand and would not reconstruct a directly scored population
    when the shared buckets have unequal lengths.

    Returns:
        Labels covering the number of shared buckets, how many of them run the
        excluded side harder, and the length-matched perplexities and gap.
    """
    buckets_covered: dict[int, list[str]] = {}
    buckets_excluded: dict[int, list[str]] = {}
    for token in covered:
        buckets_covered.setdefault(len(token), []).append(token)
    for token in excluded:
        buckets_excluded.setdefault(len(token), []).append(token)

    shared = sorted(
        length
        for length in set(buckets_covered) & set(buckets_excluded)
        if len(buckets_covered[length]) >= _MIN_BUCKET
        and len(buckets_excluded[length]) >= _MIN_BUCKET
    )
    if not shared:
        raise SystemExit(
            "refused: no length bucket holds enough tokens on both sides to "
            "length-match; the stratified claim cannot be made. No claims written."
        )

    harder = 0
    emission_weight_total = sum(len(buckets_covered[length]) * (length + 1) for length in shared)
    matched_covered = 0.0
    matched_excluded = 0.0
    for length in shared:
        covered_pp, _ = _letter_bigram_perplexity(buckets_covered[length], transition, start)
        excluded_pp, _ = _letter_bigram_perplexity(buckets_excluded[length], transition, start)
        harder += int(excluded_pp > covered_pp)
        # Per-emission perplexity counts every letter plus END. Standardizing
        # to the covered token distribution therefore requires the number of
        # covered tokens at this length times its emissions per token.
        weight = len(buckets_covered[length]) * (length + 1) / emission_weight_total
        matched_covered += weight * math.log(covered_pp)
        matched_excluded += weight * math.log(excluded_pp)

    raw_gap = math.log(_letter_bigram_perplexity(excluded, transition, start)[0]) - math.log(
        _letter_bigram_perplexity(covered, transition, start)[0]
    )
    matched_gap = matched_excluded - matched_covered
    return {
        "stage3_oov_length_buckets": len(shared),
        "stage3_oov_length_buckets_excluded_harder": harder,
        "stage3_oov_length_min_bucket": _MIN_BUCKET,
        "stage3_oov_matched_covered_perplexity": math.exp(matched_covered),
        "stage3_oov_matched_excluded_perplexity": math.exp(matched_excluded),
        "stage3_oov_raw_gap_nats": raw_gap,
        "stage3_oov_matched_gap_nats": matched_gap,
        "stage3_oov_gap_change_under_matching_pct": 100.0 * (matched_gap / raw_gap - 1.0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
