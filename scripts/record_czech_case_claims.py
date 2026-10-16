#!/usr/bin/env python3
"""Record the Czech case arm into the claims ledger (decision 77, exploratory).

Thin orchestrator and the single writer for every ``stage4_czech_*`` label.
Computation lives in ``ailm.levels.case`` (the builders and scorers, reused
unmodified) and ``ailm.levels.case_czech`` (the artifact surface).

The prospectively specified question: does the sequence mechanism's margin over the
context-free ``argmax P(case | UPOS)`` comparator survive when the labels are
real morphological case rather than a syntactic heuristic's output? Either
answer is recorded as an exploratory finding; the p-value does not join the
frozen ``SIGNIFICANCE_FAMILY``.

Refusal guards, prospectively specified in decision 77
-------------------------------------------------------
No claims are written if the observation alphabet degenerates to one symbol,
if any held-out UPOS or case value is unseen in training, if the held-out
count falls below the 420-discordant-pair arithmetic's requirement, or if the
artifact's label source is anything but 100% morphological ``feats[Case]``.

Usage
-----
    .venv/bin/python scripts/record_czech_case_claims.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ailm.ledger_lock import ledger_lock_path, single_writer_lock
from ailm.statistics import exact_two_sided_binomial_p

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "stage4_czech_case_arm"
SOURCE = "scripts/record_czech_case_claims.py"

#: English derived-arm discordance rate used by the queue's resolvability
#: arithmetic: 420 pairs at 20.4% discordance needs ~2,060 held-out tokens.
PLANNING_DISCORDANCE_RATE = 0.204


def _mcnemar(model_hits: list[bool], baseline_hits: list[bool]) -> tuple[int, int, float, float]:
    """Exact paired McNemar over aligned position vectors (decision 12).

    Also returns log10 of the p-value, computed in log space from the same
    exact binomial. At this arm's scale the p-value underflows the smallest
    positive double and ``binomtest`` returns exactly 0.0 — which is not a
    probability, so the finite log is recorded beside it rather than letting
    an underflow artifact stand alone as the measurement.
    """
    if len(model_hits) != len(baseline_hits):
        raise ValueError(f"unpaired vectors: {len(model_hits)} vs {len(baseline_hits)}")
    b = sum(1 for m, s in zip(model_hits, baseline_hits, strict=True) if m and not s)
    c = sum(1 for m, s in zip(model_hits, baseline_hits, strict=True) if s and not m)
    n = b + c
    if n == 0:
        return b, c, 1.0, 0.0
    p = exact_two_sided_binomial_p(b, c)
    # Two-sided exact tail in log space: p0 = 0.5 is symmetric, so the
    # two-sided p is twice the smaller tail (capped at 1). Computed with
    # lgamma because scipy's logcdf also underflows at this scale.
    tail = min(b, c)
    log_top = math.lgamma(n + 1) - math.lgamma(tail + 1) - math.lgamma(n - tail + 1)
    log_top += n * math.log(0.5)
    series = ratio = 1.0
    for k in range(tail, 0, -1):
        ratio *= k / (n - k + 1)
        series += ratio
        if ratio < 1e-17 * series:
            break
    log_tail = log_top + math.log(series)
    log10_p = min(math.log(2.0) + log_tail, 0.0) / math.log(10.0)
    return b, c, p, log10_p


def measure() -> dict[str, float | int]:
    """Every Czech-arm quantity, guarded by the prospectively specified refusals."""
    from collections import Counter
    from typing import cast

    from ailm.levels.case import CaseArtifact
    from ailm.levels.case_czech import (
        LABEL_SOURCE_MORPHOLOGY_PCT,
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        czech_case_accuracy,
        decode_case_positions,
        load_czech_case_artifact,
    )
    from ailm.statistics import minimum_discordant_pairs_for_equivalence

    artifact = load_czech_case_artifact()
    # The Czech artifact intentionally implements the exact data surface the
    # shared English builders consume.  The cast documents that structural
    # reuse for the type checker; no data or behavior is converted here.
    shared_artifact = cast(CaseArtifact, artifact)

    # ── Refusal guards (decision 77). Each aborts before any claim is written.
    if artifact.n_obs < 2:
        raise SystemExit(
            f"Czech arm refused: observation alphabet has size {artifact.n_obs} — "
            f"the English gold degeneracy this arm exists to escape. No claims written."
        )
    train_upos = {u for seq in artifact.train for u, _ in seq}
    train_cases = {c for seq in artifact.train for _, c in seq}
    test_upos = {u for seq in artifact.test for u, _ in seq}
    test_cases = {c for seq in artifact.test for _, c in seq}
    if test_upos - train_upos or test_cases - train_cases:
        raise SystemExit(
            f"Czech arm refused: held-out symbols unseen in training "
            f"(UPOS {sorted(test_upos - train_upos)}, cases "
            f"{sorted(test_cases - train_cases)}). No claims written."
        )
    min_pairs = minimum_discordant_pairs_for_equivalence()
    tokens_needed = math.ceil(min_pairs / PLANNING_DISCORDANCE_RATE)
    if artifact.n_test_tokens < tokens_needed:
        raise SystemExit(
            f"Czech arm refused: {artifact.n_test_tokens} held-out tokens against the "
            f"{tokens_needed} the {min_pairs}-pair arithmetic requires. No claims written."
        )
    morphology_pct = artifact.provenance.get("label_source_morphology_pct")
    if morphology_pct != LABEL_SOURCE_MORPHOLOGY_PCT:
        raise SystemExit(
            f"Czech arm refused: label source is {morphology_pct}% morphology, not 100%. "
            f"This arm exists because English gold covered 25% of scored tokens. "
            f"No claims written."
        )

    # ── The registered comparison: same builders as the English arm.
    emission = build_case_emission(shared_artifact)
    transition = build_case_transition(shared_artifact)
    start = build_case_start(shared_artifact)
    model_hits = decode_case_positions(shared_artifact, emission, transition, start)
    baseline_hits = baseline_case_positions(shared_artifact)
    if not len(model_hits) == len(baseline_hits) == artifact.n_test_tokens:
        raise SystemExit(
            f"Czech arm refused: scored positions ({len(model_hits)} model, "
            f"{len(baseline_hits)} baseline) do not align with the "
            f"{artifact.n_test_tokens} held-out tokens. No claims written."
        )
    b, c, p, log10_p = _mcnemar(model_hits, baseline_hits)

    model_pct = 100.0 * czech_case_accuracy(model_hits)
    baseline_pct = 100.0 * czech_case_accuracy(baseline_hits)

    # ── Descriptive context: the informativeness the feasibility gate measured.
    pairs = [(u, case) for seq in artifact.test for u, case in seq]
    total = len(pairs)
    joint = Counter(pairs)
    upos_marginal = Counter(u for u, _ in pairs)
    case_marginal = Counter(case for _, case in pairs)
    h_case = -sum((n / total) * math.log(n / total) for n in case_marginal.values())
    h_case_given_upos = -sum(
        (n / total) * math.log(n / upos_marginal[u]) for (u, _), n in joint.items()
    )
    majority = 100.0 * case_marginal.most_common(1)[0][1] / total

    # A context-free predictor fitted on the TEST labels themselves: the best
    # any member of the comparator class could do (as in the English arm's
    # ``cf_oracle``). The model exceeding it means the sequence mechanism beats
    # every context-free predictor of UPOS, not merely the fitted bar.
    from collections import defaultdict

    per_upos: dict[str, Counter] = defaultdict(Counter)
    for u, case in pairs:
        per_upos[u][case] += 1
    cf_oracle = 100.0 * sum(cnt.most_common(1)[0][1] for cnt in per_upos.values()) / total

    return {
        "stage4_czech_train_sequences": len(artifact.train),
        "stage4_czech_train_documents": len(artifact.train_document_ids),
        "stage4_czech_train_tokens": sum(len(s) for s in artifact.train),
        "stage4_czech_test_sequences": len(artifact.test),
        "stage4_czech_test_documents": artifact.n_test_documents,
        "stage4_czech_test_tokens": artifact.n_test_tokens,
        "stage4_czech_cases_present": len(artifact.cases_present),
        "stage4_czech_upos_alphabet_size": artifact.n_obs,
        "stage4_czech_label_source_morphology_pct": LABEL_SOURCE_MORPHOLOGY_PCT,
        "stage4_czech_label_entropy_nats": round(h_case, 4),
        "stage4_czech_conditional_entropy_nats": round(h_case_given_upos, 4),
        "stage4_czech_mutual_information_nats": round(h_case - h_case_given_upos, 4),
        "stage4_czech_majority": round(majority, 2),
        "stage4_czech_cf_oracle": round(cf_oracle, 2),
        "stage4_czech_baseline": round(baseline_pct, 2),
        "stage4_czech_accuracy": round(model_pct, 2),
        "stage4_czech_margin_pts": round(model_pct - baseline_pct, 2),
        "stage4_czech_mcnemar_b": b,
        "stage4_czech_mcnemar_c": c,
        "stage4_czech_discordant_n": b + c,
        "stage4_czech_mcnemar_p": float(f"{p:.4g}"),
        "stage4_czech_mcnemar_log10_p": round(log10_p, 1),
        "stage4_czech_min_discordant_pairs": min_pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    from ailm.claims import load_ledger

    recorded = measure()
    ledger = load_ledger(PROJECT_ROOT)

    # Epistemic status per batch (schema v2). The registered comparison is
    # exploratory by decision 77's own declaration — the confirmatory family is
    # frozen. The corpus quantities carry no inferential claim, and the pair
    # minimum is arithmetic that used no measurement.
    exploratory = {
        "stage4_czech_accuracy",
        "stage4_czech_baseline",
        "stage4_czech_margin_pts",
        "stage4_czech_mcnemar_b",
        "stage4_czech_mcnemar_c",
        "stage4_czech_discordant_n",
        "stage4_czech_mcnemar_p",
        "stage4_czech_mcnemar_log10_p",
    }
    analytic = {"stage4_czech_min_discordant_pairs"}
    for status, labels in (
        ("exploratory", exploratory),
        ("analytic", analytic),
        ("descriptive", set(recorded) - exploratory - analytic),
    ):
        batch = {label: recorded[label] for label in labels}
        ledger.record_many(batch, run_id=RUN_ID, source=SOURCE, status=status)

    print(f"Recorded {len(recorded)} Czech-arm claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\nProspectively specified question (decision 77): accuracy "
        f"{recorded['stage4_czech_accuracy']}% vs context-free "
        f"{recorded['stage4_czech_baseline']}% over "
        f"{recorded['stage4_czech_test_tokens']} held-out tokens — margin "
        f"{recorded['stage4_czech_margin_pts']} pts, McNemar "
        f"b={recorded['stage4_czech_mcnemar_b']} / c={recorded['stage4_czech_mcnemar_c']}, "
        f"p = {recorded['stage4_czech_mcnemar_p']} (exploratory; outside the frozen family)."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: two overlapping recorders lose one of the two
    # results with no error on either side, so the lock is not optional.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
