#!/usr/bin/env python3
"""Record the punctuation-type-oracle -> L3 measurement (decision entry 27).

Thin orchestrator. Computation lives in ``ailm.levels.clause_channel``; this
script sequences the measurements, applies the pre-registered refusal guards,
and writes the ledger. Single writer for every ``clause_word_*`` label.

The pre-registration (``docs/decisions.md`` entry 27) predicted that mixing a
clause-derived word statistic into the word prior **harms** the held-out
predictive, with a monotone decline (no interior optimum). The recorder holds
itself to the same standards a positive would: it refuses to record unless the
identity anchor reproduces the shipped Stage 3 perplexity exactly, the anchor
is live (a positive alpha changes the output), the scrambled control is clearly
worse than the real message (so the message is carrying signal, not just mass),
and the outcome is assessed by a paired test over the full held-out token set.

The outcome is **whatever the measurement says** — confirm, refute, or qualify
the pre-registration. This script does not tune anything toward a pass.

Usage
-----
    .venv/bin/python scripts/record_clause_channel_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.clause_channel import (  # noqa: E402
    CLAUSE_TYPES,
    ClauseMessage,
    build_clause_message,
    clustered_clause_word_summaries,
    load_clause_word_artifact,
    sweep_clause_alpha,
)
from ailm.levels.word import build_word_prior, load_word_artifact  # noqa: E402
from ailm.verification.oracle import word_path_perplexity  # noqa: E402

RUN_ID = "clause_word_channel"
SOURCE = "scripts/record_clause_channel_claims.py"

#: Mixing weights. Contains 0.0 plus more than three distinct positives, so the
#: grid guard passes; the shape of the finding is whatever this grid produces.
GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

#: Exactness tolerance for the identity-anchor vs oracle perplexity check.
ANCHOR_TOLERANCE = 1e-6

#: Paired-test significance level (per-token sign test).
ALPHA_LEVEL = 0.01

ARTIFACT = _PROJECT_ROOT / "vendor" / "brown_clause_word_counts.json"
WORD_ARTIFACT = _PROJECT_ROOT / "vendor" / "brown_word_counts.json"


def _refuse(message: str) -> int:
    print(f"REFUSING TO RECORD: {message}", file=sys.stderr)
    return 1


def _perplexity(log_probs: list[float], n_emissions: int) -> float:
    return float(np.exp(-sum(log_probs) / n_emissions))


def _two_sided_binomial_log10(improved: int, degraded: int) -> float:
    """Exact p on a log scale when the ordinary float underflows to zero."""
    discordant = improved + degraded
    if discordant == 0:
        return 0.0
    tail = min(improved, degraded)
    log_lower = float(logsumexp(binom.logpmf(np.arange(tail + 1), discordant, 0.5)))
    return min(0.0, (np.log(2.0) + log_lower) / np.log(10.0))


def main() -> int:
    # ── Guard: grid must contain 0.0 + >=3 distinct positives ────────────────
    positives = {a for a in GRID if a > 0.0}
    if 0.0 not in GRID or len(positives) < 3:
        return _refuse(f"grid {GRID} collapsed; cannot support a finding")

    try:
        bridge = load_clause_word_artifact(ARTIFACT, word_path=WORD_ARTIFACT)
    except (FileNotFoundError, ValueError) as exc:
        return _refuse(str(exc))
    train, test = bridge.train, bridge.test
    word_artifact = load_word_artifact(WORD_ARTIFACT)
    prior = build_word_prior(word_artifact)
    message = build_clause_message(train, word_artifact.word_to_index)

    tokens = [w for s in test for w in s["words"]]
    n_emissions = sum(len(w) + 1 for w in tokens)
    if not tokens:
        return _refuse("held-out token stream is empty")

    real = sweep_clause_alpha(test, prior, message, word_artifact.word_to_index, GRID)
    scrambled = sweep_clause_alpha(
        test, prior, message, word_artifact.word_to_index, GRID, scrambled=True
    )
    typefree_message = ClauseMessage(
        {sentence_type: message.fallback.copy() for sentence_type in CLAUSE_TYPES},
        list(CLAUSE_TYPES),
        message.fallback.copy(),
        message.n_vocab,
    )
    swapped_message = ClauseMessage(
        {
            CLAUSE_TYPES[0]: message.by_type[CLAUSE_TYPES[1]].copy(),
            CLAUSE_TYPES[1]: message.by_type[CLAUSE_TYPES[0]].copy(),
        },
        list(CLAUSE_TYPES),
        message.fallback.copy(),
        message.n_vocab,
    )
    typefree = sweep_clause_alpha(test, prior, typefree_message, word_artifact.word_to_index, GRID)
    swapped_type = sweep_clause_alpha(
        test, prior, swapped_message, word_artifact.word_to_index, GRID
    )

    anchor = real[0.0]

    # ── Guard 1: the identity anchor reproduces the shipped Stage 3 perplexity
    # ── over the IDENTICAL token stream (via the oracle's closed form). ─────
    oracle_ppl, _ = word_path_perplexity(tokens, word_artifact.word_to_index, prior)
    anchor_ppl = _perplexity(anchor, n_emissions)
    if abs(anchor_ppl - oracle_ppl) > ANCHOR_TOLERANCE:
        return _refuse(
            f"alpha=0 perplexity {anchor_ppl:.8f} != oracle {oracle_ppl:.8f} "
            f"(tol {ANCHOR_TOLERANCE}). The sweep anchor does not reproduce the "
            f"uncoupled Stage 3 prior, so every other weight is a comparison "
            f"against a different model."
        )

    # ── Guard 2: the anchor is live — a positive alpha must change output. ──
    if real[0.3] == anchor:
        return _refuse(
            "a positive alpha produced the anchor's exact output, so the "
            "anchor is vacuous and the sweep measures nothing."
        )

    # ── Guard 3: the scrambled control must be clearly worse than the real
    # ── message at a positive weight, proving the message carries information
    # ── rather than merely adding mass to the prior. ─────────────────────────
    probe = max((a for a in GRID if a > 0.0), key=lambda a: sum(real[a]))
    real_ppl_probe = _perplexity(real[probe], n_emissions)
    scrambled_ppl_probe = _perplexity(scrambled[probe], n_emissions)
    if scrambled_ppl_probe <= real_ppl_probe:
        return _refuse(
            f"at alpha={probe} the reversed-axis control reaches "
            f"{scrambled_ppl_probe:.4f}, not worse than the real message's "
            f"{real_ppl_probe:.4f}. The channel carries no word-structure "
            f"information, so the sweep says nothing about clause -> word."
        )

    # ── The sweep, indexed by perplexity ─────────────────────────────────────
    rows = []
    for alpha in GRID:
        ppl = _perplexity(real[alpha], n_emissions)
        # Paired sign test vs the anchor over the full held-out token set:
        # how many tokens the mixed prior made more vs less probable.
        improved = sum(1 for x, a in zip(real[alpha], anchor, strict=True) if x > a)
        degraded = sum(1 for x, a in zip(real[alpha], anchor, strict=True) if x < a)
        rows.append(
            {
                "alpha": alpha,
                "perplexity": ppl,
                "delta": ppl - anchor_ppl,
                "improved": improved,
                "degraded": degraded,
                "log10_p": _two_sided_binomial_log10(improved, degraded),
                "scrambled_ppl": _perplexity(scrambled[alpha], n_emissions),
            }
        )

    best = min(rows, key=lambda r: r["perplexity"])
    interior = 0.0 < best["alpha"] < max(GRID)
    token_clustered, loggain_clustered = clustered_clause_word_summaries(
        test, real[best["alpha"]], real[0.0]
    )

    # ── The outcome, stated honestly ──────────────────────────────────────────
    # Pre-registration predicted harm + monotone decline (no interior optimum).
    confirmed_harm = best["delta"] >= 0.0
    monotone_decline = all(
        rows[i]["perplexity"] >= rows[i - 1]["perplexity"] for i in range(1, len(rows))
    )
    # On the declared perplexity axis the message improves (negative delta).
    # Under the corrected Brown partition the best tested weight is the high
    # grid endpoint, so this sweep does not locate an optimum. The paired token
    # direction agrees, making this a concordant descriptive refutation of the
    # harm prediction.
    # It is not a fresh confirmatory population and does not identify whether
    # the gain is information, mismatch correction, or the insertion rule.
    if confirmed_harm:
        qualifier = "CONFIRMED (harm)"
    elif monotone_decline and best["alpha"] == 0.0:
        qualifier = "REFUTED (painless: no positive alpha even ties the anchor)"
    elif best["improved"] > best["degraded"]:
        qualifier = "REFUTED (predictive and token directions both improve)"
    else:
        qualifier = "HARM NOT CONFIRMED (mixed token and predictive directions)"
    if interior and best["alpha"] > 0.0:
        qualifier += "; interior optimum present"

    # Per-type message fidelity, so the reader knows the message was real.
    type_differences = {}
    for sentence_type, message_row in message.by_type.items():
        type_differences[sentence_type] = round(float(np.max(np.abs(message_row - prior))), 6)

    recorded: dict[str, float | int | str | bool] = {
        "clause_word_eval_tokens": len(tokens),
        "clause_word_eval_emissions": n_emissions,
        "clause_word_source_documents": len(bridge.document_ids),
        "clause_word_train_documents": len(bridge.train_document_indices),
        "clause_word_test_documents": len(bridge.heldout_document_indices),
        "clause_word_document_overlap": len(
            set(bridge.train_document_indices) & set(bridge.heldout_document_indices)
        ),
        "clause_word_source_categories": len(set(bridge.document_categories)),
        "clause_word_train_categories": len(
            {bridge.document_categories[index] for index in bridge.train_document_indices}
        ),
        "clause_word_test_categories": len(
            {bridge.document_categories[index] for index in bridge.heldout_document_indices}
        ),
        "clause_word_train_sentences": len(train),
        "clause_word_test_sentences": len(test),
        "clause_word_anchor_ppl": round(anchor_ppl, 6),
        "clause_word_anchor_ppl_oracle": round(oracle_ppl, 6),
        "clause_word_best_alpha": best["alpha"],
        "clause_word_best_ppl": round(best["perplexity"], 6),
        "clause_word_best_delta_ppl": round(best["delta"], 6),
        "clause_word_best_improved_tokens": best["improved"],
        "clause_word_best_degraded_tokens": best["degraded"],
        "clause_word_best_pair_log10_p": round(best["log10_p"], 6),
        "clause_word_scrambled_ppl_at_best": round(
            _perplexity(scrambled[best["alpha"]], n_emissions), 6
        ),
        # Dispersion of the per-token log-probability difference at the best
        # alpha, which is what this null was missing. The improved/degraded
        # counts bound a SHARE; the reported effect is a MEAN over per-token
        # log-losses, and a mean has no interval without a dispersion. Recording
        # the standard deviation makes the null bindable on its own scale rather
        # than on a proxy one, which is what `TODO.md` asked for.
        #
        # Token direction and aggregate perplexity now both favor the message
        # under the corrected Brown partition. A share interval still cannot
        # express the magnitude or dispersion of a mean log-loss change, so the
        # mean requires its own recorded standard deviation.
        "clause_word_best_delta_sd": round(
            float(np.std(np.asarray(real[best["alpha"]]) - np.asarray(real[0.0]), ddof=1)), 6
        ),
        "clause_word_optimum_is_interior": int(interior),
        "clause_word_confirms_harm": int(confirmed_harm),
        "clause_word_monotone_decline": int(monotone_decline),
        "clause_word_high_alpha_ppl": round(rows[-1]["perplexity"], 6),
        "clause_word_typefree_ppl_at_best": round(
            _perplexity(typefree[best["alpha"]], n_emissions), 6
        ),
        "clause_word_swapped_type_ppl_at_best": round(
            _perplexity(swapped_type[best["alpha"]], n_emissions), 6
        ),
        "clause_word_real_beats_typefree": int(
            best["perplexity"] < _perplexity(typefree[best["alpha"]], n_emissions)
        ),
        "clause_word_real_beats_swapped_type": int(
            best["perplexity"] < _perplexity(swapped_type[best["alpha"]], n_emissions)
        ),
        "clause_word_cluster_unit": "Brown source document",
        "clause_word_cluster_documents": token_clustered.n_clusters,
        "clause_word_cluster_confidence_pct": 100.0 * token_clustered.confidence,
        "clause_word_cluster_resamples": token_clustered.resamples,
        "clause_word_cluster_seed": token_clustered.seed,
        "clause_word_cluster_token_observations": token_clustered.n_observations,
        "clause_word_cluster_token_largest_document": token_clustered.largest_cluster,
        "clause_word_cluster_token_positive_documents": token_clustered.positive_clusters,
        "clause_word_cluster_token_negative_documents": token_clustered.negative_clusters,
        "clause_word_cluster_token_net_share": round(token_clustered.effect, 6),
        "clause_word_cluster_token_ci_low": round(token_clustered.ci_low, 6),
        "clause_word_cluster_token_ci_high": round(token_clustered.ci_high, 6),
        "clause_word_cluster_token_p": float(f"{token_clustered.p_value:.4g}"),
        "clause_word_cluster_token_p_floor": float(f"{token_clustered.p_floor:.4g}"),
        "clause_word_cluster_token_p_at_resolution_floor": (token_clustered.p_at_resolution_floor),
        "clause_word_cluster_loggain_emissions": loggain_clustered.n_observations,
        "clause_word_cluster_loggain_largest_document": loggain_clustered.largest_cluster,
        "clause_word_cluster_loggain_positive_documents": loggain_clustered.positive_clusters,
        "clause_word_cluster_loggain_negative_documents": loggain_clustered.negative_clusters,
        "clause_word_cluster_loggain_nats_per_emission": round(loggain_clustered.effect, 8),
        "clause_word_cluster_loggain_ci_low": round(loggain_clustered.ci_low, 8),
        "clause_word_cluster_loggain_ci_high": round(loggain_clustered.ci_high, 8),
        "clause_word_cluster_loggain_p": float(f"{loggain_clustered.p_value:.4g}"),
        "clause_word_cluster_loggain_p_floor": float(f"{loggain_clustered.p_floor:.4g}"),
        "clause_word_cluster_loggain_p_at_resolution_floor": (
            loggain_clustered.p_at_resolution_floor
        ),
    }
    for sentence_type, diff in type_differences.items():
        recorded[f"clause_word_maxdiff_{sentence_type}"] = diff

    # Record every sweep row so the full table is ledger-backed. The per-alpha
    # counts are integers and the per-alpha log10-p labels are within-comparison
    # diagnostics of the SAME hypothesis (same receiver, same held-out stream),
    # registered in EXCLUDED_FROM_FAMILY exactly as the lexical channel's
    # per-alpha mcnemar_p rows are.
    for result_row in rows:
        a = str(result_row["alpha"]).replace(".", "_")
        recorded[f"clause_word_sweep_alpha_{a}_ppl"] = round(result_row["perplexity"], 6)
        recorded[f"clause_word_sweep_alpha_{a}_delta"] = round(result_row["delta"], 6)
        recorded[f"clause_word_sweep_alpha_{a}_improved"] = result_row["improved"]
        recorded[f"clause_word_sweep_alpha_{a}_degraded"] = result_row["degraded"]
        recorded[f"clause_word_sweep_alpha_{a}_pair_log10_p"] = round(result_row["log10_p"], 6)
        recorded[f"clause_word_sweep_alpha_{a}_scrambled_ppl"] = round(
            result_row["scrambled_ppl"], 6
        )

    ledger = load_ledger(_PROJECT_ROOT)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in recorded]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=recorded, run_id=RUN_ID, source=SOURCE)],
    )

    print(f"Recorded {len(recorded)} clause_word claims:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\n{'alpha':>6} {'ppl':>10} {'delta':>10} {'scrambled':>10} "
        f"{'imp':>8} {'deg':>8} {'log10 p':>12}"
    )
    for result_row in rows:
        mark = " <- best" if result_row["alpha"] == best["alpha"] else ""
        print(
            f"{result_row['alpha']:6.2f} {result_row['perplexity']:10.5f} "
            f"{result_row['delta']:+10.5f} {result_row['scrambled_ppl']:10.5f} "
            f"{result_row['improved']:8d} {result_row['degraded']:8d} "
            f"{result_row['log10_p']:12.3f}{mark}"
        )

    print(
        f"\nEntry 27 outcome: {qualifier}. Anchor perplexity {anchor_ppl:.5f}; "
        f"best {best['perplexity']:.5f} at alpha={best['alpha']} "
        f"(delta {best['delta']:+.5f}); interior={int(interior)}. "
        f"Scrambled control at that weight: "
        f"{recorded['clause_word_scrambled_ppl_at_best']:.5f}."
    )
    if not confirmed_harm:
        print(
            "Per the pre-registration, this qualifies/withdraws the first "
            "clause of the novel-information criterion on this channel: the "
            "predicted redundancy signature did not reproduce. This end-to-end "
            "mixing experiment does not identify conditional information "
            "separately from model mismatch or the insertion rule. The outcome "
            "is recorded regardless of which way it lands."
        )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results with no error anywhere.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
