#!/usr/bin/env python3
"""Audit every stage gate for vacuity and reachability.

Two gate defects have shipped in this project, and they are opposites:

- **Stage 1** demanded ">=2 nats above uniform" — *unreachable*. The conditional
  entropy of the training distribution capped the gain achievable on it at 1.03
  nats, so the corpus does not hold the first-order information the bar demands.
  That bound is under p_train, not under the held-out sample: the finding is a
  miscalibrated threshold, not an impossibility theorem.
- **Stage 2** demanded ">=60% G2P accuracy" — *vacuous*. A context-free per-letter
  lookup table scores ~70%, so a model doing nothing hierarchical passes.

Both were found the same way: measure the **trivial baseline** and the
**achievable ceiling** before accepting a threshold. A gate is only meaningful
when the baseline fails it and the ceiling clears it.

This script makes that check a re-runnable artifact instead of something someone
happens to do twice. It reports, per stage:

    baseline  — what a no-context model scores (must FAIL the gate)
    model     — what the shipped agent scores
    ceiling   — the best comparator measured at this model order, with its
                declared ComparatorType (must PASS the gate)
    verdict   — VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED /
                BRACKETED_BOUND / BRACKETED_REFERENCE
    outcome   — PASS / FAIL (whether the model beats the threshold)
    finding   — where applicable, the expected inferential interpretation of
                that point outcome (for example an unresolved paired gain or a
                non-confirmatory clustered analysis)

Every ceiling carries a declared comparator type, because the verdicts differ
in what they license: UNREACHABLE (an impossibility claim) can only be produced
by a bound-type comparator, while a reference or oracle comparator failing the
bar yields ABOVE_REFERENCE — the threshold sits above that comparator, and
nothing stronger follows.

The speech boundary is reported separately because it is an interface contract,
not a quantitative model gate. Missing ceilings classify as UNAUDITED rather
than being silently assumed reachable.

Run:  .venv/bin/python scripts/audit_gates.py --full
      .venv/bin/python scripts/audit_gates.py --limit 800   (faster, noisier)
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.claims import load_ledger  # noqa: E402 - project path is established above
from ailm.evaluation import (  # noqa: E402
    STAGE1_EVALUATION_TEXT,
    evaluation_population_bound,
)
from ailm.gates import (  # noqa: E402 - project path is established above
    AUDITED_GATE_IDS,
    ComparatorType,
    Direction,
    GateId,
    GateVerdict,
    SmallestEffectSize,
    effect_size_is_of_interest,
    evaluate_gate,
    gate_is_baseline_relative,
    smallest_effect_size,
    stage5_nonpassing_unresolved_is_consistent,
    stage5_widened_auc_above_registered_comparators,
    stage5_widened_passing_resolved_is_consistent,
    stage5_widened_registered_gain_in_interval,
    stage6_positive_nonconfirmatory_is_consistent,
    stage6_positive_replicated_is_consistent,
    verdict_is_bracketed,
)

# pymdp emits a benign "JAX array is being set as static" UserWarning per Agent().
warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

VACUOUS = GateVerdict.VACUOUS
UNREACHABLE = GateVerdict.UNREACHABLE
ABOVE_REFERENCE = GateVerdict.ABOVE_REFERENCE
UNAUDITED = GateVerdict.UNAUDITED

#: Short display tags for the comparator types, printed beside each ceiling.
_TYPE_TAG = {
    ComparatorType.MODEL_CLASS_BOUND: "model-class bound",
    ComparatorType.ANALYTIC_BOUND: "analytic bound",
    ComparatorType.PRIVILEGED_ORACLE: "privileged oracle",
    ComparatorType.REFERENCE_MODEL: "reference model",
}


def _conditional_entropy(bigram_counts: dict, index_of: dict[str, int], n: int) -> float:
    """H(X_t | X_{t-1}) in nats from raw "prev,next" bigram counts."""
    joint = np.zeros((n, n))
    for key, cnt in bigram_counts.items():
        prev, nxt = key.split(",", 1)
        if prev in index_of and nxt in index_of:
            joint[index_of[nxt], index_of[prev]] += cnt
    joint /= joint.sum()
    p_prev = joint.sum(axis=0)
    h = 0.0
    for j in range(n):
        if p_prev[j] <= 0:
            continue
        col = joint[:, j]
        nz = col > 0
        h -= float(np.sum(col[nz] * np.log(col[nz] / p_prev[j])))
    return h


def audit_stage1() -> dict:
    """Stage 1 — letter bigram perplexity vs the context-free unigram bar."""
    from ailm.levels.letter import (
        ALPHA_TO_IDX,
        N_SYMBOLS,
        build_letter_agent,
        build_letter_transition,
        compute_perplexity,
        load_brown_counts,
        normalize_text,
    )

    counts = load_brown_counts(_PROJECT_ROOT / "vendor")
    model = compute_perplexity(
        build_letter_agent(build_letter_transition(counts)), STAGE1_EVALUATION_TEXT
    )

    freq = np.ones(N_SYMBOLS)
    for key, cnt in counts.items():
        _, curr = key.split(",", 1)
        if curr in ALPHA_TO_IDX:
            freq[ALPHA_TO_IDX[curr]] += cnt
    freq /= freq.sum()
    chars = [c for c in normalize_text(STAGE1_EVALUATION_TEXT) if c in ALPHA_TO_IDX]
    baseline = math.exp(-sum(math.log(freq[ALPHA_TO_IDX[c]]) for c in chars) / len(chars))

    # The comparator is measured on the SAME population as the baseline and the
    # model: the held-out passage's transition conditional entropy plus the
    # fixed uniform-start loss, which by [@eq:heldout_bound] lower-bounds what
    # this order-1 model class can score on exactly the all-position population.
    #
    # It used to be the Brown *training* distribution's conditional entropy.
    # That is a legitimate calibration reference and it is not $c_P$ for a gate
    # scored held-out: the row's one measured claim — that the comparator beats
    # the baseline — straddled two populations, which the gate-design section's
    # own checklist forbids. Both quantities are recorded; this row uses the one
    # that matches its baseline.
    ceiling = evaluation_population_bound(STAGE1_EVALUATION_TEXT)["perplexity"]

    return {
        "gate_id": GateId.STAGE1_LETTER,
        "stage": "1 letters (perplexity)",
        "baseline": baseline,
        "baseline_name": "unigram letter freqs",
        "model": model,
        "ceiling": ceiling,
        "ceiling_name": "order-1 entropy floor on the evaluation passage",
        "ceiling_type": ComparatorType.MODEL_CLASS_BOUND,
        # The margin of interest, in the declared SESOI's unit: on a
        # lower-is-better perplexity row the log-scale margin ln(threshold/model)
        # is positive exactly when the model beats the bar (decision 123). Here
        # the threshold is the measured baseline itself.
        "sesoi_margin": math.log(baseline / model),
        "threshold": baseline,
        "threshold_name": "beat unigram bar",
        "lower_is_better": True,
        # The gate as originally specified, for the historical record.
        "original": ">=2 nats over uniform (ppl < 4.06)",
        "original_verdict": UNREACHABLE,
    }


def audit_stage2a() -> dict:
    """Stage 2A — full-dictionary in-sample phoneme sequence protocol."""
    from ailm.levels.phoneme import (
        N_PHONES,
        full_corpus_order1_floor_perplexity,
        full_corpus_perplexity,
        full_corpus_unigram_perplexity,
        load_phoneme_counts,
    )

    counts = load_phoneme_counts(_PROJECT_ROOT / "vendor")
    model, eval_positions = full_corpus_perplexity(counts)
    baseline = full_corpus_unigram_perplexity(counts)
    ceiling, ceiling_positions = full_corpus_order1_floor_perplexity(counts)
    if ceiling_positions != eval_positions:
        raise ValueError(
            "Stage 2A model and exact order-1 bound do not share a denominator: "
            f"{eval_positions} != {ceiling_positions}"
        )

    return {
        "gate_id": GateId.STAGE2_PHONEME,
        "stage": "2A phonemes (perplexity)",
        "baseline": baseline,
        "baseline_name": "unigram phone freqs",
        "model": model,
        "ceiling": ceiling,
        "ceiling_name": "order-1 MLE floor on the same full-dictionary positions",
        "ceiling_type": ComparatorType.MODEL_CLASS_BOUND,
        "threshold": baseline,
        "threshold_name": "beat unigram bar",
        "lower_is_better": True,
        "sesoi_margin": math.log(baseline / model),
        "uniform": float(N_PHONES),
        "eval_positions": eval_positions,
        "evaluation_population": "in-sample full CMUdict dictionary",
        "original": ">=2 nats over uniform would need ppl < 5.28",
        "original_verdict": UNREACHABLE,
    }


def audit_stage2b(limit: int | None) -> dict:
    """Stage 2B — G2P accuracy vs the context-free per-letter argmax bar."""
    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.phoneme import (
        PHONES,
        _iter_pairs,
        baseline_accuracy,
        build_g2p_agent,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        g2p_accuracy,
        g2p_split,
        load_phoneme_counts,
        per_letter_argmax_baseline,
    )

    counts = load_phoneme_counts(_PROJECT_ROOT / "vendor")
    train, test = g2p_split(counts)

    emission = build_g2p_emission(train)
    transition = build_g2p_transition(train)
    start = build_g2p_start(train)

    # All three measurements must see the SAME pairs. Resolve the subsample once
    # through _iter_pairs and pass that list everywhere: mixing a strided model
    # score with a prefix-sliced baseline compares two different word sets and the
    # resulting "margin" is meaningless.
    subset = _iter_pairs(test, limit)
    model = g2p_accuracy(build_g2p_agent(emission, transition, start), subset, limit=None)
    baseline = baseline_accuracy(per_letter_argmax_baseline(train), subset)

    # Ceiling: forward-backward smoothing sees the whole word. pymdp filters, so
    # this is headroom the online architecture deliberately does not claim.
    emis = np.array(emission)[0]
    trans = np.array(transition)[0, :, :, 0]
    d0 = np.array(start).ravel()
    hits = total = 0
    for word, phones in subset:
        idx = [ALPHA_TO_IDX[c] for c in word if c in ALPHA_TO_IDX]
        if len(idx) != len(phones):
            continue
        n_steps, n_states = len(idx), len(PHONES)
        alpha = np.zeros((n_steps, n_states))
        beta = np.ones((n_steps, n_states))
        alpha[0] = d0 * emis[idx[0], :]
        alpha[0] /= alpha[0].sum()
        for t in range(1, n_steps):
            alpha[t] = (trans @ alpha[t - 1]) * emis[idx[t], :]
            alpha[t] /= alpha[t].sum()
        for t in range(n_steps - 2, -1, -1):
            beta[t] = trans.T @ (beta[t + 1] * emis[idx[t + 1], :])
            beta[t] /= beta[t].sum()
        for t in range(n_steps):
            gamma = alpha[t] * beta[t]
            hits += PHONES[int(np.argmax(gamma))] == phones[t]
            total += 1
    ceiling = hits / total

    return {
        "gate_id": GateId.STAGE2_G2P,
        "stage": "2B G2P (accuracy)",
        "baseline": baseline,
        "baseline_name": "per-letter argmax",
        "model": model,
        "ceiling": ceiling,
        "ceiling_name": "forward-backward smoothing (reference, not a bound)",
        "ceiling_type": ComparatorType.REFERENCE_MODEL,
        "threshold": baseline,
        "threshold_name": "beat argmax bar",
        "lower_is_better": False,
        # Higher-is-better accuracy: the margin of interest is the plain
        # percentage-point difference (decision 123). The audit's accuracy is a
        # fraction, so the margin is scaled to points here.
        "sesoi_margin": 100.0 * (model - baseline),
        "original": ">=60% on UD EWT (EWT has no phonemic field)",
        "original_verdict": VACUOUS,
        "eval_positions": total,
        "eval_pairs": len(subset),
        "evaluation_population": (
            "full held-out CMUdict dictionary"
            if limit is None
            else f"evenly spaced bounded subset of {len(subset)} held-out pairs"
        ),
    }


def audit_stage4(population: str = "derived") -> dict:
    """Stage 4 - the case level, scored per held-out token.

    Baseline: context-free ``argmax P(case | UPOS)``. The strongest model lacking
    the sequence mechanism this level claims.

    Ceiling: forward-backward smoothing, a strictly more powerful decode of the
    same model. Note it does NOT bound filtering here - reported as measured
    rather than assumed.

    The declared ">=60%" has population-dependent defects: VACUOUS on gold
    morphology (a constant majority predictor clears it), and ABOVE_REFERENCE
    on the derived oracle (the typed comparator is exceedable, not a model-class
    bound). See docs/decisions.md entries 16 and 75.

    The reported quantity is AGREEMENT, not accuracy: five of the eight case
    roles have no morphological ground truth in English EWT.
    """
    import numpy as np

    from ailm.cases import CaseRole
    from ailm.levels.case import (
        CASE_TO_IDX,
        N_CASES,
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        case_agreement,
        decode_case_positions,
        load_case_artifact,
    )

    artifact = load_case_artifact(population)
    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)

    model = 100.0 * case_agreement(decode_case_positions(artifact, emission, transition, start))
    baseline = 100.0 * case_agreement(baseline_case_positions(artifact))

    hits = total = 0
    for sequence in artifact.test:
        obs = [artifact.upos_to_idx[u] for u, _ in sequence]
        truth = [CASE_TO_IDX[CaseRole[c]] for _, c in sequence]
        length = len(obs)
        alpha = np.zeros((length, N_CASES))
        beta = np.ones((length, N_CASES))
        alpha[0] = start * emission[obs[0], :]
        alpha[0] /= alpha[0].sum()
        for i in range(1, length):
            alpha[i] = (transition @ alpha[i - 1]) * emission[obs[i], :]
            alpha[i] /= alpha[i].sum()
        for i in range(length - 2, -1, -1):
            beta[i] = transition.T @ (beta[i + 1] * emission[obs[i + 1], :])
            beta[i] /= beta[i].sum()
        hits += sum(int(np.argmax(alpha[i] * beta[i]) == truth[i]) for i in range(length))
        total += length

    return {
        "gate_id": GateId.STAGE4_CASE,
        "stage": f"4 case/{population} (agreement)",
        "baseline": baseline,
        "baseline_name": "context-free argmax P(case|UPOS)",
        "model": model,
        "ceiling": 100.0 * hits / total,
        "ceiling_name": "forward-backward smoothing (reference, not a bound)",
        "ceiling_type": ComparatorType.REFERENCE_MODEL,
        "ceiling_degenerate": (100.0 * hits / total) < model,
        "threshold": baseline,
        "threshold_name": "beat context-free bar",
        "lower_is_better": False,
        # Higher-is-better agreement, percentage points (decision 123).
        "sesoi_margin": model - baseline,
        "original": (
            ">=60% vs UD oracle (VACUOUS on gold; on derived it sits above every "
            "measured comparator, ABOVE_REFERENCE — no class bound exists there)"
        ),
        "original_verdict": ABOVE_REFERENCE if population == "derived" else VACUOUS,
        "eval_positions": total,
    }


#: Where ``--json`` writes the machine-readable export. Tracked in git alongside
#: the other publication surfaces of output/reports/, so the committed file is
#: evidence the stdout report alone is not.
GATE_AUDIT_JSON = _PROJECT_ROOT / "output" / "reports" / "gate_audit.json"

#: The retained clause ablation writes HERE and never into the canonical export
#: above.
#:
#: The eight-gate inventory is closed by construction: ``AUDITED_GATE_IDS`` is
#: ``tuple(GateId)``, and three suites plus the cover's honesty test pin either
#: that tuple's length or the exact set of gate ids appearing in
#: ``gate_audit.json``. Decision 114 promoted the widened-observation clause
#: model to the canonical Stage 5, so ``GateId.STAGE5_CLAUSE`` is that arm's row
#: in the canonical export. The case-only arm it displaced is a retained
#: ablation rather than a ninth shipped gate, so it takes a plain string id
#: rather than a ``GateId`` member and a file of its own. The canonical export
#: keeps exactly eight rows, and a machine consumer that wants both reads both.
GATE_AUDIT_ABLATION_JSON = _PROJECT_ROOT / "output" / "reports" / "gate_audit_ablation.json"

#: Gate id of the retained case-only clause ablation (decision 114).
#: Deliberately NOT a :class:`GateId` member.
STAGE5_CASE_ONLY_GATE_ID = "stage5_clause_case_only"


def gate_audit_row(audit: dict, evaluation) -> dict:
    """One JSON-ready object per gate, from the audit dict and its evaluation.

    Enums (``GateId``, ``ComparatorType``, ``GateVerdict``) serialize via
    ``str()`` so the export carries their stable string values, not Python
    reprs; numpy scalars are cast so ``json.dump`` never sees them.
    """
    row = {
        "gate_id": str(audit["gate_id"]),
        "stage": audit["stage"],
        "baseline": float(audit["baseline"]),
        "baseline_name": audit["baseline_name"],
        "model": float(audit["model"]),
        "ceiling": float(audit["ceiling"]),
        "ceiling_name": audit["ceiling_name"],
        "ceiling_type": str(audit["ceiling_type"]),
        "threshold": float(audit["threshold"]),
        "verdict": str(evaluation.verdict),
        "outcome": "PASS" if gate_outcome_passes(audit, evaluation) else "FAIL",
        "baseline_relative": bool(gate_is_baseline_relative(audit["baseline"], audit["threshold"])),
    }
    # The declared smallest effect size of interest (decision 123). Present
    # only where the audit computed a margin of interest in the SESOI's unit —
    # the retained case-only ablation declares none and carries no key.
    if isinstance(audit["gate_id"], GateId):
        sesoi: SmallestEffectSize = smallest_effect_size(audit["gate_id"])
        margin_value = float(audit["sesoi_margin"])
        row["margin_of_interest"] = {
            "value": margin_value,
            "unit": sesoi.unit,
            "smallest_effect_size": float(sesoi.value),
            "clears_sesoi": bool(effect_size_is_of_interest(margin_value, sesoi)),
        }
    if "eval_positions" in audit:
        row["eval_positions"] = int(audit["eval_positions"])
    if "eval_pairs" in audit:
        row["eval_pairs"] = int(audit["eval_pairs"])
    if "evaluation_population" in audit:
        row["evaluation_population"] = str(audit["evaluation_population"])
    if "generation_gate" in audit:
        gate = audit["generation_gate"]
        row["generation_gate"] = {
            "tv_estimator": "exact finite distribution",
            "tv_passes": bool(gate["tv_passes"]),
            "diversity_estimator": "sampled declared run protocol",
            "model_distinct": float(gate["model_distinct"]),
            "baseline_distinct": float(gate["baseline_distinct"]),
            "diversity_fraction": float(gate["diversity_fraction"]),
            "diversity_floor": float(gate["diversity_floor"]),
            "diversity_passes": bool(gate["diversity_passes"]),
            "sampled_words": int(gate["sampled_words"]),
            "composite_passes": bool(gate["composite_passes"]),
        }
    if "expected_outcome" in audit:
        row["finding_status"] = str(audit["expected_outcome"])
    if "cluster_ci_low" in audit:
        row["cluster_primary"] = {
            "ci_low": float(audit["cluster_ci_low"]),
            "ci_high": float(audit["cluster_ci_high"]),
            "p_value": float(audit["cluster_p"]),
            "p_method": str(audit["cluster_p_method"]),
            "unit": "document",
        }
    # Both intervals travel together or the row cannot be read. The Stage 6
    # finding status is a statement about the pair — one spans zero and one does
    # not — so exporting the development bounds alone would leave a consumer
    # unable to check the status it sits beside (decision 115).
    if "held_back_cluster" in audit:
        row["held_back_cluster"] = dict(audit["held_back_cluster"])
    if "oracle_sentence_ci_low" in audit:
        row["oracle_gain"] = {
            "n_sentences": int(audit["oracle_n_sentences"]),
            "n_documents": int(audit["oracle_n_documents"]),
            "n_positions": int(audit["oracle_n_positions"]),
            "mean_nats_per_sentence": float(audit["oracle_mean_nats_per_sentence"]),
            "sentence_ci_low": float(audit["oracle_sentence_ci_low"]),
            "sentence_ci_high": float(audit["oracle_sentence_ci_high"]),
            "pooled_nats_per_position": float(audit["oracle_pooled_nats_per_position"]),
            "position_ci_low": float(audit["oracle_position_ci_low"]),
            "position_ci_high": float(audit["oracle_position_ci_high"]),
            "confidence": float(audit["oracle_confidence"]),
            "resamples": int(audit["oracle_resamples"]),
            "seed": int(audit["oracle_seed"]),
            "resampling_unit": "document",
        }
    return row


def gate_outcome_passes(audit: dict, evaluation) -> bool:
    """Return the declared outcome, including any explicitly conjunctive arm."""
    if "generation_gate" not in audit:
        return bool(evaluation.passes)
    gate = audit["generation_gate"]
    return bool(evaluation.passes and gate["composite_passes"])


def write_gate_audit_json(rows: list[dict], path: Path) -> None:
    """Write the per-gate export, creating output/reports/ if needed."""
    atomic_write_json(path, rows, indent=2)


def expected_finding_is_consistent(audit: dict, evaluation) -> bool:
    """Whether the point gate and any inferential qualifier match the record.

    Gate placement, point outcome, and inferential status are distinct. This
    predicate keeps the retained clause ablation's unresolved outcome and the
    clustered Stage 6 outcome explicit, and gives their negative controls a
    unit-testable seam. The canonical Stage 5 also declares no expected
    outcome — decision 114 held it to the ordinary passing rule on purpose —
    but its gate id selects its promotion guard: the ordinary rule AND
    decision 114's enumerated withdrawal conditions, plus decision 112's
    prediction-1 band whose falsifier carries withdrawal authority. The
    registered prediction-2 AUC relation is recomputed and reported through
    :func:`canonical_stage5_prediction2_diagnostic` — a diagnostic, not part
    of the exit-status conjunction — and everything here is evaluated on
    values the audit recomputed rather than read.
    """
    expected = audit.get("expected_outcome")
    if expected == "nonpassing_oracle_gain_unresolved":
        return stage5_nonpassing_unresolved_is_consistent(
            verdict=evaluation.verdict,
            point_passes=evaluation.passes,
            oracle_sentence_ci_low=audit["oracle_sentence_ci_low"],
            oracle_sentence_ci_high=audit["oracle_sentence_ci_high"],
            oracle_position_ci_low=audit["oracle_position_ci_low"],
            oracle_position_ci_high=audit["oracle_position_ci_high"],
            control_accuracy=audit["control_accuracy"],
            control_mixture_perplexity=audit["control_mixture_perplexity"],
            control_typefree_perplexity=audit["control_typefree_perplexity"],
        )
    if expected == "positive_point_nonconfirmatory":
        return stage6_positive_nonconfirmatory_is_consistent(
            verdict=evaluation.verdict,
            point_passes=evaluation.passes,
            point_improvement=audit["point_improvement"],
            cluster_ci_low=audit["cluster_ci_low"],
            cluster_ci_high=audit["cluster_ci_high"],
            cluster_p=audit["cluster_p"],
        )
    if expected == "positive_point_replicated":
        held_back = audit.get("held_back_cluster")
        if held_back is None:
            return False
        return stage6_positive_replicated_is_consistent(
            verdict=evaluation.verdict,
            point_passes=evaluation.passes,
            point_improvement=audit["point_improvement"],
            cluster_ci_low=audit["cluster_ci_low"],
            cluster_ci_high=audit["cluster_ci_high"],
            held_back_ci_low=held_back["ci_low"],
            held_back_ci_high=held_back["ci_high"],
            held_back_excludes_zero=held_back["excludes_zero"],
            held_back_sign_crosses_threshold=held_back["sign_crosses_frozen_threshold"],
            held_back_sign_direction_agrees=held_back["sign_direction_agrees"],
        )
    if audit.get("gate_id") == GateId.STAGE5_CLAUSE:
        # Decision 114's guard on the canonical widened row. The ordinary rule
        # stays first: a bracketed placement plus a passing outcome, exactly
        # what Stages 1-4 are held to. On top of it, the promotion survives
        # only while decision 114's enumerated withdrawal conditions hold —
        # both document-resampled privileged-label gain intervals exclude
        # zero, the detectability control stays live at the widened alphabet,
        # and both negative controls keep rejecting — plus decision 112's
        # prediction-1 band, whose falsifier ("Withdrawn") carries withdrawal
        # authority of its own. Registered prediction 2 is deliberately NOT
        # part of this conjunction: decision 112 treats its failure as a
        # recorded outcome ("the signal is length or a single fronting
        # indicator"), and decision 114's withdrawal list does not name it, so
        # a violated AUC relation is reported as a diagnostic instead of
        # censoring the exit status. Schema integrity still fails closed:
        # nonfinite or wrong-scale AUC fields raise inside the predicate and
        # remain exit-fatal, while a finite measured AUC that simply loses to
        # a comparator is reportable without being a withdrawal.
        return bool(
            verdict_is_bracketed(evaluation.verdict)
            and gate_outcome_passes(audit, evaluation)
            and stage5_widened_passing_resolved_is_consistent(
                verdict=evaluation.verdict,
                point_passes=evaluation.passes,
                oracle_sentence_ci_low=audit["oracle_sentence_ci_low"],
                oracle_sentence_ci_high=audit["oracle_sentence_ci_high"],
                oracle_position_ci_low=audit["oracle_position_ci_low"],
                oracle_position_ci_high=audit["oracle_position_ci_high"],
                control_accuracy=audit["control_accuracy"],
                control_mixture_perplexity=audit["control_mixture_perplexity"],
                control_typefree_perplexity=audit["control_typefree_perplexity"],
                swapped_blocks_accuracy=audit["swapped_blocks_accuracy"],
                model_classification_accuracy=audit["model_classification_accuracy"],
                leak_control_divergence=audit["leak_control_divergence"],
                decoder_tolerance=audit["decoder_tolerance"],
            )
            and stage5_widened_registered_gain_in_interval(audit["oracle_mean_nats_per_sentence"])
        )
    return verdict_is_bracketed(evaluation.verdict) and gate_outcome_passes(audit, evaluation)


def canonical_stage5_prediction2_diagnostic(audit: dict) -> str | None:
    """Reportable diagnostic for a violated registered prediction 2.

    Decision 112's prediction 2 (classification AUC strictly above both the
    length-only comparator and the first-symbol-is-AUX comparator) is
    registered as a direction whose failure is a recorded outcome — the
    signal would be length or a single fronting indicator — and decision
    114's withdrawal list is exhaustive about what withdraws the promotion:
    the point gate, the two gain intervals, and the negative controls.
    Prediction 1 is the one registered prediction whose falsifier carries
    withdrawal authority ("Withdrawn"), and it stays in the canonical guard.
    The predicate still validates every AUC field (finite, 0-1 scale), so
    corrupted audit data fails closed with an exception rather than passing
    silently; only a finite, valid-scale measurement that loses to a
    comparator produces this reportable diagnostic.
    """
    if stage5_widened_auc_above_registered_comparators(
        auc=audit["auc"],
        auc_length_control=audit["auc_length_control"],
        auc_first_aux_control=audit["auc_first_aux_control"],
    ):
        return None
    return (
        f"  note      registered prediction 2 violated — AUC {audit['auc']:.4f} is not "
        f"strictly above both the length-only comparator "
        f"{audit['auc_length_control']:.4f} and the first-is-AUX comparator "
        f"{audit['auc_first_aux_control']:.4f}; a reportable finding (decision 112 "
        f"records such a failure as an outcome), not a withdrawal condition — "
        f"decision 114's withdrawal list is exhaustive"
    )


def _ceil_str(a: dict) -> str:
    """Format ceiling; warn when smoothing falls below the model (degenerate)."""
    cname = a["ceiling_name"]
    cval = a["ceiling"]
    if a.get("ceiling_degenerate", False):
        # Informational note (mirrors docs/formal.md): "ceiling" here names a
        # MEASURED comparator (forward-backward smoothing of the same model),
        # not an upper bound over the model class. The shipped filtering agent
        # legitimately scores above it, so the gate is still bracketed.
        return (
            f"  ceiling   {cval:10.4f}   {cname}  [{_TYPE_TAG[a['ceiling_type']]}]  \u26a0 BELOW model\n"
            '            note: "ceiling" is a measured comparator, not an upper\n'
            "            bound over the model class; the shipped agent scores\n"
            "            above it and the gate stays bracketed (docs/formal.md)."
        )
    return f"  ceiling   {cval:10.4f}   {cname}  [{_TYPE_TAG[a['ceiling_type']]}]"


def audit_generation(*, gamma: float | None = None) -> dict:
    """Generation — exact distributional fidelity AND realized diversity.

    Baseline: ancestral sampling from the letter bigram. The strongest generator
    with the same letter statistics and no top-down channel.

    Ceiling: 0.0, the distance a generator sampling exactly from the lexicon
    would achieve. Unlike the recognition levels this ceiling is analytic rather
    than measured, because the target distribution is known exactly.

    The first conjunct is deliberately NOT membership. Membership is green by
    construction - the preference floors any letter leaving the lexicon - so it
    is saturated at every usable precision AND at the degenerate one. A metric
    that cannot fail is not a gate. Exact TV alone is also insufficient: the
    high-precision endpoint beats the ancestral TV bar while emitting one word
    forever. The second conjunct therefore requires sampled distinct-word
    diversity to clear its declared baseline-relative floor. See decision 17.

    Args:
        gamma: Precision to audit. ``None`` reads the shipped operating point
            from the ledger; an explicit value is used by the planted
            high-precision negative control.
    """
    import numpy as np

    from ailm.generation import (
        build_generative_letter_agent,
        evaluate_generation_gate,
        exact_ancestral_lexicon_total_variation,
        exact_lexicon_total_variation,
        generate_by_policy_selection,
        generate_by_sampling,
        score_generation,
    )
    from ailm.levels.letter import build_letter_transition, load_brown_counts
    from ailm.levels.word import build_word_prior, load_word_artifact

    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]
    artifact = load_word_artifact()
    prior = build_word_prior(artifact)

    # The realized-run protocol and shipped operating point are ledger-owned.
    # Reading all three prevents the audit from claiming correspondence while
    # silently sampling a different number of words or seeds than the recorder.
    ledger = load_ledger(_PROJECT_ROOT)
    gamma_entry = ledger.get("generation_best_gamma")
    words_entry = ledger.get("generation_words_per_run")
    seeds_entry = ledger.get("generation_seeds")
    missing = [
        label
        for label, entry in (
            ("generation_best_gamma", gamma_entry),
            ("generation_words_per_run", words_entry),
            ("generation_seeds", seeds_entry),
        )
        if entry is None
    ]
    if missing:
        raise SystemExit(
            "generation audit claims are absent: "
            f"{', '.join(missing)}; run scripts/record_generation_claims.py"
        )
    if gamma_entry is None or words_entry is None or seeds_entry is None:
        raise RuntimeError("generation claim presence check failed to narrow entries")
    n_words = int(words_entry["value"])
    n_seeds = int(seeds_entry["value"])
    if n_words <= 0 or n_seeds <= 0:
        raise SystemExit("generation_words_per_run and generation_seeds must both be positive")
    seeds = tuple(range(n_seeds))
    audited_gamma = float(gamma_entry["value"]) if gamma is None else float(gamma)

    baseline_runs = [
        generate_by_sampling(habit, artifact, n_words=n_words, seed=seed) for seed in seeds
    ]
    agent = build_generative_letter_agent(habit, gamma=audited_gamma)
    model_runs = [
        generate_by_policy_selection(agent, artifact, prior, habit, n_words=n_words, seed=s)
        for s in seeds
    ]

    baseline = exact_ancestral_lexicon_total_variation(artifact, prior, habit)
    model = exact_lexicon_total_variation(artifact, prior, habit, audited_gamma)
    baseline_score = score_generation(baseline_runs)
    model_score = score_generation(model_runs)
    composite = evaluate_generation_gate(
        model_tv=model,
        baseline_tv=baseline,
        model_distinct=model_score["distinct_mean"],
        baseline_distinct=baseline_score["distinct_mean"],
    )

    return {
        "gate_id": GateId.GENERATION,
        "stage": "G generation (exact TV + realized diversity)",
        "baseline": baseline,
        "baseline_name": "exact ancestral distribution induced by B",
        "model": model,
        "ceiling": 0.0,
        "ceiling_name": "sampling exactly from the lexicon",
        "ceiling_type": ComparatorType.ANALYTIC_BOUND,
        "threshold": baseline,
        "threshold_name": "beat exact sampling-from-B TV",
        "lower_is_better": True,
        # Lower-is-better TV: the margin of interest is baseline TV minus model
        # TV, positive exactly when the policy generator beats ancestral
        # sampling from B (decision 123).
        "sesoi_margin": baseline - model,
        "original": "none declared; generation was Out of Scope until the ISA changed",
        "original_verdict": UNAUDITED,
        "generation_gate": {
            "tv_passes": composite.tv_passes,
            "model_distinct": composite.model_distinct,
            "baseline_distinct": composite.baseline_distinct,
            "diversity_fraction": composite.diversity_fraction,
            "diversity_floor": composite.diversity_floor,
            "diversity_passes": composite.diversity_passes,
            "sampled_words": n_words * len(seeds),
            "composite_passes": composite.passes,
        },
        "context": (
            f"gamma={audited_gamma:g}; TV uses exact finite distributions; "
            f"diversity uses {len(seeds)} x {n_words} seeded generated words"
        ),
    }


#: Gates for stages with no implementation yet. Declared thresholds are UNCHECKED
#: for vacuity and reachability — that is the point of listing them.
def audit_stage3() -> dict:
    """Stage 3 — the word level, scored per emission over held-out Brown tokens.

    Baseline: a letter-bigram model over the identical 27-symbol alphabet,
    trained on the identical tokens and scored on the identical positions. It is
    the strongest model that does NOT know the lexicon, which is the mechanism
    this level adds.

    Ceiling: an oracle knowing the held-out word distribution exactly. That is
    the best any closed-vocabulary word model can do, so a threshold above it
    would be UNREACHABLE.

    The originally declared "perplexity < 500 on PTB-1k" is VACUOUS by a wide
    margin — the baseline itself clears it — and named a corpus this project does
    not have. See docs/decisions.md entry 14.

    Every number here describes the in-vocabulary sublanguage; the coverage is
    reported alongside so the scope travels with the result.
    """
    import math
    from collections import Counter

    from ailm.levels.word import (
        build_letter_bigram_bar,
        load_word_artifact,
        word_perplexity,
    )
    from ailm.verification.oracle import letter_bigram_perplexity

    artifact = load_word_artifact()
    tokens = artifact.eval_tokens

    model, n_model = word_perplexity(artifact)
    transition, start = build_letter_bigram_bar(artifact)
    baseline, n_baseline = letter_bigram_perplexity(tokens, transition, start)
    if n_model != n_baseline:
        raise ValueError(
            f"model scored {n_model} emissions, baseline {n_baseline}: the two "
            f"numbers describe different position sets and cannot be compared."
        )

    held = Counter(tokens)
    ceiling = math.exp(-sum(math.log(held[w] / len(tokens)) for w in tokens) / n_model)

    return {
        "gate_id": GateId.STAGE3_WORD,
        "stage": "3 word (per-emission ppl)",
        "baseline": baseline,
        "baseline_name": "letter bigram, same positions",
        "model": model,
        "ceiling": ceiling,
        "ceiling_name": "held-out-oracle word model",
        "ceiling_type": ComparatorType.PRIVILEGED_ORACLE,
        "threshold": baseline,
        "threshold_name": "beat letter-bigram bar",
        "lower_is_better": True,
        # Lower-is-better perplexity: log-scale margin ln(threshold/model),
        # with the threshold set to the measured letter-bigram baseline.
        "sesoi_margin": math.log(baseline / model),
        "original": "perplexity < 500 on PTB-1k (baseline already clears it)",
        "original_verdict": VACUOUS,
        "eval_positions": n_model,
    }


def audit_stage5_case_only() -> dict:
    """The RETAINED case-only clause ablation - non-passing, gain unresolved.

    This was the canonical Stage 5 until decision 114 promoted the
    widened-observation arm (:func:`audit_stage5`) in its place. Nothing about
    the measurement changed: the promotion is a naming decision, and this arm
    keeps its gate, its verdict row and its consistency guard exactly as they
    were, so a run in which it silently starts passing is still a flagged
    change. It is audited beside the canonical row and exported to
    :data:`GATE_AUDIT_ABLATION_JSON`, and it is the arm the frozen significance
    family's ``stage5_mcnemar_p`` member and the held-back ``ud_ewt_test``
    replication of decision 111 both refer to.

    Baseline: the type-free case bigram — the strongest model that does not
    use sentence type (decision 2 applied to the new variable). It fails the
    gate definitionally (zero margin against itself).

    Privileged reference: the type ORACLE — every held-out sentence scored under
    its TRUE type's separately fitted chain. It is not a hard bound: finite-data
    smoothing or partial pooling can outperform the true-type chain. Measured,
    the document-disjoint remeasurement places this fitted reference on the
    failing side of the bar, as well as the shipped mixture. Whole-document
    paired-bootstrap intervals on the reference gain span zero on both
    reported scales, so its direction is unresolved at this model order; this
    is not proof of absence or an impossibility result.

    The declared gate ("free energy tracks clause length") is VACUOUS as
    stated: cumulative free energy is monotone in length under ANY model,
    uniform included — each step adds non-negative surprise, so the trivial
    baseline passes and the gate cannot fail.

    Because this arm's finding is non-passing and unresolved (decision 83),
    this audit does not demand a pass. It demands CONSISTENCY: the mixture must
    remain on the failing side, both paired intervals must span zero, and the
    detectability control must still fire. A passing mixture or a resolved gain
    direction flags a changed finding rather than being relabeled silently.
    """
    from ailm.levels.clause import (
        build_type_free_chain,
        build_type_models,
        classify_sentences,
        load_clause_artifact,
        summarize_oracle_gain,
        synthetic_control_corpus,
        type_conditional_perplexities,
    )

    artifact = load_clause_artifact()
    models = build_type_models(artifact.train)
    comparator = build_type_free_chain(artifact.train)
    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    oracle_gain = summarize_oracle_gain(
        artifact.test,
        models,
        comparator,
        clusters=artifact.test_sequence_document_indices,
    )
    classification = classify_sentences(artifact.test, models)

    control_train, control_test = synthetic_control_corpus()
    control_models = build_type_models(control_train)
    control = classify_sentences(control_test, control_models)
    control_ppl = type_conditional_perplexities(
        control_test,
        control_models,
        build_type_free_chain(control_train),
    )

    return {
        "gate_id": STAGE5_CASE_ONLY_GATE_ID,
        "stage": "5 clause, retained case-only ABLATION (non-passing; oracle gain unresolved)",
        "baseline": perplexities["type_free"],
        "baseline_name": "type-free case bigram (the bar itself)",
        "model": perplexities["mixture"],
        "ceiling": perplexities["type_oracle"],
        "ceiling_name": "type ORACLE (true type given)",
        "ceiling_type": ComparatorType.PRIVILEGED_ORACLE,
        "threshold": perplexities["type_free"],
        "threshold_name": "beat type-free bar",
        "lower_is_better": True,
        "original": "free energy tracks clause length (monotone under ANY model)",
        "original_verdict": VACUOUS,
        "eval_positions": perplexities["positions"],
        "expected_outcome": "nonpassing_oracle_gain_unresolved",
        "oracle_n_sentences": oracle_gain.n_sentences,
        "oracle_n_documents": oracle_gain.n_documents,
        "oracle_n_positions": oracle_gain.n_positions,
        "oracle_mean_nats_per_sentence": oracle_gain.mean_nats_per_sentence,
        "oracle_sentence_ci_low": oracle_gain.sentence_ci_low,
        "oracle_sentence_ci_high": oracle_gain.sentence_ci_high,
        "oracle_pooled_nats_per_position": oracle_gain.pooled_nats_per_position,
        "oracle_position_ci_low": oracle_gain.position_ci_low,
        "oracle_position_ci_high": oracle_gain.position_ci_high,
        "oracle_confidence": oracle_gain.confidence,
        "oracle_resamples": oracle_gain.resamples,
        "oracle_seed": oracle_gain.seed,
        "control_accuracy": control["accuracy"],
        "control_mixture_perplexity": control_ppl["mixture"],
        "control_typefree_perplexity": control_ppl["type_free"],
        "context": (
            f"classification {100 * classification['accuracy']:.2f}% vs majority "
            f"{100 * classification['majority_accuracy']:.2f}% "
            f"(interrogative recall {100 * classification['recalls']['interrogative']:.0f}%, "
            f"AUC {classification['auc']:.3f}); detectability control fires at "
            f"{100 * control['accuracy']:.1f}%"
        ),
    }


def audit_stage5() -> dict:
    """Stage 5 - the canonical clause gate, on the widened observation alphabet.

    The widened-observation clause model of decision 112: observation is the
    case role plus one distinguished symbol each for ``VERB`` and ``AUX``.
    Decision 114 made it the canonical Level 5, so this is the row carrying
    ``GateId.STAGE5_CLAUSE`` in the canonical export.

    Unlike the shipped ablation it displaced, this gate is a real measured
    outcome rather than a consistency check, because it was registered before
    the artifact was regenerated and could have failed. It did not: the
    mixture beats the refitted bar, and the audit therefore applies the
    ordinary consistency rule of a passing gate — a bracketed placement plus
    a passing outcome — AND, because decision 114 names this row's
    consistency check as the promotion's guard, the row's withdrawal
    conditions and its registered predictions, asserted on values this run
    recomputed rather than read from the ledger: both document-resampled
    privileged-label gain intervals excluding zero, the detectability control
    live at the widened alphabet, both negative controls rejecting, the gain
    inside the probe-registered interval, and the AUC above both registered
    comparators. A run that loses any of these exits non-zero.

    Two limits belong beside the verdict wherever it is stated, and neither is
    a defect of the measurement. It is a development-session specification,
    registration and outcome in one session, so every label it produces is
    recorded ``descriptive``. And it has no held-back replication and cannot
    acquire one, because decision 111 spent the ``ud_ewt_test`` population and
    its sentinel refuses a second pass.

    A SEPARATE arm from the retained case-only ablation, audited separately.
    This function never reads, writes, or influences
    :func:`audit_stage5_case_only`: that arm keeps its own recorded finding and
    its own consistency guard, whatever this one found.

    Baseline: the type-free pooled chain REFITTED on the identical widened
    stream — decision 2 applied to the new stream rather than carried over from
    the old one. A bar imported from the case-only alphabet would credit the
    widening with the refit. It fails the gate definitionally (zero margin
    against itself).

    Privileged reference: the type ORACLE at the widened alphabet, every
    held-out sentence scored under its true type's separately fitted chain. It
    is a receiver-sensitivity probe and not a hard bound: under misspecification
    a mixture can assign more probability than the label-selected component.

    The registered gate (entry 112) is that the mixture beats the widened
    type-free bar on held-out positions.

    One further limit on comparison. The two arms' populations are nested
    rather than equal — the widened usability rule admits sentences the
    case-only rule excludes — so per-position perplexity is not compared across
    them here or anywhere.
    """
    from ailm.levels.clause import (
        CLAUSE_TYPES,
        binary_auc,
        build_joint_emission,
        build_joint_start,
        build_joint_transition,
        build_type_free_chain,
        build_type_models,
        classify_sentences,
        sequence_log_evidence,
        summarize_oracle_gain,
        synthetic_control_corpus,
        type_conditional_perplexities,
    )
    from ailm.levels.widened import (
        AUX_SYMBOL,
        N_WIDENED_SYMBOLS,
        WIDENED_DECODER_TOLERANCE,
        WIDENED_LEAK_MASS,
        load_widened_clause_artifact,
    )
    from ailm.verification.oracle import forward_filter

    artifact = load_widened_clause_artifact()
    models = build_type_models(artifact.train, n_symbols=N_WIDENED_SYMBOLS)
    comparator = build_type_free_chain(artifact.train, n_symbols=N_WIDENED_SYMBOLS)
    perplexities = type_conditional_perplexities(artifact.test, models, comparator)
    oracle_gain = summarize_oracle_gain(
        artifact.test,
        models,
        comparator,
        clusters=artifact.test_sequence_document_indices,
        n_symbols=N_WIDENED_SYMBOLS,
        symbol_kind="widened-alphabet",
    )
    classification = classify_sentences(artifact.test, models)

    # Decision 114 makes this row's consistency check the promotion's guard, so
    # the two negative controls the recorder refuses on are RECOMPUTED here and
    # asserted by the row guard below — a harness decay that killed either
    # would otherwise let a dead control pass this audit with exit 0 forever.
    swapped_models = {
        "transitions": models["transitions"][::-1].copy(),
        "starts": models["starts"][::-1].copy(),
        "type_prior": models["type_prior"],
    }
    swapped = classify_sentences(artifact.test, swapped_models)
    emission = build_joint_emission(N_WIDENED_SYMBOLS)
    joint_b = build_joint_transition(models)
    joint_d = build_joint_start(models)
    leaked = joint_b.copy()
    for source_type in range(len(CLAUSE_TYPES)):
        target_type = 1 - source_type
        source = slice(source_type * N_WIDENED_SYMBOLS, (source_type + 1) * N_WIDENED_SYMBOLS)
        target = slice(target_type * N_WIDENED_SYMBOLS, (target_type + 1) * N_WIDENED_SYMBOLS)
        leaked[target, source] += WIDENED_LEAK_MASS
    leaked /= leaked.sum(axis=0, keepdims=True)
    leak_divergence = 0.0
    for _sentence_type, sequence in artifact.test[:50]:
        closed, _ = sequence_log_evidence(sequence, models)
        _, leaked_evidence = forward_filter(emission, leaked, joint_d, sequence)
        leak_divergence = max(leak_divergence, abs(closed - leaked_evidence))

    # Registered prediction 2's comparators, computed on the identical
    # held-out sentences and labels as the mixture AUC above.
    test_labels = np.asarray(classification["labels"])
    lengths = np.asarray([len(seq) for _t, seq in artifact.test], dtype=float)
    auc_length = binary_auc(-lengths, test_labels)
    first_is_aux = np.asarray(
        [float(seq[0] == AUX_SYMBOL) for _t, seq in artifact.test], dtype=float
    )
    auc_first_aux = binary_auc(first_is_aux, test_labels)

    control_train, control_test = synthetic_control_corpus(n_symbols=N_WIDENED_SYMBOLS)
    control_models = build_type_models(control_train, n_symbols=N_WIDENED_SYMBOLS)
    control = classify_sentences(control_test, control_models)
    control_ppl = type_conditional_perplexities(
        control_test,
        control_models,
        build_type_free_chain(control_train, n_symbols=N_WIDENED_SYMBOLS),
    )

    return {
        "gate_id": GateId.STAGE5_CLAUSE,
        "stage": "5 clause, widened observation alphabet (decisions 112, 114)",
        "baseline": perplexities["type_free"],
        "baseline_name": "type-free pooled chain REFITTED on the widened stream",
        "model": perplexities["mixture"],
        "ceiling": perplexities["type_oracle"],
        "ceiling_name": "type ORACLE at the widened alphabet (true type given)",
        "ceiling_type": ComparatorType.PRIVILEGED_ORACLE,
        "threshold": perplexities["type_free"],
        "threshold_name": "beat the widened type-free bar",
        "lower_is_better": True,
        # The declared SESOI for this row lives on the registration's own
        # estimand: the privileged-label oracle gain in nats per sentence, not
        # the definitionally hairline point-perplexity margin against the
        # refitted bar (decision 123).
        "sesoi_margin": oracle_gain.mean_nats_per_sentence,
        "original": "registered before regeneration in docs/decisions.md entry 112",
        "original_verdict": "REGISTERED",
        "eval_positions": perplexities["positions"],
        "oracle_n_sentences": oracle_gain.n_sentences,
        "oracle_n_documents": oracle_gain.n_documents,
        "oracle_n_positions": oracle_gain.n_positions,
        "oracle_mean_nats_per_sentence": oracle_gain.mean_nats_per_sentence,
        "oracle_sentence_ci_low": oracle_gain.sentence_ci_low,
        "oracle_sentence_ci_high": oracle_gain.sentence_ci_high,
        "oracle_pooled_nats_per_position": oracle_gain.pooled_nats_per_position,
        "oracle_position_ci_low": oracle_gain.position_ci_low,
        "oracle_position_ci_high": oracle_gain.position_ci_high,
        "oracle_confidence": oracle_gain.confidence,
        "oracle_resamples": oracle_gain.resamples,
        "oracle_seed": oracle_gain.seed,
        "control_accuracy": control["accuracy"],
        "control_mixture_perplexity": control_ppl["mixture"],
        "control_typefree_perplexity": control_ppl["type_free"],
        "auc": classification["auc"],
        "auc_length_control": auc_length,
        "auc_first_aux_control": auc_first_aux,
        "swapped_blocks_accuracy": swapped["accuracy"],
        "model_classification_accuracy": classification["accuracy"],
        "leak_control_divergence": leak_divergence,
        "decoder_tolerance": WIDENED_DECODER_TOLERANCE,
        "leak_mass": WIDENED_LEAK_MASS,
        "context": (
            f"classification {100 * classification['accuracy']:.2f}% vs majority "
            f"{100 * classification['majority_accuracy']:.2f}% "
            f"(interrogative recall "
            f"{100 * classification['recalls']['interrogative']:.0f}%, "
            f"AUC {classification['auc']:.3f}); detectability control fires at "
            f"{100 * control['accuracy']:.1f}%; development-session "
            f"specification, so descriptive and with no held-back replication, "
            f"and the population is nested with the case-only ablation's "
            f"rather than equal to it"
        ),
    }


def audit_stage6() -> dict:
    """Stage 6 - observed-position association, with document-level inference.

    Baseline: the type unigram — the strongest model that does not use
    document position (decision 2 on the new variable). Fails definitionally.

    Ceiling: the phase-conditional distribution FITTED ON TEST — labeled as
    such; like Stage 4's smoothing row it is a reference, not a model.

    The sentence-level point estimate clears the baseline-relative bar, but it
    is not the inferential verdict. Sentences are nested within documents, so
    the primary interval and test resample/flip documents. The channel
    intervention is a separate mechanism checked by the recorder's anchor,
    shift, and scramble guards.

    The finding status is COMPUTED, not declared (decision 115). On the
    development population the primary interval spans zero, and on its own that
    is a positive point estimate with a non-confirmatory primary analysis. The
    held-back ``ud_ewt_test`` population carries two further recorded results:
    decision 111's sign test, and decision 115's interval on the same primary
    estimand. When both point the registered way the status is
    ``positive_point_replicated``; otherwise, or when the held-back interval has
    not been recorded at all, it stays ``positive_point_nonconfirmatory``. The
    status therefore cannot be typed into agreement with a manuscript sentence:
    changing it requires changing a ledger boolean, which requires re-running a
    recorder.
    """
    import numpy as np

    from ailm.clustering import summarize_clustered_difference
    from ailm.levels.narrative import (
        build_phase_models,
        heldout_type_logloss,
        load_narrative_artifact,
        tercile,
    )

    artifact = load_narrative_artifact()
    models = build_phase_models(artifact.train)
    recognition = heldout_type_logloss(artifact.test, models)
    documents = [index for index, document in enumerate(artifact.test) for _ in document]
    clustered = summarize_clustered_difference(recognition["deltas"], documents)

    # Fitted-on-test ceiling, add-1 smoothed the same way.
    from ailm.levels.clause import N_TYPES
    from ailm.levels.narrative import N_PHASES

    counts = np.ones((N_TYPES, N_PHASES))
    for types in artifact.test:
        for index, sentence_type in enumerate(types):
            counts[sentence_type, tercile(index, len(types))] += 1.0
    fitted = counts / counts.sum(axis=0, keepdims=True)
    ceiling_losses = []
    for types in artifact.test:
        for index, sentence_type in enumerate(types):
            ceiling_losses.append(-float(np.log(fitted[sentence_type, tercile(index, len(types))])))

    # ── The held-back arm, read from the ledger rather than recomputed. ──────
    # Decision 115's recorder owns these; recomputing them here would create a
    # second implementation of the same estimand that could silently disagree
    # with the one the manuscript cites.
    ledger = load_ledger(_PROJECT_ROOT)

    def _held_back(label: str):
        entry = ledger.get(label)
        return None if entry is None else entry["value"]

    held_back_low = _held_back("confirmatory_stage6_clustered_ci_low")
    held_back_high = _held_back("confirmatory_stage6_clustered_ci_high")
    held_back_excludes = _held_back("confirmatory_stage6_clustered_excludes_zero")
    held_back_crosses = _held_back("confirmatory_stage6_sign_test_crosses_frozen_threshold")
    held_back_direction = _held_back("confirmatory_stage6_sign_test_direction_agrees")
    held_back_recorded = None not in (
        held_back_low,
        held_back_high,
        held_back_excludes,
        held_back_crosses,
        held_back_direction,
    )

    replicated = held_back_recorded and stage6_positive_replicated_is_consistent(
        verdict=GateVerdict.BRACKETED_REFERENCE,
        point_passes=True,
        point_improvement=recognition["mean_delta"],
        cluster_ci_low=clustered.ci_low,
        cluster_ci_high=clustered.ci_high,
        held_back_ci_low=held_back_low,
        held_back_ci_high=held_back_high,
        held_back_excludes_zero=held_back_excludes,
        held_back_sign_crosses_threshold=held_back_crosses,
        held_back_sign_direction_agrees=held_back_direction,
    )
    expected_outcome = (
        "positive_point_replicated" if replicated else "positive_point_nonconfirmatory"
    )
    if replicated:
        context = (
            "observed-position association only; the development-population "
            f"document-clustered primary interval [{clustered.ci_low:+.4f}, "
            f"{clustered.ci_high:+.4f}] spans zero, and the same estimand on the "
            f"held-back ud_ewt_test population [{held_back_low:+.4f}, "
            f"{held_back_high:+.4f}] does not; the held-back sign test crosses its "
            "frozen threshold in the recorded direction; live pymdp filtering is "
            "recorded separately"
        )
    else:
        context = (
            "observed-position association only; document-clustered primary "
            f"interval [{clustered.ci_low:+.4f}, {clustered.ci_high:+.4f}], "
            f"assumption-conditional sign-flip p={clustered.p_value:.4f}; live pymdp "
            "filtering is recorded separately"
        )

    row: dict = {
        "gate_id": GateId.STAGE6_NARRATIVE,
        "stage": "6 narrative (type log-loss, nats)",
        "baseline": recognition["unigram_mean"],
        "baseline_name": "type unigram (the bar itself)",
        "model": recognition["phase_mean"],
        "ceiling": float(np.mean(ceiling_losses)),
        "ceiling_name": "phase-conditional FITTED ON TEST (reference)",
        "ceiling_type": ComparatorType.PRIVILEGED_ORACLE,
        "threshold": recognition["unigram_mean"],
        "threshold_name": "beat type-unigram bar",
        "lower_is_better": True,
        # The declared SESOI's unit is nats per sentence of type log-loss; the
        # development point improvement is that margin (decision 123).
        "sesoi_margin": recognition["mean_delta"],
        "original": "narrative C constrains clause priors (audited by recorder guards)",
        "original_verdict": UNAUDITED,
        "eval_positions": recognition["n_sentences"],
        "expected_outcome": expected_outcome,
        "point_improvement": recognition["mean_delta"],
        "cluster_ci_low": clustered.ci_low,
        "cluster_ci_high": clustered.ci_high,
        "cluster_p": clustered.p_value,
        "cluster_p_method": "cluster sign-flip sensitivity under joint central symmetry",
        "context": context,
    }
    if held_back_recorded:
        row["held_back_cluster"] = {
            "population": "ud_ewt_test",
            "decision": 115,
            "ci_low": float(held_back_low),
            "ci_high": float(held_back_high),
            "excludes_zero": bool(held_back_excludes),
            "sign_crosses_frozen_threshold": bool(held_back_crosses),
            "sign_direction_agrees": held_back_direction,
            "unit": "document",
        }
    return row


UNAUDITED_STAGES = [
    (
        "7 speech",
        "Whisper -> phoneme -> TTS round trip — measures Whisper/TTS, not this "
        "project's model; re-scoped to the boundary interface contract "
        "(docs/design/speech_boundary.md, tests/test_speech_boundary.py)",
    ),
]


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI contract; omitted limits mean the full population."""
    parser = argparse.ArgumentParser(description=__doc__)
    population = parser.add_mutually_exclusive_group()
    population.add_argument(
        "--limit",
        type=int,
        default=None,
        help="G2P eval pairs (evenly-spaced stride). Larger is slower and tighter. "
        "NOTE: this is NOT a deterministic prefix - the alphabetical head inflates "
        "the G2P margin 3.1x (see docs/decisions.md entry 4).",
    )
    population.add_argument(
        "--full",
        action="store_const",
        const=None,
        dest="limit",
        help="audit the complete held-out G2P population (also the default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="also write the per-gate audit as JSON to "
        f"{GATE_AUDIT_JSON.relative_to(_PROJECT_ROOT)}, reusing the audits "
        "this run already computed",
    )
    parser.add_argument(
        "--widened",
        action="store_true",
        help="ACCEPTED AND IGNORED, kept so a runbook written before decision "
        "114 does not fail. It used to add the widened-observation clause arm "
        "as an extra row; that arm is now the canonical Stage 5 and the "
        "retained case-only ablation is audited beside it, both on every run",
    )
    return parser


def print_g2p_population_note(limit: int | None) -> None:
    """Report the population selected by the same limit passed to the scorer."""
    if limit is None:
        print(
            "NOTE  2B runs on the full held-out CMUdict population.\n"
            "      Significance belongs to the paired gate statistic recorded by\n"
            "      scripts/record_stage2_claims.py, not to comparator placement alone."
        )
    else:
        print(
            "NOTE  2B runs on a bounded subsample, which classifies the gate\n"
            "      (baseline vs reference vs threshold) but does not establish significance.\n"
            "      Use the complete held-out population and paired gate statistic in\n"
            "      scripts/record_stage2_claims.py for the recorded comparison."
        )


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")

    audits = [
        audit_stage1(),
        audit_stage2a(),
        audit_stage2b(args.limit),
        audit_stage3(),
        audit_stage4("derived"),
        audit_generation(),
        audit_stage5(),
        audit_stage6(),
    ]
    observed_ids = tuple(a["gate_id"] for a in audits)
    if observed_ids != AUDITED_GATE_IDS:
        print(
            "Gate inventory drift: expected "
            f"{[gate.value for gate in AUDITED_GATE_IDS]}, got "
            f"{[gate.value for gate in observed_ids]}",
            file=sys.stderr,
        )
        return 2

    if args.widened:
        print(
            "note: --widened is accepted and ignored since decision 114. The "
            "widened-observation clause arm is the canonical Stage 5 row and "
            "the retained case-only ablation is audited beside it on every run.",
            file=sys.stderr,
        )

    # The retained case-only ablation is appended only AFTER the inventory
    # equality check above, so the registered eight-gate inventory is exactly
    # what it was. Decision 114 made it an ablation rather than a ninth shipped
    # gate, and it is written to its own JSON file. It is audited
    # unconditionally: its consistency guard is the thing that keeps its
    # non-passing record honest, and a guard behind a flag is a guard that does
    # not run.
    audits = audits + [audit_stage5_case_only()]

    print("=" * 78)
    print("STAGE GATE AUDIT — is each gate falsifiable AND achievable?")
    print("=" * 78)

    exit_code = 0
    baseline_relative = 0
    json_rows: list[dict] = []
    for a in audits:
        direction: Direction = "lower_is_better" if a["lower_is_better"] else "higher_is_better"
        evaluation = evaluate_gate(
            baseline=a["baseline"],
            model=a["model"],
            ceiling=a["ceiling"],
            threshold=a["threshold"],
            direction=direction,
            ceiling_type=a["ceiling_type"],
        )
        json_rows.append(gate_audit_row(a, evaluation))
        verdict = evaluation.verdict
        arrow = "lower better" if a["lower_is_better"] else "higher better"
        print(f"\n{a['stage']}  [{a['gate_id']}]  ({arrow})")
        print(f"  baseline  {a['baseline']:10.4f}   {a['baseline_name']}")
        print(f"  model     {a['model']:10.4f}")
        print(_ceil_str(a))
        if "uniform" in a:
            print(f"  (uniform  {a['uniform']:10.4f}   crippled comparator)")
        if "eval_positions" in a:
            print(f"  positions {a['eval_positions']:10d}")

        # Placement and model outcome are deliberately separate.
        scalar_passes = evaluation.passes
        passes = gate_outcome_passes(a, evaluation)
        margin = evaluation.margin
        print(
            f"  GATE      {a['threshold_name']}: "
            f"{'PASS' if scalar_passes else 'FAIL'} (margin {margin:+.4f})"
        )
        # The declared smallest effect size of interest, beside the verdict so
        # a hairline positive margin cannot read as a finding (decision 123).
        if isinstance(a["gate_id"], GateId):
            sesoi = smallest_effect_size(a["gate_id"])
            interest_margin = float(a["sesoi_margin"])
            standing = "clears" if effect_size_is_of_interest(interest_margin, sesoi) else "BELOW"
            print(
                f"  EFFECT    {interest_margin:+.4f} {sesoi.unit} vs declared SESOI "
                f"{sesoi.value:g} {sesoi.unit}: {standing}"
            )
        if "generation_gate" in a:
            generation_gate = a["generation_gate"]
            diversity_margin = (
                generation_gate["model_distinct"] - generation_gate["diversity_floor"]
            )
            print(
                "  GATE      retain more than "
                f"{100.0 * generation_gate['diversity_fraction']:.4g}% of ancestral "
                "sampled diversity: "
                f"{'PASS' if generation_gate['diversity_passes'] else 'FAIL'} "
                f"(margin {diversity_margin:+.4f} distinct words)"
            )
        # Every row here sets threshold := baseline, which makes VACUOUS
        # logically unreachable under the strict comparison and leaves the
        # ceiling as the only measured input to the verdict. Print that beside
        # the verdict so the artifact carries the caveat the manuscript and the
        # Lean proof already carry, rather than reading as four live outcomes.
        predetermined = gate_is_baseline_relative(a["baseline"], a["threshold"])
        if predetermined:
            baseline_relative += 1
            placement_reading = (
                "ceiling beats baseline"
                if verdict_is_bracketed(verdict)
                else "measured reference does not beat baseline"
            )
            print(
                f"  VERDICT   {verdict}   (threshold := baseline, so VACUOUS "
                f"could not fire; {placement_reading})"
            )
        else:
            print(f"  VERDICT   {verdict}")
        outcome = "PASS" if passes else "FAIL"
        print(f"  OUTCOME   {outcome}")
        print(f"  original  {a['original']}  ->  was {a['original_verdict']}")
        if "context" in a:
            print(f"  context   {a['context']}")

        if a.get("expected_outcome") == "nonpassing_oracle_gain_unresolved":
            # The document-disjoint correction places both the mixture and
            # privileged fitted reference on the failing side; both
            # whole-document-bootstrap oracle-gain intervals span zero and the
            # planted control fires. Any change is a changed finding, not a
            # status typo.
            if not expected_finding_is_consistent(a, evaluation):
                print(
                    "  !!        case-only ablation expected finding changed — review the "
                    "point gate, paired oracle-gain intervals, and control before recording."
                )
                exit_code = 1
            else:
                print(
                    "  status    ABLATION: gate non-passing and above fitted reference; "
                    "privileged-label gain unresolved on both document-resampled scales; "
                    "replicated non-passing on the held-back ud_ewt_test split"
                )
        elif a.get("expected_outcome") == "positive_point_nonconfirmatory":
            if not expected_finding_is_consistent(a, evaluation):
                print(
                    "  !!        Stage 6 expected finding changed — review the "
                    "point gate and document-clustered primary analysis before recording."
                )
                exit_code = 1
            else:
                print(
                    "  status    positive point estimate; document-clustered "
                    "primary analysis non-confirmatory"
                )
        elif a.get("expected_outcome") == "positive_point_replicated":
            if not expected_finding_is_consistent(a, evaluation):
                print(
                    "  !!        Stage 6 expected finding changed — review the point "
                    "gate, the development-population interval, and the held-back "
                    "interval and sign test before recording."
                )
                exit_code = 1
            else:
                print(
                    "  status    positive point estimate; development-population "
                    "clustered interval spans zero; the same estimand on the "
                    "held-back population does not, and the held-back sign test "
                    "crosses its frozen threshold in the recorded direction"
                )
        elif a.get("gate_id") == GateId.STAGE5_CLAUSE:
            # Decision 114: the canonical row's consistency check is the
            # promotion's guard. A run that loses the ordinary passing rule or
            # any withdrawal condition exits non-zero rather than printing a
            # changed status.
            if not expected_finding_is_consistent(a, evaluation):
                print(
                    "  !!        canonical widened row changed — review the point "
                    "gate, both document-resampled privileged-label gain intervals, "
                    "the detectability and negative controls, and the registered "
                    "prediction interval before recording (decision 114)."
                )
                exit_code = 1
            else:
                print(
                    "  status    CANONICAL LEVEL 5: bracketed pass; both "
                    "document-resampled privileged-label gain intervals exclude "
                    "zero; detectability and negative controls live; registered "
                    "prediction-1 interval holds. Development-session "
                    "specification, descriptive, no held-back replication "
                    "(decisions 111, 112, 114)."
                )
                diagnostic = canonical_stage5_prediction2_diagnostic(a)
                if diagnostic is not None:
                    print(diagnostic)
        elif not expected_finding_is_consistent(a, evaluation):
            exit_code = 1

    if args.json:
        # Written from the audits this run already computed — never a second
        # audit pass, which at full --limit would double a multi-minute run.
        #
        # The two exports are kept apart on purpose. The canonical file's set of
        # gate ids is pinned by the cover's honesty test, which refuses an id it
        # does not know; appending the ablation row there would fail that test
        # and would silently widen an inventory three other suites treat as
        # closed.
        canonical = [row for row in json_rows if row["gate_id"] != STAGE5_CASE_ONLY_GATE_ID]
        ablation_rows = [row for row in json_rows if row["gate_id"] == STAGE5_CASE_ONLY_GATE_ID]
        write_gate_audit_json(canonical, GATE_AUDIT_JSON)
        print(f"\nWrote {len(canonical)} gate rows to {GATE_AUDIT_JSON}")
        if ablation_rows:
            write_gate_audit_json(ablation_rows, GATE_AUDIT_ABLATION_JSON)
            print(f"Wrote {len(ablation_rows)} ablation row to {GATE_AUDIT_ABLATION_JSON}")

    print("\n" + "-" * 78)
    print(
        f"NOTE  {baseline_relative} of {len(audits)} rows set the threshold to the measured\n"
        "      baseline. On those rows VACUOUS cannot fire; BRACKETED means the\n"
        "      comparator beats the baseline, while ABOVE_REFERENCE means the\n"
        "      measured, exceedable comparator does not.\n"
        "      Whether the shipped model beats the baseline is the OUTCOME line,\n"
        "      which is measured and can fail. See manuscript section Gate Design."
    )
    print("-" * 78)
    print_g2p_population_note(args.limit)
    print("-" * 78)
    print("UNAUDITED — declared but never checked for vacuity or reachability:")
    for name, threshold in UNAUDITED_STAGES:
        print(f"  {UNAUDITED:<11} {name:<13} {threshold}")
    print("-" * 78)
    print(
        "\nA gate is bracketed only when the trivial baseline FAILS it and the\n"
        "measured ceiling CLEARS it; BRACKETED_BOUND when that ceiling is a\n"
        "model-class or analytic bound, BRACKETED_REFERENCE when it is a\n"
        "reference model or privileged oracle that could still be exceeded.\n"
        "Audit stage N before building stage N."
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
