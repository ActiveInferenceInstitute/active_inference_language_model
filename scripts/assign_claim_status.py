#!/usr/bin/env python3
"""Assign an epistemic status to every ledger claim that lacks one.

Why this script exists
----------------------
The external audit asked for a machine-readable claims table whose entries say
what *kind* of evidence each number is: ``confirmatory``, ``exploratory``,
``descriptive``, or ``analytic`` (schema ``ailm-claims-v2``,
``ailm.claims.CLAIM_STATUSES``). The recorders that produced the existing
entries predate the field, so the classification is applied mechanically from
what each recorder *is*. The rule is authoritative rather than fill-only: the
script also repairs an existing status when a recorder's own declared design or
the evidence assessment changes. ``tests/test_claims_status.py`` fails on both
missing and misclassified entries and names this script as the fix.

The assignment rule
-------------------
By recorder of origin (the entry's ``source``), with a small explicit list of
per-label overrides for closed-form quantities recorded by otherwise-empirical
recorders:

* ``confirmatory`` — ONLY an arm whose immutable registration commit strictly
  predates its outcome commit and whose registered rule was used. There are two:
  latent-state G2P learning (entry 29, ``record_g2p_learning_claims.py``) and the
  ``ud_ewt_test`` replication of the four EWT members of the frozen
  ``SIGNIFICANCE_FAMILY`` (entry 111, ``record_confirmatory_replication.py``),
  which scored a held-back population once under the sentinel
  ``confirmatory_run_completed_ud_ewt_test``. Its verdicts and their sufficient
  counts are confirmatory; its population census, its two controls and its
  descriptive rates are not, and are listed in ``DESCRIPTIVE_LABELS`` so a
  source-wide promotion cannot turn a sample size into an inferential claim.
  ``data/registration_provenance.json`` is machine-checked against Git history.
* ``exploratory`` — test-selected sweep results and diagnostic/subsample arms: the
  lexical, coupling and privileged punctuation-type-oracle→word-prior sweeps
  (their headline rows are selected on the evaluation they are reported on), the
  eval-subsample comparison, the
  budgeted-attention probe, the explicitly exploratory family-disjoint G2P
  diagnostic, the Stage 2 length-matched subset/subsample labels, and the Czech
  document-clustered and wild-bootstrap effect and inferential summaries.  The
  Czech protocol was specified within its development session but has no
  independently timestamped prior artifact, so neither its token-level nor
  clustered result is confirmatory.
* ``analytic`` — closed-form or derived quantities: the complexity ledger
  (operation and byte counts, decision 21) and the entropy floors / uniform
  baselines listed in ``ANALYTIC_LABELS``.
* ``descriptive`` — everything else. Ambiguity resolves DOWNWARD, never up: a
  measured quantity that arguably deserves a stronger tag stays descriptive
  until a decision entry earns it more. The Stage 6 sentence-sign result is
  descriptive: its registered primary estimand was document-level mean log loss,
  whose interval spans zero, while the sign test can pass for a worse scrambled
  model. Plans outside the confirmatory list stay at their mechanical tier.
  A PRE-REGISTRATION heading in ``docs/decisions.md`` does not promote by
  itself: the operative rule is that the registration commit strictly predates
  the outcome commit per ``data/registration_provenance.json``. Entries 25 and
  27 satisfy that rule and remain at their mechanical tier because their
  registered outcomes were not the confirming ones (25's primary interval
  spans zero; 27's predicted harm did not reproduce); entries 22, 28, 36, 38
  and 42 carry the heading but are not in the provenance registry; and entries
  18, 46, 49, 51, 73, 74, 77 and 78 are same-commit in-session specifications.
  Promoting any of them is a deliberate act, not a default.

Idempotent and reconciling: entries already carrying the expected status are
untouched; missing or stale statuses are repaired. Statuses are written through
``ClaimsLedger.assign_status``, which can alter nothing but the ``status`` field,
under the same single-writer lock every recorder takes.

Usage
-----
    .venv/bin/python scripts/assign_claim_status.py
    .venv/bin/python scripts/assign_claim_status.py --check
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.claims import CLAIM_STATUSES, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

#: Recorder of origin -> status. Sources absent from this table are
#: ``descriptive`` — the deliberate downward default for anything ambiguous.
STATUS_BY_SOURCE: dict[str, str] = {
    # Independently timestamped preregistration scored under its registered rule.
    "scripts/record_g2p_learning_claims.py": "confirmatory",  # decision 29
    # The one-run replication of the four ud_ewt_test family members against a
    # held-back population, registered by decision 111 in a commit that strictly
    # precedes its outcome commit.
    "scripts/record_confirmatory_replication.py": "confirmatory",  # decision 111
    # Decision 115: the held-back Stage 6 document-clustered interval. Its
    # registration commit strictly precedes its outcome commit, and it is
    # descriptive anyway, which is the interesting part. Decision 111 had already
    # SPENT this population, so an estimand registered afterwards is registered by
    # someone who knows that population's other results -- here, that its sign test
    # came out strongly positive. That knowledge is exactly what preregistration
    # exists to exclude, so the downward default applies and the entry says so in
    # its own words. Listed explicitly rather than left to the absent-source
    # default: a reader checking why this arm is not confirmatory despite its
    # commit ordering should find the answer here, not in a silence.
    "scripts/record_confirmatory_stage6_clustered_claims.py": "descriptive",  # decision 115
    # These plans and outcomes first appear in the same commit. Their within-session
    # ordering can be documented, but cannot earn preregistered/confirmatory status.
    "scripts/record_l1_context_control_claims.py": "exploratory",  # decision 73
    "scripts/record_fitted_channel_claims.py": "exploratory",  # decision 49
    # Test-selected sweeps and subsample arms.
    "scripts/record_lexical_channel_claims.py": "exploratory",
    "scripts/record_coupling_claims.py": "exploratory",
    "scripts/record_clause_channel_claims.py": "exploratory",
    # Decision 121 registers the case-word influence sweep before any run. Its
    # population is the already-scored EWT dev documents in a new
    # representation, so the outcome is development evidence under a fresh
    # registration — the protocol's own scope warning, recorded as exploratory.
    "scripts/record_case_word_influence_claims.py": "exploratory",  # decision 121
    "scripts/record_subsample_claims.py": "exploratory",
    "scripts/record_budgeted_attention_claims.py": "exploratory",
    # The recorder explicitly labels this split exploratory / diagnostic. Its
    # decision entry documents the diagnostic design; it does not promote the
    # resulting measurements to confirmatory.
    "scripts/record_g2p_famsplit_claims.py": "exploratory",
    # The Czech case-marking treebank arm declares itself exploratory in its
    # own decisions.md entry ("Status unchanged: exploratory" — its p-value
    # sits outside the frozen SIGNIFICANCE_FAMILY).
    "scripts/record_czech_case_claims.py": "exploratory",
    # Closed-form arithmetic: operation and byte counts, never timings.
    "scripts/record_complexity_claims.py": "analytic",
    # Protocol P2's stipulated aligned spelling-to-pronunciation comparator
    # arm (docs/extension_protocols.md §3.4): a development-session offline
    # measurement on the frozen artifact — exploratory by protocol, never
    # confirmatory, whatever the gates' outcome.
    "scripts/record_aligned_g2p_comparator_claims.py": "exploratory",
    # Protocol P2's zero-shot probe arms (docs/extension_protocols.md §3.4):
    # reported descriptively under declared families per §3.5, never
    # confirmatory — listed explicitly rather than left to the default.
    "scripts/record_aligned_g2p_probe_claims.py": "descriptive",
}

CONFIRMATORY_DECISION_BY_SOURCE: dict[str, str] = {
    "scripts/record_g2p_learning_claims.py": "29",
    "scripts/record_confirmatory_replication.py": "111",
}

#: Per-label overrides for closed-form quantities that live inside otherwise
#: empirical recorders. Entropy floors and uniform baselines are computed, not
#: sampled: the uniform bar is the alphabet size, and each floor is the exact
#: conditional entropy of the counts it is derived from.
ANALYTIC_LABELS: frozenset[str] = frozenset(
    {
        "stage1_uniform_baseline",
        "stage2_phoneme_uniform_baseline",
        "stage3_uniform_baseline",
        "bench_ctx_uniform_baseline",
        "bench_synth_uniform_baseline",
        "stage1_bigram_floor_nats",
        "stage1_bigram_floor_perplexity",
        "stage1_heldout_all_position_entropy_nats",
        "stage1_heldout_conditional_entropy_nats",
        "stage1_heldout_floor_perplexity",
        "stage1_heldout_max_gain_nats",
        "stage1_max_bigram_gain_nats",
        "stage2_phoneme_all_position_floor_perplexity",
        "stage2_phoneme_all_position_max_gain_nats",
        "stage2_phoneme_floor_perplexity",
        "stage4_czech_min_discordant_pairs",
        "generation_shipped_information_gain_nats",
        "generation_stochastic_control_information_gain_nats",
        "generation_flat_likelihood_control_information_gain_nats",
        "generation_max_information_gain_possible_nats",
        "generation_word_state_ig_empty_nats",
        "generation_word_state_max_ig_spread_nats",
        "generation_word_state_max_pragmatic_spread_nats",
        "generation_word_state_max_g_spread_nats",
        "generation_word_state_policy_habit_max_abs_dev",
        # The closed-form policy posterior checked against pymdp's own
        # ``infer_policies`` over every current letter and a fixed prefix set:
        # a deterministic comparison of two evaluations, no sampling (decision 95).
        "generation_policy_posterior_pymdp_max_abs_dev",
        "generation_word_state_control_ig_spread_nats",
        "generation_word_state_control_g_spread_nats",
        "generation_word_state_ig_depth1_nats",
        # The P2 comparator arm's analytic uniform floors (the
        # claim:phoneme_count_39 inventory inverse and its word-length
        # power): computed, not sampled, like every other uniform bar here.
        "bench_synthg2p_comparator_uniform_position_floor",
        "bench_synthg2p_comparator_uniform_exact_floor",
        # The P2 probe arms' pre-registered response-contract floor (the
        # artifact's own 39-phone inventory uniform error rate, 1 - 1/39):
        # computed, not sampled.
        "bench_synthg2p_probe_contract_floor",
        "generation_word_state_ig_depth2_nats",
        "generation_word_state_ig_depth3_nats",
        "generation_word_state_ig_depth4_nats",
    }
)

#: Exploratory outcomes written beside descriptive diagnostics by the same
#: recorder. Keeping the list explicit prevents a source-wide promotion from
#: turning sample sizes and weight-selection markers into inferential claims.
EXPLORATORY_LABELS: frozenset[str] = frozenset(
    {
        # Deliberately non-canonical Stage 2 populations. The authoritative
        # Stage 2 recorder owns these alongside the headline measurements, but
        # their subset/subsample selection keeps their status exploratory.
        "stage2_phoneme_subset_perplexity",
        "stage2_phoneme_subset_unigram_baseline",
        "stage2_g2p_subsample_positions",
        "stage2_g2p_subsample_mcnemar_p",
        "stage2_g2p_subsample_mcnemar_b",
        "stage2_g2p_subsample_mcnemar_c",
        "channel_proper_loss_letter_predictive_nats",
        "channel_proper_loss_letter_predictive_realized_nats",
        "channel_proper_loss_lexicon_evidence_nats",
        "channel_proper_loss_lexicon_evidence_realized_nats",
        "channel_proper_loss_lexicon_predictive_nats",
        "channel_proper_loss_lexicon_predictive_realized_nats",
        "channel_proper_loss_geometric_lexicon_nats",
        "channel_proper_loss_geometric_lexicon_realized_nats",
        "channel_proper_loss_letter_evidence_nats",
        "channel_proper_loss_letter_evidence_realized_nats",
        # The operating point and its realized sample summaries were selected
        # by minimizing exact total variation over the reported gamma grid.
        # Deterministic evaluation removes Monte Carlo noise; it does not turn
        # a same-population selection into a confirmatory result.
        "generation_best_gamma",
        "generation_best_membership_pct",
        "generation_best_distinct",
        "generation_best_tv",
        "generation_tv_improvement",
        "generation_sampled_best_gamma",
        "generation_sampled_best_tv",
        "generation_sampled_baseline_tv",
        # Decision 77 and its outcome first enter Git in the same commit.  The
        # cluster recorder also writes design/count diagnostics; only the
        # measured effect and uncertainty summaries are inferential outcomes.
        "cluster_stage4_czech_effect",
        "cluster_stage4_czech_ci_low",
        "cluster_stage4_czech_ci_high",
        "cluster_stage4_czech_p",
        "cluster_stage4_czech_z",
        "cluster_stage4_czech_p_at_resolution_floor",
        # The wild-cluster bootstrap beside it declares the same Czech design,
        # and its recorder marks exactly its two intervals exploratory
        # (wild_czech_exploratory_suffixes = {"ci_low", "ci_high"}); the
        # verdict-agreement marker stays descriptive. Mirroring that rule here
        # keeps the two writers from disagreeing about the same label.
        "wild_stage4_czech_ci_low",
        "wild_stage4_czech_ci_high",
    }
)

#: Descriptive corpus facts emitted by an otherwise exploratory recorder.
DESCRIPTIVE_LABELS: frozenset[str] = frozenset(
    {
        # Frozen method configuration emitted by the exploratory family-split
        # recorder is descriptive, not itself an exploratory outcome.
        "stage2_g2p_split_stride",
        "stage2_g2p_famsplit_prefix_length",
        "stage2_g2p_famsplit_family_stride",
        "stage4_czech_train_sequences",
        "stage4_czech_train_documents",
        "stage4_czech_train_tokens",
        "stage4_czech_test_sequences",
        "stage4_czech_test_documents",
        "stage4_czech_test_tokens",
        "stage4_czech_cases_present",
        "stage4_czech_upos_alphabet_size",
        "stage4_czech_label_source_morphology_pct",
        "stage4_czech_label_entropy_nats",
        "stage4_czech_conditional_entropy_nats",
        "stage4_czech_mutual_information_nats",
        "stage4_czech_majority",
        "stage4_czech_cf_oracle",
        # The decision-111 replication's non-inferential rows. Its recorder is
        # confirmatory by source, so without these the population census, the
        # frozen protocol constants, the two control arms and the descriptive
        # agreement rates would all inherit that status. A sentence count is not
        # a pre-registered prediction, and a control is evidence that the
        # instrument works rather than a result about English.
        "confirmatory_population",
        "confirmatory_registration_decision",
        "confirmatory_frozen_alpha",
        "confirmatory_frozen_family_size",
        "confirmatory_members_scored",
        "confirmatory_census_sentences",
        "confirmatory_census_tokens",
        "confirmatory_census_documents",
        "confirmatory_control_positive_favored",
        "confirmatory_control_positive_opposed",
        "confirmatory_control_positive_p",
        "confirmatory_control_negative_favored",
        "confirmatory_control_negative_opposed",
        "confirmatory_control_negative_p",
        "confirmatory_control_negative_seed",
        "confirmatory_stage4_derived_positions",
        "confirmatory_stage4_derived_documents",
        "confirmatory_stage4_derived_agreement_pct",
        "confirmatory_stage4_derived_baseline_pct",
        "confirmatory_stage4_gold_positions",
        "confirmatory_stage4_gold_documents",
        "confirmatory_stage4_gold_agreement_pct",
        "confirmatory_stage4_gold_baseline_pct",
        "confirmatory_stage5_sentences",
        "confirmatory_stage5_documents",
        "confirmatory_stage5_classification_pct",
        "confirmatory_stage5_majority_pct",
        "confirmatory_stage6_sentences",
        "confirmatory_stage6_documents",
        "confirmatory_stage6_phase_logloss",
        "confirmatory_stage6_unigram_logloss",
        "confirmatory_stage6_improvement_nats",
        # The four per-member direction-agreement markers are post-hoc
        # comparisons against the development results, not pre-registered
        # held-back predictions: decision 111 registers each member's
        # direction and threshold crossing as the prediction, and these
        # markers only report whether the observed discordance pointed the
        # same way as the development one. A comparison against results the
        # registration already knew is not an outcome, so they ride with the
        # census and the controls in the descriptive class. The production
        # ledger still records them as confirmatory until the final
        # executor's regeneration; tests/test_claims_status.py carries the
        # explicit pending map for that interim state.
        "confirmatory_stage4_derived_mcnemar_direction_agrees",
        "confirmatory_stage4_gold_mcnemar_direction_agrees",
        "confirmatory_stage5_mcnemar_direction_agrees",
        "confirmatory_stage6_sign_test_direction_agrees",
    }
)


def status_for(label: str, source: str) -> str:
    """The mechanical assignment rule, one claim at a time."""
    if label in ANALYTIC_LABELS:
        return "analytic"
    if label in EXPLORATORY_LABELS:
        return "exploratory"
    if label in DESCRIPTIVE_LABELS:
        return "descriptive"
    return STATUS_BY_SOURCE.get(source, "descriptive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 listing claims that still lack a status",
    )
    args = parser.parse_args()

    unknown_statuses = set(STATUS_BY_SOURCE.values()) - CLAIM_STATUSES
    if unknown_statuses:
        raise RuntimeError(f"assignment rule contains unknown statuses: {sorted(unknown_statuses)}")
    overlaps = (
        (ANALYTIC_LABELS & EXPLORATORY_LABELS)
        | (ANALYTIC_LABELS & DESCRIPTIVE_LABELS)
        | (EXPLORATORY_LABELS & DESCRIPTIVE_LABELS)
    )
    if overlaps:
        raise RuntimeError(f"labels carry contradictory status overrides: {sorted(overlaps)}")

    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        ledger = load_ledger(PROJECT_ROOT)
        pending = {}
        for label in ledger.all_labels():
            entry = ledger.get(label) or {}
            expected = status_for(label, entry.get("source", ""))
            if entry.get("status") != expected:
                pending[label] = expected

        if args.check:
            if pending:
                print(
                    f"{len(pending)} claim(s) lack or carry the wrong status: "
                    f"{sorted(pending)[:10]}..."
                )
                return 1
            print("every claim carries the status required by the assignment rule")
            return 0

        if not pending:
            print("nothing to do: every claim already matches the assignment rule")
            return 0

        ledger.assign_status(pending)

    counts = collections.Counter(pending.values())
    total = len(ledger)
    print(f"assigned status to {len(pending)} of {total} claims:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
