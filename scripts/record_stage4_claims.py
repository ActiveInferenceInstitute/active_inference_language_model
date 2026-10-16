#!/usr/bin/env python3
"""Record every Stage 4 number into the claims ledger, from the run that measures it.

Thin orchestrator. Computation lives in ``ailm.levels.case``; this script
sequences the measurements over **both** populations and writes the ledger. It is
the single writer for every ``stage4_*`` label.

Why both populations are measured and neither is merged
-------------------------------------------------------
``gold`` carries real English morphology and covers 3 of the 8 case roles.
``derived`` covers all 8 but its labels come from a documented syntactic
heuristic, so a score against it is *agreement*, never accuracy. Merging them
would produce a number that is part measurement and part heuristic with no way
to tell which part is which.

The pair is also what shows the declared ">=60%" gate carrying two
population-dependent defects — vacuous on gold, above-reference on derived.
That is the Stage 4 finding, and it only exists because both are measured
(``docs/decisions.md`` entries 16 and 75).

Usage
-----
    python scripts/record_stage4_claims.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ailm.gates import STAGE4_CASE_GATE_SESOI_PTS
from ailm.ledger_lock import ledger_lock_path, single_writer_lock
from ailm.statistics import exact_two_sided_binomial_p

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "stage4_case_gate"
SOURCE = "scripts/record_stage4_claims.py"

#: The originally declared Stage 4 threshold, kept so its verdict is re-derived.
DECLARED_GATE_PCT = 60.0


def _mcnemar(model_hits: list[bool], baseline_hits: list[bool]) -> tuple[int, int, float]:
    """Exact paired McNemar over aligned position vectors (decision 12).

    Returns:
        ``(b, c, p)`` — positions where only the model is right, only the
        baseline is right, and the two-sided exact binomial p-value.

    Raises:
        ValueError: If the vectors are not aligned. An unpaired comparison
            cannot establish the claim in either direction.
    """
    if len(model_hits) != len(baseline_hits):
        raise ValueError(f"unpaired vectors: {len(model_hits)} vs {len(baseline_hits)}")
    b = sum(1 for m, s in zip(model_hits, baseline_hits, strict=True) if m and not s)
    c = sum(1 for m, s in zip(model_hits, baseline_hits, strict=True) if s and not m)
    return b, c, exact_two_sided_binomial_p(b, c)


def _stronger_decodes(artifact, emission, transition, start) -> dict[str, float]:
    """Forward-backward and Viterbi over the same model, plus a fitted-on-test bar.

    These are strictly more powerful *inference procedures* over the same
    generative model, so they are the natural ceiling candidates for a
    classification metric (``docs/gate_design.md``). Reported as variants rather
    than assumed to bound filtering — at this level they do not, and saying so
    is the honest reading.
    """
    from collections import Counter, defaultdict

    from ailm.cases import CaseRole
    from ailm.levels.case import CASE_TO_IDX, N_CASES

    fb_hits = viterbi_hits = total = 0
    for sequence in artifact.test:
        obs = [artifact.upos_to_idx[u] for u, _ in sequence]
        truth = [CASE_TO_IDX[CaseRole[c]] for _, c in sequence]
        length = len(obs)

        alpha = np.zeros((length, N_CASES))
        beta = np.ones((length, N_CASES))
        alpha[0] = start * emission[obs[0], :]
        alpha[0] /= alpha[0].sum()
        for t in range(1, length):
            alpha[t] = (transition @ alpha[t - 1]) * emission[obs[t], :]
            alpha[t] /= alpha[t].sum()
        for t in range(length - 2, -1, -1):
            beta[t] = transition.T @ (beta[t + 1] * emission[obs[t + 1], :])
            beta[t] /= beta[t].sum()
        fb_hits += sum(int(np.argmax(alpha[t] * beta[t]) == truth[t]) for t in range(length))

        delta = np.log(start) + np.log(emission[obs[0], :])
        back = np.zeros((length, N_CASES), dtype=int)
        for t in range(1, length):
            scored = delta[None, :] + np.log(transition)
            back[t] = scored.argmax(axis=1)
            delta = scored.max(axis=1) + np.log(emission[obs[t], :])
        path = [int(delta.argmax())]
        for t in range(length - 1, 0, -1):
            path.append(int(back[t][path[-1]]))
        path.reverse()
        viterbi_hits += sum(int(path[t] == truth[t]) for t in range(length))
        total += length

    # A context-free predictor fitted on the TEST labels: the best any model
    # seeing only UPOS could do. It cannot bound a context-aware model, but it
    # bounds the comparator class, which is what makes "unreachable" defensible.
    per_tag: dict[str, Counter] = defaultdict(Counter)
    for sequence in artifact.test:
        for upos, case in sequence:
            per_tag[upos][case] += 1
    oracle = sum(counts.most_common(1)[0][1] for counts in per_tag.values())

    return {
        "smoothing": 100.0 * fb_hits / total,
        "viterbi": 100.0 * viterbi_hits / total,
        "cf_oracle": 100.0 * oracle / total,
    }


def measure(population: str) -> dict[str, float | int]:
    """Every Stage 4 quantity for one population."""
    from collections import Counter

    from ailm.levels.case import (
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

    model_hits = decode_case_positions(artifact, emission, transition, start)
    baseline_hits = baseline_case_positions(artifact)
    b, c, p = _mcnemar(model_hits, baseline_hits)

    truth = Counter(case for seq in artifact.test for _, case in seq)
    majority = 100.0 * truth.most_common(1)[0][1] / artifact.n_test_tokens
    stronger = _stronger_decodes(artifact, emission, transition, start)

    model_pct = 100.0 * case_agreement(model_hits)
    baseline_pct = 100.0 * case_agreement(baseline_hits)
    return {
        f"stage4_{population}_test_tokens": artifact.n_test_tokens,
        f"stage4_{population}_test_sentences": len(artifact.test),
        f"stage4_{population}_train_sentences": len(artifact.train),
        f"stage4_{population}_test_documents": len(artifact.test_document_ids),
        f"stage4_{population}_train_documents": len(artifact.train_document_ids),
        f"stage4_{population}_cases_present": len(artifact.cases_present),
        f"stage4_{population}_majority": round(majority, 2),
        f"stage4_{population}_baseline": round(baseline_pct, 2),
        f"stage4_{population}_agreement": round(model_pct, 2),
        f"stage4_{population}_margin_pts": round(model_pct - baseline_pct, 2),
        f"stage4_{population}_smoothing": round(stronger["smoothing"], 2),
        f"stage4_{population}_viterbi": round(stronger["viterbi"], 2),
        f"stage4_{population}_cf_oracle": round(stronger["cf_oracle"], 2),
        f"stage4_{population}_mcnemar_b": b,
        f"stage4_{population}_mcnemar_c": c,
        f"stage4_{population}_mcnemar_p": float(f"{p:.4g}"),
    }


def measure_oracle_path_strata() -> dict[str, float | int]:
    """Partition the derived gate by which oracle path assigned each label.

    The aggregate ``stage4_derived_margin_pts`` is a mixture, and this is the
    partition that shows over what. The three paths are not three samples of one
    population: they differ in where the label came from, and therefore in what
    agreeing with it means.

    Recorded because the level's admitted margin is carried by one of them. On
    the ``deprel`` path the label is a deterministic function of a syntactic
    relation strongly predictable from the observed UPOS tag; on the
    ``morphology`` path it is real English case marking; on the historical
    ``adposition`` ledger path (``adposition_token`` in the oracle) it is
    attached to the ADP token rather than transferred to the nominal head.

    The ``morphology`` stratum is a PRE-REGISTERED CONTROL, not a result. Its
    discordant-pair count is far below the minimum this project requires to
    resolve an equivalence, so its outcome must be read as *not shown* and never
    as *absent*. Recording it as a control is what stops a small null from being
    quoted as evidence of no effect.

    Returns:
        One margin, both McNemar counts, the pair total, the p-value, and the
        stratum size for each of the three paths.
    """
    from collections import Counter

    from ailm.levels.case import (
        ORACLE_PATHS,
        baseline_case_positions,
        build_case_emission,
        build_case_start,
        build_case_transition,
        decode_case_positions,
        load_case_artifact,
        oracle_path_strata,
    )

    artifact = load_case_artifact("derived")
    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)
    model_hits = decode_case_positions(artifact, emission, transition, start)
    baseline_hits = baseline_case_positions(artifact)
    strata = oracle_path_strata()

    # Alignment is the whole correctness argument here. A stratification that is
    # off by one token still sums to the right total and attributes every result
    # to the wrong path, which is a defect no aggregate check can see.
    if not len(strata) == len(model_hits) == len(baseline_hits):
        raise SystemExit(
            f"Stage 4 refused: oracle-path strata ({len(strata)}) do not align with "
            f"the scored positions ({len(model_hits)}). The reconstruction in "
            f"ailm.levels.case.oracle_path_strata has drifted from the artifact "
            f"generator. No claims written."
        )
    sizes = Counter(strata)
    if set(sizes) - set(ORACLE_PATHS):
        raise SystemExit(
            f"Stage 4 refused: unknown oracle path(s) {set(sizes) - set(ORACLE_PATHS)}."
        )

    # The mechanism behind the partition, measured rather than told as a story.
    # A margin needs the observation to say something about the label. On two of
    # the three paths it says nothing at all, and for a structural reason: the
    # path's own definition fixes the UPOS tag, so the observation alphabet
    # within that stratum is a single symbol and the mutual information is
    # exactly zero rather than small. That is the same degeneracy already
    # recorded for the gold population (`stage4_gold_upos_alphabet_size`),
    # recurring inside the derived one where nothing had looked for it.
    pairs = [(upos, case) for sequence in artifact.test for upos, case in sequence]
    if len(pairs) != len(strata):
        raise SystemExit("Stage 4 refused: label pairs do not align with the strata.")

    # The historical ledger prefix remains ``adposition`` so existing claims and
    # manuscript tokens retain identity. The internal path name states the
    # construct precisely.
    claim_prefix = {
        "morphology": "morphology",
        "deprel": "deprel",
        "adposition_token": "adposition",
    }

    values: dict[str, float | int] = {}
    for path in ORACLE_PATHS:
        prefix = claim_prefix[path]
        subset = [pair for pair, name in zip(pairs, strata, strict=True) if name == path]
        total = len(subset)
        joint = Counter(subset)
        observed = Counter(upos for upos, _case in subset)
        labels = Counter(case for _upos, case in subset)
        entropy = -sum((v / total) * math.log(v / total) for v in labels.values())
        conditional = -sum(
            (v / total) * math.log(v / observed[upos]) for (upos, _c), v in joint.items()
        )
        values.update(
            {
                f"stage4_path_{prefix}_upos_alphabet_size": len(observed),
                f"stage4_path_{prefix}_label_entropy_nats": round(entropy, 6),
                f"stage4_path_{prefix}_mutual_information_nats": round(entropy - conditional, 6),
            }
        )

    for path in ORACLE_PATHS:
        prefix = claim_prefix[path]
        index = [i for i, name in enumerate(strata) if name == path]
        model = [model_hits[i] for i in index]
        baseline = [baseline_hits[i] for i in index]
        b = sum(1 for m, s in zip(model, baseline, strict=True) if m and not s)
        c = sum(1 for m, s in zip(model, baseline, strict=True) if s and not m)
        discordant = b + c
        p = exact_two_sided_binomial_p(b, c)
        model_pct = 100.0 * sum(model) / len(model)
        baseline_pct = 100.0 * sum(baseline) / len(baseline)
        values.update(
            {
                f"stage4_path_{prefix}_tokens": len(index),
                f"stage4_path_{prefix}_agreement": round(model_pct, 2),
                f"stage4_path_{prefix}_baseline": round(baseline_pct, 2),
                f"stage4_path_{prefix}_margin_pts": round(model_pct - baseline_pct, 2),
                f"stage4_path_{prefix}_mcnemar_b": b,
                f"stage4_path_{prefix}_mcnemar_c": c,
                f"stage4_path_{prefix}_discordant_n": discordant,
                f"stage4_path_{prefix}_mcnemar_p": float(f"{p:.4g}"),
            }
        )
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    from ailm.cases import CaseRole
    from ailm.claims import load_ledger
    from ailm.levels.case import load_case_artifact

    recorded: dict[str, float | int] = {}
    for population in ("gold", "derived"):
        recorded.update(measure(population))
    recorded.update(measure_oracle_path_strata())

    # The partition must reproduce the aggregate it partitions. If the strata do
    # not sum back to the recorded token count, one of the two is measuring a
    # different population and the mixture reading below would be wrong.
    strata_tokens = sum(
        recorded[f"stage4_path_{path}_tokens"] for path in ("morphology", "deprel", "adposition")
    )
    if strata_tokens != recorded["stage4_derived_test_tokens"]:
        raise SystemExit(
            f"Stage 4 refused: oracle-path strata sum to {strata_tokens} tokens against "
            f"{recorded['stage4_derived_test_tokens']} in the derived population. "
            f"No claims written."
        )

    gold = load_case_artifact("gold")
    derived = load_case_artifact("derived")
    scope = derived.oracle_scope_audit
    scope_paths = scope["derived_path_tokens"]
    fidelity = scope["heuristic_vs_gold"]
    adposition = scope["adposition_token"]
    if not isinstance(scope_paths, dict) or not isinstance(fidelity, dict):
        raise SystemExit("Stage 4 refused: validated scope-audit mappings were lost.")
    fidelity_paths = fidelity.get("source_path_tokens")
    if not isinstance(fidelity_paths, dict) or not isinstance(adposition, dict):
        raise SystemExit("Stage 4 refused: validated fidelity/adposition mappings were lost.")
    head_labeled = adposition["head_also_labeled_tokens"]
    head_disagreements = adposition["head_role_disagreements"]
    adposition_tokens = adposition["tokens"]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (head_labeled, adposition_tokens)
    ):
        raise SystemExit(
            "Stage 4 refused: adposition head percentages lack a positive denominator."
        )
    recorded.update(
        {
            "stage4_n_cases": len(CaseRole),
            "stage4_cases_without_morphology": len(CaseRole) - len(gold.cases_present),
            "stage4_heuristic_gold_agreement_pct": round(derived.heuristic_gold_agreement_pct, 1),
            "stage4_heuristic_gold_tokens": derived.heuristic_vs_gold["tokens_with_both"],
            "stage4_upos_alphabet_size": derived.n_obs,
            # Size one makes the emission uninformative. The transition can
            # still distinguish positions, so the shipped null is empirical.
            "stage4_gold_upos_alphabet_size": gold.n_obs,
            "stage4_declared_gate": DECLARED_GATE_PCT,
            "stage4_case_gate_sesoi_pts": STAGE4_CASE_GATE_SESOI_PTS,
            "stage4_ud_sentences": derived.provenance["sentences"],
            "stage4_ud_tokens": derived.provenance["tokens"],
            "stage4_ud_documents": derived.provenance["documents"],
            "stage4_document_split_train": derived.provenance["train_documents"],
            "stage4_document_split_test": derived.provenance["test_documents"],
            # Decision 84: sufficient counts delimiting the English fidelity
            # measurement. These cover the complete parsed EWT dev shard before
            # the minimum-sequence filter used for model scoring.
            "stage4_oracle_full_derived_tokens": scope["full_derived_tokens"],
            "stage4_oracle_full_path_morphology_tokens": scope_paths["morphology"],
            "stage4_oracle_full_path_deprel_tokens": scope_paths["deprel"],
            "stage4_oracle_full_path_adposition_token_tokens": scope_paths["adposition_token"],
            "stage4_heuristic_gold_deprel_tokens": fidelity_paths["deprel"],
            "stage4_heuristic_gold_adposition_token_tokens": fidelity_paths["adposition_token"],
            "stage4_adposition_token_morphology_tokens": adposition["morphological_case_tokens"],
            "stage4_adposition_token_head_resolved_tokens": adposition["head_resolved_tokens"],
            "stage4_adposition_token_head_unresolved_tokens": adposition["head_unresolved_tokens"],
            "stage4_adposition_token_head_also_labeled_tokens": head_labeled,
            "stage4_adposition_token_head_role_agreements": adposition["head_role_agreements"],
            "stage4_adposition_token_head_role_disagreements": head_disagreements,
            "stage4_adposition_token_head_also_labeled_pct": round(
                100.0 * head_labeled / adposition_tokens, 2
            ),
            "stage4_adposition_token_head_role_disagreement_pct": round(
                100.0 * head_disagreements / head_labeled, 2
            ),
        }
    )

    ledger = load_ledger(PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE, status="descriptive")

    print(f"Recorded {len(recorded)} Stage 4 claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    gold_maj = recorded["stage4_gold_majority"]
    derived_model = recorded["stage4_derived_agreement"]
    print(
        f"\nThe declared '>={DECLARED_GATE_PCT:.0f}%' has two population-dependent defects:\n"
        f"  gold    — a constant majority predictor scores {gold_maj}%, which "
        f"clears it: VACUOUS.\n"
        f"  derived — the model reaches {derived_model}%, and so does every "
        f"stronger decode of it: above every measured (exceedable) comparator."
    )
    print(
        f"\nGold population carries only {recorded['stage4_gold_test_tokens']} "
        f"held-out tokens, and the paired test finds "
        f"{recorded['stage4_gold_mcnemar_b']}/{recorded['stage4_gold_mcnemar_c']} "
        f"discordant pairs — the model and the context-free baseline agree on "
        f"every one. Measured null, not an omission."
    )
    print(
        f"\nDerived gate: {derived_model}% against "
        f"{recorded['stage4_derived_baseline']}% over "
        f"{recorded['stage4_derived_test_tokens']} positions, McNemar "
        f"p = {recorded['stage4_derived_mcnemar_p']}."
    )
    print(
        f"On the morphologically marked overlap, the syntactic labels agree "
        f"{recorded['stage4_heuristic_gold_agreement_pct']}% of the time. That "
        f"overlap contains {recorded['stage4_heuristic_gold_adposition_token_tokens']} "
        f"token-level adposition-route examples, so it cannot validate that path. "
        f"{recorded['stage4_cases_without_morphology']} of "
        f"{recorded['stage4_n_cases']} cases have no morphological ground truth "
        f"in English EWT at all."
    )
    print(
        "Token-level adposition audit: "
        f"{recorded['stage4_oracle_full_path_adposition_token_tokens']} tokens; "
        f"{head_labeled} also have a separately labeled head; "
        f"{head_disagreements} head/token roles disagree. Existing labels are preserved."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
