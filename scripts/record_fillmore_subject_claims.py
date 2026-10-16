#!/usr/bin/env python3
"""Record the decision-42 Fillmore subject re-run into the claims ledger.

Thin orchestrator and single writer for every ``fillmore_subject_*`` label.
Computation lives in ``scripts/fillmore_subject_rerun.py`` plus the existing
Level 4 builders and scorers; this script sequences the measurements and
refuses to write claims when the registration's own conditions fail.

The registered comparison
-------------------------
Per-token scoring over EWT dev sentences whose single ``nsubj`` token and every
role-bearing candidate sit inside the L4 UPOS observation alphabet. Three paths
in one table, none optional:

* **model**     -- Level 4 forward filtering on UPOS-only observations trained
  on the artifact train split, decoded per position; Fillmore's hierarchy then
  selects the subject among the decoded roles.
* **baseline**  -- leftmost-role positional rule over the heuristic roles: the
  strongest rule using no case knowledge and no sequence model.
* **control**   -- the derived assignment itself: expected near-perfect by
  construction because ``UD_DEPREL_TO_CASE`` maps ``nsubj`` to NOM. Reported
  as a control, never as a result.

Statuses (data/README.md): descriptive throughout. Decision 42's plan enters
Git in the same commit as this recorder and its outcome, so temporal
provenance cannot support a confirmatory tag; the confirmatory family stays
frozen. Corpus and design counts carry no inferential claim.

Usage
-----
    .venv/bin/python scripts/record_fillmore_subject_claims.py
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Direct-script execution puts scripts/ (not the repo root) on sys.path, which
# hides the ``scripts`` namespace package and the ``ailm`` package; put the
# repo root ahead of it, exactly as pytest's pythonpath=["src"] plus rootdir do.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.ledger_lock import (  # noqa: E402 - project path established above
    ledger_lock_path,
    single_writer_lock,
)

RUN_ID = "stage4_fillmore_subject_rerun"
SOURCE = "scripts/record_fillmore_subject_claims.py"

#: Decision 42's entry bar. Asserted against the live statistics function so a
#: change there fails loudly HERE instead of silently re-scoping the arm.
EXPECTED_MIN_PAIRS = 420


def _minimum_pairs() -> int:
    """Resolve the equivalence-minimum function from ``ailm.statistics``."""
    statistics = importlib.import_module("ailm.statistics")
    return int(statistics.minimum_discordant_pairs_for_equivalence())


def collect_sentences():
    """Scorable dev sentences under the registered per-token rule.

    Returns:
        One namedtuple per scorable sentence carrying the gold-derived role
        assignment, UPOS tags, gold subject index, and sentence id. A sentence
        enters only with exactly one ``nsubj`` whose token and every
        role-bearing candidate are inside the alphabet.
    """
    from collections import namedtuple

    from ailm.corpus.ud import case_oracle_assignment, load_ud_ewt_dev
    from ailm.levels.case import load_case_artifact
    from scripts.fillmore_subject_rerun import GOLD_CASE_MAP, sentence_is_alphabet_complete

    Scorable = namedtuple("Scorable", "sent_id upos roles subject_index")

    artifact = load_case_artifact("derived")
    alphabet = frozenset(artifact.upos_tags)
    out = []
    for sentence in load_ud_ewt_dev():
        subjects = [t for t in sentence.tokens if t.deprel == "nsubj"]
        if len(subjects) != 1:
            continue
        subject_index = sentence.tokens.index(subjects[0])
        roles = {}
        upos = {}
        for i, token in enumerate(sentence.tokens):
            upos[i] = token.upos
            role_name = GOLD_CASE_MAP.get(token.case_morphological)
            if role_name is None:
                role = case_oracle_assignment(token)[0]
                role_name = role.name if role is not None else None
            if role_name is not None:
                roles[i] = role_name
        if sentence_is_alphabet_complete(upos, roles, alphabet, subject_index=subject_index):
            out.append(
                Scorable(
                    sent_id=sentence.sent_id,
                    upos=dict(upos),
                    roles=dict(roles),
                    subject_index=subject_index,
                )
            )
    return out


def _forward_picks(artifact, emission, transition, start, upos_seq):
    """Forward-filter argmax picks for one UPOS sequence, or ``None``.

    The same recursion :func:`ailm.levels.case.decode_case_positions` applies,
    but bound to an arbitrary sequence rather than iterating the artifact test
    split: the registered population spans ALL of dev, not the stride-5 share.
    """
    import numpy as np

    try:
        obs_idx = [artifact.upos_to_idx[u] for u in upos_seq]
    except KeyError:
        return None
    belief = start * emission[obs_idx[0], :]
    denom = belief.sum()
    if not np.isfinite(denom) or denom <= 0:
        return None
    belief = belief / denom
    picks = [int(np.argmax(belief))]
    for t in range(1, len(obs_idx)):
        belief = (transition @ belief) * emission[obs_idx[t], :]
        denom = belief.sum()
        if not np.isfinite(denom) or denom <= 0:
            return None
        belief = belief / denom
        picks.append(int(np.argmax(belief)))
    return picks


def score_all(sentences):
    """Model vs baseline vs control results, aligned with ``sentences``.

    Returns:
        Three lists of aggregate-ready entries: ``{"correct": bool}`` where
        the path named a token, ``None`` where it refused.
    """
    from ailm.cases import CaseRole
    from ailm.levels.case import (
        build_case_emission,
        build_case_start,
        build_case_transition,
        fillmore_subject,
        load_case_artifact,
    )

    artifact = load_case_artifact("derived")
    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)

    case_names = [case.name for case in CaseRole]
    model_results = []
    baseline_results = []
    control_results = []

    for item in sentences:
        positions = sorted(item.roles)

        # MODEL path: decode the whole sentence from UPOS only, read the
        # decoded roles at the role-bearing positions, let the hierarchy pick.
        picks = _forward_picks(
            artifact,
            emission,
            transition,
            start,
            [item.upos[i] for i in range(len(item.upos))],
        )
        if picks is not None and positions:
            decoded_names = {p: case_names[picks[p]] for p in positions}
            choice = fillmore_subject(
                [CaseRole[decoded_names[p]] if p in decoded_names else None for p in positions]
            )
            # The hierarchy returns an index into the scored sequence; map it
            # back through `positions`, the candidate set the rule scores over.
            model_ok = choice is not None and positions[choice] == item.subject_index
            model_results.append({"correct": model_ok})
        else:
            model_results.append(None)

        # BASELINE path: leftmost ROLE-BEARING position -- no case knowledge,
        # no sequence model. Registered because a leftmost-TOKEN variant
        # cannot separate subjecthood from sentence-initial position.
        base_choice = positions[0] if positions else None
        baseline_results.append(
            {"correct": base_choice == item.subject_index} if base_choice is not None else None
        )

        # CONTROL path: the derived assignment at these same tokens. NOM lands
        # on the nsubj token whenever deprel->NOM fired, so near-perfect is
        # circularity by construction, never a result.
        control_choice = (
            fillmore_subject([CaseRole[item.roles[p]] for p in positions]) if positions else None
        )
        control_results.append(
            {"correct": control_choice == positions.index(item.subject_index)}
            if control_choice is not None
            else None
        )

    return model_results, baseline_results, control_results


def measure() -> dict[str, float | int]:
    """Every Fillmore-subject-rerun quantity, guarded by the registration."""
    from scripts.fillmore_subject_rerun import entry_bar_verdict, per_token_aggregate

    min_pairs = _minimum_pairs()
    if min_pairs != EXPECTED_MIN_PAIRS:
        raise SystemExit(
            f"Fillmore re-run refused: live equivalence minimum is {min_pairs}, "
            f"not the {EXPECTED_MIN_PAIRS} decision 42 registered. Re-check the "
            f"registration before recording. No claims written."
        )

    sentences = collect_sentences()
    model_r, baseline_r, control_r = score_all(sentences)
    model = per_token_aggregate(model_r)
    baseline = per_token_aggregate(baseline_r)
    control = per_token_aggregate(control_r)

    enters, verdict = entry_bar_verdict(model["predicted"], min_pairs)
    if not enters:
        print(
            f"Fillmore re-run: {model['predicted']} predicted of {model['scored']} "
            f"scorable sentences -- BELOW the {min_pairs}-pair entry bar. The "
            f"count is recorded either way, per decision 42."
        )

    return {
        "fillmore_subject_scorable_sentences": model["scored"],
        "fillmore_subject_predicted_sentences": model["predicted"],
        "fillmore_subject_model_correct": model["correct"],
        "fillmore_subject_model_accuracy": model["accuracy"],
        "fillmore_subject_model_coverage": model["coverage"],
        "fillmore_subject_baseline_correct": baseline["correct"],
        "fillmore_subject_baseline_accuracy": baseline["accuracy"],
        "fillmore_subject_baseline_coverage": baseline["coverage"],
        "fillmore_subject_control_correct": control["correct"],
        "fillmore_subject_control_accuracy": control["accuracy"],
        "fillmore_subject_control_coverage": control["coverage"],
        "fillmore_subject_min_equivalence_pairs": min_pairs,
        "fillmore_subject_enters_main_line": int(enters),
        "fillmore_subject_resolvable": int(verdict == "resolvable"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    from ailm.claims import load_ledger

    recorded = measure()
    ledger = load_ledger(PROJECT_ROOT)

    # Descriptive throughout: the plan enters Git with this recorder and its
    # outcome, so temporal provenance caps this arm below confirmatory, and
    # recorders pass status= directly (data/README.md).
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE, status="descriptive")

    print(f"Recorded {len(recorded)} Fillmore-subject claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in sorted(recorded.items()):
        print(f"  {label:<{width}} = {value}")

    print(
        f"\nDecision-42 one-table comparison over "
        f"{recorded['fillmore_subject_scorable_sentences']} scorable / "
        f"{recorded['fillmore_subject_predicted_sentences']} predicted sentences:\n"
        f"  model           accuracy {recorded['fillmore_subject_model_accuracy']:.4f} "
        f"(coverage {recorded['fillmore_subject_model_coverage']})\n"
        f"  leftmost-role   accuracy {recorded['fillmore_subject_baseline_accuracy']:.4f} "
        f"(coverage {recorded['fillmore_subject_baseline_coverage']})\n"
        f"  derived control accuracy {recorded['fillmore_subject_control_accuracy']:.4f} "
        f"(coverage {recorded['fillmore_subject_control_coverage']})\n"
        f"Entry bar: predicted count against "
        f"{recorded['fillmore_subject_min_equivalence_pairs']} equivalence pairs; "
        f"enters main line = {bool(recorded['fillmore_subject_enters_main_line'])}."
    )
    return 0


if __name__ == "__main__":
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
