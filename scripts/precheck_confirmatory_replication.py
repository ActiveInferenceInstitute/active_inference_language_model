#!/usr/bin/env python3
"""Precheck for the decision-79 confirmatory replication run -- PREP ONLY, no scoring.

Decision 79 (docs/decisions.md) froze the confirmatory replication protocol:
the 16-member ``SIGNIFICANCE_FAMILY`` is to be scored exactly once against
held-back populations, gated by ``confirmatory_run_completed_<population>``
sentinel claims that make a second pass refuse to run. This script is the
pre-flight for that run. It prints its dry-run summary to stdout and writes
nothing unless ``--report PATH`` is given, in which case the same summary is also
published to that path -- it scores nothing, vendors nothing, and creates no
sentinel.

Decision 91 (docs/decisions.md) retired the run by exhaustion for the eight
external_lexicon members: that population is provably empty, because upstream
CMUdict ``HEAD`` is byte-identical to the vendored pin. Decision 111 is the
successor registration for the other four: the ``ud_ewt_test`` members were
scored exactly once against the whole of ``en_ewt-ud-test.conllu``, and the
sentinel ``confirmatory_run_completed_ud_ewt_test`` records that the population
is spent.

**This script therefore exits 1 from now on, and that is the correct outcome.**
Its first check is the sentinel guard, and the sentinel exists. A prep script for
a run that has already happened has nothing left to prepare; what it still does
is fail loudly if anyone tries to prepare a second pass over a spent population,
and catch a drift in the frozen mapping while doing so. Read a nonzero exit here
as "the replication is done", not as a broken gate.

Checks performed:

1. Sentinel refusal guards: no ``confirmatory_run_completed_*`` claim exists
   in the ledger for any registered population, and no such marker file or
   directory exists anywhere in project content -- the filesystem scan prunes
   only the ``.git`` and ``.venv`` trees before descending. A present sentinel
   means the run already happened -- the confirmatory recorder must refuse,
   and so does this precheck. A missing or unreadable ledger is not a pass:
   an inconclusive guard refuses readiness (fail-closed), which is a different
   outcome from the input census below, where a missing optional artifact is
   only a GAP.
2. Input census per population: which required raw/source artifacts are on disk
   and which are missing (a missing artifact is reported as a GAP, not an error
   of this script -- sourcing/re-vendoring is Principal-gated per decision 79).
3. Protocol integrity: ``ailm.statistics.CONFIRMATORY_REPLICATION`` must map
   every family member exactly once, match decision 79's enumerated mapping,
   use only legal dispositions, and keep ``BONFERRONI_ALPHA`` corrected for the
   frozen family size.

Exit codes: 0 = protocol intact (missing data artifacts print as GAP rows but
do not flip the exit code, because re-vendoring without authorization would be
the violation; they block the *run*, not this preparation). 1 = protocol
mismatch, a sentinel already present in the ledger or on disk, or a sentinel
ledger that is missing or unreadable -- an inconclusive guard cannot certify
readiness and is reported as a hard failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _artifact_io import atomic_write_text  # noqa: E402 - project path established above

from ailm.statistics import (  # noqa: E402 - project path established above
    BONFERRONI_ALPHA,
    CONFIRMATORY_DISPOSITIONS,
    CONFIRMATORY_POPULATIONS,
    CONFIRMATORY_REPLICATION,
    FAMILYWISE_ALPHA,
    SIGNIFICANCE_FAMILY,
)

#: Decision 79's census, transcribed verbatim from docs/decisions.md entry 79.
#: The frozen code mapping must equal this exactly.
DECISION_79_ENUMERATION: dict[str, str] = {
    # CMUdict members (8): population external_lexicon.
    "channel_letter_predictive_paired_p": "external_lexicon",
    "channel_lexicon_evidence_paired_p": "external_lexicon",
    "fitted_paired_p": "external_lexicon",
    "geometric_letter_paired_p": "external_lexicon",
    "geometric_lexicon_paired_p": "external_lexicon",
    "coupling_mcnemar_p": "external_lexicon",
    "lexical_best_mcnemar_p": "external_lexicon",
    "stage2_g2p_mcnemar_p": "external_lexicon",
    # UD EWT members (4): population ud_ewt_test.
    "stage4_derived_mcnemar_p": "ud_ewt_test",
    "stage4_gold_mcnemar_p": "ud_ewt_test",
    "stage5_mcnemar_p": "ud_ewt_test",
    "stage6_sign_test_p": "ud_ewt_test",
    # Split diagnostics (4): not re-scored, by construction.
    "stage2_g2p_subsample_mcnemar_p": "split_diagnostic",
    "subsample_prefix_mcnemar_p": "split_diagnostic",
    "subsample_stride3_mcnemar_p": "split_diagnostic",
    "subsample_stride5_mcnemar_p": "split_diagnostic",
}

#: Test identity and sidedness frozen by decision 79 ("the same endpoint ...
#: the exact conditional test (McNemar's exact binomial; the Stage 6 sign
#: test), two-sided"). Every *_mcnemar_p member is exact-McNemar/two-sided;
#: stage6_sign_test_p is the exact sign test / two-sided.
FROZEN_TESTS: dict[str, tuple[str, str]] = {
    label: ("sign test (exact)", "two-sided")
    if label == "stage6_sign_test_p"
    else ("McNemar exact binomial", "two-sided")
    for label in DECISION_79_ENUMERATION
}

#: Population source, endpoint partners, registered rule per population.
POPULATION_SOURCES = {
    "ud_ewt_test": (
        "en_ewt-ud-test.conllu at the pinned upstream revision, re-vendored "
        "through generate_case_counts.py / generate_clause_counts.py / the "
        "narrative counts generator with identical oracles, exclusion rules and "
        "minimum-length filters; model parameters read unchanged from the "
        "existing dev-train fits; nothing re-fit, no stride applied."
    ),
    "external_lexicon": (
        "word/pronunciation pairs present in a newer upstream CMUdict revision "
        "and absent from vendor/cmudict.dict, passed through the identical "
        "equal-length filter, scored once against the frozen train-side fit; "
        "shift diagnostics (initial-letter TVD, shared-prefix %) reported beside "
        "the verdict."
    ),
    "split_diagnostic": (
        "Not re-scored, by construction: each is a re-measurement of a shipped "
        "subsample and has no analogue on a fresh population; the full-set "
        "member carries the replication."
    ),
}

SENTINEL_CLAIM_PREFIX = "confirmatory_run_completed_"

#: (path relative to repo root, why it matters, needed_for)
INPUT_CENSUS: list[tuple[str, str, str]] = [
    (
        "vendor/en_ewt-ud-dev.conllu",
        "dev corpus already consumed; its stride split defined the dev-train fits "
        "whose parameters the confirmatory run reads unchanged",
        "ud_ewt_test",
    ),
    (
        "vendor/en_ewt-ud-test.conllu",
        "THE held-back EWT partition per decision 79. Must NOT be vendored until "
        "Principal authorizes; if present, verify its provenance block before any run.",
        "ud_ewt_test",
    ),
    (
        "scripts/generate_case_counts.py",
        "generator for the case-count artifact (stage4 members)",
        "ud_ewt_test",
    ),
    (
        "scripts/generate_clause_counts.py",
        "generator for the clause-count artifact (stage5/stage6 members)",
        "ud_ewt_test",
    ),
    (
        "scripts/generate_narrative_counts.py",
        "narrative-counts generator named by decision 79 for the re-vendor",
        "ud_ewt_test",
    ),
    (
        "vendor/cmudict.dict",
        "vendored CMUdict revision baseline; defines what 'upstream additions' means",
        "external_lexicon",
    ),
    (
        "vendor/cmudict_counts.json",
        "derived counts over the lexicon the channel fits consumed",
        "external_lexicon",
    ),
]

lines: list[str] = []


def emit(line: str = "") -> None:
    lines.append(line)
    print(line)


#: Directory trees the filesystem sentinel scan never descends into. ``.git``
#: is VCS bookkeeping -- the previous scan walked it wholesale and then
#: filtered its paths out afterwards -- and ``.venv`` is the project's
#: dependency environment (``.venv/release-cache`` alone turned the unpruned
#: ``**`` glob into a walk over tens of thousands of unrelated entries).
#: Nested occurrences of either name are pruned too, matching the previous
#: ``".git" not in rel.parts`` contract. Everything else stays in scope: a
#: ``confirmatory_run_completed_*`` marker beside real project content --
#: under ``output/``, ``cache/``, ``data/``, ``handoff/``, the scratchpad, an
#: ordinary subdirectory -- is evidence wherever it sits, and this unit
#: deliberately adds no exclusion beyond these two.
_PRUNED_DIRECTORY_NAMES = frozenset({".git", ".venv"})


def _find_file_sentinels(root: Path) -> tuple[list[str], list[str]]:
    """Return repo-relative ``confirmatory_run_completed_*`` marker names.

    The tree is walked top-down with the pruned directories removed from the
    descent, so dependency and VCS contents are never traversed rather than
    traversed and filtered afterwards. Directories are not followed through
    symlinks: a sentinel is evidence only when it lives inside this project's
    own tree, and a link out of it would let a foreign tree masquerade as one.
    A directory whose name matches the pattern is itself a marker; it is
    reported and not descended into.

    Returns ``(hits, unreadable)``. An entry in ``unreadable`` means a
    directory listing failed and the scan is incomplete -- the caller must
    treat that as a hard failure, because an incomplete scan cannot certify
    that no marker exists.
    """
    hits: list[str] = []
    unreadable: list[str] = []

    def _on_error(error: OSError) -> None:
        unreadable.append(getattr(error, "filename", "") or str(error))

    for dirpath, dirnames, filenames in os.walk(root, onerror=_on_error):
        here = Path(dirpath)
        marker_dirs = [d for d in dirnames if d.startswith(SENTINEL_CLAIM_PREFIX)]
        hits.extend((here / d).relative_to(root).as_posix() for d in marker_dirs)
        # Prune before descending: never enter .git/.venv, a reported marker
        # directory, or a directory whose listing could not be read.
        dirnames[:] = sorted(
            d for d in dirnames if d not in _PRUNED_DIRECTORY_NAMES and d not in marker_dirs
        )
        hits.extend(
            (here / f).relative_to(root).as_posix()
            for f in filenames
            if f.startswith(SENTINEL_CLAIM_PREFIX)
        )
    return sorted(hits), sorted(unreadable)


def _ledger_sentinels(claims_path: Path) -> tuple[list[str], str | None]:
    """Read sentinel claim labels from the ledger.

    Returns ``(labels, problem)``. ``problem`` is ``None`` when the ledger was
    read and its shape was understood, and otherwise describes why the
    sentinel guard is inconclusive: the file is missing, the bytes do not
    parse, or the claims section is neither the current label->entry mapping
    nor the accepted older list-of-entries export.

    The current shape maps label -> entry, so the labels are the keys. An
    earlier revision iterated it expecting a list of entry objects carrying a
    ``name`` field; iterating a dict yields its keys, every key failed the
    isinstance check, and the guard read an empty name for every claim. It
    could not fire, and it did not: the decision 111 sentinel landed in the
    ledger and this precheck still reported "no sentinel found". A guard that
    cannot fail is worse than no guard, because it is reported as a pass. The
    list shape is still accepted so an older ledger export is not silently
    skipped, but its entries must be well-formed entry objects carrying a
    usable name or label; anything else is a problem, not a pass. An entry
    carrying both fields must alias one claim: identical strings are
    accepted, conflicting strings make the guard's read undecidable -- the
    sentinel could sit in the key that name precedence silently drops -- so
    the entry, not one of its aliases, is refused.
    """
    if not claims_path.exists():
        return [], f"{claims_path} not found"
    try:
        ledger = json.loads(claims_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], f"could not parse {claims_path}: {exc}"
    entries = ledger.get("claims", ledger) if isinstance(ledger, dict) else ledger
    if isinstance(entries, dict):
        return [str(label) for label in entries], None
    if isinstance(entries, list):
        labels = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                return [], (
                    f"{claims_path} list export entry {index} is "
                    f"{type(entry).__name__}, not an entry object"
                )
            present = {key: entry[key] for key in ("name", "label") if key in entry}
            if not present:
                return [], (f"{claims_path} list export entry {index} carries no name or label")
            if any(not isinstance(v, str) or not v.strip() for v in present.values()):
                return [], (
                    f"{claims_path} list export entry {index} carries no usable "
                    "string name or label"
                )
            if len(set(present.values())) > 1:
                return [], (
                    f"{claims_path} list export entry {index} carries conflicting "
                    f"aliases {sorted(present.items())!r}; which one names the claim "
                    "is undecidable, so the guard refuses rather than prefer one"
                )
            labels.append(next(iter(present.values())))
        return labels, None
    return [], f"{claims_path} claims section has unexpected shape {type(entries).__name__}"


def _alpha_drift_problem(
    bonferroni_alpha: float, familywise_alpha: float, family_size: int
) -> str | None:
    """The frozen-threshold drift diagnostic, or None when the contract holds.

    The frozen contract is exact: ``BONFERRONI_ALPHA`` is the familywise alpha
    divided by the frozen family size. ``ailm.statistics`` derives it exactly
    that way and the suite pins it with exact equality, so one equality check
    subsumes the old magnitude sanity checks -- including the algebraically
    ``>= 0.05`` artifact that could never fire beside them, and a
    half-corrected ``FAMILYWISE_ALPHA / 2`` that both magnitude checks would
    have accepted. Any other value -- looser divisor, stale family size,
    uncorrected familywise alpha -- is a protocol violation, and the
    diagnostic states the derived expectation instead of a hardcoded number.
    """
    expected = familywise_alpha / family_size
    if bonferroni_alpha == expected:
        return None
    return (
        f"BONFERRONI_ALPHA is {bonferroni_alpha!r}; the frozen contract is the "
        "familywise alpha over the frozen family size "
        f"({familywise_alpha!r} / {family_size} = {expected!r})"
    )


def _member_mapping_row(
    label: str,
    code_map: dict[str, str],
    sources: dict[str, str],
    tests: dict[str, tuple[str, str]],
) -> list[str]:
    """The check-4 lines for one enumerated label, tolerant of mapping drift.

    A label present in decision 79's enumeration but absent from
    ``CONFIRMATORY_REPLICATION`` is exactly the transcription drift this
    precheck exists to catch, so the row reports the gap -- disposition "?"
    with no registered population source -- instead of crashing with a
    KeyError that would bury the collected MISMATCH verdict in a traceback.
    Legitimate dispositions render their registered population source
    unchanged, and the missing-population distinctions (a census GAP is not
    an error, a split diagnostic is not rescored) are untouched here.
    """
    disposition = code_map.get(label)
    shown = disposition if disposition is not None else "?"
    source = sources.get(
        shown,
        f"<no registered population for disposition {shown!r}: the mapping has drifted>",
    )
    test, sided = tests[label]
    return [
        f"    {label}",
        f"      disposition: {shown}",
        f"      source:     {source}",
        f"      test:       {test}; sidedness: {sided}",
        "",
    ]


def main(repo_root: Path = REPO_ROOT) -> int:
    hard_failures: list[str] = []
    gaps: list[str] = []

    emit("=== Confirmatory replication precheck (decision 79) ===")
    emit("PREP ONLY -- performs no scoring, vendors nothing, writes no sentinel.")
    emit()

    # --- Check 1: sentinel refusal guards ------------------------------------
    emit("[1] Sentinel refusal guards (confirmatory_run_completed_*)")
    sentinels: set[str] = set()
    claims_path = repo_root / "data" / "claims.json"
    labels, ledger_problem = _ledger_sentinels(claims_path)
    if ledger_problem is None:
        sentinels.update(label for label in labels if label.startswith(SENTINEL_CLAIM_PREFIX))
    else:
        # Fail closed: an unreadable sentinel ledger means the guard cannot
        # certify that no population was scored, so readiness is refused
        # rather than assumed. This is deliberately distinct from the input
        # census below, where a missing optional artifact is a GAP, not an
        # error.
        emit(f"    LEDGER GUARD INCONCLUSIVE: {ledger_problem}.")
        emit(
            "      A guard that cannot read the ledger cannot certify readiness; "
            "refusing here is fail-closed, not a census gap."
        )
        hard_failures.append(
            f"sentinel ledger unreadable ({ledger_problem}) -- cannot certify readiness"
        )
    file_hits, scan_errors = _find_file_sentinels(repo_root)
    sentinels.update(file_hits)
    for error in scan_errors:
        emit(f"    SCAN INCOMPLETE: {error}")
        hard_failures.append(f"sentinel filesystem scan incomplete ({error})")
    if sentinels:
        for s in sorted(sentinels):
            emit(f"    SENTINEL PRESENT: {s}")
            emit(
                "      This population has been scored under its registration and is "
                "spent. A refusal here is the protocol working, not a defect."
            )
            hard_failures.append(
                f"sentinel already exists ({s}) -- recorder must refuse; do not run again"
            )
    elif ledger_problem is None:
        emit("    OK: no confirmatory_run_completed_* sentinel found for any population.")
    else:
        emit("    No filesystem marker found; the ledger guard above remains inconclusive.")
    emit()

    # --- Check 2: input census -------------------------------------------------
    emit("[2] Required-input census")
    for rel, why, needed_for in INPUT_CENSUS:
        state = "PRESENT" if (repo_root / rel).exists() else "ABSENT"
        flag = "" if state == "PRESENT" else "   <-- GAP"
        emit(f"    [{state}] {rel}  (needed for: {needed_for}){flag}")
        emit(f"           {why[:110]}")
        if state == "ABSENT":
            gaps.append(f"{rel} (needed for population '{needed_for}')")
    emit()
    emit("    Gap summary:")
    for g in gaps:
        emit(f"      - {g}")
    if not gaps:
        emit("      - none; every census-listed artifact is on disk")
    else:
        emit(
            "    NOTE: these gaps are the expected pre-sourcing state. Decision 79 "
            "and this lane's brief forbid re-vendoring outside Principal-gated steps."
        )
    emit()

    # --- Check 3: protocol integrity -------------------------------------------
    emit("[3] Frozen protocol integrity (ailm.statistics.CONFIRMATORY_REPLICATION)")
    code_map = dict(CONFIRMATORY_REPLICATION)
    problems: list[str] = []
    if len(code_map) != len(CONFIRMATORY_REPLICATION):
        problems.append("duplicate member in CONFIRMATORY_REPLICATION")
    if sorted(code_map) != sorted(SIGNIFICANCE_FAMILY):
        problems.append("protocol labels != SIGNIFICANCE_FAMILY")
    illegal = {d for d in code_map.values()} - set(CONFIRMATORY_DISPOSITIONS)
    if illegal:
        problems.append(f"undeclared dispositions: {sorted(illegal)}")
    for k, want in DECISION_79_ENUMERATION.items():
        got = code_map.get(k)
        if got != want:
            problems.append(f"{k}: code maps to '{got}' but decision 79 says '{want}'")
    for k, v in code_map.items():
        if k not in DECISION_79_ENUMERATION:
            problems.append(f"{k}: mapped to '{v}' in code but absent from decision-79 enumeration")
    emit(f"    family size: {len(SIGNIFICANCE_FAMILY)}; BONFERRONI_ALPHA = {BONFERRONI_ALPHA!r}")
    drift = _alpha_drift_problem(BONFERRONI_ALPHA, FAMILYWISE_ALPHA, len(SIGNIFICANCE_FAMILY))
    if drift is not None:
        problems.append(drift)
    if problems:
        for p in problems:
            emit(f"    MISMATCH: {p}")
            hard_failures.append(f"protocol integrity: {p}")
    else:
        emit(
            f"    OK: all {len(code_map)} members map exactly once, matching decision 79's "
            "enumeration; dispositions legal; threshold frozen."
        )
    emit()
    for pop in CONFIRMATORY_POPULATIONS:
        members = [member for member, d in CONFIRMATORY_REPLICATION if d == pop]
        emit(f"    {pop}: {len(members)} members -> {' | '.join(sorted(members))}")
    diagnostics = [member for member, d in CONFIRMATORY_REPLICATION if d == "split_diagnostic"]
    emit(f"    split_diagnostic: {len(diagnostics)} members -> {' | '.join(sorted(diagnostics))}")
    emit()

    # --- Check 4: per-member mapping table -------------------------------------
    emit("[4] Frozen per-member mapping (member | disposition/source | test | sidedness)")
    emit("    " + "=" * 110)
    for label in sorted(DECISION_79_ENUMERATION):
        for line in _member_mapping_row(label, code_map, POPULATION_SOURCES, FROZEN_TESTS):
            emit(line)
    emit(
        f"    Threshold: single scoring at BONFERRONI_ALPHA = {BONFERRONI_ALPHA:.6f}; "
        "Holm reported beside it as today."
    )
    emit(
        "    Discipline: one run per member ever; sentinel-first recording; no "
        "development-population label overwritten; an uncomputable member (e.g. "
        "b = c = 0 gold comparison) is recorded as the outcome, not redesigned around."
    )
    emit()

    # --- Verdict ---------------------------------------------------------------
    emit("=== Verdict ===")
    if hard_failures:
        emit(f"HARD FAIL ({len(hard_failures)} protocol violations) -- do not proceed:")
        for h in hard_failures:
            emit(f"  ! {h}")
        emit()
        return 1
    emit(
        "PROTOCOL FROZEN: code matches decision 79 and no sentinel is present. The "
        "external_lexicon population is empty and retired by decision 91 (upstream "
        "CMUdict HEAD is byte-identical to the vendored pin). The ud_ewt_test "
        "population is registered by decision 111 and unscored in this checkout; the "
        "split diagnostics have no analogue on a fresh population and are not rescored."
    )
    if gaps:
        emit(
            f"DATA GAPS REMAIN ({len(gaps)}) -- the run stays blocked until Principal-gated sourcing lands:"
        )
        for g in gaps:
            emit(f"  - {g}")
    else:
        emit("All census-listed artifacts present.")
    emit("Exit 0: prep-complete. Scoring and re-vendoring are NOT performed here.")
    emit()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-flight for the decision-79 confirmatory replication; scores nothing."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="also publish the printed summary to this file (default: stdout only)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    exit_code = main()
    if args.report is not None:
        atomic_write_text(args.report, "\n".join(lines) + "\n")
        print(f"[report written to {args.report}]")
    sys.exit(exit_code)
