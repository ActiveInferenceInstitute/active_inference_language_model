#!/usr/bin/env python3
"""Record the formal layer's own honesty numbers, from a real Lean build.

Thin orchestrator. This script is the single writer for every ``formal_*``
label, and it is the only thing standing between the ``formal/`` directory and
the failure mode it exists to prevent: a proof tree that *looks* machine-checked
and is not.

What the numbers mean
---------------------
``formal_theorem_count``
    Named ``theorem``/``lemma`` declarations across ``formal/AILM/**/*.lean``. A
    count, not a quality measure — forty trivial theorems are worth less than one
    that excludes a possible world. It is here so that a silent *deletion* is
    visible: a proof layer that quietly shrank would otherwise still report
    "build green, zero incomplete proofs".

``formal_instance_theorem_count`` / ``formal_theory_theorem_count``
    The same count split by what the theorems are *about*:
    ``AILM/Instances.lean`` holds the ones proved about recorded ledger values,
    every other module holds general theory. Both are recorded because prose in
    ``formal/README.md`` and ``HANDOFF.md`` used to hand-type them, which is the
    drift this ledger exists to prevent — the numbers were correct on the day
    they were written and had no way to stay correct. They sum to
    ``formal_theorem_count`` by construction, and the split is worth having
    because "40 theorems about measurements" and "40 theorems about arithmetic"
    are very different claims about a formal layer.

``formal_example_count``
    Anonymous ``example`` blocks over the same files. These are the
    satisfiability witnesses: a conditional theorem whose hypothesis is
    unsatisfiable compiles clean and proves nothing, so the witnesses are what
    make the conditionals mean something.

``formal_sorry_free``
    Whether a word-boundary search for either placeholder tactic over
    ``formal/AILM/**/*.lean`` comes back empty. Recorded **honestly**, including
    when it is false — see "Why this records before it refuses" below.

``formal_axioms_beyond_core``
    How many *distinct* axioms outside ``propext``, ``Quot.sound`` and
    ``funext`` appear anywhere in ``formal/axiom_report.txt``. Recorded
    alongside ``formal_axiom_names_beyond_core`` so the count is auditable
    rather than a bare number a reader has to trust.

    This is expected to be **1**, and the one is ``Classical.choice``. That is
    not a lapse: Lean core's own ``Rat.add_comm`` reports
    ``[propext, Classical.choice, Quot.sound]``, so a gate demanding only the
    three-axiom set would be unsatisfiable against core ``Rat`` and would fail
    for reasons having nothing to do with this project's proofs. What matters
    is that ``sorryAx`` (an incomplete proof) and ``Lean.ofReduceBool`` (the
    trusted-code axiom ``native_decide`` introduces) never appear.

``formal_native_decide_uses``
    Occurrences of ``native_decide`` in ``formal/AILM/**/*.lean``. Expected 0.
    Recorded rather than asserted in prose so "everything closes by
    kernel-checked ``decide``" is a ledger-backed claim.

``formal_axiom_report_verdicts``
    Lines in the axiom report carrying a verdict. Guards the vacuity failure
    mode named in ``Scripts/AxiomReport.lean``: a report covering two of eighty
    theorems reports clean.

Guards — the recorder refuses to record when
--------------------------------------------
1. ``lake build`` fails. Every number above describes a tree that compiled; if
   it did not compile, the theorem count is a count of text.
2. The axiom report is stale. The report is regenerated into a temporary file
   and compared against the committed one. A report describing a tree that has
   since changed is worse than no report, because it reads as current.
3. The axiom report does not cover every declaration. Coverage is checked
   **both ways**: every named theorem and instance in ``formal/AILM/**/*.lean``
   must have a report entry, and every report entry must correspond to a real
   declaration. An enumeration only constrains what someone remembered to
   enumerate, so neither direction is left to memory.
4. A declaration form the scanner cannot read appears anywhere in the tree. The
   patterns are anchored to column zero and to a known modifier set, so an
   indented declaration or one behind an ``@[...]`` attribute would be invisible
   to the declaration count *and* to the coverage check in (3) — and invisible to
   the coverage check means the axiom gate keeps reporting clean for a theorem
   nobody checked. The scanner refuses rather than guesses.

Why this records before it refuses (on the incomplete-proof check)
------------------------------------------------------------------
``formal_sorry_free`` is only worth storing if it *can* be false. A recorder
that refused to write whenever it found an incomplete proof would make the
label a constant — true by construction, informative about nothing, exactly the
"metric green by construction" defect this repository audits for elsewhere. So
the honest value is written and the script then exits nonzero, which fails the
suite loudly while leaving a truthful ledger behind.

The build guard is different and does refuse, because a failed build makes the
other numbers meaningless rather than merely bad.

When the Lean toolchain is absent
---------------------------------
The script prints what it needs and exits nonzero **without recording
anything**. It does not fabricate or silently preserve a green workflow: a
machine with no ``elan`` cannot verify these claims, so publication acceptance
must fail until the pinned toolchain is available. Previously recorded values
stay untouched; no new claim is made on a machine that cannot check one.

Usage
-----
    .venv/bin/python scripts/record_formal_claims.py
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from _artifact_io import atomic_write_bytes  # noqa: E402

from ailm.claims import load_ledger  # noqa: E402 - project path is established above
from ailm.formal_inventory import (  # noqa: E402
    _EXAMPLE,
    _NATIVE_DECIDE,
    declaration_inventory,
    module_theorem_counts,
    render_axiom_script,
)
from ailm.formal_inventory import (  # noqa: E402
    _INSTANCE as _INSTANCE,
)
from ailm.formal_inventory import (  # noqa: E402
    _THEOREM as _THEOREM,
)
from ailm.formal_inventory import (  # noqa: E402
    _UNSCANNABLE as _UNSCANNABLE,
)
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "formal_layer_v1"
SOURCE = "scripts/record_formal_claims.py"

FORMAL_DIR = _PROJECT_ROOT / "formal"
LIBRARY_DIR = FORMAL_DIR / "AILM"
#: The one module whose theorems are about recorded values rather than theory.
INSTANCES_FILE = "Instances.lean"
AXIOM_REPORT = FORMAL_DIR / "axiom_report.txt"
AXIOM_SCRIPT = Path("Scripts") / "AxiomReport.lean"

#: Axioms that count as core for ``formal_axioms_beyond_core``. Anything else is
#: counted and named, including ``Classical.choice`` — which is expected and
#: explained in the module docstring, not silently exempted here. An exemption
#: baked into the detector is an exemption nobody re-reads.
CORE_AXIOMS = frozenset({"propext", "Quot.sound", "funext"})

#: Word-boundary anchored so ``admits`` cannot match the placeholder tactic. A
#: bare substring search flagged a theorem named ``..._admits_no_...`` as an
#: incomplete proof, which is a false positive that reads exactly like a real
#: one. The search deliberately covers comments too: a scanner that parsed them
#: out would be one more component that could be wrong in the direction of
#: reporting clean.
_PLACEHOLDER = re.compile(r"\b(?:sorry|admit)\b")

#: One axiom verdict line. ``re.MULTILINE`` is load-bearing: without it ``^``
#: anchors to the start of the whole report and exactly one line matches, which
#: makes the coverage check below fire on almost every declaration. It failed
#: closed the first time, but a parser that under-matches is one edit away from
#: being a coverage check that passes because it found nothing to compare.
_VERDICT = re.compile(
    r"^'([^']+)' (?:depends on axioms: \[(.*)\]|does not depend on any axioms)",
    re.MULTILINE,
)


def _refuse(message: str) -> int:
    print(f"REFUSING TO RECORD: {message}", file=sys.stderr)
    return 1


def _library_sources() -> list[Path]:
    """Every Lean source under the library root, at any depth.

    ``rglob``, not ``glob``: a module added under ``AILM/Sub/`` would otherwise
    escape the declaration count, the placeholder scan AND the axiom-report
    coverage check at once, and all three would keep reporting clean.
    """
    return sorted(LIBRARY_DIR.rglob("*.lean"))


def _run_lake(args: list[str], *, capture: bool) -> subprocess.CompletedProcess[str]:
    """Invoke lake in ``formal/`` with the toolchain the directory pins."""
    return subprocess.run(
        ["lake", *args],
        cwd=str(FORMAL_DIR),
        capture_output=capture,
        text=True,
        check=False,
    )


def _parse_report(text: str) -> tuple[dict[str, set[str]], int]:
    """Map each reported declaration to its axiom set, and count the verdicts.

    ``#print axioms`` wraps long axiom lists across lines, so the text is
    un-wrapped before matching. A parser that silently dropped wrapped entries
    would under-count exactly the entries with the most axioms.
    """
    flattened = re.sub(r",\n\s+", ", ", text)
    reported: dict[str, set[str]] = {}
    for match in _VERDICT.finditer(flattened):
        name, axioms = match.group(1), match.group(2)
        reported[name] = {a.strip() for a in axioms.split(",") if a.strip()} if axioms else set()
    return reported, len(reported)


def main() -> int:
    if not LIBRARY_DIR.is_dir():
        return _refuse(f"{LIBRARY_DIR} does not exist; there is no formal layer to measure.")

    sources = _library_sources()
    if not sources:
        return _refuse(
            f"{LIBRARY_DIR} contains no .lean files. A zero-file scan satisfies every "
            f"check below vacuously."
        )

    if shutil.which("lake") is None:
        return _refuse(
            "lake is not on PATH, so the formal layer cannot be verified on this "
            "machine and NOTHING WILL BE RECORDED. Install the toolchain the "
            "formal/lean-toolchain file pins (leanprover/lean4:v4.35.0) via elan, "
            "then re-run. Previously recorded formal_* claims are left untouched."
        )

    text_by_file = {
        path.relative_to(FORMAL_DIR): path.read_text(encoding="utf-8") for path in sources
    }
    try:
        inventory = declaration_inventory(text_by_file)
        expected_script = render_axiom_script(text_by_file)
    except ValueError as error:
        return _refuse(str(error))
    script_path = FORMAL_DIR / AXIOM_SCRIPT
    if not script_path.is_file() or script_path.read_text(encoding="utf-8") != expected_script:
        return _refuse(
            "Scripts/AxiomReport.lean is stale; run "
            "scripts/record_formal_claims.py --generate-axiom-script, then "
            "formal/Scripts/axiom_report.sh before recording."
        )

    # ── Guard 1: the tree must compile ───────────────────────────────────────
    build = _run_lake(["build"], capture=True)
    if build.returncode != 0:
        return _refuse(
            "`lake build` failed, so every number below would describe text rather "
            f"than a checked proof.\n{build.stdout[-1500:]}\n{build.stderr[-1500:]}"
        )

    # ── Guard 2: the committed axiom report must describe the tree that just built ──
    regenerated = _run_lake(["env", "lean", str(AXIOM_SCRIPT)], capture=True)
    if regenerated.returncode != 0:
        return _refuse(
            "regenerating the axiom report failed — most likely a #print axioms "
            f"entry naming a declaration that no longer exists.\n{regenerated.stderr[-1500:]}"
        )
    if not AXIOM_REPORT.exists():
        return _refuse(f"{AXIOM_REPORT} is missing; run formal/Scripts/axiom_report.sh.")
    committed = AXIOM_REPORT.read_text(encoding="utf-8")
    if committed != regenerated.stdout:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(regenerated.stdout)
            drift_path = handle.name
        return _refuse(
            f"{AXIOM_REPORT} is stale — it does not match a fresh run. A stale axiom "
            f"report is worse than none, because it reads as current. Fresh output "
            f"written to {drift_path}; regenerate with formal/Scripts/axiom_report.sh."
        )

    reported, verdicts = _parse_report(committed)
    if verdicts == 0:
        return _refuse(
            f"{AXIOM_REPORT} contains no verdict lines. An axiom gate that checks "
            f"nothing reports clean."
        )

    # ── Guard 3: report coverage, checked in both directions ─────────────────
    declared = {name for names in inventory.values() for name in names}

    unreported = sorted(declared - set(reported))
    if unreported:
        return _refuse(
            f"these declarations have no entry in {AXIOM_REPORT.name}: {unreported}. "
            f"A report covering some of the tree reports clean for the rest — regenerate the "
            f"driver and report with formal/Scripts/axiom_report.sh."
        )
    phantom = sorted(set(reported) - declared)
    if phantom:
        return _refuse(
            f"{AXIOM_REPORT.name} reports on declarations this scan cannot find in "
            f"formal/AILM/*.lean: {phantom}. Either the report is stale or the "
            f"declaration scanner is missing real declarations — both make the "
            f"coverage check above unreliable."
        )

    # ── The measurements ─────────────────────────────────────────────────────
    per_file = {path: len(_THEOREM.findall(text)) for path, text in text_by_file.items()}
    theorem_count = sum(per_file.values())
    instance_theorems = sum(
        count for path, count in per_file.items() if path.name == INSTANCES_FILE
    )
    if instance_theorems == 0:
        return _refuse(
            f"no theorems were found in {INSTANCES_FILE}. Either the ledger-facing "
            f"module was renamed or the declaration scanner has stopped matching; "
            f"recording a zero for formal_instance_theorem_count would report an "
            f"empty ledger bridge as a measurement."
        )
    example_count = sum(len(_EXAMPLE.findall(text)) for text in text_by_file.values())
    native_decide_uses = sum(len(_NATIVE_DECIDE.findall(text)) for text in text_by_file.values())

    placeholders = {
        str(path): len(_PLACEHOLDER.findall(text))
        for path, text in text_by_file.items()
        if _PLACEHOLDER.search(text)
    }
    sorry_free = not placeholders

    beyond_core = sorted({axiom for axioms in reported.values() for axiom in axioms} - CORE_AXIOMS)

    values: dict[str, object] = {
        "formal_theorem_count": theorem_count,
        "formal_instance_theorem_count": instance_theorems,
        "formal_theory_theorem_count": theorem_count - instance_theorems,
        "formal_example_count": example_count,
        "formal_sorry_free": sorry_free,
        "formal_axioms_beyond_core": len(beyond_core),
        "formal_axiom_names_beyond_core": beyond_core,
        "formal_axiom_report_verdicts": verdicts,
        "formal_native_decide_uses": native_decide_uses,
        **module_theorem_counts(text_by_file),
    }

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(values)} formal-layer claims from a green `lake build`:\n")
    width = max(len(label) for label in values)
    for label, value in sorted(values.items()):
        print(f"  {label.ljust(width)} = {value}")

    if "sorryAx" in beyond_core:
        print(
            "\nREPORTED AXIOMS INCLUDE sorryAx: some proof in this tree is "
            "incomplete. The values above are recorded truthfully; the exit code "
            "is nonzero because the tree is not in a shippable state.",
            file=sys.stderr,
        )
        return 1
    if "Lean.ofReduceBool" in beyond_core:
        print(
            "\nREPORTED AXIOMS INCLUDE Lean.ofReduceBool: some proof closed by "
            "native_decide and is now trusting the compiler rather than the kernel. "
            "Document it in formal/README.md or replace it with decide.",
            file=sys.stderr,
        )
        return 1
    if not sorry_free:
        print(
            f"\nINCOMPLETE PROOFS FOUND in {placeholders}. The value recorded above "
            f"is the honest one; the exit code is nonzero because zero is the bar.",
            file=sys.stderr,
        )
        return 1
    return 0


def command_line(argv: list[str] | None = None) -> int:
    """Generate/check the declaration driver, or record after a verified build."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--generate-axiom-script", action="store_true")
    modes.add_argument("--check-axiom-script", action="store_true")
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=FORMAL_DIR,
        help="alternate formal tree for driver generation/checking only",
    )
    args = parser.parse_args(argv)
    if args.generate_axiom_script or args.check_axiom_script:
        formal_dir = args.formal_dir
        try:
            texts = {
                path.relative_to(formal_dir): path.read_text(encoding="utf-8")
                for path in sorted((formal_dir / "AILM").rglob("*.lean"))
            }
            expected = render_axiom_script(texts).encode("utf-8")
            target = formal_dir / AXIOM_SCRIPT
            if args.check_axiom_script:
                if not target.is_file() or target.read_bytes() != expected:
                    return _refuse(f"{target} is missing or stale; regenerate the axiom script")
                print(f"Axiom script matches the declaration inventory: {target}")
            else:
                atomic_write_bytes(target, expected)
                print(f"Generated axiom script: {target}")
        except (OSError, ValueError) as error:
            return _refuse(str(error))
        return 0
    if args.formal_dir != FORMAL_DIR:
        parser.error("--formal-dir requires a driver generation/check mode")
    # The ledger is read and rewritten whole; recorders must not overlap.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        return main()


if __name__ == "__main__":
    raise SystemExit(command_line())
