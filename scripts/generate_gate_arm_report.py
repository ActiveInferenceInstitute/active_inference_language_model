"""Render the measured gate-arm table from release receipts.

Every arm in this repository is a profile of ``scripts/verify_release.py``, and
every profile writes a receipt whose steps carry their own wall time. This script
turns a directory of those receipts into one document, so the numbers a reader
compares are produced by the run that measured them instead of being typed into
prose and going stale.

``--check`` compares the committed Markdown with the committed record, and
compares both with the live command inventory: an arm or a step that no longer
exists in ``scripts/verify_release.py`` fails, so the document cannot describe a
gate the runner does not have.

Usage:
    .venv/bin/python scripts/generate_gate_arm_report.py --from-receipts output/reports
    .venv/bin/python scripts/generate_gate_arm_report.py --check
    .venv/bin/python scripts/generate_gate_arm_report.py --print
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "verify_release.py"
GENERATED = ROOT / "docs" / "_generated"
RECORD_PATH = GENERATED / "gate_arms.json"
DOCUMENT_PATH = GENERATED / "gate_arms.md"
SCHEMA = "ailm-gate-arms/v1"

#: The arms this document reports, in the order a developer meets them. Each one
#: is a profile of the release runner; the profile name is the arm name.
ARMS = (
    ("lint", "format and lint"),
    ("types", "static typing"),
    ("docs", "documentation index, links, and hydration"),
    ("quick", "fast tests"),
    ("measurement", "end-to-end measurement tests"),
    ("contract", "wheel and real-socket contract tests"),
    ("gnn", "pinned upstream GNN conformance"),
    ("full", "publication acceptance"),
)

#: Profiles whose step inventory ``--check`` can print without a template
#: checkout or a Lean toolchain. ``full`` needs both, so its inventory is not
#: re-derived here; its receipt is the record.
LISTABLE = frozenset({"lint", "types", "docs", "quick", "measurement", "contract", "gnn"})

COMMAND = ".venv/bin/python -I -S scripts/verify_release.py {profile}"


def _runner(*arguments: str) -> str:
    """Run the release runner with the project interpreter and return stdout."""

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(RUNNER), *arguments],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"verify_release.py {' '.join(arguments)} failed: {result.stderr}")
    return result.stdout


def _declared_profiles() -> set[str]:
    """The profile names the runner's own argument parser accepts."""

    help_text = _runner("--help")
    for group in re.findall(r"\{([a-z][a-z,\s]*)\}", help_text):
        names = {name.strip() for name in group.split(",") if name.strip()}
        if "quick" in names:
            return names
    raise RuntimeError("verify_release.py --help does not list its profiles")


def _step_inventory(profile: str) -> list[str]:
    """The step names one profile would run, from the runner's list mode."""

    rows = [line for line in _runner(profile, "--list").splitlines() if line.strip()]
    return [row.split("\t")[0] for row in rows]


def _arm_record(arm: str, description: str, receipt: Path) -> dict[str, object]:
    """One arm's row, taken from its receipt when the receipt exists."""

    row: dict[str, object] = {
        "arm": arm,
        "profile": arm,
        "description": description,
        "command": COMMAND.format(profile=arm),
        "receipt": str(receipt.relative_to(ROOT)) if receipt.is_relative_to(ROOT) else str(receipt),
        "measured": False,
        "total_seconds": None,
        "pytest_passed": None,
        "steps": [],
        "note": "",
    }
    if not receipt.is_file():
        row["note"] = "no receipt in the supplied directory"
        return row
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("profile") != arm:
        raise ValueError(f"{receipt} records profile {payload.get('profile')!r}, not {arm!r}")
    steps = [
        {
            "name": str(step["name"]),
            "duration_seconds": float(step["duration_seconds"]),
            "status": str(step["status"]),
            "pytest_passed": step.get("pytest_passed"),
        }
        for step in payload.get("steps", ())
    ]
    if not steps:
        # A profile that refused on a prerequisite has no measured step: recording
        # it as a run would report an arm that never gated anything.
        row["measured"] = False
        row["note"] = f"refused before any step: {payload.get('refusal')}"
        return row
    passed = [step["pytest_passed"] for step in steps if step["pytest_passed"] is not None]
    row.update(
        {
            "measured": True,
            "success": bool(payload.get("success")),
            "refusal": payload.get("refusal"),
            "total_seconds": round(sum(step["duration_seconds"] for step in steps), 3),
            "pytest_passed": sum(passed) if passed else None,
            "steps": steps,
        }
    )
    if payload.get("refusal"):
        row["note"] = f"refused before any step: {payload['refusal']}"
    elif not payload.get("success"):
        row["note"] = "the run did not succeed; see its receipt"
    return row


def build_record(receipts: Path, baseline_receipt: Path | None = None) -> dict[str, object]:
    """Collect every arm's measurement from ``receipts``."""

    record: dict[str, object] = {
        "schema": SCHEMA,
        "recorded": date.today().isoformat(),
        "host": f"{platform.system().lower()}-{platform.machine()}",
        "interpreter": f"python {platform.python_version()}",
        "arms": [
            _arm_record(arm, description, receipts / f"{arm}.json") for arm, description in ARMS
        ],
    }
    if baseline_receipt is not None:
        measured = _baseline(baseline_receipt)
        if measured is not None:
            record["baseline"] = measured
    return record


#: Which steps of the pre-split ``quick`` receipt each arm now owns. The arm runs
#: exactly those commands, so their recorded durations are its measured baseline.
#: An arm whose steps were never separable there shows a blank rather than a
#: guessed number, and no whole-profile total is compared: the baseline run
#: stopped at a failing step, so it recorded a different step set.
_BASELINE_PROJECTION: dict[str, tuple[str, ...]] = {
    "lint": ("format", "lint"),
    "types": ("types",),
}


def _baseline(receipt: Path) -> dict[str, object] | None:
    """Project the pre-split quick receipt onto the arms that replaced it."""

    if not receipt.is_file():
        return None
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    steps = payload.get("steps", ())
    durations = {str(step["name"]): float(step["duration_seconds"]) for step in steps}
    if not durations:
        return None
    tests_quick = next((step for step in steps if step["name"] == "tests-quick"), None)
    return {
        "receipt": (
            str(receipt.relative_to(ROOT)) if receipt.is_relative_to(ROOT) else str(receipt)
        ),
        "recorded_steps": len(durations),
        "total_seconds": round(sum(durations.values()), 3),
        "projected": {
            arm: round(sum(durations[name] for name in names), 3)
            for arm, names in _BASELINE_PROJECTION.items()
            if all(name in durations for name in names)
        },
        "tests_quick": (
            {
                "seconds": round(float(tests_quick["duration_seconds"]), 3),
                "passed": tests_quick.get("pytest_passed"),
            }
            if tests_quick is not None
            else None
        ),
    }


def _seconds(value: object) -> str:
    return "—" if value is None else f"{float(value):.1f}"


def render(record: dict[str, object]) -> str:
    """Render the committed Markdown from the committed record."""

    arms = record["arms"]
    baseline = record.get("baseline") or {}
    before: dict[str, object] = dict(baseline.get("projected") or {})
    lines = [
        "# Gate arms — measured wall time",
        "",
        "Generated by [`scripts/generate_gate_arm_report.py`](../../scripts/generate_gate_arm_report.py)",
        "from the release receipts named in the table. Do not edit this file by hand:",
        "`--check` fails when it disagrees with [`gate_arms.json`](gate_arms.json), and",
        "when a recorded arm or step no longer exists in the runner's own inventory.",
        "",
        f"- Host: `{record['host']}`",
        f"- Interpreter: `{record['interpreter']}`",
        f"- Recorded: `{record['recorded']}`",
        "- Every arm is one profile of `.venv/bin/python -I -S scripts/verify_release.py`;",
        "  the same command runs locally and in the hosted job that owns the arm.",
        "",
        "| Arm | What it gates | Measured | Before (s) | After (s) | Tests passed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        measured = "yes" if arm["measured"] else "no"
        tests = "—" if arm["pytest_passed"] is None else str(arm["pytest_passed"])
        lines.append(
            f"| `{arm['arm']}` | {arm['description']} | {measured} | "
            f"{_seconds(before.get(arm['arm']))} | "
            f"{_seconds(arm['total_seconds'])} | {tests} |"
        )
    lines.append("")
    if baseline:
        lines.append(
            "Before column: the steps each arm owns, projected from the pre-split quick "
            f"receipt `{baseline['receipt']}` ({baseline['recorded_steps']} recorded steps, "
            f"{_seconds(baseline['total_seconds'])} total). An arm whose steps were not "
            "separable there shows `—`; its own work is unchanged by the split. No "
            "whole-profile total is compared, because the baseline run stopped at a "
            "failing step and therefore recorded a different step set."
        )
        lines.append("")
        tests_quick = baseline.get("tests_quick")
        after_tests = next(
            (
                step
                for arm in arms
                if arm["arm"] == "quick"
                for step in arm["steps"]
                if step["name"] == "tests-quick"
            ),
            None,
        )
        if tests_quick and after_tests is not None:
            # A receipt records a pass count only for a step that exited zero, so
            # the before count is absent when the baseline run stopped on a
            # failure. Saying "None passed" would read as zero.
            before_count = (
                "" if tests_quick["passed"] is None else f" and {tests_quick['passed']} passed"
            )
            lines.append(
                f"The test step itself: {_seconds(tests_quick['seconds'])} s"
                f"{before_count} before the split, "
                f"{_seconds(after_tests['duration_seconds'])} s and "
                f"{after_tests['pytest_passed']} passed after it, with the documentation "
                "file the `docs` arm now owns excluded from the second run."
            )
            lines.append("")
    lines.append("## Per-step wall time")
    lines.append("")
    for arm in arms:
        lines.append(f"### `{arm['arm']}`")
        lines.append("")
        if not arm["steps"]:
            lines.append(f"No step was measured. {arm['note']}".rstrip())
            lines.append("")
            continue
        if arm["note"]:
            lines.append(arm["note"])
            lines.append("")
        lines.append("| Step | Status | Wall time (s) | Tests passed |")
        lines.append("| --- | --- | --- | --- |")
        for step in arm["steps"]:
            passed = "—" if step["pytest_passed"] is None else str(step["pytest_passed"])
            lines.append(
                f"| `{step['name']}` | {step['status']} | "
                f"{_seconds(step['duration_seconds'])} | {passed} |"
            )
        lines.append("")
    lines.append("## Refresh")
    lines.append("")
    lines.append("```bash")
    lines.append("# One receipt per arm, named for the profile it ran.")
    for arm, _description in ARMS:
        lines.append(f"{COMMAND.format(profile=arm)} --receipt output/reports/{arm}.json")
    lines.append(
        ".venv/bin/python scripts/generate_gate_arm_report.py --from-receipts output/reports"
    )
    lines.append(".venv/bin/python scripts/generate_gate_arm_report.py --check")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write(record: dict[str, object]) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    DOCUMENT_PATH.write_text(render(record), encoding="utf-8")


def check() -> list[str]:
    """Return every way the committed document disagrees with its sources."""

    problems: list[str] = []
    if not RECORD_PATH.is_file() or not DOCUMENT_PATH.is_file():
        return [f"missing generated file: {RECORD_PATH} or {DOCUMENT_PATH}"]
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        problems.append(f"{RECORD_PATH} declares schema {record.get('schema')!r}, not {SCHEMA!r}")
    if DOCUMENT_PATH.read_text(encoding="utf-8") != render(record):
        problems.append(f"{DOCUMENT_PATH} is stale; re-run --from-receipts")
    declared = _declared_profiles()
    for arm in record["arms"]:
        profile = arm["profile"]
        if profile not in declared:
            problems.append(f"arm {arm['arm']!r} names profile {profile!r}, which the runner lacks")
            continue
        if not arm["measured"] or profile not in LISTABLE:
            continue
        inventory = set(_step_inventory(profile))
        for step in arm["steps"]:
            if step["name"] not in inventory:
                problems.append(
                    f"arm {arm['arm']!r} records step {step['name']!r}, which profile "
                    f"{profile!r} no longer runs"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-receipts",
        type=Path,
        metavar="DIR",
        help="collect <arm>.json receipts from DIR and write the record and the document",
    )
    parser.add_argument(
        "--baseline-receipt",
        type=Path,
        metavar="PATH",
        help="the pre-split quick receipt whose recorded steps are the before column",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed document against the record and the live inventory",
    )
    parser.add_argument(
        "--print",
        dest="print_document",
        action="store_true",
        help="print the document the committed record renders",
    )
    args = parser.parse_args(argv)
    if args.from_receipts is not None:
        receipts = args.from_receipts.expanduser()
        receipts = receipts if receipts.is_absolute() else (Path.cwd() / receipts).absolute()
        record = build_record(receipts, args.baseline_receipt)
        write(record)
        print(f"wrote {RECORD_PATH.relative_to(ROOT)} and {DOCUMENT_PATH.relative_to(ROOT)}")
        return 0
    if args.check:
        problems = check()
        for problem in problems:
            print(problem, file=sys.stderr)
        print("gate arm report is current" if not problems else "gate arm report is stale")
        return 0 if not problems else 1
    if args.print_document:
        print(render(json.loads(RECORD_PATH.read_text(encoding="utf-8"))))
        return 0
    parser.error("choose --from-receipts DIR, --check, or --print")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
