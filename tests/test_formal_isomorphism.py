"""The Lean formal layer must stay bound to the ledger, and must stay checkable.

What this file guards
---------------------
``formal/AILM/LedgerData.lean`` is the one Lean module containing measured
values, and it is generated. Two things can go wrong with a generated file that
holds numbers a proof depends on:

1. **Someone edits it.** A hand edit is invisible — the file still compiles,
   the proofs still close, and the tree now certifies numbers nobody measured.
2. **Someone doesn't.** A recorder re-runs, a value moves in
   ``data/claims.json``, and the committed Lean keeps proving things about the
   old number. The build stays green the entire time.

``test_regeneration_is_byte_identical`` catches both, and it is the reason the
generated file may be trusted at all.

Byte identity alone is not enough, though: it proves the generator is
*deterministic*, not that it converts *correctly*. A generator that consistently
dropped a decimal place would pass it forever. So
``test_every_literal_matches_the_ledger_exactly`` re-derives each literal from
``data/claims.json`` through an independently written conversion and compares —
a second oracle rather than a restatement of the first.

Non-vacuity
-----------
Every check here has a positive control, because each one would otherwise pass
in the degenerate case:

* ``test_a_mutated_ledger_changes_the_output`` — without it, a generator that
  ignored the ledger entirely and emitted a constant string would satisfy the
  byte-identity test perfectly.
* ``test_literal_scan_finds_the_expected_population`` — without it, a literal
  scanner whose regex stopped matching would report zero mismatches.
* ``test_axiom_scanner_flags_a_planted_axiom`` — without it, an axiom check that
  parsed nothing would report a clean tree.
* ``test_the_negative_control_does_not_compile`` — without it, "lake build
  succeeded" would rest on a toolchain nobody has asked to fail.

Toolchain
---------
Everything that needs Lean is gated on ``shutil.which("lake")`` and skips with
the exact toolchain requirement named, mirroring how
``tests/test_gnn_conformance.py`` treats its optional upstream checkout. The
ledger-binding tests above that line need no toolchain and always run, so a
machine without ``elan`` still cannot silently desynchronise the Lean from the
measurements.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAL_DIR = PROJECT_ROOT / "formal"
LIBRARY_DIR = FORMAL_DIR / "AILM"
GENERATED = LIBRARY_DIR / "LedgerData.lean"
BONFERRONI = LIBRARY_DIR / "Bonferroni.lean"
HOLM = LIBRARY_DIR / "Holm.lean"
AXIOM_REPORT = FORMAL_DIR / "axiom_report.txt"
LEDGER = PROJECT_ROOT / "data" / "claims.json"
GENERATOR = PROJECT_ROOT / "scripts" / "generate_lean_ledger.py"
MUST_FAIL = FORMAL_DIR / "Controls" / "MustFail.lean"
FORMAL_DOC = PROJECT_ROOT / "docs" / "formal.md"
FORMAL_README = FORMAL_DIR / "README.md"

TOOLCHAIN_FILE = FORMAL_DIR / "lean-toolchain"
_TOOLCHAIN = TOOLCHAIN_FILE.read_text(encoding="utf-8").strip() if TOOLCHAIN_FILE.exists() else "?"

_NEEDS_LAKE = pytest.mark.skipif(
    shutil.which("lake") is None,
    reason=(
        f"lake is not on PATH. The formal layer needs the elan-managed toolchain "
        f"pinned in formal/lean-toolchain ({_TOOLCHAIN}); install elan and re-run "
        f"to exercise these checks."
    ),
)


def _formal_deferral_documentation_defects(docs: str, readme: str) -> list[str]:
    """Keep the two formal scope documents aligned after the paid-off proof."""

    defects: list[str] = []
    retired = "Only one item is *deferred* rather than descoped"
    if retired in docs or retired in readme:
        defects.append("retired-one-item-deferred")
    if "No current item is *deferred* rather than descoped" not in docs:
        defects.append("docs-no-current-deferral")
    if "No exclusion on the list is *deferred*" not in readme:
        defects.append("readme-no-current-deferral")
    return defects


def test_formal_scope_documents_agree_that_no_item_is_currently_deferred() -> None:
    docs = FORMAL_DOC.read_text(encoding="utf-8")
    readme = FORMAL_README.read_text(encoding="utf-8")
    assert _formal_deferral_documentation_defects(docs, readme) == []

    # NEGATIVE CONTROL: the exact retired contradiction must be detected.
    planted = docs + "\nOnly one item is *deferred* rather than descoped.\n"
    assert "retired-one-item-deferred" in _formal_deferral_documentation_defects(planted, readme)


#: Axioms a proof in this tree may depend on.
#:
#: ``Classical.choice`` is here for a measured reason rather than a permissive
#: one: Lean core's own ``Rat.add_comm`` reports
#: ``[propext, Classical.choice, Quot.sound]``, so a stricter list would be
#: unsatisfiable against core ``Rat``. See formal/README.md.
ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound", "funext", "Classical.choice"})

#: Named separately from ``ALLOWED_AXIOMS`` on purpose. Widening the allow-list
#: is a one-word edit; these two must survive it, because each marks a specific
#: loss of guarantee: ``sorryAx`` an incomplete proof, ``Lean.ofReduceBool`` a
#: proof that trusts the compiler instead of the kernel.
FORBIDDEN_AXIOMS = ("sorryAx", "Lean.ofReduceBool")

#: Word-boundary anchored so ``admits`` cannot match the placeholder tactic.
#: A bare substring search produced exactly that false positive here once.
_PLACEHOLDER = re.compile(r"\b(?:sorry|admit)\b")

#: ``def NAME : Rat := lit N D`` preceded by a ``(ledger: `label`)`` comment.
_SCALAR = re.compile(
    r"\(ledger: `([a-z0-9_]+)`\)\. -/\ndef \w+ : Rat := lit \((-?\d+)\) (\d+)|"
    r"\(ledger: `([a-z0-9_]+)`\)\. -/\ndef \w+ : Rat := lit (\d+) (\d+)"
)

#: A list entry: ``lit N D  -- <note>  (label)``.
#:
#: The label group is anchored to end-of-line with a *greedy* ``.*``. A
#: non-greedy match captured ``anchor`` out of the note ``-- alpha 0 (anchor)``
#: instead of the trailing claim label — a parenthesised word that looks exactly
#: like a label and is not one.
_LIST_ENTRY = re.compile(
    r"^\s*[\[,] lit (\(-?\d+\)|\d+) (\d+)\s+-- .*\(([a-z0-9_]+)\)$",
    re.MULTILINE,
)

#: Any literal at all, for the default-deny accounting below.
#:
#: Operands must be *numeric*. The generated header explains the conversion
#: policy in prose ("emitted as ``lit n d``"), and a pattern keyed on the bare
#: constructor name flags that sentence as an unexplained measurement — a false
#: positive that reads exactly like a real one. ``lit n d`` with letters is not
#: a literal; ``lit 3 4`` is. Both cases are pinned by the positive control.
_ANY_LITERAL = re.compile(r"\blit\s+(?:\(-?\d+\)|\d+)\s+\d+")

#: The marker a *structural* literal must carry. Structural literals name no
#: ledger label because no recorder measured them — they are the pre-registered
#: sweep grid. Marking them is what keeps "has no label" from being the same
#: observable state as "should have had a label and lost it".
_STRUCTURAL_MARK = "[structural]"

#: A scalar ``def``, which is only accounted for when the line above it carries
#: the ``(ledger: `label`)`` provenance comment.
_SCALAR_DEF_LINE = re.compile(r"^def \w+ : Rat := lit (?:\(-?\d+\)|\d+) \d+$")
_LEDGER_PROVENANCE = re.compile(r"\(ledger: `[a-z0-9_]+`\)\. -/$")
_WITNESS_LENGTH = re.compile(r"witnessFamily_length\s*:\s*witnessFamily\.length\s*=\s*(\d+)")
_WITNESS_THRESHOLD = re.compile(
    r"bonferroniThreshold \(1 / 100\) witnessFamily\.length\s*=\s*1 / (\d+)"
)
#: The Holm module's synthetic step-down witness: its family, its familywise
#: level, and the two rejection counts it proves. Anchored on the theorem names
#: rather than on the expressions alone, so the containment and strictness
#: theorems, which mention the same terms under ``≤`` and ``<``, cannot be
#: mistaken for a count.
_HOLM_FAMILY = re.compile(r"def witnessFamilySorted : List Rat := \[([^\]]+)\]")
_HOLM_COUNT = re.compile(
    r"theorem witnessFamilySorted_holm_count :\s*"
    r"holmRejectCount \((\d+) / (\d+)\) witnessFamilySorted\s*=\s*(\d+)"
)
_HOLM_BONFERRONI_COUNT = re.compile(
    r"theorem witnessFamilySorted_bonferroni_count :\s*"
    r"bonferroniRejectCount \((\d+) / (\d+)\) witnessFamilySorted\s*=\s*(\d+)"
)


def _claims_text() -> dict[str, object]:
    """Ledger values with decimals kept as exact source text, as the generator does."""
    with LEDGER.open(encoding="utf-8") as handle:
        payload = json.load(handle, parse_float=str)
    return {label: entry["value"] for label, entry in payload["claims"].items()}


def _exact(value: object) -> Fraction:
    """The conversion policy, written independently of the generator's copy."""
    if isinstance(value, bool):  # pragma: no cover - guarded by the caller's labels
        raise AssertionError("a boolean is not a rational measurement")
    if isinstance(value, int):
        return Fraction(value, 1)
    assert isinstance(value, str), f"unexpected ledger value type {type(value).__name__}"
    return Fraction(value)


def _scan_literals(text: str) -> list[tuple[str, Fraction]]:
    """Every ``lit`` in the generated file, paired with the label it cites."""
    found: list[tuple[str, Fraction]] = []
    for match in _SCALAR.finditer(text):
        groups = [g for g in match.groups() if g is not None]
        label, numerator, denominator = groups[0], groups[1].strip("()"), groups[2]
        found.append((label, Fraction(int(numerator), int(denominator))))
    for match in _LIST_ENTRY.finditer(text):
        numerator, denominator, label = match.group(1).strip("()"), match.group(2), match.group(3)
        found.append((label, Fraction(int(numerator), int(denominator))))
    return found


def _unexplained_literals(text: str) -> list[tuple[int, str]]:
    """Every literal in a generated module that is neither ledger-backed nor marked.

    Written as a default-deny classifier rather than as an allow-list scan. The
    two scanners above answer "which literals cite a ledger label"; this answers
    the complementary and stricter question "is there a literal here that cites
    nothing", which is the one a silent formatting change could turn from *no*
    to *yes* without any other check noticing.
    """
    lines = text.splitlines()
    unexplained: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if not _ANY_LITERAL.search(line):
            continue
        if _STRUCTURAL_MARK in line:
            continue
        if _LIST_ENTRY.match(line):
            continue
        previous = lines[index - 1] if index else ""
        if _SCALAR_DEF_LINE.match(line) and _LEDGER_PROVENANCE.search(previous):
            continue
        unexplained.append((index + 1, line))
    return unexplained


def _bonferroni_witness_constants(text: str) -> tuple[int, int]:
    """Read the family size and exact denominator proved by the Lean witness."""
    length = _WITNESS_LENGTH.search(text)
    threshold = _WITNESS_THRESHOLD.search(text)
    if length is None or threshold is None:
        raise ValueError("Bonferroni.lean is missing its witnessed length or threshold theorem")
    return int(length.group(1)), int(threshold.group(1))


def _holm_witness_constants(text: str) -> tuple[Fraction, tuple[Fraction, ...], int, int]:
    """Read the Holm witness family, its level, and the two counts Lean proves.

    Raises rather than returning a default on a miss. A parser that quietly
    returned an empty family would make every assertion downstream vacuous, and
    vacuous is the direction this whole file exists to fail away from.
    """
    family = _HOLM_FAMILY.search(text)
    holm = _HOLM_COUNT.search(text)
    bonferroni = _HOLM_BONFERRONI_COUNT.search(text)
    if family is None or holm is None or bonferroni is None:
        raise ValueError("Holm.lean is missing its witness family or one of its count theorems")
    p_values = tuple(Fraction(entry.strip()) for entry in family.group(1).split(","))
    alpha = Fraction(int(holm.group(1)), int(holm.group(2)))
    if alpha != Fraction(int(bonferroni.group(1)), int(bonferroni.group(2))):
        raise ValueError("the two Holm count theorems are stated at different familywise levels")
    return alpha, p_values, int(holm.group(3)), int(bonferroni.group(3))


def _run_generator(destination: Path, ledger: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--out", str(destination), "--ledger", str(ledger)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_lake(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["lake", *args],
        cwd=str(FORMAL_DIR),
        capture_output=True,
        text=True,
        check=False,
    )


def _report_axioms(text: str) -> set[str]:
    """Every axiom named anywhere in an axiom report, un-wrapping long lists."""
    flattened = re.sub(r",\n\s+", ", ", text)
    axioms: set[str] = set()
    for match in re.finditer(r"^'[^']+' depends on axioms: \[(.*)\]", flattened, re.MULTILINE):
        axioms |= {name.strip() for name in match.group(1).split(",") if name.strip()}
    return axioms


# ── The ledger binding: no toolchain required ────────────────────────────────


def test_the_generated_module_exists():
    """Guard vacuity: every check below would pass on an absent file."""
    assert GENERATED.exists(), (
        f"{GENERATED} is missing. Generate it with "
        f"`.venv/bin/python scripts/generate_lean_ledger.py`."
    )


def test_regeneration_is_byte_identical(tmp_path: Path):
    """THE binding check: the committed Lean must be what the ledger produces.

    Fails if the file was hand-edited, and fails if the ledger moved without a
    regeneration. Both are silent otherwise — the tree compiles either way.
    """
    fresh = tmp_path / "LedgerData.lean"
    result = _run_generator(fresh, LEDGER)
    assert result.returncode == 0, f"generator failed:\n{result.stderr}"
    assert fresh.read_bytes() == GENERATED.read_bytes(), (
        "formal/AILM/LedgerData.lean differs from a fresh generation. Either it "
        "was edited by hand — the numbers in it would then be unmeasured — or the "
        "ledger changed and the file was not regenerated, in which case the Lean "
        "proofs are about stale values. Run "
        "`.venv/bin/python scripts/generate_lean_ledger.py`."
    )


def test_a_mutated_ledger_changes_the_output(tmp_path: Path):
    """POSITIVE CONTROL: the generator must actually read the ledger.

    Without this, a generator that ignored ``data/claims.json`` and emitted a
    fixed string would satisfy the byte-identity test above forever.
    """
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    original = payload["claims"]["stage1_unigram_baseline"]["value"]
    payload["claims"]["stage1_unigram_baseline"]["value"] = float(original) + 1.0
    mutated_ledger = tmp_path / "claims.json"
    mutated_ledger.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    mutated_out = tmp_path / "Mutated.lean"
    result = _run_generator(mutated_out, mutated_ledger)
    assert result.returncode == 0, f"generator failed on the mutated ledger:\n{result.stderr}"
    assert mutated_out.read_bytes() != GENERATED.read_bytes(), (
        "Changing a ledger value left the generated Lean unchanged. The generator "
        "is not reading the ledger, so byte identity proves nothing."
    )


def test_literal_scan_finds_the_expected_population():
    """POSITIVE CONTROL for the scanner the next test depends on.

    A regex that stopped matching would make ``test_every_literal_matches...``
    pass by checking nothing at all.
    """
    found = _scan_literals(GENERATED.read_text(encoding="utf-8"))
    assert len(found) >= 30, (
        f"only {len(found)} ledger-backed literals were located in {GENERATED.name}; "
        f"the scanner has stopped matching and the comparison below is vacuous."
    )
    labels = {label for label, _ in found}
    for expected in ("lexical_sweep_alpha_0_3_accuracy", "stage1_unigram_baseline"):
        assert expected in labels, f"{expected} was not located by the literal scanner"


def test_every_literal_matches_the_ledger_exactly():
    """An independent oracle for the conversion, not just for determinism.

    Byte identity says the generator repeats itself. This says it is *right*: a
    generator that consistently truncated a decimal would pass the identity
    check forever and fail here on the first value.
    """
    claims = _claims_text()
    mismatches = []
    for label, value in _scan_literals(GENERATED.read_text(encoding="utf-8")):
        assert label in claims, f"{GENERATED.name} cites unknown claim label {label!r}"
        expected = _exact(claims[label])
        if value != expected:
            mismatches.append((label, str(value), str(expected)))
    assert mismatches == [], (
        f"generated literals disagree with the ledger: {mismatches}. The conversion "
        f"must be exact — decimal text to Fraction to lowest terms — with no "
        f"rounding anywhere."
    )


def test_every_generated_literal_is_accounted_for():
    """DEFAULT-DENY: no number may sit in the generated module unexplained.

    ``test_every_literal_matches_the_ledger_exactly`` checks the literals the
    scanners *find*. It says nothing about a literal they miss — and a literal
    they miss is exactly the dangerous one, because a hand-added or
    reformatted value that no regex matches is a number in a machine-checked
    proof with no recorded provenance, passing every other check in this file.

    So every literal must be one of two things: a ledger-backed value citing its
    label, or a structural constant explicitly marked as such. There is no third
    kind, and this is where that is enforced rather than asserted.
    """
    unexplained = _unexplained_literals(GENERATED.read_text(encoding="utf-8"))
    assert unexplained == [], (
        f"literals in {GENERATED.name} cite no ledger label and are not marked "
        f"{_STRUCTURAL_MARK}: {unexplained}. Either emit the provenance comment the "
        f"scanners read, or mark the value structural — a number in this file with "
        f"neither is unmeasured."
    )


# Importing the real scipy statistics stack can exceed 60 s on the shared drive.
@pytest.mark.timeout(300)
def test_bonferroni_witness_tracks_the_executable_family():
    """The hand-authored Lean arithmetic must use today's Python family size."""
    from ailm.statistics import FAMILYWISE_ALPHA, SIGNIFICANCE_FAMILY

    witnessed_n, witnessed_denominator = _bonferroni_witness_constants(
        BONFERRONI.read_text(encoding="utf-8")
    )
    expected_n = len(SIGNIFICANCE_FAMILY)
    expected_denominator = int(Fraction(expected_n, 1) / Fraction(str(FAMILYWISE_ALPHA)))
    assert witnessed_n == expected_n
    assert witnessed_denominator == expected_denominator


# Importing the real scipy statistics stack can exceed 60 s on the shared drive.
@pytest.mark.timeout(300)
def test_bonferroni_witness_drift_control_is_live():
    """POSITIVE CONTROL: old family arithmetic must disagree observably."""
    from ailm.statistics import FAMILYWISE_ALPHA, SIGNIFICANCE_FAMILY

    planted_stale = """
theorem witnessFamily_length : witnessFamily.length = 11 := by decide
theorem witnessFamily_corrected_threshold :
    bonferroniThreshold (1 / 100) witnessFamily.length = 1 / 1100 := by
"""
    assert _bonferroni_witness_constants(planted_stale) == (11, 1100)
    current = (
        len(SIGNIFICANCE_FAMILY),
        int(Fraction(len(SIGNIFICANCE_FAMILY), 1) / Fraction(str(FAMILYWISE_ALPHA))),
    )
    assert _bonferroni_witness_constants(planted_stale) != current


# Importing the real scipy statistics stack can exceed 60 s on the shared drive.
@pytest.mark.timeout(300)
def test_holm_witness_agrees_with_the_executable_step_down():
    """The Lean step-down witness must survive the real Python procedure.

    ``AILM/Holm.lean`` proves two rejection counts on a synthetic ascending
    family. Those numbers are hand-authored, so on their own they show only that
    Lean can count its own definition. This runs ``holm_step_down`` on the same
    family at the same familywise level and requires the same two counts, which
    is the only thing binding the Lean definition to the procedure the project
    actually executes.

    The strictness assertion is the "converse can fail" half of the manuscript
    sentence: a witness on which the two corrections happened to agree would
    satisfy the containment theorem while exercising nothing.
    """
    from ailm.statistics import FAMILYWISE_ALPHA, holm_step_down

    alpha, p_values, holm_count, bonferroni_count = _holm_witness_constants(
        HOLM.read_text(encoding="utf-8")
    )
    assert alpha == Fraction(str(FAMILYWISE_ALPHA)), (
        "the Lean witness is stated at a familywise level the Python no longer uses"
    )

    family = {f"m{index}": float(p) for index, p in enumerate(p_values)}
    decisions = holm_step_down(family, alpha=float(alpha))
    assert sum(1 for decision in decisions if decision.rejected) == holm_count

    fixed_bar = alpha / len(p_values)
    assert sum(1 for p in p_values if p < fixed_bar) == bonferroni_count
    assert bonferroni_count < holm_count, (
        "the witness no longer separates the two corrections, so the strict "
        "containment theorem is exercised only at equality"
    )


# Importing the real scipy statistics stack can exceed 60 s on the shared drive.
@pytest.mark.timeout(300)
def test_holm_witness_drift_control_is_live():
    """POSITIVE CONTROL: a stale Lean count must disagree observably.

    Without it, a scanner whose regex stopped matching would raise rather than
    silently pass, but a scanner that matched and returned the *planted* numbers
    unchanged would report agreement forever.
    """
    from ailm.statistics import holm_step_down

    planted_stale = """
def witnessFamilySorted : List Rat := [1/1000, 1/500, 1/250, 1/150]
theorem witnessFamilySorted_holm_count : holmRejectCount (1 / 100) witnessFamilySorted = 3 := by
theorem witnessFamilySorted_bonferroni_count :
    bonferroniRejectCount (1 / 100) witnessFamilySorted = 2 := by
"""
    alpha, p_values, holm_count, _ = _holm_witness_constants(planted_stale)
    assert holm_count == 3
    decisions = holm_step_down(
        {f"m{index}": float(p) for index, p in enumerate(p_values)}, alpha=float(alpha)
    )
    assert sum(1 for decision in decisions if decision.rejected) != holm_count


def test_the_literal_accounting_flags_an_unexplained_value():
    """POSITIVE CONTROL for the check above.

    Without it, a classifier that accepted everything (a broken regex, an
    early ``continue``) would report a clean file forever.
    """
    planted = (
        "  the conversion emits `lit n d`, which is prose and not a value\n"
        "/-- a real one (ledger: `stage1_unigram_baseline`). -/\n"
        "def fine : Rat := lit 1 2\n"
        "def sneaky : Rat := lit 3 4\n"
        "  [ lit 5 6   -- a structural one  [structural]\n"
        "  [ lit 7 8   -- a ledger one  (coupling_alpha0_accuracy)\n"
    )
    flagged = _unexplained_literals(planted)
    assert [line for _, line in flagged] == ["def sneaky : Rat := lit 3 4"], (
        f"the accounting classifier flagged {flagged}; it must flag the "
        f"provenance-free literal, and none of: the prose mention, the "
        f"ledger-backed scalar, the structural entry, the ledger list entry."
    )


def _recorder_module() -> ModuleType:
    """Import ``scripts/record_formal_claims.py`` by path.

    ``scripts/`` is not a package, and the alternative — re-implementing its
    patterns here — would test a copy rather than the thing that runs.
    """
    path = PROJECT_ROOT / "scripts" / "record_formal_claims.py"
    spec = importlib.util.spec_from_file_location("_record_formal_claims", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.timeout(240)
def test_formal_recorder_fails_when_the_pinned_toolchain_is_unavailable(tmp_path):
    """NEGATIVE CONTROL: 'nothing recorded' must never be a successful run."""

    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "record_formal_claims.py")],
        cwd=PROJECT_ROOT,
        env={"PATH": str(empty_path), "HOME": str(tmp_path), "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode != 0
    assert "NOTHING WILL BE RECORDED" in result.stderr
    assert "REFUSING TO RECORD" in result.stderr


def test_the_declaration_scanner_reads_the_forms_it_claims_to():
    """The recorder's declaration patterns must cover the modifiers they allow.

    The patterns are column-anchored, which is what keeps them from matching the
    word "theorem" inside a doc comment. This pins the other side: the forms
    they are documented to handle really are handled, so the axiom-report
    coverage check cannot be narrowed by a modifier nobody tested.
    """
    recorder = _recorder_module()
    for source, expected in (
        ("theorem plain : True := trivial", ["plain"]),
        ("private theorem hidden : True := trivial", ["hidden"]),
        ("protected theorem guarded : True := trivial", ["guarded"]),
        ("lemma alt : True := trivial", ["alt"]),
        ("  theorem indented : True := trivial", []),
        ("-- a comment about theorem naming", []),
    ):
        assert recorder._THEOREM.findall(source) == expected, (
            f"declaration scanner read {source!r} as "
            f"{recorder._THEOREM.findall(source)}, expected {expected}"
        )


def test_the_unscannable_guard_fires_on_forms_the_scanner_cannot_read():
    """POSITIVE CONTROL for the recorder's fail-closed refusal.

    A declaration the scanner cannot see is a declaration the axiom report is
    never *required* to cover — the coverage check would keep reporting clean
    for a theorem nobody checked. So the recorder refuses on those forms rather
    than guessing, and this is where that refusal is made to fire. Without it,
    a pattern that matched nothing would be indistinguishable from a clean tree.
    """
    recorder = _recorder_module()
    for hidden in (
        "namespace X\n  theorem indented : True := trivial\n",
        "  instance nested : Inhabited Nat := ⟨0⟩\n",
        "@[simp]\ntheorem attributed : True := trivial\n",
        "  lemma tucked_away : True := trivial\n",
    ):
        assert recorder._UNSCANNABLE.search(hidden), (
            f"the fail-closed guard did not fire on {hidden!r}. This form is "
            f"invisible to the declaration patterns, so it would also be invisible "
            f"to the axiom-report coverage check."
        )
    for fine in (
        "theorem ordinary : True := trivial\n",
        "private instance inst : Inhabited Nat := ⟨0⟩\n",
        "/-- a doc comment mentioning theorem naming -/\n",
        "example : True := trivial\n",
    ):
        assert not recorder._UNSCANNABLE.search(fine), (
            f"the fail-closed guard fired on {fine!r}, which the scanner reads "
            f"perfectly well. A guard that refuses on ordinary code gets disabled."
        )


def test_the_recorder_scans_the_library_recursively():
    """A module under ``AILM/Sub/`` must not escape the scan.

    ``glob`` instead of ``rglob`` would drop it from the declaration count, the
    placeholder scan and the axiom-report coverage check simultaneously — three
    gates going quiet on one file, with no failure anywhere.
    """
    recorder = _recorder_module()
    assert "rglob" in (recorder._library_sources.__doc__ or ""), (
        "the recorder's source discovery no longer documents recursive globbing"
    )
    assert set(recorder._library_sources()) == set(LIBRARY_DIR.rglob("*.lean")), (
        "the recorder and this test disagree about which files make up the library"
    )


def test_the_generator_check_mode_agrees():
    """``--check`` is what a pre-commit or CI step would run; it must not lie."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"generator --check reported stale:\n{result.stdout}{result.stderr}"
    )


def test_axiom_scanner_flags_a_planted_axiom():
    """POSITIVE CONTROL for the axiom parser used by the toolchain tests below."""
    planted = (
        "'AILM.fine' depends on axioms: [propext, Quot.sound]\n"
        "'AILM.bad' depends on axioms: [propext, sorryAx]\n"
        "'AILM.clean' does not depend on any axioms\n"
    )
    assert _report_axioms(planted) == {"propext", "Quot.sound", "sorryAx"}


def test_axiom_scanner_unwraps_a_split_verdict():
    """POSITIVE CONTROL: long axiom lists wrap, and a parser that dropped the
    wrapped entries would under-report exactly the entries with most axioms."""
    wrapped = "'AILM.x' depends on axioms: [propext,\n  Classical.choice,\n  Quot.sound]\n"
    assert _report_axioms(wrapped) == {"propext", "Classical.choice", "Quot.sound"}


# ── Everything below needs the Lean toolchain ────────────────────────────────


@_NEEDS_LAKE
@pytest.mark.timeout(600)
def test_lake_build_succeeds():
    """The proofs must actually check. Without this, every claim above is about
    text that happens to sit in .lean files."""
    result = _run_lake(["build"])
    assert result.returncode == 0, (
        f"`lake build` failed in formal/:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )


@_NEEDS_LAKE
@pytest.mark.timeout(600)
def test_the_negative_control_does_not_compile():
    """POSITIVE CONTROL for the build itself.

    ``Controls/MustFail.lean`` states four falsehoods, one of them about the
    recorded coupling accuracies. It sits outside the lake target, so a green
    build never sees it; it is invoked directly and must be rejected. The
    assertion is on the rejection TEXT, not only the exit code, so a typo or a
    missing import cannot satisfy it.
    """
    result = _run_lake(["env", "lean", str(Path("Controls") / "MustFail.lean")])
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"{MUST_FAIL} compiled. The toolchain is accepting statements this project "
        f"proves false, so no green build in formal/ is evidence of anything.\n{combined}"
    )
    refutations = combined.count("proved that the proposition")
    assert refutations >= 4, (
        f"expected four distinct `decide` refutations, saw {refutations}. The control "
        f"may be failing for an unrelated reason (a typo, a missing import) rather "
        f"than because the statements are false.\n{combined}"
    )
    assert "couplingRecorded" in combined, (
        "the ledger-backed control did not fire. Without it, the controls only show "
        "that `decide` can reject invented numbers, not recorded ones.\n" + combined
    )


@_NEEDS_LAKE
def test_no_incomplete_proofs():
    """Zero placeholder tactics across the shipped library.

    ``sorry`` is a *warning* in Lean, not an error, so a nonzero exit code alone
    would not catch it — which is why this is a text search rather than a build
    assertion.
    """
    offenders = {
        str(path.relative_to(FORMAL_DIR)): _PLACEHOLDER.findall(path.read_text(encoding="utf-8"))
        for path in sorted(LIBRARY_DIR.rglob("*.lean"))
        if _PLACEHOLDER.search(path.read_text(encoding="utf-8"))
    }
    assert offenders == {}, f"incomplete proofs (or near-miss names) found: {offenders}"


@_NEEDS_LAKE
def test_the_placeholder_scan_covers_a_non_empty_file_set():
    """NON-VACUITY: an empty glob satisfies the check above trivially.

    ``rglob``, not ``glob``: a module under ``AILM/Sub/`` would otherwise escape
    the placeholder scan and the axiom-report coverage check together.
    """
    sources = list(LIBRARY_DIR.rglob("*.lean"))
    assert len(sources) >= 5, f"only {len(sources)} library files scanned"
    assert _PLACEHOLDER.search("proof by sorry"), "the placeholder pattern matches nothing"
    assert not _PLACEHOLDER.search("stage4Gold_admits_no_sound_threshold"), (
        "the placeholder pattern is not word-boundary anchored and will report a "
        "false incomplete proof on any identifier containing 'admits'."
    )


@_NEEDS_LAKE
def test_axiom_report_exists_and_is_populated():
    """A report with no verdicts is the vacuity failure mode: it reports clean."""
    assert AXIOM_REPORT.exists(), (
        f"{AXIOM_REPORT} is missing; regenerate with formal/Scripts/axiom_report.sh"
    )
    text = AXIOM_REPORT.read_text(encoding="utf-8")
    verdicts = len(re.findall(r"depends on axioms|does not depend on any axioms", text))
    assert verdicts >= 80, (
        f"the axiom report carries {verdicts} verdicts but the library declares "
        f"considerably more. A report covering part of the tree reports clean for "
        f"the rest."
    )


@_NEEDS_LAKE
def test_no_unexplained_axioms():
    """Every axiom reached by any proof must be one of the four allowed ones."""
    unexpected = sorted(_report_axioms(AXIOM_REPORT.read_text(encoding="utf-8")) - ALLOWED_AXIOMS)
    assert unexpected == [], (
        f"proofs depend on unexplained axioms {unexpected}. Either remove the "
        f"dependency or document it in formal/README.md and add it here with the "
        f"reason — never silently."
    )


@_NEEDS_LAKE
@pytest.mark.parametrize("axiom", FORBIDDEN_AXIOMS)
def test_the_two_axioms_that_may_never_appear(axiom: str):
    """Checked separately from the allow-list so widening that list cannot
    silently admit an incomplete proof or a compiler-trusting one."""
    assert axiom not in AXIOM_REPORT.read_text(encoding="utf-8"), (
        f"{axiom} appears in the axiom report. "
        + (
            "Some proof is incomplete."
            if axiom == "sorryAx"
            else "Some proof closed by native_decide and now trusts the compiler "
            "rather than the kernel."
        )
    )


@_NEEDS_LAKE
def test_the_toolchain_is_pinned_and_dependency_free():
    """Offline by construction: no `[[require]]`, so no Mathlib and no network.

    If a dependency were ever added, `lake build` would need the network and the
    checks above would start passing or failing for reasons unrelated to these
    proofs.
    """
    assert TOOLCHAIN_FILE.exists(), "formal/lean-toolchain is missing; the build is unpinned"
    assert _TOOLCHAIN.startswith("leanprover/lean4:v"), f"unexpected toolchain {_TOOLCHAIN!r}"
    # A bare substring search for "[[require]]" matches the lakefile's own
    # comment explaining that there is no such stanza. The check has to look at
    # TOML structure — an uncommented table header — not at text.
    stanzas = [
        line
        for line in (FORMAL_DIR / "lakefile.toml").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("[[require]]")
    ]
    assert stanzas == [], (
        f"formal/lakefile.toml declares a dependency ({stanzas}). The formal layer "
        f"is deliberately dependency-free so the build is offline and fast."
    )
    manifest = json.loads((FORMAL_DIR / "lake-manifest.json").read_text(encoding="utf-8"))
    assert manifest["packages"] == [], f"unexpected lake packages: {manifest['packages']}"


# The report driver is generated, but compilation and coverage remain separate.
def test_axiom_driver_tracks_added_and_removed_declarations(tmp_path):
    recorder = _recorder_module()
    library = tmp_path / "AILM" / "Sub"
    library.mkdir(parents=True)
    source = library / "Control.lean"
    source.write_text("namespace AILM\ntheorem first : True := trivial\nend AILM\n")
    command = [sys.executable, str(Path(recorder.__file__)), "--formal-dir", str(tmp_path)]
    target = tmp_path / "Scripts" / "AxiomReport.lean"
    missing = subprocess.run([*command, "--check-axiom-script"], capture_output=True, text=True)
    assert missing.returncode == 1
    assert "missing or stale" in missing.stderr
    assert not target.exists()
    for body in (
        "theorem first : True := trivial\n",
        "theorem first : True := trivial\nlemma added : True := trivial\n",
        "lemma added : True := trivial\ninstance decidableTrue : Decidable True := inferInstance\n",
    ):
        source.write_text("namespace AILM\n" + body + "end AILM\n")
        generated = subprocess.run(
            [*command, "--generate-axiom-script"], capture_output=True, text=True
        )
        assert generated.returncode == 0, generated.stdout + generated.stderr
        content = target.read_text()
        expected = {"AILM." + line.split()[1] for line in body.splitlines()}
        assert set(re.findall(r"^#print axioms (\S+)", content, re.MULTILINE)) == expected
        assert "AILM.Sub.Control" in content
        checked = subprocess.run([*command, "--check-axiom-script"], capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr
    identity = target.stat().st_ino
    prior = target.read_bytes()
    source.write_text("namespace AILM\n  theorem invisible : True := trivial\nend AILM\n")
    refused = subprocess.run([*command, "--generate-axiom-script"], capture_output=True, text=True)
    assert refused.returncode == 1
    assert "scanner cannot read" in refused.stderr
    assert target.read_bytes() == prior and target.stat().st_ino == identity


def test_canonical_axiom_driver_is_fresh():
    from ailm.formal_inventory import render_axiom_script

    texts = {path.relative_to(FORMAL_DIR): path.read_text() for path in LIBRARY_DIR.rglob("*.lean")}
    actual = (FORMAL_DIR / "Scripts" / "AxiomReport.lean").read_text()
    assert actual == render_axiom_script(texts), "regenerate the axiom driver before publication"


def test_canonical_module_counts_are_bound_to_the_recorded_ledger():
    from ailm.claims import load_ledger
    from ailm.formal_inventory import module_theorem_counts

    texts = {path.relative_to(FORMAL_DIR): path.read_text() for path in LIBRARY_DIR.rglob("*.lean")}
    counts = module_theorem_counts(texts)
    ledger = load_ledger(PROJECT_ROOT)
    for label, value in counts.items():
        entry = ledger.get(label)
        assert entry is not None and entry["value"] == value, label
    total = ledger.get("formal_theorem_count")
    assert total is not None and sum(counts.values()) == total["value"]


@pytest.mark.parametrize(
    "texts,match",
    [
        ({}, "no .lean files"),
        ({Path("AILM/Empty.lean"): "namespace AILM\nend AILM\n"}, "no named theorems"),
        (
            {
                Path("AILM/One.lean"): "theorem repeated : True := trivial\n",
                Path("AILM/Two.lean"): "lemma repeated : True := trivial\n",
            },
            "duplicate declaration",
        ),
    ],
)
def test_axiom_driver_refuses_incomplete_or_ambiguous_inventories(texts, match):
    from ailm.formal_inventory import render_axiom_script

    with pytest.raises(ValueError, match=match):
        render_axiom_script(texts)


def test_module_theorem_counts_include_witnesses_but_not_instances_or_examples():
    from ailm.formal_inventory import module_theorem_counts

    texts = {
        Path("AILM/ControlModule.lean"): (
            "theorem result : True := trivial\n"
            "/-! ## Satisfiability witnesses -/\n"
            "lemma witness : True := trivial\n"
            "instance inhabited : Inhabited Nat := inferInstance\n"
            "example : True := trivial\n"
        ),
        Path("AILM/Constants.lean"): "def constant : Nat := 1\n",
    }
    assert module_theorem_counts(texts) == {
        "formal_module_constants_theorem_count": 0,
        "formal_module_control_module_theorem_count": 2,
    }
    texts[Path("AILM/ControlModule.lean")] += "theorem added : True := trivial\n"
    assert module_theorem_counts(texts)["formal_module_control_module_theorem_count"] == 3


def test_module_theorem_counts_refuse_colliding_paths():
    from ailm.formal_inventory import module_theorem_counts

    with pytest.raises(ValueError, match="duplicate module claim label"):
        module_theorem_counts(
            {
                Path("AILM/Sub/Control.lean"): "theorem first : True := trivial\n",
                Path("AILM/Sub_Control.lean"): "theorem second : True := trivial\n",
            }
        )


@_NEEDS_LAKE
@pytest.mark.timeout(1200)
def test_failed_lean_report_preserves_the_published_report(tmp_path):
    # A real Lean error in the report step must not truncate an earlier receipt.
    formal = tmp_path / "formal"
    shutil.copytree(FORMAL_DIR, formal, ignore=shutil.ignore_patterns(".lake", "axiom_report.txt"))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("record_formal_claims.py", "_artifact_io.py"):
        shutil.copyfile(PROJECT_ROOT / "scripts" / name, scripts / name)
    # This module compiles when reached, but is absent from import AILM. The
    # scanner includes it and the print command fails on its unknown identifier.
    (formal / "AILM" / "Unimported.lean").write_text(
        "namespace AILM\ntheorem unimportedControl : True := trivial\nend AILM\n"
    )
    report = formal / "axiom_report.txt"
    report.write_text("previous verified report control\n")
    result = subprocess.run(
        ["bash", str(formal / "Scripts" / "axiom_report.sh")],
        capture_output=True,
        text=True,
        env={**os.environ, "AILM_PYTHON": sys.executable},
        timeout=600,
    )
    assert result.returncode != 0, "an unimported declaration unexpectedly printed axioms"
    assert "unimportedControl" in result.stdout + result.stderr
    assert report.read_text() == "previous verified report control\n"


def _writer_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A hermetic copy of the writer's world: formal tree, scripts, staged prior report.

    The copy has no ``.lake``, so the only ``lake`` on PATH is the fake below; a
    fake can demonstrate that the build stage ran, never publish a report — the
    publication step sits behind the verdict check, and no fixture gives it a
    passing report.
    """
    formal = tmp_path / "formal"
    shutil.copytree(
        FORMAL_DIR,
        formal,
        ignore=shutil.ignore_patterns(".lake", "axiom_report.txt", "__pycache__"),
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("record_formal_claims.py", "_artifact_io.py"):
        shutil.copyfile(PROJECT_ROOT / "scripts" / name, scripts / name)
    report = formal / "axiom_report.txt"
    report.write_text("previous verified report control\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "dirname", "mktemp", "rm", "cat", "mv"):
        resolved = shutil.which(tool)
        assert resolved is not None, f"the hermetic fixture needs {tool} on PATH"
        (bin_dir / tool).symlink_to(resolved)
    return formal, report, bin_dir


def _run_writer(formal: Path, bin_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run the writer with only the fixture's tools on PATH."""
    return subprocess.run(
        ["bash", str(formal / "Scripts" / "axiom_report.sh")],
        capture_output=True,
        text=True,
        env={
            "PATH": str(bin_dir),
            "AILM_PYTHON": sys.executable,
            "HOME": os.environ.get("HOME", ""),
            "TMPDIR": os.environ.get("TMPDIR", ""),
        },
        timeout=480,
    )


@pytest.mark.timeout(960)
def test_a_missing_rg_is_a_missing_tool_refused_before_any_work(tmp_path):
    """Missing ripgrep must read as a missing tool, never as an empty report (S19)."""
    formal, report, bin_dir = _writer_fixture(tmp_path)
    fake_lake = bin_dir / "lake"
    fake_lake.write_text("#!/bin/bash\ntouch lake-ran.marker\nexit 0\n")
    fake_lake.chmod(0o755)
    result = _run_writer(formal, bin_dir)
    assert result.returncode == 2, result.stderr
    assert "rg not found" in result.stderr
    assert "no axiom verdicts" not in result.stderr
    assert not (formal / "lake-ran.marker").exists(), (
        "the missing-tool refusal must fire before the build runs"
    )
    assert report.read_text() == "previous verified report control\n"


@pytest.mark.timeout(960)
def test_a_failing_rg_is_a_tool_failure_not_vacuity(tmp_path):
    """An rg that runs and fails must be reported with its status, not as vacuity (S19)."""
    formal, report, bin_dir = _writer_fixture(tmp_path)
    fake_lake = bin_dir / "lake"
    fake_lake.write_text("#!/bin/bash\nexit 0\n")
    fake_lake.chmod(0o755)
    fake_rg = bin_dir / "rg"
    fake_rg.write_text("#!/bin/bash\nexit 2\n")
    fake_rg.chmod(0o755)
    result = _run_writer(formal, bin_dir)
    assert result.returncode == 4, result.stderr
    assert "rg failed (exit 2)" in result.stderr
    assert "no axiom verdicts" not in result.stderr
    assert report.read_text() == "previous verified report control\n"
    assert not list(formal.glob("axiom_report.*.txt")), "the staged temp must not survive"


@pytest.mark.timeout(960)
def test_zero_verdicts_from_a_working_rg_still_refuse_as_vacuity(tmp_path):
    """The genuine vacuity refusal keeps its meaning and its exit once rg is working."""
    formal, report, bin_dir = _writer_fixture(tmp_path)
    fake_lake = bin_dir / "lake"
    fake_lake.write_text('#!/bin/bash\necho "lean stage: no verdict lines"\nexit 0\n')
    fake_lake.chmod(0o755)
    real_rg = shutil.which("rg")
    assert real_rg is not None, "the vacuity control needs a real rg"
    (bin_dir / "rg").symlink_to(real_rg)
    result = _run_writer(formal, bin_dir)
    assert result.returncode == 3, result.stderr
    assert "axiom_report.txt contains no axiom verdicts — refusing" in result.stderr
    assert "rg failed" not in result.stderr
    assert "rg not found" not in result.stderr
    assert report.read_text() == "previous verified report control\n"


@pytest.mark.parametrize(
    "declaration",
    [
        "  private theorem hidden : True := trivial",
        "@[simp] theorem decorated : True := trivial",
        "theorem α : True := trivial",
        "theorem xα : True := trivial",
        "instance : Decidable True := inferInstance",
    ],
)
def test_axiom_driver_refuses_unreadable_declarations_even_beside_valid_ones(declaration):
    from ailm.formal_inventory import render_axiom_script

    text = "theorem visible : True := trivial\n" + declaration + "\n"
    with pytest.raises(ValueError, match="scanner cannot read"):
        render_axiom_script({Path("AILM/Control.lean"): text})
