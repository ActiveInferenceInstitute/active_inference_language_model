"""Shared Lean declaration inventory and deterministic axiom-driver rendering.

This is a deliberately bounded source scanner, not a Lean parser. The library
uses column-zero named declarations in the AILM namespace. Unknown indented or
attribute-prefixed forms are refused. Compilation and the recorder's two-way
comparison with Lean's output remain necessary even for a fresh driver.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

#: Declaration modifiers this scanner understands ahead of a declaration
#: keyword. Kept explicit rather than as a permissive ``.*`` so that a form
#: nobody anticipated trips ``_UNSCANNABLE`` below instead of being silently
#: half-matched.
_MODIFIERS = r"(?:private\s+|protected\s+|noncomputable\s+)*"

_THEOREM = re.compile(
    rf"^{_MODIFIERS}(?:theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'!?]*)", re.MULTILINE
)
_INSTANCE = re.compile(rf"^{_MODIFIERS}instance\s+([A-Za-z_][A-Za-z0-9_'!?]*)", re.MULTILINE)
_EXAMPLE = re.compile(rf"^{_MODIFIERS}example\b", re.MULTILINE)
_NATIVE_DECIDE = re.compile(r"\bnative_decide\b")

#: Declaration forms this scanner would MISS, so it refuses instead.
#:
#: The patterns above are anchored to column zero and to a known modifier set.
#: That is deliberate — an unanchored scan matches the word "theorem" inside doc
#: comments — but it means an indented declaration, or one behind an attribute
#: like ``@[simp]``, would be invisible to BOTH the declaration count and the
#: axiom-report coverage check. Invisible to the coverage check is the dangerous
#: half: a theorem the scanner cannot see is a theorem the report is not
#: required to cover, and the gate would go on reporting clean.
#:
#: So the scanner does not guess. Anything that looks like a declaration it was
#: not built to read is a refusal, and whoever adds the form updates the scanner.
_UNSCANNABLE = re.compile(
    rf"^(?:[ \t]+{_MODIFIERS}(?:theorem|lemma|instance|example)\b|@\[[^\n]*)",
    re.MULTILINE,
)

AXIOM_SCRIPT_HEADER = """/-
  Generated axiom report driver; do not hand-edit.
  Writer: scripts/record_formal_claims.py --generate-axiom-script

  Run from `formal/`:

      ./Scripts/axiom_report.sh

  Prints the axiom dependencies of **every named theorem** in the tree. This
  file is deliberately outside the `AILM/` library root so `lake build` does not
  compile it; the report is a separate, explicit step.

  ## Allowed axiom set

      propext, Classical.choice, Quot.sound, funext

  `Classical.choice` is on that list for a measured reason, not a permissive
  one: Lean **core's own** `Rat.add_comm` reports
  `[propext, Classical.choice, Quot.sound]`. A gate demanding only
  `propext / Quot.sound / funext` would be unsatisfiable against core `Rat` and
  would fail for reasons that have nothing to do with this project's proofs.

  What the allowlist deliberately excludes:

  * `sorryAx`   — an incomplete proof. `sorry` is a *warning*, not an error, so
                  a nonzero exit code alone would not catch it.
  * `Lean.ofReduceBool` — the trusted-code axiom `native_decide` introduces.
                  `native_decide` is not used anywhere in this tree: every
                  computed fact closes by `decide`, which is kernel-checked.
                  If this axiom ever appears in the report, some proof started
                  trusting the compiler instead of the kernel.

  Entries are generated from the same declaration scan the recorder uses.
  Its independent comparison with Lean output still checks coverage both ways.
  Source scanning alone is not evidence of a compiled or axiom-checked proof.
-/
"""


def declaration_inventory(text_by_file: Mapping[Path, str]) -> dict[Path, tuple[str, ...]]:
    """Return every theorem and named instance, refusing ambiguous inventories."""
    if not text_by_file:
        raise ValueError("the formal library contains no .lean files")
    inventory: dict[Path, tuple[str, ...]] = {}
    seen: set[str] = set()
    for path, text in sorted(text_by_file.items()):
        unsupported = sorted({match.strip() for match in _UNSCANNABLE.findall(text)})
        if unsupported:
            raise ValueError(f"declaration forms this scanner cannot read in {path}: {unsupported}")
        declaration_lines = re.findall(
            rf"^{_MODIFIERS}(?:theorem|lemma|instance)\b[^\n]*", text, re.MULTILINE
        )
        for line in declaration_lines:
            match = _THEOREM.match(line) or _INSTANCE.match(line)
            if match is None or (line[match.end() :] and line[match.end()] not in " \t:("):
                raise ValueError(f"declaration name this scanner cannot read in {path}: {line}")
        matches = sorted(
            [*_THEOREM.finditer(text), *_INSTANCE.finditer(text)], key=lambda m: m.start()
        )
        names = tuple(f"AILM.{match.group(1)}" for match in matches)
        for name in names:
            if name in seen:
                raise ValueError(f"duplicate declaration {name} in {path}")
            seen.add(name)
        inventory[path] = names
    if not seen:
        raise ValueError("the formal library contains no named theorems or instances")
    return inventory


def render_axiom_script(text_by_file: Mapping[Path, str]) -> str:
    """Emit one print command per scanned declaration, in stable module order."""
    inventory = declaration_inventory(text_by_file)
    lines = [AXIOM_SCRIPT_HEADER + "import AILM", ""]
    for path, names in inventory.items():
        if not names:
            continue
        lines.append(f"/-! ## {path.with_suffix('').as_posix().replace('/', '.')} -/")
        lines.extend(f"#print axioms {name}" for name in names)
        lines.append("")
    return "\n".join(lines)


def module_theorem_counts(text_by_file: Mapping[Path, str]) -> dict[str, int]:
    """Derive total named-theorem counts per module, including named witnesses."""
    declaration_inventory(text_by_file)
    counts: dict[str, int] = {}
    for path, text in sorted(text_by_file.items()):
        # Paths are relative to formal/, including the AILM library component.
        module = "_".join(path.with_suffix("").parts[1:])
        slug = re.sub(r"(?<!^)(?=[A-Z])", "_", module).lower()
        label = f"formal_module_{slug}_theorem_count"
        if label in counts:
            raise ValueError(f"duplicate module claim label {label}")
        counts[label] = len(_THEOREM.findall(text))
    return counts
