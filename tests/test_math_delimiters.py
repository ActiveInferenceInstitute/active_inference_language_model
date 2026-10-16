"""Math delimiters match the region they sit in.

``SYNTAX.md`` states the rule and the reason: pandoc passes a raw ``amsthm``
environment through verbatim, so the renderer's Unicode remap reads it next and
its math-region detector recognizes ``\\(``, ``\\[``, and ``\\begin{equation}``
— not ``$``. Dollar math inside such a block is therefore treated as prose and
every command in it is silently rewritten.

The converse failure is the one that motivated this file. Pandoc's markdown
reader does not accept single-backslash TeX delimiters, so ``\\[ ... \\]`` in
ordinary prose is escaped to a literal ``{[} ... {]}`` and the equation is
emitted as text. The main PDF path swallowed that; the Beamer path did not, and
the error it reported named a font shape rather than the delimiter.

Both directions are one rule about agreement between a delimiter and its
region, so both are checked here.

Guards on the guard: the file list must be non-empty, and a planted violation of
each direction must be caught.
"""

from __future__ import annotations

import re
from pathlib import Path

from ailm.docs_hydration import strip_code

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = PROJECT_ROOT / "docs" / "manuscript"

#: Documentation *about* the manuscript quotes both forms to explain them.
_EXCLUDED = {"SYNTAX.md", "AGENTS.md", "README.md", "preamble.md"}

#: Raw LaTeX environments pandoc passes through verbatim.
_RAW_ENVIRONMENTS = ("definition", "theorem", "proposition", "lemma", "corollary")

_BEGIN = re.compile(r"\\begin\{(" + "|".join(_RAW_ENVIRONMENTS) + r")\}")
_END = re.compile(r"\\end\{(" + "|".join(_RAW_ENVIRONMENTS) + r")\}")

#: ``\(`` or ``\[`` used as a math delimiter. A doubled backslash is a line
#: break inside a matrix, not a delimiter, so it must not match.
_BACKSLASH_MATH = re.compile(r"(?<!\\)\\[(\[]")

#: Any dollar sign, single or doubled.
_DOLLAR_MATH = re.compile(r"(?<!\\)\$")


def _raw_regions(text: str) -> list[bool]:
    """Return, per line, whether that line sits inside a raw LaTeX environment."""
    inside = False
    flags: list[bool] = []
    for line in text.splitlines():
        opened = bool(_BEGIN.search(line))
        closed = bool(_END.search(line))
        flags.append(inside or opened)
        if opened:
            inside = True
        if closed:
            inside = False
    return flags


def find_violations(text: str, name: str = "<memory>") -> list[str]:
    """Report every delimiter that disagrees with the region containing it."""
    stripped = strip_code(text)
    flags = _raw_regions(stripped)
    violations: list[str] = []
    for number, (line, in_raw) in enumerate(
        zip(stripped.splitlines(), flags, strict=True), start=1
    ):
        if in_raw:
            if _DOLLAR_MATH.search(line):
                violations.append(
                    f"{name}:{number}: dollar math inside a raw LaTeX environment "
                    f"(use \\( \\) or \\[ \\]): {line.strip()}"
                )
        elif _BACKSLASH_MATH.search(line):
            violations.append(
                f"{name}:{number}: backslash math delimiter in markdown prose "
                f"(use $ or $$): {line.strip()}"
            )
    return violations


def _scanned_files() -> list[Path]:
    return sorted(p for p in MANUSCRIPT.glob("*.md") if p.name not in _EXCLUDED)


def test_the_scan_covers_files() -> None:
    """An empty file list would pass any rule."""
    assert len(_scanned_files()) > 10


def test_every_delimiter_matches_its_region() -> None:
    violations: list[str] = []
    for path in _scanned_files():
        violations.extend(find_violations(path.read_text(encoding="utf-8"), path.name))
    assert not violations, "\n".join(violations)


def test_backslash_math_in_prose_is_caught() -> None:
    """Negative control for the direction that broke the Beamer render."""
    planted = "The reduction is\n\n\\[\n  a = b\n\\]\n"
    assert find_violations(planted)


def test_dollar_math_in_a_raw_environment_is_caught() -> None:
    """Negative control for the direction ``SYNTAX.md`` documents."""
    planted = "\\begin{proposition}[Planted]\nLet $t$ be a threshold.\n\\end{proposition}\n"
    assert find_violations(planted)


def test_conforming_text_passes() -> None:
    """Positive control: the rule does not fire on correct usage."""
    conforming = (
        "The reduction is $$a = b$$ for inline $x$.\n"
        "\n"
        "\\begin{proposition}[Conforming]\n"
        "Let \\(t\\) be a threshold, so \\[a = b\\].\n"
        "\\end{proposition}\n"
    )
    assert not find_violations(conforming)
