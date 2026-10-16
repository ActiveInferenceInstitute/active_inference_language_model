"""The bibliography and the prose must agree in both directions.

Nothing checked this. A `[@key]` with no entry renders as a bare `[?]` in the
PDF, and an entry nobody cites sits in the file indefinitely — one did, for a
probe that had been retired from the benchmark, and it was found by reading
rather than by a gate. Both failures are silent in extracted text.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = PROJECT_ROOT / "docs" / "manuscript"
BIB = MANUSCRIPT / "references.bib"

#: Not sections: documentation about the manuscript rather than of it.
_EXCLUDED = {"SYNTAX.md", "AGENTS.md", "README.md", "preamble.md"}

#: Entry keys. Anchored at line start so a key mentioned inside the file's
#: ``@comment`` provenance block is not mistaken for a declaration.
_ENTRY_RE = re.compile(r"^@[a-zA-Z]+\{([^,\s]+),", re.MULTILINE)

#: Pandoc citation syntax: ``[@key]``, ``[@a; @b]``, ``@key``, ``[-@key]``.
_CITE_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_:.#$%&+?<>~/-]*)")

#: Cross-reference labels share the ``@`` sigil with citations and are not bib
#: keys: ``[@sec:x]``, ``[@fig:x]``, ``[@eq:x]``, ``[@tbl:x]``. Excluded by
#: prefix rather than by "contains a colon", which would silently swallow a real
#: key; :func:`test_no_entry_uses_a_cross_reference_prefix` proves the exclusion
#: never hides one.
_XREF_PREFIXES = ("sec:", "fig:", "eq:", "tbl:", "lst:")

# Fields that BibTeX emits as ordinary LaTeX prose.  A raw underscore in one
# of these fields enters math mode, corrupts the bibliography, and can still
# leave behind a PDF that a permissive multi-pass renderer accepts.
_PROSE_FIELD_RE = re.compile(
    r"(?m)^\s*(?P<field>author|title|journal|booktitle|series|publisher|"
    r"institution|address|note)\s*=\s*\{"
)


def _raw_prose_underscores(text: str) -> list[tuple[str, str]]:
    """Return prose fields containing an underscore not escaped for LaTeX."""
    defects: list[tuple[str, str]] = []
    for match in _PROSE_FIELD_RE.finditer(text):
        opening = match.end() - 1
        depth = 0
        value_end = None
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    value_end = index
                    break
        assert value_end is not None, f"unterminated {match.group('field')} field"
        value = text[opening + 1 : value_end]
        raw = re.search(r"(?<!\\)_", value)
        if raw is not None:
            defects.append((match.group("field"), value.strip()))
    return defects


def _entries() -> set[str]:
    return set(_ENTRY_RE.findall(BIB.read_text(encoding="utf-8")))


def _cited() -> set[str]:
    cited: set[str] = set()
    for path in sorted(MANUSCRIPT.glob("*.md")):
        if path.name in _EXCLUDED:
            continue
        text = path.read_text(encoding="utf-8")
        # Fenced code and inline code can contain @ for other reasons.
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]*`", " ", text)
        cited |= {
            key.rstrip(".,;:)")
            for key in _CITE_RE.findall(text)
            if not key.startswith(_XREF_PREFIXES)
        }
    return cited


def test_the_scan_finds_entries_and_citations() -> None:
    """NON-VACUITY: two empty sets satisfy every rule below.

    Stated as a floor here deliberately, unlike the caption gate: there is no
    independent population to derive this from, so the honest form is a check
    that both scanners returned something rather than an exact count that would
    have to be edited on every citation added.
    """
    assert len(_entries()) > 50, "the entry scanner found almost nothing; the .bib format changed"
    assert len(_cited()) > 20, "the citation scanner found almost nothing; the syntax changed"
    assert "friedman2024cerebrum" in _cited()


def test_every_citation_has_an_entry() -> None:
    """A cited key with no entry renders as `[?]` and nothing else complains."""
    missing = sorted(_cited() - _entries())
    assert missing == [], (
        f"the manuscript cites {missing}, which references.bib does not define. "
        f"Each renders as a bare marker in the PDF, which extracted text does "
        f"not distinguish from a real citation."
    )


def test_every_entry_is_cited() -> None:
    """An uncited entry is a claim about the reading behind the paper that the
    paper does not make. One survived a probe's retirement from the benchmark
    and was removed by hand; this is what would have caught it."""
    orphans = sorted(_entries() - _cited())
    assert orphans == [], (
        f"references.bib defines {orphans}, which nothing cites. Either cite the "
        f"work or remove the entry; an uncited entry outlives whatever it was "
        f"added for."
    )


def test_no_entry_uses_a_cross_reference_prefix() -> None:
    """The exclusion above must not be able to hide a real bibliography key.

    Cross-reference labels are dropped from the citation scan by prefix. If an
    entry key ever began with one of those prefixes, it would be dropped too,
    and both directions of the agreement check would pass on it vacuously.
    """
    shadowed = sorted(key for key in _entries() if key.startswith(_XREF_PREFIXES))
    assert shadowed == [], (
        f"these bibliography keys begin with a cross-reference prefix and would "
        f"be invisible to the citation scan: {shadowed}. Rename them."
    )


def test_planted_violations_are_caught() -> None:
    """POSITIVE CONTROL: both directions of the rule can actually fail."""
    entries, cited = {"real_key"}, {"real_key", "ghost_key"}
    assert sorted(cited - entries) == ["ghost_key"]
    assert sorted({"real_key", "unused_key"} - cited) == ["unused_key"]


def test_no_entry_carries_an_empty_field() -> None:
    """The file's policy is to OMIT what could not be confirmed, never to fill it.

    An empty brace is the failure mode that policy invites: it looks like a
    recorded value, renders as nothing, and reads to a later editor as a field
    someone forgot rather than one deliberately withheld.
    """
    empties = re.findall(r"^\s*([a-z]+)\s*=\s*\{\s*\},?\s*$", BIB.read_text(), re.MULTILINE)
    assert empties == [], (
        f"references.bib has empty fields {empties}. The policy is to omit an "
        f"unconfirmed field entirely, so that its absence is visible."
    )


def test_bibliography_prose_escapes_latex_underscores() -> None:
    """A generated PDF is not proof when TeX recovered from a prose error."""
    defects = _raw_prose_underscores(BIB.read_text(encoding="utf-8"))
    assert defects == [], (
        f"bibliography prose contains raw underscores that make LaTeX enter math mode: {defects}"
    )


def test_bibliography_underscore_guard_is_live() -> None:
    """NEGATIVE CONTROL: the guard rejects the exact failure seen in a note."""
    planted = "@misc{x,\n  note = {file planted_name.pdf}\n}\n"
    assert _raw_prose_underscores(planted) == [("note", "file planted_name.pdf")]
