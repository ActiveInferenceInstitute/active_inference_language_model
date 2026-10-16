"""Numbers in documentation, resolved from the ledger rather than typed.

## The two halves of the problem

The manuscript has had this solved since Stage 1: prose carries ``{{TOKEN}}``,
the renderer substitutes from ``data/claims.json``, and an unresolved token in a
PDF is a loud failure. Documentation had only half the mechanism. The convention
is ``claim:<label>`` and ``tests/test_claims_discipline.py`` checks that every
tag resolves — but the tag is *not* substituted anywhere, so a doc that wants to
show a reader an actual number has to type it.

Authors type it. Decision 13's vocabulary table was typed markdown, and by the
time anyone rebuilt it every one of its four node counts was wrong (the then-shipped
2,000-word vocabulary yielded 5,693 nodes against a printed 5,700; the current
vendored lexicon, reduced to 38,007 types on 2026-08-23, regenerates 5,643) and two of its
four coverage figures did not round from the ledger values recorded in the very
same row. Nothing regenerated the table, so nothing noticed. The conclusion
never depended on those digits, which is precisely why the drift survived.

So this module supplies both halves:

**Hydration** — :func:`hydrate` substitutes every ``claim:<label>`` with its
formatted value, using the same formatter the manuscript uses, so a reader gets
numbers and an author never types one. ``scripts/z_render_docs.py`` writes the
hydrated tree to ``output/docs/``.

**Detection** — :func:`find_hardcoded_values` reads the other direction: a
numeral in prose that *exactly* equals a recorded claim is a copy of that claim,
and copies drift. This is the check that would have caught decision 13.

## Why the detector is deliberately narrow

A ledger of 1,241 claims contains many small values — 2, 8, 16, 0.5 — that
collide with ordinary prose by coincidence. A detector that flagged those would
be noise, and a noisy check gets suppressed wholesale, which is worse than no
check. So a numeral is only suspected when it is **distinctive**: three or more
decimal places, or a thousands-separated integer. Those effectively never
coincide. The cost is that a typed ``0.13`` is invisible to this check; the
benefit is that every report is real.

Exemptions live in :data:`LITERAL_EXEMPTIONS` and each one states why. A
suppression without a reason is how a check stops meaning anything.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ailm.value_formatting import format_ledger_value

__all__ = [
    "CLAIM_TAG_RE",
    "COINCIDENTAL_MATCHES",
    "DISTINCTIVE_NUMERAL_RE",
    "LITERAL_EXEMPTIONS",
    "Hardcode",
    "find_absorbed_tags",
    "find_hardcoded_values",
    "hydrate",
    "is_exempt",
    "strip_code",
]

#: A claim reference in prose. Matches the pattern ``ailm.claims`` scans for.
CLAIM_TAG_RE = re.compile(r"claim:([A-Za-z0-9_\-]+)")

#: A numeral distinctive enough that matching a ledger value is not coincidence:
#: three or more decimal places, or a thousands-separated integer.
DISTINCTIVE_NUMERAL_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+\.\d{3,})(?![\w.])")

#: Files where a literal numeral matching a claim is correct, with the reason.
#: Keyed by repository-relative path.
LITERAL_EXEMPTIONS: dict[str, str] = {
    "docs/research_queue.md": (
        "Proposer-stage document. Its Evidence sub-bullets quote what a probe or "
        "scratchpad run measured, and the whole value of that record is that the "
        "figure can be compared against the ledger later. Substituting the ledger "
        "value would make every entry agree with the ledger by construction and "
        "delete the comparison — which is how the 153/58 exclusion-count error was "
        "found in the first place."
    ),
    "docs/syntax_guide.md": (
        "Documents the mechanism itself. Its examples must show literal tags and "
        "literal values side by side, or the guide cannot demonstrate what it "
        "describes."
    ),
    "docs/testing_philosophy.md": (
        "Quotes numbers from worked examples of failure modes, several of which are "
        "deliberately values the ledger no longer holds."
    ),
    "CHANGELOG.md": (
        "A dated record of what each release measured. Several entries quote a "
        "superseded value beside the current one on purpose — `stage5_auc 0.7024 "
        "to 0.7037` is the whole content of that line — and a live reference would "
        "rewrite both halves on every re-record, turning a history into a mirror of "
        "the present."
    ),
    "docs/development_history.md": (
        "The verbatim pre-1.0.0 changelog history moved out of `CHANGELOG.md`; the "
        "same dated record, exempt for the same reason."
    ),
}

#: Individual numerals whose matching label is a coincidence rather than a copy,
#: as ``(path, literal, anchor): reason``. The anchor is a distinctive substring
#: of the line the numeral sits on.
#:
#: **Keyed on content, not on line number.** Absolute line numbers were the
#: original key and they drifted four times in a single session: every append to
#: ``docs/decisions.md`` shifted five exemptions, and each time the literal check
#: fired and each time the fix was to renumber. The literal check was doing its
#: job -- it is what stopped a moved key from silently suppressing whatever number
#: happened to land on that line -- but a key that changes whenever an unrelated
#: paragraph is added is a key that will eventually be renumbered carelessly.
#:
#: An anchor is stable under insertion anywhere else in the file and still cannot
#: drift onto a different number: it must appear on the same line as the literal,
#: and :func:`find_hardcoded_values` requires both. If the sentence is rewritten
#: the exemption stops applying and the scan fails, which is the correct outcome:
#: a reworded line deserves a fresh judgment about whether the coincidence holds.
COINCIDENTAL_MATCHES: dict[tuple[str, str, str], str] = {
    (
        "docs/decisions.md",
        "1,322",
        "raw positions held only 480 distinct ones",
    ): "a raw position count, not the McNemar discordant count whose value it equals",
    (
        "docs/decisions.md",
        "0.000",
        "channel_factorial_anchor_accuracy",
    ): "a zero margin in a results table, not a cache-freshness fraction",
    (
        "docs/decisions.md",
        "1.0000",
        "nothing anywhere: p = 0.5966",
    ): "one of four p-values listed in a row, not an argmax-agreement rate",
    (
        "docs/decisions.md",
        "0.3900",
        "below the comparator, having closed",
    ): "a fidelity delta below a comparator, not a crossover target",
    (
        "docs/decisions.md",
        "2,000",
        "claim:stage3_trie_nodes_v2000",
    ): "a row label of the vocabulary ladder, which is the table's independent "
    "axis rather than a measurement; the rungs are fixed by TRIE_LADDER_SIZES "
    "in the recorder, and one of four rungs as a reference beside three "
    "literals reads worse than four literals",
    (
        "docs/design/stage3_word.md",
        "2,000",
        "claim:stage3_trie_nodes_v2000",
    ): "the same ladder axis in the second copy of the table",
    (
        "docs/decisions.md",
        "5,643",
        "claim:stage3_trie_nodes_v2000 now regenerates as",
    ): "the decision record deliberately types the ledger value it reconciled to; "
    "the sentence's content is that the regenerated number (5,643) differs from "
    "the historical receipt (5,693) — substituting a live reference would erase "
    "the very discrepancy the entry documents",
    (
        "docs/decisions.md",
        "0.025",
        "replayed in a new process 100/0 in 0.025 s",
    ): "a wall-clock timing quoted as measured evidence in the decision record, "
    "coincidentally equal to bench_synth_context_family_alpha; timings are not "
    "ledger-owned quantities",
    (
        "docs/decisions.md",
        "135,166",
        "exactly), 135,166 lines,",
    ): "a raw download line count recorded as byte-level verification evidence "
    "against the pinned sha256, coincidentally equal to "
    "cmudict_pronunciation_entry_count after its recount",
}


class Hardcode:
    """One numeral in prose that duplicates a recorded claim."""

    __slots__ = ("column", "label", "line", "path", "raw", "value")

    def __init__(self, path: str, line: int, column: int, raw: str, value: float, label: str):
        self.path, self.line, self.column = path, line, column
        self.raw, self.value, self.label = raw, value, label

    def __repr__(self) -> str:
        return f"{self.path}:{self.line} {self.raw!r} duplicates claim:{self.label}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Hardcode) and repr(self) == repr(other)

    def __hash__(self) -> int:
        return hash(repr(self))


def is_exempt(path: str) -> bool:
    """Whether a repository-relative path is exempt, per :data:`LITERAL_EXEMPTIONS`."""
    return path in LITERAL_EXEMPTIONS


def strip_code(text: str) -> str:
    """Blank out fenced and inline code, preserving line and column offsets.

    A numeral inside a code block is a literal the code needs, not a claim in
    prose. Blanking rather than deleting keeps every reported line number
    truthful, which matters because the report is what someone edits from.
    """

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    text = re.sub(r"```.*?```", blank, text, flags=re.S)
    return re.sub(r"`[^`\n]*`", blank, text)


def _numeric_claims(ledger_claims: dict[str, Any]) -> dict[str, list[str]]:
    """Ledger values keyed by their exact decimal rendering."""
    index: dict[str, list[str]] = {}
    for label, entry in ledger_claims.items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        index.setdefault(f"{float(value):.10g}", []).append(label)
    return index


def find_hardcoded_values(
    text: str, ledger_claims: dict[str, Any], *, path: str = "<text>"
) -> list[Hardcode]:
    """Distinctive numerals in ``text`` that exactly equal a recorded claim.

    Args:
        text: Markdown source.
        ledger_claims: The ``claims`` mapping from ``data/claims.json``.
        path: Repository-relative path, used in the report.

    Returns:
        One :class:`Hardcode` per occurrence, in source order.
    """
    index = _numeric_claims(ledger_claims)
    scannable = strip_code(text)
    found: list[Hardcode] = []
    for match in DISTINCTIVE_NUMERAL_RE.finditer(scannable):
        raw = match.group(1)
        value = float(raw.replace(",", ""))
        labels = index.get(f"{value:.10g}")
        if not labels:
            continue
        line = scannable.count("\n", 0, match.start()) + 1
        line_start = scannable.rfind("\n", 0, match.start()) + 1
        line_end = scannable.find("\n", match.start())
        line_text = scannable[line_start : line_end if line_end != -1 else len(scannable)]
        if any(
            key_path == path and key_raw == raw and anchor in line_text
            for key_path, key_raw, anchor in COINCIDENTAL_MATCHES
        ):
            continue
        column = match.start() - (scannable.rfind("\n", 0, match.start()) + 1)
        found.append(Hardcode(path, line, column, raw, value, sorted(labels)[0]))
    return found


def hydrate(
    text: str,
    ledger_claims: dict[str, Any],
    *,
    strict: bool = True,
    reference_variables: Mapping[str, str] | None = None,
) -> str:
    """Replace every ``claim:<label>`` with the label's formatted value.

    Uses ``ailm.value_formatting.format_ledger_value``, deliberately: a number must
    read the same in a decision record and in the paper, and two formatters would
    eventually disagree about whether 42820 carries a thousands separator.

    Args:
        text: Markdown source carrying ``claim:`` tags.
        ledger_claims: The ``claims`` mapping from ``data/claims.json``.
        reference_variables: Canonical source metadata and stable local protocol destinations.
            Generic manuscript token examples are preserved. Reference tokens always fail closed.
        strict: Raise on a tag with no ledger entry. Left true everywhere real;
            an unresolved tag rendered into a document reads as a fabricated
            result exactly the way an unresolved ``{{TOKEN}}`` does in the PDF.

    Returns:
        The text with every resolvable tag substituted.

    Raises:
        KeyError: If ``strict`` and a tag names no recorded claim.
    """

    def substitute(match: re.Match[str]) -> str:
        label = match.group(1)
        entry = ledger_claims.get(label)
        if entry is None:
            if strict:
                raise KeyError(f"claim:{label} has no ledger entry")
            return match.group(0)
        value = entry.get("value") if isinstance(entry, dict) else entry
        return format_ledger_value(value)

    def reference(match: re.Match[str]) -> str:
        name = match.group(1)
        if reference_variables is None or name not in reference_variables:
            raise KeyError(f"reference token {name} has no canonical producer")
        value = reference_variables[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"reference token {name} has an empty or malformed value")
        return value

    resolved = CLAIM_TAG_RE.sub(substitute, text)
    return re.sub(
        r"\{\{((?:CEREBRUM|COMPOSITIONAL_CASE|PROTOCOL)_[^{}\n]*)\}\}", reference, resolved
    )


def find_absorbed_tags(text: str, ledger_claims: dict[str, Any]) -> list[str]:
    """Tags that swallowed the word after them, so they resolve to nothing.

    The label pattern accepts hyphens, so ``claim:<label>-word``
    parses as one label rather than a tag followed by ``-word``. Writing a
    reference immediately before a hyphenated word therefore produces a tag that
    names no claim — which the strict hydrator refuses, but only at render time,
    and only for the tree it renders.

    This reports the class directly: any tag that does not resolve but whose
    prefix up to a hyphen does. Three of these were introduced in a single
    mechanical conversion pass, all of the form ``2,000-word``.

    Args:
        text: Markdown source.
        ledger_claims: The ``claims`` mapping from ``data/claims.json``.

    Returns:
        The offending full labels.
    """
    absorbed: list[str] = []
    for match in CLAIM_TAG_RE.finditer(text):
        label = match.group(1)
        if label in ledger_claims or "-" not in label:
            continue
        if label.split("-", 1)[0] in ledger_claims:
            absorbed.append(label)
    return absorbed


def scan_repository(root: Path, ledger_claims: dict[str, Any]) -> dict[str, list[Hardcode]]:
    """Every non-exempt markdown file under ``docs/`` and the top-level docs.

    ``docs/manuscript/`` is deliberately absent: it is scanned by the token
    machinery, which is stricter — a manuscript numeral does not need to match a
    claim to be wrong there, it needs only to be a numeral.
    """
    targets: list[Path] = sorted((root / "docs").rglob("*.md"))
    targets += [root / "README.md", root / "AGENTS.md", root / "CHANGELOG.md"]

    report: dict[str, list[Hardcode]] = {}
    for path in targets:
        if not path.exists():
            continue
        relative = path.relative_to(root).as_posix()
        if is_exempt(relative):
            continue
        hits = find_hardcoded_values(path.read_text(encoding="utf-8"), ledger_claims, path=relative)
        if hits:
            report[relative] = hits
    return report
