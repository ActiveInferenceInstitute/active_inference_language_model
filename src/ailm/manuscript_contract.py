"""Fail-closed contract for AILM manuscript variable injection.

The template renderer deliberately leaves unknown ``{{TOKEN}}`` markers in its
output and logs a warning.  That behavior is useful for generic projects, but it
is not safe for AILM: a surviving marker occupies the grammatical position of a
scientific result.  This module supplies the project-side contract around that
generic writer.

The checks are independent of ``infrastructure`` so they can run from the AILM
sidecar and in an installed package.  The token grammar is intentionally
byte-for-byte equivalent to the renderer's public grammar: an uppercase ASCII
letter followed by uppercase letters, digits, or underscores.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, NamedTuple

__all__ = [
    "ARTICLE_TYPOGRAPHY_INPUT",
    "ARTICLE_TYPOGRAPHY_NAME",
    "article_typography_bytes",
    "article_typography_required",
    "EXCLUDED_MANUSCRIPT_DOCS",
    "MANUSCRIPT_TOKEN_RE",
    "TOKEN_NAME_RE",
    "ManuscriptContractError",
    "HardcodedManuscriptVariable",
    "find_hardcoded_manuscript_variables",
    "manuscript_source_files",
    "resolve_manuscript_text",
    "validate_manuscript_sources",
    "validate_no_hardcoded_manuscript_variables",
    "validate_resolved_manuscript_tree",
]


TOKEN_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
"""The token-name grammar shared with the template renderer."""

MANUSCRIPT_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
"""A complete manuscript substitution marker."""

_ANY_DOUBLE_BRACE_RE = re.compile(r"\{\{([^{}\n]*)\}\}")
_PROSE_NUMBER = (
    r"[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)
_COUNT_LITERAL = (
    r"(?:zero|one|single|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|\d+)"
)
_ORDINAL_LITERAL = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|(?:[1-9]\d*)(?:st|nd|rd|th))"
)
_PLURAL_COUNT_LITERAL = (
    r"(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion|"
    r"[2-9]\d*)"
)

# These files explain the authoring mechanism and contain literal examples.  The
# generic renderer excludes the same names.  Keeping the set here explicit lets
# the isomorphism test compare the two boundaries without importing template
# infrastructure into ``src/``.
# MANUSCRIPT_STATUS.md joins the excluded set (2026-08-31): every manuscript
# directory now carries the status ledger per repo convention; it is
# documentation, never a rendered section.
EXCLUDED_MANUSCRIPT_DOCS = frozenset(
    {"AGENTS.md", "MANUSCRIPT_STATUS.md", "README.md", "SYNTAX.md"}
)


class ManuscriptContractError(RuntimeError):
    """Raised when authored text, variables, or resolved output violate the contract."""


class HardcodedManuscriptVariable(NamedTuple):
    """One authored value that bypasses the manuscript variable map."""

    path: Path
    line_number: int
    category: str
    literal: str
    line: str


def manuscript_source_files(project_root: Path) -> tuple[Path, ...]:
    """Return every Markdown input the renderer is allowed to consume.

    A missing or empty manuscript is an error rather than a successful scan of
    zero files.  Numbered sections are required independently of auxiliary
    Markdown such as ``preamble.md``.
    """
    manuscript_dir = Path(project_root) / "docs" / "manuscript"
    if not manuscript_dir.is_dir():
        raise ManuscriptContractError(f"manuscript directory not found: {manuscript_dir}")

    files = tuple(
        path
        for path in sorted(manuscript_dir.glob("*.md"))
        if path.name not in EXCLUDED_MANUSCRIPT_DOCS
    )
    numbered = tuple(path for path in files if path.name[:1].isdigit())
    if not numbered:
        raise ManuscriptContractError(
            f"no numbered manuscript sections found under {manuscript_dir}; "
            "refusing a vacuous variable check"
        )
    return files


def _malformed_markers(text: str) -> list[str]:
    """Return double-brace expressions the renderer would silently ignore."""
    malformed = [
        match.group(0)
        for match in _ANY_DOUBLE_BRACE_RE.finditer(text)
        if TOKEN_NAME_RE.fullmatch(match.group(1)) is None
    ]

    # Also catch unbalanced brace openers/closers.  Remove every balanced
    # expression first so the check is about residue, not the valid markers.
    residue = _ANY_DOUBLE_BRACE_RE.sub("", text)
    if "{{" in residue:
        malformed.append("unclosed '{{'")
    if re.search(r"\{\{\{[A-Z][A-Z0-9_]*\}\}\}?|\{\{[A-Z][A-Z0-9_]*\}\}\}", text):
        malformed.append("brace run adjacent to a marker")
    depth = 0
    for character in residue:
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                malformed.append("unmatched closing brace")
                break
    if depth > 0:
        malformed.append("unmatched opening brace")
    return malformed


def _validate_variable_map(variables: Mapping[str, str]) -> None:
    invalid_names = sorted(name for name in variables if TOKEN_NAME_RE.fullmatch(name) is None)
    if invalid_names:
        raise ManuscriptContractError(f"invalid manuscript variable names: {invalid_names}")

    non_strings = sorted(name for name, value in variables.items() if not isinstance(value, str))
    if non_strings:
        raise ManuscriptContractError(f"manuscript variable values must be strings: {non_strings}")

    recursive = sorted(
        name
        for name, value in variables.items()
        if isinstance(value, str) and ("{{" in value or "}}" in value)
    )
    if recursive:
        raise ManuscriptContractError(
            f"manuscript variable values may not contain token braces: {recursive}"
        )


def _validate_display_equations(text: str, path: Path) -> None:
    """Require the portable, automatically numbered form for every display."""

    # Code examples are not manuscript equations. Match the closing fence to
    # its opener so a shorter fence inside an example cannot expose its body.
    def hide_code(match: re.Match[str]) -> str:
        if re.fullmatch(r"\s*\{=[^}]+\}\s*", match["info"]):
            if "$$" in match[0]:
                raise ManuscriptContractError(f"display equation trapped in raw block: {path}")
            return match[0]
        return ""

    prose = re.sub(
        r"(?m)^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)\n"
        r".*?^ {0,3}(?P=fence)[ \t]*$",
        hide_code,
        text,
        flags=re.DOTALL,
    )
    if re.search(
        r"\\\[|\\begin\{(?:equation|align|alignat|gather|multline|displaymath)\*?\}", prose
    ):
        raise ManuscriptContractError(f"raw display equation must use labeled Markdown: {path}")
    for environment in re.finditer(
        r"\\begin\{(?P<kind>definition|theorem|proposition)\}"
        r"(?P<body>.*?)\\end\{(?P=kind)\}",
        prose,
        re.DOTALL,
    ):
        if "$$" in environment["body"]:
            raise ManuscriptContractError(f"display equation trapped in raw theorem: {path}")
    displays = list(re.finditer(r"\$\$(.*?)\$\$", prose, re.DOTALL))
    if prose.count("$$") != 2 * len(displays):
        raise ManuscriptContractError(f"unclosed display equation: {path}")
    labels: list[str] = []
    for display in displays:
        label = re.match(r"[ \t]*\{#eq:([A-Za-z0-9_]+)\}", prose[display.end() :])
        if label is None or not display[1].strip():
            raise ManuscriptContractError(f"empty or unlabeled display equation: {path}")
        if re.search(r"\\(?:tag|notag|nonumber)\b", display[1]):
            raise ManuscriptContractError(f"manual display equation numbering: {path}")
        labels.append(label[1])
    if len(labels) != len(set(labels)):
        raise ManuscriptContractError(f"duplicate display equation label: {path}")


def validate_manuscript_sources(project_root: Path, variables: Mapping[str, str]) -> frozenset[str]:
    """Validate authored markers and return the exact set of used token names.

    This is the pre-injection half of the contract.  It runs before the generic
    writer can turn an unknown token into a warning and success exit code.
    """
    _validate_variable_map(variables)
    used: set[str] = set()
    malformed: dict[str, list[str]] = {}

    root = Path(project_root)
    for path in manuscript_source_files(root):
        text = path.read_text(encoding="utf-8")
        if path.name[:1].isdigit():
            _validate_display_equations(text, path)
        bad = _malformed_markers(text)
        if bad:
            malformed[path.relative_to(root).as_posix()] = bad
        used.update(MANUSCRIPT_TOKEN_RE.findall(text))

    if malformed:
        raise ManuscriptContractError(
            f"malformed manuscript variable markers (the renderer would ignore them): {malformed}"
        )
    if not used:
        raise ManuscriptContractError(
            "the numbered manuscript uses no variables; refusing a vacuous injection check"
        )

    missing = sorted(used - set(variables))
    if missing:
        raise ManuscriptContractError(f"manuscript variables have no producer: {missing}")

    empty = sorted(name for name in used if not variables[name].strip())
    if empty:
        raise ManuscriptContractError(f"manuscript variables resolve to empty text: {empty}")
    return frozenset(used)


def _variable_value_patterns(
    variables: Mapping[str, str],
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Build conservative literal checks from the live injection map.

    The generic quantitative grammar can detect a number with a unit, but not a
    bare coefficient or error estimate in ordinary prose. The variable map is
    the missing evidence: if a source sentence types the exact value that a live
    numeric token owns, in a scientific-result context, it bypasses injection.
    Count phrases are handled by the noun-specific grammars below. Values are
    escaped as data and matched longest-first.
    """
    numeric_values: set[str] = set()
    for value in variables.values():
        stripped = value.strip()
        if re.fullmatch(_PROSE_NUMBER, stripped) is not None:
            numeric_values.add(stripped)
            if "," in stripped:
                numeric_values.add(stripped.replace(",", ""))

    patterns: list[tuple[str, re.Pattern[str]]] = []
    scientific_context = (
        r"(?:result|measurement|estimate|coefficient|tolerance|error|probability|"
        r"accuracy|agreement|perplexity|gain|margin|effect|rate|share|fraction|mass|"
        r"divergence|drift|alpha|AUC|ECE|total[- ]variation)"
    )
    if numeric_values:
        alternatives = "|".join(
            re.escape(value)
            for value in sorted(numeric_values, key=lambda item: (-len(item), item))
        )
        patterns.append(
            (
                "mapped-numeric-value",
                re.compile(
                    rf"\b{scientific_context}\b"
                    r"(?:\s+(?:estimate|value|term|coefficient))?\s*"
                    r"(?:is|was|equals?|=|:|at|of|reaches?|measures?)\s*"
                    rf"(?<![A-Za-z0-9_.-])(?:{alternatives})"
                    r"(?![A-Za-z0-9_]|[.]\d)",
                    re.I,
                ),
            )
        )
    return tuple(patterns)


def find_hardcoded_manuscript_variables(
    project_root: Path, variables: Mapping[str, str]
) -> tuple[HardcodedManuscriptVariable, ...]:
    """Find high-signal results and known mutable identities typed in source.

    This is intentionally context-aware rather than a ban on digits.  Equations,
    citations, Stage/Level identifiers, semantic versions in source metadata,
    and figure widths are structural forms handled by the shared quantitative
    scanner.  What remains includes percentages, sample/statistic assignments,
    scientific units and rates.  A second pass catches mutable product identity
    and code-derived plural counts even though they are not measurements.
    """
    from ailm.claims import scan_bare_quantitative_claims

    _validate_variable_map(variables)

    root = Path(project_root)
    files = tuple(path for path in manuscript_source_files(root) if path.name[:1].isdigit())
    found: list[HardcodedManuscriptVariable] = [
        HardcodedManuscriptVariable(
            finding.path,
            finding.line_number,
            finding.category,
            finding.literal,
            finding.line,
        )
        for finding in scan_bare_quantitative_claims(*files)
    ]

    # These values are mutable configuration identities rather than quantities,
    # so the numeric grammar cannot see them.  Each regex names the syntactic
    # context to avoid treating an unrelated 1.4 or year as the same variable.
    owned_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "additive-smoothing",
            re.compile(
                rf"\badd-(?:{_PROSE_NUMBER}|{_COUNT_LITERAL})\s+"
                r"(?:Laplace|smoothing|count\s+emission)\b",
                re.I,
            ),
        ),
        (
            "external-model-display",
            re.compile(
                r"\b(?:Gemma|Qwen|Llama|Mistral|Phi|DeepSeek|GPT|Claude|"
                r"Falcon|OLMo|Command|Granite)(?:\s+[A-Za-z0-9.-]+){0,3}"
                r"\s+\d+(?:\.\d+)?\s*[BM]\b",
                re.IGNORECASE,
            ),
        ),
        (
            "external-model-tag",
            re.compile(r"`[a-z][a-z0-9._-]+:[a-z0-9._-]+`"),
        ),
        (
            "pymdp-version",
            re.compile(
                r"\b(?:inferactively-pymdp|pymdp)(?:\s+version)?\s*`?v?"
                r"\d+\.\d+\.\d+\b",
                re.IGNORECASE,
            ),
        ),
        (
            "cerebrum-version",
            re.compile(r"\bCEREBRUM\s+v\d+(?:\.\d+)+\b", re.IGNORECASE),
        ),
        (
            "literature-cutoff",
            re.compile(
                r"\bthrough\s+(?:January|February|March|April|May|June|July|"
                r"August|September|October|November|December)\s+(?:19|20)\d{2}\b",
                re.IGNORECASE,
            ),
        ),
        (
            "inline-math-statistic",
            re.compile(
                rf"\$\s*(?:p|q|n)\s*(?:<=|>=|=|<|>|≤|≥)\s*{_PROSE_NUMBER}"
                r"(?:e[+-]?\d+)?\s*\$",
                re.IGNORECASE,
            ),
        ),
        (
            "inline-math-result",
            re.compile(
                rf"(?:\$\s*{_PROSE_NUMBER}\s*\$|\\\(\s*{_PROSE_NUMBER}\s*\\\))"
                r"\s*(?:\\?%|nats?|bits?|perplexity\b|positions?\b|tokens?\b|"
                r"sentences?\b|documents?\b)",
                re.IGNORECASE,
            ),
        ),
        (
            "inline-math-percentage",
            re.compile(
                rf"(?:\$\s*{_PROSE_NUMBER}\s*\\?%\s*\$|"
                rf"\\\(\s*{_PROSE_NUMBER}\s*\\?%\s*\\\))",
                re.IGNORECASE,
            ),
        ),
        (
            "inline-math-p-value",
            re.compile(
                rf"\bp(?:-value)?\s+(?:was|is)\s+"
                rf"(?:\$\s*{_PROSE_NUMBER}(?:e[+-]?\d+)?\s*\$|"
                rf"\\\(\s*{_PROSE_NUMBER}(?:e[+-]?\d+)?\s*\\\))",
                re.IGNORECASE,
            ),
        ),
        (
            "inline-code-result",
            re.compile(
                rf"`\s*\$?\s*{_PROSE_NUMBER}\s*(?:%|percent(?:age)?(?:\s+points?)?)"
                r"\s*\$?\s*`",
                re.IGNORECASE,
            ),
        ),
        (
            "subsample-stride",
            re.compile(r"\bstride\s*\[\s*::\s*\d+\s*\]", re.IGNORECASE),
        ),
        (
            "inline-learning-rate-assignment",
            re.compile(
                rf"\b(?:lr|learning[_ -]?rate)\s*(?:=|:=|equals?|is)\s*"
                rf"(?:{_COUNT_LITERAL})\b",
                re.I,
            ),
        ),
        (
            "scientific-notation-result",
            re.compile(
                r"\b(?:result|measurement|estimate|effect|error|tolerance|drift|"
                r"divergence|probability|accuracy|agreement)\b[^.\n]{0,48}"
                r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)[eE][+-]?\d+\b",
                re.I,
            ),
        ),
        (
            "digest-prefix-width",
            re.compile(
                rf"\b(?:sha-?256|checksum|digest)[^.\n]{{0,48}}\b"
                rf"(?:first|leading|prefix(?:\s+of)?)\s+{_COUNT_LITERAL}"
                r"(?:\s+(?:characters?|digits?|bytes?))?\b",
                re.I,
            ),
        ),
    )
    count_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "built-level-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:[A-Za-z-]+\s+){{0,4}}"
                r"(?:built\s+levels?|levels?\s+(?:are\s+|were\s+)?built)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "representation-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:[A-Za-z-]+\s+){{0,3}}representations?\b",
                re.I,
            ),
        ),
        (
            "channel-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}(?:\s+of\s+(?:the\s+)?{_COUNT_LITERAL})?\s+"
                r"(?:(?:cross|inter)[- ]level\s+|measured\s+|comprehension\s+)?channels?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "probe-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:(?:served|external|reference)\s+)?probes?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "external-model-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+external\s+models?\b", re.IGNORECASE),
        ),
        (
            "contrast-type-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+(?:paired\s+)?contrast\s+types?\b", re.I),
        ),
        (
            "narrative-phase-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:(?:narrative|document-position)\s+)?phases?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "miswiring-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}(?:\s+of\s+(?:the\s+)?{_COUNT_LITERAL})?\s+"
                r"(?:(?:finite-perplexity|surviving|word-level|empirically)\s+){0,4}"
                r"mis-?wirings?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "nat-threshold-count",
            re.compile(
                rf"\b(?:gain(?:\s+of|\s+(?:is|was|remains))?[^.\n]{{0,28}}"
                rf"\b{_COUNT_LITERAL}\s+nats?|{_COUNT_LITERAL}[- ]nat\s+"
                r"(?:gate|target|threshold|wording))\b",
                re.I,
            ),
        ),
        (
            "nat-comparison-count",
            re.compile(
                rf"\b(?:(?:under|below|less\s+than)\s+(?:a\s+)?"
                rf"{_COUNT_LITERAL}\s+nats?|(?:gain|nats?)[^.\n]{{0,72}}\b"
                rf"(?:again\s+)?(?:under|below|less\s+than)\s+(?:a\s+)?"
                rf"{_COUNT_LITERAL})\b",
                re.I,
            ),
        ),
        (
            "prefix-width-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}[- ]character"
                r"(?:\s+[A-Za-z-]+){0,3}\s+prefix(?:es|\s+famil(?:y|ies))?\b",
                re.I,
            ),
        ),
        (
            "prefix-minimum-length-count",
            re.compile(
                rf"\b(?:minimum\s+)?length\s+(?:of\s+)?(?:at\s+least\s+)?"
                rf"{_COUNT_LITERAL}(?:\s+(?:characters?|or\s+more))\b",
                re.I,
            ),
        ),
        (
            "selection-ordinal-count",
            re.compile(
                rf"\bevery\s+{_ORDINAL_LITERAL}\s+"
                r"(?:[A-Za-z-]+\s+){0,4}(?:family|index|pair|item|word)s?\b",
                re.I,
            ),
        ),
        (
            "phone-bucket-boundary",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+phones?\s+or\s+(?:fewer|less|more)\b",
                re.I,
            ),
        ),
        (
            "phone-bucket-range",
            re.compile(
                rf"\b(?:short|mid(?:-length)?|long)\s+words?\s*\([^)]*\b"
                rf"{_COUNT_LITERAL}(?:\s+phones?)?\s+(?:or\s+(?:fewer|less|more)|"
                rf"(?:to|through|[-–—])\s+{_COUNT_LITERAL}(?:\s+phones?)?)[^)]*\)",
                re.I,
            ),
        ),
        (
            "degenerate-distinct-count",
            re.compile(
                r"\b(?:collaps(?:e|es|ed|ing)|emit(?:s|ted|ting)?|"
                r"repeat(?:s|ed|ing)?)[^.\n]{0,64}\b(?:a\s+)?(?:one|single)\s+"
                r"(?:(?:repeated|identical|same)\s+)?words?\b",
                re.I,
            ),
        ),
        (
            "unrejectable-member-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:of\s+)?(?:the\s+)?"
                r"(?:(?:declared|family(?:'s)?)\s+){0,2}members?\s+"
                r"(?:could\s+not|cannot|can\s+never|(?:is|are)\s+unable\s+to)\s+"
                r"(?:have\s+)?reject(?:ed|ing)?\b",
                re.I,
            ),
        ),
        (
            "family-member-partitive-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+of\s+(?:its|the|this)\s+"
                rf"(?:{_COUNT_LITERAL}\s+)?members?\b",
                re.I,
            ),
        ),
        (
            "confirmatory-member-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+members\s+of\s+the\s+family\b", re.I),
        ),
        (
            "family-members-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+family\s+members\b", re.I),
        ),
        (
            "ordinal-list-anaphora",
            re.compile(
                rf"\b(?:none|all|each)\s+of\s+(?:the\s+)?(?:first|last)\s+"
                rf"{_COUNT_LITERAL}[^.]{{0,160}}\b(?:the\s+)?{_ORDINAL_LITERAL}\b",
                re.I,
            ),
        ),
        (
            "supplement-structure-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:supplements?|subsections?)\s+"
                r"(?:follow|comprise|cover|report|are\s+included)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "declared-task-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+(?:declared\s+)?tasks?\b", re.I),
        ),
        (
            "composition-seam-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:(?:exact|composition|measured)\s+)*seams?\b",
                re.I,
            ),
        ),
        (
            "further-seam-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+further\s+(?:[a-z-]+\s+){{0,3}}seams?\b", re.I),
        ),
        (
            "code-owned-structure-count",
            re.compile(
                rf"\b{_PLURAL_COUNT_LITERAL}"
                rf"(?:\s+of\s+(?:the\s+)?{_COUNT_LITERAL})?\s+"
                r"(?:[A-Za-z-]+\s+){0,3}(?:rows?|cells?|arms?|statistics?|controls?|"
                r"matrices?|comparators?|effects?|thresholds?|examples?|roles?|mappings?|"
                r"paths?|models?|letters?|strata|guards?|facets?)\b",
                re.I,
            ),
        ),
        (
            "measured-zero-count",
            re.compile(
                r"\bzero\s+(?:(?:decision|document)\s+)?"
                r"(?:disagreements?|overlaps?|mutual[- ]information)\b",
                re.I,
            ),
        ),
        (
            "measured-separately-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+measured\s+separately\b", re.I),
        ),
        (
            "storage-width-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+bytes?\s+per\b", re.I),
        ),
        (
            "spelled-percentage",
            re.compile(rf"\b{_COUNT_LITERAL}(?:\s+hundred)?\s+percent\b", re.I),
        ),
        (
            "spelled-measured-range",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:to|through)\s+{_COUNT_LITERAL}\s+"
                r"(?:letters?|tokens?|positions?|sentences?|documents?|levels?)\b",
                re.I,
            ),
        ),
        (
            "spelled-duration",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+years?\s+"
                r"(?:old|earlier|later|before|after)\b",
                re.I,
            ),
        ),
        (
            "spelled-approximate-fraction",
            re.compile(
                r"\b(?:about|approximately|near(?:ly)?|roughly)\s+(?:a|one)\s+"
                r"(?:half|third|quarter|fifth|sixth|seventh|eighth|ninth|tenth|"
                r"twentieth|twenty-fifth|hundredth)\b",
                re.I,
            ),
        ),
        (
            "learning-rate-literal",
            re.compile(rf"\blearning\s+rate[^.\n]{{0,32}}\b{_COUNT_LITERAL}\b", re.I),
        ),
        (
            "historical-year-bound",
            re.compile(r"\bnone\s+(?:is|are|was|were)\s+later\s+than\s+(?:19|20)\d{2}\b", re.I),
        ),
        (
            "corrupted-control-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:[A-Za-z-]+\s+){{0,3}}"
                r"(?:corrupted|broken)\s+(?:variants?|emissions?|controls?)\b",
                re.I,
            ),
        ),
        (
            "claim-strength-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:(?:claim[- ]strength)\s+labels?|"
                r"(?:claim\s+)?strengths?)\b",
                re.I,
            ),
        ),
        (
            "simultaneous-level-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+levels?\s+at\s+once\b", re.I),
        ),
        (
            "subsample-reanalysis-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+subsample\s+re[- ]?analy(?:ses|sis|zes)\b",
                re.I,
            ),
        ),
        (
            "prior-corpus-system-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+(?:documented\s+)?earlier\s+"
                r"corpus-fitted\s+systems?\b",
                re.I,
            ),
        ),
        (
            "registered-stride-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+(?:registered\s+)?strides?\b", re.I),
        ),
        (
            "comparison-count",
            re.compile(rf"\b{_PLURAL_COUNT_LITERAL}\s+(?:primary\s+)?comparisons?\b", re.I),
        ),
        (
            "literature-result-count",
            re.compile(rf"\b{_COUNT_LITERAL}\s+(?:works?|papers?|systems?)\s+above\b", re.I),
        ),
        (
            "audit-inventory-count",
            re.compile(rf"\ball\s+{_COUNT_LITERAL}\s+failed\s+(?:their\s+)?audits?\b", re.I),
        ),
        (
            "cascade-anaphora-count",
            re.compile(
                rf"\b(?:the\s+)?latter\s+{_COUNT_LITERAL}[^.\n]{{0,48}}"
                r"not\s+executed\s+as\s+a\s+connected\s+cascade\b",
                re.I,
            ),
        ),
        (
            "level-partitive-count",
            re.compile(rf"\bof\s+the\s+{_COUNT_LITERAL},\s+it\s+is\s+the\s+one\b", re.I),
        ),
        (
            "shared-receiver-count",
            re.compile(
                rf"\b{_COUNT_LITERAL}\s+of\s+them\s+enter\s+the\s+same\s+"
                r"(?:[A-Za-z-]+\s+){0,2}receiver\b",
                re.I,
            ),
        ),
        (
            "ordinal-channel-count",
            re.compile(
                r"\b(?:a|the)\s+(?:first|second|third|fourth|fifth|sixth|seventh|"
                r"eighth|ninth|tenth)\s+channel\s+helps\b",
                re.I,
            ),
        ),
        (
            "remaining-channel-count",
            re.compile(
                rf"\bthe\s+remaining\s+{_COUNT_LITERAL}\s+are\s+(?:the\s+)?"
                r"(?:less|more)\s+[A-Za-z-]+(?:\s+ones)?\b",
                re.I,
            ),
        ),
        (
            "derived-value-anaphora-count",
            re.compile(
                rf"\b(?:largest\s+of\s+the\s+{_COUNT_LITERAL}\s+derived\s+values|"
                rf"bounds?\s+the\s+other\s+{_COUNT_LITERAL})\b",
                re.I,
            ),
        ),
    )

    mapped_patterns = _variable_value_patterns(variables)
    for path in files:
        text = MANUSCRIPT_TOKEN_RE.sub("", path.read_text(encoding="utf-8"))
        count_text = re.sub(
            r"\b(?:Stage|Level|L)\s*-?\d+(?:\.\d+)*[A-Za-z]?\b",
            lambda match: " " * len(match.group(0)),
            text,
            flags=re.IGNORECASE,
        )
        # Exact mapped-value checks concern prose, not formal notation or code.
        # Replace excluded spans with equal-width spaces so match offsets still
        # identify the correct source line in ``text``.
        mapped_text = count_text
        for _mask in (
            re.compile(r"`+[^`\n]*`+"),
            re.compile(r"(?<!\\)\$(?!\$)[^$\n]*(?<!\\)\$"),
            re.compile(r"\\\([^\n]*?\\\)"),
        ):
            mapped_text = _mask.sub(lambda match: " " * len(match.group(0)), mapped_text)
        for category, pattern in owned_patterns:
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line = text[line_start : line_end if line_end != -1 else len(text)].strip()
                found.append(
                    HardcodedManuscriptVariable(
                        path=path,
                        line_number=line_number,
                        category=category,
                        literal=match.group(0),
                        line=line,
                    )
                )
        for category, pattern in count_patterns:
            for match in pattern.finditer(count_text):
                line_number = count_text.count("\n", 0, match.start()) + 1
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line = text[line_start : line_end if line_end != -1 else len(text)].strip()
                found.append(
                    HardcodedManuscriptVariable(
                        path=path,
                        line_number=line_number,
                        category=category,
                        literal=match.group(0),
                        line=line,
                    )
                )
        for category, pattern in mapped_patterns:
            for match in pattern.finditer(mapped_text):
                line_number = mapped_text.count("\n", 0, match.start()) + 1
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                line = text[line_start : line_end if line_end != -1 else len(text)].strip()
                found.append(
                    HardcodedManuscriptVariable(
                        path=path,
                        line_number=line_number,
                        category=category,
                        literal=match.group(0),
                        line=line,
                    )
                )

    # Multiple grammars may legitimately overlap on the same source span.  One
    # edit location is one finding, with the more specific first category kept.
    unique: dict[tuple[Path, int, str], HardcodedManuscriptVariable] = {}
    for finding in found:
        unique.setdefault((finding.path, finding.line_number, finding.literal), finding)
    return tuple(
        sorted(unique.values(), key=lambda item: (str(item.path), item.line_number, item.literal))
    )


def validate_no_hardcoded_manuscript_variables(
    project_root: Path, variables: Mapping[str, str]
) -> None:
    """Fail when manuscript source bypasses a result/configuration token."""
    findings = find_hardcoded_manuscript_variables(project_root, variables)
    if findings:
        preview = "; ".join(
            f"{finding.path.name}:{finding.line_number} [{finding.category}] {finding.literal!r}"
            for finding in findings[:20]
        )
        suffix = f"; plus {len(findings) - 20} more" if len(findings) > 20 else ""
        raise ManuscriptContractError(
            "hard-coded manuscript variables bypass the injection map: " + preview + suffix
        )


def resolve_manuscript_text(text: str, variables: Mapping[str, str]) -> str:
    """Resolve one source string after its token coverage has been validated."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        try:
            return variables[name]
        except KeyError as exc:  # defensive: callers may use this function directly
            raise ManuscriptContractError(f"manuscript variable has no producer: {name}") from exc

    return MANUSCRIPT_TOKEN_RE.sub(replace, text)


ARTICLE_TYPOGRAPHY_NAME = "article_typography.sty"
"""The single generated LaTeX auxiliary admitted by the resolved-tree contract."""

ARTICLE_TYPOGRAPHY_INPUT = r"\input{../manuscript/article_typography.sty}"
"""Fixed input path relative to the renderer's output/pdf or output/slides directory."""


def article_typography_required(project_root: Path) -> bool:
    """Identify the exact authored include, rejecting alternative style paths."""
    preamble = Path(project_root) / "docs" / "manuscript" / "preamble.md"
    if not preamble.is_file():
        return False
    text = preamble.read_text(encoding="utf-8")
    if ARTICLE_TYPOGRAPHY_NAME not in text:
        if Path(ARTICLE_TYPOGRAPHY_NAME).stem + "." in text:
            raise ManuscriptContractError(
                "article typography requires its exact single input directive"
            )
        return False
    if text.count(ARTICLE_TYPOGRAPHY_NAME) != 1 or ARTICLE_TYPOGRAPHY_INPUT not in text:
        raise ManuscriptContractError(
            "article typography requires its exact single input directive"
        )
    return True


def article_typography_bytes(project_root: Path, variables: Mapping[str, str]) -> bytes:
    """Generate canonical article style from validated config and producer parity."""
    # Import lazily: the variable loader shares this module's token grammar.
    from ailm.manuscript_variables.literature import _load_rendering_variables

    expected = _load_rendering_variables(Path(project_root))
    differing = sorted(name for name, value in expected.items() if variables.get(name) != value)
    if differing:
        raise ManuscriptContractError(
            f"article typography variables differ from rendering configuration: {differing}"
        )
    text = (
        "% Generated article typography from docs/manuscript/config.yaml; do not edit.\n"
        "\\csname @ifclassloaded\\endcsname{beamer}{}{%\n"
        f"\\geometry{{top={expected['META_MARGIN_TOP_IN']}in,"
        f"bottom={expected['META_MARGIN_BOTTOM_IN']}in,"
        f"left={expected['META_MARGIN_LEFT_IN']}in,right={expected['META_MARGIN_RIGHT_IN']}in}}\n"
        f"\\renewcommand{{\\cftsecfont}}{{\\fontsize{{{expected['META_TOC_FONT_PT']}}}"
        f"{{{expected['META_TOC_BASELINE_PT']}}}\\selectfont\\bfseries}}\n"
        f"\\renewcommand{{\\cftsecpagefont}}{{\\fontsize{{{expected['META_TOC_FONT_PT']}}}"
        f"{{{expected['META_TOC_BASELINE_PT']}}}\\selectfont\\bfseries}}\n"
        "\\AtBeginDocument{\\renewcommand{\\normalsize}{"
        f"\\fontsize{{{expected['META_BODY_FONT_PT']}}}{{{expected['META_BODY_BASELINE_PT']}}}"
        "\\selectfont}\\normalsize}\n"
        "}\n"
    )
    return text.encode("utf-8")


def validate_resolved_manuscript_tree(
    project_root: Path,
    variables: Mapping[str, str],
    resolved_dir: Path | None = None,
) -> None:
    """Prove that the on-disk resolved tree is the exact source substitution.

    A token-free output alone is insufficient: it may be stale, omit a source
    section, or contain a manual edit.  This check independently substitutes
    each current source file and requires byte identity with the corresponding
    output, while also rejecting missing and extra Markdown files.
    """
    root = Path(project_root)
    validate_manuscript_sources(root, variables)
    target = root / "output" / "manuscript" if resolved_dir is None else Path(resolved_dir)
    if not target.is_dir():
        raise ManuscriptContractError(f"resolved manuscript directory not found: {target}")

    sources = manuscript_source_files(root)
    auxiliary_sources = tuple(
        path for path in (root / "docs" / "manuscript" / "config.yaml",) if path.is_file()
    ) + tuple(sorted((root / "docs" / "manuscript").glob("*.bib")))
    expected_names = {Path(path.name) for path in (*sources, *auxiliary_sources)}
    typography_required = article_typography_required(root)
    if typography_required:
        expected_names.add(Path(ARTICLE_TYPOGRAPHY_NAME))
    actual_names = {path.relative_to(target) for path in target.rglob("*")}
    missing_files = sorted(str(path) for path in expected_names - actual_names)
    extra_files = sorted(str(path) for path in actual_names - expected_names)
    if missing_files or extra_files:
        raise ManuscriptContractError(
            "resolved manuscript file set differs from source: "
            f"missing={missing_files}, extra={extra_files}"
        )

    drifted: list[str] = []
    unresolved: dict[str, list[str]] = {}
    malformed: dict[str, list[str]] = {}
    for source in sources:
        output = target / source.name
        actual = output.read_text(encoding="utf-8")
        expected = resolve_manuscript_text(source.read_text(encoding="utf-8"), variables)
        if actual != expected:
            drifted.append(source.name)
        remaining = sorted(set(MANUSCRIPT_TOKEN_RE.findall(actual)))
        if remaining:
            unresolved[source.name] = remaining
        bad = _malformed_markers(actual)
        if bad:
            malformed[source.name] = bad

    for source in auxiliary_sources:
        output = target / source.name
        if output.read_bytes() != source.read_bytes():
            drifted.append(source.name)

    if typography_required and (
        target / ARTICLE_TYPOGRAPHY_NAME
    ).read_bytes() != article_typography_bytes(root, variables):
        drifted.append(ARTICLE_TYPOGRAPHY_NAME)

    if drifted or unresolved or malformed:
        raise ManuscriptContractError(
            "resolved manuscript is not the exact, token-free source substitution: "
            f"drifted={drifted}, unresolved={unresolved}, malformed={malformed}"
        )
