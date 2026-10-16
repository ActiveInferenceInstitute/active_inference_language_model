"""Finite protocol companion mapping and complete local HTML target checks.

Context files are delivered source views, not another documentation website.
Only files directly referenced by the whole protocol page enter this package.
The existing provider remains the owner of rendered headings and their IDs.
"""

from __future__ import annotations

import html
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Source paths, not mutable measurements or a second source-identity store.
CONTEXT_SOURCES = frozenset(
    {
        "docs/design/README.md",
        "docs/design/connected_runtime.md",
        "docs/reference/data_statement.md",
        "src/ailm/benchmarks/README.md",
        "TODO.md",
        "docs/manuscript/98_supplement_external_benchmarks.md",
        "docs/provenance/sources.md",
        "docs/provenance/AGENTS.md",
        "docs/gate_design.md",
        "docs/testing_philosophy.md",
        "src/ailm/benchmarks/fitted/README.md",
        "docs/architecture.md",
        "src/ailm/corpus/AGENTS.md",
        "docs/README.md",
        "data/claims.json",
    }
)

# Old authored navigation selects numbered source sections. The actual ID is
# read from the provider's whole HTML, never computed with a slug algorithm.
PROTOCOL_SECTIONS = {
    "5-ownership-grant-boundaries-and-what-this-milestone-did-not-do": "5.",
    "2-protocol-p1--a-fitted-external-probe-arm": "2.",
    "3-protocol-p2--stipulated-aligned-spelling-to-pronunciation-design-b": "3.",
    "4-protocol-p3--the-l4l3-pword--case-role-bridge-account": "4.",
    "28-external-authorization-and-resource-requirements": "2.8",
}
DATA_SECTION = "Universal Dependencies English GUM — connected evaluation"

# Only the complete source-view page receives this rule. Its text blocks reuse
# the provider's pre colors instead of the provider's inline-code background.
SOURCE_VIEW_STYLE = (
    '<style id="ailm-source-view-style">\n'
    "#main-content pre.text > code { color: inherit; background-color: inherit; }\n"
    "</style>\n"
)

_REFERENCE_ID = re.compile(r"(?:sec|fig|tbl|eq):[A-Za-z0-9_.-]+")
_REFERENCE_MARKER = re.compile(r"\[\s*@\s*(?:sec|fig|tbl|eq)\b")
_EQUATION_DEFINITION_MARKER = re.compile(r"\{\s*#\s*eq\b")
_REFERENCE_PREFIX = re.compile(
    r"\b(?:sec\.|fig\.|tbl\.|eq\.|section|figure|table|equation)\s+$", re.I
)
_LITERAL_TAGS = frozenset({"pre", "code", "script", "style"})
_DISPLAY_BLOCKS = frozenset(
    {
        "html",
        "body",
        "address",
        "article",
        "aside",
        "blockquote",
        "caption",
        "col",
        "colgroup",
        "dd",
        "details",
        "dialog",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "header",
        "hgroup",
        "hr",
        "legend",
        "li",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "summary",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
        "br",
        "center",
        "dir",
        "frameset",
        "noframes",
    }
    | {f"h{level}" for level in range(1, 7)}
)


def _reference_kind(prefix: str) -> str:
    aliases = {"section": "sec", "figure": "fig", "table": "tbl", "equation": "eq"}
    value = prefix.strip().rstrip(".").lower()
    return aliases.get(value, value)


class HtmlInventory(HTMLParser):
    """Read IDs, heading text and every URI-bearing package attribute."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.targets: list[str] = []
        self.headings: list[tuple[str, str]] = []
        self._heading: tuple[str, str] | None = None
        self._text: list[str] = []
        self._literal: list[str] = []
        self.visible_text: list[str] = []
        self.reference_citations: list[str] = []
        self.numbered_ids: dict[str, str] = {}
        self.feed(text)
        self.close()
        if self._literal:
            raise ValueError("unclosed literal HTML region")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # HTMLParser normalizes names but preserves duplicate attributes.
        # Browsers keep the first; dict(attrs) would silently keep the last.
        values: dict[str, str | None] = {}
        for name, value in attrs:
            if name in values:
                raise ValueError(f"duplicate HTML attribute: {name}")
            values[name] = value
        if tag == "base" and "href" in values:
            raise ValueError("unsupported HTML base href")
        identity = values.get("id")
        if identity:
            if identity in self.ids:
                raise ValueError(f"ambiguous HTML id: {identity}")
            self.ids.add(identity)
            if values.get("data-number") is not None:
                self.numbered_ids[identity] = str(values["data-number"])
        for attribute in ("href", "src", "poster"):
            value = values.get(attribute)
            if value is not None:
                self.targets.append(value)
        if re.fullmatch(r"h[1-6]", tag) and identity:
            self._heading = tag, identity
            self._text = []
        if tag in _LITERAL_TAGS:
            self._literal.append(tag)
        if not self._literal and values.get("data-cites"):
            cites = str(values["data-cites"])
            if re.search(r"\b(?:sec|fig|tbl|eq):", cites):
                self.reference_citations.append(cites)

    def handle_data(self, data: str) -> None:
        if self._heading:
            self._text.append(data)
        if not self._literal:
            self.visible_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._heading and self._heading[0] == tag:
            self.headings.append((self._heading[1], " ".join("".join(self._text).split())))
            self._heading = None
        if tag in _LITERAL_TAGS:
            if not self._literal or self._literal[-1] != tag:
                raise ValueError(f"malformed literal HTML region: {tag}")
            self._literal.pop()


@dataclass(frozen=True)
class _ReferenceLink:
    target: str
    prefix: str
    number: str
    markup: str


class _ReferenceHtml(HtmlInventory):
    """Locate exact provider spans without rewriting surrounding HTML bytes."""

    def __init__(self, text: str) -> None:
        self.source = text
        # HTMLParser's positions count LF, not every Unicode splitlines separator.
        self.offsets = [0, *(match.end() for match in re.finditer("\n", text))]
        self.links: list[_ReferenceLink] = []
        self.references: list[tuple[int, int, str]] = []
        self._link: tuple[int, str, str] | None = None
        self._citation: tuple[int, str] | None = None
        self._content: list[str] = []
        self._display_context = ""
        super().__init__(text)
        if self._link or self._citation or self._literal:
            raise ValueError("unclosed reference or literal HTML region")

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.offsets[line - 1] + column

    def _end_offset(self, tag: str) -> int:
        start = self._offset()
        closing = re.match(r"</" + re.escape(tag) + r"\s*>", self.source[start:], re.I)
        if closing is None:
            raise ValueError(f"malformed reference closing tag: {tag}")
        return start + closing.end()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DISPLAY_BLOCKS:
            self._display_context = ""
        super().handle_starttag(tag, attrs)
        if self._literal:
            return
        if self._citation:
            raise ValueError("malformed reference citation contains nested HTML")
        values = dict(attrs)  # Duplicate attributes were refused by the parent.
        if tag == "a":
            if self._link:
                raise ValueError("nested reference link")
            target = values.get("href") or ""
            if _REFERENCE_ID.fullmatch(unquote(urlsplit(target).fragment)):
                preceding = self.source[max(0, self._offset() - 32) : self._offset()]
                prefix = _REFERENCE_PREFIX.search(preceding) or _REFERENCE_PREFIX.search(
                    self._display_context
                )
                if not (values.get("id") or "").startswith("toc-"):
                    self._link = self._offset(), target, prefix.group() if prefix else ""
                    self._content = []
        if values.get("data-cites") and re.search(
            r"\b(?:sec|fig|tbl|eq):", str(values["data-cites"])
        ):
            label = str(values["data-cites"])
            if tag != "span" or "citation" not in (values.get("class") or "").split():
                raise ValueError("malformed reference citation element")
            if not _REFERENCE_ID.fullmatch(label) or self._link:
                raise ValueError(f"malformed reference citation: {label}")
            self._citation = self._offset(), label
            self._content = []

    def handle_data(self, data: str) -> None:
        super().handle_data(data)
        if not (set(self._literal) & {"script", "style"}):
            self._display_context = (self._display_context + data)[-128:]
            if self._link or self._citation:
                self._content.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._citation:
            if tag != "span":
                raise ValueError("malformed reference citation closing tag")
            start, label = self._citation
            if "".join(self._content).strip() != "[@" + label + "]":
                raise ValueError(f"malformed reference citation text: {label}")
            self.references.append((start, self._end_offset(tag), label))
            self._citation = None
        if self._link and tag == "a":
            start, target, prefix = self._link
            number = " ".join("".join(self._content).split())
            numeric = bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", number))
            if not numeric and prefix:
                raise ValueError(f"reference link has no authoritative number: {target}")
            if numeric:
                self.links.append(
                    _ReferenceLink(
                        target, prefix, number, self.source[start : self._end_offset(tag)]
                    )
                )
            self._link = None
        super().handle_endtag(tag)
        if tag in _DISPLAY_BLOCKS:
            self._display_context = ""


def _reference_link_markup(reference: _ReferenceLink, href: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return match[1] + match[2] + html.escape(href, quote=True) + match[2]

    markup = re.sub(
        r'(\bhref\s*=\s*)(["\'])(.*?)\2', replace, reference.markup, count=1, flags=re.S
    )
    if HtmlInventory(markup).targets != [href]:
        raise ValueError("unsupported authoritative href attribute")
    return markup


@dataclass(frozen=True, slots=True)
class _PublicationSpan:
    start: int
    end: int
    section: str
    attributes: Mapping[str, str | None]


class _PublicationHtml(_ReferenceHtml):
    """Retain occurrence positions for transport from the combined edition."""

    def __init__(self, text: str) -> None:
        self.section = ""
        self.sections: set[str] = set()
        self.spans: list[_PublicationSpan] = []
        self.data: list[tuple[int, str, str]] = []
        self.anchors: list[tuple[str, str]] = []
        self.blocks: list[tuple[int, int]] = []
        self.block_tags: dict[int, str] = {}
        self.block_ids: dict[int, str] = {}
        self._blocks: list[tuple[str, int]] = []
        self._anchor: tuple[str, list[str]] | None = None
        self._spans: list[tuple[int, str, dict[str, str | None]]] = []
        super().__init__(text)
        if self._spans:
            raise ValueError("unclosed publication span")
        if self._anchor:
            raise ValueError("unclosed publication link")
        self.spans.sort(key=lambda span: span.start)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        super().handle_starttag(tag, attrs)
        if self._literal:
            return
        values = dict(attrs)
        if tag in _DISPLAY_BLOCKS - {"br", "hr", "col"}:
            self._blocks.append((tag, self._offset()))
            self.block_tags[self._offset()] = tag
            if values.get("id"):
                self.block_ids[self._offset()] = str(values["id"])
        if tag == "a":
            if self._anchor:
                raise ValueError("nested publication link")
            self._anchor = values.get("href") or "", []
        if values.get("data-cites") is not None:
            if tag != "span" or "citation" not in (values.get("class") or "").split():
                raise ValueError("malformed publication citation element")
            if any("citation" in (attrs.get("class") or "").split() for _, _, attrs in self._spans):
                raise ValueError("nested publication citation")
        if re.fullmatch(r"h[1-6]", tag) and values.get("id"):
            self.section = str(values["id"])
            self.sections.add(self.section)
        if tag == "span":
            self._spans.append((self._offset(), self.section, values))

    def handle_endtag(self, tag: str) -> None:
        if not self._literal and self._blocks and self._blocks[-1][0] == tag:
            _, start = self._blocks.pop()
            self.blocks.append((start, self._end_offset(tag)))
        if not self._literal and tag == "a" and self._anchor is not None:
            target, content = self._anchor
            self.anchors.append((target, "".join(content)))
            self._anchor = None
        if not self._literal and tag == "span":
            if not self._spans:
                raise ValueError("unmatched publication span")
            start, section, attrs = self._spans.pop()
            self.spans.append(_PublicationSpan(start, self._end_offset(tag), section, attrs))
        super().handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self._literal:
            if self._anchor is not None:
                self._anchor[1].append(data)
            start = self._offset()
            end = self.source.find("<", start)
            self.data.append((start, self.source[start : end if end >= 0 else None], self.section))
        super().handle_data(data)


def _citation_spans(parsed: _PublicationHtml) -> list[_PublicationSpan]:
    return [
        span
        for span in parsed.spans
        if "citation" in (span.attributes.get("class") or "").split()
        and not re.search(
            r"\b(?:sec|fig|tbl|eq|def|thm|lem|cor|prop):", span.attributes.get("data-cites") or ""
        )
    ]


def _citation_markup(parsed: _PublicationHtml, span: _PublicationSpan, prefix: str) -> str:
    """Relocate only actual bibliography href attributes in a citeproc span."""
    markup = parsed.source[span.start : span.end]
    keys = (span.attributes.get("data-cites") or "").split()
    inventory = _PublicationHtml(markup)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("missing or duplicate publication citation metadata")
    if sorted(inventory.targets) != sorted("#ref-" + key for key in keys):
        raise ValueError("publication citation lacks authoritative bibliography links")
    if any(not text.strip() for _, text in inventory.anchors):
        raise ValueError("publication citation has an empty bibliography link")
    if any(
        re.search(r"(?<![\w@])@" + re.escape(key) + r"(?![\w:.-])", "".join(inventory.visible_text))
        for key in keys
    ):
        raise ValueError("combined publication citation retains raw source notation")
    if any("ref-" + key not in parsed.ids for key in keys):
        raise ValueError("publication citation has missing bibliography anchor")
    return re.sub(
        r'(\bhref\s*=\s*)(["\'])(#ref-[^"\']+)\2',
        lambda match: match[1] + match[2] + prefix + match[3] + match[2],
        markup,
    )


def _citation_context(
    parsed: _PublicationHtml,
    start: int,
    end: int,
    occurrences: list[tuple[int, int]],
    authority: Mapping[str, _ReferenceLink],
) -> tuple[str, str]:
    """Require matching surrounding prose, excluding only transported notation."""
    blocks = [(a, b) for a, b in parsed.blocks if a <= start < end <= b]
    if not blocks:
        raise ValueError("publication citation has no enclosing prose block")
    left, right = min(blocks, key=lambda block: block[1] - block[0])
    substitutions = [(a, b, "") for a, b in occurrences]
    substitutions.extend(
        (span.start, span.end, "")
        for span in parsed.spans
        if "header-section-number" in (span.attributes.get("class") or "").split()
        or (span.attributes.get("id") or "").startswith("eq:")
        or set((span.attributes.get("class") or "").split()) == {"math", "display"}
    )
    for a, b, label in parsed.references:
        if label not in authority:
            raise ValueError(f"missing authoritative combined reference: {label}")
        reference = authority[label]
        substitutions.append((a, b, reference.prefix + reference.markup))

    def prose(a: int, b: int) -> str:
        cursor = a
        chunks: list[str] = []
        for first, last, replacement in sorted(substitutions, key=lambda item: (item[0], -item[1])):
            if a <= first < last <= b and first >= cursor:
                chunks.extend((parsed.source[cursor:first], replacement))
                cursor = last
        chunks.append(parsed.source[cursor:b])
        # Each side may end within an inline wrapper. HTMLParser's text reader
        # deliberately accepts such fragments; whole-document structure was read above.
        visible = "".join(HtmlInventory("".join(chunks)).visible_text)
        visible = re.sub(r"\{#eq:[A-Za-z0-9_.-]+\}", "", visible)
        return " ".join(visible.split())

    before = prose(left, start)
    if parsed.block_tags[left] in {"caption", "figcaption"}:
        parents = [
            parsed.block_ids[a]
            for a, b in blocks
            if a in parsed.block_ids and parsed.block_ids[a] in authority
        ]
        counter = re.match(r"^([A-Za-z]+)\s+(\d+(?:\.\d+)*):\s*", before)
        if counter and len(parents) == 1:
            reference = authority[parents[0]]
            if (
                counter[2] != reference.number
                or _reference_kind(counter[1]) != parents[0].split(":")[0]
            ):
                raise ValueError("publication citation caption counter differs")
            before = before[counter.end() :]
    return before, prose(end, right)


def _equation_wrap_identity(tex: str) -> str:
    """Treat ordinary Pandoc line wraps as one space, retaining every separator.

    This is not a TeX whitespace canonicalizer: tabs and repeated spaces stay
    exact, so command terminators and text spacing cannot disappear. Comments,
    verbatim or catcode-sensitive content, escaped newlines and paragraph
    breaks require byte identity rather than an inferred TeX equivalence.
    """
    if re.search(
        r"%|\^\^|\\(?:verb|obeylines|obeyspaces|catcode|endlinechar)"
        r"|(?<!\\)(?:\\\\)*\\\n|\n[ \t]*\n",
        tex,
    ):
        return tex
    return tex.replace("\n", " ")


def _publication_markup_replacements(
    parsed: _PublicationHtml,
    combined: _PublicationHtml,
    index: Path,
    page: Path,
    authority: Mapping[str, _ReferenceLink],
) -> list[tuple[int, int, str]]:
    """Bind complete heading-local occurrence sequences, never guess citation style.

    The individual provider discards bracket-citation presentation, retaining
    only ordered keys. The combined citeproc span is therefore the presentation
    authority for that exact occurrence, including mode, prefix and suffix.
    """
    prefix = posixpath.relpath(index, page.parent)
    canonical: dict[str, list[_PublicationSpan]] = {}
    for span in _citation_spans(combined):
        _citation_markup(combined, span, "")
        canonical.setdefault(span.section, []).append(span)
    citations = _citation_spans(parsed)
    found: dict[str, list[tuple[int, int, list[str], str]]] = {}
    for span in citations:
        keys = (span.attributes.get("data-cites") or "").split()
        markup = parsed.source[span.start : span.end]
        found.setdefault(span.section, []).append((span.start, span.end, keys, markup))
    bibliography = {identity[4:] for identity in combined.ids if identity.startswith("ref-")}
    for start, raw, section in parsed.data:
        if any(span.start <= start < span.end for span in citations):
            continue
        if any(
            re.search(r"(?<![\w@])@" + re.escape(key) + r"(?![\w:.-])", html.unescape(raw))
            for key in bibliography
        ):
            raise ValueError("unmarked raw publication citation")
        for match in re.finditer(r"\[([^\[\]]+)\]", raw):
            keys = [key.strip() for key in html.unescape(match[1]).split(";")]
            if not any(key in bibliography for key in keys):
                continue
            if not all(key in bibliography for key in keys):
                raise ValueError("unknown key in publication citation group")
            found.setdefault(section, []).append(
                (start + match.start(), start + match.end(), keys, match[0])
            )
    replacements: list[tuple[int, int, str]] = []
    canonical_occurrences = [
        (span.start, span.end) for spans in canonical.values() for span in spans
    ]
    actual_occurrences = [(a, b) for values in found.values() for a, b, _, _ in values]
    presentations: dict[tuple[str, tuple[str, ...], tuple[str, str]], str] = {}
    for section in parsed.sections | found.keys():
        expected = canonical.get(section, [])
        actual = sorted(found.get(section, []))
        if (expected or actual) and (not section or section not in combined.sections):
            raise ValueError("publication citations have no authoritative section")
        if len(expected) != len(actual):
            raise ValueError(f"publication citation occurrence count differs: {section}")
        for target, (start, end, keys, markup) in zip(expected, actual, strict=True):
            if keys != (target.attributes.get("data-cites") or "").split():
                raise ValueError(f"publication citation occurrence order differs: {section}")
            context = _citation_context(
                combined, target.start, target.end, canonical_occurrences, authority
            )
            if context != _citation_context(parsed, start, end, actual_occurrences, authority):
                raise ValueError(f"publication citation prose context differs: {section}")
            replacement = _citation_markup(combined, target, prefix)
            identity = section, tuple(keys), context
            if identity in presentations and presentations[identity] != replacement:
                raise ValueError(f"ambiguous publication citation presentation: {section}")
            presentations[identity] = replacement
            if markup == replacement:
                continue
            visible = "".join(HtmlInventory(markup).visible_text)
            raw_group = "[" + "; ".join(keys) + "]"
            if " ".join(visible.split()) not in {raw_group, "@" + " ".join(keys)}:
                raise ValueError(f"malformed or conflicting publication citation: {section}")
            replacements.append((start, end, replacement))
    definitions: list[tuple[int, int, str, bool]] = [
        (span.start, span.end, str(span.attributes["id"]), False)
        for span in parsed.spans
        if (span.attributes.get("id") or "").startswith("eq:")
    ]
    for start, raw, _section in parsed.data:
        for match in re.finditer(r"\{#eq[^}]*\}", raw):
            definitions.append((start + match.start(), start + match.end(), match[0][2:-1], True))
    expected_definitions = {
        str(span.attributes["id"])
        for span in combined.spans
        if (span.attributes.get("id") or "").startswith("eq:") and span.section in parsed.sections
    }
    if {label for _, _, label, _ in definitions} != expected_definitions:
        raise ValueError("publication equation definition inventory differs")
    if len(definitions) != len(expected_definitions):
        raise ValueError("duplicate publication equation definition")
    for position, end, label, raw_definition in definitions:
        if not _REFERENCE_ID.fullmatch(label) or label not in authority:
            raise ValueError(f"missing or malformed equation definition: {label}")
        targets = [span for span in combined.spans if span.attributes.get("id") == label]
        if len(targets) != 1:
            raise ValueError(f"equation definition has no unique target: {label}")
        target = targets[0]
        if (
            not raw_definition
            and parsed.source[position:end] == combined.source[target.start : target.end]
        ):
            if not any(
                span.start == position and span.section == target.section for span in parsed.spans
            ):
                raise ValueError(f"equation definition section differs: {label}")
            continue
        preceding = [
            span
            for span in parsed.spans
            if span.end <= position
            and set((span.attributes.get("class") or "").split()) == {"math", "display"}
            and not parsed.source[span.end : position].strip()
        ]
        if len(preceding) != 1:
            raise ValueError(f"equation definition has no unique display: {label}")
        equations = [
            span
            for span in combined.spans
            if target.start < span.start < span.end < target.end
            and set((span.attributes.get("class") or "").split()) == {"math", "display"}
        ]
        if len(equations) != 1 or target.section != preceding[0].section:
            raise ValueError(f"equation definition section or math differs: {label}")
        original = "".join(
            HtmlInventory(parsed.source[preceding[0].start : preceding[0].end]).visible_text
        )
        numbered = "".join(
            HtmlInventory(combined.source[equations[0].start : equations[0].end]).visible_text
        )
        number = authority[label].number
        unnumbered = re.sub(r"\\qquad\{\(" + re.escape(number) + r"\)\}(?=\\\]\s*$)", "", numbered)
        if unnumbered == numbered or _equation_wrap_identity(original) != _equation_wrap_identity(
            unnumbered
        ):
            raise ValueError(f"equation definition math or numbering differs: {label}")
        reference = authority[label]
        replacement = (
            '<span id="'
            + html.escape(label, quote=True)
            + '">'
            + reference.prefix
            + _reference_link_markup(reference, prefix + "#" + label)
            + "</span>"
        )
        if raw_definition:
            replacements.append((position, end, replacement))
        elif parsed.source[position:end] != replacement:
            raise ValueError(f"conflicting publication equation definition: {label}")
    return replacements


def publication_reference_repairs(directory: Path) -> dict[Path, str]:
    """Derive individual-page repairs from actual combined links and anchors.

    This returns a complete validated write plan; it never changes files.
    Code examples retain their literal source notation. Missing, conflicting
    or malformed provider references fail before any publication write.
    """
    directory = directory.absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"publication HTML must be a real directory: {directory}")
    index = directory / "index.html"
    if index.is_symlink() or not index.is_file():
        raise ValueError("publication HTML has no ordinary combined index.html")
    combined = _PublicationHtml(index.read_bytes().decode("utf-8"))
    if combined.reference_citations or _REFERENCE_MARKER.search("".join(combined.visible_text)):
        raise ValueError("combined index has unresolved publication references")
    authority: dict[str, _ReferenceLink] = {}
    for link in combined.links:
        target = urlsplit(link.target)
        if target.path or target.scheme or target.netloc or target.query:
            continue
        label = unquote(target.fragment)
        if label not in combined.ids or _reference_kind(link.prefix) != label.split(":")[0]:
            raise ValueError(f"combined reference anchor inconsistency: {label}")
        if label in combined.numbered_ids and combined.numbered_ids[label] != link.number:
            raise ValueError(f"combined reference numbering inconsistency: {label}")
        if label in authority and authority[label].number != link.number:
            raise ValueError(f"ambiguous combined reference presentation: {label}")
        authority[label] = link
    repairs: dict[Path, str] = {}
    prospective: dict[Path, str] = {}
    pages = sorted(directory.rglob("*.html"))
    for page in pages:
        if not page.is_file() or any(
            p.is_symlink()
            for p in (page, *page.parents)
            if p == directory or directory in p.parents
        ):
            raise ValueError(f"publication page is not an ordinary file: {page}")
        text = page.read_bytes().decode("utf-8")
        parsed = _PublicationHtml(text)
        replacements = (
            []
            if page == index
            else _publication_markup_replacements(parsed, combined, index, page, authority)
        )
        for start, end, label in parsed.references:
            if label not in authority:
                raise ValueError(f"missing authoritative combined reference: {label}")
            reference = authority[label]
            href = posixpath.relpath(index, page.parent) + "#" + label
            markup = _reference_link_markup(reference, href)
            replacements.append((start, end, reference.prefix + markup))
        for start, end, replacement in sorted(replacements, reverse=True):
            text = text[:start] + replacement + text[end:]
        checked = _ReferenceHtml(text)
        if (
            checked.reference_citations
            or _REFERENCE_MARKER.search("".join(checked.visible_text))
            or _EQUATION_DEFINITION_MARKER.search("".join(checked.visible_text))
        ):
            raise ValueError(f"unresolved publication reference: {page.name}")
        for link in checked.links:
            target = urlsplit(link.target)
            local = Path(posixpath.normpath(str(page.parent / unquote(target.path))))
            if local != index or target.scheme or target.netloc:
                continue
            label = unquote(target.fragment)
            if target.query:
                raise ValueError(f"unsupported publication reference query: {page.name}: {label}")
            if label not in authority or link.number != authority[label].number:
                raise ValueError(f"publication reference numbering differs: {page.name}: {label}")
            if not link.prefix or _reference_kind(link.prefix) != _reference_kind(
                authority[label].prefix
            ):
                raise ValueError(f"publication reference kind differs: {page.name}: {label}")
        if text != parsed.source:
            repairs[page] = text
        prospective[page] = text
    _validate_html_targets(
        pages, directory.parent.resolve(), allow_external=True, texts=prospective
    )
    return repairs


def heading_id(inventory: HtmlInventory, text: str, *, numbered: bool = False) -> str:
    """Require a unique actual rendered heading, selected by source text."""
    matches = [
        identity
        for identity, heading in inventory.headings
        if (heading.startswith(text + " ") if numbered else heading == text)
    ]
    if len(matches) != 1:
        raise ValueError(f"supporting source heading is absent or ambiguous: {text!r}")
    return matches[0]


def context_source(href: str) -> str:
    """Resolve one protocol link to its registered, directly referenced source."""
    path = unquote(urlsplit(href).path)
    source = posixpath.normpath(posixpath.join("docs", path))
    if path.endswith("/"):
        source += "/README.md"
    if source not in CONTEXT_SOURCES:
        raise ValueError(f"unregistered supporting context source: {href!r} -> {source}")
    return source


def relocate_protocol_html(text: str, data_id: str, data_page: str) -> str:
    """Replace only URI attribute spans; all provider content remains intact."""
    inventory = HtmlInventory(text)
    replacements: dict[str, str] = {}
    for target in inventory.targets:
        link = urlsplit(target)
        if link.scheme or link.netloc:
            continue
        if not link.path:
            if link.fragment in PROTOCOL_SECTIONS:
                replacements[target] = "#" + heading_id(
                    inventory, PROTOCOL_SECTIONS[link.fragment], numbered=True
                )
            continue
        if link.path == "favicon.ico":
            continue
        source = context_source(target)
        destination = "context/" + source
        if link.fragment:
            if source != "docs/reference/data_statement.md" or link.fragment != (
                "universal-dependencies-english-gum--connected-evaluation"
            ):
                raise ValueError(f"unregistered supporting context fragment: {target}")
            destination = data_page + "#" + data_id
        if link.query:
            raise ValueError(f"unsupported local supporting query: {target}")
        replacements[target] = destination

    def replace(match: re.Match[str]) -> str:
        target = html.unescape(match.group(2))
        return (
            match.group(1)
            + html.escape(replacements.get(target, target), quote=True)
            + match.group(3)
        )

    return re.sub(r'(\b(?:href|src|poster)\s*=\s*")([^"\n]*)(")', replace, text)


def source_view_markdown(text: str, section: str, identity: str) -> str:
    """Show the complete hydrated Markdown with a provider-backed section anchor.

    The original Markdown is also delivered verbatim. Fenced source views keep
    checkout-relative links literal instead of recursively publishing a website.
    The section anchor sits at its actual source heading, not at the top of a
    partial excerpt. Fence length is derived from the source to preserve it.
    """
    headings = list(re.finditer(r"^#{1,6} " + re.escape(section) + r"\s*$", text, re.M))
    if len(headings) != 1:
        raise ValueError(f"supporting source section is absent or ambiguous: {section!r}")
    offset = headings[0].start()
    fence = "`" * max(3, max((len(m.group()) for m in re.finditer(r"`+", text)), default=0) + 1)
    before, after = text[:offset], text[offset:]
    return (
        "# Hydrated source context\n\n"
        + fence
        + "text\n"
        + before
        + "\n"
        + fence
        + "\n\n[]{#"
        + identity
        + "}\n\n"
        + fence
        + "text\n"
        + after
        + "\n"
        + fence
        + "\n"
    )


def style_source_view_html(text: str) -> str:
    """Inherit provider colors without changing source text, geometry or IDs."""
    if text.count("</head>") != 1 or 'id="ailm-source-view-style"' in text:
        raise ValueError("source-view provider head is absent, ambiguous or already styled")
    return text.replace("</head>", SOURCE_VIEW_STYLE + "</head>", 1)


def validate_supporting_package(directory: Path) -> int:
    """Validate every actual HTML href/src/poster against a relocated package.

    Local paths must be ordinary files inside this package, including all path
    components. Fragments must be actual IDs in the delivered HTML. Remote URIs
    are not registered in this finite companion. A newly introduced external
    target fails closed rather than exempting an undelivered checkout reference.
    """
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"supporting package must be a real directory: {directory}")
    directory = directory.absolute()
    root = directory.resolve()
    pages = sorted(directory.rglob("*.html"))
    if not pages:
        raise ValueError("supporting package contains no complete HTML pages")
    return _validate_html_targets(pages, root, allow_external=False)


def validate_publication_html(directory: Path) -> int:
    """Check all delivered HTML consumers and assets at their actual locations.

    Figure assets are siblings of the web edition under output, whereas the
    complete protocol companion is beside its consumers. Remote publication
    references are retained; only local target resolution is checked here.
    """
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"publication HTML must be a real directory: {directory}")
    directory = directory.absolute()
    if not (directory / "index.html").is_file():
        raise ValueError("publication HTML has no combined index.html consumer")
    if publication_reference_repairs(directory):
        raise ValueError("unresolved publication references in individual HTML")
    from ailm.reference_presentation import PROTOCOL_REFERENCE_ANCHORS, PROTOCOL_SUPPORTING_HTML

    incoming = {
        PROTOCOL_SUPPORTING_HTML + "#" + anchor for anchor in PROTOCOL_REFERENCE_ANCHORS.values()
    }
    missing = incoming - set(HtmlInventory((directory / "index.html").read_text()).targets)
    if missing:
        raise ValueError(
            f"publication HTML is missing registered protocol consumers: {sorted(missing)}"
        )
    return _validate_html_targets(
        sorted(directory.rglob("*.html")), directory.parent.resolve(), allow_external=True
    )


def _validate_html_targets(
    pages: list[Path], root: Path, *, allow_external: bool, texts: Mapping[Path, str] | None = None
) -> int:
    count = 0
    target_ids: dict[Path, set[str]] = {}
    for page in pages:
        if page.is_symlink() or not page.is_file():
            raise ValueError(f"supporting page is not an ordinary file: {page}")
        inventory = HtmlInventory(
            texts[page] if texts is not None and page in texts else page.read_text(encoding="utf-8")
        )
        if (
            inventory.reference_citations
            or _REFERENCE_MARKER.search("".join(inventory.visible_text))
            or _EQUATION_DEFINITION_MARKER.search("".join(inventory.visible_text))
        ):
            raise ValueError(f"unresolved publication reference: {page.name}")
        for href in inventory.targets:
            link = urlsplit(href)
            # On an HTTP edition, http:relative keeps the browser's origin.
            # It cannot be exempted as an absolute remote publication URL.
            if link.scheme == "http" and not link.netloc:
                raise ValueError(f"unsupported authority-free HTTP target: {href}")
            if link.scheme or link.netloc:
                if allow_external and link.scheme in {"http", "https", "mailto", "data", ""}:
                    continue
                raise ValueError(f"unregistered external supporting target: {href}")
            target = page.parent / unquote(link.path) if link.path else page
            try:
                relative = target.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError(f"supporting target escapes package: {href}") from exc
            # Check the lexical path as well; resolve must not hide symlinks.
            lexical = Path(posixpath.normpath(str(target)))
            if lexical != root / relative or any(
                component.is_symlink()
                for component in (lexical, *lexical.parents)
                if component == root or root in component.parents
            ):
                raise ValueError(f"supporting target is not an ordinary file: {href}")
            if not target.is_file():
                raise ValueError(f"missing supporting target: {page.name}: {href}")
            if link.fragment:
                if target.suffix != ".html":
                    raise ValueError(f"missing supporting anchor: {page.name}: {href}")
                if lexical not in target_ids:
                    target_ids[lexical] = HtmlInventory(
                        texts[lexical]
                        if texts is not None and lexical in texts
                        else target.read_text(encoding="utf-8")
                    ).ids
                if unquote(link.fragment) not in target_ids[lexical]:
                    raise ValueError(f"missing supporting anchor: {page.name}: {href}")
            count += 1
    return count
