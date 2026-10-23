"""Real complete-package positives and production failure controls."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

from ailm.docs_hydration import hydrate
from ailm.reference_presentation import _load_reference_variables
from ailm.supporting_package import (
    CONTEXT_SOURCES,
    DATA_SECTION,
    HtmlInventory,
    context_source,
    heading_id,
    provider_asset,
    relocate_protocol_html,
    source_view_markdown,
    style_source_view_html,
    validate_publication_html,
    validate_supporting_package,
)

ROOT = Path(__file__).resolve().parents[1]


def _markup_web(tmp_path):
    web, page, _ = _reference_web(tmp_path)
    index = web / "index.html"
    parenthetical = (
        '<span class="citation" data-cites="alpha beta">(see <a href="#ref-alpha" '
        'role="doc-biblioref">Alpha 2001</a>; <a href="#ref-beta" '
        'role="doc-biblioref">Beta 2002</a>, pp. 4–5)</span>'
    )
    narrative = (
        '<span class="citation" data-cites="alpha">Alpha (<a href="#ref-alpha" '
        'role="doc-biblioref">2001</a>)</span>'
    )
    math = r'<span class="math display">\[x + y = z\]</span>'
    equation = '<span id="eq:real">' + math.replace(r"\]", r"\qquad{(23)}\]") + "</span>"
    heading = '<h2 id="section-real">Real section</h2>'
    index.write_text(
        index.read_text()
        + heading
        + "<p>é\u2028"
        + parenthetical
        + " / "
        + narrative
        + equation
        + "</p>"
        + 'eq.\u00a0<a href="#eq:real">23</a>'
        + '<div id="ref-alpha">Alpha bibliography</div><div id="ref-beta">Beta bibliography</div>'
    )
    raw_narrative = '<span class="citation" data-cites="alpha">@alpha</span>'
    literal = "".join(
        f"<{tag}>[alpha; beta] @alpha {{#eq:real}}</{tag}>"
        for tag in ("pre", "code", "script", "style")
    )
    before = "<!-- keep bytes -->\r\n" + heading + '<p title="unchanged">é\u2028'
    after = "</p>\r\n" + literal + "<p>user@example.org @unrelated [ordinary prose]</p>"
    original = before + "[alpha;\n beta] / " + raw_narrative + " " + math + " {#eq:real}" + after
    expected = (
        before
        + parenthetical.replace('href="#', 'href="index.html#')
        + " / "
        + narrative.replace('href="#', 'href="index.html#')
        + " "
        + math
        + ' <span id="eq:real">eq.\u00a0<a href="index.html#eq:real">23</a></span>'
        + after
    )
    page.write_bytes(original.encode())
    return web, page, original, expected


def test_equation_and_occurrence_citation_transport_preserves_exact_non_targets(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, original, expected = _markup_web(tmp_path)
    with pytest.raises(ValueError, match="unresolved publication references"):
        validate_publication_html(web)
    assert publication_reference_repairs(web) == {page: expected}
    assert page.read_bytes() == original.encode()
    page.write_bytes(expected.encode())
    assert publication_reference_repairs(web) == {}
    assert validate_publication_html(web) > 0


@pytest.mark.parametrize(
    "target,old,new",
    [
        ("page", "{#eq:real}", ""),
        ("page", "{#eq:real}", "{#eq:missing}"),
        ("page", "{#eq:real}", "{#eq:real} {#eq:real}"),
        ("page", "{#eq:real}", "{#eq:real"),
        ("page", "{#eq:real}", "{# eq:real}"),
        ("page", "{#eq:real}", "text {#eq:real}"),
        ("page", "x + y = z", "x - y = z"),
        ("page", "[alpha;\n beta]", ""),
        ("page", "[alpha;\n beta]", "[beta; alpha]"),
        ("page", "[alpha;\n beta]", "[alpha; missing]"),
        ("page", "[alpha;\n beta]", "[alpha; beta] [alpha; beta]"),
        ("page", 'data-cites="alpha"', 'data-cites="beta"'),
        ("page", ">@alpha</span>", ">@beta</span>"),
        ("page", 'class="citation" data-cites="alpha"', 'data-cites="alpha"'),
        (
            "page",
            'class="citation" data-cites="alpha"',
            'class="citation" data-cites="alpha" data-cites="beta"',
        ),
        ("page", 'id="section-real"', 'id="unknown-section"'),
        ("index", 'id="ref-alpha"', 'id="ref-absent"'),
        ("index", 'href="#ref-alpha"', 'href="#ref-beta"'),
        ("index", 'data-cites="alpha beta"', 'data-cites="alpha alpha"'),
        ("index", r"\qquad{(23)}", r"\qquad{(24)}"),
        ("index", 'id="eq:real"', 'id="eq:absent"'),
        ("index", 'id="ref-alpha"', 'id="ref-beta"'),
        ("index", 'class="citation" data-cites="alpha beta"', 'class="citation"'),
        ("index", 'role="doc-biblioref">Alpha 2001</a>', 'role="doc-biblioref"></a>'),
        ("index", ">Alpha 2001</a>", ">@alpha</a>"),
        ("page", '<span class="citation" data-cites="alpha">@alpha</span>', "@alpha"),
    ],
)
def test_equation_and_citation_real_file_refusals_preserve_input(tmp_path, target, old, new):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _, _ = _markup_web(tmp_path)
    path = page if target == "page" else web / "index.html"
    text = path.read_bytes().decode()
    assert old in text
    path.write_bytes(text.replace(old, new, 1).encode())
    before = {p: p.read_bytes() for p in web.rglob("*.html")}
    for check in (publication_reference_repairs, validate_publication_html):
        with pytest.raises(ValueError):
            check(web)
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize(
    "old,new",
    [
        ("x + y = z", "x - y = z"),
        ("index.html#eq:real", "index.html#eq:example"),
        (">23</a>", ">24</a>"),
        ("Alpha 2001", "Invented 2099"),
        ("index.html#ref-alpha", "index.html#ref-beta"),
    ],
)
def test_repaired_publication_rejects_stale_math_citations_and_equation_targets(tmp_path, old, new):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _, expected = _markup_web(tmp_path)
    assert old in expected
    page.write_bytes(expected.replace(old, new, 1).encode())
    with pytest.raises(ValueError):
        publication_reference_repairs(web)


def test_already_correct_native_equation_is_preserved(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, original, expected = _markup_web(tmp_path)
    native = (
        '<span id="eq:real"><span class="math display">\\[x + y = z\\qquad{(23)}\\]</span></span>'
    )
    raw = r'<span class="math display">\[x + y = z\]</span> {#eq:real}'
    fixed = (
        r'<span class="math display">\[x + y = z\]</span> <span id="eq:real">eq.'
        + '\u00a0<a href="index.html#eq:real">23</a></span>'
    )
    page.write_bytes(original.replace(raw, native).encode())
    assert publication_reference_repairs(web)[page] == expected.replace(fixed, native)


@pytest.mark.parametrize("repaired", [False, True])
@pytest.mark.parametrize(
    "formula,wrapped",
    [
        ("x + y = z", "x +\ny = z"),
        ("x +\ny = z", "x + y = z"),
        (r"x \le t", "x \\le\nt"),
        (r"\frac{x + y}{z}", "\\frac{x +\ny}{z}"),
        (r"\text{two  words} + x", "\\text{two  words} +\nx"),
        (r"x\ y + z", "x\\ y +\nz"),
        ("x " + "\\" * 2 + " y", "x " + "\\" * 2 + "\ny"),
    ],
)
def test_equation_identity_accepts_single_line_wraps(tmp_path, repaired, formula, wrapped):
    from ailm.supporting_package import publication_reference_repairs

    web, page, original, expected = _markup_web(tmp_path)
    index = web / "index.html"
    index.write_bytes(index.read_bytes().replace(b"x + y = z", formula.encode()))
    source = expected if repaired else original
    page.write_bytes(source.replace("x + y = z", wrapped).encode())
    before = {path: path.read_bytes() for path in web.rglob("*.html")}
    plan = publication_reference_repairs(web)
    assert plan == ({} if repaired else {page: expected.replace("x + y = z", wrapped)})
    assert {path: path.read_bytes() for path in before} == before
    if not repaired:
        page.write_bytes(plan[page].encode())
    assert publication_reference_repairs(web) == {}
    assert validate_publication_html(web) > 0


@pytest.mark.parametrize("repaired", [False, True])
@pytest.mark.parametrize(
    "formula,corrupt",
    [
        (r"x \le t", r"x \let"),
        ("x \\le\nt", r"x \let"),
        ("x \\le\tt", r"x \let"),
        (r"\text{two words}", r"\text{twowords}"),
        (r"\verb|a b|", r"\verb|ab|"),
        ("x % comment\n+ y", "x % comment + y"),
        (r"x\ y", r"x\y"),
        (r"\text{two  words}", r"\text{two words}"),
        (r"\verb|a b|", "\\verb|a\nb|"),
        ("x\n\n+ y", "x  + y"),
        ("x\n \n+ y", "x   + y"),
        ("x\\\ny", "x\\ y"),
        ("\\obeylines x\ny", r"\obeylines x y"),
        ("\\obeyspaces x\ny", r"\obeyspaces x y"),
        ("\\catcode`x=10 x\ny", r"\catcode`x=10 x y"),
        ("\\endlinechar=32 x\ny", r"\endlinechar=32 x y"),
        ("x " + "\\" * 3 + "\ny", "x " + "\\" * 3 + " y"),
        ("x ^^M\ny", "x ^^M y"),
        ("x +\ty", "x + y"),
    ],
)
def test_equation_identity_preserves_tex_spacing(tmp_path, repaired, formula, corrupt):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _, _ = _markup_web(tmp_path)
    index = web / "index.html"
    for path in (index, page):
        path.write_bytes(path.read_bytes().replace(b"x + y = z", formula.encode()))
    canonical = index.read_bytes()
    plan = publication_reference_repairs(web)
    assert set(plan) == {page}
    if repaired:
        page.write_bytes(plan[page].encode())
        assert publication_reference_repairs(web) == {}
        assert validate_publication_html(web) > 0
    original = page.read_bytes()
    assert formula.encode() in original
    page.write_bytes(original.replace(formula.encode(), corrupt.encode(), 1))
    for check in (publication_reference_repairs, validate_publication_html):
        with pytest.raises(ValueError, match="equation definition math or numbering differs"):
            check(web)
    assert index.read_bytes() == canonical


@pytest.mark.parametrize(
    "mutation", ["none", "reorder-context", "changed-context", "extra", "ambiguous", "identical"]
)
@pytest.mark.template_context
def test_actual_citeproc_modes_prefixes_suffixes_and_multi_citations_are_transported(
    tmp_path, mutation
):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    bib = tmp_path / "references.bib"
    bib.write_text(
        "@book{alpha, author={Smith, Jane}, title={Alpha}, year={2001}}\n"
        "@book{beta, author={Jones, Bill}, title={Beta}, year={2002}}\n"
    )
    source = tmp_path / "source.md"
    source.write_text(
        "# Citations {#citation-modes}\n\n"
        "Narrative @alpha. Normal [@alpha]. Suppressed [-@alpha]. "
        "Prefixed [see @alpha, p. 4]. Multiple [@alpha; @beta, pp. 2–3].\n"
    )
    if mutation == "ambiguous":
        source.write_text("# Citations {#citation-modes}\n\n[@alpha]\n\n[-@alpha]\n")
    if mutation == "identical":
        source.write_text("# Citations {#citation-modes}\n\n[@alpha]\n\n[@alpha]\n")
    canonical = tmp_path / "canonical.html"
    rendered = subprocess.run(
        [
            "pandoc",
            str(source),
            "-f",
            "markdown",
            "-t",
            "html5",
            "--citeproc",
            "-M",
            "link-citations=true",
            "--bibliography",
            str(bib),
            "-o",
            str(canonical),
        ],
        capture_output=True,
        text=True,
    )
    assert rendered.returncode == 0, rendered.stderr
    provider = os.environ["TEMPLATE_REPO_ROOT"]
    flattened = tmp_path / "flattened.md"
    preprocess = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from pathlib import Path; import sys; "
            "from infrastructure.rendering._web_markdown_preprocess import render_pandoc_citations; "
            "Path(sys.argv[2]).write_text(render_pandoc_citations(Path(sys.argv[1]).read_text()))",
            str(source),
            str(flattened),
        ],
        env={**os.environ, "PYTHONPATH": provider},
        capture_output=True,
        text=True,
    )
    assert preprocess.returncode == 0, preprocess.stderr
    raw = tmp_path / "individual.html"
    rendered = subprocess.run(
        ["pandoc", str(flattened), "-f", "markdown", "-t", "html5", "-o", str(raw)],
        capture_output=True,
        text=True,
    )
    assert rendered.returncode == 0, rendered.stderr
    combined = canonical.read_text()
    index = web / "index.html"
    index.write_text(index.read_text() + combined)
    page.write_bytes(raw.read_bytes())
    if mutation == "reorder-context":
        page.write_text(
            page.read_text().replace(
                "Normal [alpha]. Suppressed [alpha].", "Suppressed [alpha]. Normal [alpha]."
            )
        )
    if mutation == "changed-context":
        page.write_text(page.read_text().replace("Prefixed", "Changed"))
    if mutation == "extra":
        page.write_text(page.read_text().replace("Normal [alpha].", "Normal [alpha] [alpha]."))
    if mutation not in {"none", "identical"}:
        with pytest.raises(
            ValueError, match="citation (prose context|occurrence count|presentation)"
        ):
            publication_reference_repairs(web)
        return
    spans = re.findall(r'<span\s+class="citation".*?</span>', combined, re.S)
    assert len(spans) == (5 if mutation == "none" else 2)
    if mutation == "none":
        assert "see " in spans[3] and "4" in spans[3] and "2–3" in spans[4]
        assert spans[0] != spans[1] != spans[2]
    repaired = publication_reference_repairs(web)[page]
    assert re.findall(r'<span\s+class="citation".*?</span>', repaired, re.S) == [
        span.replace('href="#', 'href="index.html#') for span in spans
    ]
    page.write_bytes(repaired.encode())
    assert publication_reference_repairs(web) == {}
    assert validate_publication_html(web) > 0


def _reference_web(tmp_path):
    from ailm.reference_presentation import PROTOCOL_REFERENCE_ANCHORS, PROTOCOL_SUPPORTING_HTML

    web = tmp_path / "web"
    web.mkdir()
    companion = web / PROTOCOL_SUPPORTING_HTML
    companion.parent.mkdir()
    companion.write_text(
        "".join(f'<h2 id="{a}">Protocol</h2>' for a in PROTOCOL_REFERENCE_ANCHORS.values())
    )
    incoming = "".join(
        f'<a href="{PROTOCOL_SUPPORTING_HTML}#{a}">Protocol</a>'
        for a in PROTOCOL_REFERENCE_ANCHORS.values()
    )
    index = (
        '<h2 id="sec:example" data-number="4.2">Actual heading</h2>'
        '<div id="fig:example"></div><table id="tbl:example"></table>'
        '<span id="eq:example"></span>'
        'sec.\u00a0<a href="#sec:example">4.2</a> '
        'fig.\u00a0<a href="#fig:example">17</a> '
        'tbl.\u00a0<a href="#tbl:example">8</a> '
        'eq.\u00a0<a href="#eq:example">12</a>' + incoming
    )
    (web / "index.html").write_text(index)
    page = web / "individual.html"
    citations = " ".join(
        f'<span class="citation" data-cites="{kind}:example">[@{kind}:example]</span>'
        for kind in ("sec", "fig", "tbl", "eq")
    )
    literal = (
        "<pre><code>[@sec:literal]</code></pre><code>[@fig:literal]</code>"
        '<script>const example="[@eq:literal]";</script>'
        "<style>/* [@tbl:literal] */</style>"
    )
    page.write_text("<p>" + citations + "</p>" + literal)
    return web, page, literal


def test_global_reference_plan_uses_real_index_numbers_and_preserves_literal_regions(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, literal = _reference_web(tmp_path)
    original = page.read_text()
    with pytest.raises(ValueError, match="unresolved publication reference"):
        validate_publication_html(web)
    plan = publication_reference_repairs(web)
    assert page.read_text() == original  # A write plan never publishes partial repairs.
    assert set(plan) == {page}
    repaired = plan[page]
    assert repaired.endswith(literal)
    for kind, number in (("sec", "4.2"), ("fig", "17"), ("tbl", "8"), ("eq", "12")):
        assert f'{kind}.\u00a0<a href="index.html#{kind}:example">{number}</a>' in repaired
    page.write_text(repaired)
    assert publication_reference_repairs(web) == {}
    assert validate_publication_html(web) > 0


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing-target", "anchor inconsistency"),
        ("missing-presentation", "missing authoritative"),
        ("duplicate-id", "ambiguous HTML id"),
        ("duplicate-href", "duplicate HTML attribute"),
        ("duplicate-class", "duplicate HTML attribute"),
        ("duplicate-cites", "duplicate HTML attribute"),
        ("conflicting-number", "ambiguous combined"),
        ("wrong-heading-number", "numbering inconsistency"),
        ("wrong-prefix", "anchor inconsistency"),
        ("unknown-label", "missing authoritative"),
        ("malformed-text", "malformed reference citation text"),
        ("multiple-labels", "malformed reference citation"),
        ("nested-markup", "malformed reference citation"),
        ("unclosed-citation", "unclosed reference"),
        ("bare-marker", "unresolved publication reference"),
        ("split-marker", "unresolved publication reference"),
        ("unclosed-literal", "unclosed literal"),
    ],
)
def test_global_reference_real_file_refusals_do_not_write(tmp_path, mutation, expected):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    index = web / "index.html"
    source, combined = page.read_text(), index.read_text()
    if mutation == "missing-target":
        combined = combined.replace('id="sec:example"', 'id="removed"')
    elif mutation == "missing-presentation":
        combined = combined.replace('sec.\u00a0<a href="#sec:example">4.2</a>', "")
    elif mutation == "duplicate-id":
        combined += '<h2 id="sec:example">Duplicate</h2>'
    elif mutation == "duplicate-href":
        combined = combined.replace('href="#sec:example"', 'href="#sec:example" href="#other"')
    elif mutation == "duplicate-class":
        source = source.replace('class="citation"', 'class="citation" class="other"', 1)
    elif mutation == "duplicate-cites":
        source = source.replace(
            'data-cites="sec:example"', 'data-cites="sec:example" data-cites="eq:example"'
        )
    elif mutation == "conflicting-number":
        combined = (
            combined.replace(' data-number="4.2"', "") + ' sec.\u00a0<a href="#sec:example">9</a>'
        )
    elif mutation == "wrong-heading-number":
        combined = combined.replace('data-number="4.2"', 'data-number="9"')
    elif mutation == "wrong-prefix":
        combined = combined.replace("sec.\u00a0<a", "eq.\u00a0<a", 1)
    elif mutation == "unknown-label":
        source = source.replace("sec:example", "sec:absent")
    elif mutation == "malformed-text":
        source = source.replace("[@sec:example]", "[@sec:other]")
    elif mutation == "multiple-labels":
        source = source.replace('data-cites="sec:example"', 'data-cites="sec:example sec:other"')
    elif mutation == "nested-markup":
        source = source.replace("[@sec:example]", "<strong>[@sec:example]</strong>")
    elif mutation == "unclosed-citation":
        source = '<span class="citation" data-cites="sec:example">[@sec:example]'
    elif mutation == "bare-marker":
        source += "<p>[@sec:example]</p>"
    elif mutation == "split-marker":
        source += "<p>[@sec:<em>example</em>]</p>"
    else:
        source += "<pre>literal source"
    page.write_text(source)
    index.write_text(combined)
    with pytest.raises(ValueError, match=expected):
        publication_reference_repairs(web)
    assert page.read_text() == source
    assert index.read_text() == combined


def test_stale_global_reference_display_number_fails_after_index_changes(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    page.write_text(publication_reference_repairs(web)[page])
    index = web / "index.html"
    index.write_text(index.read_text().replace("4.2", "4.3"))
    with pytest.raises(ValueError, match="publication reference numbering differs"):
        publication_reference_repairs(web)


def test_nested_page_uses_relative_index_and_refuses_stale_numbers_and_queries(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, original, _ = _reference_web(tmp_path)
    nested = web / "nested"
    nested.mkdir()
    page = nested / "individual.html"
    original.rename(page)
    page.write_text(publication_reference_repairs(web)[page])
    assert 'href="../index.html#sec:example"' in page.read_text()
    assert publication_reference_repairs(web) == {}
    good = page.read_text()
    page.write_text(good.replace(">4.2</a>", ">99</a>"))
    with pytest.raises(ValueError, match="publication reference numbering differs"):
        publication_reference_repairs(web)
    page.write_text(good.replace("../index.html#sec:example", "../index.html?q#sec:example"))
    with pytest.raises(ValueError, match="unsupported publication reference query"):
        publication_reference_repairs(web)


def test_missing_asset_refuses_complete_plan_before_any_write(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    page.write_text(page.read_text() + '<img src="missing.png">')
    bad = page.read_bytes()
    with pytest.raises(ValueError, match="missing supporting target"):
        publication_reference_repairs(web)
    assert page.read_bytes() == bad


@pytest.mark.parametrize(
    "prefix",
    ["fig.\u00a0", "Figure ", "<em>fig.</em>\u00a0", "fig.<span>&nbsp;</span>", "", "sec.<br>"],
)
def test_resolved_reference_kind_and_missing_prefix_refused_by_every_production_mode(
    tmp_path, prefix
):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    bad = "<p>" + prefix + '<a href="index.html#sec:example"><strong>4.2</strong></a></p>'
    page.write_text(bad)
    for check in (publication_reference_repairs, validate_publication_html):
        with pytest.raises(ValueError, match="publication reference kind differs"):
            check(web)
    argv = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/z_render_docs.py"),
        "--publication-references",
        "--web-dir",
        str(web),
    ]
    for flags in ([], ["--check"]):
        result = subprocess.run([*argv, *flags], capture_output=True, timeout=30)
        assert result.returncode == 1
        assert b"publication reference kind differs" in result.stderr
        assert page.read_text() == bad


@pytest.mark.parametrize("prefix", ["sec.\u00a0", "Section ", "<em>sec.</em>\u00a0", "SEC. "])
def test_legitimate_visible_kind_markup_and_descriptive_navigation_remain_unchanged(
    tmp_path, prefix
):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    good = (
        "<p>" + prefix + '<a href="index.html#sec:example"><strong>4.2</strong></a></p>'
        '<p><a href="index.html#sec:example">Actual heading</a></p>'
    )
    page.write_text(good)
    assert publication_reference_repairs(web) == {}
    assert validate_publication_html(web) > 0
    assert page.read_text() == good


def test_visible_prefix_cannot_be_borrowed_from_a_previous_paragraph(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    page.write_text('<p>sec.</p><p><a href="index.html#sec:example">4.2</a></p>')
    with pytest.raises(ValueError, match="publication reference kind differs"):
        publication_reference_repairs(web)


@pytest.mark.parametrize(
    "tag",
    [
        "section",
        "article",
        "blockquote",
        "main",
        "aside",
        "figure",
        "address",
        "details",
        "dialog",
        "dl",
        "fieldset",
        "footer",
        "form",
        "header",
        "hgroup",
        "legend",
        "menu",
        "nav",
        "ol",
        "search",
        "summary",
        "table",
        "tbody",
        "thead",
        "tfoot",
        "tr",
        "ul",
        "center",
        "dir",
    ],
)
def test_block_container_prefix_scope_and_inline_positive_controls(tmp_path, tag):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    link = '<a href="index.html#sec:example"><strong>4.2</strong></a>'

    def container(content):
        if tag in {"ol", "ul", "menu", "dir"}:
            return f"<{tag}><li>{content}</li></{tag}>"
        if tag == "dl":
            return f"<dl><dt>{content}</dt></dl>"
        if tag == "table":
            return f"<table><tbody><tr><td>{content}</td></tr></tbody></table>"
        if tag in {"tbody", "thead", "tfoot"}:
            return f"<table><{tag}><tr><td>{content}</td></tr></{tag}></table>"
        if tag == "tr":
            return f"<table><tbody><tr><td>{content}</td></tr></tbody></table>"
        if tag == "legend":
            return f"<fieldset><legend>{content}</legend></fieldset>"
        if tag == "summary":
            return f"<details><summary>{content}</summary></details>"
        if tag == "details":
            return f"<details><summary>Context</summary>{content}</details>"
        if tag == "hgroup":
            return f"<hgroup><h2>{content}</h2></hgroup>"
        return f"<{tag}>{content}</{tag}>"

    bad = container("sec.") + container(" " + link)
    page.write_text(bad)
    argv = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/z_render_docs.py"),
        "--publication-references",
        "--web-dir",
        str(web),
    ]
    for check in (publication_reference_repairs, validate_publication_html):
        with pytest.raises(ValueError, match="publication reference kind differs"):
            check(web)
    for flags in ([], ["--check"]):
        rejected = subprocess.run([*argv, *flags], capture_output=True, timeout=30)
        assert rejected.returncode == 1
        assert b"publication reference kind differs" in rejected.stderr
        assert page.read_text() == bad
    good = container("<em>sec.</em>\u00a0" + link)
    page.write_text(good)
    assert publication_reference_repairs(web) == {}
    assert validate_publication_html(web) > 0
    accepted = subprocess.run([*argv, "--check"], capture_output=True, timeout=30)
    assert accepted.returncode == 0, accepted.stderr.decode()
    assert page.read_text() == good


@pytest.mark.parametrize("separator", ["\r\n", "\r", "\n", "\u2028"])
def test_reference_span_repair_preserves_unrelated_utf8_bytes_and_line_separators(
    tmp_path, separator
):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    before = ("unchanged α" + separator + "<p>Before ").encode()
    after = (" after.</p>" + separator + "<p>second</p>\n<p>tail</p>").encode()
    raw = before + b'<span class="citation" data-cites="sec:example">[@sec:example]</span >' + after
    page.write_bytes(raw)
    expected = before + 'sec.\u00a0<a href="index.html#sec:example">4.2</a>'.encode() + after
    assert publication_reference_repairs(web)[page].encode() == expected
    argv = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/z_render_docs.py"),
        "--publication-references",
        "--web-dir",
        str(web),
    ]
    written = subprocess.run(argv, capture_output=True, timeout=30)
    assert written.returncode == 0, written.stderr.decode()
    assert page.read_bytes() == expected
    checked = subprocess.run([*argv, "--check"], capture_output=True, timeout=30)
    assert checked.returncode == 0, checked.stderr.decode()
    assert page.read_bytes() == expected


@pytest.mark.parametrize("closing", ["</span >", "</span\t>", "</SPAN\n >", "</span \r\n >"])
def test_actual_closing_tag_ranges_preserve_exact_prose_and_cli_checks(tmp_path, closing):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    page.write_text(
        '<p>Before <span class="citation" data-cites="sec:example">[@sec:example]'
        + closing
        + " after.</p>"
    )
    expected = '<p>Before sec.\u00a0<a href="index.html#sec:example">4.2</a> after.</p>'
    assert publication_reference_repairs(web)[page] == expected
    argv = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/z_render_docs.py"),
        "--publication-references",
        "--web-dir",
        str(web),
    ]
    written = subprocess.run(argv, capture_output=True, timeout=30)
    assert written.returncode == 0, written.stderr.decode()
    assert page.read_text() == expected
    checked = subprocess.run([*argv, "--check"], capture_output=True, timeout=30)
    assert checked.returncode == 0, checked.stderr.decode()
    assert page.read_text() == expected


def test_authoritative_link_closing_whitespace_and_nested_number_markup_are_preserved(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    index = web / "index.html"
    index.write_text(index.read_text().replace(">4.2</a>", "><em>4.2</em></a >"))
    page.write_text(publication_reference_repairs(web)[page])
    assert 'href="index.html#sec:example"><em>4.2</em></a >' in page.read_text()
    assert publication_reference_repairs(web) == {}
    assert validate_publication_html(web) > 0


def test_malformed_closing_tag_attributes_are_rejected_without_a_write(tmp_path):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    bad = page.read_text().replace("</span>", "</span extra>", 1)
    page.write_text(bad)
    with pytest.raises(ValueError, match="malformed reference closing tag"):
        publication_reference_repairs(web)
    assert page.read_text() == bad


@pytest.mark.parametrize("quote", ['"', "'"])
def test_real_combined_href_quote_forms_keep_authoritative_presentation(tmp_path, quote):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    index = web / "index.html"
    index.write_text(
        index.read_text().replace('href="#sec:example"', f"href={quote}#sec:example{quote}")
    )
    page.write_text(publication_reference_repairs(web)[page])
    assert f"href={quote}index.html#sec:example{quote}" in page.read_text()
    assert publication_reference_repairs(web) == {}


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_publication_page_requires_an_ordinary_file(tmp_path, kind):
    from ailm.supporting_package import publication_reference_repairs

    web, page, _ = _reference_web(tmp_path)
    page.rename(web / "retained")
    if kind == "directory":
        page.mkdir()
    elif kind == "symlink":
        page.symlink_to(web / "retained")
    else:
        os.mkfifo(page)
    with pytest.raises(ValueError, match="not an ordinary file"):
        publication_reference_repairs(web)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--publication-references", "--supporting-protocols"],
        ["--publication-references", "--prune-retired"],
        ["--publication-references", "--dry-run"],
        ["--web-dir", "unselected"],
    ],
)
def test_publication_reference_cli_rejects_conflicting_phases(arguments):
    child = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/z_render_docs.py"), *arguments],
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 2
    assert b"error:" in child.stderr


def test_publication_reference_cli_write_check_and_failure_controls(tmp_path):
    web, page, literal = _reference_web(tmp_path)
    argv = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/z_render_docs.py"),
        "--publication-references",
        "--web-dir",
        str(web),
    ]
    original = page.read_bytes()
    stale = subprocess.run([*argv, "--check"], capture_output=True, timeout=30)
    assert stale.returncode == 1
    assert b"unresolved publication references" in stale.stderr
    assert page.read_bytes() == original
    published = subprocess.run(argv, capture_output=True, timeout=30)
    assert published.returncode == 0, published.stderr.decode()
    assert page.read_text().endswith(literal)
    repaired = page.read_bytes()
    checked = subprocess.run([*argv, "--check"], capture_output=True, timeout=30)
    assert checked.returncode == 0, checked.stderr.decode()
    assert page.read_bytes() == repaired
    index = web / "index.html"
    index.write_text(index.read_text().replace('id="sec:example"', 'id="removed"'))
    invalid = subprocess.run(argv, capture_output=True, timeout=30)
    assert invalid.returncode == 1
    assert b"anchor inconsistency" in invalid.stderr
    assert page.read_bytes() == repaired


@pytest.mark.parametrize("tag", ["p", "strong", "span"])
def test_existing_html_validator_rejects_visible_reference_markup(tmp_path, tag):
    page = tmp_path / "page.html"
    page.write_text(f"<{tag}>[@sec:missing]</{tag}>")
    with pytest.raises(ValueError, match="unresolved publication reference"):
        validate_supporting_package(tmp_path)
    page.write_text("<pre><code>[@sec:example]</code></pre><code>[@sec:example]</code>")
    assert validate_supporting_package(tmp_path) == 0


class _SourceBlocks(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.active = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        if tag == "pre":
            self.active = True
            self.blocks.append("")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.blocks[-1] += data


def _source(tmp_path: Path) -> Path:
    claims = json.loads((ROOT / "data/claims.json").read_text())["claims"]
    source = tmp_path / "extension_protocols.md"
    source.write_text(
        hydrate(
            (ROOT / "docs/extension_protocols.md").read_text(),
            claims,
            reference_variables=_load_reference_variables(ROOT),
        )
    )
    return source


@pytest.mark.template_context
def test_complete_relocated_package_and_production_mutations(tmp_path):
    from scripts.z_generate_manuscript_variables import render_supporting_protocol_html

    source = _source(tmp_path)
    original = tmp_path / "build" / "ailm-supporting-docs"
    assert render_supporting_protocol_html(source, original) == 0
    relocated = tmp_path / "Downloads" / "AILM-R15" / "ailm-supporting-docs"
    shutil.copytree(original, relocated)
    count = validate_supporting_package(relocated)
    assert count >= 24
    assert render_supporting_protocol_html(source, relocated, check_only=True) == 0
    context = relocated / "context"
    assert {p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()} == (
        CONTEXT_SOURCES
    )
    assert (context / "data/claims.json").read_bytes() == (ROOT / "data/claims.json").read_bytes()
    page = relocated / "docs__extension_protocols.html"
    html = page.read_text()
    inventory = HtmlInventory(html)
    assert not any(href.startswith("../") for href in inventory.targets)
    assert all(
        "context/" in href
        or href.startswith("#")
        or provider_asset(href) is not None
        or href.startswith("reference__data_statement.html#")
        for href in inventory.targets
    )
    # Every provider asset the page references was delivered beside it, and
    # the delivered extras are nothing but such assets.
    delivered_assets = {
        p.relative_to(relocated).as_posix()
        for p in relocated.rglob("*")
        if p.is_file() and not p.relative_to(relocated).as_posix().startswith("context/")
    } - {page.name, "reference__data_statement.html", "favicon.ico"}
    assert all(provider_asset(name) is not None for name in delivered_assets)
    assert {provider_asset(h) for h in inventory.targets if provider_asset(h)} - {
        "favicon.ico"
    } <= (delivered_assets)

    # The production entrypoint and its local-target validator both refuse a
    # real missing delivered context file, not just a test-side oracle.
    target = context / "docs/design/README.md"
    content = target.read_bytes()
    target.rename(target.with_suffix(".retained"))
    with pytest.raises(ValueError, match="missing supporting target"):
        validate_supporting_package(relocated)
    assert render_supporting_protocol_html(source, relocated, check_only=True) == 1
    target.with_suffix(".retained").rename(target)
    assert target.read_bytes() == content

    identity = heading_id(inventory, "5.", numbered=True)
    page.write_text(html.replace(f'id="{identity}"', 'id="removed-anchor-control"'))
    with pytest.raises(ValueError, match="missing supporting anchor"):
        validate_supporting_package(relocated)
    assert render_supporting_protocol_html(source, relocated, check_only=True) == 1
    page.write_text(html)
    data_page = relocated / "reference__data_statement.html"
    data_html = data_page.read_text()
    data_text = (context / "docs/reference/data_statement.md").read_text()
    offset = re.search(r"^## " + re.escape(DATA_SECTION) + "$", data_text, re.M).start()
    blocks = _SourceBlocks(data_html).blocks
    assert len(blocks) == 2
    assert blocks[0] == data_text[:offset]
    assert blocks[1] == data_text[offset:]
    data_id = next(
        href.split("#", 1)[1] for href in inventory.targets if href.startswith(data_page.name + "#")
    )
    data_page.write_text(data_html.replace(f'id="{data_id}"', 'id="removed-data-anchor"'))
    with pytest.raises(ValueError, match="missing supporting anchor"):
        validate_supporting_package(relocated)
    assert render_supporting_protocol_html(source, relocated, check_only=True) == 1
    data_page.write_text(data_html)
    assert validate_supporting_package(relocated) == count

    # Break a fresh render, so the production target check must fail even with
    # freshly generated artifact hashes and all incoming protocol spans intact.
    good = source.read_text()
    source.write_text(good + "\n\n[Broken internal navigation](#absent-whole-page-id)\n")
    with pytest.raises(ValueError, match="missing supporting anchor"):
        render_supporting_protocol_html(source, tmp_path / "fresh-broken-anchor")
    source.write_text(good + "\n\n[Undelivered context](absent-context.md)\n")
    with pytest.raises(ValueError, match="unregistered supporting context source"):
        render_supporting_protocol_html(source, tmp_path / "fresh-broken-target")
    source.write_text(re.sub(r"^## 5\..*$", "", good, flags=re.M))
    with pytest.raises(ValueError, match="source heading is absent"):
        render_supporting_protocol_html(source, tmp_path / "fresh-removed-heading")


@pytest.mark.parametrize("attribute", ["href", "src", "poster"])
@pytest.mark.parametrize("target", ["asset.ico", "section.html#section", "#here", ""])
def test_every_uri_attribute_has_live_file_and_fragment_controls(tmp_path, attribute, target):
    (tmp_path / "asset.ico").write_bytes(b"real asset")
    section = tmp_path / "section.html"
    section.write_text('<h1 id="section">Section</h1>')
    page = tmp_path / "page.html"
    page.write_text(f'<h1 id="here">Here</h1><a {attribute}="{target}">link</a>')
    assert validate_supporting_package(tmp_path) == 1
    page.write_text(f'<a {attribute}="absent.ico">link</a>')
    with pytest.raises(ValueError, match="missing supporting target"):
        validate_supporting_package(tmp_path)
    page.write_text(f'<a {attribute}="section.html#absent">link</a>')
    with pytest.raises(ValueError, match="missing supporting anchor"):
        validate_supporting_package(tmp_path)


@pytest.mark.parametrize("target", ["../outside.ico", "%2e%2e/outside.ico", "/outside.ico"])
def test_local_target_cannot_escape_package(tmp_path, target):
    package = tmp_path / "package"
    package.mkdir()
    (tmp_path / "outside.ico").write_bytes(b"outside")
    (package / "page.html").write_text(f'<a href="{target}">escape</a>')
    with pytest.raises(ValueError, match="escapes package"):
        validate_supporting_package(package)


@pytest.mark.parametrize("kind", ["file", "directory", "page", "package"])
def test_package_paths_refuse_real_symlinks(tmp_path, kind):
    package = tmp_path / "package"
    package.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "asset.ico").write_bytes(b"outside")
    page = package / "page.html"
    page.write_text('<a href="asset.ico">asset</a>')
    if kind == "file":
        (package / "asset.ico").symlink_to(other / "asset.ico")
    elif kind == "directory":
        (package / "linked").symlink_to(other, target_is_directory=True)
        page.write_text('<a href="linked/asset.ico">asset</a>')
    elif kind == "page":
        (other / "empty.html").write_text("<h1>source</h1>")
        page.rename(package / "retained.page")
        page.symlink_to(other / "empty.html")
    else:
        linked = tmp_path / "linked-package"
        linked.symlink_to(package, target_is_directory=True)
        package = linked
    with pytest.raises(ValueError, match="ordinary file|escapes package|real directory"):
        validate_supporting_package(package)


def test_complete_source_view_preserves_text_and_uses_actual_section_identity():
    text = (ROOT / "docs/reference/data_statement.md").read_text()
    identity = "provider-emitted-identity"
    view = source_view_markdown(text, DATA_SECTION, identity)
    fence = re.search(r"^(`+)text$", view, re.M).group(1)
    chunks = re.findall(re.escape(fence) + r"text\n(.*?)\n" + re.escape(fence), view, re.S)
    assert "".join(chunks) == text
    assert view.index("[]{#" + identity + "}") < view.index("## " + DATA_SECTION)
    with pytest.raises(ValueError, match="absent or ambiguous"):
        source_view_markdown(text, "Absent source section", identity)


def test_heading_ids_are_read_from_provider_html_without_a_slug_algorithm():
    inventory = HtmlInventory('<h2 id="arbitrary-provider-id">2. Protocol <code>P1</code></h2>')
    assert heading_id(inventory, "2.", numbered=True) == "arbitrary-provider-id"
    assert heading_id(inventory, "2. Protocol P1") == "arbitrary-provider-id"
    with pytest.raises(ValueError, match="absent or ambiguous"):
        heading_id(inventory, "3.", numbered=True)
    duplicate = HtmlInventory('<h2 id="a">2. One</h2><h2 id="b">2. Two</h2>')
    with pytest.raises(ValueError, match="absent or ambiguous"):
        heading_id(duplicate, "2.", numbered=True)


@pytest.mark.parametrize(
    "markup",
    [
        '<script data-template-source-font-loader src="assets/source-web-fonts.js"></script>',
        '<link rel="stylesheet" href="assets/site.css">',
        '<img src="./assets/logo.svg">',
        '<link rel="icon" href="favicon.ico">',
    ],
)
def test_provider_assets_relocate_unchanged_without_context_registration(markup):
    """A provider improvement that adds a resource must not read as an unregistered source."""
    assert relocate_protocol_html(markup, "data-id", "data.html") == markup


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("assets/source-web-fonts.js", "assets/source-web-fonts.js"),
        ("./assets/site.css", "assets/site.css"),
        ("assets/fonts/source.woff2", "assets/fonts/source.woff2"),
        ("favicon.ico", "favicon.ico"),
        ("../assets/escape.js", None),
        ("assets/../../escape.js", None),
        ("/assets/absolute.js", None),
        ("assets/site.css?v=2", None),
        ("assets/site.css#fragment", None),
        ("design/README.md", None),
        ("context/data/claims.json", None),
        ("https://example.org/remote.js", None),
        ("", None),
    ],
)
def test_provider_asset_classification_is_local_contained_and_suffix_bound(href, expected):
    assert provider_asset(href) == expected


def test_escaping_or_decorated_assets_keep_the_source_context_rules():
    """NEGATIVE CONTROLS: the carve-out admits only contained, plain resource paths."""
    with pytest.raises(ValueError, match="unregistered supporting context source"):
        relocate_protocol_html(
            '<script src="../assets/escape.js"></script>', "data-id", "data.html"
        )
    with pytest.raises(ValueError, match="unregistered supporting context source"):
        relocate_protocol_html('<link href="assets/site.css?v=2">', "data-id", "data.html")
    with pytest.raises(ValueError, match="unregistered supporting context source"):
        relocate_protocol_html('<a href="assets/notes.md">notes</a>', "data-id", "data.html")


def test_a_referenced_provider_asset_must_be_delivered_as_an_ordinary_file(tmp_path):
    package = tmp_path / "package"
    (package / "assets").mkdir(parents=True)
    (package / "page.html").write_text(
        '<html><body><script src="assets/fonts.js"></script></body></html>', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing supporting target"):
        validate_supporting_package(package)
    (package / "assets" / "fonts.js").write_text("/* provider asset */\n", encoding="utf-8")
    assert validate_supporting_package(package) == 1


def test_mapping_preserves_provider_bytes_except_registered_attribute_values():
    text = '<h2 id="provider-id">5. Ownership</h2><a href="#5-ownership-grant-boundaries-and-what-this-milestone-did-not-do">§5</a>'
    assert relocate_protocol_html(text, "data-id", "data.html") == text.replace(
        'href="#5-ownership-grant-boundaries-and-what-this-milestone-did-not-do"',
        'href="#provider-id"',
    )
    assert context_source("design/") == "docs/design/README.md"
    with pytest.raises(ValueError, match="unregistered"):
        context_source("absent.md")
    with pytest.raises(ValueError, match="unregistered supporting context fragment"):
        relocate_protocol_html('<a href="README.md#absent">link</a>', "data-id", "data.html")
    with pytest.raises(ValueError, match="unsupported local supporting query"):
        relocate_protocol_html('<a href="README.md?query">link</a>', "data-id", "data.html")
    remote = '<a href="https://example.org/source#section">source</a>'
    assert relocate_protocol_html(remote, "data-id", "data.html") == remote


def test_empty_packages_and_non_html_fragments_fail(tmp_path):
    with pytest.raises(ValueError, match="no complete HTML"):
        validate_supporting_package(tmp_path)
    (tmp_path / "raw.md").write_text("# Section")
    (tmp_path / "page.html").write_text('<a href="raw.md#section">link</a>')
    with pytest.raises(ValueError, match="missing supporting anchor"):
        validate_supporting_package(tmp_path)


def test_unregistered_external_targets_and_internal_symlinks_fail_closed(tmp_path):
    page = tmp_path / "page.html"
    for uri in ("https://example.org/source", "//example.org/source", "file:///outside"):
        page.write_text(f'<a href="{uri}">unregistered destination</a>')
        with pytest.raises(ValueError, match="unregistered external supporting target"):
            validate_supporting_package(tmp_path)
    (tmp_path / "real.ico").write_bytes(b"real asset")
    (tmp_path / "linked.ico").symlink_to(tmp_path / "real.ico")
    page.write_text('<a href="linked.ico">linked asset</a>')
    with pytest.raises(ValueError, match="not an ordinary file"):
        validate_supporting_package(tmp_path)


def test_source_view_style_preserves_entire_provider_document():
    page = (
        '<html><head></head><body><pre class="text"><code>whole source</code></pre></body></html>'
    )
    from ailm.supporting_package import SOURCE_VIEW_STYLE

    corrected = style_source_view_html(page)
    assert corrected.replace(SOURCE_VIEW_STYLE, "") == page
    for invalid in ("<body>no head</body>", page + "</head>", corrected):
        with pytest.raises(ValueError, match="provider head"):
            style_source_view_html(invalid)


@pytest.mark.parametrize("attribute", ["href", "src", "poster"])
def test_actual_publication_targets_include_sibling_assets_and_all_pages(tmp_path, attribute):
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    figures = web.parent / "figures"
    figures.mkdir()
    asset = figures / "real.png"
    asset.write_bytes(b"real local asset")
    from ailm.reference_presentation import PROTOCOL_REFERENCE_ANCHORS, PROTOCOL_SUPPORTING_HTML

    companion = web / PROTOCOL_SUPPORTING_HTML
    companion.parent.mkdir()
    companion.write_text(
        "".join(
            f'<h2 id="{anchor}">Protocol</h2>' for anchor in PROTOCOL_REFERENCE_ANCHORS.values()
        )
    )
    incoming = "".join(
        f'<a href="{PROTOCOL_SUPPORTING_HTML}#{anchor}">Protocol</a>'
        for anchor in PROTOCOL_REFERENCE_ANCHORS.values()
    )
    (web / "index.html").write_text(incoming + '<a href="section.html#actual">section</a>')
    section = web / "section.html"
    section.write_text(f'<h1 id="actual">Actual</h1><img {attribute}="../figures/real.png">')
    assert validate_publication_html(web) == 2 + len(PROTOCOL_REFERENCE_ANCHORS)
    section.write_text(
        section.read_text() + '<a href="https://example.org/retained-reference">Reference</a>'
    )
    assert validate_publication_html(web) == 2 + len(PROTOCOL_REFERENCE_ANCHORS)
    section.write_text(
        section.read_text() + '<a href="http://example.org/retained-reference">Reference</a>'
    )
    assert validate_publication_html(web) == 2 + len(PROTOCOL_REFERENCE_ANCHORS)
    section.write_text(section.read_text() + '<a href="file:///outside">Unsafe</a>')
    with pytest.raises(ValueError, match="unregistered external supporting target"):
        validate_publication_html(web)
    section.write_text(f'<h1 id="actual">Actual</h1><img {attribute}="../figures/real.png">')
    asset.rename(asset.with_suffix(".retained"))
    with pytest.raises(ValueError, match="missing supporting target"):
        validate_publication_html(web)
    asset.with_suffix(".retained").rename(asset)
    section.write_text('<h1 id="removed">Actual</h1>')
    with pytest.raises(ValueError, match="missing supporting anchor"):
        validate_publication_html(web)
    section.write_text(f'<h1 id="actual">Actual</h1><a {attribute}="../../../outside">escape</a>')
    with pytest.raises(ValueError, match="escapes package"):
        validate_publication_html(web)
    # Isolate the next refusal: an earlier escaped target must not remain live.
    section.write_text('<h1 id="actual">Actual</h1>')
    (web / "index.html").write_text("<html><body>Empty consumer</body></html>")
    with pytest.raises(ValueError, match="missing registered protocol consumers"):
        validate_publication_html(web)
    (web / "index.html").rename(web / "retained-index")
    with pytest.raises(ValueError, match="no combined index"):
        validate_publication_html(web)


def test_publication_directory_refuses_absent_paths_and_real_symlinks(tmp_path):
    with pytest.raises(ValueError, match="real directory"):
        validate_publication_html(tmp_path / "absent")
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        validate_publication_html(linked)


@pytest.mark.template_context
def test_production_cli_delivers_and_checks_both_editions_and_actual_consumers(tmp_path):
    """Use an isolated real source tree and subprocess; no imported globals change."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("src", "scripts", "docs", "data", "vendor"):
        shutil.copytree(ROOT / name, project / name)
    for name in ("README.md", "AGENTS.md", "CHANGELOG.md", "TODO.md"):
        shutil.copy2(ROOT / name, project / name)
    for name in ("web", "figures"):
        shutil.copytree(ROOT / "output" / name, project / "output" / name)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def run(*arguments, phase="--supporting-protocols"):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(project / "scripts/z_render_docs.py"),
                phase,
                *arguments,
            ],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        with (tmp_path / "production-commands.log").open("a") as log:
            log.write(f"argv={completed.args!r}\nexit={completed.returncode}\n")
            log.write(completed.stdout + completed.stderr)
        return completed

    generated = run()
    assert generated.returncode == 0, generated.stdout + generated.stderr
    references = run(phase="--publication-references")
    assert references.returncode == 0, references.stdout + references.stderr
    checked = run("--check")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    bundles = [project / "output" / edition / "ailm-supporting-docs" for edition in ("pdf", "web")]
    inventory = {
        p.relative_to(bundles[0]): p.read_bytes() for p in bundles[0].rglob("*") if p.is_file()
    }
    assert inventory
    assert {
        p.relative_to(bundles[1]): p.read_bytes() for p in bundles[1].rglob("*") if p.is_file()
    } == inventory
    for bundle in bundles:
        for name in ("docs__extension_protocols.html", "favicon.ico", "context/data/claims.json"):
            target = bundle / name
            original = target.read_bytes()
            target.rename(target.with_name(target.name + ".retained"))
            assert run("--check").returncode == 1
            target.with_name(target.name + ".retained").rename(target)
            target.write_bytes(original + b"\nstale-control")
            assert run("--check").returncode == 1
            target.write_bytes(original)
        page = bundle / "docs__extension_protocols.html"
        original = page.read_text()
        page.write_text(original.replace('id="protocol-p3-bridge"', 'id="removed-control"'))
        assert run("--check").returncode == 1
        page.write_text(original)
        view = bundle / "reference__data_statement.html"
        original_view = view.read_text()
        from ailm.supporting_package import SOURCE_VIEW_STYLE

        view.write_text(original_view.replace(SOURCE_VIEW_STYLE, ""))
        assert run("--check").returncode == 1
        view.write_text(original_view)
    # Entire actual consumers, not only the three authored link paragraphs.
    section = project / "output/web/manuscript__01_introduction.html"
    original = section.read_text()
    section.write_text(
        original + '<a href="ailm-supporting-docs/docs__extension_protocols.html#absent">bad</a>'
    )
    assert run("--check").returncode == 1
    assert run(phase="--publication-references").returncode == 1
    section.write_text(original)
    assert run("--check").returncode == 0
    # Malformed navigation outside the original consumer paragraphs must be
    # refused by both real production modes, including normalized case names.
    controls = []
    for attribute in ("href", "src", "poster"):
        for first, last in (
            ("absent-independent-target.html", "index.html"),
            ("index.html", "absent-independent-target.html"),
        ):
            for second_name in (attribute, attribute.upper()):
                tag = {"href": "a", "src": "img", "poster": "video"}[attribute]
                controls.append(
                    f'<{tag} {attribute}="{first}" {second_name}="{last}">duplicate</{tag}>'
                )
    controls.extend(
        [
            '<a id="browser-first" id="phantom-last">duplicate ID</a><a href="#phantom-last">go</a>',
            '<a ID="browser-first" id="phantom-last">duplicate ID</a><a href="#phantom-last">go</a>',
            '<a id="repeated-control">first</a><p id="repeated-control">second</p><a href="#repeated-control">go</a>',
        ]
    )
    controls.extend(
        [
            '<base href="ailm-supporting-docs/context/docs/design/README.md"><a href="index.html">base target</a>',
            '<a href="http:absent-independent-target.html">relative scheme target</a>',
            '<img src="http:absent-independent-target.html">',
            '<video poster="http:absent-independent-target.html"></video>',
        ]
    )
    for number, control in enumerate(controls):
        section.write_text(original + control)
        shutil.copy2(section, tmp_path / f"duplicate-consumer-{number}.html")
        for arguments in ((), ("--check",)):
            result = run(*arguments, phase="--publication-references")
            assert result.returncode == 1, result.stdout + result.stderr
            assert any(
                message in result.stderr
                for message in (
                    "duplicate HTML attribute",
                    "ambiguous HTML id",
                    "unsupported HTML base href",
                    "unsupported authority-free HTTP target",
                )
            )
        section.write_text(original)
    assert run().returncode == 0
    assert run(phase="--publication-references").returncode == 0
    assert run("--check").returncode == 0


@pytest.mark.parametrize("attribute", ["href", "src", "poster", "id"])
@pytest.mark.parametrize("mixed_case", [False, True])
@pytest.mark.parametrize(
    "values",
    [
        ("absent.html", "section.html"),
        ("section.html", "absent.html"),
        ("section.html", "section.html"),
        (None, "section.html"),
        ("", "section.html"),
    ],
)
def test_duplicate_navigation_attributes_fail_closed(tmp_path, attribute, mixed_case, values):
    (tmp_path / "section.html").write_text('<h1 id="section">Section</h1>')
    page = tmp_path / "page.html"
    name = attribute.upper() if mixed_case else attribute
    attrs = " ".join(
        key if value is None else f'{key}="{value}"'
        for key, value in zip((attribute, name), values, strict=True)
    )
    tag = {"href": "a", "src": "img", "poster": "video", "id": "h2"}[attribute]
    page.write_text(f"<{tag} {attrs}>target</{tag}>")
    with pytest.raises(ValueError, match=f"duplicate HTML attribute: {attribute}"):
        validate_supporting_package(tmp_path)


@pytest.mark.parametrize("second", ["section", "sect&#105;on"])
def test_duplicate_id_values_make_local_fragment_ambiguous(tmp_path, second):
    page = tmp_path / "page.html"
    page.write_text(
        f'<h1 id="section">First</h1><p id="{second}">Second</p><a href="#section">go</a>'
    )
    with pytest.raises(ValueError, match="ambiguous HTML id: section"):
        validate_supporting_package(tmp_path)


def test_unambiguous_single_attributes_and_page_scoped_ids_pass(tmp_path):
    (tmp_path / "page.html").write_text(
        '<h1 ID="section">Section</h1><a HREF="other.html#section">go</a>'
    )
    (tmp_path / "other.html").write_text(
        '<h1 id="section">Other</h1><img SRC="asset.ico"/><video POSTER="asset.ico"></video>'
    )
    (tmp_path / "asset.ico").write_bytes(b"real asset")
    assert validate_supporting_package(tmp_path) == 3


@pytest.mark.parametrize("name", ["base", "BASE"])
def test_base_href_cannot_rebase_existing_local_targets(tmp_path, name):
    (tmp_path / "context").mkdir()
    (tmp_path / "context/source.md").write_text("real context source")
    (tmp_path / "target.html").write_text("real local destination")
    page = tmp_path / "page.html"
    page.write_text(f'<{name} href="context/source.md"><a href="target.html">go</a>')
    with pytest.raises(ValueError, match="unsupported HTML base href"):
        validate_supporting_package(tmp_path)


def test_base_without_href_preserves_local_resolution(tmp_path):
    (tmp_path / "target.html").write_text("real local destination")
    (tmp_path / "page.html").write_text('<base target="_self"><a href="target.html">go</a>')
    assert validate_supporting_package(tmp_path) == 1


@pytest.mark.parametrize("attribute", ["href", "src", "poster"])
@pytest.mark.parametrize("scheme", ["http", "HTTP"])
def test_authority_free_http_target_is_not_a_remote_exemption(tmp_path, attribute, scheme):
    (tmp_path / "target.html").write_text("real local destination")
    (tmp_path / "page.html").write_text(f'<a {attribute}="{scheme}:target.html">go</a>')
    with pytest.raises(ValueError, match="unsupported authority-free HTTP target"):
        validate_supporting_package(tmp_path)
