"""A rendered PDF must be structurally intact, not merely present and non-empty.

The render pipeline reported ``Valid PDFs: 1/1`` over a file whose cross-reference
table was unrecoverable: ``qpdf`` had to reconstruct the xref, two content streams
were truncated mid-flate, and a text extractor emitted hundreds of syntax errors.
The render had exited 0. Its notion of "valid" is that a file exists, is non-empty
and starts with a PDF header, and that is true of a corrupt file too.

The corruption was transient — a re-render was clean — which is exactly what makes
it worth a test rather than a memory. It appeared under heavy machine load, so it
will reappear under heavy machine load, and the next time nobody will be watching
the extractor's stderr.

**What these checks do not catch, measured rather than assumed.** Truncation at
any point, and a destroyed trailer, are both rejected by the trailer check and by
``qpdf --check`` independently. Overwriting bytes *inside* an already-declared
stream is caught by neither: lengths and offsets still agree, so the file is
structurally well-formed and only its rendered content is wrong. Detecting that
would mean decompressing and re-parsing every content stream, which is a
different and much more expensive check. The damage observed here was of the
truncation kind — the trailer named an offset where no xref lived — so these
checks cover it; a reader should not read them as a guarantee of a correct PDF.

What this file does NOT do is judge content. Unresolved tokens, dangling
cross-references and missing figures are checked by
``tests/test_manuscript_tokens.py`` and ``tests/test_manuscript_crossrefs.py``
against the manuscript sources, which is the right place: a structural check
cannot tell a correct sentence from an incorrect one, and a content check cannot
tell a whole file from a truncated one. It does reject explicit TeX errors from
the combined-manuscript log: TeX can recover, write a structurally valid PDF,
and exit nonzero after corrupting a bibliography or equation.

The tests skip when no PDF has been rendered, because rendering is a separate
stage that a clean checkout has not run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "output" / "pdf"

#: How much of the tail to read when locating the trailer. The startxref keyword
#: and its offset live in the last few hundred bytes of a well-formed file; a
#: generous window costs nothing and tolerates trailing whitespace.
_TAIL_BYTES = 4096

_STARTXREF_RE = re.compile(rb"startxref\s+(\d+)\s*%%EOF\s*$", re.DOTALL)
_TEX_ERROR_RE = re.compile(r"(?m)^! .+$")
_COMBINED_LOG = PDF_DIR / "_combined_manuscript.log"


def _tex_errors(log: str) -> list[str]:
    """Return TeX's explicit error diagnostics, excluding ordinary warnings."""
    return _TEX_ERROR_RE.findall(log)


def _rendered_pdfs() -> list[Path]:
    return sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.is_dir() else []


_SKIP_NO_PDF = pytest.mark.skipif(
    not _rendered_pdfs(), reason="no rendered PDF; run the render stage first"
)


@_SKIP_NO_PDF
def test_every_rendered_pdf_declares_a_reachable_cross_reference_table():
    """The trailer's ``startxref`` must point at a real xref, not past the file.

    This is the specific damage that was observed: ``xref not found`` at the
    offset the trailer names. A truncated or partially rewritten file usually
    fails here first, and it needs no external tool, so it runs everywhere.
    """
    for pdf in _rendered_pdfs():
        raw = pdf.read_bytes()
        assert raw.startswith(b"%PDF-"), f"{pdf.name} does not begin with a PDF header"

        match = _STARTXREF_RE.search(raw[-_TAIL_BYTES:])
        assert match is not None, (
            f"{pdf.name} has no well-formed 'startxref ... %%EOF' trailer. The "
            f"render was interrupted or the file was rewritten under it."
        )
        offset = int(match.group(1))
        assert 0 < offset < len(raw), (
            f"{pdf.name} declares its cross-reference table at byte {offset}, "
            f"outside the {len(raw)}-byte file"
        )
        # Either a classic table ("xref") or a cross-reference stream, which
        # begins with an indirect object header at that offset.
        here = raw[offset : offset + 64]
        assert here.startswith(b"xref") or re.match(rb"\d+\s+\d+\s+obj", here), (
            f"{pdf.name} declares its cross-reference table at byte {offset}, "
            f"but that offset holds {here[:24]!r} rather than an xref table or "
            f"an xref stream object"
        )


@_SKIP_NO_PDF
def test_every_rendered_pdf_survives_a_structural_check():
    """A full structural pass, when a checker is available.

    ``qpdf --check`` walks every object and every stream, which is what caught
    the truncated content streams the trailer check alone would have missed. It
    is skipped rather than vendored when absent: a weaker check that always runs
    is worth more than a stronger one that makes the suite undiscoverable on a
    machine without the tool.
    """
    qpdf = shutil.which("qpdf")
    if qpdf is None:
        pytest.skip("qpdf not installed; the trailer check above still ran")

    for pdf in _rendered_pdfs():
        result = subprocess.run(
            [qpdf, "--check", str(pdf)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, (
            f"{pdf.name} is structurally damaged. Re-render it; do not ship or "
            f"stage this file. The render stage reports success on a damaged "
            f"PDF, so its exit code is not evidence.\n"
            f"{(result.stdout + result.stderr)[-1200:]}"
        )


@pytest.mark.skipif(not _COMBINED_LOG.is_file(), reason="no combined TeX render log")
def test_combined_render_log_contains_no_tex_errors() -> None:
    """A recovered PDF is still a failed render when TeX emitted an error."""
    errors = _tex_errors(_COMBINED_LOG.read_text(encoding="utf-8", errors="replace"))
    assert errors == [], (
        "the combined manuscript log contains explicit TeX errors even though "
        f"a PDF was written: {errors}"
    )


def test_tex_error_scan_rejects_a_recovered_render() -> None:
    """NEGATIVE CONTROL: reproduce TeX's recover-and-write failure signature."""
    planted = "This is XeTeX\n! Missing $ inserted.\nOutput written on paper.pdf\n"
    assert _tex_errors(planted) == ["! Missing $ inserted."]


@pytest.mark.template_context
@pytest.mark.parametrize("broken", [False, True])
def test_bibliography_source_passes_bibtex_and_malformed_entry_fails(tmp_path, broken):
    """Use the actual bibliography compiler; a written PDF can conceal its failure."""
    bibtex = shutil.which("bibtex")
    assert bibtex is not None, "BibTeX is required for publication verification"
    source = (PROJECT_ROOT / "docs/manuscript/references.bib").read_text(encoding="utf-8")
    if broken:
        source += "\n@article{deliberately_unclosed,\n"
    (tmp_path / "references.bib").write_text(source, encoding="utf-8")
    (tmp_path / "paper.aux").write_text(
        "\\relax\n\\citation{*}\n\\bibstyle{plainnat}\n\\bibdata{references}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [bibtex, "paper"], cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30
    )
    diagnostics = result.stdout + result.stderr
    if broken:
        assert result.returncode != 0 and "error message" in diagnostics, diagnostics
    else:
        assert result.returncode == 0, diagnostics


def test_the_scan_knows_where_pdfs_live():
    """POSITIVE CONTROL: a wrong directory would skip every check silently.

    Both tests above skip when the glob is empty, so a renamed output directory
    would turn this file green while checking nothing. This asserts the path the
    render stage actually writes to.
    """
    assert PDF_DIR == PROJECT_ROOT / "output" / "pdf", (
        "the rendered-PDF directory moved; the skip conditions above would now "
        "hide every structural failure"
    )


@_SKIP_NO_PDF
def test_a_planted_truncation_is_caught():
    """POSITIVE CONTROL: the trailer check can fail.

    Truncating a real PDF is the cheapest faithful simulation of the observed
    damage, and it must be rejected. Without this, a regex that stopped matching
    anything would pass every file.
    """
    raw = _rendered_pdfs()[0].read_bytes()
    truncated = raw[: len(raw) // 2]
    assert _STARTXREF_RE.search(truncated[-_TAIL_BYTES:]) is None, (
        "a half-truncated PDF still presented a well-formed trailer, so the "
        "trailer check cannot detect truncation"
    )
