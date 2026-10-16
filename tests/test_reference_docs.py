"""Real metadata mutations and Markdown destinations for supporting source accounts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ailm.manuscript_variables.literature import (
    InvalidReferenceSourceError,
    _load_reference_variables,
)

ROOT = Path(__file__).resolve().parents[1]


def _snapshot(tmp_path: Path) -> tuple[Path, dict]:
    target = tmp_path / "data/reference_sources.json"
    target.parent.mkdir(parents=True)
    payload = json.loads((ROOT / "data/reference_sources.json").read_text())
    target.write_text(json.dumps(payload))
    return target, payload


@pytest.mark.parametrize(
    ("key", "token"),
    [
        ("case_transition_free_energy", "CEREBRUM_TRANSFORMATION_SUPPLEMENT"),
        ("expected_free_energy_transition", "CEREBRUM_EFE_TRANSITION_SUPPLEMENT"),
        ("core_function_table", "CEREBRUM_CORE_FUNCTION_TABLE"),
    ],
)
def test_new_locators_follow_real_metadata(tmp_path, key, token):
    target, payload = _snapshot(tmp_path)
    evidence = payload["sources"]["cerebrum"]["pdf_evidence"][key]
    values = _load_reference_variables(tmp_path)
    assert values[token] == evidence["locator"].split(" ", 1)[1]
    assert (
        values[token + "_URL"]
        == payload["sources"]["cerebrum"]["deposited_file"]["download_url"]
        + f"#page={evidence['physical_page']}"
    )
    evidence["locator"] = "Table 2" if key == "core_function_table" else "Supplement 9.7"
    evidence["physical_page"] += 1
    target.write_text(json.dumps(payload))
    changed = _load_reference_variables(tmp_path)
    assert changed[token] == evidence["locator"].split(" ", 1)[1]
    assert changed[token + "_PAGES"] == str(evidence["physical_page"])
    assert changed[token + "_URL"] != values[token + "_URL"]


@pytest.mark.parametrize("defect", ["missing", "malformed", "page-zero", "page-beyond-extent"])
@pytest.mark.parametrize(
    "key", ["case_transition_free_energy", "expected_free_energy_transition", "core_function_table"]
)
def test_new_locators_refuse_defects(tmp_path, key, defect):
    target, payload = _snapshot(tmp_path)
    evidence = payload["sources"]["cerebrum"]["pdf_evidence"][key]
    if defect == "missing":
        evidence.pop("locator")
    elif defect == "malformed":
        evidence["locator"] = "section unknown"
    else:
        evidence["physical_page"] = 0 if defect == "page-zero" else 10000
    target.write_text(json.dumps(payload))
    with pytest.raises(InvalidReferenceSourceError):
        _load_reference_variables(tmp_path)


def _html(text: str) -> str:
    pandoc = shutil.which("pandoc")
    assert pandoc, "real Pandoc required for destination proof"
    return subprocess.run(
        [pandoc, "-f", "markdown", "-t", "html"],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _verify_protocol_links(source: str, target_html: str) -> None:
    ids = set(re.findall(r'\bid="([^"]+)"', target_html))
    from ailm.manuscript_contract import resolve_manuscript_text
    from ailm.manuscript_variables import generate_manuscript_variables

    resolved = resolve_manuscript_text(source, generate_manuscript_variables(ROOT))
    destinations = re.findall(
        r"\]\(ailm-supporting-docs/docs__extension_protocols\.html#([^)]*)\)", resolved
    )
    assert destinations, "protocol consumers supply no links"
    assert set(destinations) <= ids, f"unresolved protocol destinations: {set(destinations) - ids}"


def test_protocol_consumers_target_real_markdown_ids_and_break_when_removed():
    target = (ROOT / "docs/extension_protocols.md").read_text()
    html = _html(target)
    for rel in [
        "docs/manuscript/05_discussion.md",
        "docs/manuscript/98_supplement_external_benchmarks.md",
    ]:
        source = (ROOT / rel).read_text()
        _verify_protocol_links(source, html)
        broken = (
            source.replace(
                "{{PROTOCOL_P3_BRIDGE_URL}}",
                "ailm-supporting-docs/docs__extension_protocols.html#absent-protocol",
            )
            .replace(
                "{{PROTOCOL_P1_FIT_EVALUATION_URL}}",
                "ailm-supporting-docs/docs__extension_protocols.html#absent-protocol",
            )
            .replace(
                "{{PROTOCOL_P2_ALIGNED_CORRESPONDENCE_URL}}",
                "ailm-supporting-docs/docs__extension_protocols.html#absent-protocol",
            )
        )
        with pytest.raises(AssertionError, match="unresolved protocol destinations"):
            _verify_protocol_links(broken, html)
    # Renumbering the document changes no explicit destination.
    renumbered = re.sub(r"^(#{2,3}) [\d.]+ ", r"\1 ", target, flags=re.M)
    _verify_protocol_links(
        (ROOT / "docs/manuscript/05_discussion.md").read_text(), _html(renumbered)
    )


def test_actual_supporting_prose_follows_mutated_source_label_and_page(tmp_path):
    from ailm.manuscript_contract import resolve_manuscript_text

    target, payload = _snapshot(tmp_path)
    source = (ROOT / "docs/design/cerebrum_mapping.md").read_text()
    initial = resolve_manuscript_text(source, _load_reference_variables(tmp_path))
    entry = payload["sources"]["cerebrum"]["pdf_evidence"]["case_transition_free_energy"]
    label, page = entry["locator"], entry["physical_page"]
    url = payload["sources"]["cerebrum"]["deposited_file"]["download_url"]
    assert f"[{label}]({url}#page={page})" in initial
    entry["locator"] = "Supplement 9.7"
    entry["physical_page"] += 1
    target.write_text(json.dumps(payload))
    changed = resolve_manuscript_text(source, _load_reference_variables(tmp_path))
    assert f"[{entry['locator']}]({url}#page={entry['physical_page']})" in changed
    assert f"[{label}]({url}#page={page})" not in changed
    assert "physical page " + str(entry["physical_page"]) in changed


def test_reference_hydration_is_live_on_actual_supporting_document(tmp_path):
    from ailm.docs_hydration import hydrate

    target, payload = _snapshot(tmp_path)
    source = (ROOT / "docs/design/cerebrum_mapping.md").read_text()
    initial = hydrate(source, {}, reference_variables=_load_reference_variables(tmp_path))
    entry = payload["sources"]["cerebrum"]["pdf_evidence"]["pomdp_tuple"]
    entry["locator"] = "Supplement 9.8"
    entry["physical_page"] += 1
    target.write_text(json.dumps(payload))
    changed = hydrate(source, {}, reference_variables=_load_reference_variables(tmp_path))
    assert "[Supplement 9.8]" in changed and "#page=91)" in changed
    assert initial != changed
    with pytest.raises(KeyError, match="no canonical producer"):
        hydrate(source, {})
    with pytest.raises(KeyError, match="no canonical producer"):
        hydrate("{{CEREBRUM_missing}}", {}, reference_variables=_load_reference_variables(tmp_path))
    assert (
        hydrate("{{TOKEN}} is a generic syntax example", {})
        == "{{TOKEN}} is a generic syntax example"
    )


def _pdf_uris(value):
    if isinstance(value, dict):
        if value.get("/S") == "/URI":
            yield value["/URI"].removeprefix("u:")
        for child in value.values():
            yield from _pdf_uris(child)
    elif isinstance(value, list):
        for child in value:
            yield from _pdf_uris(child)


def _validate_relocated_pdf_links(pdf: Path, html: str, uris: list[str]) -> None:
    from urllib.parse import urlsplit

    ids = set(re.findall(r'\bid="([^"]+)"', html))
    assert len(set(uris)) == 3, "the real PDF must contain all three distinct protocol destinations"
    for uri in uris:
        link = urlsplit(uri)
        assert not link.scheme and not link.netloc
        assert (pdf.parent / link.path).is_file(), f"missing relocated protocol page: {uri}"
        assert link.fragment in ids, f"missing relocated protocol anchor: {uri}"


@pytest.mark.template_context
def test_whole_protocol_page_and_actual_pdf_destinations_survive_relocation(tmp_path):
    from ailm.docs_hydration import hydrate
    from ailm.manuscript_contract import resolve_manuscript_text
    from ailm.manuscript_variables import generate_manuscript_variables
    from scripts.z_generate_manuscript_variables import render_supporting_protocol_html

    values = generate_manuscript_variables(ROOT)
    claims = json.loads((ROOT / "data/claims.json").read_text())["claims"]
    canonical = (ROOT / "docs/extension_protocols.md").read_text()
    hydrated = tmp_path / "extension_protocols.md"
    hydrated.write_text(
        hydrate(canonical, claims, reference_variables=_load_reference_variables(ROOT))
    )
    package = tmp_path / "build-package"
    companion = package / "ailm-supporting-docs"
    assert render_supporting_protocol_html(hydrated, companion) == 0
    assert render_supporting_protocol_html(hydrated, companion, check_only=True) == 0
    linked_output = tmp_path / "linked-package"
    linked_output.symlink_to(companion, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        render_supporting_protocol_html(hydrated, linked_output, check_only=True)
    target_page = companion / "docs__extension_protocols.html"
    current_bytes = target_page.read_bytes()
    target_page.rename(target_page.with_suffix(".missing-control"))
    assert render_supporting_protocol_html(hydrated, companion, check_only=True) == 1
    target_page.with_suffix(".missing-control").rename(target_page)
    target_page.write_bytes(current_bytes + b"\n<!-- planted stale output -->\n")
    assert render_supporting_protocol_html(hydrated, companion, check_only=True) == 1
    target_page.write_bytes(current_bytes)
    paragraphs = []
    for name in ["05_discussion.md", "98_supplement_external_benchmarks.md"]:
        text = (ROOT / "docs/manuscript" / name).read_text()
        paragraphs.extend(
            resolve_manuscript_text(p, values)
            for p in text.split("\n\n")
            if re.search(r"\{\{PROTOCOL_[A-Z0-9_]+_URL\}\}", p)
        )
    assert len(paragraphs) == 3
    source = tmp_path / "protocol-consumer-passages.md"
    source.write_text("\n\n".join(paragraphs))
    pdf = package / "protocol-consumer-passages.pdf"
    subprocess.run(
        ["pandoc", str(source), "-o", str(pdf), "--pdf-engine=xelatex"],
        check=True,
        capture_output=True,
        text=True,
    )
    objects = subprocess.run(
        ["qpdf", "--json", str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    (tmp_path / "pdf-objects.json").write_text(objects)
    uris = list(_pdf_uris(json.loads(objects)))
    # Move only the portable package. No checkout-relative file enters this directory.
    downloads = tmp_path / "relocated" / "Downloads" / "AILM-R15"
    downloads.parent.mkdir(parents=True)
    shutil.copytree(package, downloads)
    relocated = downloads / pdf.name
    page = downloads / "ailm-supporting-docs/docs__extension_protocols.html"
    html = page.read_text()
    _validate_relocated_pdf_links(relocated, html, uris)
    with pytest.raises(AssertionError, match="missing relocated protocol anchor"):
        _validate_relocated_pdf_links(
            relocated, html.replace('id="protocol-p3-bridge"', 'id="broken-protocol"'), uris
        )
    page.rename(page.with_suffix(".retained-html"))
    with pytest.raises(AssertionError, match="missing relocated protocol page"):
        _validate_relocated_pdf_links(relocated, html, uris)
    page.with_suffix(".retained-html").rename(page)
    # The production generator itself refuses a canonical document missing a stable label.
    hydrated.write_text(hydrated.read_text().replace("[]{#protocol-p3-bridge}", ""))
    with pytest.raises(RuntimeError, match="missing registered anchors"):
        render_supporting_protocol_html(hydrated, tmp_path / "broken-package")


def test_reference_errors_preserve_historical_identity_and_real_catch_contract(tmp_path):
    from ailm.manuscript_variables._exceptions import (
        DuplicateInputKeyError as legacy_duplicate_error,
    )
    from ailm.manuscript_variables._exceptions import (
        InvalidReferenceSourceError as legacy_reference_error,
    )
    from ailm.reference_presentation import DuplicateInputKeyError as shared_duplicate_error
    from ailm.reference_presentation import InvalidReferenceSourceError as shared_reference_error
    from ailm.reference_presentation import _load_reference_variables as stdlib_presenter

    assert legacy_reference_error is shared_reference_error
    assert legacy_duplicate_error is shared_duplicate_error
    assert shared_reference_error.__module__ == "ailm.manuscript_variables._exceptions"
    assert shared_duplicate_error.__module__ == "ailm.manuscript_variables._exceptions"
    target, _ = _snapshot(tmp_path)
    target.write_text('{"_schema":"ailm-reference-sources-v1", "_schema":"duplicate"}')
    with pytest.raises(legacy_duplicate_error, match="duplicate JSON key"):
        stdlib_presenter(tmp_path)
    target.write_text('{"_schema":"wrong"}')
    with pytest.raises(legacy_reference_error, match="must declare schema"):
        stdlib_presenter(tmp_path)
