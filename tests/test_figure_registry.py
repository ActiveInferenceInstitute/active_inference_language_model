"""The generated-figure registry, and the boundary it is published across.

`scripts/generate_figure_registry.py` is the one place this project imports the
template's `infrastructure.documentation` builder. The tests here cover the
project-owned half — how the specs are assembled from real sources — and the
boundary rules, but deliberately not the envelope's schema: that belongs to the
template, is validated by the template's own builder, and re-asserting its shape
here would create a second definition free to drift from the first.

Everything reachable without a template checkout runs unconditionally. The
publishing step itself is skipped when `infrastructure` is absent, because a
bare checkout must still be able to run the suite.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_figure_registry import (  # noqa: E402
    SCHEMA_VERSION,
    FigureSpec,
    _is_manuscript_section,
    build_specs,
    collect_manuscript_figures,
    load_provenance,
    resolve_template_root,
)

REGISTRY = REPO_ROOT / "output" / "figures" / "figure_registry.json"
PROVENANCE = REPO_ROOT / "output" / "figures" / "figure_provenance.json"

needs_figures = pytest.mark.skipif(
    not PROVENANCE.exists(), reason="figures not generated in this checkout"
)


def _infrastructure_imports(source: str) -> list[tuple[int, str]]:
    """Return real imports of the template-only package from Python source."""

    offenders: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "infrastructure" or alias.name.startswith("infrastructure."):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "infrastructure" or node.module.startswith("infrastructure."):
                offenders.append((node.lineno, node.module))
    return offenders


def test_invalid_explicit_template_root_never_falls_back(tmp_path, monkeypatch):
    """A misspelled authoritative path must not select another checkout."""
    from scripts.z_generate_manuscript_variables import _resolve_template_root

    invalid = tmp_path / "missing-template"
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(invalid))
    assert _resolve_template_root() is None
    assert resolve_template_root() is None
    invalid.mkdir()
    assert resolve_template_root() is None


def test_valid_explicit_template_root_is_used_by_both_publishers(tmp_path, monkeypatch):
    from scripts.z_generate_manuscript_variables import _resolve_template_root

    template = tmp_path / "chosen-template"
    infrastructure = template / "infrastructure"
    infrastructure.mkdir(parents=True)
    (infrastructure / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("TEMPLATE_REPO_ROOT", str(template))
    assert resolve_template_root() == _resolve_template_root() == template


# ── The project-owned facts ──────────────────────────────────────────────────


@needs_figures
def test_provenance_supplies_filenames_and_generators():
    """`generated_by` must name a real figure module, not a placeholder."""
    produced = load_provenance()
    assert produced, "the provenance receipt registered no figures"
    assert all(name.endswith(".png") for name in produced)
    assert all(module.startswith("ailm.figures.") for module in produced.values())


def test_captions_and_labels_come_from_the_manuscript():
    """POSITIVE: real figures are found with their labels and captions.

    NEGATIVE CONTROL: the scan must not reach `SYNTAX.md`, whose worked example
    points at `name.png` — a file that is not a figure and was never generated.
    Including it made the builder claim a file that cannot exist.
    """
    found = collect_manuscript_figures()
    assert len(found) > 20, f"only {len(found)} figures found in the manuscript"
    assert "name.png" not in found, "the syntax guide's example leaked into the scan"

    for filename, (label, caption) in found.items():
        assert filename.endswith(".png"), filename
        assert label.startswith("fig:"), label
        assert caption.strip(), f"{label} has an empty caption"
        assert "\n" not in caption, f"{label} caption was not collapsed to one line"


@pytest.mark.parametrize("same_section", [False, True])
@pytest.mark.parametrize(
    ("filename", "label", "message"),
    [
        ("first.png", "fig:second", "duplicate figure filename"),
        ("second.png", "fig:first", "duplicate figure label"),
        ("first.png", "fig:first", "duplicate figure filename"),
    ],
)
def test_manuscript_scan_rejects_duplicate_references(
    tmp_path, same_section, filename, label, message
):
    manuscript = tmp_path / "docs" / "manuscript"
    manuscript.mkdir(parents=True)
    first = "![First caption](../output/figures/first.png){#fig:first}\n"
    second = f"![Second caption](../output/figures/{filename}){{#{label}}}\n"
    (manuscript / "01_first.md").write_text(first + (second if same_section else ""))
    if not same_section:
        (manuscript / "02_second.md").write_text(second)
    with pytest.raises(ValueError, match=message) as error:
        collect_manuscript_figures(tmp_path)
    assert "01_first.md" in str(error.value)
    if not same_section:
        assert "02_second.md" in str(error.value)


def test_section_detection_accepts_sections_and_rejects_supporting_docs():
    """The rule is stated over the general class, not by naming SYNTAX.md."""
    assert _is_manuscript_section(Path("03_methods.md"))
    assert _is_manuscript_section(Path("07a_clause_level.md"))
    assert _is_manuscript_section(Path("98_supplement_external_benchmarks.md"))

    assert not _is_manuscript_section(Path("SYNTAX.md"))
    assert not _is_manuscript_section(Path("AGENTS.md"))
    assert not _is_manuscript_section(Path("README.md"))


@needs_figures
def test_every_referenced_figure_was_actually_generated():
    """A referenced-but-ungenerated figure is fatal; the reverse is tolerated.

    The paper printing an image nobody produced is an error. Producing an image
    nobody cites is not — cover art and diagnostic plates exist — and the
    template's builder documents the same tolerance.
    """
    specs = build_specs()
    assert specs, "no figure specs were built"

    produced = set(load_provenance())
    referenced = {spec.filename for spec in specs}
    assert referenced <= produced, f"referenced but never generated: {referenced - produced}"

    for spec in specs:
        assert (REPO_ROOT / "output" / "figures" / spec.filename).is_file()


@needs_figures
def test_specs_are_unique_on_label_and_filename():
    """The builder rejects duplicates; catching them here names the offender."""
    specs = build_specs()
    labels = [spec.label for spec in specs]
    filenames = [spec.filename for spec in specs]
    assert len(set(labels)) == len(labels), "duplicate figure label"
    assert len(set(filenames)) == len(filenames), "duplicate figure filename"


def test_figure_spec_satisfies_the_template_protocol_structurally():
    """The dataclass carries the fields `FigureSpecLike` reads.

    Checked without importing the protocol, so this passes in a bare checkout.
    That is the point of the dataclass not inheriting from it. ``alt_text`` is
    optional in the protocol and required here: the template's builder skips the
    field when it is absent, and the renderer then fails closed on every figure,
    so this project treats it as mandatory rather than discovering the omission
    at the HTML stage.
    """
    spec = FigureSpec(label="fig:x", filename="x.png", caption="c", generated_by="m", alt_text="a")
    for field in ("label", "filename", "caption", "generated_by", "alt_text"):
        assert isinstance(getattr(spec, field), str)


@pytest.mark.parametrize(
    ("content", "message"),
    (
        ('{"schema":"one","schema":"two"}\n', "duplicate JSON key"),
        ('{"schema":NaN}\n', "non-finite JSON constant"),
    ),
)
def test_registry_rejects_ambiguous_or_nonfinite_provenance_json(
    tmp_path: Path, content: str, message: str
) -> None:
    """NEGATIVE CONTROL: the registry uses the generator's strict JSON contract."""
    receipt = tmp_path / "output" / "figures" / "figure_provenance.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_provenance(tmp_path)


def test_registry_rejects_an_unexpected_top_level_receipt_field(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: an extensible parser must not silently weaken v3."""
    receipt = tmp_path / "output" / "figures" / "figure_provenance.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        '{"schema":"ailm-figure-provenance-v3","unexpected":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="is not ailm-figure-provenance-v3"):
        load_provenance(tmp_path)


@pytest.mark.parametrize("kind", ("symlink", "directory"))
def test_registry_rejects_nonordinary_provenance_paths(tmp_path: Path, kind: str) -> None:
    """NEGATIVE CONTROL: a named path is not evidence that regular bytes were read."""
    receipt = tmp_path / "output" / "figures" / "figure_provenance.json"
    receipt.parent.mkdir(parents=True)
    if kind == "symlink":
        real = tmp_path / "real-receipt.json"
        real.write_text("{}\n", encoding="utf-8")
        receipt.symlink_to(real)
    else:
        receipt.mkdir()

    with pytest.raises(ValueError, match="regular"):
        load_provenance(tmp_path)


@pytest.mark.timeout(30)
def test_registry_rejects_a_fifo_receipt_in_a_bounded_real_process(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: a FIFO cannot block registry publication indefinitely."""
    receipt = tmp_path / "output" / "figures" / "figure_provenance.json"
    receipt.parent.mkdir(parents=True)
    os.mkfifo(receipt)
    script = (
        "from pathlib import Path; "
        "from scripts.generate_figure_registry import load_provenance; "
        "load_provenance(Path(__import__('sys').argv[1]))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "must be a regular file" in result.stderr
    assert receipt.is_fifo()


# ── The published artifact ───────────────────────────────────────────────────


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not published in this checkout")
def test_the_published_registry_matches_the_current_specs():
    """What is on disk must be what the current sources would produce."""
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION

    published = {entry["label"]: entry for entry in payload["figures"]}
    for spec in build_specs():
        assert spec.label in published, (
            f"{spec.label} is built but not published. Run "
            f"`.venv/bin/python scripts/generate_figure_registry.py`."
        )
        entry = published[spec.label]
        # Every one of these fails the same way and has the same remedy, and the
        # caption one fires on any wording change to a figure. Saying so in the
        # message turns a raw diff of two long captions into a one-line fix; not
        # saying it cost three cycles in a single session.
        stale = (
            f"the published registry is stale for {spec.label}. Re-run "
            f"`.venv/bin/python scripts/generate_figure_registry.py` after any "
            f"edit to a figure's caption, filename or generator."
        )
        assert entry["filename"] == spec.filename, stale
        assert entry["generated_by"] == spec.generated_by, stale
        assert entry["caption"] == spec.caption, stale


@needs_figures
def test_registry_cli_uses_the_sidecar_figure_contract() -> None:
    """The template's unrelated ``scripts`` package cannot shadow this one."""
    template_root = resolve_template_root()
    if template_root is None:
        pytest.skip("template checkout absent")
    env = dict(os.environ)
    env["TEMPLATE_REPO_ROOT"] = str(template_root)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_figure_registry.py"), "--check"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr


# ── The import boundary ──────────────────────────────────────────────────────


def test_the_template_root_resolver_agrees_with_the_variables_script():
    """One answer to "where is the template", not two.

    Both scripts resolve it the same way and in the same order. Two resolvers
    that could disagree is a defect waiting for the day they do.
    """
    from scripts.z_generate_manuscript_variables import _resolve_template_root

    assert resolve_template_root() == _resolve_template_root()


def test_nothing_under_src_imports_infrastructure():
    """The architecture rule, enforced rather than documented.

    `src/ailm/` must run in a bare checkout with no template beside it. Scripts
    are the sanctioned boundary; the package is not. Written over the whole
    package so a new module is covered the moment it is added.
    """
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "ailm").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line, module in _infrastructure_imports(text):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{line}: imports {module}")
    assert not offenders, "src/ must never import infrastructure:\n" + "\n".join(offenders)

    # NEGATIVE CONTROL: aliases and submodules must fail; comments and strings
    # that merely document the package must not.
    planted = "import infrastructure.documentation as docs\n"
    assert _infrastructure_imports(planted) == [(1, "infrastructure.documentation")]
    harmless = "# import infrastructure\nmessage = 'from infrastructure import docs'\n"
    assert _infrastructure_imports(harmless) == []


def test_the_publisher_reports_a_missing_template_rather_than_crashing(tmp_path):
    """No template checkout is a clear message and exit 2, not a traceback.

    The figure pipeline does not need `infrastructure`; only this optional
    publishing step does, and a bare checkout must be told that plainly.
    """
    bare_repo = tmp_path / "bare-repo"
    scripts = bare_repo / "scripts"
    scripts.mkdir(parents=True)
    copied_script = scripts / "generate_figure_registry.py"
    shutil.copy2(REPO_ROOT / "scripts" / copied_script.name, copied_script)

    isolated_home = tmp_path / "home"
    isolated_home.mkdir()
    env = dict(os.environ)
    env.pop("TEMPLATE_REPO_ROOT", None)
    env["HOME"] = str(isolated_home)
    result = subprocess.run(
        [sys.executable, str(copied_script), "--check"],
        cwd=bare_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "TEMPLATE_REPO_ROOT" in result.stderr
    assert "Traceback" not in result.stderr
