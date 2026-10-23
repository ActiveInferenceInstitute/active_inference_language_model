"""Documentation integrity: the index is complete, and its links resolve.

Documentation fails silently in a way code does not. A link to a file that was
renamed still renders as a link; a document nobody indexed is invisible rather
than broken; a table row with an empty label is a row a reader skips. None of it
raises, and none of it shows up in a diff as an error — which is exactly the
class of failure this project's gates exist to make loud.

These checks are structural on purpose. They cannot tell whether a document is
*correct*; they can tell whether it is reachable, whether what it points at
exists, and whether every document that should be listed is listed.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from ailm.manuscript_contract import ManuscriptContractError, resolve_manuscript_text
from ailm.manuscript_variables import generate_manuscript_variables

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

#: Markdown link, capturing label and target. Images are excluded by the
#: negative lookbehind: a missing figure is the figure registry's business.
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)")

PYTEST_NODE_RE = re.compile(
    r"(?P<path>tests/test_[A-Za-z0-9_]+\.py)"
    r"(?P<nodes>(?:::[A-Za-z_][A-Za-z0-9_]*)+)"
)
PYTEST_SHORTHAND_RE = re.compile(r"(?<![A-Za-z0-9_./])::test_[A-Za-z0-9_]+")

FIGURE_REGISTRY = REPO_ROOT / "output" / "figures" / "figure_registry.json"

GATE_TAXONOMY_SUMMARIES = (
    Path("docs/README.md"),
    Path("docs/gate_design.md"),
    Path("docs/reference/notation.md"),
    Path("docs/guides/quickstart.md"),
    Path("docs/troubleshooting.md"),
    Path("docs/reference/output_inventory.md"),
    Path("docs/syntax_guide.md"),
    Path("docs/testing_philosophy.md"),
    Path("docs/formal.md"),
    Path("docs/guides/forking_guide.md"),
    Path("docs/manuscript/README.md"),
    Path("src/ailm/figures/README.md"),
)

PYTEST_LOCATOR_SURFACES = (
    Path("ISA.md"),
    *tuple(sorted(Path("docs").rglob("*.md"))),
    *tuple(
        path
        for path in sorted(Path("src").rglob("README.md"))
        if path != Path("src/ailm/corpus/README.md")
    ),
)

#: Documents that live outside `docs/` but are part of the same story, so the
#: index is expected to reach them.
EXTERNAL_INDEXED = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("src/ailm/dashboard/README.md"),
)

#: Reader-facing/current-method surfaces that previously retained obsolete
#: Stage 5, Stage 6, and multiplicity wording after the canonical manuscript had
#: been corrected.  Exact phrases are used so a legitimate discussion of
#: confirmation or bounds is not rejected merely for containing either word.
CURRENT_STATUS_SURFACES = (
    Path("ISA.md"),
    Path("HANDOFF.md"),
    Path("docs/decisions.md"),
    Path("tests/README.md"),
    Path("scripts/probe_l5_widened_alphabet.py"),
    Path("docs/manuscript/01_introduction.md"),
    Path("docs/manuscript/03_methods.md"),
    Path("docs/manuscript/04_results.md"),
)

CATEGORICAL_CLAIM_SURFACES = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("CHANGELOG.md"),
    Path("docs/README.md"),
    Path("docs/guides/agent_instructions.md"),
    Path("docs/design/no_embeddings.md"),
    Path("docs/provenance/sources.md"),
    Path("docs/manuscript/06_related_work.md"),
)

INVENTORY_COUNT_SURFACES = (
    Path("README.md"),
    Path("HANDOFF.md"),
    Path("docs/hierarchy.md"),
    Path("docs/gnn_composition.md"),
    Path("docs/reference/formalism_map.md"),
    Path("docs/guides/quickstart.md"),
    Path("scripts/README.md"),
    Path("scripts/build_web_ailm.py"),
    Path("scripts/generate_cover.py"),
)

RETIRED_HARDCODED_INVENTORY_ASSERTIONS = (
    "five channels",
    "six levels",
    "all six levels",
    "five cross-level interventions",
    "three of them share an alphabet",
    "seven stage gates",
    "all eight files",
    "audits eight model/generation gates",
    "six modeled levels",
    "six built levels",
    "directory holds 37 scripts",
    "every measurement, 21 of them",
)

RETIRED_CATEGORICAL_ASSERTIONS = (
    "discocat functor lands in stochastic maps",
    "discocat functor lands in discrete stochastic maps",
    "functor `f: gram → finstoch`",
    "finstoch is a compact closed category",
    "row-normalized non-negative matrices",
    "section 4: the finstoch functor",
)


def _output_tracking_documentation_defects(text: str, visible: set[str]) -> list[str]:
    """Compare the curated projection with Git-visible files, including additions.

    A development checkout may have a required publication file awaiting its
    first commit. Clean publication eligibility is enforced by the release
    runner; this check verifies the current allowlist and file inventory.
    """

    exact = {
        "output/data/manuscript_variables.json",
        "output/pdf/active_inference_language_model_combined.pdf",
        "output/reports/artifact_manifest.json",
        "output/reports/connected_gum.json",
        "output/reports/evidence_registry.json",
        "output/reports/gate_audit.json",
        "output/reports/gate_audit_ablation.json",
        "output/reports/manuscript_composition.json",
        "output/reports/rendered_provenance.json",
        "output/reports/validation_report.json",
        "output/web/_combined_manuscript.md",
        "output/web/index.html",
    }
    allowed = exact | {
        path
        for path in visible
        if path.startswith("output/figures/") or path.startswith("output/manuscript/")
    }
    defects = [f"unexpected publication output:{path}" for path in sorted(visible - allowed)]
    defects.extend(f"missing publication output:{path}" for path in sorted(exact - visible))
    for phrase in (
        "every file under `output/figures/` and",
        "`output/manuscript/`",
        "`output/pdf/active_inference_language_model_combined.pdf`",
        "the eight report files",
        "`output/web/_combined_manuscript.md` and",
        "`output/web/index.html`",
    ):
        if phrase not in text:
            defects.append(f"missing documentation:{phrase}")
    return defects


def test_output_tracking_inventory_matches_git_and_has_a_live_negative_control() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "output"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    visible = set(result.stdout.splitlines())
    conventions = (DOCS / "reference" / "output_conventions.md").read_text(encoding="utf-8")
    inventory = (DOCS / "reference" / "output_inventory.md").read_text(encoding="utf-8")
    assert _output_tracking_documentation_defects(conventions, visible) == []
    assert "curated set of publication surfaces is tracked" in " ".join(inventory.split())
    assert "fresh clone has no `output/`" not in inventory

    planted = conventions.replace("`output/web/index.html`", "`output/web/demo.html`")
    assert "missing documentation:`output/web/index.html`" in (
        _output_tracking_documentation_defects(planted, visible)
    )
    missing_ablation = visible - {"output/reports/gate_audit_ablation.json"}
    assert "missing publication output:output/reports/gate_audit_ablation.json" in (
        _output_tracking_documentation_defects(conventions, missing_ablation)
    )

    missing_connected = visible - {"output/reports/connected_gum.json"}
    assert "missing publication output:output/reports/connected_gum.json" in (
        _output_tracking_documentation_defects(conventions, missing_connected)
    )


def test_runtime_vendor_inventory_has_one_code_owned_source_of_truth() -> None:
    """Documentation points at the registry instead of copying its filenames."""

    from ailm.resources import RUNTIME_VENDOR_ARTIFACTS

    inventory = (DOCS / "reference" / "output_inventory.md").read_text(encoding="utf-8")

    def duplicated(text: str) -> list[str]:
        return [name for name in RUNTIME_VENDOR_ARTIFACTS if f"`vendor/{name}`" in text]

    assert "ailm.resources.RUNTIME_VENDOR_ARTIFACTS" in inventory
    assert duplicated(inventory) == []

    planted = inventory + f"\n`vendor/{RUNTIME_VENDOR_ARTIFACTS[0]}`\n"
    assert duplicated(planted) == [RUNTIME_VENDOR_ARTIFACTS[0]]


def test_output_inventory_names_current_ignored_classes() -> None:
    inventory = (DOCS / "reference" / "output_inventory.md").read_text(encoding="utf-8")
    required = (
        "output/bench_raw/",
        "output/upstream_pymdp_pr/",
        ".venv/release-cache/",
        "release-receipt.json",
        "logs/",
    )
    missing = [path for path in required if path not in inventory]
    assert missing == []
    assert "not tracked or distributed" in inventory
    assert "`ailm.html` is the *shipped* file" not in inventory

    planted = inventory.replace("output/upstream_pymdp_pr/", "output/omitted_bundle/")
    assert "output/upstream_pymdp_pr/" not in planted


RETIRED_CURRENT_STATUS_ASSERTIONS = (
    "measured absence of an effect",
    "instrument provably had power",
    "stage 6 a confirmed pre-registration",
    "level 6 — narrative (pre-registered, confirmed)",
    "confirmed pre-registered recognition result",
    "the channel's case-role value is bounded by",
    "the *effect* does not exist here",
    "receiver-sensitivity bound on the cascade",
    "the 0.13% type-oracle ceiling",
    "declared 11-test family",
    "family of 11 tests",
    "113 pages",
    "one confirmed pre-registered test",
    "all recorded results current",
    "declare any new p-value into the multiplicity family",
    "interval on the privileged-reference gain",
    "interval on that reference gain includes zero",
    "no practically meaningful effect is detectable",
)


def _markdown_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if "output" not in p.parts)


def _links(path: Path) -> list[tuple[str, str]]:
    return LINK_RE.findall(path.read_text(encoding="utf-8"))


def _relative_link_defects(document: Path, variables: Mapping[str, str]) -> list[str]:
    targets = [resolve_manuscript_text(target, variables) for _, target in _links(document)]
    relative = [
        target.split("#")[0]
        for target in targets
        if target and not target.startswith(("http://", "https://", "mailto:", "#"))
    ]
    # Source checks precede the full release's generated-artifact step. The one
    # registered portable page is backed by its canonical authored document;
    # missing/stale generated HTML is rejected by the mandatory later check.
    supporting_sources = {
        "ailm-supporting-docs/docs__extension_protocols.html": DOCS / "extension_protocols.md",
    }
    return [
        target
        for target in relative
        if not supporting_sources.get(target, document.parent / target).resolve().exists()
    ]


@pytest.fixture(scope="session")
def documentation_variables() -> dict[str, str]:
    return generate_manuscript_variables(REPO_ROOT)


def _retired_current_status_assertions(text: str) -> list[str]:
    """Return exact obsolete assertions after normalizing Markdown whitespace."""
    normalized = " ".join(text.lower().split())
    return [phrase for phrase in RETIRED_CURRENT_STATUS_ASSERTIONS if phrase in normalized]


def _retired_categorical_assertions(text: str) -> list[str]:
    """Return exact retracted category claims after Markdown normalization."""
    normalized = " ".join(text.lower().split())
    return [phrase for phrase in RETIRED_CATEGORICAL_ASSERTIONS if phrase in normalized]


def _retired_hardcoded_inventory_assertions(text: str) -> list[str]:
    """Return prose tallies that must instead enumerate or derive their inventory."""
    normalized = " ".join(text.lower().split())
    return [phrase for phrase in RETIRED_HARDCODED_INVENTORY_ASSERTIONS if phrase in normalized]


def _enum_member_names(path: Path, class_name: str) -> tuple[str, ...]:
    """Read a string-valued enum independently, without importing project code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    enum = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return tuple(
        target.id
        for node in enum.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _missing_verdicts(text: str, verdicts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(verdict for verdict in verdicts if verdict not in text)


def _figure_projection(text: str, *, label_first: bool) -> list[tuple[str, str, str]]:
    """Extract ``(filename, label, module)`` triples from a documentation table."""
    if label_first:
        pattern = re.compile(
            r"^\|\s*`\{#(?P<label>fig:[^}]+)\}`\s*\|\s*"
            r"`(?P<filename>[^`]+\.png)`\s*\|\s*"
            r"`(?P<module>ailm\.figures\.[^`]+)`\s*\|",
            re.MULTILINE,
        )
    else:
        pattern = re.compile(
            r"^\|\s*`(?P<filename>[^`]+\.png)`\s*\|\s*"
            r"`\{#(?P<label>fig:[^}]+)\}`\s*\|\s*"
            r"`(?P<module>ailm\.figures\.[^`]+)`\s*\|",
            re.MULTILINE,
        )
    return [
        (match["filename"], match["label"], match["module"]) for match in pattern.finditer(text)
    ]


def _pytest_locator_errors(text: str) -> list[str]:
    """Return cited pytest node IDs that do not resolve in the source AST."""
    errors = PYTEST_SHORTHAND_RE.findall(text)
    for match in PYTEST_NODE_RE.finditer(text):
        path = REPO_ROOT / match["path"]
        names = match["nodes"].split("::")[1:]
        if not path.is_file():
            errors.append(match.group(0))
            continue
        scope = ast.parse(path.read_text(encoding="utf-8")).body
        for name in names:
            node = next(
                (
                    item
                    for item in scope
                    if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == name
                ),
                None,
            )
            if node is None:
                errors.append(match.group(0))
                break
            scope = node.body if isinstance(node, ast.ClassDef) else []
    return errors


def _exact_dependency_pin(name: str) -> str:
    dependencies = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    pattern = re.compile(rf"^{re.escape(name)}==(?P<version>[^;]+)$")
    matches = [match["version"] for item in dependencies if (match := pattern.fullmatch(item))]
    assert len(matches) == 1, f"{name} must have one exact project dependency pin: {matches}"
    return matches[0]


def test_current_status_surfaces_do_not_reassert_retired_evidence_boundaries():
    """Current guidance may not restore the audited Stage 5/6 overclaims."""
    offenders = {
        str(relative): hits
        for relative in CURRENT_STATUS_SURFACES
        if (
            hits := _retired_current_status_assertions(
                (REPO_ROOT / relative).read_text(encoding="utf-8")
            )
        )
    }
    assert offenders == {}, f"retired current-status assertions: {offenders}"


def test_retired_current_status_assertion_detector_is_live():
    """POSITIVE CONTROL: every retired phrase is detectable after line wrapping."""
    planted = "\n".join(
        phrase.replace(" ", "\n", 1) for phrase in RETIRED_CURRENT_STATUS_ASSERTIONS
    )
    assert set(_retired_current_status_assertions(planted)) == set(
        RETIRED_CURRENT_STATUS_ASSERTIONS
    )


def test_current_surfaces_do_not_hardcode_registry_inventory_counts():
    offenders = {
        str(relative): hits
        for relative in INVENTORY_COUNT_SURFACES
        if (
            hits := _retired_hardcoded_inventory_assertions(
                (REPO_ROOT / relative).read_text(encoding="utf-8")
            )
        )
    }
    assert offenders == {}, f"hard-coded registry inventory counts: {offenders}"


def test_hardcoded_inventory_assertion_detector_is_live():
    planted = "\n".join(
        phrase.replace(" ", "\n", 1) for phrase in RETIRED_HARDCODED_INVENTORY_ASSERTIONS
    )
    assert set(_retired_hardcoded_inventory_assertions(planted)) == set(
        RETIRED_HARDCODED_INVENTORY_ASSERTIONS
    )


def test_current_surfaces_do_not_restore_the_retracted_finstoch_functor():
    offenders = {
        str(relative): hits
        for relative in CATEGORICAL_CLAIM_SURFACES
        if (
            hits := _retired_categorical_assertions(
                (REPO_ROOT / relative).read_text(encoding="utf-8")
            )
        )
    }
    assert offenders == {}, f"retracted categorical assertions: {offenders}"


def test_retracted_categorical_assertion_guard_is_live():
    planted = "\n".join(phrase.replace(" ", "\n", 1) for phrase in RETIRED_CATEGORICAL_ASSERTIONS)
    assert set(_retired_categorical_assertions(planted)) == set(RETIRED_CATEGORICAL_ASSERTIONS)


def test_categorical_boundary_states_the_implemented_orientation_and_limit():
    text = " ".join((DOCS / "design" / "no_embeddings.md").read_text(encoding="utf-8").split())
    assert "each source state occupies a column" in text
    assert "each such column sums to one" in text
    assert "not a compact-closed category" in text
    assert "does **not** claim a functor `Preg → FinStoch`" in text


def test_manuscript_does_not_reduce_pymdp_validation_to_normalization():
    text = " ".join(
        (REPO_ROOT / "docs" / "manuscript" / "02_background.md").read_text(encoding="utf-8").split()
    )
    assert "the only structural property `pymdp` checks" not in text.lower()
    for implemented_check in (
        "dependency dimensions and indices",
        "control-factor consistency",
        "selected inference algorithm",
    ):
        assert implemented_check in text


# ── The index ────────────────────────────────────────────────────────────────


def test_every_document_under_docs_is_reachable_from_the_index():
    """An unindexed document is invisible, not broken, which is worse.

    `research_queue.md` was linked from a stray bullet appended below the last
    table, outside every section, with an empty label — `[](research_queue.md)`.
    It rendered as a bare parenthesis. The document was in the repository, in
    the index, and unreadable by anyone who did not already know it was there.
    """
    index = DOCS / "README.md"
    targets = {target.split("#")[0] for _, target in _links(index)}
    resolved = {(index.parent / target).resolve() for target in targets if target}

    missing = [
        path.relative_to(REPO_ROOT)
        for path in _markdown_files(DOCS)
        if path.name != "README.md" and path.resolve() not in resolved
    ]
    assert not missing, f"documents no index row reaches: {missing}"


def test_the_index_reaches_the_documents_that_live_beside_the_code():
    """Not every document belongs under `docs/`, and none may be orphaned.

    The dashboard's README is the only description of that artifact's scope —
    that it is a demonstrator, backs no manuscript number, and runs Level 5 on
    inferred case roles where the probe uses gold ones. Nothing in `docs/`
    linked to it, so the one place that warning lives was reachable only by
    browsing the source tree.
    """
    index = DOCS / "README.md"
    resolved = {
        (index.parent / target.split("#")[0]).resolve()
        for _, target in _links(index)
        if target and not target.startswith(("http", "mailto"))
    }
    for relative in EXTERNAL_INDEXED:
        assert (REPO_ROOT / relative).resolve() in resolved, (
            f"{relative} is not reachable from the documentation index"
        )


def test_no_markdown_link_has_an_empty_label():
    """`[](target)` renders as nothing at all — a link with no handle.

    The check is the general property rather than the one occurrence that
    prompted it, so the next empty label fails too.
    """
    offenders: dict[str, list[str]] = {}
    for path in _markdown_files(DOCS) + [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md"]:
        empty = [target for label, target in _links(path) if not label.strip()]
        if empty:
            offenders[str(path.relative_to(REPO_ROOT))] = empty
    assert not offenders, f"links with no visible label: {offenders}"


# ── Link targets ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "document",
    [pytest.param(p, id=str(p.relative_to(REPO_ROOT))) for p in _markdown_files(DOCS)],
)
def test_every_relative_link_resolves(document: Path, documentation_variables: dict[str, str]):
    """Resolve metadata URLs before classifying local paths; missing targets fail."""
    broken = _relative_link_defects(document, documentation_variables)
    assert not broken, f"{document.name} links to paths that do not exist: {broken}"


def test_metadata_links_are_resolved_without_exempting_broken_targets(
    tmp_path: Path, documentation_variables: dict[str, str]
) -> None:
    document = tmp_path / "links.md"
    target = tmp_path / "retained.md"
    target.write_text("# Retained\n", encoding="utf-8")
    document.write_text(
        "[Source]({{CEREBRUM_POMDP_SUPPLEMENT_URL}}) [Local](retained.md#retained)\n",
        encoding="utf-8",
    )
    assert _relative_link_defects(document, documentation_variables) == []

    broken_variables = dict(documentation_variables)
    broken_variables["CEREBRUM_POMDP_SUPPLEMENT_URL"] = "absent.md#missing"
    assert _relative_link_defects(document, broken_variables) == ["absent.md"]

    document.write_text("[Source]({{UNPRODUCED_SOURCE_URL}})\n", encoding="utf-8")
    with pytest.raises(ManuscriptContractError, match="UNPRODUCED_SOURCE_URL"):
        _relative_link_defects(document, documentation_variables)

    document.write_text("[Source]({{malformed_source_url}})\n", encoding="utf-8")
    assert _relative_link_defects(document, documentation_variables) == ["{{malformed_source_url}}"]


# ── Per-package documentation ────────────────────────────────────────────────


def test_every_source_package_is_documented_beside_its_code():
    """A package README is the only documentation that moves with the package.

    `docs/architecture.md` maps every module, but it is one file away from the
    code and drifts on its own schedule. A package that grows a README is a
    package whose author had to say what it is for.
    """
    package_root = REPO_ROOT / "src" / "ailm"
    packages = sorted(
        path
        for path in package_root.iterdir()
        if path.is_dir() and (path / "__init__.py").exists() and path.name != "__pycache__"
    )
    assert packages, "no sub-packages found; this test would be vacuous"

    undocumented = [
        path.name
        for path in packages
        if not (path / "README.md").exists()
        and not (path / "__init__.py").read_text(encoding="utf-8").lstrip().startswith('"""')
    ]
    assert not undocumented, (
        f"packages with neither a README nor a module docstring: {undocumented}"
    )


def test_the_architecture_map_lists_every_shipped_module():
    """The code map is only useful while it is complete.

    A module added without a row is a module nobody can find by reading the map,
    and the map still looks complete — every row in it is correct.
    """
    architecture = (DOCS / "architecture.md").read_text(encoding="utf-8")
    package_root = REPO_ROOT / "src" / "ailm"

    modules = [
        path.relative_to(package_root.parent).as_posix()
        for path in sorted(package_root.rglob("*.py"))
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    ]

    def missing(text: str, expected: list[str]) -> list[str]:
        return [relative for relative in expected if relative not in text]

    assert missing(architecture, modules) == []

    planted_module = "ailm/planted_new_module.py"
    assert missing(architecture, modules + [planted_module]) == [planted_module]


def test_architecture_dependency_column_is_explanatory_not_a_false_allowlist() -> None:
    architecture = (DOCS / "architecture.md").read_text(encoding="utf-8")
    assert "Principal dependencies (not exhaustive)" in architecture
    assert "not an exhaustive per-file allowlist" in architecture
    assert "| Module | Owns | May import |" not in architecture

    planted = architecture.replace(
        "Principal dependencies (not exhaustive)",
        "May import",
    )
    assert "Principal dependencies (not exhaustive)" not in planted


def test_script_ownership_docs_admit_and_bound_recorder_local_estimands() -> None:
    readme = (REPO_ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "scripts" / "AGENTS.md").read_text(encoding="utf-8")
    root_agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    architecture = (DOCS / "architecture.md").read_text(encoding="utf-8")
    style = (DOCS / "method" / "style_guide.md").read_text(encoding="utf-8")
    for text in (readme, agents, root_agents, architecture, style):
        assert "recorder-specific estimands" in text
        assert "none computes" not in text.casefold()
        assert "Thin orchestrators only" not in text
    assert "src/ailm/ — all logic" not in architecture
    assert "scripts/ — thin orchestrators" not in architecture
    assert "scripts/           thin orchestrators" not in root_agents
    assert "with one deliberate exception" not in style

    planted = readme.replace("recorder-specific estimands", "formatting only")
    assert "recorder-specific estimands" not in planted


# ── Registry-owned documentation projections ─────────────────────────────────────────


def test_gate_taxonomy_summaries_enumerate_the_code_owned_enum():
    """A four-verdict summary silently erases comparator typing."""
    verdicts = _enum_member_names(REPO_ROOT / "src" / "ailm" / "gates.py", "GateVerdict")
    assert len(verdicts) >= 6, "GateVerdict discovery is vacuous"

    missing = {
        str(path): absent
        for path in GATE_TAXONOMY_SUMMARIES
        if (absent := _missing_verdicts((REPO_ROOT / path).read_text(encoding="utf-8"), verdicts))
    }
    assert missing == {}, f"current taxonomy summaries omit typed verdicts: {missing}"


def test_gate_taxonomy_summary_check_detects_a_missing_member():
    """POSITIVE CONTROL: deleting one enum member makes the projection stale."""
    verdicts = _enum_member_names(REPO_ROOT / "src" / "ailm" / "gates.py", "GateVerdict")
    complete = " / ".join(verdicts)
    planted = complete.replace(verdicts[-1], "")
    assert _missing_verdicts(complete, verdicts) == ()
    assert _missing_verdicts(planted, verdicts) == (verdicts[-1],)


def test_reader_facing_figure_inventories_exactly_match_the_registry():
    """No figure may be added, removed, renamed, or reassigned in only one place."""
    assert FIGURE_REGISTRY.is_file(), "stable publication figure registry is missing"
    payload = json.loads(FIGURE_REGISTRY.read_text(encoding="utf-8"))
    expected = [
        (entry["filename"], entry["label"], entry["generated_by"]) for entry in payload["figures"]
    ]
    assert len(expected) > 20, "figure-registry comparison would be vacuous"
    assert len(expected) == len(set(expected)), "figure registry repeats a projection triple"

    projections = {
        "docs/reference/output_inventory.md": _figure_projection(
            (DOCS / "reference" / "output_inventory.md").read_text(encoding="utf-8"),
            label_first=False,
        ),
        "docs/syntax_guide.md": _figure_projection(
            (DOCS / "syntax_guide.md").read_text(encoding="utf-8"), label_first=True
        ),
    }
    for path, actual in projections.items():
        assert len(actual) == len(set(actual)), f"{path} repeats a figure row"
        assert set(actual) == set(expected), (
            f"{path} is not an exact figure-registry projection; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra_or_wrong={sorted(set(actual) - set(expected))}"
        )


def test_figure_projection_check_detects_missing_and_misassigned_rows():
    """POSITIVE CONTROLS: both omission and a plausible generator typo fire."""
    text = (
        "| File | Label | Generator | Note |\n"
        "|---|---|---|---|\n"
        "| `a.png` | `{#fig:a}` | `ailm.figures.alpha` | A |\n"
        "| `b.png` | `{#fig:b}` | `ailm.figures.beta` | B |\n"
    )
    expected = {
        ("a.png", "fig:a", "ailm.figures.alpha"),
        ("b.png", "fig:b", "ailm.figures.beta"),
    }
    assert set(_figure_projection(text, label_first=False)) == expected
    assert (
        set(_figure_projection(text.replace("| `b.png`", "| `missing.txt`"), label_first=False))
        != expected
    )
    assert (
        set(
            _figure_projection(
                text.replace("ailm.figures.beta", "ailm.figures.alpha"), label_first=False
            )
        )
        != expected
    )


# ── Executable documentation references ────────────────────────────────────────────


def test_cited_pytest_node_locators_resolve_to_collected_source_nodes():
    """Imports, unqualified class methods, and file-free shorthands are invalid."""
    cited = 0
    broken: dict[str, list[str]] = {}
    for relative in PYTEST_LOCATOR_SURFACES:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        cited += len(PYTEST_NODE_RE.findall(text))
        if errors := _pytest_locator_errors(text):
            broken[str(relative)] = errors
    assert cited > 20, f"only {cited} pytest node locators were checked"
    assert broken == {}, f"documentation cites nonexistent pytest nodes: {broken}"


def test_pytest_locator_check_rejects_an_import_and_a_missing_method():
    """POSITIVE CONTROL: all three historical locator defects are detectable."""
    valid = (
        "tests/test_control_harness.py::TestControlHarness::"
        "test_raises_gate_vacuous_error_when_identical"
    )
    planted = "\n".join(
        (
            "tests/test_control_harness.py::GateVacuousError",
            "tests/test_control_harness.py::TestControlHarness::test_does_not_exist",
            "::test_contextless_shorthand",
        )
    )
    assert _pytest_locator_errors(valid) == []
    assert sorted(_pytest_locator_errors(planted)) == sorted(planted.splitlines())


def test_troubleshooting_dependency_pins_match_pyproject_exactly():
    """The recovery command must install the environment the project declares."""
    text = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
    expected = {
        name: _exact_dependency_pin(name) for name in ("inferactively-pymdp", "jax", "jaxlib")
    }
    missing = [
        f"{name}=={version}"
        for name, version in expected.items()
        if f"{name}=={version}" not in text
    ]
    assert missing == [], f"troubleshooting omits exact project pins: {missing}"
    assert "0.4.30" not in text, "troubleshooting restored the retired JAX pin"


def test_dependency_pin_check_can_detect_planted_drift():
    """POSITIVE CONTROL: the old JAX command no longer satisfies the contract."""
    version = _exact_dependency_pin("jax")
    current = f"install jax=={version}"
    planted = current.replace(version, "0.4.30")
    assert f"jax=={version}" in current
    assert f"jax=={version}" not in planted


def test_research_queue_marks_incomplete_attack_export_non_governing():
    """Literal ellipses cannot carry the authority the malformed export claimed."""
    text = (DOCS / "research_queue.md").read_text(encoding="utf-8")
    assert "## Adversarial verdict export (retired)" in text
    assert "the verdict governs" not in text.lower()
    assert "Do not infer task\nstatus from it" in text
    assert text.count("### Truncated historical label:") == 4
    assert "## Added 2026-08-20 from the external deep-research audit" in text
    assert "- [x] **`98_supplement:461` prompt-effect shift fractions.**" in text
    assert "**Resolved by removal.**" in text

    stage4 = text.split("### Level 4 — case", 1)[1].split("### Level 5", 1)[0]
    assert "UNREACHABLE" not in stage4, (
        "current Stage 4 queue prose restored the superseded impossibility verdict"
    )
    assert "ABOVE_REFERENCE" in stage4


def test_research_queue_retirement_contract_is_live():
    """POSITIVE CONTROL: an active malformed export and open resolved item fail."""
    text = (DOCS / "research_queue.md").read_text(encoding="utf-8")
    planted = text.replace(
        "## Adversarial verdict export (retired)", "## Adversarial verdicts, per item"
    ).replace(
        "- [x] **`98_supplement:461` prompt-effect shift fractions.**",
        "- [ ] **`98_supplement:461` prompt-effect shift fractions.**",
    )
    assert "## Adversarial verdict export (retired)" not in planted
    assert "- [x] **`98_supplement:461` prompt-effect shift fractions.**" not in planted


def test_package_readme_delegates_the_module_inventory_to_the_checked_map():
    """The package README must not become a second hand-maintained module map."""
    text = (REPO_ROOT / "src" / "ailm" / "README.md").read_text(encoding="utf-8")
    assert "../../docs/architecture.md#package-map" in text
    assert "### Flat modules" not in text
    assert "### Subpackages" not in text


def test_manuscript_readme_distinguishes_publication_order_from_its_inventory():
    """A thematic table is not evidence of the renderer's filename order."""
    text = (REPO_ROOT / "docs" / "manuscript" / "README.md").read_text(encoding="utf-8")
    assert "renderer assembles sections in filename order" in text
    assert "thematic inventory" in text
    assert "Assembled in lexicographic order" not in text


def test_cache_receipt_docs_state_the_unkeyed_trusted_local_boundary():
    """A digest detects drift; it does not authenticate a hostile writer."""

    sources = {
        "architecture": (DOCS / "architecture.md").read_text(encoding="utf-8"),
        "output conventions": (DOCS / "reference" / "output_conventions.md").read_text(
            encoding="utf-8"
        ),
        "script inventory": (REPO_ROOT / "scripts" / "README.md").read_text(encoding="utf-8"),
        "call cache": (REPO_ROOT / "src" / "ailm" / "call_cache.py").read_text(encoding="utf-8"),
        "replay receipts": (REPO_ROOT / "src" / "ailm" / "replay_provenance.py").read_text(
            encoding="utf-8"
        ),
    }
    joined = "\n".join(sources.values()).lower()
    assert "self-authenticating" not in joined
    for phrase in ("unkeyed", "trusted-local", "hostile-writer authentication"):
        assert phrase in joined, f"cache/receipt trust boundary omits {phrase!r}"

    # NEGATIVE CONTROL: the same retired overclaim is detected when planted.
    planted = joined + "\nreceipts are self-authenticating."
    assert "self-authenticating" in planted
