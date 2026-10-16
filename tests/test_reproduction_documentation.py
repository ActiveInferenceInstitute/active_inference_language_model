"""Executable contracts for the documented clean reproduction order.

These checks are deliberately independent of the release runner.  A correct
orchestrator does not make a stale copy-paste command safe, and a documentation
check that cannot reject a planted bad ordering or status-masking pipeline proves
nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_provenance_checksums import DOCUMENTED_ARTIFACTS

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A ``generate_*counts.py`` reference as the reproduction table writes it: the
#: full script name, or the ``..._<rest>_counts.py`` abbreviation the table uses
#: after the first full name in a row.
COUNTS_GENERATOR_TOKEN = re.compile(r"(?:generate|\.\.\.)(_[a-z_]*counts)\.py")

REPRODUCTION_SURFACES = (
    Path("HANDOFF.md"),
    Path("docs/guides/quickstart.md"),
    Path("docs/guides/agent_instructions.md"),
    Path("docs/reference/formalism_map.md"),
    Path("docs/reference/output_conventions.md"),
    Path("docs/syntax_guide.md"),
    Path("docs/troubleshooting.md"),
    Path("docs/manuscript/07_reproducibility.md"),
    Path("docs/manuscript/AGENTS.md"),
    Path("docs/manuscript/README.md"),
    Path("scripts/AGENTS.md"),
    Path("tests/AGENTS.md"),
    Path("src/ailm/figures/AGENTS.md"),
    Path("src/ailm/figures/README.md"),
    Path("docs/guides/forking_guide.md"),
    Path("tests/README.md"),
)

STATUS_MASKING_PATTERNS = (
    re.compile(r"2>&1\s*\|"),
    re.compile(r";\s*echo\b"),
    re.compile(r"&&\s*echo\b"),
    re.compile(r"\|\|\s*echo\b"),
    re.compile(r"\|\s*(?:grep|rg|tee|head|tail)\b"),
)

LEAN_CONTROL_NODE = (
    ".venv/bin/python -m pytest "
    "tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile "
    "-v --no-cov"
)
LEAN_CONTROL_SURFACES = (
    Path("HANDOFF.md"),
    Path("docs/reference/output_conventions.md"),
    Path("docs/formal.md"),
    Path("docs/manuscript/07_reproducibility.md"),
    Path("formal/README.md"),
    Path("formal/Controls/MustFail.lean"),
    Path("scripts/README.md"),
)
LEAN_PLACEHOLDER_COMMAND = (
    ".venv/bin/python -m pytest "
    "tests/test_formal_isomorphism.py::test_no_incomplete_proofs "
    "tests/test_formal_isomorphism.py::test_the_placeholder_scan_covers_a_non_empty_file_set "
    "-v --no-cov"
)
LEAN_PLACEHOLDER_SURFACES = (Path("docs/formal.md"), Path("formal/README.md"))
RAW_LEAN_CONTROL = re.compile(r"\blake\b[^\n]*\benv\s+lean\b[^\n]*\bMustFail\.lean\b")


def _read(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _ordering_defect(text: str, before: str, after: str) -> str | None:
    """Return a useful defect when either command is absent or reversed."""
    if before not in text:
        return f"missing prerequisite: {before}"
    if after not in text:
        return f"missing consumer: {after}"
    if text.index(before) > text.index(after):
        return f"consumer appears before prerequisite: {after}"
    return None


def _status_masking_defects(text: str) -> tuple[str, ...]:
    return tuple(pattern.pattern for pattern in STATUS_MASKING_PATTERNS if pattern.search(text))


def _raw_lean_control_lines(text: str) -> tuple[str, ...]:
    """Return raw Lean mutation runs, including shell-negated invocations."""

    return tuple(line.strip() for line in text.splitlines() if RAW_LEAN_CONTROL.search(line))


def test_manuscript_variables_precede_every_documented_figure_workflow() -> None:
    cases = {
        Path("docs/guides/quickstart.md"): (
            "## 5. Hydrate the manuscript variables",
            "## 6. Generate the figures",
        ),
        Path("docs/manuscript/07_reproducibility.md"): (
            "| Manuscript variables |",
            "| Figures |",
        ),
        Path("docs/manuscript/AGENTS.md"): (
            "`.venv/bin/python scripts/z_generate_manuscript_variables.py`",
            "`MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py`",
        ),
        Path("scripts/AGENTS.md"): (
            ".venv/bin/python scripts/z_generate_manuscript_variables.py\n",
            "MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py\n",
        ),
        Path("src/ailm/figures/AGENTS.md"): (
            ".venv/bin/python scripts/z_generate_manuscript_variables.py --print-only\n",
            "MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py\n",
        ),
        Path("src/ailm/figures/README.md"): (
            ".venv/bin/python scripts/z_generate_manuscript_variables.py --print-only\n",
            "MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py\n",
        ),
        Path("docs/guides/forking_guide.md"): (
            ".venv/bin/python scripts/z_generate_manuscript_variables.py --print-only",
            "MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --list",
        ),
        Path("docs/manuscript/README.md"): (
            ".venv/bin/python scripts/z_generate_manuscript_variables.py",
            "MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py",
        ),
        Path("HANDOFF.md"): (
            ".venv/bin/python scripts/z_generate_manuscript_variables.py",
            "MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py",
        ),
    }
    defects = {
        str(path): defect
        for path, (before, after) in cases.items()
        if (defect := _ordering_defect(_read(path), before, after)) is not None
    }
    assert defects == {}

    planted = "generate figures\nhydrate variables\n"
    assert _ordering_defect(planted, "hydrate variables", "generate figures") is not None


def test_clean_suite_docs_materialize_web_artifacts_before_pytest() -> None:
    cases = {
        Path("docs/guides/quickstart.md"): (
            ".venv/bin/python scripts/export_web_model.py",
            ".venv/bin/python scripts/build_web_ailm.py",
            ".venv/bin/python -m pytest tests/ -v",
        ),
        Path("tests/AGENTS.md"): (
            ".venv/bin/python scripts/export_web_model.py",
            ".venv/bin/python scripts/build_web_ailm.py",
            ".venv/bin/python -m pytest tests/ --cov=ailm",
        ),
        Path("tests/README.md"): (
            ".venv/bin/python scripts/export_web_model.py",
            ".venv/bin/python scripts/build_web_ailm.py",
            ".venv/bin/python -m pytest tests/ --cov=ailm",
        ),
        Path("docs/manuscript/07_reproducibility.md"): (
            "| Export the web model |",
            "| Build the web page |",
            ".venv/bin/python -m pytest tests/ --cov=ailm",
        ),
        Path("docs/guides/agent_instructions.md"): (
            ".venv/bin/python scripts/export_web_model.py",
            ".venv/bin/python scripts/build_web_ailm.py",
            ".venv/bin/python -m pytest tests/ -q",
        ),
        Path("README.md"): (
            ".venv/bin/python scripts/export_web_model.py",
            ".venv/bin/python scripts/build_web_ailm.py",
            ".venv/bin/python -m pytest tests/ -v --cov=ailm",
        ),
        Path("AGENTS.md"): (
            ".venv/bin/python scripts/export_web_model.py",
            ".venv/bin/python scripts/build_web_ailm.py",
            ".venv/bin/python -m pytest tests/ -v --tb=short",
        ),
        Path("HANDOFF.md"): (
            ".venv/bin/python scripts/export_web_model.py",
            ".venv/bin/python scripts/build_web_ailm.py",
            ".venv/bin/python -m pytest tests/ -q",
        ),
    }
    defects: dict[str, str] = {}
    for path, (export, build, suite) in cases.items():
        text = _read(path)
        defect = _ordering_defect(text, export, build) or _ordering_defect(text, build, suite)
        if defect is not None:
            defects[str(path)] = defect
    assert defects == {}

    planted = "build web\nexport model\nrun suite\n"
    assert _ordering_defect(planted, "export model", "build web") is not None


def test_every_claim_phase_and_status_assignment_precede_manuscript_consumers() -> None:
    reproduction = _read(Path("docs/manuscript/07_reproducibility.md"))
    ordered = (
        "orchestrate_claim_recorders.py core",
        "orchestrate_claim_recorders.py network",
        "orchestrate_claim_recorders.py aggregate",
        "orchestrate_claim_recorders.py formal",
        "scripts/assign_claim_status.py",
        "scripts/z_generate_manuscript_variables.py",
        "scripts/z_render_docs.py",
        "scripts/generate_figures.py",
        "scripts/generate_figure_registry.py",
        "scripts/generate_cover.py",
        "scripts/export_web_model.py",
        "scripts/build_web_ailm.py",
        "scripts/pipeline/stage_03_render.py",
        "scripts/pipeline/stage_04_validate.py",
        ".venv/bin/python -m pytest tests/ --cov=ailm",
        "scripts/verify_release.py full",
    )
    for before, after in zip(ordered, ordered[1:], strict=False):
        assert _ordering_defect(reproduction, before, after) is None
        # Every declared edge must reject a prerequisite moved after its
        # consumer, including the rendered-artifact requirements of the suite.
        moved = reproduction.replace(before, "withheld-prerequisite") + f"\n{before}\n"
        assert _ordering_defect(moved, before, after) is not None

    planted = (
        reproduction.replace(
            "orchestrate_claim_recorders.py network",
            "orchestrate_claim_recorders.py deferred-network",
        )
        + "\norchestrate_claim_recorders.py network\n"
    )
    assert (
        _ordering_defect(
            planted,
            "orchestrate_claim_recorders.py network",
            "scripts/z_generate_manuscript_variables.py",
        )
        is not None
    )


def test_reproduction_prose_names_the_network_phase_without_stale_row_position() -> None:
    reproduction = _read(Path("docs/manuscript/07_reproducibility.md"))
    scripts_readme = _read(Path("scripts/README.md"))
    prose = " ".join(reproduction.split())
    scripts_prose = " ".join(scripts_readme.split())

    assert "the network claim phase" in prose
    assert "That network phase" in prose
    assert "the last row" not in prose
    assert "That last row" not in prose
    assert "topologically orders the declared producer→consumer edges" in scripts_prose
    assert "the only constraint inside step 2" not in scripts_prose

    planted = prose.replace("the network claim phase", "the last row", 1)
    assert "the last row" in planted


def test_complete_reproduction_names_every_corpus_artifact_generator() -> None:
    reproduction = _read(Path("docs/manuscript/07_reproducibility.md"))
    generators = sorted(path.name for path in (REPO_ROOT / "scripts").glob("generate_*counts.py"))
    generators.append("generate_heldout_stream.py")
    documented = {
        name
        for name in generators
        if name in reproduction or f"...{name.removeprefix('generate')}" in reproduction
    }
    missing = [name for name in generators if name not in documented]
    assert missing == []

    planted_name = generators[0]
    planted_token = (
        planted_name
        if planted_name in reproduction
        else f"...{planted_name.removeprefix('generate')}"
    )
    planted = reproduction.replace(planted_token, "..._omitted_counts.py")
    assert planted_token not in planted


def _named_counts_generators(text: str) -> tuple[str, ...]:
    """Every ``generate_*counts.py`` the text names, in full or abbreviated form."""
    return tuple(f"generate{match.group(1)}.py" for match in COUNTS_GENERATOR_TOKEN.finditer(text))


#: A provenance section documenting a ``data/``-resident artifact: the heading
#: names the directory and the body binds at least one artifact digest.
_DATA_SECTION = re.compile(r"^### .*`(data/[a-z0-9_]+/)`.*$", re.M)
_DATA_DIGEST = re.compile(r"\*\*Artifact sha256 \(`data/[^`]+`\):\*\*")


def _data_resident_documented_generators() -> set[str]:
    """Counts generators that produce a digest-bound artifact under ``data/``.

    ``DOCUMENTED_ARTIFACTS`` is vendor-only by design, because ``vendor/`` is the
    packaged surface. Development-session arms (the widened clause artifact,
    decision 112) and the held-back confirmatory population (decision 111) are
    deliberately not packaged and live under ``data/``; their generators are
    documented in ``docs/provenance/sources.md`` in sections whose heading
    names the ``data/`` directory and whose body binds each artifact's digest.
    A generator named there counts as documented; one named in the reproduction
    table without such a section does not.
    """
    text = _read(Path("docs/provenance/sources.md"))
    documented: set[str] = set()
    starts = [m.start() for m in _DATA_SECTION.finditer(text)]
    for start in starts:
        end = text.find("\n### ", start + 1)
        section = text[start : end if end != -1 else len(text)]
        if not _DATA_DIGEST.search(section):
            continue
        documented.update(_named_counts_generators(section))
    return documented


def test_every_named_counts_generator_is_the_documented_generator_of_a_vendored_artifact() -> None:
    """A regenerator the table names must be the generator of something vendored.

    The reproduction table once listed a ``generate_*counts.py`` that no vendored
    artifact named as its generator: it wrote a file nothing consumed, and the row
    presented it as a byte-identical regeneration of the inputs. The guard is
    written against the class rather than the removed script, so it rejects any
    named generator that ``DOCUMENTED_ARTIFACTS`` does not know.
    """
    reproduction = _read(Path("docs/manuscript/07_reproducibility.md"))
    documented = {Path(script).name for script in DOCUMENTED_ARTIFACTS.values()}
    documented |= _data_resident_documented_generators()
    named = _named_counts_generators(reproduction)
    assert named, "the reproduction table names no counts generator at all"
    undocumented = sorted(set(named) - documented)
    assert undocumented == [], (
        "the reproduction table names generators that produce no documented vendored "
        f"artifact: {undocumented}"
    )

    planted = reproduction.replace(
        "`..._phoneme_counts.py`", "`..._phoneme_counts.py`, `..._orphan_counts.py`", 1
    )
    assert planted != reproduction, "the planted control found no row to plant into"
    assert "generate_orphan_counts.py" in _named_counts_generators(planted)
    assert "generate_orphan_counts.py" not in documented
    # The data-resident route is not a loophole: it admits only generators named
    # in a digest-bound data/ section. The two arms that live there are admitted
    # (a section may also name the vendored generator it imports from), and a
    # planted orphan is not.
    data_resident = _data_resident_documented_generators()
    assert {
        "generate_clause_widened_counts.py",
        "generate_confirmatory_ewt_counts.py",
    } <= data_resident
    assert "generate_orphan_counts.py" not in data_resident


def test_reproduction_commands_do_not_mask_upstream_exit_status() -> None:
    defects = {
        str(path): _status_masking_defects(_read(path))
        for path in REPRODUCTION_SURFACES
        if _status_masking_defects(_read(path))
    }
    assert defects == {}

    planted = "generator 2>&1 | grep -v warning\naudit; echo $?\n"
    assert _status_masking_defects(planted)


def test_documented_lean_mutation_control_validates_expected_diagnostics() -> None:
    missing_node = {
        str(path): LEAN_CONTROL_NODE not in _read(path) for path in LEAN_CONTROL_SURFACES
    }
    assert not any(missing_node.values()), missing_node

    raw_invocations = {
        str(path): lines
        for path in LEAN_CONTROL_SURFACES
        if (lines := _raw_lean_control_lines(_read(path)))
    }
    assert raw_invocations == {}

    direct = "cd formal && lake build && lake env lean Controls/MustFail.lean"
    negated = "! (cd formal && lake env lean Controls/MustFail.lean)"
    assert _raw_lean_control_lines(direct) == (direct,)
    assert _raw_lean_control_lines(negated) == (negated,)
    assert _raw_lean_control_lines(LEAN_CONTROL_NODE) == ()


def test_formal_placeholder_scan_is_non_vacuous_on_every_current_surface() -> None:
    for path in LEAN_PLACEHOLDER_SURFACES:
        assert LEAN_PLACEHOLDER_COMMAND in _read(path), path

    planted = LEAN_PLACEHOLDER_COMMAND.replace(
        "test_the_placeholder_scan_covers_a_non_empty_file_set",
        "test_planted_no_inventory_control",
    )
    assert LEAN_PLACEHOLDER_COMMAND not in planted


def test_quickstart_executes_tools_and_treats_skip_inventory_as_dynamic() -> None:
    text = _read(Path("docs/guides/quickstart.md"))
    assert "`mmdc --version`" in text
    assert "which mmdc" not in text
    assert "cd formal && lake --version" in text
    assert "do not trust a fixed census" in text
    assert "A clean run reports **2 skipped**" not in text
    assert "ailm.resources.RUNTIME_VENDOR_ARTIFACTS" in text

    planted = text.replace("`mmdc --version`", "`which mmdc`")
    assert "`mmdc --version`" not in planted


def test_split_and_stage5_guidance_name_the_independent_units() -> None:
    guide = _read(Path("docs/guides/forking_guide.md"))
    sampling = "independent sampling unit before the assignment rule"
    assignment = "deterministic stable\n  hash or a unit-level stratification"
    assert _ordering_defect(guide, sampling, assignment) is None
    assert "positional** split only when an explicit leakage analysis" in guide

    reproduction = _read(Path("docs/manuscript/07_reproducibility.md"))
    tests_readme = _read(Path("tests/README.md"))
    assert "paired whole-document cluster-bootstrap intervals" in reproduction
    assert "paired whole-document cluster-bootstrap gain intervals" in tests_readme

    planted = guide.replace("independent sampling unit", "individual observation")
    assert sampling not in planted
