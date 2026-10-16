"""Every label resolves, and every table and figure is pointed at by a sentence.

Two failure modes, both silent in a Pandoc build. A reference to a label that
does not exist renders as a literal ``[@tbl:missing]`` in the PDF, which no test
notices and a skim-read reader mistakes for a citation. A table or figure that no
sentence references floats: it is numbered and typeset, but nothing in the prose
tells the reader when to look at it, so the number it carries is unusable and the
result it holds is one the argument never actually invokes.

The second rule found four orphaned tables when it was written. It binds tables,
figures, and equations rather than sections, because a section is reachable
through the table of contents and a float is not.

Guards on the guard: the scan must find a substantial number of labels (an empty
scan passes any rule), and planted violations of both rules must be caught.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = PROJECT_ROOT / "docs" / "manuscript"
FORMALISM_MAP = PROJECT_ROOT / "docs" / "reference" / "formalism_map.md"

#: Files excluded: documentation *about* the manuscript, not part of it. Their
#: label mentions are examples, so they neither define nor satisfy a reference.
_EXCLUDED = {"SYNTAX.md", "AGENTS.md", "README.md", "preamble.md", "MANUSCRIPT_STATUS.md"}

#: A label definition — ``{#tbl:x}`` alone, or ``{#fig:x width=90%}`` with
#: attributes. The trailing class is what separates a definition from a
#: reference; matching ``}`` only would miss every figure in the manuscript.
_DEF_RE = re.compile(r"\{#((?:sec|fig|tbl|eq):[A-Za-z0-9_]+)[ }]")

#: A bracketed reference group: ``[@fig:x]``, or the grouped form Pandoc also
#: accepts, ``[@fig:a; @fig:b]``. Requiring the closing bracket immediately after
#: one label — as this pattern used to — silently drops BOTH labels of a grouped
#: reference, which then reads as two undefined references and two unreferenced
#: floats at once. The group is matched first and the labels extracted from it,
#: so the two forms are treated identically.
_REF_GROUP_RE = re.compile(r"\[@[^\[\]]*\]")

_PIPE_DELIMITER_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")

#: A label inside a reference group. A group may mix cross-references with
#: ordinary citations; only the four prefixed kinds are cross-references.
_REF_RE = re.compile(r"@((?:sec|fig|tbl|eq):[A-Za-z0-9_]+)")


def _malformed_pipe_table_starts(text: str) -> list[int]:
    """Return table starts whose next row is not Pandoc's delimiter row."""
    lines = text.splitlines()
    malformed: list[int] = []
    in_fence = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.lstrip().startswith("|"):
            continue
        previous = lines[index - 1].strip() if index else ""
        if previous.startswith("|"):
            continue
        # A generated row roster may hydrate into multiple pipe rows before a
        # final authored total row, so that authored row is not a table start.
        if previous.startswith("{{") and previous.endswith("_TABLE_ROWS}}"):
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if _PIPE_DELIMITER_RE.fullmatch(following) is None:
            malformed.append(index + 1)
    return malformed


#: Pandoc does not parse Markdown inside raw theorem-like LaTeX environments.
#: A token such as ``[@eq:efe]`` there survives literally into the PDF even when
#: the label itself exists, so the global defined/reference set cannot detect it.
_RAW_LATEX_ENV_RE = re.compile(
    r"\\begin\{(?P<kind>definition|theorem|proposition)\}"
    r"(?P<body>.*?)"
    r"\\end\{(?P=kind)\}",
    re.DOTALL,
)

#: Exact titled theorem declarations.  The title is part of the manuscript's
#: public formalism inventory: a generic "Definition" row can otherwise remain
#: green after the definition it purports to map has been replaced.
_THEOREM_DECL_RE = re.compile(r"\\begin\{(?P<kind>definition|proposition)\}\[(?P<title>[^\]\n]+)\]")

#: The first cell of a formalism-map row.  Equations retain their manuscript
#: label; theorem-like rows retain their exact environment kind and title.
_FORMALISM_MAP_ROW_RE = re.compile(
    r"^\| (?P<entry>`eq:[A-Za-z0-9_]+`|(?:Definition|Proposition) \([^|]+\)) \|"
)

#: Floats must be invoked by prose. Sections are reachable via the contents page,
#: so an unreferenced section heading is navigable and not a defect.
_FLOAT_KINDS = ("tbl", "fig", "eq")


def _sources() -> list[Path]:
    return sorted(p for p in MANUSCRIPT.glob("*.md") if p.name not in _EXCLUDED)


def _labels() -> tuple[dict[str, str], dict[str, list[str]]]:
    """``(defined -> file, referenced -> files)`` across the manuscript."""
    defined: dict[str, str] = {}
    referenced: dict[str, list[str]] = defaultdict(list)
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for match in _DEF_RE.finditer(text):
            defined[match.group(1)] = path.name
        for group in _REF_GROUP_RE.finditer(text):
            for match in _REF_RE.finditer(group.group(0)):
                referenced[match.group(1)].append(path.name)
    return defined, dict(referenced)


def _source_formalism_inventory() -> dict[str, list[str]]:
    """Derive equations and titled theorem-like declarations from source."""
    defined, _referenced = _labels()
    inventory: dict[str, list[str]] = {
        "equations": sorted(label for label in defined if label.startswith("eq:")),
        "definitions": [],
        "propositions": [],
    }
    for path in _sources():
        for match in _THEOREM_DECL_RE.finditer(path.read_text(encoding="utf-8")):
            inventory[f"{match.group('kind')}s"].append(match.group("title"))
    inventory["definitions"].sort()
    inventory["propositions"].sort()
    return inventory


def _mapped_formalism_inventory(text: str) -> dict[str, list[str]]:
    """Read only formalism rows, ignoring prose mentions and audit tables."""
    inventory: dict[str, list[str]] = {
        "equations": [],
        "definitions": [],
        "propositions": [],
    }
    for line in text.splitlines():
        match = _FORMALISM_MAP_ROW_RE.match(line)
        if match is None:
            continue
        entry = match.group("entry")
        if entry.startswith("`eq:"):
            inventory["equations"].append(entry.strip("`"))
            continue
        kind, title = entry.split(" (", maxsplit=1)
        inventory[f"{kind.lower()}s"].append(title[:-1])
    for entries in inventory.values():
        entries.sort()
    return inventory


def _formalism_inventory_drift(
    source: dict[str, list[str]], mapped: dict[str, list[str]]
) -> list[str]:
    """Report missing, phantom, and duplicated rows without masking any class."""
    problems: list[str] = []
    for kind in ("equations", "definitions", "propositions"):
        source_entries = set(source[kind])
        mapped_entries = set(mapped[kind])
        missing = sorted(source_entries - mapped_entries)
        extra = sorted(mapped_entries - source_entries)
        duplicates = sorted(entry for entry, count in Counter(mapped[kind]).items() if count > 1)
        if missing:
            problems.append(f"{kind}: missing {missing}")
        if extra:
            problems.append(f"{kind}: phantom {extra}")
        if duplicates:
            problems.append(f"{kind}: duplicated {duplicates}")
    return problems


def test_the_scan_finds_the_manuscript_labels():
    """POSITIVE CONTROL: an empty label set passes both rules below."""
    defined, referenced = _labels()
    assert len(defined) >= 50, f"only {len(defined)} labels defined; scan is broken"
    assert len(referenced) >= 50, f"only {len(referenced)} labels referenced"


def test_formalism_map_is_the_exact_source_derived_inventory():
    """Every source formalism has one map row, with no generic or phantom rows."""
    source = _source_formalism_inventory()
    assert {kind: len(entries) for kind, entries in source.items()} == {
        "equations": 25,
        "definitions": 2,
        "propositions": 2,
    }, (
        "the expected inventory changed; inspect each new or removed formalism before updating the map"
    )

    mapped = _mapped_formalism_inventory(FORMALISM_MAP.read_text(encoding="utf-8"))
    assert _formalism_inventory_drift(source, mapped) == []


def test_formalism_inventory_check_rejects_one_omitted_equation_row():
    """NEGATIVE CONTROL: deleting a real row must identify the exact omission."""
    source = _source_formalism_inventory()
    text = FORMALISM_MAP.read_text(encoding="utf-8")
    marker = "| `eq:paired_difference` |"
    lines = text.splitlines()
    pruned_lines = [line for line in lines if not line.startswith(marker)]
    assert len(pruned_lines) == len(lines) - 1, "negative-control target is not one unique row"

    mapped = _mapped_formalism_inventory("\n".join(pruned_lines))
    assert _formalism_inventory_drift(source, mapped) == [
        "equations: missing ['eq:paired_difference']"
    ]


def test_every_cross_reference_resolves():
    """A reference to a missing label renders as literal text in the PDF."""
    defined, referenced = _labels()
    dangling = {
        label: sorted(set(files)) for label, files in referenced.items() if label not in defined
    }
    assert dangling == {}, (
        f"cross-references with no definition: {dangling}. Pandoc renders these "
        f"as literal '[@label]' in the PDF. Define the label or fix the spelling."
    )


def _raw_latex_reference_tokens(text: str) -> list[str]:
    """Pandoc reference groups trapped inside raw theorem-like environments."""
    return [
        group.group(0)
        for block in _RAW_LATEX_ENV_RE.finditer(text)
        for group in _REF_GROUP_RE.finditer(block.group("body"))
    ]


def test_raw_latex_reference_scan_has_live_controls():
    """A detector that never fires would bless the literal-token PDF defect."""
    planted = r"\begin{proposition}See [@eq:missing].\end{proposition}"
    safe = "See [@eq:outside].\n" + r"\begin{proposition}No raw reference.\end{proposition}"
    assert _raw_latex_reference_tokens(planted) == ["[@eq:missing]"]
    assert _raw_latex_reference_tokens(safe) == []


def test_no_pandoc_reference_is_trapped_in_raw_latex():
    """Markdown cross-references in raw LaTeX render as literal ``[@...]``."""
    offenders: list[str] = []
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        for token in _raw_latex_reference_tokens(text):
            offenders.append(f"{path.name}: {token}")
    assert offenders == [], (
        f"Pandoc references inside raw theorem-like LaTeX: {offenders}. Pandoc "
        "passes these blocks through verbatim, so the token ships literally in "
        "the PDF. Move the reference into surrounding Markdown or restate it "
        "without a Markdown token."
    )


def test_every_float_is_referenced_by_the_prose():
    """A table or figure no sentence points at is numbered but unusable."""
    defined, referenced = _labels()
    orphans = {
        label: file
        for label, file in defined.items()
        if label.split(":")[0] in _FLOAT_KINDS and label not in referenced
    }
    assert orphans == {}, (
        f"floats defined but never referenced: {orphans}. Each is typeset and "
        f"numbered with nothing in the prose sending the reader to it. Add a "
        f"[@label] at the point in the argument where the float is the evidence."
    )


#: ``Figure [@fig:x]`` — pandoc-crossref already expands the reference to
#: "Figure 7", so a hand-written prefix renders as "Figure Figure 7". The plural
#: forms are banned for the same reason: pandoc-crossref expands a grouped
#: ``[@fig:a; @fig:b]`` to "Figures 3 and 4" by itself.
_DOUBLE_PREFIX_RE = re.compile(
    r"\b(?:Figures?|Figs?\.|Tables?|Sections?|Equations?|Eqs?\.|"
    r"figures?|tables?|sections?|equations?)\s+\[@"
    r"(?:sec|fig|tbl|eq):",
)


def test_no_reference_carries_a_hand_written_prefix():
    """pandoc-crossref supplies the word; writing it too yields 'Figure Figure 7'."""
    offenders = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _DOUBLE_PREFIX_RE.search(line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == [], (
        f"hand-written reference prefixes at {offenders}. pandoc-crossref expands "
        f"[@fig:x] to 'Figure N' on its own — drop the preceding word."
    )


def test_planted_violations_are_caught():
    """POSITIVE CONTROL: both rules can fail."""
    assert _DEF_RE.search("...](../../output/figures/x.png){#fig:x width=90%}")
    assert _DEF_RE.search(": A caption. {#tbl:x}")
    assert _REF_RE.search("as [@tbl:x] shows")
    # A citation is not a cross-reference and must not be mistaken for one.
    assert not _REF_RE.search("[@friston2017active]")
    # A grouped reference contributes both labels, not neither.
    grouped = "compare [@fig:a; @fig:b] and [@friston2017active]"
    found = [
        match.group(1)
        for group in _REF_GROUP_RE.finditer(grouped)
        for match in _REF_RE.finditer(group.group(0))
    ]
    assert found == ["fig:a", "fig:b"]
    assert _DOUBLE_PREFIX_RE.search("The intervals in Figure [@fig:external_benchmarks] are")
    assert _DOUBLE_PREFIX_RE.search("Figures [@fig:a; @fig:b] both show")
    assert _DOUBLE_PREFIX_RE.search("see Tables [@tbl:a] and [@tbl:b]")
    assert not _DOUBLE_PREFIX_RE.search("The intervals in [@fig:external_benchmarks] are")
    # A citation preceded by such a word is ordinary prose, not a double prefix.
    assert not _DOUBLE_PREFIX_RE.search("see section [@shannon1948]")
    assert not _DOUBLE_PREFIX_RE.search("see sections [@shannon1948; @friston2017active]")


def test_no_interval_joins_two_tokens_that_can_render_a_triple_hyphen() -> None:
    """A `--` between two tokens is an en dash unless the right one is negative.

    Then it is three hyphens, which pandoc's smart extension folds into an em
    dash — and the em dash eats the minus sign. The published interval
    `-0.383605---0.308352` printed as `-0.383605—0.308352`, an interval that
    appears to span zero, in the same parenthesis as a p of 3e-31 and a sentence
    saying the probe was far below the comparator.

    Guarded over the class rather than the instance: any `{{A}}--{{B}}` whose
    right-hand label currently holds a negative value fails, whichever section it
    is in and whoever writes it next.
    """
    import json

    ledger = json.loads((PROJECT_ROOT / "data" / "claims.json").read_text(encoding="utf-8"))[
        "claims"
    ]
    offenders = []
    for path in sorted((PROJECT_ROOT / "docs" / "manuscript").glob("*.md")):
        for match in re.finditer(
            r"\{\{([A-Z0-9_]+)\}\}--\{\{([A-Z0-9_]+)\}\}", path.read_text(encoding="utf-8")
        ):
            right = match.group(2).lower()
            entry = ledger.get(right)
            if entry is None:
                continue
            try:
                value = float(entry["value"])
            except (TypeError, ValueError):
                continue
            if value < 0:
                offenders.append(f"{path.name}: {match.group(0)} (right side = {value})")
    assert offenders == [], (
        "these token pairs joined by '--' have a negative right-hand value, so the "
        "rendered text carries three hyphens and the minus sign is lost to an em "
        f"dash: {offenders}. Use ' to ' or an explicit en dash instead."
    )


def test_the_assembly_tables_list_every_section_that_exists():
    """`README.md` and `SYNTAX.md` enumerate sections by filename, by hand.

    Neither is generated and nothing validated them, so they drifted: the
    supplement had been part of the manuscript for as long as it existed and
    `README.md` never listed it. A hand-maintained index of a directory is a
    second copy of that directory, and the second copy is the one that goes
    stale silently.

    NEGATIVE CONTROL: the tables must list a substantial number of sections, or
    an empty table would satisfy a subset check and prove nothing.
    """
    import re
    from pathlib import Path

    manuscript = Path(__file__).resolve().parents[1] / "docs" / "manuscript"
    #: Author-facing files, not sections of the paper.
    NOT_SECTIONS = {"README.md", "AGENTS.md", "SYNTAX.md", "preamble.md", "MANUSCRIPT_STATUS.md"}

    actual = {p.name for p in manuscript.glob("*.md")} - NOT_SECTIONS
    assert len(actual) >= 12, "too few sections for this check to constrain anything"

    for table in ("README.md", "SYNTAX.md"):
        text = (manuscript / table).read_text(encoding="utf-8")
        listed = set(re.findall(r"(\d+[a-z]?_[a-z_0-9]+\.md)", text))
        assert listed, f"{table} enumerates no sections"

        missing = sorted(actual - listed)
        assert not missing, (
            f"{table} does not list {missing}. A section absent from the assembly "
            f"table is invisible to anyone reading the index instead of the "
            f"directory."
        )
        phantom = sorted(listed - actual)
        assert not phantom, (
            f"{table} lists {phantom}, which no longer exists. A renamed section "
            f"leaves the old name behind in every hand-maintained table."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rendered_provenance_drift(root: Path, report: dict) -> list[str]:
    """Independently recompute every file identity carried by the render bridge."""
    problems: list[str] = []

    def check(relative_raw: object, expected_raw: object, label: str) -> None:
        relative = Path(str(relative_raw))
        expected = str(expected_raw)
        if relative.is_absolute() or ".." in relative.parts:
            problems.append(f"{label}: unsafe path {relative}")
            return
        path = root / relative
        if not path.is_file():
            problems.append(f"{label}: missing {relative}")
            return
        actual = _sha256(path)
        if actual != expected:
            problems.append(f"{label}: {relative} records {expected}, disk has {actual}")

    consumed = report.get("consumed_manuscript")
    if not isinstance(consumed, list) or not consumed:
        return ["consumed_manuscript is absent or empty"]
    for index, row in enumerate(consumed):
        if not isinstance(row, dict):
            problems.append(f"consumed_manuscript[{index}] is not an object")
            continue
        check(row.get("source_path"), row.get("source_sha256"), f"source row {index}")
        check(row.get("rendered_path"), row.get("rendered_sha256"), f"rendered row {index}")

    evidence = report.get("evidence_files")
    if not isinstance(evidence, dict) or not evidence:
        problems.append("evidence_files is absent or empty")
    else:
        for name, row in evidence.items():
            if not isinstance(row, dict):
                problems.append(f"evidence file {name} is not an object")
                continue
            check(row.get("path"), row.get("sha256"), f"evidence file {name}")
    return problems


def test_rendered_provenance_matches_every_consumed_file_on_disk():
    """The tracked provenance bridge must describe this exact publication tree.

    Stage 03 renders the publication; Stage 04 writes source and hydrated hashes
    into ``rendered_provenance.json`` after validating it.
    A later source edit otherwise leaves a plausible, internally complete report
    beside a PDF assembled from a different tree. Recomputing every digest here
    makes stale provenance fail the ordinary sidecar suite, not only an optional
    template validation run.
    """
    report_path = PROJECT_ROOT / "output" / "reports" / "rendered_provenance.json"
    assert report_path.is_file(), "rendered provenance is absent; run template Stages 03 then 04"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("schema_version") == "template-rendered-provenance-v2"

    rows = report.get("consumed_manuscript")
    assert isinstance(rows, list) and len(rows) >= 12, (
        "rendered provenance names too few manuscript inputs to constrain the publication"
    )
    assert _rendered_provenance_drift(PROJECT_ROOT, report) == []

    expected_sources = {path.relative_to(PROJECT_ROOT).as_posix() for path in _sources()}
    observed_sources = {str(row.get("source_path")) for row in rows if isinstance(row, dict)}
    assert observed_sources == expected_sources, (
        "rendered provenance does not cover the exact authored manuscript inventory"
    )

    composition_path = PROJECT_ROOT / "output" / "reports" / "manuscript_composition.json"
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    composed_paths = {str(row["path"]) for row in composition["ordered_inputs"]}
    rendered_paths = {str(row.get("rendered_path")) for row in rows if isinstance(row, dict)}
    assert rendered_paths == composed_paths, (
        "rendered provenance and the composition manifest name different hydrated inputs"
    )


def test_rendered_provenance_drift_detector_is_live(tmp_path: Path):
    """NEGATIVE CONTROL: one forged digest must be rejected, then its real value accepted."""
    source = tmp_path / "docs" / "manuscript" / "section.md"
    rendered = tmp_path / "output" / "manuscript" / "section.md"
    evidence = tmp_path / "output" / "reports" / "evidence.json"
    for path, content in ((source, "source"), (rendered, "rendered"), (evidence, "{}")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    report = {
        "consumed_manuscript": [
            {
                "source_path": "docs/manuscript/section.md",
                "source_sha256": "0" * 64,
                "rendered_path": "output/manuscript/section.md",
                "rendered_sha256": _sha256(rendered),
            }
        ],
        "evidence_files": {
            "probe": {"path": "output/reports/evidence.json", "sha256": _sha256(evidence)}
        },
    }
    drift = _rendered_provenance_drift(tmp_path, report)
    assert len(drift) == 1 and "source row 0" in drift[0]

    report["consumed_manuscript"][0]["source_sha256"] = _sha256(source)
    assert _rendered_provenance_drift(tmp_path, report) == []


def test_every_supplement_section_renders_after_every_body_section():
    """A demoted section that renders in the body fails nothing without this.

    Section order is filename sort order, and the supplement is distinguished
    only by its `98` prefix. Nothing asserted that a file *named* a supplement
    actually lands after the body: no ordering check, no page-count assertion,
    no bucket record in any committed artifact. A section demoted with the wrong
    prefix renders between the conclusion and the references, the suite stays
    green, the render succeeds, and the defect is visible only to someone
    reading the table of contents.

    That is the exact hollow-green shape this project's gate doctrine exists to
    refuse, and it matters now because sections are about to be demoted.

    NEGATIVE CONTROL: the composition must contain both kinds of section, or the
    ordering claim is vacuous.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    composition = root / "output" / "reports" / "manuscript_composition.json"
    if not composition.exists():
        pytest.skip("no composition record; run the render stage first")

    ordered = json.loads(composition.read_text(encoding="utf-8"))["ordered_inputs"]
    assert ordered, "the composition record lists no inputs"

    def kind(path: str) -> str:
        name = Path(path).name
        if "supplement" in name:
            return "supplement"
        if "references" in name:
            return "references"
        return "body"

    positions = {"body": [], "supplement": [], "references": []}
    for entry in ordered:
        positions[kind(entry["path"])].append(entry["ordinal"])

    assert positions["body"], "no body sections found"
    assert positions["supplement"], (
        "no supplement section found; this check would pass vacuously and the "
        "restructure it protects would be unverified"
    )

    last_body = max(positions["body"])
    first_supplement = min(positions["supplement"])
    assert first_supplement > last_body, (
        f"a supplement section renders at ordinal {first_supplement}, before the "
        f"last body section at {last_body}. Section order is filename sort order, "
        f"so a demoted section needs a prefix that sorts after every body file."
    )

    if positions["references"]:
        assert min(positions["references"]) > max(positions["supplement"]), (
            "the bibliography renders before the supplement"
        )


def test_every_section_filename_starts_with_a_digit():
    """Two guards in this repository select sections by leading digit.

    `tests/test_manuscript_tokens.py` globs `[0-9]*.md` for the unresolved-token
    audit, and `scripts/generate_figure_registry.py` returns
    `path.name[0].isdigit()`. A section named with any other leading character
    renders perfectly well — the template's own documentation offers an `S01_`
    supplemental bucket — while dropping out of both guards. Its unresolved
    `{{TOKEN}}` markers stop being audited and its figures stop being
    registered, and the registry's own `len(found) > 20` assertion keeps
    passing.

    That is a hollow-green hole a filename can open, which is why the naming
    convention is enforced rather than documented.

    NEGATIVE CONTROL: the sections must exist and be numerous, or a convention
    check over an empty set proves nothing.
    """
    from pathlib import Path

    manuscript = Path(__file__).resolve().parents[1] / "docs" / "manuscript"
    NOT_SECTIONS = {"README.md", "AGENTS.md", "SYNTAX.md", "preamble.md", "MANUSCRIPT_STATUS.md"}

    sections = sorted(p.name for p in manuscript.glob("*.md") if p.name not in NOT_SECTIONS)
    assert len(sections) >= 12, "too few sections for this check to constrain anything"

    offenders = [name for name in sections if not name[0].isdigit()]
    assert not offenders, (
        f"section filenames that do not start with a digit: {offenders}. Both the "
        f"token audit and the figure registry select by leading digit, so these "
        f"would render into the paper unaudited."
    )

    # The supplement bucket is a literal `98_` prefix in the renderer, not a
    # numeric comparison: `98a_` falls through and renders as a body section.
    # Verified by executing the renderer's own discovery function.
    mislabeled = [name for name in sections if "supplement" in name and not name.startswith("98_")]
    assert not mislabeled, (
        f"sections named as supplements that will render in the BODY: {mislabeled}. "
        f"The renderer tests `stem.startswith('98_')` literally, so `98a_` and "
        f"`13_supplement_` both land among the body sections."
    )


def test_the_h1_registry_matches_the_actual_section_headings():
    """`SYNTAX.md` stores every section's H1 by hand, and nothing read it.

    Two tests exclude the file by name, so a retitling pass lands green while
    leaving the registry wrong. It had drifted on ten of twenty-two rows — every
    title changed in an earlier pass, plus the file rename — and the drift was
    invisible because the registry is documentation about the manuscript rather
    than part of it.

    A hand-maintained copy of something derivable is a second source of truth,
    and this is the check that keeps the copy honest.

    NEGATIVE CONTROL: every section must have a labeled H1, so a section whose
    heading lost its label fails here rather than silently dropping out of the
    comparison.
    """
    import re
    from pathlib import Path

    manuscript = Path(__file__).resolve().parents[1] / "docs" / "manuscript"
    NOT_SECTIONS = {"README.md", "AGENTS.md", "SYNTAX.md", "preamble.md", "MANUSCRIPT_STATUS.md"}

    actual = {}
    for path in sorted(manuscript.glob("*.md")):
        if path.name in NOT_SECTIONS:
            continue
        heading = re.search(
            r"^# (.+?)\s*\{(#sec:[a-z_0-9]+)\}\s*$",
            path.read_text(encoding="utf-8"),
            re.M,
        )
        assert heading, f"{path.name} has no labeled H1; every section needs one"
        actual[path.name] = (heading.group(1), heading.group(2))

    assert len(actual) >= 12, "too few sections for this check to constrain anything"

    registry = {}
    for row in re.finditer(
        r"^\| `([^`]+\.md)` \| (.+?) \| `\{(#sec:[a-z_0-9]+)\}` \|$",
        (manuscript / "SYNTAX.md").read_text(encoding="utf-8"),
        re.M,
    ):
        registry[row.group(1)] = (row.group(2).strip(), row.group(3))

    assert registry, "SYNTAX.md holds no H1 registry rows"
    drift = {
        name: {"registry": registry.get(name), "actual": heading}
        for name, heading in actual.items()
        if registry.get(name) != heading
    }
    assert not drift, (
        f"SYNTAX.md disagrees with the section headings for {sorted(drift)}. "
        f"Regenerate the table from the files rather than editing it by hand."
    )

    phantom = sorted(set(registry) - set(actual))
    assert not phantom, f"SYNTAX.md lists sections that no longer exist: {phantom}"


def test_every_authored_pipe_table_has_a_pandoc_delimiter_row() -> None:
    """A second header/spanner row makes Pandoc print the whole table literally."""
    defects: dict[str, list[int]] = {}
    for path in sorted(MANUSCRIPT.glob("*.md")):
        if path.name in _EXCLUDED:
            continue
        lines = _malformed_pipe_table_starts(path.read_text(encoding="utf-8"))
        if lines:
            defects[path.name] = lines
    assert defects == {}, (
        "pipe-table starts must be followed immediately by an alignment delimiter; "
        f"these blocks render as literal pipe text: {defects}"
    )


def test_pipe_table_grammar_guard_rejects_a_spanner_header() -> None:
    """NEGATIVE CONTROL: reproduce the two-header-row publication defect."""
    planted = "| | Model A | |\n| k | score | p |\n|---:|---:|---:|\n"
    assert _malformed_pipe_table_starts(planted) == [1]
