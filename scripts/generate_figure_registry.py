#!/usr/bin/env python3
"""Publish ``output/figures/figure_registry.json`` via the template's own builder.

Thin orchestrator. The registry envelope, its schema and every validity rule
belong to ``infrastructure.documentation.generated_figure_registry`` in the
template repository; this script only assembles the project-owned facts and
hands them over.

## Why the builder is imported rather than reimplemented

The registry is consumed by the template's output validator, and its accepted
shape is defined by the template, not here. Hand-writing the envelope from the
outside would be a guess at another repository's schema that happens to pass
today and drifts silently the moment the template changes it. Importing the
builder means the schema has exactly one definition and this project cannot
disagree with it: if a spec is malformed, ``build_generated_figure_registry``
raises ``FigureRegistryError`` here rather than the validator rejecting the
output two stages later.

## Why this is a script and not part of ``src/``

``docs/architecture.md`` states that ``src/ailm/`` must never import
``infrastructure``: the package is available at render time and at no other
time, and ``src/`` has to work in a bare checkout with no template beside it.
Scripts are the sanctioned boundary — ``scripts/z_generate_manuscript_variables.py``
already imports ``infrastructure.rendering.manuscript_injection`` after
resolving the template root, and this script uses the same resolver for the same
reason. Nothing under ``src/ailm/`` gains a template dependency, and running the
figure pipeline without a template checkout still works; only this optional
publishing step needs one.

## Where the facts come from

Neither field is invented:

- **filename** and **generated_by** come from ``output/figures/figure_provenance.json``,
  written by ``scripts/generate_figures.py`` on the run that produced the images.
- **label** and **caption** come from the manuscript's own figure references,
  ``![caption](../output/figures/name.png){#fig:label}``, which are the captions
  a reader sees and the labels pandoc-crossref resolves.
- **alt_text** comes from ``data/figure_alt_text.yaml``, authored per label. It is
  deliberately not derived from the caption: a caption here carries population,
  provenance and falsification condition and runs to hundreds of words, which is
  the wrong thing to read aloud in place of a picture. Without this field the
  template's accessibility layer fails the HTML render closed, which is how the
  omission was found.

So the registry describes the figures that were actually generated, captioned as
the paper actually captions them. A referenced figure that was never generated
is an error; unreferenced cover art and diagnostic figures are permitted.
Repeated labels or filenames are rejected before either can overwrite a caption.

Usage
-----
    .venv/bin/python scripts/generate_figure_registry.py
    .venv/bin/python scripts/generate_figure_registry.py --check   # verify only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Schema identifier stamped into the registry. Namespaced to this project
#: because the template's validator accepts any non-empty string and a shared
#: value would make two projects' registries indistinguishable.
SCHEMA_VERSION = "ailm-figure-registry-v1"

#: The manuscript's figure syntax: `![caption](../../output/figures/x.png){#fig:x ...}`.
#: The leading `../` run is matched by depth-agnostic repetition rather than a fixed
#: count, so moving the manuscript directory cannot silently empty the registry.
#: Captions in this project routinely run to several hundred words and contain
#: brackets and parentheses, so the caption group is matched lazily up to the
#: image path rather than by balancing delimiters.
_FIGURE_PATTERN = re.compile(
    r"!\[(?P<caption>.*?)\]\((?:\.\./)+output/figures/(?P<filename>[^)]+?)\)\{#(?P<label>fig:[A-Za-z0-9_:-]+)",
    re.DOTALL,
)


@dataclass(frozen=True)
class FigureSpec:
    """One figure, satisfying the template's ``FigureSpecLike`` protocol.

    The protocol is structural, so this class does not import it — which keeps
    the dataclass usable in a bare checkout where ``infrastructure`` is absent.
    """

    label: str
    filename: str
    caption: str
    generated_by: str
    alt_text: str


def resolve_template_root() -> Path | None:
    """Locate the template repo root providing the ``infrastructure`` package.

    Same order of preference as ``scripts/z_generate_manuscript_variables.py``,
    deliberately: two different answers to "where is the template" would be a
    defect waiting to happen.

    1. ``TEMPLATE_REPO_ROOT`` (set by the renderer — the authoritative answer)
    2. A walk-up from this file, for invocation from inside the template
    3. The ``template`` checkout beside the ``projects`` sidecar root, for the
       canonical ``<root>/template`` + ``<root>/projects/working/<name>`` layout
    4. The conventional home-directory checkout, for invocation elsewhere
    """
    env_root = os.environ.get("TEMPLATE_REPO_ROOT", "").strip()
    if env_root:
        # An explicit checkout is authoritative, as in manuscript hydration.
        # Falling back after a typo would publish through a different engine.
        candidate = Path(env_root)
        return candidate if (candidate / "infrastructure" / "__init__.py").is_file() else None
    # The candidate order is byte-identical in scripts/z_generate_manuscript_variables.py
    # and scripts/verify_release.py; tests/test_figure_registry.py holds them to it.
    candidates = list(_PROJECT_ROOT.parents)
    candidates.extend(parent / "template" for parent in _PROJECT_ROOT.parents[2:3])
    candidates.append(Path.home() / "Documents" / "GitHub" / "template")

    for candidate in candidates:
        if (candidate / "infrastructure" / "__init__.py").is_file():
            return candidate
    return None


def load_provenance(project_root: Path = _PROJECT_ROOT) -> dict[str, str]:
    """``{basename: generating module}`` from an authenticated figure receipt.

    Raises:
        FileNotFoundError: If the provenance file is absent — the figures have
            not been generated, and a registry built without them would claim
            files nobody produced.
    """
    from scripts import generate_figures as contract
    from scripts._artifact_io import AuthenticatedFileError, read_strict_json_regular_file

    path = project_root / "output" / "figures" / contract._PROVENANCE_NAME
    try:
        payload, _receipt_digest = read_strict_json_regular_file(
            path, description="figure provenance receipt"
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{path} is absent. Run scripts/generate_figures.py before publishing the registry."
        ) from error
    except AuthenticatedFileError as error:
        raise ValueError(str(error)) from error
    expected_fields = {
        "schema",
        "environment",
        "manuscript_variables",
        "shared_sources",
        "code_dependencies",
        "computation_inputs",
        "figures",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or payload.get("schema") != contract._PROVENANCE_SCHEMA
    ):
        raise ValueError(
            f"{path} is not {contract._PROVENANCE_SCHEMA}; run scripts/generate_figures.py "
            "before publishing the registry"
        )
    environment = payload.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError(f"{path} has no numeric-environment record")

    variables = payload.get("manuscript_variables")
    if (
        not isinstance(variables, dict)
        or set(variables) != {"path", "sha256"}
        or variables.get("path") != contract._MANUSCRIPT_VARIABLES_PATH
    ):
        raise ValueError(f"{path} has no canonical manuscript-variable input record")
    _require_digest(project_root / variables["path"], variables.get("sha256"), path)

    try:
        expected_record_sets = {
            "shared_sources": contract._shared_source_records(project_root),
            "code_dependencies": contract._code_dependency_records(project_root),
            "computation_inputs": contract._computation_input_records(project_root),
        }
    except contract.FigureProvenanceError as error:
        raise ValueError(str(error)) from error
    for field, expected in expected_record_sets.items():
        recorded = payload.get(field)
        if recorded != expected:
            raise ValueError(
                f"{path} has stale, malformed, or incomplete {field.replace('_', ' ')} records"
            )

    figures = payload.get("figures")
    if not isinstance(figures, dict) or not figures:
        raise ValueError(f"{path} has no figure records")

    # `figures` is keyed by figure name; every value authenticates both its
    # generating source module and the portable PNG paths the run published.
    produced: dict[str, str] = {}
    for name, entry in figures.items():
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not isinstance(entry, dict)
            or set(entry) != {"module", "source", "outputs"}
        ):
            raise ValueError(f"{path} has a malformed record for {name!r}")
        module = entry.get("module")
        if not isinstance(module, str) or not module.isidentifier():
            raise ValueError(f"{path} has an unsafe generator module for {name!r}")
        source = entry.get("source")
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "sha256"}
            or source.get("path") != (f"src/ailm/figures/{module}.py")
        ):
            raise ValueError(f"{path} has a noncanonical source for {name!r}")
        _require_digest(project_root / source["path"], source.get("sha256"), path)
        outputs = entry.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"{path} has no authenticated outputs for {name!r}")
        for output in outputs:
            if not isinstance(output, dict) or set(output) != {"path", "sha256"}:
                raise ValueError(f"{path} has a malformed output for {name!r}")
            raw = output.get("path")
            if not contract._safe_output_name(raw):
                raise ValueError(f"{path} has unsafe output path {raw!r}")
            if not isinstance(raw, str):
                raise ValueError(f"{path} has a non-string output path {raw!r}")
            _require_digest(
                project_root / "output" / "figures" / raw,
                output.get("sha256"),
                path,
            )
            if raw in produced:
                raise ValueError(f"{path} claims duplicate output {raw!r}")
            produced[raw] = f"ailm.figures.{module}"
    try:
        published = contract._published_png_names(project_root / "output" / "figures")
    except contract.FigureProvenanceError as error:
        raise ValueError(str(error)) from error
    if published != set(produced):
        raise ValueError(f"{path} does not authenticate the exact published PNG inventory")
    return produced


def _require_digest(target: Path, expected: object, receipt: Path) -> None:
    """Require one regular file to match the receipt's lowercase SHA-256."""
    from scripts._artifact_io import (
        AuthenticatedFileError,
        sha256_authenticated_regular_file,
    )

    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{receipt} has an invalid SHA-256 for {target}")
    try:
        actual = sha256_authenticated_regular_file(
            target,
            description=f"artifact named by {receipt.name}",
        )
    except (AuthenticatedFileError, FileNotFoundError) as error:
        raise ValueError(f"{receipt} names unavailable or symlinked file {target}") from error
    if actual != expected:
        raise ValueError(f"{target} does not match {receipt}")


#: Files in `docs/manuscript/` that are not manuscript sections. `SYNTAX.md` carries
#: a worked example of the figure syntax pointing at `name.png`, which is not a
#: figure and was never generated; scanning it would make the registry claim a
#: file that cannot exist. The rendered sections are the numbered ones, so the
#: rule is stated over the general class rather than by naming SYNTAX.md.
_NON_SECTION_FILES = frozenset({"AGENTS.md", "README.md", "SYNTAX.md"})


def _is_manuscript_section(path: Path) -> bool:
    """Whether a file under `docs/manuscript/` is a rendered section.

    Sections are numbered (`03_methods.md`, `07a_clause_level.md`); the
    supporting documents beside them are not.
    """
    return path.name not in _NON_SECTION_FILES and path.name[0].isdigit()


def collect_manuscript_figures(project_root: Path = _PROJECT_ROOT) -> dict[str, tuple[str, str]]:
    """``{basename: (label, caption)}`` from the manuscript's figure references.

    Only rendered sections are scanned; see ``_is_manuscript_section``.
    Each filename and label must occur exactly once across those sections.
    """
    found: dict[str, tuple[str, str]] = {}
    filenames: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    sections = [
        p
        for p in sorted((project_root / "docs" / "manuscript").glob("*.md"))
        if _is_manuscript_section(p)
    ]
    if not sections:
        raise ValueError("no numbered manuscript sections found; the scan would be empty")
    for source in sections:
        for match in _FIGURE_PATTERN.finditer(source.read_text(encoding="utf-8")):
            filename = match.group("filename").strip()
            label = match.group("label")
            for kind, value, locations in (
                ("filename", filename, filenames),
                ("label", label, labels),
            ):
                if value in locations:
                    raise ValueError(
                        f"duplicate figure {kind} {value!r} in "
                        f"{locations[value].relative_to(project_root)} and "
                        f"{source.relative_to(project_root)}"
                    )
                locations[value] = source
            caption = " ".join(match.group("caption").split())
            found[filename] = (label, caption)
    return found


def build_specs(project_root: Path = _PROJECT_ROOT) -> list[FigureSpec]:
    """Join the provenance receipt to the manuscript's captions.

    Raises:
        ValueError: If a referenced figure was never generated. That direction
            is fatal because the paper would print an image it cannot produce.
            The reverse — generated but unreferenced — is allowed and skipped:
            cover art and diagnostic plates are produced without being cited,
            and the template's own builder documents the same tolerance.
    """
    produced = load_provenance(project_root)
    referenced = collect_manuscript_figures(project_root)

    missing = sorted(name for name in referenced if name not in produced)
    if missing:
        raise ValueError(
            f"the manuscript references figures that were never generated: {', '.join(missing)}"
        )

    alt_texts = load_alt_texts(project_root)
    unlabeled = sorted(label for label, _ in referenced.values() if label not in alt_texts)
    if unlabeled:
        raise ValueError(
            "these referenced figures have no entry in data/figure_alt_text.yaml: "
            f"{', '.join(unlabeled)}"
        )

    return [
        FigureSpec(
            label=label,
            filename=filename,
            caption=caption,
            generated_by=produced[filename],
            alt_text=alt_texts[label],
        )
        for filename, (label, caption) in sorted(referenced.items())
    ]


def load_alt_texts(project_root: Path = _PROJECT_ROOT) -> dict[str, str]:
    """Authored one-sentence image alternatives, keyed by figure label."""
    import yaml

    source = project_root / "data" / "figure_alt_text.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    entries = (payload or {}).get("figures")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{source} does not declare a non-empty 'figures' mapping")
    return {label: " ".join(str(text).split()) for label, text in entries.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the generated-figure registry.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build and validate the registry without writing it.",
    )
    args = parser.parse_args()

    template_root = resolve_template_root()
    if template_root is None:
        print(
            "error: could not locate a template checkout providing the "
            "'infrastructure' package. Set TEMPLATE_REPO_ROOT to your template "
            "repository root. The figure pipeline itself does not need this; only "
            "publishing the registry does.",
            file=sys.stderr,
        )
        return 2
    figures_dir = _PROJECT_ROOT / "output" / "figures"
    # Authenticate the project-owned inputs before adding the surrounding
    # template checkout to ``sys.path``. Both repositories have a top-level
    # ``scripts`` directory; adding the template first made the lazy imports in
    # ``load_provenance`` resolve against the wrong repository.
    sys.path.insert(0, str(_PROJECT_ROOT))
    try:
        specs = build_specs()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The local ``scripts`` namespace is now cached and cannot be shadowed by
    # the template's unrelated regular package. The template path is needed
    # only for its authoritative registry builder.
    sys.path.insert(0, str(template_root))
    from infrastructure.documentation import (
        FigureRegistryError,
        build_generated_figure_registry,
        write_generated_figure_registry,
    )

    generated = [figures_dir / spec.filename for spec in specs]
    try:
        registry = build_generated_figure_registry(specs, generated, schema_version=SCHEMA_VERSION)
    except FigureRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"template root: {template_root}")
    print(f"figures declared: {len(registry['figures'])}")
    if args.check:
        print("--check: registry builds and validates; not written.")
        return 0

    path = write_generated_figure_registry(
        figures_dir / "figure_registry.json",
        specs,
        generated,
        schema_version=SCHEMA_VERSION,
    )
    print(f"wrote {path.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
