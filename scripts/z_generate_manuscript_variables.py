#!/usr/bin/env python3
"""Thin orchestrator: generate and inject AILM manuscript ``{{TOKEN}}`` values.

The renderer invokes this automatically before assembling the manuscript
(``infrastructure.rendering._manuscript_source.run_manuscript_variable_script``),
using the project's own ``.venv`` and passing ``TEMPLATE_REPO_ROOT``.

All computation lives in ``src/ailm/manuscript_variables``; all substitution lives
in ``infrastructure.rendering.manuscript_injection``. This file only wires them.

## Why TEMPLATE_REPO_ROOT rather than a relative walk-up

The canonical exemplars live *inside* the template repo, so they can find
``infrastructure`` with ``Path(__file__).parents[3]``. AILM does not: it is a
sidecar project at ``~/Documents/GitHub/projects/working/`` that is *symlinked*
into ``template/projects/working/``. A relative walk-up from the real path lands
in ``~/Documents/GitHub/projects/``, which has no ``infrastructure`` package, so
the import fails — or worse, silently finds an unrelated one.

``TEMPLATE_REPO_ROOT`` is set by the renderer precisely for this case. This script
requires it (with a walk-up fallback for direct manual invocation) and fails
loudly when ``infrastructure`` cannot be located, rather than emitting a
manuscript full of unresolved ``{{TOKEN}}`` markers.

Exit codes:
    0   variables written and injected
    1   unresolvable template root, missing claim, or injection failure

When ``SOURCE_DATE_EPOCH`` is absent, direct invocation derives it from the
last audited Git commit that touched a manuscript *input* (every tracked path
outside the regenerable ``output/`` tree) before generating variables. Using
that commit rather than ``HEAD`` gives regeneration a fixed point: a commit
that only refreshes ``output/`` does not move the epoch that stamped it. An
explicit value remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


class ManuscriptPublicationError(RuntimeError):
    """Raised when a staged manuscript publication cannot be committed safely."""


#: Git pathspec naming the manuscript's inputs: the whole tracked tree except
#: the regenerable ``output/`` directory. ``scripts/verify_release.py`` carries
#: the same pathspec for the release receipt; the two must not drift, and
#: ``tests/test_manuscript_tokens.py`` derives its expected epoch from this one.
INPUT_EPOCH_PATHSPEC: tuple[str, ...] = (":/", ":(exclude,top)output")


def _ensure_source_date_epoch(
    environment: MutableMapping[str, str] | None = None,
    *,
    project_root: Path = _PROJECT_ROOT,
) -> str:
    """Use an explicit reproducibility epoch or derive one from the input commit.

    Direct invocation previously fell through to the variable generator's
    wall-clock fallback, and later to the ``HEAD`` commit epoch. ``HEAD`` had no
    fixed point: committing regenerated ``output/`` moved ``HEAD``, which changed
    the stamp, which dirtied ``output/`` again. The epoch is therefore the
    committer time of the last commit reachable from ``HEAD`` that touched any
    path outside ``output/``. An explicitly supplied non-empty value remains
    authoritative; only an absent or blank value is filled here.
    """
    active_environment = os.environ if environment is None else environment
    configured = active_environment.get("SOURCE_DATE_EPOCH", "").strip()
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "HEAD", "--", *INPUT_EPOCH_PATHSPEC],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ManuscriptPublicationError(
            f"cannot derive SOURCE_DATE_EPOCH from the audited input commit: {exc}"
        ) from exc
    epoch = completed.stdout.strip()
    if not epoch:
        raise ManuscriptPublicationError(
            "cannot derive SOURCE_DATE_EPOCH: no commit reachable from HEAD touches a "
            "manuscript input outside output/"
        )
    try:
        parsed = int(epoch)
    except ValueError as exc:
        raise ManuscriptPublicationError(
            f"git returned a non-integer input-commit timestamp: {epoch!r}"
        ) from exc
    if parsed < 0 or str(parsed) != epoch:
        raise ManuscriptPublicationError(
            f"git returned an invalid input-commit timestamp: {epoch!r}"
        )
    active_environment["SOURCE_DATE_EPOCH"] = epoch
    return epoch


def render_supporting_protocol_html(
    hydrated_source: Path,
    output_dir: Path,
    *,
    check_only: bool = False,
    additional_output_dirs: tuple[Path, ...] = (),
) -> int:
    """Render the canonical protocol account as a sibling of the local PDF.

    Uses the existing provider's whole-page renderer, including its attribution
    guards. No checkout path or unpublished remote is a publication destination.
    The caller supplies the freshly checked hydrated canonical document.
    Additional destinations share this invocation's fresh provider render; each
    destination is independently compared and validated.
    """
    import tempfile

    from ailm.reference_presentation import (
        PROTOCOL_REFERENCE_ANCHORS,
        PROTOCOL_SUPPORTING_HTML,
    )
    from scripts._artifact_io import _open_or_create_directory, atomic_write_bytes

    template_root = _resolve_template_root()
    if template_root is None:
        raise RuntimeError("supporting protocol HTML requires the configured template provider")
    sys.path.insert(0, str(template_root))
    from infrastructure.rendering.config import RenderingConfig
    from infrastructure.rendering.web_renderer import WebRenderer

    output_dirs = (output_dir, *additional_output_dirs)
    for destination in output_dirs:
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError(f"supporting protocol output must be a real directory: {destination}")
        descriptor = _open_or_create_directory(destination.parent)
        os.close(descriptor)
    with tempfile.TemporaryDirectory(prefix=".protocol-html-", dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        source = staged / "docs" / "extension_protocols.md"
        source.parent.mkdir()
        source.write_bytes(hydrated_source.read_bytes())
        renderer = WebRenderer(RenderingConfig(web_dir=str(staged / "html")))
        page = renderer.render(source)
        if page.name != Path(PROTOCOL_SUPPORTING_HTML).name:
            raise RuntimeError("supporting protocol renderer changed the registered filename")
        html = page.read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([^"\n]+)"', html))
        missing = set(PROTOCOL_REFERENCE_ANCHORS.values()) - ids
        if missing:
            raise RuntimeError(
                f"supporting protocol HTML is missing registered anchors: {sorted(missing)}"
            )
        from ailm.docs_hydration import hydrate
        from ailm.manuscript_contract import resolve_manuscript_text
        from ailm.manuscript_variables import generate_manuscript_variables
        from ailm.reference_presentation import _load_reference_variables
        from ailm.supporting_package import (
            CONTEXT_SOURCES,
            DATA_SECTION,
            HtmlInventory,
            heading_id,
            provider_asset,
            relocate_protocol_html,
            source_view_markdown,
            style_source_view_html,
            validate_supporting_package,
        )

        claims = json.loads((_PROJECT_ROOT / "data/claims.json").read_text())["claims"]
        references = _load_reference_variables(_PROJECT_ROOT)
        variables = generate_manuscript_variables(_PROJECT_ROOT)
        context: dict[str, bytes] = {}
        for relative in sorted(CONTEXT_SOURCES):
            original = _PROJECT_ROOT / relative
            if original.is_symlink() or not original.is_file():
                raise ValueError(f"supporting context source is not an ordinary file: {relative}")
            content = original.read_bytes()
            if original.suffix == ".md":
                prose = content.decode("utf-8")
                if relative.startswith("docs/manuscript/"):
                    prose = resolve_manuscript_text(prose, variables)
                content = hydrate(prose, claims, reference_variables=references).encode("utf-8")
            context["context/" + relative] = content

        # Obtain the section ID from a real whole provider render. The delivered
        # source view contains the entire exact hydrated Markdown, with that ID
        # at the source heading; its links remain readable literal source context.
        data_source = staged / "docs" / "reference" / "data_statement.md"
        data_source.parent.mkdir()
        data_text = context["context/docs/reference/data_statement.md"].decode("utf-8")
        data_source.write_text(data_text, encoding="utf-8")
        data_page = renderer.render(data_source)
        data_id = heading_id(HtmlInventory(data_page.read_text(encoding="utf-8")), DATA_SECTION)
        data_source.write_text(
            source_view_markdown(data_text, DATA_SECTION, data_id), encoding="utf-8"
        )
        data_page = renderer.render(data_source)
        data_page.write_text(
            style_source_view_html(data_page.read_text(encoding="utf-8")), encoding="utf-8"
        )
        page.write_text(relocate_protocol_html(html, data_id, data_page.name), encoding="utf-8")
        for relative, content in context.items():
            target = page.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        expected = {
            file.relative_to(page.parent).as_posix(): file.read_bytes()
            for file in page.parent.rglob("*")
            if file.is_file()
        }
        # The provider's own assets travel with the pages that reference them:
        # every referenced asset, plus the resource files the provider emitted
        # in the same directories (a stylesheet's fonts, a loader's data). The
        # package validator below still requires each referenced asset to be an
        # ordinary file inside the package, so an undelivered asset fails closed.
        referenced_assets = {
            asset
            for delivered in (page, data_page)
            for target in HtmlInventory(delivered.read_text(encoding="utf-8")).targets
            if (asset := provider_asset(target)) is not None
        }
        asset_directories = {
            posixpath.dirname(asset) for asset in referenced_assets if posixpath.dirname(asset)
        }
        provider_assets = {
            name
            for name in expected
            if provider_asset(name) is not None
            and (name in referenced_assets or posixpath.dirname(name) in asset_directories)
        }
        expected_names = {page.name, data_page.name, "favicon.ico", *context, *provider_assets}
        if set(expected) != expected_names:
            raise RuntimeError(
                f"supporting renderer emitted unexpected artifacts: {sorted(expected)}"
            )
        validate_supporting_package(page.parent)
        for destination in output_dirs:
            stale = []
            for name, content in expected.items():
                target = destination / name
                if target.is_symlink() or (target.exists() and not target.is_file()):
                    raise ValueError(
                        f"supporting protocol target is not an ordinary file: {target}"
                    )
                if not target.exists() or target.read_bytes() != content:
                    stale.append(name)
            if check_only:
                if stale:
                    print(
                        f"ERROR: supporting protocol artifacts are stale: {stale}", file=sys.stderr
                    )
                    return 1
                validate_supporting_package(destination)
            else:
                for name in stale:
                    atomic_write_bytes(destination / name, expected[name])
                validate_supporting_package(destination)
    print("Supporting protocol HTML and stable anchors are current.")
    return 0


def _resolve_template_root(
    environment: Mapping[str, str] | None = None, *, project_root: Path | None = None
) -> Path | None:
    """Locate the template repo root that provides the ``infrastructure`` package.

    Order of preference:
    1. ``TEMPLATE_REPO_ROOT`` (set by the renderer — the authoritative answer)
    2. The repository's ``.template-root`` file, the per-machine declaration
       (``ailm.template_root``); like the environment value, it is final
    3. A walk-up from this file, for manual invocation from inside the template
    4. The ``template`` checkout beside the ``projects`` sidecar root, for the
       canonical ``<root>/template`` + ``<root>/projects/working/<name>`` layout
    5. The conventional home-directory checkout, for manual invocation elsewhere

    Returns:
        The directory containing ``infrastructure/``, or None if not found.
    """
    from ailm.template_root import configured_template_root

    root = _PROJECT_ROOT if project_root is None else project_root
    configured, _source = configured_template_root(root, environment)
    if configured is not None:
        # An explicitly supplied root (environment or ``.template-root``) is
        # authoritative.  Falling through from a typo to another checkout can
        # import a different renderer and make a failed render look successful.
        return configured if (configured / "infrastructure" / "__init__.py").is_file() else None

    # The candidate order is byte-identical in scripts/generate_figure_registry.py
    # and scripts/verify_release.py; tests/test_figure_registry.py holds them to it.
    candidates = list(root.parents)
    candidates.extend(parent / "template" for parent in root.parents[2:3])
    candidates.append(Path.home() / "Documents" / "GitHub" / "template")

    for candidate in candidates:
        if (candidate / "infrastructure" / "__init__.py").is_file():
            return candidate
    return None


def _canonical_variable_bytes(variables: Mapping[str, str]) -> bytes:
    """Return the only accepted byte representation of the variable map."""
    return (json.dumps(dict(variables), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_canonical_variable_map(path: Path) -> dict[str, str]:
    """Load one canonical variable map, rejecting duplicate keys and byte drift."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise ManuscriptPublicationError(
                    f"duplicate key {key!r} in manuscript variable map: {path}"
                )
            parsed[key] = value
        return parsed

    try:
        raw = path.read_bytes()
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManuscriptPublicationError(f"invalid manuscript variable map {path}: {exc}") from exc

    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ManuscriptPublicationError(
            f"manuscript variable map must be a JSON object of string values: {path}"
        )
    typed_value: dict[str, str] = value
    if raw != _canonical_variable_bytes(typed_value):
        raise ManuscriptPublicationError(
            f"manuscript variable map is valid JSON but not canonical bytes: {path}"
        )
    return typed_value


def _stage_variable_map(target: Path, variables: Mapping[str, str]) -> Path:
    """Write and validate a sibling staging file for the variable map."""
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".stage",
    )
    staged = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(
                handle.fileno(),
                target.stat().st_mode & 0o777 if target.is_file() else 0o644,
            )
            handle.write(_canonical_variable_bytes(variables))
            handle.flush()
            os.fsync(handle.fileno())
        if _load_canonical_variable_map(staged) != dict(variables):
            raise ManuscriptPublicationError("staged manuscript variable map changed while writing")
        return staged
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _reserve_absent_sibling(target: Path, label: str) -> Path:
    """Reserve, then release, a collision-resistant path beside *target*."""
    descriptor, raw_path = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=f".{label}",
    )
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def _stage_resolved_tree(
    project_root: Path,
    variables: Mapping[str, str],
    write_resolved_manuscript_tree: Callable[[Path, dict[str, str]], Path],
    validate_resolved_manuscript_tree: Callable[[Path, Mapping[str, str], Path], None],
) -> Path:
    """Build the complete resolved tree away from the live publication path."""
    output_root = project_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "manuscript"
    staged = _reserve_absent_sibling(target, "stage")

    try:
        with tempfile.TemporaryDirectory(
            dir=output_root,
            prefix=".manuscript-build.",
        ) as temporary_root:
            build_root = Path(temporary_root)
            shutil.copytree(project_root / "docs" / "manuscript", build_root / "manuscript")
            rendered = Path(write_resolved_manuscript_tree(build_root, dict(variables)))
            if not rendered.is_dir():
                raise ManuscriptPublicationError(
                    f"template writer did not produce a resolved manuscript tree: {rendered}"
                )
            # The generic writer copies auxiliary preamble bytes without injection.
            # Resolve from authored input before the exact-output contract runs.
            from ailm.manuscript_contract import resolve_manuscript_text

            preamble = project_root / "docs" / "manuscript" / "preamble.md"
            if preamble.is_file():
                (rendered / "preamble.md").write_text(
                    resolve_manuscript_text(preamble.read_text(encoding="utf-8"), variables),
                    encoding="utf-8",
                )
            from ailm.manuscript_contract import (
                ARTICLE_TYPOGRAPHY_NAME,
                article_typography_bytes,
                article_typography_required,
            )

            if article_typography_required(project_root):
                (rendered / ARTICLE_TYPOGRAPHY_NAME).write_bytes(
                    article_typography_bytes(project_root, variables)
                )
            os.replace(rendered, staged)
        validate_resolved_manuscript_tree(project_root, variables, staged)
        return staged
    except BaseException:
        if staged.is_dir() and not staged.is_symlink():
            shutil.rmtree(staged)
        elif staged.exists() or staged.is_symlink():
            staged.unlink()
        raise


def _remove_exact_path(path: Path) -> None:
    """Remove one known publication or staging path without following symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes before declaring the transaction complete."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_staged_artifacts(
    *,
    staged_json: Path,
    staged_tree: Path,
    target_json: Path,
    target_tree: Path,
    precommit_check: Callable[[], None],
) -> None:
    """Publish the staged pair and restore both prior artifacts on any failure.

    ``precommit_check`` runs after both replacements but before either backup is
    discarded.  Production uses it to re-read and revalidate the published
    pair; tests use the same seam to prove that a late failure rolls back both
    artifacts without a mocking framework.
    """
    if staged_json.parent != target_json.parent or staged_tree.parent != target_tree.parent:
        raise ManuscriptPublicationError("staged artifacts must be siblings of their targets")
    if not staged_json.is_file() or staged_json.is_symlink():
        raise ManuscriptPublicationError(
            f"staged variable map is not a regular file: {staged_json}"
        )
    if not staged_tree.is_dir() or staged_tree.is_symlink():
        raise ManuscriptPublicationError(f"staged manuscript is not a directory: {staged_tree}")
    if target_json.exists() and (not target_json.is_file() or target_json.is_symlink()):
        raise ManuscriptPublicationError(
            f"variable-map target is not a regular file: {target_json}"
        )
    if target_tree.exists() and (not target_tree.is_dir() or target_tree.is_symlink()):
        raise ManuscriptPublicationError(f"manuscript target is not a directory: {target_tree}")

    json_backup = _reserve_absent_sibling(target_json, "rollback")
    tree_backup = _reserve_absent_sibling(target_tree, "rollback")
    json_had_prior = target_json.exists()
    tree_had_prior = target_tree.exists()
    json_backed_up = False
    tree_backed_up = False
    json_published = False
    tree_published = False

    try:
        if json_had_prior:
            os.replace(target_json, json_backup)
            json_backed_up = True
        if tree_had_prior:
            os.replace(target_tree, tree_backup)
            tree_backed_up = True

        os.replace(staged_json, target_json)
        json_published = True
        os.replace(staged_tree, target_tree)
        tree_published = True
        _fsync_directory(target_json.parent)
        _fsync_directory(target_tree.parent)
        precommit_check()
    except BaseException as publication_error:
        rollback_errors: list[str] = []
        try:
            if tree_published:
                _remove_exact_path(target_tree)
            if tree_backed_up:
                os.replace(tree_backup, target_tree)
        except BaseException as exc:  # keep attempting the independent JSON rollback
            rollback_errors.append(f"tree: {exc}")
        try:
            if json_published:
                _remove_exact_path(target_json)
            if json_backed_up:
                os.replace(json_backup, target_json)
        except BaseException as exc:
            rollback_errors.append(f"variable map: {exc}")
        if not rollback_errors:
            _fsync_directory(target_json.parent)
            _fsync_directory(target_tree.parent)
            raise
        raise ManuscriptPublicationError(
            "manuscript publication failed and rollback was incomplete; "
            f"recoverable backups are {json_backup} and {tree_backup}; "
            f"rollback errors={rollback_errors}"
        ) from publication_error

    _remove_exact_path(json_backup)
    _remove_exact_path(tree_backup)
    _fsync_directory(target_json.parent)
    _fsync_directory(target_tree.parent)


def _publish_staged_variable_map(staged: Path, target: Path) -> None:
    """Atomically publish the map-only mode after canonical-byte validation."""
    if staged.parent != target.parent:
        raise ManuscriptPublicationError("staged variable map must be a sibling of its target")
    _load_canonical_variable_map(staged)
    os.replace(staged, target)
    _fsync_directory(target.parent)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate manuscript variables for active_inference_language_model"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Write output/data/manuscript_variables.json but skip tree injection "
        "(useful for inspecting tokens without a full render).",
    )
    args = parser.parse_args()

    try:
        _ensure_source_date_epoch()
    except ManuscriptPublicationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    from ailm.manuscript_contract import (
        ManuscriptContractError,
        validate_manuscript_sources,
        validate_no_hardcoded_manuscript_variables,
        validate_resolved_manuscript_tree,
    )
    from ailm.manuscript_variables import (
        ManuscriptVariableError,
        generate_manuscript_variables,
    )

    try:
        variables = generate_manuscript_variables(_PROJECT_ROOT)
        validate_manuscript_sources(_PROJECT_ROOT, variables)
        validate_no_hardcoded_manuscript_variables(_PROJECT_ROOT, variables)
    except (ManuscriptVariableError, ManuscriptContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    out_path = _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"
    if args.print_only:
        staged_json: Path | None = None
        try:
            staged_json = _stage_variable_map(out_path, variables)
            _publish_staged_variable_map(staged_json, out_path)
            staged_json = None
        except (ManuscriptPublicationError, OSError) as exc:
            print(f"ERROR: manuscript variable publication failed: {exc}", file=sys.stderr)
            return 1
        finally:
            if staged_json is not None:
                staged_json.unlink(missing_ok=True)
        print(str(out_path))
        return 0

    template_root = _resolve_template_root()
    if template_root is None:
        configured = os.environ.get("TEMPLATE_REPO_ROOT", "").strip()
        detail = (
            f"explicit TEMPLATE_REPO_ROOT is invalid: {configured}"
            if configured
            else "no template checkout was discovered"
        )
        print(
            f"ERROR: {detail}; could not locate the template repo root providing the "
            "'infrastructure' package. Set TEMPLATE_REPO_ROOT to your template "
            "checkout. Refusing to continue: without injection the manuscript "
            "would render with unresolved {{TOKEN}} markers, which read as "
            "fabricated results.",
            file=sys.stderr,
        )
        return 1
    sys.path.insert(0, str(template_root))

    try:
        from infrastructure.rendering.manuscript_injection import (
            write_resolved_manuscript_tree,
        )
    except ImportError as exc:
        print(
            f"ERROR: found a template root at {template_root} but could not import "
            f"infrastructure.rendering.manuscript_injection: {exc}",
            file=sys.stderr,
        )
        return 1

    staged_json = None
    staged_tree: Path | None = None
    try:
        staged_json = _stage_variable_map(out_path, variables)
        staged_tree = _stage_resolved_tree(
            _PROJECT_ROOT,
            variables,
            write_resolved_manuscript_tree,
            validate_resolved_manuscript_tree,
        )

        def validate_published_pair() -> None:
            if _load_canonical_variable_map(out_path) != variables:
                raise ManuscriptPublicationError(
                    "published manuscript variable map differs from the generated map"
                )
            validate_resolved_manuscript_tree(
                _PROJECT_ROOT,
                variables,
                _PROJECT_ROOT / "output" / "manuscript",
            )

        _publish_staged_artifacts(
            staged_json=staged_json,
            staged_tree=staged_tree,
            target_json=out_path,
            target_tree=_PROJECT_ROOT / "output" / "manuscript",
            precommit_check=validate_published_pair,
        )
        staged_json = None
        staged_tree = None
    except (ManuscriptContractError, ManuscriptPublicationError, OSError) as exc:
        print(f"ERROR: manuscript injection contract failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if staged_json is not None:
            staged_json.unlink(missing_ok=True)
        if staged_tree is not None:
            _remove_exact_path(staged_tree)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
