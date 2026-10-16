#!/usr/bin/env python3
"""Generate every AILM manuscript figure.

Thin orchestrator: it discovers the generators in :mod:`ailm.figures`, calls them, and
prints each written path to stdout for manifest collection. All computation — the
numbers, the styling, the layout, the legibility assertions — lives in ``src/ailm/``.
Nothing here decides what a figure says.

Discovery is dynamic rather than a hardcoded registry: every public ``generate_*``
function in every non-underscore module of ``ailm.figures`` is a figure. That way a new
figure module is picked up by adding it, with no second list to drift out of sync, and
modules owned by different authors compose without editing this file.

The figure modules use two shapes for a generator, and both are supported by one rule
rather than by a per-module special case:

- ``generate_x(...) -> Path`` — the generator writes its own PNG and returns the path.
- ``generate_x_figure(...) -> Figure`` — the generator builds the figure and this script
  writes it, under the stem ``x``, through the same ``_common.save_figure`` the writing
  generators use. The stem is the function name without the ``generate_`` prefix and the
  ``_figure`` suffix, which is exactly the name each module's own ``write_*_figures``
  helper uses, so neither convention produces a differently-named file.

Module-level ``write_*_figures`` aggregates are deliberately *not* discovered: they would
re-generate figures this script has already produced individually, and they cannot be
selected one at a time with ``--only``.

Ordering is deterministic — modules sorted by name, functions sorted within a module —
so the stdout manifest is stable across runs. Generators write into a temporary sibling
directory. Only after every selected generator succeeds and the complete bounded code,
data/spec inputs, and PNG bytes have been authenticated is each PNG atomically replaced
and the v3 provenance receipt published last. The receipt is a fail-detect commit marker,
not a batch filesystem transaction: an interrupted multi-file publication is rejected on
the next validation because its old receipt cannot authenticate the mixed PNG set.
Consequently stdout contains final paths only; a failed generation or publication emits
no manifest paths.

Usage
-----
    MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
    MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only perplexity_ladder
    MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only results --list

Exit status is non-zero if any figure fails, if ``--only`` matches nothing, or if a
figure module cannot be imported. A missing figure must break the build rather than
leave the manuscript referencing a PNG that is not there.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import pkgutil
import stat
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

# Headless rendering must be settled before matplotlib is imported anywhere down the
# import chain, so it is set here rather than left to the caller's environment.
os.environ.setdefault("MPLBACKEND", "Agg")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts._artifact_io import (  # noqa: E402
    AuthenticatedFileError,
    read_strict_json_regular_file,
    sha256_authenticated_regular_file,
)

_PREFIX = "generate_"
_BUILDER_SUFFIX = "_figure"


def _figure_name(function_name: str) -> str:
    """Derive the output stem from a generator's name.

    ``generate_perplexity_ladder`` → ``perplexity_ladder`` and
    ``generate_hierarchy_figure`` → ``hierarchy``, matching the stems each module's own
    ``write_*_figures`` helper writes.
    """
    stem = function_name[len(_PREFIX) :]
    if stem.endswith(_BUILDER_SUFFIX) and len(stem) > len(_BUILDER_SUFFIX):
        stem = stem[: -len(_BUILDER_SUFFIX)]
    return stem


class Generator(NamedTuple):
    """One discovered figure generator."""

    name: str
    module: str
    call: Callable[..., Any]


def discover_generators(package_name: str = "ailm.figures") -> list[Generator]:
    """Find every ``generate_*`` function in every module of the figures package.

    Args:
        package_name: Dotted package to scan.

    Returns:
        Generators sorted by ``(module, name)``.

    Raises:
        ImportError: If the package or one of its modules cannot be imported. A figure
            module that fails to import is a build failure, not a figure to skip.
    """
    package = importlib.import_module(package_name)
    found: list[Generator] = []

    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{info.name}")
        for attr in sorted(vars(module)):
            if not attr.startswith(_PREFIX):
                continue
            candidate = getattr(module, attr)
            # Only functions defined in this module — a re-exported generator would
            # otherwise be run once per module that imports it.
            if not callable(candidate) or getattr(candidate, "__module__", "") != (
                f"{package_name}.{info.name}"
            ):
                continue
            found.append(Generator(name=_figure_name(attr), module=info.name, call=candidate))
    return found


def _matches(generator: Generator, selectors: list[str]) -> bool:
    """Whether a generator is selected by any ``--only`` value (name or module)."""
    return generator.name in selectors or generator.module in selectors


def _invoke(generator: Generator, output_dir: Path | None) -> list[Path]:
    """Call one generator and return the path(s) it produced.

    Accepts either shape of generator: one that writes its own PNG and returns a path, or
    one that returns an unwritten ``Figure``, which is written here under the generator's
    derived stem. ``output_dir`` is forwarded only to generators that accept it.

    Args:
        generator: A discovered generator.
        output_dir: Destination directory, or None for each generator's own default.

    Returns:
        Paths written, in the order produced.

    Raises:
        TypeError: If the generator returns something that is neither a path, a list of
            paths, nor a matplotlib ``Figure``.
    """
    from matplotlib.figure import Figure

    from ailm.figures._common import save_figure

    parameters = inspect.signature(generator.call).parameters
    if output_dir is not None and "output_dir" in parameters:
        result = generator.call(output_dir=output_dir)
    else:
        result = generator.call()

    if isinstance(result, Figure):
        import matplotlib.pyplot as plt

        try:
            return [save_figure(result, generator.name, output_dir)]
        finally:
            plt.close(result)
    if isinstance(result, (str, Path)):
        return [Path(result)]
    if isinstance(result, (list, tuple)):
        return [Path(item) for item in result]
    raise TypeError(
        f"{generator.module}.{_PREFIX}{generator.name} returned "
        f"{type(result).__name__}; a figure generator must return a Path, a list of "
        "Paths, or an unwritten matplotlib Figure."
    )


_PROVENANCE_SCHEMA = "ailm-figure-provenance-v3"
_PROVENANCE_NAME = "figure_provenance.json"
_MANUSCRIPT_VARIABLES_PATH = "output/data/manuscript_variables.json"
_SOURCE_DIRECTORY = Path("src/ailm/figures")
_SHARED_SOURCE_PATHS = (
    Path("scripts/generate_figures.py"),
    Path("scripts/_artifact_io.py"),
    Path("src/ailm/figures/_common.py"),
    Path("src/ailm/viz_config.py"),
)
_CODE_DEPENDENCY_ROOT = Path("src/ailm")
_COMPUTATION_INPUT_ROOTS = (
    (Path("data"), frozenset({".json", ".npz", ".yaml", ".yml", ".png"})),
    (Path("src/ailm/dashboard/static"), frozenset({".html", ".css", ".js"})),
    (Path("vendor"), frozenset({".json", ".npz", ".yaml", ".yml"})),
    (Path("gnn/models"), frozenset({".md"})),
    (Path("gnn/mutations"), frozenset({".md"})),
)
_SHA256_HEXDIGITS = frozenset("0123456789abcdef")


class FigureProvenanceError(RuntimeError):
    """A figure build cannot establish an authenticated publication state."""


def _read_json(path: Path, *, description: str) -> object:
    """Read strict JSON from the same ordinary-file descriptor that is hashed."""
    try:
        payload, _digest = read_strict_json_regular_file(path, description=description)
    except (AuthenticatedFileError, FileNotFoundError) as error:
        raise FigureProvenanceError(str(error)) from error
    return payload


def _read_json_with_digest(path: Path, *, description: str) -> tuple[object, str]:
    """Read and hash strict JSON once, through one authenticated descriptor."""
    try:
        return read_strict_json_regular_file(path, description=description)
    except (AuthenticatedFileError, FileNotFoundError) as error:
        raise FigureProvenanceError(str(error)) from error


def _sha256_regular_file(path: Path, *, description: str) -> str:
    """Hash one ordinary file without following or blocking on special files."""
    try:
        return sha256_authenticated_regular_file(path, description=description)
    except (AuthenticatedFileError, FileNotFoundError) as error:
        raise FigureProvenanceError(str(error)) from error


def _valid_sha256(value: object) -> bool:
    """Whether *value* is one canonical lowercase SHA-256 hex digest."""
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEXDIGITS


def _manuscript_variables_record() -> dict[str, str]:
    """Authenticate the one canonical token-map input consumed by figures."""
    from ailm.figures._common import manuscript_variables_path

    target = manuscript_variables_path()
    payload, digest = _read_json_with_digest(
        target, description="canonical manuscript variable map"
    )
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()
    ):
        raise FigureProvenanceError(
            f"{target}: canonical manuscript variable map must be a JSON object "
            "whose keys and values are strings"
        )
    return {
        "path": _MANUSCRIPT_VARIABLES_PATH,
        "sha256": digest,
    }


def _generator_source_record(generator: Generator) -> dict[str, str]:
    """Bind a discovered generator to its canonical checked-in source bytes."""
    raw_source = inspect.getsourcefile(generator.call)
    if raw_source is None:
        raise FigureProvenanceError(
            f"{generator.name}: cannot locate source for ailm.figures.{generator.module}"
        )
    source = Path(raw_source)
    if source.is_symlink():
        raise FigureProvenanceError(f"{source}: generator source must not be a symlink")
    try:
        relative = source.resolve(strict=True).relative_to(_PROJECT_ROOT.resolve(strict=True))
    except (FileNotFoundError, ValueError) as error:
        raise FigureProvenanceError(
            f"{source}: generator source is outside the AILM checkout"
        ) from error
    expected = _SOURCE_DIRECTORY / f"{generator.module}.py"
    if relative != expected:
        raise FigureProvenanceError(
            f"{generator.name}: source resolved to {relative.as_posix()!r}, expected "
            f"{expected.as_posix()!r}"
        )
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_regular_file(source, description="generator source"),
    }


def _generator_sources(generators: list[Generator]) -> dict[str, dict[str, str]]:
    """Snapshot every discovered generator source, rejecting ambiguous names."""
    records: dict[str, dict[str, str]] = {}
    for generator in generators:
        if generator.name in records:
            raise FigureProvenanceError(
                f"duplicate figure name {generator.name!r}; provenance keys must be unique"
            )
        records[generator.name] = _generator_source_record(generator)
    return records


def _shared_source_records(project_root: Path = _PROJECT_ROOT) -> list[dict[str, str]]:
    """Authenticate the bounded shared rendering code every generator depends on."""
    records: list[dict[str, str]] = []
    for relative in _SHARED_SOURCE_PATHS:
        target = project_root / relative
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256_regular_file(target, description="shared rendering source"),
            }
        )
    return records


def _code_dependency_paths(project_root: Path = _PROJECT_ROOT) -> list[Path]:
    """Return every project Python source that can enter a figure computation.

    The inventory intentionally covers all of ``src/ailm`` rather than trying
    to keep a hand-written import list. Figure generators contain local imports,
    and their transitive model imports evolve; an over-inclusive rebuild is safe,
    while an omitted computation module permits a mixed-code selective receipt.
    Directory and ``.py`` symlinks are rejected rather than silently skipped.
    """
    source_root = project_root / _CODE_DEPENDENCY_ROOT
    try:
        root_metadata = source_root.lstat()
    except OSError as error:
        raise FigureProvenanceError(
            f"{source_root}: cannot inspect the AILM code dependency root"
        ) from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise FigureProvenanceError(
            f"{source_root}: AILM code dependency root must be a real directory"
        )

    def walk_error(error: OSError) -> None:
        raise FigureProvenanceError(
            f"{source_root}: cannot enumerate AILM code dependencies: {error}"
        ) from error

    paths: list[Path] = []
    for raw_directory, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False, onerror=walk_error
    ):
        directory = Path(raw_directory)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = directory / name
            try:
                metadata = child.lstat()
            except OSError as error:
                raise FigureProvenanceError(
                    f"{child}: cannot inspect code dependency directory"
                ) from error
            if not stat.S_ISDIR(metadata.st_mode):
                raise FigureProvenanceError(
                    f"{child}: code dependency directories must not be symlinks or special files"
                )
        for name in file_names:
            if not name.endswith(".py"):
                continue
            paths.append((directory / name).relative_to(project_root))
    if not paths:
        raise FigureProvenanceError(f"{source_root}: no AILM Python sources were found")
    return sorted(paths, key=lambda path: path.as_posix())


def _code_dependency_records(project_root: Path = _PROJECT_ROOT) -> list[dict[str, str]]:
    """Authenticate the complete deterministic AILM Python source inventory."""
    return [
        {
            "path": relative.as_posix(),
            "sha256": _sha256_regular_file(
                project_root / relative, description="AILM code dependency"
            ),
        }
        for relative in _code_dependency_paths(project_root)
    ]


def _computation_input_paths(project_root: Path = _PROJECT_ROOT) -> list[Path]:
    """Return the complete bounded inventory of non-code figure inputs.

    The source trees are intentionally broader than the current set of imports:
    selective reuse may over-invalidate, but it must never preserve a PNG made
    from old corpus, measurement, agreement, or GNN-spec bytes.
    """
    paths: list[Path] = []
    for relative_root, suffixes in _COMPUTATION_INPUT_ROOTS:
        source_root = project_root / relative_root
        try:
            root_metadata = source_root.lstat()
        except OSError as error:
            raise FigureProvenanceError(
                f"{source_root}: cannot inspect computation-input root"
            ) from error
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise FigureProvenanceError(
                f"{source_root}: computation-input root must be a real directory"
            )

        def walk_error(error: OSError, *, root: Path = source_root) -> None:
            raise FigureProvenanceError(
                f"{root}: cannot enumerate computation inputs: {error}"
            ) from error

        for raw_directory, directory_names, file_names in os.walk(
            source_root, topdown=True, followlinks=False, onerror=walk_error
        ):
            directory = Path(raw_directory)
            directory_names.sort()
            file_names.sort()
            for name in directory_names:
                child = directory / name
                try:
                    metadata = child.lstat()
                except OSError as error:
                    raise FigureProvenanceError(
                        f"{child}: cannot inspect computation-input directory"
                    ) from error
                if not stat.S_ISDIR(metadata.st_mode):
                    raise FigureProvenanceError(
                        f"{child}: computation-input directories must not be symlinks "
                        "or special files"
                    )
            for name in file_names:
                candidate = directory / name
                if candidate.suffix.lower() in suffixes:
                    paths.append(candidate.relative_to(project_root))
    if not paths:
        raise FigureProvenanceError("no bounded figure computation inputs were found")
    paths.append(Path("output/web/ailm.html"))
    return sorted(paths, key=lambda path: path.as_posix())


def _computation_input_records(project_root: Path = _PROJECT_ROOT) -> list[dict[str, str]]:
    """Authenticate the deterministic non-code input inventory."""
    return [
        {
            "path": relative.as_posix(),
            "sha256": _sha256_regular_file(
                project_root / relative, description="figure computation input"
            ),
        }
        for relative in _computation_input_paths(project_root)
    ]


def _safe_output_name(value: object) -> bool:
    """Whether *value* is a portable PNG basename beside the receipt."""
    return (
        isinstance(value, str)
        and value.endswith(".png")
        and value not in {".png", "..png"}
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
    )


def _published_png_names(directory: Path) -> set[str]:
    """Return every published PNG basename, rejecting non-regular entries."""
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return set()
    except OSError as error:
        raise FigureProvenanceError(
            f"{directory}: cannot inspect published figure outputs"
        ) from error

    names: set[str] = set()
    for entry in entries:
        if entry.suffix.lower() != ".png":
            continue
        try:
            metadata = entry.lstat()
        except OSError as error:
            raise FigureProvenanceError(f"{entry}: cannot inspect figure output") from error
        if not stat.S_ISREG(metadata.st_mode):
            raise FigureProvenanceError(f"{entry}: published figure output is not a regular file")
        names.add(entry.name)
    return names


def _recorded_output_names(records: Mapping[str, Mapping[str, object]]) -> set[str]:
    """Collect the portable output names in already validated records."""
    names: set[str] = set()
    for record in records.values():
        outputs = record.get("outputs")
        if not isinstance(outputs, list):
            raise FigureProvenanceError("validated figure record lost its output list")
        for output in outputs:
            if not isinstance(output, dict):
                raise FigureProvenanceError("validated figure record has a non-object output")
            name = output.get("path")
            if not isinstance(name, str):
                raise FigureProvenanceError("validated figure record has a non-string output path")
            names.add(name)
    return names


def _environment_summary() -> dict[str, object]:
    """The exact numeric environment the figures were computed in.

    Recorded so "the figures ran on real JAX/pymdp, on this backend, at these
    versions" is a checkable artifact rather than an assumption. The pins live
    in ``pyproject.toml``; ``tests/test_figure_provenance.py`` holds this
    summary against them.
    """
    import importlib.metadata as importlib_metadata

    import jax
    import matplotlib
    import numpy

    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": importlib_metadata.version("jaxlib"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "pymdp": importlib_metadata.version("inferactively-pymdp"),
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _validated_previous_records(
    directory: Path,
    environment: Mapping[str, object],
    manuscript_variables: Mapping[str, str],
    generators: Mapping[str, Generator],
    generator_sources: Mapping[str, Mapping[str, str]],
    shared_sources: list[dict[str, str]],
    code_dependencies: list[dict[str, str]],
    computation_inputs: list[dict[str, str]],
) -> dict[str, dict[str, object]]:
    """Authenticate every input and output a selective build would preserve.

    A malformed receipt must not be treated as an empty one: doing so lets
    ``--only`` erase the provenance for every figure it did not regenerate.
    Likewise, a receipt from another numeric environment cannot be relabeled
    as if all of its untouched figures were produced by the current stack. The
    validation happens before the staging directory is created, so a stale
    input, changed generator, or mutated PNG cannot touch published output.
    """
    target = directory / _PROVENANCE_NAME
    try:
        target.lstat()
    except FileNotFoundError:
        unreceipted = _published_png_names(directory)
        if unreceipted:
            raise FigureProvenanceError(
                f"{directory}: selective build found PNGs without a provenance receipt: "
                f"{sorted(unreceipted)}; run a full figure build"
            ) from None
        return {}
    except OSError as error:
        raise FigureProvenanceError(f"{target}: cannot inspect existing provenance") from error
    try:
        payload = _read_json(target, description="existing figure provenance")
    except FigureProvenanceError as error:
        raise FigureProvenanceError(
            f"{target}: existing provenance is unreadable; run a full figure build "
            "instead of merging a selective build"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema",
            "environment",
            "manuscript_variables",
            "shared_sources",
            "code_dependencies",
            "computation_inputs",
            "figures",
        }
        or payload.get("schema") != _PROVENANCE_SCHEMA
    ):
        raise FigureProvenanceError(
            f"{target}: existing provenance has an unsupported schema; run a full build"
        )
    if payload.get("environment") != dict(environment):
        raise FigureProvenanceError(
            f"{target}: existing figures were produced by a different numeric environment; "
            "run a full build so the global environment remains truthful"
        )
    recorded_variables = payload.get("manuscript_variables")
    if (
        not isinstance(recorded_variables, dict)
        or set(recorded_variables) != {"path", "sha256"}
        or recorded_variables.get("path") != _MANUSCRIPT_VARIABLES_PATH
        or not _valid_sha256(recorded_variables.get("sha256"))
    ):
        raise FigureProvenanceError(
            f"{target}: malformed canonical manuscript-variable input record"
        )
    if recorded_variables != dict(manuscript_variables):
        raise FigureProvenanceError(
            f"{target}: canonical manuscript variable map digest changed since the "
            "receipt was written; run a full figure build"
        )
    recorded_shared_sources = payload.get("shared_sources")
    if not isinstance(recorded_shared_sources, list) or any(
        not isinstance(source, dict)
        or set(source) != {"path", "sha256"}
        or not isinstance(source.get("path"), str)
        or not _valid_sha256(source.get("sha256"))
        for source in recorded_shared_sources
    ):
        raise FigureProvenanceError(f"{target}: malformed shared rendering source records")
    if recorded_shared_sources != shared_sources:
        raise FigureProvenanceError(
            f"{target}: shared rendering source path or digest changed since the receipt "
            "was written; run a full figure build"
        )
    recorded_code_dependencies = payload.get("code_dependencies")
    if not isinstance(recorded_code_dependencies, list) or any(
        not isinstance(source, dict)
        or set(source) != {"path", "sha256"}
        or not isinstance(source.get("path"), str)
        or not _valid_sha256(source.get("sha256"))
        for source in recorded_code_dependencies
    ):
        raise FigureProvenanceError(f"{target}: malformed AILM code dependency records")
    if recorded_code_dependencies != code_dependencies:
        raise FigureProvenanceError(
            f"{target}: AILM code dependency inventory or digest changed since the receipt "
            "was written; run a full figure build"
        )
    recorded_computation_inputs = payload.get("computation_inputs")
    if not isinstance(recorded_computation_inputs, list) or any(
        not isinstance(source, dict)
        or set(source) != {"path", "sha256"}
        or not isinstance(source.get("path"), str)
        or not _valid_sha256(source.get("sha256"))
        for source in recorded_computation_inputs
    ):
        raise FigureProvenanceError(f"{target}: malformed computation-input records")
    if recorded_computation_inputs != computation_inputs:
        raise FigureProvenanceError(
            f"{target}: figure computation-input inventory or digest changed since the "
            "receipt was written; run a full figure build"
        )
    figures = payload.get("figures")
    if not isinstance(figures, dict):
        raise FigureProvenanceError(f"{target}: provenance 'figures' must be an object")

    validated: dict[str, dict[str, object]] = {}
    claimed_outputs: set[str] = set()
    for name, record in figures.items():
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise FigureProvenanceError(f"{target}: unsafe figure key {name!r}")
        if name not in generators or name not in generator_sources:
            raise FigureProvenanceError(
                f"{target}: figure {name!r} has no currently discovered generator"
            )
        if not isinstance(record, dict) or set(record) != {"module", "source", "outputs"}:
            raise FigureProvenanceError(f"{target}: malformed record for figure {name!r}")
        module = record["module"]
        source = record["source"]
        outputs = record["outputs"]
        if module != generators[name].module:
            raise FigureProvenanceError(f"{target}: generator module changed for figure {name!r}")
        if (
            not isinstance(source, dict)
            or set(source) != {"path", "sha256"}
            or not isinstance(source.get("path"), str)
            or not _valid_sha256(source.get("sha256"))
        ):
            raise FigureProvenanceError(
                f"{target}: malformed generator source record for figure {name!r}"
            )
        if source != dict(generator_sources[name]):
            raise FigureProvenanceError(
                f"{target}: generator source path or digest changed for figure {name!r}; "
                "run a full figure build"
            )
        if (
            not isinstance(outputs, list)
            or not outputs
            or any(
                not isinstance(output, dict)
                or set(output) != {"path", "sha256"}
                or not _safe_output_name(output.get("path"))
                or not _valid_sha256(output.get("sha256"))
                for output in outputs
            )
        ):
            raise FigureProvenanceError(f"{target}: invalid outputs for figure {name!r}")

        validated_outputs: list[dict[str, str]] = []
        local_outputs: set[str] = set()
        for output in outputs:
            output_name = output["path"]
            expected_digest = output["sha256"]
            if not isinstance(output_name, str) or not isinstance(expected_digest, str):
                raise FigureProvenanceError(
                    f"{target}: validated output record changed type for figure {name!r}"
                )
            if output_name in local_outputs or output_name in claimed_outputs:
                raise FigureProvenanceError(
                    f"{target}: duplicate output path {output_name!r} in provenance"
                )
            actual_digest = _sha256_regular_file(
                directory / output_name, description=f"figure output {output_name}"
            )
            if actual_digest != expected_digest:
                raise FigureProvenanceError(
                    f"{directory / output_name}: figure output digest changed since the "
                    "receipt was written; run a full figure build"
                )
            local_outputs.add(output_name)
            claimed_outputs.add(output_name)
            validated_outputs.append({"path": output_name, "sha256": expected_digest})
        validated[name] = {
            "module": module,
            "source": dict(source),
            "outputs": validated_outputs,
        }
    published_outputs = _published_png_names(directory)
    if published_outputs != claimed_outputs:
        raise FigureProvenanceError(
            f"{target}: receipt/output inventory mismatch; claimed={sorted(claimed_outputs)}, "
            f"published={sorted(published_outputs)}; run a full figure build"
        )
    return validated


def _provenance_bytes(
    environment: Mapping[str, object],
    manuscript_variables: Mapping[str, str],
    shared_sources: list[dict[str, str]],
    code_dependencies: list[dict[str, str]],
    computation_inputs: list[dict[str, str]],
    records: dict[str, dict[str, object]],
    *,
    previous: Mapping[str, dict[str, object]] | None = None,
) -> tuple[bytes, dict[str, dict[str, object]]]:
    """Serialize a portable deterministic receipt and return its merged records.

    A full run rewrites the file from scratch; an ``--only`` run merges its
    records into the existing entries so a selective regeneration does not
    erase the provenance of figures it did not touch.
    """
    figures: dict[str, dict[str, object]] = dict(previous or {})
    figures.update(records)
    payload = {
        "schema": _PROVENANCE_SCHEMA,
        "environment": dict(environment),
        "manuscript_variables": dict(manuscript_variables),
        "shared_sources": list(shared_sources),
        "code_dependencies": list(code_dependencies),
        "computation_inputs": list(computation_inputs),
        "figures": dict(sorted(figures.items())),
    }
    return (
        (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        figures,
    )


def _staged_outputs(stage: Path, paths: list[Path]) -> list[dict[str, str]]:
    """Authenticate one generator's returned files inside the staging directory."""
    if not paths:
        raise FigureProvenanceError("figure generator returned no output paths")
    outputs: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        if not _safe_output_name(path.name) or path.parent.resolve() != stage.resolve():
            raise FigureProvenanceError(
                f"generator returned {path}, which is outside the figure staging directory"
            )
        if path.name in seen:
            raise FigureProvenanceError(f"generator returned duplicate output {path.name!r}")
        seen.add(path.name)
        outputs.append(
            {
                "path": path.name,
                "sha256": _sha256_regular_file(path, description="staged figure output"),
            }
        )
    return outputs


def _validate_destination(directory: Path) -> None:
    """Reject a pre-existing publication path that is not an ordinary directory."""
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise FigureProvenanceError(f"{directory}: cannot inspect output directory") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise FigureProvenanceError(
            f"{directory}: figure output destination must be a real directory, not a symlink"
        )


def _assert_inputs_unchanged(
    manuscript_variables: Mapping[str, str],
    generators: list[Generator],
    generator_sources: Mapping[str, Mapping[str, str]],
    shared_sources: list[dict[str, str]],
    code_dependencies: list[dict[str, str]],
    computation_inputs: list[dict[str, str]],
) -> None:
    """Re-authenticate inputs after rendering and before publication."""
    if _manuscript_variables_record() != dict(manuscript_variables):
        raise FigureProvenanceError(
            "canonical manuscript variable map changed while figures were rendering"
        )
    if _shared_source_records() != shared_sources:
        raise FigureProvenanceError("shared rendering source changed while figures were rendering")
    if _code_dependency_records() != code_dependencies:
        raise FigureProvenanceError("AILM code dependency changed while figures were rendering")
    if _computation_input_records() != computation_inputs:
        raise FigureProvenanceError("figure computation input changed while figures were rendering")
    for generator in generators:
        if _generator_source_record(generator) != dict(generator_sources[generator.name]):
            raise FigureProvenanceError(
                f"generator source changed while {generator.name!r} was rendering"
            )


def _publish(
    stage: Path,
    directory: Path,
    records: Mapping[str, dict[str, object]],
    receipt: bytes,
) -> list[Path]:
    """Atomically replace each PNG, then publish the receipt as the commit marker."""
    from scripts._artifact_io import atomic_write_bytes

    _validate_destination(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final_paths: list[Path] = []
    for record in records.values():
        outputs = record.get("outputs")
        if not isinstance(outputs, list):
            raise FigureProvenanceError("publication record lost its output list")
        for output in outputs:
            if not isinstance(output, dict):
                raise FigureProvenanceError("publication record has a non-object output")
            name = output.get("path")
            expected_digest = output.get("sha256")
            if not isinstance(name, str) or not isinstance(expected_digest, str):
                raise FigureProvenanceError(
                    "publication record has an invalid output path or digest"
                )
            staged = stage / name
            content = staged.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected_digest:
                raise FigureProvenanceError(
                    f"{staged}: staged output changed before atomic publication"
                )
            final = directory / name
            atomic_write_bytes(final, content)
            if (
                _sha256_regular_file(final, description="published figure output")
                != expected_digest
            ):
                raise FigureProvenanceError(
                    f"{final}: atomic publication did not preserve the staged bytes"
                )
            final_paths.append(final)
    atomic_write_bytes(directory / _PROVENANCE_NAME, receipt)
    return final_paths


def main(argv: list[str] | None = None) -> int:
    """Run the selected generators.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status: 0 when every selected figure was written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help="Figure name or module to generate. Repeatable; comma-separated accepted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to each generator's own output/figures/.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the discovered figures and exit without generating anything.",
    )
    args = parser.parse_args(argv)

    selectors = [
        part.strip() for value in args.only for part in str(value).split(",") if part.strip()
    ]

    try:
        generators = discover_generators()
    except ImportError:
        traceback.print_exc()
        print(
            "FAILED: could not import ailm.figures. The figure modules are part of "
            "the build, so this is a hard failure.",
            file=sys.stderr,
        )
        return 1

    if not generators:
        print("FAILED: no generate_* functions found in ailm.figures.", file=sys.stderr)
        return 1

    selected = [g for g in generators if _matches(g, selectors)] if selectors else generators

    if selectors and not selected:
        print(
            f"FAILED: --only {selectors} matched nothing. Available figures: "
            f"{', '.join(g.name for g in generators)}",
            file=sys.stderr,
        )
        return 1

    if args.list:
        for generator in selected:
            print(f"{generator.name}\t({generator.module})", file=sys.stderr)
        return 0

    environment = _environment_summary()
    print(
        "environment: jax {jax} ({jax_backend}), jaxlib {jaxlib}, pymdp {pymdp}, "
        "numpy {numpy}, matplotlib {matplotlib}".format(**environment),
        file=sys.stderr,
    )

    from ailm.figures._common import default_output_dir

    provenance_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    provenance_dir = provenance_dir.expanduser().absolute()
    try:
        _validate_destination(provenance_dir)
        manuscript_variables = _manuscript_variables_record()
        generators_by_name = {generator.name: generator for generator in generators}
        source_records = _generator_sources(generators)
        shared_sources = _shared_source_records()
        code_dependencies = _code_dependency_records()
        computation_inputs = _computation_input_records()
        previous_records: dict[str, dict[str, object]] = {}
        if selectors:
            previous_records = _validated_previous_records(
                provenance_dir,
                environment,
                manuscript_variables,
                generators_by_name,
                source_records,
                shared_sources,
                code_dependencies,
                computation_inputs,
            )
    except FigureProvenanceError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1

    # The staging directory is a sibling of the final directory: every
    # generator can finish (or fail) without changing a published PNG. Only a
    # completely authenticated stage reaches the atomic publication boundary.
    provenance_dir.parent.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    records: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(
        prefix=f".{provenance_dir.name}.stage-", dir=provenance_dir.parent
    ) as raw_stage:
        stage = Path(raw_stage)
        claimed_outputs: set[str] = set()
        for generator in selected:
            started = time.monotonic()
            try:
                paths = _invoke(generator, stage)
                outputs = _staged_outputs(stage, paths)
                output_names = {output["path"] for output in outputs}
                duplicates = output_names & claimed_outputs
                if duplicates:
                    raise FigureProvenanceError(
                        f"multiple generators returned the same outputs: {sorted(duplicates)}"
                    )
                claimed_outputs.update(output_names)
            except Exception:
                traceback.print_exc()
                print(
                    f"FAILED {generator.name} ({generator.module})",
                    file=sys.stderr,
                )
                failures.append(generator.name)
                continue
            wall = time.monotonic() - started
            records[generator.name] = {
                "module": generator.module,
                "source": dict(source_records[generator.name]),
                "outputs": outputs,
            }
            print(f"  {generator.name}: {wall:.2f}s", file=sys.stderr)

        staged_names = {
            path.name
            for path in stage.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        }
        if staged_names != claimed_outputs:
            failures.append("unreported_staged_outputs")
            print(
                "FAILED: staged PNGs differ from generator-returned outputs: "
                f"staged={sorted(staged_names)}, returned={sorted(claimed_outputs)}",
                file=sys.stderr,
            )

        if failures:
            print(
                f"{len(selected) - len(failures)}/{len(selected)} figures staged; "
                f"published 0; failed: {', '.join(failures)}",
                file=sys.stderr,
            )
            return 1

        try:
            _assert_inputs_unchanged(
                manuscript_variables,
                selected,
                source_records,
                shared_sources,
                code_dependencies,
                computation_inputs,
            )
            if selectors:
                # Detect a concurrent edit after the first preflight. A selective
                # build must merge the exact state it authenticated, not merely
                # another independently valid state.
                current_previous = _validated_previous_records(
                    provenance_dir,
                    environment,
                    manuscript_variables,
                    generators_by_name,
                    source_records,
                    shared_sources,
                    code_dependencies,
                    computation_inputs,
                )
                if current_previous != previous_records:
                    raise FigureProvenanceError(
                        "existing provenance changed while figures were rendering"
                    )
            receipt, merged_records = _provenance_bytes(
                environment,
                manuscript_variables,
                shared_sources,
                code_dependencies,
                computation_inputs,
                records,
                previous=previous_records if selectors else None,
            )
            final_inventory = _published_png_names(provenance_dir) | _recorded_output_names(records)
            expected_inventory = _recorded_output_names(merged_records)
            if final_inventory != expected_inventory:
                raise FigureProvenanceError(
                    "publication would leave PNGs outside the provenance receipt; "
                    f"future={sorted(final_inventory)}, receipt={sorted(expected_inventory)}"
                )
            final_paths = _publish(stage, provenance_dir, records, receipt)
            authenticated = _validated_previous_records(
                provenance_dir,
                environment,
                manuscript_variables,
                generators_by_name,
                source_records,
                shared_sources,
                code_dependencies,
                computation_inputs,
            )
            if authenticated != merged_records:
                raise FigureProvenanceError(
                    "published provenance did not round-trip to the authenticated records"
                )
        except (FigureProvenanceError, OSError) as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1

    # stdout remains the final manifest: one final absolute path per line, no
    # temporary staging paths and no path from a failed publication.
    for path in final_paths:
        print(path)
    provenance = provenance_dir / _PROVENANCE_NAME
    print(f"provenance: {provenance}", file=sys.stderr)
    print(f"{len(selected)}/{len(selected)} figures written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
