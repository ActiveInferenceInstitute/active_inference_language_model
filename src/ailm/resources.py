"""Resolve committed and downloaded corpus data across installation modes.

The canonical copies live in the repository's top-level ``vendor/`` directory.
Selected runtime artifacts are embedded under ``ailm/_vendor/`` when Hatch builds
the wheel.  Loaders use :func:`resolve_vendored_artifact` rather than deriving a
repository root from ``__file__``; that derivation points outside ``site-packages``
after installation and made every default loader fail. Registered connected-runtime
inputs retain their source paths under ``data/`` and ship under ``ailm/_data/``.

Raw corpora are intentionally not embedded in the wheel.  A source checkout keeps
its historical ``vendor/`` download destination, while an installed distribution
uses a writable per-user cache.  ``AILM_CORPUS_CACHE_DIR`` is the explicit override
for both modes.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TypeAlias

VendorResource: TypeAlias = Path | Traversable

#: Complete committed JSON resource surface shipped in the wheel. Raw corpus
#: downloads and staging directories are intentionally absent: loaders that
#: acquire those inputs retain their explicit download/path contracts, while
#: every derived artifact needed by shipped runtime and analysis code is here.
RUNTIME_VENDOR_ARTIFACTS: tuple[str, ...] = (
    "brown_clause_word_counts.json",
    "brown_counts.json",
    "brown_heldout_stream.json",
    "brown_online_drift.json",
    "brown_word_counts.json",
    "brown_word_counts_full.json",
    "cmudict_counts.json",
    "synthetic_aligned_g2p_counts.json",
    "synthetic_counts.json",
    "synthetic_phone_counts.json",
    "ud_czech_case_counts.json",
    "ud_ewt_case_counts.json",
    "ud_ewt_clause_counts.json",
    "ud_ewt_narrative_counts.json",
)

#: Canonical development-artifact paths now consumed by the connected runtime.
RUNTIME_DATA_ARTIFACTS: tuple[str, ...] = (
    "case_word/ud_ewt_case_word_counts.json",
    "widened/ud_ewt_clause_widened_counts.json",
)

_PACKAGE_VENDOR_DIRECTORY = "_vendor"
_SOURCE_VENDOR_DIRECTORY = Path(__file__).resolve().parents[2] / "vendor"
RAW_CORPUS_CACHE_ENV = "AILM_CORPUS_CACHE_DIR"


def _validate_basename(filename: str) -> None:
    """Reject traversal and directory inputs at every resource boundary."""
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"resource filename must be a basename, got {filename!r}")


def _source_checkout_available() -> bool:
    """Whether this module is executing from the repository's ``src/`` tree."""
    project_root = _SOURCE_VENDOR_DIRECTORY.parent
    return (project_root / "pyproject.toml").is_file() and (
        project_root / "src" / "ailm" / "resources.py"
    ).is_file()


def _platform_user_cache_directory(
    environment: Mapping[str, str],
    *,
    platform: str,
    operating_system: str,
    home: Path,
) -> Path:
    """Return the conventional cache directory without adding a dependency."""
    if operating_system == "nt":
        configured = environment.get("LOCALAPPDATA")
        root = Path(configured).expanduser() if configured else home / "AppData" / "Local"
    elif platform == "darwin":
        root = home / "Library" / "Caches"
    else:
        configured = environment.get("XDG_CACHE_HOME")
        candidate = Path(configured).expanduser() if configured else None
        # The XDG specification requires an absolute path.  Ignore an invalid
        # relative ambient value rather than making loader behavior depend on cwd.
        root = candidate if candidate is not None and candidate.is_absolute() else home / ".cache"
    return root / "ailm" / "corpora"


def default_raw_corpus_directory() -> Path:
    """Return the destination for raw corpus downloads.

    An explicit ``AILM_CORPUS_CACHE_DIR`` wins in every installation mode and
    must be absolute.  Without it, a source checkout retains ``vendor/`` while an
    installed wheel uses the platform's per-user cache directory.
    """
    configured = os.environ.get(RAW_CORPUS_CACHE_ENV)
    if configured is not None:
        if not configured.strip():
            raise ValueError(f"{RAW_CORPUS_CACHE_ENV} must not be empty")
        directory = Path(configured).expanduser()
        if not directory.is_absolute():
            raise ValueError(f"{RAW_CORPUS_CACHE_ENV} must be an absolute path: {configured!r}")
        return directory
    if _source_checkout_available():
        return _SOURCE_VENDOR_DIRECTORY
    return _platform_user_cache_directory(
        os.environ,
        platform=sys.platform,
        operating_system=os.name,
        home=Path.home(),
    )


def resolve_raw_corpus_path(filename: str) -> Path:
    """Return the local path owned by one downloadable raw corpus."""
    _validate_basename(filename)
    return default_raw_corpus_directory() / filename


def resolve_vendored_artifact(
    filename: str,
    *,
    path: Path | str | None = None,
    directory: Path | str | None = None,
) -> VendorResource:
    """Return a readable resource for one vendored artifact.

    Explicit caller paths retain their historical precedence.  With no override,
    an installed package reads the artifact embedded in the wheel; an editable or
    direct source checkout reads the canonical top-level ``vendor/`` copy.

    Args:
        filename: Artifact basename, never a path.
        path: Optional complete artifact path.
        directory: Optional directory containing ``filename``.

    Raises:
        ValueError: If ``filename`` is not a basename or both override forms are
            supplied.
    """
    _validate_basename(filename)
    if path is not None and directory is not None:
        raise ValueError("path and directory are mutually exclusive")
    if path is not None:
        return Path(path)
    if directory is not None:
        return Path(directory) / filename

    packaged = files("ailm").joinpath(_PACKAGE_VENDOR_DIRECTORY, filename)
    return packaged if packaged.is_file() else _SOURCE_VENDOR_DIRECTORY / filename


def resolve_data_artifact(relative: str) -> VendorResource:
    """Resolve an explicitly registered runtime input without changing its source path."""
    if relative not in RUNTIME_DATA_ARTIFACTS:
        raise ValueError(f"unregistered runtime data artifact: {relative!r}")
    packaged = files("ailm").joinpath("_data", *relative.split("/"))
    if _source_checkout_available():
        return _SOURCE_VENDOR_DIRECTORY.parent / "data" / relative
    return packaged
