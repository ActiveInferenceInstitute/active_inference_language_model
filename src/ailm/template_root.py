"""Where the ``docxology/template`` checkout is, declared once per machine.

AILM renders through that checkout's ``infrastructure`` package, but it is a
standalone sidecar: nothing about this repository's own location says where
the template is. Three scripts need the answer (manuscript hydration, the
figure registry and the release verifier) and they must agree, so the explicit
sources live here and each script applies its own validation to the result.

Two explicit sources, in order:

1. ``TEMPLATE_REPO_ROOT`` in the environment. The renderer sets it for the
   hydration subprocess, and an operator can set it for one command.
2. A ``.template-root`` file at the repository root holding one path. It is
   gitignored, because the location is a property of the machine rather than
   of the project, and it is what makes the choice persistent across commands.

An explicit source is final: a path that does not hold a template is an error
for the caller to report, never a reason to search elsewhere, because falling
through from a typo to another checkout would publish through a different
renderer. Only when neither source is set do the scripts fall back to their
conventional candidate locations.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

TEMPLATE_ROOT_ENV = "TEMPLATE_REPO_ROOT"
TEMPLATE_ROOT_FILE = ".template-root"


def template_root_file(project_root: Path) -> Path:
    """Return the per-machine configuration file for ``project_root``."""
    return project_root / TEMPLATE_ROOT_FILE


def read_template_root_file(path: Path) -> Path:
    """Return the one path a ``.template-root`` file declares.

    Blank lines and ``#`` comments are ignored. Exactly one path must remain;
    ``~`` is expanded, and a relative path is taken relative to the file's own
    directory.

    Raises:
        ValueError: when the file declares no path or more than one.
    """
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(values) != 1:
        raise ValueError(
            f"{path} must declare exactly one template checkout path, found {len(values)}"
        )
    declared = Path(values[0]).expanduser()
    return declared if declared.is_absolute() else path.parent / declared


def configured_template_root(
    project_root: Path, environment: Mapping[str, str] | None = None
) -> tuple[Path | None, str]:
    """Return the explicitly configured template root and the source that named it.

    Returns ``(None, "")`` when neither source is set, so the caller may fall
    back to its candidate walk. The returned path is not validated here: each
    caller decides what a usable template looks like for its purpose, and an
    explicit path that fails that check is final rather than a fallback.
    """
    active = os.environ if environment is None else environment
    explicit = active.get(TEMPLATE_ROOT_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser(), TEMPLATE_ROOT_ENV
    configured = template_root_file(project_root)
    if configured.is_file():
        return read_template_root_file(configured), TEMPLATE_ROOT_FILE
    return None, ""
