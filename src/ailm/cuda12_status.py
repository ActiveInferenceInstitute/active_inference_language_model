"""Runtime status probe for the optional ``cuda12`` dependency extra.

``pyproject.toml`` declares ``cuda12 = ["jax[cuda12]==0.4.38"]`` so a single
``uv sync --extra cuda12`` (or ``pip install "jax[cuda12]==0.4.38"``) turns on
GPU support on linux x86_64 without touching CPU-only environments. Until
2026-08-26 nothing at runtime referenced that extra; this module is the first
opt-in runtime touchpoint. It never imports CUDA packages unconditionally:
CPU-only environments get a structured "absent" report instead of an error.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

EXPECTED_PLUGIN_VERSION = "0.4.38"

_CUDA_DISTRIBUTIONS = (
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "nvidia-cublas-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
)


def _distribution_version(name: str) -> str | None:
    """Return the installed version of ``name``, or None when absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cuda12_status() -> dict[str, Any]:
    """Probe whether the cuda12 plugin set is installed and imports cleanly.

    Returns a structured dict:

    - ``extra_installed``: every known cuda12 distribution importable/metadata-visible
    - ``plugin_version_match``: plugin/pjrt versions equal the 0.4.38 environment pin
      (None when not installed)
    - ``missing``: distributions from the measured lock closure that are absent
    - ``components``: per-distribution installed version map (None = absent)
    - ``import_error``: exception text when importing the plugin fails, else None
    - ``backend``: current ``jax.default_backend()`` when jax imports, else None

    This function is read-only and side-effect free.
    """
    components: dict[str, str | None] = {
        name: _distribution_version(name) for name in _CUDA_DISTRIBUTIONS
    }
    missing = sorted(n for n, v in components.items() if v is None)
    extra_installed = not missing

    plugin_version = components["jax-cuda12-plugin"]
    pjrt_version = components["jax-cuda12-pjrt"]
    version_match: bool | None = None
    if plugin_version is not None and pjrt_version is not None:
        version_match = plugin_version == pjrt_version == EXPECTED_PLUGIN_VERSION

    import_error: str | None = None
    if plugin_version is not None:
        try:
            __import__("jax_cuda12_plugin")
        except Exception as exc:  # pragma: no cover - exercised only on broken GPU installs
            import_error = f"{type(exc).__name__}: {exc}"

    backend: str | None = None
    try:
        import jax

        backend = jax.default_backend()
    except Exception as exc:  # pragma: no cover - exercised only on broken jax installs
        import_error = f"jax import failed: {type(exc).__name__}: {exc}"

    return {
        "extra_installed": extra_installed,
        "plugin_version_match": version_match,
        "expected_plugin_version": EXPECTED_PLUGIN_VERSION,
        "missing": missing,
        "components": components,
        "import_error": import_error,
        "backend": backend,
    }
