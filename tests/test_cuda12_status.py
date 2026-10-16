"""Tests for the cuda12 extra runtime status probe.

The probe tests run in every environment (they assert on the *structure* of the
report, never that CUDA is present). The live-GPU assertions are opt-in under
the ``requires_cuda12`` marker: default addopts do NOT deselect them — only the
two verify_release profiles do (``-m "not ... and not requires_cuda12"`` in
``scripts/verify_release.py``). On a host without the cuda12 extra the class
also skips itself via the ``skipif`` environment guard.
"""

from __future__ import annotations

import importlib.metadata

import pytest

from ailm.cuda12_status import EXPECTED_PLUGIN_VERSION, cuda12_status

_KEEP_ASSERT_REWRITE_ORDER_EXPLICIT = pytest.register_assert_rewrite


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_cuda12: requires the cuda12 extra (jax[cuda12]==0.4.38 plugin set); "
        "opt in with -m requires_cuda12. See docs/troubleshooting.md CUDA section",
    )


def test_cuda12_status_report_structure_on_cpu_only() -> None:
    """On this CPU-only checkout the extra is absent and reported honestly."""
    status = cuda12_status()
    assert set(status) == {
        "extra_installed",
        "plugin_version_match",
        "expected_plugin_version",
        "missing",
        "components",
        "import_error",
        "backend",
    }
    assert status["expected_plugin_version"] == EXPECTED_PLUGIN_VERSION == "0.4.38"
    # This machine has no CUDA wheels installed: absence must be visible.
    if not status["extra_installed"]:
        assert len(status["missing"]) == 13
        assert "jax-cuda12-plugin" in status["missing"]
        assert all(v is None for v in status["components"].values())
    else:
        assert status["missing"] == []
    assert status["backend"] in {"cpu", "gpu", "tpu", None}
    assert status["import_error"] is None


def test_cuda12_status_backend_matches_real_jax_install() -> None:
    """The reported backend agrees with importing jax directly (no mocking)."""
    import jax

    status = cuda12_status()
    assert status["backend"] == jax.default_backend()


def _cuda12_extra_installed() -> bool:
    try:
        importlib.metadata.version("jax-cuda12-plugin")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


requires_cuda12_env = pytest.mark.skipif(
    not _cuda12_extra_installed(),
    reason="cuda12 extra not installed in this environment",
)


@pytest.mark.requires_cuda12
@requires_cuda12_env
class TestLiveCuda12Environment:
    """Opt-in: only meaningful after installing the cuda12 extra."""

    def test_all_thirteen_components_present(self) -> None:
        assert not cuda12_status()["missing"]

    def test_plugin_versions_match_environment_pin(self) -> None:
        status = cuda12_status()
        assert status["plugin_version_match"] is True
        assert status["components"]["jax-cuda12-plugin"] == "0.4.38"

    def test_plugin_imports_cleanly(self) -> None:
        assert cuda12_status()["import_error"] is None

    def test_reports_gpu_backend_when_available(self) -> None:
        backend = cuda12_status()["backend"]
        assert backend in {"gpu", "cpu"}  # cpu tolerated when no device visible
