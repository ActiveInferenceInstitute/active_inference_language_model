"""Tests for scripts/probe_pymdp_sparse.py.

Timeouts: the repo default (60 s) cannot even import jax on cold caches, and
pytest-timeout runs per test — so the heavy imports happen at module
collection time, outside any per-test budget, and the module carries the
documented module-level mark for the bounded pymdp walks themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_pymdp_sparse import (  # noqa: E402
    check_unpatched_failure_modes,
    run_equivalence_probe,
)

pytestmark = pytest.mark.timeout(300)


def test_unpatched_failure_modes_detected():
    """Verify that unpatched blocking call sites are accurately identified."""
    failures = check_unpatched_failure_modes()
    assert len(failures) >= 2
    sites = [f["site"] for f in failures]
    assert "pymdp/agent.py:263" in sites
    assert "pymdp/utils.py:109" in sites


def test_sparse_equivalence_probe():
    """Verify that with minimal patches, sparse B matches dense exactly."""
    results = run_equivalence_probe()
    assert results["max_diff"] < 1e-6
    assert results["all_supports_match"] is True
