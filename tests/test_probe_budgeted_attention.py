"""Tests for scripts/probe_budgeted_attention.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ailm.levels.phoneme import (  # noqa: E402
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    g2p_split,
    load_phoneme_counts,
)
from scripts.probe_budgeted_attention import (  # noqa: E402
    _filter_with_observed,
    _uniform_phases,
)


def test_budgeted_attention_probe_helpers():
    """Verify that schedule fitting, uniform phases, and filtering execute correctly."""
    counts = load_phoneme_counts()
    train_pairs, _test_pairs = g2p_split(counts)

    emission = np.asarray(build_g2p_emission(train_pairs)[0], dtype=np.float64)
    transition = np.asarray(build_g2p_transition(train_pairs)[0], dtype=np.float64)[:, :, 0]
    start = np.asarray(build_g2p_start(train_pairs)[0], dtype=np.float64)

    # Test uniform phases helper
    phases = _uniform_phases(length=6, budget=3)
    assert len(phases) > 0
    for p in phases:
        assert len(p) == 3
        for idx in p:
            assert 0 <= idx < 6

    # Test small synthetic sequence filtering
    sample_letters = [0, 1, 2, 3]  # e.g. a, b, c, d
    posteriors = _filter_with_observed(
        sample_letters,
        frozenset({0, 2}),
        emission,
        transition,
        start,
    )
    assert posteriors.shape == (4, transition.shape[0])
    assert np.allclose(posteriors.sum(axis=1), 1.0)
