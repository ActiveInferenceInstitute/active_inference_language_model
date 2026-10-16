"""Column-stochastic normalization and validation shared by the levels.

Every level builds its ``A`` and ``B`` matrices in the same orientation,
``[next_or_observation, current_state]``, and turns counts into a distribution
over axis 0 the same way. The one-line normalizer used to be written into each
builder separately and the validator was duplicated between the coupling
channel and the oracle; a change to one copy could not reach the others, and
nothing would have raised. This module is the single definition the levels
share.

Convention: a matrix ``M`` is column-stochastic when ``M[:, j]`` is a
distribution over the produced variable for every ``j``. The independent
instrument at :mod:`ailm.verification.oracle` deliberately keeps its own copy
of the validator, because it must share no code with the levels it audits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Tolerance for stochasticity checks.
STOCHASTIC_TOL: float = 1e-6


def column_stochastic(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize a ``(n_next, n_current)`` count matrix so every column sums to 1.

    This is exactly ``counts / counts.sum(axis=0, keepdims=True)``: no
    coercion, no clamping, so a caller's float64 result is bit-identical to the
    inline expression it replaces. A column of all zeros divides to NaN, which
    is why every caller smooths its counts first.

    Args:
        counts: Non-negative counts with no all-zero column.

    Returns:
        The same shape, normalized down axis 0 (the "next"/observation axis).
    """
    return counts / counts.sum(axis=0, keepdims=True)


def validate_column_stochastic(
    matrix: object, name: str, *, tol: float = STOCHASTIC_TOL
) -> NDArray[np.float64]:
    """Coerce ``matrix`` to float64 and check every column is a distribution.

    Args:
        matrix: Candidate 2-D array.
        name: Parameter name for error messages.
        tol: Largest tolerated deviation of any column sum from 1.

    Returns:
        The validated float64 array.

    Raises:
        ValueError: If the array is not 2-D/finite/non-negative, or any column
            deviates from summing to 1 by more than ``tol``.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite entries")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} contains negative probabilities")
    col_sums = arr.sum(axis=0)
    worst = int(np.argmax(np.abs(col_sums - 1.0)))
    if abs(float(col_sums[worst]) - 1.0) > tol:
        raise ValueError(
            f"{name} is not column-stochastic: column {worst} sums to "
            f"{float(col_sums[worst])!r} (tolerance {tol})."
        )
    return arr
