"""Tunable hyperparameters for the AILM level builders.

Why this module exists
----------------------
Every builder in :mod:`ailm.levels.letter` and :mod:`ailm.levels.phoneme` began
life with its hyperparameters written into the source: ``np.ones(...)`` was the
add-1 Laplace pseudocount, and the held-out split stride was whatever the
vendored artifact happened to carry. Nothing could be varied without editing
code, so no sensitivity statement could be made and no reviewer could probe
robustness. This module is the single place those quantities now live.

The knobs
---------
``smoothing``
    The add-k pseudocount applied to every count table before normalization.
    With pseudocount ``k`` over ``V`` symbols, a cell with count ``c`` in a
    column whose observed total is ``n`` becomes ``(c + k) / (n + kV)``. That is
    a convex mixture of the maximum-likelihood column and the uniform column::

        p_k = lambda * p_mle + (1 - lambda) * uniform,   lambda = n / (n + kV)

    so ``k`` is a dial from "trust the corpus" (``k -> 0``) to "trust nothing"
    (``k -> inf``, every column exactly uniform). ``lambda`` is strictly
    decreasing in ``k``, and ``KL(p_k || uniform)`` is therefore strictly
    decreasing in ``k`` whenever the MLE column is not already uniform. That
    monotonicity is asserted in ``tests/test_config.py`` rather than described
    here and left unchecked.

``emission_smoothing``
    A separate pseudocount for the G2P emission ``A[0, letter, phone]``.
    ``None`` (the default) means "use ``smoothing``", which is what keeps the
    default configuration byte-identical to the original hardcoded behavior.
    It exists because the emission table is a different shape and a different
    density from the transition tables — 30 x 39 letter-phone alignments versus
    39 x 39 phone bigrams — so the amount of mass a reviewer wants to reserve
    for unseen cells is not obviously the same quantity in both.

``test_every``
    The held-out stride for :func:`ailm.levels.phoneme.g2p_split`: index ``i``
    is TEST iff ``i % test_every == 0``, everything else is TRAIN. ``None`` (the
    default) means "read it from the vendored artifact", which is where the rule
    used to be locked. Setting it varies the held-out fraction, which is
    approximately ``1 / test_every``. Values below 2 are rejected: at 1 every
    pair is TEST, the TRAIN split is empty, and every matrix built from it
    collapses to uniform — a model that is silently vacuous rather than loudly
    broken.

Validation
----------
A pseudocount must be finite and strictly positive. Zero or negative
pseudocounts are rejected at construction, not at use, because the failure they
produce downstream is quiet: ``k = 0`` restores unsmoothed MLE tables in which
an unseen transition has probability exactly zero, so a single unseen bigram in
held-out text drives the cross-entropy to infinity and the perplexity to ``inf``
or ``nan``. A negative pseudocount can produce negative "probabilities" that
still normalize to columns summing to 1, which pymdp's ``_validate()`` accepts.

``not math.isfinite(x)`` is checked explicitly and *before* the sign test: a
bare ``x <= 0`` guard is False for ``nan``, so NaN would otherwise pass a
fail-closed-looking check and propagate into every cell of the matrix.

The module is stdlib-only on purpose so that importing a configuration never
drags in JAX, numpy or pymdp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DEFAULT_LEVEL_CONFIG", "LevelConfig"]


def _validate_pseudocount(field: str, value: object) -> None:
    """Reject anything that is not a finite, strictly positive real pseudocount.

    Args:
        field: Name of the dataclass field, quoted back in the error message so
            a failure names the offending knob rather than "a value".
        value: The candidate pseudocount.

    Raises:
        ValueError: If ``value`` is not a real number, is not finite, or is not
            strictly positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"LevelConfig.{field} must be a real number, got {value!r} "
            f"of type {type(value).__name__}."
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        # Checked before the sign test on purpose: `nan <= 0` is False, so a
        # lone `<= 0` guard would let NaN through and poison every matrix cell.
        raise ValueError(
            f"LevelConfig.{field} must be finite, got {value!r}. A non-finite "
            f"pseudocount makes every smoothed count non-finite."
        )
    if numeric <= 0.0:
        raise ValueError(
            f"LevelConfig.{field} must be > 0, got {value!r}. A zero pseudocount "
            f"leaves unseen cells at probability 0, which makes held-out "
            f"perplexity infinite; a negative one produces negative "
            f"probabilities that still normalize and still pass pymdp's "
            f"_validate()."
        )


@dataclass(frozen=True, slots=True)
class LevelConfig:
    """Hyperparameters shared by the letter- and phoneme-level builders.

    Immutable and hashable, so a configuration can be passed around, compared,
    and used as a dict key in a sweep without any chance of a builder mutating
    the caller's settings.

    Attributes:
        smoothing: Add-k pseudocount for the count tables (letter bigrams, phone
            bigrams, phone unigrams, G2P start prior). ``1.0`` reproduces the
            original add-1 Laplace behavior exactly.
        emission_smoothing: Add-k pseudocount for the G2P letter-given-phone
            emission. ``None`` means "use :attr:`smoothing`".
        test_every: Held-out stride for the G2P split. ``None`` means "use the
            value recorded in the vendored artifact".

    Raises:
        ValueError: From ``__post_init__`` if a pseudocount is not finite and
            strictly positive, or if ``test_every`` is not an integer >= 2.
    """

    smoothing: float = 1.0
    emission_smoothing: float | None = None
    test_every: int | None = None

    def __post_init__(self) -> None:
        """Validate every field, naming the offending field and value on failure."""
        _validate_pseudocount("smoothing", self.smoothing)
        if self.emission_smoothing is not None:
            _validate_pseudocount("emission_smoothing", self.emission_smoothing)
        if self.test_every is not None:
            if isinstance(self.test_every, bool) or not isinstance(self.test_every, int):
                raise ValueError(
                    f"LevelConfig.test_every must be an int or None, got "
                    f"{self.test_every!r} of type {type(self.test_every).__name__}. "
                    f"The split rule is 'index i is TEST iff i % test_every == 0', "
                    f"which is only defined for integers."
                )
            if self.test_every < 2:
                raise ValueError(
                    f"LevelConfig.test_every must be >= 2, got {self.test_every!r}. "
                    f"At 1 every pair is held out, the TRAIN split is empty, and "
                    f"every matrix built from it is uniform — vacuous rather than "
                    f"broken."
                )

    @property
    def resolved_emission_smoothing(self) -> float:
        """The pseudocount the G2P emission actually uses.

        Returns:
            :attr:`emission_smoothing` when set, otherwise :attr:`smoothing`.
        """
        if self.emission_smoothing is None:
            return float(self.smoothing)
        return float(self.emission_smoothing)


#: The documented default: add-1 Laplace everywhere, held-out stride taken from
#: the vendored artifact. Every builder called with no configuration uses this
#: instance, and the resulting matrices are byte-identical to the ones the
#: hardcoded implementations produced (pinned by digest in tests/test_config.py).
DEFAULT_LEVEL_CONFIG: LevelConfig = LevelConfig()
