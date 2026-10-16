"""Level-1 to Level-2 coupling: the letter model as a soft observation for G2P.

What this module is
-------------------
A downward message from the letter level into the G2P level's likelihood — the
smallest honest version of message passing *between* two agents, as opposed to
the cross-representation inference L2B already performs inside one agent:

  - The L1 letter agent (identity ``A``, Brown letter-bigram ``B``, uniform ``D``)
    produces a predictive distribution over the letter at position ``t`` from
    letters ``0 .. t-1``. That predictive distribution *is* the L1 empirical
    prior — the same object ``update_empirical_prior`` returns.
  - The L2B G2P agent normally consumes a **hard** letter index at position
    ``t``: its evidence vector over phones is the single emission row
    ``A[letter_t, :]``.
  - Coupling replaces that hard index with a **soft** observation: a distribution
    ``q(letter)`` over the 30 letters, marginalized through the emission
    (:func:`soft_observation_likelihood`). The distribution is a per-position
    mixture of the observed letter (one-hot) and the L1 predictive, weighted by
    ``alpha``.

``alpha = 0`` is pure hard observation and reproduces the shipped uncoupled decode
**exactly**; ``alpha = 1`` discards the observed glyph entirely and listens only to
the letter model's expectation. Everything in between is a real downward message.

The measured outcome, stated up front
-------------------------------------
Coupling does **not** help, and it is not marginal. Recorded by
``scripts/record_coupling_claims.py`` over claim:coupling_eval_positions held-out
positions: the best positive alpha on the fixed grid is
claim:coupling_best_alpha, which reaches claim:coupling_best_accuracy% against
claim:coupling_alpha0_accuracy% for the uncoupled anchor — a margin of
claim:coupling_margin_pts points, i.e. a **loss**, significant in the direction of
harm at paired-McNemar p = claim:coupling_mcnemar_p. Accuracy falls monotonically
as ``alpha`` rises. The recorded support flag is claim:coupling_supported.

The reason is structural: at position ``t`` the G2P agent already *observes*
letter ``t``, so the L1 predictive — a function of letters ``0 .. t-1`` only —
carries strictly less information about letter ``t`` than the glyph does, and
mixing it in dilutes exact evidence with a guess. The channel is nonetheless not
inert, which is what makes this a measurement rather than a wiring bug; the full
evidence, including the scrambled-control and message-quality numbers, is in
``docs/decisions.md``. The module is kept, gated, and off by default: the
recommendation that follows is to correct the architecture figure's arrow label,
not to ship a coupling that costs accuracy. A coupling form that adds an
*independent* channel is not what the figure claims and is not measured here.

Why the coupling is still gated rather than deleted
---------------------------------------------------
A null needs the same controls as a positive, or "we measured no effect" is
indistinguishable from "we measured nothing":

  - **alpha = 0 identity.** The soft recursion here is an independent
    implementation of the forward filter, so at ``alpha = 0`` it must agree
    position-for-position with :func:`ailm.verification.oracle.forward_filter`
    driven by hard observations, with zero differences. Without that anchor the
    sweep compares two unrelated decoders and the null is unfalsifiable.
  - **A scrambled-message control.** :func:`scramble_letter_distribution`
    reverses the letter axis of the L1 predictive — a fixed permutation, so the
    message keeps its entropy and loses only its meaning. The real message must
    beat the scrambled one, or the sweep is measuring the effect of adding *any*
    extra mass to the likelihood.
  - **A grid guard.** :func:`validate_alpha_grid` refuses a sweep that does not
    contain ``0.0`` plus at least three distinct positive values, so "no alpha
    beat the baseline" cannot be produced by a sweep that collapsed to a single
    point.

Orientation conventions (pinned — a transpose here voids every number)
---------------------------------------------------------------------
All matrices are plain float64 numpy in the orientation
:mod:`ailm.verification.oracle` expects, i.e. the pymdp layout with the leading
batch axis stripped:

``emission``    ``(30, 39)``, ``emission[l, p] = P(letter l | phone p)``,
                columns over letters sum to 1.
``transition``  ``(39, 39)``, ``transition[i, j] = P(next phone i | cur phone j)``,
                columns sum to 1.
``start``       ``(39,)`` over the first phone.
``letter_transition`` ``(30, 30)``, ``[i, j] = P(next letter i | cur letter j)``.
``letter_start``      ``(30,)``, uniform, mirroring
                :func:`ailm.levels.letter.build_letter_agent`'s ``D``.

pymdp is never imported here, not even lazily. The whole point of computing the
sweep in numpy is that a full-set pymdp pass is ~42.8k ``infer_states`` calls;
the license for substituting numpy is the alpha=0 identity against the
implementation-independent reference decoder (the oracle), which is checked in
``tests/test_coupling.py`` and again in
``scripts/record_coupling_claims.py`` before anything is recorded.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ailm.control import GateVacuousError
from ailm.levels._stochastic import STOCHASTIC_TOL, validate_column_stochastic
from ailm.levels.letter import (
    ALPHA_TO_IDX,
    IDX_TO_ALPHA,
    N_SYMBOLS,
    build_letter_transition,
)

# ``_iter_pairs`` and ``_word_letters`` are imported despite the underscore, for
# the same reason ``tests/test_stage2_gate.py`` imports ``_iter_pairs``: a coupled
# decode is only comparable with :func:`ailm.levels.phoneme.baseline_positions`
# and :func:`ailm.levels.phoneme.decode_positions` if it scores *exactly* the same
# positions in the same order. Re-deriving the stride formula or the letter
# normalization here would let the two drift apart silently — which is the failure
# mode the paired comparison exists to exclude.
from ailm.levels.phoneme import (
    IDX_TO_PHONE,
    N_PHONES,
    _iter_pairs,
    _word_letters,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
)
from ailm.verification.oracle import forward_filter, identity_emission

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ALPHA_GRID",
    "G2PMatrices",
    "LetterMatrices",
    "PositionDiagnostic",
    "build_g2p_matrices",
    "build_letter_matrices",
    "couple_letter_to_phoneme",
    "coupled_accuracy",
    "coupled_positions",
    "letter_predictive_distributions",
    "mcnemar",
    "mix_letter_distribution",
    "scramble_letter_distribution",
    "soft_observation_likelihood",
    "sweep_alpha",
    "uncoupled_positions",
    "validate_alpha_grid",
]

#: The fixed sweep grid. Contains 0.0 (the identity anchor) plus five distinct
#: positive values, so :func:`validate_alpha_grid` accepts it. Fixed in advance
#: and never tuned: the finding is whatever this grid measures.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

#: Minimum number of distinct positive alphas a legitimate sweep must contain.
_MIN_POSITIVE_ALPHAS: int = 3

#: Smallest probability fed to ``log``, matching the oracle's policy.
_LOG_FLOOR: float = 1e-300

# ── MODEL BUNDLES ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LetterMatrices:
    """The L1 letter model in the oracle's numpy orientation.

    Attributes:
        transition: ``(30, 30)``, ``[i, j] = P(next letter i | cur letter j)``.
        start: ``(30,)`` distribution over the first letter.
    """

    transition: NDArray[np.float64]
    start: NDArray[np.float64]


@dataclass(frozen=True)
class G2PMatrices:
    """The L2B G2P model in the oracle's numpy orientation.

    Attributes:
        emission: ``(30, 39)``, ``[l, p] = P(letter l | phone p)``.
        transition: ``(39, 39)``, ``[i, j] = P(next phone i | cur phone j)``.
        start: ``(39,)`` distribution over the first phone.
    """

    emission: NDArray[np.float64]
    transition: NDArray[np.float64]
    start: NDArray[np.float64]


@dataclass(frozen=True)
class PositionDiagnostic:
    """Per-position record of what the coupled decode actually did.

    Attributes:
        index: Position within the word, 0-based.
        letter: The observed letter symbol.
        decoded_phone: The argmax phone at this position.
        posterior_max: Probability mass the phone posterior put on
            ``decoded_phone``.
        letter_predictive_prob: Probability the L1 predictive assigned to the
            letter that was actually observed. This is the diagnostic that shows
            whether the downward message was any good at this position: near 1
            means L1 already knew the letter, near 0 means the message was
            actively misleading and ``alpha`` mass was spent on a wrong guess.
        mixture_prob_observed: Probability the *mixed* ``q(letter)`` placed on the
            observed letter, i.e. ``(1 - alpha) + alpha * letter_predictive_prob``.
        log_evidence: ``log P(evidence_t | evidence_<t)`` for this position, in
            nats. Not comparable across alphas as a model score — a soft
            observation is a different observation, not the same one scored
            differently — so it is a diagnostic, never a gate metric.
    """

    index: int
    letter: str
    decoded_phone: str
    posterior_max: float
    letter_predictive_prob: float
    mixture_prob_observed: float
    log_evidence: float


# ── VALIDATION ────────────────────────────────────────────────────────────────


def _validate_distribution(vector: object, length: int, name: str) -> NDArray[np.float64]:
    """Coerce ``vector`` to a float64 probability vector of the given length.

    Args:
        vector: Candidate 1-D array.
        length: Required length.
        name: Parameter name for error messages.

    Returns:
        The validated float64 vector.

    Raises:
        ValueError: On wrong shape, non-finite or negative entries, or a total
            that deviates from 1 by more than ``1e-6``.
    """
    arr = np.asarray(vector, dtype=np.float64).ravel()
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {np.asarray(vector).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite entries")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} contains negative probabilities")
    total = float(arr.sum())
    if abs(total - 1.0) > STOCHASTIC_TOL:
        raise ValueError(f"{name} must sum to 1, got {total!r} (tolerance {STOCHASTIC_TOL})")
    return arr


def _validate_alpha(alpha: object) -> float:
    """Coerce ``alpha`` to a float in ``[0, 1]``.

    Args:
        alpha: Candidate mixture weight.

    Returns:
        The validated float.

    Raises:
        ValueError: If ``alpha`` is not a finite real number in ``[0, 1]``.
    """
    if isinstance(alpha, bool):
        raise ValueError(f"alpha must be a real number in [0, 1], got {alpha!r}")
    try:
        value = float(alpha)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alpha must be a real number in [0, 1], got {alpha!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"alpha must be finite, got {value!r}")
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {value!r}")
    return value


def validate_alpha_grid(grid: object) -> tuple[float, ...]:
    """Validate a sweep grid and return it as a tuple of floats.

    A sweep that has collapsed to a single point, or that has lost its ``0.0``
    anchor, can report "no alpha beat the baseline" without having looked. This
    function makes that impossible: the grid must contain ``0.0`` and at least
    :data:`_MIN_POSITIVE_ALPHAS` distinct positive values.

    Args:
        grid: Iterable of candidate alphas.

    Returns:
        The grid as a tuple of validated floats, in the order given.

    Raises:
        ValueError: If any entry is not a real number in ``[0, 1]``.
        GateVacuousError: If ``0.0`` is absent, or fewer than three distinct
            positive values are present.
    """
    if isinstance(grid, (str, bytes)) or not isinstance(grid, Iterable):
        raise ValueError(
            f"alpha grid must be a non-string iterable of candidate values, got {grid!r}"
        )
    values = tuple(_validate_alpha(a) for a in grid)

    if not any(v == 0.0 for v in values):
        raise GateVacuousError(
            "alpha grid has no 0.0 entry. Without the uncoupled anchor there is "
            "nothing for the coupled decodes to be compared against, so every "
            f"conclusion drawn from the sweep would be unfounded. Grid: {values}"
        )
    positive = {v for v in values if v > 0.0}
    if len(positive) < _MIN_POSITIVE_ALPHAS:
        raise GateVacuousError(
            f"alpha grid has only {len(positive)} distinct positive value(s) "
            f"({sorted(positive)}), fewer than the required "
            f"{_MIN_POSITIVE_ALPHAS}. A sweep that has collapsed to a single "
            "point can report 'no alpha helped' without having swept."
        )
    return values


# ── MODEL CONSTRUCTION ────────────────────────────────────────────────────────


def build_letter_matrices(brown_counts: dict) -> LetterMatrices:
    """Build the L1 letter model as numpy, from Brown bigram counts.

    Wraps :func:`ailm.levels.letter.build_letter_transition` — which returns the
    column-stochastic bigram matrix that :func:`ailm.levels.letter.build_letter_agent`
    installs as ``B`` — and pairs it with the uniform ``D`` that function uses.
    Nothing is re-derived here, so the coupled L1 message comes from the shipped
    letter model rather than from a look-alike.

    Args:
        brown_counts: Payload from :func:`ailm.levels.letter.load_brown_counts`,
            i.e. ``{"prev,curr": count}``.

    Returns:
        A :class:`LetterMatrices` bundle.

    Raises:
        ValueError: If the resulting transition is not column-stochastic.
    """
    transition = np.asarray(build_letter_transition(brown_counts), dtype=np.float64)[0]
    # D in build_letter_agent is uniform over the 30 letters. Mirrored rather than
    # re-derived: a different start prior would silently change position 0.
    start = np.ones(N_SYMBOLS, dtype=np.float64) / N_SYMBOLS
    return LetterMatrices(
        transition=validate_column_stochastic(transition, "letter_transition"),
        start=_validate_distribution(start, N_SYMBOLS, "letter_start"),
    )


def build_g2p_matrices(train_pairs: list[tuple[str, list[str]]]) -> G2PMatrices:
    """Build the L2B G2P model as numpy, from the TRAIN split.

    Wraps the shipped builders in :mod:`ailm.levels.phoneme` so the coupled
    decoder and the pymdp agent are parameterized by the same matrices. In
    particular the transition comes from :func:`~ailm.levels.phoneme.build_g2p_transition`,
    which counts bigrams inside the training split only — using the artifact's
    full-corpus ``phone_bigrams`` here would leak the held-out words.

    Args:
        train_pairs: ``(word, [phones])`` items from
            :func:`ailm.levels.phoneme.g2p_split`.

    Returns:
        A :class:`G2PMatrices` bundle.

    Raises:
        ValueError: If any matrix fails its stochasticity or shape check.
    """
    emission = np.asarray(build_g2p_emission(train_pairs), dtype=np.float64)[0]
    transition = np.asarray(build_g2p_transition(train_pairs), dtype=np.float64)[0, :, :, 0]
    start = np.asarray(build_g2p_start(train_pairs), dtype=np.float64).ravel()

    # The two shape checks below are unreachable while the phoneme builders keep
    # their documented shapes, and they are deliberately kept anyway: they are the
    # tripwire for a future change to those builders. Do not "cover" them by
    # loosening them — a transposed emission is the one Stage 2 mis-wiring that an
    # accuracy bar provably cannot see.
    emission = validate_column_stochastic(emission, "emission")
    if emission.shape != (N_SYMBOLS, N_PHONES):
        raise ValueError(
            f"emission must have shape ({N_SYMBOLS}, {N_PHONES}) — "
            f"[letter, phone] = P(letter | phone) — got {emission.shape}"
        )
    transition = validate_column_stochastic(transition, "transition")
    if transition.shape != (N_PHONES, N_PHONES):
        raise ValueError(
            f"transition must have shape ({N_PHONES}, {N_PHONES}), got {transition.shape}"
        )
    return G2PMatrices(
        emission=emission,
        transition=transition,
        start=_validate_distribution(start, N_PHONES, "start"),
    )


# ── THE DOWNWARD MESSAGE ──────────────────────────────────────────────────────


def letter_predictive_distributions(
    letter_matrices: LetterMatrices,
    letter_indices: object,
) -> NDArray[np.float64]:
    """The L1 empirical prior at every position: ``P(letter_t | letters_<t)``.

    This is exactly what :meth:`pymdp.agent.Agent.update_empirical_prior` returns
    inside :func:`ailm.levels.letter.compute_perplexity`, computed through the
    implementation-independent reference decoder (the oracle) instead of through
    pymdp:

        ``predictive[0] = start``
        ``predictive[t] = transition @ posterior[t-1]``   for ``t >= 1``

    The posteriors come from :func:`ailm.verification.oracle.forward_filter` under
    an identity emission, so under the shipped L1 model each posterior is one-hot
    on the letter just observed and ``predictive[t]`` is a column of the bigram
    matrix. It is computed through the filter anyway rather than read off as a
    column, so this stays correct if the letter level's ``A`` ever stops being the
    identity.

    Position ``t`` never sees letter ``t``. That is the whole content of the
    finding: the message is strictly weaker evidence about letter ``t`` than the
    observed glyph, so mixing it in can only dilute.

    Args:
        letter_matrices: The L1 model.
        letter_indices: Length-``T >= 1`` sequence of letter indices.

    Returns:
        ``(T, 30)`` array whose rows sum to 1.

    Raises:
        ValueError: If ``letter_indices`` is empty or holds an out-of-range index.
    """
    indices = np.asarray(letter_indices, dtype=np.int64).ravel()
    if indices.size == 0:
        raise ValueError("letter_indices must be non-empty")

    posteriors, _ = forward_filter(
        identity_emission(N_SYMBOLS),
        letter_matrices.transition,
        letter_matrices.start,
        indices,
    )
    predictive = np.empty((indices.size, N_SYMBOLS), dtype=np.float64)
    predictive[0] = letter_matrices.start
    if indices.size > 1:
        # transition @ posteriors[:-1].T -> (30, T-1); transpose back to rows.
        predictive[1:] = (letter_matrices.transition @ posteriors[:-1].T).T
    return predictive


def scramble_letter_distribution(
    distribution: NDArray[np.float64] | np.ndarray,
) -> NDArray[np.float64]:
    """Reverse the letter axis — the negative control on the coupling channel.

    Reversal is a fixed involutive permutation of the 30 letters, so the scrambled
    message has *identical* entropy, identical peakedness, and identical
    normalization to the real one. Only its correspondence to the letter alphabet
    is destroyed. That is what makes it the right control: if the real message and
    the scrambled message change accuracy by the same amount, the change is an
    artifact of adding *any* extra probability mass to the likelihood, not of
    adding letter-model information, and neither a positive nor a null result
    about coupling could be read off the sweep.

    A permutation was chosen over "replace with uniform" deliberately. Uniform is
    a weaker control: it changes the entropy of the message as well as its
    content, so a difference between real and uniform is confounded.

    Args:
        distribution: ``(30,)`` distribution over letters, or ``(T, 30)`` rows.

    Returns:
        The same shape with the last axis reversed.

    Raises:
        ValueError: If the last axis is not of length 30.
    """
    arr = np.asarray(distribution, dtype=np.float64)
    if arr.shape[-1] != N_SYMBOLS:
        raise ValueError(
            f"distribution's last axis must have length {N_SYMBOLS} (the letter "
            f"alphabet), got shape {arr.shape}"
        )
    return np.ascontiguousarray(arr[..., ::-1])


def mix_letter_distribution(
    observed_index: int,
    predictive: NDArray[np.float64] | np.ndarray,
    alpha: float,
) -> NDArray[np.float64]:
    """Mix the observed letter (one-hot) with the L1 predictive.

    ``q = (1 - alpha) * onehot(observed_index) + alpha * predictive``.

    At ``alpha = 0`` this is exactly ``onehot(observed_index)``, which is what
    makes the alpha=0 identity anchor hold bit-for-bit rather than approximately:
    the one-hot branch is produced by the same arithmetic, and
    ``1.0 * 1.0 + 0.0 * p`` is exact in IEEE-754.

    Args:
        observed_index: Index of the letter actually observed.
        predictive: ``(30,)`` L1 predictive distribution.
        alpha: Mixture weight on the predictive, in ``[0, 1]``.

    Returns:
        ``(30,)`` distribution summing to 1.

    Raises:
        ValueError: On an out-of-range index, a malformed predictive, or an alpha
            outside ``[0, 1]``.
    """
    weight = _validate_alpha(alpha)
    pred = _validate_distribution(predictive, N_SYMBOLS, "predictive")
    index = int(observed_index)
    if not (0 <= index < N_SYMBOLS):
        raise ValueError(f"observed_index must lie in [0, {N_SYMBOLS}), got {observed_index!r}")
    mixed = weight * pred
    mixed[index] += 1.0 - weight
    return mixed


def soft_observation_likelihood(
    emission: NDArray[np.float64] | np.ndarray,
    letter_distribution: NDArray[np.float64] | np.ndarray,
) -> NDArray[np.float64]:
    """Marginalize the G2P emission over a distribution on letters.

    ``P(evidence | phone p) = sum_l P(letter l | phone p) * q(letter l)``, i.e.
    ``emission.T @ q``.

    With ``q`` one-hot on letter ``l0`` this reduces **exactly** to
    ``emission[l0, :]`` — the hard-observation column the shipped G2P decoder
    uses. That reduction is what makes the generalization faithful rather than
    merely plausible, and ``tests/test_coupling.py`` asserts it with ``==`` on
    every one of the 30 letters, not with a tolerance.

    The result is a likelihood, not a distribution: it does not sum to 1 over
    phones and must not be normalized. Normalizing it would turn the evidence
    into a prior and silently change the model.

    Args:
        emission: ``(30, 39)`` with ``[l, p] = P(letter l | phone p)``,
            column-stochastic over letters.
        letter_distribution: ``(30,)`` distribution over letters.

    Returns:
        ``(39,)`` non-negative likelihood vector over phones.

    Raises:
        ValueError: If the emission is not a column-stochastic 2-D array, or the
            letter distribution is not a distribution of matching length.
    """
    emis = validate_column_stochastic(emission, "emission")
    q = _validate_distribution(letter_distribution, emis.shape[0], "letter_distribution")
    return emis.T @ q


# ── THE COUPLED DECODER ───────────────────────────────────────────────────────


def _soft_forward(
    g2p: G2PMatrices,
    letter_indices: NDArray[np.int64],
    predictive: NDArray[np.float64],
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Forward filter over phones driven by soft letter observations.

    The recursion is the oracle's, with the hard emission column replaced by a
    marginalized likelihood::

        phone_pred[0] = start
        lik[t]        = emission.T @ q[t]
        Z[t]          = sum_p lik[t, p] * phone_pred[t, p]
        posterior[t]  = lik[t] * phone_pred[t] / Z[t]
        phone_pred[t+1] = transition @ posterior[t]

    Validation is the caller's job so a full-set sweep does not re-check the same
    three matrices 39,000 times. For the same reason the mixture and the one-hot
    likelihood are inlined rather than routed through
    :func:`mix_letter_distribution` and :func:`soft_observation_likelihood`, which
    validate on every call. The duplication is a drift risk, so it is pinned by a
    test: the ``mixtures`` returned here are asserted equal to
    :func:`mix_letter_distribution` output, and the ``alpha = 0`` likelihood
    short circuit is asserted equal to
    :func:`soft_observation_likelihood` on a one-hot.

    Args:
        g2p: Pre-validated G2P matrices.
        letter_indices: ``(T,)`` observed letter indices.
        predictive: ``(T, 30)`` L1 predictive rows, already scrambled if the
            control is being run.
        alpha: Pre-validated mixture weight.

    Returns:
        ``(posteriors, mixtures, log_evidence)`` of shapes ``(T, 39)``,
        ``(T, 30)`` and ``(T,)``.
    """
    n_steps = int(letter_indices.shape[0])
    posteriors = np.empty((n_steps, N_PHONES), dtype=np.float64)
    mixtures = np.empty((n_steps, N_SYMBOLS), dtype=np.float64)
    log_evidence = np.empty(n_steps, dtype=np.float64)

    emission_t = g2p.emission.T  # (39, 30), reused every step
    phone_pred = g2p.start

    for t in range(n_steps):
        index = int(letter_indices[t])
        if alpha == 0.0:
            # Exact one-hot short circuit. Algebraically identical to the mixture
            # below, and it is what makes the alpha=0 anchor bit-exact against the
            # oracle's hard-observation path rather than merely close.
            mixed = np.zeros(N_SYMBOLS, dtype=np.float64)
            mixed[index] = 1.0
            likelihood = g2p.emission[index, :]
        else:
            mixed = alpha * predictive[t]
            mixed[index] += 1.0 - alpha
            likelihood = emission_t @ mixed

        unnormalized = likelihood * phone_pred
        norm = float(unnormalized.sum())
        log_evidence[t] = float(np.log(max(norm, _LOG_FLOOR)))
        if norm <= 0.0:
            # Cannot occur with add-1 smoothed matrices; mirrored from the oracle
            # so a future unsmoothed model degrades the same way in both.
            logger.warning(
                "_soft_forward: zero evidence at t=%d (letter index %d); "
                "falling back to the predictive belief",
                t,
                index,
            )
            posteriors[t] = phone_pred
        else:
            posteriors[t] = unnormalized / norm
        mixtures[t] = mixed
        if t + 1 < n_steps:
            phone_pred = g2p.transition @ posteriors[t]

    return posteriors, mixtures, log_evidence


def couple_letter_to_phoneme(
    letter_matrices: LetterMatrices,
    g2p_matrices: G2PMatrices,
    word: str,
    alpha: float = 0.0,
    scramble: bool = False,
) -> tuple[list[str], list[PositionDiagnostic]]:
    """Decode a word to phones with the L1 letter message mixed into the evidence.

    At each position the evidence handed to the G2P filter is
    ``soft_observation_likelihood(emission, q_t)`` where
    ``q_t = (1 - alpha) * onehot(letter_t) + alpha * L1_predictive_t``.

    ``alpha`` defaults to ``0.0``, so the default behavior of this function is
    provably the uncoupled decode: the same phones
    :func:`ailm.levels.phoneme.decode_phonemes` produces. Coupling is opt-in and
    is never the silent default.

    Args:
        letter_matrices: The L1 model, from :func:`build_letter_matrices`.
        g2p_matrices: The L2B model, from :func:`build_g2p_matrices`.
        word: Raw word. Normalized with
            :func:`ailm.levels.phoneme._word_letters`, the same normalization the
            uncoupled decoder and the context-free baseline use.
        alpha: Weight on the L1 message, in ``[0, 1]``. ``0`` is uncoupled, ``1``
            ignores the observed glyph entirely.
        scramble: If True, reverse the letter axis of the L1 predictive before
            mixing — the negative control described in
            :func:`scramble_letter_distribution`. Has no effect at ``alpha = 0``,
            which is itself a useful check: the control and the real coupling must
            coincide exactly at the anchor.

    Returns:
        ``(phones, diagnostics)``. ``phones`` has one entry per retained letter;
        ``diagnostics`` is the aligned list of :class:`PositionDiagnostic`. An
        empty word returns ``([], [])``.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]``.
    """
    weight = _validate_alpha(alpha)
    letters = _word_letters(word)
    if not letters:
        return [], []

    indices = np.asarray([ALPHA_TO_IDX[c] for c in letters], dtype=np.int64)
    predictive = letter_predictive_distributions(letter_matrices, indices)
    if scramble:
        predictive = scramble_letter_distribution(predictive)

    posteriors, mixtures, log_evidence = _soft_forward(g2p_matrices, indices, predictive, weight)

    phones: list[str] = []
    diagnostics: list[PositionDiagnostic] = []
    for t, index in enumerate(indices):
        phone_idx = int(np.argmax(posteriors[t]))
        phones.append(IDX_TO_PHONE[phone_idx])
        diagnostics.append(
            PositionDiagnostic(
                index=t,
                letter=IDX_TO_ALPHA[int(index)],
                decoded_phone=IDX_TO_PHONE[phone_idx],
                posterior_max=float(posteriors[t, phone_idx]),
                letter_predictive_prob=float(predictive[t, int(index)]),
                mixture_prob_observed=float(mixtures[t, int(index)]),
                log_evidence=float(log_evidence[t]),
            )
        )
    return phones, diagnostics


def coupled_positions(
    letter_matrices: LetterMatrices,
    g2p_matrices: G2PMatrices,
    pairs: list[tuple[str, list[str]]],
    alpha: float = 0.0,
    limit: int | None = None,
    scramble: bool = False,
) -> list[bool]:
    """Per-position correctness of the coupled decoder.

    Mirrors :func:`ailm.levels.phoneme.decode_positions`, including ``limit``, and
    resolves the subsample through the *same* :func:`ailm.levels.phoneme._iter_pairs`
    — an evenly spaced stride, never a prefix, because the vendored ``g2p_pairs``
    list is alphabetically sorted (see ``docs/decisions.md``). Sharing the
    subsample function is also what keeps the returned vector aligned
    position-for-position with :func:`ailm.levels.phoneme.baseline_positions` and
    :func:`uncoupled_positions` on the same ``pairs`` and ``limit``, which is what
    makes a paired test valid.

    Args:
        letter_matrices: The L1 model.
        g2p_matrices: The L2B model.
        pairs: ``(word, [phones])`` evaluation items.
        alpha: Weight on the L1 message, in ``[0, 1]``.
        limit: Optional bound on evaluation pairs, resolved as a stride.
        scramble: Run the scrambled-message control instead of real coupling.

    Returns:
        One bool per position, words in order and letters in order.

    Raises:
        ValueError: If a decoded length differs from its reference length. The
            corpus is length-matched by construction, so a mismatch is a
            normalization bug rather than a data condition to tolerate.
    """
    weight = _validate_alpha(alpha)
    correct: list[bool] = []
    for word, phones in _iter_pairs(pairs, limit):
        decoded, _ = couple_letter_to_phoneme(
            letter_matrices, g2p_matrices, word, alpha=weight, scramble=scramble
        )
        if len(decoded) != len(phones):
            raise ValueError(
                f"Coupled decode length {len(decoded)} != reference length "
                f"{len(phones)} for word {word!r}. The G2P corpus is "
                f"length-matched by construction, so this is a normalization "
                f"bug: decoded={decoded}, reference={phones}."
            )
        correct.extend(d == r for d, r in zip(decoded, phones, strict=True))
    return correct


def coupled_accuracy(
    letter_matrices: LetterMatrices,
    g2p_matrices: G2PMatrices,
    pairs: list[tuple[str, list[str]]],
    alpha: float = 0.0,
    limit: int | None = None,
    scramble: bool = False,
) -> float:
    """Position-level accuracy of the coupled decoder.

    Mirrors :func:`ailm.levels.phoneme.g2p_accuracy`, including ``limit``.

    Args:
        letter_matrices: The L1 model.
        g2p_matrices: The L2B model.
        pairs: ``(word, [phones])`` evaluation items.
        alpha: Weight on the L1 message, in ``[0, 1]``.
        limit: Optional bound on evaluation pairs, resolved as a stride.
        scramble: Run the scrambled-message control instead of real coupling.

    Returns:
        Fraction of positions decoded correctly, or 0.0 if there are none.

    Raises:
        ValueError: If a decoded length differs from its reference length.
    """
    positions = coupled_positions(
        letter_matrices,
        g2p_matrices,
        pairs,
        alpha=alpha,
        limit=limit,
        scramble=scramble,
    )
    if not positions:
        return 0.0
    return sum(positions) / len(positions)


def uncoupled_positions(
    g2p_matrices: G2PMatrices,
    pairs: list[tuple[str, list[str]]],
    limit: int | None = None,
) -> list[bool]:
    """Per-position correctness of the **uncoupled** decoder, via the oracle.

    This is the alpha=0 reference, and it is deliberately computed by a different
    code path from :func:`coupled_positions`: hard observation indices straight
    into :func:`ailm.verification.oracle.forward_filter`, exactly as
    ``scripts/record_stage2_claims.py`` does. Comparing the two is therefore a
    differential check between two implementations rather than a function
    compared with itself.

    Args:
        g2p_matrices: The L2B model.
        pairs: ``(word, [phones])`` evaluation items.
        limit: Optional bound on evaluation pairs, resolved as a stride through
            :func:`ailm.levels.phoneme._iter_pairs`.

    Returns:
        One bool per position, aligned with :func:`coupled_positions` and
        :func:`ailm.levels.phoneme.baseline_positions` on the same arguments.

    Raises:
        ValueError: If a decoded length differs from its reference length.
    """
    correct: list[bool] = []
    for word, phones in _iter_pairs(pairs, limit):
        letters = _word_letters(word)
        if len(letters) != len(phones):
            raise ValueError(
                f"Retained-letter count {len(letters)} != reference length "
                f"{len(phones)} for word {word!r}. The G2P corpus is "
                f"length-matched by construction, so this is a normalization bug."
            )
        indices = [ALPHA_TO_IDX[c] for c in letters]
        posteriors, _ = forward_filter(
            g2p_matrices.emission,
            g2p_matrices.transition,
            g2p_matrices.start,
            indices,
        )
        decoded = [IDX_TO_PHONE[int(np.argmax(row))] for row in posteriors]
        correct.extend(d == r for d, r in zip(decoded, phones, strict=True))
    return correct


# ── SWEEP AND STATISTICS ──────────────────────────────────────────────────────


def sweep_alpha(
    letter_matrices: LetterMatrices,
    g2p_matrices: G2PMatrices,
    pairs: list[tuple[str, list[str]]],
    grid: object = DEFAULT_ALPHA_GRID,
    limit: int | None = None,
    scramble: bool = False,
) -> dict[float, list[bool]]:
    """Decode ``pairs`` once per alpha in ``grid`` and return the position vectors.

    Every returned vector covers the same positions in the same order, so any two
    of them — and :func:`uncoupled_positions` / :func:`ailm.levels.phoneme.baseline_positions`
    on the same arguments — can be compared pairwise.

    Args:
        letter_matrices: The L1 model.
        g2p_matrices: The L2B model.
        pairs: ``(word, [phones])`` evaluation items.
        grid: Alphas to sweep. Validated by :func:`validate_alpha_grid`.
        limit: Optional bound on evaluation pairs, resolved as a stride.
        scramble: Run the scrambled-message control at every alpha.

    Returns:
        ``{alpha: positions}``. Duplicate alphas in ``grid`` collapse to one key.

    Raises:
        GateVacuousError: If the grid lacks ``0.0`` or three distinct positives.
        ValueError: If a decoded length differs from its reference length.
    """
    alphas = validate_alpha_grid(grid)
    results: dict[float, list[bool]] = {}
    for alpha in alphas:
        if alpha in results:
            continue
        results[alpha] = coupled_positions(
            letter_matrices,
            g2p_matrices,
            pairs,
            alpha=alpha,
            limit=limit,
            scramble=scramble,
        )
    # Unreachable while every alpha is decoded over the same pairs and limit, and
    # kept as a tripwire on exactly that assumption: a de-paired sweep would make
    # every margin computed downstream meaningless while still producing numbers.
    lengths = {len(v) for v in results.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"Sweep produced position vectors of differing lengths {lengths}. "
            "The alphas are not comparable pairwise, so no margin computed from "
            "them would be a paired margin."
        )
    return results


def mcnemar(
    treatment: list[bool],
    control: list[bool],
) -> tuple[int, int, float]:
    """Exact paired McNemar test on two aligned correctness vectors.

    Identical procedure to ``scripts/record_stage2_claims.py``: count the
    discordant pairs and run a two-sided exact binomial test at ``p = 0.5``. The
    concordant pairs carry no information about a difference and are excluded, so
    a half-point aggregate margin becomes testable.

    ``b`` is the count of positions where ``treatment`` is right and ``control``
    is wrong; ``c`` is the reverse. ``b == c == 0`` returns ``p = 1.0`` — with no
    discordant pairs there is no evidence of a difference, which is exactly what
    the self-comparison control asserts.

    Args:
        treatment: Correctness vector for the candidate decoder.
        control: Correctness vector for the reference decoder, same length and
            same position order.

    Returns:
        ``(b, c, p_value)``.

    Raises:
        ValueError: If the two vectors have different lengths — an unpaired
            comparison, which this test is not valid for.
    """
    if len(treatment) != len(control):
        raise ValueError(
            f"McNemar needs paired vectors: got {len(treatment)} and "
            f"{len(control)} positions. An unpaired comparison of two accuracies "
            "cannot establish a fraction-of-a-point margin."
        )
    b = sum(1 for t, c in zip(treatment, control, strict=True) if t and not c)
    c_count = sum(1 for t, c in zip(treatment, control, strict=True) if c and not t)
    from ailm.statistics import exact_two_sided_binomial_p

    return b, c_count, exact_two_sided_binomial_p(b, c_count)
