"""Closed-form HMM oracle — an independent instrument for checking pymdp.

Why this module exists
----------------------
Stage 1 shipped a bug that pymdp could not see: the learned bigram matrix was
placed in the emission ``A`` with an identity ``B``. That wiring is *anti*-
informative rather than loudly broken — observing symbol ``c`` yields a
posterior over the symbol that *preceded* ``c``, and an identity ``B`` carries
that backward-looking belief forward as the next-symbol prediction. A bigram
matrix is column-stochastic, so ``pymdp._validate()`` accepts it in either
slot. The defect measured claim:stage1_anti_informative_perplexity against
claim:stage1_letter_perplexity for correct wiring, and on a small in-sample
corpus scored *worse than uniform*, yet still slipped under a naive threshold.

What caught it was a hand-rolled numpy Markov chain that disagreed with pymdp's
perplexity by a large margin. That instrument was a throwaway scratch script.
This module is that instrument promoted to a tested, first-class dependency so
every level from Stage 2 onward can be cross-checked numerically rather than
structurally.

**This module must never import pymdp or jax.** Its entire value is being an
independent implementation. If it shared code with the thing it audits, it
would inherit the flaw it audits for.

Conventions (pinned — a transpose here silently invalidates every cross-check)
-----------------------------------------------------------------------------
``emission``
    Shape ``(n_obs, n_states)``. ``emission[i, j] = P(obs = i | state = j)``.
    Column-stochastic: ``emission[:, j].sum() == 1`` for every state ``j``.

``transition``
    Shape ``(n_states, n_states)``. ``transition[i, j] = P(next = i | cur = j)``.
    Column-stochastic: ``transition[:, j].sum() == 1`` for every current state ``j``.

``start``
    Shape ``(n_states,)``, sums to 1. This is the distribution over
    ``state_0`` *before* any observation, so the predictive distribution at
    ``t = 0`` is ``start`` itself — the transition is **not** applied first.

These match the pymdp layout used elsewhere in AILM once the leading batch
dimension is stripped: ``A[0]`` is ``(n_obs, n_states)`` and ``B[0, :, :, 0]``
is ``(n_next, n_current)``, both validated along the axis that indexes the
*produced* variable (pymdp's ``axis=1``, this module's ``axis=0``).

Generative story
----------------
::

    state_0 ~ start
    obs_t   ~ emission[:, state_t]
    state_{t+1} ~ transition[:, state_t]

Numerical policy
----------------
Forward and backward messages are renormalized at every timestep and the
log-evidence is accumulated from the normalizers, so a 5000-symbol sequence
does not underflow. Viterbi runs entirely in log space. Probabilities are
clipped to :data:`_LOG_FLOOR` before any ``log``.
"""

from __future__ import annotations

import logging
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

__all__ = [
    "forward_backward",
    "forward_filter",
    "identity_emission",
    "letter_bigram_perplexity",
    "markov_chain_perplexity",
    "sequence_perplexity",
    "viterbi",
    "word_path_logprob",
    "word_path_perplexity",
]

#: Smallest probability fed to ``log``. Below float64's ~1e-308 denormal floor
#: would produce ``-inf``; 1e-300 keeps ``log`` finite (about -690.8 nats).
_LOG_FLOOR: float = 1e-300

#: Tolerance for stochasticity checks on emission / transition / start.
_STOCHASTIC_TOL: float = 1e-6


# ── VALIDATION ────────────────────────────────────────────────────────────────


def _as_2d(array: NDArray[np.float64] | np.ndarray, name: str) -> NDArray[np.float64]:
    """Coerce ``array`` to a finite 2-D float64 array.

    Args:
        array: Candidate matrix.
        name: Parameter name, used in error messages.

    Returns:
        A float64 ``(rows, cols)`` array.

    Raises:
        ValueError: If the input is not 2-D, is empty, or holds NaN/inf.
    """
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite entries")
    return arr


def _validate_column_stochastic(matrix: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Validate that every column of ``matrix`` is a probability distribution.

    Args:
        matrix: A 2-D array whose columns must sum to 1.
        name: Parameter name, used in error messages.

    Returns:
        The validated float64 matrix.

    Raises:
        ValueError: If ``matrix`` is not 2-D/finite, holds negative entries, or
            has any column whose sum deviates from 1 by more than
            :data:`_STOCHASTIC_TOL`.
    """
    arr = _as_2d(matrix, name)
    if np.any(arr < 0.0):
        raise ValueError(f"{name} contains negative probabilities")
    col_sums = arr.sum(axis=0)
    deviation = np.abs(col_sums - 1.0)
    if np.any(deviation > _STOCHASTIC_TOL):
        worst = int(np.argmax(deviation))
        raise ValueError(
            f"{name} is not column-stochastic: column {worst} sums to "
            f"{col_sums[worst]!r} (tolerance {_STOCHASTIC_TOL}). "
            f"{name}[:, j] must be a distribution over the produced variable."
        )
    return arr


def _validate_start(start: NDArray[np.float64] | np.ndarray, n_states: int) -> NDArray[np.float64]:
    """Validate the initial state distribution.

    Args:
        start: Candidate ``(n_states,)`` distribution.
        n_states: Expected length.

    Returns:
        The validated float64 vector.

    Raises:
        ValueError: If ``start`` is not 1-D of length ``n_states``, is not
            finite/non-negative, or does not sum to 1 within tolerance.
    """
    vec = np.asarray(start, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"start must be 1-D, got shape {vec.shape}")
    if vec.shape[0] != n_states:
        raise ValueError(f"start has length {vec.shape[0]} but the model has {n_states} states")
    if not np.all(np.isfinite(vec)):
        raise ValueError("start contains NaN or infinite entries")
    if np.any(vec < 0.0):
        raise ValueError("start contains negative probabilities")
    total = float(vec.sum())
    if abs(total - 1.0) > _STOCHASTIC_TOL:
        raise ValueError(f"start must sum to 1, got {total!r} (tolerance {_STOCHASTIC_TOL})")
    return vec


def _validate_obs(obs_indices: object, n_obs: int) -> NDArray[np.int64]:
    """Validate and coerce an observation index sequence.

    Args:
        obs_indices: Sequence of integer observation indices.
        n_obs: Number of observation values; valid indices are ``0..n_obs-1``.

    Returns:
        A 1-D int64 array of length ``T >= 1``.

    Raises:
        ValueError: If the sequence is empty, not 1-D, non-integral, or contains
            an index outside ``[0, n_obs)``.
    """
    arr = np.asarray(obs_indices)
    if arr.ndim != 1:
        raise ValueError(f"obs_indices must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("obs_indices must be non-empty")
    if not np.issubdtype(arr.dtype, np.integer):
        # Reject 3.5 but accept an all-integral float array (e.g. from np.zeros).
        float_arr = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(float_arr)) or np.any(float_arr != np.floor(float_arr)):
            raise ValueError(f"obs_indices must contain integers, got dtype {arr.dtype}")
        arr = float_arr.astype(np.int64)
    idx = arr.astype(np.int64, copy=False)
    if int(idx.min()) < 0 or int(idx.max()) >= n_obs:
        raise ValueError(
            f"obs_indices out of range: values must lie in [0, {n_obs}), got "
            f"min={int(idx.min())}, max={int(idx.max())}"
        )
    return idx


def _validate_model(
    emission: NDArray[np.float64] | np.ndarray,
    transition: NDArray[np.float64] | np.ndarray,
    start: NDArray[np.float64] | np.ndarray,
    obs_indices: object,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Validate a full ``(emission, transition, start, obs)`` bundle.

    Args:
        emission: ``(n_obs, n_states)`` column-stochastic emission.
        transition: ``(n_states, n_states)`` column-stochastic transition.
        start: ``(n_states,)`` initial distribution.
        obs_indices: Observation index sequence.

    Returns:
        The four validated arrays, in the same order.

    Raises:
        ValueError: On any stochasticity, shape, or index violation.
    """
    emis = _validate_column_stochastic(emission, "emission")
    trans = _validate_column_stochastic(transition, "transition")
    n_obs, n_states = emis.shape
    if trans.shape != (n_states, n_states):
        raise ValueError(
            f"transition must be square with side n_states={n_states} to match "
            f"emission's shape {emis.shape}, got {trans.shape}"
        )
    start_vec = _validate_start(start, n_states)
    obs = _validate_obs(obs_indices, n_obs)
    return emis, trans, start_vec, obs


# ── FORWARD FILTERING ─────────────────────────────────────────────────────────


def forward_filter(
    emission: NDArray[np.float64] | np.ndarray,
    transition: NDArray[np.float64] | np.ndarray,
    start: NDArray[np.float64] | np.ndarray,
    obs_indices: object,
) -> tuple[NDArray[np.float64], float]:
    """Forward filtering (the scaled alpha recursion).

    ``posteriors[t] = P(state_t | obs_0 .. obs_t)`` and
    ``log_evidence = sum_t log P(obs_t | obs_0 .. obs_{t-1})`` in nats.

    The recursion is::

        pred_0 = start
        pred_t = transition @ posteriors[t-1]          for t >= 1
        Z_t    = sum_j emission[obs_t, j] * pred_t[j]  = P(obs_t | obs_<t)
        posteriors[t] = emission[obs_t, :] * pred_t / Z_t

    Each step renormalizes, so the messages cannot underflow; the evidence is
    accumulated from ``log Z_t`` instead of from a shrinking joint.

    Args:
        emission: ``(n_obs, n_states)``, column-stochastic over ``n_obs``.
        transition: ``(n_states, n_states)`` with
            ``transition[i, j] = P(next = i | cur = j)``, column-stochastic over ``i``.
        start: ``(n_states,)`` distribution over ``state_0``, sums to 1.
        obs_indices: Length-``T`` sequence of observation indices.

    Returns:
        ``(posteriors, log_evidence)`` where ``posteriors`` has shape
        ``(T, n_states)`` with rows summing to 1, and ``log_evidence`` is a
        float in nats.

    Raises:
        ValueError: If any input violates the shape / stochasticity / index
            contract documented above.
    """
    emis, trans, start_vec, obs = _validate_model(emission, transition, start, obs_indices)
    n_steps = int(obs.shape[0])
    n_states = int(emis.shape[1])

    posteriors = np.empty((n_steps, n_states), dtype=np.float64)
    log_evidence = 0.0
    predictive = start_vec

    for t in range(n_steps):
        unnormalized = emis[obs[t], :] * predictive
        norm = float(unnormalized.sum())
        log_evidence += float(np.log(max(norm, _LOG_FLOOR)))
        if norm <= 0.0:
            # Every state assigns zero probability to this observation. Fall back
            # to the predictive so the belief stays a distribution and later
            # timesteps remain interpretable; the log-evidence already recorded
            # the floor penalty for this step.
            logger.warning(
                "forward_filter: zero likelihood for obs=%d at t=%d; "
                "falling back to the predictive belief",
                int(obs[t]),
                t,
            )
            posteriors[t] = predictive
        else:
            posteriors[t] = unnormalized / norm
        if t + 1 < n_steps:
            predictive = trans @ posteriors[t]

    return posteriors, log_evidence


def forward_backward(
    emission: NDArray[np.float64] | np.ndarray,
    transition: NDArray[np.float64] | np.ndarray,
    start: NDArray[np.float64] | np.ndarray,
    obs_indices: object,
) -> NDArray[np.float64]:
    """Smoothed posteriors ``P(state_t | obs_0 .. obs_{T-1})``.

    Runs :func:`forward_filter` for the alphas, then the backward recursion::

        beta_{T-1}(j) = 1
        beta_t(j)     = sum_i transition[i, j] * emission[obs_{t+1}, i] * beta_{t+1}(i)

    and combines them as ``smoothed_t ∝ posteriors[t] * beta_t``. Each ``beta_t``
    is rescaled to sum to 1; the scale factor is state-independent, so it
    cancels exactly in the ``smoothed_t`` normalization while preventing
    underflow over long sequences.

    At ``t = T - 1`` there is no future evidence, so the smoothed posterior
    equals the filtered one. At earlier timesteps the two differ whenever
    future observations are informative — see ``tests/test_oracle.py`` for the
    non-vacuity control on that.

    Args:
        emission: ``(n_obs, n_states)``, column-stochastic over ``n_obs``.
        transition: ``(n_states, n_states)``, ``transition[i, j] = P(next=i | cur=j)``.
        start: ``(n_states,)`` distribution over ``state_0``.
        obs_indices: Length-``T`` sequence of observation indices.

    Returns:
        ``(T, n_states)`` array of smoothed posteriors, rows summing to 1.

    Raises:
        ValueError: If any input violates the documented contract.
    """
    emis, trans, start_vec, obs = _validate_model(emission, transition, start, obs_indices)
    filtered, _ = forward_filter(emis, trans, start_vec, obs)
    n_steps = int(obs.shape[0])
    n_states = int(emis.shape[1])

    betas = np.ones((n_steps, n_states), dtype=np.float64)
    for t in range(n_steps - 2, -1, -1):
        weighted = emis[obs[t + 1], :] * betas[t + 1]  # over next state i
        raw = trans.T @ weighted  # raw[j] = sum_i T[i, j] * weighted[i]
        scale = float(raw.sum())
        betas[t] = raw / scale if scale > 0.0 else np.ones(n_states) / n_states

    smoothed = filtered * betas
    row_sums = smoothed.sum(axis=1, keepdims=True)
    degenerate = row_sums[:, 0] <= 0.0
    if np.any(degenerate):
        logger.warning(
            "forward_backward: %d timestep(s) had zero smoothed mass; "
            "falling back to the filtered belief there",
            int(degenerate.sum()),
        )
        smoothed[degenerate] = filtered[degenerate]
        row_sums = smoothed.sum(axis=1, keepdims=True)
    return smoothed / row_sums


def viterbi(
    emission: NDArray[np.float64] | np.ndarray,
    transition: NDArray[np.float64] | np.ndarray,
    start: NDArray[np.float64] | np.ndarray,
    obs_indices: object,
) -> list[int]:
    """Most likely state path (MAP path) via the Viterbi recursion in log space.

    Maximizes ``log P(states, obs)`` over whole paths::

        delta_0[i] = log start[i] + log emission[obs_0, i]
        delta_t[i] = log emission[obs_t, i]
                     + max_j (log transition[i, j] + delta_{t-1}[j])

    This is the joint-MAP path, which is not in general the sequence of
    per-timestep argmaxes of :func:`forward_backward`.

    Args:
        emission: ``(n_obs, n_states)``, column-stochastic over ``n_obs``.
        transition: ``(n_states, n_states)``, ``transition[i, j] = P(next=i | cur=j)``.
        start: ``(n_states,)`` distribution over ``state_0``.
        obs_indices: Length-``T`` sequence of observation indices.

    Returns:
        List of ``T`` state indices — the highest-joint-probability path. Ties
        resolve to the lowest state index (``np.argmax`` semantics).

    Raises:
        ValueError: If any input violates the documented contract.
    """
    emis, trans, start_vec, obs = _validate_model(emission, transition, start, obs_indices)
    n_steps = int(obs.shape[0])
    n_states = int(emis.shape[1])

    log_emis = np.log(np.maximum(emis, _LOG_FLOOR))
    log_trans = np.log(np.maximum(trans, _LOG_FLOOR))
    log_start = np.log(np.maximum(start_vec, _LOG_FLOOR))

    delta = log_start + log_emis[obs[0], :]
    backpointers = np.zeros((n_steps, n_states), dtype=np.int64)

    for t in range(1, n_steps):
        # scores[i, j] = log P(next=i | cur=j) + delta_{t-1}[j]
        scores = log_trans + delta[np.newaxis, :]
        best_prev = np.argmax(scores, axis=1)
        backpointers[t] = best_prev
        delta = log_emis[obs[t], :] + scores[np.arange(n_states), best_prev]

    path = np.zeros(n_steps, dtype=np.int64)
    path[n_steps - 1] = int(np.argmax(delta))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backpointers[t, path[t]]
    return [int(s) for s in path]


# ── PERPLEXITY ────────────────────────────────────────────────────────────────


def sequence_perplexity(
    emission: NDArray[np.float64] | np.ndarray,
    transition: NDArray[np.float64] | np.ndarray,
    start: NDArray[np.float64] | np.ndarray,
    obs_indices: object,
) -> float:
    """Per-symbol perplexity ``exp(-log_evidence / T)`` from :func:`forward_filter`.

    Analytic anchors this must reproduce exactly:

    * Uniform transition, uniform start, identity emission over ``n`` symbols
      gives perplexity ``n`` for *any* observation sequence.
    * A deterministic chain observed along its own trajectory gives ``1.0``.

    Args:
        emission: ``(n_obs, n_states)``, column-stochastic over ``n_obs``.
        transition: ``(n_states, n_states)``, ``transition[i, j] = P(next=i | cur=j)``.
        start: ``(n_states,)`` distribution over ``state_0``.
        obs_indices: Length-``T`` sequence of observation indices.

    Returns:
        Perplexity in symbols. Lower is better; ``n_obs`` is the uniform bar.

    Raises:
        ValueError: If any input violates the documented contract.
    """
    _, log_evidence = forward_filter(emission, transition, start, obs_indices)
    n_steps = len(np.asarray(obs_indices))
    return float(np.exp(-log_evidence / n_steps))


def identity_emission(n: int) -> NDArray[np.float64]:
    """Return the ``(n, n)`` identity — the emission for a directly-observed state.

    Args:
        n: Number of states (equivalently observations).

    Returns:
        ``(n, n)`` float64 identity matrix, trivially column-stochastic.

    Raises:
        ValueError: If ``n`` is not a positive integer.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise ValueError(f"n must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return np.eye(int(n), dtype=np.float64)


def markov_chain_perplexity(
    transition: NDArray[np.float64] | np.ndarray,
    start: NDArray[np.float64] | np.ndarray,
    obs_indices: object,
) -> float:
    """Perplexity of a *directly observed* Markov chain.

    Convenience wrapper: identical by construction to
    ``sequence_perplexity(identity_emission(n), transition, start, obs)``.
    This is the shape used to audit a bigram / n-gram language model, where the
    symbol *is* the state and the learned statistics live in the transition.

    Args:
        transition: ``(n_states, n_states)``, ``transition[i, j] = P(next=i | cur=j)``.
        start: ``(n_states,)`` distribution over the first symbol.
        obs_indices: Length-``T`` sequence of symbol indices.

    Returns:
        Per-symbol perplexity.

    Raises:
        ValueError: If any input violates the documented contract.
    """
    trans = _validate_column_stochastic(transition, "transition")
    if trans.shape[0] != trans.shape[1]:
        raise ValueError(f"transition must be square, got {trans.shape}")
    return sequence_perplexity(identity_emission(trans.shape[0]), trans, start, obs_indices)


# ── WORD LEVEL (Stage 3) ──────────────────────────────────────────────────────


def word_path_logprob(
    word: str,
    word_index: int,
    start_word: NDArray[np.float64] | np.ndarray,
    n_letters: int = 26,
) -> float:
    """Log-probability the word level assigns to a word's full emission path.

    The Stage 3 generative model factorizes the hidden state into word identity
    and position. ``B_word`` is the identity, ``B_pos`` is the +1 shift, and the
    emission is deterministic — letter ``k`` of word ``w`` at position ``k``, the
    end-of-word symbol thereafter. With no stochasticity anywhere except the
    initial distribution over words, the probability of the whole path collapses
    to ``start_word[word_index]``.

    That collapse is the identity the level is pinned by. It is derived here,
    without pymdp, so that the agreement in
    ``tests/test_stage3_gate.py::test_path_probability_equals_the_word_unigram``
    is between two independent derivations rather than one implementation
    agreeing with itself. Three of the five mis-wirings enumerated in
    ``docs/decisions.md`` entry 14 still beat the level's accuracy bar; all three
    change the word prior, so all three fail *this*.

    Args:
        word: The word, lowercase ASCII letters only.
        word_index: Its index in the vocabulary, indexing ``start_word``.
        start_word: ``(n_vocab,)`` distribution over word identity, summing to 1.
        n_letters: Size of the letter alphabet, excluding the end symbol.

    Returns:
        Natural log-probability of the emission path.

    Raises:
        ValueError: If the word is not lowercase ASCII, the index is out of
            range, or ``start_word`` is not a distribution.
    """
    if not word or not all("a" <= ch <= "z" for ch in word):
        raise ValueError(f"word must be non-empty lowercase ASCII, got {word!r}")
    prior = np.asarray(start_word, dtype=np.float64).ravel()
    if not np.all(np.isfinite(prior)):
        raise ValueError("start_word contains non-finite values")
    if abs(float(prior.sum()) - 1.0) > _STOCHASTIC_TOL:
        raise ValueError(f"start_word must sum to 1, got {float(prior.sum())!r}")
    if not 0 <= word_index < prior.size:
        raise ValueError(f"word_index {word_index} outside vocabulary of {prior.size}")
    if n_letters < 1:
        raise ValueError(f"n_letters must be positive, got {n_letters}")
    return float(np.log(max(float(prior[word_index]), _LOG_FLOOR)))


def word_path_perplexity(
    tokens: list[str],
    word_to_index: dict[str, int],
    start_word: NDArray[np.float64] | np.ndarray,
) -> tuple[float, int]:
    """Per-emission perplexity of the word level over a token stream.

    An *emission* is one letter or the end-of-word symbol, so a token of length
    ``L`` contributes ``L + 1`` emissions. Reporting per emission rather than per
    token is what makes the number comparable with the letter-bigram bar, which
    is scored on the identical positions.

    This quantity is **not** comparable with ``claim:stage1_letter_perplexity``:
    that is over 30 running-text symbols including space and punctuation, this is
    over 27. Perplexities over different alphabets do not belong in one table.

    Args:
        tokens: Held-out word tokens, every one of which must be in vocabulary.
        word_to_index: Vocabulary map.
        start_word: ``(n_vocab,)`` distribution over word identity.

    Returns:
        ``(perplexity, n_emissions)``.

    Raises:
        ValueError: If ``tokens`` is empty or contains an out-of-vocabulary word.
            Falling back to a smoothed value would report a number for a model
            that is *undefined* on that token — the closed-vocabulary bound is
            the finding, not an inconvenience.
    """
    if not tokens:
        raise ValueError("tokens is empty; there is nothing to score")
    total_logp = 0.0
    n_emissions = 0
    for token in tokens:
        index = word_to_index.get(token)
        if index is None:
            raise ValueError(
                f"{token!r} is out of vocabulary. The word level assigns it no "
                f"probability at all; restrict the evaluation stream to "
                f"in-vocabulary tokens and report the coverage."
            )
        total_logp += word_path_logprob(token, index, start_word)
        n_emissions += len(token) + 1
    return float(np.exp(-total_logp / n_emissions)), n_emissions


def letter_bigram_perplexity(
    tokens: list[str],
    transition: NDArray[np.float64] | np.ndarray,
    start: NDArray[np.float64] | np.ndarray,
) -> tuple[float, int]:
    """Per-emission perplexity of the letter-bigram bar on the identical positions.

    The bar for Stage 3: the strongest model that does *not* know the lexicon.
    It is trained on the same tokens as the level and scored over the same
    emissions, so the comparison isolates lexical knowledge rather than any
    difference in corpus or position set (decision 2).

    Tokens are validated against the same predicate
    :func:`word_path_logprob` applies, because the two are the two sides of one
    comparison. ``ord(ch) - 97`` maps code points 70..96 (``'F'``..``'`'``) to
    -27..-1, which NumPy accepts as wrap-around indices, so an unvalidated
    ``'aF'`` would score exactly like ``'aa'`` and be reported as a measurement.
    The model side already raises on such a token; a bar that does not would make
    the comparison fail open on exactly the input that breaks it.

    Args:
        tokens: The same held-out tokens the level is scored on, lowercase ASCII
            letters only.
        transition: ``(27, 27)`` column-stochastic, index 26 is end-of-word.
        start: ``(27,)`` distribution over the first symbol of a word.

    Returns:
        ``(perplexity, n_emissions)``.

    Raises:
        ValueError: If ``tokens`` is empty, a token is not a non-empty lowercase
            ASCII-letter word, or the matrices violate the contract.
    """
    if not tokens:
        raise ValueError("tokens is empty; there is nothing to score")
    for position, token in enumerate(tokens):
        if not token or not all("a" <= ch <= "z" for ch in token):
            raise ValueError(
                f"tokens[{position}] = {token!r} must be a non-empty lowercase ASCII "
                f"word. Indexing it as ord(ch) - 97 would produce a negative index "
                f"that NumPy reads from the far end of the row, scoring a different "
                f"symbol without any error."
            )
    trans = _validate_column_stochastic(transition, "transition")
    n_obs = trans.shape[0]
    first = _validate_start(start, n_obs)
    end_index = n_obs - 1

    total_logp = 0.0
    n_emissions = 0
    for token in tokens:
        seq = [ord(ch) - 97 for ch in token] + [end_index]
        total_logp += float(np.log(max(float(first[seq[0]]), _LOG_FLOOR)))
        for prev, nxt in pairwise(seq):
            total_logp += float(np.log(max(float(trans[nxt, prev]), _LOG_FLOOR)))
        n_emissions += len(seq)
    return float(np.exp(-total_logp / n_emissions)), n_emissions
