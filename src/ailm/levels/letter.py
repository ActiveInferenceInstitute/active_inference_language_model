"""Letter-level active inference agent for AILM Stage 1.

A single-factor, single-modality pymdp generative model at the character level,
parameterized by Brown corpus letter bigram statistics:

  - Hidden factor:      "current letter" — 30 states  (ALPHABET)
  - Observation modality: "current letter" — 30 observations (ALPHABET)
  - A matrix (1, 30, 30): identity — observing a letter reveals the state exactly
  - B matrix (1, 30, 30, 1): B[0, i, j, 0] = P(next letter i | current letter j)
  - D: uniform prior over the initial letter

The bigram counts parameterize the TRANSITION (B), not the emission (A). Putting
them in A instead makes the model *anti*-informative — the posterior would be
over the letter that *preceded* the observation — and pymdp accepts either
placement, so nothing structural objects. See ``docs/decisions.md``.

Inference cycle (per letter)
-----------------------------
1. empirical_prior = predictive belief over the current letter (from B @ q_{t-1})
2. obs = current letter index
3. qs = agent.infer_states(obs, empirical_prior)  — collapses onto the observed
   letter, since A is the identity
4. new_prior = agent.update_empirical_prior(action=0, qs)  — pushes the posterior
   through B, producing P(next letter | observed letter)

Perplexity
----------
P(letter_t | prior_t) = sum_j A[0, letter_t, j] * prior_t[j]
Cross-entropy H = -mean(log P(letter_t | prior_t))
Perplexity = exp(H)

Uniform baseline: exp(log(30)) = 30.0. The gate is stated against the measured
context-free unigram baseline and the order-1 information floor rather than
against a margin below uniform; ``docs/decisions.md`` records why, and
``docs/gate_design.md`` is the procedure.

Unlike the L2 models, this level does NOT reset the empirical prior per word: its
corpus is running text, where the transition across a space is a real letter
bigram (space is one of the thirty ALPHABET symbols).

ALPHABET
--------
Exactly 30 symbols: a-z (26) + space + period + apostrophe + hyphen.
Sorted order (used for deterministic indexing):
  [' ', "'", '-', '.', 'a', 'b', ..., 'z']
  idx: 0    1    2    3    4    5  ...  29
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig
from ailm.corpus.brown import (
    BROWN_BIGRAM_ARTIFACT_SCHEMA,
    validate_brown_artifact_provenance,
)
from ailm.levels._stochastic import column_stochastic
from ailm.resources import resolve_vendored_artifact

if TYPE_CHECKING:
    from pymdp.agent import Agent as PymdpAgent

logger = logging.getLogger(__name__)

# ── ALPHABET ─────────────────────────────────────────────────────────────────
# Exactly 30 symbols: lowercase a-z + space + period + apostrophe + hyphen.
ALPHABET: frozenset[str] = frozenset("abcdefghijklmnopqrstuvwxyz ." + "'" + "-")
if len(ALPHABET) != 30:  # module-load invariant; explicit raise survives `python -O`
    raise AssertionError(f"BUG: ALPHABET has {len(ALPHABET)} symbols, expected 30")

#: Sorted list of ALPHABET symbols for deterministic indexing.
ALPHA_LIST: list[str] = sorted(ALPHABET)

#: Character → integer index mapping.
ALPHA_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(ALPHA_LIST)}

#: Integer index → character mapping.
IDX_TO_ALPHA: dict[int, str] = {i: c for i, c in enumerate(ALPHA_LIST)}

_ALPHA_SET: frozenset[str] = ALPHABET  # internal alias for speed

N_SYMBOLS: int = len(ALPHA_LIST)  # 30


# ── TEXT NORMALIZATION ────────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Normalize text to the ALPHABET character set.

    Lowercases the text, then replaces every character not in ALPHABET with
    a space. Does NOT collapse consecutive spaces — the caller can do that
    if needed, but for bigram counting preserving positions is simpler.

    Args:
        text: Raw text (any casing, any characters).

    Returns:
        String containing only ALPHABET characters (lowercase letters,
        space, period, apostrophe, hyphen).
    """
    lowered = text.lower()
    return "".join(c if c in _ALPHA_SET else " " for c in lowered)


# ── CORPUS TRAINING ───────────────────────────────────────────────────────────


def train_on_corpus(sentences: list[str]) -> dict[tuple[str, str], int]:
    """Compute letter bigram counts from a list of sentence strings.

    Normalizes each sentence to ALPHABET, then counts consecutive character
    pairs (prev_char, curr_char). Only pairs where both characters are in
    ALPHABET are counted (after normalization all chars are in ALPHABET, so
    this is always satisfied).

    Args:
        sentences: List of sentence strings (raw text; will be normalized).

    Returns:
        Dict mapping (prev_char, curr_char) → count. Empty if no sentences.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for sent in sentences:
        text = normalize_text(sent)
        for i in range(len(text) - 1):
            counts[(text[i], text[i + 1])] += 1
    return dict(counts)


# ── LETTER TRANSITION CONSTRUCTION ────────────────────────────────────────────


def build_letter_transition(
    counts: dict,
    *,
    config: LevelConfig | None = None,
) -> jnp.ndarray:
    """Build the column-stochastic letter transition from bigram counts.

    The returned matrix parameterizes the agent's B transition:
      B[0, i, j, 0] = P(next letter i | current letter j).
    The builder returns the batch axis but leaves the action axis to
    :func:`build_letter_agent`; the agent's A emission is the identity.

    Column j must sum to 1 (normalized). Add-k smoothing ensures no zero
    probabilities even for unseen bigrams: a cell with count ``c`` in a column
    of observed total ``n`` becomes ``(c + k) / (n + k * 30)``, so an *unseen*
    bigram lands at exactly ``k / (n + k * 30)``.

    Keys can be either tuple pairs (prev, curr) or JSON-serialised strings
    "prev,curr" (as produced by json.dumps with comma-joined keys).

    Args:
        counts: Dict mapping bigram pairs to counts. Supports:
                - tuple keys:  {('a', 'b'): 5, ...}
                - string keys: {"a,b": 5, ...}  (from JSON round-trip)
        config: Keyword-only. Supplies the pseudocount ``k`` via
            :attr:`ailm.config.LevelConfig.smoothing`. ``None`` uses
            :data:`ailm.config.DEFAULT_LEVEL_CONFIG` (k = 1, add-1 Laplace).

    Returns:
        jnp.Array of shape (1, 30, 30) — column-stochastic.
    """
    cfg = DEFAULT_LEVEL_CONFIG if config is None else config
    # Add-k smoothing: every cell starts at the pseudocount.
    count_matrix = np.full((N_SYMBOLS, N_SYMBOLS), float(cfg.smoothing), dtype=np.float64)

    for key, cnt in counts.items():
        if isinstance(key, str):
            # JSON-serialised key: "prev_char,curr_char"
            # Handle single-char keys: both chars should be 1-char each
            parts = key.split(",", 1)
            if len(parts) != 2:
                continue
            prev_char, curr_char = parts[0], parts[1]
        else:
            # Tuple key
            prev_char, curr_char = key[0], key[1]

        if prev_char in ALPHA_TO_IDX and curr_char in ALPHA_TO_IDX:
            j = ALPHA_TO_IDX[prev_char]  # state = previous letter
            i = ALPHA_TO_IDX[curr_char]  # observation = current letter
            count_matrix[i, j] += float(cnt)

    a_matrix = column_stochastic(count_matrix)  # shape (30, 30)

    # Add batch dimension → (1, 30, 30)
    return jnp.array(a_matrix[np.newaxis, :, :])


# Compatibility for historical notebooks and callers. This is the same function
# object, not a second estimator; new code names the transition it constructs.
build_unigram_a = build_letter_transition


# ── AGENT CONSTRUCTION ────────────────────────────────────────────────────────


def build_letter_agent(a_matrix: jnp.ndarray) -> PymdpAgent:
    """Build a pymdp Agent from a (1, 30, 30) column-stochastic bigram matrix.

    The agent uses:
    - A: identity emission (observing a letter reveals the letter state)
    - B: the supplied bigram matrix as the transition, P(next | current)
    - C: zero preferences (no goal-directed behavior at this level)
    - D: uniform prior over the initial letter

    The keyword ``a_matrix`` is retained for compatibility with historical
    callers, but it parameterizes the *transition*. See the module docstring for why
    placing it in A instead is anti-informative.

    Args:
        a_matrix: jnp.Array of shape (1, 30, 30) — column-stochastic, as
                  returned by :func:`build_letter_transition`.

    Returns:
        pymdp.agent.Agent configured for letter-level inference.

    Raises:
        ValueError: If ``a_matrix`` does not have shape (1, 30, 30).
    """
    from pymdp.agent import Agent

    # Ensure a_matrix is a JAX array with correct shape
    a_jnp = jnp.array(a_matrix) if not hasattr(a_matrix, "shape") else a_matrix
    if a_jnp.shape != (1, N_SYMBOLS, N_SYMBOLS):
        raise ValueError(
            f"Expected bigram matrix shape (1, {N_SYMBOLS}, {N_SYMBOLS}), got {a_jnp.shape}"
        )

    # B: bigram transition — P(next letter = i | current letter = j).
    # This is where the corpus statistics live: the *dynamics* of the letter
    # chain, not the emission. Shape (batch=1, next=30, current=30, actions=1).
    # pymdp validates normalization along axis=1, which is satisfied because
    # build_letter_transition returns a column-stochastic matrix.
    B = [a_jnp[:, :, :, jnp.newaxis]]

    # A: identity emission — observing a letter reveals the letter state exactly.
    # Shape (batch=1, n_obs=30, n_states=30).
    A = [jnp.eye(N_SYMBOLS)[jnp.newaxis, :, :]]

    # C: zero log-preferences (no preference over observations)
    C = [jnp.zeros((1, N_SYMBOLS))]

    # D: uniform prior over initial state (previous letter unknown)
    D = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]

    return Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        num_controls=[1],
        batch_size=1,
    )


# ── PERPLEXITY ────────────────────────────────────────────────────────────────


def compute_perplexity(agent: PymdpAgent, text: str) -> float:
    """Compute per-letter perplexity of text under the agent.

    Perplexity = exp(cross-entropy in nats per letter).
    Uniform baseline: exp(log(30)) = 30.0 for a 30-symbol alphabet.

    Algorithm:
    1. Normalize text to ALPHABET.
    2. For each character c_t with prior π_t:
       a. Compute predictive prob: P(c_t | π_t) = sum_j A[0, c_t, j] * π_t[j]
       b. Accumulate log P(c_t | π_t)
       c. Update posterior: qs_t = infer_states([c_t], π_t)
       d. Next prior: π_{t+1} = update_empirical_prior(0, qs_t)
    3. Perplexity = exp(-mean log probs)

    Fewer than two scorable characters: a documented refusal, not an error
    ----------------------------------------------------------------------
    An input with fewer than two scorable characters returns ``float(N_SYMBOLS)``
    — the uniform value — rather than raising. This diverges from
    :func:`ailm.verification.oracle.forward_filter`, which raises on an empty
    sequence, and the divergence is deliberate rather than accidental:

    * ``N_SYMBOLS`` sits on the **failing** side of every Stage 1 comparison. A
      gate asks whether the model beats the uniform bar; returning exactly the
      bar can never pass one, so a degenerate population cannot be laundered into
      a passing measurement through this path.
    * The value is pinned as a contract by ``tests/test_letter.py``
      (``test_empty_text_returns_n_symbols``,
      ``test_single_character_returns_n_symbols``,
      ``test_single_char_after_normalization_returns_n_symbols``).

    The residual risk this leaves is a *reporting* one: an evaluation stream that
    was silently emptied upstream is reported as a measured uniform perplexity
    rather than as an error. Callers that build the stream themselves should
    check its length before scoring; the recorders in ``scripts/`` score a
    fixture-pinned text whose length is asserted elsewhere.

    Args:
        agent: A trained pymdp letter agent (from build_letter_agent).
        text: Text to evaluate. Will be normalized to ALPHABET.

    Returns:
        Perplexity (float). Lower = better. Uniform baseline = 30.0.
        Returns ``float(N_SYMBOLS)`` for fewer than two scorable characters; see
        the refusal note above.
    """
    normalized = normalize_text(text)
    chars = [c for c in normalized if c in _ALPHA_SET]

    if len(chars) < 2:
        # Fail-closed by value: the uniform bar can never clear a gate that asks
        # the model to beat the uniform bar. See the docstring note.
        return float(N_SYMBOLS)

    # Initialize empirical prior: uniform over all 30 states
    empirical_prior: list[jnp.ndarray] = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]

    a_mat = agent.A[0]  # shape (1, 30, 30)
    log_probs: list[float] = []
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    for char in chars:
        obs_idx = ALPHA_TO_IDX[char]

        # Predictive distribution: P(obs=i | prior) = A[0, i, :] @ prior_vec
        # empirical_prior[0] is (1, 30): the initial value is built that way above
        # and pymdp's update_empirical_prior returns the same batch-leading shape,
        # so ravel() is unconditional — it is the identity on an already-flat array.
        prior_vec = empirical_prior[0].ravel()  # -> (30,)

        # a_mat[0] is (30, 30); a_mat[0, obs_idx, :] is the obs_idx-th row → (30,)
        a_row = a_mat[0, obs_idx, :]  # P(obs=char | state=j) for each j
        pred_prob = jnp.dot(a_row, prior_vec)  # scalar

        # Numerical stability clamp
        pred_prob = jnp.clip(pred_prob, 1e-10, 1.0)
        log_probs.append(float(jnp.log(pred_prob)))

        # Bayesian update: posterior belief over states given observation
        obs = [jnp.array([obs_idx])]
        qs = agent.infer_states(obs, empirical_prior)

        # Next prior = posterior (state = what letter we just processed)
        empirical_prior = agent.update_empirical_prior(action, qs)

    # Cross-entropy in nats per letter
    cross_entropy = -sum(log_probs) / len(log_probs)
    return float(jnp.exp(jnp.array(cross_entropy)))


# ── CORPUS CACHE ──────────────────────────────────────────────────────────────


def load_brown_counts(vendor_dir: Path | None = None) -> dict:
    """Load cached Brown corpus bigram counts from vendor/brown_counts.json.

    The file is generated offline by scripts/generate_counts.py and
    committed to vendor/. Tests must not make network or corpus calls — they
    load from this cache.

    The versioned JSON envelope retains the pinned raw-source identity. Its
    ``counts`` keys are "prev_char,curr_char" strings (e.g. "a,b"), which
    :func:`build_letter_transition` accepts directly.

    Args:
        vendor_dir: Path to the vendor/ directory. If None, reads the embedded
            wheel resource when installed and the top-level ``vendor/`` copy in
            a source checkout.

    Returns:
        Dict mapping "prev,curr" → count (integers).

    Raises:
        FileNotFoundError: If vendor/brown_counts.json does not exist.
    """
    counts_file = resolve_vendored_artifact("brown_counts.json", directory=vendor_dir)
    if not counts_file.is_file():
        raise FileNotFoundError(
            f"Brown counts not found at {counts_file}. "
            "Run scripts/generate_counts.py to generate it."
        )
    payload = json.loads(counts_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != BROWN_BIGRAM_ARTIFACT_SCHEMA:
        schema = payload.get("schema") if isinstance(payload, dict) else None
        raise ValueError(
            f"Unexpected Brown counts schema {schema!r}; regenerate with "
            "scripts/generate_counts.py from the pinned raw source"
        )
    validate_brown_artifact_provenance(payload)

    provenance = payload.get("provenance")
    counts = payload.get("counts")
    if not isinstance(provenance, Mapping) or not isinstance(counts, dict) or not counts:
        raise ValueError("Brown counts artifact must contain non-empty provenance and counts")
    if provenance.get("canonical") is not True:
        raise ValueError("Brown counts artifact is explicitly noncanonical")
    if provenance.get("alphabet") != ALPHA_LIST:
        raise ValueError("Brown counts artifact alphabet does not match Level 1")

    for key, value in counts.items():
        if not isinstance(key, str):
            raise ValueError(f"Brown bigram key is not text: {key!r}")
        parts = key.split(",", 1)
        if len(parts) != 2 or parts[0] not in ALPHABET or parts[1] not in ALPHABET:
            raise ValueError(f"Brown artifact contains an invalid bigram key: {key!r}")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Brown artifact contains an invalid count for {key!r}: {value!r}")

    total = sum(counts.values())
    if provenance.get("distinct_bigrams") != len(counts):
        raise ValueError("Brown distinct-bigram provenance does not match counts")
    if provenance.get("bigram_tokens") != total:
        raise ValueError("Brown bigram-token provenance does not match counts")
    if provenance.get("normalized_characters") != total + 1:
        raise ValueError("Brown normalized-character provenance does not match counts")
    return counts
