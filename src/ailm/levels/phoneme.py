"""Phoneme-level active inference agent for AILM Stage 2.

Two generative models live in this module, both single-factor / single-modality
pymdp agents over the 39 canonical ARPAbet phones.

**Part A — the phoneme sequence model** (the Stage 1 letter model, one level down):

  - Hidden factor:        "current phone" — 39 states  (PHONES)
  - Observation modality: "current phone" — 39 observations (PHONES)
  - A (1, 39, 39):        identity — observing a phone reveals the state exactly
  - B (1, 39, 39, 1):     B[0, i, j, 0] = P(next phone i | current phone j)
  - D (1, 39):            phone unigram distribution (the start prior)

**Part B — the G2P model** (the hierarchical letter <-> phoneme coupling). Here
the two levels genuinely differ, which is the point of Stage 2:

  - Hidden factor:        "current phone"  — 39 states  (PHONES)
  - Observation modality: "current letter" — 30 observations (letter.ALPHA_LIST)
  - A (1, 30, 39):        A[0, l, p] = P(letter l | phone p) — a real emission
  - B (1, 39, 39, 1):     phone bigram transition, trained on the TRAIN split only
  - D (1, 39):            distribution over the first phone of a training word

Inference cycle (per position, both models)
-------------------------------------------
1. ``prior_t`` = predictive belief over the current phone (``B @ q_{t-1}``, D at 0)
2. ``qs = agent.infer_states(obs_t, prior_t)`` — ``obs_t`` is a phone in Part A,
   a letter in Part B
3. ``prior_{t+1} = agent.update_empirical_prior(action=0, qs)``

Filtering, not smoothing: position ``t`` sees observations ``1..t`` only. Part A
scores ``P(phone_t | prior_t) = sum_j A[0, phone_t, j] * prior_t[j]`` and reports
``exp(-mean log P)``; Part B argmaxes the posterior at each position. The prior
resets to D at every word boundary — a word boundary is not a phoneme transition.

The invariant
-------------
Learned sequence dynamics live in **B**; the emission is **A**. pymdp cannot
enforce it (a bigram matrix satisfies ``_validate()`` in either slot), and beating
the context-free baseline is only a *necessary* condition — at least one mis-wired
emission clears that bar. So both halves are verified numerically against
closed-form oracles, :mod:`ailm.gnn` checks each live agent against its GNN spec,
and the comparator is the context-free :func:`per_letter_argmax_baseline`, never
uniform. Evidence for all of it: ``docs/decisions.md``.

Arithmetic is float32 (JAX's default), so analytic anchors land within about 2e-7
relative — compare with a tolerance, never for equality.

PHONES
------
The 39 canonical ARPAbet phones with stress digits stripped, sorted for
deterministic indexing. Sourced from
:data:`ailm.corpus.cmudict.CANONICAL_ARPABET` so this module imports cleanly
without the vendored artifact present:
  ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', ..., 'Z', 'ZH']
  idx:   0     1     2     3     4     5    6    7   ...   37    38
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig
from ailm.corpus.cmudict import CANONICAL_ARPABET
from ailm.levels._stochastic import column_stochastic as _column_stochastic
from ailm.levels.letter import ALPHA_LIST, ALPHA_TO_IDX, N_SYMBOLS, normalize_text
from ailm.resources import resolve_vendored_artifact
from ailm.statistics import exact_two_sided_binomial_p

if TYPE_CHECKING:
    from pymdp.agent import Agent as PymdpAgent

logger = logging.getLogger(__name__)

#: Public surface. ``ALPHA_LIST``, ``ALPHA_TO_IDX`` and ``N_SYMBOLS`` are
#: deliberate re-exports of the letter level: the G2P emission is indexed by
#: letter, so callers and gate tests need the same indexing this module used.
__all__ = [
    "ALPHA_LIST",
    "ALPHA_TO_IDX",
    "IDX_TO_PHONE",
    "N_PHONES",
    "N_SYMBOLS",
    "PHONES",
    "PHONE_TO_IDX",
    "baseline_accuracy",
    "baseline_positions",
    "build_g2p_agent",
    "build_g2p_emission",
    "build_g2p_start",
    "build_g2p_transition",
    "build_phone_transition",
    "build_phone_unigram",
    "build_phoneme_agent",
    "compute_phoneme_perplexity",
    "decode_phonemes",
    "decode_positions",
    "full_corpus_order1_floor_perplexity",
    "full_corpus_perplexity",
    "full_corpus_unigram_perplexity",
    "g2p_accuracy",
    "g2p_family_split",
    "g2p_split",
    "load_phoneme_counts",
    "per_letter_argmax_baseline",
]

# ── PHONEME INVENTORY ────────────────────────────────────────────────────────
#: The 39 canonical ARPAbet phones, sorted for deterministic indexing.
PHONES: tuple[str, ...] = tuple(sorted(CANONICAL_ARPABET))
if len(PHONES) != 39:  # module-load invariant; explicit raise survives `python -O`
    raise AssertionError(f"BUG: PHONES has {len(PHONES)} symbols, expected 39")

#: Phone symbol → integer index mapping.
PHONE_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(PHONES)}

#: Integer index → phone symbol mapping.
IDX_TO_PHONE: dict[int, str] = {i: p for i, p in enumerate(PHONES)}

N_PHONES: int = len(PHONES)  # 39

#: On-disk schema tag of the vendored Stage 2 artifact.
_COUNTS_SCHEMA = "ailm-cmudict-counts-v1"

#: Generator script named in the FileNotFoundError message.
_GENERATOR = "scripts/generate_phoneme_counts.py"


# ── VENDORED ARTIFACT ────────────────────────────────────────────────────────


def load_phoneme_counts(vendor_dir: Path | None = None) -> dict:
    """Load cached CMUdict phoneme statistics from ``vendor/cmudict_counts.json``.

    The file is generated offline and committed to ``vendor/``. Tests must not
    make network or corpus calls — they load from this cache.

    Payload keys (schema ``ailm-cmudict-counts-v1``):
      - ``phones``:         sorted list of the 39 canonical ARPAbet phones
      - ``phone_unigrams``: ``{"AA": count, ...}``
      - ``phone_bigrams``:  ``{"AA,B": count, ...}`` where the key is "prev,next"
      - ``g2p_pairs``:      ``[["cat", "K AE T"], ...]``, sorted, letters lowercase
      - ``test_every``:     held-out stride; index ``i`` is TEST iff ``i % n == 0``
      - ``min_word_len``, ``source_base_words``: provenance

    Args:
        vendor_dir: Path to the ``vendor/`` directory. If None, reads the
            embedded wheel resource when installed and the top-level ``vendor/``
            copy in a source checkout.

    Returns:
        The parsed JSON payload.

    Raises:
        FileNotFoundError: If ``vendor/cmudict_counts.json`` does not exist.
        ValueError: If the payload's ``_schema`` is not ``ailm-cmudict-counts-v1``,
            or its ``phones`` list disagrees with :data:`PHONES` (which would
            silently misindex every matrix built from it).
    """
    counts_file = resolve_vendored_artifact("cmudict_counts.json", directory=vendor_dir)
    if not counts_file.is_file():
        raise FileNotFoundError(
            f"Phoneme counts not found at {counts_file}. Run {_GENERATOR} to generate it."
        )
    payload = json.loads(counts_file.read_text(encoding="utf-8"))

    schema = payload.get("_schema")
    if schema != _COUNTS_SCHEMA:
        raise ValueError(
            f"{counts_file} has _schema={schema!r}, expected {_COUNTS_SCHEMA!r}. "
            f"Regenerate it with {_GENERATOR}."
        )

    phones = payload.get("phones")
    if phones is not None and tuple(phones) != PHONES:
        raise ValueError(
            f"{counts_file} 'phones' disagrees with PHONES. Every matrix built "
            f"from this artifact would be misindexed. "
            f"Missing from artifact: {sorted(set(PHONES) - set(phones))}; "
            f"unexpected in artifact: {sorted(set(phones) - set(PHONES))}."
        )

    return payload


# ── PART A: PHONEME SEQUENCE MODEL ───────────────────────────────────────────


def _split_bigram_key(key: object) -> tuple[str, str] | None:
    """Normalize a bigram key to a ``(prev, next)`` pair.

    Accepts the JSON-serialised ``"prev,next"`` string form and the in-memory
    ``("prev", "next")`` tuple form. ARPAbet symbols never contain a comma, so
    a single split is unambiguous.

    Args:
        key: A bigram key from a counts mapping.

    Returns:
        ``(prev, next)``, or None if the key is not a well-formed pair.
    """
    if isinstance(key, str):
        parts = key.split(",", 1)
        if len(parts) != 2:
            return None
        return parts[0], parts[1]
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return str(key[0]), str(key[1])
    return None


def build_phone_transition(
    bigrams: dict,
    *,
    config: LevelConfig | None = None,
) -> jnp.ndarray:
    """Build the phone transition matrix B from phone bigram counts.

    ``B[0, i, j, 0] = P(next phone = i | current phone = j)``, with add-k
    smoothing so no transition has zero probability. pymdp validates B along
    axis 1, so column ``j`` sums to 1 over the NEXT index — which is exactly the
    normalization a transition needs and *not* what an emission would need.

    Args:
        bigrams: Mapping of bigram → count. Keys may be ``"prev,next"`` strings
            (as stored in the vendored JSON) or ``("prev", "next")`` tuples.
            Unknown phone symbols are ignored.
        config: Keyword-only. Supplies the pseudocount ``k`` via
            :attr:`ailm.config.LevelConfig.smoothing`. ``None`` uses
            :data:`ailm.config.DEFAULT_LEVEL_CONFIG` (k = 1, add-1 Laplace).

    Returns:
        jnp array of shape ``(1, 39, 39, 1)``, column-stochastic along axis 1.
    """
    cfg = DEFAULT_LEVEL_CONFIG if config is None else config
    # add-k: every cell starts at the pseudocount.
    counts = np.full((N_PHONES, N_PHONES), float(cfg.smoothing), dtype=np.float64)

    for key, cnt in bigrams.items():
        pair = _split_bigram_key(key)
        if pair is None:
            continue
        prev_phone, next_phone = pair
        if prev_phone in PHONE_TO_IDX and next_phone in PHONE_TO_IDX:
            j = PHONE_TO_IDX[prev_phone]  # current state
            i = PHONE_TO_IDX[next_phone]  # next state
            counts[i, j] += float(cnt)

    # Guard against corrupted vendor data: negative counts produce column-stochastic
    # matrices with negative probabilities, which pymdp silently accepts (it only
    # checks column sums). This invariant is pinned by a strict xfail in
    # tests/test_phoneme.py:2020.
    if not (counts >= 0).all():
        raise ValueError(
            f"Negative count in phone bigram data. counts.min() = {counts.min()}. "
            f"Corrupted vendor artifact or invalid custom lexicon."
        )

    normalized = _column_stochastic(counts)  # (39, 39)
    # (batch=1, next=39, current=39, actions=1)
    return jnp.array(normalized[np.newaxis, :, :, np.newaxis])


def build_phone_unigram(
    unigrams: dict,
    *,
    config: LevelConfig | None = None,
) -> jnp.ndarray:
    """Build the start/prior distribution D from phone unigram counts.

    Args:
        unigrams: Mapping of phone symbol → count. Add-k smoothed; unknown
            symbols are ignored.
        config: Keyword-only. Supplies the pseudocount ``k`` via
            :attr:`ailm.config.LevelConfig.smoothing`. ``None`` uses
            :data:`ailm.config.DEFAULT_LEVEL_CONFIG` (k = 1, add-1 Laplace).

    Returns:
        jnp array of shape ``(1, 39)`` summing to 1.
    """
    cfg = DEFAULT_LEVEL_CONFIG if config is None else config
    # add-k: every cell starts at the pseudocount.
    counts = np.full(N_PHONES, float(cfg.smoothing), dtype=np.float64)
    for phone, cnt in unigrams.items():
        idx = PHONE_TO_IDX.get(str(phone))
        if idx is not None:
            counts[idx] += float(cnt)
    return jnp.array((counts / counts.sum())[np.newaxis, :])


def build_phoneme_agent(
    transition: jnp.ndarray,
    start: jnp.ndarray | None = None,
) -> PymdpAgent:
    """Build the Part A phoneme sequence agent.

    A is the identity emission (observing a phone reveals the phone state), B is
    the supplied bigram transition, C is flat, and D is the start prior. See the
    module docstring for why the learned statistics belong in B and not in A.

    Args:
        transition: ``(1, 39, 39, 1)`` transition from
            :func:`build_phone_transition`.
        start: Optional ``(1, 39)`` start prior from
            :func:`build_phone_unigram`. Defaults to uniform.

    Returns:
        A configured ``pymdp.agent.Agent``.

    Raises:
        ValueError: If ``transition`` does not have shape ``(1, 39, 39, 1)``,
            or ``start`` does not have shape ``(1, 39)``.
    """
    from pymdp.agent import Agent

    transition = jnp.asarray(transition)
    if transition.shape != (1, N_PHONES, N_PHONES, 1):
        raise ValueError(
            f"Expected phone transition shape (1, {N_PHONES}, {N_PHONES}, 1), "
            f"got {transition.shape}. B[0, i, j, 0] must be "
            f"P(next phone i | current phone j)."
        )

    if start is None:
        d_matrix = jnp.ones((1, N_PHONES)) / N_PHONES
    else:
        d_matrix = jnp.asarray(start)
        if d_matrix.shape != (1, N_PHONES):
            raise ValueError(f"Expected start prior shape (1, {N_PHONES}), got {d_matrix.shape}.")

    # A: identity emission — the observed phone *is* the hidden phone.
    A = [jnp.eye(N_PHONES)[jnp.newaxis, :, :]]
    B = [transition]
    C = [jnp.zeros((1, N_PHONES))]
    D = [d_matrix]

    return Agent(A=A, B=B, C=C, D=D, num_controls=[1], batch_size=1)


def compute_phoneme_perplexity(
    agent: PymdpAgent,
    phone_sequences: list[list[str]],
) -> float:
    """Compute per-phone perplexity across a list of phone sequences.

    Each sequence is independent: the empirical prior is reset to D at the start
    of every sequence, because a word boundary is not a phoneme transition.
    Log probabilities are accumulated over *all* positions in *all* sequences and
    divided by the total position count, so the result is a corpus-level
    per-phone perplexity rather than a mean of per-word perplexities.

    ``P(phone_t | prior_t) = sum_j A[0, phone_t, j] * prior_t[j]``, which reduces
    to ``prior_t[phone_t]`` under the identity emission. It is computed through A
    anyway so the metric stays correct if the emission is ever generalized.

    This is an all-positions, smoothed perplexity. The transitions-only
    conditional-entropy floor is a different quantity over a different position
    set, so the two are not comparable — see ``docs/decisions.md``.

    Zero scorable positions: a documented refusal, not an error
    ----------------------------------------------------------
    An empty population returns ``float(N_PHONES)`` — the uniform value — rather
    than raising, unlike :func:`ailm.verification.oracle.forward_filter`, which
    raises on an empty sequence. The same reasoning applies here as at the letter
    level and in :func:`full_corpus_perplexity`: ``N_PHONES`` is the bar every
    Stage 2 comparison asks the model to beat, so it sits on the failing side and
    a degenerate population cannot be turned into a passing measurement through
    this path. The value is pinned as a contract by ``tests/test_phoneme.py``
    (``test_empty_sequence_list_returns_n_phones``,
    ``test_all_empty_sequences_return_n_phones``) and, for the closed-form twin,
    by ``tests/test_phoneme_full_corpus.py``
    (``test_no_scorable_position_returns_the_uniform_value``).

    The residual risk is a reporting one: a stream emptied upstream is reported as
    a measured uniform perplexity rather than as an error. Callers that assemble
    their own population should assert its size before scoring.

    Args:
        agent: A phoneme agent from :func:`build_phoneme_agent`.
        phone_sequences: Sequences of ARPAbet phone symbols. Sequences with
            fewer than 1 phone are skipped.

    Returns:
        Perplexity in phones (lower is better). Returns ``float(N_PHONES)`` —
        the uniform value — if no scorable position exists. Because the
        arithmetic runs in JAX float32, analytic anchors are reproduced to about
        2e-7 relative rather than exactly: assert with a tolerance.

    Raises:
        KeyError: If a sequence contains a symbol outside :data:`PHONES`.
    """
    a_mat = jnp.asarray(agent.A[0])[0]  # (39, 39)
    d_prior = [jnp.asarray(agent.D[0])]
    action = jnp.zeros((1, 1), dtype=jnp.int32)

    total_logp = 0.0
    total_positions = 0

    for seq in phone_sequences:
        if len(seq) < 1:
            continue

        empirical_prior = d_prior
        for phone in seq:
            if phone not in PHONE_TO_IDX:
                raise KeyError(
                    f"Unknown phone {phone!r} is not one of the {N_PHONES} "
                    f"canonical ARPAbet phones in PHONES."
                )
            obs_idx = PHONE_TO_IDX[phone]

            prior_vec = jnp.asarray(empirical_prior[0]).ravel()  # (39,)
            pred_prob = jnp.dot(a_mat[obs_idx, :], prior_vec)
            pred_prob = jnp.clip(pred_prob, 1e-12, 1.0)
            total_logp += float(jnp.log(pred_prob))
            total_positions += 1

            qs = agent.infer_states([jnp.array([obs_idx])], empirical_prior)
            empirical_prior = agent.update_empirical_prior(action, qs)

    # Fail-closed by value, not by raising: the uniform value is the bar every
    # comparison here asks the model to beat, so a degenerate population cannot
    # produce a passing measurement. See compute_phoneme_perplexity's docstring.
    if total_positions == 0:
        return float(N_PHONES)
    return float(np.exp(-total_logp / total_positions))


def full_corpus_perplexity(counts: dict) -> tuple[float, int]:
    """Exact full-dictionary Part A perplexity, in closed form from the counts.

    The headline number is measured over the whole dictionary rather than over the
    artifact's ``g2p_pairs`` sequence list: that list is the length-matched G2P
    subset, which is biased toward short, transparent-orthography words and
    flatters the model. Walking the whole dictionary through pymdp instead is
    exact but costs one ``infer_states`` call per phone position, far too slow for
    a gate. See ``docs/decisions.md``.

    The closed form removes both problems. With the prior reset at every word
    boundary, position 0 of each word is scored under the start prior ``D`` and
    every later position under a transition, so::

        log L = SUM_p  initials[p] * log D[p]
              + SUM_ab bigrams[a,b] * log B[b | a]
        n     = SUM initials + SUM bigrams

    ``phone_initials`` and ``phone_bigrams`` are therefore *exact sufficient
    statistics* — no approximation and no sampling. ``tests/test_phoneme_full_corpus.py``
    checks the identity against a sequence-by-sequence walk of the real dictionary.

    Args:
        counts: The vendored artifact (see :func:`load_phoneme_counts`). Must carry
            a ``phone_initials`` key — artifacts generated before that key was
            added cannot support a full-corpus measurement.

    Returns:
        ``(perplexity, n_positions)``.

    Raises:
        ValueError: If ``phone_initials`` is absent, rather than silently falling
            back to the biased subset.
    """
    if "phone_initials" not in counts:
        raise ValueError(
            "The vendored artifact has no 'phone_initials' key, so the full-corpus "
            "perplexity cannot be computed. Regenerate it with "
            "scripts/generate_phoneme_counts.py. Refusing to fall back to the "
            "length-matched subset, which flatters the model by ~1.46 perplexity "
            "units."
        )

    transition = np.asarray(build_phone_transition(counts["phone_bigrams"]))[0, :, :, 0]
    start = np.asarray(build_phone_unigram(counts["phone_unigrams"])).ravel()

    total_logp = 0.0
    total_positions = 0

    for phone, count in counts["phone_initials"].items():
        if phone in PHONE_TO_IDX:
            total_logp += count * float(np.log(start[PHONE_TO_IDX[phone]]))
            total_positions += count

    for key, count in counts["phone_bigrams"].items():
        pair = _split_bigram_key(key)
        if pair is None:
            continue
        prev_phone, next_phone = pair
        if prev_phone in PHONE_TO_IDX and next_phone in PHONE_TO_IDX:
            probability = transition[PHONE_TO_IDX[next_phone], PHONE_TO_IDX[prev_phone]]
            total_logp += count * float(np.log(probability))
            total_positions += count

    # Fail-closed by value, not by raising: the uniform value is the bar every
    # comparison here asks the model to beat, so a degenerate population cannot
    # produce a passing measurement. See compute_phoneme_perplexity's docstring.
    if total_positions == 0:
        return float(N_PHONES), 0
    return float(np.exp(-total_logp / total_positions)), total_positions


def full_corpus_unigram_perplexity(counts: dict) -> float:
    """The context-free Part A bar over the same full dictionary.

    Must be measured on the identical position set as
    :func:`full_corpus_perplexity`, or the comparison is between two corpora
    rather than two models.
    """
    if "phone_initials" not in counts:
        raise ValueError(
            "The vendored artifact has no 'phone_initials' key. Regenerate it with "
            "scripts/generate_phoneme_counts.py."
        )

    unigram = np.asarray(build_phone_unigram(counts["phone_unigrams"])).ravel()

    total_logp = 0.0
    total_positions = 0
    # Every position is scored under the same context-free distribution, so the
    # position multiset is all that matters: word-initial phones plus the "next"
    # phone of every transition.
    for phone, count in counts["phone_initials"].items():
        if phone in PHONE_TO_IDX:
            total_logp += count * float(np.log(unigram[PHONE_TO_IDX[phone]]))
            total_positions += count
    for key, count in counts["phone_bigrams"].items():
        pair = _split_bigram_key(key)
        if pair is None:
            continue
        _, next_phone = pair
        if next_phone in PHONE_TO_IDX:
            total_logp += count * float(np.log(unigram[PHONE_TO_IDX[next_phone]]))
            total_positions += count

    # Fail-closed by value, not by raising: the uniform value is the bar every
    # comparison here asks the model to beat, so a degenerate population cannot
    # produce a passing measurement. See compute_phoneme_perplexity's docstring.
    if total_positions == 0:
        return float(N_PHONES)
    return float(np.exp(-total_logp / total_positions))


def full_corpus_order1_floor_perplexity(counts: dict) -> tuple[float, int]:
    """Exact unsmoothed order-1 MLE floor on the full Part A population.

    Unlike the historical transition-only conditional entropy, this bound
    scores every independently reset word start under the maximum-likelihood
    start distribution and every later phone under the maximum-likelihood
    transition column.  It therefore has the same denominator and reset
    convention as :func:`full_corpus_perplexity` and is a genuine model-class
    lower bound for that reported all-position metric.

    Returns:
        ``(perplexity, n_positions)`` over the exact sufficient statistics.

    Raises:
        ValueError: If the sufficient statistics are absent, malformed,
            negative, foreign to the canonical inventory, or empty.
    """
    if "phone_initials" not in counts:
        raise ValueError(
            "The vendored artifact has no 'phone_initials' key. Regenerate it with "
            "scripts/generate_phoneme_counts.py."
        )

    starts = np.zeros(N_PHONES, dtype=np.int64)
    transitions = np.zeros((N_PHONES, N_PHONES), dtype=np.int64)

    def checked_count(raw: object, *, context: str) -> int:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"{context} must be a non-negative integer, got {raw!r}")
        return raw

    for phone, raw_count in counts["phone_initials"].items():
        if phone not in PHONE_TO_IDX:
            raise ValueError(f"phone_initials contains non-canonical phone {phone!r}")
        starts[PHONE_TO_IDX[phone]] += checked_count(
            raw_count, context=f"phone_initials[{phone!r}]"
        )

    for key, raw_count in counts["phone_bigrams"].items():
        pair = _split_bigram_key(key)
        if pair is None:
            raise ValueError(f"phone_bigrams contains malformed key {key!r}")
        previous, current = pair
        if previous not in PHONE_TO_IDX or current not in PHONE_TO_IDX:
            raise ValueError(f"phone_bigrams contains non-canonical pair {pair!r}")
        transitions[PHONE_TO_IDX[current], PHONE_TO_IDX[previous]] += checked_count(
            raw_count, context=f"phone_bigrams[{key!r}]"
        )

    start_total = int(starts.sum())
    transition_total = int(transitions.sum())
    positions = start_total + transition_total
    if start_total == 0 or positions == 0:
        raise ValueError("full-corpus order-1 floor requires at least one word start")

    negative_log_likelihood = 0.0
    for count in starts[starts > 0]:
        negative_log_likelihood -= int(count) * float(np.log(int(count) / start_total))
    for previous_idx in range(N_PHONES):
        column = transitions[:, previous_idx]
        column_total = int(column.sum())
        if column_total == 0:
            continue
        for count in column[column > 0]:
            negative_log_likelihood -= int(count) * float(np.log(int(count) / column_total))
    return float(np.exp(negative_log_likelihood / positions)), positions


# ── PART B: G2P (LETTER → PHONEME COUPLING) ──────────────────────────────────


#: Prefix-family protocol for decision 74's family-disjoint diagnostic. The
#: split implementation, both recorders that report prefix overlap, and the
#: manuscript consume these constants rather than redeclaring ``4`` and ``5``.
G2P_FAMILY_PREFIX_LENGTH = 4
G2P_FAMILY_STRIDE = 5


def g2p_split(
    counts: dict,
    *,
    config: LevelConfig | None = None,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Split the vendored G2P pairs into (train, test) on index.

    The artifact's ``g2p_pairs`` list is already sorted and the split is defined
    by position, so this function must NOT re-sort: index ``i`` is TEST iff
    ``i % test_every == 0``, everything else is TRAIN.

    ``config.test_every`` overrides the artifact's stride so a reviewer can vary
    the held-out fraction (approximately ``1 / test_every``) without regenerating
    the artifact. The artifact remains the default and the recorded provenance.

    Args:
        counts: Payload from :func:`load_phoneme_counts`.
        config: Keyword-only. When
            :attr:`ailm.config.LevelConfig.test_every` is set it replaces the
            artifact's stride. ``None``, or a config leaving ``test_every`` at
            ``None``, reads ``counts["test_every"]`` exactly as before.

    Returns:
        ``(train, test)``, each a list of ``(word, [phones])``. Words are
        lowercase; phone strings in the artifact are split on whitespace.

    Raises:
        KeyError: If ``g2p_pairs`` or ``test_every`` is absent.
        ValueError: If the effective ``test_every`` is not a positive integer.
    """
    cfg = DEFAULT_LEVEL_CONFIG if config is None else config
    pairs = counts["g2p_pairs"]
    # The artifact key is read and validated unconditionally: an override must
    # not turn a missing or malformed provenance field into a silent pass.
    test_every = counts["test_every"]
    if isinstance(test_every, bool) or not isinstance(test_every, int) or test_every < 1:
        raise ValueError(f"test_every must be a positive integer, got {test_every!r}.")
    if cfg.test_every is not None:
        test_every = cfg.test_every

    train: list[tuple[str, list[str]]] = []
    test: list[tuple[str, list[str]]] = []
    for i, entry in enumerate(pairs):
        word, phones = entry[0], entry[1]
        phone_list = phones.split() if isinstance(phones, str) else list(phones)
        item = (str(word), phone_list)
        if i % test_every == 0:
            test.append(item)
        else:
            train.append(item)
    return train, test


def g2p_family_split(
    counts: dict,
    *,
    prefix_length: int = G2P_FAMILY_PREFIX_LENGTH,
    family_stride: int = G2P_FAMILY_STRIDE,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Split the vendored G2P pairs with whole prefix families on one side.

    The family-disjoint arm of decision 74. :func:`g2p_split` interleaves at the
    word level, which leaves ``stage2_g2p_split_shared_prefix_pct`` of held-out
    words sharing a four-character prefix with some training word — morphological
    neighbors like ``catalog`` / ``catalogs`` straddling the split. This split
    removes them by construction: pairs are grouped by their first
    ``prefix_length`` characters (the whole word when shorter — the same
    grouping unit as the recorded shared-prefix diagnostic), the sorted family
    keys are strided, and every family lands wholly in TRAIN or wholly in TEST.
    No held-out word of length >= ``prefix_length`` can share its prefix with a
    training word.

    Deterministic and seedless like :func:`g2p_split`: sorted family keys, index
    ``f % family_stride == 0`` is TEST. Pairs keep their original list order
    within each side, and no equal-length filtering happens here — downstream
    consumers apply the identical filter they apply to the shipped split.

    Args:
        counts: Payload from :func:`load_phoneme_counts`.
        prefix_length: Family key length in characters. Keyword-only.
        family_stride: Every ``family_stride``-th sorted family is TEST.
            Keyword-only; the default mirrors the shipped split's word-level
            ``test_every`` of 5.

    Returns:
        ``(train, test)``, each a list of ``(word, [phones])``.

    Raises:
        KeyError: If ``g2p_pairs`` is absent.
        ValueError: If ``prefix_length`` or ``family_stride`` is not a positive
            integer, or either side of the split comes out empty.
    """
    for name, value in (("prefix_length", prefix_length), ("family_stride", family_stride)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")

    pairs = counts["g2p_pairs"]
    families = sorted({str(entry[0])[:prefix_length] for entry in pairs})
    test_families = {family for i, family in enumerate(families) if i % family_stride == 0}

    train: list[tuple[str, list[str]]] = []
    test: list[tuple[str, list[str]]] = []
    for entry in pairs:
        word, phones = str(entry[0]), entry[1]
        phone_list = phones.split() if isinstance(phones, str) else list(phones)
        item = (word, phone_list)
        if word[:prefix_length] in test_families:
            test.append(item)
        else:
            train.append(item)

    if not train or not test:
        raise ValueError(
            f"family split produced {len(train)} train and {len(test)} test pairs; "
            f"an empty side cannot support a paired comparison."
        )
    return train, test


def _word_letters(word: str) -> list[str]:
    """Normalize a word and return the retained ALPHABET symbols.

    Normalization is :func:`ailm.levels.letter.normalize_text`, which lowercases
    and maps every out-of-alphabet character to a space (itself an ALPHABET
    symbol). Filtering to ALPHABET membership afterwards is therefore
    length-preserving, which is what keeps a decode aligned position-for-position
    with its reference pronunciation.

    Args:
        word: Raw word.

    Returns:
        List of single-character ALPHABET symbols.
    """
    return [c for c in normalize_text(word) if c in ALPHA_TO_IDX]


def build_g2p_emission(
    train_pairs: list[tuple[str, list[str]]],
    *,
    config: LevelConfig | None = None,
) -> jnp.ndarray:
    """Build the G2P emission matrix A from aligned (letter, phone) pairs.

    ``A[0, l, p] = P(letter = l | phone = p)``, add-k smoothed and
    column-stochastic over letters (``A[0, :, p]`` sums to 1) — the normalization
    pymdp validates along axis 1. This is a genuine emission: 30 observations
    over 39 hidden states, so unlike Part A it cannot be confused with a
    transition by shape alone.

    Only 1:1 aligned pairs contribute; a pair whose letter count differs from its
    phone count is skipped with a warning rather than silently truncated.

    Args:
        train_pairs: ``(word, [phones])`` items from :func:`g2p_split`.
        config: Keyword-only. The emission takes its pseudocount from
            :attr:`ailm.config.LevelConfig.resolved_emission_smoothing`, i.e.
            ``emission_smoothing`` when set and ``smoothing`` otherwise — the
            emission is a different shape and density from the transitions, so it
            gets its own dial. ``None`` uses
            :data:`ailm.config.DEFAULT_LEVEL_CONFIG` (k = 1, add-1 Laplace).

    Returns:
        jnp array of shape ``(1, 30, 39)``.
    """
    cfg = DEFAULT_LEVEL_CONFIG if config is None else config
    # add-k: every cell starts at the pseudocount.
    counts = np.full((N_SYMBOLS, N_PHONES), cfg.resolved_emission_smoothing, dtype=np.float64)
    skipped = 0

    for word, phones in train_pairs:
        letters = _word_letters(word)
        if len(letters) != len(phones):
            skipped += 1
            continue
        for letter, phone in zip(letters, phones, strict=True):
            p_idx = PHONE_TO_IDX.get(phone)
            if p_idx is None:
                continue
            counts[ALPHA_TO_IDX[letter], p_idx] += 1.0

    if skipped:
        logger.warning(
            "build_g2p_emission skipped %d/%d training pairs whose letter count "
            "did not equal their phone count.",
            skipped,
            len(train_pairs),
        )

    normalized = _column_stochastic(counts)  # (30, 39), columns over letters
    return jnp.array(normalized[np.newaxis, :, :])


def build_g2p_transition(
    train_pairs: list[tuple[str, list[str]]],
    *,
    config: LevelConfig | None = None,
) -> jnp.ndarray:
    """Build the G2P phone transition B from bigrams inside the TRAIN split only.

    Deliberately not the full-corpus ``phone_bigrams`` from the artifact: those
    are computed over every primary pronunciation, including the held-out test
    words, and using them would leak test statistics into the decoder.

    Args:
        train_pairs: ``(word, [phones])`` items from :func:`g2p_split`.
        config: Keyword-only, forwarded to :func:`build_phone_transition`, so the
            transition follows ``smoothing`` and not ``emission_smoothing``.

    Returns:
        jnp array of shape ``(1, 39, 39, 1)``, column-stochastic along axis 1.
    """
    bigrams: dict[tuple[str, str], int] = defaultdict(int)
    for _word, phones in train_pairs:
        for a, b in pairwise(phones):
            bigrams[(a, b)] += 1
    return build_phone_transition(dict(bigrams), config=config)


def build_g2p_start(
    train_pairs: list[tuple[str, list[str]]],
    *,
    config: LevelConfig | None = None,
) -> jnp.ndarray:
    """Build the G2P start prior D from the first phone of each training word.

    Args:
        train_pairs: ``(word, [phones])`` items from :func:`g2p_split`.
        config: Keyword-only, forwarded to :func:`build_phone_unigram`, so the
            start prior follows ``smoothing``.

    Returns:
        jnp array of shape ``(1, 39)`` summing to 1, add-k smoothed.
    """
    firsts: dict[str, int] = defaultdict(int)
    for _word, phones in train_pairs:
        if phones:
            firsts[phones[0]] += 1
    return build_phone_unigram(dict(firsts), config=config)


def build_g2p_agent(
    emission: jnp.ndarray,
    transition: jnp.ndarray,
    start: jnp.ndarray,
) -> PymdpAgent:
    """Build the Part B G2P agent: letters observed, phones hidden.

    Args:
        emission: ``(1, 30, 39)`` from :func:`build_g2p_emission`.
        transition: ``(1, 39, 39, 1)`` from :func:`build_g2p_transition`.
        start: ``(1, 39)`` from :func:`build_g2p_start`.

    Returns:
        A configured ``pymdp.agent.Agent``.

    Raises:
        ValueError: If any argument has the wrong shape. The emission check
            is what distinguishes this model from Part A: a ``(1, 39, 39)``
            square matrix passed as the emission would mean the letter and phone
            levels had been collapsed.
    """
    from pymdp.agent import Agent

    emission = jnp.asarray(emission)
    transition = jnp.asarray(transition)
    start = jnp.asarray(start)

    if emission.shape != (1, N_SYMBOLS, N_PHONES):
        raise ValueError(
            f"Expected G2P emission shape (1, {N_SYMBOLS}, {N_PHONES}) — "
            f"A[0, letter, phone] = P(letter | phone) — got {emission.shape}."
        )
    if transition.shape != (1, N_PHONES, N_PHONES, 1):
        raise ValueError(
            f"Expected G2P transition shape (1, {N_PHONES}, {N_PHONES}, 1), got {transition.shape}."
        )
    if start.shape != (1, N_PHONES):
        raise ValueError(f"Expected G2P start shape (1, {N_PHONES}), got {start.shape}.")

    return Agent(
        A=[emission],
        B=[transition],
        C=[jnp.zeros((1, N_SYMBOLS))],
        D=[start],
        num_controls=[1],
        batch_size=1,
    )


def decode_phonemes(agent: PymdpAgent, word: str) -> list[str]:
    """Decode a word to phones by forward filtering with pymdp.

    For each letter in order: ``infer_states`` under the current empirical prior,
    argmax the posterior to pick the phone, then push the posterior through B with
    ``update_empirical_prior`` to obtain the prior for the next position. This is
    filtering, not smoothing — position ``t`` sees letters ``1..t`` only.

    Args:
        agent: A G2P agent from :func:`build_g2p_agent`.
        word: Raw word; normalized via :func:`_word_letters`.

    Returns:
        Phone symbols, one per retained letter. Empty input returns ``[]``.
    """
    letters = _word_letters(word)
    if not letters:
        return []

    empirical_prior = [jnp.asarray(agent.D[0])]
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    decoded: list[str] = []

    for letter in letters:
        obs_idx = ALPHA_TO_IDX[letter]
        qs = agent.infer_states([jnp.array([obs_idx])], empirical_prior)
        posterior = jnp.asarray(qs[0]).ravel()  # (39,)
        decoded.append(IDX_TO_PHONE[int(jnp.argmax(posterior))])
        empirical_prior = agent.update_empirical_prior(action, qs)

    return decoded


def _iter_pairs(
    pairs: list[tuple[str, list[str]]],
    limit: int | None,
) -> list[tuple[str, list[str]]]:
    """Return a deterministic, alphabet-spanning subsample of ``pairs``.

    Subsampling is an evenly spaced **stride** — every kth pair — never a prefix
    and never random: the vendored ``g2p_pairs`` list is sorted alphabetically, so
    a prefix is reproducible but not representative, and it overstates the G2P
    margin. See ``docs/decisions.md``.

    Two decoders given the same ``limit`` see exactly the same positions in the
    same order, so paired comparisons remain valid.

    A caution the stride does not remove: any subsample of this eval is
    **underpowered** for the G2P gate. Use ``limit`` for fast smoke checks;
    compute gate statistics on the full set via ``ailm.verification.oracle`` after
    proving the two decoders agree exactly. See
    ``scripts/record_stage2_claims.py``.

    Args:
        pairs: Evaluation pairs.
        limit: Approximate number of pairs to keep, or None for all of them.

    Returns:
        A stride subsample of at most ``limit`` pairs, in the original order, or
        all of ``pairs`` when ``limit`` is None or exceeds the input length.

    Raises:
        ValueError: If ``limit`` is negative.
    """
    if limit is None:
        return list(pairs)
    if limit < 0:
        raise ValueError(f"limit must be non-negative or None, got {limit}.")
    if limit == 0:
        return []
    if limit >= len(pairs):
        return list(pairs)
    # Evenly spaced indices across the WHOLE list. A plain pairs[::stride][:limit]
    # would span the range and then truncate the tail back off, dropping every word
    # from the far end of the alphabet.
    return [pairs[i * len(pairs) // limit] for i in range(limit)]


def g2p_accuracy(
    agent: PymdpAgent,
    pairs: list[tuple[str, list[str]]],
    limit: int | None = None,
) -> float:
    """Position-level G2P accuracy of the pymdp filtering decoder.

    Args:
        agent: A G2P agent from :func:`build_g2p_agent`.
        pairs: ``(word, [phones])`` evaluation items.
        limit: Optional bound on evaluation pairs. Subsampling is an evenly
            spaced stride, not a prefix (see :func:`_iter_pairs`).

    Returns:
        Fraction of positions whose decoded phone equals the reference phone.
        Returns 0.0 if there are no positions.

    Raises:
        ValueError: If a decoded length differs from its reference length. The
            corpus is length-matched by construction, so a mismatch is a bug in
            normalization, not a data condition to be tolerated.
    """
    positions = decode_positions(agent, pairs, limit=limit)
    if not positions:
        return 0.0
    return sum(positions) / len(positions)


def per_letter_argmax_baseline(
    train_pairs: list[tuple[str, list[str]]],
) -> dict[str, str]:
    """Build the context-free bar: the single most frequent phone per letter.

    This is the comparator every context-aware decoder must beat. Uniform-random
    and majority-phone comparators are crippled — this lookup table is the honest
    baseline. See ``docs/decisions.md``.

    Ties are broken by sorted phone name so the mapping is deterministic.

    Args:
        train_pairs: ``(word, [phones])`` items from :func:`g2p_split`.

    Returns:
        Mapping of letter → most frequent aligned phone. Letters never observed
        in training are absent from the mapping.
    """
    tallies: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for word, phones in train_pairs:
        letters = _word_letters(word)
        if len(letters) != len(phones):
            continue
        for letter, phone in zip(letters, phones, strict=True):
            if phone in PHONE_TO_IDX:
                tallies[letter][phone] += 1

    mapping: dict[str, str] = {}
    for letter, phone_counts in tallies.items():
        # max() on (count, negated-name) is ambiguous for strings, so sort
        # candidates by name first and let max() keep the first maximum.
        best = max(sorted(phone_counts.items()), key=lambda kv: kv[1])
        mapping[letter] = best[0]
    return mapping


def baseline_accuracy(
    mapping: dict[str, str],
    pairs: list[tuple[str, list[str]]],
    limit: int | None = None,
) -> float:
    """Position-level accuracy of the context-free letter → phone mapping.

    Args:
        mapping: Output of :func:`per_letter_argmax_baseline`.
        pairs: ``(word, [phones])`` evaluation items.
        limit: Optional bound on evaluation pairs. Subsampling is an evenly
            spaced stride, not a prefix (see :func:`_iter_pairs`).

    Returns:
        Fraction of positions predicted correctly. A letter absent from the
        mapping counts as wrong rather than raising. Returns 0.0 if there are no
        positions.

    Raises:
        ValueError: If a word's retained-letter count differs from its reference
            phone count.
    """
    positions = baseline_positions(mapping, pairs, limit=limit)
    if not positions:
        return 0.0
    return sum(positions) / len(positions)


def decode_positions(
    agent: PymdpAgent,
    pairs: list[tuple[str, list[str]]],
    limit: int | None = None,
) -> list[bool]:
    """Per-position correctness of the pymdp filtering decoder.

    Aligned position-for-position with :func:`baseline_positions` called on the
    same ``pairs`` and the same ``limit``: same order, same length. That pairing
    is what lets the gate run McNemar's test on the discordant pairs — the
    filtering margin over the context-free bar is a fraction of a point, so an
    unpaired comparison of two aggregate accuracies cannot establish it. See
    ``docs/decisions.md``.

    Cost is one ``infer_states`` plus one ``update_empirical_prior`` per position,
    so the full test split takes minutes. Use ``limit`` to bound a test run; the
    stride is deterministic, so a subset result is reproducible, but the margin
    over the baseline wobbles on small subsets.

    Args:
        agent: A G2P agent from :func:`build_g2p_agent`.
        pairs: ``(word, [phones])`` evaluation items.
        limit: Optional bound on evaluation pairs. Subsampling is an evenly
            spaced stride, not a prefix (see :func:`_iter_pairs`).

    Returns:
        One bool per position, words in order and letters in order.

    Raises:
        ValueError: If a decoded length differs from its reference length.
    """
    correct: list[bool] = []
    for word, phones in _iter_pairs(pairs, limit):
        decoded = decode_phonemes(agent, word)
        if len(decoded) != len(phones):
            raise ValueError(
                f"Decoded length {len(decoded)} != reference length "
                f"{len(phones)} for word {word!r}. The G2P corpus is "
                f"length-matched by construction, so this is a normalization "
                f"bug: decoded={decoded}, reference={phones}."
            )
        correct.extend(d == r for d, r in zip(decoded, phones, strict=True))
    return correct


def baseline_positions(
    mapping: dict[str, str],
    pairs: list[tuple[str, list[str]]],
    limit: int | None = None,
) -> list[bool]:
    """Per-position correctness of the context-free mapping.

    Aligned position-for-position with :func:`decode_positions` on the same
    ``pairs`` and ``limit``. See that function for why the pairing matters.

    Args:
        mapping: Output of :func:`per_letter_argmax_baseline`.
        pairs: ``(word, [phones])`` evaluation items.
        limit: Optional bound on evaluation pairs. Subsampling is an evenly
            spaced stride, not a prefix (see :func:`_iter_pairs`).

    Returns:
        One bool per position, words in order and letters in order. A letter
        absent from ``mapping`` yields False.

    Raises:
        ValueError: If a word's retained-letter count differs from its reference
            phone count.
    """
    correct: list[bool] = []
    for word, phones in _iter_pairs(pairs, limit):
        letters = _word_letters(word)
        if len(letters) != len(phones):
            raise ValueError(
                f"Retained-letter count {len(letters)} != reference length "
                f"{len(phones)} for word {word!r}. The G2P corpus is "
                f"length-matched by construction, so this is a normalization "
                f"bug: letters={letters}, reference={phones}."
            )
        correct.extend(
            mapping.get(letter) == phone for letter, phone in zip(letters, phones, strict=True)
        )
    return correct


# ── PART A HELD-OUT WORD SPLIT ───────────────────────────────────────────────
#
# The full-corpus Part A numbers above are an in-sample dictionary protocol
# (decision 6): the same primary pronunciations supply the counts that fit the
# model and the sufficient statistics that are scored. This section adds the
# additive machinery for the held-out arm: a seedless word split, the exact
# sufficient statistics of a sample of words, and the closed-form scorer that
# fits on one count set and scores another. Nothing above this section is
# modified; the recorded held-out numbers live in the Stage 2 recorder.


from collections.abc import Iterable, Sequence  # noqa: E402

#: Every Nth sorted dictionary word is HELD OUT of the Part A fit. Mirrors the
#: artifact's ``test_every`` G2P stride so both Stage 2 splits hold out the
#: same fraction of their populations.
PARTA_HELDOUT_STRIDE = 5


def part_a_counts(sequences: Iterable[Sequence[str]]) -> dict:
    """Exact Part A sufficient statistics from an iterable of phone sequences.

    Mirrors ``scripts/generate_phoneme_counts.py``: word-initial phone counts,
    phone unigram counts, and phone bigram counts keyed ``"prev,next"``. With
    the per-word prior reset (decision 7) these three tables are exact
    sufficient statistics for the all-positions perplexity of ``sequences`` —
    which is what lets a held-out word split be scored in closed form without
    vendoring per-word sequences.

    Args:
        sequences: Phone sequences, one per word, in a fixed order. Unknown
            phone symbols are ignored, matching the builders.

    Returns:
        Dict with ``phone_unigrams``, ``phone_bigrams`` and ``phone_initials``
        keys, in the artifact's schema, ready for :func:`part_a_split_perplexity`.
    """
    unigrams: dict[str, int] = defaultdict(int)
    bigrams: dict[str, int] = defaultdict(int)
    initials: dict[str, int] = defaultdict(int)
    for seq in sequences:
        known = [phone for phone in seq if phone in PHONE_TO_IDX]
        if not known:
            continue
        initials[known[0]] += 1
        for phone in known:
            unigrams[phone] += 1
        for prev, nxt in pairwise(known):
            bigrams[f"{prev},{nxt}"] += 1
    return {
        "phone_unigrams": dict(unigrams),
        "phone_bigrams": dict(bigrams),
        "phone_initials": dict(initials),
    }


def phoneme_heldout_split(
    words: Sequence[str],
    *,
    stride: int = PARTA_HELDOUT_STRIDE,
) -> tuple[list[str], list[str]]:
    """Split an ordered word list into ``(train, held_out)`` by even stride.

    The held-out arm of the Part A measurement. Index ``i`` is HELD OUT iff
    ``i % stride == 0`` — the artifact's own ``test_every`` convention, one
    level up. The input must already be in the caller's canonical order
    (``sorted`` dictionary words); this function must NOT re-sort, because the
    split is defined by position. A stride spans the whole alphabet, where a
    prefix would concentrate the held-out words in one initial-letter
    neighborhood (decision 4).

    Args:
        words: Ordered dictionary words, no duplicates.
        stride: Keyword-only. Held-out stride; approximately one word in
            ``stride`` is held out.

    Returns:
        ``(train, held_out)``, each preserving the input order.

    Raises:
        ValueError: If ``stride`` is not a positive integer, or either side of
            the split comes out empty.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be a positive integer, got {stride!r}.")
    held_out = [word for i, word in enumerate(words) if i % stride == 0]
    train = [word for i, word in enumerate(words) if i % stride != 0]
    if not held_out or not train:
        raise ValueError(
            f"held-out split produced {len(train)} train and {len(held_out)} "
            f"words at stride {stride}; an empty side cannot support a "
            f"held-out measurement."
        )
    return train, held_out


def part_a_split_perplexity(fit_counts: dict, score_counts: dict) -> tuple[float, float, int]:
    """Closed-form Part A perplexity and unigram bar across a count split.

    Fits the Part A model — the add-k-smoothed phone bigram :math:`B` and the
    start prior :math:`D` from the production builders — on ``fit_counts`` and
    scores the closed form on ``score_counts``. With the prior reset at every
    word boundary, position 0 of each scored word is charged under the fitted
    :math:`D` and every later position under a fitted transition, so the
    scored tables are exact sufficient statistics and nothing is approximated.
    The context-free bar re-fits nothing but the unigram distribution on the
    same ``fit_counts`` and scores the identical position set, so model and bar
    differ by the transition information alone, on one denominator.

    Passing the same dict as both arguments degenerates to the in-sample
    closed form: the result must then agree with :func:`full_corpus_perplexity`
    and :func:`full_corpus_unigram_perplexity` to within `1e-9` relative. The
    Stage 2 recorder asserts exactly that as its positive control.

    Args:
        fit_counts: Counts the model is fit on. Needs ``phone_unigrams`` and
            ``phone_bigrams``.
        score_counts: Counts the closed form is scored on. Needs
            ``phone_initials`` and ``phone_bigrams``.

    Returns:
        ``(model_perplexity, unigram_bar_perplexity, n_positions)``. With no
        scorable position, returns the uniform value for both perplexities and
        ``0`` positions.

    Raises:
        ValueError: If a required table is absent, mirroring
            :func:`full_corpus_perplexity`'s refusal to fall back silently.
    """
    for name in ("phone_unigrams", "phone_bigrams"):
        if name not in fit_counts:
            raise ValueError(f"fit_counts has no {name!r} table.")
    for name in ("phone_initials", "phone_bigrams"):
        if name not in score_counts:
            raise ValueError(f"score_counts has no {name!r} table.")

    transition = np.asarray(build_phone_transition(fit_counts["phone_bigrams"]))[0, :, :, 0]
    start = np.asarray(build_phone_unigram(fit_counts["phone_unigrams"])).ravel()
    unigram = start

    model_logp = 0.0
    bar_logp = 0.0
    total_positions = 0
    for phone, count in score_counts["phone_initials"].items():
        if phone in PHONE_TO_IDX:
            model_logp += count * float(np.log(start[PHONE_TO_IDX[phone]]))
            bar_logp += count * float(np.log(unigram[PHONE_TO_IDX[phone]]))
            total_positions += count
    for key, count in score_counts["phone_bigrams"].items():
        pair = _split_bigram_key(key)
        if pair is None:
            continue
        prev_phone, next_phone = pair
        if prev_phone in PHONE_TO_IDX and next_phone in PHONE_TO_IDX:
            next_idx = PHONE_TO_IDX[next_phone]
            model_logp += count * float(np.log(transition[next_idx, PHONE_TO_IDX[prev_phone]]))
            bar_logp += count * float(np.log(unigram[next_idx]))
            total_positions += count

    # Fail-closed by value, not by raising: the uniform value is the bar every
    # comparison here asks the model to beat, so a degenerate population cannot
    # produce a passing measurement. See compute_phoneme_perplexity's docstring.
    if total_positions == 0:
        return float(N_PHONES), float(N_PHONES), 0
    return (
        float(np.exp(-model_logp / total_positions)),
        float(np.exp(-bar_logp / total_positions)),
        total_positions,
    )


# ── FAMILY-GROUPED G2P SENSITIVITY ARM (decision 102) ────────────────────────
#
# Decision 74's famsplit arm pools the held-out set into one number. This
# section adds the additive machinery for reporting the same split, decoder,
# and comparator per morphological family: a per-family record type, a
# deterministic family-grouping of held-out pairs, and the summary the
# recorder's guards read. Nothing above is modified; the recorded numbers
# live in the Stage 2 recorder.


def g2p_family_group(
    pairs: list[tuple[str, list[str]]],
    *,
    prefix_length: int = G2P_FAMILY_PREFIX_LENGTH,
) -> dict[str, list[tuple[str, list[str]]]]:
    """Group held-out G2P pairs by their prefix-family key.

    The family-grouped sensitivity arm of decision 102: the reporting unit of
    the family-disjoint split, lifted from one pooled population to one group
    per morphological family. The key is the first ``prefix_length``
    characters (the whole word when shorter) — the same deterministic
    grouping unit :func:`g2p_family_split` assigns sides with, derived from
    the vendored pair list alone, so no new grouping rule is introduced.

    Args:
        pairs: ``(word, [phones])`` items, e.g. the test side of
            :func:`g2p_family_split`.
        prefix_length: Family key length in characters. Keyword-only.

    Returns:
        Mapping of family key to that family's pairs, in each family's
        original list order. Keys are sorted (dict insertion order follows
        first appearance, which for a sorted pair list is alphabetical).

    Raises:
        ValueError: If ``prefix_length`` is not a positive integer.
    """
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, int) or prefix_length < 1:
        raise ValueError(f"prefix_length must be a positive integer, got {prefix_length!r}.")
    groups: dict[str, list[tuple[str, list[str]]]] = {}
    for word, phones in pairs:
        phone_list = phones.split() if isinstance(phones, str) else list(phones)
        groups.setdefault(str(word)[:prefix_length], []).append((str(word), phone_list))
    return groups


def g2p_family_margin_record(
    family: str,
    pairs: list[tuple[str, list[str]]],
    model_hits: list[bool],
    baseline_hits: list[bool],
) -> dict[str, float | int | str]:
    """Build one family's margin record from its paired correctness vectors.

    Args:
        family: The family key, carried into the record for reporting.
        pairs: The family's ``(word, [phones])`` items.
        model_hits: Per-position decoder correctness, aligned position for
            position with ``baseline_hits``.
        baseline_hits: Per-position comparator correctness, same alignment.

    Returns:
        Dict with ``family``, ``words``, ``eval_positions``, ``model_correct``,
        ``baseline_correct``, ``accuracy``, ``baseline_accuracy``,
        ``margin_pts``, ``mcnemar_b``, ``mcnemar_c``, and ``mcnemar_p``.

    Raises:
        ValueError: If the two hit vectors are empty or differ in length —
            an unpaired comparison cannot feed McNemar.
    """
    if len(model_hits) != len(baseline_hits) or not model_hits:
        raise ValueError(
            f"family {family!r}: model scored {len(model_hits)} positions against "
            f"the comparator's {len(baseline_hits)}. An unpaired or empty "
            f"comparison cannot feed McNemar."
        )
    b = sum(1 for m, s in zip(model_hits, baseline_hits, strict=True) if m and not s)
    c = sum(1 for m, s in zip(model_hits, baseline_hits, strict=True) if s and not m)
    accuracy = 100.0 * sum(model_hits) / len(model_hits)
    baseline = 100.0 * sum(baseline_hits) / len(baseline_hits)
    return {
        "family": family,
        "words": len(pairs),
        "eval_positions": len(model_hits),
        "model_correct": sum(model_hits),
        "baseline_correct": sum(baseline_hits),
        "accuracy": round(accuracy, 3),
        "baseline_accuracy": round(baseline, 3),
        "margin_pts": round(accuracy - baseline, 3),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": float(f"{exact_two_sided_binomial_p(b, c):.4g}"),
    }


def g2p_family_margin_summary(
    family_results: list[dict[str, float | int | str]],
    *,
    min_positions: int,
) -> dict[str, float | int]:
    """Summarize per-family margin records for the family-grouped arm.

    Args:
        family_results: One record per family from
            :func:`g2p_family_margin_record`.
        min_positions: Minimum evaluation positions a family needs to enter
            the sign-share statistic.

    Returns:
        Dict with ``families``, ``evaluated_families`` (those meeting
        ``min_positions``), ``positive_margin_families``,
        ``positive_margin_share``, and the position-weighted pool the
        positive control checks: ``pool_positions``, ``pool_model_correct``,
        ``pool_baseline_correct``, ``pool_accuracy``,
        ``pool_baseline_accuracy``, ``pool_margin_pts``.

    Raises:
        ValueError: If ``min_positions`` is not a positive integer.
    """
    if isinstance(min_positions, bool) or not isinstance(min_positions, int) or min_positions < 1:
        raise ValueError(f"min_positions must be a positive integer, got {min_positions!r}.")
    total_model = sum(int(r["model_correct"]) for r in family_results)
    total_base = sum(int(r["baseline_correct"]) for r in family_results)
    total_positions = sum(int(r["eval_positions"]) for r in family_results)
    evaluated = [r for r in family_results if int(r["eval_positions"]) >= min_positions]
    positive = sum(1 for r in evaluated if float(r["margin_pts"]) > 0)
    pool_accuracy = 100.0 * total_model / total_positions if total_positions else 0.0
    pool_baseline = 100.0 * total_base / total_positions if total_positions else 0.0
    return {
        "families": len(family_results),
        "evaluated_families": len(evaluated),
        "positive_margin_families": positive,
        "positive_margin_share": round(positive / len(evaluated), 4) if evaluated else 0.0,
        "pool_positions": total_positions,
        "pool_model_correct": total_model,
        "pool_baseline_correct": total_base,
        "pool_accuracy": round(pool_accuracy, 3),
        "pool_baseline_accuracy": round(pool_baseline, 3),
        "pool_margin_pts": round(pool_accuracy - pool_baseline, 3),
    }
