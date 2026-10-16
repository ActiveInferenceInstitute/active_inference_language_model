"""Computational complexity of every AILM level — exact counts, not adjectives.

What this module measures, and how it can be wrong
--------------------------------------------------
Big-O statements are easy to assert and hard to falsify. This module makes the
project's complexity claims falsifiable the same way its accuracy claims are:
every asymptotic statement is backed by an **exact closed-form operation-count
formula**, and every formula is checked against an **instrumented execution**
of the same recursion over the real shipped workload. The instrumented counter
increments its tallies only at statements that actually execute — the final
step of a sequence performs no transition push, a per-word reset costs no
push, the zero-likelihood fallback skips the normalizing division — so the
formula must reproduce the executed branch structure, not just multiply two
numbers. A formula with the classic off-by-one (``T`` transition pushes rather
than ``T - 1``) fails the reconciliation, and the test suite carries exactly
that wrong formula as a negative control.

Why operations and bytes, not seconds
-------------------------------------
Wall-clock time is not deterministic, and AILM's ledger discipline (ISC-4)
requires that re-running a recorder be a no-op. Scalar operation counts and
array bytes are exact, reproducible, machine-independent, and — unlike a
timing — can be *derived* independently and required to agree to the last
integer. Timings may appear in prose as context; they may never be claims.

The cost model
--------------
Counts are kept per category — ``mul``, ``add``, ``div``, ``log``, ``exp``,
``cmp`` — over the scaled alpha recursion that every recognition level runs
(``ailm.verification.oracle.forward_filter``):

    likelihood row:   S multiplies
    normalizer:       S - 1 adds, 1 log (evidence), S divides
    transition push:  S^2 multiplies, S(S - 1) adds   (T - 1 times per sequence)
    decode argmax:    S - 1 comparisons               (when decoding)

so a sequence of length ``T`` over ``S`` hidden states costs
``T*S + (T-1)*S^2`` multiplies — Θ(T·S²) — and a corpus of ``n`` sequences
totalling ``T`` positions costs ``T*S + (T-n)*S^2`` because every sequence
withholds one push. The word level is the deliberate exception: its factors
are deterministic, so scoring a token collapses to one prior lookup — one log
and one accumulate, Θ(1) per token regardless of vocabulary size — which is
the complexity payoff of the Stage 3 factorization.

Memory is derived from the **GNN spec**: every role-carrying variable's
declared dimensions, times four bytes (float32, the live agents' dtype). The
live agent, built by ``ailm.spec_agent.build_agent_from_spec``, must occupy
exactly that many bytes — a second place the spec is required to be
constructively true, not descriptively plausible.

There is a second, independent route to the same bytes.
:func:`spec_memory_bytes_theory` sums declared dimensions blindly, whichever
variables a spec happens to carry; :func:`agent_memory_bytes_from_shape`
composes the total structurally from a level's ``(O, [S_p], A)`` shape alone,
so it also asserts *which* arrays exist and how big each must be. The two
agree only if both the blind sum and the structural model are right, which is
why they are kept as separate derivations rather than one calling the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FLOAT_BYTES",
    "OpCount",
    "agent_memory_bytes",
    "agent_memory_bytes_from_shape",
    "count_filtering_ops",
    "count_word_path_ops",
    "dense_joint_transition_bytes",
    "filtering_ops_theory",
    "spec_memory_bytes_theory",
    "step_ops_terms",
    "word_factored_transition_bytes",
    "word_path_ops_theory",
]

#: Every live AILM agent stores float32 (JAX's default), so four bytes per entry.
FLOAT_BYTES: int = 4


@dataclass
class OpCount:
    """Scalar operations by category. Integer-exact; equality is exact.

    Attributes:
        mul: Multiplications.
        add: Additions (including subtractions).
        div: Divisions.
        log: Logarithms.
        exp: Exponentials.
        cmp: Comparisons (argmax decoding).
    """

    mul: int = 0
    add: int = 0
    div: int = 0
    log: int = 0
    exp: int = 0
    cmp: int = 0

    def __add__(self, other: OpCount) -> OpCount:
        return OpCount(
            mul=self.mul + other.mul,
            add=self.add + other.add,
            div=self.div + other.div,
            log=self.log + other.log,
            exp=self.exp + other.exp,
            cmp=self.cmp + other.cmp,
        )

    @property
    def total(self) -> int:
        """Total scalar operations across every category."""
        return self.mul + self.add + self.div + self.log + self.exp + self.cmp

    def as_dict(self) -> dict[str, int]:
        """Category counts plus the total, JSON-ready."""
        return {
            "mul": self.mul,
            "add": self.add,
            "div": self.div,
            "log": self.log,
            "exp": self.exp,
            "cmp": self.cmp,
            "total": self.total,
        }


# ── FILTERING: THEORY ────────────────────────────────────────────────────────


def filtering_ops_theory(n_states: int, lengths: Sequence[int], *, decode: bool = False) -> OpCount:
    """Closed-form operation count for filtering a corpus of sequences.

    Every sequence resets the prior to ``D`` (free) and performs one transition
    push per step **except the last** — pushing after the final observation
    would predict a symbol nothing scores. That ``T - 1`` is the part of the
    formula that is easiest to state wrongly and is exactly what the
    instrumented counter checks.

    Args:
        n_states: Hidden-state count ``S``.
        lengths: Length of every sequence in the workload.
        decode: Count the per-position argmax of a decoder.

    Returns:
        Exact counts for the whole workload.

    Raises:
        ValueError: On a non-positive state count or an empty/invalid workload.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be positive, got {n_states}")
    if not lengths:
        raise ValueError("empty workload; the count would be vacuously zero")
    if any(t < 1 for t in lengths):
        raise ValueError("every sequence must have at least one position")

    s = int(n_states)
    total = int(sum(lengths))
    n_sequences = len(lengths)
    pushes = total - n_sequences  # one withheld per sequence

    return OpCount(
        mul=total * s + pushes * s * s,
        add=total * (s - 1) + pushes * s * (s - 1),
        div=total * s,
        log=total,
        cmp=total * (s - 1) if decode else 0,
    )


def step_ops_terms(n_states: int, *, decode: bool) -> dict[str, int]:
    """Per-position scalar operations, split by the matrix that causes them.

    Collecting :math:`\\mathbf{A}`'s likelihood row, the normalization and the
    evidence logarithm gives :math:`3S`, paid at every scored position. The
    :math:`\\mathbf{B}` push costs :math:`S^2` multiplies and :math:`S(S-1)`
    adds — :math:`2S^2 - S` — and is paid at every position **except the last
    of each sequence**. A decoder adds :math:`S - 1` comparisons. A position
    that carries a push therefore costs exactly :math:`2S^2 + 2S`, and the last
    position of a sequence costs :math:`3S`.

    This is the same cost model as :func:`filtering_ops_theory` sliced the
    other way: that function totals a workload by operation *category*, this
    one prices a single position by operation *cause*. The two slices are not
    interchangeable, which is why this returns a mapping rather than an
    :class:`OpCount` — an ``OpCount`` would have to put ``3S`` in three
    different fields and lose the attribution that is the whole point.

    Args:
        n_states: Hidden-state count ``S``.
        decode: Whether the level runs a per-position argmax.

    Returns:
        Mapping with keys ``emission`` (the ``A`` row, its normalizer and the
        evidence log), ``push`` (the ``B`` product) and ``decode``.

    Raises:
        ValueError: On a non-positive state count, which would make every term
            below vacuously non-negative rather than wrong.
    """
    if n_states < 1:
        raise ValueError(f"n_states must be positive, got {n_states}")
    s = int(n_states)
    return {
        "emission": 3 * s,
        "push": 2 * s * s - s,
        "decode": s - 1 if decode else 0,
    }


# ── FILTERING: INSTRUMENTED EXECUTION ────────────────────────────────────────


def count_filtering_ops(
    emission: np.ndarray,
    transition: np.ndarray,
    start: np.ndarray,
    sequences: Iterable[Sequence[int]],
    *,
    decode: bool = False,
) -> tuple[OpCount, float]:
    """Run the scaled alpha recursion and count what actually executes.

    This is the empirical side of the reconciliation: the recursion really
    runs — real likelihood rows, real normalizers, real pushes — and the
    counter increments beside each statement with the size of the array that
    statement touched. Branches count only when taken: the final step of every
    sequence performs no push, and a zero-likelihood step performs no
    normalizing division (the belief falls back to the predictive, matching
    ``ailm.verification.oracle.forward_filter``).

    Args:
        emission: ``(n_obs, S)`` column-stochastic over observations.
        transition: ``(S, S)``, ``transition[i, j] = P(next i | current j)``.
        start: ``(S,)`` prior, reset at the head of every sequence.
        sequences: Observation-index sequences (the per-word reset is real:
            each sequence starts from ``start`` again).
        decode: Also perform and count the per-position argmax.

    Returns:
        ``(counts, log_evidence)`` — the executed tallies and the total log
        evidence in nats, so callers can confirm real inference happened.

    Raises:
        ValueError: On shape disagreement or an empty workload.
    """
    emis = np.asarray(emission, dtype=np.float64)
    trans = np.asarray(transition, dtype=np.float64)
    prior = np.asarray(start, dtype=np.float64).ravel()
    s = prior.shape[0]
    if emis.ndim != 2 or emis.shape[1] != s:
        raise ValueError(f"emission {emis.shape} does not span {s} states")
    if trans.shape != (s, s):
        raise ValueError(f"transition {trans.shape} is not ({s}, {s})")

    ops = OpCount()
    log_evidence = 0.0
    n_sequences = 0

    for sequence in sequences:
        n_sequences += 1
        predictive = prior
        length = len(sequence)
        for position, obs in enumerate(sequence):
            row = emis[int(obs), :]
            unnormalized = row * predictive
            ops.mul += s
            norm = float(unnormalized.sum())
            ops.add += s - 1
            log_evidence += float(np.log(max(norm, 1e-300)))
            ops.log += 1
            if norm > 0.0:
                posterior = unnormalized / norm
                ops.div += s
            else:  # pragma: no cover - degenerate inputs only
                posterior = predictive
            if decode:
                int(np.argmax(posterior))
                ops.cmp += s - 1
            if position + 1 < length:
                predictive = trans @ posterior
                ops.mul += s * s
                ops.add += s * (s - 1)

    if n_sequences == 0:
        raise ValueError("empty workload; nothing executed, nothing to count")
    return ops, log_evidence


# ── THE WORD LEVEL: DETERMINISM COLLAPSES THE RECURSION ──────────────────────


def word_path_ops_theory(n_tokens: int) -> OpCount:
    """Scoring cost of the Stage 3 closed form: Θ(1) per token.

    The factorization makes both transitions and the emission deterministic, so
    a token's path probability *is* its prior mass (``docs/decisions.md`` entry
    13): one logarithm and one accumulate per token, independent of the
    vocabulary size. This is the payoff the dense-joint counterfactual in
    :func:`dense_joint_transition_bytes` prices.

    Args:
        n_tokens: Number of scored tokens.

    Raises:
        ValueError: On an empty workload.
    """
    if n_tokens < 1:
        raise ValueError("empty workload; the count would be vacuously zero")
    return OpCount(log=n_tokens, add=n_tokens)


def count_word_path_ops(
    tokens: Sequence[str],
    word_to_index: dict[str, int],
    start_word: np.ndarray,
) -> tuple[OpCount, float]:
    """Score the real token stream through the shipped closed form, counting.

    Executes the same per-token collapse as
    ``ailm.verification.oracle.word_path_perplexity`` — a prior lookup, a log,
    an accumulate — and tallies beside the execution.

    Args:
        tokens: In-vocabulary held-out tokens.
        word_to_index: Vocabulary map.
        start_word: ``(n_vocab,)`` prior over word identity.

    Returns:
        ``(counts, total_log_probability)``.

    Raises:
        ValueError: On an empty stream or an out-of-vocabulary token, matching
            the closed-vocabulary contract of the scorer it mirrors.
    """
    if not tokens:
        raise ValueError("tokens is empty; there is nothing to score")
    prior = np.asarray(start_word, dtype=np.float64).ravel()
    ops = OpCount()
    total_logp = 0.0
    for token in tokens:
        index = word_to_index.get(token)
        if index is None:
            raise ValueError(f"{token!r} is out of vocabulary")
        total_logp += float(np.log(max(float(prior[index]), 1e-300)))
        ops.log += 1
        ops.add += 1
    return ops, total_logp


# ── MEMORY: THE SPEC PRICES THE AGENT ────────────────────────────────────────

#: Ontology roles whose variables are materialised as live agent arrays.
_MATERIALISED_ROLES: frozenset[str] = frozenset(
    {
        "RecognitionMatrix",
        "TransitionMatrix",
        "LogPreferenceVector",
        "Prior",
        "PolicyPrior",
    }
)


def spec_memory_bytes_theory(spec: Any) -> int:
    """Bytes the live agent's parameter arrays must occupy, from the spec alone.

    Sums the product of every role-carrying variable's declared dimensions,
    times :data:`FLOAT_BYTES`. Trailing singleton axes contribute a factor of
    one, and the batch axis pymdp adds is also size one, so the declared
    product *is* the live element count — there is nothing to normalize.

    One term comes from the runtime rather than the spec: pymdp materialises a
    policy prior ``E`` for **every** agent, so a level that declares no
    ``PolicyPrior`` still carries one float32 per policy — a single element for
    AILM's single-action recognition levels. The formula adds that term
    explicitly (``num_policies = num_controls`` at policy length one) rather
    than excluding ``E`` from the measured side, because filtering the
    measurement to fit the prediction is how a 4-byte gap becomes an unnoticed
    4-gigabyte one at a level with a real action set.

    Args:
        spec: A parsed ``ailm.gnn.GnnSpec``.

    Returns:
        Exact byte count.

    Raises:
        ValueError: If the spec has no role-carrying variables at all — a
            "memory footprint" of an unpinned spec would be vacuously zero.
    """
    total_elements = 0
    counted = 0
    declares_policy_prior = False
    for name in spec.variables:
        role = spec.ontology.get(name)
        if role in _MATERIALISED_ROLES:
            counted += 1
            total_elements += int(np.prod(spec.variable_dims(name)))
            if role == "PolicyPrior":
                declares_policy_prior = True
    if counted == 0:
        raise ValueError(
            f"{spec.model_name!r} annotates no materialised roles; its memory "
            f"footprint is not defined."
        )
    if not declares_policy_prior:
        num_controls = spec.parameters.get("num_controls", 1)
        total_elements += int(num_controls) if isinstance(num_controls, int) else 1
    return total_elements * FLOAT_BYTES


def agent_memory_bytes(agent: Any) -> int:
    """Bytes actually occupied by a live agent's parameter arrays.

    Sums ``nbytes`` over every array in ``A``, ``B``, ``C``, ``D`` and — when
    present — ``E``. This is the measured side of the memory reconciliation;
    :func:`spec_memory_bytes_theory` is the predicted side.
    """
    total = 0
    for attribute in ("A", "B", "C", "D"):
        for item in getattr(agent, attribute):
            total += int(np.asarray(item).nbytes)
    e_value = getattr(agent, "E", None)
    if e_value is not None:
        total += int(np.asarray(e_value).nbytes)
    return total


def agent_memory_bytes_from_shape(
    n_obs: int,
    factor_sizes: Sequence[int],
    *,
    n_actions: int = 1,
) -> int:
    """Live-agent parameter bytes from the level's declared shape alone.

    One closed form prices every level of the hierarchy::

        4 · ( O·∏ₚ Sₚ  +  A·∑ₚ Sₚ²  +  O  +  ∑ₚ Sₚ  +  A )

    term by term: ``A`` the emission, dense over the **joint** of every hidden
    factor and therefore carrying the product; ``B`` one transition per factor
    per action, carrying the sum of squares; ``C`` over observations, ``D`` over
    each factor's states, and the policy prior ``E`` pymdp materialises for
    every agent — one scalar at the single-action recognition levels, which is
    the term whose omission once left the reconciliation four bytes short at
    every level (``docs/decisions.md`` entry 21).

    The product and the sum are the whole factorization argument in one line:
    factoring moves the transition from ``(∏ₚ Sₚ)²`` to ``∑ₚ Sₚ²`` and leaves
    the emission's product exactly where it was.

    This is the shape-side route to the bytes :func:`spec_memory_bytes_theory`
    derives from a parsed spec, and the two must agree at every level where
    both apply; ``ailm.figures.complexity.complexity_memory_data`` draws that
    agreement and raises when it stops holding.

    Args:
        n_obs: Observation count ``O`` of the level's single modality.
        factor_sizes: Size of every hidden factor.
        n_actions: Control states, hence policies at policy length one.

    Returns:
        Exact byte count for a float32 agent.

    Raises:
        ValueError: On an empty factor list or a non-positive size — a level
            with no factors would price at the ``O + A`` terms alone and look
            like a very cheap agent rather than an unpinned one.
    """
    sizes = [int(s) for s in factor_sizes]
    if not sizes:
        raise ValueError("a level with no hidden factors has no memory footprint")
    if n_obs < 1 or n_actions < 1 or any(s < 1 for s in sizes):
        raise ValueError("observation count, action count and factor sizes must be positive")
    joint = 1
    for size in sizes:
        joint *= size
    emission = n_obs * joint
    transition = n_actions * sum(size * size for size in sizes)
    preference = n_obs
    prior = sum(sizes)
    policy_prior = n_actions
    return (emission + transition + preference + prior + policy_prior) * FLOAT_BYTES


def dense_joint_transition_bytes(n_vocab: int, n_positions: int) -> int:
    """Bytes a dense transition over the UNfactored Stage 3 joint would need.

    The counterfactual that prices the factorization: a single hidden factor
    over ``(word, position)`` pairs has ``V·P`` states, and pymdp's dense
    transition over it is ``(V·P)²`` float32 entries. At the shipped
    configuration that is two orders of magnitude beyond the factored agent,
    and it grows with ``V²P²`` rather than ``V²``.

    Args:
        n_vocab: Vocabulary size ``V``.
        n_positions: Position-factor size ``P``.

    Raises:
        ValueError: On non-positive sizes.
    """
    if n_vocab < 1 or n_positions < 1:
        raise ValueError("sizes must be positive")
    joint = n_vocab * n_positions
    return joint * joint * FLOAT_BYTES


def word_factored_transition_bytes(n_vocab: int, n_positions: int) -> int:
    """Bytes of the factored Stage 3 transitions: ``V² + P²`` float32 entries.

    The ``V²`` identity over word states dominates and is the term that sets
    the pymdp instantiation cap recorded at ``claim:stage3_max_agent_vocab``
    (see ``ailm.levels.word.MAX_AGENT_VOCAB_BYTES``).

    Raises:
        ValueError: On non-positive sizes.
    """
    if n_vocab < 1 or n_positions < 1:
        raise ValueError("sizes must be positive")
    return (n_vocab * n_vocab + n_positions * n_positions) * FLOAT_BYTES


def dense_joint_transition_nonzeros(n_vocab: int, n_positions: int) -> int:
    """Nonzero entries a transition over the UNfactored Stage 3 joint needs.

    The same counterfactual as :func:`dense_joint_transition_bytes`, priced by
    what the array *contains* rather than by what it occupies. The joint
    transition is a deterministic map: from each ``(word, position)`` state
    exactly one successor is reachable, so exactly one entry per source column
    is nonzero and the count is ``V·P`` rather than ``(V·P)²``.

    Args:
        n_vocab: Vocabulary size ``V``.
        n_positions: Position-factor size ``P``.

    Returns:
        Structurally nonzero entries.

    Raises:
        ValueError: On non-positive sizes.
    """
    if n_vocab < 1 or n_positions < 1:
        raise ValueError("sizes must be positive")
    return n_vocab * n_positions


def word_factored_transition_nonzeros(n_vocab: int, n_positions: int) -> int:
    """Nonzero entries the factored Stage 3 transitions need: ``V + P``.

    Both factored transitions are deterministic too — the word transition is the
    identity, the position transition advances one slot — so each contributes one
    nonzero per source state.

    Raises:
        ValueError: On non-positive sizes.
    """
    if n_vocab < 1 or n_positions < 1:
        raise ValueError("sizes must be positive")
    return n_vocab + n_positions


def joint_over_factored_ratio(n_vocab: int, n_positions: int, *, sparse: bool = False) -> float:
    """How much more the unfactored joint costs than the factored pair.

    Two routes, because the answer differs by an exponent and the difference is
    a property of the *storage*, not of the factorization:

    **Dense** — ``(V·P)² / (V² + P²)``, which is ``P²`` up to the negligible
    ``P²`` term in the denominator. This is the number the manuscript reports,
    and it is correct arithmetic.

    **Sparse** — ``V·P / (V + P)``, which tends to ``P`` as ``V`` grows. Both
    transitions are deterministic maps with one nonzero per source state, so
    under any storage that does not materialize structural zeros the joint costs
    a factor of roughly ``P`` more, not ``P²``.

    The exponent therefore belongs to the dense encoding rather than to the
    decision to factor. Stating it the other way credits the factorization with
    a saving that a change of representation would have delivered on its own,
    which is a claim about this project's implementation dressed as a claim
    about hierarchical models. Decision 58 records why the dense encoding is
    nonetheless the one that ships: no sparse path is reachable through pymdp's
    public ``Agent`` API.

    Args:
        n_vocab: Vocabulary size ``V``.
        n_positions: Position-factor size ``P``.
        sparse: Price by nonzero count rather than by dense bytes.

    Returns:
        The ratio, joint over factored.

    Raises:
        ValueError: On non-positive sizes.
    """
    if sparse:
        return dense_joint_transition_nonzeros(n_vocab, n_positions) / (
            word_factored_transition_nonzeros(n_vocab, n_positions)
        )
    return dense_joint_transition_bytes(n_vocab, n_positions) / (
        word_factored_transition_bytes(n_vocab, n_positions)
    )
