"""The channel factorial: message source crossed with insertion point.

[@sec:hierarchy] states that the factorial version of the channel experiment
"crosses message source with insertion point and reports all four cells", and
then that "the two cells that would separate source from insertion point are not
built". Decisions 18 and 19 — the novel-information criterion — therefore rest on
a diagonal: the letter message enters at the evidence site and harms, the lexical
message enters at the predictive site and helps, and source moves with site.

This module is the executor for all four cells. It is a **dispatcher, not an
implementation**: every cell routes into the shipped primitives
(:func:`ailm.levels.coupling.mix_letter_distribution`,
:func:`ailm.levels.coupling.soft_observation_likelihood`,
:func:`ailm.levels.lexical_channel.coupled_g2p_positions`) and reimplements none
of them. That is the whole binding, and it is gated before use: the two cells
that already exist must reproduce their shipped per-position hit vectors exactly,
not their accuracies. Two decoders can agree to four decimal places on an
aggregate while disagreeing on thousands of positions in both directions, and an
executor that merely reproduces the number is measuring itself.

Three things this factorial does NOT de-confound, registered in decision 46 and
repeated here because a caller of this module is exactly who needs to know:

* The mixing weight is not a common dose across sites. The evidence vector's
  total mass varies by a factor of about twenty-five across letters, so a declared
  weight is a different effective weight depending on the observed letter.
  :func:`run_channel` scale-matches and reports both readings.
* The predictive site compounds along the word and the evidence site does not.
  They are not two levels of one factor.
* Source is collinear with conditioning window: the letter message is exclusive
  of the current letter by construction, the lexical message is inclusive of it
  plus absolute position.

The alphabet transport is the fourth, and it is this module's own finding. The
evidence site consumes a distribution over letters and the predictive site one
over phones, so both off-diagonal cells must move a message between alphabets
while neither diagonal cell does. See :func:`letters_to_phones`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Imported for the signature of ``run_cell``, which is declared before it is
# built so that its gate can exist and fail first. ``noqa`` rather than deletion:
# removing them would hide which primitives this module is contracted to dispatch
# into, and that contract is the whole point of the file.
from ailm.levels.coupling import (  # noqa: F401
    G2PMatrices,
    LetterMatrices,
    letter_predictive_distributions,
    mix_letter_distribution,
    soft_observation_likelihood,
)
from ailm.levels.lexical_channel import (  # noqa: F401
    LexicalMessage,
    coupled_g2p_positions,
    coupled_g2p_posteriors,
    lexical_message_for,
)

#: Floor on a message coordinate under geometric pooling. A product of experts
#: treats a zero as a veto, so an unsmoothed zero would delete a phone the
#: emission still supports; this keeps the veto from being an artifact of a count
#: that happened to be absent from the training split.
_MESSAGE_FLOOR = 1e-12

__all__ = [
    "CELLS",
    "ChannelCell",
    "letters_to_phones",
    "phones_to_letters",
    "reverse_emission",
]

Source = Literal["letter", "lexicon"]
Site = Literal["evidence", "predictive"]

#: The four cells, diagonal first. The two diagonal cells are the shipped
#: experiments and exist only to gate the executor; the two off-diagonal cells are
#: the ones [@sec:hierarchy] names as unbuilt.
CELLS: tuple[tuple[Source, Site], ...] = (
    ("letter", "evidence"),
    ("lexicon", "predictive"),
    ("letter", "predictive"),
    ("lexicon", "evidence"),
)


@dataclass(frozen=True)
class ChannelCell:
    """One cell's identity and what it required to be built."""

    source: Source
    site: Site
    transported: bool
    shipped: bool


def reverse_emission(
    emission: NDArray[np.float64] | np.ndarray,
    phone_prior: NDArray[np.float64] | np.ndarray,
) -> NDArray[np.float64]:
    """``P(phone | letter)`` by Bayes inversion under the model's own phone prior.

    The emission is ``P(letter | phone)``, column-stochastic over letters, so
    ``emission.T @ q`` for a letter distribution ``q`` returns a *likelihood* over
    phones that does not sum to one — about 1.87 on this artifact. Normalizing it
    is not available: :func:`soft_observation_likelihood` states that doing so
    "would turn the evidence into a prior and silently change the model", and a
    cell built that way would answer a different question from the other three.

    The inversion uses only parameters the model already holds, which is what
    makes it a declared choice rather than a new fit. It is registered in decision
    46 as a model choice, with the alternative — the normalized likelihood — named
    as the sensitivity check the arm owes.

    Raises:
        ValueError: If any letter carries zero evidence mass under the prior, so
            that its conditional is undefined. Returning a uniform row there would
            invent a message for a letter the model says nothing about.
    """
    emis = np.asarray(emission, dtype=np.float64)
    prior = np.asarray(phone_prior, dtype=np.float64)
    if emis.ndim != 2 or prior.shape != (emis.shape[1],):
        raise ValueError(f"emission {emis.shape} and prior {prior.shape} do not compose")
    joint = emis * prior[None, :]
    mass = joint.sum(axis=1)
    dead = np.flatnonzero(mass <= 0.0)
    if dead.size:
        raise ValueError(
            f"letters {dead.tolist()} carry no evidence mass under the phone prior, so "
            f"P(phone | letter) is undefined for them; a uniform row here would invent a "
            f"message for a letter the model makes no statement about"
        )
    return joint / mass[:, None]


def letters_to_phones(
    letter_rows: NDArray[np.float64] | np.ndarray,
    emission: NDArray[np.float64] | np.ndarray,
    phone_prior: NDArray[np.float64] | np.ndarray,
) -> NDArray[np.float64]:
    """Carry per-position letter distributions to phone distributions.

    Used by the ``letter`` source at the ``predictive`` site, which is one of the
    two cells [@sec:hierarchy] names as unbuilt. Returns rows that sum to one, so
    the predictive site receives the same kind of object the lexical source gives
    it.
    """
    rows = np.asarray(letter_rows, dtype=np.float64)
    reverse = reverse_emission(emission, phone_prior)
    carried = rows @ reverse
    return carried / carried.sum(axis=1, keepdims=True)


def phones_to_letters(
    phone_rows: NDArray[np.float64] | np.ndarray,
    emission: NDArray[np.float64] | np.ndarray,
) -> NDArray[np.float64]:
    """Carry per-position phone distributions to letter distributions.

    Used by the ``lexicon`` source at the ``evidence`` site. This direction needs
    no inversion: the emission is column-stochastic over letters, so mixing it
    against a phone distribution already yields a proper letter distribution. The
    asymmetry with :func:`letters_to_phones` is not a nuisance — it is why the two
    off-diagonal cells are not mirror images of each other, and decision 46
    records it.
    """
    rows = np.asarray(phone_rows, dtype=np.float64)
    emis = np.asarray(emission, dtype=np.float64)
    carried = rows @ emis.T
    return carried / carried.sum(axis=1, keepdims=True)


def effective_weight(alpha: float, evidence_mass: float) -> float:
    """The weight a declared ``alpha`` actually applies at the evidence site.

    The evidence vector is a likelihood whose total mass varies across letters, so
    mixing at a declared alpha applies a different effective weight depending on
    which letter was observed. Reporting only the declared grid would make the
    "common weight grid" claim false across sites. Both are recorded.
    """
    if evidence_mass <= 0.0:
        return 0.0
    return alpha / (alpha + (1.0 - alpha) / evidence_mass)


def _geometric_posteriors(
    emission: NDArray[np.float64],
    transition: NDArray[np.float64],
    start: NDArray[np.float64],
    letter_indices: list[int],
    message_rows: NDArray[np.float64],
    exponent: float,
) -> NDArray[np.float64]:
    """Per-position posteriors of the geometric (product-of-experts) recursion.

    The single implementation: :func:`_geometric_positions` reads the argmax off
    these rows and the proper-loss pass reads the gold phone's mass from the same
    rows, so the two axes cannot be computed from recursions that have drifted.

    A product of experts rather than a mixture: the posterior is proportional to
    ``A[l] * m**exponent * pred``. Decision 47 proves that inserting the message at
    the evidence site and at the predictive site are the *same* computation under
    this rule, because multiplication is associative and the intermediate
    normalization cancels; :func:`geometric_site_equivalence` measures how far
    apart they land in floating point, and the recorder writes that magnitude to
    the ledger rather than anyone typing it here. So this function takes no site
    argument: there is only one cell per source, and offering a choice would
    invite a caller to believe it mattered.

    ``exponent = 0`` reduces to the uncoupled decode exactly, since anything to
    the zeroth power is one. That is the identity anchor, and it is exact rather
    than approximate.

    The floor on the message guards ``0 ** negative`` and, more importantly, keeps
    a zero in the message from annihilating a phone the emission still supports.
    Under a product of experts a single zero is a veto, which is the rule's
    sharpest property and its most dangerous one.
    """
    predictive = np.asarray(start, dtype=np.float64).copy()
    posteriors = np.empty((len(letter_indices), predictive.shape[0]), dtype=np.float64)
    for position, letter in enumerate(letter_indices):
        if exponent > 0.0:
            weighted = np.power(np.maximum(message_rows[position], _MESSAGE_FLOOR), exponent)
        else:
            weighted = 1.0
        posterior = emission[letter, :] * weighted * predictive
        mass = float(posterior.sum())
        if mass <= 0.0:
            posterior = predictive.copy()
            mass = float(posterior.sum())
        posterior = posterior / mass
        posteriors[position] = posterior
        predictive = transition @ posterior
    return posteriors


def _geometric_positions(
    emission: NDArray[np.float64],
    transition: NDArray[np.float64],
    start: NDArray[np.float64],
    letter_indices: list[int],
    truth: list[int],
    message_rows: NDArray[np.float64],
    exponent: float,
) -> list[bool]:
    """Hit vector of the geometric recursion; see :func:`_geometric_posteriors`."""
    posteriors = _geometric_posteriors(
        emission, transition, start, letter_indices, message_rows, exponent
    )
    return [int(np.argmax(row)) == gold for row, gold in zip(posteriors, truth, strict=True)]


def geometric_site_equivalence(
    emission: NDArray[np.float64],
    transition: NDArray[np.float64],
    start: NDArray[np.float64],
    letter_indices: list[int],
    message_rows: NDArray[np.float64],
    exponent: float,
) -> tuple[float, int]:
    """How far apart the two insertion sites actually are under geometric pooling.

    Decision 47 proves they are the same computation: multiplication is
    associative and the intermediate normalization cancels, so pooling the
    message into the evidence and pooling it into the predictive prior differ
    only by floating-point ordering. A proof is not a measurement, and the
    manuscript reports a magnitude, so the magnitude is measured here rather
    than asserted — it was a typed ``1.7e-16`` in the paper and in this module's
    own docstring, backed by nothing, until this existed.

    The two variants are written out separately on purpose. Sharing an
    implementation between them would make the equivalence true by construction
    and the measurement worthless.

    Args:
        emission: Letter-given-phone emission.
        transition: Phone transition.
        start: Phone start prior.
        letter_indices: Observed letters, in order.
        message_rows: One message row per position.
        exponent: Pooling exponent. At zero both variants reduce to the
            uncoupled decode; the measured difference is floating-point
            rounding (~1e-16, one extra renormalization on the predictive
            side), not an exact zero.

    Returns:
        ``(max_abs_difference, argmax_disagreements)`` over every position.
    """

    def at_evidence() -> list[NDArray[np.float64]]:
        """Posteriors that weight the top-down message by ``exponent`` (evidence arm)."""
        predictive = np.asarray(start, dtype=np.float64).copy()
        out = []
        for position, letter in enumerate(letter_indices):
            weighted = (
                np.power(np.maximum(message_rows[position], _MESSAGE_FLOOR), exponent)
                if exponent > 0.0
                else 1.0
            )
            posterior = (emission[letter, :] * weighted) * predictive
            posterior = posterior / posterior.sum()
            out.append(posterior)
            predictive = transition @ posterior
        return out

    def at_predictive() -> list[NDArray[np.float64]]:
        """Posteriors from the predictive (message-weighting-disabled) arm."""
        predictive = np.asarray(start, dtype=np.float64).copy()
        out = []
        for position, letter in enumerate(letter_indices):
            weighted = (
                np.power(np.maximum(message_rows[position], _MESSAGE_FLOOR), exponent)
                if exponent > 0.0
                else 1.0
            )
            pooled = predictive * weighted
            pooled = pooled / pooled.sum()
            posterior = emission[letter, :] * pooled
            posterior = posterior / posterior.sum()
            out.append(posterior)
            predictive = transition @ posterior
        return out

    left, right = at_evidence(), at_predictive()
    worst = 0.0
    disagreements = 0
    for a, b in zip(left, right, strict=True):
        worst = max(worst, float(np.max(np.abs(a - b))))
        disagreements += int(np.argmax(a) != np.argmax(b))
    return worst, disagreements


def _iter_cell_posteriors(
    source: Source,
    site: Site,
    *,
    letter_matrices: LetterMatrices,
    g2p: G2PMatrices,
    message: LexicalMessage,
    pairs: list[tuple[str, list[str]]],
    alpha: float,
    phone_to_index: dict[str, int],
    limit: int | None = None,
):
    """Yield ``(posteriors, truth)`` per word for one factorial cell.

    The single dispatch behind :func:`run_cell` and :func:`run_cell_proper_loss`:
    both read off the same posterior rows, so the accuracy axis and the loss axis
    of one cell cannot be produced by recursions that have drifted apart.

    Every cell dispatches. The evidence site routes into
    :func:`ailm.levels.coupling._soft_forward`, which is the recursion the shipped
    coupled decoder runs; the predictive site routes into
    :func:`ailm.levels.lexical_channel.coupled_g2p_posteriors`, the recursion the
    shipped lexical channel runs. Nothing here reimplements a filter, which is why
    the gate in ``tests/test_channel_factorial.py`` can be an identity check
    rather than a tolerance.

    The two off-diagonal cells transport their message across alphabets first, by
    the rule decision 46 registers: :func:`letters_to_phones` for a letter message
    arriving at the predictive site, :func:`phones_to_letters` for a lexical
    message arriving at the evidence site.
    """
    from ailm.levels.coupling import _soft_forward, _validate_alpha
    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.phoneme import _iter_pairs

    weight = _validate_alpha(alpha)
    emission = np.asarray(g2p.emission, dtype=np.float64)
    prior = np.asarray(g2p.start, dtype=np.float64)

    for word, phones in _iter_pairs(pairs, limit):
        letters = np.array([ALPHA_TO_IDX[ch] for ch in word], dtype=np.int64)
        truth = [phone_to_index[p] for p in phones]

        if source == "letter":
            rows = np.asarray(
                letter_predictive_distributions(letter_matrices, letters), dtype=np.float64
            )
        else:
            rows, _ = lexical_message_for(message, word)
            rows = np.asarray(rows, dtype=np.float64)

        if site == "evidence":
            # The evidence site consumes LETTER distributions.
            if source == "lexicon":
                rows = phones_to_letters(rows, emission)
            posteriors, _mixtures, _evidence = _soft_forward(g2p, letters, rows, weight)
        else:
            # The predictive site consumes PHONE distributions.
            if source == "letter":
                rows = letters_to_phones(rows, emission, prior)
            posteriors = coupled_g2p_posteriors(
                emission,
                np.asarray(g2p.transition, dtype=np.float64),
                prior,
                letters,
                None if weight == 0.0 else rows,
                weight,
            )

        if posteriors.shape[0] != len(truth):
            raise ValueError(
                f"cell ({source}, {site}) decoded {posteriors.shape[0]} positions against "
                f"{len(truth)} reference phones for {word!r}; the corpus is "
                f"length-matched by construction, so this is a normalization bug"
            )
        yield posteriors, truth


def run_cell(
    source: Source,
    site: Site,
    *,
    letter_matrices: LetterMatrices,
    g2p: G2PMatrices,
    message: LexicalMessage,
    pairs: list[tuple[str, list[str]]],
    alpha: float,
    phone_to_index: dict[str, int],
    limit: int | None = None,
) -> list[bool]:
    """One cell's per-position hit vector, words in order and letters in order.

    Returns booleans rather than an accuracy on purpose. The paired tests this arm
    reports need the pairing, and the executor's own gate compares hit vectors
    because two decoders can agree on an aggregate while disagreeing on thousands
    of positions in both directions.

    Dispatch and transport live in :func:`_iter_cell_posteriors`; this reads the
    argmax off each row.
    """
    hits: list[bool] = []
    for posteriors, truth in _iter_cell_posteriors(
        source,
        site,
        letter_matrices=letter_matrices,
        g2p=g2p,
        message=message,
        pairs=pairs,
        alpha=alpha,
        phone_to_index=phone_to_index,
        limit=limit,
    ):
        hits.extend(
            int(np.argmax(row)) == gold for row, gold in zip(posteriors, truth, strict=True)
        )
    return hits


def run_cell_proper_loss(
    source: Source,
    site: Site,
    *,
    letter_matrices: LetterMatrices,
    g2p: G2PMatrices,
    message: LexicalMessage,
    pairs: list[tuple[str, list[str]]],
    alpha: float,
    phone_to_index: dict[str, int],
    limit: int | None = None,
) -> tuple[float, int, float]:
    """One cell scored on the receiver's proper predictive loss.

    Mean ``-log q_t(gold phone)`` in nats over the identical positions
    :func:`run_cell` scores, from the identical posterior rows, plus the position
    count and the accuracy read off the same rows — so the loss axis and the
    accuracy axis of a cell can be compared with no possibility of a
    position-set or recursion mismatch.

    The clip guards only ``log(0)``: a zero there is a real miss that must score
    catastrophically, not be smoothed into something survivable.

    Returns:
        ``(mean_nats, positions, accuracy_pct)``.

    Raises:
        ValueError: If no positions were scored.
    """
    total = 0.0
    positions = 0
    correct = 0
    for posteriors, truth in _iter_cell_posteriors(
        source,
        site,
        letter_matrices=letter_matrices,
        g2p=g2p,
        message=message,
        pairs=pairs,
        alpha=alpha,
        phone_to_index=phone_to_index,
        limit=limit,
    ):
        gold = np.asarray(truth, dtype=np.int64)
        rows = np.arange(gold.shape[0])
        gold_mass = np.clip(posteriors[rows, gold], 1e-300, 1.0)
        total += float(-np.sum(np.log(gold_mass)))
        correct += int(np.sum(np.argmax(posteriors, axis=1) == gold))
        positions += int(gold.shape[0])
    if positions == 0:
        raise ValueError(f"cell ({source}, {site}) scored no positions")
    return total / positions, positions, 100.0 * correct / positions
