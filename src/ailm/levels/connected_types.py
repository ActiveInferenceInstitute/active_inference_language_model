"""Typed observations, interventions and traces for the connected runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class ConnectedInputError(ValueError):
    """An observation or channel configuration violates the runtime contract."""


class Channel(StrEnum):
    PHASE_CLAUSE = "phase_clause"
    CLAUSE_CASE = "clause_case"
    CASE_WORD = "case_word"
    WORD_PHONE = "word_phone"
    WORD_LETTER = "word_letter"
    CASE_CLAUSE = "case_clause"
    CLAUSE_PHASE = "clause_phase"


@dataclass(frozen=True, slots=True)
class Connections:
    """Fixed channel strength and explicit intervention arms; never fitted on a run."""

    strength: float = 0.5
    disabled: frozenset[Channel] = frozenset()
    shuffled: frozenset[Channel] = frozenset()

    def __post_init__(self) -> None:
        if not np.isfinite(self.strength) or not 0 <= self.strength < 1:
            raise ConnectedInputError("strength must be finite and in [0, 1)")
        if not (self.disabled | self.shuffled) <= frozenset(Channel):
            raise ConnectedInputError("unknown channel intervention")
        if self.disabled & self.shuffled:
            raise ConnectedInputError("a channel cannot be disabled and shuffled")


@dataclass(frozen=True, slots=True)
class Token:
    """A normalized lexical token; UPOS is external metadata available before spelling.

    Evaluation fields (case, dependency, morphology, phone and sentence-type
    labels) deliberately have no slot. Raw mode leaves UPOS absent.
    """

    form: str
    upos: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch("[a-z]+", self.form) is None:
            raise ConnectedInputError("token form must contain lowercase ASCII letters only")


@dataclass(frozen=True, slots=True)
class Sentence:
    tokens: tuple[Token, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ConnectedInputError("empty sentences are not observations")


def parse_text(text: str) -> tuple[Sentence, ...]:
    """Split ASCII letter runs at punctuation boundaries; punctuation type is withheld.

    Apostrophes, hyphens, digits and non-ASCII characters delimit lexical runs.
    This explicit normalization is shared by scoring and reported token traces.
    """
    sentences: list[Sentence] = []
    current: list[Token] = []
    for chunk in re.findall(r"[a-z]+|[.!?]+", text.lower()):
        if chunk[0].isalpha():
            current.append(Token(chunk))
        elif current:
            sentences.append(Sentence(tuple(current)))
            current = []
    if current:
        sentences.append(Sentence(tuple(current)))
    if not sentences:
        raise ConnectedInputError("text contains no ASCII lexical tokens")
    return tuple(sentences)


@dataclass(frozen=True, slots=True)
class InferenceEvent:
    """Pre-observation priors and predictions, followed by live pymdp marginals."""

    level: str
    sentence: int
    token: int
    position: int
    prior: tuple[NDArray[np.float64], ...]
    observation: NDArray[np.float64]
    predictive: NDArray[np.float64]
    posterior: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True, slots=True)
class TokenResult:
    form: str
    word_prior: NDArray[np.float64]
    word_probability: float
    oov: bool
    phone_posteriors: tuple[NDArray[np.float64], ...]
    case_posterior: NDArray[np.float64]
    upos_observation: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DocumentResult:
    tokens: tuple[TokenResult, ...]
    sentence_types: tuple[NDArray[np.float64], ...]
    phase_posterior: NDArray[np.float64]
    next_word_prior: NDArray[np.float64]
    events: tuple[InferenceEvent, ...]

    @property
    def letter_log_loss(self) -> float:
        """Mean predictive loss on the normalized letters and token spaces."""
        values = [
            -float(np.log(event.predictive[np.argmax(event.observation)]))
            for event in self.events
            if event.level == "letter"
        ]
        return float(np.mean(values))

    @property
    def word_log_loss(self) -> float:
        """Closed-vocabulary loss; any OOV gives infinity, never a dropped token."""
        if any(token.oov for token in self.tokens):
            return float("inf")
        return -float(np.mean([np.log(token.word_probability) for token in self.tokens]))


@dataclass(frozen=True, slots=True)
class PolicyEvent:
    prefix: str
    preferences: NDArray[np.float64]
    habit: NDArray[np.float64]
    posterior: NDArray[np.float64]
    action: int


@dataclass(frozen=True, slots=True)
class GeneratedWord:
    word: str
    terminated: bool
    word_prior: NDArray[np.float64]
    policies: tuple[PolicyEvent, ...]
