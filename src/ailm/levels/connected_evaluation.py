"""Streaming document estimands for the locally registered connected GUM comparison."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from ailm.corpus.connected_gum import GumDocument, GumProtocolError
from ailm.levels.connected import ConnectedRuntime
from ailm.levels.connected_builder import ConnectedModel
from ailm.levels.connected_types import Channel, Connections

BOOTSTRAP_SEED: Final = 20261005
BOOTSTRAP_RESAMPLES: Final = 10000
BOOTSTRAP_CONFIDENCE: Final = 0.95
CONNECTED_STRENGTH: Final = 0.5


def validate_frozen_settings(strength: float, seed: int, resamples: int, confidence: float) -> None:
    """Refuse changes to the settings named by the authenticated local protocol."""
    if (strength, seed, resamples, confidence) != (0.5, 20261005, 10000, 0.95):
        raise GumProtocolError("estimator settings differ from the frozen protocol")


@dataclass(frozen=True, slots=True)
class ArmSummary:
    """Bounded sufficient measurements; no event histories or token priors retained."""

    letter_log_loss: float
    word_log_loss: float | None
    word_loss_status: str
    normalized_tokens: int
    oov_tokens: int
    normalized_letters: int
    scored_characters: int
    level_events: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DocumentPair:
    document_id: str
    source_sentences: int
    source_characters: int
    source_nonspace_characters: int
    integrated: ArmSummary
    disconnected: ArmSummary

    @property
    def letter_gain(self) -> float:
        """Positive means integrated prediction has lower per-document letter loss."""
        return self.disconnected.letter_log_loss - self.integrated.letter_log_loss


@dataclass(frozen=True, slots=True)
class DocumentUncertainty:
    document_mean_gain: float
    ci_low: float
    ci_high: float
    outcome: str
    seed: int = BOOTSTRAP_SEED
    resamples: int = BOOTSTRAP_RESAMPLES
    confidence: float = BOOTSTRAP_CONFIDENCE


def _score_document(runtime: ConnectedRuntime, document: GumDocument) -> ArmSummary:
    """Release each real document's full trace immediately after summarizing it."""
    result = runtime.infer_text(document.text)
    loss = result.letter_log_loss
    if not np.isfinite(loss):
        raise GumProtocolError(f"nonfinite predictive letter loss: {document.document_id}")
    oov = sum(token.oov for token in result.tokens)
    word_loss = result.word_log_loss
    if not oov and not np.isfinite(word_loss):
        raise GumProtocolError(f"nonfinite in-vocabulary word loss: {document.document_id}")
    events = Counter(event.level for event in result.events)
    if set(events) - {"letter", "phone", "word", "case", "clause", "phase"} or not {
        "letter",
        "phone",
        "case",
        "clause",
        "phase",
    } <= set(events):
        raise GumProtocolError(f"incomplete inference level inventory: {document.document_id}")
    events.setdefault("word", 0)
    return ArmSummary(
        loss,
        None if oov else word_loss,
        "infinite_oov" if oov else "finite",
        len(result.tokens),
        oov,
        sum(len(token.form) for token in result.tokens),
        events["letter"],
        tuple(sorted(events.items())),
    )


def evaluate_documents(
    model: ConnectedModel, documents: Sequence[GumDocument]
) -> Iterator[DocumentPair]:
    """Yield the fixed live connected/disconnected comparison once per document.

    The arms share parameters, normalization and input. The disconnected arm
    disables every named channel while retaining the fixed raw observation
    encoder. A runtime failure propagates; no document is skipped or substituted.
    """
    validate_frozen_settings(
        CONNECTED_STRENGTH, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, BOOTSTRAP_CONFIDENCE
    )
    if len(documents) < 2 or len({document.document_id for document in documents}) != len(
        documents
    ):
        raise GumProtocolError("evaluation requires distinct paired documents")
    integrated = ConnectedRuntime(model, Connections(strength=CONNECTED_STRENGTH))
    disconnected = ConnectedRuntime(
        model, Connections(strength=CONNECTED_STRENGTH, disabled=frozenset(Channel))
    )
    for document in documents:
        full = _score_document(integrated, document)
        control = _score_document(disconnected, document)
        if (full.normalized_tokens, full.oov_tokens, full.scored_characters) != (
            control.normalized_tokens,
            control.oov_tokens,
            control.scored_characters,
        ):
            raise GumProtocolError(f"unpaired observation population: {document.document_id}")
        yield DocumentPair(
            document.document_id,
            len(document.sentences),
            len(document.text),
            sum(not char.isspace() for char in document.text),
            full,
            control,
        )


def document_uncertainty(differences: Sequence[float]) -> DocumentUncertainty:
    """Registered equal-document percentile bootstrap with a fixed local RNG stream."""
    validate_frozen_settings(
        CONNECTED_STRENGTH, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, BOOTSTRAP_CONFIDENCE
    )
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise GumProtocolError("uncertainty requires finite paired document differences")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [(1 - BOOTSTRAP_CONFIDENCE) / 2, (1 + BOOTSTRAP_CONFIDENCE) / 2])
    outcome = "unresolved"
    if low > 0:
        outcome = "integrated_lower_loss"
    elif high < 0:
        outcome = "integrated_higher_loss"
    return DocumentUncertainty(float(values.mean()), float(low), float(high), outcome)
