"""Assemble GNN-specified agents and train-only kernels for causal composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ailm.cases import CaseRole
from ailm.corpus.case_word import load_case_word_artifact
from ailm.levels.case import load_case_artifact
from ailm.levels.connected_types import ConnectedInputError
from ailm.levels.phoneme import PHONE_TO_IDX, g2p_split, load_phoneme_counts
from ailm.levels.word import WordArtifact, load_word_artifact
from ailm.spec_agent import build_agent_from_spec

if TYPE_CHECKING:
    from pymdp.agent import Agent


@dataclass(frozen=True, slots=True)
class ConnectedModel:
    """Fixed agents and normalized kernels; all estimation precedes document inference."""

    letter: Agent
    phone: Agent
    word: Agent
    case: Agent
    clause: Agent
    phase: Agent
    generator: Agent
    lexicon: WordArtifact
    upos: tuple[str, ...]
    word_given_case: NDArray[np.float64]
    upos_given_word: NDArray[np.float64]
    unknown_upos: NDArray[np.float64]
    phone_given_word_position: NDArray[np.float64]

    def __post_init__(self) -> None:
        vocabulary = len(self.lexicon.vocab)
        n_phone = int(self.phone.D[0].shape[-1])
        expected = (
            (self.word_given_case, (vocabulary, len(CaseRole)), 0),
            (self.upos_given_word, (len(self.upos), vocabulary), 0),
            (self.unknown_upos, (len(self.upos),), 0),
            (
                self.phone_given_word_position,
                (self.lexicon.n_positions, n_phone, vocabulary),
                1,
            ),
        )
        for matrix, shape, axis in expected:
            if (
                matrix.shape != shape
                or not np.isfinite(matrix).all()
                or (matrix < 0).any()
                or not np.allclose(matrix.sum(axis=axis), 1, atol=1e-6)
            ):
                raise ConnectedInputError(f"invalid connected kernel: expected {shape}")
            matrix.setflags(write=False)


def load_connected_model() -> ConnectedModel:
    """Build canonical agents from GNN and fit bridges exclusively from existing train rows.

    Unsupported role columns use the lexical unigram. Unpronounced words and
    positions beyond a training pronunciation use the phone start law. These
    backoffs are part of the model, never selected using inference inputs.
    """
    lexicon = load_word_artifact()
    case_artifact = load_case_artifact()
    bridge = load_case_word_artifact()
    agents = [
        build_agent_from_spec(name)
        for name in (
            "L1_letter",
            "L2b_grapheme_to_phoneme",
            "L3_word",
            "L4_case",
            "L5w_clause_widened",
            "L6_narrative",
            "L1g_letter_generation",
        )
    ]
    letter, phone, word, case, clause, phase, generator = agents
    vocabulary = len(lexicon.vocab)
    prior = np.asarray(word.D[0], dtype=np.float64).ravel()
    prior /= prior.sum()
    word_given_case = np.tile(prior[:, None], (1, len(CaseRole)))
    for role_index, role in enumerate(CaseRole):
        counts = bridge.sender_map.get(role.name, {})
        if counts:
            column = np.array([counts.get(w, 0) for w in lexicon.vocab], dtype=np.float64)
            word_given_case[:, role_index] = column / column.sum()

    upos = tuple(case_artifact.upos_tags)
    upos_index = {name: index for index, name in enumerate(upos)}
    lexical_upos = np.ones((len(upos), vocabulary), dtype=np.float64)
    unknown_upos = np.ones(len(upos), dtype=np.float64)
    for document in bridge.train_documents:
        for row in document["rows"]:
            if row.upos in upos_index:
                index = upos_index[row.upos]
                unknown_upos[index] += 1
                word_index = lexicon.word_to_index.get(row.word_form.lower())
                if word_index is not None:
                    lexical_upos[index, word_index] += 1
    lexical_upos /= lexical_upos.sum(axis=0, keepdims=True)
    unknown_upos /= unknown_upos.sum()

    phone_start = np.asarray(phone.D[0], dtype=np.float64).ravel()
    phone_start /= phone_start.sum()
    phone_kernel = np.tile(phone_start[None, :, None], (lexicon.n_positions, 1, vocabulary))
    train, _heldout = g2p_split(load_phoneme_counts())
    pronunciations: dict[str, list[list[str]]] = {}
    for spelling, phones in train:
        pronunciations.setdefault(spelling, []).append(phones)
    for index, spelling in enumerate(lexicon.vocab):
        variants = pronunciations.get(spelling, [])
        for position in range(lexicon.n_positions):
            phone_counts = np.zeros(len(phone_start), dtype=np.float64)
            for phones in variants:
                if position < len(phones):
                    phone_counts[PHONE_TO_IDX[phones[position]]] += 1
            if phone_counts.sum() > 0:
                phone_kernel[position, :, index] = phone_counts / phone_counts.sum()
    return ConnectedModel(
        letter,
        phone,
        word,
        case,
        clause,
        phase,
        generator,
        lexicon,
        upos,
        word_given_case,
        lexical_upos,
        unknown_upos,
        phone_kernel,
    )
