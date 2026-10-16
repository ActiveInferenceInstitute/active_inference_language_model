"""The widened Level 5 observation alphabet, defined once.

Level 5's shipped observation is the case role. The *widened* alphabet adds the
two parts of speech that occupy a position in surface order but carry no case —
verbs and auxiliaries — which is what lets the level see subject-auxiliary
inversion, the strongest interrogative cue in English.

## Why this module exists

The alphabet was written down three times: the symbol indices in
``ailm.dashboard.engine`` and again in ``scripts/probe_l5_widened_alphabet``,
and the chain fit in both, character for character. Nothing at runtime would
notice them diverging. A swapped ``VERB``/``AUX`` index still yields a
valid-looking posterior, just a wrong one, and a pseudocount changed in one
place would leave the demonstrator quietly modeling something the recorded
measurement never used.

So the alphabet, the extraction and the fit live here, and the probe and the
dashboard both import them. One definition, three consumers, and
``tests/test_widened.py`` asserts they produce identical chains from the same
corpus.

## The recorded arm (decision 112)

The third consumer is the widened-observation clause level: a second, separately
recorded Level 5 arm built on decision 85's authenticated document partition. It
adds a committed artifact under ``data/widened/``, a loader that validates it,
and the joint agent at the widened state count. The shipped Level 5 is untouched — its verdict is recorded
in decisions 24, 83 and 85 and its consistency guard is
``scripts/audit_gates.py::audit_stage5``.

The disjointness assertion is *strengthened* here rather than dropped, because
the widened stream admits token classes the case-only stream excluded.
``PUNCT`` still emits nothing and remains the label source, and a ``VERB`` or
``AUX`` token contributes its UPOS only — never its lemma, its ``feats`` or its
surface form. A verb's form is the single richest correlate of interrogative
mood in this corpus, so admitting it would make the widening a label leak
wearing an alphabet's clothes. :func:`assert_widened_disjointness` checks both
at artifact build time, and it takes the extractor as an argument so a planted
leak is a real negative control rather than a comment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ailm.cases import CaseRole
from ailm.corpus.ud import document_identity_sha256, validate_ud_ewt_source_metadata
from ailm.levels._stochastic import column_stochastic
from ailm.levels.clause import CLAUSE_TYPES, N_TYPES, ClauseArtifact
from ailm.resources import resolve_data_artifact

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pymdp.agent import Agent as PymdpAgent

__all__ = [
    "AUX_SYMBOL",
    "DEFAULT_WIDENED_ARTIFACT",
    "MIN_SEQUENCE_LEN",
    "N_WIDENED_JOINT",
    "N_WIDENED_SYMBOLS",
    "TYPE_BY_FINAL_PUNCT",
    "VERB_SYMBOL",
    "WIDENED_ARTIFACT_SCHEMA",
    "WIDENED_DECODER_TOLERANCE",
    "WIDENED_LEAK_MASS",
    "WIDENED_SYMBOL_LABELS",
    "WIDENED_SYMBOL_TO_IDX",
    "WidenedClauseArtifact",
    "assert_widened_disjointness",
    "build_widened_agent",
    "extract_widened_sequences",
    "extract_widened_sequences_documented",
    "extract_widened_sequences_indexed",
    "fit_markov_chain",
    "load_widened_clause_artifact",
    "widened_symbols",
]


#: Case roles occupy 0-7; the two non-case parts of speech take the next two.
VERB_SYMBOL = 8
AUX_SYMBOL = 9
N_WIDENED_SYMBOLS = 10

#: Human-readable name per symbol index, for interfaces that display a sequence.
WIDENED_SYMBOL_LABELS: tuple[str, ...] = tuple([role.name for role in CaseRole] + ["VERB", "AUX"])

#: Sentence type read off the final punctuation mark. ``"!"`` is deliberately
#: absent: it is neither modeled type, so those sentences are dropped rather
#: than forced into one.
TYPE_BY_FINAL_PUNCT: dict[str, str] = {".": "declarative", "?": "interrogative"}

#: A sequence shorter than this carries no transition at all, so it can inform a
#: start prior but never the chain the level is about.
MIN_SEQUENCE_LEN = 2

#: Symbol name -> observation index. The inverse of
#: :data:`WIDENED_SYMBOL_LABELS`, built from it rather than typed again.
WIDENED_SYMBOL_TO_IDX: dict[str, int] = {
    name: index for index, name in enumerate(WIDENED_SYMBOL_LABELS)
}

#: Joint (type x symbol) state count for the widened receiver.
N_WIDENED_JOINT: int = N_TYPES * N_WIDENED_SYMBOLS

WIDENED_ARTIFACT_SCHEMA = "ailm-clause-widened-counts-v1"

#: Off-diagonal mass planted across the type blocks by the widened arm's
#: cross-type-leak negative control, and the longest tolerated disagreement
#: between the closed-form mixture evidence and oracle filtering on the joint
#: block-diagonal chain. Defined once here and imported by the recorder, the
#: standing audit and the tests, so "the control fired" cannot mean one thing
#: in one path and another in the rest. The mass must stay large enough that a
#: correct implementation cannot absorb it into floating-point tolerance.
WIDENED_LEAK_MASS: float = 0.05
WIDENED_DECODER_TOLERANCE: float = 1e-9


#: The widened artifact lives under ``data/``, not ``vendor/``.
#:
#: ``vendor/`` is not a convenient directory of committed JSON; it *is* the
#: wheel's packaged runtime surface, and ``tests/test_wheel_resources.py`` holds
#: ``RUNTIME_VENDOR_ARTIFACTS`` equal to every committed ``vendor/*.json``. An
#: artifact placed there is packaged by that invariant's definition, so
#: "committed to vendor/ but deliberately unpackaged" is not a state the
#: repository allows — it is the invariant failing. This arm retains its
#: development-session provenance (decision 112) under ``data/``. The connected
#: runtime now consumes it, so the identical bytes are explicitly packaged
#: through ``RUNTIME_DATA_ARTIFACTS`` without moving or upgrading that evidence.
DEFAULT_WIDENED_ARTIFACT = resolve_data_artifact("widened/ud_ewt_clause_widened_counts.json")


def widened_symbols(tokens: list) -> list[int]:
    """The widened observation sequence for one sentence's tokens.

    UD ``PUNCT`` tokens are skipped: they occupy no position in the surface
    order this alphabet describes, and a model fitted with them skipped must be
    applied with them skipped.

    Args:
        tokens: UD tokens, each carrying ``upos`` and ``case_derived``.

    Returns:
        Symbol indices, one per contributing token.
    """
    sequence: list[int] = []
    for token in tokens:
        if token.upos == "PUNCT":
            continue
        if token.upos == "VERB":
            sequence.append(VERB_SYMBOL)
        elif token.upos == "AUX":
            sequence.append(AUX_SYMBOL)
        elif token.case_derived is not None:
            sequence.append(list(CaseRole).index(token.case_derived))
    return sequence


def extract_widened_sequences_indexed(sentences: list) -> list[tuple[int, str, list[int]]]:
    """``(global_index, sentence_type, widened_sequence)`` for usable sentences.

    A sentence is usable when its final punctuation names one of the two
    modeled types and it contributes at least ``MIN_SEQUENCE_LEN`` symbols.

    The index is the sentence's position in the **whole** corpus, counted before
    any exclusion. It is retained for historical audit only: decision 85 retired
    sentence-index partitioning. New split consumers must use
    :func:`extract_widened_sequences_documented` and partition the returned raw
    document identities before filtering.

    Args:
        sentences: Parsed UD sentences.

    Returns:
        Triples in corpus order, one per usable sentence.
    """
    rows: list[tuple[int, str, list[int]]] = []
    for index, sentence in enumerate(sentences):
        forms = [token.form for token in sentence.tokens]
        final = next((form for form in reversed(forms) if form in {".", "?", "!"}), None)
        if final not in TYPE_BY_FINAL_PUNCT:
            continue
        sequence = widened_symbols(sentence.tokens)
        if len(sequence) >= MIN_SEQUENCE_LEN:
            rows.append((index, TYPE_BY_FINAL_PUNCT[final], sequence))
    return rows


def extract_widened_sequences_documented(
    sentences: list,
) -> list[tuple[str, str, list[int]]]:
    """``(document_id, sentence_type, sequence)`` for usable sentences.

    The document identity comes from authenticated CoNLL-U ``# newdoc id``
    metadata. This view is the only one suitable for train/test partitioning:
    all document assignments must be fixed on the unfiltered corpus first.
    """
    indexed = extract_widened_sequences_indexed(sentences)
    return [
        (sentences[index].document_id, sentence_type, sequence)
        for index, sentence_type, sequence in indexed
    ]


def extract_widened_sequences(sentences: list) -> list[tuple[str, list[int]]]:
    """``(sentence_type, widened_sequence)`` for every usable sentence.

    The index-free view, for consumers that do not split. See
    :func:`extract_widened_sequences_indexed` for the selection rule.

    Args:
        sentences: Parsed UD sentences.

    Returns:
        Pairs in corpus order.
    """
    return [
        (stype, sequence) for _, stype, sequence in extract_widened_sequences_indexed(sentences)
    ]


def fit_markov_chain(
    sequences: list[list[int]],
    n_states: int = N_WIDENED_SYMBOLS,
    pseudocount: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit an order-1 transition and start prior with add-k smoothing.

    ``B`` is returned column-stochastic in ``B[next, current]`` orientation,
    which is pymdp's and this project's throughout. Stating it the other way is
    the orientation trap decision 1 records; the sums below are over axis 0 for
    that reason.

    Args:
        sequences: Symbol sequences to count.
        n_states: Alphabet size.
        pseudocount: Add-k smoothing weight applied to every cell.

    Returns:
        ``(B, D)`` of shapes ``(n_states, n_states)`` and ``(n_states,)``.

    Raises:
        ValueError: If ``pseudocount`` is not positive. Zero smoothing makes an
            unseen transition impossible rather than merely unlikely, which
            turns one absent bigram into an infinite log-loss.
    """
    if pseudocount <= 0:
        raise ValueError(f"pseudocount must be positive, got {pseudocount}")

    transition_counts = np.full((n_states, n_states), pseudocount, dtype=np.float64)
    start_counts = np.full(n_states, pseudocount, dtype=np.float64)

    for sequence in sequences:
        if not sequence:
            continue
        start_counts[sequence[0]] += 1.0
        for step in range(len(sequence) - 1):
            transition_counts[sequence[step + 1], sequence[step]] += 1.0

    return (
        column_stochastic(transition_counts),
        start_counts / start_counts.sum(),
    )


# ── THE STRENGTHENED DISJOINTNESS ASSERTION ──────────────────────────────────


class _WidenedExtractor(Protocol):
    """The shape :func:`widened_symbols` has, so a leak can be planted."""

    def __call__(self, tokens: list) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class _FieldStrippedToken:
    """A token exposing only the two fields the alphabet is allowed to read.

    ``lemma``, ``feats`` and ``form`` are present but replaced by one shared
    sentinel, so an extractor that consults any of them sees every token as
    identical and produces a different sequence. Removing the attributes
    instead would raise :class:`AttributeError`, which reads as a broken
    fixture rather than as a leak; a wrong answer is the signal wanted here.
    """

    upos: str
    case_derived: Any
    lemma: str = "<withheld>"
    feats: str = "<withheld>"
    form: str = "<withheld>"


def assert_widened_disjointness(
    tokens: list,
    sequence: list[int],
    *,
    extract: _WidenedExtractor = widened_symbols,
    sent_id: str = "",
) -> None:
    """Assert the widened stream cannot carry its own label source.

    Two properties, both checked by recomputation rather than asserted in
    prose, and both stated over the general class rather than over one known
    offender:

    1. **``PUNCT`` emits nothing.** Every ``PUNCT`` token in the sentence is
       offered to the extractor alone and must yield the empty sequence. The
       sentence-final punctuation form is the type label's source, so a
       punctuation token contributing any observation would put the label
       inside the observation stream — Stage 4's lesson (decision 16) at the
       new alphabet.
    2. **A ``VERB`` or ``AUX`` contributes its UPOS only.** The sequence must
       equal what the extractor produces from a field-stripped view of the same
       tokens, in which lemma, ``feats`` and surface form are one shared
       sentinel. Any extractor that reads those fields disagrees, because the
       stripped tokens no longer distinguish what it was reading.

    Args:
        tokens: The sentence's UD tokens.
        sequence: The widened sequence the generator is about to record.
        extract: The extraction under test. Defaulted to the shipped
            :func:`widened_symbols` and parameterized so a planted leak is an
            executable negative control.
        sent_id: Sentence identifier, quoted in the failure.

    Raises:
        AssertionError: If either property fails.
    """
    for token in tokens:
        if token.upos == "PUNCT" and extract([token]) != []:
            raise AssertionError(
                f"sentence {sent_id}: a PUNCT token contributes an observation, "
                f"so the sentence type label's source would leak into the "
                f"widened stream. The widened clause level cannot be built on this."
            )

    stripped = [
        _FieldStrippedToken(upos=token.upos, case_derived=token.case_derived) for token in tokens
    ]
    if extract(stripped) != sequence:
        raise AssertionError(
            f"sentence {sent_id}: the widened sequence changes when lemma, feats "
            f"and form are withheld, so the extraction reads a field outside "
            f"(upos, case_derived). A VERB or AUX may contribute its UPOS only."
        )


# ── THE VENDORED WIDENED ARTIFACT ────────────────────────────────────────────


class WidenedClauseArtifact:
    """The vendored widened Stage 5 statistics, validated on load.

    The same contract as :class:`ailm.levels.clause.ClauseArtifact` at the
    widened alphabet. The document machinery is not re-implemented: this class
    calls that one's validator, so the two artifacts cannot drift in how they
    check identity digests, split partitioning, or sequence/document alignment.

    Attributes:
        train: ``(type, [widened indices])`` training sentences.
        test: Held-out sentences, same shape.
        provenance: The generator's recorded parameters.
    """

    def __init__(self, payload: dict) -> None:
        if payload.get("schema") != WIDENED_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unexpected artifact schema {payload.get('schema')!r}; "
                f"regenerate with scripts/generate_clause_widened_counts.py"
            )
        self.provenance: dict = payload["provenance"]
        if self.provenance.get("alphabet") != "widened":
            raise ValueError(
                f"widened artifact provenance declares alphabet "
                f"{self.provenance.get('alphabet')!r}; this loader reads only the "
                f"widened alphabet of decision 112"
            )
        self.train = self._validate_split(payload["splits"]["train"], "train")
        self.test = self._validate_split(payload["splits"]["test"], "test")
        (
            self.document_ids,
            split_document_indices,
            sequence_document_indices,
        ) = ClauseArtifact._validate_documents(payload.get("documents"), payload.get("splits"))
        self.train_split_document_indices = split_document_indices["train"]
        self.test_split_document_indices = split_document_indices["test"]
        self.train_sequence_document_indices = sequence_document_indices["train"]
        self.test_sequence_document_indices = sequence_document_indices["test"]
        self.train_document_ids = [
            self.document_ids[index] for index in sorted(set(self.train_sequence_document_indices))
        ]
        self.test_document_ids = [
            self.document_ids[index] for index in sorted(set(self.test_sequence_document_indices))
        ]
        if self.provenance.get("document_identity_sha256") != document_identity_sha256(
            self.document_ids
        ):
            raise ValueError(
                "widened artifact document identity digest disagrees with its ordered ids"
            )
        if self.provenance.get("documents") != len(self.document_ids):
            raise ValueError("widened artifact provenance document count is inconsistent")
        if self.provenance.get("train_documents") != len(self.train_split_document_indices):
            raise ValueError("widened artifact provenance train-document count is inconsistent")
        if self.provenance.get("test_documents") != len(self.test_split_document_indices):
            raise ValueError("widened artifact provenance test-document count is inconsistent")
        train_types = {t for t, _ in self.train}
        if train_types != set(CLAUSE_TYPES):
            raise ValueError(
                f"training split carries types {sorted(train_types)}; both "
                f"{CLAUSE_TYPES} are required to estimate the mixture"
            )

    @staticmethod
    def _validate_split(entries: list, name: str) -> list[tuple[str, list[int]]]:
        """Validate one split, refusing any symbol outside the widened alphabet."""
        if not entries:
            raise ValueError(f"{name} split is empty")
        out: list[tuple[str, list[int]]] = []
        for entry in entries:
            sentence_type = entry["type"]
            if sentence_type not in CLAUSE_TYPES:
                raise ValueError(f"unknown sentence type {sentence_type!r}")
            symbols = entry["symbols"]
            if len(symbols) < MIN_SEQUENCE_LEN:
                raise ValueError(f"{name} split carries a sequence with no transition")
            stray = next((s for s in symbols if s not in WIDENED_SYMBOL_TO_IDX), None)
            if stray is not None:
                # Decision 16's structural guard at the widened alphabet: any
                # symbol outside the ten could smuggle the label source in.
                raise ValueError(
                    f"observation {stray!r} is outside the widened alphabet; "
                    f"the artifact was not built by the shipped generator"
                )
            out.append((sentence_type, [WIDENED_SYMBOL_TO_IDX[s] for s in symbols]))
        return out

    @property
    def n_test_documents(self) -> int:
        """Number of source documents contributing typed held-out sentences."""
        return len(self.test_document_ids)


def load_widened_clause_artifact(path: Path | str | None = None) -> WidenedClauseArtifact:
    """Load and validate the vendored widened clause artifact.

    Raises:
        FileNotFoundError: If absent — this arm has no fallback corpus.
        ValueError: On validation failure.
    """
    resolved = DEFAULT_WIDENED_ARTIFACT if path is None else Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{resolved} is absent. Run scripts/generate_clause_widened_counts.py."
        )
    artifact = WidenedClauseArtifact(json.loads(resolved.read_text(encoding="utf-8")))
    validate_ud_ewt_source_metadata(artifact.provenance)
    return artifact


# ── THE JOINT pymdp AGENT, AT THE WIDENED STATE COUNT ────────────────────────


def build_widened_agent(
    models: dict,
    *,
    transition: NDArray[np.float64] | None = None,
    start: NDArray[np.float64] | None = None,
) -> PymdpAgent:
    """Build the widened joint agent from fitted type-conditioned models.

    Structurally identical to :func:`ailm.levels.clause.build_clause_agent` and
    built by the same shipped builders at a larger alphabet, so the widened arm
    exercises the level's machinery rather than a parallel copy of it.

    Args:
        models: Fitted models from ``build_type_models(..., n_symbols=10)``.
        transition: Override for the joint ``B``. Negative controls inject
            mis-wired matrices through the shipped builder, which is decision
            15's discipline.
        start: Override for the joint ``D``.

    Raises:
        ValueError: If an override has the wrong shape.
    """
    import jax.numpy as jnp
    from pymdp.agent import Agent

    from ailm.levels.clause import (
        build_joint_emission,
        build_joint_start,
        build_joint_transition,
    )

    b = (
        build_joint_transition(models)
        if transition is None
        else np.asarray(transition, dtype=np.float64)
    )
    d = build_joint_start(models) if start is None else np.asarray(start, dtype=np.float64)
    if b.shape != (N_WIDENED_JOINT, N_WIDENED_JOINT):
        raise ValueError(
            f"transition must be ({N_WIDENED_JOINT}, {N_WIDENED_JOINT}), got {b.shape}"
        )
    if d.shape != (N_WIDENED_JOINT,):
        raise ValueError(f"start must be ({N_WIDENED_JOINT},), got {d.shape}")

    emission = build_joint_emission(N_WIDENED_SYMBOLS)
    return Agent(
        A=[jnp.array(emission, dtype=jnp.float32)[jnp.newaxis]],
        B=[jnp.array(b, dtype=jnp.float32)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, N_WIDENED_SYMBOLS))],
        D=[jnp.array(d, dtype=jnp.float32)[jnp.newaxis]],
        num_controls=[1],
        batch_size=1,
    )
