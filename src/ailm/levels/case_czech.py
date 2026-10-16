"""The Czech arm of Level 4: real case morphology as ground truth (decision 77).

Same model shape as :mod:`ailm.levels.case` — a single-factor model whose
hidden state is the case role, whose observation is the UPOS tag, with
``B = P(case_t | case_{t-1})``, ``A = P(UPOS | case)`` and a forward-filtering
decode — scored on UD Czech-PDTC, where every label is UD morphological
``feats[Case]``. The builders and scorers are *imported from the English
module unmodified*: :class:`CzechCaseArtifact` exposes the same duck-typed
surface (``train``/``test`` as ``[(upos, case_name), ...]`` sequences,
``upos_tags``, ``upos_to_idx``, ``n_obs``), so any divergence in construction
would be a bug in exactly one place rather than a fork.

Accuracy, not agreement
-----------------------
The English module names its score ``case_agreement`` because its derived
labels are a documented heuristic. Here the labels are morphology, so the
honest word is *accuracy* — :func:`czech_case_accuracy` says so, and the
constant :data:`LABEL_SOURCE_MORPHOLOGY_PCT` states the coverage this arm
exists to have (English gold morphology covered only the pronouns; Czech
covers every scored token, by construction and by a recorder guard).

What is excluded, and why
-------------------------
ADP tokens carry ``Case=`` in Czech PDT, but on a preposition the feature
records *government* — the case of its complement, not inflection of the token
itself. That is the structural analogue of the English token-level adposition
path whose margin decision 63 found reversed, so it is excluded from the population before
any model is fitted. The exclusion is recorded in the artifact's provenance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ailm.corpus.ud_czech import UD_CZECH_PDTC_SHARDS, validate_ud_czech_source_metadata
from ailm.levels.case import (
    CASES,
    N_CASES,
    baseline_case_positions,
    build_case_emission,
    build_case_start,
    build_case_transition,
    context_free_baseline,
    decode_case_positions,
)
from ailm.resources import resolve_vendored_artifact

__all__ = [
    "CZECH_ARTIFACT_SCHEMA",
    "CZECH_ADP_EXCLUDED_REASON",
    "CZECH_CORPUS_NAME",
    "CZECH_CLUSTER_HIERARCHY",
    "CZECH_DOCUMENT_IDENTITY_SOURCE",
    "CZECH_ENCODING",
    "CZECH_LABEL_SOURCE",
    "CZECH_LICENSE",
    "CZECH_MIN_SEQUENCE_LEN",
    "CZECH_NOMINAL_UPOS",
    "CZECH_OBSERVATION",
    "CZECH_SPLIT_SCOPE",
    "CZECH_STUDY_DESIGN_PROVENANCE",
    "CZECH_UD_CASE_VALUES",
    "LABEL_SOURCE_MORPHOLOGY_PCT",
    "CzechCaseArtifact",
    "baseline_case_positions",
    "build_case_emission",
    "build_case_start",
    "build_case_transition",
    "context_free_baseline",
    "czech_case_accuracy",
    "decode_case_positions",
    "load_czech_case_artifact",
    "validate_study_design_provenance",
]

CZECH_ARTIFACT_SCHEMA = "ailm-czech-case-counts-v3"

#: Static provenance values shared by the artifact writer and loader.  Keeping
#: one spelling is part of the artifact schema: a plausible-but-different corpus
#: description, observation contract, or split declaration must not be accepted
#: merely because the encoded arrays still have the expected shape.
CZECH_CORPUS_NAME = "UD Czech-PDTC (Prague Dependency Treebank - Consolidated)"
CZECH_LICENSE = "CC BY-NC-SA 4.0"
CZECH_SPLIT_SCOPE = "train-ft shard fits, dev is held out, test is not read"
CZECH_OBSERVATION = "UPOS tag only; every other CoNLL-U column withheld"
CZECH_LABEL_SOURCE = "UD morphological feats[Case] — ground truth, not a heuristic"
CZECH_ADP_EXCLUDED_REASON = (
    "Case= on a Czech preposition records government (the case of its complement), "
    "not inflection of the token — the analogue of the English token-level "
    "adposition path of decision 63"
)
CZECH_ENCODING = "token = upos_index * 8 + case_index"
CZECH_DOCUMENT_IDENTITY_SOURCE = "# newdoc id comments from the authenticated CoNLL-U shard"
CZECH_CLUSTER_HIERARCHY = "token < retained sequence < source document"

#: Nominal UPOS categories whose own forms inflect for case in this arm.  ADP
#: is intentionally absent: Czech ``Case=`` on a preposition describes the
#: governed complement rather than inflection of the observed token.
CZECH_NOMINAL_UPOS = frozenset({"ADJ", "DET", "NOUN", "NUM", "PRON", "PROPN"})

#: Every retained sequence must carry at least one case-to-case transition.
CZECH_MIN_SEQUENCE_LEN = 2

#: UD values in the exact :class:`CaseRole` order used by the shared builders.
CZECH_UD_CASE_VALUES = tuple(role.name.title() for role in CASES)

#: Honest study-design provenance for decision 77.  The population and endpoint
#: were specified prospectively within the development session, after feasibility
#: and development-set inspection, but no independently timestamped immutable
#: artifact predates the model score.  It is therefore not a preregistration.
CZECH_STUDY_DESIGN_PROVENANCE: Mapping[str, object] = MappingProxyType(
    {
        "protocol_status": "prospective-development-session-specification",
        "independently_timestamped": False,
        "immutable_prior_artifact": None,
    }
)

_STUDY_DESIGN_KEYS = frozenset(
    {"protocol_status", "independently_timestamped", "immutable_prior_artifact"}
)
_PRIOR_ARTIFACT_KEYS = frozenset({"path", "git_commit", "sha256", "committed_at"})
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def validate_study_design_provenance(provenance: object) -> dict[str, object]:
    """Validate the terminology/evidence contract for a study protocol.

    Calling a protocol ``preregistered`` is permitted only when the record
    names an independently timestamped, content-addressed artifact committed
    before measurement.  This makes the word depend on evidence rather than on
    an editable heading in ``docs/decisions.md``.  Decision 77 deliberately
    takes the other valid branch: a prospective development-session
    specification with no independent prior artifact.

    Returns:
        A plain validated copy suitable for deterministic JSON serialization.

    Raises:
        ValueError: If the object is malformed or its terminology outruns its
            immutable provenance.
    """
    if not isinstance(provenance, dict) or set(provenance) != _STUDY_DESIGN_KEYS:
        actual = sorted(provenance) if isinstance(provenance, dict) else type(provenance).__name__
        raise ValueError(
            "study_design must contain exactly protocol_status, "
            "independently_timestamped, and immutable_prior_artifact; "
            f"got {actual!r}"
        )
    status = provenance["protocol_status"]
    timestamped = provenance["independently_timestamped"]
    prior = provenance["immutable_prior_artifact"]
    if status not in {
        "prospective-development-session-specification",
        "preregistered",
    }:
        raise ValueError(f"unsupported study-design protocol_status {status!r}")
    if type(timestamped) is not bool:
        raise ValueError("study-design independently_timestamped must be a boolean")

    if status == "prospective-development-session-specification":
        if timestamped or prior is not None:
            raise ValueError(
                "a prospective development-session specification must declare "
                "independently_timestamped=false and immutable_prior_artifact=null"
            )
        return dict(provenance)

    if not timestamped or not isinstance(prior, dict) or set(prior) != _PRIOR_ARTIFACT_KEYS:
        raise ValueError(
            "preregistered terminology requires an independently timestamped "
            "immutable prior artifact with path, git_commit, sha256, and committed_at"
        )
    path = prior["path"]
    commit = prior["git_commit"]
    digest = prior["sha256"]
    committed_at = prior["committed_at"]
    if (
        not isinstance(path, str)
        or not path.strip()
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise ValueError(
            "immutable prior artifact path must be a non-empty repository-relative path"
        )
    if not isinstance(commit, str) or _HEX_40.fullmatch(commit) is None:
        raise ValueError("immutable prior artifact git_commit must be 40 lowercase hex characters")
    if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
        raise ValueError("immutable prior artifact sha256 must be 64 lowercase hex characters")
    if not isinstance(committed_at, str):
        raise ValueError("immutable prior artifact committed_at must be timezone-aware ISO text")
    try:
        parsed = datetime.fromisoformat(committed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "immutable prior artifact committed_at must be timezone-aware ISO text"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("immutable prior artifact committed_at must include a timezone")
    return {
        "protocol_status": status,
        "independently_timestamped": timestamped,
        "immutable_prior_artifact": dict(prior),
    }


#: Share of scored tokens whose label is morphological ``feats[Case]``. 100 by
#: construction — the artifact has no other label source — and asserted at load
#: time so a regenerated artifact that acquired one would refuse to load.
LABEL_SOURCE_MORPHOLOGY_PCT = 100.0

DEFAULT_ARTIFACT = resolve_vendored_artifact("ud_czech_case_counts.json")


class CzechCaseArtifact:
    """The vendored Czech case statistics, decoded to the English arm's surface.

    The artifact encodes each token as ``upos_index * 8 + case_index`` against
    its two recorded alphabets; this class decodes back to
    ``[(upos, case_name), ...]`` sequences so the builders in
    :mod:`ailm.levels.case` consume it unchanged.

    Attributes:
        train: Fitting sequences (the PDTC train-ft shard).
        test: Held-out sequences (the PDTC dev shard).
        upos_tags: Sorted observation alphabet.
        upos_to_idx: Observation index map.
        train_document_ids: Source-order document identity alphabet for train.
        test_document_ids: Source-order document identity alphabet for test.
        train_sequence_document_indices: Document index per training sequence.
        test_sequence_document_indices: Document index per held-out sequence.
        provenance: The generator's recorded parameters.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        expected_top_level = {
            "schema",
            "provenance",
            "upos_tags",
            "ud_case_values",
            "populations",
            "documents",
        }
        if not isinstance(payload, dict) or set(payload) != expected_top_level:
            actual = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
            raise ValueError(
                f"Czech artifact must contain exactly {sorted(expected_top_level)}, got {actual!r}"
            )
        schema = payload.get("schema")
        if schema != CZECH_ARTIFACT_SCHEMA:
            raise ValueError(
                f"Unexpected artifact schema {schema!r}; regenerate with "
                f"scripts/generate_czech_case_counts.py"
            )
        provenance = payload["provenance"]
        expected_provenance = {
            "corpus",
            "license",
            "raw_source",
            "split_scope",
            "observation",
            "label_source",
            "label_source_morphology_pct",
            "nominal_upos",
            "adp_excluded_reason",
            "min_sequence_len",
            "encoding",
            "document_identity_source",
            "cluster_hierarchy",
            "splits",
            "study_design",
        }
        if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
            actual_provenance: object = (
                sorted(provenance) if isinstance(provenance, dict) else type(provenance).__name__
            )
            raise ValueError(
                "Czech artifact provenance has a stale or widened surface: "
                f"expected {sorted(expected_provenance)}, got {actual_provenance!r}"
            )
        self.provenance = provenance
        expected_static_provenance: dict[str, object] = {
            "corpus": CZECH_CORPUS_NAME,
            "license": CZECH_LICENSE,
            "split_scope": CZECH_SPLIT_SCOPE,
            "observation": CZECH_OBSERVATION,
            "label_source": CZECH_LABEL_SOURCE,
            "adp_excluded_reason": CZECH_ADP_EXCLUDED_REASON,
            "document_identity_source": CZECH_DOCUMENT_IDENTITY_SOURCE,
            "cluster_hierarchy": CZECH_CLUSTER_HIERARCHY,
        }
        drifted_static = {
            field: (self.provenance[field], expected)
            for field, expected in expected_static_provenance.items()
            if self.provenance[field] != expected
        }
        if drifted_static:
            raise ValueError(
                "Czech artifact static provenance disagrees with the exact loader "
                f"contract: {drifted_static}"
            )
        raw_source = self.provenance.get("raw_source")
        if not isinstance(raw_source, dict):
            raise ValueError("the Czech artifact lacks a raw_source provenance object")
        validate_ud_czech_source_metadata(raw_source)
        morphology_pct = self.provenance.get("label_source_morphology_pct")
        if type(morphology_pct) is not float or morphology_pct != LABEL_SOURCE_MORPHOLOGY_PCT:
            raise ValueError(
                "the Czech artifact declares a non-morphological label source; "
                "this arm exists to score against morphology only (decision 77)"
            )
        study_design = validate_study_design_provenance(self.provenance["study_design"])
        if study_design != dict(CZECH_STUDY_DESIGN_PROVENANCE):
            raise ValueError(
                "Czech artifact study-design provenance does not match decision 77's "
                "development-session specification status"
            )
        min_sequence_len = self.provenance.get("min_sequence_len")
        if type(min_sequence_len) is not int or min_sequence_len != CZECH_MIN_SEQUENCE_LEN:
            raise ValueError(f"Czech artifact min_sequence_len must be {CZECH_MIN_SEQUENCE_LEN}")
        if self.provenance.get("encoding") != CZECH_ENCODING:
            raise ValueError("Czech artifact encoding contract is absent or changed")

        upos_tags = payload["upos_tags"]
        if (
            not isinstance(upos_tags, list)
            or not upos_tags
            or any(not isinstance(tag, str) or not tag for tag in upos_tags)
            or upos_tags != sorted(upos_tags)
            or len(upos_tags) != len(set(upos_tags))
        ):
            raise ValueError(
                "Czech artifact upos_tags must be a non-empty sorted unique string list"
            )
        if not set(upos_tags) <= CZECH_NOMINAL_UPOS:
            raise ValueError(
                "Czech artifact upos_tags contain a non-nominal observation: "
                f"{sorted(set(upos_tags) - CZECH_NOMINAL_UPOS)}"
            )
        if self.provenance.get("nominal_upos") != sorted(CZECH_NOMINAL_UPOS):
            raise ValueError(
                "Czech artifact nominal_upos provenance disagrees with the loader contract"
            )
        self.upos_tags = list(upos_tags)
        self.upos_to_idx: dict[str, int] = {u: i for i, u in enumerate(self.upos_tags)}
        case_values = payload["ud_case_values"]
        if not isinstance(case_values, list) or any(not isinstance(v, str) for v in case_values):
            raise ValueError("Czech artifact ud_case_values must be a string list")
        if tuple(case_values) != CZECH_UD_CASE_VALUES:
            raise ValueError(
                f"artifact case order {case_values} does not match CaseRole; "
                f"every decoded index would denote the wrong role"
            )
        case_names = [c.name for c in CASES]

        populations = payload["populations"]
        if not isinstance(populations, dict) or set(populations) != {"train", "test"}:
            actual_populations: object = (
                sorted(populations) if isinstance(populations, dict) else type(populations).__name__
            )
            raise ValueError(
                "Czech artifact populations must contain exactly train and test; "
                f"got {actual_populations!r}"
            )
        split_metadata = self.provenance.get("splits")
        if not isinstance(split_metadata, dict) or set(split_metadata) != {"train", "test"}:
            raise ValueError("Czech artifact provenance.splits must contain exactly train and test")
        documents = payload["documents"]
        if not isinstance(documents, dict) or set(documents) != {"train", "test"}:
            raise ValueError("Czech artifact documents must contain exactly train and test")
        self.document_ids: dict[str, list[str]] = {}
        self.sequence_document_indices: dict[str, list[int]] = {}

        def decode(split: str, sequences: object) -> list[list[tuple[str, str]]]:
            if not isinstance(sequences, list) or not sequences:
                raise ValueError(f"Czech artifact {split} split must be a non-empty sequence list")
            decoded: list[list[tuple[str, str]]] = []
            limit = len(self.upos_tags) * N_CASES
            for sequence_index, sequence in enumerate(sequences):
                if not isinstance(sequence, list) or len(sequence) < CZECH_MIN_SEQUENCE_LEN:
                    raise ValueError(
                        f"Czech artifact {split} sequence {sequence_index} must be a list "
                        f"of at least {CZECH_MIN_SEQUENCE_LEN} encoded tokens"
                    )
                pairs: list[tuple[str, str]] = []
                for token_index, code in enumerate(sequence):
                    if type(code) is not int or not 0 <= code < limit:
                        raise ValueError(
                            f"Czech artifact {split} sequence {sequence_index} token "
                            f"{token_index} has invalid encoded value {code!r}; expected "
                            f"an integer in [0, {limit})"
                        )
                    upos_index, case_index = divmod(code, N_CASES)
                    pairs.append((self.upos_tags[upos_index], case_names[case_index]))
                decoded.append(pairs)
            metadata = split_metadata[split]
            expected_split_keys = {
                "file",
                "sentences",
                "documents",
                "tokens",
                "case_marked_tokens",
                "retained_tokens",
                "retained_sequences",
                "retained_documents",
            }
            if not isinstance(metadata, dict) or set(metadata) != expected_split_keys:
                raise ValueError(
                    f"Czech artifact provenance.splits.{split} must contain exactly "
                    f"{sorted(expected_split_keys)}"
                )
            if metadata["file"] != UD_CZECH_PDTC_SHARDS[split].filename:
                raise ValueError(f"Czech artifact {split} split names the wrong source file")
            count_names = expected_split_keys - {"file"}
            if any(type(metadata[name]) is not int or metadata[name] < 0 for name in count_names):
                raise ValueError(
                    f"Czech artifact {split} split counts must be non-negative integers"
                )
            retained_tokens = sum(len(sequence) for sequence in decoded)
            document_mapping = documents[split]
            expected_document_keys = {"ids", "sequence_document_indices"}
            if not isinstance(document_mapping, dict) or set(document_mapping) != (
                expected_document_keys
            ):
                raise ValueError(
                    f"Czech artifact documents.{split} must contain exactly "
                    f"{sorted(expected_document_keys)}"
                )
            document_ids = document_mapping["ids"]
            sequence_document_indices = document_mapping["sequence_document_indices"]
            if (
                not isinstance(document_ids, list)
                or not document_ids
                or any(
                    not isinstance(document_id, str)
                    or not document_id
                    or document_id != document_id.strip()
                    for document_id in document_ids
                )
                or len(document_ids) != len(set(document_ids))
            ):
                raise ValueError(
                    f"Czech artifact documents.{split}.ids must be non-empty, "
                    "unique, stripped strings"
                )
            if (
                not isinstance(sequence_document_indices, list)
                or len(sequence_document_indices) != len(decoded)
                or any(
                    type(index) is not int or not 0 <= index < len(document_ids)
                    for index in sequence_document_indices
                )
            ):
                raise ValueError(
                    f"Czech artifact documents.{split}.sequence_document_indices must "
                    "give one in-range exact integer per retained sequence"
                )
            expected_indices = list(range(len(document_ids)))
            if sorted(set(sequence_document_indices)) != expected_indices:
                raise ValueError(
                    f"Czech artifact documents.{split} contains an unused or absent "
                    "document identity"
                )
            if list(dict.fromkeys(sequence_document_indices)) != expected_indices:
                raise ValueError(
                    f"Czech artifact documents.{split}.ids must follow first source use"
                )
            if len(document_ids) >= len(decoded):
                raise ValueError(
                    f"Czech artifact {split} has no outer document cluster: "
                    "per-sequence identities cannot serve as the primary sampling unit"
                )
            if (
                metadata["retained_sequences"] != len(decoded)
                or metadata["retained_tokens"] != retained_tokens
                or metadata["retained_documents"] != len(document_ids)
            ):
                raise ValueError(
                    f"Czech artifact {split} retained counts disagree with the encoded population"
                )
            if not (
                metadata["sentences"] >= metadata["retained_sequences"]
                and metadata["sentences"] >= metadata["documents"]
                and metadata["documents"] >= metadata["retained_documents"]
                and metadata["retained_sequences"] > metadata["retained_documents"]
                and metadata["tokens"] >= metadata["case_marked_tokens"]
                and metadata["case_marked_tokens"] >= metadata["retained_tokens"]
            ):
                raise ValueError(
                    f"Czech artifact {split} provenance counts are internally impossible"
                )
            self.document_ids[split] = list(document_ids)
            self.sequence_document_indices[split] = list(sequence_document_indices)
            return decoded

        self.train = decode("train", populations["train"])
        self.test = decode("test", populations["test"])
        self.train_document_ids = self.document_ids["train"]
        self.test_document_ids = self.document_ids["test"]
        self.train_sequence_document_indices = self.sequence_document_indices["train"]
        self.test_sequence_document_indices = self.sequence_document_indices["test"]
        observed_upos = {upos for sequence in self.train + self.test for upos, _case in sequence}
        if observed_upos != set(self.upos_tags):
            raise ValueError(
                "Czech artifact upos_tags must exactly equal the observations used by its populations"
            )

    @property
    def n_obs(self) -> int:
        """Observation alphabet size."""
        return len(self.upos_tags)

    @property
    def n_test_tokens(self) -> int:
        """Held-out labeled tokens — the *n* every reported score needs."""
        return sum(len(seq) for seq in self.test)

    @property
    def n_test_documents(self) -> int:
        """Contributing held-out ``# newdoc`` units behind the primary interval."""
        return len(self.test_document_ids)

    @property
    def cases_present(self) -> set[str]:
        """Case names that actually occur across both splits."""
        return {c for seq in self.train + self.test for _, c in seq}


def load_czech_case_artifact(path: Path | str | None = None) -> CzechCaseArtifact:
    """Load and validate the vendored Czech case artifact.

    Args:
        path: Artifact location; defaults to ``vendor/ud_czech_case_counts.json``.

    Returns:
        The validated artifact.

    Raises:
        FileNotFoundError: If the artifact is absent.
        ValueError: If it fails validation.
    """
    resolved = (
        DEFAULT_ARTIFACT
        if path is None
        else resolve_vendored_artifact("ud_czech_case_counts.json", path=path)
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{resolved} is absent. Run scripts/generate_czech_case_counts.py "
            f"(needs the staged UD Czech-PDTC shards)."
        )
    return CzechCaseArtifact(json.loads(resolved.read_text(encoding="utf-8")))


def czech_case_accuracy(hits: list[bool]) -> float:
    """Fraction of held-out positions matching morphological gold.

    Named *accuracy* deliberately, in contrast with
    :func:`ailm.levels.case.case_agreement`: the Czech labels are UD
    morphology, not a heuristic, so this number is the thing the English arm
    could never report (decision 77).

    Raises:
        ValueError: If ``hits`` is empty — an accuracy over no positions is not
            a small number, it is undefined.
    """
    if not hits:
        raise ValueError("no positions scored; accuracy is undefined")
    return sum(hits) / len(hits)
