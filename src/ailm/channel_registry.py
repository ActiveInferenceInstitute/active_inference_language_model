"""Canonical inventory of the receiver interventions reported by AILM.

The registry distinguishes model-level outputs from privileged-oracle senders,
and measured interventions from the two registered case-word structural seams
(the L4→L3 model-sender bridge and its privileged-oracle diagnostic), whose
evidence is structural: their evaluation records nothing, so they carry a
declared prior-only insertion site rather than a scored estimand. The
interventions are experiments layered on top of independently scored levels,
not edges along one live inference cascade. Keeping source kind, identity, and
expected evidence class in one typed registry prevents manuscript counts,
figures, and semantic tests from each maintaining a different or scientifically
inflated list.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence


class ChannelEvidenceRule(StrEnum):
    """Recorded estimand used to classify one channel."""

    GENERATION = "generation"
    LEXICAL = "lexical"
    LETTER_COUPLING = "letter-coupling"
    NARRATIVE_PRIOR = "narrative-prior"
    CLAUSE_WORD = "clause-word"
    CASE_WORD_BRIDGE = "case-word-bridge"
    CASE_WORD_ORACLE_DIAGNOSTIC = "case-word-oracle-diagnostic"


class ChannelOutcome(StrEnum):
    """Reader-facing evidence class supported by the recorded estimand."""

    HELPS = "helps"
    HARMS = "harms"
    PRIOR_ONLY = "prior-only"
    MIXED_DIRECTION = "mixed-direction"


class ChannelSourceKind(StrEnum):
    """Whether the sender value is model output or a privileged reference."""

    MODEL_OUTPUT = "model-output"
    PRIVILEGED_ORACLE = "privileged-oracle"


@dataclass(frozen=True, slots=True)
class CrossLevelChannel:
    """One independently measured source-to-receiver intervention."""

    key: str
    sender: str
    receiver: str
    insertion: str
    evidence_rule: ChannelEvidenceRule
    expected_outcome: ChannelOutcome
    comprehension_only: bool
    source_kind: ChannelSourceKind = ChannelSourceKind.MODEL_OUTPUT


CROSS_LEVEL_CHANNELS: tuple[CrossLevelChannel, ...] = (
    CrossLevelChannel(
        key="letter_coupling",
        sender="L1",
        receiver="L2B",
        insertion="observation likelihood",
        evidence_rule=ChannelEvidenceRule.LETTER_COUPLING,
        expected_outcome=ChannelOutcome.HARMS,
        comprehension_only=True,
    ),
    CrossLevelChannel(
        key="lexical",
        sender="L3",
        receiver="L2B",
        insertion="phone predictive",
        evidence_rule=ChannelEvidenceRule.LEXICAL,
        expected_outcome=ChannelOutcome.HELPS,
        comprehension_only=True,
    ),
    CrossLevelChannel(
        key="generation",
        sender="L3",
        receiver="L1g",
        insertion="preference C",
        evidence_rule=ChannelEvidenceRule.GENERATION,
        expected_outcome=ChannelOutcome.HELPS,
        comprehension_only=False,
    ),
    CrossLevelChannel(
        key="narrative_prior",
        sender="L6",
        receiver="L5",
        insertion="initial prior D",
        evidence_rule=ChannelEvidenceRule.NARRATIVE_PRIOR,
        expected_outcome=ChannelOutcome.PRIOR_ONLY,
        comprehension_only=True,
    ),
    CrossLevelChannel(
        key="clause_word",
        sender="punctuation-type oracle",
        receiver="L3",
        insertion="word prior D",
        evidence_rule=ChannelEvidenceRule.CLAUSE_WORD,
        expected_outcome=ChannelOutcome.HELPS,
        comprehension_only=True,
        source_kind=ChannelSourceKind.PRIVILEGED_ORACLE,
    ),
    CrossLevelChannel(
        key="case_word_bridge",
        sender="L4",
        receiver="L3",
        insertion="word prior D",
        evidence_rule=ChannelEvidenceRule.CASE_WORD_BRIDGE,
        expected_outcome=ChannelOutcome.PRIOR_ONLY,
        comprehension_only=True,
    ),
    CrossLevelChannel(
        key="case_word_oracle_diagnostic",
        sender="case-oracle (train routes, dev)",
        receiver="L3",
        insertion="word prior D",
        evidence_rule=ChannelEvidenceRule.CASE_WORD_ORACLE_DIAGNOSTIC,
        expected_outcome=ChannelOutcome.PRIOR_ONLY,
        comprehension_only=True,
        source_kind=ChannelSourceKind.PRIVILEGED_ORACLE,
    ),
)


def channel_by_key(
    channels: Sequence[CrossLevelChannel] = CROSS_LEVEL_CHANNELS,
) -> dict[str, CrossLevelChannel]:
    """Return the registry keyed by stable channel identifier."""
    entries = tuple(channels)
    if not entries:
        raise ValueError("cross-level channel registry cannot be empty")
    keyed = {channel.key: channel for channel in entries}
    if len(keyed) != len(entries):
        raise ValueError("cross-level channel registry contains duplicate keys")
    rules = [channel.evidence_rule for channel in entries]
    if len(set(rules)) != len(rules):
        raise ValueError("cross-level channel registry contains duplicate evidence rules")
    if any(
        not value.strip()
        for channel in entries
        for value in (channel.key, channel.sender, channel.receiver, channel.insertion)
    ):
        raise ValueError("cross-level channel registry contains an empty identity field")
    oracle_channels = [
        channel for channel in entries if channel.source_kind is ChannelSourceKind.PRIVILEGED_ORACLE
    ]
    if any(channel.sender.startswith("L") for channel in oracle_channels):
        raise ValueError("a privileged-oracle intervention cannot claim a model level as sender")
    bridge = keyed.get("case_word_bridge")
    if bridge is not None and (
        bridge.source_kind is not ChannelSourceKind.MODEL_OUTPUT
        or bridge.sender != "L4"
        or bridge.receiver != "L3"
        or not bridge.comprehension_only
    ):
        raise ValueError(
            "case_word_bridge is a model-level sender whose runtime message "
            "marginalizes the L4 role posterior into the L3 word prior; it "
            "must be declared MODEL_OUTPUT from L4 to L3, comprehension-only"
        )
    oracle_diagnostic = keyed.get("case_word_oracle_diagnostic")
    if oracle_diagnostic is not None and (
        oracle_diagnostic.source_kind is not ChannelSourceKind.PRIVILEGED_ORACLE
        or not oracle_diagnostic.sender.startswith("case-oracle")
        or oracle_diagnostic.receiver != "L3"
        or not oracle_diagnostic.comprehension_only
    ):
        raise ValueError(
            "case_word_oracle_diagnostic is the separately labeled privileged "
            "diagnostic arm; it can never replace the model-sender bridge"
        )
    clause_word = keyed.get("clause_word")
    if clause_word is not None and (
        clause_word.source_kind is not ChannelSourceKind.PRIVILEGED_ORACLE
        or clause_word.sender != "punctuation-type oracle"
    ):
        raise ValueError(
            "clause_word consumes the punctuation-defined reference type, not L5 output"
        )
    return keyed
