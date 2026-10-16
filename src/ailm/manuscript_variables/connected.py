"""Reconcile the connected GUM result without assuming a favorable direction."""

from __future__ import annotations

import re

from ailm.corpus.connected_gum import EXCLUDED_DOCUMENTS, PROTOCOL_SHA256, TEST_SHA256
from ailm.levels.connected_evaluation import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONNECTED_STRENGTH,
)
from ailm.value_formatting import ManuscriptVariableError

from ._claims import _ClaimReaders, require_close
from ._io import DIGEST_SHORT_LENGTH, _StrictVariableMap


def _produce_connected_gum(variables: _StrictVariableMap, readers: _ClaimReaders) -> None:
    """Require coherent counts, paired effect, protocol and OOV disclosure."""
    prefix = "connected_gum_"
    readers.require_namespace_provenance(
        prefix,
        source="scripts/record_connected_gum_claims.py",
        run_id="connected_gum_unexamined_v1",
    )
    counts = {}
    for suffix in (
        "documents",
        "excluded_documents",
        "source_sentences",
        "source_characters",
        "normalized_tokens",
        "oov_tokens",
        "in_vocab_tokens",
        "normalized_letters",
        "scored_characters",
    ):
        value = readers.num(prefix + suffix)
        if value < 0 or value != int(value):
            raise ManuscriptVariableError(f"connected GUM count {suffix} is invalid")
        counts[suffix] = int(value)
    if counts["documents"] < 2 or counts["normalized_tokens"] == 0:
        raise ManuscriptVariableError("connected GUM requires a nonempty paired population")
    if counts["in_vocab_tokens"] + counts["oov_tokens"] != counts["normalized_tokens"]:
        raise ManuscriptVariableError(
            "connected GUM token coverage does not partition the population"
        )
    if counts["scored_characters"] != counts["normalized_letters"] + counts["normalized_tokens"]:
        raise ManuscriptVariableError("connected GUM character denominator omits token spaces")
    require_close(
        prefix + "vocab_coverage_pct",
        readers.percentage(prefix + "vocab_coverage_pct"),
        100 * counts["in_vocab_tokens"] / counts["normalized_tokens"],
    )
    for suffix, expected in (
        ("excluded_documents", len(EXCLUDED_DOCUMENTS)),
        ("strength", CONNECTED_STRENGTH),
        ("bootstrap_seed", BOOTSTRAP_SEED),
        ("bootstrap_resamples", BOOTSTRAP_RESAMPLES),
        ("bootstrap_confidence_pct", BOOTSTRAP_CONFIDENCE * 100),
        ("corpus_sha256", TEST_SHA256),
        ("protocol_sha256", PROTOCOL_SHA256),
        ("word_loss_is_infinite", counts["oov_tokens"] > 0),
    ):
        readers.require_exact_claim(prefix + suffix, expected)
    integrated = readers.num(prefix + "document_integrated_log_loss")
    disconnected = readers.num(prefix + "document_disconnected_log_loss")
    if min(integrated, disconnected) < 0:
        raise ManuscriptVariableError("connected GUM log loss cannot be negative")
    require_close(
        prefix + "document_mean_gain",
        readers.num(prefix + "document_mean_gain"),
        disconnected - integrated,
    )
    low, high = readers.num(prefix + "document_ci_low"), readers.num(prefix + "document_ci_high")
    if low > high:
        raise ManuscriptVariableError("connected GUM interval endpoints are reversed")
    outcome = (
        "integrated_lower_loss"
        if low > 0
        else "integrated_higher_loss"
        if high < 0
        else "unresolved"
    )
    readers.require_exact_claim(prefix + "outcome", outcome)
    variables["CONNECTED_GUM_OUTCOME_TEXT"] = {
        "integrated_lower_loss": "lower integrated loss",
        "integrated_higher_loss": "higher integrated loss",
        "unresolved": "an interval spanning zero",
    }[outcome]
    variables["CONNECTED_GUM_WORD_LOSS_TEXT"] = (
        "infinite because out-of-vocabulary tokens receive zero word probability"
        if counts["oov_tokens"]
        else "finite on this closed-vocabulary population"
    )
    for suffix in (
        "corpus_sha256",
        "protocol_sha256",
        "execution_sha256",
        "runtime_sha256",
        "report_sha256",
    ):
        digest = readers.raw(prefix + suffix)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ManuscriptVariableError(f"connected GUM {suffix} is not a SHA-256 digest")
        variables[(prefix + suffix + "_short").upper()] = digest[:DIGEST_SHORT_LENGTH]
