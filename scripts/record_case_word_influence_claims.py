#!/usr/bin/env python3
"""Record the L4→L3 case-word influence sweep (decision entry 121).

Protocol P3 partition 3's causal-influence registration, executed: the
model-sender bridge (`src/ailm/case_word_channel.py`) marginalizes the case
level's own per-UPOS role posterior into a word message, the message enters
the L3 word prior through `apply_case_word_inlet`, and this recorder scores
the L3 word level's own per-token log-probability arithmetic over the
case-word artifact's test rows under every weight of the shipped alpha grid —
mirroring `scripts/record_clause_channel_claims.py`'s receiver math, with the
per-token UPOS observation in place of the per-sentence clause type.

The estimand, fixed before any run by decision entry 121: the per-token
log-gain of the mixed prior against the bit-identical alpha = 0 anchor, over
the artifact's `test` documents (`CaseWordArtifact.test_documents`) restricted
to lowercase exact-string matches against the Stage 3 receiver vocabulary, in
document order, with token direction counts and a whole-document clustered
bootstrap (`clustered_clause_word_summaries`).

The sweep loop lives here rather than in
`ailm.levels.clause_channel.sweep_clause_alpha` because the case-word message
conditions on each token's own UPOS observation (not on a sentence-level
type), and the no-message policy must skip the mixing arithmetic entirely to
keep `D_alpha = D_0` bit-identical — a `ClauseMessage` lookup can represent
neither. Everything else is the shipped machinery, unchanged: the grid,
`mix_prior` (via the inlet), `scramble_word_distribution`, and
`clustered_clause_word_summaries`.

Controls and refusals, all registered in entry 121 and checked before the
ledger is opened: the grid guard; Stage 3 vocabulary-order alignment; the
fitted sender's supported roles against the artifact's declared
`sender_support`; anchor bit-identity at alpha = 0; anchor non-vacuity; the
no-message policy (a messageless token reproduces the anchor at every alpha);
seam normalization on every mixed prior the inlet produces; and the declared
positive control — the evaluation stream's own pooled in-vocabulary unigram,
sent wherever the model-sender arm sends — must lead the scrambled-word
negative control by the pre-registered per-token margin derived offline from
the frozen artifacts (`POSITIVE_CONTROL_MARGIN`; derivation in entry 121).
Any failed gate exits via `SystemExit` before the ledger is touched: no
claims, no partial runs.

Scope, declared: development evidence under a fresh registration — never
confirmatory (the scored documents are the already-scored EWT dev population
in a new representation); cross-corpus transfer (an EWT-estimated message on
the Stage 3 Brown-receiver vocabulary) is part of the estimand, named as
such; the recorded receiver sensitivities of the other arms bound nothing
here; the case-word registry entries are registration-only structure, never
narrated as an executed cascade.

Usage:
    .venv/bin/python scripts/record_case_word_influence_claims.py

Isolated ledger for validation (never the tracked ledger):
    AILM_CLAIMS_PATH=/tmp/case_word_sweep_claims.json .venv/bin/python \\
        scripts/record_case_word_influence_claims.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.case_word_channel import (  # noqa: E402
    apply_case_word_inlet,
    build_message,
    fit_sender,
    role_posterior_from_agent,
)
from ailm.cases import CaseRole  # noqa: E402
from ailm.claims import ClaimBatch, load_ledger  # noqa: E402
from ailm.corpus.case_word import load_case_word_artifact  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.case import N_CASES, build_case_agent, load_case_artifact  # noqa: E402
from ailm.levels.clause_channel import (  # noqa: E402
    clustered_clause_word_summaries,
    scramble_word_distribution,
)
from ailm.levels.word import build_word_prior, load_word_artifact  # noqa: E402

RUN_ID = "case_word_influence_sweep"
SOURCE = "scripts/record_case_word_influence_claims.py"

#: The shipped mixing grid, reused unchanged from the clause-channel arm
#: (`scripts/record_clause_channel_claims.py`): the 0.0 anchor plus six
#: distinct positive weights, so the grid guard cannot collapse.
GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

#: Log floor for a mixed prior entry, mirroring `sweep_clause_alpha`.
LOG_FLOOR = 1e-300

#: Seam-normalization tolerance asserted on every mixed prior the inlet
#: produces. Tighter than `clause_channel.MIX_NORMALIZATION_ATOL` (which
#: admits float32-sourced rounding at the 2000-type scale) because the inlet's
#: operands here are float64 end to end.
SEAM_NORMALIZATION_ATOL = 1e-9

#: The pre-registered positive-control margin, in per-token nats: the least
#: per-alpha gap (positive control minus scrambled control) derived offline
#: from the frozen artifacts before any run. The derivation and the per-alpha
#: values are stated in decision entry 121; the binding weight is alpha = 0.1.
POSITIVE_CONTROL_MARGIN = 0.177528463

#: Float slack on the margin gate. The margin is a deterministic function of
#: the frozen artifacts and the shipped builders, so a live run reproduces it
#: to rounding; the slack absorbs that rounding and nothing else.
MARGIN_TOLERANCE = 1e-6


class CaseWordChannel(NamedTuple):
    """Everything the sweep scores, artifact-bound.

    A `NamedTuple` rather than a dataclass: this module is imported by
    spec-based loaders (the recorder test's import helper) that do not
    register the module in `sys.modules` before executing it, which dataclass
    processing requires — the same reason
    `scripts/record_synthetic_benchmark_claims.py` uses one.

    Attributes:
        vocab: The Stage 3 receiver vocabulary in receiver index order.
        word_to_index: Vocabulary word to receiver index.
        prior: The receiver's L3 word prior (`build_word_prior`).
        messages: UPOS tag -> marginalized message, or `None` where the
            no-message policy fires (a UPOS outside the case level's
            observation alphabet has no posterior row to marginalize).
        rows: Per-document evaluation rows, `{"document_index", "words"}`, in
            the shape `clustered_clause_word_summaries` consumes.
        tokens: `(lowercase in-vocabulary word, UPOS)` per scored token, in
            document then token order.
        positive_control: The declared positive control's message — the
            evaluation stream's own pooled in-vocabulary unigram.
        sender_roles: Number of roles the fitted sender supports.
    """

    vocab: list[str]
    word_to_index: dict[str, int]
    prior: np.ndarray
    messages: dict[str, np.ndarray | None]
    rows: list[dict[str, object]]
    tokens: list[tuple[str, str]]
    positive_control: np.ndarray
    sender_roles: int


def _refuse(message: str) -> None:
    """Refuse the run before the ledger is opened: `SystemExit`, no claims."""
    raise SystemExit(f"REFUSING TO RECORD: {message} No claims written.")


def _grid_guard(grid: tuple[float, ...]) -> None:
    """Entry 9's grid guard: a collapsed sweep cannot support a finding."""
    positives = {alpha for alpha in grid if alpha > 0.0}
    if 0.0 not in grid or len(positives) < 3:
        _refuse(f"grid {grid} collapsed; cannot support a finding")


def _load_channel(
    case_word_path: Path | str | None = None,
    word_path: Path | str | None = None,
    case_path: Path | str | None = None,
) -> CaseWordChannel:
    """Bind the frozen artifacts into the channel the sweep scores.

    The artifact-level refusals live here: vocabulary-order alignment with
    the Stage 3 receiver, sender support against the artifact's declared
    `sender_support`, a non-empty evaluation stream, and the two-document
    minimum the clustered bootstrap requires.
    """
    case_word = load_case_word_artifact(case_word_path)
    word_artifact = load_word_artifact(word_path)
    if list(word_artifact.vocab) != case_word.vocab:
        _refuse(
            "the case-word artifact's vocabulary order does not match the "
            "Stage 3 receiver's; regenerate the artifact"
        )
    prior = build_word_prior(word_artifact)

    case_artifact = load_case_artifact(path=case_path)
    agent = build_case_agent(case_artifact)
    posterior = role_posterior_from_agent(agent, n_upos=case_artifact.n_obs, n_cases=N_CASES)
    role_names = [role.name for role in CaseRole]
    sender = fit_sender(case_word.sender_map, case_word.vocab)
    if set(sender) != set(case_word.sender_support):
        _refuse(
            "the fitted sender's supported roles disagree with the artifact's "
            "declared sender_support; the zero-support policy would silently "
            "change the estimand"
        )

    # The evaluation stream: the test split's token rows in document order,
    # lowercased, kept when the word form is an exact-string match against
    # the Stage 3 vocabulary (an out-of-vocabulary word has no L3 state).
    tokens: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []
    for document in case_word.test_documents:
        pairs = [
            (row.word_form.lower(), row.upos)
            for row in document["rows"]
            if row.word_form.lower() in case_word.word_to_index
        ]
        rows.append(
            {
                "document_index": document["document_index"],
                "words": [word for word, _upos in pairs],
            }
        )
        tokens.extend(pairs)
    if not tokens:
        _refuse("the held-out token stream is empty")
    if len(rows) < 2:
        _refuse("the whole-document clustered bootstrap needs at least two documents")

    # Per-UPOS messages. A UPOS outside the case level's observation alphabet
    # has no posterior row to marginalize and therefore sends no message —
    # the declared no-message policy `D_alpha = D_0`, not an exclusion of the
    # token from the scored stream.
    messages: dict[str, np.ndarray | None] = {}
    for upos in {upos for _word, upos in tokens}:
        index = case_artifact.upos_to_idx.get(upos)
        messages[upos] = (
            None if index is None else build_message(posterior[index], role_names, sender)
        )

    # The declared positive control: the evaluation stream's own pooled
    # in-vocabulary unigram — the strongest role-free sender the vocabulary
    # admits, constructed to help. It is a sender swap, so the channel's
    # message-availability policy is unchanged (see `_positive_control_sender`).
    counts = np.zeros(len(case_word.vocab), dtype=np.float64)
    for word, _upos in tokens:
        counts[case_word.word_to_index[word]] += 1.0
    positive_control = counts / counts.sum()
    return CaseWordChannel(
        vocab=list(case_word.vocab),
        word_to_index=dict(case_word.word_to_index),
        prior=prior,
        messages=messages,
        rows=rows,
        tokens=tokens,
        positive_control=positive_control,
        sender_roles=len(sender),
    )


def _positive_control_sender(
    channel: CaseWordChannel,
) -> Callable[[str], np.ndarray | None]:
    """The positive control as a sender swap over the channel's own policy.

    Wherever the model-sender arm sends, the positive control sends its
    pooled unigram; wherever the no-message policy fires, it fires for the
    control too — the control replaces the sender, not the channel.
    """

    def message_of(upos: str) -> np.ndarray | None:
        return None if channel.messages[upos] is None else channel.positive_control

    return message_of


def _sweep(
    channel: CaseWordChannel,
    message_of: (Callable[[str], np.ndarray | None] | dict[str, np.ndarray | None] | None) = None,
) -> dict[float, list[float]]:
    """Per-token log-probability of the held-out word under the mixed prior.

    Returns a dict `alpha -> [log P_alpha(token)]` aligned position-for-
    position across alphas, so any two weights pair exactly — the clause
    arm's receiver arithmetic (`sweep_clause_alpha`) with the per-token
    UPOS-conditioned message in place of the sentence type, and the mixing
    performed by `apply_case_word_inlet` so the builder's own seam and
    no-message policies are the ones exercised.

    `message_of` substitutes a control's sender for the model sender's — a
    per-UPOS message mapping or a callable returning one. The
    no-message policy is structural: a `None` message returns the receiver
    prior untouched (the inlet performs no arithmetic at all on that path),
    so the token's value is the anchor's, bit for bit, at every alpha. Every
    mixed prior is checked against the seam-normalization contract.
    """
    if message_of is None:
        messages = channel.messages
    elif isinstance(message_of, dict):
        messages = message_of
    else:
        messages = {upos: message_of(upos) for upos in channel.messages}
    results: dict[float, list[float]] = {alpha: [] for alpha in GRID}
    for word, upos in channel.tokens:
        index = channel.word_to_index[word]
        message = messages[upos]
        for alpha in GRID:
            mixed = apply_case_word_inlet(channel.prior, message, alpha)
            total = float(np.sum(mixed))
            if abs(total - 1.0) > SEAM_NORMALIZATION_ATOL:
                _refuse(
                    f"the inlet produced an unnormalized prior (sum {total!r}) "
                    f"at alpha {alpha}; the seam normalization contract is broken"
                )
            results[alpha].append(float(np.log(max(mixed[index], LOG_FLOOR))))
    return results


def _anchor_identity_gate(channel: CaseWordChannel, real: dict[float, list[float]]) -> None:
    """The alpha = 0 anchor is bit-identical to the receiver prior's own
    per-token log-probability, and `mix_prior` at alpha = 0 returns the prior
    unchanged. Without this, every other weight compares against a different
    model."""
    for position, (word, _upos) in enumerate(channel.tokens):
        expected = float(np.log(max(channel.prior[channel.word_to_index[word]], LOG_FLOOR)))
        if real[0.0][position] != expected:
            _refuse(
                "the alpha=0 anchor is not bit-identical to the receiver "
                "prior's own log-probability at scored position "
                f"{position}; every other weight would be a comparison "
                "against a different model"
            )
    with_message = next(m for m in channel.messages.values() if m is not None)
    mixed_at_zero = apply_case_word_inlet(channel.prior, with_message, 0.0)
    if not np.array_equal(mixed_at_zero, channel.prior):
        _refuse(
            "mix_prior at alpha=0 is not bit-identical to the receiver prior; "
            "the identity anchor is broken"
        )


def _anchor_live_gate(real: dict[float, list[float]]) -> None:
    """A positive alpha must change the output, or the anchor is vacuous."""
    anchor = real[0.0]
    if all(real[alpha] == anchor for alpha in GRID if alpha > 0.0):
        _refuse(
            "every positive alpha produced the anchor's exact output; the "
            "anchor is vacuous and the sweep measures nothing"
        )


def _no_message_gate(channel: CaseWordChannel, real: dict[float, list[float]]) -> None:
    """A messageless token reproduces the anchor at every alpha — the
    deterministic `D_alpha = D_0` policy, verified on the scored stream."""
    anchor = real[0.0]
    for position, (_word, upos) in enumerate(channel.tokens):
        if channel.messages[upos] is not None:
            continue
        for alpha in GRID:
            if real[alpha][position] != anchor[position]:
                _refuse(
                    f"a messageless token changed under mixing at alpha "
                    f"{alpha}; the no-message policy D_alpha = D_0 is not live"
                )


def _positive_control_margin_gate(
    channel: CaseWordChannel,
    anchor: list[float],
    scrambled: dict[float, list[float]],
    positive: dict[float, list[float]],
) -> None:
    """The declared positive control must lead the scrambled-word negative
    control by the pre-registered margin at every positive alpha.

    The margin is `POSITIVE_CONTROL_MARGIN`, derived offline from the frozen
    artifacts before any run (decision entry 121 states the derivation). A
    live run that misses it means the receiver arithmetic or the channel is
    broken, and no claim may be recorded.
    """
    n_tokens = len(channel.tokens)
    for alpha in GRID:
        if alpha <= 0.0:
            continue
        positive_gain = sum(p - a for p, a in zip(positive[alpha], anchor, strict=True))
        scrambled_gain = sum(s - a for s, a in zip(scrambled[alpha], anchor, strict=True))
        gap = (positive_gain - scrambled_gain) / n_tokens
        if gap < POSITIVE_CONTROL_MARGIN - MARGIN_TOLERANCE:
            _refuse(
                f"at alpha={alpha} the declared positive control leads the "
                f"scrambled-word control by {gap:.9f} nats per token, below "
                f"the pre-registered margin {POSITIVE_CONTROL_MARGIN} "
                "(decision entry 121); the receiver arithmetic or the "
                "channel is broken"
            )


def _mean_log_gain(stream: list[float], anchor: list[float]) -> float:
    """Mean per-token log-gain of `stream` against the anchor, in nats."""
    return float(sum(s - a for s, a in zip(stream, anchor, strict=True)) / len(anchor))


def main(
    argv: list[str] | None = None,
    *,
    channel: CaseWordChannel | None = None,
) -> int:
    """Run the gates, sweep the grid, and record the `case_word_sweep_*`
    claims wholesale into the ledger named by `AILM_CLAIMS_PATH` (default:
    the tracked ledger). `channel` exists so tests can bind a small real
    fixture channel through the identical gate chain; production runs bind
    the frozen artifacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-word-artifact", type=Path, default=None)
    parser.add_argument("--word-artifact", type=Path, default=None)
    parser.add_argument("--case-artifact", type=Path, default=None)
    args = parser.parse_args(argv)

    _grid_guard(GRID)
    if channel is None:
        channel = _load_channel(args.case_word_artifact, args.word_artifact, args.case_artifact)

    real = _sweep(channel)
    anchor = real[0.0]
    _anchor_identity_gate(channel, real)
    _anchor_live_gate(real)
    _no_message_gate(channel, real)

    scrambled = _sweep(
        channel,
        {
            upos: None if message is None else scramble_word_distribution(message)
            for upos, message in channel.messages.items()
        },
    )
    positive = _sweep(channel, _positive_control_sender(channel))
    _positive_control_margin_gate(channel, anchor, scrambled, positive)

    # Whole-document clustered bootstrap at every weight: the scored tokens
    # nest inside EWT dev source documents, and treating them as independent
    # would let within-document repetition masquerade as replication.
    summaries: dict[float, object] = {}
    for alpha in GRID:
        _token_summary, loggain_summary = clustered_clause_word_summaries(
            channel.rows, real[alpha], anchor
        )
        summaries[alpha] = loggain_summary

    emissions = sum(len(word) + 1 for word, _upos in channel.tokens)
    anchor_ppl = float(np.exp(-sum(anchor) / emissions))
    no_message_tokens = sum(1 for _word, upos in channel.tokens if channel.messages[upos] is None)

    recorded: dict[str, float | int | str | bool] = {
        "case_word_sweep_scored_tokens": len(channel.tokens),
        "case_word_sweep_scored_emissions": emissions,
        "case_word_sweep_eval_documents": len(channel.rows),
        "case_word_sweep_vocab_types": len(channel.vocab),
        "case_word_sweep_sender_roles": channel.sender_roles,
        "case_word_sweep_no_message_upos": sum(
            1 for message in channel.messages.values() if message is None
        ),
        "case_word_sweep_no_message_tokens": no_message_tokens,
        "case_word_sweep_anchor_ppl": round(anchor_ppl, 6),
        "case_word_sweep_anchor_bit_identical": 1,
        "case_word_sweep_anchor_live": 1,
        "case_word_sweep_positive_control_margin_registered": POSITIVE_CONTROL_MARGIN,
    }

    best_alpha = max(GRID, key=lambda a: _mean_log_gain(real[a], anchor))
    for alpha in GRID:
        stream = np.asarray(real[alpha], dtype=np.float64)
        anchor_array = np.asarray(anchor, dtype=np.float64)
        difference = stream - anchor_array
        summary = summaries[alpha]
        scrambled_gain = _mean_log_gain(scrambled[alpha], anchor)
        positive_gain = _mean_log_gain(positive[alpha], anchor)
        gap = positive_gain - scrambled_gain
        suffix = str(alpha).replace(".", "_")
        recorded[f"case_word_sweep_alpha_{suffix}_mean_loggain"] = round(
            float(difference.mean()), 9
        )
        recorded[f"case_word_sweep_alpha_{suffix}_improved"] = int((difference > 0.0).sum())
        recorded[f"case_word_sweep_alpha_{suffix}_degraded"] = int((difference < 0.0).sum())
        recorded[f"case_word_sweep_alpha_{suffix}_tied"] = int((difference == 0.0).sum())
        recorded[f"case_word_sweep_alpha_{suffix}_scrambled_mean_loggain"] = round(
            scrambled_gain, 9
        )
        recorded[f"case_word_sweep_alpha_{suffix}_positive_control_mean_loggain"] = round(
            positive_gain, 9
        )
        recorded[f"case_word_sweep_alpha_{suffix}_positive_control_margin"] = round(gap, 9)
        recorded[f"case_word_sweep_alpha_{suffix}_margin_ok"] = int(
            alpha <= 0.0 or gap >= POSITIVE_CONTROL_MARGIN - MARGIN_TOLERANCE
        )
        recorded[f"case_word_sweep_alpha_{suffix}_cluster_loggain_effect"] = round(
            float(summary.effect), 9
        )
        recorded[f"case_word_sweep_alpha_{suffix}_cluster_loggain_ci_low"] = round(
            float(summary.ci_low), 9
        )
        recorded[f"case_word_sweep_alpha_{suffix}_cluster_loggain_ci_high"] = round(
            float(summary.ci_high), 9
        )
        recorded[f"case_word_sweep_alpha_{suffix}_cluster_loggain_observations"] = int(
            summary.n_observations
        )
        recorded[f"case_word_sweep_alpha_{suffix}_cluster_loggain_positive_documents"] = int(
            summary.positive_clusters
        )
        recorded[f"case_word_sweep_alpha_{suffix}_cluster_loggain_negative_documents"] = int(
            summary.negative_clusters
        )

    best_summary = summaries[best_alpha]
    recorded["case_word_sweep_best_alpha"] = best_alpha
    recorded["case_word_sweep_best_mean_loggain"] = round(
        _mean_log_gain(real[best_alpha], anchor), 9
    )
    recorded["case_word_sweep_cluster_documents"] = int(best_summary.n_clusters)
    recorded["case_word_sweep_cluster_resamples"] = int(best_summary.resamples)
    recorded["case_word_sweep_cluster_seed"] = int(best_summary.seed)
    recorded["case_word_sweep_cluster_confidence_pct"] = 100.0 * float(best_summary.confidence)

    ledger = load_ledger(_PROJECT_ROOT)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in recorded]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=recorded, run_id=RUN_ID, source=SOURCE)],
    )

    print(f"Recorded {len(recorded)} case_word_sweep claims:\n")
    width = max(len(label) for label in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")
    print(
        f"\nEntry 121 sweep: anchor perplexity {anchor_ppl:.6f} over "
        f"{emissions} emissions; best mean log-gain "
        f"{recorded['case_word_sweep_best_mean_loggain']:.9f} nats/token at "
        f"alpha={best_alpha}; positive-control margin gate live at "
        f"{POSITIVE_CONTROL_MARGIN} nats/token."
    )
    print(
        "Scope: development evidence under a fresh registration; "
        "cross-corpus transfer (EWT sender, Stage 3 Brown receiver) is part "
        "of the declared estimand; the registry entries are "
        "registration-only structure, never an executed cascade."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: `ClaimsLedger` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results with no error anywhere.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
