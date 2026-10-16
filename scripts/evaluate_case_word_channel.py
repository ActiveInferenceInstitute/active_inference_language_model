#!/usr/bin/env python3
"""Structural evaluation of the L4→L3 case-word bridge (model-sender arm).

Protocol P3 partition 3's thin executor check: load the canonical case-word
artifact, build the REAL case agent from the shipped case artifact, marginalize
its per-token role posterior into word messages, and run the concrete L3 prior
inlet over held-out development documents. This script reports structure only —

- anchor identity: alpha = 0 and the no-message case reproduce the receiver
  prior bit-for-bit;
- seam normalization: every mixed prior sums to one within tolerance;
- causal reachability: with support and alpha > 0 the receiver prior and its
  per-token log-probabilities change; the no-message tokens do not change;
- declared scope: development evidence, model-sender estimand only (the
  oracle-sender arm is a separately labeled diagnostic and is not executed
  here), no performance claim, no claim-recorder run, no influence sweep.

Usage:
    .venv/bin/python scripts/evaluate_case_word_channel.py [--documents N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from ailm.corpus.case_word import load_case_word_artifact  # noqa: E402
from ailm.levels.case import N_CASES, build_case_agent, load_case_artifact  # noqa: E402
from ailm.levels.word import build_word_prior, load_word_artifact  # noqa: E402

#: The declared mixing grid. 0.0 is the anchor; the positive values are the
#: structural demonstration weights, not a performance sweep.
ALPHAS = (0.0, 0.2, 0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=5)
    args = parser.parse_args(argv)

    case_word = load_case_word_artifact()
    case_artifact = load_case_artifact()
    word_artifact = load_word_artifact()

    # The message vectors are aligned to the receiver's vocabulary index order.
    if list(word_artifact.vocab) != case_word.vocab:
        raise SystemExit(
            "REFUSING TO RUN: the case-word artifact's vocabulary order does "
            "not match the Stage 3 receiver's; regenerate the artifact."
        )
    prior = build_word_prior(word_artifact)

    # The real L4 model, and its real per-token posterior under reset
    # semantics, identity-checked row by row against the Bayes expression.
    agent = build_case_agent(case_artifact)
    posterior = role_posterior_from_agent(agent, n_upos=case_artifact.n_obs, n_cases=N_CASES)
    upos_to_idx = case_artifact.upos_to_idx
    role_names = [role.name for role in CaseRole]

    sender = fit_sender(case_word.sender_map, case_word.vocab)
    supported_roles = sorted(sender)

    documents = case_word.test_documents[: args.documents]
    no_message_tokens = 0
    changed_tokens = 0
    total_tokens = 0
    for document in documents:
        for row in document["rows"]:
            upos = row.upos
            if upos not in upos_to_idx:
                continue
            total_tokens += 1
            message = build_message(posterior[upos_to_idx[upos]], role_names, sender)
            if message is None:
                no_message_tokens += 1
                continue
            for alpha in ALPHAS:
                mixed = apply_case_word_inlet(prior, message, alpha)
                if abs(float(np.sum(mixed)) - 1.0) > 1e-9:
                    raise SystemExit(
                        f"the inlet produced an unnormalized prior (sum "
                        f"{float(np.sum(mixed))!r}) at alpha {alpha}"
                    )
            if not np.array_equal(apply_case_word_inlet(prior, message, 0.0), prior):
                raise SystemExit(
                    "the alpha = 0 anchor is no longer bit-identical to the receiver prior"
                )
            if not np.array_equal(apply_case_word_inlet(prior, message, ALPHAS[-1]), prior):
                changed_tokens += 1

    print("=== L4→L3 case-word bridge — structural evaluation (model-sender arm) ===")
    print(f"receiver vocabulary          : {len(case_word.vocab)} types (Stage 3)")
    print(f"sender roles with support    : {supported_roles}")
    print(f"L4 posterior identity check  : passed for all {posterior.shape[0]} UPOS rows")
    print(f"held-out documents evaluated : {len(documents)} (dev test-split, development evidence)")
    print(f"UPOS tokens considered       : {total_tokens}")
    print(f"tokens with a message        : {total_tokens - no_message_tokens}")
    print(f"no-message tokens (D_a = D_0): {no_message_tokens}")
    print(
        f"tokens whose receiver prior changed at alpha={ALPHAS[-1]} "
        f"(and anchor-identical at alpha=0): {changed_tokens}"
    )
    print(
        "scope: development evidence only; no performance estimand scored, no "
        "claim recorded, no influence sweep run. The oracle-sender diagnostic "
        "is a separately labeled registry arm and was not executed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
