#!/usr/bin/env python3
"""Record Stage 1 quantitative claims into data/claims.json.

Every number that appears in Stage 1 prose (README, AGENTS, docs/design) is
computed here from the vendored Brown counts and written to the ledger. Prose
references them as claim:<label>; nothing is hand-typed.

Includes the information-theoretic ceiling for an order-1 letter model, which is
what shows the originally-specified ">=2 nats" gate to be unreachable at this
level rather than merely unmet.

Run:  .venv/bin/python scripts/record_stage1_claims.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import jax.numpy as jnp  # noqa: E402

from ailm.claims import load_ledger  # noqa: E402
from ailm.evaluation import (  # noqa: E402
    STAGE1_EVALUATION_TEXT,
    evaluation_population_bound,
)
from ailm.gates import (  # noqa: E402
    STAGE1_DECLARED_GATE_NATS,
    STAGE1_GATE_SESOI_NATS,
)
from ailm.levels.letter import (  # noqa: E402
    ALPHA_TO_IDX,
    N_SYMBOLS,
    build_letter_agent,
    build_letter_transition,
    compute_perplexity,
    load_brown_counts,
    normalize_text,
)

RUN_ID = "stage1_letter_gate"
SOURCE = "scripts/record_stage1_claims.py"

from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402


def bigram_information_bounds(counts: dict) -> dict[str, float]:
    """Compute uniform / unigram / conditional entropies of the Brown bigrams.

    H(X_t | X_{t-1}) is the floor for any order-1 letter model: no bigram model
    can score below its own training distribution's conditional entropy.

    Returns:
        Dict of entropies in nats and their equivalent perplexities.
    """
    joint = np.zeros((N_SYMBOLS, N_SYMBOLS))
    for key, cnt in counts.items():
        prev, curr = key.split(",", 1)
        if prev in ALPHA_TO_IDX and curr in ALPHA_TO_IDX:
            joint[ALPHA_TO_IDX[curr], ALPHA_TO_IDX[prev]] += cnt

    p_joint = joint / joint.sum()
    p_prev = p_joint.sum(axis=0)
    p_curr = p_joint.sum(axis=1)

    h_cond = 0.0
    for j in range(N_SYMBOLS):
        if p_prev[j] <= 0:
            continue
        for i in range(N_SYMBOLS):
            if p_joint[i, j] > 0:
                h_cond -= p_joint[i, j] * math.log(p_joint[i, j] / p_prev[j])

    h_unigram = -float(sum(p * math.log(p) for p in p_curr if p > 0))
    h_uniform = math.log(N_SYMBOLS)

    return {
        "h_uniform_nats": h_uniform,
        "h_unigram_nats": h_unigram,
        "h_cond_nats": h_cond,
        "ppl_uniform": math.exp(h_uniform),
        "ppl_unigram_optimal": math.exp(h_unigram),
        "ppl_bigram_floor": math.exp(h_cond),
        "max_gain_nats": h_uniform - h_cond,
    }


def unigram_baseline_perplexity(counts: dict, text: str) -> float:
    """Context-free letter-frequency perplexity on `text` (the honest baseline)."""
    freq = np.ones(N_SYMBOLS)  # add-1, matching build_letter_transition's smoothing
    for key, cnt in counts.items():
        _, curr = key.split(",", 1)
        if curr in ALPHA_TO_IDX:
            freq[ALPHA_TO_IDX[curr]] += cnt
    freq /= freq.sum()

    chars = [c for c in normalize_text(text) if c in ALPHA_TO_IDX]
    logp = sum(math.log(freq[ALPHA_TO_IDX[c]]) for c in chars)
    return math.exp(-logp / len(chars))


def anti_informative_perplexity(counts: dict, text: str) -> float:
    """Perplexity of the mis-wired variant: bigram as emission A, identity B.

    Recorded so the architecture doc can cite the measured cost of the wiring
    error rather than asserting it qualitatively.
    """
    from pymdp.agent import Agent

    matrix = build_letter_transition(counts)
    agent = Agent(
        A=[matrix],
        B=[jnp.eye(N_SYMBOLS)[jnp.newaxis, :, :, jnp.newaxis]],
        C=[jnp.zeros((1, N_SYMBOLS))],
        D=[jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS],
        num_controls=[1],
        batch_size=1,
    )
    chars = [c for c in normalize_text(text) if c in ALPHA_TO_IDX]
    prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    logps: list[float] = []
    for ch in chars:
        idx = ALPHA_TO_IDX[ch]
        p = float(jnp.clip(jnp.dot(agent.A[0][0, idx, :], prior[0].ravel()), 1e-10, 1.0))
        logps.append(math.log(p))
        qs = agent.infer_states([jnp.array([idx])], prior)
        prior = agent.update_empirical_prior(action, qs)
    return math.exp(-sum(logps) / len(logps))


def main() -> int:
    counts = load_brown_counts(_PROJECT_ROOT / "vendor")
    bounds = bigram_information_bounds(counts)

    agent = build_letter_agent(build_letter_transition(counts))
    model_ppl = compute_perplexity(agent, STAGE1_EVALUATION_TEXT)
    model_nats = math.log(model_ppl)

    uniform_agent = build_letter_agent(jnp.ones((1, N_SYMBOLS, N_SYMBOLS)) / N_SYMBOLS)
    uniform_ppl = compute_perplexity(uniform_agent, STAGE1_EVALUATION_TEXT)

    unigram_ppl = unigram_baseline_perplexity(counts, STAGE1_EVALUATION_TEXT)

    gain_nats = bounds["h_uniform_nats"] - model_nats
    captured_pct = 100.0 * gain_nats / bounds["max_gain_nats"]

    # The same-population bound. Recorded beside the training-distribution one
    # because they answer different questions and the manuscript needs both.
    heldout = evaluation_population_bound(STAGE1_EVALUATION_TEXT)
    # The historically declared gate lives once, in ailm.gates, where the figure
    # layer also reads it; the recorder publishes that constant as the ledger
    # label the figures then compare against.
    declared = STAGE1_DECLARED_GATE_NATS
    if heldout["max_gain_nats"] >= declared:
        # Refuse rather than record. The gate-design section argues that the
        # Stage 1 target was unreachable on the population the model is scored
        # on; this is the measurement that either establishes it or destroys it.
        # Recording a value that destroys it, while the prose keeps asserting
        # it, is the exact failure the ledger exists to prevent.
        raise SystemExit(
            f"REFUSING TO RECORD: the held-out passage admits a gain of "
            f"{heldout['max_gain_nats']:.4f} nats over uniform at order 1, which "
            f"reaches the declared {declared} target. The Stage 1 unreachability "
            f"argument does not hold on the evaluation population and "
            f"the gate audit in docs/manuscript/04_results.md must be rewritten rather than "
            f"re-rendered."
        )
    heldout_captured_pct = 100.0 * gain_nats / heldout["max_gain_nats"]

    ledger = load_ledger(_PROJECT_ROOT)
    recorded = {
        # measured model performance
        "stage1_letter_perplexity": round(model_ppl, 4),
        "stage1_letter_nats": round(model_nats, 4),
        "stage1_letter_bits": round(model_nats / math.log(2), 4),
        # baselines the gate compares against
        "stage1_uniform_baseline": round(uniform_ppl, 4),
        "stage1_unigram_baseline": round(unigram_ppl, 4),
        # information-theoretic context
        "stage1_bigram_floor_perplexity": round(bounds["ppl_bigram_floor"], 4),
        "stage1_bigram_floor_nats": round(bounds["h_cond_nats"], 4),
        "stage1_gain_over_uniform_nats": round(gain_nats, 4),
        "stage1_max_bigram_gain_nats": round(bounds["max_gain_nats"], 4),
        # The gate Stage 1 originally DECLARED. It is a historical fact about the
        # spec, not a measurement, but it belongs in the ledger all the same: prose
        # and figures both quote it, and src/ailm/manuscript_variables.py previously
        # hardcoded the literal to derive the shortfall ratio — a hand-typed
        # number inside the very layer that exists to make hand-typed numbers
        # impossible. The value is the single constant in ailm.gates.
        "stage1_declared_gate_nats": STAGE1_DECLARED_GATE_NATS,
        "stage1_gate_sesoi_nats": STAGE1_GATE_SESOI_NATS,
        "stage1_bigram_signal_captured_pct": round(captured_pct, 1),
        # The order-1 bound on the evaluation population itself. The conditional
        # entropy covers T-1 transitions; the all-position entropy also includes
        # the fixed uniform initial prediction that compute_perplexity scores.
        "stage1_heldout_conditional_entropy_nats": round(heldout["conditional_entropy_nats"], 4),
        "stage1_heldout_all_position_entropy_nats": round(heldout["all_position_entropy_nats"], 4),
        "stage1_heldout_floor_perplexity": round(heldout["perplexity"], 4),
        "stage1_heldout_max_gain_nats": round(heldout["max_gain_nats"], 4),
        "stage1_heldout_signal_captured_pct": round(heldout_captured_pct, 1),
        "stage1_heldout_positions": int(heldout["positions"]),
        "stage1_heldout_transitions": int(heldout["transitions"]),
        # measured cost of the historical wiring error (negative control)
        "stage1_anti_informative_perplexity": round(
            anti_informative_perplexity(counts, STAGE1_EVALUATION_TEXT), 4
        ),
        # corpus provenance
        "stage1_brown_bigram_types": len(counts),
        "stage1_brown_bigram_tokens": int(sum(counts.values())),
    }
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(recorded)} Stage 1 claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\nGate context: model {model_ppl:.4f} ppl "
        f"({model_nats:.4f} nats) vs uniform {uniform_ppl:.4f}, "
        f"unigram {unigram_ppl:.4f}, order-1 floor "
        f"{bounds['ppl_bigram_floor']:.4f}."
    )
    print(
        f"Max gain over uniform at this order is {bounds['max_gain_nats']:.4f} nats "
        f"on the TRAINING distribution and {heldout['max_gain_nats']:.4f} nats on "
        f"the EVALUATION passage itself ({int(heldout['positions'])} positions: "
        f"one uniform-start prediction plus {int(heldout['transitions'])} transitions). "
        f'The originally-specified ">={declared} nats" target exceeds both, so it '
        f"is unreachable on the population the model is scored on and not only on "
        f"the one it was fitted to."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
