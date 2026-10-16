#!/usr/bin/env python3
"""Record the third EFE degeneracy: word-state information gain that selects nothing.

Thin orchestrator. Computation lives in ``ailm.generation``; this script builds
the word-identity construction, sequences the measurements, and writes the
ledger. Single writer for every ``generation_word_state_*`` label. Decision 78
(``docs/decisions.md``) prospectively specifies the construction, prefix rule,
predictions, and refusals within the development session. The decision and this
recorder first appear in the same commit, so the result is not independently
timestamped or preregistered.

The construction under test is the research queue's epistemic-generation item,
taken at its word: hidden factor = word identity over the shipped Stage 3
lexicon, action = the emitted letter, ``A`` = the prefix-conditioned
next-symbol map, belief = the trie-node posterior. The level invariant pins
``B``: a word does not become a different word part-way through being spelled
(``build_word_transition``), so ``B[:, :, k]`` is the identity for *every*
action — the action changes what is observed, never which word is being
spelled.

What that buys, and what it cancels:

1. **Information gain is genuinely nonzero** wherever the prefix is ambiguous —
   the trie-node posterior carries real uncertainty and the next symbol
   resolves some of it. This is the half the queue item got right.
2. **It is the same for every action.** Action-independent transitions make the
   predictive state distribution — and therefore the predictive observation
   distribution — identical across actions, so both EFE terms are constants
   over the policy space and their per-action spread is exactly zero.
3. **The softmax cancels constants**, so the induced policy posterior
   ``softmax(gamma * neg_efe + ln E)`` is ``softmax(ln E)``: the pure habit.

No pymdp ``Agent`` is built. A dense per-prefix ``B`` would spend hundreds of
megabytes of the ``MAX_AGENT_VOCAB_BYTES`` budget on a constant, so the
identity is broadcast, not materialized, and everything runs through the
standalone ``expected_free_energy_terms`` path the decomposition tests already
pin against pymdp.

Usage
-----
    python scripts/record_generation_word_state_claims.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ailm.ledger_lock import ledger_lock_path, single_writer_lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "generation_word_state"
SOURCE = "scripts/record_generation_word_state_claims.py"

#: Trie depths probed beyond the empty prefix. Fixed by decision 78: at each
#: depth the recorder takes the prefix with the greatest word-prior mass among
#: nodes with at least two distinct next symbols — deterministic given the
#: artifact, so no prefix is hand-picked to flatter the result.
PROBED_DEPTHS: tuple[int, ...] = (1, 2, 3, 4)

#: Prediction (c)'s tolerance: the policy posterior must match the habit
#: softmax to machine precision. 1e-12 is the same bar the decomposition
#: equivalence tests use for "these are the same computation".
POLICY_DEVIATION_BOUND: float = 1e-12


def next_symbol_index(word: str, depth: int, alpha_to_idx: dict[str, int]) -> int:
    """The observation a word emits at ``depth``: its next letter, or the space.

    Mirrors ``lexicon_preference``'s marginalization exactly — a word that ends
    at the prefix votes for the space that terminates it.
    """
    symbol = word[depth] if depth < len(word) else " "
    return alpha_to_idx[symbol]


def select_prefixes(
    vocab: tuple[str, ...], prior: np.ndarray, alpha_to_idx: dict[str, int]
) -> list[str]:
    """The empty prefix plus the heaviest ambiguous trie node at each probed depth.

    "Ambiguous" means at least two distinct next symbols among the node's
    words: only there can the next observation carry information about word
    identity, so only there can prediction (a) be tested at all.

    Raises:
        SystemExit: If some probed depth has no ambiguous node — the rule would
            silently return fewer prefixes than the specification names.
    """
    prefixes = [""]
    for depth in PROBED_DEPTHS:
        mass: dict[str, float] = {}
        symbols: dict[str, set[int]] = {}
        for index, word in enumerate(vocab):
            if len(word) < depth:
                continue
            node = word[:depth]
            mass[node] = mass.get(node, 0.0) + float(prior[index])
            symbols.setdefault(node, set()).add(next_symbol_index(word, depth, alpha_to_idx))
        ambiguous = [node for node, seen in symbols.items() if len(seen) >= 2]
        if not ambiguous:
            raise SystemExit(
                f"REFUSING TO RECORD: no ambiguous trie node at depth {depth}. "
                f"Decision 78's rule names one prefix per depth in "
                f"{PROBED_DEPTHS}; recording a shorter list would not be the "
                f"specified measurement."
            )
        # max() with a (mass, node) key: deterministic, and ties break
        # lexicographically rather than by dict order.
        prefixes.append(max(ambiguous, key=lambda node: (mass[node], node)))
    return prefixes


def word_state_model(
    prefix: str, vocab: tuple[str, ...], prior: np.ndarray, alpha_to_idx: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The queue item's construction at one trie node: ``(A, B, belief)``.

    ``A[o, j]`` is 1 where word ``j`` of the prefix-consistent support emits
    symbol ``o`` next — deterministic, column-stochastic. ``B`` is the identity
    for every action, per the level invariant, supplied as a broadcast view so
    the empty-prefix node (the full lexicon) costs one identity matrix rather
    than a dense per-action stack.
    """
    from ailm.levels.letter import N_SYMBOLS

    support = [index for index, word in enumerate(vocab) if word.startswith(prefix)]
    belief = np.asarray(prior[support], dtype=np.float64)
    belief /= belief.sum()

    emission = np.zeros((N_SYMBOLS, len(support)), dtype=np.float64)
    for column, index in enumerate(support):
        emission[next_symbol_index(vocab[index], len(prefix), alpha_to_idx), column] = 1.0

    identity = np.eye(len(support), dtype=np.float64)
    transition = np.broadcast_to(identity[:, :, None], (len(support), len(support), N_SYMBOLS))
    return emission, transition, belief


def action_dependent_control(
    emission: np.ndarray, belief: np.ndarray, preference: np.ndarray
) -> dict[str, np.ndarray]:
    """The control that must fire: the same node with transitions that differ by action.

    Each action ``k`` mixes the identity with the cyclic shift by ``k`` (modulo
    the support size), so distinct actions genuinely induce distinct predictive
    states. If the per-action spread is still zero here, the spread computation
    cannot detect action dependence and the recorded zeros mean nothing.
    """
    from ailm.generation import expected_free_energy_terms
    from ailm.levels.letter import N_SYMBOLS

    n_states = belief.shape[0]
    identity = np.eye(n_states, dtype=np.float64)
    transition = np.empty((n_states, n_states, N_SYMBOLS), dtype=np.float64)
    for action in range(N_SYMBOLS):
        transition[:, :, action] = 0.5 * identity + 0.5 * np.roll(
            identity, action % n_states, axis=0
        )
    return expected_free_energy_terms(emission, transition, preference, state_prior=belief)


def main() -> int:
    from ailm.claims import load_ledger
    from ailm.generation import (
        SPACE_INDEX,
        expected_free_energy_terms,
        lexicon_preference,
    )
    from ailm.levels.letter import ALPHA_TO_IDX, build_letter_transition, load_brown_counts
    from ailm.levels.word import build_word_prior, load_word_artifact

    ledger = load_ledger(PROJECT_ROOT)
    best_gamma = ledger.get("generation_best_gamma")
    if best_gamma is None:
        raise SystemExit(
            "REFUSING TO RECORD: generation_best_gamma is not in the ledger. "
            "Prediction (c) is registered at the shipped operating precision; "
            "run scripts/record_generation_claims.py first."
        )
    gamma = float(best_gamma["value"])

    artifact = load_word_artifact()
    vocab = tuple(artifact.vocab)
    prior = build_word_prior(artifact)
    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]

    prefixes = select_prefixes(vocab, prior, ALPHA_TO_IDX)

    gains: dict[str, float] = {}
    supports: dict[str, int] = {}
    spreads = {"information_gain": 0.0, "pragmatic_value": 0.0, "expected_free_energy": 0.0}
    policy_deviation = 0.0

    for prefix in prefixes:
        emission, transition, belief = word_state_model(prefix, vocab, prior, ALPHA_TO_IDX)
        preference = lexicon_preference(prefix, artifact, prior)
        terms = expected_free_energy_terms(emission, transition, preference, state_prior=belief)

        gains[prefix] = float(terms["information_gain"][0])
        supports[prefix] = belief.shape[0]
        for name in spreads:
            spreads[name] = max(spreads[name], float(np.ptp(terms[name])))

        # Prediction (c): the induced policy posterior against the pure habit.
        # The habit matrix is stored float32; both softmaxes must run in
        # float64, or the comparison measures the promotion rather than the
        # cancellation.
        current = SPACE_INDEX if not prefix else ALPHA_TO_IDX[prefix[-1]]
        log_habit = np.log(np.maximum(habit[:, current].astype(np.float64), 1e-300))
        scores = -gamma * terms["expected_free_energy"] + log_habit
        scores -= scores.max()
        posterior = np.exp(scores) / np.exp(scores).sum()
        habit_scores = log_habit - log_habit.max()
        habit_posterior = np.exp(habit_scores) / np.exp(habit_scores).sum()
        policy_deviation = max(policy_deviation, float(np.max(np.abs(posterior - habit_posterior))))

    # ── Refusal conditions, in decision 78's order ────────────────────────────
    if any(spread != 0.0 for spread in spreads.values()):
        raise SystemExit(
            f"REFUSING TO RECORD: prediction (b) failed — a per-action spread is "
            f"nonzero ({spreads}). Action-independent transitions must make both "
            f"EFE terms constant across actions; a nonzero spread means the "
            f"derivation is wrong and the manuscript extension must not ship."
        )
    if policy_deviation >= POLICY_DEVIATION_BOUND:
        raise SystemExit(
            f"REFUSING TO RECORD: prediction (c) failed — the policy posterior "
            f"deviates from softmax(ln E) by {policy_deviation:.3e}, at or above "
            f"the registered {POLICY_DEVIATION_BOUND:.0e} bound."
        )
    if not all(gain > 0.0 for gain in gains.values()):
        raise SystemExit(
            f"REFUSING TO RECORD: prediction (a) failed — a selected node has no "
            f"information gain ({gains}). The selection rule was to name only "
            f"ambiguous nodes, so a zero here means the rule or the term is "
            f"broken, and 'nonzero yet action-inert' would have no exhibit."
        )

    deepest = prefixes[-1]
    emission, _, belief = word_state_model(deepest, vocab, prior, ALPHA_TO_IDX)
    control = action_dependent_control(
        emission, belief, lexicon_preference(deepest, artifact, prior)
    )
    control_ig_spread = float(np.ptp(control["information_gain"]))
    control_g_spread = float(np.ptp(control["expected_free_energy"]))
    if control_ig_spread <= 0.0 or control_g_spread <= 0.0:
        raise SystemExit(
            f"REFUSING TO RECORD: the action-dependent control did not fire "
            f"(information-gain spread {control_ig_spread:.6g}, G spread "
            f"{control_g_spread:.6g}). A control that cannot produce a nonzero "
            f"spread turns the recorded zeros into decoration."
        )

    analytic: dict[str, float] = {
        "generation_word_state_ig_empty_nats": round(gains[""], 6),
        "generation_word_state_max_ig_spread_nats": spreads["information_gain"],
        "generation_word_state_max_pragmatic_spread_nats": spreads["pragmatic_value"],
        "generation_word_state_max_g_spread_nats": spreads["expected_free_energy"],
        "generation_word_state_policy_habit_max_abs_dev": float(f"{policy_deviation:.3g}"),
        "generation_word_state_control_ig_spread_nats": round(control_ig_spread, 6),
        "generation_word_state_control_g_spread_nats": round(control_g_spread, 6),
    }
    for depth, prefix in zip(PROBED_DEPTHS, prefixes[1:], strict=True):
        analytic[f"generation_word_state_ig_depth{depth}_nats"] = round(gains[prefix], 6)

    descriptive: dict[str, float | int | str] = {
        "generation_word_state_prefix_count": len(prefixes),
        "generation_word_state_support_empty": supports[""],
    }
    for depth, prefix in zip(PROBED_DEPTHS, prefixes[1:], strict=True):
        descriptive[f"generation_word_state_prefix_depth{depth}"] = prefix
        descriptive[f"generation_word_state_support_depth{depth}"] = supports[prefix]

    ledger.record_many(analytic, run_id=RUN_ID, source=SOURCE, status="analytic")
    ledger.record_many(descriptive, run_id=RUN_ID, source=SOURCE, status="descriptive")

    recorded = {**analytic, **descriptive}
    print(f"Recorded {len(recorded)} generation_word_state claims into data/claims.json:\n")
    width = max(len(label) for label in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\nAll three prospectively specified predictions held (decision 78): information "
        f"gain is positive at every selected node (empty prefix: "
        f"{gains['']:.6f} nats), every per-action spread is exactly 0.0, and "
        f"the policy posterior matches softmax(ln E) to {policy_deviation:.3g} "
        f"at gamma={gamma}. The epistemic term is real and selects nothing."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: two overlapping recorders lose one of the
    # two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
