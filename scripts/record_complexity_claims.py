#!/usr/bin/env python3
"""Record every computational-complexity number, from the run that counts it.

Thin orchestrator; computation lives in ``ailm.complexity``. This script is the
single writer for every ``complexity_*`` label.

What makes these claims different from timings
----------------------------------------------
Nothing recorded here is a wall-clock measurement. Every value is either an
exact scalar-operation count from an **instrumented execution** of the shipped
inference recursion over the shipped workload, or an exact byte count of a
live agent's parameter arrays. Both are deterministic, so re-running this
recorder is a no-op — the idempotence the single-writer discipline (ISC-4)
requires and a timing could never satisfy.

Guards — the recorder refuses to write when
-------------------------------------------
1. Any level's closed-form operation formula disagrees with the instrumented
   count by even one operation. The formula must reproduce the executed branch
   structure: ``T - 1`` transition pushes per sequence, per-word resets, the
   decode argmax.
2. Any level's spec-derived memory formula disagrees with the live agent's
   actual bytes. The live agents are built **from their specs**
   (``ailm.spec_agent``), so this binds spec, formula, and runtime at once.
3. Any filtering run reports non-negative log evidence — evidence that real
   inference happened; a counter that never scored anything would be vacuous.

Usage
-----
    .venv/bin/python scripts/record_complexity_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.evaluation import STAGE1_EVALUATION_TEXT  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402

RUN_ID = "complexity_audit"
SOURCE = "scripts/record_complexity_claims.py"


def _filtering_workloads() -> dict[str, dict]:
    """The four filtering levels' real shipped workloads, deterministically.

    Each entry carries the model matrices (float64, exactly as the oracle
    consumes them) and the observation sequences the level's own gate scores.
    """
    from ailm.levels.case import (
        build_case_emission,
        build_case_start,
        build_case_transition,
        load_case_artifact,
    )
    from ailm.levels.letter import (
        ALPHA_TO_IDX,
        N_SYMBOLS,
        build_letter_transition,
        load_brown_counts,
        normalize_text,
    )
    from ailm.levels.phoneme import (
        N_PHONES,
        PHONE_TO_IDX,
        _word_letters,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        build_phone_transition,
        build_phone_unigram,
        g2p_split,
        load_phoneme_counts,
    )

    workloads: dict[str, dict] = {}

    # L1 — running text, one stream, no per-word reset (decision 7).
    letters = [ALPHA_TO_IDX[c] for c in normalize_text(STAGE1_EVALUATION_TEXT) if c in ALPHA_TO_IDX]
    bigram = np.asarray(build_letter_transition(load_brown_counts()), dtype=np.float64)[0]
    workloads["l1"] = {
        "states": N_SYMBOLS,
        "emission": np.eye(N_SYMBOLS),
        "transition": bigram,
        "start": np.full(N_SYMBOLS, 1.0 / N_SYMBOLS),
        "sequences": [letters],
        "decode": False,
    }

    counts = load_phoneme_counts()

    # L2A — one phone sequence per vendored pronunciation, reset per word.
    phone_sequences = []
    for _word, phones in counts["g2p_pairs"]:
        indices = [PHONE_TO_IDX[p] for p in phones.split() if p in PHONE_TO_IDX]
        if indices:
            phone_sequences.append(indices)
    workloads["l2a"] = {
        "states": N_PHONES,
        "emission": np.eye(N_PHONES),
        "transition": np.asarray(build_phone_transition(counts["phone_bigrams"]), dtype=np.float64)[
            0, :, :, 0
        ],
        "start": np.asarray(build_phone_unigram(counts["phone_unigrams"]), dtype=np.float64)[0],
        "sequences": phone_sequences,
        "decode": False,
    }

    # L2B — decode the held-out test words: filtering plus a per-position argmax.
    train, test = g2p_split(counts)
    letter_sequences = []
    for word, _phones in test:
        indices = [ALPHA_TO_IDX[c] for c in _word_letters(word)]
        if indices:
            letter_sequences.append(indices)
    workloads["l2b"] = {
        "states": N_PHONES,
        "emission": np.asarray(build_g2p_emission(train), dtype=np.float64)[0],
        "transition": np.asarray(build_g2p_transition(train), dtype=np.float64)[0, :, :, 0],
        "start": np.asarray(build_g2p_start(train), dtype=np.float64)[0],
        "sequences": letter_sequences,
        "decode": True,
    }

    # L4 — the derived test population, the one the Stage 4 gate is stated over.
    artifact = load_case_artifact()
    workloads["l4"] = {
        "states": 8,
        "emission": build_case_emission(artifact),
        "transition": build_case_transition(artifact),
        "start": build_case_start(artifact),
        "sequences": [
            [artifact.upos_to_idx[upos] for upos, _ in sequence] for sequence in artifact.test
        ],
        "decode": True,
    }

    return workloads


def main() -> int:
    from ailm.claims import load_ledger
    from ailm.complexity import (
        agent_memory_bytes,
        count_filtering_ops,
        count_word_path_ops,
        dense_joint_transition_bytes,
        dense_joint_transition_nonzeros,
        filtering_ops_theory,
        joint_over_factored_ratio,
        spec_memory_bytes_theory,
        word_factored_transition_bytes,
        word_factored_transition_nonzeros,
        word_path_ops_theory,
    )
    from ailm.gnn import load_spec
    from ailm.levels.word import build_word_prior, load_word_artifact
    from ailm.spec_agent import build_agent_from_spec

    values: dict[str, float | int] = {}

    # ── Filtering levels: instrumented count must equal the closed form ──────
    for level, load in _filtering_workloads().items():
        lengths = [len(sequence) for sequence in load["sequences"]]
        counted, log_evidence = count_filtering_ops(
            load["emission"],
            load["transition"],
            load["start"],
            load["sequences"],
            decode=load["decode"],
        )
        theory = filtering_ops_theory(load["states"], lengths, decode=load["decode"])
        if counted != theory:
            raise SystemExit(
                f"{level}: instrumented count {counted.as_dict()} != closed form "
                f"{theory.as_dict()}. One of them mis-states the executed branch "
                f"structure; refusing to record either."
            )
        if not log_evidence < 0.0:
            raise SystemExit(
                f"{level}: log evidence {log_evidence} is not negative — the "
                f"counter cannot have scored a real probabilistic workload."
            )
        total_positions = sum(lengths)
        values[f"complexity_{level}_states"] = load["states"]
        values[f"complexity_{level}_sequences"] = len(lengths)
        values[f"complexity_{level}_positions"] = total_positions
        values[f"complexity_{level}_ops_total"] = counted.total
        values[f"complexity_{level}_ops_per_position"] = round(counted.total / total_positions, 1)

    # ── The word level: determinism collapses scoring to Θ(1) per token ──────
    artifact = load_word_artifact()
    prior = build_word_prior(artifact)
    word_counted, total_logp = count_word_path_ops(
        artifact.eval_tokens, artifact.word_to_index, prior
    )
    word_theory = word_path_ops_theory(len(artifact.eval_tokens))
    if word_counted != word_theory:
        raise SystemExit(
            f"l3: instrumented count {word_counted.as_dict()} != closed form "
            f"{word_theory.as_dict()}; refusing to record either."
        )
    if not total_logp < 0.0:
        raise SystemExit("l3: total log probability is not negative")
    values["complexity_l3_tokens"] = len(artifact.eval_tokens)
    values["complexity_l3_ops_total"] = word_counted.total
    values["complexity_l3_ops_per_token"] = word_counted.total / len(artifact.eval_tokens)

    # ── Memory: the spec prices the live agent, to the byte ──────────────────
    context: dict = {}
    for level, spec_name in (
        ("l1", "L1_letter"),
        ("l1g", "L1g_letter_generation"),
        ("l2a", "L2a_phoneme_sequence"),
        ("l2b", "L2b_grapheme_to_phoneme"),
        ("l3", "L3_word"),
        ("l4", "L4_case"),
    ):
        predicted = spec_memory_bytes_theory(load_spec(spec_name))
        measured = agent_memory_bytes(build_agent_from_spec(spec_name, context=context))
        if predicted != measured:
            raise SystemExit(
                f"{spec_name}: spec-derived memory {predicted:,} B != live agent "
                f"{measured:,} B; refusing to record either."
            )
        values[f"complexity_{level}_memory_bytes"] = measured

    # ── The factorization counterfactual ─────────────────────────────────────
    n_vocab = len(artifact.vocab)
    n_positions = artifact.n_positions
    dense = dense_joint_transition_bytes(n_vocab, n_positions)
    factored = word_factored_transition_bytes(n_vocab, n_positions)
    values["complexity_l3_position_states"] = n_positions
    values["complexity_l3_dense_joint_gb"] = round(dense / 1e9, 3)
    values["complexity_l3_factored_transition_mb"] = round(factored / 1e6, 1)
    values["complexity_l3_joint_over_factored"] = round(dense / factored, 1)

    # The same counterfactual priced by nonzero count. Both transitions are
    # deterministic maps with one nonzero per source state, so the joint costs a
    # factor of roughly P more than the factored pair rather than P squared. The
    # exponent belongs to the dense encoding, not to the factorization, and
    # recording both routes is what keeps that attribution checkable.
    values["complexity_l3_joint_nonzeros"] = dense_joint_transition_nonzeros(n_vocab, n_positions)
    values["complexity_l3_factored_nonzeros"] = word_factored_transition_nonzeros(
        n_vocab, n_positions
    )
    values["complexity_l3_joint_over_factored_sparse"] = round(
        joint_over_factored_ratio(n_vocab, n_positions, sparse=True), 2
    )

    ledger = load_ledger(_PROJECT_ROOT)
    ledger.record_many(values, run_id=RUN_ID, source=SOURCE)

    print(f"Recorded {len(values)} complexity claims.")
    for label, value in sorted(values.items()):
        print(f"  {label} = {value}")
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(_PROJECT_ROOT)):
        raise SystemExit(main())
