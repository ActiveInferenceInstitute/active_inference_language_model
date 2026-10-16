#!/usr/bin/env python3
"""Record every Stage 3 number into the claims ledger, from the run that measures it.

Thin orchestrator. All computation lives in ``ailm.levels.word`` and
``ailm.verification.oracle``; this script sequences the measurements, proves the
pymdp agent and the closed form agree before trusting either, and writes the
ledger. It is the single writer for its main Stage 3 namespace; the separately
scoped ``stage3_oov_*`` denominator analysis has its own recorder. Tests verify
these values and must never overwrite them (``docs/decisions.md`` entry 10).

What it establishes, in order:

1. The trie-encoding cost that decided the state space (decision 13).
2. The comparator ladder, all on identical positions (decision 2).
3. The path identity — the pymdp agent's emission-path probability against the
   closed-form word unigram — which is what licenses computing the full-set
   statistics in closed form at all (the oracle-equivalence pattern,
   ``docs/gate_design.md``).
4. The mis-wirings, including the three that beat the bar while being wrong.

Usage
-----
    python scripts/record_stage3_claims.py [--equivalence-words N]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ailm.gates import STAGE3_WORD_GATE_SESOI_NATS
from ailm.ledger_lock import ledger_lock_path, single_writer_lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "stage3_word_gate"
SOURCE = "scripts/record_stage3_claims.py"

#: The originally declared Stage 3 threshold, kept so its verdict is re-derived
#: rather than remembered.
DECLARED_GATE_PERPLEXITY = 500.0


#: Vocabulary rungs of decision 13's trie-vs-factored table.
TRIE_LADDER_SIZES = (1000, 2000, 4000, 8000)


def measure_trie_cost(cmudict_counts: dict) -> dict[str, float]:
    """Cost of the rejected trie-node encoding, over the G2P vocabulary.

    This is the measurement that decided decision 13. It is recorded rather than
    described so the rejection stays auditable: a design alternative dismissed
    with an unmeasured "too big" is a decision nobody can check.
    """
    words = [w for w, _ in cmudict_counts["g2p_pairs"]]
    prefixes: set[str] = set()
    for word in words:
        for i in range(1, len(word) + 1):
            prefixes.add(word[:i])
    nodes = len(prefixes) + 1  # + root
    edges = len(prefixes)  # one incoming edge per non-root node
    dense_bytes = nodes * nodes * 4
    return {
        "nodes": nodes,
        "edges": edges,
        "dense_gb": dense_bytes / 1e9,
        "density_pct": 100.0 * edges / (nodes * nodes),
        # Full CMUdict is ~3.9x the length-matched vocabulary; the dense cost is
        # quadratic, so extrapolate the node count and square it.
        "dense_full_tb": ((nodes * 3.9) ** 2 * 4) / 1e12,
    }


def measure_trie_ladder(artifact, sizes: tuple[int, ...]) -> dict[str, float]:
    """Trie-node count at each vocabulary rung of decision 13's table.

    The table was typed markdown. Two of its four coverage figures did not round
    from the ledger values beside them (78.31 was printed 78.4, 89.52 was
    printed 89.6) and none of its four node counts survived a rebuild — the
    shipped 2,000-word vocabulary yielded 5,693 nodes against a printed 5,700;
    the vendored lexicon was later reduced to 38,007 types on 2026-08-23, so the
    same rung regenerates 5,643 today. Nothing regenerated the table, so nothing
    caught it drifting.

    Recording the counts makes the rejection auditable in the same way the
    G2P-vocabulary measurement above already is: a design alternative rejected
    on numbers nobody can regenerate is a decision nobody can check.

    Args:
        artifact: The word artifact, whose vocabulary is frequency ordered.
        sizes: Vocabulary rungs to measure.

    Returns:
        ``{"stage3_trie_nodes_v<N>": count}`` per rung.
    """
    if len(artifact.vocab) < max(sizes):
        raise ValueError(
            f"vocabulary of {len(artifact.vocab):,} cannot supply a rung of "
            f"{max(sizes):,}; truncating would silently record the smaller "
            f"trie under the larger rung's label"
        )
    values: dict[str, float] = {}
    for size in sizes:
        prefixes: set[str] = set()
        for word in artifact.vocab[:size]:
            for i in range(1, len(word) + 1):
                prefixes.add(word[:i])
        nodes = len(prefixes) + 1  # + root
        values[f"stage3_trie_nodes_v{size}"] = nodes
        # The dense cost is derived, and derived numbers typed by hand are how
        # this table drifted in the first place. Record it beside its source.
        values[f"stage3_trie_dense_v{size}_mb"] = round(nodes * nodes * 4 / 1e6, 1)
        values[f"stage3_factored_v{size}_mb"] = round(size * size * 4 / 1e6, 1)
    return values


def verify_path_identity(artifact, agent, prior: np.ndarray, n_words: int) -> float:
    """Largest disagreement between the live pymdp agent and the closed form.

    Walks the agent letter by letter over a stride-sampled slice of the
    vocabulary, accumulating ``log P(o_t | predictive_t)``, and compares the
    total against ``log D[0][w]``. Returns the maximum absolute difference in
    nats.

    The sampling is an evenly spaced **stride**, never a prefix: the vocabulary
    is sorted by descending frequency, so a prefix would sample only the
    commonest words and a suffix only the rarest (decision 4).

    Raises:
        ValueError: If the disagreement exceeds float32's reach. The full-set
            statistics are computed in closed form, so they describe the pymdp
            agent only while this identity holds; a warning here would be a
            substitution rather than a proof.
    """
    import jax.numpy as jnp

    from ailm.verification.oracle import word_path_logprob

    emission = np.asarray(agent.A[0])[0]
    n_pos = artifact.n_positions
    position_start = np.zeros(n_pos, dtype=np.float32)
    position_start[0] = 1.0

    stride = max(1, len(artifact.vocab) // n_words)
    sampled = artifact.vocab[::stride]

    worst = 0.0
    for word in sampled:
        belief = [
            jnp.array(np.asarray(agent.D[0]).reshape(1, -1)),
            jnp.array(position_start[np.newaxis]),
        ]
        live_logp = 0.0
        for obs in [ord(ch) - 97 for ch in word] + [26]:
            q_word = np.asarray(belief[0]).ravel()
            q_pos = np.asarray(belief[1]).ravel()
            predictive = float(np.einsum("wp,w,p->", emission[obs], q_word, q_pos))
            live_logp += math.log(max(predictive, 1e-300))
            qs = agent.infer_states([jnp.array([obs])], belief)
            belief = agent.update_empirical_prior(jnp.array([[0, 0]]), qs)
        closed = word_path_logprob(word, artifact.word_to_index[word], prior)
        worst = max(worst, abs(live_logp - closed))

    if worst > 1e-4:
        raise ValueError(
            f"The pymdp agent and the closed form disagree by {worst:.3e} nats "
            f"over {len(sampled)} words. The full-set Stage 3 statistics are "
            f"computed in closed form and would no longer describe the agent. "
            f"Refusing to record."
        )
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--equivalence-words",
        type=int,
        default=400,
        help="Vocabulary words walked through pymdp for the path identity.",
    )
    args = parser.parse_args(argv)

    from ailm.claims import load_ledger
    from ailm.levels.word import (
        build_letter_bigram_bar,
        build_word_agent,
        build_word_prior,
        load_word_artifact,
        validate_stage3_miswirings,
        word_perplexity,
    )
    from ailm.verification.oracle import letter_bigram_perplexity

    artifact = load_word_artifact()
    tokens = artifact.eval_tokens
    prior = build_word_prior(artifact)

    # ── The level, and the bar, on identical positions ────────────────────────
    model_ppl, n_emissions = word_perplexity(artifact)
    transition, start = build_letter_bigram_bar(artifact)
    bar_ppl, n_bar = letter_bigram_perplexity(tokens, transition, start)
    if n_bar != n_emissions:
        raise ValueError(
            f"The bar scored {n_bar} emissions and the model {n_emissions}. A "
            f"comparator on a different position set is not a comparator."
        )

    # ── Scale markers. Never the bar; recorded so the ladder is visible. ───────
    uniform_ppl = 27.0
    letter_counts = np.full(27, 1.0)
    for word, count in artifact.bar_train_freq.items():
        for ch in word:
            letter_counts[ord(ch) - 97] += count
        letter_counts[26] += count
    letter_unigram = letter_counts / letter_counts.sum()
    unigram_logp = sum(math.log(letter_unigram[ord(ch) - 97]) for w in tokens for ch in w) + len(
        tokens
    ) * math.log(letter_unigram[26])
    unigram_ppl = math.exp(-unigram_logp / n_emissions)

    # ── The ceiling: an oracle knowing the held-out word distribution exactly ──
    from collections import Counter

    held_freq = Counter(tokens)
    ceiling_logp = sum(math.log(held_freq[w] / len(tokens)) for w in tokens)
    ceiling_ppl = math.exp(-ceiling_logp / n_emissions)

    # ── The path identity, before any of the above is trusted ─────────────────
    agent = build_word_agent(artifact)
    identity_err = verify_path_identity(artifact, agent, prior, args.equivalence_words)

    # ── Mis-wirings: what the bar does and does not catch ─────────────────────
    vocab = artifact.vocab
    uniform_prior = np.full(len(vocab), 1.0 / len(vocab))

    length_groups = Counter(len(w) for w in vocab)
    length_prior = np.array([1.0 / (length_groups[len(w)] * len(length_groups)) for w in vocab])
    length_prior /= length_prior.sum()

    reversed_prior = prior[::-1].copy()

    def ppl_under(p: np.ndarray) -> float:
        value, _ = word_perplexity(artifact, word_prior=p)
        return value

    # Time-reversed emission: the model spells words backwards, so a token is
    # scored under the reversed-spelling path. Words that are not palindromic
    # reversals of a vocabulary entry get no mass at all.
    reversed_paths: dict[str, float] = {}
    total_freq = sum(artifact.word_freq.values())
    for word, count in artifact.word_freq.items():
        reversed_paths[word[::-1]] = count / total_freq
    reversed_logp = sum(math.log(max(reversed_paths.get(w, 0.0), 1e-12)) for w in tokens)
    time_reversed_ppl = math.exp(-reversed_logp / n_emissions)
    miswiring_values = {
        "uniform_prior": round(ppl_under(uniform_prior), 4),
        "length_prior": round(ppl_under(length_prior), 4),
        "reversed_prior": round(ppl_under(reversed_prior), 4),
        "time_reversed": round(time_reversed_ppl, 4),
    }
    miswiring_registry = validate_stage3_miswirings()
    finite_registry_keys = {arm.key for arm in miswiring_registry if arm.has_finite_score}
    if set(miswiring_values) != finite_registry_keys:
        raise ValueError(
            "Stage 3 recorder and canonical miswiring registry disagree: "
            f"measured={sorted(miswiring_values)}, registered={sorted(finite_registry_keys)}"
        )
    recorded_miswirings = {
        arm.claim_label: miswiring_values[arm.key]
        for arm in miswiring_registry
        if arm.claim_label is not None
    }

    gain_nats = math.log(bar_ppl) - math.log(model_ppl)
    headroom_nats = math.log(model_ppl) - math.log(ceiling_ppl)
    available = math.log(bar_ppl) - math.log(ceiling_ppl)

    # ── Complete length-eligible training vocabulary, measured in closed form ──
    # The shipped agent is bounded by memory; the MODEL is not. Every factor
    # except D is deterministic, so the closed form scores any vocabulary size
    # exactly -- which is what makes scaling to every length-eligible training
    # type a measurement rather than an aspiration. Recorded so the ladder from
    # the shipped sublanguage to the complete eligible training vocabulary is
    # visible instead of promised.
    full_path = PROJECT_ROOT / "vendor" / "brown_word_counts_full.json"
    full: dict[str, float | int] = {}
    if full_path.exists():
        from ailm.complexity import agent_memory_bytes_from_shape
        from ailm.levels.word import (
            MAX_AGENT_VOCAB_BYTES,
            N_OBS,
            load_word_artifact,
            scaling_rungs,
        )

        full_artifact = load_word_artifact(full_path)
        full_ppl, full_positions = word_perplexity(full_artifact)
        full_transition, full_start = build_letter_bigram_bar(full_artifact)
        full_bar, _ = letter_bigram_perplexity(
            full_artifact.eval_tokens, full_transition, full_start
        )
        n_full = len(full_artifact.vocab)
        # The ladder itself, not just its ends. Numeric coverage and perplexity
        # both rise with lexicon size, but their utility directions oppose:
        # higher coverage is desirable while higher perplexity is worse. The
        # intermediate rungs make that trade the finding rather than a choice of
        # emphasis, and they cost seconds because every factor except the word
        # prior is deterministic.
        full.update(measure_trie_ladder(full_artifact, TRIE_LADDER_SIZES))
        rungs = scaling_rungs(full_artifact)
        for rung in rungs:
            size = int(rung["vocab_size"])
            full[f"stage3_scale_{size}_coverage_pct"] = rung["coverage_pct"]
            full[f"stage3_scale_{size}_word_perplexity"] = rung["perplexity"]
            full[f"stage3_scale_{size}_bigram_baseline"] = rung["bigram_baseline"]
            full[f"stage3_scale_{size}_eval_positions"] = rung["positions"]
            full[f"stage3_scale_{size}_gain_nats"] = rung["gain_nats"]
            full[f"stage3_scale_{size}_gain_shared"] = rung["gain_nats_shared_denominator"]
        full["stage3_scale_rungs"] = len(rungs)
        full["stage3_scale_gain_rise_pct"] = round(
            100.0
            * (rungs[-1]["gain_nats_shared_denominator"] - rungs[0]["gain_nats_shared_denominator"])
            / rungs[0]["gain_nats_shared_denominator"],
            2,
        )
        full |= {
            "stage3_full_vocab_size": n_full,
            "stage3_full_coverage_pct": round(full_artifact.coverage_pct, 2),
            "stage3_full_excluded_pct": round(100.0 - full_artifact.coverage_pct, 2),
            "stage3_full_word_perplexity": round(full_ppl, 4),
            "stage3_full_bigram_baseline": round(full_bar, 4),
            "stage3_full_eval_positions": full_positions,
            "stage3_full_gain_nats": rungs[-1]["gain_nats"],
            "stage3_full_gain_shared": rungs[-1]["gain_nats_shared_denominator"],
            # Two quantities, two labels. This was ONE label spent two ways:
            # `docs/manuscript/03_methods.md` reads it as the word factor's dense
            # transition (which is what it computes) while
            # `docs/manuscript/98_supplement_complexity.md` read it as the whole live agent,
            # which is larger by the emission, the position factor and the
            # priors. A label with two readings is a number that is wrong in
            # one of the places it appears, and neither sentence looks wrong.
            "stage3_full_word_factor_gb": round(n_full * n_full * 4 / 1e9, 2),
            "stage3_full_agent_gb": round(
                agent_memory_bytes_from_shape(N_OBS, (n_full, full_artifact.n_positions)) / 1e9,
                2,
            ),
            "stage3_max_agent_vocab": int((MAX_AGENT_VOCAB_BYTES // 4) ** 0.5),
        }

    trie = measure_trie_cost(
        json.loads((PROJECT_ROOT / "vendor" / "cmudict_counts.json").read_text())
    )

    recorded: dict[str, float | int] = {
        # ── population and scope ──
        "stage3_vocab_size": len(vocab),
        "stage3_coverage_pct": round(artifact.coverage_pct, 2),
        "stage3_eval_tokens": len(tokens),
        "stage3_eval_positions": n_emissions,
        "stage3_position_max_word_len": artifact.max_word_len,
        "stage3_observed_max_word_len": artifact.observed_max_word_len,
        "stage3_train_tokens": artifact.provenance["train_tokens"],
        "stage3_train_types": artifact.provenance["train_types"],
        "stage3_source_documents": artifact.provenance["documents"],
        "stage3_train_documents": artifact.provenance["train_documents"],
        "stage3_heldout_documents": artifact.provenance["heldout_documents"],
        "stage3_document_overlap": len(
            set(artifact.train_document_indices) & set(artifact.heldout_document_indices)
        ),
        "stage3_source_categories": len(set(artifact.document_categories)),
        "stage3_train_categories": len(
            {artifact.document_categories[index] for index in artifact.train_document_indices}
        ),
        "stage3_heldout_categories": len(
            {artifact.document_categories[index] for index in artifact.heldout_document_indices}
        ),
        # ── the rejected encoding (decision 13) ──
        "stage3_trie_nodes_g2p": trie["nodes"],
        "stage3_trie_edges_g2p": trie["edges"],
        "stage3_trie_dense_gb": round(trie["dense_gb"], 1),
        "stage3_trie_density_pct": round(trie["density_pct"], 4),
        "stage3_trie_dense_full_tb": round(trie["dense_full_tb"], 2),
        # ── the comparator ladder (decision 2) ──
        "stage3_uniform_baseline": round(uniform_ppl, 4),
        "stage3_unigram_baseline": round(unigram_ppl, 4),
        "stage3_bigram_baseline": round(bar_ppl, 4),
        "stage3_word_perplexity": round(model_ppl, 4),
        "stage3_ceiling_perplexity": round(ceiling_ppl, 4),
        "stage3_gain_nats": round(gain_nats, 4),
        "stage3_headroom_nats": round(headroom_nats, 4),
        "stage3_signal_captured_pct": round(100.0 * gain_nats / available, 2),
        # ── the declared gate's verdict, re-derived ──
        "stage3_declared_gate": DECLARED_GATE_PERPLEXITY,
        "stage3_declared_gate_slack": round(DECLARED_GATE_PERPLEXITY / bar_ppl, 1),
        "stage3_word_gate_sesoi_nats": STAGE3_WORD_GATE_SESOI_NATS,
        # ── the identity that pins the level (decision 13) ──
        "stage3_path_identity_max_abs_err": float(f"{identity_err:.3g}"),
        "stage3_path_identity_words": args.equivalence_words,
        # ── mis-wirings (decision 14) ──
        **recorded_miswirings,
        **full,
    }

    from ailm.claims import ClaimBatch

    ledger = load_ledger(PROJECT_ROOT)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in recorded]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=recorded, run_id=RUN_ID, source=SOURCE)],
    )

    print(f"Recorded {len(recorded)} Stage 3 claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\nGate: model {model_ppl:.4f} vs bar {bar_ppl:.4f} over {n_emissions:,} "
        f"emissions from {len(tokens):,} tokens ({artifact.coverage_pct:.2f}% "
        f"of held-out tokens are in vocabulary)."
    )
    print(
        f"The declared 'perplexity < {DECLARED_GATE_PERPLEXITY:.0f}' is VACUOUS: "
        f"the bar itself clears it by {DECLARED_GATE_PERPLEXITY / bar_ppl:.0f}x."
    )
    print(
        f"Path identity holds to {identity_err:.3e} nats, so the closed-form "
        f"full-set statistics describe the pymdp agent."
    )
    if full:
        print(
            f"\nFull-dictionary configuration: {full['stage3_full_vocab_size']:,} "
            f"types reaching {full['stage3_full_coverage_pct']}% coverage, "
            f"perplexity {full['stage3_full_word_perplexity']} against a bar of "
            f"{full['stage3_full_bigram_baseline']}. The pymdp AGENT is not "
            f"instantiable there ({full['stage3_full_word_factor_gb']} GB word "
            f"factor, {full['stage3_full_agent_gb']} GB whole agent, "
            f"cap ~{full['stage3_max_agent_vocab']:,} types); the closed form is "
            f"exact at any size, so the model is measurable where the agent is not."
        )
    survivors = [
        arm.display_name.replace("\n", " ")
        for arm in miswiring_registry
        if arm.claim_label is not None and recorded[arm.claim_label] < bar_ppl
    ]
    finite_count = sum(arm.has_finite_score for arm in miswiring_registry)
    print(
        f"{len(survivors)} of {finite_count} finite mis-wirings still beat the bar "
        f"({', '.join(survivors)}) "
        f"— the bar is necessary, not sufficient; the path identity is what pins them."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
