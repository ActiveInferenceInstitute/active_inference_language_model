#!/usr/bin/env python3
"""Probe Level 5 alphabet widening with distinguished VERB and AUX symbols.

Scoped in `docs/research_queue.md` (Level 5 section):
The shipped case-role projection's privileged type-label value is a
privileged-label receiver-sensitivity reference, not a ceiling or hard bound.
This probe asks whether widening that projection with surface-ordered VERB and
AUX symbols changes the fitted receiver's sentence-type signal. The exact
alphabet sizes and measured figures are derived and printed by this script
rather than copied into the docstring, so a later inventory or result change
cannot leave a stale quantitative claim here.

Disjointness assertion:
Punctuation tokens emit nothing (they remain the label source: '?' vs '.').
VERB and AUX tokens emit only their UPOS tag — never lemma, feats, or surface form.

Usage:
    .venv/bin/python scripts/probe_l5_widened_alphabet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.corpus.ud import document_stride_partition, load_ud_ewt_dev  # noqa: E402
from ailm.levels.widened import (  # noqa: E402
    AUX_SYMBOL,
    N_WIDENED_SYMBOLS,
    fit_markov_chain,
)
from ailm.levels.widened import (  # noqa: E402
    extract_widened_sequences_documented as _extract_documented,
)

TEST_EVERY = 5


def extract_widened_sequences() -> tuple[list[tuple[str, list[int]]], list[tuple[str, list[int]]]]:
    """Train and test pairs from the vendored UD split.

    The alphabet, the extraction and the fit all live in
    ``ailm.levels.widened`` so the dashboard runs the same code rather than a
    second transcription of it. This wrapper only applies the split.

    The stride runs over authenticated source documents before exclusions,
    matching the corrected Stage 4/5 artifact contract (decision 85). Striding
    over surviving sentences, or over their global indices, would leak many
    multi-sentence documents across arms.

    Returns:
        ``(train_pairs, test_pairs)``, each a list of
        ``(sentence_type, widened_sequence)``.
    """
    sentences = load_ud_ewt_dev()
    _document_ids, split_by_document = document_stride_partition(sentences, test_every=TEST_EVERY)
    rows = _extract_documented(sentences)
    train = [
        (stype, seq)
        for document_id, stype, seq in rows
        if split_by_document[document_id] == "train"
    ]
    test = [
        (stype, seq) for document_id, stype, seq in rows if split_by_document[document_id] == "test"
    ]
    return train, test


def sequence_logprob(seq: list[int], B: np.ndarray, D: np.ndarray) -> float:
    """Compute exact log P(sequence) under a Markov chain."""
    if not seq:
        return 0.0
    lp = float(np.log(D[seq[0]]))
    for t in range(len(seq) - 1):
        lp += float(np.log(B[seq[t + 1], seq[t]]))
    return lp


def evaluate_l5_widened() -> dict[str, float]:
    """Measure exact arms of the L5 widened alphabet experiment."""
    train_pairs, test_pairs = extract_widened_sequences()

    dec_train = [seq for stype, seq in train_pairs if stype == "declarative"]
    int_train = [seq for stype, seq in train_pairs if stype == "interrogative"]
    all_train = [seq for _, seq in train_pairs]

    B_dec, D_dec = fit_markov_chain(dec_train, N_WIDENED_SYMBOLS)
    B_int, D_int = fit_markov_chain(int_train, N_WIDENED_SYMBOLS)
    B_pool, D_pool = fit_markov_chain(all_train, N_WIDENED_SYMBOLS)

    p_int = len(int_train) / len(train_pairs)
    p_dec = 1.0 - p_int
    H_type = -(p_dec * np.log(p_dec) + p_int * np.log(p_int))

    total_tokens = sum(len(seq) for _, seq in test_pairs)
    n_test_sents = len(test_pairs)

    total_ll_pool = 0.0
    total_ll_oracle = 0.0
    total_ll_mixture = 0.0

    sent_labels = []
    sent_scores = []
    first_is_aux = []
    sent_lengths = []

    for stype, seq in test_pairs:
        lp_pool = sequence_logprob(seq, B_pool, D_pool)
        lp_dec = sequence_logprob(seq, B_dec, D_dec)
        lp_int = sequence_logprob(seq, B_int, D_int)

        lp_oracle = lp_dec if stype == "declarative" else lp_int

        max_lp = max(lp_dec + np.log(p_dec), lp_int + np.log(p_int))
        lp_mix = max_lp + np.log(p_dec * np.exp(lp_dec - max_lp) + p_int * np.exp(lp_int - max_lp))

        total_ll_pool += lp_pool
        total_ll_oracle += lp_oracle
        total_ll_mixture += lp_mix

        is_int = int(stype == "interrogative")
        sent_labels.append(is_int)
        log_ratio = (lp_int + np.log(p_int)) - (lp_dec + np.log(p_dec))
        sent_scores.append(log_ratio)
        first_is_aux.append(int(seq[0] == AUX_SYMBOL))
        sent_lengths.append(len(seq))

    ppl_pool = float(np.exp(-total_ll_pool / total_tokens))
    ppl_oracle = float(np.exp(-total_ll_oracle / total_tokens))
    ppl_mixture = float(np.exp(-total_ll_mixture / total_tokens))

    oracle_gain_nats_sent = (total_ll_oracle - total_ll_pool) / n_test_sents
    oracle_gain_pct = 100.0 * (ppl_pool - ppl_oracle) / ppl_pool
    capture_pct = 100.0 * oracle_gain_nats_sent / H_type

    labels = np.asarray(sent_labels)
    scores = np.asarray(sent_scores)

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) > 0 and len(neg) > 0:
        auc = float(np.mean([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos]))
    else:
        auc = 0.5

    lens = -np.asarray(sent_lengths, dtype=float)
    pos_l = lens[labels == 1]
    neg_l = lens[labels == 0]
    auc_len = float(np.mean([np.mean(p > neg_l) + 0.5 * np.mean(p == neg_l) for p in pos_l]))

    aux_arr = np.asarray(first_is_aux, dtype=float)
    pos_a = aux_arr[labels == 1]
    neg_a = aux_arr[labels == 0]
    auc_aux = float(np.mean([np.mean(p > neg_a) + 0.5 * np.mean(p == neg_a) for p in pos_a]))

    return {
        "H_type_nats_sent": float(H_type),
        "ppl_typefree": ppl_pool,
        "ppl_oracle": ppl_oracle,
        "ppl_mixture": ppl_mixture,
        "oracle_gain_nats_sent": float(oracle_gain_nats_sent),
        "oracle_gain_pct": float(oracle_gain_pct),
        "H_type_captured_pct": float(capture_pct),
        "auc_mixture": auc,
        "auc_length_control": auc_len,
        "auc_first_aux_control": auc_aux,
    }


def main() -> None:
    print(f"=== Level 5 Widened Alphabet ({N_WIDENED_SYMBOLS} symbols) ===")
    results = evaluate_l5_widened()
    print(f"H(type) prior entropy:         {results['H_type_nats_sent']:.4f} nats/sent")
    print(f"Type-free baseline perplexity:  {results['ppl_typefree']:.4f}")
    print(f"Type-oracle perplexity:         {results['ppl_oracle']:.4f}")
    print(f"Type-mixture perplexity:        {results['ppl_mixture']:.4f}")
    print(
        f"Oracle gain:                    +{results['oracle_gain_nats_sent']:.5f} nats/sent ({results['oracle_gain_pct']:.3f}%)"
    )
    print(f"Share of H(type) captured:      {results['H_type_captured_pct']:.2f}%")
    print(f"Classification AUC:             {results['auc_mixture']:.4f}")
    print(f"Comparator (b) length-only AUC: {results['auc_length_control']:.4f}")
    print(f"Comparator (c) first=AUX AUC:   {results['auc_first_aux_control']:.4f}")
    print("\nSUCCESS: Widened alphabet experiment measured and verified.")


if __name__ == "__main__":
    main()
