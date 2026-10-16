"""Tests for scripts/probe_l5_widened_alphabet.py."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_l5_widened_alphabet import (  # noqa: E402
    evaluate_l5_widened,
    extract_widened_sequences,
)


def test_widened_alphabet_extraction_and_disjointness():
    """Verify sequences contain symbols in 0..9 and obey disjointness."""
    train_pairs, test_pairs = extract_widened_sequences()
    assert len(train_pairs) > 500
    assert len(test_pairs) > 100

    all_symbols = set()
    for _, seq in train_pairs + test_pairs:
        assert len(seq) >= 2
        for s in seq:
            assert 0 <= s <= 9
            all_symbols.add(s)

    # All 10 symbols should appear (8 cases + VERB (8) + AUX (9))
    assert len(all_symbols) == 10


def test_widened_alphabet_metrics_and_controls():
    """Verify that the widened alphabet clears the comparator controls."""
    results = evaluate_l5_widened()
    assert results["ppl_oracle"] < results["ppl_typefree"]
    # Decision 112 registered the probe's own 99% bootstrap CI for this gain,
    # quoted at registration as [+0.039, +0.321] nats/sentence. A clean re-run
    # through the shipped code must land inside it: a point outside means the
    # pre-computation was contaminated and the proposal is withdrawn. The
    # probe's resampling unit was never committed, so only the registered
    # band is enforced here, not the recorded arm's document-resampled
    # intervals.
    from ailm.gates import stage5_widened_registered_gain_in_interval

    assert stage5_widened_registered_gain_in_interval(
        oracle_gain_nats_per_sentence=results["oracle_gain_nats_sent"]
    ), f"probe gain {results['oracle_gain_nats_sent']} left the registered interval"
    assert results["H_type_captured_pct"] > 40.0
    assert results["auc_mixture"] > 0.85
    # The controls are the assertions with content. An absolute AUC says little
    # on its own: a corpus where interrogatives are simply shorter would hand
    # any classifier a high number, which is what comparator (b) exists to rule
    # out, and "the first symbol is an auxiliary" is a one-feature rule that
    # needs no sequence model at all, which is (c).
    assert results["auc_mixture"] > results["auc_length_control"]
    assert results["auc_mixture"] > results["auc_first_aux_control"]


def test_the_probe_inherits_the_shipped_document_partition():
    """Every widened sentence inherits an arm from its raw source document."""
    from ailm.corpus.ud import document_stride_partition, load_ud_ewt_dev
    from ailm.levels.clause import load_clause_artifact
    from ailm.levels.widened import (
        extract_widened_sequences_documented,
        extract_widened_sequences_indexed,
    )
    from scripts.probe_l5_widened_alphabet import TEST_EVERY as PROBE_STRIDE

    ARTIFACT_STRIDE = load_clause_artifact().provenance["test_every"]
    assert PROBE_STRIDE == ARTIFACT_STRIDE

    sentences = load_ud_ewt_dev()
    _ids, split = document_stride_partition(sentences, test_every=ARTIFACT_STRIDE)
    rows = extract_widened_sequences_documented(sentences)
    expected_train = [(stype, seq) for doc, stype, seq in rows if split[doc] == "train"]
    expected_test = [(stype, seq) for doc, stype, seq in rows if split[doc] == "test"]
    train_pairs, test_pairs = extract_widened_sequences()
    assert train_pairs == expected_train
    assert test_pairs == expected_test

    train_docs = {doc for doc, _, _ in rows if split[doc] == "train"}
    test_docs = {doc for doc, _, _ in rows if split[doc] == "test"}
    assert train_docs.isdisjoint(test_docs)

    # POSITIVE CONTROL: the retired global-sentence stride leaks actual docs.
    indexed = extract_widened_sequences_indexed(sentences)
    old_train = {sentences[index].document_id for index, _, _ in indexed if index % ARTIFACT_STRIDE}
    old_test = {
        sentences[index].document_id for index, _, _ in indexed if index % ARTIFACT_STRIDE == 0
    }
    assert old_train & old_test
