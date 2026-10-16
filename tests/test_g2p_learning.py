import pytest

from ailm.levels.g2p_learning import (
    _learn_agent,
    _oracle_decode_positions,
    _stride,
    run_g2p_emission_learning,
)
from ailm.levels.phoneme import N_PHONES


def test_latent_learning_moves_and_oracle_agrees(tiny_cmudict):
    train = [("cat", ["K", "AE", "T"]), ("dog", ["D", "AO", "G"])]
    test = [("cat", ["K", "AE", "T"])]
    result = run_g2p_emission_learning(train, test, learning_stride=1)
    assert result.max_movement > 0
    assert result.gold_max_abs_delta < 1e-5
    assert result.decoder_disagreements == 0


def test_invalid_learning_inputs_are_rejected():
    with pytest.raises(ValueError, match="both be non-empty"):
        run_g2p_emission_learning([], [("a", ["A"])])
    with pytest.raises(ValueError, match="both be non-empty"):
        run_g2p_emission_learning([("a", ["A"])], [])
    with pytest.raises(ValueError, match="positive"):
        run_g2p_emission_learning([("a", ["A"])], [("a", ["A"])], pseudocount=0)


def test_learning_helpers_cover_controls_and_alignment_errors():
    import numpy as np

    assert _stride([1, 2, 3, 4], 2) == [1, 3]
    with pytest.raises(ValueError, match="positive"):
        _learn_agent([("a", ["A"])], pseudocount=0, gold_phones=True)
    # A non-observable word is skipped rather than updating A.
    learned = _learn_agent([("_", ["A"])], pseudocount=1, gold_phones=True)
    assert learned.shape == (30, N_PHONES)
    transition = np.eye(N_PHONES)
    start = np.full(N_PHONES, 1 / N_PHONES)
    with pytest.raises(ValueError, match="no aligned"):
        _oracle_decode_positions(learned, transition, start, [("a", ["AA", "AA"])])
