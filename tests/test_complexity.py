"""Complexity claims: exact reconciliation of formulas against executed counts.

The project's complexity statements (``docs/manuscript/98_supplement_complexity.md``) are not
big-O adjectives: each is an exact closed-form operation-count or byte-count
formula, and each formula must reproduce, to the last integer, what an
instrumented execution of the shipped recursion actually did on the shipped
workload. This file carries the reconciliation on bounded real workloads, the
negative controls proving the reconciliation can fail — the classic off-by-one
formula (``T`` transition pushes rather than ``T - 1``), a dropped sequence, a
mis-stated decode flag — and the cross-checks binding the recorded
``complexity_*`` labels to the rest of the ledger.

Nothing here writes a claim; ``scripts/record_complexity_claims.py`` is the
single writer for every ``complexity_*`` label and refuses to record when its
own reconciliation fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.complexity import (
    FLOAT_BYTES,
    OpCount,
    agent_memory_bytes,
    count_filtering_ops,
    count_word_path_ops,
    dense_joint_transition_bytes,
    filtering_ops_theory,
    spec_memory_bytes_theory,
    word_factored_transition_bytes,
    word_path_ops_theory,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Every label the recorder writes; a missing one means the manuscript section
#: cites a measurement that was never taken.
COMPLEXITY_LABELS = (
    "complexity_l1_states",
    "complexity_l1_sequences",
    "complexity_l1_positions",
    "complexity_l1_ops_total",
    "complexity_l1_ops_per_position",
    "complexity_l2a_states",
    "complexity_l2a_sequences",
    "complexity_l2a_positions",
    "complexity_l2a_ops_total",
    "complexity_l2a_ops_per_position",
    "complexity_l2b_states",
    "complexity_l2b_sequences",
    "complexity_l2b_positions",
    "complexity_l2b_ops_total",
    "complexity_l2b_ops_per_position",
    "complexity_l3_tokens",
    "complexity_l3_ops_total",
    "complexity_l3_ops_per_token",
    "complexity_l4_states",
    "complexity_l4_sequences",
    "complexity_l4_positions",
    "complexity_l4_ops_total",
    "complexity_l4_ops_per_position",
    "complexity_l1_memory_bytes",
    "complexity_l1g_memory_bytes",
    "complexity_l2a_memory_bytes",
    "complexity_l2b_memory_bytes",
    "complexity_l3_memory_bytes",
    "complexity_l4_memory_bytes",
    "complexity_l3_position_states",
    "complexity_l3_dense_joint_gb",
    "complexity_l3_factored_transition_mb",
    "complexity_l3_joint_over_factored",
)


@pytest.fixture(scope="module")
def ledger():
    return load_ledger(PROJECT_ROOT)


@pytest.fixture(scope="module")
def case_model():
    """The full Stage 4 derived-population workload — small enough to run whole."""
    from ailm.levels.case import (
        build_case_emission,
        build_case_start,
        build_case_transition,
        load_case_artifact,
    )

    artifact = load_case_artifact()
    return {
        "emission": build_case_emission(artifact),
        "transition": build_case_transition(artifact),
        "start": build_case_start(artifact),
        "sequences": [
            [artifact.upos_to_idx[upos] for upos, _ in sequence] for sequence in artifact.test
        ],
    }


# ── THEORY == COUNTED, on real workloads ─────────────────────────────────────


def test_case_level_formula_reproduces_the_executed_count(case_model):
    """Full current L4 workload, with one real reset per artifact sequence."""
    counted, log_evidence = count_filtering_ops(
        case_model["emission"],
        case_model["transition"],
        case_model["start"],
        case_model["sequences"],
        decode=True,
    )
    theory = filtering_ops_theory(8, [len(s) for s in case_model["sequences"]], decode=True)
    assert counted == theory
    assert log_evidence < 0.0, "no evidence accumulated; the run was vacuous"


def test_letter_level_formula_reproduces_the_executed_count():
    """A single unsegmented stream — the n_sequences=1 edge of the formula."""
    from ailm.evaluation import STAGE1_EVALUATION_TEXT
    from ailm.levels.letter import (
        ALPHA_TO_IDX,
        N_SYMBOLS,
        build_letter_transition,
        load_brown_counts,
        normalize_text,
    )

    letters = [ALPHA_TO_IDX[c] for c in normalize_text(STAGE1_EVALUATION_TEXT) if c in ALPHA_TO_IDX]
    bigram = np.asarray(build_letter_transition(load_brown_counts()), dtype=np.float64)[0]
    counted, log_evidence = count_filtering_ops(
        np.eye(N_SYMBOLS), bigram, np.full(N_SYMBOLS, 1.0 / N_SYMBOLS), [letters]
    )
    assert counted == filtering_ops_theory(N_SYMBOLS, [len(letters)])
    assert log_evidence < 0.0


def test_word_level_scoring_is_two_ops_per_token():
    """The Stage 3 collapse: Θ(1) per token, independent of vocabulary size."""
    from ailm.levels.word import build_word_prior, load_word_artifact

    artifact = load_word_artifact()
    tokens = artifact.eval_tokens[:2000]
    counted, total_logp = count_word_path_ops(
        tokens, artifact.word_to_index, build_word_prior(artifact)
    )
    assert counted == word_path_ops_theory(len(tokens))
    assert counted.total == 2 * len(tokens)
    assert total_logp < 0.0


# ── NEGATIVE CONTROLS: the reconciliation can fail ───────────────────────────


def test_the_off_by_one_formula_is_rejected(case_model):
    """The classic wrong formula — a push after the final observation.

    ``T`` pushes per sequence instead of ``T - 1`` adds exactly one ``S²``
    multiply block per sequence. If the instrumented count agreed with THAT,
    the reconciliation would be decorative.
    """
    counted, _ = count_filtering_ops(
        case_model["emission"],
        case_model["transition"],
        case_model["start"],
        case_model["sequences"],
        decode=True,
    )
    lengths = [len(s) for s in case_model["sequences"]]
    right = filtering_ops_theory(8, lengths, decode=True)
    wrong = OpCount(
        mul=right.mul + len(lengths) * 8 * 8,
        add=right.add + len(lengths) * 8 * 7,
        div=right.div,
        log=right.log,
        cmp=right.cmp,
    )
    assert counted != wrong
    assert counted == right


def test_a_dropped_sequence_is_rejected(case_model):
    """Workload accounting is part of the claim, not bookkeeping."""
    counted, _ = count_filtering_ops(
        case_model["emission"],
        case_model["transition"],
        case_model["start"],
        case_model["sequences"],
    )
    lengths = [len(s) for s in case_model["sequences"]]
    assert counted != filtering_ops_theory(8, lengths[:-1])


def test_a_mis_stated_decode_flag_is_rejected(case_model):
    """Counting a decoder as a filter (or vice versa) must not reconcile."""
    counted, _ = count_filtering_ops(
        case_model["emission"],
        case_model["transition"],
        case_model["start"],
        case_model["sequences"],
        decode=True,
    )
    lengths = [len(s) for s in case_model["sequences"]]
    assert counted != filtering_ops_theory(8, lengths, decode=False)


# ── MEMORY: the spec prices the live agent, to the byte ──────────────────────


@pytest.mark.parametrize(
    "spec_name",
    [
        "L1_letter",
        "L1g_letter_generation",
        "L2a_phoneme_sequence",
        "L2b_grapheme_to_phoneme",
        "L3_word",
        "L4_case",
    ],
)
def test_spec_memory_formula_equals_live_agent_bytes(spec_name):
    from ailm.gnn import load_spec
    from ailm.spec_agent import build_agent_from_spec

    predicted = spec_memory_bytes_theory(load_spec(spec_name))
    measured = agent_memory_bytes(build_agent_from_spec(spec_name))
    assert predicted == measured, (
        f"{spec_name}: spec says {predicted:,} B, live agent occupies {measured:,} B"
    )


def test_the_memory_formula_carries_the_implicit_policy_prior():
    """NON-VACUITY: pymdp materialises E even for a single-action agent.

    Dropping that one-scalar term from the formula must break the equality —
    otherwise the formula could omit any term the measured side also omitted,
    and a real E at a controlled level would be unpriced.
    """
    from ailm.gnn import load_spec
    from ailm.spec_agent import build_agent_from_spec

    predicted = spec_memory_bytes_theory(load_spec("L4_case"))
    measured = agent_memory_bytes(build_agent_from_spec("L4_case"))
    assert predicted == measured
    assert predicted - FLOAT_BYTES != measured, (
        "removing the implicit policy-prior term still reconciles, so the "
        "term is not load-bearing and the formula is not exact"
    )


# ── LEDGER: the recorded claims are the ones these formulas produce ──────────


def test_every_complexity_label_is_recorded(ledger):
    missing = [label for label in COMPLEXITY_LABELS if not ledger.verify_label(label)]
    assert missing == [], (
        f"unrecorded complexity labels {missing}; run scripts/record_complexity_claims.py"
    )


def test_recorded_case_count_matches_a_fresh_execution(case_model, ledger):
    """Re-derive the smallest recorded workload end to end."""
    counted, _ = count_filtering_ops(
        case_model["emission"],
        case_model["transition"],
        case_model["start"],
        case_model["sequences"],
        decode=True,
    )
    assert ledger.get("complexity_l4_ops_total")["value"] == counted.total
    assert ledger.get("complexity_l4_positions")["value"] == sum(
        len(s) for s in case_model["sequences"]
    )
    assert ledger.get("complexity_l4_sequences")["value"] == len(case_model["sequences"])


def test_recorded_totals_agree_with_recorded_rates(ledger):
    """ops_per_position is a rounded quotient of the two exact integers."""
    for level in ("l1", "l2a", "l2b", "l4"):
        total = ledger.get(f"complexity_{level}_ops_total")["value"]
        positions = ledger.get(f"complexity_{level}_positions")["value"]
        rate = ledger.get(f"complexity_{level}_ops_per_position")["value"]
        assert rate == round(total / positions, 1), level


def test_complexity_workloads_are_the_gates_workloads(ledger):
    """The counted populations must be the ones the accuracy gates score.

    A complexity number over a private convenience workload would be
    uninterpretable next to the accuracy numbers; these bindings keep the two
    sets of claims about the same populations.
    """
    assert (
        ledger.get("complexity_l2b_positions")["value"]
        == ledger.get("stage2_g2p_eval_positions")["value"]
    )
    assert ledger.get("complexity_l3_tokens")["value"] == ledger.get("stage3_eval_tokens")["value"]
    assert ledger.get("complexity_l3_ops_per_token")["value"] == 2.0
    assert (
        ledger.get("complexity_l1_states")["value"] == 30
        and ledger.get("complexity_l2a_states")["value"] == 39
    )


def test_the_counterfactual_prices_agree_with_stage3(ledger):
    """The dense-joint and factored bytes must match the Stage 3 story.

    ``stage3_full_agent_gb`` was derived independently in ``ailm.levels.word``
    from the ``V²`` instantiation arithmetic; the factored formula here adds
    only the negligible ``P²`` term, so the two must agree to well under a
    percent at the full-dictionary vocabulary.
    """
    from ailm.levels.word import load_word_artifact

    artifact = load_word_artifact()
    n_vocab, n_positions = len(artifact.vocab), artifact.n_positions

    dense = dense_joint_transition_bytes(n_vocab, n_positions)
    factored = word_factored_transition_bytes(n_vocab, n_positions)
    assert ledger.get("complexity_l3_dense_joint_gb")["value"] == round(dense / 1e9, 3)
    assert ledger.get("complexity_l3_factored_transition_mb")["value"] == round(factored / 1e6, 1)
    assert ledger.get("complexity_l3_joint_over_factored")["value"] == round(dense / factored, 1)
    # The ratio is n_positions² up to the +P² correction in the denominator.
    assert abs(dense / factored - n_positions**2) < 1.0

    full_vocab = ledger.get("stage3_full_vocab_size")
    word_factor_gb = ledger.get("stage3_full_word_factor_gb")
    agent_gb = ledger.get("stage3_full_agent_gb")
    if full_vocab is None or word_factor_gb is None or agent_gb is None:
        pytest.skip("full-dictionary stage3 claims not recorded in this checkout")

    # `V² · 4` is the WORD FACTOR, not the agent. These were one label spent two
    # ways — the word level's prose read it as the transition (correct), the
    # complexity section read it as the live agent (larger by the emission, the
    # position factor and the priors). Neither sentence looked wrong, and this
    # assertion could only ever have held for one of them.
    n_vocab_full = int(full_vocab["value"])
    v_squared_gb = (n_vocab_full**2) * FLOAT_BYTES / 1e9
    assert (
        abs(v_squared_gb - float(word_factor_gb["value"])) / float(word_factor_gb["value"]) < 0.01
    )

    # NEGATIVE CONTROL: the two must NOT be equal, or splitting the label bought
    # nothing and the ambiguity would return the next time someone merged them.
    assert float(agent_gb["value"]) > float(word_factor_gb["value"]), (
        "the whole agent is not larger than its word factor; the two labels have "
        "collapsed back into one quantity"
    )


# ── UNIT BEHAVIOR of the cost model ─────────────────────────────────────────


def test_opcount_arithmetic():
    a = OpCount(mul=1, add=2, div=3, log=4, exp=5, cmp=6)
    b = OpCount(mul=10)
    assert (a + b).mul == 11
    assert a.total == 21
    assert a.as_dict()["total"] == 21


def test_theory_rejects_degenerate_workloads():
    with pytest.raises(ValueError, match="positive"):
        filtering_ops_theory(0, [3])
    with pytest.raises(ValueError, match="empty workload"):
        filtering_ops_theory(5, [])
    with pytest.raises(ValueError, match="at least one position"):
        filtering_ops_theory(5, [3, 0])
    with pytest.raises(ValueError, match="empty workload"):
        word_path_ops_theory(0)


def test_counter_rejects_shape_disagreement():
    with pytest.raises(ValueError, match="does not span"):
        count_filtering_ops(np.eye(3), np.eye(4), np.full(4, 0.25), [[0]])
    with pytest.raises(ValueError, match="is not"):
        count_filtering_ops(np.eye(4), np.eye(3), np.full(4, 0.25), [[0]])
    with pytest.raises(ValueError, match="empty workload"):
        count_filtering_ops(np.eye(3), np.eye(3), np.full(3, 1 / 3), [])


def test_word_counter_enforces_the_closed_vocabulary():
    with pytest.raises(ValueError, match="out of vocabulary"):
        count_word_path_ops(["zzz"], {"a": 0}, np.array([1.0]))
    with pytest.raises(ValueError, match="empty"):
        count_word_path_ops([], {"a": 0}, np.array([1.0]))


def test_memory_helpers_reject_degenerate_sizes():
    with pytest.raises(ValueError, match="positive"):
        dense_joint_transition_bytes(0, 4)
    with pytest.raises(ValueError, match="positive"):
        word_factored_transition_bytes(4, 0)


def test_memory_formula_refuses_an_unpinned_spec():
    from ailm.gnn import parse_gnn_string

    spec = parse_gnn_string(
        "## ModelName\n\nBare\n\n## StateSpaceBlock\n\nx[3,type=float]\n\n"
        "## ActInfOntologyAnnotation\n\nx=HiddenState\n",
        origin="<bare>",
    )
    with pytest.raises(ValueError, match="no materialised roles"):
        spec_memory_bytes_theory(spec)


# ── The factorization's exponent belongs to the storage ──────────────────────


def test_the_factorization_buys_P_under_sparse_storage_and_P_squared_under_dense():
    """POSITIVE: the two pricing routes disagree by exactly one factor of P.

    The manuscript reported the dense ratio and read it as "the factorization
    buys precisely the square of the position-factor size." The arithmetic was
    right and the reading was wrong: both Stage 3 transitions are deterministic
    maps with one nonzero per source state, so under any storage that does not
    materialize structural zeros the joint costs a factor of P more, not P².
    One factor of P is bought by factoring; the other is bought by storing
    densely.

    NEGATIVE CONTROL: the two routes must not agree. If the sparse ratio also
    came out at P², there would be nothing to re-attribute and this test would
    be asserting a distinction that does not exist.
    """
    from ailm.complexity import joint_over_factored_ratio

    for n_vocab, n_positions in ((2000, 16), (8000, 16), (38203, 16)):
        dense = joint_over_factored_ratio(n_vocab, n_positions)
        sparse = joint_over_factored_ratio(n_vocab, n_positions, sparse=True)

        # Dense is P², sparse is P, each up to the term the other size adds.
        assert dense == pytest.approx(n_positions**2, rel=1e-3)
        assert sparse == pytest.approx(n_positions, rel=0.01)

        # NEGATIVE CONTROL: they differ, and by the position-factor size.
        assert dense / sparse == pytest.approx(n_positions, rel=0.02)


def test_the_sparse_ratio_converges_to_P_as_the_vocabulary_grows():
    """The claim is asymptotic, so it is checked asymptotically.

    `V·P / (V + P)` approaches `P` from below. A ratio that drifted away from P
    with vocabulary would mean the deterministic-map argument had a size
    dependence the prose does not state.
    """
    from ailm.complexity import joint_over_factored_ratio

    n_positions = 16
    errors = [
        abs(joint_over_factored_ratio(n_vocab, n_positions, sparse=True) - n_positions)
        for n_vocab in (500, 2000, 8000, 32000)
    ]
    assert errors == sorted(errors, reverse=True), (
        f"the sparse ratio does not converge monotonically to P: {errors}"
    )
    assert errors[-1] < 0.01


def test_the_nonzero_counts_are_one_per_source_state():
    """The whole re-attribution rests on determinism, so determinism is checked.

    NEGATIVE CONTROL: the dense byte counts must NOT equal the nonzero counts,
    or the two routes would be the same route under different names.
    """
    from ailm.complexity import (
        dense_joint_transition_bytes,
        dense_joint_transition_nonzeros,
        word_factored_transition_bytes,
        word_factored_transition_nonzeros,
    )

    n_vocab, n_positions = 2000, 16
    assert dense_joint_transition_nonzeros(n_vocab, n_positions) == n_vocab * n_positions
    assert word_factored_transition_nonzeros(n_vocab, n_positions) == n_vocab + n_positions

    assert dense_joint_transition_bytes(n_vocab, n_positions) > dense_joint_transition_nonzeros(
        n_vocab, n_positions
    )
    assert word_factored_transition_bytes(n_vocab, n_positions) > word_factored_transition_nonzeros(
        n_vocab, n_positions
    )

    for bad in ((0, 16), (2000, 0), (-1, 16)):
        with pytest.raises(ValueError):
            dense_joint_transition_nonzeros(*bad)
        with pytest.raises(ValueError):
            word_factored_transition_nonzeros(*bad)
