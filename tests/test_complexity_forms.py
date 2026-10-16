"""The two collected closed forms, tested as arithmetic rather than as drawing.

``step_ops_terms`` and ``agent_memory_bytes_from_shape`` used to live in
``ailm.figures.complexity``, where the only thing that exercised them was a
figure generator. A form reached only through a plot is tested only by whether
the plot renders, and a plot renders perfectly with a wrong constant in it.
They are library code — they raise rather than draw — so they now live in
``ailm.complexity`` and are checked here against two independent routes:

* ``step_ops_terms`` prices one position by *cause* (the ``A`` row, the ``B``
  push, the decode argmax); ``filtering_ops_theory`` totals a workload by
  operation *category* (mul/add/div/log/cmp). Neither is derived from the
  other, so requiring the per-position terms to rebuild the workload total is
  a real constraint on both.
* ``agent_memory_bytes_from_shape`` composes an agent's bytes structurally
  from its ``(O, [S_p], A)`` shape; ``spec_memory_bytes_theory`` blindly sums
  the declared dimensions of whatever role-carrying variables a parsed GNN
  spec happens to carry. Agreement means the structural model got both *which*
  arrays exist and *how big* each one is right.

Every equality here is paired with a NEGATIVE CONTROL: a named, plausible
mistake, applied to the form, that must produce a different — and therefore
wrong — answer. Without those, each test would only be checking the current
output against the current output, which no defect can fail.

Nothing here writes a claim or reads the ledger; the recorded
``complexity_*`` labels are reconciled in ``tests/test_complexity.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from ailm.complexity import (
    FLOAT_BYTES,
    agent_memory_bytes_from_shape,
    filtering_ops_theory,
    spec_memory_bytes_theory,
    step_ops_terms,
)
from ailm.gnn import load_spec

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPEC_SUFFIX = ".gnn.md"

#: Every shipped level spec. Derived from the directory rather than listed, so
#: a new level joins the sweep instead of quietly escaping it; the sweep's own
#: non-emptiness is asserted by ``test_the_model_spec_sweep_is_not_empty``,
#: because a glob that matches nothing passes every test parametrised on it.
MODEL_SPEC_NAMES: tuple[str, ...] = tuple(
    sorted(
        path.name[: -len(SPEC_SUFFIX)]
        for path in (PROJECT_ROOT / "gnn" / "models").glob("*" + SPEC_SUFFIX)
    )
)

#: Filtering workloads exercised by the per-position decomposition:
#: ``(n_states, sequence lengths, decode)``. The single-sequence and
#: many-short-sequence cases matter separately: the push is withheld once per
#: sequence, so a workload of many length-1 sequences pays no push at all and
#: a workload of one long sequence pays nearly one per position.
WORKLOADS: tuple[tuple[int, tuple[int, ...], bool], ...] = (
    (2, (1,), False),
    (3, (5,), False),
    (3, (5,), True),
    (8, (1, 1, 1, 1), False),
    (8, (4, 7, 2, 9), True),
    (39, (12, 3), False),
    (1024, (2, 2), True),
)

#: The workloads that pay at least one push. Every sequence of length one
#: withholds its only push, so a workload made entirely of them leaves the push
#: term unconstrained: no total could see it wrong. The push-term negative
#: control is parametrized over this tuple so the exclusion holds by
#: construction rather than by a runtime skip, which the release profile forbids.
PUSH_WORKLOADS: tuple[tuple[int, tuple[int, ...], bool], ...] = tuple(
    workload for workload in WORKLOADS if sum(workload[1]) > len(workload[1])
)


# ── MUTANTS: the plausible mistakes each form has to survive ─────────────────


def _push_term_charging_S_squared_adds(n_states: int) -> int:
    """MUTANT: the ``B`` push priced at ``2S²`` instead of ``2S² − S``.

    Summing ``S`` products takes ``S - 1`` additions, not ``S``. Reading the
    matrix-vector product as "S² multiplies and S² adds" is the arithmetic
    slip this control catches; it inflates every filtering level by exactly
    ``S`` operations at every position that carries a push.
    """
    return 2 * n_states * n_states


def _emission_term_without_the_normalizing_divides(n_states: int) -> int:
    """MUTANT: a position priced at ``2S`` instead of ``3S``.

    The ``3S`` is ``S`` likelihood multiplies, ``S - 1`` adds for the
    normalizer, one evidence logarithm and ``S`` divides that renormalize the
    belief. Pricing only the likelihood row and its sum — the mistake of
    forgetting that a *scaled* alpha recursion renormalizes at every step —
    drops those ``S`` divides.
    """
    return 2 * n_states


def _memory_bytes_without_the_policy_prior(
    n_obs: int,
    factor_sizes: Sequence[int],
    *,
    n_actions: int = 1,
) -> int:
    """MUTANT: the memory form with pymdp's implicit ``E`` term removed.

    pymdp materialises a policy prior for every agent, including the
    single-action recognition levels where it is one float32. Omitting it
    leaves the form four bytes short at each of those levels — small enough to
    look like rounding, and the same omission at a level with a real action
    set is the whole ``E`` array (``docs/decisions.md`` entry 21).

    Written as a delta from the true form so the ONLY difference between them
    is the named term.
    """
    truth = agent_memory_bytes_from_shape(n_obs, factor_sizes, n_actions=n_actions)
    return truth - n_actions * FLOAT_BYTES


def _memory_bytes_with_summed_emission(
    n_obs: int,
    factor_sizes: Sequence[int],
    *,
    n_actions: int = 1,
) -> int:
    """MUTANT: the emission priced at ``O·∑ₚ Sₚ`` instead of ``O·∏ₚ Sₚ``.

    ``A`` is dense over the JOINT of every hidden factor, so it carries the
    product. Swapping in the sum is the mistake of pricing the emission as if
    it were factored like ``B`` is, and it under-prices a multi-factor level
    by orders of magnitude — at the word level it is the difference between
    the emission dominating the agent and looking negligible.
    """
    sizes = [int(size) for size in factor_sizes]
    truth = agent_memory_bytes_from_shape(n_obs, sizes, n_actions=n_actions)
    swap = n_obs * (sum(sizes) - math.prod(sizes))
    return truth + swap * FLOAT_BYTES


def _memory_bytes_with_a_joint_transition(
    n_obs: int,
    factor_sizes: Sequence[int],
    *,
    n_actions: int = 1,
) -> int:
    """MUTANT: transitions priced at ``A·(∏ₚ Sₚ)²`` instead of ``A·∑ₚ Sₚ²``.

    This is the unfactored counterfactual: one dense transition over the joint
    state rather than one per factor. It is what the factorization buys, so a
    form that cannot tell the two apart cannot price the factorization either.
    """
    sizes = [int(size) for size in factor_sizes]
    truth = agent_memory_bytes_from_shape(n_obs, sizes, n_actions=n_actions)
    joint = math.prod(sizes)
    swap = n_actions * (joint * joint - sum(size * size for size in sizes))
    return truth + swap * FLOAT_BYTES


# ── OPERATIONS: the per-position split must rebuild the workload total ───────


def _workload_total_from_step_terms(
    n_states: int,
    lengths: Sequence[int],
    *,
    decode: bool,
    push_term: int | None = None,
    emission_term: int | None = None,
    pushes: int | None = None,
) -> int:
    """Total a workload from the per-position terms, with substitutions.

    The keyword arguments exist so a negative control can replace exactly one
    term and leave the rest of the accounting identical.
    """
    terms = step_ops_terms(n_states, decode=decode)
    emission = terms["emission"] if emission_term is None else emission_term
    push = terms["push"] if push_term is None else push_term
    positions = sum(lengths)
    pushed = positions - len(lengths) if pushes is None else pushes
    return positions * emission + pushed * push + positions * terms["decode"]


@pytest.mark.parametrize(("n_states", "lengths", "decode"), WORKLOADS)
def test_the_per_position_terms_rebuild_the_workload_total(n_states, lengths, decode):
    """The by-cause split and the by-category total are the same cost model.

    ``step_ops_terms`` and ``filtering_ops_theory`` are written independently —
    one splits a position into A/B/decode, the other totals a workload into
    mul/add/div/log/cmp — so this equality is a constraint, not a tautology.
    """
    expected = filtering_ops_theory(n_states, list(lengths), decode=decode).total
    assert expected > 0, "a workload that costs nothing cannot falsify a cost model"
    assert _workload_total_from_step_terms(n_states, lengths, decode=decode) == expected


@pytest.mark.parametrize(("n_states", "lengths", "decode"), PUSH_WORKLOADS)
def test_a_push_charging_S_squared_adds_breaks_the_total(n_states, lengths, decode):
    """NEGATIVE CONTROL: ``2S²`` for ``2S² − S`` must change the answer.

    Workloads that pay no push at all — every sequence of length one withholds
    its only push — are excluded by construction through ``PUSH_WORKLOADS``,
    because there the term cannot be wrong in a way any total could see.
    """
    assert sum(lengths) > len(lengths), "PUSH_WORKLOADS admitted a push-free workload"
    expected = filtering_ops_theory(n_states, list(lengths), decode=decode).total
    mutated = _workload_total_from_step_terms(
        n_states,
        lengths,
        decode=decode,
        push_term=_push_term_charging_S_squared_adds(n_states),
    )
    assert mutated != expected, (
        f"S = {n_states}: charging S² adds for the B push still totals "
        f"{expected}, so the S(S-1) in the push term is not load-bearing"
    )


@pytest.mark.parametrize(("n_states", "lengths", "decode"), WORKLOADS)
def test_a_position_priced_without_the_normalizing_divides_breaks_the_total(
    n_states, lengths, decode
):
    """NEGATIVE CONTROL: ``2S`` for ``3S`` must change the answer.

    The emission term is paid at every position of every workload, so unlike
    the push there is no shape that excuses it.
    """
    expected = filtering_ops_theory(n_states, list(lengths), decode=decode).total
    mutated = _workload_total_from_step_terms(
        n_states,
        lengths,
        decode=decode,
        emission_term=_emission_term_without_the_normalizing_divides(n_states),
    )
    assert mutated != expected, (
        f"S = {n_states}: dropping the S normalizing divides still totals "
        f"{expected}, so the scaled recursion's renormalization is unpriced"
    )


@pytest.mark.parametrize(("n_states", "lengths", "decode"), WORKLOADS)
def test_pushing_at_every_position_breaks_the_total(n_states, lengths, decode):
    """NEGATIVE CONTROL: the classic off-by-one, ``T`` pushes rather than ``T − n``.

    Pushing after the final observation predicts a symbol nothing scores. The
    per-position terms are correct here and only the amortisation is wrong,
    which is exactly the failure a stacked bar drawn over a recorded rate
    would otherwise hide.
    """
    expected = filtering_ops_theory(n_states, list(lengths), decode=decode).total
    mutated = _workload_total_from_step_terms(n_states, lengths, decode=decode, pushes=sum(lengths))
    assert mutated != expected, (
        f"S = {n_states}, lengths {tuple(lengths)}: charging a push at every "
        f"position still totals {expected}, so the withheld push is invisible"
    )


@pytest.mark.parametrize("n_states", [1, 2, 8, 39, 1024])
def test_the_decode_term_is_exactly_the_argmax_comparisons(n_states):
    """Decoding adds ``S - 1`` comparisons and touches nothing else."""
    plain = step_ops_terms(n_states, decode=False)
    decoding = step_ops_terms(n_states, decode=True)
    assert plain["decode"] == 0
    assert decoding["decode"] == n_states - 1
    assert decoding["emission"] == plain["emission"]
    assert decoding["push"] == plain["push"]


@pytest.mark.parametrize("n_states", [0, -1])
def test_step_ops_terms_rejects_a_non_positive_state_count(n_states):
    """A zero-state level would price at zero and read as a very cheap level."""
    with pytest.raises(ValueError, match="positive"):
        step_ops_terms(n_states, decode=False)


# ── MEMORY: the shape form against the spec route ────────────────────────────


def _declared_shape(spec_name: str) -> tuple[int, list[int], int]:
    """Read ``(O, [S_p], A)`` off a spec's declared array dimensions.

    Args:
        spec_name: A shipped model id such as ``"L3_word"``.

    Returns:
        Observation count, hidden-factor sizes, and control states.
    """
    spec = load_spec(spec_name)
    recognition = spec.variables_with_role("RecognitionMatrix")
    transitions = spec.variables_with_role("TransitionMatrix")
    assert len(recognition) == 1, (
        f"{spec_name} declares {len(recognition)} recognition matrices; the "
        f"single-modality shape form does not apply to it"
    )
    assert transitions, f"{spec_name} declares no transition matrix"
    n_obs = spec.variable_dims(recognition[0])[0]
    factor_sizes = [spec.variable_dims(name)[0] for name in transitions]
    action_axes = {
        spec.variable_dims(name)[2] if len(spec.variable_dims(name)) > 2 else 1
        for name in transitions
    }
    assert len(action_axes) == 1, (
        f"{spec_name} declares disagreeing control axes {sorted(action_axes)}; "
        f"one action count cannot price all of its transitions"
    )
    return n_obs, factor_sizes, action_axes.pop()


def test_the_model_spec_sweep_is_not_empty():
    """A glob that matched nothing would pass every test parametrised on it."""
    assert len(MODEL_SPEC_NAMES) >= 6, (
        f"only {len(MODEL_SPEC_NAMES)} model specs found under gnn/models; the "
        f"memory sweep below would be vacuous"
    )
    assert "L3_word" in MODEL_SPEC_NAMES, (
        "the only multi-factor level is missing, so the ∏-vs-∑ controls below "
        "would have nothing to distinguish"
    )


@pytest.mark.parametrize("spec_name", MODEL_SPEC_NAMES)
def test_the_shape_form_agrees_with_the_spec_route(spec_name):
    """Two independent derivations of the same bytes, to the last byte.

    ``spec_memory_bytes_theory`` sums whatever role-carrying variables the spec
    declares; the shape form asserts which arrays must exist and how big each
    is. Agreement pins both.
    """
    n_obs, factor_sizes, n_actions = _declared_shape(spec_name)
    from_spec = spec_memory_bytes_theory(load_spec(spec_name))
    from_shape = agent_memory_bytes_from_shape(n_obs, factor_sizes, n_actions=n_actions)
    assert from_spec > 0, "a zero-byte agent would make the agreement vacuous"
    assert from_shape == from_spec, (
        f"{spec_name}: shape form prices the agent at {from_shape:,} B, the "
        f"spec sums to {from_spec:,} B"
    )


@pytest.mark.parametrize("spec_name", MODEL_SPEC_NAMES)
def test_dropping_the_policy_prior_breaks_the_agreement(spec_name):
    """NEGATIVE CONTROL: pymdp's implicit ``E`` is load-bearing at every level."""
    n_obs, factor_sizes, n_actions = _declared_shape(spec_name)
    from_spec = spec_memory_bytes_theory(load_spec(spec_name))
    mutated = _memory_bytes_without_the_policy_prior(n_obs, factor_sizes, n_actions=n_actions)
    assert mutated != from_spec, (
        f"{spec_name}: omitting the policy prior still reconciles at "
        f"{from_spec:,} B, so the E term is unpriced and a level with a real "
        f"action set would go unpriced too"
    )
    assert from_spec - mutated == n_actions * FLOAT_BYTES


def test_summing_the_emission_factors_breaks_the_agreement():
    """NEGATIVE CONTROL: ``O·∑ₚ Sₚ`` for ``O·∏ₚ Sₚ`` at the multi-factor level."""
    n_obs, factor_sizes, n_actions = _declared_shape("L3_word")
    assert len(factor_sizes) > 1, "L3 stopped being multi-factor; this control is blind"
    from_spec = spec_memory_bytes_theory(load_spec("L3_word"))
    mutated = _memory_bytes_with_summed_emission(n_obs, factor_sizes, n_actions=n_actions)
    assert mutated != from_spec, (
        f"pricing the emission over the SUM of factor sizes still reconciles "
        f"at {from_spec:,} B, so the joint the emission is dense over is not "
        f"actually in the formula"
    )
    assert mutated < from_spec, "the swap should under-price, not over-price, the emission"


@pytest.mark.parametrize("spec_name", [name for name in MODEL_SPEC_NAMES if name != "L3_word"])
def test_the_emission_swap_is_invisible_at_a_single_factor_level(spec_name):
    """Why the control above needs L3: ``∑`` and ``∏`` coincide over one factor.

    This is not a second falsification — it is the boundary of the first one,
    asserted so that nobody later "strengthens" the control by parametrising it
    over every level and gets a suite that passes while proving nothing.
    """
    n_obs, factor_sizes, n_actions = _declared_shape(spec_name)
    assert len(factor_sizes) == 1, f"{spec_name} is multi-factor; move it to the control above"
    truth = agent_memory_bytes_from_shape(n_obs, factor_sizes, n_actions=n_actions)
    assert _memory_bytes_with_summed_emission(n_obs, factor_sizes, n_actions=n_actions) == truth


def test_a_joint_transition_breaks_the_agreement():
    """NEGATIVE CONTROL: ``A·(∏ₚ Sₚ)²`` for ``A·∑ₚ Sₚ²`` — the unfactored form.

    The gap this opens IS what the factorization buys, so it is asserted to be
    large rather than merely nonzero.
    """
    n_obs, factor_sizes, n_actions = _declared_shape("L3_word")
    from_spec = spec_memory_bytes_theory(load_spec("L3_word"))
    mutated = _memory_bytes_with_a_joint_transition(n_obs, factor_sizes, n_actions=n_actions)
    assert mutated != from_spec, (
        f"a dense transition over the joint state still reconciles at "
        f"{from_spec:,} B, so the form cannot tell factored from unfactored"
    )
    assert mutated > 10 * from_spec, (
        f"the unfactored transition costs {mutated:,} B against a factored "
        f"{from_spec:,} B — under an order of magnitude, which would make the "
        f"factorization argument the manuscript draws a weak one"
    )


def test_the_shape_form_rejects_a_level_without_hidden_factors():
    """No factors would price at ``O + A`` and read as a very cheap agent."""
    with pytest.raises(ValueError, match="no hidden factors"):
        agent_memory_bytes_from_shape(27, [])


@pytest.mark.parametrize(
    ("n_obs", "factor_sizes", "n_actions"),
    [(0, [4], 1), (27, [0], 1), (27, [4], 0), (-1, [4], 1), (27, [4, -3], 1)],
)
def test_the_shape_form_rejects_degenerate_sizes(n_obs, factor_sizes, n_actions):
    """A non-positive size makes every term below it meaningless, not small."""
    with pytest.raises(ValueError, match="positive"):
        agent_memory_bytes_from_shape(n_obs, factor_sizes, n_actions=n_actions)
