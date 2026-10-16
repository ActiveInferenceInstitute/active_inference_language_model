"""Generation gate — policy selection as generation, and what admits it.

The claim
---------
AILM generates by **selecting actions that minimize expected free energy**, with
the letter level's ``C`` supplied top-down by the lexicon. This is the first
place in the project where a level's preferences come from another level rather
than being flat, and the first where the policy machinery is exercised at all.

The gate, and the metric that could not be the gate
---------------------------------------------------
The obvious metric — the fraction of generated tokens that are real words — is
**green by construction**. ``lexicon_preference`` floors any letter that would
leave the trie, so at any usable ``gamma`` the generator essentially cannot emit
a non-word. Measured, membership is 100% at every ``gamma`` from 0.5 upward *and*
at the degenerate one that emits a single word forever. A metric that cannot fail
is not a gate, however good the number looks; that finding is asserted here
(``test_membership_is_green_by_construction_and_so_is_not_the_gate``) rather than
quietly worked around.

The gate is **exact total-variation distance to the true word distribution**,
against the exact distribution induced by ancestral sampling from the letter
bigram — the strongest generator with the same letter statistics and no top-down
channel — paired with a sampled diversity floor.

The finding: an interior optimum
--------------------------------
Distance to the true distribution is U-shaped in the policy precision ``gamma``,
and both endpoints fail for *opposite* reasons: ``gamma -> 0`` reduces exactly to
the baseline (no goal), while the shipped artifact's high-precision greedy
prefix path collapses onto one word (no language). That greedy word need not be
the global prior mode. No recognition gate in this project is non-vacuous in
both directions; this one is, and the sweep is what shows it.

Claims discipline
-----------------
This file **verifies**; ``scripts/record_generation_claims.py`` writes.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ailm.claims import load_ledger
from ailm.control import ControlHarness
from ailm.generation import (
    GENERATION_DIVERSITY_FRACTION,
    SPACE_INDEX,
    GenerationResult,
    build_generative_letter_agent,
    efe_policy_posterior,
    evaluate_generation_gate,
    exact_ancestral_lexicon_total_variation,
    exact_lexicon_total_variation,
    exact_sweep_precision,
    generate_by_policy_selection,
    generate_by_sampling,
    lexicon_preference,
    lexicon_total_variation,
    score_generation,
    sweep_precision,
)
from ailm.gnn import describe_shapes
from ailm.levels.letter import N_SYMBOLS, build_letter_transition, load_brown_counts
from ailm.levels.word import build_word_prior, load_word_artifact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORD_ARTIFACT = PROJECT_ROOT / "vendor" / "brown_word_counts.json"
BROWN_COUNTS = PROJECT_ROOT / "vendor" / "brown_counts.json"
GENERATION_SOURCE = PROJECT_ROOT / "src" / "ailm" / "generation.py"

SEEDS = (0, 1, 2)
N_WORDS = 200


def _recorded_best_gamma() -> float:
    entry = load_ledger(PROJECT_ROOT).get("generation_best_gamma")
    if entry is None:
        raise RuntimeError(
            "generation_best_gamma is absent; run scripts/record_generation_claims.py"
        )
    return float(entry["value"])


BEST_GAMMA = _recorded_best_gamma()

pytestmark = pytest.mark.skipif(
    not (WORD_ARTIFACT.exists() and BROWN_COUNTS.exists()),
    reason="vendored artifacts absent; run the generators",
)


def _generation_gate_overview_defects(source: str) -> list[str]:
    """Reject the retired membership-gate account on the live module surface."""

    normalized = " ".join(source.split())
    defects: list[str] = []
    for retired in (
        "Why the metric is lexicon membership",
        "The gate is the fraction of generated word tokens that are real words",
    ):
        if retired in normalized:
            defects.append(retired)
    for required in (
        "Why the gate is distributional fidelity, not lexicon membership",
        "exact total-variation distance to the lexicon word distribution",
        "Membership is therefore reported only as a diagnostic",
    ):
        if required not in normalized:
            defects.append(f"missing:{required}")
    return defects


def test_live_generation_overview_names_the_actual_gate() -> None:
    source = GENERATION_SOURCE.read_text(encoding="utf-8")
    assert _generation_gate_overview_defects(source) == []

    # NEGATIVE CONTROL: the historical wording must make this guard fail.
    planted = source.replace(
        "Why the gate is distributional fidelity, not lexicon membership",
        "Why the metric is lexicon membership",
        1,
    )
    assert "Why the metric is lexicon membership" in _generation_gate_overview_defects(planted)


def habit_agent(habit: np.ndarray):
    """The shipped generative agent, for the conformance checks."""
    return build_generative_letter_agent(habit)


@pytest.fixture(scope="module")
def habit() -> np.ndarray:
    return np.asarray(build_letter_transition(load_brown_counts()))[0]


@pytest.fixture(scope="module")
def artifact():
    return load_word_artifact()


@pytest.fixture(scope="module")
def prior(artifact) -> np.ndarray:
    return build_word_prior(artifact)


@pytest.fixture(scope="module")
def baseline(habit, artifact) -> list[GenerationResult]:
    return [generate_by_sampling(habit, artifact, n_words=N_WORDS, seed=s) for s in SEEDS]


@pytest.fixture(scope="module")
def model(habit, artifact, prior) -> list[GenerationResult]:
    agent = build_generative_letter_agent(habit, gamma=BEST_GAMMA)
    return [
        generate_by_policy_selection(agent, artifact, prior, habit, n_words=N_WORDS, seed=s)
        for s in SEEDS
    ]


# ══ THE GATE ══════════════════════════════════════════════════════════════════


def test_generation_beats_the_sampling_baseline(habit, artifact, prior):
    """(1) Exact induced distributions must show the top-down improvement."""
    model_tv = exact_lexicon_total_variation(artifact, prior, habit, BEST_GAMMA)
    baseline_tv = exact_ancestral_lexicon_total_variation(artifact, prior, habit)
    assert model_tv < baseline_tv, (
        f"Generated distribution is {model_tv:.4f} from the lexicon's, against "
        f"{baseline_tv:.4f} for ancestral sampling. The top-down channel bought "
        f"nothing."
    )


def test_generation_clears_a_diversity_floor(model, baseline):
    """(2) Distance alone is not sufficient — see the degenerate case below."""
    scored = score_generation(model)
    baseline_scored = score_generation(baseline)
    gate = evaluate_generation_gate(
        model_tv=0.0,
        baseline_tv=1.0,
        model_distinct=scored["distinct_mean"],
        baseline_distinct=baseline_scored["distinct_mean"],
    )
    assert gate.diversity_passes, (
        f"Only {scored['distinct_mean']:.0f} distinct words against "
        f"{baseline_scored['distinct_mean']:.0f} for the baseline: the generator "
        f"has collapsed onto a handful of words."
    )
    assert gate.diversity_floor == pytest.approx(
        GENERATION_DIVERSITY_FRACTION * baseline_scored["distinct_mean"]
    )


def test_the_gate_is_live_under_the_control_harness(habit, artifact, prior):
    """The gate must DISCRIMINATE, not merely be satisfied."""
    ControlHarness("generation").assert_gate_is_live(
        good_value=exact_lexicon_total_variation(artifact, prior, habit, BEST_GAMMA),
        bad_value=exact_ancestral_lexicon_total_variation(artifact, prior, habit),
        direction="lower_is_better",
    )


# ══ THE FINDING: AN INTERIOR OPTIMUM ══════════════════════════════════════════


def test_the_precision_optimum_is_interior(habit, artifact, prior):
    """(3) Both endpoints must generate worse than the middle.

    This is the property no recognition gate in the project has: failure in two
    opposite directions, so neither "no goal" nor "pure goal" can pass.
    """
    gammas = (0.1, 0.5, 1.0, 4.0, 16.0)
    rows = exact_sweep_precision(artifact, prior, habit, gammas)
    best = min(rows, key=lambda r: r["total_variation"])
    assert 0.1 < best["gamma"] < 16.0, (
        f"The best gamma is {best['gamma']}, at an endpoint. The U-shape claim "
        f"rests on an interior optimum."
    )
    assert rows[0]["total_variation"] > best["total_variation"], (
        "The low-precision endpoint is not worse than the optimum."
    )
    assert rows[-1]["total_variation"] > best["total_variation"], (
        "The high-precision endpoint is not worse than the optimum."
    )


def test_high_precision_collapses_diversity(habit, artifact, prior):
    """The shipped high-gamma failure mode is degeneracy, not noise."""
    agent = build_generative_letter_agent(habit, gamma=16.0)
    runs = [
        generate_by_policy_selection(agent, artifact, prior, habit, n_words=N_WORDS, seed=s)
        for s in SEEDS
    ]
    assert score_generation(runs)["distinct_mean"] <= 2.0, (
        "The shipped high-precision greedy path should collapse diversity; if it "
        "no longer does, the degeneracy claim in decisions.md needs revisiting."
    )


def test_sweep_refuses_an_unsupportable_claim(habit, artifact, prior):
    """A U-shape cannot be shown from two points, nor a diagnostic from one seed."""
    with pytest.raises(ValueError, match="three distinct positive gammas"):
        exact_sweep_precision(artifact, prior, habit, (1.0, 2.0))
    with pytest.raises(ValueError, match="three distinct positive gammas"):
        sweep_precision(artifact, prior, habit, (1.0, 2.0), seeds=SEEDS, n_words=10)
    with pytest.raises(ValueError, match="cherry-pick"):
        sweep_precision(artifact, prior, habit, (0.5, 1.0, 2.0), seeds=(0,), n_words=10)


# ══ WHY MEMBERSHIP IS NOT THE GATE ════════════════════════════════════════════


def test_membership_is_green_by_construction_and_so_is_not_the_gate(habit, artifact, prior):
    """THE INSUFFICIENCY, recorded in executable form.

    ``lexicon_preference`` floors any letter leaving the trie, so membership
    cannot fail at a usable precision — including at the degenerate precision
    that emits one word forever. Had membership been the gate, the degenerate
    generator would have passed it perfectly.

    If this test ever fails because membership stopped being saturated, the
    finding changed and ``docs/decisions.md`` entry 17 needs revisiting. It is
    not a license to promote membership to the gate.
    """
    for gamma in (0.5, 1.0, 16.0):
        agent = build_generative_letter_agent(habit, gamma=gamma)
        runs = [
            generate_by_policy_selection(agent, artifact, prior, habit, n_words=100, seed=s)
            for s in SEEDS
        ]
        rate = score_generation(runs)["membership_mean"]
        assert rate > 0.99, (
            f"Membership at gamma={gamma} is {rate:.3f}, not saturated. The "
            f"green-by-construction finding no longer holds."
        )


def test_degenerate_precision_fails_the_composite_generation_gate(habit, artifact, prior):
    """The mis-wiring the distance metric alone does NOT catch.

    Every level in this project records what its guards miss. Here it is the
    degenerate generator: it emits one word forever, and still scores a *better*
    distance than ancestral sampling. Only the diversity floor rejects it.
    """
    agent = build_generative_letter_agent(habit, gamma=16.0)
    runs = [
        generate_by_policy_selection(agent, artifact, prior, habit, n_words=N_WORDS, seed=s)
        for s in SEEDS
    ]
    degenerate_tv = exact_lexicon_total_variation(artifact, prior, habit, 16.0)
    baseline_tv = exact_ancestral_lexicon_total_variation(artifact, prior, habit)
    baseline_runs = [generate_by_sampling(habit, artifact, n_words=N_WORDS, seed=s) for s in SEEDS]
    gate = evaluate_generation_gate(
        model_tv=degenerate_tv,
        baseline_tv=baseline_tv,
        model_distinct=score_generation(runs)["distinct_mean"],
        baseline_distinct=score_generation(baseline_runs)["distinct_mean"],
    )
    assert gate.tv_passes, (
        "The degenerate generator no longer beats the baseline on distance. That "
        "would make the diversity floor redundant — check before removing it."
    )
    assert not gate.diversity_passes
    assert not gate.passes, "the scalar TV arm must never certify the collapsed endpoint"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_tv", float("nan")),
        ("baseline_tv", 1.1),
        ("model_distinct", 0.0),
        ("baseline_distinct", float("inf")),
        ("diversity_fraction", 0.0),
    ),
)
def test_composite_generation_gate_rejects_invalid_metrics(field, value):
    values = {
        "model_tv": 0.3,
        "baseline_tv": 0.9,
        "model_distinct": 100.0,
        "baseline_distinct": 300.0,
        "diversity_fraction": GENERATION_DIVERSITY_FRACTION,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        evaluate_generation_gate(**values)


# ══ CONTROLS ══════════════════════════════════════════════════════════════════


def test_flat_preference_reproduces_the_baseline_exactly(habit, artifact, prior, baseline):
    """THE IDENTITY ANCHOR. With ``C = 0`` the policy posterior IS the habit.

    Character-for-character, not approximately: the two are the same stochastic
    process driven by the same seed. Anything less would mean the generator is
    influenced by something other than ``C`` and ``E``, and the whole
    interpretation of the sweep would be unsound.
    """
    agent = build_generative_letter_agent(habit, gamma=BEST_GAMMA)
    flat = [
        generate_by_policy_selection(
            agent, artifact, prior, habit, n_words=N_WORDS, seed=s, flat=True
        )
        for s in SEEDS
    ]
    for run, reference in zip(flat, baseline, strict=True):
        assert run.text == reference.text, (
            "A flat preference did not reproduce ancestral sampling exactly."
        )


def test_the_anchor_is_not_vacuous(habit, artifact, prior, baseline):
    """The real preference must actually change the output.

    Without this, a bug making ``lexicon_preference`` always return zeros would
    satisfy the anchor above and look like a passing control.
    """
    agent = build_generative_letter_agent(habit, gamma=BEST_GAMMA)
    real = generate_by_policy_selection(
        agent, artifact, prior, habit, n_words=N_WORDS, seed=SEEDS[0]
    )
    assert real.text != baseline[0].text, (
        "The real preference produced the baseline's output exactly, so the "
        "top-down channel is carrying nothing."
    )


def test_scrambled_preference_destroys_membership(habit, artifact, prior):
    """The entropy-preserving control.

    Reversing the letter axis is a fixed involutive permutation, so the
    preference keeps its entropy and loses only its meaning — strictly stronger
    than replacing it with uniform, which would confound content with entropy.
    """
    agent = build_generative_letter_agent(habit, gamma=BEST_GAMMA)
    scrambled = [
        generate_by_policy_selection(
            agent, artifact, prior, habit, n_words=100, seed=s, scrambled=True
        )
        for s in SEEDS
    ]
    rate = score_generation(scrambled)["membership_mean"]
    assert rate < 0.05, (
        f"The scrambled preference still reached {100 * rate:.1f}% membership. A "
        f"channel whose meaning can be destroyed without consequence is not "
        f"carrying meaning."
    )


# ══ VERIFICATION AGAINST PYMDP ════════════════════════════════════════════════


def _load_generation_recorder():
    """Import ``scripts/record_generation_claims.py`` by path — scripts/ is not a package."""
    import importlib.util
    import sys

    for extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(
        "ailm_record_generation_claims", PROJECT_ROOT / "scripts" / "record_generation_claims.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_form_efe_matches_pymdp_infer_policies(habit, artifact, prior):
    """The oracle-equivalence pattern, applied to policy selection.

    The closed form is what drives generation, because it is fast enough to run
    thousands of steps. It is only allowed to stand in for pymdp because the two
    are checked against each other here — an equivalence whose failure is a
    warning would be a substitution rather than a proof.

    The population is the recorder's own: every current letter paired with each
    fixed prefix in ``POLICY_CHECK_PREFIXES`` — the same tuple
    ``scripts/record_generation_claims.py`` measures and records as
    ``generation_policy_posterior_pymdp_states_checked`` — so the unit check and
    the recorded measurement describe one population rather than a subset. The
    count is derived here (prefixes × alphabet) and pinned against the ledger,
    never copied.
    """
    recorder = _load_generation_recorder()
    ledger = load_ledger(PROJECT_ROOT)
    recorded_states = int(ledger.get("generation_policy_posterior_pymdp_states_checked")["value"])
    assert len(recorder.POLICY_CHECK_PREFIXES) * N_SYMBOLS == recorded_states, (
        f"the recorder measures {recorded_states} (letter, prefix) states but "
        f"{len(recorder.POLICY_CHECK_PREFIXES)} prefixes × {N_SYMBOLS} letters is "
        f"{len(recorder.POLICY_CHECK_PREFIXES) * N_SYMBOLS}; one population or one "
        "comment is wrong"
    )
    agent = build_generative_letter_agent(habit, gamma=BEST_GAMMA)
    worst = 0.0
    states = 0
    for prefix in recorder.POLICY_CHECK_PREFIXES:
        preference = lexicon_preference(prefix, artifact, prior)
        for current in range(N_SYMBOLS):
            installed = eqx.tree_at(
                lambda m: (m.C, m.E),
                agent,
                (
                    [jnp.array(preference, dtype=jnp.float32)[jnp.newaxis]],
                    jnp.array(habit[:, current], dtype=jnp.float32)[jnp.newaxis],
                ),
            )
            one_hot = np.eye(N_SYMBOLS, dtype=np.float32)[current]
            q_pi, _ = installed.infer_policies([jnp.array(one_hot.reshape(1, 1, N_SYMBOLS))])
            mine = efe_policy_posterior(agent, current, preference, habit)
            worst = max(worst, float(np.max(np.abs(np.asarray(q_pi).ravel() - mine))))
            states += 1
    assert states == recorded_states, (
        f"the test paired {states} states against a recorded population of "
        f"{recorded_states}; the two populations drifted apart"
    )
    assert worst < 1e-5, (
        f"The closed-form policy posterior disagrees with pymdp by {worst:.3e}. "
        f"The generated text would no longer describe the pymdp agent."
    )


# ══ STRUCTURE ═════════════════════════════════════════════════════════════════


def test_the_agent_has_a_real_action_set(habit):
    """A single trivial action is why generation was previously unexercised."""
    agent = build_generative_letter_agent(habit)
    transition = np.asarray(agent.B[0])[0]
    assert transition.shape == (N_SYMBOLS, N_SYMBOLS, N_SYMBOLS), (
        f"B is {transition.shape}; the generative agent needs one control per "
        f"letter, not the single trivial action the recognition agent has."
    )


def test_the_controllable_transition_is_deterministic(habit):
    """Action ``k`` must set the next letter to ``k`` from any current letter."""
    transition = np.asarray(build_generative_letter_agent(habit).B[0])[0]
    for action in range(N_SYMBOLS):
        column = transition[:, :, action]
        assert np.allclose(column[action, :], 1.0), (
            f"action {action} does not deterministically emit letter {action}"
        )
        assert np.isclose(column.sum(), N_SYMBOLS), (
            f"action {action} puts mass on more than one next letter"
        )


def test_a_transposed_bigram_is_rejected(habit):
    """The orientation guard, matching every other level (decision 1)."""
    with pytest.raises(ValueError, match="column-stochastic"):
        build_generative_letter_agent(habit.T)


@pytest.mark.parametrize("gamma", [0.0, -1.0, np.nan, np.inf])
def test_agent_builder_rejects_invalid_policy_precision(habit, gamma):
    with pytest.raises(ValueError, match="positive finite"):
        build_generative_letter_agent(habit, gamma=gamma)


def test_closed_form_policy_posterior_validates_agent_state(habit):
    """The fast oracle must fail closed at each public probability boundary."""
    agent = build_generative_letter_agent(habit)
    preference = np.zeros(N_SYMBOLS, dtype=np.float64)

    invalid_gamma_agent = eqx.tree_at(lambda model: model.gamma, agent, np.array([0.0]))
    with pytest.raises(ValueError, match="agent gamma"):
        efe_policy_posterior(invalid_gamma_agent, SPACE_INDEX, preference, habit)
    with pytest.raises(ValueError, match="current_letter"):
        efe_policy_posterior(agent, -1, preference, habit)

    invalid_preference = preference.copy()
    invalid_preference[0] = np.nan
    with pytest.raises(ValueError, match="finite log-preference"):
        efe_policy_posterior(agent, SPACE_INDEX, invalid_preference, habit)


@pytest.mark.parametrize("mutation", ["negative", "nan", "inf"])
def test_sum_preserving_invalid_habits_fail_closed(habit, artifact, mutation):
    bad = habit.copy()
    if mutation == "negative":
        bad[0, 0], bad[1, 0] = -0.5, 1.5
    elif mutation == "nan":
        bad[0, 0] = np.nan
    else:
        bad[0, 0] = np.inf

    with pytest.raises(ValueError, match="finite|non-negative"):
        build_generative_letter_agent(bad)
    with pytest.raises(ValueError, match="finite|non-negative"):
        generate_by_sampling(bad, artifact, n_words=1, seed=0)


def test_preference_outside_the_lexicon_expresses_nothing(artifact, prior):
    """A prefix that has left the lexicon must not invent a preference."""
    preference = lexicon_preference("zzq", artifact, prior)
    assert np.allclose(preference, 0.0), (
        "A prefix with no lexicon support produced a non-flat preference, so the "
        "channel is asserting something the lexicon does not license."
    )


def test_scoring_nothing_raises(artifact):
    """A rate over zero words is undefined, not zero."""
    with pytest.raises(ValueError, match="undefined"):
        _ = GenerationResult("", frozenset(artifact.vocab)).membership_rate
    with pytest.raises(ValueError, match="no runs"):
        score_generation([])


# ══ GNN CONFORMANCE ═══════════════════════════════════════════════════════════


def test_live_generative_agent_conforms_to_its_spec(habit):
    """The generative configuration has its own spec, because it is its own model.

    Same state space as ``L1_letter``, different control structure and a
    different home for the learned table: under a controllable ``B`` the bigram
    is the policy prior ``E``, not the transition. A spec annotating it
    ``TransitionMatrix`` here would describe a model that cannot choose what to
    emit.
    """
    from ailm.gnn import assert_conforms, load_spec

    assert_conforms(load_spec("L1g_letter_generation"), describe_shapes(habit_agent(habit)))


@pytest.mark.parametrize("mutation", ["no_action_set", "habit_as_transition", "learned_emission"])
def test_generation_spec_mutations_are_rejected(habit, mutation):
    """NEGATIVE CONTROLS on conformance for the generative model."""
    from ailm.gnn import check_conformance, load_spec

    violations = check_conformance(
        load_spec(f"L1g_letter_generation@{mutation}"),
        describe_shapes(habit_agent(habit)),
    )
    assert violations, (
        f"L1g_letter_generation@{mutation} passed conformance against the correct "
        f"live agent, so that check is vacuous for the property it targets."
    )


# ══ CLAIMS LEDGER — verify, never write ══════════════════════════════════════


def test_generation_claims_verified(habit, artifact, prior):
    """Recorded generation values must come from the recorder and be self-consistent."""
    ledger = load_ledger(PROJECT_ROOT)
    required = (
        "generation_baseline_tv",
        "generation_sampled_baseline_tv",
        "generation_best_tv",
        "generation_best_gamma",
        "generation_best_membership_pct",
        "generation_best_distinct",
        "generation_words_per_run",
        "generation_seeds",
        "generation_degenerate_tv",
        "generation_degenerate_distinct",
        "generation_flat_control_tv",
        "generation_tv_improvement",
        "generation_scrambled_control_membership_pct",
    )
    values: dict[str, float] = {}
    for label in required:
        entry = ledger.get(label)
        assert entry is not None, (
            f"{label} is absent. Run scripts/record_generation_claims.py — this "
            f"test may not write it."
        )
        assert entry["source"].startswith("scripts/"), (
            f"{label} was last written by {entry['source']!r}. Recorders write; tests verify."
        )
        values[label] = float(entry["value"])

    # The ledger must not be able to hold an internally inconsistent story.
    assert values["generation_best_tv"] < values["generation_baseline_tv"]
    assert values["generation_flat_control_tv"] == pytest.approx(
        values["generation_baseline_tv"]
    ), (
        "The flat control's distance differs from the baseline's, but the flat control IS the baseline."
    )
    assert values["generation_degenerate_distinct"] <= 2.0
    assert values["generation_scrambled_control_membership_pct"] < 5.0

    # Canonical baseline, model, control, and improvement must all be exact and
    # compatible. The pooled Monte Carlo baseline remains visible only under its
    # explicitly sampled diagnostic label.
    exact_baseline = exact_ancestral_lexicon_total_variation(artifact, prior, habit)
    exact_model = exact_lexicon_total_variation(artifact, prior, habit, BEST_GAMMA)
    assert values["generation_baseline_tv"] == round(exact_baseline, 4)
    assert values["generation_best_tv"] == round(exact_model, 4)
    assert values["generation_tv_improvement"] == round(exact_baseline - exact_model, 4)
    diagnostic_baseline_runs = [
        generate_by_sampling(
            habit,
            artifact,
            n_words=int(values["generation_words_per_run"]),
            seed=seed,
        )
        for seed in range(int(values["generation_seeds"]))
    ]
    assert values["generation_sampled_baseline_tv"] == round(
        lexicon_total_variation(diagnostic_baseline_runs, artifact, prior),
        4,
    )

    # One operating-point row must come from one operating point. This replay
    # catches the former defect where the exact TV/gamma came from 0.55 while
    # membership and diversity were silently copied from the nearest sampled
    # grid row at 0.5.
    assert values["generation_best_gamma"] == BEST_GAMMA
    agent = build_generative_letter_agent(habit, gamma=BEST_GAMMA)
    operating_runs = [
        generate_by_policy_selection(
            agent,
            artifact,
            prior,
            habit,
            n_words=int(values["generation_words_per_run"]),
            seed=seed,
        )
        for seed in range(int(values["generation_seeds"]))
    ]
    operating = score_generation(operating_runs)
    assert values["generation_best_membership_pct"] == round(
        100.0 * operating["membership_mean"], 2
    )
    assert values["generation_best_distinct"] == round(operating["distinct_mean"], 1)


def _load_generation_recorder_module():
    """Import ``scripts/record_generation_claims.py`` by path — scripts/ is not a package."""
    import importlib.util
    import sys

    for extra in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(
        "ailm_record_generation_claims_m2a",
        PROJECT_ROOT / "scripts" / "record_generation_claims.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scrambled_membership_refusal_passes_legitimate_values():
    """A finite rate strictly below the declared floor is a legitimate pass."""
    recorder = _load_generation_recorder_module()
    assert recorder.scrambled_membership_refusal(0.0) is None
    assert recorder.scrambled_membership_refusal(0.04) is None
    assert recorder.scrambled_membership_refusal(0.049) is None


def test_scrambled_membership_refusal_fires_at_and_above_the_boundary():
    """The declared floor is inclusive: exactly 0.05 refuses, as does anything
    above it. The units are the scored fraction; the message reports percent."""
    recorder = _load_generation_recorder_module()
    at_boundary = recorder.scrambled_membership_refusal(0.05)
    assert at_boundary is not None and "0.05" not in at_boundary and "5.00%" in at_boundary
    above = recorder.scrambled_membership_refusal(0.06)
    assert above is not None and "6.00%" in above and "Refusing to record" in above
    assert recorder.scrambled_membership_refusal(0.06) == recorder.scrambled_membership_refusal(
        0.06, recorder.SCRAMBLED_MEMBERSHIP_REFUSAL_FLOOR
    )


def test_scrambled_membership_refusal_fails_closed_on_nonfinite_values():
    """A nonfinite measurement cannot be judged green: NaN and the infinities
    refuse, they never pass through the ``< floor`` comparison (which NaN
    would silently satisfy)."""
    import math

    recorder = _load_generation_recorder_module()
    for bad in (math.nan, math.inf, -math.inf):
        refusal = recorder.scrambled_membership_refusal(bad)
        assert refusal is not None and "nonfinite" in refusal


def test_scrambled_membership_refusal_fails_closed_on_out_of_range_values():
    """A membership share is a mean of per-run rates in [0, 1]; values outside
    are schema violations and refuse rather than pass."""
    recorder = _load_generation_recorder_module()
    for bad in (-0.1, 1.1, 100.0):
        refusal = recorder.scrambled_membership_refusal(bad)
        assert refusal is not None and "outside the [0, 1] rate range" in refusal


def test_main_wires_the_named_refusal_at_the_pre_write_location():
    """The seam is the one main() consults before the ledger write: the
    recorder's main must call ``scrambled_membership_refusal`` and must not
    re-derive the floor comparison inline."""
    import inspect

    recorder = _load_generation_recorder_module()
    source = inspect.getsource(recorder.main)
    assert 'scrambled_membership_refusal(scrambled["membership_mean"])' in source
    assert "raise ValueError(refusal)" in source
    assert source.index("scrambled_membership_refusal") < source.index("ledger.record")
