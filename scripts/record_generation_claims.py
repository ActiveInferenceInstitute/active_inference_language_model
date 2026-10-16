#!/usr/bin/env python3
"""Record every generation number into the claims ledger, from the run that measures it.

Thin orchestrator. Computation lives in ``ailm.generation``; this script
sequences the measurements and writes the ledger. Single writer for every
``generation_*`` label.

What it establishes, in order:

1. The **baseline** — the exact finite-lexicon distribution induced by ancestral
   sampling from the letter bigram, the strongest generator with the same letter
   statistics and no top-down channel.
2. The **precision sweep**, which is the finding: distance to the true word
   distribution is U-shaped in ``gamma``, so both "no goal" and "pure goal"
   generate badly and the optimum is interior.
3. The **controls** — a flat ``C`` must reproduce the baseline exactly, and an
   entropy-preserving scrambled ``C`` must fail.
4. The **insufficiency** — the degenerate high-``gamma`` generator, which beats
   the baseline on distance while emitting one word forever. Recorded because
   every gate in this project states what it does *not* catch.
5. The **policy-posterior attribution** — the generator samples from the
   closed-form ``efe_policy_posterior`` and never calls ``agent.infer_policies``;
   that substitution is licensed only by agreement with pymdp's own policy
   inference, so the worst deviation over every current letter and a fixed
   prefix set is measured, refused above ``POLICY_POSTERIOR_ATOL``, and
   recorded (decision 95).

Usage
-----
    python scripts/record_generation_claims.py [--words N] [--seeds N]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ailm.gates import GENERATION_GATE_SESOI_TV
from ailm.ledger_lock import ledger_lock_path, single_writer_lock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ID = "generation_gate"
SOURCE = "scripts/record_generation_claims.py"

#: Precisions swept by the *sampled* generator. Spans both failure modes so the
#: interior optimum is visible rather than asserted. Kept as it was so the
#: convergence check below compares like with like.
GAMMAS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0)

#: Precisions evaluated exactly. Denser than the sampled grid because computing
#: a point costs about as much as one sampled run and carries no estimator
#: noise, and because the sampled grid's own spacing was too coarse to resolve
#: the nearby operating points reliably. The sampled operating point moved under
#: exact scoring; the U-shape did not.
EXACT_GAMMAS: tuple[float, ...] = (
    0.1,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.85,
    1.0,
    1.25,
    1.5,
    2.0,
    4.0,
    8.0,
    16.0,
)

#: The degenerate point, kept in the sweep and reported separately.
DEGENERATE_GAMMA: float = 16.0

#: Prefixes at which the closed-form policy posterior is compared with
#: ``agent.infer_policies``, paired with every current letter. The same fixed
#: set ``tests/test_generation.py`` uses, so the recorded measurement and the
#: unit check describe one population: an empty prefix (word start), two
#: mid-word prefixes with different continuation supports, and a one-letter one.
POLICY_CHECK_PREFIXES: tuple[str, ...] = ("", "th", "co", "s")

#: Tolerance on the worst absolute deviation between the two policy posteriors.
#: The Agent evaluates in float32; the closed form in float64. Rounding is the
#: only admissible difference, and a wrong reduction would exceed this by
#: orders of magnitude.
POLICY_POSTERIOR_ATOL: float = 1e-5

#: The scrambled-preference control must DESTROY lexicon membership: the
#: verify-time guard (``tests/test_generation.py``) refuses a recorded value at
#: or above this share, and the recorder now refuses to write one in the first
#: place. One threshold, two enforcement points, so the control cannot be live
#: under one path and dead under the other.
SCRAMBLED_MEMBERSHIP_REFUSAL_FLOOR: float = 0.05


def scrambled_membership_refusal(
    membership_mean: float,
    floor: float = SCRAMBLED_MEMBERSHIP_REFUSAL_FLOOR,
) -> str | None:
    """The scrambled-preference control's record-time refusal, as a predicate.

    Returns the refusal message when the control must block the write, and
    ``None`` when the value is a legitimate pass (a finite rate below the
    floor). A membership share is a mean of per-run rates in ``[0, 1]``, so a
    nonfinite value or anything outside that interval cannot be judged green
    and refuses too: this is a real gate, not a formality a malformed
    measurement could slip past, and the current recorded result being 0.0
    does not make the control green by construction.

    Args:
        membership_mean: The scrambled runs' mean lexicon-membership rate
            (fraction on the 0-1 scale, as scored by ``score_generation``).
        floor: The declared refusal floor (fraction; the verify-time guard in
            ``tests/test_generation.py`` pins the same 0.05 threshold).
    """
    if not math.isfinite(membership_mean):
        return (
            f"The scrambled-preference control measured a nonfinite membership "
            f"({membership_mean!r}); a measurement that cannot be read cannot "
            f"be judged green. Refusing to record."
        )
    if membership_mean < 0.0 or membership_mean > 1.0:
        return (
            f"The scrambled-preference control measured membership "
            f"{membership_mean!r}, outside the [0, 1] rate range. "
            f"Refusing to record."
        )
    if membership_mean < floor:
        return None
    return (
        "The scrambled-preference control still reached "
        f"{100.0 * membership_mean:.2f}% lexicon membership "
        f"(refusal floor {100.0 * floor:.0f}%). A channel whose meaning "
        "survives an entropy-preserving scramble of its preference is not "
        "carrying meaning. Refusing to record."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    import equinox as eqx
    import jax.numpy as jnp

    from ailm.claims import load_ledger
    from ailm.generation import (
        GENERATION_DIVERSITY_FRACTION,
        build_generative_letter_agent,
        efe_policy_posterior,
        evaluate_generation_gate,
        exact_ancestral_lexicon_total_variation,
        exact_sweep_precision,
        expected_free_energy_terms,
        generate_by_policy_selection,
        generate_by_sampling,
        lexicon_preference,
        lexicon_total_variation,
        score_generation,
        sweep_precision,
    )
    from ailm.levels.letter import N_SYMBOLS, build_letter_transition, load_brown_counts
    from ailm.levels.word import build_word_prior, load_word_artifact

    seeds = tuple(range(args.seeds))
    habit = np.asarray(build_letter_transition(load_brown_counts()))[0]
    artifact = load_word_artifact()
    prior = build_word_prior(artifact)

    # ── Baseline: no top-down channel ─────────────────────────────────────────
    baseline_runs = [
        generate_by_sampling(habit, artifact, n_words=args.words, seed=s) for s in seeds
    ]
    baseline = score_generation(baseline_runs)
    sampled_baseline_tv = lexicon_total_variation(baseline_runs, artifact, prior)
    baseline_tv = exact_ancestral_lexicon_total_variation(artifact, prior, habit)

    # ── The sweep, which is the finding ───────────────────────────────────────
    #
    # Computed, not sampled. The generator's state is the emitted prefix alone,
    # so its induced distribution over the lexicon is exact; the sampled sweep
    # below is kept as a convergence cross-check, which is the same
    # oracle-equivalence pattern every other numeric claim in this project uses.
    #
    # This changes the recorded operating point: the pooled empirical
    # distribution had fewer tokens than target types and no uncertainty, so it
    # could not resolve nearby candidates. The interior optimum — the actual
    # claim — survives exact scoring.
    exact_rows = exact_sweep_precision(artifact, prior, habit, EXACT_GAMMAS)
    best = min(exact_rows, key=lambda r: r["total_variation"])
    degenerate = next(r for r in exact_rows if r["gamma"] == DEGENERATE_GAMMA)
    if not 0.0 < best["gamma"] < max(EXACT_GAMMAS):
        raise ValueError(
            f"The best gamma is {best['gamma']}, at an endpoint of the swept "
            f"range. The U-shape claim rests on an INTERIOR optimum; either the "
            f"range is too narrow or the finding has changed."
        )
    if not best["total_variation"] < baseline_tv:
        raise SystemExit(
            "REFUSING TO RECORD: the exact policy-induced distribution does not "
            "beat the exact ancestral distribution. The canonical improvement "
            "must compare compatible estimators and be strictly positive; got "
            f"model={best['total_variation']:.8f}, baseline={baseline_tv:.8f}."
        )

    # The sampled sweep, now a cross-check rather than the measurement. The gap
    # is reported so a reader can see how far the sampled estimator sits from
    # the quantity it estimates, at the sample size the sweep actually used.
    rows = sweep_precision(artifact, prior, habit, GAMMAS, seeds=seeds, n_words=args.words)
    exact_at = {row["gamma"]: row["total_variation"] for row in exact_rows}
    mc_gap = max(
        abs(row["total_variation"] - exact_at[row["gamma"]])
        for row in rows
        if row["gamma"] in exact_at
    )
    sampled_best = min(rows, key=lambda r: r["total_variation"])

    # Membership and diversity still come from the sampled generator: they are
    # properties of realized runs, not of the induced distribution. Sample the
    # exact shipped precision rather than borrowing the nearest row from the
    # coarser diagnostic grid. The former recorder silently borrowed diversity
    # from a neighboring sampled precision, creating one ledger row from two
    # operating points.
    agent = build_generative_letter_agent(habit, gamma=best["gamma"])
    operating_runs = [
        generate_by_policy_selection(agent, artifact, prior, habit, n_words=args.words, seed=seed)
        for seed in seeds
    ]
    operating_score = score_generation(operating_runs)
    operating = {
        "gamma": float(best["gamma"]),
        "membership": operating_score["membership_mean"],
        "distinct": operating_score["distinct_mean"],
    }
    gate = evaluate_generation_gate(
        model_tv=best["total_variation"],
        baseline_tv=baseline_tv,
        model_distinct=operating["distinct"],
        baseline_distinct=baseline["distinct_mean"],
    )
    if not gate.passes:
        raise SystemExit(
            "REFUSING TO RECORD: the shipped generation operating point does not "
            "pass both declared gate conjuncts; "
            f"tv_passes={gate.tv_passes}, diversity_passes={gate.diversity_passes}, "
            f"distinct={gate.model_distinct:.4g}, floor={gate.diversity_floor:.4g}."
        )

    degenerate_score = next(r for r in rows if r["gamma"] == DEGENERATE_GAMMA)
    degenerate_gate = evaluate_generation_gate(
        model_tv=degenerate["total_variation"],
        baseline_tv=baseline_tv,
        model_distinct=degenerate_score["distinct"],
        baseline_distinct=baseline["distinct_mean"],
    )
    if not (
        degenerate_gate.tv_passes
        and not degenerate_gate.diversity_passes
        and not degenerate_gate.passes
    ):
        raise SystemExit(
            "REFUSING TO RECORD: the high-precision negative control no longer "
            "proves that exact TV alone is insufficient; "
            f"tv_passes={degenerate_gate.tv_passes}, "
            f"diversity_passes={degenerate_gate.diversity_passes}."
        )

    # ── Controls ──────────────────────────────────────────────────────────────
    flat_runs = [
        generate_by_policy_selection(
            agent, artifact, prior, habit, n_words=args.words, seed=s, flat=True
        )
        for s in seeds
    ]
    scrambled_runs = [
        generate_by_policy_selection(
            agent, artifact, prior, habit, n_words=args.words, seed=s, scrambled=True
        )
        for s in seeds
    ]
    flat = score_generation(flat_runs)
    scrambled = score_generation(scrambled_runs)

    # A flat preference must reproduce the baseline EXACTLY: with C = 0 the
    # policy posterior is the habit, so the two samplers are the same process
    # driven by the same seed. An approximate match would mean something else is
    # influencing the choice.
    exact_match = all(f.text == b.text for f, b in zip(flat_runs, baseline_runs, strict=True))
    if not exact_match:
        raise ValueError(
            "A flat preference did NOT reproduce the sampling baseline "
            "character-for-character. With C = 0 the policy posterior reduces to "
            "the habit, so any difference means the generator is using something "
            "other than C and E. Refusing to record."
        )

    refusal = scrambled_membership_refusal(scrambled["membership_mean"])
    if refusal is not None:
        raise ValueError(refusal)
    # ── The closed-form policy posterior vs pymdp's own infer_policies ────────
    #
    # ``generate_by_policy_selection`` never calls ``agent.infer_policies``: the
    # distribution it samples actions from is ``efe_policy_posterior``, a closed
    # form fast enough to run thousands of steps. That substitution is licensed
    # only by agreement with pymdp's own policy inference, and a check that
    # lives in a unit test is not a recorded measurement. Every current letter
    # is paired with each fixed prefix, ``C`` and ``E`` are installed on the
    # shipped agent exactly as the generator's sampler would see them, and the
    # worst deviation between the two posteriors is refused above tolerance
    # and recorded otherwise.
    policy_checked = 0
    policy_worst = 0.0
    for prefix in POLICY_CHECK_PREFIXES:
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
            closed_form = efe_policy_posterior(agent, current, preference, habit)
            policy_worst = max(
                policy_worst, float(np.max(np.abs(np.asarray(q_pi).ravel() - closed_form)))
            )
            policy_checked += 1
    if policy_worst >= POLICY_POSTERIOR_ATOL:
        raise SystemExit(
            "REFUSING TO RECORD: the closed-form policy posterior that drives "
            f"generation disagrees with pymdp's infer_policies by {policy_worst:.3e} "
            f"(tolerance {POLICY_POSTERIOR_ATOL:.0e}) over {policy_checked} "
            "(letter, prefix) states. The generated text would no longer describe "
            "the pymdp agent."
        )

    # ── The information-gain term, measured rather than asserted ──────────────
    #
    # The generation section claims that under this construction the expected
    # free energy has no information-gain term, so policy selection reduces to a
    # preference-weighted softmax over the habit. That is an analytic claim, and
    # a claim of the form "this quantity is zero" is exactly the kind that reads
    # as true whether or not anyone computed it. It is computed here.
    #
    # The two controls establish that the zero is a fact about the construction
    # and not about the arithmetic: making the transition stochastic while
    # keeping the identity likelihood must produce a *positive* gain, and then
    # flattening the likelihood must return it to zero. If the first control did
    # not fire, this measurement could not distinguish the claim from a
    # function that returns zeros.
    identity_emission = np.eye(N_SYMBOLS)
    shipped_transition = np.zeros((N_SYMBOLS, N_SYMBOLS, N_SYMBOLS))
    for action in range(N_SYMBOLS):
        shipped_transition[action, :, action] = 1.0
    stochastic_transition = np.full((N_SYMBOLS, N_SYMBOLS, N_SYMBOLS), 1.0 / N_SYMBOLS)
    goal = lexicon_preference("", artifact, prior)

    shipped_terms = expected_free_energy_terms(identity_emission, shipped_transition, goal)
    stochastic_terms = expected_free_energy_terms(identity_emission, stochastic_transition, goal)
    flat_likelihood_terms = expected_free_energy_terms(
        np.full((N_SYMBOLS, N_SYMBOLS), 1.0 / N_SYMBOLS), stochastic_transition, goal
    )

    shipped_gain = float(np.max(np.abs(shipped_terms["information_gain"])))
    stochastic_gain = float(np.max(stochastic_terms["information_gain"]))
    if stochastic_gain <= shipped_gain:
        raise SystemExit(
            f"REFUSING TO RECORD: the stochastic-transition control produced an "
            f"information gain of {stochastic_gain:.6g}, no greater than the "
            f"shipped construction's {shipped_gain:.6g}. The control exists to "
            f"show the term CAN be positive; without that, recording a zero for "
            f"the shipped case establishes nothing about the construction."
        )

    recorded: dict[str, float | int] = {
        # The analytic reduction, measured. See [@sec:generation].
        "generation_shipped_information_gain_nats": round(shipped_gain, 12),
        "generation_stochastic_control_information_gain_nats": round(stochastic_gain, 6),
        "generation_flat_likelihood_control_information_gain_nats": round(
            float(np.max(np.abs(flat_likelihood_terms["information_gain"]))), 12
        ),
        "generation_max_information_gain_possible_nats": round(math.log(N_SYMBOLS), 6),
        "generation_words_per_run": args.words,
        "generation_seeds": len(seeds),
        "generation_gamma_points": len(GAMMAS),
        "generation_exact_grid_points": len(EXACT_GAMMAS),
        # How far the sampled estimator sits from the quantity it estimates, at
        # the sample size the sweep uses. Recorded so the demotion of the
        # sampled sweep is a measurement rather than an assertion.
        "generation_mc_exact_max_abs_gap": round(mc_gap, 4),
        "generation_sampled_best_gamma": sampled_best["gamma"],
        "generation_sampled_best_tv": round(sampled_best["total_variation"], 4),
        "generation_sampled_baseline_tv": round(sampled_baseline_tv, 4),
        # baseline
        "generation_baseline_membership_pct": round(100.0 * baseline["membership_mean"], 2),
        "generation_baseline_distinct": round(baseline["distinct_mean"], 1),
        "generation_baseline_tv": round(baseline_tv, 4),
        "generation_diversity_fraction_pct": round(100.0 * GENERATION_DIVERSITY_FRACTION, 4),
        "generation_diversity_floor": round(gate.diversity_floor, 4),
        # the shipped operating point
        "generation_best_gamma": best["gamma"],
        "generation_best_membership_pct": round(100.0 * operating["membership"], 2),
        "generation_best_distinct": round(operating["distinct"], 1),
        "generation_best_tv": round(best["total_variation"], 4),
        "generation_tv_improvement": round(baseline_tv - best["total_variation"], 4),
        "generation_gate_sesoi_tv": GENERATION_GATE_SESOI_TV,
        # the sweep's endpoints, which fail for opposite reasons
        "generation_low_gamma": min(GAMMAS),
        "generation_low_gamma_tv": round(exact_at[min(GAMMAS)], 4),
        "generation_low_gamma_membership_pct": round(100.0 * rows[0]["membership"], 2),
        "generation_degenerate_gamma": DEGENERATE_GAMMA,
        "generation_degenerate_tv": round(degenerate["total_variation"], 4),
        "generation_degenerate_distinct": round(degenerate_score["distinct"], 1),
        "generation_degenerate_membership_pct": round(100.0 * degenerate_score["membership"], 2),
        # controls
        # C = 0 is analytically the same ancestral process, and the seeded
        # character identity above proves the two implementations agree. Use the
        # exact ancestral distance here; relabeling its sampled realization as an
        # exact control would recreate the estimator mismatch this recorder fixes.
        "generation_flat_control_tv": round(baseline_tv, 4),
        "generation_flat_control_membership_pct": round(100.0 * flat["membership_mean"], 2),
        "generation_scrambled_control_membership_pct": round(
            100.0 * scrambled["membership_mean"], 2
        ),
        "generation_scrambled_control_distinct": round(scrambled["distinct_mean"], 1),
    }

    ledger = load_ledger(PROJECT_ROOT)
    ledger.record_many(recorded, run_id=RUN_ID, source=SOURCE)
    # The attribution measurement carries its own statuses: the deviation is a
    # deterministic comparison of two evaluations (analytic); the state count
    # is a size (descriptive). Neither is sampled, so neither inherits the
    # sweep's default.
    ledger.record(
        "generation_policy_posterior_pymdp_max_abs_dev",
        float(f"{policy_worst:.3g}"),
        run_id=RUN_ID,
        source=SOURCE,
        status="analytic",
    )
    ledger.record(
        "generation_policy_posterior_pymdp_states_checked",
        policy_checked,
        run_id=RUN_ID,
        source=SOURCE,
        status="descriptive",
    )
    recorded["generation_policy_posterior_pymdp_max_abs_dev"] = float(f"{policy_worst:.3g}")
    recorded["generation_policy_posterior_pymdp_states_checked"] = policy_checked

    print(f"Recorded {len(recorded)} generation claims into data/claims.json:\n")
    width = max(len(k) for k in recorded)
    for label, value in recorded.items():
        print(f"  {label:<{width}} = {value}")

    print(
        f"\n{'gamma':>8} {'member%':>9} {'distinct':>9} {'TV(diagnostic MC)':>17} {'TV(exact)':>9}"
    )
    print(
        f"{'baseline':>8} {100 * baseline['membership_mean']:8.2f}% "
        f"{baseline['distinct_mean']:9.1f} {sampled_baseline_tv:17.4f} "
        f"{baseline_tv:8.4f}"
    )
    for row in rows:
        exact = exact_at.get(row["gamma"])
        mark = "  <- best" if row["gamma"] == best["gamma"] else ""
        exact_col = f" {exact:8.4f}" if exact is not None else " " * 9
        print(
            f"{row['gamma']:8.2f} {100 * row['membership']:8.2f}% "
            f"{row['distinct']:9.1f} {row['total_variation']:17.4f}{exact_col}{mark}"
        )
    print(
        f"\nSampled argmin gamma={sampled_best['gamma']} "
        f"(TV {sampled_best['total_variation']:.4f}); exact argmin "
        f"gamma={best['gamma']} (TV {best['total_variation']:.4f}). Largest "
        f"sampled-to-exact gap on the shared grid: {mc_gap:.4f}."
    )
    print(
        f"Exact ancestral baseline TV={baseline_tv:.4f}; its diagnostic pooled "
        f"sample estimate is {sampled_baseline_tv:.4f}. The canonical improvement "
        f"subtracts the exact model TV from the exact baseline TV."
    )

    print(
        f"\nThe optimum is INTERIOR at gamma={best['gamma']}: both endpoints "
        f"generate badly for opposite reasons. At gamma={min(EXACT_GAMMAS)} the goal is "
        f"too weak and the generator reverts toward the habit; at "
        f"gamma={DEGENERATE_GAMMA} it collapses onto "
        f"{next(r for r in rows if r['gamma'] == DEGENERATE_GAMMA)['distinct']:.0f} "
        f"distinct word(s)."
    )
    print(
        f"\nMembership is NOT the gate: it is "
        f"{100 * operating['membership']:.1f}% at the operating point and "
        f"{100 * next(r for r in rows if r['gamma'] == DEGENERATE_GAMMA)['membership']:.1f}% "
        f"at the degenerate one, so it "
        f"cannot fail. The gate is total-variation distance to the true word "
        f"distribution, paired with a diversity floor."
    )
    print(
        f"\nControls: a flat C reproduced the baseline character-for-character; "
        f"the scrambled C reached "
        f"{100 * scrambled['membership_mean']:.2f}% membership."
    )
    return 0


if __name__ == "__main__":
    # The ledger is single-writer: ``ClaimsLedger`` reads the whole file at
    # construction and writes the whole file at the end, so two overlapping
    # recorders lose one of the two results, with no error on either side.
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
