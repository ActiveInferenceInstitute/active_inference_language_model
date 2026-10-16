# Formalism map

Every labeled equation, definition, and proposition in the manuscript, mapped to
the code that computes it and the test that exercises it — or to the sentence that
plainly says nothing computes it.

The current source-derived inventory (`equations=25`, `definitions=2`,
`propositions=2`) is enforced exactly by
[`test_manuscript_crossrefs.py`](../../tests/test_manuscript_crossrefs.py): this file
must have one and only one row for every formalism, and no phantom row. A row is
only `IMPLEMENTED` if a named function can be pointed at *and* the arithmetic in
that function was read and found to match the equation. Name similarity is not
evidence.

Why this file exists: a manuscript that states a formalism it does not compute has
made an unfalsifiable claim, and the formal apparatus of an active-inference paper
is exactly where that is easiest to do accidentally. Re-derive this map when an
equation is added, moved, or relabeled.

Every reference below names a **function, class, or label** rather than a line
number. Earlier revisions cited `file.py:NNN`, and every one of those numbers had
drifted by the time anyone checked — a row that points at the wrong line is not
evidence, and worse, it looks like evidence. A symbol name either resolves or it
does not, and `rg -n` turns it back into a line whenever a line is what you
want.

Several of these formalisms are additionally **machine-checked** in Lean 4 under
`formal/` — `eq:gate_satisfies`, `eq:gate_verdict`, the correction half of
`eq:filtering`, the Bonferroni arithmetic, and the Holm step-down comparison
behind it — with the Lean ↔ equation ↔
Python ↔ test correspondence tabulated in [`formal.md`](../formal.md). A Lean proof
does not change any row's status here: this file is about whether the *shipped
code* computes the equation, which is a different question from whether the
equation's logical content is proved.

## Status vocabulary

| Status | Meaning |
|---|---|
| `IMPLEMENTED` | A named function computes it, the arithmetic matches, and a test exercises it. |
| `IMPLEMENTED (inert)` | The function exists, is correct, and is tested — but no shipped level calls it. |
| `HONESTLY DEFERRED` | The manuscript states the formalism *and* states plainly that no built level exercises it. |
| `OVERCLAIMED` | Presented as active or as characterizing the shipped system, but no code computes it. A defect. |

## The map

| Label | What it states | Status | Implementation | Exercising test | Introduced in |
|---|---|---|---|---|---|
| `eq:vfe` | Variational free energy, complexity + accuracy, bounding $-\log p(o)$ | `IMPLEMENTED` (minimizer only) | [`forward_filter`](../../src/ailm/verification/oracle.py) closed-form posterior; [`compute_perplexity`](../../src/ailm/levels/letter.py) | [`test_hand_computed_forward_pass`](../../tests/test_oracle.py) | [`02_background.md`](../manuscript/02_background.md) |
| `eq:efe` | Expected free energy over policies, epistemic + pragmatic | `IMPLEMENTED` | [`efe_policy_posterior`](../../src/ailm/generation.py), checked against pymdp's own `infer_policies` | [`test_closed_form_efe_matches_pymdp_infer_policies`](../../tests/test_generation.py) | [`02_background.md`](../manuscript/02_background.md) |
| `eq:policy_posterior` | $q(\pi)=\sigma(\gamma\,\mathrm{neg\text{-}EFE}+\ln\mathbf{E})$ | `IMPLEMENTED` | [`efe_policy_posterior`](../../src/ailm/generation.py) evaluates the normalized policy posterior from precision, preferences, and the learned habit | [`test_closed_form_efe_matches_pymdp_infer_policies`](../../tests/test_generation.py) checks the closed form against pymdp policy inference | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:policy_reduction` | Under deterministic controllable transitions and identity emission, $q(\pi=k)=\sigma(\gamma\mathbf{C}[k]+\log\mathbf{E}[k])$ | `IMPLEMENTED` | [`expected_free_energy_terms`](../../src/ailm/generation.py) computes the two EFE terms; [`efe_policy_posterior`](../../src/ailm/generation.py) computes the reduced posterior | [`test_the_reduction_reproduces_the_shipped_policy_posterior`](../../tests/test_free_energy_decomposition.py) | [`04_results.md`](../manuscript/04_results.md) |
| Proposition (The information-gain term vanishes) | A deterministic policy-conditioned transition makes the predictive state a point mass, so its outcome-conditioned KL is zero; identity emission further reduces pragmatic value to $-\mathbf{C}[k]$ | `IMPLEMENTED` (measured with discriminating controls) | [`expected_free_energy_terms`](../../src/ailm/generation.py) computes information gain and pragmatic value separately for every action | [`test_the_shipped_construction_has_no_information_gain`](../../tests/test_free_energy_decomposition.py), with stochastic-transition, flat-likelihood, and deterministic-transition-only controls beside it | [`04_results.md`](../manuscript/04_results.md) |
| `eq:generative` | The discrete POMDP joint factorization | `IMPLEMENTED` | [`build_letter_agent`](../../src/ailm/levels/letter.py), [`build_phoneme_agent`](../../src/ailm/levels/phoneme.py), [`build_g2p_agent`](../../src/ailm/levels/phoneme.py) — each declaring `A_dependencies` | [`TestADependenciesShape`](../../tests/test_pymdp_smoke.py) | [`02_background.md`](../manuscript/02_background.md) |
| `eq:pomdp` | $\mathcal{M} = \langle A, B, C, D\rangle$ | `IMPLEMENTED` | [`build_letter_agent`](../../src/ailm/levels/letter.py) constructs all four slots in one place | [`test_stage1_bigram_statistics_live_in_the_transition`](../../tests/test_stage1_gate.py) | [`02_background.md`](../manuscript/02_background.md) |
| Definition (The AILM matrix tuple) | $A_{ij}=p(o{=}i\mid s{=}j)$, $B^{(u)}_{ij}=p(s_{t+1}{=}i\mid s_t{=}j,u)$, $C_i=\log\tilde p(o{=}i)$, $D_j=p(s_1{=}j)$, column-stochastic | `IMPLEMENTED` | [`build_letter_transition`](../../src/ailm/levels/letter.py), [`build_g2p_emission`](../../src/ailm/levels/phoneme.py), [`column_stochastic`](../../src/ailm/levels/_stochastic.py) | [`test_stage2_g2p_emission_orientation_is_pinned`](../../tests/test_stage2_gate.py) | [`02_background.md`](../manuscript/02_background.md) |
| `eq:filtering` | Predictive prior through $B$, posterior correction through $A$ | `IMPLEMENTED` (twice, independently) | [`compute_perplexity`](../../src/ailm/levels/letter.py), [`compute_phoneme_perplexity`](../../src/ailm/levels/phoneme.py), [`decode_phonemes`](../../src/ailm/levels/phoneme.py); independently [`forward_filter`](../../src/ailm/verification/oracle.py) | [`test_hand_computed_evidence_equals_brute_force_marginal`](../../tests/test_oracle.py); [`test_stage1_gate.py`](../../tests/test_stage1_gate.py) | [`02_background.md`](../manuscript/02_background.md) |
| `eq:perplexity` | $\exp(-\frac1T\sum_t\log\sum_j A_{o_tj}\rho_t(j))$ over the predictive prior $\rho_t$ | `IMPLEMENTED` | [`compute_perplexity`](../../src/ailm/levels/letter.py), [`compute_phoneme_perplexity`](../../src/ailm/levels/phoneme.py), [`sequence_perplexity`](../../src/ailm/verification/oracle.py) | [`test_uniform_chain_perplexity_is_exactly_n`](../../tests/test_oracle.py), plus the deterministic-chain anchor beside it | [`02_background.md`](../manuscript/02_background.md) |
| `eq:conditional_entropy` | $H(X_t\mid X_{t-1})$ and $\mathrm{PPL}_{\min}$ | `IMPLEMENTED` (three copies) | [`bigram_information_bounds`](../../scripts/record_stage1_claims.py), [`phone_information_bounds`](../../scripts/record_stage2_claims.py), [`_conditional_entropy`](../../scripts/audit_gates.py) | [`test_stage2_two_nat_target_is_provably_unreachable`](../../tests/test_stage2_gate.py) | [`02_background.md`](../manuscript/02_background.md) |
| `eq:case_free_energy` | AILM's scalar NOM/ACC scaffold: $F_c[q]=\beta_c D_{\mathrm{KL}}[q\|p]-\alpha_c\mathbb{E}_q[\log p(o\mid s)]$; not a source formula for all cases | `IMPLEMENTED (inert; NOM/ACC only)` | [`case_conditioned_free_energy`](../../src/ailm/cases.py); project-owned defaults in `AILM_SCALAR_CASE_WEIGHTS`; distinct source formulations in `CEREBRUM_CASE_FORMULATIONS` | [`TestCaseFreeEnergy`](../../tests/test_cases.py), [`TestCaseFreeEnergySourceFidelity`](../../tests/test_cases.py) | [`02_background.md`](../manuscript/02_background.md) |
| Proposition (Sequence statistics belong in \(\mathbf{B}\)) | $M$ in $B$ predicts forward; $M$ in $A$ yields a posterior over the *preceding* symbol; both are column-stochastic | `IMPLEMENTED` (measured, not asserted) | [`anti_informative_perplexity`](../../scripts/record_stage1_claims.py) | [`test_stage1_unigram_gate_rejects_the_anti_informative_wiring`](../../tests/test_stage1_gate.py) | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:step_cost` | Per-step arithmetic cost of the filtering recursion: $T\,S + (T-n)\,S^2$ multiplications over $n$ sequences totalling $T$ positions | `IMPLEMENTED` (formula and executed count reconciled) | [`filtering_ops_theory`](../../src/ailm/complexity.py) states the formula; [`count_filtering_ops`](../../src/ailm/complexity.py) increments beside each statement of the recursion as it executes | [`test_letter_level_formula_reproduces_the_executed_count`](../../tests/test_complexity.py), with [`test_the_off_by_one_formula_is_rejected`](../../tests/test_complexity.py) carrying the wrong formula as a live control | [`98_supplement_complexity.md`](../manuscript/98_supplement_complexity.md) |
| `eq:forward_recursion` | The oracle's array-layout form of `eq:filtering` | `IMPLEMENTED` | [`forward_filter`](../../src/ailm/verification/oracle.py) | [`test_transition_orientation_survives_doubly_stochastic_transpose`](../../tests/test_oracle.py) | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:decoder_equivalence` | $\sum_t \mathbf{1}[\hat p^{\text{pymdp}}_t \neq \hat p^{\text{oracle}}_t] = 0$ | `IMPLEMENTED` (refuses to record on failure) | [`verify_decoder_equivalence`](../../scripts/record_stage2_claims.py), enforced in that script's `main` | [`test_stage2_g2p_pymdp_matches_oracle`](../../tests/test_stage2_gate.py) | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:mcnemar` | McNemar exact test: two-sided exact binomial on $b$ of $n=b+c$ discordant pairs at $p=\tfrac12$ | `IMPLEMENTED` (one count-level implementation, one validating benchmark wrapper) | [`exact_two_sided_binomial_p`](../../src/ailm/statistics.py) is the central count-level helper; equivalent direct SciPy sites are disclosed below; [`exact_mcnemar_p`](../../src/ailm/benchmarks/scoring.py) validates benchmark inputs before delegating to it | [`test_exact_mcnemar_wrapper_matches_the_central_scipy_contract`](../../tests/test_statistics.py) checks the wrapper, central helper, and independent SciPy result over degenerate, tied, asymmetric, and shipped sufficient counts; stage-specific tests bind their recorded sufficient counts | [`98_supplement_statistics.md`](../manuscript/98_supplement_statistics.md) |
| `eq:paired_difference` | $\delta=(2s-1)(b+c)/n$, mapping the exact discordant-pair share interval to the paired accuracy-difference scale | `IMPLEMENTED` | [`paired_difference_interval`](../../src/ailm/benchmarks/scoring.py) applies the exact Clopper–Pearson share interval and this affine mapping | [`test_paired_difference_interval_brackets_the_point_estimate`](../../tests/test_benchmarks.py), with swap symmetry, zero-discordance, and impossible-count controls beside it | [`98_supplement_statistics.md`](../manuscript/98_supplement_statistics.md) |
| `eq:bits_per_char` | $H_2=\log_2\mathrm{PPL}$ | `IMPLEMENTED` | [`record_stage1_claims.py::main`](../../scripts/record_stage1_claims.py) | [`test_stage1_bits_and_signal_capture_are_exact_ledger_derivations`](../../tests/test_stage1_gate.py), including a natural-log wrong-unit control | [`04_results.md`](../manuscript/04_results.md) |
| `eq:signal_captured` | $\eta = (\log\mathrm{PPL}_{\text{uniform}}-\log\mathrm{PPL}_{\text{model}})/(\log\mathrm{PPL}_{\text{uniform}}-\log\mathrm{PPL}_{\text{floor}})$ | `IMPLEMENTED` | [`record_stage1_claims.py::main`](../../scripts/record_stage1_claims.py) | [`test_stage1_bits_and_signal_capture_are_exact_ledger_derivations`](../../tests/test_stage1_gate.py), including a planted wrong-denominator control | [`04_results.md`](../manuscript/04_results.md) |
| `eq:gate_satisfies` | Declared $x\succeq t$ uses $x\le t$ (lower better) or $x\ge t$ (higher better); the audit applies strict $x\succ t$ | `IMPLEMENTED` | [`gate_passes`](../../src/ailm/gates.py) — the strict audit implementation, called through [`evaluate_gate`](../../src/ailm/gates.py) by `scripts/audit_gates.py` | [`test_gates.py`](../../tests/test_gates.py); [`test_gate_taxonomy_values_match_the_ledger`](../../tests/test_figures_results.py) | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:reference_no_impossibility` | A failing exceedable reference does not imply that the admissible model class cannot pass | `IMPLEMENTED` and machine-checked | [`classify_gate`](../../src/ailm/gates.py) emits `ABOVE_REFERENCE`, never `UNREACHABLE`, for a failing exceedable comparator; [`unreachable_requires_bound`](../../formal/AILM/GateVerdict.lean) proves the licensing rule | [`test_only_a_bound_comparator_licenses_an_impossibility_verdict`](../../tests/test_gates.py), with the untyped-comparator fail-closed control beside it | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:gate_verdict` | `VACUOUS` if $b\succ t$; if $c\not\succ t$, `UNREACHABLE` only for a bound and `ABOVE_REFERENCE` for an exceedable comparator; otherwise `BRACKETED_BOUND` or `BRACKETED_REFERENCE`; absent $c$ yields `UNAUDITED` | `IMPLEMENTED` (one implementation, two wrappers) | [`classify_gate`](../../src/ailm/gates.py). `scripts/audit_gates.py::_verdict` and `ailm.figures.gates._classify` delegate to it, so the audit and figure cannot drift apart | [`test_gates.py`](../../tests/test_gates.py); [`test_gate_taxonomy_values_match_the_ledger`](../../tests/test_figures_results.py), with comparator-type and non-vacuity controls beside it | [`03_methods.md`](../manuscript/03_methods.md) |
| Definition (Threshold placement) | the baseline must fail, a typed comparator must pass, and the resulting bracketed verdict retains whether that comparator is a bound or an exceedable reference | `IMPLEMENTED` | same as `eq:gate_verdict` | same as `eq:gate_verdict` | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:heldout_bound` | Held-out order-1 all-position cross-entropy is the fixed uniform-start loss plus empirical transition conditional entropy and a non-negative conditional KL term, divided by all scored positions | `IMPLEMENTED` | [`evaluation_population_bound`](../../src/ailm/evaluation.py) computes the transition entropy, adds the initial `log(|A|)` loss, and reports the exact lower bound on the same all-position denominator as `compute_perplexity` | [`test_the_bound_is_below_the_declared_gate_on_the_real_passage`](../../tests/test_evaluation_population_bound.py), [`test_all_position_bound_includes_the_uniform_initial_prediction`](../../tests/test_evaluation_population_bound.py), and [`test_the_bound_really_bounds_the_shipped_model`](../../tests/test_evaluation_population_bound.py), with predictable, unpredictable, dropped-initial, wrong-population, and degenerate-passage controls | [`98_supplement_audit_trails.md`](../manuscript/98_supplement_audit_trails.md) |

| `eq:matrix_entries` | The emission, action-conditioned transition, preference and initial-state entries of the matrix tuple | `IMPLEMENTED` | Same builders and column-stochastic constraints as Definition (The AILM matrix tuple) above | Same matrix-orientation and normalization controls as that definition | [`02_background.md`](../manuscript/02_background.md) |
| `eq:message_log_loss` | Conditional information plus the change in receiver misspecification decomposes realized log-loss reduction | `HONESTLY DEFERRED` (joint empirical decomposition) | Explanatory identity; the channel sweeps do not separately estimate all its terms | No shipped test of the complete empirical decomposition is claimed | [`03_methods.md`](../manuscript/03_methods.md) |
| `eq:transition_cross_entropy` | Empirical transition cross-entropy equals conditional entropy plus conditional KL | `IMPLEMENTED` (bound) | [`evaluation_population_bound`](../../src/ailm/evaluation.py) computes the entropy lower bound; no separate KL estimate is claimed | [`test_the_bound_really_bounds_the_shipped_model`](../../tests/test_evaluation_population_bound.py) | [`98_supplement_audit_trails.md`](../manuscript/98_supplement_audit_trails.md) |
| `eq:agent_memory` | Parameter-array bytes from observation, action and hidden-factor sizes | `IMPLEMENTED` | [`agent_memory_bytes_from_shape`](../../src/ailm/complexity.py) sums emission, transition, preference, initial-state and policy-prior storage | [`test_the_memory_formula_carries_the_implicit_policy_prior`](../../tests/test_complexity.py) and live-agent reconciliation | [`98_supplement_complexity.md`](../manuscript/98_supplement_complexity.md) |

## The statistic the gate runs

`eq:mcnemar` is the two-sided exact binomial probability on the discordant pairs —
McNemar's exact test. The authoritative count-level implementation is
`ailm.statistics.exact_two_sided_binomial_p(b, c)`, including the defined
zero-discordance result. `ailm.benchmarks.scoring.exact_mcnemar_p` validates its
public inputs and delegates to that implementation. The Stage 2 recorder and
statistical-ledger reconstruction use the central helper.

Some older stage, channel, and diagnostic recorders and their independent tests
remain direct SciPy sites. They use the equivalent two-sided exact call
`scipy.stats.binomtest(b, b + c, 0.5)` (usually with an explicit
zero-discordance branch). The direct sites include the Stage 4–6 recorders,
subsample and family-split diagnostics, Czech case, the coupling level, and the
lexical/coupling/clause-channel analyses. This mixed implementation inventory is
intentional and visible; reproduce the current source locations rather than
trusting a hand-maintained call count:

```bash
rg -n '\bbinomtest\(' src scripts tests --glob '*.py'
```

One direct site is deliberately different:
`scripts/record_statistical_claims.py` passes `alternative="greater"` for the
directional Stage 6 sign test and records it under a distinct claim. It is not a
second implementation of the two-sided McNemar statistic. The displayed closed
form remains exact rather than approximate:

$$p = \min\Big(1,\ 2\textstyle\sum_{k=0}^{\min(b,c)} \binom{n}{k} 2^{-n}\Big),\qquad n = b+c$$

The doubling is exact because the null distribution is symmetric about $n/2$.
The parity grid in `tests/test_statistics.py` binds the central helper and the
benchmark wrapper to SciPy for tied, asymmetric, boundary, and shipped-count
inputs; stage-specific controls bind the remaining direct sites.

The manuscript previously displayed the continuity-corrected $\chi^2$ approximation
here instead. That is recorded in
[`decisions.md` entry 12](../decisions.md#12--the-paired-test-is-mcnemars-exact-test-not-the-chi-square-approximation),
with the reason the exact form is the right choice.

## Scope notes: what is exercised, and what remains scaffold

The formal apparatus above is wider than the shipped capability. That gap is
intentional and is stated in the manuscript; it is restated here so that reading
this file cannot leave the opposite impression.

### Generation, and policy selection under `eq:efe`

**L3→L1g generation is built.** `src/ailm/generation.py` implements
`generate_by_policy_selection` and `generate_by_sampling`. The policy-selection
path evaluates the policy posterior in closed form (`efe_policy_posterior`)
under `eq:efe` and is verified against `agent.infer_policies`; L3's lexicon
supplies the next-letter `C`, and a Brown-bigram policy prior supplies `E`. L6
and L5 do not participate. Results are reported over claim:generation_seeds
fixed seeds, with membership, total variation, and distinct-word counts across
gamma-sweep controls. A scrambled-`C` control tests the lexical seam.
`tests/test_generation.py` verifies the closed-form `efe_policy_posterior`
matches pymdp's own `infer_policies`, and `scripts/record_generation_claims.py`
records the worst deviation as claim:generation_policy_posterior_pymdp_max_abs_dev
over claim:generation_policy_posterior_pymdp_states_checked (letter, prefix)
states, refusing to write above tolerance (decision 95).

The historical generation sweep above remains distinct from
`ConnectedRuntime.generate_next_word`, which uses the context-conditioned word
prior and calls real `infer_policies`/`sample_action` through the compiled or
eager Agent path. `tests/test_connected.py` checks its policy law and bounded
termination. This establishes the implemented policy path, not fluent narrative
generation; [the connected contract](../design/connected_runtime.md) states its
scope and clocks.

Design rationale: [`design/generation_problem.md`](../design/generation_problem.md).

### The CEREBRUM case layer, and `eq:case_free_energy`

[`src/ailm/cases.py`](../../src/ailm/cases.py) is Stage 4 scaffolding. The
`claim:stage0_case_roles` linguistic roles and `claim:stage0_function_roles`
scoped CEREBRUM model roles exist as separate enumerations. The source reuses
`[GEN]` for core Genitive and a novel Generative role; their serialized values
remain distinct. CEREBRUM [Supplement {{CEREBRUM_CASE_SUPPLEMENT}}]({{CEREBRUM_CASE_SUPPLEMENT_URL}}) does **not** assign a generic scalar
$(\alpha_c,\beta_c)$ pair to every case: NOM weights the likelihood term, ACC
weights the complexity term, and GEN/DAT/INS/LOC/ABL/VOC introduce distinct
auxiliary latent variables and precision relations. `CEREBRUM_CASE_FORMULATIONS`
records those structural distinctions.

AILM separately implements `eq:case_free_energy` as a deliberately narrow scalar
approximation for NOM and ACC. `AILM_SCALAR_CASE_WEIGHTS` contains its
project-owned numerical defaults; requesting a non-scalar source case fails
closed. [`case_conditioned_free_energy`](../../src/ailm/cases.py) implements that
equation. [`transformation_cost`](../../src/ailm/cases.py) implements [Supplement {{CEREBRUM_TRANSFORMATION_SUPPLEMENT}}]({{CEREBRUM_TRANSFORMATION_SUPPLEMENT_URL}})'s distinct, unscaled
$F_{\mathrm{trans}}=F_{\mathrm{target}}+D_{\mathrm{KL}}(q_{\mathrm{target}}\|q_{\mathrm{source}})$,
while [`expected_free_energy_transition`](../../src/ailm/cases.py) implements
[Supplement {{CEREBRUM_EFE_TRANSITION_SUPPLEMENT}}]({{CEREBRUM_EFE_TRANSITION_SUPPLEMENT_URL}})'s separately $\gamma$-scaled expected-free-energy transition.
Each contract is independently tested in [`tests/test_cases.py`](../../tests/test_cases.py)
and bound to exact-PDF locators by
[`tests/test_reference_source_fidelity.py`](../../tests/test_reference_source_fidelity.py).

**They are called by nothing in the comprehension path.** No shipped decoders invoke
`case_conditioned_free_energy`, `transformation_cost`, or
`expected_free_energy_transition`. Level 4's gate (case agreement) and this
source-bounded CEREBRUM scaffold are separate.

Level 4 is built with recorded agreement outcomes: `src/ailm/levels/case.py` constructs a proper pymdp agent
over the 8 syntactic case roles (UD-derived), observes UPOS tags through a learned emission, and scores agreement
against both a gold morphology oracle and a derived oracle
([@sec:case_level]). The NOM/ACC scalar approximation and the source formulation
metadata are unused by that path — the case *assignment* model is live and measured.
The manuscript's
architecture table marks Level 4 as agreement-reported, which is the current state.

The manuscript is clear about the level: the architecture table marks Level 4 as
built with agreement reported, while the CEREBRUM-derived formulation metadata and
AILM-owned NOM/ACC scalar approximation are its unused scaffold.

The UD case oracle is separately disclaimed as a research claim rather than ground
truth, in code at [`cases.py:127`](../../src/ailm/cases.py) and in prose at
[`05_discussion.md`](../manuscript/05_discussion.md). Mapping rationale:
[`design/cerebrum_mapping.md`](../design/cerebrum_mapping.md).

### Whole-sequence inference

The oracle implements [`forward_backward`](../../src/ailm/verification/oracle.py) and
[`viterbi`](../../src/ailm/verification/oracle.py), and their scores are recorded as
`claim:stage2_g2p_smoothing_ceiling` and `claim:stage2_g2p_viterbi_ceiling`. Neither
is a shipped capability: both are *ceilings the streaming architecture deliberately
does not claim*, because `eq:filtering` conditions only on observations up to $t$.
Reported, not claimed — [`05_discussion.md`](../manuscript/05_discussion.md) works
the gap through.

### Later levels and the speech boundary

Current status:
- **L3 (word)** — trie-constrained paths; letter-bigram bar on identical positions; gate BRACKETED_REFERENCE (held-out-oracle comparator)
- **L4 (case)** — case-role agreement uses a NumPy filter that the live-pymdp recursion reproduces at every held-out position, with the agreement recorded (decision 94)
- **L5 (clause)** — canonical joint type × widened-symbol model with exact mixture scoring and a passing development-session gate (decisions 112, 114); no held-back replication. The case-only arm remains a non-passing retained ablation.
- **L6 (`narrative` in code)** — document-position terciles intervene on the clause `D` prior; the document-clustered development interval spans zero. The separately registered held-back interval excludes zero at its declared coverage, not family-corrected coverage (decision 115).
- **L7 (speech)** — re-scoped to boundary interface contract; tested in `tests/test_speech_boundary.py`

`scripts/audit_gates.py` audits every registered model/generation gate through
L6, reports each typed verdict from the code-owned registry, and reports the L7
model gate as UNAUDITED because Stage 7 is an interface contract rather than a
model.

### Connected conditional inference

The independent `_exact_marginals` enumerator in `tests/test_connected.py`
reconstructs each event’s joint state table from its incoming factor priors and
emission, using the pinned expected-log evidence convention. The runtime’s
predictions and posterior marginals are checked against that table. This is
exact conditional agreement at each update; it does not extend the manuscript’s
single-agent factorization into a globally inferred hierarchy. The frozen GUM
comparison is a separate performance measurement with explicit OOV coverage and
complete source/input binding.

## Auto-numbering and cross-reference audit

The source-side inventory is derived continuously by
`tests/test_manuscript_crossrefs.py`, not copied into a second hand-maintained
list. The PDF-side theorem numbers below are a retained historical inspection with
`pdftotext` on 2026-08-23, not the current numbering or release verdict. That
snapshot exposed a source/render boundary: its PDF said “The four matrices”
while the authored environment said “The AILM matrix tuple”. Current source
parity and post-render contracts, followed by exact-product independent PDF QA,
are authoritative for the current render; the historical observations remain
provenance.

| Check | Result |
|---|---|
| Formalisms hand-numbered in prose | none — no literal `Definition N`, `Equation N`, `Theorem N`, or `Proposition N` anywhere in `docs/manuscript/[0-9]*.md`. The one `Figure 13` match cites a figure in an external source paper. |
| Equation inventory | The source-derived inventory and map both contain exactly `21` unique equation labels; the parity test carries a planted one-row omission and reports the exact missing label |
| Theorem-like numbers in the inspected PDF | `Definition 3.1 (The four matrices)`, `Proposition 4.1 (Sequence statistics belong in B)`, `Definition 5.1 (Threshold placement)`, `Proposition 12.1 (The information-gain term vanishes)` — section-scoped, generated by the `amsthm` declarations in [`docs/manuscript/preamble.md`](../manuscript/preamble.md). The source-derived map binds the exact current titles independently of rendered numbering |
| Unresolved references | zero `??` in the extracted PDF text |
| Dangling `[@...]` references and unreferenced floats | **superseded by a test, not by this hand audit.** [`tests/test_manuscript_crossrefs.py`](../../tests/test_manuscript_crossrefs.py) asserts continuously that every `[@…]` resolves and that every table, figure and equation is pointed at by a sentence, and it carries planted violations of both rules so the scan cannot pass by finding nothing. Sections are exempt by design — a section is reachable through the contents page, a float is not — which is why `sec:abstract`, `sec:introduction`, `sec:conclusion` and `sec:references` are unreferenced without being defects |

## Boundary note

**`eq:gate_verdict` at $c = t$.** The definition makes the comparator fail when
$c \not\succ t$, which for a lower-is-better metric means $c \geq t$ — a
comparator equal to its bar establishes no bracket. The resulting verdict is
`UNREACHABLE` only when $c$ is a model-class or analytic bound; an exceedable
reference yields `ABOVE_REFERENCE`. `ailm.gates.gate_passes` treats
equality as a failure in both directions and says so in its docstring, and
`classify_gate` calls it for the ceiling as well as the baseline, so the code and
the manuscript now agree at the boundary. Earlier revisions of both wrappers
tested `ceiling > threshold` and classified that case `BRACKETED`; that
disagreement is closed by there being one implementation to agree with.

## Reproducing this audit

```bash
# label definitions, references, and dangling refs — the continuous version
.venv/bin/python -m pytest tests/test_manuscript_crossrefs.py -q

# the same two rules by hand, when you want to see the labels themselves
rg -n '\{#eq:' docs/manuscript --glob '[0-9]*.md'
rg -no '@eq:[a-z_]*' docs/manuscript --glob '[0-9]*.md'

# the rendered artifact: numbering and unresolved references
AILM_FORMALISM_TEXT="$(mktemp)"
pdftotext -layout output/pdf/active_inference_language_model_combined.pdf \
  "$AILM_FORMALISM_TEXT"
rg 'Definition|Proposition|\?\?' "$AILM_FORMALISM_TEXT"

# eq:mcnemar — reproduce the central helper and deliberate direct sites
rg -n '\bbinomtest\(' src scripts tests --glob '*.py'
rg -n 'chi2|chisquare|abs\(b - c\)|abs\(b-c\)' src scripts tests --glob '*.py'
```

Numbers referenced here are ledger labels resolved from
[`data/claims.json`](../../data/claims.json); see
[`testing_philosophy.md`](../testing_philosophy.md) for the claims discipline that
forbids hand-typing them.
