---
name: active_inference_language_model
phase: build
tier: E5
stage_current: 6
stage_total: 7
---

# ISA — Active Inference Language Model

Historical acceptance-contract record for what *done* was specified to mean.
Current scientific status is governed by [`docs/hierarchy.md`](docs/hierarchy.md)
and the chronology in [`docs/decisions.md`](docs/decisions.md). Every criterion
below names a probe that could falsify it; a criterion with no such probe is not a criterion.
Working queue lives in [`TODO.md`](TODO.md), decision evidence in
[`docs/decisions.md`](docs/decisions.md), and the archived 2026-08-21 cold-start
snapshot in [`HANDOFF.md`](HANDOFF.md), which is provenance rather than a current
handoff.

## Problem

Mechanistic interpretability of language models is hard because the models are not
mechanistic. Interpretation is retrofitted onto weights that were never conditional
probabilities of anything, so an explanation can be plausible, unfalsifiable, and
wrong at once. Separately, staged builds routinely certify themselves with gates
that cannot be failed — or cannot be passed — and neither defect announces itself:
the tests are green either way.

## Vision

A language model whose every parameter is an interpretable conditional probability
in an explicit generative model, built level by level, where each level's claim to
work survives a test that could have shown it did not. The capability cost is
accepted openly. The euphoric surprise is that "we verified this" becomes a
checkable statement rather than a reassurance.

## Out of Scope

Competitive perplexity or benchmark standing. Any neural component, learned
embedding, or transformer. Multilinguality. Speech I/O. Beating published
grapheme-to-phoneme systems. This is an interpretability and verification
artifact, not a language technology.

**Text generation moved IN SCOPE on 2026-07-30** at the principal's direction:
both recognition and generation are required, and generation will be used
extensively. It is now part of the Goal below rather than an exclusion, and
Stage 6 is built and audited; Stage 7 remains an enforced speech-interface
boundary rather than a seventh linguistic model. The apparatus that was specified-but-unexercised —
`eq:efe`, `C`, the policy space — is exercised by
`gnn/models/L1g_letter_generation.gnn.md` and `src/ailm/generation.py`. See
`docs/decisions.md` entry 17 for the gate and the two failure modes it must
straddle.

## Principles

- **Hard-to-vary equals testable.** A claim is load-bearing if and only if a named
  probe could falsify it.
- **A threshold means nothing without two measurements.** The trivial baseline and
  the achievable ceiling are what make a gate a gate.
- **Verify against an independent implementation, not against a threshold.**
  Agreement between two derivations is evidence; clearing a number is not.
- **A null outcome is a result.** A failed comparison is reported with its
  uncertainty and controls as a bound; it is never promoted into measured
  absence.
- **Reproducible is not representative.** Determinism of a sample says nothing
  about whether it stands for the population.

## Constraints

- pymdp 1.0.4 is the only generative model. No mocks, no mocking framework; all
  tests use real pymdp and real corpora.
- `src/ailm/` imports no `infrastructure` (template layer contract).
  `src/ailm/verification/` additionally imports no pymdp — its value is independence.
- Coverage floor claim:stage0_coverage_floor_pct% branch coverage
  (`pyproject.toml` `fail_under`, read into the ledger by
  `scripts/record_stage0_claims.py` so the two cannot disagree;
  `branch = true`). The current run belongs in
  `output/reports/release-receipt.json`; this instrument does not freeze a
  copied coverage result.
- Every number in prose is a `claim:<label>` backed by `data/claims.json`; every
  number in the manuscript is a `{{TOKEN}}` resolved from it. Recorders write
  claims; tests verify them.
- Commits go to the sidecar repository only, never to the template repo.
- Prospectively, Stage N+1 may not begin until Stage N's gate passes with a
  baseline and a live negative control. The historical build did not
  consistently follow that rule; the audit chronology in `docs/decisions.md`
  governs what was retroactively re-certified.

## Dependencies

- `inferactively-pymdp==1.0.4` — the generative engine.
- Brown corpus (NLTK) → `vendor/brown_counts.json`; CMU Pronouncing Dictionary →
  `vendor/cmudict_counts.json`. Both committed, both byte-identical on
  regeneration, both checksum-enforced.
- Template repo render pipeline (`scripts/pipeline/stage_03_render.py`) and
  `infrastructure.rendering.manuscript_injection`, reached via `TEMPLATE_REPO_ROOT`.
- GeneralizedNotationNotation upstream — spec grammar and, optionally, its parser
  for the interoperability check.

## Goal

Six modeled linguistic levels — letter, phoneme, word, case, clause and
narrative — plus a speech interface boundary, each held to an explicit
verification contract. Stages 0–6 are built and audited. Stage 5 is non-passing
as recorded on its registered case-only clause arm, with an unresolved privileged-label gain; its
canonical widened-observation arm (decisions 112 and 114) passes as a
development-session specification and has no held-back
replication because the held-back population was spent. Stage 6 has a positive
pre-registered point estimate whose document-clustered primary interval spans
zero on the development population (decisions 25–26 and their correction);
under decision 111 the sentence-level sign test replicates on the held-back
split, and the separately registered document-level estimand (decision 115)
excludes zero there. Stage 7's boundary contract is enforced in code.

**Both directions, without inventing a cascade.** Recognition updates beliefs
within each separately evaluated level. The L3→L1g generation seam supplies the
generation Agent's `C`, after which actions are selected by minimizing expected
free energy. The L6→L5 and L5→L3 experiments instead intervene on `D`, and the
three seams are independently measured rather than connected into one run. The
artifact is a rendered manuscript in which no quantitative sentence can drift
from the measurement that produced it.

## Criteria

All criteria below are met; probes named. (Stages 0–6 built and audited; Stage 7 is an enforced interface contract — see Goal.)

**Infrastructure**

- [x] ISC-1: Claims ledger enforced against **real** repository prose, not fixtures
      — probe `tests/test_claims_discipline.py`, which carries a positive control
      planting an unbacked tag.
- [x] ISC-2: Every gate has a positive **and** negative control; a vacuous gate
      raises — probe
      `tests/test_control_harness.py::TestControlHarness::test_raises_gate_vacuous_error_when_identical`.
- [x] ISC-3: A closed-form oracle exists that shares no code with pymdp and
      agrees with it — probe `tests/test_oracle.py`, the coverage gate over
      `verification/`, and cross-checks in the stage gates.
- [x] ISC-4: Exactly one producer writes any claim label; re-running the recorders
      is a no-op — probe `tests/test_claims_single_writer.py`.
- [x] ISC-5: Vendored artifacts regenerate byte-identically and their documented
      checksums match the files — probe `tests/test_provenance_checksums.py`.

**Level 1 — letters**

- [x] ISC-6: Letter perplexity beats the context-free unigram bar on held-out
      prose — probe `tests/test_stage1_gate.py`. `claim:stage1_letter_perplexity`
      against `claim:stage1_unigram_baseline`.
- [x] ISC-7: The learned bigram sits in the transition `B`, never the emission `A`,
      and a swapped agent fails — probe the anti-informative negative control in
      `tests/test_stage1_gate.py` plus `tests/test_gnn_conformance.py`.
- [x] ISC-8: The model does not beat its own order-1 information floor — probe the
      floor assertion in `tests/test_stage1_gate.py`.

**Level 2 — phonemes**

- [x] ISC-9: Phone-sequence perplexity beats the context-free unigram bar over the
      **full** dictionary in closed form — probe
      `tests/test_phoneme_full_corpus.py`, verified against a sequence walk.
- [x] ISC-10: G2P accuracy beats the per-letter argmax bar with a **significant
      paired** test over all held-out positions — probe `tests/test_stage2_gate.py`.
      `claim:stage2_g2p_mcnemar_p` over `claim:stage2_g2p_eval_positions`.
- [x] ISC-11: The significance machinery reports *no effect* when given two
      identical predictors — probe the self-vs-self control in
      `tests/test_stage2_gate.py`.
- [x] ISC-12: pymdp and the oracle decode identically, licensing oracle-computed
      full-set statistics — probe `test_stage2_decoder_equivalence`;
      `claim:stage2_decoder_disagreements` is 0.
- [x] ISC-13: A bounded eval subsample spans the alphabet — probe
      `test_stage2_eval_subsample_spans_the_alphabet`.

**Hierarchical composition**

- [x] ISC-40: The project has a **criterion** for when an inter-level channel
      helps, not merely a list of channels that did and did not — the
      novel-information criterion, `docs/decisions.md` entry 18.
- [x] ISC-41: That criterion made a **pre-registered prediction** which was
      recorded before the measurement and then met on the development
      population: a lexical message into L2B helps by
      `claim:lexical_best_margin_pts` points at
      p = `claim:lexical_best_mcnemar_p`, with the predicted interior optimum.
      It is a retrospective corrected-threshold survivor, not an untouched-
      population confirmation —
      probe `tests/test_lexical_channel.py::test_the_lexical_channel_helps` and
      `tests/test_lexical_channel.py::test_the_sweep_does_not_fall_monotonically_and_the_argmax_is_interior`;
      the companion
      `tests/test_lexical_channel.py::test_the_interior_argmax_is_not_evidence_that_over_weighting_costs_accuracy`
      bounds what the interior optimum shows (an ordering, not a reversal).
- [x] ISC-42: The comparison is **controlled** — same receiving level as the
      harmful channel of entry 9, different sender — so the evidence bears on the
      criterion rather than on the two channels separately.
- [x] ISC-43: The channel is not circular: it conditions on the letter *prefix*,
      not word identity, and is estimated on train and evaluated on held-out
      pairs, so it can be and is wrong — probe
      `tests/test_lexical_channel.py::test_the_channel_actually_conditions_on_prefixes`.

**Generation**

- [x] ISC-35: A level has a **non-trivial action set and a non-flat `C`**, and
      the policy machinery is exercised rather than specified — probe
      `tests/test_generation.py::test_the_agent_has_a_real_action_set` and
      `gnn/models/L1g_letter_generation.gnn.md`.
- [x] ISC-36: Generated output beats a **sampling-from-`B` baseline** on a stated
      metric, and the metric is not one that cannot fail: membership is green by
      construction and is therefore rejected as the gate in favor of
      total-variation distance to the true word distribution — probe
      `tests/test_generation.py::test_generation_beats_the_sampling_baseline` and
      `tests/test_generation.py::test_membership_is_green_by_construction_and_so_is_not_the_gate`.
- [x] ISC-37: The gate fails in **both** directions. The precision optimum is
      interior, so neither "no goal" nor "pure goal" passes — probe
      `tests/test_generation.py::test_the_precision_optimum_is_interior`. No recognition gate here has
      this property.
- [x] ISC-38: The top-down channel has an **exact identity anchor**: with `C = 0`
      the generator reproduces ancestral sampling character-for-character, and
      the recorder refuses to write if it does not — probe
      `tests/test_generation.py::test_flat_preference_reproduces_the_baseline_exactly` plus its
      non-vacuity companion.
- [x] ISC-39: The closed-form policy posterior agrees with pymdp's own
      `infer_policies`, so the generated text describes the pymdp agent — probe
      `tests/test_generation.py::test_closed_form_efe_matches_pymdp_infer_policies`.

**Specification and configurability**

- [x] ISC-14: Every level has a GNN spec that parses with the **upstream** parser
      and whose declared structure is checked against the **live** agent — probe
      `tests/test_gnn_conformance.py`, including a shape-identical A/B swap that
      must fail.
- [x] ISC-33: Every GNN spec's identity — hierarchy position, base model, and
      injected mutation — is derivable from its filename by a grammar enforced
      **default-deny** over the whole tree, and every mutation is a spec that must
      FAIL conformance against the correct live agent — probe
      `tests/test_gnn_registry.py::test_no_spec_file_escapes_the_grammar` and the
      per-level `test_spec_mutations_are_rejected` cases.
- [x] ISC-34: Conformance generalizes to multi-factor models without loosening
      the single-factor rules: one transition and one prior **per factor**, each
      bound to its factor by name, and a modality's state axes checked in factor
      order so transposing two factors is a violation rather than a reshape —
      probe `tests/test_stage3_gate.py::test_live_agent_conforms_to_its_spec` and
      `L3_word@transposed_factors`.
- [x] ISC-15: Every configuration knob is provably *consumed*: changing it changes
      the artifact in the direction the maths predicts, and the default is
      bit-identical to the prior behavior — probe `tests/test_config.py`.
- [x] ISC-44: The spec **builds** the agent: `ailm.spec_agent.build_agent_from_spec`
      constructs a live pymdp agent from a GNN spec alone, resolving every
      `Learned(...)` form through an explicit **default-deny** registry keyed on
      the complete provenance keywords *and* the ontology role — a registered
      generator annotated on the wrong slot is refused at construction — probe
      `tests/test_spec_agent.py::test_a_registered_provenance_on_the_wrong_role_is_refused`
      and `tests/test_spec_agent.py::test_an_unregistered_provenance_is_refused`.
- [x] ISC-45: For every shipped level the spec-built agent is **bit-identical**
      — shapes, dtypes, bytes across `A`, `B`, `C`, `D`, `E` — to the
      builder-built agent every recorded claim came from, with a
      planted-perturbation control proving the comparator can fail, and a
      discovery guard refusing a canonical spec with no identity test — probe
      the six `test_*_spec_build_is_bit_identical` cases plus
      `tests/test_spec_agent.py::test_every_canonical_spec_is_covered_by_an_identity_test`.
- [x] ISC-46: Every mutation spec either **fails construction** or constructs an
      agent that is provably not the shipped model, and the partition between
      those outcomes is asserted exactly rather than left to drift — probe
      `tests/test_spec_agent.py::test_no_mutation_reproduces_the_shipped_model`
      and `tests/test_spec_agent.py::test_the_mutation_partition_is_exact`. Decision 20 records the one
      divergence construction caught that conformance could not: the L1g spec's
      `E` misdescribed the constructed agent.

**Computational complexity**

- [x] ISC-47: Every complexity claim is an exact reconciliation: the closed-form
      operation count must equal an **instrumented execution** of the shipped
      recursion on the shipped gate workload, to the last integer, and the
      recorder refuses to write on any disagreement — probe
      `tests/test_complexity.py` theory==counted cases plus the off-by-one,
      dropped-sequence, and mis-stated-decode negative controls that must fail.
      No claim is a timing: wall-clock cannot satisfy ISC-4's idempotence
      (decision 21).
- [x] ISC-48: Every level's memory is **spec-priced**: bytes derived from the
      GNN spec's declared dimensions (plus pymdp's implicit one-scalar policy
      prior) must equal the live spec-built agent's actual bytes, and the
      formula must not reconcile with the implicit-`E` term removed — probe
      `tests/test_complexity.py::test_spec_memory_formula_equals_live_agent_bytes` and
      `tests/test_complexity.py::test_the_memory_formula_carries_the_implicit_policy_prior`.
- [x] ISC-49: Complexity workloads ARE the gate workloads — the counted
      populations are bound to the accuracy claims' populations
      (`complexity_l2b_positions` = `stage2_g2p_eval_positions`,
      `complexity_l3_tokens` = `stage3_eval_tokens`) so cost and accuracy
      describe the same measurements — probe
      `tests/test_complexity.py::test_complexity_workloads_are_the_gates_workloads`.

**Parameter learning (W2)**

- [x] ISC-50: pymdp's learning machinery is exercised and anchored: the
      prediction that `infer_parameters` implements conjugate Dirichlet
      counting under observed states was **pre-registered and committed before
      the measurement** (`c760357`, decision 22) and then verified on the
      development population — matched
      priors agree to `claim:learning_matched_max_abs_delta`, a mismatched
      prior breaks agreement by `claim:learning_mismatched_max_abs_delta` and
      re-matching restores it, with identical held-out perplexity as the
      corollary — probe `tests/test_learning.py`, including the toy anchor
      that checks the Dirichlet increments at integer resolution and the
      lr≠1 control proving the learning rate is load-bearing. Learned matrices
      write only `learning_*` labels, never `stage*` ones (entry 10 at
      parameter scale).

**Figures and captions**

- [x] ISC-51: Every figure caption outside a frozen, shrink-only legacy
      allowlist states what would falsify its panel, and the scan proves its
      own non-vacuity (finds the figures, catches a planted non-compliant
      caption, refuses stale allowlist entries) — probe
      `tests/test_caption_standard.py`; standard in `docs/method/style_guide.md`.
- [x] ISC-52: The interpretability claim is *shown*, not only argued: Level
      4's entire generative model is rendered cell-by-cell with real labels,
      Level 3's factors and induced-trie narrowing are drawn from the live
      builders over the real held-out stream, and the generation regimes are
      shown as actual generated text — probes: the five generators in
      `ailm.figures.{interpretability,samples,learning}`, each with a
      refuse-to-draw guard binding it to the ledger or to the property it
      illustrates.

**Manuscript**

- [x] ISC-16: No quantitative sentence contains a literal; every number resolves
      from the ledger — probe `tests/test_manuscript_tokens.py`.
- [x] ISC-17: The rendered PDF contains zero unresolved cross-references and zero
      unresolved tokens — probe pdftotext scan of `output/pdf/*.pdf`.
- [x] ISC-18: Every figure is free of overlapping and clipped text, and discovery
      covers every shipped figure — probe `tests/test_figure_layout.py`.
- [x] ISC-19: Every declared formalism is IMPLEMENTED or explicitly marked not
      exercised — probe `docs/reference/formalism_map.md` against the code.

**Level 3 — words**

- [x] ISC-20: Word level admitted by a gate derived from a measured baseline and
      a measured ceiling — **not** the borrowed constant 500, which the measured
      bar clears by `claim:stage3_declared_gate_slack`x — probe
      `tests/test_stage3_gate.py`, `scripts/audit_gates.py::audit_stage3`.
- [x] ISC-25: The level's parameters are pinned by a closed-form identity, not by
      the accuracy bar: the emission-path probability equals the word's unigram
      probability to `claim:stage3_path_identity_max_abs_err` nats between a live
      pymdp agent and an oracle importing neither pymdp nor jax — probe
      `tests/test_stage3_gate.py::test_path_probability_equals_the_word_unigram`, with a planted-perturbation
      non-vacuity control.
- [x] ISC-26: **Three of four** mis-wirings clear the accuracy bar, and that fact
      is asserted in executable form rather than remarked on — probe
      `tests/test_stage3_gate.py::test_the_bar_does_not_catch_the_prior_miswirings`. A future reader cannot
      conclude that `assert model < bar` is a sufficient gate.
- [x] ISC-27: The trie the design specifies is *induced* by the factorization
      rather than abandoned: the posterior's support after a prefix equals the
      vocabulary words carrying it, and narrows as the prefix grows — probe
      `tests/test_stage3_gate.py::test_posterior_support_is_the_trie_node` plus its non-vacuity companion.
- [x] ISC-28: The closed-vocabulary bound is reported, not hidden.
      `claim:stage3_coverage_pct`% of held-out tokens are in vocabulary and the
      scorer **raises** on the rest rather than smoothing — probe
      `tests/test_stage3_gate.py::test_out_of_vocabulary_raises_rather_than_smoothing`.

**Level 4 — cases**

- [x] ISC-21: Case level reports *agreement* with the UD oracle, never accuracy.
      The function is named `case_agreement`; `claim:stage4_cases_without_morphology`
      of 8 roles have no morphological ground truth in English EWT — probe
      `tests/test_stage4_gate.py::test_five_of_eight_cases_have_no_morphological_ground_truth`.
- [x] ISC-29: The model never observes the fields its labels are derived from.
      This is the only mis-wiring in the project that would make a model look
      *better*, so the guard is structural rather than numeric — probe
      `tests/test_stage4_gate.py::test_the_observation_never_carries_the_label_source`.
- [x] ISC-30: The declared gate's **two population-dependent verdicts** are re-derived from
      one threshold — vacuous on gold, above-reference on derived — probe
      `tests/test_stage4_gate.py::test_the_two_verdicts_come_from_one_threshold`.
- [x] ISC-31: The gold-population null is shown to be empirical rather than
      promoted to an impossibility: the observation alphabet has size
      `claim:stage4_gold_upos_alphabet_size`, but an alternating one-tag control
      proves the transition can still create discordance — probes
      `tests/test_stage4_gate.py::test_the_gold_observation_alphabet_is_a_single_tag`
      and `tests/test_stage4_gate.py::test_a_single_tag_does_not_force_decoder_baseline_agreement`.
- [x] ISC-32: The syntactic heuristic's agreement with gold morphology is measured at
      `claim:stage4_heuristic_gold_agreement_pct`%, computed by bypassing the
      oracle's morphological shortcut. Its overlap includes
      `claim:stage4_heuristic_gold_adposition_token_tokens` token-level
      adposition-route examples, so it is not a bound on the full derived
      population; decision 84 records the path and ADP/head sufficient counts — probe
      `tests/test_stage4_gate.py::test_the_heuristic_fidelity_is_measured_on_its_named_overlap`.

**Level 5 — clauses (non-passing; privileged-label gain unresolved)**

- [x] ISC-22: The clause level's declared free-energy claim was audited and
      shown VACUOUS executably (cumulative free energy is monotone under ANY
      model), and the honest replacement is a **non-passing gate with an
      unresolved paired gain**: the privileged-label reference has
      a point gain of `claim:stage5_oracle_gain_nats_per_sentence` nats per
      sentence, with a `claim:stage5_oracle_gain_ci_confidence_pct`% paired
      whole-document-bootstrap interval from
      `claim:stage5_oracle_gain_sentence_ci_low` to
      `claim:stage5_oracle_gain_sentence_ci_high`, resampling
      `claim:stage5_oracle_gain_bootstrap_documents` contributing documents.
      The position-weighted interval also spans zero; both the mixture and the
      fitted privileged reference fail the derived bar (`ABOVE_REFERENCE`), and the
      identical harness fires
      decisively on larger, balanced planted synthetic types. That reference is
      a receiver-sensitivity probe, not a hard information bound, and the
      synthetic operating point does not prove sensitivity to the real effect — probes
      `tests/test_stage5_gate.py` and
      `scripts/audit_gates.py::audit_stage5`, which enforces the non-passing and
      unresolved result rather than demanding a pass. Decisions 24, 83, and 85.

**Level 6 — narrative (positive point estimate; primary non-confirmatory)**

- [x] ISC-23: The narrative level's independent top-down `D`-prior intervention
      demonstrably changes clause priors — the α=0 anchor reproduces the
      clause prior byte-for-byte, the
      α=1 opening-phase message shifts P(interrogative) by
      `claim:stage6_opening_interrogative_shift`, and the phase-permuted
      scramble lands elsewhere. The registered recognition score has a positive
      point estimate of `claim:stage6_logloss_improvement_nats` nats, but the
      primary document-clustered interval from
      `claim:cluster_stage6_logloss_ci_low` to
      `claim:cluster_stage6_logloss_ci_high` spans zero at
      p = `claim:cluster_stage6_logloss_p`, so it is imprecise and
      non-confirmatory; no target-effect power analysis is recorded. The sentence sign test is secondary, and Stage 5's
      `claim:stage5_privileged_label_gain_pct`% privileged-label value is a sensitivity
      reference rather than a mathematical bound on the intervention — probes
      `tests/test_stage6_gate.py`, decisions 25–26 and their correction.

**Level 7 — speech (an interface, not a model)**

- [x] ISC-24: The speech level's DSP boundary is stated as outside pymdp
      (`docs/design/speech_boundary.md`) and the stated boundary is now
      *enforced*: the boundary vocabulary is the Level 2 inventory by
      identity, round-trip fidelity holds over every vendored CMUdict
      pronunciation, stress digits are stripped, and everything else is
      rejected fail-closed — probe `tests/test_speech_boundary.py`. The
      declared "Whisper → phoneme → TTS round trip" gate is re-scoped in the
      audit: it would measure someone else's models.

## Open Criteria — protocol v4 and the fairness program

Everything above is met. These are not, and each names the probe that would
settle it. A criterion here with no such probe is a wish and should be deleted
rather than carried.

### Landing the census

- [ ] OSC-1: The v4 ledger differs from `handoff/claims.pre-v4-census.json` only
      in ways decision 36 predicted **or** decision 39's probe retirement
      accounts for, and any value that moved is explained rather than absorbed —
      probe: diff the two ledgers, partition the difference into those two
      classes, and read the first against decision 36's five registered
      predictions. The retirement class is bounded in advance: exactly the
      retired probe's labels disappear and no surviving probe's label changes
      because of it, so a moved value on a served probe is a finding and not
      a consequence of the retirement. Widening either class after seeing the
      diff is what this criterion exists to prevent.
- [ ] OSC-2: No sentence in the manuscript is falsified by the new values —
      probe: `tests/test_benchmark_interpretations.py`, which re-derives the
      supplement's English claims from the ledger. It is the tripwire, so it runs
      first, before anything is regenerated.
- [ ] OSC-3: Both benchmark recorders describe one run — probe: re-record the
      synthetic arm and confirm `bench_protocol_version` agrees across the
      `bench_` namespace.
- [ ] OSC-4: The two places stating that no ledger label carries the per-model
      rule divergence no longer say so — probe: grep
      `98_supplement_external_benchmarks.md` and `13_limitations.md` for the
      claim, which the run makes false.

### Bounding what is currently unbounded

- [x] OSC-5: Every null the project reports is either bounded or explicitly named
      unbindable with the missing quantity stated — probe: all three arms now
      carry the missing quantity, recorded rather than approximated. The
      clause → word channel records `clause_word_best_delta_sd`, the dispersion
      of the per-token log-probability difference, which is what a mean needs and
      a share count cannot supply (decision 65). The two subsample arms record
      `subsample_stride{5,3}_mcnemar_{b,c}` (decision 64). Bounds are derived at
      render time from the counts, never stored beside them.
- [x] OSC-6: The project states one coverage level deliberately, or states why it
      carries two — probe: one level, chosen. `SHARE_CONFIDENCE` is now
      `DIFFERENCE_CONFIDENCE` and decision 66 records why, including that the two
      are one interval under an affine rescaling and sat backwards on their roles.
      The label no longer carries the level: `stat_*_ci99_*` became
      `stat_*_share_ci_*`, because a name encoding a coverage goes false the
      moment the coverage moves and nothing in the ledger can notice.

### Fairness

- [ ] OSC-7: At least one benchmark arm satisfies "equal opportunity to fit the
      task" — probe: the supplement's own fairness table, which currently records
      that condition as holding nowhere. A few-shot or fitted-probe arm would be
      the first version where it holds, and this is the largest remaining threat
      to the comparison.
- [ ] OSC-8: Contamination is controlled in more than the synthetic arm — probe:
      the same table. Brown and CMUdict exposure is undeclared and unmeasurable
      for the probes and, if present, acts in their favor.
- [ ] OSC-9: The calibration gate tests the quantity it cares about rather than a
      proxy for it — probe: the offline sweep showed the declared captured-mass
      floor is not conservative and that a conservative one is unreachable through
      the API. The dual-rule check measures the right quantity directly; adopting
      it as the gate is a decision entry plus a re-run.

## Bridge Criteria

Verified across seams this project does not own.

- [x] BISC-1: The manuscript renders through the template pipeline for a
      **symlinked sidecar** project, resolving `infrastructure` via
      `TEMPLATE_REPO_ROOT` rather than a relative walk-up — probe a real render
      from the template repo.
- [x] BISC-2: AILM's GNN specs are accepted by the upstream GNN parser, and the
      two readers agree on dimensions, ontology, and parameters — probe the
      interoperability test, which skips cleanly when the checkout is absent.
- [x] BISC-3: The template citation-validator fix is **committed upstream**
      (template repo `5bf4fa69a`). Its focused and full validation, ruff and mypy
      passed at the time of that historical upstream commit; re-run the template
      repository's current gates before treating that receipt as current. It was
      not pushed from this project — the remote is public, which is the
      principal's call, not this project's.

## Test Strategy

Real corpora, real pymdp, no mocks. Three independent guards on the load-bearing
wiring invariant: a closed-form numpy oracle, a measured context-free comparator,
and GNN conformance naming each matrix's role. Every gate pairs a positive control
with a negative control that must fail; a gate whose negative control passes raises
`GateVacuousError` rather than reporting success. Thin margins are established by
paired tests over the full held-out set, never by aggregate differences on a
subsample. Where a full pymdp pass is infeasible, decoder equivalence with the
oracle is proven first and the oracle stands in.

## Features

| name | description | satisfies | depends_on | parallelizable |
|---|---|---|---|---|
| letter level | 30-symbol bigram agent | ISC-6..8 | infrastructure | no |
| phoneme level | phone bigram + G2P | ISC-9..13 | letter level | no |
| oracle | independent closed-form inference | ISC-3 | — | yes |
| gnn layer | specs + conformance checker | ISC-14 | levels | yes |
| config | consumed hyperparameters | ISC-15 | levels | yes |
| manuscript | ledger-bound prose + figures | ISC-16..19 | all levels | yes |
| word level | trie-constrained paths | ISC-20 | phoneme level | no |

## Decisions

Full evidence in [`docs/decisions.md`](docs/decisions.md). Standing decisions:
learned dynamics in `B`; context-free comparator not uniform; thresholds derived
not borrowed; eval subsampling by even stride; thin margins by paired test; Part A
over the full dictionary in closed form; filtering not smoothing; recorders write
and tests verify; inter-level coupling implemented, measured harmful, not shipped.

## Changelog

**2026-07-30 — the spec describes a model the constructor never built**
- conjectured: the L1g spec's `E=Learned(bigram)` describes the generative
  agent's policy prior.
- refuted by: the constructed agent's `E` is uniform; the habit enters per step
  as a function argument, exactly as `C` receives the top-down preference. No
  conformance check binds `E`, so the divergence was invisible until
  `build_agent_from_spec` had to construct what the spec declared.
- learned: conformance can bridge two sources of truth and still let them
  describe different models; only making one source *build* the artifact forces
  every declaration to be constructively true.
- criterion now: ISC-44..46 — the spec constructs the agent behind a
  default-deny registry, bit-identity is the migration gate, and mutations must
  fail construction or construct a provably different model.

**2026-07-30 — the triangle stated wrongly, caught by recording the number**
- conjectured: the narrative phase transition, left-to-right by construction,
  has a zero strictly-LOWER triangle.
- refuted by: the recorded mass below the diagonal was nonzero — because in
  B[next, current] orientation a non-decreasing phase puts its mass ON and
  BELOW the diagonal; it is the strictly UPPER triangle that vanishes, and
  does (claim:stage6_transition_above_diagonal).
- learned: the orientation trap of decision 1 arrives in prose as readily as
  in matrices, and the defense is identical — record the invariant and let
  the number contradict the sentence.
- criterion now: the recorder refuses to write if the upper triangle is
  non-zero; `tests/test_stage6_gate.py` asserts it.

**2026-07-30 — the memory formula that was four bytes short**
- conjectured: a live agent's parameter bytes equal the sum of its spec's
  declared dimensions times four.
- refuted by: every single-action level measured exactly four bytes over the
  formula — pymdp materialises a policy prior `E` even when the policy space is
  one action, one float32 scalar.
- learned: when prediction and measurement disagree, the temptation is to
  filter the measurement (skip `E` when no `PolicyPrior` is declared); the
  honest fix prices the term in the formula, because a filtered measurement
  would also have hidden a real unpriced `E` at a controlled level.
- criterion now: ISC-48, with the term's removal as a tested negative control.

**2026-07-29 — the declared Stage 1 gate**
- conjectured: letter perplexity should beat uniform by
  claim:stage1_declared_gate_nats nats.
- refuted by: the conditional entropy of the Brown bigram distribution caps the
  achievable gain at `claim:stage1_max_bigram_gain_nats` nats — roughly half the
  demand. No order-1 letter model can pass.
- learned: a threshold borrowed rather than derived can be unreachable, and the
  honest response to a correct implementation then looks identical to the response
  to a broken one.
- criterion now: ISC-6 gates against the measured context-free unigram bar.

**2026-07-29 — the declared Stage 2 gate**
- conjectured: G2P should reach a fixed percentage target on the UD EWT oracle.
- refuted by: UD EWT carries no phonemic annotation at all, so it cannot score a
  phoneme task; and a context-free per-letter lookup table already reaches
  `claim:stage2_g2p_baseline_accuracy`%, so the declared threshold admits a model that does nothing
  hierarchical.
- learned: gates fail in two opposite directions — unreachable and vacuous — and
  the same two measurements detect both.
- criterion now: ISC-10 gates against the per-letter argmax bar with a paired test.

**2026-07-29 — where the learned matrix goes**
- conjectured: the bigram table belongs in the emission `A` with an identity `B`.
- refuted by: that wiring is anti-informative — observing a symbol yields a
  posterior over its *predecessor*. Measured
  `claim:stage1_anti_informative_perplexity` against
  `claim:stage1_letter_perplexity`, and worse than uniform in-sample, while still
  clearing a naive threshold. pymdp validates normalization only, so it accepts
  either slot.
- learned: a structurally valid model can be semantically inverted; only an
  independent numeric derivation catches it.
- criterion now: ISC-7, guarded three ways including GNN role annotation.

**2026-07-29 — eval subsampling**
- conjectured: a deterministic prefix is a sound bounded eval because it is
  reproducible.
- refuted by: the pair list is sorted alphabetically, so the prefix held only
  a/b/c words and overstated the G2P margin threefold; a *representative* stride of
  the same size was then underpowered.
- learned: reproducible and representative are independent properties, and a thin
  margin needs the full set plus a paired test.
- criterion now: ISC-13 and the paired requirement in ISC-10.

**2026-07-29 — inter-level coupling**
- conjectured: passing the letter level's predictive distribution into the G2P
  likelihood should improve phoneme inference.
- refuted by: measured significantly worse — `claim:coupling_margin_pts` points,
  McNemar p = `claim:coupling_mcnemar_p`, with the alpha=0 identity anchor exact
  and the scrambled control confirming the message did carry letter information.
- learned: at position *t* the G2P level already observes letter *t*, while the
  letter predictive depends only on `0..t-1` — strictly less information about the
  observed symbol. The composition is a shared alphabet, not a message channel.
- criterion now: no coupling ISC. `claim:coupling_supported` is 0 and the
  architecture figure describes what the code does.

**2026-07-29 — claim provenance**
- conjectured: a gate test may record the values it measures.
- refuted by: a recorder and a test wrote the same label from different
  measurements, so the stored value depended on execution order and a rendered PDF
  asserted the wrong number while every test passed.
- learned: single-writer discipline is a correctness property of the ledger, not a
  tidiness preference.
- criterion now: ISC-4.

## Verification

Run from the project root with the project venv:

```
.venv/bin/python -I -S scripts/verify_release.py full --keep-going # canonical acceptance; receipt -> output/reports/release-receipt.json
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing
.venv/bin/python scripts/audit_gates.py
.venv/bin/python -m pytest tests/test_claims_discipline.py \
    tests/test_claims_single_writer.py tests/test_manuscript_tokens.py \
    tests/test_gnn_conformance.py -q
```

Render and verify the artifact from the template repo:

```
cd /path/to/template
UV_NO_CACHE=1 uv run --locked python scripts/pipeline/stage_03_render.py \
    --project working/active_inference_language_model
```

Then confirm the PDF, not the exit code: zero `??` in extracted text, zero
unresolved `{{TOKEN}}`, and every figure present. A render can exit 0 with every
cross-reference unresolved.

Current verification is deliberately not copied into this instrument. Read
`output/reports/release-receipt.json` for the last executed release profile and
`output/reports/gate_audit.json` for the last typed gate export, then verify that
each receipt describes the current `HEAD` and input state. A historical green or
red leg is evidence about that run only. The same rule applies to suite counts,
coverage, rendered page counts, and formal theorem counts: obtain them from their
own receipts or claim-backed outputs rather than from a hand-maintained summary
here.

Re-verify the suite, receipt, formal build, and extracted PDF before trusting a
publication state.
