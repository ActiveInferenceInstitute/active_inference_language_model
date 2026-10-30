/-
  Generated axiom report driver; do not hand-edit.
  Writer: scripts/record_formal_claims.py --generate-axiom-script

  Run from `formal/`:

      ./Scripts/axiom_report.sh

  Prints the axiom dependencies of **every named theorem** in the tree. This
  file is deliberately outside the `AILM/` library root so `lake build` does not
  compile it; the report is a separate, explicit step.

  ## Allowed axiom set

      propext, Classical.choice, Quot.sound, funext

  `Classical.choice` is on that list for a measured reason, not a permissive
  one: Lean **core's own** `Rat.add_comm` reports
  `[propext, Classical.choice, Quot.sound]`. A gate demanding only
  `propext / Quot.sound / funext` would be unsatisfiable against core `Rat` and
  would fail for reasons that have nothing to do with this project's proofs.

  What the allowlist deliberately excludes:

  * `sorryAx`   — an incomplete proof. `sorry` is a *warning*, not an error, so
                  a nonzero exit code alone would not catch it.
  * `Lean.ofReduceBool` — the trusted-code axiom `native_decide` introduces.
                  `native_decide` is not used anywhere in this tree: every
                  computed fact closes by `decide`, which is kernel-checked.
                  If this axiom ever appears in the report, some proof started
                  trusting the compiler instead of the kernel.

  Entries are generated from the same declaration scan the recorder uses.
  Its independent comparison with Lean output still checks coverage both ways.
  Source scanning alone is not evidence of a compiled or axiom-checked proof.
-/
import AILM

/-! ## AILM.Bonferroni -/
#print axioms AILM.sum_le_length_mul
#print axioms AILM.bonferroni_union_bound
#print axioms AILM.bonferroni_threshold_scales
#print axioms AILM.bonferroni_union_bound_div
#print axioms AILM.witnessFamily_length
#print axioms AILM.witnessFamily_bounded
#print axioms AILM.witnessFamily_scale
#print axioms AILM.witnessFamily_bound
#print axioms AILM.witnessFamily_nonempty
#print axioms AILM.witnessFamily_corrected_threshold
#print axioms AILM.witnessFamily_bounded_by_corrected_threshold
#print axioms AILM.witnessFamily_bound_div
#print axioms AILM.scaleHypothesisIsLoadBearing
#print axioms AILM.zeroFamilyThresholdCollapses

/-! ## AILM.Filtering -/
#print axioms AILM.sum_div
#print axioms AILM.normalize_sum
#print axioms AILM.normalize_nonneg
#print axioms AILM.normalize_isDistribution
#print axioms AILM.evWitness_sum
#print axioms AILM.evWitness_not_distribution
#print axioms AILM.evWitness_sum_ne_zero
#print axioms AILM.evWitness_pos
#print axioms AILM.evWitness_nonneg
#print axioms AILM.evWitness_normalizes
#print axioms AILM.zeroMassCounterexample
#print axioms AILM.smul_length
#print axioms AILM.vadd_length
#print axioms AILM.predict_length
#print axioms AILM.smul_sum
#print axioms AILM.zeroVec_sum
#print axioms AILM.vadd_sum
#print axioms AILM.predict_sum
#print axioms AILM.smul_nonneg
#print axioms AILM.zeroVec_nonneg
#print axioms AILM.vadd_nonneg
#print axioms AILM.predict_nonneg
#print axioms AILM.predict_isDistribution
#print axioms AILM.transWitness_length
#print axioms AILM.transWitness_width
#print axioms AILM.transWitness_columns
#print axioms AILM.priorWitness_isDistribution
#print axioms AILM.predictWitness_mixes
#print axioms AILM.predictWitness_isDistribution
#print axioms AILM.raggedColumnLosesMass
#print axioms AILM.raggedTransition_meets_the_other_hypotheses
#print axioms AILM.leakyColumnLosesMass
#print axioms AILM.leakyTransition_meets_the_other_hypotheses
#print axioms AILM.missingColumnLosesMass
#print axioms AILM.shortTransition_meets_the_other_hypotheses

/-! ## AILM.GateVerdict -/
#print axioms AILM.instDecidablePasses
#print axioms AILM.instDecidableSatisfies
#print axioms AILM.passes_irrefl
#print axioms AILM.satisfies_of_passes
#print axioms AILM.passes_iff_satisfies_and_ne
#print axioms AILM.classify_some_cases
#print axioms AILM.classify_none_cases
#print axioms AILM.bracketedFor_isBracketed
#print axioms AILM.beyondFor_not_isBracketed
#print axioms AILM.bracketedFor_ne_beyondFor
#print axioms AILM.beyondFor_eq_unreachable_iff
#print axioms AILM.beyondFor_eq_aboveReference_iff
#print axioms AILM.bracketed_iff
#print axioms AILM.vacuous_iff
#print axioms AILM.no_co_fire
#print axioms AILM.degenerate_no_gate
#print axioms AILM.unreachable_of_bound_fails
#print axioms AILM.above_reference_of_reference_fails
#print axioms AILM.bracketed_at_baseline_bar_iff
#print axioms AILM.unreachable_requires_bound
#print axioms AILM.above_reference_requires_exceedable
#print axioms AILM.no_co_fire_hypothesis_inhabited
#print axioms AILM.unreachable_hypotheses_inhabited

/-! ## AILM.Holm -/
#print axioms AILM.inv_antitone_of_pos
#print axioms AILM.div_antitone_in_denominator
#print axioms AILM.bonferroni_le_holm_threshold
#print axioms AILM.bonferroni_le_holm_threshold_at_every_rank
#print axioms AILM.holm_threshold_monotone_in_rank
#print axioms AILM.holm_rank_one_is_bonferroni
#print axioms AILM.stepDownCount_mono_in_bar
#print axioms AILM.countBelow_eq_zero_of_all_at_least
#print axioms AILM.sorted_countBelow_eq_stepDown_const
#print axioms AILM.holm_rejects_every_bonferroni_rejection
#print axioms AILM.witnessFamilySorted_length
#print axioms AILM.witnessFamilySorted_ascending
#print axioms AILM.witnessFamilySorted_bars
#print axioms AILM.witnessFamilySorted_fixed_bar
#print axioms AILM.witnessFamilySorted_holm_count
#print axioms AILM.witnessFamilySorted_bonferroni_count
#print axioms AILM.witnessFamilySorted_containment
#print axioms AILM.witnessFamilySorted_holm_rejects_strictly_more
#print axioms AILM.witnessFamilySorted_bars_dominate_the_fixed_bar
#print axioms AILM.sortednessIsLoadBearing
#print axioms AILM.holm_rank_one_hypothesis_is_load_bearing
#print axioms AILM.holm_rank_positivity_is_load_bearing
#print axioms AILM.inversionPositivityIsLoadBearing

/-! ## AILM.Instances -/
#print axioms AILM.lexicalSweep_interiorOptimum
#print axioms AILM.lexicalSweep_peak_is_recorded_best
#print axioms AILM.lexicalSweep_anchor_is_recorded
#print axioms AILM.lexical_peak_index_selects_the_recorded_best_alpha
#print axioms AILM.lexical_interior_flag_agrees_with_the_sweep
#print axioms AILM.lexicalSweep_not_noInteriorGain
#print axioms AILM.lexicalSweep_not_monotone
#print axioms AILM.couplingRecorded_declines_across_recorded_points
#print axioms AILM.couplingRecorded_noInteriorGain
#print axioms AILM.couplingRecorded_not_interiorOptimum
#print axioms AILM.couplingRecorded_entries_are_the_recorded_scalars
#print axioms AILM.couplingRecorded_covers_part_of_the_grid
#print axioms AILM.couplingBestAlpha_is_between_the_recorded_endpoints
#print axioms AILM.channel_anchors_agree_with_stage2
#print axioms AILM.stage1_bracketed
#print axioms AILM.stage2Phoneme_bracketed
#print axioms AILM.stage2G2p_bracketed
#print axioms AILM.stage3Word_bracketed
#print axioms AILM.stage4Derived_bracketed
#print axioms AILM.stage5Clause_above_reference
#print axioms AILM.stage1_model_passes
#print axioms AILM.stage2Phoneme_model_passes
#print axioms AILM.stage2G2p_model_passes
#print axioms AILM.stage3Word_model_passes
#print axioms AILM.stage4Derived_model_passes
#print axioms AILM.stage5Clause_model_fails
#print axioms AILM.stage4Derived_declared_gate_above_reference
#print axioms AILM.stage4Gold_declared_gate_vacuous
#print axioms AILM.stage4Gold_is_degenerate
#print axioms AILM.stage4Gold_no_threshold_is_sound
#print axioms AILM.stage3Word_declared_gate_vacuous
#print axioms AILM.stage1_ceiling_misses_declared_gate
#print axioms AILM.stage1_declared_gate_unreachable_for_every_bounded_model
#print axioms AILM.stage1_shipped_model_is_bounded_by_the_ceiling
#print axioms AILM.stage1_shipped_model_was_unreachable
#print axioms AILM.manufacturedAnchorWins_not_interior
#print axioms AILM.manufacturedAnchorWins_noInteriorGain
#print axioms AILM.lexicalSweepFromPeak_not_interior
#print axioms AILM.lexicalSweepFromPeak_noInteriorGain
#print axioms AILM.stage1_bar_at_ceiling_is_unreachable
#print axioms AILM.stage1_loose_bar_is_vacuous
#print axioms AILM.stage1_without_ceiling_is_unaudited
#print axioms AILM.stage1_population_reaches_every_verdict
#print axioms AILM.stage1_direction_is_load_bearing
#print axioms AILM.stage1_comparator_kind_is_load_bearing

/-! ## AILM.PairedDifference -/
#print axioms AILM.pairedDelta_discordant_form
#print axioms AILM.pairedDelta_pos
#print axioms AILM.pairedDelta_neg
#print axioms AILM.pairedDelta_zero
#print axioms AILM.witness_pos
#print axioms AILM.witness_neg
#print axioms AILM.witness_discordant_form
#print axioms AILM.pairedDelta_pos_drops_imbalance
#print axioms AILM.pairedDelta_pos_drops_imbalance_companion
#print axioms AILM.pairedDelta_pos_drops_nonempty
#print axioms AILM.pairedDelta_pos_drops_nonempty_companion
#print axioms AILM.pairedDelta_neg_drops_imbalance
#print axioms AILM.pairedDelta_neg_drops_imbalance_companion
#print axioms AILM.pairedDelta_neg_drops_nonempty
#print axioms AILM.pairedDelta_neg_drops_nonempty_companion

/-! ## AILM.SweepShape -/
#print axioms AILM.instDecidablePeakAt
#print axioms AILM.instDecidableInteriorOptimum
#print axioms AILM.instDecidableNoInteriorGain
#print axioms AILM.instDecidableMonotoneDecline
#print axioms AILM.monotone_decline_no_interior_gain
#print axioms AILM.interior_excludes_no_gain
#print axioms AILM.interior_excludes_monotone
#print axioms AILM.interiorWitness_interior
#print axioms AILM.interiorWitness_not_no_gain
#print axioms AILM.interiorWitness_not_monotone
#print axioms AILM.monotoneWitness_monotone
#print axioms AILM.monotoneWitness_no_gain
#print axioms AILM.monotoneWitness_not_interior
#print axioms AILM.noGainButNotMonotone_no_gain
#print axioms AILM.noGainButNotMonotone_not_monotone
