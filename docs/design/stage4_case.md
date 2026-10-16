# Stage 4 — The Case-Role Level

Why the case level observes almost nothing, why its declared gate fails in two
opposite directions at once, and why one of its two populations cannot carry a
gate at all.

This page describes the standalone stage experiments. The [connected runtime](connected_runtime.md)
reuses the case agent but supplies optional external UPOS before a word or a
train-fitted uncertain UPOS encoding after raw spelling. That conditional
observation contract does not replace these stage populations or labels.

Decision records: [`../decisions.md`](../decisions.md) entries 15, 16, and 84.
Gate procedure: [`../gate_design.md`](../gate_design.md).

- [The model](#the-model)
- [The circularity trap](#the-circularity-trap)
- [Two populations, never merged](#two-populations-never-merged)
- [One threshold, both defects](#one-threshold-two-population-dependent-defects)
- [The shipped gold arm is an empirical null](#the-shipped-gold-arm-is-an-empirical-null)
- [A measurement that was green by construction](#a-measurement-that-was-green-by-construction)

---

## The model

| slot | content |
|---|---|
| hidden factor | the token's syntactic case role (UD-derived) — 8 states |
| observation | the token's UPOS tag — claim:stage4_upos_alphabet_size on the gated population |
| `A` | **learned** `P(UPOS | case)` |
| `B` | **learned** case-role transition along the sentence |
| `D` | case distribution at a sentence's first labeled token |

Structurally this is L2B one level up: two different alphabets, observations of
one and inference over the other. As at every level below, the learned sequence
dynamics live in the transition.

The state space is 8 case roles as a **single** factor, not "8 case roles ×
n_words". The latter hits exactly the wall [`stage3_word.md`](stage3_word.md)
records, and word identity is already Level 3's factor — putting it here again
would duplicate the lexicon in a second level rather than compose with it.

## The circularity trap

**The observation is the UPOS tag and nothing else.** `deprel` and `feats[Case]`
are withheld from the model, and the restriction is recorded in the vendored
artifact's own `provenance.observation` field.

The reason is that the evaluation labels are *derived from* those fields by
`ailm.cases.UD_DEPREL_TO_CASE`. A model observing `deprel` would be reading its
own answer key: agreement would approach perfection and would measure nothing.

**Why this trap is worse than every mis-wiring below it.** Every earlier defect in
this project made a model *worse* — anti-informative, collapsed, upside down —
and a suspicious number is something a reviewer can pull on. This one makes the
model *excellent*. Nothing looks broken, the gate passes comfortably, and the
result is pure circularity. There is no numeric signature to notice, so the only
possible guard is structural: the observation alphabet is checked for what it is
**not allowed to contain**
(`test_the_observation_never_carries_the_label_source`).

This generalizes beyond this project. Wherever evaluation labels are derived from
features rather than annotated independently, the feature set and the label
derivation must be disjoint, and the disjointness needs a test — because the
failure mode is indistinguishable from success.

## Two populations, never merged

| population | labels from | tokens | cases |
|---|---|---|---|
| **gold** | UD `feats[Case]` — real morphology | claim:stage4_gold_test_tokens held out | claim:stage4_gold_cases_present of 8 |
| **derived** | `deprel` + token-level adposition heuristic — a *research claim* | claim:stage4_derived_test_tokens held out | claim:stage4_derived_cases_present of 8 |

claim:stage4_cases_without_morphology of the eight roles — Dat, Ins, Loc, Abl,
Voc — have **no morphological ground truth in English EWT at all**. No amount of
modeling effort changes that; a different corpus with richer case morphology
would, at the cost of changing the project's language scope.

A score against the derived labels is therefore **agreement**, never accuracy.
The scoring function is named `case_agreement` so that writing the wrong word
takes deliberate effort. The available validation is partial: where both
labelings exist (claim:stage4_heuristic_gold_tokens tokens), the syntactic
heuristic agrees with real morphology
claim:stage4_heuristic_gold_agreement_pct% of the time. That overlap contains
claim:stage4_heuristic_gold_deprel_tokens dependency-route tokens and
claim:stage4_heuristic_gold_adposition_token_tokens token-level adposition-route
tokens. It is not a fidelity bound for the full derived population.

The path composition and head structure are recorded as sufficient counts over
the complete parsed development shard. The oracle labels
claim:stage4_oracle_full_derived_tokens tokens:
claim:stage4_oracle_full_path_morphology_tokens through morphology,
claim:stage4_oracle_full_path_deprel_tokens through `deprel`, and
claim:stage4_oracle_full_path_adposition_token_tokens by the ADP token's own
lemma. That last route does not consult the head. Of those token-level
adposition assignments, claim:stage4_adposition_token_head_also_labeled_tokens
have a head the oracle separately labels, and the two roles disagree for
claim:stage4_adposition_token_head_role_disagreements pairs. The audit preserves
the historical labels; transferring a role to the nominal head would change the
research population and requires a new run.

Nothing merges the two. A merged number would be part measurement and part
heuristic with no way to tell which part is which.

## One threshold, two population-dependent defects

The declared gate of claim:stage4_declared_gate% against the UD 8-case oracle
fails in **both** directions at
once, and which one you observe depends only on the population:

| population | majority | context-free bar | **model** | smoothing | Viterbi | verdict |
|---|---|---|---|---|---|---|
| gold | claim:stage4_gold_majority% | claim:stage4_gold_baseline% | **claim:stage4_gold_agreement%** | claim:stage4_gold_smoothing% | claim:stage4_gold_viterbi% | **VACUOUS** |
| derived | claim:stage4_derived_majority% | claim:stage4_derived_baseline% | **claim:stage4_derived_agreement%** | claim:stage4_derived_smoothing% | claim:stage4_derived_viterbi% | **ABOVE_REFERENCE** |

On gold a constant majority predictor already clears that threshold. On derived the
threshold sits above the model, above both strictly-more-powerful decodes of it,
and above a context-free predictor *fitted on the test labels themselves*
(claim:stage4_derived_cf_oracle%). Those comparators are exceedable references,
not bounds over the model class, so the result does not license an impossibility
claim. Decision 75 supersedes decision 16's older `UNREACHABLE` label.

This is the clearest evidence in the project for a claim the earlier stages only
gesture at: **a threshold without a named population is not a threshold.**

The restated gate: beat the context-free `argmax P(case | UPOS)` bar on the
derived population, defended by a paired McNemar test —
claim:stage4_derived_agreement% against claim:stage4_derived_baseline%, a margin
of claim:stage4_derived_margin_pts points over
claim:stage4_derived_test_tokens positions,
claim:stage4_derived_mcnemar_b versus claim:stage4_derived_mcnemar_c discordant,
p = claim:stage4_derived_mcnemar_p.

Note also that forward-backward smoothing does **not** bound filtering here
(claim:stage4_derived_smoothing% against claim:stage4_derived_agreement%). A
strictly more powerful inference procedure is not guaranteed to give better
per-position argmax accuracy under a mis-specified model. Both are recorded as
variants; neither is reported as a ceiling it does not bound.

## The shipped gold arm is an empirical null

On the gold population the model and the context-free baseline make **identical**
predictions on every token: claim:stage4_gold_mcnemar_b and
claim:stage4_gold_mcnemar_c discordant pairs.

Every gold-morphology token in English EWT is a `PRON`, so the gold observation
alphabet has size claim:stage4_gold_upos_alphabet_size. A constant observation
carries no state information through the emission, but the learned transition
still evolves the filtering prior. A synthetic alternating one-tag population
therefore makes the sequence decoder and context-free baseline disagree. The
committed split's zero discordance is an empirical result of its fitted
transition, start prior, and held-out labels, not the only outcome a one-tag
population permits. `test_a_single_tag_does_not_force_decoder_baseline_agreement`
is the negative control for that scope; the derived-population control separately
confirms the paired comparison detects discordance when present.

No gate is stated on the gold population, and that is a measured conclusion.

## A measurement that was green by construction

Computing the heuristic-versus-gold fidelity required one correction worth
recording, because the natural implementation is wrong in a way that looks right.

`_apply_case_oracle` consults `feats[Case]` **first**, falling back to the
syntactic rules. Calling it on each token and comparing against the gold
morphology therefore returns the gold label itself on exactly the tokens being
compared. The measurement reads as **perfect** — a perfect-looking validation that
confirms nothing at all.

The syntactic rules have to be applied directly, bypassing the morphological
shortcut, which is the comparison the number was always supposed to make. The
honest value is claim:stage4_heuristic_gold_agreement_pct%, with the coverage
boundary above recorded beside it rather than inferred from that percentage.

`test_the_heuristic_fidelity_is_measured_and_bounded` asserts the value is not
perfect, with a message naming this failure mode — because a perfect agreement
metric is far more likely to be measuring
itself than to be perfect.

## The decision-42 subject-selection re-run: measured and set aside

The registered re-run of decision 42 — Fillmore's subject hierarchy scored
per token over alphabet-complete sentences — has now been executed, and its
one-table outcome is recorded as a measurement that does not resolve.

| Path | Accuracy | Coverage |
| --- | --- | --- |
| Model (L4 forward filter, UPOS only) | claim:fillmore_subject_model_accuracy | claim:fillmore_subject_model_coverage |
| Leftmost-role positional baseline | claim:fillmore_subject_baseline_accuracy | claim:fillmore_subject_baseline_coverage |
| Derived-assignment control (circularity) | claim:fillmore_subject_control_accuracy | claim:fillmore_subject_control_coverage |

The arm is set aside rather than reported as a finding because of decision
42's own entry bar. The model path ventures a prediction on only
claim:fillmore_subject_predicted_sentences of
claim:fillmore_subject_scorable_sentences scorable sentences: the per-token
rule moves the *scorable* population from nine to hundreds, but the decode
recursion can produce an in-alphabet role at a candidate position only when
the whole prefix up to it decodes inside the alphabet, so the *predictable*
population stays below the 420-pair equivalence minimum and the accuracy
over nine predictions is the same no-sample number decision 42 refused to
treat as a finding. Enters main line =
claim:fillmore_subject_enters_main_line; the count is recorded either way,
exactly as the registration required.

The falsifier logic did fire where it could. The derived control scores
claim:fillmore_subject_control_accuracy — near-perfect circularity by
construction, as registered — while the leftmost-role baseline scores
claim:fillmore_subject_baseline_accuracy at full coverage. A resolvable
version of this arm needs the train-split fit described in
`docs/research_queue.md`, which is a different experiment because it vendors
a new corpus split.
