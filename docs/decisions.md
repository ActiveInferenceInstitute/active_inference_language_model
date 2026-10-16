# Decisions

The single record of the decisions behind AILM's numbers.

Each entry states the decision, the alternative it displaced, the measurement
that settled it, and the guard that now prevents a regression. Measurements are
cited as `claim:<label>` and resolved from `data/claims.json`; a label with no
ledger entry fails `tests/test_claims_discipline.py`, so nothing here can be a
number someone remembered.

This file is where development history lives. The code keeps at most one
sentence of rationale per decision plus a pointer here, which is enough to stop
an innocent revert without turning a module docstring into a changelog.
Forward-looking guidance on how to *specify* a gate is a separate document:
[`gate_design.md`](gate_design.md).

| # | Decision |
|---|---|
| [1](#1--learned-sequence-dynamics-go-in-b-the-emission-is-a) | Learned sequence dynamics go in B; the emission is A |
| [2](#2--the-comparator-is-the-context-free-baseline-not-uniform) | The comparator is the context-free baseline, not uniform |
| [3](#3--gate-thresholds-are-derived-from-a-measured-baseline-and-ceiling) | Gate thresholds are derived from a measured baseline and ceiling |
| [4](#4--eval-subsampling-is-an-evenly-spaced-stride-never-a-prefix) | Eval subsampling is an evenly spaced stride, never a prefix |
| [5](#5--a-thin-margin-is-defended-by-a-paired-test) | A thin margin is defended by a paired test |
| [6](#6--part-a-perplexity-is-measured-over-the-full-dictionary-in-closed-form) | Part A perplexity is measured over the full dictionary, in closed form |
| [7](#7--level-1-does-not-reset-the-prior-per-word-l2a-and-l2b-do) | Level 1 does not reset the prior per word; L2A and L2B do |
| [8](#8--filtering-not-smoothing) | Filtering, not smoothing |
| [9](#9--inter-level-soft-observation-coupling-is-a-recorded-negative-result) | Inter-level soft-observation coupling is a recorded negative result |
| [10](#10--recorders-write-claims-tests-verify-them) | Recorders write claims; tests verify them |
| [11](#11--hyperparameters-are-configurable-with-consumption-tests) | Hyperparameters are configurable, with consumption tests |
| [12](#12--the-paired-test-is-mcnemars-exact-test-not-the-chi-square-approximation) | The paired test is McNemar's exact test, not the chi-square approximation |
| [13](#13--level-3s-state-space-is-word-identity--position-not-trie-nodes) | Level 3's state space is word identity × position, not trie nodes |
| [14](#14--stage-3s-gate-is-the-letter-bigram-bar-not-perplexity--500) | Stage 3's gate is the letter-bigram bar, not "perplexity < 500" |
| [15](#15--the-case-level-may-not-observe-the-fields-its-labels-are-derived-from) | The case level may not observe the fields its labels are derived from |
| [16](#16--stage-4s-declared-gate-is-vacuous-and-unreachable-at-once) | Historical Stage 4 classification; derived `UNREACHABLE` was superseded by decision 75's `ABOVE_REFERENCE` |
| [17](#17--generation-is-policy-selection-and-its-gate-is-not-membership) | Generation is policy selection, and its gate is not membership |
| [18](#18--prospective-development-session-specification-when-an-inter-level-channel-helps--a-principle-and-its-in-session-test) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: when an inter-level channel helps — a principle, and its in-session test |
| [19](#19--the-criterion-survived-its-test-l3--l2b-helps-non-monotonically) | The criterion survived its test: L3 → L2B helps, non-monotonically |
| [20](#20--the-spec-builds-the-agent-behind-a-default-deny-registry-and-a-bit-identity-gate) | The spec builds the agent, behind a default-deny registry and a bit-identity gate |
| [21](#21--complexity-claims-are-operation-and-byte-counts-never-timings) | Complexity claims are operation and byte counts, never timings |
| [22](#22--pre-registration-pymdp-parameter-learning-is-conjugate-counting-under-observed-states) | PRE-REGISTRATION: pymdp parameter learning is conjugate counting under observed states |
| [23](#23--confirmed-the-agents-have-now-learned-and-learning-is-counting) | CONFIRMED: the agents have now learned, and learning is counting |
| [24](#24--stage-5-the-clause-level-is-a-recorded-null-with-a-detectability-control) | Stage 5: the clause level is a recorded null, with a detectability control |
| [25](#25--pre-registration-the-narrative-level-and-a-receiver-sensitivity-reference) | PRE-REGISTRATION: the narrative level, and a receiver-sensitivity reference |
| [26](#26--stage-6-positive-point-estimate-working-intervention-and-non-confirmatory-primary-analysis) | Stage 6: positive point estimate, working intervention, and non-confirmatory primary analysis |
| [27](#27--pre-registration-the-clause--word-channel-is-predicted-to-harm-falsification-of-the-first-clause) | PRE-REGISTRATION: the clause → word channel is predicted to harm (falsification of the first clause) |
| [28](#28--pre-registration-online-learning-on-non-stationary-text-beats-a-frozen-control-that-must-lose) | PRE-REGISTRATION: online learning on non-stationary text beats a frozen control that must lose |
| [29](#29--pre-registration-g2p-emission-learning-with-unobserved-states) | PRE-REGISTRATION: G2P emission learning with unobserved states |
| [30](#30--the-external-benchmark-is-a-reference-frame-and-its-protocol-must-not-tilt-it) | The external benchmark is a reference frame, and its protocol must not tilt it |
| [31](#31--the-interpretability-claim-is-demonstrated-on-a-named-column-not-asserted) | The interpretability claim is demonstrated on a named column, not asserted |
| [32](#32--the-captured-mass-floor-located-the-declared-floor-is-not-conservative) | The captured-mass floor, located: the declared floor is not conservative |
| [33](#33--post-hoc-power-for-three-secondary-arms-is-derived-not-remeasured) | Post-hoc power for three secondary arms is derived, not remeasured |
| [34](#34--gnn-conformance-binds-all-eight-shipped-specs-default-deny) | GNN conformance binds all eight shipped specs, default-deny |
| [35](#35--clearing-and-rewriting-a-ledger-namespace-is-one-atomic-write) | Clearing and rewriting a ledger namespace is one atomic write |
| [36](#36--pre-registration-what-the-protocol-v4-census-is-expected-to-show) | PRE-REGISTRATION: what the protocol v4 census is expected to show |
| [37](#37--two-coverage-levels-are-one-interval-at-two-defaults-harmonise-to-one) | Two coverage levels are one interval at two defaults; harmonise to one |
| [38](#38--pre-registration-what-stays-in-the-main-line-and-what-is-archived) | PRE-REGISTRATION: what stays in the main line, and what is archived |
| [39](#39--a-retired-probe-leaves-the-registry-not-a-named-exclusion) | A retired probe leaves the registry, not a named exclusion |
| [40](#40--resumability-belongs-at-the-model-call-not-at-the-model) | Resumability belongs at the model call, not at the model |
| [41](#41--conformance-checks-the-configuration-not-only-the-matrices) | Conformance checks the configuration, not only the matrices |
| [42](#42--pre-registration-fillmores-subject-hierarchy-as-an-external-check-on-l4) | PRE-REGISTRATION: Fillmore's subject hierarchy as an external check on L4 |
| [43](#43--entry-38-applied-and-the-criterion-corrected-in-the-applying) | Entry 38 applied, and the criterion corrected in the applying |
| [44](#44--the-red-team-findings-and-what-they-cost) | The red-team findings, and what they cost |
| [45](#45--two-multiplicity-families-corrected-at-two-levels-on-purpose) | Two multiplicity families, corrected at two levels, on purpose |
| [46](#46--prospective-development-session-specification-the-channel-factorial-and-what-it-cannot-de-confound) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the channel factorial, and what it cannot de-confound |
| [47](#47--the-insertion-site-is-an-artifact-of-the-combination-rule) | The insertion site is an artifact of the combination rule |
| [48](#48--the-upstream-interoperability-check-must-not-live-in-a-temporary-directory) | The upstream interoperability check must not live in a temporary directory |
| [49](#49--three-hyperparameters-were-chosen-now-they-are-fitted) | Three hyperparameters were chosen; now they are fitted |
| [50](#50--the-combination-rule-belongs-in-the-spec-and-only-there) | The combination rule belongs in the spec, and only there |
| [51](#51--prospective-development-session-specification-the-in-context-arm-and-the-answer-key-it-must-not-be-given) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the in-context arm, and the answer key it must not be given |
| [52](#52--the-stratificational-identity-claim-narrowed-to-what-the-papers-own-text-supports) | The stratificational identity claim, narrowed to what the paper's own text supports |
| [53](#53--the-scaling-ladder-measures-a-moving-population-and-the-ladder-says-so) | The scaling ladder measures a moving population, and the ladder says so |
| [54](#54--budgeted-attention-killed-by-its-own-comparator-and-by-what-peeking-costs) | Budgeted attention: killed by its own comparator, and by what "peeking" costs |
| [55](#55--the-coverage-effect-is-not-length-and-the-first-account-of-it-was-a-causal-claim-with-no-control) | The coverage effect is not length, and the first account of it was a causal claim with no control |
| [56](#56--what-a-twenty-eight-agent-adversarial-audit-found-and-the-three-guards-that-could-not-fail) | What a twenty-eight-agent adversarial audit found, and the three guards that could not fail |
| [57](#57--the-in-context-result-replicates-on-a-second-probe-and-the-family-grew-to-say-so) | The in-context result replicates on a second probe, and the family grew to say so |
| [58](#58--pymdp-contains-sparse-contraction-machinery-the-blocker-is-at-the-agent-api) | pymdp contains sparse contraction machinery; the blocker is at the Agent API |
| [59](#59--the-letter-at-evidence-channel-is-scored-on-the-receivers-proper-loss-and-both-axes-agree) | the letter-at-evidence channel is scored on the receiver's proper loss, and both axes agree |
| [60](#60--parameters-are-counted-three-ways-and-every-level-is-declared-as-a-gnn-file) | parameters are counted three ways, and every level is declared as a GNN file |
| [61](#61--five-typed-numerals-became-claims-and-the-templates-own-builder-publishes-the-figure-registry) | five typed numerals became claims, and the template's own builder publishes the figure registry |
| [62](#62--three-oversized-modules-became-packages-along-seams-that-already-existed) | three oversized modules became packages, along seams that already existed |
| [63](#63--stage-4s-derived-margin-is-a-mixture-and-one-oracle-path-carries-all-of-it) | Stage 4's derived margin is a mixture, and one oracle path carries all of it |
| [64](#64--two-subsample-nulls-become-bindable-by-recording-counts-that-were-being-discarded) | two subsample nulls become bindable by recording counts that were being discarded |
| [65](#65--the-original-token-level-oracle-intervention-mean-becomes-bindable-on-its-own-scale) | the original token-level oracle-intervention mean becomes bindable on its own scale |
| [66](#66--the-two-coverage-levels-are-harmonized-and-the-label-stops-carrying-the-level) | the two coverage levels are harmonized, and the label stops carrying the level |
| [67](#67--level-4s-margin-can-only-live-where-it-does-and-the-reason-is-an-alphabet-of-one) | Level 4's margin can only live where it does, and the reason is an alphabet of one |
| [68](#68--scorer-audit-l4-gains-bounded-live-pymdp-equivalence-l6-later-gains-a-distinct-live-score) | Scorer audit: L4 gains bounded live-pymdp equivalence; L6 later gains a distinct live score |
| [69](#69--the-comparator-was-defensible-rather-than-strongest-and-the-figure-order-let-it-ship) | the comparator was defensible rather than strongest, and the figure order let it ship |
| [70](#70--the-suites-cost-is-one-test-and-six-of-its-eight-skips-were-the-environment) | the suite's cost is one test, and six of its eight skips were the environment |
| [71](#71--holm-joins-bonferroni-and-moves-one-member-across-the-line) | Holm joins Bonferroni, and moves one member across the line |
| [72](#72--a-caption-is-not-an-image-alternative) | A caption is not an image alternative |
| [73](#73--prospective-development-session-specification-beating-the-unigram-bar-is-not-yet-using-the-previous-letter) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: beating the unigram bar is not yet using the previous letter |
| [74](#74--prospective-development-session-specification-a-family-disjoint-g2p-split-priced-against-the-shipped-interleave) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: a family-disjoint G2P split, priced against the shipped interleave |
| [75](#75--the-gate-taxonomy-carries-its-comparators-type-and-impossibility-now-requires-a-bound) | the gate taxonomy carries its comparator's type, and impossibility now requires a bound |
| [76](#76--l4-human-validation-is-declined-the-case-marking-treebank-arm-supersedes-it) | L4 human validation is declined; the case-marking treebank arm supersedes it |
| [77](#77--prospective-development-session-specification-level-4-scored-against-real-case-morphology-on-czech) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: Level 4 scored against real case morphology, on Czech |
| [78](#78--prospective-development-session-specification-the-third-degeneracy--action-independent-transitions-cancel-a-nonzero-epistemic-term) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the third degeneracy — action-independent transitions cancel a nonzero epistemic term |
| [77 (outcome)](#77-outcome--the-margin-survives-on-real-case-morphology) | the margin survives on real case morphology |
| [79](#79--pre-registration-the-confirmatory-replication-protocol-with-an-honest-census-of-what-can-still-be-held-back) | PRE-REGISTRATION: the confirmatory replication protocol, with an honest census of what can still be held back |
| [80](#80--every-ledger-claim-carries-a-machine-readable-epistemic-status) | every ledger claim carries a machine-readable epistemic status |
| [81](#81--stochastic-kernel-composition-is-not-a-discocat-functor) | stochastic-kernel composition is not a DisCoCat functor |
| [82](#82--exact-source-identities-and-the-cerebrum-crosswalk-boundary) | exact source identities and the CEREBRUM crosswalk boundary |
| [83](#83--stage-5-is-non-passing-the-privileged-label-gain-remains-unresolved) | Stage 5 is non-passing; the privileged-label gain remains unresolved |
| [84](#84--stage-4s-morphological-overlap-does-not-bound-its-token-level-adposition-construct) | Stage 4's morphological overlap does not bound its token-level adposition construct |
| [85](#85--english-ewt-is-split-and-resampled-by-authenticated-source-document) | English EWT is split and resampled by authenticated source document |
| [86](#86--stage-6-retains-and-authenticates-the-document-identity-it-resamples) | Stage 6 retains and authenticates the document identity it resamples |
| [87](#87--stage-3-uses-an-authenticated-category-stratified-brown-document-partition) | Stage 3 uses an authenticated category-stratified Brown document partition |
| [88](#88--the-stage-3-partition-correction-makes-the-clause-to-word-result-directionally-helpful) | The Stage 3 partition correction makes the clause-to-word result directionally helpful |
| [89](#89--the-typed-word-prior-arm-is-a-privileged-oracle-intervention-not-l5-output) | the typed word-prior arm is a privileged-oracle intervention, not L5 output |
| [90](#90--the-synthetic-v2-contract-bounds-exact-artifact-overlap-claims-under-independent-trajectories) | the synthetic v2 contract bounds exact-artifact-overlap claims under independent trajectories |
| [91](#91--the-confirmatory-replication-is-retired-by-exhaustion-the-external_lexicon-population-has-no-held-back-partition) | the confirmatory replication is retired by exhaustion: the external_lexicon population has no held-back partition |
| [92](#92--l5-and-l6-move-from-array-evidence-to-measured-pymdp-inference-evidence-p1-2026-08-31) | L5 and L6 move from array evidence to measured pymdp inference evidence (P1, 2026-08-31) |
| [93](#93--the-calibration-gate-is-the-direct-dual-rule-divergence-check-adopted-2026-08-31) | the calibration gate is the direct dual-rule divergence check, adopted (2026-08-31) |
| [94](#94--l4-and-l5-live-pymdp-equivalence-are-posterior-tolerance-gates-on-the-full-held-out-set-2026-09-01) | L4 and L5 live-pymdp equivalence are posterior-tolerance gates on the full held-out set (2026-09-01) |
| [95](#95--the-generator-samples-the-closed-form-policy-posterior-its-agreement-with-infer_policies-is-recorded-2026-09-01) | the generator samples the closed-form policy posterior; its agreement with infer_policies is recorded (2026-09-01) |
| [96](#96--manuscript-claim-strength-corrections-the-calibration-gate-the-word-state-generator-online-learning-the-prior-only-channel-and-the-abstract-2026-09-01) | Manuscript claim-strength corrections: the calibration gate, the word-state generator, online learning, the prior-only channel, and the abstract (2026-09-01) |
| [97](#97--the-manuscript-states-the-replications-retirement-the-ud_ewt_test-members-await-a-principal-decision-2026-09-01) | The manuscript states the replication's retirement; the ud_ewt_test members await a principal decision (2026-09-01) |
| [98](#98--statistics-prose-bound-to-the-ledger-benchmark-power-cluster-directions-stage-5-coverage-and-partition-p-values-2026-09-01) | Statistics prose bound to the ledger: benchmark power, cluster directions, Stage 5 coverage, and partition p-values (2026-09-01) |
| [99](#99--registration-provenance-inventories-protocols-18-25-and-27-confirmatory-means-registration-commit-before-outcome-commit-2026-09-01) | Registration provenance inventories protocols 18, 25 and 27; confirmatory means registration commit before outcome commit (2026-09-01) |
| [100](#100--figures-draw-only-what-they-claim-bounded-gutter-contained-verdicts-ledger-sourced-declared-gate-named-withheld-probes-drawn-intervals-typed-figure-package-2026-09-02) | Figures draw only what they claim: bounded gutter, contained verdicts, ledger-sourced declared gate, named withheld probes, drawn intervals, typed figure package (2026-09-02) |
| [101](#101--every-stratified-benchmark-bar-carries-a-wilson-interval-on-its-own-denominator-2026-09-02) | Every stratified benchmark bar carries a Wilson interval on its own denominator (2026-09-02) |
| [102](#102--prospective-development-session-specification-the-family-grouped-g2p-sensitivity-arm-its-heterogeneity-prediction-is-refuted-2026-09-02) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the family-grouped G2P sensitivity arm; its heterogeneity prediction is refuted (2026-09-02) |
| [103](#103--part-a-perplexity-is-also-measured-on-a-held-out-word-split-in-closed-form-2026-09-02) | Part A perplexity is also measured on a held-out word split, in closed form (2026-09-02) |
| [104](#104--the-wild-cluster-bootstrap-is-a-recorded-sensitivity-arm-beside-every-clustered-interval-2026-09-02) | The wild cluster bootstrap is a recorded sensitivity arm beside every clustered interval (2026-09-02) |
| [105](#105--the-orphan-gate-measures-its-own-dark-share-three-real-consumption-surfaces-and-a-per-family-threshold-2026-09-02) | The orphan gate measures its own dark share: three real consumption surfaces and a per-family threshold (2026-09-02) |
| [106](#106--conformances-live-inputs-are-a-typed-record-not-a-positional-signature-2026-09-02) | Conformance's live inputs are a typed record, not a positional signature (2026-09-02) |
| [107](#107--generate_manuscript_variables-is-a-pure-orchestrator-over-per-section-producers-held-to-a-zero-diff-oracle-at-every-step-2026-09-02) | generate_manuscript_variables is a pure orchestrator over per-section producers, held to a zero-diff oracle at every step (2026-09-02) |
| [108](#108--prospective-development-session-specification-the-stipulated-phone-chain-beside-the-letter-chain-the-fitted-comparator-beats-every-registered-probe-2026-09-02) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the stipulated phone chain beside the letter chain; the fitted comparator beats every registered probe (2026-09-02) |
| [109](#109--the-holm-step-down-comparison-is-machine-checked-beside-the-bonferroni-arithmetic-2026-09-05) | The Holm step-down comparison is machine-checked beside the Bonferroni arithmetic (2026-09-05) |
| [110](#110--the-lexical-channels-backoff-matched-the-wrong-context-corrected-to-the-documented-longest-suffix-rule-every-dependent-arm-re-recorded-2026-09-05) | The lexical channel's backoff matched the wrong context; corrected to the documented longest-suffix rule, every dependent arm re-recorded (2026-09-05) |
| [111](#111--pre-registration-the-ud_ewt_test-confirmatory-replication-of-the-four-ewt-family-members-2026-09-05) | PRE-REGISTRATION: the ud_ewt_test confirmatory replication of the four EWT family members (2026-09-05) |
| [112](#112--prospective-development-session-specification-the-widened-observation-clause-level-case-roles-plus-surface-ordered-verb-and-aux-on-the-document-disjoint-partition-2026-09-05) | PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the widened-observation clause level (case roles plus surface-ordered VERB and AUX) on the document-disjoint partition (2026-09-05) |
| [113](#113--rule-a-reply-field-that-describes-the-servers-own-cache-is-not-part-of-the-answer-2026-09-06) | RULE: a reply field that describes the server's own cache is not part of the answer (2026-09-06) |
| [114](#114--the-widened-observation-clause-model-is-the-canonical-level-5-the-case-only-model-is-the-retained-ablation-2026-09-06) | The widened-observation clause model is the canonical Level 5; the case-only model is the retained ablation (2026-09-06) |
| [115](#115--pre-registration-the-document-clustered-interval-of-the-level-6-improvement-on-the-held-back-ud_ewt_test-population-2026-09-06) | PRE-REGISTRATION: the document-clustered interval of the Level 6 improvement on the held-back ud_ewt_test population (2026-09-06) |
| [116](#116--rule-a-verdict-word-on-a-figure-is-read-from-a-recorded-boolean-the-confirmatory-replication-gets-its-own-figure-2026-09-07) | RULE: a verdict word on a figure is read from a recorded boolean; the confirmatory replication gets its own figure (2026-09-07) |
| [117](#117--rule-a-family-exclusion-is-justified-by-a-checked-invariant-not-by-prose-the-factorials-diagonal-cells-are-one-hypothesis-under-a-second-selection-grid-2026-09-07) | RULE: a family exclusion is justified by a checked invariant, not by prose; the factorial's diagonal cells are one hypothesis under a second selection grid (2026-09-07) |
| [118](#118--the-parameter-census-follows-canonical-level-5-and-distinguishes-simplex-capacity-from-fitted-parameters-2026-09-07) | The parameter census follows canonical Level 5 and distinguishes simplex capacity from fitted parameters (2026-09-07) |
| [119](#119--make-the-in-context-comparison-visible-and-gate-reference-rule-crossings-2026-09-07) | Make the in-context comparison visible and gate reference-rule crossings (2026-09-07) |
| [120](#120--correct-the-stipulated-phone-context-interpretation-without-changing-its-measurement-2026-09-07) | Correct the stipulated phone context interpretation without changing its measurement (2026-09-07) |
| [121](#121--pre-registration-the-case-word-influence-sweep--the-l4l3-bridges-causal-influence-estimand-controls-and-refusals-2026-09-14) | PRE-REGISTRATION: the case-word influence sweep — the L4→L3 bridge's causal-influence estimand, controls, and refusals (2026-09-14) |
| [122](#122--measured-the-p1-warm-up-compute-gate--the-census-budget-does-not-clear-at-the-measured-rate-without-the-declared-efficiency-options-2026-09-15) | MEASURED: the P1 warm-up compute gate — the census budget does not clear at the measured rate without the declared efficiency options (2026-09-15) |
| [123](#123--rule-every-audited-gate-declares-a-smallest-effect-size-of-interest-and-the-audit-reads-the-margin-against-it-2026-09-18) | RULE: every audited gate declares a smallest effect size of interest, and the audit reads the margin against it (2026-09-18) |
| [124](#124--decision-re-measure-the-benchmark-arms-under-the-updated-local-gemma34b-identity-rather-than-restoring-the-old-binary-2026-09-18) | DECISION: re-measure the benchmark arms under the updated local gemma3:4b identity rather than restoring the old binary (2026-09-18) |
| [125](#125--decision-re-verify-the-warm-ollama-core-after-the-transport-re-bind-rather-than-carrying-the-09-18-run-ids-2026-09-19) | DECISION: re-verify the warm Ollama core after the transport re-bind rather than carrying the 09-18 run ids (2026-09-19) |
| [126](#126--decision-archive-the-moved-gemma34b-metadata-identity-and-re-measure-its-stratum-rather-than-repinning-the-derived-hash-2026-09-20) | DECISION: archive the moved gemma3:4b metadata identity and re-measure its stratum rather than repinning the derived hash (2026-09-20) |
| [127](#127--decision-per-scored-call-cache-namespaces-protocol-v3-and-the-hosted-comparator-switched-to-a-non-reasoning-model-2026-09-21) | DECISION: per-scored-call cache namespaces, protocol v3, and the hosted comparator switched to a non-reasoning model (2026-09-21) |
| [128](#128--finding-the-stipulated-phone-arms-one-symbol-direction-is-not-run-stable-on-the-hosted-transport-the-interpretive-sentence-is-retired-2026-09-22) | FINDING: the stipulated phone arm one-symbol direction is not run-stable on the hosted transport; the interpretive sentence is retired (2026-09-22) |
| [129](#129--decision-the-local-model-identity-binds-to-the-weights-blob-and-every-local-stratum-is-archived-and-re-measured-2026-09-28) | DECISION: the local model identity binds to the weights blob, and every local stratum is archived and re-measured (2026-09-28) |
| [130](#130--finding-which-phone-arm-context-contrasts-clear-is-not-run-stable-the-supplement-reports-the-run-outcome-2026-09-29) | FINDING: which phone-arm context contrasts clear is not run-stable; the supplement reports the run outcome (2026-09-29) |
| [131](#131--finding-an-os-lapack-update-moved-an-analytic-protocol-fields-last-bits-float-registry-fields-bind-within-a-relative-tolerance-2026-09-30) | FINDING: an OS LAPACK update moved an analytic protocol field's last bits; float registry fields bind within a relative tolerance (2026-09-30) |
| [132](#132--implemented-a-conditional-connected-runtime-and-a-locally-registered-gum-comparison-2026-10-05) | IMPLEMENTED: a conditional connected runtime and a locally registered GUM comparison (2026-10-05) |
| [133](#133--correction-hosted-run-names-did-not-identify-local-model-weights-2026-10-05) | CORRECTION: hosted run names did not identify local model weights (2026-10-05) |
| [134](#134--correction-hosted-recorders-have-no-local-model-preflight-or-warm-up-2026-10-05) | CORRECTION: hosted recorders have no local-model preflight or warm-up (2026-10-05) |
| [135](#135--prospective-correction-explicit-local-loader-and-context-capacity-2026-10-05) | PROSPECTIVE CORRECTION: explicit local loader and context capacity (2026-10-05) |
| [136](#136--prospective-correction-project-owned-local-benchmark-endpoint-2026-10-05) | PROSPECTIVE CORRECTION: project-owned local benchmark endpoint (2026-10-05) |
| [137](#137--prospective-correction-compare-cached-logprob-candidates-within-their-emitted-positions-2026-10-06) | PROSPECTIVE CORRECTION: compare cached logprob candidates within their emitted positions (2026-10-06) |
| [138](#138--operational-continuation-preserve-interrupted-scopes-and-verify-the-current-upstream-checkout-2026-10-07) | OPERATIONAL CONTINUATION: preserve interrupted scopes and verify the current upstream checkout (2026-10-07) |
| [139](#139--operational-selection-amendment-current-gnn-interoperability-after-release-recovery-2026-10-09) | OPERATIONAL SELECTION AMENDMENT: current GNN interoperability after release recovery (2026-10-09) |
| [140](#140--publication-metadata-alignment-institute-release-destination-2026-10-14) | PUBLICATION METADATA ALIGNMENT: Institute release destination (2026-10-14) |

---

## 1 — Learned sequence dynamics go in B; the emission is A

**Decision.** In every level, the corpus-learned sequence statistics parameterize
the transition `B`, and `A` is the emission. At L1 and L2A the emission is the
identity (the observed symbol *is* the hidden symbol); at L2B the emission is a
genuine learned letter-given-phone table over a different alphabet from the
hidden one.

**Alternative rejected.** Installing the learned bigram matrix as `A` with an
identity `B`. It is an attractive mistake because a bigram table is
column-stochastic and therefore satisfies pymdp's `_validate()` in either slot,
so nothing structural objects.

**Why it is wrong, not merely different.** Under that wiring, observing symbol
`c` yields a posterior over the symbol that *preceded* `c`, and an identity `B`
carries that backward-looking belief forward as the next-symbol prediction. The
model is anti-informative rather than weak.

**Measurement.** At the letter level, the anti-informative wiring scores
claim:stage1_anti_informative_perplexity against
claim:stage1_letter_perplexity for the shipped formulation, with uniform at
claim:stage1_uniform_baseline. On a small in-sample corpus the same mis-wiring
scored *worse than uniform* while still passing the threshold that was in force
at the time.

The G2P model makes the trap available in a second flavour, and three
mis-wirings were evaluated against the context-free bar
claim:stage2_g2p_baseline_accuracy%:

- a time-reversed `B` — "observing a letter tells you which phone came *before*
  it" — falls below the bar and is caught;
- a uniform, information-free `A` collapses accuracy and is caught;
- an `A` normalized over phones instead of letters, then renormalized so pymdp
  still validates it, **clears the bar while being wrong**.

So "beats the context-free baseline" is a necessary condition, not a sufficient
one. What actually pins the emission down is the column-sum invariant on `A`
(`A[0, :, p]` sums to 1 *before* any renormalization) plus numerical agreement
with a closed-form oracle.

**Guards.**

- `ailm.gnn` checks each live agent against its GNN spec, including that the
  matrix annotated `TransitionMatrix` really is the learned one:
  `tests/test_gnn_conformance.py::TestSpecFilesParse::test_directly_observed_levels_declare_an_identity_emission`,
  `tests/test_gnn_conformance.py::TestLiveAgentsConform::test_g2p_live_emission_is_obs_by_states`,
  `tests/test_gnn_conformance.py::TestPositiveControls::test_transposed_g2p_emission_fails_the_g2p_spec`.
- `tests/test_stage1_gate.py::test_stage1_bigram_statistics_live_in_the_transition`
  and `tests/test_stage2_gate.py::test_stage2_phoneme_statistics_live_in_the_transition`.
- Negative controls that must fail:
  `tests/test_stage2_gate.py::test_stage2_g2p_negative_control_uniform_emission`,
  `tests/test_stage2_gate.py::test_stage2_g2p_negative_control_anti_informative_transition`,
  `tests/test_stage2_gate.py::test_stage2_g2p_emission_orientation_is_pinned`,
  `tests/test_stage1_gate.py::test_stage1_unigram_gate_rejects_the_anti_informative_wiring`,
  `tests/test_coupling.py::test_time_reversed_transition_is_caught`.
- Shape assertions in `build_g2p_agent` and `build_phoneme_agent`, and the
  orientation checks in `ailm.levels.coupling.build_g2p_matrices` — deliberately
  unreachable while the builders keep their documented shapes, and kept as
  tripwires on exactly that assumption.

---

## 2 — The comparator is the context-free baseline, not uniform

**Decision.** Every level is scored against the best model that does *not* do
the thing being claimed. For a context-aware decoder that is a context-free
predictor: per-letter argmax at L2B, the unigram distribution at L1 and L2A.

**Alternative rejected.** Uniform. A uniform-random phone guess and a
majority-phone constant predictor both sit far below anything the corpus
supports, so beating them is not evidence of context sensitivity — it is a
crippled comparator, and a gate placed against one is vacuous by construction.

**Measurement.** At L2B the honest bar reaches
claim:stage2_g2p_baseline_accuracy% and forward filtering reaches
claim:stage2_g2p_accuracy%. At L1 uniform is claim:stage1_uniform_baseline
while the context-free unigram model already reaches
claim:stage1_unigram_baseline. At L2A uniform is
claim:stage2_phoneme_uniform_baseline and the phone unigram model reaches
claim:stage2_phoneme_unigram_baseline, against
claim:stage2_phoneme_perplexity for the bigram model — the whole gain available
over uniform is under one nat.

**Guards.** `per_letter_argmax_baseline` and `full_corpus_unigram_perplexity`
are the comparators the gates use, and the context-free bar is measured on the
*identical* position set as the model
(`tests/test_phoneme_full_corpus.py::TestRealArtifact::test_model_beats_the_context_free_bar_on_identical_positions`,
`tests/test_stage1_gate.py::test_stage1_beats_unigram_letter_frequency_baseline`,
`tests/test_stage2_gate.py::test_stage2_phoneme_beats_context_free_unigram`,
`tests/test_stage2_gate.py::test_stage2_g2p_beats_context_free_baseline`).

---

## 3 — Gate thresholds are derived from a measured baseline and ceiling

**Decision.** A threshold is computed from two measurements — what a trivial
no-context model already scores, and what the best model of this order could
possibly score — and never borrowed from a convention or another level.

**Alternative rejected.** Round numbers carried over from elsewhere. AILM
shipped one defect of each kind, in opposite directions:

- **UNREACHABLE.** The declared Stage 1 target of
  claim:stage1_declared_gate_nats nats below uniform demands more than the
  maximum order-1 gain available, claim:stage1_max_bigram_gain_nats nats. No
  implementation of a letter bigram model could pass it. The exact all-position
  arithmetic at L2A gives
  claim:stage2_phoneme_all_position_max_gain_nats nats of headroom against the
  same declared target; even the more permissive transitions-only relaxation is
  only claim:stage2_phoneme_max_gain_nats nats.
- **VACUOUS.** The originally declared G2P threshold sat *below* the
  context-free bar claim:stage2_g2p_baseline_accuracy%, so the lookup table it
  was supposed to beat already passed it. (It was also stated over a treebank
  with no phonemic field at all.)

**Measurement.** The order-1 information floor is
claim:stage1_bigram_floor_perplexity at L1 and
claim:stage2_phoneme_all_position_floor_perplexity at L2A; the historical
transitions-only diagnostic is claim:stage2_phoneme_floor_perplexity. The shipped models reach
claim:stage1_letter_perplexity and claim:stage2_phoneme_perplexity, capturing
claim:stage1_bigram_signal_captured_pct% of the available order-1 signal at L1.
At L2B the achievable ceilings are claim:stage2_g2p_smoothing_ceiling% for
forward–backward smoothing and claim:stage2_g2p_viterbi_ceiling% for Viterbi.

**Guards.** `scripts/audit_gates.py` makes the classification re-runnable;
`ailm.figures.gates._classify` mirrors its verdict function so the figure and
the audit cannot disagree, and `gate_taxonomy_data` re-derives each verdict from
the ledger and raises if it stops matching what the manuscript reports.
`tests/test_stage2_gate.py::test_stage2_two_nat_target_is_provably_unreachable`
and `tests/test_stage2_gate.py::test_stage2_gate_constants_are_derived_not_typed` pin the arithmetic;
`tests/test_stage1_gate.py::test_stage1_gate_constants_are_consistent` does the
same at L1. L2A's gate now compares its all-position perplexity to an exact
all-position MLE bound on the same in-sample dictionary protocol. Its separate
transitions-only conditional-entropy diagnostic remains a different quantity;
a gate comparing the all-position model against that diagnostic can never pass.

---

## 4 — Eval subsampling is an evenly spaced stride, never a prefix

**Decision.** `ailm.levels.phoneme._iter_pairs` is the single subsample selector
for every decoder, and it returns evenly spaced indices across the whole list.

**Alternative rejected.** A deterministic prefix. It is reproducible, which is
easy to mistake for representative. The vendored `g2p_pairs` list is sorted
alphabetically, so a prefix contains only words beginning a, b or c.

**Measurement.** The full held-out set gives a margin of
claim:stage2_g2p_margin_pts points over the context-free bar across
claim:stage2_g2p_eval_positions positions. An alphabetical prefix of a fraction
of that size reports a margin several times larger — a comfortable-looking pass
— while a stride of the same size lands near the truth and spans most of the
alphabet's initials. Those rows live in `SUBSAMPLE_COMPARISON`
in `src/ailm/figures/gates.py` and are re-runnable through
`scripts/audit_gates.py --limit`. That debt is closed: `scripts/record_subsample_claims.py`
(run_id `subsample_bias`) re-recorded every row from a measured run —
claim:subsample_full_margin_pts, claim:subsample_prefix_margin_pts,
claim:subsample_stride5_margin_pts and claim:subsample_stride3_margin_pts points
(full, prefix, stride·5, stride·3) — each row now carries `ledger_backed: True`,
and `subsample_bias_data` verifies every row against the ledger at draw time
(`test_every_subsample_row_is_verified_against_the_ledger`; see CHANGELOG.md
“Fixed — the last `ledger_backed: False` rows”). Nothing in this figure is
papered over; agreement with the ledger is enforced, not stated.

A caution the stride does not remove: any subsample of this eval is
**underpowered**. The full set reaches claim:stage2_g2p_mcnemar_p, while a
stride covering claim:stage2_g2p_subsample_positions positions reaches only
claim:stage2_g2p_subsample_mcnemar_p. `limit` is for smoke checks; gate
statistics are computed on the full set after proving the fast path agrees with
the reference decoder exactly.

**Guards.** Every decoder resolves `limit` through the same `_iter_pairs`, which
is what keeps the position vectors aligned for a paired test:
`tests/test_coupling.py::test_limit_resolves_as_a_stride_not_a_prefix`,
`tests/test_phoneme_full_corpus.py::TestRealArtifact::test_full_corpus_is_harder_than_the_length_matched_subset`.
`fig:subsample_bias` states the argument in the manuscript, and its
significance level is pinned to the gate's own level by
`tests/test_figure_layout.py` rather than set to the looser convention — at the
looser level a representative stride reads as significant, which is precisely
the conclusion the panel exists to deny.

---

## 5 — A thin margin is defended by a paired test

**Decision.** Where a margin is a fraction of a point, the two decoders are run
position-for-position over the same evaluation positions in the same order and
compared with an exact paired McNemar test on the discordant pairs.

**Alternative rejected.** Comparing two aggregate accuracies. At this effect
size the difference of two aggregates is indistinguishable from noise, so an
unpaired comparison cannot establish the claim in either direction.

**Measurement.** claim:stage2_g2p_accuracy% against
claim:stage2_g2p_baseline_accuracy% is a margin of
claim:stage2_g2p_margin_pts points. Pairing over
claim:stage2_g2p_eval_positions positions yields
claim:stage2_g2p_mcnemar_b positions where only filtering is right against
claim:stage2_g2p_mcnemar_c where only the baseline is, giving
p = claim:stage2_g2p_mcnemar_p. The pairing carries the claim; the aggregate
margin alone would not.

**Guards.** `decode_positions` / `baseline_positions` / `coupled_positions` /
`uncoupled_positions` all return vectors aligned by construction, and
`ailm.levels.coupling.mcnemar` raises rather than silently accepting unpaired
vectors (`tests/test_coupling.py::test_mcnemar_rejects_unpaired_vectors`). The
test is proved live in both directions: a self-comparison finds
claim:coupling_null_mcnemar_discordant discordant pairs
(`tests/test_coupling.py::test_mcnemar_self_comparison_finds_no_discordant_pairs`) and a planted
difference is detected (`tests/test_coupling.py::test_mcnemar_detects_a_planted_difference`,
`tests/test_stage2_gate.py::test_stage2_mcnemar_can_return_no_effect`). A sweep
whose position vectors differ in length raises instead of producing a margin.

---

## 6 — Part A perplexity is measured over the full dictionary, in closed form

**Decision.** L2A's headline perplexity is computed from the vendored sufficient
statistics — word-initial phone counts plus phone bigram counts — over every
primary pronunciation in the dictionary.

**Alternative rejected.** Walking the artifact's `g2p_pairs` sequences through
pymdp. That list is the length-matched G2P subset, biased toward short words
with transparent orthography, and it *flatters* the model. Walking the whole
dictionary through pymdp instead is exact but costs roughly one `infer_states`
call per phone position, far too slow for a gate.

**Measurement.** The full dictionary gives claim:stage2_phoneme_perplexity over
claim:stage2_phoneme_eval_positions positions; the length-matched subset gives
claim:stage2_phoneme_subset_perplexity against its own context-free bar
claim:stage2_phoneme_subset_unigram_baseline. Reporting the subset number as
"the" phoneme perplexity would reproduce the subsample-bias error of decision 4
in a second place. Only claim:stage2_g2p_length_matched_pct% of dictionary
entries are length-matched at all.

**Why the closed form is exact.** With the prior reset at every word boundary,
position 0 of each word is scored under the start prior `D` and every later
position under a transition, so `phone_initials` and `phone_bigrams` are exact
sufficient statistics — no approximation and no sampling.

**Guards.** `full_corpus_perplexity` raises when `phone_initials` is absent
rather than falling back to the biased subset
(`tests/test_phoneme_full_corpus.py::TestMissingInitialsGuard::test_missing_phone_initials_raises`), the
closed form is checked against a sequence-by-sequence walk
(`tests/test_phoneme_full_corpus.py::TestClosedFormEqualsSequenceWalk::test_closed_form_matches_walk`,
`tests/test_phoneme_full_corpus.py::TestAnalyticAnchors::test_positions_counted_are_initials_plus_bigrams`), and the position count is
asserted to cover the whole dictionary (`tests/test_phoneme_full_corpus.py::TestRealArtifact::test_positions_cover_the_whole_dictionary`).

---

## 7 — Level 1 does not reset the prior per word; L2A and L2B do

**Decision.** L1 scores continuous prose as one stream with no per-word reset.
L2A and L2B reset the empirical prior to `D` at the start of every word.

**Rationale.** The unit of the L2 models is the dictionary entry: a word
boundary there is not a phoneme transition, and the first phone of a word has no
predecessor to be predicted from. L1's corpus is running text, where the
transition across a space is a real letter bigram — space is one of the thirty
ALPHABET symbols.

**Alternative rejected.** Treating the dictionary as one continuous phone
stream, so each word's first phone is predicted from the previous word's last
phone. Measured, that is *worse* than claim:stage2_phoneme_perplexity, which
pays the unigram price at position 0 — the reset is correct rather than merely
convenient.

**Guards.** `compute_phoneme_perplexity` resets to `D` per sequence and
accumulates over all positions of all sequences, so its result is a corpus-level
per-phone perplexity rather than a mean of per-word perplexities. The closed
form in decision 6 is only valid under that reset — its `phone_initials` term
*is* the reset — so
`tests/test_phoneme_full_corpus.py::TestClosedFormEqualsSequenceWalk::test_closed_form_matches_walk` and
`tests/test_phoneme_full_corpus.py::TestClosedFormEqualsSequenceWalk::test_single_phone_words_are_scored_under_the_start_prior` fail if either side
changes convention.

---

## 8 — Filtering, not smoothing

**Decision.** The shipped decoders are strictly causal forward filters: position
`t` sees observations `1..t`. The level's reported accuracy is the filtering
accuracy.

**Alternative rejected.** Reporting a forward–backward or Viterbi number as the
level's result. Those are the achievable *ceilings* at this model order, and
they are recorded as such — but both need the whole word before they can emit
position 0, which is a different architecture from the streaming one AILM
claims.

**Measurement.** Filtering reaches claim:stage2_g2p_accuracy% against a
smoothing ceiling of claim:stage2_g2p_smoothing_ceiling% and a Viterbi ceiling
of claim:stage2_g2p_viterbi_ceiling%.

**Guards.** `decode_phonemes` and `ailm.verification.oracle.forward_filter`
never look ahead, and the two are checked against each other
(`tests/test_stage2_gate.py::test_stage2_g2p_pymdp_matches_oracle`,
`tests/test_stage2_gate.py::test_stage2_phoneme_pymdp_matches_oracle`, recorded as
claim:stage2_decoder_disagreements). The ceilings enter the gate taxonomy as
ceilings, never as the model's score.

---

## 9 — Inter-level soft-observation coupling is a recorded negative result

**Decision.** `ailm.levels.coupling` implements a downward message from L1 into
L2B's likelihood — the L1 predictive over letters, mixed with the observed glyph
by weight `alpha` and marginalized through the G2P emission. It is kept,
gated, and **not** enabled: `alpha` defaults to 0, which is provably the
uncoupled decode. The recommendation that follows is to correct the
architecture figure's arrow label, not to ship the coupling.

**Alternative rejected.** Two of them. Shipping the coupling, which costs
accuracy; and deleting the module once the result came out negative, which would
leave the figure's hierarchical claim untested and the null unfalsifiable.

**Measurement.** Over claim:coupling_eval_positions held-out positions with a
grid of claim:coupling_grid_size alphas, the best positive alpha is
claim:coupling_best_alpha, reaching claim:coupling_best_accuracy% against
claim:coupling_alpha0_accuracy% for the uncoupled anchor — a margin of
claim:coupling_margin_pts points, i.e. a loss, significant in the direction of
harm at paired-McNemar p = claim:coupling_mcnemar_p
(claim:coupling_mcnemar_b versus claim:coupling_mcnemar_c discordant
positions). Accuracy falls monotonically as `alpha` rises. The recorded support
flag is claim:coupling_supported.

**Why it is a measurement and not a wiring bug.** The channel carries real
information: at `alpha = 1`, where the glyph is discarded entirely, the real
message scores claim:coupling_alpha1_accuracy% against
claim:coupling_alpha1_scrambled_accuracy% for the reversed-axis control. The
message simply is not *new* information — it assigns mean probability
claim:coupling_message_mean_prob to the letter actually observed and its argmax
picks that letter claim:coupling_message_argmax_pct% of the time, so spending
observation weight on it substitutes weak evidence for a certainty. The reason
is structural: at position `t` the G2P agent already observes letter `t`, and
the L1 predictive is a function of letters `0..t-1` only. A coupling form that
adds an *independent* channel is not what the figure claims and is not measured
here.

**Guards.** A null needs the same controls as a positive, or "we measured no
effect" is indistinguishable from "we measured nothing":

- **The `alpha = 0` identity.** The numpy soft recursion must agree
  position-for-position with the independent oracle driven by hard observations,
  with claim:coupling_alpha0_identity_diffs differences, and with the pymdp
  agent at claim:coupling_pymdp_disagreements disagreements
  (`tests/test_coupling.py::test_alpha_zero_reproduces_uncoupled_decode_exactly`,
  `tests/test_coupling.py::test_alpha_zero_matches_the_pymdp_agent`). That anchor is what licenses
  computing the sweep in numpy at all.
- **The anchor is not vacuous.** `tests/test_coupling.py::test_anchor_is_not_vacuous_alpha_above_zero_does_differ`.
- **A scrambled-message control.** Reversing the letter axis is a fixed
  involutive permutation, so the message keeps its entropy and loses only its
  meaning — a strictly stronger control than replacing it with uniform, which
  would confound content with entropy
  (`tests/test_coupling.py::test_real_message_beats_scrambled_at_full_weight`,
  `tests/test_coupling.py::test_scrambled_message_does_LESS_damage_at_intermediate_alphas`).
- **A grid guard.** `validate_alpha_grid` raises `GateVacuousError` unless the
  grid contains `0.0` plus at least three distinct positive values, so "no alpha
  beat the baseline" cannot come from a sweep that collapsed to a single point.
- **Liveness under a broken emission.** `tests/test_coupling.py::test_coupled_decoder_is_live_under_a_broken_emission`.

---

## 10 — Recorders write claims; tests verify them

**Decision.** Every quantitative statement in prose, docstrings, figures and the
manuscript is a `claim:<label>` resolved from `data/claims.json`. Exactly one
`scripts/record_*_claims.py` writes each label, from the run that produced the
value. Tests re-derive and compare; tests never record.

**Alternative rejected.** Hand-typed numerals in prose. They cannot be
re-derived, they drift silently, and a stale one is indistinguishable from a
fresh one to a reader.

**Evidence that the discipline needs enforcing against the real tree.** The
scanner existed from Stage 0 and was exercised only against `tmp_path`
fixtures. Every test passed while claim:phoneme_count_39 sat unbacked in
README.md, AGENTS.md and `docs/provenance/sources.md`. A gate that only runs on
synthetic inputs constrains nothing about the shipped artifact.

**Guards.** `tests/test_claims_discipline.py` points the scanner at
README/AGENTS/CHANGELOG/ISA/TODO, all of `docs/`, and all of `src/`, with a
positive control that plants an unbacked tag and requires it to be flagged, a
negative control on a backed tag, and a non-vacuity check that the scanned set
is non-empty. `tests/test_claims_single_writer.py` pins each canonical label to
its recorder. `ailm.manuscript_variables.REQUIRED_CLAIMS` fails the render with
`MissingClaimError` rather than emitting an unresolved `{{TOKEN}}` or a
plausible default — an unresolved token in a rendered PDF is a fabricated
result. `tests/test_manuscript_tokens.py` checks the token layer against the
ledger. Recorded claims are re-derived from fresh measurements by
`tests/test_stage1_gate.py::test_stage1_claim_verified`,
`tests/test_stage2_gate.py::test_stage2_claims_recorded` and
`tests/test_coupling.py::test_recorded_claims_match_a_fresh_measurement`.

The rule when a claim fails: record the measurement again from the run that
produced it. Never delete the tag from prose to make the gate green.

---

## 11 — Hyperparameters are configurable, with consumption tests

**Decision.** `ailm.config.LevelConfig` holds the tunable quantities —
`smoothing` (the add-k pseudocount), `emission_smoothing` (a separate dial for
the G2P emission, which is a different shape and density from the transition
tables), and `test_every` (the held-out stride). Every builder takes a
keyword-only `config`, and `DEFAULT_LEVEL_CONFIG` reproduces the original add-1
behavior byte for byte.

**Alternative rejected.** Hyperparameters written into the source — the add-1
pseudocount as a literal, and the held-out stride readable only from the
vendored artifact. Nothing could be varied without editing code, so no
sensitivity statement could be made and no reviewer could probe robustness.

**Measurement.** The defaults reproduce claim:stage1_letter_perplexity and
claim:stage2_phoneme_perplexity, and the shipped split is
claim:stage2_g2p_train_pairs train pairs against
claim:stage2_g2p_test_pairs test pairs.

**Guards.** A configuration knob is only real if changing it changes an output,
so `tests/test_config.py` flips each dial and asserts the consequence rather
than the naming: `test_every_artifact_changes_when_smoothing_changes`,
`test_emission_smoothing_moves_the_emission`,
`test_emission_smoothing_leaves_the_transition_and_start_untouched`,
`test_test_every_flows_into_the_trained_matrices`,
`test_test_every_sets_the_split_sizes_exactly`, plus monotonicity sweeps
(`test_letter_kl_from_uniform_decreases_monotonically_with_smoothing`,
`test_larger_emission_smoothing_flattens_the_emission_monotonically`,
`test_held_out_fraction_decreases_monotonically_with_test_every`) and endpoint
checks (`test_huge_smoothing_drives_the_letter_columns_to_uniform`,
`test_flattening_the_emission_collapses_g2p_accuracy`). Digest tests pin the
default build against pre-change references and are themselves checked for
discrimination (`test_reference_digests_are_discriminating`). Validation is
fail-closed on non-finite values *before* the sign test, because a bare
`x <= 0` guard is False for NaN. `g2p_split` validates the artifact's
`test_every` unconditionally, so an override cannot turn a malformed provenance
field into a silent pass
(`test_a_bad_artifact_stride_still_raises_even_with_an_override`).

---

## 12 — The paired test is McNemar's exact test, not the chi-square approximation

**Decision.** The discordant pairs are scored with the two-sided *exact binomial*
probability of `b` successes in `n = b + c` trials at `p = 0.5`, computed by
`scipy.stats.binomtest(b, b + c, 0.5)`. In closed form, with the doubling exact
because the null is symmetric about `n/2`:

```
p = min(1, 2 * sum_{k=0}^{min(b,c)} C(n, k) * 2^-n)
```

**Alternative rejected.** The continuity-corrected chi-square approximation
`X^2 = (|b - c| - 1)^2 / (b + c)`. It is an approximation to the exact test, and
its continuity correction exists only to patch that approximation. On the
subsamples this project also reports, `b + c` is small enough that the
approximation is least trustworthy precisely where it would be relied on. The
exact test costs nothing here and has no free parameter.

**Measurement.** On the shipped counts (`claim:stage2_g2p_mcnemar_b` against
`claim:stage2_g2p_mcnemar_c`) the closed form above and `binomtest` agree to the
last bit, as they do on the degenerate cases `b = c` and `c = 0`. The two
statistics also land on the same verdict here — the defect was never a wrong
number.

**Guards.** `tests/test_stage2_gate.py::_mcnemar` and
`ailm.levels.coupling.mcnemar` are the only two implementations and both call
`binomtest`; `grep -rn 'chi2\|chisquare' src/ scripts/ tests/` returns nothing.
`docs/formalism_map.md` maps `eq:mcnemar` to those call sites and is the file to
re-derive if the statistic ever changes.

**History.** For one revision the manuscript displayed the chi-square form as
`eq:mcnemar` while every implementation ran the exact binomial, so a reader would
have taken the reported *p* to be the tail probability of a statistic nothing
computed. The equation now states what runs. This is the same "documented, never
applied" failure the claims ledger had one layer up (entry 10), and it is why
`docs/formalism_map.md` exists: an equation in a paper is a claim, and it needs a
named function behind it like any other.

---

## 13 — Level 3's state space is word identity × position, not trie nodes

**Decision.** The word level factorizes the hidden state into two pymdp factors —
**word identity** (`N_VOCAB` states) and **position within the word**
(`MAX_WORD_LEN + 1` states) — with a single letter modality depending on both
through `A_dependencies = [[0, 1]]`. The vocabulary is bounded to the
claim:stage3_vocab_size most frequent alphabetic Brown word types, so every
Stage 3 number describes that sublanguage and says so.

The trie the design document calls for is not discarded; it is *induced*. After
observing a prefix, the posterior over the word factor is supported exactly on
the set of vocabulary words sharing that prefix — which is the trie node. The
two formulations are the same model in different coordinates, and
`tests/test_stage3_gate.py::test_posterior_support_is_the_trie_node` checks that
correspondence rather than assuming it.

**Alternative rejected — the trie *is* the state space (dense `B`).**
`pymdp.agent.Agent` takes dense `jnp` arrays and no sparse path is reachable
through its public API. Decision 58 corrects what that does *not* mean: the
sparse contraction machinery exists in pymdp's JAX backend, and the blocker is
the `Agent` API rather than the method. Measured
over the length-matched G2P vocabulary the trie has claim:stage3_trie_nodes_g2p
nodes and claim:stage3_trie_edges_g2p edges, so a dense transition is
claim:stage3_trie_dense_gb GB at float32 and
claim:stage3_trie_density_pct% occupied. Extrapolated to full CMUdict it is
roughly claim:stage3_trie_dense_full_tb TB. This is not a performance problem to
optimize; it is a representation that does not fit.

**Alternative rejected — bound the vocabulary *and* keep trie nodes as states.**
This does fit, and it was measured rather than dismissed:

| vocabulary | trie nodes | dense `B` | word-factor `B` | held-out token coverage |
|---|---|---|---|---|
| 1,000 | claim:stage3_trie_nodes_v1000 | claim:stage3_trie_dense_v1000_mb MB | claim:stage3_factored_v1000_mb MB | claim:stage3_scale_1000_coverage_pct% |
| 2,000 | claim:stage3_trie_nodes_v2000 | claim:stage3_trie_dense_v2000_mb MB | claim:stage3_factored_v2000_mb MB | claim:stage3_scale_2000_coverage_pct% |
| 4,000 | claim:stage3_trie_nodes_v4000 | claim:stage3_trie_dense_v4000_mb MB | claim:stage3_factored_v4000_mb MB | claim:stage3_scale_4000_coverage_pct% |
| 8,000 | claim:stage3_trie_nodes_v8000 | claim:stage3_trie_dense_v8000_mb MB | claim:stage3_factored_v8000_mb MB | claim:stage3_scale_8000_coverage_pct% |

Every cell above is a ledger reference. It was typed markdown until a rebuild
disagreed with all four node counts (against the shipped claim:stage3_vocab_size word
vocabulary, claim:stage3_trie_nodes_v2000 now regenerates as 5,643; the handoff receipt
records the historically measured 5,693 against that table's printed 5,700) and two of the four
coverage figures did not round from the ledger values recorded beside them —
78.31 was printed 78.4, and 89.52 was printed 89.6. Nothing regenerated the
table, so nothing noticed it drifting away from the measurements it summarizes.
The conclusion below never depended on those digits, which is exactly why the
drift survived so long.

The trie-node encoding costs an order of magnitude more memory **under dense
storage** for the *same* vocabulary and the same coverage, because it spends states on prefixes rather
than on words. Probed live, a trie agent at the claim:stage3_vocab_size word rung reached 1.33 GB resident
where the factored agent at the same vocabulary sits under 0.7 GB. The
factorization also puts word identity in the model explicitly, which is what a
word level ought to have — under the trie encoding the word is only recoverable
by asking which terminal node was reached.

**Alternative rejected — keep the trie outside pymdp as a constraint mask.**
Cheapest of all, and it removes the level's reason to exist: the lexicon would
no longer be part of the generative model, so nothing about the claim "this is a
word-level active inference agent" would be testable.

**Why the factorization is exact rather than an approximation.** With `B_word`
the identity, `B_pos` the +1 shift, and `A` deterministic, the probability the
model assigns to the full emission path of word *w* is exactly `D_word[w]` — the
word's unigram probability. Nothing is lost relative to a perfect closed-vocabulary
word model, and the identity is checkable in closed form rather than argued:
`ailm.verification.oracle.word_path_logprob` derives it without importing pymdp,
and claim:stage3_path_identity_max_abs_err is the largest disagreement over the
whole vocabulary.

**What the bound costs, stated rather than hidden.** claim:stage3_coverage_pct%
of held-out alphabetic Brown tokens are in vocabulary; the model is **undefined**
on the rest, and every Stage 3 number is conditioned on that population. This is
the same population-scoping discipline as the G2P length-matched subset
(decision 6), applied where it is load-bearing rather than incidental.

**Guards.** `tests/test_stage3_gate.py::test_path_probability_equals_the_word_unigram`
(the identity above), `tests/test_stage3_gate.py::test_posterior_support_is_the_trie_node`,
`tests/test_stage3_gate.py::test_b_word_is_exactly_the_identity`,
`tests/test_stage3_gate.py::test_b_position_advances_and_is_not_time_reversed`,
`tests/test_stage3_gate.py::test_emission_is_normalized_over_observations_not_over_words`, and
`tests/test_gnn_conformance.py` against `gnn/models/L3_word.gnn.md`.

---

## 14 — Stage 3's gate is the letter-bigram bar, not "perplexity < 500"

**Decision.** Stage 3 is admitted by beating a **letter-bigram model over the
identical alphabet and same whole-document training partition, scored on the
identical held-out in-vocabulary positions**. The bar is fitted to every
training type, whereas the word prior is fitted to its bounded lexicon, so the
non-lexical comparator has broader training support. The metric is per-emission
perplexity, where an emission is a letter or the end-of-word symbol.

**Alternative rejected — the declared "perplexity < 500 on PTB-1k".** Measured,
the letter-bigram bar is claim:stage3_bigram_baseline, which clears 500 by a
factor of claim:stage3_declared_gate_slack. The declared gate is **VACUOUS** —
the Stage 2 defect exactly, and the third time a borrowed round constant has
turned out to constrain nothing. It also named a corpus this project does not
have: `nltk`'s `treebank` is the 5% sample distributed under research terms, not
the LDC-licensed PTB. Brown is already vendored, checksummed and byte-reproducible,
and is the corpus Level 1 is trained on, so Stage 3 is measured there and the
second corpus is not introduced.

**Measurement.** Over claim:stage3_eval_positions emissions from
claim:stage3_eval_tokens held-out tokens:

| comparator | uses the lexicon? | per-emission perplexity |
|---|---|---|
| Uniform over 27 symbols | no | claim:stage3_uniform_baseline |
| Context-free letter unigram | no | claim:stage3_unigram_baseline |
| **Letter bigram — the bar** | no | **claim:stage3_bigram_baseline** |
| **Word × position model (this level)** | yes | **claim:stage3_word_perplexity** |
| Held-out-oracle word model — the ceiling | yes | claim:stage3_ceiling_perplexity |

The gain over the bar is claim:stage3_gain_nats nats per emission and the
remaining headroom to the ceiling is claim:stage3_headroom_nats nats, so the
level captures claim:stage3_signal_captured_pct% of what a closed-vocabulary word
model can capture. The margin is wide by construction — a lexicon is a great deal
of information — so unlike Stage 2 this gate does **not** need a paired test, and
that is a measured conclusion rather than an omission.

**Why the bar is not enough, and what actually pins the level down.** Level 2
taught that beating the baseline is necessary, not sufficient. At Level 3 that
lesson is sharper, because the lexicon constraint is so strong that nearly any
word prior beats a letter model. Five mis-wirings were built and scored against
the bar:

| mis-wiring | per-emission perplexity | caught by the bar? |
|---|---|---|
| Uniform `D` over words — the lexicon kept, the frequencies discarded | claim:stage3_miswire_uniform_prior | **no** |
| `D` derived from word length instead of frequency | claim:stage3_miswire_length_prior | **no** |
| Frequency ranking reversed — the rarest words made the likeliest | claim:stage3_miswire_reversed_prior | **no** |
| Time-reversed emission — letters emitted back to front | claim:stage3_miswire_time_reversed | yes |
| End-of-word emitted one position early | infinite (assigns 0 to every true path) | yes |

**Three of five clear the bar**, and the third is the striking one: inverting the
frequency ranking so that the rarest words are predicted most confidently still
scores claim:stage3_miswire_reversed_prior against a bar of
claim:stage3_bigram_baseline. A gate whose only guard is `assert model < bar`
admits a model whose word prior is upside down.

What pins the level down is the closed-form identity in decision 13 — the path
probability *must* equal the word's unigram probability, to
claim:stage3_path_identity_max_abs_err — plus the per-axis structural invariants.
All three surviving mis-wirings change the word prior, so all three fail that
identity. This is the level's version of Stage 2's `A`-normalized-over-phones
variant, and it is why the mis-wiring enumeration is part of building a stage
rather than a follow-up.

**Guards.** `scripts/audit_gates.py::audit_stage3` re-runs the classification;
`tests/test_stage3_gate.py` carries one negative control per row of the table
above, each asserted to fail, plus
`tests/test_stage3_gate.py::test_the_bar_does_not_catch_the_prior_miswirings` — a control on the *control
set*, which records in an executable form that three of them survive it.


---

## 15 — The case level may not observe the fields its labels are derived from

**Decision.** Stage 4's observation is the token's **UPOS tag and nothing else**.
The hidden factor is the case role, `A` is `P(UPOS | case)`, and `B` is the
case-role transition along the sentence. `deprel` and `feats[Case]` are withheld
from the model.

**Alternative rejected.** Observing `deprel`, which is the obvious richer
feature and would raise every reported number. It is disqualified because the
evaluation labels are *derived from* `deprel` by
`ailm.cases.UD_DEPREL_TO_CASE`. A model observing it would be reading its own
answer key: agreement would approach 100% and would measure nothing about case
structure at all.

**Why this trap is worse than the ones below it.** Every earlier mis-wiring in
this project made a model *worse* — anti-informative, collapsed, upside down —
and a suspicious number is something a reviewer can pull on. This one makes the
model *excellent*. Nothing looks broken, the gate passes comfortably, and the
result is pure circularity. There is no numeric signature to notice, so the only
possible guard is structural: the observation alphabet must be checked for what
it is not allowed to contain.

**Measurement.** With UPOS alone the level reaches
claim:stage4_derived_agreement% against a context-free bar of
claim:stage4_derived_baseline%. The point of the entry is that the alternative
was not measured and must not be: a number produced by reading the label source
is not a weaker result, it is not a result.

**State space.** 8 case roles as a single factor, not "8 case roles x n_words".
The latter hits exactly the wall decision 13 records, and the word identity is
already Level 3's factor — putting it here again would duplicate the lexicon in
a second level rather than compose with it.

**Guards.** `ailm.levels.case.FORBIDDEN_OBSERVATIONS` names the withheld fields
as a constant, `scripts/generate_case_counts.py` emits UPOS only, and
`tests/test_stage4_gate.py::test_the_observation_never_carries_the_label_source`
asserts the vendored artifact contains no `deprel` or `feats` value. The
artifact's own `provenance.observation` field records the restriction so a
regenerated artifact carries its own terms.

---

## 16 — Stage 4's declared gate is vacuous *and* unreachable at once

**Historical record; verdict superseded by decision 75.** This entry preserves
the reasoning and language used at the time. Its derived-population comparator
was later typed as exceedable, so `UNREACHABLE` is not the current licensed
verdict; `ABOVE_REFERENCE` is.

**Decision.** Stage 4 is admitted by beating the **context-free
`argmax P(case | UPOS)` bar on the derived population**, defended by a paired
McNemar test. The result is reported as **agreement**, never accuracy. No gate is
stated on the gold-morphology population, and the reason is measured rather than
asserted.

**Alternative rejected — the declared ">=60% vs the UD 8-case oracle".** It fails
in **both** directions at once, and which one you get depends only on which
population you score it against:

| population | *n* (held out) | cases | majority | context-free bar | **filtering model** | smoothing | Viterbi | verdict on ">=60%" |
|---|---|---|---|---|---|---|---|---|
| gold morphology | claim:stage4_gold_test_tokens | 3 of 8 | claim:stage4_gold_majority% | claim:stage4_gold_baseline% | **claim:stage4_gold_agreement%** | claim:stage4_gold_smoothing% | claim:stage4_gold_viterbi% | **VACUOUS** |
| derived oracle | claim:stage4_derived_test_tokens | 8 of 8 | claim:stage4_derived_majority% | claim:stage4_derived_baseline% | **claim:stage4_derived_agreement%** | claim:stage4_derived_smoothing% | claim:stage4_derived_viterbi% | **ABOVE_REFERENCE** (retyped by decision 75) |

On gold a constant "NOM" predictor already clears 60%. On derived the threshold
sits above the model, above both strictly-more-powerful decodes of that model,
and above a context-free predictor *fitted on the test labels themselves*
(claim:stage4_derived_cf_oracle%). The original entry treated those values as a
model-order ceiling; decision 75 shows that they are exceedable references and
withdraws the impossibility claim. The population contrast remains, and
it is the clearest evidence that a threshold without a named population is not a
threshold.

**The gold population cannot support a gate at all.** After a held-out split it
holds claim:stage4_gold_test_tokens tokens, and the paired test finds
claim:stage4_gold_mcnemar_b positions where only the model is right against
claim:stage4_gold_mcnemar_c where only the baseline is — **zero discordant pairs
in both directions**. The model and the context-free baseline make identical
predictions on every gold token. Every gold-morphology token in English EWT is a
`PRON`, so the gold observation alphabet has size
claim:stage4_gold_upos_alphabet_size. That removes state information from the
emission, but it does not erase the learned transition: filtering still evolves
the prior through `B`. An alternating one-tag counterexample makes the decoder
and context-free baseline disagree. The committed split's zero discordance is
therefore a measured empirical null of its fitted transition, start prior, and
held-out labels, not the only outcome a one-tag population permits. It is
reported as a result with that scope.

**Historical fidelity interpretation; superseded by decision 84.** Where
both labelings exist, the syntactic heuristic agrees with real morphology on
claim:stage4_heuristic_gold_agreement_pct% of tokens. This validates that
morphologically marked overlap; decision 84 shows that the overlap contains no
token-level adposition-route examples and withdraws the inference that it bounds
every derived number. Dat/Ins/Loc/Abl/Voc —
claim:stage4_cases_without_morphology of the eight — have
no morphological ground truth in English EWT in principle. No amount of
modeling effort changes that; a different corpus with richer case morphology
would, at the cost of changing the project's language scope.

**Measurement.** On the derived population the level reaches
claim:stage4_derived_agreement% against claim:stage4_derived_baseline%, a margin
of claim:stage4_derived_margin_pts points over
claim:stage4_derived_test_tokens positions, with McNemar
claim:stage4_derived_mcnemar_b versus claim:stage4_derived_mcnemar_c discordant
positions at p = claim:stage4_derived_mcnemar_p.

**Guards.** `scripts/audit_gates.py::audit_stage4` re-derives the classification
for both populations; `tests/test_stage4_gate.py` carries the paired test, the
zero-discordance null on gold, a self-comparison control that must report no
effect, and negative controls for the emission and transition mis-wirings. The
function that reports the score is named `case_agreement`, never
`case_accuracy`, so writing the wrong word takes deliberate effort.


---

## 17 — Generation is policy selection, and its gate is not membership

**Decision.** AILM generates by selecting actions that minimize expected free
energy at a **controllable** letter level: one action per symbol, with
`B[next, current, k] = delta(next == k)`, the learned Brown bigram as the policy
prior `E`, and `C` supplied top-down by the lexicon. The gate is **exact
total-variation distance to the true word distribution** against the exact word
distribution induced by ancestral sampling from the bigram, paired with a
sampled diversity floor.

**Scope.** Generation was Out of Scope until the principal moved it in
(2026-07-30). The ISA is the system of record and now lists it under Goal;
Stages 6 and 7, which depend on it, are unblocked.

**Alternative rejected — lexicon membership as the metric.** It is the obvious
choice and it is **green by construction**. `lexicon_preference` floors any
letter that would leave the trie, so at any usable precision the generator
essentially cannot emit a non-word. Measured, membership is
claim:generation_best_membership_pct% at the operating point *and*
claim:generation_degenerate_membership_pct% at the degenerate precision that
emits claim:generation_degenerate_distinct distinct word(s) forever. A metric
that cannot fail is not a gate, however good the number looks — and had
membership been the gate, the degenerate generator would have passed it
perfectly. This is the fourth time in this project that the first metric anyone
reaches for turns out not to constrain anything.

**Alternative rejected — a single trivial action.** Every recognition level has
one action, so `infer_policies` had nothing to select between and `eq:efe` was
specified but exercised by nothing. Making the transition controllable is what
turns policy selection from a metaphor for generation into generation: choosing
an action *is* choosing the next symbol.

**Where the corpus statistics go, and why it moves.** At every recognition level
the learned bigram parameterizes the transition. Under a controllable `B` the
transition is a structural constant — a deterministic consequence of the chosen
action — and the same table appears as the policy prior `E` instead. The table is
identical; its role is not. It is the agent's *habit*: what it would do absent a
goal. A spec annotating it `TransitionMatrix` here would describe a model that
cannot choose what to emit, which is why `L1g_letter_generation` is a separate
model rather than a configuration flag on `L1_letter`.

**Measurement, and the finding.** Distance to the true word distribution is
**U-shaped** in the policy precision `gamma`, so the optimum is *interior*:

| precision | membership | distinct words | TV distance |
|---|---|---|---|
| baseline — sampling from `B` | claim:generation_baseline_membership_pct% | claim:generation_baseline_distinct | claim:generation_baseline_tv |
| claim:generation_low_gamma — goal too weak | claim:generation_low_gamma_membership_pct% | — | claim:generation_low_gamma_tv |
| **claim:generation_best_gamma — shipped** | **claim:generation_best_membership_pct%** | **claim:generation_best_distinct** | **claim:generation_best_tv** |
| claim:generation_degenerate_gamma — collapsed | claim:generation_degenerate_membership_pct% | claim:generation_degenerate_distinct | claim:generation_degenerate_tv |

Every distance in the table uses the same exact estimator. For the policy rows,
it multiplies action-posterior probabilities along every named lexicon path; for
the baseline it multiplies Brown-transition probabilities along those paths.
Each calculation conditions out an immediate empty attempt, matching the
non-empty word tokens retained by `GenerationResult`, and assigns all remaining
path mass to an explicit out-of-lexicon bucket. Membership and distinct-word
columns are sampled diagnostics of realized runs. The former pooled baseline TV
estimate remains recorded only under the explicitly diagnostic
`generation_sampled_baseline_tv` label; it is not the comparator used in the
canonical improvement.

Both endpoints generate badly for **opposite** reasons: as `gamma` goes to zero
the policy posterior becomes the habit and the model *is* the baseline. At high
precision the policy takes the locally most preferred trie branch at each prefix,
with the habit breaking ties. This need not recover the globally most probable
word; a three-word counterexample selects a lower-prior word. On the shipped
Brown artifact the greedy path collapses onto one repeated word. **No recognition
gate in this project fails in two directions.** This one does, which makes it the
only gate here that cannot be passed by turning a dial to an extreme.

**What the distance metric still does not catch.** The degenerate generator
scores claim:generation_degenerate_tv, which *beats* the baseline's
claim:generation_baseline_tv while emitting one word forever. Distance alone
would admit it; only the strict claim:generation_diversity_fraction_pct%-of-baseline
floor (claim:generation_diversity_floor distinct words under the declared sampled
protocol) rejects it. Recorded here because every
gate in this project states what its guards miss —
`tests/test_generation.py::test_degenerate_precision_fails_the_composite_generation_gate`
holds that fact in executable form.

**Guards.**

- **The identity anchor.** With `C = 0` the policy posterior reduces to the
  habit, so the generator must reproduce ancestral sampling
  **character-for-character** under the same seed, not approximately
  (`tests/test_generation.py::test_flat_preference_reproduces_the_baseline_exactly`). The recorder
  refuses to write if it does not. That anchor is what licenses interpreting the
  whole sweep as an effect of `C`.
- **The anchor is not vacuous** — the real preference must change the output
  (`tests/test_generation.py::test_the_anchor_is_not_vacuous`).
- **An entropy-preserving scrambled control.** Reversing the letter axis is a
  fixed involutive permutation, so the preference keeps its entropy and loses
  only its meaning; it reaches
  claim:generation_scrambled_control_membership_pct% membership. Replacing `C`
  with uniform instead would confound content with entropy — the same argument as
  entry 9.
- **Agreement with pymdp.** The closed-form policy posterior drives generation
  because it is fast enough for thousands of steps; it is checked against
  `agent.infer_policies` at several states
  (`tests/test_generation.py::test_closed_form_efe_matches_pymdp_infer_policies`). An equivalence whose
  failure was a warning would be a substitution rather than a proof.
- **A sweep that refuses an unsupportable claim.** `exact_sweep_precision` raises
  unless given enough distinct positive precisions to exhibit an interior point;
  the retained diagnostic `sweep_precision` additionally requires multiple
  seeds. A U-shape cannot be shown from only its endpoints, and a one-seed Monte
  Carlo diagnostic is a cherry-pick.
- `scripts/audit_gates.py::audit_generation` re-runs the classification, and
  `gnn/models/L1g_letter_generation.gnn.md` carries three mutations that must
  fail conformance.

**Why the earlier coupling null does not apply here.** Entry 9 records an
inter-level channel that measured *harmful*. The reason was structural: it
carried strictly less information about the observed symbol than the observation
itself. This channel is not in that position — it carries the lexicon, which the
letter level does not have in any form. The two results are consistent, and the
distinction is the one entry 9 already named.


---

## 18 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: when an inter-level channel helps — a principle, and its in-session test

**Correction (2026-09-05).** The measurement this entry's prediction was tested against was taken through a defective backoff lookup; entry 110 records the defect, the corrected rule, and the re-recorded outcome. The prediction's first half (the channel helps) held under the correction and held more strongly; its second half (an interior optimum as the signature of novelty) is not supported by the corrected sweep, which helps at every tested weight and does not distinguish its interior argmax from a plateau. Nothing below is edited; it is the record of what was predicted.

**Provenance correction (2026-09-01).** This entry, `scripts/record_lexical_channel_claims.py`, `src/ailm/levels/lexical_channel.py`, the recorded sweep and entry 19 first enter Git history in the same commit (`bc070dea6a475f238b68092f0cceafbc3ad8b548`). The prediction below was written before the sweep ran, and the text preserves that order, but no independently timestamped immutable artifact establishes it outside the session. Under the terminology rule of `data/registration_provenance.json`, the channel is therefore *prospectively specified in-session*, not pre-registered; protocol `"18"` is inventoried there as a same-commit specification with `confirmatory_status_eligible=false`, and the manuscript, README and `docs/hierarchy.md` now say so in the same words. No measurement changes.

**The question this entry exists to answer.** AILM is a *hierarchy*, and the
central research question is what makes one. Four levels are built, and two
inter-level channels have now been measured with opposite outcomes:

| channel | direction | outcome |
|---|---|---|
| L1 letters → L2B G2P likelihood (entry 9) | bottom-up | **significantly harmful**, claim:coupling_margin_pts pts, p = claim:coupling_mcnemar_p |
| L3 lexicon → L1 preferences (entry 17) | top-down | **works**, claim:generation_baseline_tv → claim:generation_best_tv |

Two data points are an anecdote unless something explains both. This entry states
the explanation, derives a prediction from it, and then tests that prediction on
a channel neither result covers — with the prediction written down **before** the
measurement, because a principle that is only ever fitted to results already in
hand explains nothing.

**The principle — the novel-information criterion.**

> An inter-level message improves the receiving level **iff it carries
> information the receiver does not already have access to**. Message strength,
> direction, and architectural elegance are all irrelevant to the sign of the
> effect; only novelty is.
>
> *[Status note, 2026-08-04: preserved verbatim as the original pre-registered
> phrasing. Live documents (docs/manuscript/09_hierarchy.md) now state this as an
> evidence-bounded working hypothesis; its measured support is entries 9, 17,
> 19 and 25–26, and universality is not asserted.]*

It accounts for both existing results, and it accounts for them *structurally*
rather than by appeal to the numbers:

- **L1 → L2B failed** because at position `t` the G2P level already observes
  letter `t`, while L1's predictive is a function of letters `0..t-1` only. The
  message is a strictly weaker statement about the very symbol the receiver can
  see. Spending observation weight on it substitutes weak evidence for a
  certainty, which is why accuracy fell *monotonically* in the mixing weight
  rather than peaking somewhere.
- **L3 → L1g succeeded** because the lexicon is information the generation agent does
  not possess in any form. A bigram cannot represent "only these letter sequences
  are words," at any order below the length of the longest word.

The criterion also explains why the second channel needed a *precision* dial and
the first did not: novel information has an optimal weight, redundant information
does not, and entry 17's interior optimum is what that looks like when measured.

**The prospectively specified prediction.** The criterion is only worth having if it
predicts. The channel chosen to test it is **L3 → L2B**: a lexical prefix
statistic into the same receiving level the first channel harmed. Holding the
receiver fixed while changing only the sender is what makes the comparison
controlled.

The message is `P(phone at position t | the word's letter prefix 0..t)`,
estimated from the G2P **train** split and applied to held-out pairs. It is
lexical: it encodes which letter sequences are words and how those words are
pronounced.

> **Prediction, recorded before the measurement: this channel HELPS.** The
> lexicon is information the G2P level lacks entirely — its `A` is a per-letter
> emission and its `B` a phone bigram, and neither can represent whole-word
> orthographic constraints. If the criterion is right, the effect should be
> positive, and it should be non-monotone in the mixing weight (an interior
> optimum), because that is the signature of novel rather than redundant
> information.
>
> **What would falsify the criterion:** a negative or null effect here. That
> outcome would mean novelty is not what distinguishes the two recorded results,
> and the criterion would be withdrawn rather than qualified — the measured
> outcome is recorded below either way.

**Why this is not circular.** The message must not simply hand over the answer.
Conditioning on full word *identity* would do exactly that: the lexicon maps word
to pronunciation, so G2P would become a dictionary lookup and the "improvement"
would be the Stage 4 circularity trap in a new costume (entry 15). The message is
therefore conditioned on the letter *prefix* only, estimated on train pairs and
evaluated on held-out ones, so it is a genuine lexical generalization rather than
a lookup.

**The measured outcome is recorded in the section that follows this one**, added
after the experiment ran. Guards mirror entry 9's, because a positive result
needs the same controls a null does: an `alpha = 0` identity anchor, a
non-vacuity check that `alpha > 0` differs, a scrambled-message control that
preserves entropy while destroying meaning, and a paired McNemar test over the
full held-out position set.


---

## 19 — The criterion survived its test: L3 → L2B helps, non-monotonically

**Correction (2026-09-05).** The table and the second numbered finding below were measured through the defective backoff lookup that entry 110 corrects. The `claim:` tags in this entry resolve to the corrected ledger, so the table now shows the corrected sweep; the prose around it, kept as written, describes the defective measurement's shape. Under the corrected rule the channel helps at every tested weight, the high-weight reversal does not exist, and the interior argmax is an ordering the sweep cannot distinguish from a plateau. The first finding stands; the second is withdrawn.

**Outcome (as recorded on 2026-08-01, before the correction).** Entry 18's prediction is **confirmed**, on both counts it made.

| mixing weight | G2P accuracy | vs anchor | scrambled control | McNemar *b* / *c* | *p* |
|---|---|---|---|---|---|
| `0.0` — the anchor | claim:lexical_anchor_accuracy% | — | identical by construction | claim:lexical_sweep_alpha_0_0_mcnemar_b / claim:lexical_sweep_alpha_0_0_mcnemar_c | claim:lexical_sweep_alpha_0_0_mcnemar_p |
| `0.1` | claim:lexical_sweep_alpha_0_1_accuracy% | +claim:lexical_sweep_alpha_0_1_delta | claim:lexical_sweep_alpha_0_1_scrambled% | claim:lexical_sweep_alpha_0_1_mcnemar_b / claim:lexical_sweep_alpha_0_1_mcnemar_c | claim:lexical_sweep_alpha_0_1_mcnemar_p |
| `0.2` | claim:lexical_sweep_alpha_0_2_accuracy% | +claim:lexical_sweep_alpha_0_2_delta | claim:lexical_sweep_alpha_0_2_scrambled% | claim:lexical_sweep_alpha_0_2_mcnemar_b / claim:lexical_sweep_alpha_0_2_mcnemar_c | claim:lexical_sweep_alpha_0_2_mcnemar_p |
| **claim:lexical_best_alpha — best** | **claim:lexical_best_accuracy%** | **+claim:lexical_best_margin_pts** | claim:lexical_best_scrambled_accuracy% | claim:lexical_best_mcnemar_b / claim:lexical_best_mcnemar_c | claim:lexical_best_mcnemar_p |
| `0.5` | claim:lexical_sweep_alpha_0_5_accuracy% | +claim:lexical_sweep_alpha_0_5_delta | claim:lexical_sweep_alpha_0_5_scrambled% | claim:lexical_sweep_alpha_0_5_mcnemar_b / claim:lexical_sweep_alpha_0_5_mcnemar_c | claim:lexical_sweep_alpha_0_5_mcnemar_p |
| `0.7` | claim:lexical_sweep_alpha_0_7_accuracy% | +claim:lexical_sweep_alpha_0_7_delta | claim:lexical_sweep_alpha_0_7_scrambled% | claim:lexical_sweep_alpha_0_7_mcnemar_b / claim:lexical_sweep_alpha_0_7_mcnemar_c | claim:lexical_sweep_alpha_0_7_mcnemar_p |
| claim:lexical_high_alpha | claim:lexical_high_alpha_accuracy% | claim:lexical_high_alpha_margin_pts | claim:lexical_sweep_alpha_0_9_scrambled% | claim:lexical_sweep_alpha_0_9_mcnemar_b / claim:lexical_sweep_alpha_0_9_mcnemar_c | claim:lexical_sweep_alpha_0_9_mcnemar_p |

Over claim:lexical_eval_positions held-out positions. The channel had a real
lexical prefix to condition on at claim:lexical_positions_with_prefix_pct% of
them, from claim:lexical_prefix_tables estimated tables, so a null could not have
come from a channel that was silently backing off.

**Both predicted signatures appeared.**

1. **It helps.** claim:lexical_best_margin_pts points at
   alpha = claim:lexical_best_alpha, at p = claim:lexical_best_mcnemar_p. For
   scale, the entire Stage 2 G2P margin over its own context-free baseline is
   claim:stage2_g2p_margin_pts points — so the lexical channel is worth several
   times what the phoneme level's own headline result is worth.
2. **The optimum is interior**, and the effect *reverses* at high weight:
   claim:lexical_high_alpha_margin_pts points at
   alpha = claim:lexical_high_alpha. That is the predicted signature of novel
   information — it has an optimal weight — and it is exactly what entry 9's
   channel did **not** do. That one fell monotonically, because redundant
   information has no optimal weight above zero.

**The controlled comparison.** The two channels share a receiving level. Only the
sender differs:

| | entry 9: L1 → L2B | entry 19: L3 → L2B |
|---|---|---|
| message | letter predictive over `0..t-1` | lexical prefix statistic |
| novel to the receiver? | **no** — L2B observes letter `t` directly | **yes** — L2B has no lexicon |
| shape in the mixing weight | monotone decline | **interior optimum** |
| sign | claim:coupling_margin_pts pts (harm) | +claim:lexical_best_margin_pts pts (help) |

Holding the receiver fixed is what makes this evidence about the *criterion*
rather than about the two channels separately. Architecture, direction and
strength are matched; novelty is the variable, and it predicted the sign
correctly in advance.

**The criterion, restated now that it has survived a test.**

> An inter-level message improves the receiving level iff it carries information
> the receiver does not already have. Its optimal weight is strictly between zero
> and one when it is novel, and zero when it is redundant — so the *shape* of the
> weight sweep diagnoses novelty independently of the sign of the effect.

That second sentence is the part worth keeping. It converts a question about
whether to build a channel — expensive, and answerable only after building it —
into a reading of a curve. A monotone decline says the message is redundant and
no amount of tuning will rescue it. An interior optimum says the message is
carrying something the receiver lacks, and the only remaining question is how
loudly to say it.

**Why this is not the circularity trap.** Conditioning on full word *identity*
would make G2P a dictionary lookup — the lexicon maps word to pronunciation — and
the "improvement" would be entry 15's trap in a new costume. The message
conditions on the letter *prefix* only, is estimated on the train split and
evaluated on held-out pairs, so it can be wrong. It is: the best configuration
still misses claim:lexical_best_accuracy% of positions, which is what a
generalization looks like and a lookup does not.

**Guards.** `scripts/record_lexical_channel_claims.py` refuses to record unless
the `alpha = 0` anchor is **bit-identical** to the uncoupled decode, a positive
alpha changes the output, and the channel conditioned on a real prefix at most
positions. The scrambled control reverses the phone axis — a fixed involutive
permutation preserving entropy while destroying meaning — and loses
claim:lexical_best_vs_scrambled_pts points against the real message at the
operating point, widening monotonically with weight.
`tests/test_lexical_channel.py` carries all of it plus the sweep-validation
checks that refuse a collapsed grid.


## 20 — The spec builds the agent, behind a default-deny registry and a bit-identity gate

**Decision.** `ailm.spec_agent.build_agent_from_spec` constructs a live pymdp
agent from a GNN spec alone: dimensions from `StateSpaceBlock`, counts from
`ModelParameters`, slot roles from `ActInfOntologyAnnotation`, structural forms
materialised directly, and every `Learned(...)` form resolved through
`LEARNED_REGISTRY` — an explicit table keyed on the **complete** keyword set of
the form *together with the ontology role of the variable it parameterizes*.
Every resolver calls the same shipped building-block functions the hand
builders call, so there is one implementation of every learned matrix.

**The alternative it displaced.** Two sources of truth per level — the spec and
the builder — bridged by conformance. Conformance pins structure and roles but
cannot notice a builder and a spec drifting into two different-but-both-valid
models. It also displaced the tempting shortcut of resolving
`Learned(generator=...)` by importing whatever path the string names: executing
a named path is a far wider capability than parsing one, and a mutated spec
could name anything. The registry is default-deny; an unregistered provenance —
or a registered one annotated on a role it is not registered for — refuses to
construct.

**The measurement that settled it.** `tests/test_spec_agent.py`: for all six
shipped levels (L1, L1g, L2A, L2B, L3, L4) the spec-built agent is
**bit-identical** to the builder-built one — same shapes, same dtypes, same
bytes across `A`, `B`, `C`, `D`, and `E` — with a planted-perturbation control
proving the comparator can fail. The gate ran, and passed, *before* any use of
the spec path; migrating first and proving second would have quietly rebased
the claims ledger onto a model nothing had measured.

**What construction caught that conformance could not.** The shipped
`L1g_letter_generation` spec declared `E=Learned(...)` — the Brown bigram as
the policy prior. The constructed agent's `E` is **uniform**: the habit is
installed per step by the generation loop
(`efe_policy_posterior(agent, current, preference, habit)`), exactly as `C`
receives the top-down preference per step. Conformance never noticed because no
shape check binds `E`. The spec was corrected to `E=Uniform(30)` with the
per-step provenance in a comment, mirroring the spec's own `C=Zeros`
convention. The recorded generation numbers are untouched — they came from the
live agent, which was always built this way; it was the *description* that was
wrong.

**Mutations now fail construction, or construct a different model.** Eight of
the eleven mutation specs refuse to construct — the transposed-factor and
swapped-alphabet mutations die on a shape contradiction with the artifact their
own citation resolves to, the drifting-word and learned-emission mutations die
in the registry, the unsized factor and the action-set mutations die on
`ModelParameters` — and the three that do construct (`frozen_position`,
`frozen_transition`, `uninformed_prior`) are internally consistent specs of
*wrong* models, guarded by the assertion that no mutation's constructed agent
is bit-identical to its base. The partition is asserted exactly, so a mutation
drifting between classes is a visible event.

**What was deliberately not done.** The hand builders are **not** retired. They
are parameterized constructors — the negative controls and sweeps inject
mis-wired matrices *through the shipped builder* on purpose (entry 15's
discipline), and a spec path cannot accept an arbitrary matrix without
reopening the hole the registry closes. The builders stay as the parameterized
surface; the spec path is the canonical zero-argument constructor; and the
bit-identity gate binds them permanently — if either side drifts, the suite
fails. Retiring the builders would now remove a guard, not a redundancy.

**Guard.** `tests/test_spec_agent.py`: the six identity gates, a discovery
check that refuses a seventh canonical spec with no identity test, the
comparator's planted-perturbation/dtype/missing-slot controls, the doctored
A/B-swap spec that must be refused **by role**, and the exact mutation
partition. `build_agent_from_spec` additionally runs `assert_conforms` on its
own output, so the construction and conformance layers cannot disagree
silently.

## 21 — Complexity claims are operation and byte counts, never timings

**Decision.** The manuscript's computational-complexity section
(`docs/manuscript/98_supplement_complexity.md`) states costs as exact scalar-operation counts
and exact byte counts. Every asymptotic statement is backed by a closed-form
formula (`ailm.complexity`), every formula is reconciled against an
**instrumented execution** of the shipped recursion on the shipped gate
workload, and every memory figure is derived from the level's GNN spec and
required to equal the live agent's actual bytes. The single writer is
`scripts/record_complexity_claims.py`, which refuses to record if any
reconciliation misses by one operation or one byte.

**The alternative it displaced.** Wall-clock benchmarks. A timing is not
deterministic, so it cannot satisfy the ledger's idempotence discipline
(ISC-4: re-running a recorder is a no-op), it binds the claim to one machine,
and it cannot be derived twice by independent routes and required to *agree
exactly* — which is the project's standard for a verified number. Timings may
appear in prose as context; they may never be claims.

**Why the reconciliation is not green by construction.** The counter does not
apply the formula per sequence; it increments beside each statement of the
recursion **as it executes** — the final step of a sequence performs no
transition push, a per-word reset costs nothing, decode argmaxes count only
when decoding. The formula must therefore reproduce the executed branch
structure from workload summary statistics alone: `T·S + (T−n)·S²` multiplies
for `n` sequences totalling `T` positions. The classic off-by-one (a push
after the final observation) fails the reconciliation, and
`tests/test_complexity.py` carries exactly that wrong formula, a dropped
sequence, and a mis-stated decode flag as negative controls that must fail.

**What the numbers say.** The filtering levels run at Θ(T·S²): from
claim:complexity_l4_ops_per_position operations per position at the 8-state
case level to claim:complexity_l2b_ops_per_position at the 39-state G2P
decoder — measured over the identical populations the accuracy gates score
(enforced: `complexity_l2b_positions` equals `stage2_g2p_eval_positions`, and
`complexity_l3_tokens` equals `stage3_eval_tokens`). The word level is the
payoff: determinism collapses scoring to claim:complexity_l3_ops_per_token
operations per token at any vocabulary size, while the factorization prices
memory at claim:complexity_l3_factored_transition_mb MB against the
claim:complexity_l3_dense_joint_gb GB dense joint — a ratio of
claim:complexity_l3_joint_over_factored, which is exactly P² **under dense storage**. By nonzero count the same ratio is claim:complexity_l3_joint_over_factored_sparse, which tends to P: both transitions are deterministic maps with one nonzero per source state, so one factor of P is bought by factoring and the other by storing densely (decision 58). The whole stack
fits in claim:complexity_l3_memory_bytes bytes or less per level:
interpretability here costs accuracy, not compute.

**Memory is spec-priced.** `spec_memory_bytes_theory` sums each role-carrying
variable's declared dimensions from the GNN spec (times four bytes), plus the
one-scalar policy prior pymdp materialises even for single-action agents —
a term discovered because the first reconciliation missed by exactly four
bytes per recognition level. Dropping that term is a tested negative control:
the formula must not reconcile without it. This binds spec, formula, and
runtime in a second place after decision 20's bit-identity gate.

**The per-symbol decomposition, attributed to the matrix that causes it.**
`step_ops_terms` splits one position's cost by its source. The `A` likelihood
row, its normalizer and the evidence logarithm collect to exactly `3S`. The `B`
push costs `S²` multiplies and `S(S−1)` adds, so `2S² − S`. A decoding level
adds `S − 1` comparisons. Before decoding, a position that carries a push
therefore costs `2S² + 2S` and the last position of every sequence costs `3S`,
because the push is precisely the term a sequence withholds — the same `T − n`
the workload
formula turns on, restated per position. Multiplied out over `n` sequences and
`T` positions the decomposition reproduces `filtering_ops_theory` exactly, at
every state count and both decode settings, so it is that closed form in
attributed form and not a second estimate of it.

Attribution is what the aggregate cannot give. `Θ(T·S²)` says a level is
quadratic in the state space; it does not say the quadratic term is the
transition push and the linear term is the emission and its normalization, and
it does not bound a single position. The decomposition does both, and the bound
is the envelope `[3S, 2S² + 2S + decode]` that every scored position must fall
inside. That envelope is drawn beside the recorded rates, so a level whose
recorded operations-per-position rate falls outside it has changed its recursion
rather than its workload — a distinction an aggregate count hides, because a
workload change moves the same rate for an innocent reason.

**Memory has a second derivation from a different declaration.**
`agent_memory_bytes_from_shape` prices a level from its declared shape alone:

```
4 · ( O·∏ₚ Sₚ  +  A·∑ₚ Sₚ²  +  O  +  ∑ₚ Sₚ  +  A )
```

`A` dense over the joint of every hidden factor, one `B` per factor per action,
`C` over observations, `D` over each factor's states, and the one policy prior
pymdp materialises for every agent. This is worth recording because it reaches
the byte count `spec_memory_bytes_theory` already derives, from a different
starting point: the spec route sums the dimensions the GNN file declares for
each role-annotated variable, and the shape route multiplies out the
observation-alphabet size, the hidden-factor sizes and the action count the
manuscript states. Neither reads the other's input. The two agree to the byte at
all six shipped levels — claim:complexity_l4_memory_bytes at the case level,
claim:complexity_l1_memory_bytes at letters, claim:complexity_l2b_memory_bytes
at G2P, claim:complexity_l2a_memory_bytes at phonemes,
claim:complexity_l1g_memory_bytes at generation, and
claim:complexity_l3_memory_bytes at words — and the shape route is the one
`complexity_memory_data` runs at draw time, so the panel raises rather than
illustrating a configuration the ledger has stopped recording.

The formula also carries the Stage 3 argument in one line: factoring moves the
transition from `(∏ₚ Sₚ)²` to `∑ₚ Sₚ²` and leaves the emission's product
exactly where it was. At claim:stage3_vocab_size words by
claim:complexity_l3_position_states positions that is
claim:complexity_l3_factored_transition_mb MB against
claim:complexity_l3_dense_joint_gb GB.

**What the second derivation does not buy.** The two routes are independent in
where the dimensions come from, not in the cost model: both add the policy-prior
term explicitly, because both encode the same discovered fact about what pymdp
materialises. A spec edit that changes a declared dimension without changing the
stated shape is caught, and so is the reverse. A wrong belief about which arrays
a live agent holds would be wrong in both and caught by neither. That is why the
recorder still compares against `agent_memory_bytes` on the built agent rather
than against the second formula — the measurement is the arbiter, and the two
formulas are checks on each other.

**Guards.** The recorder's refuse-to-record checks; the exact-equality tests
and negative controls in `tests/test_complexity.py`; the figure data functions
(`ailm.figures.complexity`), which recompute the closed form from workload
tokens at draw time and raise rather than draw a panel disagreeing with the
ledger, and which likewise re-derive every memory row from its shape tokens and
raise on one byte of disagreement.

**Falsifier.** Any of these refutes the entry, and each is arithmetic rather
than a judgement: a decomposition whose terms fail to sum to
`filtering_ops_theory` on a workload, at any state count or either decode
setting; a recorded operations-per-position rate outside its level's envelope
`[3S, 2S² + 2S + decode]`; the spec route and the shape route disagreeing by one
byte at any shipped level; either of them disagreeing with the live agent's
`nbytes`; or a timing appearing anywhere in the ledger, which is the failure
this entry exists to prevent and the one no equality check would notice.

## 22 — PRE-REGISTRATION: pymdp parameter learning is conjugate counting under observed states

**Status: prediction recorded BEFORE the measurement ran.** This entry is
committed to git before `ailm.learning` executes a single update. The outcome
entry (23) may confirm it, refute it, or qualify it — but it may not rewrite
this one.

**The gap being closed.** `learn_A`, `learn_B`, `pA`, `pB` and
`infer_parameters` appear nowhere outside the Stage 0 API smoke tests. Every
matrix in the project is estimated by counting outside pymdp and installed
fixed: the agents infer, but have never learned (W2, `TODO.md`).

**Setup.** The letter level, because its identity emission makes learning
supervised-equivalent: the filtered posterior at every step is a one-hot on the
observed letter, so pymdp's Dirichlet accumulation over posterior outer
products should reduce to accumulation of observed bigram indicators. The
learning stream is the Stage 1 held-out text (`HELD_OUT_TEXT`, the population
named by `claim:complexity_l1_positions` = claim:complexity_l1_positions positions) — used here as a
*training* stream for an equivalence claim about two estimators on identical
data, not as evidence of model quality. The comparator is the shipped counting
path: `build_unigram_a(train_on_corpus(stream))`, add-k with k matching the
Dirichlet prior.

**Predictions, with what would withdraw the principle.**

1. **P1 (exact agreement).** With `pB` initialized to all-ones (k = 1),
   `lr_pB = 1`, and beliefs collected by running the live agent's own
   `infer_states` over the stream, the posterior-mean transition returned by
   `infer_parameters` equals the add-1 counted matrix to float32 tolerance
   (max |Δ| ≤ 1e-5). Not "close": add-k counting *is* the conjugate posterior
   mean of a categorical–Dirichlet model, and under one-hot beliefs the two
   computations are the same arithmetic in different code.
2. **P2 (the prior is the only difference).** With `pB` initialized to
   all-twos while the comparator still counts with k = 1, P1's equality must
   FAIL (max |Δ| > 1e-5) — and re-counting the comparator with k = 2 must
   restore it. If equality survives a mismatched prior, the comparison is
   vacuous and nothing was tested.
3. **P3 (non-vacuity).** The learned matrix must differ from the prior's
   expected value (uniform): learning must have moved something.
4. **Corollary.** Because P1 is exact, the learned-B agent and the counted-B
   agent must produce *identical* held-out perplexity on any evaluation
   stream; the TODO's originally scoped "learned vs counted perplexity gate"
   is therefore expected to be an identity, and is asserted as one rather than
   reported as an empirical margin.

**Withdrawal condition.** If P1 fails with matched priors, the principle
"pymdp's `infer_parameters` implements conjugate Dirichlet counting under
observed states" is false for the pinned pymdp (1.0.3 at pre-registration, 1.0.4 at re-verification) — perhaps a learning-rate scaling, a
smoothing pass, or a normalization this analysis missed. In that case the
outcome entry records the measured deviation and traces which implementation
departs from the conjugate arithmetic; the deviation, not the agreement,
becomes the finding, and no learned matrix may ship until it is explained.

**What this deliberately does not claim.** Nothing here is evidence that
learning *helps*: agreement with counting is a correctness anchor, not a
capability. The interesting regime — online learning on non-stationary text
against a frozen control that must lose — remains scoped in `TODO.md` and needs
its own pre-registration.

## 23 — CONFIRMED: the agents have now learned, and learning is counting

**Outcome of entry 22's pre-registration.** The prediction was committed at
`c760357`, before `ailm.learning` existed; this entry records what the
measurement then showed. Every clause held.

| clause | predicted | measured |
|---|---|---|
| P1 — matched priors (k = claim:learning_pseudocount) | max \|Δ\| ≤ claim:learning_tolerance | claim:learning_matched_max_abs_delta |
| P2 — mismatched prior (k = claim:learning_mismatched_pseudocount vs 1) | must exceed tolerance | claim:learning_mismatched_max_abs_delta |
| P2 — comparator re-matched | restored ≤ tolerance | claim:learning_restored_max_abs_delta |
| P3 — moved from initialization | > 0.1 | claim:learning_moved_from_initial |
| corollary — held-out perplexity | identical | claim:learning_learned_perplexity vs claim:learning_counted_perplexity, \|Δ\| = claim:learning_perplexity_abs_delta |

Over claim:learning_stream_positions positions of the Stage 1 held-out text,
pymdp's own loop — `infer_states` collecting filtered beliefs,
`infer_parameters` updating a Dirichlet over `B` — reproduced the shipped
`build_unigram_a(train_on_corpus(...))` counting path to float32 rounding.
The toy anchor in `tests/test_learning.py` verifies the mechanism at integer
resolution: the posterior Dirichlet is the prior plus exactly one unit per
observed transition.

**What was learned beyond the confirmation.** The agreement is only evidence
because it can fail, and the controls show *where* it fails: a mismatched
prior (P2), a comparator counted on different data, and — pinned by its own
test — a learning rate other than 1, which breaks the conjugate identity.
`lr_pB` is therefore not a tuning knob for this project: any value except 1
makes the learned matrix an estimator of nothing the ledger describes.

**Standing decision.** Learned matrices are a verified *mechanism*, not a
shipped parameterization: every shipped matrix remains count-based, and the
learned path writes its own `learning_*` labels, never the `stage*` ones —
the two-writers discipline (entry 10) applied at parameter scale, exactly as
the W2 scoping predicted it would need to be. The regime where learning could
*earn* its way into a shipped model — online updates on non-stationary text
against a frozen control that must lose — remains open in `TODO.md` and
requires its own pre-registration before any measurement.

**Guards.** `scripts/record_learning_claims.py` refuses to record unless all
five clauses hold; `tests/test_learning.py` re-runs P1–P3 on a bounded slice,
carries the harness controls, and binds `data/learning_agreement.npz` (the
matrices the figure draws) to the recorded deviations exactly.

## 24 — Stage 5: the clause level is a recorded null, with a detectability control

**Current status.** This entry records the interpretation that originally
shipped. Decision 83 corrects its unsupported *resolved null* and *negligible*
language: the mixture is non-passing, while the paired privileged-label gain
remains unresolved. Decision 85 then corrects the sentence-level split and
resampling population: the fitted privileged reference's point changes sign and
no longer clears the bar, so the current derived placement is
`ABOVE_REFERENCE`, not bracketed. The detectability control below remains
necessary, but does not establish equivalence or absence.

**The original point-estimate finding.** Measured on UD EWT at bigram order over
claim:stage5_test_positions held-out positions in claim:stage5_test_sentences
typed sentences (claim:stage5_test_interrogatives interrogative):

| quantity | value |
|---|---|
| type-free bar (per-position perplexity) | claim:stage5_typefree_perplexity |
| mixture (the clause model) | claim:stage5_mixture_perplexity — *worse* |
| type ORACLE (true type given) | claim:stage5_type_oracle_perplexity |
| privileged-label reference gain over the bar | claim:stage5_privileged_label_gain_pct% |
| MAP classification vs majority | claim:stage5_classification_pct% vs claim:stage5_majority_baseline_pct% |
| interrogative recall | claim:stage5_interrogative_recall_pct% |
| rank signal (AUC) | claim:stage5_auc |

The privileged-label row is the finding: selecting the fitted chain with the
reference sentence-type label changes case-role perplexity by the signed
reduction claim:stage5_privileged_label_gain_pct percent; the current negative sign means
the fitted reference is worse. This is a receiver-sensitivity probe, not
a hard information bound: under fitted-model misspecification, a mixture can
assign more probability than the label-selected component. The mixture sits
slightly *worse* than the type-free bar. With
claim:stage5_train_interrogatives interrogative training sentences, that result
is consistent with limited-sample estimation variance, but this fitted split
does not isolate variance from smoothing, misspecification, or distribution
shift. MAP classification recovers none of the held-out interrogatives; the
measured rank signal does not translate into the reported classification result.

**Why the instrument is not generically dead.** The identical harness on a
synthetic corpus whose two types genuinely differ classifies at
claim:stage5_control_classification_pct% against a
claim:stage5_control_majority_baseline_pct% majority and drops
mixture perplexity from claim:stage5_control_typefree_perplexity to
claim:stage5_control_mixture_perplexity — the detectability control
(failure-to-reject needs a positive control). That balanced, larger control
shows that the machinery can recover a deliberately large planted signal; it
does not establish power for the real-data effect at the shipped operating
point. Every interpretation is bounded to this EWT sample, its
punctuation-derived labels, this receiver order, and the default decision rule;
it does not extend to English case-role dynamics in general.

**The declared gate was vacuous.** "Free energy tracks clause length" cannot
fail: cumulative free energy is monotone in length under ANY model — each
step adds non-negative surprise — so the trivial baseline passes it. Shown
executably in `tests/test_stage5_gate.py`. The derived gate (beat the
type-free bar) is currently `ABOVE_REFERENCE` — the baseline fails it
definitionally, and neither the fitted privileged reference nor the shipped
model clears it. Because that reference is exceedable, this is not an
impossibility claim. `scripts/audit_gates.py::audit_stage5`
enforces the current non-passing/unresolved contract rather than demanding a
pass: a mixture that started beating the bar, a fitted reference that crossed
it, or a paired interval that moved off zero would flag as a changed finding.

**Standing decision, corrected by entry 83.** The level ships as machinery plus
a non-passing gate outcome with unresolved privileged-label gain: the joint
(type × case) agent, its
GNN spec (`L5_clause`), spec-build bit-identity, closed-form/matrix conformance,
and the full control set all hold. The shipped mixture does not clear the
type-free bar, and the default MAP rule does not improve on its majority
reference at this sample, labeling rule and receiver order. The weak rank signal
and low-discordance classification bound forbid the stronger claim that the
effect does not exist. Decision 83 adds the missing paired uncertainty analysis;
both reported intervals span zero, and no equivalence margin was registered, so
negligibility and absence remain unsupported. The headline score is a
closed-form mixture, not live pymdp inference.
Stage 6 builds on the type variable as a **label** (it exists regardless of
whether case roles predict it), not on Stage 5's posterior.

**Guards.** `scripts/record_stage5_claims.py` refuses to record unless the
closed-form checks hold and the detectability control fires;
`tests/test_stage5_gate.py` re-measures the outcome and its paired bootstrap
intervals, carries the vacuity demonstration, the positive and negative
interval controls, the swapped-blocks and cross-type-leak controls, the
disjointness guard (the observation alphabet is exactly the eight case
roles; PUNCT carries none), and the conformance checks for `L5_clause` and
its mutations.

## 25 — PRE-REGISTRATION: the narrative level, and a receiver-sensitivity reference

**Status: prediction recorded BEFORE the held-out measurement ran.** Committed
before `scripts/record_stage6_claims.py` existed. Train-side statistics are
cited below; no test-split number had been computed when this entry was
written. Entry 26 records the outcome.

**The level.** One step up from clauses: the document. Observation = the
sentence-type sequence (Level 5's latent variable, one level down); hidden
state = the narrative **phase**, operationalised deterministically as the
document-position tercile (opening/middle/closing) — a function of position,
so every parameter is a count and no EM is fitted. Artifact:
`vendor/ud_ewt_narrative_counts.json`
(claim:stage6_train_documents train / claim:stage6_test_documents test documents).

**The train-side observation motivating P1.** Interrogatives concentrate in
document openings relative to the later terciles. The outcome recorder owns the
exact fitted shifts reported in entry 26; this historical preregistration does
not duplicate unledgered train-side numerals. Web discourse in this corpus opens
with questions more often than it continues with them.

**Predictions.**

1. **P1 (recognition).** On the held-out documents, per-sentence type
   log-loss under the phase-conditional model P(type | tercile) is LOWER than
   under the type-unigram bar, i.e. document position genuinely informs
   sentence type out of sample. Withdrawal: a non-positive mean improvement
   means the tercile pattern was noise, and the narrative level has no
   recognition content at this operationalisation — recorded as a null with
   the same rigor as Stage 5's.
2. **P2 (the channel mechanism, ISC-23).** Blending the narrative prediction
   into the clause level's type prior — `D'(α) = (1−α)·P_clause(type) +
   α·P(type | phase)`, then re-forming the joint prior — shifts
   P(interrogative) by a strictly positive amount at α = 1 for at least the
   opening phase, is EXACTLY the unmodified prior at α = 0 (the identity
   anchor, byte-for-byte), and a scrambled control (phases permuted) produces
   a different shift than the real message. This is the proven top-down
   mechanism of `docs/decisions.md` entry 17, applied one level up.
3. **P3 (receiver-sensitivity reference).** The privileged sentence-type arm at
   Stage 5 records how sensitive the fitted receiver is to selecting a
   type-specific chain. Its gain is claim:stage5_privileged_label_gain_pct percent. That
   motivates expecting a small narrative→clause effect on case-role perplexity,
   but it is not a mathematical upper bound: a mixture or intervention can beat
   a label-selected fitted component under misspecification. The stable part of
   the criterion is qualitative: novelty at the sender is insufficient when the
   receiver is insensitive to the message variable.

**Withdrawal condition.** P1 failing withdraws the narrative level's
recognition claim, not the mechanism. P2 failing means the prior-intervention
plumbing is broken and Stage 6 may not be recorded at all. P3 is a sensitivity
expectation, not a threshold or proof obligation.

## 26 — Stage 6: positive point estimate, working intervention, and non-confirmatory primary analysis

**Outcome of entry 25's pre-registration.** The directional point estimate was
positive and the prior-intervention anchors behaved as specified. That does not
make the recognition result confirmatory: the development population and
analysis family were reused, and the correct primary unit is the document.

**P1 — positive point estimate, not confirmation.** Over
claim:stage6_test_sentences held-out sentences in
claim:stage6_test_documents documents, the phase-conditional model scores
claim:stage6_phase_logloss nats per sentence against the unigram bar's
claim:stage6_unigram_logloss — an improvement of
claim:stage6_logloss_improvement_nats nats, positive on
claim:stage6_sign_test_positive sentences against
claim:stage6_sign_test_negative (sign test
p = claim:stage6_sign_test_p). That sign statistic treats sentences as units and
is secondary. In the primary document-clustered log-loss analysis, the interval
from claim:cluster_stage6_logloss_ci_low to
claim:cluster_stage6_logloss_ci_high spans zero and the p-value is
claim:cluster_stage6_logloss_p under a whole-document sign-flip sensitivity
analysis that is exact only under joint central symmetry of document differences.
The document-clustered interval spans zero, so the supported statement is the
positive point estimate, not a population-level effect claim. No target-effect
power analysis was recorded, so this entry makes no separate power claim.

**A distinct live-Agent description, not a replacement estimand.** The registered
score above selects the known tercile from sentence position and never calls
`infer_states`. A separately recorded sequential score resets the real Agent to
its declared `D` for each document, charges predictive loss before each sentence,
then advances the posterior through `infer_states` and
`update_empirical_prior`. It records claim:stage6_live_filter_logloss nats,
an improvement of claim:stage6_live_filter_improvement_nats, with
claim:stage6_live_filter_sign_positive favorable sentences against
claim:stage6_live_filter_sign_negative and a descriptive sign-test p-value of
claim:stage6_live_filter_sign_p. This post-registered execution diagnostic is
outside the frozen 16-member significance family and carries no corrected
inferential verdict. Those rows establish that the live L6 model is
executed and transition-sensitive; they do not turn the observed-position score
into latent-state inference or supersede the non-confirmatory clustered primary.

**P2 — the intervention works, with the anchor exact.** The α = 0 channel reproduces the
clause level's joint prior byte-for-byte (the recorder refuses to run
otherwise); at α = 1 the opening-phase message moves P(interrogative) by
+claim:stage6_opening_interrogative_shift from the static
claim:stage6_static_interrogative_mass, the middle phase by
claim:stage6_middle_interrogative_shift, and the phase-permuted scramble
produces a different shift (claim:stage6_scrambled_opening_shift at the
opening) — so the effect depends on which phase says what, not on the
plumbing.

**P3 — retained as a sensitivity interpretation, not a derivation.** The channel
demonstrably changes the clause level's *type* prior. Stage 5's privileged-label
reference indicates near-nil sensitivity of the fitted case-role likelihood to
that variable, which makes a small delivered effect plausible. It does not cap
the intervention mathematically. Entry 18's useful heuristic remains: novelty
to the receiver is not enough when the receiver's likelihood is insensitive.

**The defect the measurement caught.** The first draft of
`build_phase_transition`'s docstring claimed the strictly-*lower* triangle of
the left-to-right phase transition is zero. Recording the number showed 0.739
of the mass there — because in ``B[next, current]`` orientation a
non-decreasing phase puts mass ON and BELOW the diagonal; it is the strictly
**upper** triangle that must vanish, and does
(claim:stage6_transition_above_diagonal). The orientation trap of decision 1,
caught in this project's own prose by the discipline of recording rather than
asserting. The recorder now refuses to write if the upper triangle is
non-zero.

**Guards.** The recorder's three refusal guards (exact anchor, non-vacuous
shift, distinguishable scramble); `tests/test_stage6_gate.py` re-measures P1,
asserts the structural upper-triangle zero, checks conformance of
`L6_narrative` and rejection of its mutation, and binds the recorded claims
to fresh executions.

---

## 27 — PRE-REGISTRATION: the clause → word channel is predicted to harm (falsification of the first clause)

**Status: prediction recorded BEFORE any measurement.** The manuscript already
states this prediction textually ([`docs/manuscript/03_methods.md`](manuscript/03_methods.md),
the cheapest remaining falsification test of the first clause); this entry
makes it the formal pre-registration in the decision record, so an outcome entry
may confirm it, refute it, or withdraw the clause — but may not rewrite this one.

**The claim under test.** The novel-information criterion (entry 18; refined to
two clauses by entries 25–26) says a message helps iff it is **novel to the
receiver** *and* the receiving likelihood is **sensitive to it** (novelty prices
the message, sensitivity prices the delivery). Four channels are measured: entry
9 L1 → L2B harms (the receiver already observes the letter directly), the
L3 → L1g lexicon-preference seam makes generation work at all, entry 19 L3 → L2B helps
with an interior optimum (novel), and entry 26 narrative → clause is interpreted
against the receiver's near-nil privileged-reference sensitivity. That reference
is not a mathematical bound. The **first clause** (novel ⇒ helps;
redundant ⇒ harms) rests on just those channels. This entry is the cheapest
remaining test of it.

**The channel.** A clause level (Stage 5, L5) passing its own posterior down to
the word level (Stage 3, L3). L3 already observes the words it recognizes — its
emission resolves word identity deterministically from the letters, and its
learned quantity is the word prior `D[0]` (`P(word)`). A clause-derived statistic
about the sentential/clausal context in which words appear is therefore, at the
receiving side, information the word level already possesses: the words are
literally observed. Mixing such a message into `D[0]` substitutes a redundant,
lower-resolution statement for what the receiver can already read exactly.

**Prediction.** The channel **harms**: mixing a clause-derived word-distribution
statistic into L3 prior `D[0]` with weight `alpha` yields a non-positive effect
on the word level predictive, and the *shape* of the sweep is a monotone decline
(no interior optimum) — the redundant-information signature of entry 19, which
says a redundant message has no optimal weight above zero.

**Why the effect is measured on the predictive, not recognition accuracy.** L3
word recognition is deterministic given the letters, so accuracy is pinned and
cannot register harm. The meaningful quantity is the *predictive*: the held-out
perplexity of the prior `D[0]` over in-vocabulary tokens (the same axis decision
6 and ISC-28 govern), or the posterior concentration before the letters resolve
the word. Harm on that axis is the prediction; this is stated here so the outcome
cannot quietly move the needle to a metric that cannot move.

**What would withdraw the principle (falsification).** If a clause → word message
**improves** the held-out predictive, or shows an **interior optimum** in the
mixing weight, then one of the two load-bearing readings fails: either the
receiver did *not* already possess the sentential information (so redundant ⇒
harm is false here), or shape does not diagnose novelty as claimed. Either way
the first clause is **withdrawn or qualified**, and the measured outcome is
recorded below regardless of which way it lands.

**Design (mirroring entries 9, 18, 19).** Receiver held fixed (L3); message =
punctuation-type-conditioned word statistic estimated on the **train** split and
selected by the held-out reference punctuation type, so it is a privileged-oracle
intervention and can still be wrong; linear mixing into `D[0]` by weight `alpha`. Guards,
all taken from the established channel harness: an `alpha = 0` identity anchor
bit-identical to the uncoupled prior; a non-vacuity check that `alpha > 0`
differs; a scrambled control (entropy-preserving permutation of the clause
statistic) that produces a different effect; a paired test over the full held-out
token set; and a grid guard that refuses a collapsed sweep (entry 9). A recorded
null needs the same controls a recorded positive does.

**What this deliberately does not claim.** It is not evidence about learning, and
it is not a recommendation about whether to build the channel — harm is the
prediction, and a recorded harm is a finding, not a verdict.

**Outcome chronology (corrected by decision 88).** The channel was built and
first measured on the bridge
corpus `vendor/brown_clause_word_counts.json` (Brown sentences typed by
sentence-final punctuation, restricted to the Stage 3 word vocabulary; see
`docs/provenance/sources.md`). The pre-registered prediction — **harm, with a
monotone decline and no interior optimum** — was **not confirmed**, and per
the withdrawal condition above this qualifies the first clause of the
novel-information criterion on this channel.

**Sender correction (decision 89).** The implemented arm reads each held-out
sentence's punctuation-derived reference type. It never calls the L5 Agent or
uses an L5 posterior. Its measured source is therefore a privileged
punctuation-type oracle, not L5; any earlier `L5→L3` label is historical design
shorthand and is superseded for scientific attribution.

The original artifact produced a mixed-direction reading. That reading is
historical, not current: decision 87 replaced the leaking Stage 3 token split
with an authenticated category-stratified document partition, and decision 88
records the downstream remeasurement. Active `clause_word_*` labels refer only
to the corrected artifact; the retired values are not preserved as if they
were current claims.

On the *declared* axis (held-out perplexity of the prior `D[0]` over
in-vocabulary tokens), the clause-typed word message slightly *improves* the
predictive: anchor claim:clause_word_anchor_ppl, best
claim:clause_word_best_ppl at alpha = claim:clause_word_best_alpha
(delta claim:clause_word_best_delta_ppl perplexity). That selected value is the
high tested endpoint, not an interior optimum
(claim:clause_word_optimum_is_interior). Perplexity improves at every positive
grid step, so the sweep does not locate an optimum and does not reproduce either
predicted signature — harm, or the redundant-information monotone-harm shape
(claim:clause_word_monotone_decline).

Two readings of the corrected non-confirmation are pinned by the controls:
- **The message is real, not inert.** The entropy-preserving reversed-axis
  control is far worse at the best weight
  (claim:clause_word_scrambled_ppl_at_best vs
  claim:clause_word_best_ppl), so the sweep measured a genuine
  word-structure message, not the effect of adding arbitrary mass to the
  prior. The per-type message differs from the marginal prior by up to
  claim:clause_word_maxdiff_interrogative (interrogative) — the type
  conditioning carries actual information.
- **Both directional summaries now favor the message.** At the best weight,
  claim:clause_word_best_improved_tokens tokens are made more probable against
  claim:clause_word_best_degraded_tokens made less probable, while aggregate
  perplexity also falls. The exact nested-token diagnostic is retained only on
  its log10 scale (claim:clause_word_best_pair_log10_p); it is not treated as an
  independent-token inferential result. Whole-document uncertainty is reported
  under decision 89, outside the frozen confirmatory family; changing the corpus
  partition does not retroactively create a new confirmatory hypothesis.

**What this does and does not claim.** This is not evidence about learning. It
is a measured reading on one channel at one corpus and order: the predicted
harm did not reproduce, and on the declared axis the effect is a small gain
whose best tested weight is the high grid endpoint. The sweep does not identify
an optimum. The improvement is the outcome that entry 18's first clause treats
as withdrawing/qualifying it. The first clause remains supported by the channels
that confirmed it (entry 19's L3->L2B help; entry 9's measured L1->L2B harm);
the clause -> word result adds that a *coarse, type-conditioned* message into
a receiver whose prior is a marginal — not already type-structured — is not
cleanly redundant, which is a qualification, not a contradiction, of the
criterion's first clause. The corrected direction is descriptive remeasurement,
not a fresh confirmatory population.


---

## 28 — PRE-REGISTRATION: online learning on non-stationary text beats a frozen control that must lose

**Status: prediction recorded BEFORE the measurement.** W2 open half
([`TODO.md`](../TODO.md)): decisions 22–23 established that pymdp
`infer_parameters` is conjugate counting under *observed* states on a stationary
held-out stream — a correctness anchor, not a capability. This entry
pre-registers the increment where learning could actually *earn* its way in: an
agent that keeps learning on a changing distribution should track a drift that a
frozen (non-learning) agent cannot. Committed before any update runs on the
non-stationary stream.

**Setup (scoped to where conjugacy is exact).** The letter level, whose identity
emission makes beliefs path-supervised-equivalent, so conjugate counting is
exact for every step (decision 22). A non-stationary stream is constructed by
training on a first corpus, then continuing under a second whose letter-bigram
transition statistics genuinely differ (a principled, injected drift — measured,
not asserted, by a control below). The **online** agent continues pymdp
`infer_parameters` updates over the drifted segment. The **frozen** control is
the add-k counted `B` fixed from the first distribution only — it must lose.

**Predictions, with what would withdraw the principle.**

1. **P1 (the drift is real).** The two segments differ in held-out letter
   perplexity under the frozen matrix by more than tolerance — the drift is
   measurable, so a gap later cannot be blamed on a stream that never changed.
2. **P2 (frozen must lose).** Measured on held-out text from the *second*
   distribution after the drift point, the online agent perplexity is strictly
   lower than the frozen control. This is the load-bearing clause: online
   conjugate updates recover the drift.
3. **P3 (identity before the drift, non-vacuity).** On held-out text from the
   *first* distribution, the two agents score identically (the learning rate,
   at 1, makes the online path equal to counting — decision 23) — no advantage to
   learning on stationary text, which is exactly why this regime, not stationary
   learning, is the test. And a re-fit control (the frozen matrix re-estimated on
   the second segment) scores at least as well as the online agent, showing the
   drift itself, not estimator noise, is what the online path tracks.

**Withdrawal condition.** If P2 fails — the online agent does not beat the frozen
control on the post-drift held-out set, or ties — the claim that online conjugate
learning recovers non-stationarity is withdrawn or qualified, and the measured
deviation becomes the finding, recorded below like any other. P1 failing would
withdraw the premise (no drift to track); P3 failing would reveal a plumbing or
pre-registration defect.

**What this deliberately does not claim.** It is not a claim about tracking
hierarchical structure or about unobserved states (the G2P `learn_A` case, whose
states are not observed, is explicitly warned in `TODO.md`, and conjugate
agreement is NOT the expected outcome there). It claims only that online
conjugate updates track a measurable drift better than a frozen point estimate —
the minimal regime in which learning beats not-learning.

**Guards.** The recorder refuses to record unless P1, P2 and P3 all hold;
`tests` re-measure P1–P3 on a bounded slice of the same stream; the identity hook
(P3 first half) and the re-fit control are bound so the drift cannot be
manufactured by the update count. A null needs the same controls a positive does.

**Outcome (2026-08-06): CONFIRMED.** The experiment was built and measured on
`vendor/brown_online_drift.json` — two Brown genres (government, then romance)
whose letter-bigram statistics genuinely differ, filtered and learned through
the real pymdp `infer_parameters` path across the drift point. Every
pre-registered clause held:

| clause | prediction | measured |
|---|---|---|
| P1 — the drift is real | frozen perplexity moves > claim:online_drift_tolerance between genres | claim:online_max_drift (frozen claim:online_ppl_frozen_pre pre vs claim:online_ppl_frozen_post post) |
| P2 — frozen must lose | online < frozen on post-drift held-out | claim:online_ppl_online_post vs claim:online_ppl_frozen_post, gap claim:online_frozen_vs_post_gap_ppl |
| P3 — re-fit control | re-fit ≥ online | claim:online_ppl_refit_post (≤ online); pre-drift identity holds at lr=1 |
| measured stream | — | claim:online_pre_learn_positions + claim:online_post_learn_positions learn, claim:online_pre_hold_positions + claim:online_post_hold_positions hold, k = claim:online_pseudocount |

The online agent — filtered through the pre genre, then *continuing*
`infer_parameters` over the post genre — reaches claim:online_ppl_online_post
on post-drift held-out text, beating the frozen control's
claim:online_ppl_frozen_post by claim:online_frozen_vs_post_gap_ppl perplexity
(= the frozen, fixed from the first genre only, loses, as required). The
re-fit control (frozen matrix re-estimated on the second genre alone) scores
claim:online_ppl_refit_post — at least as well as the online path — and the
identity hook (the online matrix as it stood *at* the drift point equals the
frozen matrix to float32 rounding, P3) confirms the learning rate of 1 makes
online equal counting, so the advantage comes from tracking the drift, not
from estimator noise. The claim is recorded as confirmed
(claim:online_confirmed) against the pre-registered withdrawal condition: P2
did NOT fail.

**Caveats, stated with the confirmation.** The effect is measured on the
letter level where conjugacy is exact and the drift is a *principled, injected*
genre change — the minimal regime in which learning beats not-learning. It is
not a claim about unobserved states (the G2P `learn_A` case remains warned in
`TODO.md`), and the gap is a property of this corpus and this stream length,
not a universal margin.


## 29 — PRE-REGISTRATION: G2P emission learning with unobserved states

**Status: prediction recorded BEFORE the measurement.** This entry is committed
before the G2P `learn_A` experiment or its recorder runs. The outcome may confirm,
refute, or withdraw the prediction below; it may not rewrite this entry.

**Question.** Does pymdp's `infer_parameters` learn a G2P emission matrix `A`
from letter observations when the phone states are unobserved? Unlike entry 22,
the hidden phone posterior is not one-hot: it is inferred from the current letter,
the phone transition, and the start prior. Therefore conjugate agreement with an
external count table is **not** the expected outcome and is not the gate.

**Controlled design.** Use the vendored CMUdict 1:1 G2P train/test split. Build the
real G2P agent from the train split with its shipped transition and start prior,
then filter the held-out-aligned training words with the live `infer_states` and
`update_empirical_prior` loop. Call `infer_parameters` once with `learn_A=True`,
`pA` initialized to a positive symmetric Dirichlet concentration, `lr_pA=1`, and
`pB`/`pD` fixed. Evaluate the learned emission on the disjoint test pairs with
the same forward decoder and positions as the fixed-emission baseline. The
recorder must also run an independent numpy forward decoder and refuse to record
unless the pymdp and numpy paths agree on every decoded test position.

**Predictions.** (P1) The learned emission moves from its symmetric initialization
(non-vacuity), and the live decoder is not identical to the fixed-emission
control. (P2) Because states are unobserved, the learned matrix will **not** be
required to equal the externally counted emission within the observed-state
tolerance; the measured deviation is the result, not a failed gate. (P3) The
learned and fixed decoders will be compared on held-out G2P accuracy and paired
McNemar counts, but no positive direction is predicted: improvement, null, or
degradation are all admissible outcomes. (P4) A frozen-state negative control
that supplies the true phone one-hot beliefs must recover the entry-22-style
conjugate counting identity; this control verifies that the learning machinery
can move A when its state evidence is observed and that any non-agreement in the
main lane is attributable to latent-state inference rather than a broken
parameter update.

**Withdrawal conditions and guards.** Withdraw the experiment without recording
claims if the train/test split overlaps, if the learned path uses future letters,
if `infer_parameters` is not actually called with `learn_A=True`, if the initial
emission is not symmetric and positive, if the learned matrix does not move, or
if the independent decoder disagrees with pymdp. A positive result is not required
for the experiment to be informative. The fixed-emission baseline, the frozen-state
identity control, and the full held-out paired comparison must all be recorded
under distinct `g2p_learn_A_*` labels by one recorder.

**What this does not claim.** It is not evidence that unsupervised emission
learning improves G2P generally, nor that latent-state learning is identifiable
from this corpus. It is a bounded mechanism experiment whose primary result is
the observed movement and held-out comparison under this exact model and split.

**OUTCOME (2026-08-07, recorded from the run).** Latent-state emission learning
is measured and does **not** improve held-out decoding; the learned emission is
substantially worse than same-data counting, and the controls localise the cause
to latent-state inference. All claims are recorded under `g2p_learn_A_*` labels by
`scripts/record_g2p_learning_claims.py` (run `g2p_learn_A_latent_states`).

- **P1 held (non-vacuity).** The learned emission moved
  claim:g2p_learn_A_max_movement from the symmetric uniform initialization over
  claim:g2p_learn_A_train_pairs training words (every 16th train pair — an even
  stride, never a prefix).
- **P2 measured, direction not predicted.** On the full held-out test split
  (claim:g2p_learn_A_eval_positions positions) the latent-state learned emission
  scores claim:g2p_learn_A_learned_accuracy% while add-1 counting over the *same*
  learning lane scores claim:g2p_learn_A_fixed_accuracy%. The deviation from
  same-lane counting is claim:g2p_learn_A_max_count_deviation (max abs).
- **P3, the paired comparison.** The learned decoder is right on
  claim:g2p_learn_A_mcnemar_learned_only positions that the counted decoder
  misses and wrong on claim:g2p_learn_A_mcnemar_fixed_only that it gets — a
  one-sided collapse, not a trade. No positive direction was predicted, and none
  was found (`g2p_learn_A_confirmed` = 0).
- **P4, the observed-state control, holds exactly.** Feeding the true phone
  one-hot beliefs to the same `infer_parameters` machinery reproduces same-lane
  counting to claim:g2p_learn_A_gold_max_abs_delta (max abs, float32 tolerance);
  the oracle re-derivation scores claim:g2p_learn_A_oracle_gold_lane_accuracy%,
  identical to the counted emission. The pymdp decoder and the independent numpy
  oracle agree at every one of the claim:g2p_learn_A_eval_positions evaluation
  positions (`g2p_learn_A_decoder_disagreements` = 0), so the recorded numbers
  describe the agent, not the oracle.

**What this outcome does and does not say.** It is a controlled, pre-registered
negative: under this model order, this split, and a weak symmetric Dirichlet
prior, pymdp's latent-state emission update is harmful to held-out G2P decoding
relative to counting the same data — the observed-state control proves the
machinery is sound, so the harm is attributable to the latent-state inference
itself, exactly the case entry 22 explicitly declined to extrapolate to. It is
not a claim that latent-state learning can never help G2P; it is a claim that this
bounded experiment's answer is no, with controls that make the no trustworthy.

---

## 30 — The external benchmark is a reference frame, and its protocol must not tilt it

**Context.** The external benchmark places AILM beside zero-shot Ollama probes
on the three next-symbol tasks. Its purpose is a reference frame: AILM has no
pretraining, so the probes are expected to be stronger, and a comparison that
flatters AILM is a worse outcome for this paper than one that does not.
Extending it — more positions, paired intervals, calibration, a second probe —
required first establishing that the existing numbers measured what they
claimed. Three defects meant they did not, and all three ran in the same
direction: toward flattering AILM.

**Defect 1: the letter prompt contradicted the letter inventory.** `ALPHABET`
has thirty symbols: a–z plus space, apostrophe, hyphen and period. The gold
stream, the response schema and the scorer all used those thirty. The prompt
asked for "one lowercase letter". A space is the single most frequent gold
symbol in the held-out passage, so on nearly a third of scored positions the
probe was asked for an answer the prompt had told it not to give — and
`parse_letter_answer` stripped whitespace before checking for length one, so a
correct space answer would have been recorded as *invalid and wrong* had the
probe given one. A zero unparsable count was read as evidence of a clean
contract; it was evidence the probe was obeying an instruction that cost it
those positions. v3 states the inventory in the prompt, delimits the sequence so
a trailing space is visible, and parses a lone space as the symbol it is.

**Defect 2: the v2 letter fixture was one passage repeated three times.** Its
1,322 raw positions held only 480 distinct ones. Both scored systems are
deterministic, which makes a repeated position the same observation counted
again: proportion intervals come out too narrow and the exact McNemar test is
anti-conservative, because discordant pairs arrive in perfectly correlated
blocks. Those two figures describe the superseded fixture and are deliberately
written as plain numerals rather than claim tags, because the tags would resolve
to the current fixture and make the sentence contradict itself.

v3 deduplicates the pool before sampling, and keeps deduplication as the
sampling contract even now that the fixture no longer repeats. The 480-position
ceiling was real while it lasted: it made the letter contrast unresolvable at
any `n`. Extending the held-out prose removed it, and the current pool holds
claim:bench_letter_pool_unique distinct positions.

**Defect 3: the phoneme comparator was fitted on its own test set.** The
vendored `phone_bigrams` are pooled over every CMUdict base word, and the
phoneme positions were drawn from those same words. AILM's phoneme number was
train-on-test against a zero-shot probe. v3 draws phoneme positions from the
Stage 2 held-out split and recomputes the comparator's phone statistics from the
training split alone. The leak turned out to be diffuse rather than large: on
identical held-out positions the pooled-statistics comparator and the train-only
one differ in the third decimal place, because the scored words are a small
share of the pooled counts. This is a **validity** defect, not a magnitude one,
and nothing here should be read as claiming the old phoneme number was inflated
by a large amount. The fix is not contingent on the size of the effect it
removes: a comparator fitted on its own test set is not reportable whatever the
difference turns out to be.

**Decision.** Bump to `external-benchmark-v3` and fix all three, rather than
re-record v2 at a larger sample. More precision on a biased estimate is worse
than less precision on an honest one: it moves an interval that was never
centerd on the right quantity, and it does so in a way that reads as rigor.
The v2 numbers are superseded, not merely refreshed, and no v2 value should be
compared to a v3 one.

**What else v3 adds, and the reasoning behind each.**

- **Sample size raised to claim:bench_letter_positions per task.** Under v2 the
  letter contrast had too few discordant pairs to separate a real gap from
  noise, and its exact p was consistent with either. The v3 sample is *not* the
  pool maximum — the pool holds claim:bench_letter_pool_unique positions, twenty
  times larger. It is the largest round count the measurement budget supports at
  roughly forty-five minutes per three-model run, chosen because it is where the
  phoneme and G2P contrasts resolve decisively; the letter arm remains
  underpowered there, as the power table below states.
- **Paired difference intervals.** Two overlapping per-group Wilson intervals do
  not answer the paired question, and reporting only them invites the reader to
  eyeball an overlap and conclude "no difference". Each contrast now also
  carries the exact conditional interval on the paired difference, built by
  Clopper-Pearson on the discordant share and mapped back to the difference
  scale.
- **Power at a fixed reference share, never at the observed effect.** Power at
  the observed effect is a monotone transform of the p-value (Hoenig & Heisey
  2001) and carries no evidence the p-value does not. The benchmark mirrors the
  repository's existing `stat_*_power_at_reference` pattern instead.
- **A declared benchmark family.** The benchmark's paired p-values are excluded
  from the repository-wide significance family because they test an external
  reference point rather than a claim this paper makes. Excluded is not
  uncorrected: the recorder declares its own family and Bonferroni threshold,
  and `tests/test_statistics.py` refuses a bench p-value that is not in it. The
  p-value scan there was also widened — it matched only labels *ending* in
  `_p`, so every `bench_*_paired_p_<model>` label had sat outside the closure
  check while its exclusion entry read as though it were load-bearing.
- **Context sweep at claim:bench_letter_window_short and
  claim:bench_letter_window_long characters.** The comparator is order one and
  therefore context-invariant by construction; the sweep is where the probe's
  architecture can show an advantage the comparator cannot answer. The long
  window is held below the passage repetition period, or the widened prefix
  would contain the target itself at the aligned offset.
- **Both letter inventories, from one run.** Fixing the prompt makes the space
  symbol answerable, but it does not settle which inventory the letter task
  should score against, and the answer is not neutral: the thirty-symbol stream
  and its twenty-six alphabetic subset can rank the systems differently. Both
  arms are recorded from the same calls — the alphabetic arm is a restriction of
  the per-position outcomes, not a second run — so a reader can see whether a
  letter conclusion is about the models or about the inventory.
- **Reference points, and no ladder.** `gemma3:4b` and `qwen2.5:3b` are
  reference points from different model families, not rungs of one scale. Any
  probe set ordered by parameter count invites the scale-ladder misreading, so
  the panel states explicitly that no trend may be fitted: probes differ in
  training recipe as well as size, and a difference between any two is a
  difference between two models and nothing more.
- **Calibration.** Accuracy compares argmaxes; ECE and the Brier score compare
  distributions, which is the one axis on which a bigram column and a
  transformer's logprobs are commensurable. Ollama reports at most twenty
  candidates for a thirty-symbol inventory, so the completion of the remaining
  mass is stated rather than hidden.
- **A captured-mass floor on reportability.** Truncation is not merely a caveat:
  when most of the reported distribution is completion rather than measurement,
  the completion pushes confidence toward its floor, and on a usually-wrong
  model lower confidence *reduces* the measured calibration error. The
  unmeasurable model would then read as the best-calibrated one. Calibration is
  therefore reported only above claim:bench_calib_mass_floor observed share, a
  model below it gets no calibration labels at all rather than labels with a
  warning, and two positive controls pin the hazard so the guard cannot pass
  vacuously. No probe in the served registry falls below the floor, so the gate
  rejects nothing here — which is a statement about this run's probes and not
  evidence that the floor is in the right place. Entry 32 is where it was
  located; the controls are what keep the guard from passing vacuously in the
  meantime.
- **Seed replicates are a determinism check, not a variance estimate.**
  Decoding is greedy at temperature zero, so the seed cannot reach the output.
  The measured spread is expected to be exactly zero, and it is recorded as
  evidence for that rather than presented as run-to-run variability.

**The letter task's power limit, and what moved it.** Only discordant pairs
inform the paired contrast. The original letter fixture was one passage repeated
three times, which after deduplication left too few distinct positions for a
reference-sized effect to be decidable at any `n` — the contrast was unresolved
by construction rather than by choice of sample size. That limit has since been
removed by extending the held-out prose: the pool now holds
claim:bench_letter_pool_unique distinct positions, enough that the fixture is no
longer what binds.

The binding constraint is now compute, and it should be stated in those terms
rather than left implicit. Every figure below is power against the declared
reference share at the observed discordance rate, and each is labeled with the
alpha it assumes — the two columns differ by enough that carrying a number
across them would misstate the design by a wide margin.

| sampled positions | power at alpha 0.05 | power at the corrected 0.002778 |
|---:|---:|---:|
| 100 | 0.045 | 0.006 |
| 400 | 0.151 | 0.024 |
| 1000 | 0.397 | 0.094 |
| 2000 | 0.661 | 0.282 |
| 4000 | 0.924 | 0.670 |
| 6000 | 0.988 | 0.884 |
| claim:bench_letter_pool_unique (full pool) | 0.998 | 0.967 |

The benchmark reports its paired tests against the corrected threshold, so the
right-hand column is the one that governs: conventional power arrives near
n=4000, corrected power near n=6000. Each position costs one model call per arm
per model, so a decisive letter run is a multi-hour measurement rather than a
different flag. The recorded n is an honest budget choice, and the letter
contrast should be read as underpowered at this sample size — a different and
weaker statement than the one this entry made when the fixture was the
bottleneck.

**Falsifier.** A v3 letter run whose invalid count is non-zero on space-gold
positions, a sampled position that appears twice, a phoneme comparator whose
statistics include a scored word, a context window at or beyond the passage
period, a bench p-value absent from the recorder's declared family, or a
calibration figure reported for a model whose observed share falls below the
declared floor — any of these refutes the protocol as described here.

## 31 — The interpretability claim is demonstrated on a named column, not asserted

**Context.** The manuscript's opening argument is that this architecture answers
"why did the model predict that?" by naming a column of `B` and reading the
numbers in it. Every section downstream depends on that being true, and no
section showed it happening. A reader had to take the central methodological
claim of the paper on the author's word, in a paper whose whole thesis is that
its numbers should never have to be taken on anyone's word.

**Decision.** Record one worked example and walk it in prose.
`scripts/record_worked_example_claims.py` reads the shipped letter level's
transition matrix and records the ranked successors of two context letters:
`q`, where the answer is near-deterministic, and `t`, where it is not. The
manuscript prints both from those claims, so the demonstration is a measurement
subject to the same ledger discipline as every other number.

Two contexts rather than one, deliberately. The `q` column alone is a better
advertisement and a worse description: it would leave a reader with the
impression that the level is typically certain, when the ordinary case is a
genuine distribution over successors. The example is chosen so a reader who
knows English orthography can check it without running anything, which is the
property being claimed — an audit available to the reader, not only to the
author.

**Why the values are strings.** These are the only ledger entries whose value is
a symbol rather than a measurement of one. Space and period are recorded under
readable names rather than as literal `" "` and `"."`, which would be invisible
in the rendered PDF and unreviewable in a diff.

**Falsifier.** `tests/test_worked_example.py` rebuilds the transition matrix
from the vendored counts and re-derives each recorded rank and probability. A
recorded successor that is not the successor the matrix ranks there, a column
that does not sum to one, a `q` column that has stopped being near-deterministic,
or a `t` column that has become so — any of these refutes the paragraph, and the
last two refute its argument even while every printed number stays correct.

## 32 — The captured-mass floor, located: the declared floor is not conservative

**Context.** Entry 30 put a captured-mass floor on calibration reportability:
a probe whose answer-slot distribution is less than
claim:bench_calib_mass_floor *observed* gets no calibration labels at all,
rather than labels with a warning. The reasoning was sound — below some
captured mass the completion rule rather than the model decides the reported
number — but one half is where a distribution stops being mostly a
measurement, which is a definition. It is not a measurement of where the two
completion rules actually stop disagreeing, and those are different
quantities. The measured probes sit at
claim:bench_calib_captured_mass_gemma3 and
claim:bench_calib_captured_mass_qwen25, so the live protocol observes almost
nothing across the interval the floor sits in and cannot say whether the floor
is in the right place. A principled threshold that has never been located is
still a guess about the thing it guards.

**Decision.** Locate it, offline, with no model call.
`scripts/record_calibration_floor_claims.py` takes AILM's own letter
transition columns — real next-symbol distributions over the same inventory of
claim:bench_floor_inventory_size symbols that the benchmark scores,
deterministic, and already the comparator the calibration arm reports — and at
each point on a grid of target captured masses truncates a column to the
smallest top-k reaching that mass, then reconstructs it under both completion
rules the project has: the shipped min-bound rule, read out of
`_distribution_from_candidates` rather than restated, and an observed-only
rule that renormalizes the survivors and gives an absent symbol nothing. Both
reconstructions are scored with the two metrics the benchmark reports, and the
disagreement between them is the measured quantity. The negligibility
threshold claim:bench_floor_epsilon is declared next to the grid rather than
chosen once the curve was visible: a threshold picked from a plot locates
whichever floor was wanted.

**What it found, which is not the comfortable answer.** The smallest captured
mass above which neither metric again disagrees by more than the threshold is
claim:bench_floor_located_captured, which is claim:bench_floor_located_ratio
times the declared claim:bench_floor_declared_floor. The declared floor is
therefore **not conservative**; it sits at roughly half the mass at which the
two rules converge. At the declared floor the rules still differ by
claim:bench_floor_atdeclared_ece_divergence in ECE and
claim:bench_floor_atdeclared_brier_divergence in Brier. AILM's whole recorded
ECE is claim:bench_calib_ece_ailm, so on this distribution family the
completion rule's contribution at the declared floor is a large fraction of
the effect the calibration arm exists to report.

**The located floor is unreachable in principle, not merely unmet.** Reaching
it needs claim:bench_floor_located_k of claim:bench_floor_inventory_size
candidates on average against the server's cap of claim:bench_floor_cap, and
claim:bench_floor_located_over_cap_fraction of positions need more than the
cap. No setting of this API returns that many, so a floor placed there would
withhold calibration for every model, always. At the declared floor the same
statistic is claim:bench_floor_atdeclared_k candidates, which the cap supplies
easily — the declared floor is satisfiable because it is low, not because it
is right. That asymmetry is the honest shape of the problem: the reachable
threshold is the one that does not separate rule from model, and the
separating threshold is the one this API cannot reach.

**A methodological trap, reported because the obvious locator falls into it.**
Each metric's *signed* rule difference crosses zero exactly once —
claim:bench_floor_ece_sign_changes and claim:bench_floor_brier_sign_changes
sign changes across the whole grid — at claim:bench_floor_naive_ece_captured
and claim:bench_floor_naive_brier_captured captured mass, which is essentially
the declared floor. At that crossing the two reconstructions are still
claim:bench_floor_naive_ece_rule_tv apart in total variation: they are nearly
disjoint distributions that happen to score the same. The mechanism is
opposite bias, not agreement. Min-bound flattens the distribution and
understates confidence; observed-only sharpens it and overstates confidence;
their difference passes through zero on the way from one bias to the other. A
locator that took the first crossing would have "confirmed" the declared floor
to three decimals, by coincidence, and the confirmation would have been the
most persuasive artifact in the section. `locate_divergence_floor` therefore
requires a **sustained** crossing — scanning down for the last grid point at
which the divergence exceeds the threshold — and records the naive answer
beside the located one so the size of the trap stays visible rather than being
described.

Two further shapes are recorded rather than assumed, because a floor argument
quietly presupposes both. The metric divergences are not monotone in captured
mass (claim:bench_floor_ece_increases and claim:bench_floor_brier_increases
rising steps), while the rule divergence in total variation is
(claim:bench_floor_tv_increases). And below
claim:bench_floor_fidelity_crossover_captured captured mass the observed-only
rule reconstructs the true column *better* than the shipped min-bound rule:
min-bound's conservativeness is a property of the bound, not a claim to be the
closer reconstruction.

**Identity control, and a sample-size control.** At full capture truncation
removes nothing and the two rules must agree exactly. The grid's top point
reproduces the recorded claim:bench_calib_ece_ailm and
claim:bench_calib_brier_ailm as claim:bench_floor_identity_ece and
claim:bench_floor_identity_brier, with a rule divergence under
claim:bench_floor_identity_tolerance. The sweep is therefore exercising the
shipped reconstruction path rather than a reimplementation that agrees with it
by construction. A second sweep at claim:bench_floor_check_positions
positions, four times the primary claim:bench_floor_positions, locates
claim:bench_floor_check_located_captured — so the located floor is a property
of the completion rules and not of the sample size that ECE's binning would
otherwise import.

**What this does not license.** It does not say the reported calibration
numbers are wrong. The sweep's ground truth is AILM's own letter columns,
which are comparatively flat: mean stated confidence
claim:bench_calib_confidence_ailm, against
claim:bench_calib_confidence_gemma3 and claim:bench_calib_confidence_qwen25
for the two reported probes. A peaked distribution loses far less to
truncation than a flat one, so the measured probes diverge much less at their
own captured masses than this sweep predicts at those masses. The correct
conclusion is narrower and more useful than "the numbers are wrong": captured
mass is a **proxy** for rule-dependence, the proxy is not conservative on at
least one realiztic distribution family, and what actually licenses the two
reported figures is the direct per-model check that these particular systems
are close to rule-insensitive above the floor — not the floor itself.

**Decision, second half: the constant stays and the successor is registered.**
`CALIBRATION_MASS_FLOOR` is not changed here and the calibration arm is not
withdrawn. Moving the constant to the located value on this evidence would
withhold both reportable systems on the strength of a distribution family
neither belongs to, and would do it with a threshold no API run can satisfy —
trading a proxy that is too loose for a gate that is vacuous in the other
direction. The recommended successor is a **direct per-model divergence
gate**: reconstruct each probe's own recorded answer-slot distribution under
both rules, and report calibration only where the metric disagreement on that
model's own data falls below claim:bench_floor_epsilon. Captured mass then
becomes a diagnostic printed beside the figure rather than the thing deciding
it.

Three pieces of evidence would justify adopting it, and this pass produced
none of them. First, the per-model divergence recorded as a ledger label for
every reported system: today the rule-insensitivity of Gemma 3 4B and Qwen 2.5
3B above the floor is stated in prose, and a gate cannot rest on a number that
is not in `data/claims.json`. Second, a positive control in which the gate
*rejects* a system — a starved probe whose divergence exceeds the threshold —
since a gate that has never refused anything is a comment. Third, a
demonstration that the per-model check does not inherit the sign-change trap:
a single scalar disagreement evaluated at one captured mass is exactly the
quantity that can pass through zero while the reconstructions are far apart,
so the gate must read the rule's total variation alongside the metric
disagreement, as claim:bench_floor_naive_ece_rule_tv shows is necessary.

**Falsifier.** The recorder refuses to write any claim unless truncation at
full mass is the identity, the divergence exceeds the threshold somewhere on
the grid, and the located floor is not at the bottom of the grid — so a run
that writes claims has already ruled out a vacuous bound and a grid that
starts above the transition. Beyond that: a located floor at or below
claim:bench_floor_declared_floor would make the declared floor conservative
after all and this entry wrong; a sustained and a first crossing that
coincided would remove the trap and with it the reason for the locator's
shape; a top-of-grid divergence above claim:bench_floor_identity_tolerance
would mean the sweep is scoring something other than the shipped
reconstruction; and a located floor that moved materially between
claim:bench_floor_positions and claim:bench_floor_check_positions would make
it a property of the sample rather than of the rules.
`tests/test_calibration_floor.py` holds each of these.

## 33 — Post-hoc power for three secondary arms is derived, not remeasured

**Context.** Three arms of the external benchmark reported a contrast with no
detectability figure beside it: the alphabetic-only letter restriction over
claim:bench_letter_az_positions positions, the prompt-format arm, and the
context-window sweep between claim:bench_letter_window_short and
claim:bench_letter_window_long characters. Each recorded its discordant counts
and its exact p, and none recorded power. The supplement could therefore say
two systems differed, or did not, and say nothing at all about whether the
pair count could have found a difference of the size the benchmark declares as
its reference. An unqualified "no difference" is the worst of the available
readings and was the one the text invited.

**Decision.** Derive the six missing figures at render time rather than
remeasure them. `ailm.benchmarks.paired_power` is a pure function of `b` and
`c` — the recorder passes it nothing else — so the power of an arm whose
counts are already in the ledger needs no model call, only the same call at a
different moment. `paired_power_from_counts` in
`src/ailm/manuscript_variables.py` wraps that function and emits
`BENCH_LETTER_AZ_POWER_*`, `BENCH_LETTER_PROMPT_POWER_*` and
`BENCH_LETTER_CTX_POWER_*` for each served probe.

**Why derivation needs a license, and what the license is.** A derived number
that merely resembles the recorded one is the defect worth guarding against: a
derivation using a different alpha, a different reference share, or the
observed share instead of the reference share would still emit six plausible
probabilities, the render would still succeed, and the supplement would report
detectability computed two ways in one table.
`tests/test_manuscript_variables.py::TestDerivedPairedPower` closes that by
running the derivation on the arms that *do* carry a recorded power — the
letter, phoneme and G2P arms, six arm-probe pairs — and requiring it to
reproduce the recorder's own values, both as floats and as rendered text. That
agreement is the whole warrant for trusting the three arms where no recorded
value exists to check against. The derivation wraps `paired_power` rather than
reimplementing it for the same reason: a reimplementation is precisely what
the equivalence test would have to catch, and catching it later is worse than
not writing it.

The block carries its own non-vacuity and its own expiry. A stub returning a
constant would satisfy any range check, so the derivation is required to
distinguish pair counts and to buy power with more of them; the token wiring is
checked by moving one recorded count and requiring exactly one token to move;
and a test asserts that these three arms still have *no* recorded power claim,
so if a recorder ever starts writing `bench_letter_ctx_power_*` the derived
token is deleted rather than left to shadow the measurement. A fractional
count raises instead of being truncated, because a rate label read where a
count label was meant would otherwise return a power for a pair count nobody
measured.

**What the six figures say, stated precisely.** Power here is evaluated at
the fixed reference share claim:bench_power_reference_share and alpha
claim:bench_power_alpha, never at the observed effect — power at the observed
effect is a monotone transform of the p-value (Hoenig & Heisey 2001) and adds
nothing. All six sit far below any conventional adequacy level. The reading
that follows is therefore about nulls: for the prompt-format and
context-window arms, which returned no detectable difference, the contrast is
**unresolved rather than absent**, and the text must say so. The alphabetic
arm is the case that needs care in the other direction: its power at the
reference share is equally low, but its recorded exact p
(claim:bench_letter_az_paired_p_gemma3,
claim:bench_letter_az_paired_p_qwen25)
is decisive against the corrected threshold, because the observed discordant
split is nowhere near the reference share. Low power at a small reference
effect does not retract an observed large one; it only forbids reading a null
as a null.

**Falsifier.** A recorded and a derived power that disagree on any of the six
control pairs refutes the derivation outright — and the correct response is
not to loosen the tolerance but to conclude that the recorder computes power
from something other than `b` and `c`, in which case deriving it from `b` and
`c` is the wrong derivation. A derived token outside the open unit interval, a
derived token colliding with a ledger label, a token that fails to move when
its own counts move, or one of these three arms acquiring a recorded power
claim — each fails a named test in `TestDerivedPairedPower`.

## 34 — GNN conformance binds all eight shipped specs, default-deny

**Context.** `tests/test_gnn_conformance.py` is the third independent guard on
AILM's wiring orientation: it checks a live pymdp agent against the semantic
roles the spec declares, and its positive controls include the historical
anti-informative A/B swap. It bound three specs — `L1_letter`,
`L2a_phoneme_sequence` and `L2b_grapheme_to_phoneme` — while eight ship under
`gnn/models/`. The gap was invisible because the file enumerated the specs it
covered instead of the directory, so the five uncovered levels were not
excluded on purpose; nobody had to decide anything for them to go unchecked.

**Decision.** Bind all eight against live agents built from the real vendored
artifacts, and make coverage default-deny: `TestEveryShippedSpecIsCovered`
reads `gnn/models/` off disk and requires the covered set to equal what is
there minus `UNCHECKED_SPECS`, an exclusion map that is currently empty and
whose entries must name the failing slot rather than merely admit a failure. A
spec added later cannot go unchecked by being forgotten. A floor on the
discovered count keeps a scan that returned nothing from satisfying every set
comparison, and the multi-factor level `L3_word` gets its own parallel
assertions rather than loosening the single-factor ones.

**What extending the coverage surfaced.** Four checks turned out to be
narrower than they read, and all four were invisible while only three specs
ran.

- **An `artifact=` may name a builder, not only a path.** The provenance check
  resolved every cited artifact as a repository file. That held for the three
  covered specs because all three cite vendored JSON. Five citations across
  four of the remaining specs name a bare symbol instead — the structural
  forms cite the function that builds them — and each would have failed an
  `is_file()` assertion. The check now resolves both shapes: a value
  containing a `/` must be a file, and a bare symbol must be defined in the
  generator the same entry names. The second is the stronger of the two, since
  renaming that function breaks it, and a positive control asserts it can fail.
- **The two readers disagree on one variable name.** With
  `L1g_letter_generation` in scope, the upstream GeneralizedNotationNotation
  parser normalizes the ASCII identifier `pi` to `π`. That is a notation
  difference, not a disagreement about the model, so it is recorded in
  `UPSTREAM_NAME_ALIASES` and applied when the two readers are compared — and
  then pinned, because an alias is exactly where a real disagreement could
  hide. A separate test requires the rename set to be `π` on that one spec and
  empty everywhere else, so an unexercised alias must be deleted rather than
  left behind.
- **A declared policy prior `E` was going unverified.** `describe_shapes`
  extracts A, B, C and D only, so `check_conformance` never reads `E`, and
  `L1g_letter_generation` is the one shipped model that declares one. Its whole
  argument for annotating the Brown bigram as `PolicyPrior` instead of
  `TransitionMatrix` — entry 20 — is that the bigram is installed into `E` per
  generation step while the construction-time `E` is uniform. Unchecked, that
  distinction rested on prose. `policy_prior_violations` now checks the
  declared `E` against the live agent, refuses an unrecognized declared form
  rather than passing it, and is held non-vacuous by a test asserting that
  exactly one shipped spec declares an `E` at all.
- **The upstream agreement test compared four hardcoded keys.** It checked
  `num_hidden_states`, `num_obs`, `num_factors` and `num_modalities`, which
  happened to be every parameter the three covered specs declare. `L3_word` has
  no scalar `num_hidden_states` and `L1g_letter_generation` adds `num_controls`
  and `gamma`, so a fixed list would silently stop covering whatever a spec
  declares next. It now iterates every `ModelParameters` key AILM reads, with a
  floor so the loop cannot go vacuous.

**Scope, unchanged.** Conformance pins structure and declared roles. It does
not verify a learned probability, and no assertion in that file should ever be
cited as evidence that a number is right.

**Falsifier.** A spec on disk that is neither covered nor named in
`UNCHECKED_SPECS`, a name in `SPEC_NAMES` that is not on disk, an exclusion for
a spec that no longer ships, or a covered spec with no live agent — each fails
`TestEveryShippedSpecIsCovered`. Beyond coverage: a `Learned` form whose cited
generator or builder no longer resolves, a live `E` that is not uniform at
construction, an upstream rename other than the recorded one, or an upstream
parameter value disagreeing with AILM's on any key either reader sees. The
upstream tests skip cleanly when no checkout is present, so "GNN compatible"
is demonstrated where one exists and claimed nowhere else.

## 35 — Clearing and rewriting a ledger namespace is one atomic write

**Context.** A recorder that owns a label namespace has to delete the labels
its previous run produced and this run does not, then write this run's
results. Spelled as `remove_labels` followed by `record_many`, those are two
whole-file replacements with a window between them. A crash or a kill in that
window commits the deletion without the replacement: the ledger loses the
retired labels *and* the new ones, and nothing on disk records that it
happened. The next reader sees a smaller ledger that is internally consistent
and passes every schema check.

**Decision.** `ClaimsLedger.replace_claims` takes the removals and one
`ClaimBatch` per run id, builds a single candidate payload from both halves,
and hands it to one atomic write. Any other process observes the file as
entirely pre-run or entirely post-run. Removals apply before batch entries, so
a label in both ends up recorded with its new value, and every validation —
batch types, per-batch entry checks, two batches claiming one label, a call
that would do nothing — runs before any state changes.
`scripts/record_external_benchmark_claims.py` uses it, which is the recorder
that most needs it: it groups its labels into one run id per model and task,
so its write is many batches and one file replacement rather than one batch
per replacement. `scripts/record_calibration_floor_claims.py` uses it too.

**Now everywhere, and enforced rather than swept.**
`scripts/record_synthetic_benchmark_claims.py` was the last recorder retaining
the split sequence and has been converted, so no recorder in `scripts/` calls
`remove_labels` any more. That is checkable rather than asserted, which is what
turned this entry's original falsifier from a description into a guard: a second
recorder adopting the split sequence was named here as the symptom to watch for,
and a scan for it is cheaper and stricter than watching.

**Falsifier.** `tests/test_claims_ledger.py::TestReplaceClaims` exercises the
method directly: a spy over `_write_atomic` asserts it is called exactly once,
snapshots the file at call time and requires it to still equal the pre-run bytes,
and requires the payload handed in to be the complete post-run ledger. Twelve
invalid-input cases require the file to be byte-identical afterwards, and a
positive control drives the old `remove_labels`-then-`record_many` sequence
through the same spy to exhibit the durable half-applied state it used to leave.
`tests/test_claims_single_writer.py` additionally requires that no recorder
reintroduce the split sequence. The argument no longer rests on reading
`_write_atomic`. It would be refuted by a recorder writing the ledger twice for
one logical replacement, or by any of those cases leaving the file changed. The
test that would have settled it interrupts between the two halves and asserts the
on-disk ledger is unchanged, with a companion showing
that the `remove_labels`-then-`record_many` sequence fails the same
interruption. Until that exists, a second recorder adopting the split sequence,
or a `replace_claims` call that leaves a temporary file behind, are the
observable symptoms that would refute it.

## 36 — PRE-REGISTRATION: what the protocol v4 census is expected to show

**Written before the values exist.** The census was launched and this entry was
written while it was still running, with the ledger frozen at v3 and the new
values not yet on disk. That ordering is the point: a benchmark whose analysis is
decided after its numbers arrive is a benchmark that can always be made to agree
with whatever it produced, and this project has already caught itself reporting a
withheld model as best-calibrated because a completion rule flattered it.

**What changed.** The letter arm is a census of every distinct position in the
fixture rather than a 400-position sample, taking its power against the corrected
threshold from figures near a tenth to figures near one; and each
probe's calibration now carries a direct two-rule reconstruction check beside the
captured-mass gate. Phoneme and G2P stay at their previous size, where they were
already decisive.

**Predictions, each with what would refute it.**

1. *The full-inventory letter arm still favors the comparator, and still for the
   wrong reason.* The gap is compliance, not prediction: two probes emit no
   non-alphabetic symbol at all, and roughly a fifth of positions have one as the
   target. Refuted if the arm reverses, or if the non-alphabetic accuracies move
   off their floors — either would mean the compliance account was wrong.
2. *The alphabetic arm remains the informative one, and the census does not
   change its direction.* A census removes sampling error, not confounding.
   Refuted if a contrast changes sign, which would indicate the 400-position
   sample had been unrepresentative rather than merely imprecise.
3. *Phoneme and G2P are unchanged beyond noise.* Their positions were not
   resampled. Refuted by any movement, which would indicate non-determinism in a
   decode that was measured as exactly reproducible.
4. *The context-window and prompt-format nulls become bounded rather than
   resolved.* Their pair counts grow with the census, so their intervals narrow;
   the registered expectation is that they still contain zero. Refuted if either
   clears the corrected threshold, which would be a real finding and would mean
   the earlier arm was underpowered rather than null.
5. *The direct dual-rule check agrees with the captured-mass gate on which
   systems are reportable, and disagrees with it on why.* The expectation is a
   small divergence for the two probes above the floor and a large one for the
   probe below it. Refuted if a probe below the mass floor shows small rule
   divergence, which would show the proxy withholding a system the direct check
   would admit — the strongest possible argument for replacing the gate, and the
   outcome this entry would most like to be wrong about.

**OUTCOME — prediction 4 is REFUTED.** Recorded here, against the prediction, so
that a reader meets the refutation where the prediction is rather than in a later
entry that frames it favorably. Three of the four contrasts cleared the
corrected threshold: context-window on both probes and prompt-format on Gemma 3
4B. Only prompt-format on Qwen 2.5 3B remained a null. By the terms written above
this "would be a real finding and would mean the earlier arm was underpowered
rather than null", and that is the correct reading: the earlier arm was
underpowered, and this entry's expectation that the intervals would still contain
zero was wrong.

The refutation is more useful than the prediction would have been. It is direct
evidence for the reporting convention of [@sec:bounds] — an unresolved contrast
read as an absent effect is a mistake this project would have made three times in
one section if the earlier nulls had been reported as nulls rather than as
bounds.

**OUTCOME — prediction 1 is qualified.** The full-inventory arm still favors the
comparator and still on compliance grounds, but the phrasing "two probes emit no
non-alphabetic symbol at all" did not survive: over the full pool the weaker
probe is right on a small non-zero share, and the stronger one was already far
off any floor before the census. The compliance account holds; the sentence
describing it was written for a three-probe inventory and was stale before the
census ran.

**What will not be claimed whatever the census returns.** Not competitiveness:
the fairness conditions that fail before the run fail identically after it, since
sample size is not one of them. Not that a null is an absence: a narrower interval
containing zero is a tighter bound and nothing more. Not a trend across the three
probes, which are separate model families and not rungs of a ladder.

**Falsifier for this entry as a whole.** If the analysis performed after the
census differs from the one registered here — a new arm promoted to a headline, a
threshold relaxed, a prediction quietly dropped — then this pre-registration was
decoration. The commit history of this file is the check, and the values landing
after it are what make the ordering verifiable rather than asserted.

## 37 — Two coverage levels are one interval at two defaults; harmonise to one

**The split, stated exactly.** Paired difference bounds are reported at
`DIFFERENCE_CONFIDENCE`, printed through `BOUND_CONFIDENCE_PCT`. Discordant
share intervals are reported at `SHARE_CONFIDENCE`, printed through
`SHARE_CONFIDENCE_PCT`, and that level comes from the recorded
`stat_*_ci99_*` labels rather than from an argument. Both are tokenised, so
neither can drift out of the prose it annotates, and the manuscript can no
longer print a coverage its intervals do not have. What tokenising did not do is
decide anything: it fixed an inherited split in place and made it visible.

**Where the split came from.** `clopper_pearson_interval` defaults to one level.
`paired_difference_interval` *calls* it and defaults to another, passing its own
level down. So one function wraps the other and silently changes the wrapped
default. Two recorders written at different times each took the default in front
of them, and the resulting levels were recorded into label names. Nothing chose
two levels; two levels are what you get when a wrapper disagrees with what it
wraps.

**Decision: they should be harmonised, to one marginal level plus the corrected
one.** Not because two numbers are untidy, but for two reasons that are checkable
rather than aesthetic.

**First — the scale argument does not reach coverage.** The manuscript is right
that the two scales answer different questions: the discordant share `b/(b+c)`
has null one half, the paired difference `(b−c)/n` has null zero, and they are
different estimands. But their *intervals* are not two intervals.
`paired_difference_interval` constructs the difference bound by taking the
Clopper–Pearson interval on the share and mapping it through
`delta = (2·share − 1)·(b+c)/n`. That map is affine in the share with positive
slope, so it is a bijection and it transports coverage exactly: the image of a
share interval at some level is a difference interval at the *same* level, and
never at another. Coverage is a property of the interval, not of the coordinate
the interval is written in. Two estimands therefore license two scales, two
nulls, and two readings — and give no reason at all for two levels. A defense
of the split on "different questions on different scales" would have to point at
some second construction that produces the difference bound independently of the
share interval, and there is none: `ailm.benchmarks` is the single
implementation, and it is the map.

**Second — the levels are assigned to their roles backwards.** The stricter
level lands on the four primary effects in `tbl:multiplicity_family`, where the
interval sits beside a verdict already read at the corrected level and is a
precision statement about an effect that is not in doubt: every one of those
arms has a recorded lower endpoint above one half — claim:stat_stage2_share_ci_low,
claim:stat_stage4_share_ci_low, claim:stat_coupling_harm_share_ci_low,
claim:stat_lexical_share_ci_low. The looser level lands on the nulls and near-nulls
in `tbl:derived_bounds` and `fig:null_bounds`, where the interval *is* the
result — "no effect" is replaced by "at most this large", and nothing else
carries the claim. The equivalence machinery is defined there too:
`minimum_discordant_pairs_for_equivalence` computes how many discordant pairs an
equivalence claim needs *at that level*, so the level directly sets which arms
are allowed to say more than "bounded". The place where coverage is load-bearing
has the looser level and the place where it is nearly decorative has the
stricter one. That is not a defensible division of labour; it is the signature
of a division nobody made.

**Why nothing looks wrong today.** The two levels are applied to disjoint sets of
arms. No arm in the manuscript prints both a share interval and a difference
bound — the lexical rows are two different mixing weights, not one arm on two
scales — so no reader can currently set the two against each other and notice.
That is the argument for fixing it rather than for leaving it. The contradiction
is latent, it surfaces on the first arm reported on both scales, and by
construction it surfaces as an arm that is bounded away from its null on one
scale and consistent with it on the other, in the same document, from the same
counts.

**Which level to converge on, and why it is not obvious.** The project already
carries a third level, `CORRECTED_CONFIDENCE`, and that one *is* chosen: it is
one minus claim:stat_bonferroni_alpha, derived from the declared familywise rate
over a family of claim:stat_family_size tests, and it is the level every verdict
is read at, recorded as the `stat_*_corrected_ci_excludes_half` booleans. The
marginal level is therefore presentational, and the conservative reading is not
lost by loosening it — it already lives at the corrected level, which is wider
than both candidates. Converge on `DIFFERENCE_CONFIDENCE`: it is the level the
equivalence threshold, the derived-bounds table and the null-bounds figure are
all defined and tested at, and it is the level the larger recorded set already
carries, so the cheaper convergence is also the one that leaves the
census-produced benchmark arms untouched. The honest cost of that direction is
that it narrows every recorded share interval, and narrowing a recorded interval
strengthens it; that is acceptable here only because the decision-grade reading
moves nowhere, and it would not be acceptable if any verdict were read at the
marginal level.

**What it would cost, which is why this entry decides and does not execute.**
Harmonising changes recorded numbers, and recorded numbers are the ledger's to
change. Ten labels move: the `ci99` low and high endpoints for the `stage2`,
`stage4`, `coupling_harm` and `lexical` arms, written by
`scripts/record_statistical_claims.py`, plus the `stage5_auc` pair from
`scripts/record_stage5_claims.py`. The names encode the level, so the labels are
renamed as well as re-valued, and the single-writer map in
`tests/test_claims_single_writer.py`, `ailm.manuscript_variables.REQUIRED_CLAIMS`
and every citing site move with them.
`SHARE_CONFIDENCE` and `SHARE_CONFIDENCE_PCT` are then deleted rather than
retargeted, because a token that always equals another token is a second name for
one convention and will drift back apart the first time someone edits one of them.

Two complications belong in the estimate rather than in the surprise afterwards.
The Stage 5 pair is not a Clopper–Pearson share interval at all — it is a
bootstrap percentile interval on an AUC — so the `ci99` suffix currently names a
coverage shared across two different constructions on two different quantities.
Harmonising the coverage does not harmonise the construction, and leaving that
pair behind would strand a label named `ci99` beside labels named `ci95`. And
all of this is a re-record: it needs the single-writer lock on
`data/claims.json`, which the benchmark census holds. The work is blocked, the
direction is not, and the split is declared here so that it is carried
deliberately in the meantime instead of silently.

**Falsifier.** Take the Stage 2 arm, whose share interval is recorded and whose
counts are recorded: map claim:stat_stage2_share_ci_low and
claim:stat_stage2_share_ci_high through `delta = (2·share − 1)·(b+c)/n` with
`b` = claim:stage2_g2p_mcnemar_b, `c` = claim:stage2_g2p_mcnemar_c and
`n` = claim:stage2_g2p_eval_positions. If the result equals what
`paired_difference_interval` returns for those counts at its own default, then
the two levels are two independent constructions, the scale argument reaches
coverage after all, and this entry is wrong. It does not equal it: the mapped
interval is the bound at `SHARE_CONFIDENCE` and is strictly wider than the one at
`DIFFERENCE_CONFIDENCE`, at every arm where both can be computed, which is what
makes them one interval at two defaults. The second thing that would refute this
entry is a decision rather than an identity: an arm reported at the marginal
level whose verdict is read there rather than at `CORRECTED_CONFIDENCE`, which
would make the marginal level load-bearing and make loosening it a real loss
instead of a presentational one.

## 38 — PRE-REGISTRATION: what stays in the main line, and what is archived

**Written before the census lands**, for the same reason entry 36 was: a rule for
deciding which experiments belong in the paper is worth nothing if it is chosen
after seeing which experiments it would remove.

**The problem.** A paper that reports every measurement it took, including the
ones that could not have answered their own question, reads as a review of the
authors' learning rather than as a result. The manuscript has accumulated arms of
exactly that kind.

**The criterion, and it is already computed.** An arm belongs in the main line if
it could have resolved its question *at any outcome*. Operationally: an arm whose
discordant-pair count falls below the minimum needed for an equivalence claim at
the declared coverage could never have supported a conclusion in either
direction, whatever the data had shown. That minimum is
anchored on the declared reference share claim:stat_reference_share and is drawn
on [@fig:null_bounds] as a shaded region; the figure exists precisely because a bounded null and an
unresolvable one look identical in a table.

**Why this is not applied yet.** The two arms that fail the criterion at the
current sample size — the letter context-window sweep and the prompt-format
sensitivity arm — are driven by `letter_n`, the same parameter the running census
raised from a 400-position sample to the whole deduplicated pool. Both are
therefore being measured at roughly twenty times their previous size right now.
Projected forward from their recorded discordance rates, both land above the
threshold rather than below it. Archiving them today would delete the arms the
running experiment is in the middle of making informative, which is the opposite
of the intent.

**The rule, then, in order.** When the census lands, recompute the criterion for
every arm. An arm that clears it stays in the main line and is reported as a
result. An arm that still fails it is archived: removed from the manuscript
entirely, not demoted to a caveat, with its recorded claims retired from the
ledger by their owning recorder and its outcome preserved here, in this file,
which is the record of what was tried and what it showed. The paper keeps
registered negative results, which are findings; it does not keep
underpowered arms, which are not.

**Falsifier.** If an arm is retained after failing the criterion, or archived
without its outcome being recorded here, this entry was decoration. The check is
mechanical: every arm reported in [@sec:external_benchmarks] should have a
discordant-pair count at or above the drawn minimum, and every arm archived
should appear in this file with the count that disqualified it.

## 39 — A retired probe leaves the registry, not a named exclusion

**Context.** The external reference frame served three probes and now serves two:
`gemma3:4b` and `qwen2.5:3b`. The retirement is not a bookkeeping change. The
declared correction family is six contrast types on each probe, so its size
scales with the probe count; dropping one loosens the Bonferroni threshold every
interpretive guard in the supplement reads p-values against. It also emptied the
captured-mass gate's reject side, because the retired probe was the only system
that floor ever excluded — which is why [@sec:external_benchmarks] now reports
that the gate does not discriminate in this run rather than that it passed.

**The trap this entry exists to name.** The obvious way to enforce a retirement
is to forbid the name: assert that no ledger tag contains the retired probe's
key, that the supplement contains no occurrence of its display name. Three such
guards were written and all three were wrong in the same way. A guard that spells out the model it excludes
reintroduces the text it exists to remove, so the name survives its own deletion
in the one place a reader is most likely to trust. Worse, it only ever catches
that model: a fourth probe swapped in under a different key passes every one of
them, and every interpretive threshold in the section silently changes meaning.
The same shape as counting instead of comparing — a size check is satisfied by
any two members.

**Decision.** State the rule over the registry, never over a retired member.
`tests/test_benchmarks.py` asserts the model registry by exact set equality
rather than by size. The supplement guard extracts every model-name-shaped token
in the section and requires each to be one the registry serves, so it fails on
any unregistered probe rather than on one named string. Count wordings are
generated from the registry's length, so a probe added or retired needs no edit
to the guard that would otherwise go stale. Prose that carried the count — the
derived-power block's "nine figures", the supplement's probe tallies — is
restated over `len(PROBES)` or over the arms themselves.

**What is deliberately kept.** `handoff/claims.*.json` are pre-run ledger
snapshots and still carry the retired labels; they are the diff target that
proves what the census replaced, and rewriting provenance to match the present
would destroy the only evidence the transition happened. This file keeps the
retired probe's outcome for the same reason entry 38 requires it — the record of
what was tried is not legacy, it is the negative result.

**What the restatement then found.** Generating the expected label set instead
of listing it surfaced a drift that had been invisible: the multiplicity
exclusion table carried a documented reason for the alphabetic and prompt-format
arms on one probe and nothing at all for the other. Both p-values were therefore
corrected by no procedure, and every existing check stayed green — because
closure was tested against the ledger, and the ledger was missing the same two
rows. A hand-maintained list compared against another hand-maintained list agrees
on the entries neither has.

The fix is the same shape as the rest of this entry. Closure is now stated over
the *recorders*: each declares the probes it serves and the family templates it
writes, and their product is every bench p-value the repository can produce, each
of which must be in the corrected family or carry a documented exclusion before
it is measured rather than after. The rule remains producer-owned rather than
inferred from whatever rows happen to be in the ledger.

**Correction (2026-08-24).** The one-probe synthetic exception described by the
original decision is no longer the live contract. The in-context recorder
authenticates a zero-shot synthetic row for every registered probe, so a
one-probe producer made the network phase impossible to complete. The synthetic
producer now derives the complete shared registry, measures all probes or none,
and corrects one paired contrast per registered probe. The source-expanded
closure rule above caught the missing Qwen row before a partial receipt could be
published; tests now also assert exact producer/consumer registry equality.

**Falsifier.** A guard here that names a specific model is this entry undone: it
will pass on the day a different unregistered probe appears in the section. The
check is mechanical — the registry-derived guards must fail when a foreign model
name is inserted into the supplement, and `tests/test_benchmark_interpretations.py`
carries that positive control. Separately, if the ledger ever holds a paired
p-value outside six contrasts on each served probe, the declared family and the
recorded family have diverged and every threshold in the section is being read
against the wrong denominator.

## 40 — Resumability belongs at the model call, not at the model

**Written before the run that depends on it**, following entries 36 and 38: a
rule about how a measurement may be reused is worth nothing if it is chosen after
seeing which reuse it would permit.

**Context.** The protocol v4 census was halted at 3h15m and lost all of it. The
recorder did have a resume mechanism — a per-model sidecar, written after that
model's every arm completed — and it did not help, because with the letter arm a
full-pool census the first model *is* most of the run. The checkpoint boundary
and the failure boundary were the same place. That is the general defect, and it
is not about benchmarks: a checkpoint between top-level units protects nothing
when one unit's cost dwarfs the others, and it looks like protection the whole
time it is failing to provide any.

**Correction to the original purity premise.** The first version of this entry
called a temperature-zero, fixed-seed request a pure function of its arguments.
That is stronger than the serving behavior supports. The harness later measured
that a model load or an intervening workload can change server state, and
answer-slot log probabilities drift even when the greedy answer reproduces. A
cache record is therefore an exact snapshot of one observed reply under an
authenticated request and model identity, not proof that the live server is
history-independent.

**The seam.** Every cacheable model call passes through
`ailm.benchmarks.ollama._post_generate`, so the snapshot store goes there and
resume stops being a recorder-level code path. Rerunning an identical command
replays the observed replies already on disk and measures only missing request
keys. The `ask_ollama*` helpers differ in what they read out of a reply, not in
how they obtain one, so one wrapper covers every cacheable arm; a test asserts
they still share it, because a helper that obtained its own reply would silently
opt that arm out of resumability.

**State is controlled rather than wished away.** `benchmark_model` discards the
first call after a model load or different workload. `repeat_agreement` then
makes adjacent uncached pairs under matched server state, and the release audit
re-queries a content-bound stratified sample of stored calls, requiring exact
answer agreement and bounded distributional drift. No auxiliary warmer or other
Ollama-backed task may run concurrently with those controls: a file lock protects
the append-only store from interleaved bytes, not the model server from an
intervening workload.

**Identity is the digest, never the tag.** `gemma3:4b` is a mutable pointer, and
the same tag addresses a different build after a pull. Keying a cache on it is
the same name-versus-identity confusion as deciding ownership by substring match,
and it fails in the direction hardest to notice: a re-pulled model is precisely
when fresh measurements matter most, and precisely when a tag-keyed store would
answer from the old build. The key is the content-bound build digest from
`/api/show`, alongside the payload, the endpoint and the protocol version. A
direct declared digest is preferred. When the server omits it, the stable
manifest is hashed only if the resolved Modelfile's `FROM` line contains a
validated SHA-256 content blob; license, template, capability, tensor, or model
metadata alone never become an identity.

**An unknown identity degrades to no caching, not to caching under a guess.**
When the digest cannot be read, the call yields no key; a keyless call is
measured and not stored, and the omission is counted so it appears in the
receipt rather than looking like a call that never happened. The tempting
fallback is to key on the tag in that case, which is why `model_digest` does not
offer it.

**The receipt separates hits from misses, and that is not bookkeeping.** A run
that replayed every call and a run that measured every call emit byte-identical
claims. The difference is real, it is invisible in the ledger, and it is exactly
the difference between evidence and a file read. So the counts are recorded as
claims rather than left in a gitignored artifact, and the manuscript states what
fraction of the reported numbers the run actually measured.

**A crash can only cost the last incomplete append.** The store is append-only
and each completed record is one fsynced line, so an interruption may leave an
unterminated malformed final fragment while every earlier record remains intact.
The loader recovers only that shape and truncates it before the next append. A
malformed interior line, a terminated malformed final line, an unsupported
schema, or an invalid complete record rejects the store. Broadly skipping
unparsable lines would turn corruption into a plausible-looking partial cache
and is not the contract.

**The audit, and why the store needs one.** A resumed run otherwise proves only
that a file was read back. `--verify-cache N` replays recorded payloads against
the live server and requires exact agreement on the answer string plus the
declared log-space tolerance on shared answer-slot candidates. That is the
experiment that can falsify the belief that the key and controlled server state
name everything relevant to the recorded result. It is wired into the full
release profile and run as a recorder precondition, so claims cannot be written
on a store the server no longer agrees with. It is deliberately not gated in
GitHub CI, which has no model server; a check that can only be skipped there is
not a check.

**Falsifier.** Corrupting a stored reply must make the audit exit non-zero, and
breaking the shared seam must fail the seam test — both are positive controls
with measured outcomes, not assertions. A key component that does not change the
key when it changes is decoration, and each is tested for separation
individually. If a resumed run ever produces a different result from an
uninterrupted one on the same inputs, the key is incomplete and the store must be
discarded rather than the difference explained.

**Measured before this entry was believed.** 100 real `gemma3:4b` calls, six
phases: cold 100 measured in 72.8 s; replayed in a new process 100/0 in 0.025 s
with an identical answer hash; SIGKILL at 28 s then resume gave 28 replayed and
72 measured with that same hash; a changed seed reused nothing; a disabled store
read nothing; and 15 of 15 replayed calls matched the live server byte for byte.


## 41 — Conformance checks the configuration, not only the matrices

**Context.** GNN conformance compared every declared slot against the live pymdp
agent — shapes, structural forms, the policy prior, the precision — and reported
clean. Entry 34 closed the last gap of that kind by making `E` and `gamma` travel
with the description. Auditing what `describe_shapes` still did *not* read turned
up a different kind of claim it could not see at all.

**The finding.** An agent can carry a correct non-flat `C`, a correct `E`, and a
genuinely controllable `B`, and still not minimize expected free energy over any
of them, because whether the preference and epistemic terms enter the policy
score is a flag on the agent rather than a value in any slot. With
`use_utility=False` the preference vector is inert: no shape changes, no entry
changes, conformance passes, and the manuscript's claim that generation selects
by minimizing expected free energy over a non-flat preference is false. The same
blindness covered `use_states_info_gain` (without it, policy selection is utility
maximization, not EFE minimization), the agent's own `num_controls` (the shape
reduction drops trailing size-one axes and its docstring already conceded it does
not compare the control count), and `policy_len` with the enumerated policy count
— `E = Uniform(30)` is coherent only while a policy is one action deep, since at
depth two the space is 900 and a 30-entry prior is a different claim wearing the
same shape.

**Decision.** `describe_shapes` carries the inference configuration alongside the
arrays, and `_check_inference_configuration` runs before the factor dispatch so a
factorized model cannot escape it by gaining a factor. The checks apply only to a
spec that declares itself controlled (`num_controls > 1`). That restriction is
the point rather than a weakening: every comprehension level runs with a flat `C`
and a single action *by design*, and that design is exactly what the paper claims
about them, so a rule demanding a preference term of every model would fail all
six for conforming as intended — which is how a guard becomes something someone
deletes.

**Falsifier.** Four named mutants, one per way the configuration can lie:
`use_utility` off, `use_states_info_gain` off, a control count that disagrees with
the spec, and a policy prior sized against a policy space the agent did not
build. Each must produce a violation naming its own failure. A fifth test guards
the guard by asserting `describe_shapes` still reports every field the mutants
flip, since a check cannot fire on a field the description stopped carrying, and
a sixth asserts an uncontrolled level is *not* asked for a preference term.

## 42 — PRE-REGISTRATION: Fillmore's subject hierarchy as an external check on L4

**Written before the design is changed**, and specifically before the change that
would make the arm resolvable, because a rule chosen after seeing that an arm
returned nothing is not a rule.

**The idea.** Fillmore's case grammar puts case relations in the base as
primitives and treats "subject" as a surface notion selected from them: if there
is an agentive it becomes the subject, otherwise the instrumental, otherwise the
objective. That hierarchy is a prediction, and it is a prediction about a label
this project's L4 model is forbidden to observe. `deprel` is in
`FORBIDDEN_OBSERVATIONS`, so scoring the model's UPOS-only case assignment
against which token bears `nsubj` measures something the model was never given —
the first genuinely external validity check this level has had.

**The circularity that had to be ruled out first.** The *derived* assignment
cannot serve. `UD_DEPREL_TO_CASE` maps `nsubj` to NOM, so on that path the token
bearing `nsubj` is by construction the token bearing NOM and the hierarchy's
first clause selects it every time. This is the same trap the level has already
sprung once, when the case oracle consulted `feats[Case]` first and returned the
gold label on exactly the tokens being compared. The derived path is therefore
kept and reported as a control that is *expected* to be near-perfect, not as a
result. It is not perfectly circular, because `nsubj:pass` maps to ACC, and the
size of that gap is itself informative about how much of the mapping is
relabeling.

**What was measured, and why it is not reported as a finding.** Implemented as
`fillmore_subject` and `subject_prediction_accuracy` with thirteen tests. Run
over UD EWT dev, the derived control behaves as predicted and a leftmost-role
positional baseline is well below it. The model path is a different matter: the
L4 artifact's UPOS alphabet is smaller than UD's full tag set, and the first
scoring rule dropped any sentence containing an out-of-alphabet tag. That left
nine scorable sentences. Nine is not a small sample, it is no sample — by
entry 38's criterion the arm could not have resolved its question at any
outcome, so its accuracy is not recorded here and must not reach the manuscript.

**The decision, stated before the re-run.** The drop rule is the defect, not the
result. Score per *token* over the sentences whose subject token and candidate
set are entirely in the alphabet, rather than discarding a whole sentence for one
unseen tag elsewhere in it. That change is registered now, along with what will
count: the arm enters the main line only if its scorable-sentence count clears
the minimum an equivalence claim needs at the declared coverage, and only if the
model path is compared against both the positional baseline and the derived
control in the same table. Any other rescue of the sample size — widening the
alphabet, relaxing the single-subject restriction, pooling with a second corpus —
is a different experiment and needs its own entry.

**Falsifier.** If the model path scores near the derived control, the observation
alphabet has leaked and the result is the old circularity with more steps; the
response is to find the leak, not to report the number. If it scores at or below
the positional baseline, the level's case assignment carries nothing that
transfers to an unobserved grammatical relation, and that is a real negative
result about this level worth reporting as one.

## 43 — Entry 38 applied, and the criterion corrected in the applying

**The census landed, so the informativeness criterion of entry 38 was recomputed
for every benchmark arm.** Applied literally, it archives the phoneme and G2P
arms — the two arms whose contrasts are the most decisively significant in the
section. That result is absurd, and the absurdity is in the criterion rather than
in the arms.

**What entry 38 got wrong.** It said an arm belongs in the main line if it could
have resolved its question *at any outcome*, and operationalised that as the
discordant-pair count reaching the minimum an equivalence claim needs. Those are
not the same test. The equivalence minimum is what a *null* needs in order to
mean anything; an arm that observes a large effect has resolved its question in
the other direction, on far fewer pairs, and asking it to also have been capable
of a bounded null is asking it to have been a different experiment. The phoneme
and G2P arms carry 51 to 82 discordant pairs and p-values that clear the
corrected threshold decisively; they resolved.

**The corrected criterion.** The equivalence minimum applies to an arm *whose
result is a null*. An arm reporting a significant effect is retained on that
effect. An arm reporting a null with discordant pairs at or above the minimum
reports a bounded null, which is a finding. An arm reporting a null below the
minimum reports an unresolvable one.

**On the disposition of an unresolvable null, and a correction to the first
version of this entry.** As first written, this entry said such an arm is "a
candidate for archiving only if nothing else in the section depends on it". That
was a second change, smuggled in on the first one's justification, and it was
wrong twice over. Entry 38 had said explicitly that a failing arm is "removed
from the manuscript entirely, *not demoted to a caveat*", and the replacement is
precisely the caveat-demotion it forbade by name. It was also the only part of
this entry that altered a real outcome: the statistical correction above rescued
none of entry 38's own nominees, because those arms cleared on the census alone,
while the disposition change is what retained the one arm that did not.

The rule is therefore restored to entry 38's, with one clarification entry 38
did not need to make and this one does. An arm reporting an unresolvable null is
**not archived** when it is a *control* rather than a result: the prompt-format
contrast on Qwen 2.5 3B exists to bound how much of a probe's letter accuracy is
an artifact of prompt wording, and deleting it would remove the only evidence
about that threat on that probe while leaving the threat in place. It is
retained, labeled unresolvable, and it makes no claim. An arm reporting an
unresolvable null as a *result* is archived, as entry 38 said.

That distinction is a real change to entry 38 and is argued here rather than
assumed. Its cost is that "control" is now a load-bearing word: an author who
wants to keep a failing arm can call it a control. The check against that is
mechanical and stated in entry 38's own terms — a control must be named as one
in the pre-registration that introduced the arm, not after its result is known.
The prompt-format arm qualifies because entry 36 registered it as a protocol
sensitivity diagnostic, "a separate protocol-sensitivity question, not a
model-versus-comparator hypothesis", before the census ran.

**Applied, at the census.** No arm is archived. The two arms entry 38 nominated —
the letter context-window sweep and the prompt-format sensitivity arm — were
nominated because at 400 sampled positions they produced nulls on 18 to 51
discordant pairs. Over the full pool they carry 335 to 752, and three of their
four contrasts now clear the corrected threshold outright, which is the outcome
entry 38 predicted and the reason it declined to archive them early. Exactly one
contrast remains a null: prompt-format on Qwen 2.5 3B, at 335 discordant pairs
against a minimum of 420. It is an unresolvable null, it is labeled as one in
[@sec:external_benchmarks], and it stays.

**Falsifier, replacing entry 38's.** Entry 38's check — every reported arm should
have a discordant count at or above the drawn minimum — is withdrawn, because it
fails on arms that resolved. The replacement: every arm reporting a *null* must
either carry discordant pairs at or above the minimum, or be described in the
text as unresolved rather than as absent. An arm reporting a significant effect
is exempt, and an arm claiming a bounded null on fewer pairs than the minimum is
the defect this now catches.

## 44 — The red-team findings, and what they cost

An adversarial pass over the landed census, the new related-work section, the
call store, entry 43 and the scaling ladder returned five findings that survived
independent verification. They are recorded here in full, including the two that
were mine and were wrong, because a red team whose findings are absorbed silently
produces a paper that looks like it was always right.

**1. A registered prediction was refuted and booked as confirmed.** Entry 36's
fourth prediction states that the context-window and prompt-format nulls "become
bounded rather than resolved", *"Refuted if either clears the corrected
threshold"*. Three of the four contrasts cleared it. Entry 43 then described that
outcome as "the outcome entry 38 predicted", pointing at a different entry's
different threshold. The refutation is now recorded against entry 36 itself,
where a reader meets the prediction, rather than in a later entry that frames it
favorably.

**2. The determinism probe could not fail.** `repeat_agreement` issued the same
request twice through the memoized path, so the second call was answered from the
store and agreement was analytically 1.0 whenever the cache was on — while the
supplement cited exactly those values as evidence that the server reproduces its
replies. Verified by counting server calls: two identical requests produced one.
The probe was first repaired by calling the uncached poster directly on both
legs. Receipt hardening subsequently preserved that freshness while storing the
two replies under distinct cache-purpose protocols: neither trial can satisfy
the other, and both can now be replayed to rederive the reliability claim. A
measurement whose value is forced by the mechanism it measures is worse than no
measurement, because it reads as a passing check.

**3. The scaling ladder's stated mechanism was false and its headline was a
denominator artifact.** The prose said the comparator is "re-fit at every rung";
it is estimated once from the full training frequencies and re-scored, and its
matrices are bit-identical across rungs. Separately, per-emission gain is an
average over each rung's surviving positions, and a smaller lexicon survives on a
more frequent and easier subset, so the "nearly flat, non-monotone" gain is a
property of the denominator. On the shared denominator the coverage axis already
uses, the gain rises strictly and by more than seventy percent. Both series are
now recorded and both appear in the text and the figure title. The old falsifier —
that the two curves must not cross — was unfirable on nested frequency-ranked
prefixes of a Zipfian corpus, and is replaced by one that can fail.

**4. A resolved contrast was described as a bound in one section and as
established in another.** [@sec:verification] listed both context-window arms
among the results that "sit below" the equivalence minimum and are therefore "a
bound and nothing more", while their pair counts are above it and both contrasts
clear the corrected threshold. The figure had already refused to draw them; the
verification section had not been told. The table retains them because its
membership rule is provenance rather than verdict, and the text now says so.

**5. Two scholarly overclaims in the new related-work material.** "Twenty years
earlier" understated the gap to Reich by four decades. And the claim that the
roles carrying no English morphology are Fillmore deep cases "in exactly that
sense" is true of three of them; ablative is not in his inventory and vocative
cannot be a deep case by his definition, since it is not a relation between a
nominal and the predicate. The lineage is now stated as partial with the
exceptions named. A direct quotation from an unpaginated technical report was
also replaced by paraphrase, matching the bibliography's own rule against
locators it cannot supply.

**What did not survive the attack.** Two charges were checked and rejected. The
multiplicity loosening from retiring a probe changed no verdict: the interval
between the old and new thresholds is empty, so no contrast is resolved only
because the family shrank. And the correction in entry 43 rescued none of entry
38's own nominees — those arms cleared on the census alone — which is the
opposite signature from a threshold moved to save a case.

**Falsifier.** Each finding above is mechanical. The determinism probe must make
two server calls per position; the ladder must record both gain series and its
monotone one must rise; no section may call an arm above the equivalence minimum
a bound; and entry 36 must carry its own refutation. If a later revision removes
any of these without a superseding entry, this one was decoration.

## 45 — Two multiplicity families, corrected at two levels, on purpose

**The defect.** The verification section was titled "One family, one corrected
threshold" while the project declared two: the repository-wide family of
`stat_family_size` tests at familywise 0.01, and the external benchmark's family
of `bench_family_size` contrasts at familywise 0.05. The benchmark's per-test bar
is about five times looser. Neither declaration referenced the other, so a reader
could not tell whether a given p-value had cleared a strict bar or a lenient one.

**Decision.** Keep both levels and argue them, rather than harmonising to one.
The two families answer different questions. The repository-wide family holds the
hypotheses this paper's own conclusions rest on, where a false positive is a false
claim about AILM, so it is strict. The benchmark family is an external reference
frame the paper explicitly declines to draw conclusions from, and whose own
summary states it does not certify competitiveness; a false positive there
mis-describes a probe. Harmonising to the strict level would impose the cost of a
claim on a description, and harmonising to the loose one would weaken every claim
the paper actually makes.

**Two rules follow, both now stated in the manuscript.** No test belongs to both
families — the benchmark contrasts are excluded from `SIGNIFICANCE_FAMILY` by
name with a recorded reason, so nothing is corrected twice and nothing escapes
correction. And no benchmark result may be promoted into a claim about AILM
without being re-judged at the repository level, because the threshold it cleared
is not the threshold this paper's claims are held to.

**Falsifier.** A label appearing in both families, or in neither, refutes the
first rule; `tests/test_statistics.py` already scans for the second case and now
has a reason to. A sentence anywhere in the manuscript that cites a benchmark
p-value as evidence for a claim about AILM refutes the second, and that is a
reading check rather than a mechanical one, which is why the rule is written
where the families are declared rather than left to a test.

## 46 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the channel factorial, and what it cannot de-confound

**Provenance correction (2026-08-23).** The rule was written before the executor
ran its four cells within the development session, but this entry and the
executor first appear in the same Git commit. That ordering is not independently
timestamped. The design is therefore a prospective in-session specification,
not a preregistration and not confirmatory evidence; the exact Git-bound
classification is in `data/registration_provenance.json`.

**The defect this repairs is one the paper already admits.** [@sec:hierarchy]
states that the factorial version of the channel experiment "crosses message
source with insertion point and reports all four cells", and then that "the two
cells that would separate source from insertion point are not built". Decisions
18 and 19 — the novel-information criterion, one of this paper's two
methodological contributions — therefore rest on a diagonal. The L1→L2B coupling
puts a *letter* message at the *evidence* site and harms; the L3→L2B lexical
channel puts a *lexicon* message at the *predictive* site and helps. Source and
site move together, so the recorded 2.27-point gap between them is unattributed.

**The design.** Four cells: {letter, lexicon} source × {evidence, predictive}
site, one executor dispatching into the shipped primitives and reimplementing
none of them, on the same receiver, weight grid and evaluation positions.

**Three differences the factorial does NOT remove, specified before the cells so they cannot
be discovered later as caveats.** First, the mixing weight is not a common dose
across sites: the evidence vector's total mass varies by a factor of about
twenty-five across letters, so a declared weight is a different effective weight
depending on which letter was observed, and the executor must scale-match and
report both the raw and the effective sweep. Second, the predictive site
compounds along the word and the evidence site does not, so the two sites are not
two levels of one factor. Third, source is collinear with conditioning window:
the letter message is exclusive of the current letter by construction and the
lexical message is inclusive of it plus absolute position. **The factorial
de-confounds site, and only site.**

**AMENDMENT, made during implementation and before any cell was measured: the two
off-diagonal cells need an alphabet transport that neither diagonal cell needs.**
This is specified rather than patched in silently, because discovering it
mid-measurement and fixing it quietly is how a factorial stops being one.

The two sites do not consume the same kind of object. The evidence site takes a
distribution over the 30 **letters** and marginalizes the emission over it; the
predictive site takes a distribution over the 39 **phones** and mixes it into the
phone predictive. The letter source produces letters and the lexical source
produces phones, so on the diagonal each message already lives in the alphabet its
site consumes. Both off-diagonal cells must move a message between alphabets, and
that transport is a modeling choice present in neither recorded cell.

The two directions are not symmetric, and the asymmetry is the finding. Phone to
letter is clean: the emission is column-stochastic over letters, so `emission @ r`
carries a phone distribution to a proper letter distribution with no new
parameters and no inversion. Letter to phone is not: `emission.T @ q` returns a
*likelihood* over phones, summing to about 1.87 rather than 1, and `coupling.py`
states in its own docstring that normalizing that likelihood "would turn the
evidence into a prior and silently change the model". Taking the easy route there
would have built a cell that answers a different question from the other three.

**Specified choice.** The letter-to-phone transport is the Bayes inversion of the
emission under the model's own start prior over phones,
$P(\text{phone} \mid \text{letter}) \propto P(\text{letter} \mid \text{phone})\,P(\text{phone})$,
which is a proper distribution for every letter (verified: all 30 letters carry
non-zero evidence mass, row sums within 3e-16 of one) and introduces no parameter
the model does not already hold. It is a genuine model choice and it is declared
here rather than defended later.

**What this does to the reading of the factorial.** The site main effect is now
partly a transport effect on two of four cells, so the de-confounding claim
narrows again: the factorial separates site from source *given a declared
transport*, and a different transport is a different experiment. Any cell whose
verdict flips under a plausible alternative transport — the normalized likelihood
being the obvious one — must be reported as transport-sensitive. That test is
cheap and is part of the arm, not a follow-up.

**Controls.** The entropy-preserving axis reversal, with a per-cell directional
rule derived from that cell's own measurement. The shared scramble threshold is
explicitly *not* used: it is unreachable in three of four cells, and in the
coupling cell the recorded scramble already beats the real message, so a shared
gate there would rebuild the defect of decision 16. The uniform-transport
baseline is dropped as uninformative in both directions.

**Specified before scoring: how a significant-but-scramble-explained cell is
reported.** A cell may return a small significant gain whose scrambled control
recovers much of it, meaning most of the effect is regularisation rather than
information. This project has no precedent for that case and it is a likely one.
Such a cell is reported as *regularisation-dominated*, with the scrambled control
in the same sentence as the headline, and it may not be cited as evidence for the
novel-information criterion in either direction.

**Family arithmetic, in the same commit as the measurement.** Two new paired
p-values join the repository-wide family, moving its size from 11 to 13 and the
corrected threshold with it. `tests/test_statistics.py` hardcodes the current
size, and the five current survivors must be re-checked at the new threshold in
that same commit. Decision 45 forbids a label joining both families; these join
the statistical one.

**Deliberately NOT built: the GNN composition layer.** The proposal that produced
this entry also proposed declaring the realization relation as a spec-level
object with new ontology roles. There is no live pymdp object for a channel —
the coupling module never imports pymdp — so conformance would have nothing to
read those roles off, and they would be unchecked declarations in a directory the
spec discovery does not scan. `gnn/README.md` says the layer is a guard rather
than documentation. Adding unguarded roles to it is the weakening this project's
own doctrine forbids. The binding is the bit-identity gate against the shipped
path, which needs no spec.

**OUTCOME — the same message has opposite signs at the two sites.** The executor
passed its gate: it reproduces both shipped cells bit-identically over all claim:channel_factorial_positions
positions, zero disagreements. The completed 2×2, each cell at its best *engaged*
weight against the claim:channel_factorial_anchor_accuracy uncoupled anchor:

| cell               | weight | accuracy | margin | b/c      | p         |
|--------------------|-------:|---------:|-------:|---------:|----------:|
| letter@evidence    |    0.1 |  claim:channel_letter_evidence_best_accuracy | −0.264 |  217/330 | 1.5e-06   |
| letter@predictive  |    0.1 |  claim:channel_letter_predictive_best_accuracy | +0.103 |  502/458 | 0.17      |
| lexicon@evidence   |    0.1 |  claim:channel_lexicon_evidence_best_accuracy | −0.171 |  120/193 | 4.4e-05   |
| lexicon@predictive |    0.3 |  claim:channel_lexicon_predictive_best_accuracy | +claim:lexical_best_margin_pts | 1600/741 | 5.6e-72   |

**A defect in the first pass of this recorder, corrected before anything was
written.** It selected the best weight over the whole grid including zero. Zero
weight *is* the anchor, so for a harmful cell the maximum was the anchor itself:
both evidence cells reported claim:channel_factorial_anchor_accuracy with 0/0 discordant pairs and p = 1, and a
channel that damages the model appeared in the table as doing nothing at all. The
shipped coupling recorder had always selected over positive weights, which is why
it reports claim:channel_letter_evidence_best_accuracy rather than the anchor. Selection is now over engaged weights
only. The first pass would have replaced a significant harm with a null.

**The finding, which is sharper than the interaction framing it replaces.** Both
evidence-site cells *significantly harm*. The lexical message — the one that buys
two points at the predictive site — harms by 0.17 points at the evidence site, at
p = 4.4e-05. Same message, same receiver, same weight grid, opposite sign, and the
only thing that differs is where it enters. The letter message at the predictive
site buys 0.10 points and does not clear significance.

Decomposed: the source effect is +0.09 points at the evidence site and +1.90 at
the predictive site; the site effect is +0.37 for the letter source and +2.18 for
the lexical one; the interaction is 1.81 points and the recorded diagonal gap is
2.27.

**What this does to decisions 18 and 19.** Those entries read the diagonal as
evidence about message *content*: a message carrying information the receiver
lacks helps, one carrying what it already has does not. The factorial says the
insertion point carries at least as much of the effect. A message demonstrably
full of novel information still harms at the evidence site. The criterion is
therefore qualified, not withdrawn: it predicted the L3→L2B pair correctly, and
the factorial now explains that the prediction succeeded at one site and would
have failed at the other.

**QUALIFIED BY DECISION 47, recorded the same day.** Everything above holds for
the *linear* mixing rule, which all four cells share and which this design
therefore could not see. Under geometric pooling the two sites are provably the
same computation — maximum posterior difference 1.7e-16, zero argmax
disagreements over 13,016 position-comparisons — so the site factor measured here
is an artifact of the combination rule rather than a property of where a message
enters. The measurements stand; the conclusion narrows from "the insertion point
carries the effect" to "linear mixing makes the insertion point carry the effect".
Read entry 47 before citing this one.

**Why this is a mechanism rather than a curiosity.** The evidence site mixes a
message into the observation likelihood, which is where the model's only hard
information about the current position lives. Softening it trades certain
evidence for a prior, and the trade is bad regardless of how good the prior is.
The predictive site mixes into the state prior, which is where a prior belongs.
Stated that way the result is unsurprising in hindsight and was not predicted by
anyone here in advance, which is the reason it was worth measuring.

**Family arithmetic, applied.** Two of the four cells are numerically identical to
family members already declared: letter@evidence *is* `coupling_mcnemar_p` and
lexicon@predictive *is* `lexical_best_mcnemar_p`, to every recorded digit, because
those are the shipped experiments the executor was gated against. Listing them
again would count one hypothesis twice and tighten the threshold on a duplicate.
Only the two genuinely new cells join, moving the family from 11 to 13 and the
corrected threshold to 7.7e-04. The survivor count rises from 5 to 6: the
lexicon-at-evidence harm clears the corrected threshold, so a *harm* is now a
familywise-confirmed result.

**Falsifier.** If the four cells do not reproduce the two recorded ones exactly —
per-position hit vectors, not accuracies — the executor is a reimplementation and
the factorial measures it rather than the channels. If all four cells at weight
zero are not byte-identical to each other and to the uncoupled decode, the grid
does not share an origin and no cell is comparable to another. And if the site
main effect turns out to exceed the declared fraction of the recorded gap, the
attribution in decisions 18 and 19 is confounded and those entries must be
qualified rather than this one.

## 47 — The insertion site is an artifact of the combination rule

**A derivation, not a pre-registration.** The sign is provable in two lines and has
now been measured, so writing this as a registered prediction after the fact would
be worse than having none.

**What it says.** Decision 46 crossed message source with insertion site and
concluded that the site carries most of the effect. That 2×2 held one factor
fixed in all four cells: the *combination rule*, which in every shipped channel is
linear mixing, $(1-\alpha)x + \alpha m$. Under the geometric rule — log-linear
pooling, a product of experts, and the rule `pymdp` itself uses when it combines
modalities by summing log-likelihoods — **the two sites are the same computation.**

At the evidence site the posterior is proportional to
$A[l,\cdot] \cdot m^{\alpha} \cdot \text{pred}$; at the predictive site it is
$A[l,\cdot] \cdot (\text{pred} \cdot m^{\alpha})$. Multiplication is associative,
the intermediate normalization cancels, and the posterior is what propagates
forward in both recursions, so the whole trajectory is identical. Measured on the
shipped matrices: maximum posterior difference $1.7\times10^{-16}$ at every
exponent, and zero argmax disagreements over 13,016 position-comparisons across
500 held-out words at four exponents.

**A second consequence, also provable.** At the evidence site a *letter* message
meets a one-hot observation, and geometric pooling of a one-hot with anything is
that one-hot: every other coordinate is multiplied by zero. So under the geometric
rule the L1→L2B channel of decision 18 is *inert* by construction rather than
harmful. Verified at exponents 0.1, 0.5 and 0.9: one non-zero entry, always the
observed letter.

**What this does to decision 46, which was recorded hours earlier.** Its
measurements stand and its conclusion narrows sharply. "Both evidence-site cells
significantly harm" is true *of linear mixing* and of nothing more general. The
harm is dilution: linear mixing trades away a certainty — the hard observation
likelihood — in proportion to $\alpha$, and no message however good repays that
trade. Geometric pooling never dilutes a certainty; it reweights within it. So the
site effect decision 46 measured is real, and its cause is the rule rather than
the site.

The same reasoning reaches back further. Decision 18 recorded the L1 coupling as
harmful and read it as a message carrying nothing the receiver lacked. Decision 46
corrected that to a statement about the insertion point. This entry corrects it
again: under the combination rule the library itself uses, that channel cannot
harm because it cannot act at all. The measured harm is the whole of what a
dilution rule permits, and all of it is damage.

**Why the factorial could not have seen this.** A 2×2 that varies source and site
while holding the rule fixed cannot discover that the rule is what makes site
matter. That is a general lesson about factorials rather than about this one: the
factor you hold fixed in every cell is invisible to the design, and the honest
response on finding it is to name it as a third dimension rather than to widen the
claim the 2×2 supports.

**OUTCOME of the exponent sweep, run after this entry was written and reported
here rather than in a new one, because it answers exactly the question this entry
left open.** All arms on the same claim:channel_factorial_positions positions against the same claim:channel_factorial_anchor_accuracy
anchor, each at its best engaged exponent:

| arm | exponent | accuracy | margin | at exponent 1 | p |
|---|---:|---:|---:|---:|---:|
| lexical message | 0.5 | claim:fitted_default_accuracy | +2.074 | +1.985 | 2.8e-80 |
| letter message | 0.1 | claim:geometric_letter_best_accuracy | −0.187 | −2.158 | 5.4e-03 |
| scrambled lexical | 0.1 | claim:geometric_scrambled_best_accuracy | −0.077 | −8.438 | 8.1e-03 |
| uniform | any | claim:channel_factorial_anchor_accuracy | **0.000** | **0.000** | 1 |

**On performance the two rules are a tie.** Geometric peaks at claim:fitted_default_accuracy against
linear's claim:channel_lexicon_predictive_best_accuracy, a difference of 0.068 points. Nothing here recommends switching
the shipped decoder, and this entry does not.

**On interpretation the two rules are not close, and that is the result.** The
uniform arm returns the anchor to every recorded digit, at every exponent, on zero
discordant pairs — the inertness the entry above predicted, now measured. Under
linear mixing a uniform message moves posteriors materially, so *any* linear
result is a mixture of information and regularisation separable only by argument.
Under geometric pooling the separation is structural, and every remaining effect
is content.

**Which vindicates decision 18, by a route decision 18 could not have taken.**
That entry recorded the L1→L2B letter message as harmful and read the harm as the
message carrying nothing the receiver lacked. Decision 46 reinterpreted it as an
insertion-point effect; this entry showed the site effect was a rule artifact.
Under a rule with provably no dilution the letter message *still harms*, by 0.19
points at its best exponent and 2.16 at full weight, monotonically worse as the
message is weighted more. Monotone degradation under a rule that cannot dilute is
content, not arithmetic. The original reading was right, and it took two
corrections and a third rule to establish it on evidence rather than on a
plausible story.

The scrambled control behaves as the criterion requires: it loses to the real
message at every exponent and its loss widens monotonically to 8.44 points at full
weight, so the lexical gain is correspondence to the phone inventory rather than
smoothing.

**What is still not claimed.** The two rules were compared at one receiver on one
level. Nothing here says geometric pooling is generally preferable, and the
shipped decoder is unchanged, so no recorded number moved.

**Falsifier.** If a geometric implementation ever produces different posteriors at
the two sites, it is not implementing geometric pooling — the identity is
algebraic, so a disagreement is a bug and never a finding. If a letter message
under the geometric rule moves any argmax at the evidence site, the one-hot is not
surviving the pool and the same conclusion follows.


## 48 — The upstream interoperability check must not live in a temporary directory

**What happened.** Thirty-three tests verify that AILM's specs parse with the
*real* GeneralizedNotationNotation parser rather than only with this repository's
strict subset reader. That check is what makes "expressed in GNN" a claim about
the notation instead of about a private dialect, and [@sec:architecture] rests on
it.

Their checkout defaulted to `/tmp/claude/gnn_probe`. Temporary-directory cleanup
reaped the parser sources while leaving the package directory and its
`__pycache__` in place, so the checkout looked present and was hollow. The tests
went from running to skipping, the suite's skip count drifted from 2 to 35, and
nothing announced that a manuscript claim had lost its evidence.

**The skip then reported the wrong cause.** It said "no checkout" at a path where
a checkout plainly sat. An absent checkout is the normal condition on a fresh
machine and an easy fix; a hollowed one is a broken environment. Reporting the
second as the first sends a reader to install something already installed. This is
the second probe in as many days to explain its own failure incorrectly, after the
cache audit whose fixed five-record sample could never reach a calibration record.

**Decision.** Search a list of candidate roots, most durable first, and treat a
temporary directory only as a fallback for an existing probe checkout. A durable
clone now lives outside the repository and is found without any environment
variable. The skip message distinguishes three states — available, hollowed,
absent — rather than two.

**What was NOT done, deliberately.** No file was hand-placed to make the count go
up. A green count produced by fabricating the artifact under test would be worse
than an honest skip, and the honest skip is what surfaced the problem.

**Falsifier.** A temporary directory appearing before a durable one in the
candidate order refutes the decision, and a test asserts the ordering rather than
the specific paths. If the skip message ever again names a cause that is not the
cause, the three-state helper has regressed; a test asserts each branch says
something true of the state it describes.

## 49 — Three hyperparameters were chosen; now they are fitted

**Provenance correction (2026-08-23).** The inner-split rule and the fitted
recorder first appear in the same Git commit. The protocol was prospectively
specified within that development session, but no immutable prior commit
establishes its ordering independently. Its claims are therefore exploratory,
not preregistered or confirmatory; `data/registration_provenance.json` binds the
classification to Git history.

**The gap.** `build_lexical_message` takes `max_prefix`, `min_count` and
`smoothing`, all set by hand and never fitted, and the pooling exponent of entry
47 was read off a declared grid. Four numbers governing this project's largest
architectural gain were nobody's measurement.

**Protocol, and the leak guard is the design.** Selection runs on an inner split
of TRAIN — every fifth pair to dev, the rest to fit — with the G2P matrices built
from the fit half alone, so no scored position and no statistic derived from one
is visible while the hyperparameters are chosen. The recorder refuses to run if
any selection word also appears in the held-out set. TEST is scored exactly once,
after selection closes, over an 81-point grid. Choosing a value on the positions
the headline is reported on is the single most likely way an arm like this
produces a fake gain, which is why the protocol is recorded rather than described.

**Result.** Selection picks `max_prefix` 6, `min_count` 2, `smoothing` 0.5 and
exponent 0.75. On the held-out set the fitted configuration scores claim:fitted_accuracy against
the hand-chosen claim:fitted_default_accuracy and the uncoupled anchor claim:channel_factorial_anchor_accuracy, a paired gain of
**0.663 points** on 827/543 discordant pairs at *p* = 1.7e-14. That is the best
grapheme-to-phoneme number this project has recorded, and it is 2.74 points above
the uncoupled decode.

**Evidence that the fit generalized rather than memorised its selection set.** The
gain on inner-dev was 0.717 points and the gain on the untouched held-out set was
0.663. Two similar numbers are what a fit that transferred looks like; a large dev
gain beside a small test gain would be the opposite, and both are recorded so a
reader can see which happened rather than being told.

**What moved and what did not.** Only `min_count` and `smoothing` changed from
their hand-set values; `max_prefix` was already at the selected value, so the
person who chose 6 chose well. The gain comes from admitting prefixes seen twice
rather than three times, and from halving the pseudocount — that is, from trusting
sparse evidence slightly more, which is the direction a hand-set prior usually
errs against.

**Not adopted as the shipped default, deliberately.** Changing
`build_lexical_message`'s defaults would move every recorded lexical-channel
number in the ledger and every sentence that cites one, in the same revision that
first measures the fit. Entry 32 declined exactly that combination for the
calibration gate, on the grounds that it leaves no version of the paper in which
the declared and the measured configuration can be compared. The fitted
configuration is recorded as a measured result; adopting it is a separate change
needing its own entry and a full re-record.

**Falsifier.** If a selection word ever appears in the held-out set the recorder
refuses, so a leak cannot be silent. If the dev gain and the test gain diverge
materially on a re-run, the fit is selecting noise and the grid is too large for
the inner split. And if the fitted configuration ever fails to beat the hand-set
one on the held-out set, this entry is withdrawn rather than the grid widened.

## 50 — The combination rule belongs in the spec, and only there

**Why the spec and not a conformance check.** Decision 47 established that the
combination rule, not the insertion site, decides whether an inter-level message
can help. That makes the rule load-bearing, and it is exactly the kind of fact the
GNN layer exists to declare — with one property that settles where it must live:
**the rule is invisible in the matrices.** The same emission, transition and prior
serve both rules; only the code combining them differs. There is nothing to read
off a live agent, so a conformance check against one could never see it.

That is the opposite situation from decision 41, where the inference-configuration
flags *were* readable off the agent and belonged in `describe_shapes`. The two
entries together give the rule for where a declaration goes: if a live model can
be interrogated for it, check it against the model; if it cannot, the spec is the
only place the claim can exist, and the check is that the declaration names
something the project implements.

**Decision.** `ModelParameters` may declare `combination_rule`, optional because
most levels carry no inter-level message and should not be made to declare
something they do not use. `L2b_grapheme_to_phoneme` declares `linear`, which is
the rule its recorded channel numbers were measured under.

**Layering, asserted rather than assumed.** A key with no value is a malformed
spec and the parser rejects it before conformance runs; a well-formed spec naming
an unimplemented rule is a conformance failure. Both are tested, including the
case-sensitivity of the name — a spec declaring `Linear` must fail, because a
tolerant match would let a spec name a rule the code then fails to dispatch on,
which is an invented name wearing a familiar spelling.

**A guard I did not know about caught this.** Adding the parameter failed
`test_every_model_parameter_on_disk_has_a_reader`, which requires every declared
`ModelParameters` key to be inventoried against whatever reads it. A parameter
nobody reads is decoration that looks like a check, and the inventory is what
stops the spec layer from accumulating it. The entry for `combination_rule` names
its reader and records why it is deliberately not read off an agent.

**Falsifier.** If the shipped decoder ever changes rule, this declaration and
every recorded lexical-channel number move together or the spec is lying, and a
test asserts the declaration matches the rule those numbers were measured under.
A rule listed in `COMBINATION_RULES` with no implementation behind it would pass
conformance and fail at run time, so a test pins the vocabulary to the two code
paths that exist.

## 51 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the in-context arm, and the answer key it must not be given

**Status correction.** This design was written before its first measurement in
the same development history, but it was not independently timestamped or
content-addressed before access to the outcome. It therefore records ordering
within a development session, not a preregistration, and cannot earn
`preregistered` or `confirmatory` status. The older heading overstated that
provenance.

**The objection it tried to answer.** AILM estimates a transition from training
data while the external probe meets the task through its prompt. The original
arm gave the probe labeled transition examples and withheld AILM's fitted
posterior column: the scientific question was whether the probe could infer the
mapping from observations, not whether it could copy the comparator's answer.
A mapping-deranged control preserved the displayed prefixes and label multiset
while breaking their correspondence.

**What the audit changed.** The first protocol treated longer prefixes as
examples even though their internal transitions exposed additional chain data,
described a stipulated transition as if it excluded every form of prior
exposure, and claimed token-statistic parity without authenticating either
model's tokenizer. Only character-level structure and the displayed-label
multiset were controlled. Those defects make the original protocol historical
development evidence, not the current measurement contract.

**Current disposition.** The retired numerical outcome is not repeated here:
its ledger labels are no longer current, and copying the old values into a
documentation exception would recreate the hard-coded-result path this project
forbids. Decision 90 owns the replacement protocol, its negative controls, and
the only active synthetic results. The unregistered diagnostic-English request
path was removed because no publication recorder consumed its keys or could bind
results from it to the claims ledger.

## 52 — The stratificational identity claim, narrowed to what the paper's own text supports

**The claim as written.** "Lamb's strata and AILM's levels are recognizably the
same idea." Reviewed against the rest of the manuscript, this failed on two
counts at once, and the second is the interesting one.

**It was true at an abstraction that carries no information.** "Several levels,
each with its own alphabet, related by a correspondence" describes the
hierarchical hidden Markov model and a transducer cascade equally well, and the
section compares against both at the level of *mechanism* — a single automaton
with an induced hierarchy, versus a composition algebra that never materialises
the joint state space. Beside those, an analogy was the weakest comparison in a
section whose other comparisons are structural.

**The premise for narrowing it was itself wrong, and checking that mattered.**
The queue entry asserted that the paper's own Level 4 and Level 6 sections argue
those levels are not strata in Lamb's sense. They do not. `07_case_level.md`
says the opposite in as many words — Level 4 is "structurally the grapheme-to-
phoneme model one level up" — and Level 4's uninformativeness is scoped strictly
to the *gold* population, not to the level. This historical review also called
Level 6 `stage-confirmed`; entry 26's later audit correction supersedes that
wording with a preregistered positive point estimate and a non-confirmatory
document-clustered primary analysis. The word "realize" appeared in neither
level file. Writing the narrowing from the queue entry's premise would have put
false statements about the paper into its own related-work section; retaining
the obsolete Stage 6 verdict would have done the same after the correction.

**What the text does support**, and what the claim now says: the specifically
stratificational content is the *realization relation*, and AILM has it at three
of five seams — where the upper level's observation alphabet **is** the lower
level's hidden state space — and lacks it at two, the word level reading its own
within-word alphabet and the case level reading treebank annotation. Three of
five is the size of the claim.

**The comparison's most useful output was not the resemblance.** The reported
receiver interventions are L1→L2B, L3→L2B, L3→L1g, L6→L5 and the privileged
punctuation-type-oracle→L3 arm. Four sources are model outputs; the last is not.
Level 4 appears in none of them, in either direction. That is a structural fact about the shipped
system that no section had stated, and the stratificational frame is what
surfaced it: the one level whose observations come from annotation is also the
one level with no realization relation measured at all.

**Two scholarly liabilities, closed rather than left implicit.** Reich's
finiteness thesis was deployed without the objection it has always faced;
@chomsky1956three is now cited as that objection, with the note that the two are
about different objects — a body of utterances against a generative capacity —
and that this paper takes no side because nothing it measures could. And every
source in the section is primary and none is later than 1973, so it speaks to
what the program proposed and is silent on how the field answered it,
including on why a framework this close in shape did not persist. That silence
is now disclosed in the section rather than inferable from the dates in the
bibliography.

## 53 — The scaling ladder measures a moving population, and the ladder says so

**What the sweep does.** claim:stage3_scale_rungs nested frequency-ranked
lexicons run from the configured bottom rung to every length-eligible training
type, each a prefix of the next and each scored on the held-out tokens it covers.
`SCALING_RUNGS` declares the interior sizes and `scaling_rungs` appends that
complete eligible vocabulary; an entry that counted the declaration rather
than the sweep would describe a ladder the figure does not draw. Numeric
coverage and perplexity both rise, while their utility directions oppose.
Coverage rises from
claim:stage3_scale_500_coverage_pct% to claim:stage3_full_coverage_pct%, and word
perplexity rises with it from claim:stage3_scale_500_word_perplexity to
claim:stage3_full_word_perplexity.

**Why the headline was nearly wrong.** Read per emission, the gain over the
letter-bigram bar is flat: claim:stage3_scale_500_gain_nats nats at the bottom rung against claim:stage3_full_gain_nats at the
top. That reads as "the level's advantage does not scale", and it is an artifact
of the denominator. Each rung is scored on a *different* population — claim:stage3_scale_500_eval_positions
emissions at 500 types against claim:stage3_full_eval_positions at the ceiling — so a per-emission mean is
an average over a set that grows as it is measured. On one shared denominator the
same numbers are strictly monotone and rise
claim:stage3_scale_gain_rise_pct%, from claim:stage3_scale_500_gain_shared to
claim:stage3_full_gain_shared. Both scales are
in the figure's title for that reason; quoting either alone is quoting a
denominator.

**The comparator is not re-fit, and the first version of this entry said it was.**
`build_letter_bigram_bar` is estimated once on the full training frequencies and
re-scored per rung; the matrices are bit-identical across rungs
(`np.array_equal`). The bar's curve therefore moves *only* with the evaluation
population, which is what makes it a probe of the population rather than of the
lexicon. Its rise, claim:stage3_scale_500_bigram_baseline → claim:stage3_full_bigram_baseline, is the same fact the ladder's real finding
rests on.

**Selection on coverage, now quantified rather than inferred.** Every rung selects
its own evaluation set by the same rule the level uses, so the ladder is a
sequence of measurements on progressively harder text. That was arguable from the
bar's curve and is now measured directly: at the shipped claim:stage3_vocab_size word rung the bar
reaches claim:stage3_bigram_baseline on the covered sublanguage against
claim:stage3_oov_bar_full_perplexity on the whole held-out stream and
claim:stage3_oov_bar_excluded_perplexity on the excluded remainder (decision 51's sibling artifact,
`vendor/brown_heldout_stream.json`). The mechanism is **not** length, though the
first version of this entry said it was; see decision 55.

**What the ladder therefore licenses.** That the word level's advantage over a
lexicon-free comparator is preserved as the lexicon grows and the text gets
harder, on a shared denominator. Not that the advantage is constant per emission,
and not that any rung's number describes running English: every rung describes
its own sublanguage, and the ceiling rung still excludes
claim:stage3_full_excluded_pct% of held-out tokens.

**Falsifier.** If the bar's matrices ever differ between rungs, the third series
stops being a population probe and the entry above is wrong. If the shared
denominator gain stops being monotone, the ladder's only surviving positive claim
goes with it. A rung present in the ledger and absent from the figure would
falsify the panel, which is why the figure discovers its rungs from the token
namespace rather than from a list.

## 54 — Budgeted attention: killed by its own comparator, and by what "peeking" costs

**The proposal.** Under a budget of `B` observations for a word of length `L`,
choose *which* letters to look at by expected free energy rather than by a fixed
schedule. The headline was a margin over "uniform stride".

**The comparator was the whole margin.** "Uniform stride" is not one arm. Its
phase is an unstated free parameter, and the spread across phases on the G2P
test split is 7.66 points at length 4 and **12.31 at length 5** — an order of
magnitude larger than any margin the proposal claimed. A comparison that fixes
the best phase for one side and an arbitrary phase for the other is reporting a
choice as a measurement.

**Against an honest bar the arm loses.** The bar here is the *exhaustively
optimal* fixed schedule per (length, budget), fitted on the train split: for
each shape, the observation set maximizing train accuracy, searched over all
`C(L, B)` subsets rather than approximated. It uses zero inference at decode
time and strictly less data-dependence than the adaptive arm. Over claim:budget_scored_positions
held-out positions at a half budget it scores claim:budget_fixed_accuracy% against the adaptive
arm's claim:budget_adaptive_accuracy% — **−1.08 points for adaptivity**. Every level in this project
has to beat the strongest thing that does not use the mechanism it is arguing
for; this one does not.

**The near-miss is the part worth keeping.** The first version of this probe
scored each candidate position by the entropy remaining after conditioning on
the letter *actually there*, and reported **+3.90 points for adaptivity**. That
is not a policy: it consumes the observation in order to decide whether to pay
for it, which makes the budget fictional. Corrected to marginalize over what the
candidate might show, under the model's own predictive distribution, the same
code returns −1.08. The peeking arm is retained in the script as a declared
upper bound and sits **+4.99 above** the honest one, which is the size of the
error that was one line away from being published as a result. It has the
property that makes this project's sharpest traps sharp: it produced an
impressive number rather than a degraded one.

**Decision.** No level, no ledger entry, no recorded null arm — a null in the
ledger would imply a level exists to be null about. The probe ships as
`scripts/probe_budgeted_attention.py` so the kill-check is re-runnable, and
[@sec:limitations] carries one paragraph.

**Falsifier.** If the adaptive arm ever beats the exhaustively-fitted fixed
schedule on held-out positions at any budget fraction, this entry is wrong and
the level is worth building. The probe takes `--budget-fraction`, so that is one
command rather than an argument.

## 55 — The coverage effect is not length, and the first account of it was a causal claim with no control

**What was written.** Decision 53 and the word level's new coverage paragraphs
attributed the gap between the covered sublanguage and the excluded remainder to
word length: covered tokens average 3.61 letters against 6.70, so a
frequency-ranked lexicon filters for short words, and short words are easier for
a letter model. Two marginal means, one causal conclusion, no control.

**Measurement correction (2026-08-24).** The first stratified recorder then
weighted each length bucket by its covered token count. That does not reconstruct
the metric the paper reports: per-emission perplexity gives a length-$L$ token
$L+1$ observations. The standardization now uses `n_L * (L + 1)` emission
weights, and a planted unequal-length control independently reconstructs the
direct score while demonstrating that the former token weights fail. The
direction below survives; its recorded magnitude is smaller.

**The control refutes it.** Scoring each length bucket separately, the excluded
tokens are harder in claim:stage3_oov_length_buckets_excluded_harder of the
claim:stage3_oov_length_buckets buckets holding at least
claim:stage3_oov_length_min_bucket tokens on both sides. Standardizing both
per-emission scores to the covered length distribution uses emission weights
`n_L * (L + 1)`, not token weights: it gives
claim:stage3_oov_matched_covered_perplexity against
claim:stage3_oov_matched_excluded_perplexity. Matching on length does not shrink
the gap; it **widens** it from claim:stage3_oov_raw_gap_nats nats to
claim:stage3_oov_matched_gap_nats, a change of
claim:stage3_oov_gap_change_under_matching_pct%. On this per-emission estimand,
length was masking part of the effect rather than causing it.

**Why this one is worth an entry rather than a quiet edit.** It is the exact
failure this repository's standard exists to prevent, committed in the paragraph
that introduced a new measurement — and committed while writing prose whose
stated purpose was to replace a definition with a measurement. The subordinate
claim was also checkable and also false: within the excluded population,
per-emission perplexity *falls* with length, from 27.25 at two letters to 12.75
at eleven, so "short words are easier for a letter model" is not true of this
data at all.

**What replaces it.** The stratified comparison is recorded rather than the two
means alone: `stage3_oov_length_buckets`,
`stage3_oov_length_buckets_excluded_harder`, the two matched perplexities, both
gaps, and the percentage change. The manuscript now states that the effect
survives length-matching and grows under it, and that the mechanism is
**unmeasured**. The plausible account — that in-vocabulary tokens are the
frequent, orthographically typical ones whose transitions dominate the
token-weighted counts the comparator is estimated from — is written down as a
hypothesis this section does not test.

**Falsifier and guard.** One test asserts the excluded side is harder in a
majority of shared length buckets; a planted unequal-length control separately
requires the weighted bucket scores to reconstruct direct per-emission scoring
and proves that the former token weights do not. If either stops holding, the
paragraph is wrong. The old claim had a test too
(`excluded_pp > covered_pp`) and it passed throughout, because it pinned the
direction of the raw gap and the raw gap was never in doubt. A guard on the
conclusion is not a guard on the reasoning that reached it.

## 56 — What a twenty-eight-agent adversarial audit found, and the three guards that could not fail

Six review dimensions over the newly landed work, each candidate finding then
attacked by an independent skeptic instructed to refute it. Twenty-two
candidates, **twelve confirmed and ten refuted** — a refutation rate worth
recording, because it is the reason the verify stage exists: five of the ten
refuted findings were plausible, specific, and wrong, including three that
claimed a family-size contradiction that does not exist and one that claimed a
misquotation that turned out to be accurate.

**The scientific error is decision 55**, recorded separately: the coverage
effect was attributed to word length by a causal claim with no control, and the
control refutes it.

**Two defects reached the published PDF.** An interval joining two negative
tokens with `--` rendered as `-0.383605—0.308352`: three hyphens became an em
dash and ate the upper bound's minus sign, so a 95% interval that excludes zero
printed as one that spans it, in the same parenthesis as a p of 3e-31. And the
scaling-ladder caption asserted the comparator is "re-fit at each rung" four
lines above the sentence saying it is not — the correction from decision 53 had
been applied to the body and to half the caption. The figure module's docstring
joined the claim to its own negation with "so".

**Three guards could not fail, and each failed differently.**

The British-spelling guard used `\b` on both sides of every pattern. A word
boundary cannot match between a letter and an underscore, so it missed
`normalise_text`, `summarise_paired_effect` and `stat_stage2_favoured_ratio` —
identifiers being exactly what the sweep renamed. It would have reported the tree
clean while every renamed function still carried the old spelling.

The replay exclusion's upper-bound anchors were recorders that import nothing
from `ailm.benchmarks`, so they are insensitive to the predicate by
construction. Appending one pure helper to `_SERVER_ENTRY_POINTS` dropped the
hermetic `record_calibration_floor_claims.py` out of replay — the slow recorder
somebody has a live incentive to exclude — with all four guards green. The
anchors are now discovered as recorders that import from that module *without*
importing a server entry point, and the mutation now fails.

The decisions index paired nothing. Row labels were compared to the set of
heading numbers and row anchors to the set of real anchors, so a row could name
one decision and link to another, and an index where every row linked to the
same heading passed. A dead link reads as a missing decision; a link to the
wrong decision reads as the right one.

**The pattern under all three.** Each guard checked a *conclusion* while the
thing that could go wrong was the *reasoning*. The old held-out-stream test
asserted `excluded_pp > covered_pp`, which was true throughout and stayed true
while the prose attributed the gap to the wrong cause. Two further tautologies
came out with them: a value compared to a recomputation of itself, and a set
difference empty by construction.

## 57 — The in-context result replicates on a second probe, and the family grew to say so

**Why a second probe.** Every sentence of decision 51's outcome rested on one
model. An interface finding on one system is an anecdote about that system.

**It replicates, and more cleanly than the first.** Qwen 2.5 3B answers **nothing**
correctly zero-shot on this chain and clears the corrected threshold against
zero-shot at *every* rung including k=4 (p = 1.49e-8 at four examples,
9.10e-13 at forty-eight). Against the derangement of the same examples it clears
nothing anywhere: p = 0.5966, 1.0000, 0.2800, 1.0000. Across both probes,
**eight of eight** mapping contrasts fail to approach significance, and the
control is numerically ahead in six of the eight cells. At the headline k the two
probes sit −0.3525 and −0.3900 below the comparator, having closed 18.50% and
16.58% of their zero-shot gaps.

**What the second probe adds that the first could not.** Gemma's lift did not
clear correction at k=4, so "the examples help" rested on the three larger rungs.
Qwen clears at every rung from a floor of exactly zero, which makes the lift
unambiguous — and makes the mapping null harder to explain away as a weak
manipulation. The examples are demonstrably doing something; what they are not
doing is conveying the chain.

**The family grew, and the recorder had to be changed to know it.** The declared
size was `len(FAMILY_TEMPLATES)` while one probe was registered. Registering a
second doubles the tests the recorder declares, and leaving the size alone would
have halved the threshold every "clears correction" sentence is read against.
Size and alpha are now templates times probes: 18 members at 0.05/18 = 2.778e-3.
Every conclusion above is stated against the tighter threshold, and every one
survives it.

**Partial runs are refused rather than supported.** There is no `--model` flag.
The stale sweep clears every `bench_ctx_` label a run did not produce, so scoring
one probe and writing would delete the other's arms while leaving a two-probe
alpha in place. Both probes are measured in one process or nothing is written,
and a test asserts the flag does not exist.


## 58 — pymdp contains sparse contraction machinery; the blocker is at the Agent API

**The false claim in earlier prose.** Several sections and docstrings asserted
that "pymdp takes dense arrays and has no sparse path." That is false as
written. `pymdp`'s JAX backend contains sparse contraction machinery in
`maths.py` (`spm_dot_sparse`, `factor_dot_flex`), sparse belief handling in
`inference.py` (`JAXSparse`), and sparse-dependency FPI in `algos.py`.

**The exact boundary where sparse B is blocked.** While the underlying math
routines contract sparse tensors without densification, sparse arrays cannot be
passed through the public `Agent` API:
1. `pymdp.agent.Agent.__init__` (line 263): calls
   `jnp.broadcast_to(b_f, (batch_size,) + b_f.shape)`, which raises
   `TypeError: broadcast_to requires ndarray or scalar arguments, got BCOO`.
2. `pymdp.agent.Agent._validate` (line 1183) → `pymdp.utils.validate_normalization`
   (line 109): calls `jnp.sum(tensor, axis=axis)`, which raises
   `TypeError: sum requires ndarray or scalar arguments, got BCOO`.
3. `pymdp.control.compute_expected_state` (line 324): contracts `qs` against
   sparse `B` and returns a BCOO single-factor belief. Feeding that BCOO back as
   a prior on the next step fails in `factor_dot` because belief propagation
   expects dense arrays for 1D probability vectors.

**Verified equivalence under the generated three-site patch.** A re-runnable
probe (`scripts/probe_pymdp_sparse.py`) copies the installed package, applies the
exact candidate patch with `git apply --check` followed by `git apply`, and then
imports that copy in a fresh process. The three sites are:
- `validate_normalization` summing sparsely via `sparse.sparsify(jnp.sum)(tensor, axis=axis)`
- `compute_expected_state` densifying its 1D returned marginal via `maths._to_dense_if_sparse`
- `Agent.__init__` preserving a caller-supplied leading BCOO batch dimension and
  rejecting the unbatched form explicitly instead of passing it to
  `jnp.broadcast_to`

With those changes, a sparse-B Stage 3 Agent runs with **0.000e+00 max posterior difference** against
the dense Agent across test words and produces exact identical trie-node supports
at prefixes `th` (30), `the` (11), `co` (83), and `s` (218).

The probe performs no runtime reassignment of pymdp functions. A structural
negative control in `tests/test_upstream_pymdp_bundle.py` plants such a
replacement and requires the guard to reject it; the measured equivalence is
therefore evidence about the applied candidate patch rather than a patched test
instrument hidden inside the process.

**Correction.** Every manuscript and codebase occurrence now states the
boundary rather than the denial: no sparse path is reachable through pymdp's
public `Agent` API. The manuscript also states the positive half, which this
entry recorded and the paper did not — that the contraction machinery exists and
the transition contraction is already sparse-capable. A reader of the paper
could not previously tell that the restriction was an interface boundary rather
than a limit of the method. `tests/test_pymdp_sparse.py` pins both halves: it
re-runs the probe, and it refuses any unqualified denial reappearing in `src/`,
`docs/` or `docs/manuscript/` — written against the class of statement rather than
the sentence removed, so a rephrasing fails too.

**Sparse `B` is DECLINED for the shipped Stage 3 level.** It was built, measured,
proved exactly equivalent, and is not adopted. The arithmetic, computed by the
probe at the shipped configuration ($V$ = claim:stage3_vocab_size, $P$ =
claim:complexity_l3_position_states):

| Array | Dense | As BCOO |
|---|---:|---:|
| Word identity `B[0]` | 16.0 MB | 24.2 kB |
| Whole agent | claim:complexity_l3_memory_bytes bytes | — |

Sparsity saves 16.0 MB of a 19.5 MB agent, on a level nothing is memory
constrained by, at the shipped default. (The BCOO figure is index-dtype
dependent: 12 bytes per nonzero at float32 data with int32 indices. Quote the
route, not only the number.) There is no operational consumer, and the adoption
cost lands in this project's most heavily reconciled machinery —
`complexity_l3_memory_bytes` is currently reconciled three independent ways, and
a sparse `B` would have to be reconciled the same three ways plus the two
patches, which are not upstream.

**What refutes this decline.** A queued consumer appearing. The specific
condition is any Level 3 variant in which `B[0]` is not the identity — a learned
within-word transition, an open-vocabulary UNK state, a morphemic segmentation —
since each of those makes the word transition carry statistics rather than
structure, and the 16.0 MB stops being an identity matrix nobody reads.

**What this does not rescue.** The trie encoding. Decision 13's conclusion
survives and only its stated reason changes: from "does not fit" to "costs
multiples more nonzeros and still hides word identity." The trie's failure was
never storage.

## 59 — the letter-at-evidence channel is scored on the receiver's proper loss, and both axes agree

**What was missing.** [@sec:hierarchy]'s working-hypothesis box is stated "under
a fixed receiver and a proper predictive loss", and every channel number in the
paper was an argmax accuracy. Accuracy is a threshold statistic on the
receiver's posterior: it cannot see mass moving without the argmax moving, and
it can move either way on near-ties while the receiver's belief about the target
gets worse. The section's central diagnostic — "a monotone decline over the
whole grid means no weight rescues the message" — was therefore a statement
about a statistic, not demonstrably about a channel.

**The decision.** Re-score one channel on the declared axis rather than restate
the diagnostic. `scripts/record_channel_predictive_loss_claims.py` is the single
writer for `channel_proper_loss_*`; the computation lives in
`scripts/probe_channel_predictive_loss.py`. The cell is **letter@evidence** from
decision 46, chosen because its accuracy is already recorded, so both axes come
from the same claim:channel_proper_loss_positions positions with no population
change between them.

**Which cell, and how that is enforced.** The recorder refuses to write unless
it reproduces the recorded accuracy for that cell and the uncoupled anchor to
four decimals. A probe measuring a neighboring channel would produce plausible
loss numbers the prose would then misattribute, and no other check would catch
it. The void condition from the research queue is a separate guard: the
weight-zero posteriors must be bit-identical, position by position, to a plain
hard-evidence filter written independently of the coupling code, so the anchor
is provably the uncoupled decode rather than assumed to be.

**The measurement.** The uncoupled anchor costs
claim:channel_proper_loss_anchor_nats nats per position. No engaged weight
improves on it: the best gives claim:channel_proper_loss_best_realized_nats and
the worst claim:channel_proper_loss_worst_realized_nats. Accuracy on the
identical positions declines monotonically as well
(claim:channel_proper_loss_accuracy_gain_pts points at its best engaged weight).
**Sign and shape class agree between the two axes**, so for this cell the
diagnostic is a property of the channel. That is a result about the diagnostic,
and it could have gone the other way — a disagreement is precisely the
falsifier the research queue registered for this arm.

**Why the harm is structural.** The oracle-sender arm replaces the L1 message
with a point mass on the letter actually present. The receiver's loss does not
move: the spread across every weight is
claim:channel_proper_loss_oracle_spread, exactly, because
$(1-\alpha)\delta_x + \alpha\delta_x = \delta_x$. The receiver's sensitivity to
the variable this message carries is claim:channel_proper_loss_sensitivity_nats
— it already observes the letter — so the realized effect is bounded above by
zero and the only available outcomes were no effect or harm. This is the
hypothesis box's own *sensitivity* term, measured rather than argued.

**What this does not settle.** One channel of five, on one of the two axes. It
is not $I(Y; M \mid X)$: everything here is still an end-to-end intervention at
a fixed mixing rule, which is the confound [@sec:hierarchy] already names. And
it says nothing about letter@**predictive**, the cell that resolves neither way
on accuracy, because that cell mixes into the prior rather than the evidence and
a prior is not an observation — the redundancy argument above does not transfer.

**The guard.** `tests/test_channel_proper_loss_claims.py` re-measures and
compares against the ledger, and asserts the sign is still a harm. A negative
result is the kind that decays quietly: if the channel ever started helping, the
manuscript paragraph resting on "no engaged weight improves on it" would be
false while every structural test still passed.

## 60 — parameters are counted three ways, and every level is declared as a GNN file

**The question the paper could not answer.** [@sec:architecture] argues for
factorization on storage grounds and prices the flat alternative in GB and TB,
but the shipped model had no parameter count anywhere. `ailm.complexity` prices
arrays in *bytes*, which is a different quantity: bytes are a property of the
representation, and a reader asking "how many parameters" is asking about the
model.

**The decision.** `ailm.parameters` counts every level through its own shipped
builder and reports three numbers per array rather than one, because each is
wrong on its own. **Stored** is the dense entry count. **Nonzero** is the
entries the model can place mass on. **Free** is what a fit could move: each
distribution loses one degree of freedom to the simplex constraint, so a column
with $k$ live entries carries $k-1$ and a deterministic column carries none.

**Why one number would not do.** Reporting stored alone makes Level 3 look like a level of
claim:params_l3_stored parameters when it fits only
claim:params_l3_free of them — its word transition is an identity padded out to
the square of the vocabulary and its emission is the lexicon itself, one letter
per word-position pair. Reporting nonzero alone still charges a one-hot emission
for every entry it has, though a lookup table written as a matrix has entries
and no freedom. The free count is the one comparable to a parameters column
reported for a neural model, and it is claim:params_total_free for the whole
hierarchy against claim:params_total_stored stored entries —
claim:params_total_density_pct% of which are non-zero.

**Orientation is asserted, not assumed.** `count_array` checks that each array
sums to one along the axis its kind declares before counting anything, and
raises otherwise. The orientation trap — summing the wrong axis and getting a
plausible number — is the most repeated defect in this log, and a parameter
count taken off a mis-oriented array is wrong without looking wrong. A
row-stochastic array handed in as column-stochastic is rejected; its transpose
is accepted.

**Preferences are excluded.** `C` totals
claim:params_total_preference_entries entries and appears in no column. It does
not normalize, nothing estimates it from a corpus, and folding a preference into
a parameter count would inflate the model by an amount carrying no information
about the data. It is measured so the exclusion is visible rather than silent.

**Every level is declared as a GNN file, and the declaration is checked.**
`ailm.parameters.GNN_SPEC_FILES` is the registry binding each level model to the
spec under `gnn/models/` that declares it, and it is in bijection with that
directory: a spec nothing registers, or a level with no spec, fails the test.
`check_model_against_gnn` compares every built array against the product of the
dimensions its spec declares, mapping the spec's multi-factor names (`B_f0`,
`B_f1`) onto the built agent's (`B[0]`, `B[1]`) — a mapping that only
understood the single-factor `B` would silently skip Level 3's second factor and
report conformance having compared nothing. All
claim:params_gnn_specs specs conform to the built agents, with
claim:params_gnn_conformance_failures discrepancies.

This is worth checking because the two are independent by construction: the spec
does not generate the builder, both are written by hand, and nothing at runtime
would notice them disagreeing on a size. The agent would run, the gate would
report a number, and the file the manuscript points at as that level's
definition would describe a different model than the one measured. Each level
section of the paper now names its spec, and the paths are checked against the
registry — prose and registry are two hand-maintained lists of one fact, and the
path is prose rather than a token, so hydration cannot catch a drift.

**The guard.** `tests/test_parameters.py` pins the counting rule with a
deterministic-versus-uniform pair, rejects a mis-oriented array while accepting
its transpose, checks that every total is the sum of its parts and that free
never exceeds nonzero nor nonzero exceeds stored, and detects a planted
size disagreement in a copied spec while the unmodified spec still conforms.
`scripts/record_parameter_claims.py` is the single writer for `params_*` and
refuses to write on any of those failures.

## 61 — five typed numerals became claims, and the template's own builder publishes the figure registry

**Two failures of the same kind, found by running a validator the project had
been ignoring.** `stage_04_validate.py` reported three critical issues. None was
noise.

**The five numerals.** `docs/manuscript/05_discussion.md` printed the
budgeted-attention results as bare numbers, transcribed by hand from a probe's
stdout: the fixed-schedule and adaptive accuracies, the adaptivity loss, the
peeking headroom, and the largest uniform-stride phase spread.
[@sec:reproducibility] states that every number in the paper is a token resolved
from the ledger, and these five were not. Nothing caught it:
`tests/test_claims_discipline.py` scans for `claim:` tags with no backing entry,
and a bare numeral has no tag to scan — the gate is shaped to catch a broken
reference, not a missing one.

The fix is the one this project's discipline already prescribes.
`scripts/probe_budgeted_attention.py` now exposes
`measure_budgeted_attention`, so its numbers have a caller other than stdout,
and `scripts/record_budgeted_attention_claims.py` is the single writer for
`budget_*`. The prose is tokenized. The recorded result is unchanged and
negative: the train-fitted fixed schedule beats the adaptive policy by
claim:budget_adaptivity_loss_pts points over
claim:budget_scored_positions positions.

**The figure registry.** The template's output validator expects
`output/figures/figure_registry.json` built from project-declared figure specs.
This project emitted only its own `figure_provenance.json` and the check failed.
`scripts/generate_figure_registry.py` now publishes the registry by **importing
the template's builder** rather than hand-writing its envelope: the schema is
defined by the template and consumed by its validator, so writing it from the
outside would be a guess at another repository's contract that passes today and
drifts the moment the template changes it. A malformed spec now raises
`FigureRegistryError` at publication instead of surfacing two stages later.

Neither field is invented. `filename` and `generated_by` come from the
provenance receipt written by the run that produced the images; `label` and
`caption` come from the manuscript's own figure references. The scan covers
numbered sections only — `docs/manuscript/SYNTAX.md` carries a worked example
pointing at `name.png`, and including it made the builder claim a file that
cannot exist.

**The import boundary held.** `docs/architecture.md` forbids `src/ailm/` from
importing `infrastructure`, and it still does not: the publisher is a script,
resolving the template root exactly as
`scripts/z_generate_manuscript_variables.py` does, and a test asserts no module
under `src/` imports the package. Without a template checkout the publisher
exits 2 with a message; the figure pipeline itself is unaffected.

**A file for literals that are not measurements.**
`data/manuscript_literals.yaml` declares stable entry identifiers in the
project’s own decision log. Deposited-document locators have since moved to
`data/reference_sources.json`, the literature cutoff and claim policy to
`data/literature_search.json`, and executable design constants to code-owned
registries, so each value now comes from the narrowest source that can validate
it. The remaining literals are grounded through the template's evidence
collector, which reads project YAML. The
property worth keeping is that a bare *measured* number still fails: the only
way to make one pass is to write down, in that file, why it is not a
measurement.

**A template-side bug, fixed template-side.** `Rendered provenance inputs`
first failed with `SOURCE_SYMLINK_ESCAPE`: this project lives in the private
sidecar and is symlinked into `template/projects/working/`, so the snapshot
resolved the symlink, found the tree outside the template repository, and
refused to read the very tree it had been asked to describe.
`infrastructure/validation/rendered_snapshot.py` now takes an optional
`extra_root` containment root, and the project walk passes the project it is
already walking; a symlink escaping both the repository and the declared project
is still refused, and a caller declaring no extra root keeps the original
boundary exactly. The template's own validation suite passes, including a new
regression test covering all three cases.

**A decision reversed, deliberately.** With the escape fixed the check reported
`ARTIFACT_MANIFEST_EMPTY`: it binds the report to a manifest of the *committed*
output tree, and `output/` was ignored outright, so the manifest was empty and
the check could not run. Passing it by un-ignoring without committing would have
listed files no fresh clone could produce — the exact drift
`collect_current_artifact_manifest` documents itself as guarding against, and a
hollow green.

So the artifacts are tracked, following the template's own stable/transient
split: manuscript, figures, the combined PDF, web,
the report JSONs and the manuscript variable map are committed; slides,
`bench_raw` and LaTeX intermediates stay ignored, as does `cache/` (~190 MB),
which holds the memoized model calls and is an input rather than an output.
Seventeen MB across sixty-seven files, against 229 MB in the tree.
`docs/output_conventions.md` is rewritten to state the new position rather than
left arguing the old one.

One trap worth recording: `!output/` must lead the allowlist. This repository's
`core.excludesFile` is the sidecar's `.gitignore`, which excludes the `output/`
*directory*, and git does not descend into an excluded directory — so without
re-including it first, every negation below is dead and `check-ignore` reports
the parent's rule for every path. The template's own `.gitignore` uses the same
`!output/` then `output/*` pair for the same reason.

What did NOT change is the rule that matters: no generated artifact is ever
hand-edited to make a check pass. What changed is which artifacts ship. All nine
checks now pass.

**The guard.** `tests/test_figure_registry.py` covers the project-owned half —
spec assembly, the section-scan rule, the referenced-but-ungenerated error — and
deliberately not the envelope's schema, which belongs to the template and would
become a second definition free to drift. It also asserts the two template-root
resolvers agree and that `src/` imports no `infrastructure`.

## 62 — three oversized modules became packages, along seams that already existed

**The three.** `gnn.py` at 2,405 lines, `benchmarks.py` at 2,291, and
`manuscript_variables.py` at 1,517 — against a 65-module mean of 483. Each was
doing several jobs, and the largest was mostly not code at all.

**`ailm.gnn`** split along the `# ──` section markers its own author had already
drawn: `types`, `parsing`, `shapes`, `conformance`, `registry`. The dependency
order is linear and acyclic. `parsing` never touches a live agent, so a spec can
now be read with no pymdp in reach, which was true before and is now
structurally enforced rather than merely observed.

**`ailm.benchmarks`** had no such markers, so the boundaries came from where the
file's own ordering already changed concern: `constants`, `prompts`, `scoring`,
`ollama`, `corpora`, `comparators`, `calibration`, `harness`.

**`ailm.required_claims`** is the 955-line `REQUIRED_CLAIMS` tuple, which was 63%
of `manuscript_variables.py` and buried the ~500 lines of hydration machinery
under a list. It is data, not logic. `manuscript_variables.REQUIRED_CLAIMS` still
resolves.

**Every public API is unchanged**, re-exported from each package's `__init__`,
including the underscore-prefixed internals the test suite reaches for to
exercise individual checks in isolation.

**Two defects the split exposed, both the same shape.** `spec_dir()` and the
benchmark cache root computed the project root as `parents[N]`. Moving a module
one level deeper made both resolve to `src/gnn/models/` and `src/vendor/` —
plausible-looking paths, so every caller reported "no such file" for artifacts
that had not moved. Both now walk up until they find a directory holding the
markers they need, and say so when there is none. A hop count is a load-bearing
assumption about file depth that nothing states and nothing checks.

**One test the split broke for a real reason.** `test_call_cache.py`
monkeypatched `benchmarks.post_generate_uncached` to count server calls. A
function in `harness` resolves that name in its own module globals, so patching
the package attribute stopped intercepting it and the probe hit a real server.
The patch now targets the module where the name is looked up. Two source-reading
assertions that opened `src/ailm/benchmarks.py` by path now use
`inspect.getsource`, which survives any future move.

**The guard.** `tests/test_parameters.py` gained a docs-wide check that every
`gnn/...` path printed anywhere under `docs/` resolves and is registered; two
references to `gnn/ailm_word_level.md` had survived the specs moving to
`gnn/models/` because a path in inline code is neither a hydrated token nor a
markdown link.

## 63 — Stage 4's derived margin is a mixture, and one oracle path carries all of it

**The finding.** `stage4_derived_margin_pts` = claim:stage4_derived_margin_pts is
an aggregate over three populations whose labels have different provenance. The
oracle in `ailm.corpus.ud._apply_case_oracle` assigns each label by one of three
paths in strict precedence: morphological `feats[Case]`, then a deterministic
dependency-relation map, then an ADP token's own lemma. The last route labels
that ADP token and does not transfer the role to its head. Partitioned that way,
the margin is not a property of the level.

| path | tokens | baseline | model | margin | discordant | *p* |
|---|---:|---:|---:|---:|---:|---:|
| dependency relation | claim:stage4_path_deprel_tokens | claim:stage4_path_deprel_baseline | claim:stage4_path_deprel_agreement | claim:stage4_path_deprel_margin_pts | claim:stage4_path_deprel_discordant_n | claim:stage4_path_deprel_mcnemar_p |
| morphology (control) | claim:stage4_path_morphology_tokens | claim:stage4_path_morphology_baseline | claim:stage4_path_morphology_agreement | claim:stage4_path_morphology_margin_pts | claim:stage4_path_morphology_discordant_n | claim:stage4_path_morphology_mcnemar_p |
| adposition | claim:stage4_path_adposition_tokens | claim:stage4_path_adposition_baseline | claim:stage4_path_adposition_agreement | claim:stage4_path_adposition_margin_pts | claim:stage4_path_adposition_discordant_n | claim:stage4_path_adposition_mcnemar_p |

**What it is not.** Not circularity in the model. The model observes the UPOS tag
and nothing else; `FORBIDDEN_OBSERVATIONS` blocks the label's source fields and
the vendored artifact withholds them. The stratification recomputes those fields
from the treebank to *partition an evaluation*, and the partition never reaches
the model. Reading a field to stratify a result and reading it to make a
prediction are different acts, and only the second is the trap this project named.

**What it is.** On the dependency-relation path the label is a deterministic
renaming of a syntactic relation, and a first-order model over part-of-speech
tags recovers syntactic relations well. That stratum demonstrates a result about
word order wearing case's name. On the morphology path, where the label is real
English case marking, the effect is not shown. On the token-level adposition path, where the
label sits on the preposition rather than the nominal it marks, it reverses.

**The morphology stratum is registered as a control, not a result.** Its
discordant-pair count is far below the minimum required to resolve an
equivalence, so its outcome is *not shown* and must never be reported as
*absent*. Registering it this way before reading it is what stops a small null
from being quoted as evidence of no effect.

**The ceiling-effect objection, answered rather than deflected.** The morphology
stratum starts from a much higher baseline and so has less room to improve.
Normalizing each margin by its available headroom, the dependency-relation path
captures roughly a fifth of what is reachable and the morphology path roughly a
twenty-fifth. The gap narrows under that normalization and does not close, so the
difference is not only a ceiling effect.

**Nothing recorded is withdrawn.** The aggregate is correct arithmetic, it
reproduces from the partition exactly, and the gate-audit verdict is unchanged.
What changes is what the number licenses. `05_discussion.md` already said
agreement with an unvalidated heuristic is not accuracy; that qualitative
disclaimer is now a measured distribution a reader can inspect.

**The guard that should have caught this was hollow.**
`tests/test_stage4_gate.py::test_a_deprel_observation_would_be_near_perfect`
carried a docstring saying it "builds the circular predictor the guard forbids
and shows it scores far above the honest model." Its body asserted
`len(set(UD_DEPREL_TO_CASE.values())) >= 5` and `model < 60`. It built no
predictor and recorded no number, so the demonstration the docstring promised did
not exist and the suite reported it as covered. It now builds the predictor. A
model permitted the forbidden observation scores 72.81% against the honest
model's claim:stage4_derived_agreement% and the context-free bar's
claim:stage4_derived_baseline%, so Level 4 closes about 22% of the distance
between knowing nothing and knowing the label's own generator. That is the number
the docstring always claimed to report.

**Refusals added.** The recorder aborts if the strata fail to align
element-for-element with the scored positions, and if they fail to sum back to
`stage4_derived_test_tokens`. A stratification off by one token still sums to a
plausible total while attributing every result to the wrong path, and no
aggregate check can see it.

**Multiplicity treatment: excluded from the declared family, and why.** The three
stratum p-values do not join `SIGNIFICANCE_FAMILY`. They partition a hypothesis
already in it: every position they score is a position
`stage4_derived_mcnemar_p` already scored, and the strata sum back to the derived
population exactly. Admitting them would count the same positions twice and
tighten the corrected threshold for every established result on the strength of a
diagnostic. That is the same trade the cluster-aware re-analyses already refuse,
for the same reason.

The cost of that choice is stated rather than hidden: none of the three may be
reported as a confirmed effect. The dependency-relation p-value is quoted to show
the aggregate is a mixture, which is a claim about decomposition and not about
significance. The other two are controls whose outcomes are *not shown*.

## 64 — two subsample nulls become bindable by recording counts that were being discarded

**What was wrong.** `record_subsample_claims.py` computed the discordant counts
`b` and `c` for every arm and passed them straight into `binomtest` without
recording them. The arms therefore carried a margin and a p-value but no
interval, because `paired_difference_bound_from_counts` needs the counts and
there was nowhere to read them from. A null with no bound cannot be told apart
from an unmeasured one, which is the whole difference between "no effect was
found" and "no effect could have been found".

**What changed.** The counts are recorded for the two subsample arms:
claim:subsample_stride5_mcnemar_b / claim:subsample_stride5_mcnemar_c over
claim:subsample_stride5_positions positions, and
claim:subsample_stride3_mcnemar_b / claim:subsample_stride3_mcnemar_c over
claim:subsample_stride3_positions.

**Not recorded for the `full` arm, deliberately.** That subset *is* the Stage 2
G2P comparison, and its discordant counts already have exactly one writer as
`stage2_g2p_mcnemar_b`/`_c`. Recording them here would give one quantity two
producers, and the two would drift the first time either recorder's population
changed. The single-writer rule is not a convention to work around when a second
copy would be convenient.

**The bounds are derived, never recorded.** The counts are the measurement. An
interval stored beside them would be a second copy of the same fact with its own
opportunity to go stale, so the interval is computed at render time from the
counts, exactly as the seven existing paired-bound arms are.

**What the bounds show.** The stride-5 arm is a genuine null that now has an
interval around it. The stride-3 arm's interval **excludes zero**, which its
p-value alone did not make visible. That interval is at the difference-scale coverage
`DIFFERENCE_CONFIDENCE` declares, and it is uncorrected, and this project judges its declared
family at a Bonferroni-corrected threshold, so this is explicitly *not* a
confirmed effect and the manuscript does not report it as one. What it does
establish is that the arm was never a null: it was an unmeasured interval being
read as one.

**Refusal added.** The recorder aborts if `100(b-c)/n` disagrees with the margin
recorded beside it. Counts and margin that disagree would mean the two were
computed over different populations, and a bound derived from them would be an
interval around a number nobody measured.

## 65 — the original token-level oracle-intervention mean becomes bindable on its own scale

**What was missing at this point in the chronology.** The punctuation-type-oracle
intervention recorded improved and degraded
token counts, which bound a *share*, while the reported effect
(claim:clause_word_best_delta_ppl on the perplexity scale) is a *mean* over
per-token log-losses. A mean has no interval without a dispersion, so the arm
carried an effect size and a p-value and no bound. `TODO.md` asked for the
missing quantity to be recorded rather than the bound approximated on a proxy
scale, and that is what this does.

**What changed.** `sweep_clause_alpha` already returns per-token
log-probabilities aligned position-for-position across alphas, so the dispersion
was one line from data the recorder already held.
claim:clause_word_best_delta_sd is the standard deviation of the per-token
log-probability difference between the best alpha and the anchor, over
claim:clause_word_eval_tokens tokens.

**What the original bound said, and what now supersedes it.** The token-level
dispersion made an interval on the mean log-loss scale computable, but did not
make it population-valid because tokens are nested within sampled source
documents. Decision 89 now supplies the current whole-document bootstrap
interval and resolution-limited document sign-flip sensitivity. The token-level
standard-error construction below remains historical method chronology, not the
current uncertainty statement.

**Three reasons that is a bound and not a confirmation.** First, the interval is
computed at the token level while the corpus samples sentences, and this project
already records that its paired tests are computed at units nested inside the
units their corpora sample; a token-level standard error understates the true
one. Second, it is uncorrected, and the declared family is judged at a
Bonferroni-corrected threshold. Third, the effect is tiny in absolute terms:
claim:clause_word_best_delta_ppl of perplexity against an anchor of
claim:clause_word_anchor_ppl.

**Correction to the old sign reading.** Under the corrected Stage 3 partition,
more tokens improve (claim:clause_word_best_improved_tokens) than degrade
(claim:clause_word_best_degraded_tokens), and aggregate perplexity improves as
well. A share interval still cannot express the magnitude or dispersion of the
mean, which is the substantive argument for recording the dispersion rather
than leaning on the counts already present. Decision 88 supersedes only the
retired direction and population reading, not this measurement-scale argument.

## 66 — the two coverage levels are harmonized, and the label stops carrying the level

**What decision 37 resolved, now executed.** The discordant-share intervals read
at 0.99 and the paired-difference bounds at 0.95, for no reason beyond having
been introduced by different recorders. They also sat backwards on their roles:
the stricter level went to the primary effects, which are already judged against
`BONFERRONI_ALPHA` and do not need their interval to carry extra coverage too,
while the looser went to the nulls, where the interval *is* the whole claim. And
they are one interval, not two: `paired_difference_interval` maps the
Clopper-Pearson share interval through an affine rescaling, which preserves
coverage exactly. `SHARE_CONFIDENCE` is now `DIFFERENCE_CONFIDENCE`.

**A third literal, which no scoping had found.** `summarize_paired_effect`
carried `confidence: float = 0.99` as a bare default that never referenced
`SHARE_CONFIDENCE` at all, so the constant and the function that produced every
recorded interval agreed only by coincidence. It now reads the constant.

**Scope corrected before acting.** The scoping said ten labels across two
recorders. It is eight across one. `stage5_auc_ci99_{low,high}` matches on *name*
only: it is a bootstrap percentile interval with its own declared level, recorded
as claim:stage5_auc_ci_confidence_pct, and is not a discordant-share interval at
all. Harmonizing it would have been a name match standing in for an identity
match, which is the failure this project has recorded before.

**The label no longer carries the level.** `stat_*_ci99_*` became
`stat_*_share_ci_*`. A name that encodes a coverage goes false the moment the
coverage moves, and nothing in the ledger can notice; the level now lives in
`SHARE_CONFIDENCE` alone, which is the only place it can drift. The name keeps
`share` because the two *scales* remain distinct even where the level does not:
a share lives in [0, 1] and a paired difference in [-1, 1].

**Nothing concluded differently.** All four intervals reproduce exactly from
`clopper_pearson_interval` at the new level, all four narrowed as a lower
coverage requires, and all four excluded 0.5 before and after. The
`corrected_ci_excludes_half` flags are computed at `CORRECTED_CONFIDENCE` and are
untouched. This changes the width of four reported intervals and no verdict.

| arm | old, 0.99 | new, 0.95 |
|---|---|---|
| Stage 2B | [0.5206, 0.5720] | [claim:stat_stage2_share_ci_low, claim:stat_stage2_share_ci_high] |
| Stage 4 derived | [0.5879, 0.7351] | [claim:stat_stage4_share_ci_low, claim:stat_stage4_share_ci_high] |
| lexical channel | [0.6581, 0.7080] | [claim:stat_lexical_share_ci_low, claim:stat_lexical_share_ci_high] |
| coupling harm | [0.5478, 0.6570] | [claim:stat_coupling_harm_share_ci_low, claim:stat_coupling_harm_share_ci_high] |

**One test was parameterized rather than re-pinned.**
`test_real_stage2_summary_matches_the_documented_anchor` hardcoded the 0.99
interval while reading the level from a global, which made it a test of the
global. Its own docstring already records this lesson for `alpha`, which it
passes explicitly after breaking twice as the family grew. Confidence now gets
the same treatment, so the anchor tests the function again and a real regression
in the interval still fails it. The pinned numbers are unchanged.

## 67 — Level 4's margin can only live where it does, and the reason is an alphabet of one

**The refinement.** Decision 63 partitioned the derived gate by oracle path and
found the margin carried entirely by the dependency-relation stratum, *not shown*
on morphology, reversed on adposition. That reading was correct and too weak. The
observation is the UPOS tag, so a margin driven by the emission requires the tag
to carry information about the label. Measured within each stratum:

| oracle path | UPOS alphabet | H(case) | I(case; UPOS) |
|---|---:|---:|---:|
| dependency relation | claim:stage4_path_deprel_upos_alphabet_size | claim:stage4_path_deprel_label_entropy_nats | claim:stage4_path_deprel_mutual_information_nats |
| morphology | claim:stage4_path_morphology_upos_alphabet_size | claim:stage4_path_morphology_label_entropy_nats | claim:stage4_path_morphology_mutual_information_nats |
| adposition | claim:stage4_path_adposition_upos_alphabet_size | claim:stage4_path_adposition_label_entropy_nats | claim:stage4_path_adposition_mutual_information_nats |

**Exactly zero, not small.** The mutual information vanishes on two paths for a
structural reason rather than a statistical one: each of those paths *defines*
the tag it selects on. The morphology path fires on `feats[Case]`, which in
English EWT marks pronouns, so every token in that stratum is `PRON`. The
token-level adposition path fires only when `upos == ADP`, so every token there is
`ADP`. The
observation alphabet inside each is a single symbol and the emission cannot
separate any two states.

**This is a known degeneracy, found in a new place.** `stage4_gold_upos_alphabet_size`
records the same alphabet-of-one for the gold population, and decision 16 already
reports every predictor scoring alike there. What was not known is that it recurs
*inside* the derived population, because nothing had partitioned it. The
manuscript's gold-population section and its new stratification section now name
the same mechanism.

**What it changes.** The claim strengthens from "the margin happens to live on
the dependency-relation path" to "it can only live there". On the other two paths
no emission-driven effect is possible, so whatever they show is attributable to
the case-to-case transition alone. That also explains why those strata are not
flatly zero: the transition still varies with sequence position while the
context-free baseline does not.

**What it does not change.** No recorded number moves. The aggregate, the three
stratum margins and every verdict stand exactly as decision 63 recorded them.
This adds the mechanism beneath them and nothing else.

**What it cannot resolve.** It says nothing about whether the case level would
work on a language that marks case on nouns. That question needs a case-marking
treebank and is carried in `TODO.md` under Major; the measurement here is the
pre-build veto that item declares, computed on the corpus actually in hand.

## 68 — scorer audit: L4 gains bounded live-pymdp equivalence; L6 later gains a distinct live score

**Version.** Originally audited on `inferactively-pymdp==1.0.3`. Re-verified on
2026-08-29 after the pin moved to `inferactively-pymdp==1.0.4` (published on PyPI
2026-08-23; no GitHub release notes at re-verification time, so behavior was
checked against the installed package, not a changelog): `infer_parameters`
remains the learning API at `agent.py:421`, `update_A`/`update_B` exist only
under `pymdp.legacy`, and the four pymdp-routed scoring paths are unchanged.
The original 1.0.3 upstream check (latest release `v1.0.3`, published 2026-06-15)
is retained below as a dated record.

**Which levels routed a recorded score through pymdp at the time of this audit.**
Four paths did: `compute_perplexity` (L1), `compute_phoneme_perplexity` (L2A),
and both `decode_positions` and `decode_phonemes` (L2B) call `infer_states` and
`update_empirical_prior`. The other headline paths did not: L3's
`word_perplexity` is an exact closed form; L4 uses a NumPy forward filter; L5 an
exact mixture; and L6's registered observed-position score indexes the fitted
conditional table directly.

**That split must be disclosed without laundering construction into inference.**
`infer_states` is one call per observation. The exact/NumPy paths can describe
the same arrays without being calls through pymdp, and each level needs its own
evidence before its score can be attributed to live inference.

**What was actually being checked, and what was not.** The Agents were built and
their *shapes* conformance-checked against the GNN specs. At L5,
`test_closed_form_oracle_and_live_agent_agree` went further and walked the
agent's matrices by hand. Its comment is exact -- "the live agent's float32
matrices walk to the same evidence" -- and its name is not: extracting `agent.A`,
`agent.B` and `agent.D` and multiplying them in numpy exercises pymdp's *storage*,
never its inference. At that point no test called `infer_states` on L4, L5 or L6
and compared the result to a reported score.

**Partially closed.** `tests/test_pymdp_inference_equivalence.py` runs pymdp's own
`infer_states` and `update_empirical_prior` at Level 4 and requires the argmax
sequence to reproduce the numpy decoder position for position. It agrees at every
one of the 184 positions in the compared prefix. Two controls sit beside it: a
normalization check, because an argmax comparison would pass just as well over
unnormalized beliefs, and an `isinstance(agent, Agent)` assertion at L4, L5 and
L6, because a builder returning a namespace with the right attribute names would
satisfy every shape-conformance test in the suite.

At L3, a separate sampled live-pymdp path-identity test supports the exact word
score. At L5, the named test walks arrays extracted from the Agent in NumPy; it
does not call pymdp inference. L5 therefore remains model-array evidence, not
headline-score inference equivalence.

**Subsequent L6 amendment.** Stage 6 now also records a *separate* descriptive
sequential predictive score through the real Agent. It resets `D` per document,
scores `A @ prior` before observing each sentence type, and then calls
`infer_states` and `update_empirical_prior`; a transition replacement is a
negative control. This closes the narrower question of whether an L6 live score
exists and depends on `B`. It is not an equivalence test for the registered
observed-position score, whose true-tercile covariate gives it a different
estimand and code path.

**What it cannot resolve.** The L4 comparison is over a bounded prefix, since one
call per observation is the scaling limit that motivated the numpy path in the
first place. A prefix that agrees everywhere is not full-set equality, and the
manuscript already states that distinction for the L2B decoder equivalence. This
raises L4 from shape conformance to inference evidence over a stated subset. It
does not do so for L5, and it is not a full-set proof. L6's later live score is
direct execution evidence for its own sequential estimand, not evidence that the
registered observed-position estimator is equivalent to live inference.

## 69 — the comparator was defensible rather than strongest, and the figure order let it ship

**Two findings from one comprehensive audit, and they compound.**

**The comparator.** Requirement 1 of the gate checklist asks for a comparator that
is context-free and not uniform. It never asks for the *strongest* comparator that
is still not the model. Searching that class instead of picking from it: a
left-context lookup, fitted on training data alone as argmax over
`(previous observed symbol, current observed symbol) -> label` with per-symbol
backoff, scores claim:comparator_stage2_lookup_accuracy% at Stage 2B against the
shipped model's claim:stage2_g2p_accuracy% and the declared bar's
claim:comparator_stage2_unigram_accuracy%. It has
claim:comparator_stage2_lookup_parameters parameters, no latent state, no
inference at decode time, and strictly less information than the filtering decode.
It beats the model by claim:comparator_stage2_over_model_pts points, several times
the model's own recorded margin. At Level 4 a table of
claim:comparator_stage4_lookup_parameters parameters beats it by
claim:comparator_stage4_over_model_pts.

**No recorded number is withdrawn**; all of them reproduce. What is withdrawn is
the interpretation that clearing the declared bar showed the latent-state
machinery doing the work. The taxonomy of `eq:gate_verdict` is untouched. What
this shows is that the *check* is not yet a procedure: a procedure would enumerate
the comparator class and place the threshold above its strongest member.

**The controls, including the weak one.** The probe's unigram backoff must
reproduce each shipped context-free bar, and does to within 0.01 points, which is
what establishes it scores the gate's own population. The derangement holds the
key set, label multiset and parameter count fixed and changes only the mapping: at
Stage 2B it collapses to claim:comparator_stage2_deranged_accuracy%, at Level 4 it
reaches claim:comparator_stage4_deranged_accuracy%, above the bar. So a
substantial share of the Level 4 result is context-conditional label frequency
rather than the mapping, and that half should be discounted relative to Stage 2B's.

**The figure order.** `docs/rendering_pipeline.md` has always stated the
constraint: record claims, hydrate tokens, generate figures, render.
`scripts/verify_release.py` ran figures *before* manuscript-variables. Because
`output/data/manuscript_variables.json` is tracked, a stale map never raises
`MissingVariableError`; it silently supplies the previous ledger. When decision 66
moved the share intervals from 0.99 to 0.95, `multiplicity_forest.png` shipped
drawing 99% whiskers over a ledger holding 95%, past a green suite, in the figure
whose entire subject is the multiplicity correction. `multiplicity.py:346` also
hardcoded the level in its axis label while line 371 interpolated the constant, so
regenerating alone would have produced a panel disagreeing with itself.

**Both halves fixed.** The literal is gone, the figure regenerated and inspected,
the steps swapped, and the guard rewritten: it asserted set membership, which
cannot see order, and now asserts the index of `manuscript-variables` is below
that of `figures`.

## 70 — the suite's cost is one test, and six of its eight skips were the environment

**Speed.** Half the suite ran in 28:08 while the other half ran 2,541 tests in
4:25. Six times the wall clock for half the tests, so the cost was concentrated
rather than spread. It is one test:
`test_claims_single_writer.py::test_rerunning_the_recorders_is_a_no_op` launches
every claims recorder as its own subprocess against an isolated ledger, each
paying full JAX and `pymdp` import before doing any work, and one of them
re-measures Stage 2. Deselecting it takes that half from **28:08 to 2:53**.

**It stays serial.** The recorders write one ledger under a single-writer lock,
and the property under test is that no recorder overwrites another's label. Give
each recorder its own copy so they can run in parallel and the test can no longer
see the conflict it exists to catch. Faster would mean weaker, so it stays slow.

**The fix is discoverability, not optimization.** The `recorder_replay` marker
already existed, was already declared in `pyproject.toml`, and
`scripts/verify_release.py` already excluded it from the quick profile and kept it
in the full one. What was missing is that none of this was visible from where
somebody iterating would look. `docs/quickstart.md` now carries the command, the
measured saving, and the reason the test cannot be parallelized, so the next
reader does not rediscover it by watching a run take half an hour.

**This also explains the runs that kept dying.** Full-suite runs were repeatedly
killed at 11%, which is inside this file. They were not dying *near* it.

**Skips.** Eight, of which six were environmental and are now zero. Four tests in
`test_benchmarks.py` and two elsewhere guard on `ollama_available("gemma3:4b")`,
which `skipif` evaluates at collection time, so a sandbox blocking
`localhost:11434` silently converted them to no-ops. Run where the model server is
reachable, `test_benchmarks.py` reports 70 passed and 0 skipped. That is the
dangerous shape of skip: a benchmark check that quietly does not run is
indistinguishable, in a summary line, from one that passes.

**The remaining two are correct and should stay.** Both are in
`test_complexity_forms.py`, where the parametrized workloads of length one carry
no push term, so the negative control cannot fail for a reason that means
anything. Skipping beats asserting something vacuous. The quickstart now records
that a clean run reports exactly two skips, which turns any third into a signal
rather than noise.

**Subsequent correction.** Reachability is no longer evaluated by a `skipif` at
collection time. Live Ollama tests require the explicit
`AILM_RUN_LIVE_OLLAMA_TESTS=1` contract, then check model availability inside the
test body. The default suite therefore makes no speculative call to a user's model
server, while an opted-in run still reports an unavailable model as an explicit
skip reason. Recorder tests additionally bind `AILM_CALL_CACHE_PATH` to their
isolated ledger directory, so a test cannot warm or append to the publication
store. The fixed skip census above is a historical observation, not a current
acceptance criterion; the quickstart now requires reviewing the reported reasons.


## 71 — Holm joins Bonferroni, and moves one member across the line

**The manuscript had already scoped this open.** The Stage 6 fragility list named
Holm's step-down correction as one of four directions the verdict could move in,
observed that it can only admit more members than Bonferroni and never fewer, and
then declined to claim anything: "whether it would admit *this* member is not
recorded, and is therefore not claimed in either direction here." Recording it
closes the one direction that was left open rather than closed.

**It admits it.** Over the same claim:stat_family_size declared members at the
same familywise claim:stat_familywise_alpha, Holm rejects
claim:stat_holm_survivor_count where the fixed threshold rejects
claim:stat_family_survivor_count. The gain is `stage6_sign_test_p`: at rank
claim:stat_holm_stage6_rank of the ascending family it is compared against
claim:stat_holm_stage6_threshold rather than claim:stat_bonferroni_alpha, and its
recorded p of claim:stage6_sign_test_p crosses the first and not the second.

**This one is different in kind from the other three fragilities.** Sidedness and
the choice of estimand change what is being tested; collapsing the subsample
re-analyzes changes what the family contains. Holm changes only how a fixed
family of fixed two-sided p-values is compared against a fixed familywise level.
Its verdict is therefore available under the declared rule rather than by
relaxing it, which is why it is reported as a result and the other three remain
reported as fragilities.

**Both corrections stay.** Not as a hedge: they have different properties and the
project needs both. `BONFERRONI_ALPHA` is a constant, so a regression test can
assert against it and a reader can check one p-value against one number. A Holm
threshold depends on where every other member fell, so a verdict can change
because an unrelated measurement moved — which is informative in a paper and
unusable in an automated gate. The recorder refuses to write a Holm survivor set
that is not a superset of the Bonferroni one, because that outcome is impossible
and would mean the implementation is wrong.

**What it does not buy.** Every caveat on the fixed threshold transfers intact.
The family was assembled retrospectively on the development population, the
paired units nest inside the units their corpora sample, and a familywise
correction bounds a fixed family's Type-I rate rather than restoring
independence to a reused split. Stage 6 moves from *nominally significant* to
*step-down survivor*, and neither label means confirmed.

## 72 — a caption is not an image alternative

**The convention was wrong on its own terms.** `docs/syntax_guide.md` said
captions "appear in the PDF and as alt text in HTML", so no figure carried a
separate alternative. But a caption in this paper names the population, the
provenance and the falsification condition; several run past three hundred
words. Read aloud in place of a picture, the one thing a screen-reader user
wants — what is on the canvas — arrives last if at all.

**Nothing on this side of the boundary noticed.** The full suite was green while
the HTML stage refused fourteen files, once per figure, because the template's
accessibility layer fails closed when a registry record exists without
`metadata.alt_text` and this project's generator never wrote the field. A gap
visible only to another repository's renderer is a gap with no gate.

**Alternatives are authored, not derived.** `data/figure_alt_text.yaml` holds one
sentence per label. Deriving them from the first sentence of each caption was
the cheap option and was rejected: five of the thirty-two open on a `{{TOKEN}}`
or inline math, so the derived text would read a ledger number aloud through a
channel no claim gate can audit, and it would go stale silently when the number
moved. `tests/test_figure_alt_text.py` bans tokens, citation keys and math
outright, bounds the length so a pasted caption fails, and checks the authored
map and the published registry name the same figures in both directions.

**The registry generator now requires the field.** The template's builder treats
`alt_text` as optional and simply omits it when absent, which is what let the
omission ship. `FigureSpec` makes it mandatory, so a new figure without an
alternative fails at generation rather than at somebody else's render.

## 73 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: beating the unigram bar is not yet using the previous letter

**Provenance correction (2026-08-23).** The prediction and recorder first appear
in the same Git commit. Their within-session ordering may document prospective
intent, but it is not independently timestamped and cannot earn preregistered or
confirmatory status. The measurement is exploratory; the exact commit identity
is checked from `data/registration_provenance.json`.

**The gap.** An external audit read the Stage 1 prose — "a model can beat uniform
while carrying no context at all, but it cannot beat unigram letter frequencies
without using the previous letter" — as a theorem, and it is not one. The unigram
bar is *estimated on the training counts* and *scored on a finite held-out
passage*. On a finite sample a context-free distribution other than the
training-estimated unigram can score below that bar through distribution shift
alone; nothing about clearing the bar, by itself, attributes the gain to
conditioning on the previous letter. The comparison claim (the agent scores
claim:stage1_letter_perplexity against the unigram bar's
claim:stage1_unigram_baseline on the same positions) stands exactly as recorded.
What needs repair is the *attribution* sentence, and prose alone cannot repair
it: attribution is an experimental claim, so it gets a control.

**The control: destroy the context, keep everything else.** Take the trained
letter transition — the (30, 30) column-stochastic matrix whose column *j* is
the conditional distribution over successors of letter *j* — and rotate its
columns by one position over the sorted index order. The multiset of conditional
distributions is preserved exactly (same 30 columns, same normalization, same
parameter count); the only thing destroyed is *which context indexes which
conditional*, and a rotation over 30 distinct indices has no fixed points, so
every context is mis-assigned. This deranged model still reads the previous
letter at every position; it just conditions on a systematically wrong lookup.
Beside it, one more comparator is measured: the context-marginalized
distribution, the trained transition's columns averaged under the training
unigram weights — a context-free distribution *derived from the same trained
matrix*, which is exactly the class of "other context-free distribution" the
audit's objection invokes. Both are scored on the identical held-out positions
the Stage 1 gate scores (the `HELD_OUT_TEXT` passage, through the production
`compute_perplexity` path for the transition-shaped models and the closed form
for the context-free ones).

**What will be recorded**, all by the one single-writer recorder under
`l1_context_*` labels: the scored position count, the freshly re-measured
trained and unigram perplexities (guards, see below), the deranged-context
perplexity, the context-marginalized perplexity, both gaps to the unigram bar,
and a confirmation flag.

**Refusal conditions.** The recorder refuses to write anything if (a) its fresh
trained perplexity does not reproduce claim:stage1_letter_perplexity within the
gate test's own tolerance, or its fresh unigram bar does not reproduce
claim:stage1_unigram_baseline within 0.01 — either mismatch means it is scoring
a different population than the gate; (b) the deranged model scores *better*
than the trained model — mis-assigned context beating correctly-assigned context
means the measurement, not the model, is broken; (c) the context-marginalized
matrix is not actually context-free (its predictive must be identical at every
position by construction).

**The falsifiable prediction.** (P1) The deranged-context model scores **no
better than the unigram bar** on the held-out passage — destroying the
context-to-conditional assignment costs at least the whole bigram margin. If the
derangement lands at or worse than unigram, the attribution stands in its
repaired, control-carried form: the trained model's margin over unigram is
attributable to correctly-assigned previous-letter conditioning, because the
same conditionals mis-assigned forfeit it. (P2, the specified falsifier) If the
deranged model *beats* the unigram bar, that is recorded — flag 0, claims
written, not an error — and the L1 attribution is weakened to a pure comparison
claim: the prose must then say the gain cannot be localized to the
context-conditional assignment on this passage. No prediction is specified for
the context-marginalized comparator in either direction; whether a context-free
derivative of the bigram lands above or below the count-estimated unigram is
itself the audit's point, and its value is reported as measured.

**The prose change this licenses.** `docs/manuscript/03a_gate_design.md` and
`docs/manuscript/04_letter_level.md` replace "cannot beat unigram without using the
previous letter" with the comparison claim plus this control's recorded numbers.
The manuscript's attribution sentence cites the control, not the bar.

**OUTCOME (2026-08-20, recorded from the run).** P1 held, and not narrowly. On
the claim:l1_context_positions gate positions the deranged-context model scores
claim:l1_context_derangement_perplexity against the unigram bar's
claim:l1_context_unigram_perplexity — mis-assigning the conditionals does not
merely forfeit the trained model's margin, it lands far above uniform, because
peaked conditionals indexed by the wrong context are confidently wrong.
`l1_context_control_confirmed` = 1, and both population guards reproduced their
shipped values (trained claim:l1_context_trained_perplexity against
claim:stage1_letter_perplexity; the bar to within 0.001). The unregistered
comparator delivered the audit's point as a measurement: the trained
transition's context-marginalized distribution scores
claim:l1_context_marginal_perplexity, which is
claim:l1_context_marginal_minus_unigram units from the count-estimated bar — a
context-free distribution, derived rather than count-estimated, sitting
(microscopically) *below* the training unigram on this passage. Beating the bar
is therefore demonstrably not proof of context use; surviving the destruction
control is. The manuscript's attribution sentences now carry the control's
numbers, per the prose change this entry licenses.

## 74 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: a family-disjoint G2P split, priced against the shipped interleave

**Provenance correction (2026-08-23).** The direction and recorder first appear
in the same Git commit. Their within-session ordering may document prospective
intent, but it is not independently timestamped and cannot earn preregistered or
confirmatory status. This remains an exploratory robustness diagnostic; the
exact commit identity is checked from `data/registration_provenance.json`.

**The exposure.** Decision 4 chose an interleaved stride for the G2P split, and
the ledger records what the interleave does not remove:
claim:stage2_g2p_split_shared_prefix_pct% of held-out words share a
four-character prefix with some training word. The manuscript states this as a
bound on independence rather than a verdict of contamination — on a task whose
subject is letter-to-phone regularity the overlap is partly the phenomenon — but
stating a bound is weaker than measuring what sits inside it. The Stage 2B
margin (claim:stage2_g2p_margin_pts points, McNemar p =
claim:stage2_g2p_mcnemar_p) could be carried in unknown part by morphological
neighbors: `catalog` in training teaching `catalogs` in test.

**The arm: assign whole prefix families to one side.** Group the vendored
`g2p_pairs` by their first four characters (the word itself when shorter — the
same grouping unit as the recorded shared-prefix diagnostic). Sort the family
keys, assign every fifth family to TEST and the rest to TRAIN — the same
deterministic-stride, no-seed construction as the shipped split, lifted from the
word level to the family level. Everything downstream is identical to the
shipped pipeline: the same equal-length pair filter, the same emission /
transition / start fit on the train side only, the same per-letter argmax
comparator fitted on the same train side, the same forward-filter decode over
the same positions, the same exact McNemar on the discordant pairs. The decode
runs through the closed-form numpy oracle whose position-identity with the
pymdp agent was proven at the canonical split (decision 10's machinery in
`scripts/record_stage2_claims.py`); this arm re-fits the matrices but not the
model class, and it is recorded as a diagnostic, so the equivalence proof is
inherited rather than re-run.

**What will be recorded**, under `stage2_g2p_famsplit_*` labels by one
single-writer recorder: the family count and stride, train/test pair counts, the
new split's shared-prefix percentage and initial-letter TVD (the same two
diagnostics the shipped split records), evaluation positions, model and
context-free-lookup accuracies, the margin, the discordant counts, and the exact
McNemar p.

**Refusal conditions.** The recorder refuses to write anything if (a) the
family-disjoint split's shared-prefix percentage is not far below the shipped
82.5 — the construction makes it exactly zero for words of length four or more,
so anything above 5% means the split function does not do what this entry says;
(b) the model and comparator position lists differ in length — an unpaired
comparison cannot feed McNemar; (c) either side of the split is empty.

**Multiplicity, declared now.** The famsplit p-value does **not** join
`SIGNIFICANCE_FAMILY`. That family is frozen and its Bonferroni threshold is
load-bearing; this arm re-tests the *same* Stage 2B hypothesis on a stricter
split rather than posing a new one, so entering it would count one hypothesis
twice. It is recorded as an exploratory robustness diagnostic, excluded from the
repository-wide family with a documented reason in `tests/test_statistics.py`,
and the manuscript reports it as a bound on the leakage story, not as a second
significance claim.

**The falsifiable prediction.** (P1, direction specified) The famsplit margin
**shrinks** relative to the shipped margin of claim:stage2_g2p_margin_pts points:
some nonzero part of the Stage 2B margin is morphological-neighbor transfer, and
removing the neighbors removes that part. Whether the remainder survives — a
positive margin, a null, or a reversal — is the question, and all three are
admissible outcomes. (P2) The context-free comparator itself drops on the
disjoint split (its per-letter table also benefits from neighbors), so the
margin, not either accuracy alone, is the quantity the prediction is about.

**What this does not claim.** It is not a replacement for the shipped split —
the shipped interleave measures the dictionary the level actually models,
neighbors included, and its diagnostics are recorded. It is a bound: whatever
the famsplit margin is, morphological-neighbor transfer cannot account for more
of the shipped margin than the difference between the two.


**OUTCOME (2026-08-20, recorded from the run).** P1 held: the margin shrinks and
survives. With whole four-character families assigned to one side (the split's
own shared-prefix diagnostic reads claim:stage2_g2p_famsplit_shared_prefix_pct%,
exactly the construction's zero), the model scores
claim:stage2_g2p_famsplit_accuracy% against the train-side lookup's
claim:stage2_g2p_famsplit_baseline_accuracy% over
claim:stage2_g2p_famsplit_eval_positions positions — a margin of
claim:stage2_g2p_famsplit_margin_pts points against the shipped
claim:stage2_g2p_margin_pts. About thirty percent of the shipped margin is
therefore attributable to morphological-neighbor transfer, and about seventy
percent is not: the discordant counts are claim:stage2_g2p_famsplit_mcnemar_b
against claim:stage2_g2p_famsplit_mcnemar_c, exact McNemar p =
claim:stage2_g2p_famsplit_mcnemar_p. P2 held too — the comparator drops
alongside the model (both accuracies are lower than their interleave
counterparts), which is why the margin and not either accuracy carries the
finding. The price the family assignment pays is visible in the split
diagnostics: assigning whole alphabetical families moves the initial-letter TVD
to claim:stage2_g2p_famsplit_initial_tvd, well above the interleave's
claim:stage2_g2p_split_initial_tvd — a real covariate cost, recorded rather than
hidden, and the reason this arm bounds the shipped result instead of replacing
it. Per the multiplicity declaration above, the p-value is exploratory: it is
excluded from the frozen family and the manuscript cites it as a robustness
bound, not as a familywise survivor.

## 75 — the gate taxonomy carries its comparator's type, and impossibility now requires a bound

**Context.** An external deep-research audit of the manuscript found a formal
inconsistency this project's own prose had already half-admitted: Definition
"Threshold placement" licensed `UNREACHABLE` whenever the measured ceiling
failed the threshold, while the gate-audit table's own caption said a reference
comparator "can be exceeded, and at Stage 4 it is". Those two positions cannot
support the same verdict: `c_reference ≺ t` does not bound the model class, so
no impossibility claim follows from it. The comparator-type column existed as
prose; the classifier, the Lean development, and the verdict vocabulary did not
carry it.

**Decision.** The typing is now structural, end to end. `ComparatorType` in
`src/ailm/gates.py` declares every ceiling one of `model_class_bound`,
`analytic_bound`, `privileged_oracle`, `reference_model`; `classify_gate`
refuses an untyped measured ceiling (fail-closed), and the verdict set is six:
`VACUOUS`, `UNREACHABLE` (bound fails t), `ABOVE_REFERENCE` (exceedable
comparator fails t — placement, no impossibility), `UNAUDITED`,
`BRACKETED_BOUND`, `BRACKETED_REFERENCE`. All eight audited rows declare their
types (`scripts/audit_gates.py`, mirrored in `output/reports/gate_audit.json`):
Stage 1, 2A, and generation carry bounds; 2B and 4 carry the smoothing
reference; 3, 5, and 6 carry privileged oracles. The Lean development was
rewritten to match — `ComparatorKind` with `isBound`, the six-case `Verdict`,
and two new theorems that are the repair itself: `unreachable_requires_bound`
(an impossibility verdict is derivable *only* from a bound-kind comparator,
whatever the numbers) and `above_reference_requires_exceedable` (its dual: a
bound is never softened into a reference statement). NC10 exhibits the kind
being load-bearing on recorded data: the same Stage 1 triple classifies
`UNREACHABLE` under its bound kind and `ABOVE_REFERENCE` with the kind flipped.

**One recorded verdict changed.** The Stage 4 declared gate (>=60%) on the
derived population was `UNREACHABLE`; it is now `ABOVE_REFERENCE`
(`stage4Derived_declared_gate_above_reference` in `formal/AILM/Instances.lean`),
because every comparator measured there — smoothing, Viterbi, the test-fitted
context-free lookup — is exceedable. The bar sits above everything measured,
and that is all the record supports. Stage 1's `UNREACHABLE` stands: its
comparator is the evaluation passage's own conditional entropy, a model-class
bound on those positions by the held-out-bound decomposition.

**Also adopted from the same audit finding.** `evaluate_gate` takes an optional
`min_effect` (default 0.0, reproducing the strict comparison exactly): a
prospective smallest effect of interest on the metric's own scale, so a future
gate's outcome can require "beats the baseline by enough to matter" rather than
"beats it at all". The eight retrospective rows use the default deliberately —
their thresholds were restated after measurement, and a nonzero `min_effect`
there would be as retrospective as the threshold.

**What this does not change.** No recorded number moved. The baseline-relative
disclosure (decision on `gate_is_baseline_relative`) is unchanged: with
`t := b`, a bracketed verdict of either strength still carries exactly one
measured comparison. The earlier rationale for freezing the verdict enum —
"a new case would invalidate the Lean proofs" — was written for the
baseline-relative predicate, not for this split; the proofs were not a reason
to keep a taxonomy that overclaims, so they were rewritten instead
(126 theorems, sorry-free, `formal/axiom_report.txt` regenerated at 132
verdicts).

## 76 — L4 human validation is declined; the case-marking treebank arm supersedes it

**Context.** The external audit asked for construct validation of the L4 role
labels, offering two routes: human annotation of a stratified token sample, or
ground truth from a treebank that marks case morphologically. TODO carried the
choice as "run it or decline it with an argument".

**Decision (principal, 2026-08-21).** Declined. The argument: human annotators
recruited to apply the eight-role ontology from its own definitions would be
scored against the same heuristic construct under test — high agreement would
certify that the definitions are teachable, not that the construct is
grammatical case, and decision 63 already shows the derived margin lives in the
stratum whose labels a dependency rule generated. A case-marking treebank gives
the level ground truth rather than agreement with a heuristic, which is why
that arm survived adversarial review as the strongest item in the queue while
annotation did not appear in it at all. One route answers the construct
question; the other measures inter-rater teachability of a definition. The
treebank arm is retained; the annotation arm is closed, not deferred.

## 77 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: Level 4 scored against real case morphology, on Czech

**Status: exploratory.** The confirmatory family (`SIGNIFICANCE_FAMILY`) is
frozen and this arm's p-values do not join it. Whatever the outcome, it is
reported as an exploratory finding beside the English results, never as a
confirmed effect, and no English number moves.

**Provenance correction (2026-08-23).** The population, comparator, endpoint,
and refusal rules were specified prospectively within the development session
before the sequence-model score, but only after feasibility and development-set
inspection. Decision 77, the generator, model surface, artifact, recorder, and
result first enter Git history in the same commit. No independently timestamped
immutable prior artifact establishes the ordering outside that session, so the
earlier `PRE-REGISTRATION` heading overclaimed the provenance. The artifact now
carries an exact `study_design` object stating
`prospective-development-session-specification`,
`independently_timestamped=false`, and `immutable_prior_artifact=null`; its
loader permits `preregistered` terminology only with a content-addressed,
timezone-stamped prior artifact. This correction changes no measurement and
leaves the arm exploratory.

**Sampling-unit correction (2026-08-23).** The first clustered reanalysis
mistook each retained sentence-level sequence for an independent unit. The
authenticated dev shard carries `# newdoc id` metadata: its
claim:stage4_czech_test_sequences retained sequences nest within
claim:stage4_czech_test_documents contributing source documents. The outer
document is therefore the primary bootstrap unit. Artifact schema v3 retains
both the source document identities and one document index per sequence; its
loader rejects a planted one-document-per-sequence mapping. The earlier
sequence-clustered interval is retired rather than retained as a competing
primary.

**The question, prospectively specified before the sequence-model score.** Does
the sequence mechanism's margin over a context-free comparator survive when
the labels are real morphological case rather than the output of a syntactic
heuristic? Decision 63 showed the English derived margin lives entirely in the
stratum whose labels a dependency rule generated; the morphology stratum was an
unresolvable control. This arm is the same model shape on a corpus where every
label is morphology, sized so the answer resolves either way.

**Corpus and feasibility, measured before this entry was written.** UD
Czech-PDTC (staged locally; dev 22,666 sentences, train-ft shard, test
untouched). The four informativeness checks from
`scripts/probe_l4_case_marking_treebank.py`, run on the subsequently specified population
(below): UPOS alphabet over gold-case tokens |Σ| = 6 (English gold: 1);
H(Case | UPOS) = claim:stage4_czech_conditional_entropy_nats nats with
I = claim:stage4_czech_mutual_information_nats nats (English gold: 0); a
context-free bar of claim:stage4_czech_baseline% against a
claim:stage4_czech_cf_oracle% fit-on-test ceiling and a
claim:stage4_czech_majority% majority — headroom,
not a ceiling (the English deprel rung died at a 99.59% bar); held-out gold
tokens claim:stage4_czech_test_tokens against the ~2,060 the
420-discordant-pair arithmetic requires (82x). Czech carries 7 of the 8 CEREBRUM cases (no ablative); Voc is rare
(69 dev tokens).

**The specified population, including one exclusion.** Scored tokens are those
with UPOS in {ADJ, DET, NOUN, NUM, PRON, PROPN} and `feats[Case]` in the
8-value UD inventory. ADP is excluded although Czech PDT annotates `Case=` on
prepositions: there the feature records *government* — the case of the
complement, not inflection of the token itself — which is the structural
analogue of the English token-level adposition path whose margin decision 63 found
reversed. Including it would re-import the one label source this arm exists to
escape. Sentences contribute their scored tokens in order; sequences shorter
than 2 are dropped (as in the English arm).

**The vendor rule, and why it does not spring the case-oracle trap.** The
vendored artifact (`vendor/ud_czech_case_counts.json`, generator
`scripts/generate_czech_case_counts.py`) retains UPOS (the observation) and
`feats[Case]` mapped to the 8-role inventory (the label), plus the authenticated
`# newdoc id` alphabet and one document index per retained sequence. FORM,
LEMMA, XPOS, HEAD, DEPREL and MISC are not in the artifact at all. Decision
15's trap was an observation that reads the label's *generator* —
English derived labels were a deterministic function of deprel + adposition
lemma. Here the label comes from morphology and the observation is a different
column whose information about the label is measured at
claim:stage4_czech_mutual_information_nats nats; nothing
deterministic connects them, and the artifact physically cannot leak fields it
does not carry. The raw .conllu files (90 MB, CC BY-NC-SA) are not vendored;
the derived counts artifact is the committed surface, as for every other corpus
here.

**Split.** The train-ft shard (claim:stage4_czech_train_sequences usable sequences
within claim:stage4_czech_train_documents source documents,
claim:stage4_czech_train_tokens gold tokens) fits the model and the comparator;
dev (claim:stage4_czech_test_sequences retained sequences within
claim:stage4_czech_test_documents contributing documents) is the held-out scoring set; test is not read
by any script in this arm. No stride is needed — the corpus ships the split.

**Model.** The identical construction to the English arm, reusing the builders
in `ailm.levels.case` unmodified via a duck-typed artifact
(`src/ailm/levels/case_czech.py`): B = P(case_t | case_{t-1}) column-stochastic
with the default smoothing, A = P(UPOS | case), D = the first-token case
distribution, forward-filtering decode (decision 8).

**Comparator.** Context-free argmax P(case | UPOS), fitted on the same train-ft
tokens, scored on the identical held-out positions (decision 2).

**Endpoint and sampling unit.** Per-token accuracy against morphological gold —
accuracy, not agreement, because the labels are ground truth rather than a
research claim. The token-weighted margin is the estimand. Its primary
uncertainty resamples each authenticated source document as one outer cluster.
A whole-document sign-flip sensitivity analysis is also recorded; unlike
pair-level McNemar, it is finite-sample exact only if independent whole-document
difference vectors are jointly centrally symmetric. The position-level exact McNemar (decision 12),
recorded with b, c and a finite log-tail, is secondary and conditional on
treating tokens as independent despite their nesting in sequences and documents.

**Refusal guards in the single writer** (`scripts/record_czech_case_claims.py`):
no claims are written if the observation alphabet degenerates to one symbol, if
any dev UPOS or case value is unseen in training, if the held-out count falls
below the 420-pair arithmetic's requirement, or if any scored token's label
source is anything but morphological `feats[Case]` (the coverage this arm
exists to have; English gold covered 25% of scored tokens, this must be 100%).

**What either outcome means.** A surviving margin: the sequence mechanism adds
information about real case beyond what UPOS alone carries, on ground truth,
where the English arm could show it only against its own heuristic. A collapsed
margin: the English margin's dependence on heuristic labels generalizes — the
mechanism has no demonstrated purchase on real case anywhere, which is the
strongest form of this level's null. Both are recorded; neither rescues or
retires any English number.

**What this arm cannot resolve** (from the queue entry, carried forward): it
cannot rescue a single English number; it makes L4 the stack's only non-English
level, so decision 52's orphan finding worsens — measurable at the price of
unconnectable, accepted here explicitly; and it replaces one mapping question
(deprel → role) with another (UD `Case=Ins` → INS), which stays a research
claim about the inventory even though the label itself is morphology.

## 78 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the third degeneracy — action-independent transitions cancel a nonzero epistemic term

**Provenance correction (2026-08-23).** The derivation and recorder first appear
in the same Git commit. Their within-session ordering may document prospective
intent, but it is not independently timestamped and cannot earn preregistered or
confirmatory status. These claims remain analytic or descriptive; the exact
commit identity is checked from `data/registration_provenance.json`.

**Context.** The adversarial pass on the research queue's epistemic-generation
item returned BUILD_MODIFIED (docs/research_queue.md, "Epistemic generation").
The item proposes a generative agent whose hidden factor is word identity, so
that "information gain is nonzero by construction", and expects the epistemic
term to then do work in policy selection. The skeptic's derivation says it
cannot — not because the information gain is zero, but because it is the *same*
for every action. This entry states that derivation's testable consequences
before the in-session re-verification, rather than trusting either the item or
the skeptic.

**The construction.** Hidden factor: word identity over the shipped Stage 3
lexicon. Action: the emitted letter, one per symbol. `A`: the
prefix-conditioned next-letter map — each vocabulary word consistent with the
committed prefix emits its next symbol (the following letter, or the
terminating space) deterministically, exactly the marginalization
`lexicon_preference` performs. `B[:, :, k] = I` for every action `k`, because
the level invariant is that a word does not become a different word part-way
through being spelled (`build_word_transition`,
`src/ailm/levels/word.py`); choosing which letter to emit changes what is
*observed*, never which word is being spelled. Belief: the trie-node posterior
— the word prior restricted to the prefix-consistent support, renormalized.
Everything runs through the standalone `expected_free_energy_terms` path
(`src/ailm/generation.py`), never a pymdp `Agent`: a dense per-prefix `B`
would spend hundreds of megabytes of the `MAX_AGENT_VOCAB_BYTES` budget on a
constant, so the identity is broadcast, not materialized.

**Prefix selection rule, fixed in advance.** The empty prefix, plus, for each
depth 1 through 4, the prefix with the greatest word-prior mass among trie
nodes at that depth having at least two distinct next symbols. Deterministic
given the artifact; no hand-picked prefixes.

**Predictions.**

- **(a)** Per-action information gain is strictly positive at every selected
  node — the trie-node posterior carries real uncertainty and the next-symbol
  observation resolves some of it.
- **(b)** The per-action *spread* (max minus min) of information gain, of
  pragmatic value, and of `G` is exactly `0.0` at every prefix.
  Action-independent transitions make the predictive state distribution — and
  therefore the predictive observation distribution — identical across
  actions, so both terms of [@eq:efe] are constants over the policy space.
- **(c)** The induced policy posterior `softmax(gamma * neg_efe + ln E)`
  equals `softmax(ln E)` — the pure habit — to machine precision (max absolute
  deviation below 1e-12) at the shipped operating precision.

**Refusal conditions.** The recorder refuses to write anything if (b) or (c)
fails — a nonzero spread means the derivation is wrong and the manuscript
extension must not ship. It also refuses if (a) fails at any selected node
(the rule would have selected an unambiguous node, and "nonzero yet
action-inert" would have no exhibit), and if the action-DEPENDENT control —
same states, same belief, same `A` and `C`, a transition that genuinely
differs by action — fails to produce a strictly positive per-action spread of
both information gain and `G`. A control that cannot fire turns the recorded
zeros into decoration, the same rule the generation recorder applies to its
stochastic-transition control.

**Status of the claims: analytic/descriptive.** No inferential test is run or
specified. The recorded quantities instantiate an identity: they are
measured so the vanishing spread is demonstrated on the shipped artifact
rather than asserted, and so a property test can reject the general class
(any action-independent transition stack, not the identity specifically).

**What this licenses.** The zero-information region of
`docs/manuscript/08_generation.md` gains the third degeneracy: a hidden state
carrying real uncertainty still cannot make the epistemic term action-bearing
when transitions are action-independent — nonzero information gain that
differentiates nothing. The stated consequence: action-dependent epistemic
value over a fixed word state requires an environment that already holds the
true word — interrogation, not generation — with the budgeted-attention arm
cross-referenced as the measured regime where information gain alone drives
selection and loses.

**OUTCOME (2026-08-21, recorded from the run).** All three predictions held,
and the refusal machinery earned its keep once: the recorder's first run
refused on (c) at a measured deviation near 3e-8, which turned out to be the
float32 habit matrix — the reference softmax was computed at float32 precision
while the induced one had been promoted, so the comparison was measuring the
promotion, not the cancellation. Both softmaxes now run in float64 and the
deviation is claim:generation_word_state_policy_habit_max_abs_dev. (a) The
selection rule chose *t*, *th*, *the*, *with* — agreeing with the skeptic's
independently chosen nodes where they overlap, to six decimals — and
information gain is positive at every node:
claim:generation_word_state_ig_empty_nats nats at the empty prefix over the
full support of claim:generation_word_state_support_empty words, down to
claim:generation_word_state_ig_depth4_nats nats at *with*. (b) The per-action
spread of information gain, pragmatic value, and G is exactly 0.0 at every
prefix — bitwise, not merely small — and the action-dependent control spread G
by claim:generation_word_state_control_g_spread_nats nats, so the zeros were
recorded by machinery that can report a nonzero.
`tests/test_generation_word_state.py` pins the general class with randomized
instances and a negative control. The manuscript extension this entry licenses
is in `docs/manuscript/08_generation.md`; the queue item's verdict
(BUILD_MODIFIED) and what was built instead are recorded at
`docs/research_queue.md`.

## 77 (outcome) — the margin survives on real case morphology

**The prospectively specified question resolved, in the surviving direction.** On UD
Czech-PDTC, with every label read from morphological `feats[Case]` and the
identical builders the English arm uses, the sequence model scores
claim:stage4_czech_accuracy% against the context-free
`argmax P(case | UPOS)` comparator's claim:stage4_czech_baseline% over
claim:stage4_czech_test_tokens held-out tokens. The primary document-clustered
analysis preserves an effect of claim:cluster_stage4_czech_effect percentage points over
claim:cluster_stage4_czech_clusters contributing source documents. Its
claim:cluster_stage4_czech_confidence_pct% percentile cluster-bootstrap interval
is [claim:cluster_stage4_czech_ci_low,
claim:cluster_stage4_czech_ci_high]. Its assumption-conditional whole-document
sign-flip sensitivity analysis reaches
the claim:cluster_stage4_czech_p_floor Monte Carlo resolution floor using
claim:cluster_stage4_czech_resamples draws at seed
claim:cluster_stage4_czech_seed. The finite statement is therefore
`p <= floor` under joint central symmetry; it is not a zero probability and the
symmetry is stronger than a zero expected document sum.

**Secondary conditional analysis.** Treating tokens as independent despite
their nesting in sequences and documents gives a margin of
claim:stage4_czech_margin_pts percentage points, McNemar
b = claim:stage4_czech_mcnemar_b against c = claim:stage4_czech_mcnemar_c,
with claim:stage4_czech_discordant_n discordant pairs. The exact two-sided
p-value underflows a double, so the ledger carries the finite log beside it:
log10 p = claim:stage4_czech_mcnemar_log10_p, computed in log space from the
same exact binomial. This conditional tail does not replace the
document-clustered primary.

**The stronger reading, and its measured limit.** The model also sits above
claim:stage4_czech_cf_oracle% — a context-free predictor fitted on the test
labels themselves, the ceiling of the whole comparator class — so the margin
is not a better lookup; it is attributable to the case-to-case transition.
That is the mechanism Level 4 always claimed, shown for the first time against
labels no heuristic generated, where English could show it only on the stratum
its own dependency rule labeled (decision 63). The limit is equally measured:
I(case; UPOS) = claim:stage4_czech_mutual_information_nats nats against a
label entropy of claim:stage4_czech_label_entropy_nats, so the absolute level
is low and a model observing word forms would sit far higher — the claim is
about the transition structure, not about case tagging being solved.

**Both negative controls transfer.** A cyclically deranged emission and a
deranged transition both fall below the context-free baseline in
`tests/test_czech_case.py`, so the surviving margin is destroyed by exactly the
mis-wirings that should destroy it.

**Status unchanged: exploratory.** Both p-values stay outside the frozen
`SIGNIFICANCE_FAMILY`; no English number moves; the arm's declared
non-resolutions (orphan level, inventory mapping as a residual research claim)
stand as prospectively specified development-session limitations. The vendored artifact is
`vendor/ud_czech_case_counts.json` (sha256 in `docs/provenance/sources.md`,
byte-identical on regeneration); the staged raw shards under
`vendor/staging_ud_czech/` are now gitignored and remain uncommitted.

## 79 — PRE-REGISTRATION: the confirmatory replication protocol, with an honest census of what can still be held back

**Status: registered before the run exists.** The external audit's central
statistical objection stands and is restated in the manuscript's own words: the
sixteen-member `SIGNIFICANCE_FAMILY` was assembled retrospectively, on the
population every gate was revised against, and no correction factor turns that
into confirmation. The only repair is a frozen family run exactly once against
data nothing here has scored. This entry registers that run — its populations,
its per-member endpoints, and its one-run discipline — and records, bluntly,
which members can and cannot be re-scored from the artifacts this repository
vendors. The freeze is executable: `CONFIRMATORY_REPLICATION` in
`src/ailm/statistics.py` maps every family member to exactly one declared
population, `tests/test_statistics.py` holds it equal to the family, and this
entry is the registration it points at.

**The census, member by member.** A held-back partition must come from data
outside every scored evaluation set *and* outside every training fit; the
census below is of the vendored artifacts as they stand.

- **CMUdict members (8): no held-back partition exists.**
  `channel_letter_predictive_paired_p`, `channel_lexicon_evidence_paired_p`,
  `fitted_paired_p`, `geometric_letter_paired_p`, `geometric_lexicon_paired_p`,
  `coupling_mcnemar_p`, `lexical_best_mcnemar_p`, `stage2_g2p_mcnemar_p`. The
  vendored `g2p_pairs` are exhausted twice over: the interleave's train side
  (indices not divisible by the stride) fitted every matrix and message, and
  its test side is scored in full by `scripts/record_stage2_claims.py` and every
  channel recorder. Decision 74's family-disjoint re-partition reassigned the
  same pairs and has itself been scored (outcome recorded 2026-08-20), so it is
  spent as well. The pairs excluded by the equal-length filter are not usable —
  unalignable pairs are a different construct, not held-back data of the same
  experiment. Fitting a reduced model on part of train to free the rest would
  score a model that is not the shipped one — a new experiment, not a
  replication. Population: `external_lexicon` — the registered candidate rule is
  the set of word/pronunciation pairs present in a newer upstream CMUdict
  revision and absent from the vendored one, passed through the identical
  equal-length filter and scored once against the frozen train-side fit. The
  covariate cost is registered with it: upstream additions skew toward proper
  nouns and neologisms, so that run measures the frozen model on a shifted
  lexicon, and the entry scoring it must report the shift diagnostics
  (initial-letter TVD, shared-prefix percentage) beside the verdict.

- **UD EWT members (4): a genuine held-back population exists, outside the
  vendored artifacts.** `stage4_derived_mcnemar_p`, `stage4_gold_mcnemar_p`,
  `stage5_mcnemar_p`, `stage6_sign_test_p`. Every EWT artifact here is built
  from `en_ewt-ud-dev.conllu` alone, split by the vendored stride; the dev-test
  side is scored in full and the dev-train side fitted every level. But the
  corpus's own `en_ewt-ud-test.conllu` has never been read by any script in
  this repository. Population: `ud_ewt_test` — the partition rule is the whole
  of that file, vendored through the same generators
  (`generate_case_counts.py`, `generate_clause_counts.py`, the narrative
  counts generator) with the identical oracles, exclusion rules and
  minimum-length filters, the *entire* re-vendored corpus serving as evaluation
  and the model parameters read unchanged from the existing dev-train fits.
  Nothing is re-fit, no stride is applied, and the UD release is pinned in the
  artifact's provenance block at vendor time.

- **Split diagnostics (4): no replication analogue, by construction.**
  `stage2_g2p_subsample_mcnemar_p`, `subsample_prefix_mcnemar_p`,
  `subsample_stride3_mcnemar_p`, `subsample_stride5_mcnemar_p`. Each is a
  deliberate re-measurement of the Stage 2 comparison over a subset of the
  shipped test split — the prefix member exists *because* its subsample is
  biased. On a fresh population the full-set member is the hypothesis; a
  diagnostic of the old split's internal structure has nothing to replicate.
  Disposition: `split_diagnostic`, meaning: not re-scored; the full-set member
  carries the replication.

**The frozen discipline, applying to every future confirmatory recorder.** The
family is `SIGNIFICANCE_FAMILY` as it stands at this entry (16 members; its
Bonferroni threshold moves only if the family is edited, which this entry
forbids for the replication run). Per member: the same endpoint (the paired
comparison its label records today), the same exact conditional test (McNemar's
exact binomial; the Stage 6 sign test), two-sided, judged at the frozen
`BONFERRONI_ALPHA`, with Holm reported beside it as today. Channel members are
scored at their recorded weights — frozen now, in the ledger, before any new
data exist, which retires the test-selection caveat for the replication run.
One run per member, ever: the recorder that performs it must write a sentinel
claim first (`confirmatory_run_completed_<population>`) and refuse to run when
the sentinel exists, because a second pass over a held-back set un-holds it.
Its labels are recorded under `confirmatory_*` with epistemic status
`confirmatory`, and no development-population label is overwritten. If any
member cannot be computed on its population (e.g. the gold-case comparison
again yields b = c = 0), that is recorded as the outcome, not re-designed
around.

**What was changed in the manuscript now, ahead of any run.** The supplement's
multiplicity section names the family's verdicts development-population
inference at the point the family is introduced, states the census above, and
points here; the results section carries the same label and pointer. The
claim-strength vocabulary is unchanged — *corrected-threshold survivor* was
already defined as a development-population term.

**Falsifier.** If any script in this repository is found to have read
`en_ewt-ud-test.conllu` before the confirmatory vendor lands, that population is
burned and this entry's `ud_ewt_test` rows revert to "no held-back partition
exists"; the registration does not get to shrug. If the replication run is ever
performed with a family, endpoint, weight or threshold differing from the
frozen ones, its results are exploratory by this entry's own rule, whatever
their p-values.

## 80 — every ledger claim carries a machine-readable epistemic status

**Decision.** The claims ledger schema moves to `ailm-claims-v2`: every entry
may — and in the committed ledger must — carry a fourth field, `status`, one of
exactly four values (`ailm.claims.CLAIM_STATUSES`): **confirmatory** (a
pre-registered prediction scored under its registered rule), **exploratory**
(test-selected sweep results and subsample arms — headlines chosen on the
evaluation they are reported on), **descriptive** (measured quantities carrying
no inferential claim), **analytic** (closed-form or derived quantities). This
answers the external audit's ask for a machine-readable claims table in which a
reader, or a test, can distinguish the nested-arm results from the
selected-weight sweeps without reading prose.

**The assignment is mechanical, by recorder of origin.** The mapping lives in
`scripts/assign_claim_status.py`; `tests/test_claims_status.py` computes the
current census directly from `data/claims.json` rather than freezing mutable
counts in this decision prose. `confirmatory` is granted only to entry 29's
latent-state G2P learning recorder: its immutable registration commit strictly
predates the outcome commit. Entries 49 and 73 first appear with their recorders
in the same commits and are exploratory.
Ambiguity resolves DOWNWARD, always: the other pre-registrations keep their
mechanical tier (`exploratory` for selected or diagnostic arms, `descriptive`
for the rest), because promoting a status is a deliberate act with its own
decision entry, never a default. The Czech case arm is `exploratory` on the
strength of its own entry's words.

**Three properties make the field trustworthy rather than decorative.**

1. *It cannot erode.* The field is optional on write — recorders predate it and
   a required field would have broken every concurrent writer mid-run — but
   `tests/test_claims_status.py` fails while any committed claim is
   unclassified, and a re-record that names no status inherits the stored one,
   so a recorder replay is still a byte-level no-op. A recorder that wants to
   set the status passes `status=` to `record` / `record_many` / `ClaimBatch`;
   `assign_claim_status.py` classifies whatever arrives without one.
2. *It cannot inflate.* The string `confirmatory` may not appear in a recorder
   module unless that module's docstring cites the pre-registration decision
   entry that earns it (grep-based, positive-controlled, in the style of the
   entry-35 split-sequence scan), every ledger entry carrying the status must
   trace to a source whose docstring cites a decision number, and the
   confirmatory share is capped below 20% of the ledger — the same
   exception-stays-the-exception guard the test-owned allowlist carries.
3. *It cannot fabricate.* Statuses are written through
   `ClaimsLedger.assign_status`, which can alter nothing but the `status` field
   — assigning a status is a judgement about a measurement that already
   exists, not a measurement — and takes the same single-writer lock as every
   recorder.

**What did not move.** `formal/AILM/LedgerData.lean` is byte-identical: the
generator reads only `value`, so the formal layer is untouched by
classification, which is correct — a proof about a number does not change when
the number's evidential class is named. A v1 ledger file is still readable
(v2 adds only the optional field, so the legacy tag upgrades losslessly on
load); a genuinely unknown schema still refuses, as before.

**Relation to entry 79.** The taxonomy here is the vocabulary entry 79's
replication protocol needs: today's `confirmatory` claims come from the one
independently timestamped prior-commit arm, and the held-back-partition run 79 registers is the
mechanism by which members of the current `descriptive`/`exploratory` majority
could ever earn the label — under a new decision entry, cited in the recorder
that scores it, never by editing the mapping's default.

## 81 — stochastic-kernel composition is not a DisCoCat functor

**Correction.** Earlier guidance said that AILM redirected the DisCoCat meaning
functor from `FVect` into `FinStoch`, described the stochastic arrays as
row-normalized, and called `FinStoch` compact closed. Every part of that bridge
was wrong. Under the repository convention a pymdp likelihood or transition is
column-stochastic: a column is the conditional distribution over targets for a
fixed source. Compatible kernels compose, but `FinStoch` is a causal Markov
category rather than a compact-closed one. It therefore does not provide the
cups and caps needed to interpret a pregroup reduction and satisfy the snake
identities.

**Source audit.** The exact archived *Compositional Approaches to Linguistic
Case for Cognitive Modeling* paper defines its meaning functor in the
`FVect`/`FinHilb` setting; it contains no `FinStoch` functor. The optional
companion repository revision inspected for this correction is pinned in
`docs/provenance/sources.md` and implements the same linear target. The old
attribution to the paper has therefore been removed rather than weakened.

**Implemented boundary.** AILM has no categorical meaning-functor
implementation, and no recorded result depends on one. Its no-embeddings claim
is direct: the measured paths use named finite states and conditional kernels,
not learned semantic vectors. `docs/design/no_embeddings.md` now calls kernel
products stochastic composition and lists the obligations a future categorical
bridge would have to discharge. One possible research route uses a
compact-closed ambient matrix category and an explicit normalization boundary;
another stays in a Markov category and does not claim compact pregroup
semantics. Neither is part of the shipped model.

**Guards.** `tests/test_documentation.py` scans every current claim surface for
the retired assertions, proves that scanner live with a planted copy, and pins
the column orientation and compact-closure disclaimer in the canonical design
document. `docs/manuscript/06_related_work.md`, `README.md`, `AGENTS.md`, and the
source-provenance record all state the same narrower boundary.

## 82 — exact source identities and the CEREBRUM crosswalk boundary

**Correction.** The case scaffold formerly attributed a rowwise
CEREBRUM-to-pymdp crosswalk to Figure 13 and represented every core case with a
generic scalar $(\alpha_c,\beta_c)$ pair. Neither attribution survives inspection
of the deposited CEREBRUM PDF. The correction binds the implementation and prose
to the exact deposited artifacts rather than to record labels or a truncated
text export.

**Source identity is two-layered.** The Zenodo record for CEREBRUM identifies the
deposit as version 1.4 dated 2025-04-16 and names
`CEREBRUM_1-4_April-16-2025.pdf`; the title page printed in that exact file says
version 1.2 dated 2025-04-12. The record for *Compositional Approaches to
Linguistic Case for Cognitive Modeling* identifies version `v1`, dated
2026-04-23, while the deposited PDF's title page reads *Cognitive Diagrams:
Reviewing Categorical Accounts of Linguistic Case*, dated 2026-04-22, with no
printed version. `data/reference_sources.json` retains both identities, the
version DOI and concept DOI, exact filename, byte extent, page extent, MD5, and
SHA-256 for each deposit. The optional companion repository is a third identity,
pinned separately by revision; none is silently substituted for another.

**What the exact CEREBRUM PDF supports.** Figure 13 on physical page 21 contains
only the caption “Active Inference Integration Framework.” It does not print a
symbol crosswalk. Supplement 8.1 on physical page 90 instead defines
$(S,A,T,\Omega,O,F,\pi)$ as hidden states, actions, transition dynamics,
observations, likelihood, a free-energy functional, and precision parameters.
AILM preserves that tuple verbatim. Its correspondence to pymdp is now named
`AILM_PYMDP_HEURISTIC_CROSSWALK` and owned by this project: source $T$ is related
to pymdp `B`, and source $O$ to pymdp `A`; pymdp `C`, `D`, and `E` have no direct
analogue in the source tuple. Source $A$ remains the action space, source $F$ is
not pymdp `G`, and source precision $\pi$ is never mapped to pymdp `E`.

**The two `[GEN]` roles remain distinct.** CEREBRUM Table 1 gives core Genitive a
source/possessor model function. Supplement 2.12 separately introduces a novel
Generative role and reuses the abbreviation `[GEN]`. `CaseRole.GEN` remains the
linguistic Genitive. The scoped `CerebrumFunctionRole` inventory represents both
the core Genitive and novel Generative roles, preserves their shared source
abbreviation, and gives them distinct serialized values. No table asserts
rowwise equivalence between linguistic cases and model-function roles.

**The case formulations stay structurally different.** Supplement 11 weights
the likelihood term for NOM and the complexity term for ACC. GEN, DAT, INS, LOC,
ABL, and VOC instead augment the latent state with distinct variables and state
case-specific precision relations. `CEREBRUM_CASE_FORMULATIONS` records those
differences. AILM's scalar implementation is explicitly narrowed to NOM and ACC;
its numeric weights are project-owned defaults, remain unused by the measured
Level 4 comprehension path, and requests for the six non-scalar cases fail
closed rather than inventing source parameters.

**The two transition equations remain separate.** Supplement 8.2.3 defines the
transition free energy as target free energy plus the unscaled directed KL term.
Supplement 8.8.3 separately defines an expected-free-energy transition with a
$\gamma$-scaled KL term. `transformation_cost` implements the former;
`expected_free_energy_transition` implements the latter. Sharing a directed-KL
helper does not collapse their equations.

**Alternatives rejected.** Retaining the old dictionary with a disclaimer would
leave a false source attribution callable by downstream code. Giving pymdp
`C`/`D`/`E` invented source analogues, mapping precision to the policy prior,
assigning scalar weights to all eight cases, or merging the two `[GEN]` meanings
would each erase an observed source distinction. They are removed or made to
fail closed instead.

**Guards.** `tests/test_reference_source_fidelity.py` validates the offline
artifact-and-locator snapshot, binds it to the code and manuscript literals,
scans current claim surfaces for the retired attribution, and carries planted
positive controls. `tests/test_cases.py` independently rejects the precision-to-
`E` mapping, the collapsed `[GEN]` inventory, a generic all-case scalar model,
and conflation of the unscaled and $\gamma$-scaled transition equations.

## 83 — Stage 5 is non-passing; the privileged-label gain remains unresolved

**Correction.** Decision 24 and downstream publication surfaces called Stage 5
a resolved or bounded null and described the true-type privileged-label gain as
negligible. The recorder had no sampling interval for that paired gain and no
equivalence margin or smallest effect size of interest. A failed mixture gate
and a live large-effect control do not license absence, equivalence, or
negligibility. Those claims are withdrawn.

**Population and estimand at the time.** This correction originally reused the
then-current sentence-stride population. Decision 85 supersedes that population
with claim:stage5_test_sentences held-out typed EWT sentences from
claim:stage5_test_documents contributing source documents and
claim:stage5_test_positions case-role positions. The estimand, fitted
type-conditioned chains, type-free comparator, and true-type labels are
unchanged. For sentence \(i\), the paired difference is

\[
\Delta_i = \log P(o_i \mid \text{fitted true-type chain})
           - \log P(o_i \mid \text{type-free chain}).
\]

The primary summary gives every sentence equal weight: the mean is
claim:stage5_oracle_gain_nats_per_sentence nats per sentence and its
claim:stage5_oracle_gain_ci_confidence_pct% percentile bootstrap interval is
[claim:stage5_oracle_gain_sentence_ci_low,
claim:stage5_oracle_gain_sentence_ci_high]. The companion summary preserves the
position-weighted perplexity estimand: it divides the resampled sum of
\(\Delta_i\) by the resampled sum of sentence lengths. Its point is
claim:stage5_oracle_gain_nats_per_position nats per position and its interval is
[claim:stage5_oracle_gain_position_ci_low,
claim:stage5_oracle_gain_position_ci_high]. This companion point equals the log
ratio of type-free to privileged-label perplexity.

**Resampling contract, superseded by decision 85.** The original correction
treated one sentence as independent. The current contract treats one
authenticated source document as the outer resampling unit; all sentences and
case positions in a selected document move together. Each replicate applies one
vector of document indices to both arms, and the position-weighted numerator
and denominator use those same indices. The production helper resamples
claim:stage5_oracle_gain_bootstrap_documents documents for
claim:stage5_oracle_gain_bootstrap_resamples
replicates and seed claim:stage5_oracle_gain_bootstrap_seed. The recorder writes
the estimates, endpoints, coverage, replicate count, and seed; the manuscript
derives its qualitative interval language from the endpoints rather than
hard-coding a verdict.

**Result.** Both document-bootstrap intervals span zero. The mixture and fitted
privileged reference also remain on the failing side of the type-free gate. The
supported status is therefore **non-passing and `ABOVE_REFERENCE`;
privileged-label gain unresolved on this split**. This is not an equivalence
result: no equivalence margin was declared, and the observed intervals would
not by themselves define one. The balanced synthetic control shows that the
same machinery and interval can resolve a deliberately large positive effect;
swapping its type-conditioned chains resolves a negative effect. Those controls
establish a live instrument for large planted effects, not power at the EWT
operating point.

**Guards.** `ailm.levels.clause.summarize_oracle_gain` owns the paired estimator
and validates every resampling and probability input. The real Stage 5 recorder
is the sole writer of all nine uncertainty claims and is a byte-level no-op on
replay. `tests/test_stage5_gate.py` independently re-derives the fixed-seed
whole-document replicates and both exact percentile endpoints, checks the perplexity-ratio
identity, exercises positive and negative controls, and forces malformed
contracts and evidence to fail. `scripts/audit_gates.py` treats either a point
pass, a fitted reference crossing the bar, or an interval that moves off zero
as a changed finding. Figure and
manuscript builders likewise refuse a stale unresolved verdict when the
endpoints no longer support it. Planted multi-sentence-document controls show
that the retired sentence bootstrap can exclude zero while the corrected
document bootstrap spans it. The interval coverage
claim:stage5_oracle_gain_ci_confidence_pct% is inherited from the Stage 5 AUC
contract (`STAGE5_BOOTSTRAP_CONFIDENCE`), which decision 66 kept separate from
the repository-wide `DIFFERENCE_CONFIDENCE` harmonization because these are
percentile intervals rather than the share interval; the verdicts above are
invariant to that choice of level.

## 84 — Stage 4's morphological overlap does not bound its token-level adposition construct

**Correction.** The Stage 4 heuristic-versus-gold result was described as a
fidelity ceiling on every score against the derived labels. It is not. The
comparison covers claim:stage4_heuristic_gold_tokens morphologically marked
tokens, of which claim:stage4_heuristic_gold_deprel_tokens receive their
syntactic comparison label through the dependency-relation route and
claim:stage4_heuristic_gold_adposition_token_tokens through the token-level
adposition route. The percentage validates only that named overlap.

**The construct that the old reading hid.** Across the complete parsed EWT
development shard, the oracle labels claim:stage4_oracle_full_derived_tokens
tokens. Its path composition is
claim:stage4_oracle_full_path_morphology_tokens morphology,
claim:stage4_oracle_full_path_deprel_tokens dependency relation, and
claim:stage4_oracle_full_path_adposition_token_tokens token-level adposition.
The final route maps the ADP token's own lemma and labels that ADP token. It does
not consult or relabel the token whose UD ID appears in `head`.

That distinction is material rather than terminological. No token on the
token-level adposition route carries gold `Case` morphology. At the same time,
claim:stage4_adposition_token_head_also_labeled_tokens of those ADP tokens have
a head the oracle separately labels, and
claim:stage4_adposition_token_head_role_disagreements of the ADP/head role pairs
disagree. The scheme can therefore assign two roles to the two tokens of one
prepositional construction. English EWT supplies no morphological fidelity
estimate for the ADP-token label.

**Alternative rejected — silently transfer the adposition role to its nominal
head.** That may be a defensible future construct, but it changes which tokens
carry labels, the transition sequences, both fitted matrices, every Stage 4
score, and the downstream clause/narrative artifacts. It is a new research
population requiring a new recorded run. This correction preserves the
historical labels and gates and narrows only what their existing validation
licenses.

**Implementation and evidence.** `_apply_case_oracle` now takes exactly one
token, matching what it always used; the unused sentence parameter and the false
head-lookup docstring are gone. `case_oracle_assignment` returns both the role
and the explicit `adposition_token` path, while `case_oracle_scope_audit`
resolves heads separately by CoNLL-U token ID. The artifact retains the path,
fidelity, head-resolution, duplication, agreement, and disagreement sufficient
counts. `scripts/record_stage4_claims.py` is their sole ledger writer, and the
manuscript consumes their mechanically generated tokens.

**Guards.** `tests/test_ud.py` exercises the real EWT census, a deliberately
shuffled/non-contiguous head-ID example, a missing-head branch, and a duplicate-ID
refusal. The planted inline ``to`` example proves the ADP token and nominal head
keep separate roles. `tests/test_stage4_gate.py` validates every sufficient-count
identity, proves a corrupted head partition fails, and binds the ledger entries
back to the vendored artifact. The canonical generator must still reproduce that
artifact byte for byte.

## 85 — English EWT is split and resampled by authenticated source document

**Correction.** The English Stage 4 and Stage 5 artifacts formerly applied a
global sentence-index stride. EWT contains claim:ewt_boundary_source_documents
authenticated `# newdoc id` blocks, many containing multiple sentences, so the
retired rule placed claim:ewt_boundary_stage4_retired_overlap_documents source
documents in both Stage 4 arms and
claim:ewt_boundary_stage5_retired_overlap_documents in both Stage 5 arms. A
sentence-disjoint split was not document-disjoint, and sentence bootstrap
treated repeated observations from one source as independent.

**Current contract.** `parse_conllu` carries the stateful raw `# newdoc id`
through every sentence. `document_stride_partition` builds one ordered identity
alphabet, rejects missing or reappearing blocks, partitions those identities
before any case-population, type, punctuation, or minimum-length filter, and
asserts disjoint arms. Both artifacts retain the ordered IDs, an unambiguous
digest, the full split indices, and one document index per retained sequence.
Their loaders reject duplicate IDs, count drift, mappings outside an arm, and
any overlap. The current Stage 4 and Stage 5 overlap counts are respectively
claim:ewt_boundary_stage4_current_overlap_documents and
claim:ewt_boundary_stage5_current_overlap_documents.

**Exact before and after.** The before arm is not a remembered table:
`scripts/record_ewt_document_correction_claims.py` reconstructs the retired
sentence stride from the pinned raw CoNLL-U source and re-runs the current
scorers. Stage 4 changes from
claim:ewt_boundary_stage4_before_test_sentences sequences and
claim:ewt_boundary_stage4_before_test_tokens tokens at
claim:ewt_boundary_stage4_before_agreement% agreement versus
claim:ewt_boundary_stage4_before_baseline% (margin
claim:ewt_boundary_stage4_before_margin_pts points) to
claim:ewt_boundary_stage4_after_test_sentences sequences and
claim:ewt_boundary_stage4_after_test_tokens tokens at
claim:ewt_boundary_stage4_after_agreement% versus
claim:ewt_boundary_stage4_after_baseline% (margin
claim:ewt_boundary_stage4_after_margin_pts points). Its primary interval changes
from the retired sentence-cluster
[claim:ewt_boundary_stage4_before_sentence_cluster_ci_low,
claim:ewt_boundary_stage4_before_sentence_cluster_ci_high] to the current
document-cluster [claim:cluster_stage4_derived_ci_low,
claim:cluster_stage4_derived_ci_high].

Stage 5 changes from claim:ewt_boundary_stage5_before_test_sentences typed test
sentences and claim:ewt_boundary_stage5_before_test_positions positions to
claim:ewt_boundary_stage5_after_test_sentences and
claim:ewt_boundary_stage5_after_test_positions. The paired fitted-reference gain
changes from claim:ewt_boundary_stage5_before_gain_nats_per_sentence nats per
sentence, interval [claim:ewt_boundary_stage5_before_gain_sentence_ci_low,
claim:ewt_boundary_stage5_before_gain_sentence_ci_high] under sentence
resampling, to claim:ewt_boundary_stage5_after_gain_nats_per_sentence, interval
[claim:ewt_boundary_stage5_after_gain_sentence_ci_low,
claim:ewt_boundary_stage5_after_gain_sentence_ci_high] under document
resampling. AUC changes from claim:ewt_boundary_stage5_before_auc, interval
[claim:ewt_boundary_stage5_before_auc_ci_low,
claim:ewt_boundary_stage5_before_auc_ci_high], to
claim:ewt_boundary_stage5_after_auc, interval
[claim:ewt_boundary_stage5_after_auc_ci_low,
claim:ewt_boundary_stage5_after_auc_ci_high]. Both gain intervals remain
unresolved, but the point changes sign and the fitted privileged reference no
longer clears the type-free bar. The Stage 5 gate placement is therefore now
`ABOVE_REFERENCE`, not `BRACKETED_REFERENCE`.

**Inference contract.** Stage 4's English gold and derived effects use source
document as the primary cluster. Stage 5's paired gain, MAP-accuracy contrast,
and AUC all resample whole documents; the point estimands remain per sentence or
per position as named. Sentencewise McNemar counts remain secondary conditional
descriptions and do not substitute for population inference.

**Controls.** The real-corpus negative control proves the retired sentence
stride leaks many multi-sentence documents while the canonical split has zero
overlap. Planted repeated-sentence documents are stronger: individual-sentence
bootstrap can falsely exclude zero for the paired gain or AUC while
whole-document bootstrap includes the null. Cross-arm document-index mutations
must be refused by both loaders. Artifact regeneration is byte-identical, and
the correction recorder is the sole writer of every `ewt_boundary_*` claim.

## 86 — Stage 6 retains and authenticates the document identity it resamples

**Correction.** The original narrative artifact wrote a `doc` string beside
each sequence but its loader discarded that field. Downstream clustered
inference enumerated list positions as if they were document identities. A
duplicate or cross-arm alias could therefore masquerade as an independent
cluster while the artifact still loaded. The generator also applied its stride
after the minimum typed-sentence filter, so eligibility could change which raw
documents entered each arm.

**Current contract.** The narrative-counts v2 generator first partitions the
complete authenticated `# newdoc id` alphabet using the same shared EWT
document-stride helper as Stages 4 and 5. Typing and the minimum arc length are
applied only afterward. The artifact retains the ordered raw identities, their
digest, an exhaustive split partition, one document index per eligible
sequence, and the human-readable ID beside that sequence. The loader requires
all representations to agree and exposes the validated IDs to the clustered
recorder; list position is no longer a substitute for identity.

**Inference and outcome.** The point estimand remains the observation-weighted
mean sentence log-loss difference. Source document remains the resampling and
sign-flip unit. Re-recording owns any resulting changes to
claim:stage6_train_documents, claim:stage6_test_documents,
claim:stage6_test_sentences, and the `cluster_stage6_*` family; no value is
copied into this decision. The primary interpretation remains conditional on
the re-recorded interval and cannot be promoted from non-confirmatory by a
schema repair.

**Controls.** Planted duplicate IDs, duplicate sequence indices, cross-split
mappings, entry/index aliases, digest drift, and provenance-count drift all
fail closed. The canonical partition is asserted disjoint and exhaustive, and
raw-corpus regeneration must reproduce the artifact byte for byte.

## 87 — Stage 3 uses an authenticated category-stratified Brown document partition

**Correction.** The retired Stage 3 writer flattened Brown into one alphabetic
token stream and cut that stream at the target fraction. That cut fell inside a
source file, so the same authenticated document contributed tokens to training
and evaluation. Token-disjoint observations are not a document-disjoint
evaluation population, and a future document-clustered analysis could not be
reconstructed because neither derived artifact retained document membership.

**Current contract.** `build_brown_document_partition` authenticates the pinned
archive, normalizes each Brown file separately, groups identities by the Brown
category metadata carried in that archive, and selects the nearest target
whole-document count within each category under a revision-bound SHA-256
ordering. Selection depends on immutable source identity, never token length,
labels, or model output. The complete population has
claim:stage3_source_documents ordered source identities;
claim:stage3_train_documents enter training and
claim:stage3_heldout_documents enter evaluation, with
claim:stage3_document_overlap overlap. All claim:stage3_source_categories source
categories contribute, with claim:stage3_train_categories represented in
training and claim:stage3_heldout_categories in evaluation. Both word artifacts and the unfiltered
held-out denominator carry the ordered identities, an unambiguous length-framed
digest, category assignments, per-document normalized-token counts, an exhaustive split, and a
document index aligned to every retained token.

**Consequences.** The vocabulary, priors, evaluation stream, baseline, ceiling,
scaling ladder, out-of-vocabulary analysis, and generation distribution were
re-recorded from the corrected artifacts. No old value is transcribed into this
decision. Stable full-dictionary labels own the ladder endpoint; dynamic labels
whose names embed a measured vocabulary size are removed when that size changes,
so a corpus correction cannot leave a plausible stale endpoint in the ledger.

**Controls.** The loader rejects duplicate or reordered identities, digest
drift, cross-arm aliases, non-exhaustive splits, a partition other than the
declared category-stratified revision-bound rule, per-document count drift, a token
mapped to a training document, or an overfull document mapping. Independent
tests reconstruct both artifacts from the pinned archive byte for byte and
prove that filtering the unbounded held-out stream reproduces the scoring tokens
and their document indices element-for-element.

## 88 — The Stage 3 partition correction makes the clause-to-word result directionally helpful

**Correction.** The clause-to-word bridge consumes the Stage 3 vocabulary and
prior, so decision 87 changed its input population. Regenerating
`vendor/brown_clause_word_counts.json` and replaying its sole recorder changes
the old mixed-direction reading. At the selected high endpoint, aggregate
perplexity changes by claim:clause_word_best_delta_ppl from
claim:clause_word_anchor_ppl to claim:clause_word_best_ppl, while
claim:clause_word_best_improved_tokens tokens improve against
claim:clause_word_best_degraded_tokens degraded. Both directional summaries now
favor the message. The tested curve improves throughout the positive grid, so
the sweep does not locate an optimum. The pre-registered monotone-harm prediction remains
falsified; the evidential class is `helps`, not `mixed-direction`.

**Epistemic boundary.** This is a correction-driven remeasurement on a corpus
and analysis family already inspected, not a new confirmatory population. It
supports the descriptive direction of this channel under the corrected
partition. It does not establish a general cross-level law, isolate conditional
information from mismatch correction or linear mixing, or upgrade any Stage 5
or Stage 6 inference.

**Propagation.** The typed channel registry, generated manuscript tallies,
manuscript interpretation, architecture documentation, recorder diagnostics,
and per-token direction tests all read the corrected class. Decision 27 retains
the pre-registration and outcome chronology; decision 65 retains the reason a
mean needs its own dispersion, while their retired sign interpretations are
explicitly superseded here.

**Guards.** `classify_cross_level_channels` derives the channel class from both
the aggregate perplexity delta and the improved/degraded ordering, then refuses
manuscript generation if that class diverges from the typed registry. The real
channel test replays the selected row, requires the current token direction,
and reconstructs every recorded sweep count. Planted opposite-direction counts
must return `mixed-direction`, so deleting the distinction cannot make the gate
green.

## 89 — The typed word-prior arm is a privileged-oracle intervention, not L5 output

**Correction.** The typed word-prior code indexes each held-out Brown sentence
by its punctuation-derived reference type and selects the corresponding
train-estimated word distribution. It imports no L5 inference function, runs no
Clause Agent, and consumes no L5 posterior. Calling this result `L5→L3` therefore
over-attributed a privileged-label intervention to the model stack. The measured
source is now named **punctuation-type oracle→L3** everywhere current results are
summarized.

**What survives.** The receiver, insertion rule, document-disjoint population,
alpha sweep, type-free and label-swapped controls, and recorded helpful direction
are unchanged. The result still falsifies the harm prediction for this oracle
intervention. It does not establish that L5 could deliver the message, especially
while Stage 5's own gate is non-passing.

**Uncertainty correction.** Tokens are nested in claim:clause_word_test_documents
held-out Brown documents. The primary descriptive uncertainty therefore
resamples whole documents. The mean gain is
claim:clause_word_cluster_loggain_nats_per_emission nats per emission with a
claim:clause_word_cluster_confidence_pct% document-bootstrap interval of
[claim:clause_word_cluster_loggain_ci_low,
claim:clause_word_cluster_loggain_ci_high]. The assumption-conditional document
sign-flip sensitivity reaches only the declared Monte Carlo resolution floor
claim:clause_word_cluster_loggain_p_floor; it is reported as at or below the
procedure's resolution floor, never as `p = 0`.

**Controls.** The typed registry records the source kind and rejects any
`clause_word` entry that claims L5 as the sender. The real bridge loader proves
document disjointness and exact Stage 3 vocabulary identity. At the selected
weight the real typed message beats both the type-free bridge marginal
(claim:clause_word_typefree_ppl_at_best) and a declarative/interrogative label
swap (claim:clause_word_swapped_type_ppl_at_best); the reversed vocabulary-axis
control remains a separate wiring mutation. Tests plant a false L5 sender, a
false model-output source kind, cross-arm sentence moves, and broken type controls.

## 90 — The synthetic v2 contract bounds exact-artifact-overlap claims under independent trajectories

**Correction.** The synthetic arm does not prove that a model has never seen the
same sequence, a similar alphabet, or a related transition rule. What is
checkable is narrower: the scored artifacts have local provenance from a
stipulated transition, and the AILM training and evaluation draws are disjoint.
That construction reduces exact-artifact overlap as an explanation; it cannot
establish absence of coincident or structurally similar material from an
external model's undisclosed pretraining corpus. Current prose therefore uses
*stipulated-language exact-artifact-overlap control* only for this bounded
design property and makes no general contamination or non-exposure claim.

**Population.** Every scored position is generated from its own deterministic
trajectory under the frozen protocol registry. The recorder does not generate
one long stream and then select nearby windows from it. It records the position
generator, seed, trajectory length, and target-start total-variation distance,
and receipt validation reconstructs the ordered population. The full-prefix and
short-prefix arms score the same targets in alternating order; their paired
interval is a context-width sensitivity analysis, not a randomized causal
effect.

**Demonstrations.** Each in-context example exposes exactly one labeled
order-one transition. No unrelated within-prefix transition remains in either
treatment or control. Example draws use position-derived deterministic seeds, so
the same `(rung, prefix)` draw is invariant to insertion or reordering elsewhere
in the evaluation. The character-level derangement preserves prefixes and the
label multiset while destroying every prefix-to-label pairing; it does not claim
tokenizer-level equality. A separate deterministic mapping learner must beat its
deranged control before the external-model arm is publishable, proving the data
manipulation carries recoverable signal without treating that offline learner as
evidence about the model interface.

**Prompt and parser contract.** The response schema accepts only the frozen
synthetic inventory and one exact JSON object; surrounding text, case changes,
padded symbols, extra keys, and multiple lines are refusals. Requests carry an
explicit context window and response budget. Character length is only a
conservative preflight diagnostic; the decisive capacity guard reads each
server response's actual prompt-token count and refuses publication if any
scored prompt could have been truncated.

**Multiplicity and execution.** Family sizes are derived from the Cartesian
product of the registered probes and contrast templates rather than copied into
prose. Arm order, warm-up policy, call protocol, endpoint, immutable model
identity, source manifest, exact cache keys, and the complete ordered execution
trace are receipt-bound. Canonical publication disables scored cache reads and
requires the recorded hit/miss census; an existing local cache may persist as a
receipt store but cannot silently replace fresh scored calls.

**Epistemic boundary.** This is a post-audit replacement measured on a fully
inspected synthetic construction, not an independently registered confirmatory
experiment. Its intervals and corrected paired tests are descriptive of the
registered probes, prompt schema, and stipulated chain. The generated supplement
owns every current numerical result through injected ledger variables; this
decision deliberately contains no copied outcome table.

**Guards.** Contract tests plant shared-trajectory populations, wrong target
starts, unstable example draws, non-derangeable controls, parser near-misses,
omitted probes and grid rungs, trace reordering, cache hits, model-identity drift,
source drift between measurement and receipt writing, and prompt-token overflow.
Each mutation must fail before the corresponding claims can pass a release.

## 91 — the confirmatory replication is retired by exhaustion: the external_lexicon population has no held-back partition

**Outcome of a run that never happened.** The decision-79 confirmatory
replication was executed up to its first data touch on 2026-08-26 and stopped
by its own registered discipline, twice, before any population byte was read.
Both stops are recorded here because they are results, not incidents.

**Stop 1 (divergence).** The evening lane brief directed sourcing the
`external_lexicon` population from WikiPron (`eng_latn_us_broad.tsv`, mined
Wiktionary IPA). Decision 79 registers that population as *upstream CMUdict
additions*. These are different populations: the frozen channel/fitted/
geometric/coupling endpoints are defined over stress-stripped ARPAbet sequences,
so an IPA-to-ARPAbet layer would be a new construct injected between population
and endpoint — exactly what 79 forbids redesigning around — and the identical
equal-length filter (letter count == ARPAbet phone count) has no defined
meaning over IPA transcriptions with syllable dots and length marks. The run's
falsifier clause makes any execution on a non-frozen population exploratory
regardless of p-values. Held. The Principal then ruled Option A: source exactly
as registered.

**Stop 2 (exhaustion), measured live 2026-08-26 evening.** Under Option A the
first execute-time step is fetching a newer upstream CMUdict revision to diff
against `vendor/cmudict.dict`. There is none:

1. cmusphinx/cmudict default branch HEAD = `74790861f652…` — the vendored pin
   `CMUDICT_REVISION` itself.
2. The only other branch (`kal-make-ps-dict`) is behind master; its unique
   commits add only `make_ps_dict.py` and touch no dictionary file.
3. Commit history filtered to path `cmudict.dict`: the last commit that ever
   touched the file is `0f8072f81430` (2024-12-17), which predates the vendored
   revision; compare API `0f8072f...7479086` shows ahead_by=2 with only
   `make_ps_dict.py` changed.
4. Raw download at pinned HEAD: sha256
   `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`
   (= `CMUDICT_RAW_SHA256` exactly), 135,166 lines, `cmp` against
   `vendor/cmudict.dict` byte-identical.

The set {entries present upstream AND absent from the vendored file} is
therefore EMPTY. **The census error in decision 79 was temporal optimism, not
classification error**: at registration time upstream could have grown since
the pin; it factually did not, and as of this entry it provably cannot for
*this* baseline unless the baseline itself moves.

**Disposition of the members.** The eight `external_lexicon` family members
(`channel_letter_predictive_paired_p`, `channel_lexicon_evidence_paired_p`,
`fitted_paired_p`, `geometric_letter_paired_p`, `geometric_lexicon_paired_p`,
`coupling_mcnemar_p`, `lexical_best_mcnemar_p`, `stage2_g2p_mcnemar_p`) revert
to the same status as their dev-population state: *no held-back partition
exists*, in the census language 79 already uses for them. The four
`ud_ewt_test` members (`stage4_derived_mcnemar_p`, `stage4_gold_mcnemar_p`,
`stage5_mcnemar_p`, `stage6_sign_test_p`) are different and this entry is
deliberate about them: **the partition survives, unburned** — no sentence of
`en_ewt-ud-test.conllu` has been read by any script (re-verified during the
aborted run; repo-wide grep unchanged), no sentinel exists for either
population, and no scoring touched it. They remain rescoring-eligible under a
future protocol; nothing in this entry spends them.

**Sentinels and statuses.** No `confirmatory_run_completed_*` sentinel exists;
none is written, because rule-gated sentinels record completed populations and
neither completed. Protocol "79" stays
`independently_timestamped_pending_outcome` in
`data/registration_provenance.json` pending the Principal's choice between
the two resolutions below; if retirement is confirmed, its outcome fields may
be closed with this entry cited rather than a commit hash, because there is no
outcome commit to cite. `BONFERRONI_ALPHA` and the family are untouched: the
multiplicity vocabulary — development-population inference at
{{STAT_FAMILYWISE_ALPHA}} familywise, per-test {{STAT_BONFERRONI_ALPHA}} — remains exact.
(These braces are prose convention here, not hydrated tokens.)

**What changes in the manuscript.** Nothing yet, and this entry names why:
`docs/manuscript/04_results.md` currently says the confirmatory run "is
pre-registered rather than performed" and points at `CONFIRMATORY_REPLICATION`
and this log; that sentence remains true under this entry. What becomes false
on full retirement is the forward-looking hope that the upgrade path exists.
If the Principal retires the replication outright, the supplement's
multiplicity section drops the pointer to a future run and states instead that
the one candidate held-back population outside the vendored artifacts was
examined at the registered boundary and found empty by upstream stasis
(citing this entry); `04_results.md`'s final sentence then reads
development-population inference without an open promissory note. If the
Principal re-registers, the new entry supersedes this disposition and the
prose keeps a pointer — to the new entry, not to 79. Either way the honest
sentence is the one 79 already put in the manuscript's mouth: no result in
this paper is confirmation, and the word "confirmatory" appears only in
descriptions of the machinery that would have earned it.

**Falsifier carried forward.** 79's falsifier still binds: if any script reads
`en_ewt-ud-test.conllu` before a successor registration vendors it through the
identical generators, that population burns and any successor entry must say
so. This entry adds its own falsifier: if upstream CMUdict later gains commits
touching `cmudict.dict`, the *delta-from-vendored-pin* population becomes
non-empty again mechanically — but a future run must then re-register with the
diff-size covariate shift reported beside the verdict, per 79's own
covariate-cost paragraph, and must NOT treat mere upstream motion as automatic
revival of this protocol.

## 92 — L5 and L6 move from array evidence to measured pymdp inference evidence (P1, 2026-08-31)

**What was true before.** Entry 68's residual, restated by
`tests/test_pymdp_inference_equivalence.py`'s module docstring: L5's gate walked
`agent.A/B/D` by hand (storage, not inference) and scored everything in closed
form; L6 had a registered observed-position score from a conditional table plus
a separately recorded live-filter run, with no test attributing that run to
pymdp's inference path and no measurement of how the two estimands relate.

**What is measured now.** `scripts/record_pymdp_equivalence_claims.py` (single
writer, `descriptive` status) records, on bounded slices of the canonical
held-out splits:

- L5: over {{STAGE5_PYMDP_PREFIX_SENTENCES}} sentences /
  {{STAGE5_PYMDP_POSITIONS}} positions, live `infer_states` +
  `update_empirical_prior` agrees with the NumPy joint forward filter on the
  sentence-type posterior argmax with
  {{STAGE5_PYMDP_TYPE_ARGMAX_DISAGREEMENTS}} disagreements. The recorder
  refuses to write if the count is non-zero or any belief is unnormalized —
  the equivalence is a gate, not an observation.
- L6: the live sequential estimand and the registered observed-position
  estimand, scored over the identical
  {{STAGE6_PYMDP_PREFIX_SENTENCES}}-sentence slice, differ by
  {{STAGE6_PYMDP_LIVE_VS_OBSERVED_DELTA_NATS}} nats and agree on the
  per-sentence sign on a {{STAGE6_PYMDP_LIVE_VS_OBSERVED_SIGN_AGREEMENT}}
  share of sentences.
  They are different estimands by construction (latent-phase belief carried
  through `B` versus the known position), so this is characterization; the
  manuscript now says which estimand is which with the number attached.

**Verification surface.** `tests/test_pymdp_l5_l6_equivalence.py` adds the L5
equivalence test plus two negative controls (belief normalization; a
type-block-swapped `B` that must visibly break the agreement) and an L6
attribution test (every `infer_states` belief equals the manual Bayes step over
the agent's own stored float32 matrices). The bounded-slice caveat of entry 68
carries over verbatim: prefix agreement is evidence over a stated subset, not a
proof of full-set equality; the full stream remains verified by the NumPy
closed form with the documented float32 conjugacy tolerance.

**Family discipline.** No new `_p` label is introduced, so
`SIGNIFICANCE_FAMILY` and `EXCLUDED_FROM_FAMILY` are untouched. The eight new
labels are `descriptive` recorder-owned quantities, wired through
`REQUIRED_CLAIMS`; the spot-check `CANONICAL_LABELS` names the two
headline-bearing ones.

**What stays array evidence.** L5's headline full-set score is still the
closed-form mixture (now with a live equivalence check at the type-posterior
level on the stated slice); L6's registered primary analysis is still the
observed-position score with its document-clustered interval. Nothing in this
entry converts a descriptive diagnostic into confirmatory inference.

## 93 — The calibration gate is the direct dual-rule divergence check, adopted (2026-08-31)

Decision 36 pre-registered the v4 protocol with a registered change it deliberately deferred: *"gate the calibration arm on this divergence exceeding the rule epsilon rather than on captured mass falling below the floor."* The deferral had one reason — switching the gate in the same revision that first measures it would leave no version of the paper in which the declared rule and the measured rule can be compared side by side. The v4 record is that comparison. It is now complete, and the gate is adopted.

**The census.** The direct dual-rule check (`dual_rule_calibration`), computed for every probe on its own recorded answer-slot candidates, gives:

| Probe | ECE divergence | Brier divergence | Rule TV (mean) | Captured mass (mean) | Captured-mass verdict | Direct-check verdict |
|---|---:|---:|---:|---:|---|---|
| gemma3 | 0.01274 | 0.01213 | 0.0219 | 0.861 | admitted (floor 0.5) | **withheld** (both divergences exceed epsilon 0.01) |
| qwen25 | 0.00586 | 0.00685 | 0.0062 | 0.996 | admitted (floor 0.5) | admitted (inside epsilon) |
| smollm | 0.12939 | 0.11759 | — | 0.318 | withheld (below floor 0.5) | **withheld** (divergences an order of magnitude past epsilon) |

The smollm row is recorded under the adopted gate by the same revision that registers the third probe family; the two instruct-probe rows are the census decision 36's pre-registration anticipated, and they are unchanged by the adoption. The proxy and the direct check still disagree on **exactly one system**: gemma3, which the floor admits and the divergence withholds. SmolLM is withheld by both rules, and its recorded-but-unpublished expected calibration error is the lowest figure in the ledger — the flattening trap the floor subsection describes, demonstrated rather than argued.

**The two rules disagree on exactly one system.** The captured-mass proxy admitted both instruct probes; the direct check withholds gemma3's calibration, because both of its metric divergences exceed the declared negligibility threshold of 0.01. This is the resolution of decision 36's pre-registered comparison: the proxy is not conservative in the direction the offline sweep predicted — it admits a system whose reported calibration number moves by more than the declared tolerance when the completion rule is swapped.

**What changes in code.** `calibration.measurement` now decides `measurable` from the rule check (`rule_check["agrees"]`), not from `mean_captured >= CALIBRATION_MASS_FLOOR`. The floor constant is retired from gating and retained as provenance — the starved-fraction diagnostic and the floor-locating sweep still cite it. `dual_rule_calibration` is computed for every probe regardless of its verdict, so a withheld calibration is an audited one. The recorder now records the calibration metrics for every probe whether the gate admits it or not: a withheld system is one whose figures sit in the ledger while the reporting rule declines to publish them as a row, which keeps the gate's verdict recomputable from the ledger instead of frozen into it.

**What changes in prose.** The supplement's calibration subsections report the gate as adopted: the floor subsection's "registered change, not applied in this revision" paragraph becomes past tense, the calibration table reports only gate-admitted systems, and the withheld probes' rows are replaced by prose that cites their recorded divergences rather than leaving the absence implicit.

**Family discipline.** No new `_p` label is introduced and no significance family changes. The measurability flags remain binary indicators in `REQUIRED_CLAIMS`; their meaning, not their type, changed.

## 94 — L4 and L5 live-pymdp equivalence are posterior-tolerance gates on the full held-out set (2026-09-01)

**What was vacuous.** Entry 92's L5 gate compared live `infer_states` /
`update_empirical_prior` with the NumPy joint forward filter at one statistic,
the sentence-type posterior argmax, on a bounded prefix. On that prefix — and,
it turns out, on the whole held-out split — the filter's type argmax is the
majority type at every position. A statistic that never changes cannot fail: an
identity transition (no case dynamics at all) also scores zero type-argmax
disagreements on the same slice while its joint posterior departs from the
filter by a wide margin. The recorder would have accepted that model. The
block-swap negative control of entry 92 catches a *mis-wired* `B`; it cannot
catch a `B` that carries no information, which is the class the argmax
statistic is blind to.

**What is measured now.** `scripts/record_pymdp_equivalence_claims.py` walks
the *entire* held-out split at both levels and records, beside the argmax
counts, the largest absolute deviation between the live posterior and the
NumPy filter over every position:

- L5: claim:stage5_pymdp_prefix_sentences sentences and
  claim:stage5_pymdp_positions positions (the label keeps its bounded-era name
  so the manuscript tokens stay stable; the value is the full split), with
  claim:stage5_pymdp_type_argmax_disagreements argmax disagreements and a
  largest joint-posterior deviation of claim:stage5_pymdp_posterior_max_abs_dev.
- L4, previously a bounded test-only comparison that no ledger label recorded
  and that `03_methods.md` pointed at as "agreement reported" in a section
  that reported nothing: claim:stage4_pymdp_sentences sentences and
  claim:stage4_pymdp_positions positions, claim:stage4_pymdp_argmax_disagreements
  argmax disagreements, and a largest posterior deviation of
  claim:stage4_pymdp_posterior_max_abs_dev. The recorder's inline filter is
  bound to `decode_case_positions` hit for hit, so the comparison is against
  the recorded Stage 4 score and not a parallel implementation of it.

The recorder refuses to write if any argmax count is non-zero, if any
deviation exceeds the float32 rounding tolerance it declares, if a belief is
unnormalized, if the L4 walk covers a different population than the scorer, or
if the inline L4 filter drifts from the shipped scorer.

**Verification surface.** `tests/test_pymdp_l5_l6_equivalence.py` now asserts
the joint posterior at every slice position within the same tolerance and adds
`test_identity_transition_breaks_the_posterior_agreement`, which pins both
halves of the finding: the identity `B` must score zero type-argmax
disagreements on the slice (proving the argmax count is not sufficient) and a
joint-posterior deviation far above tolerance (proving the tolerance detects
the class the block swap cannot). `tests/test_figures_inference_evidence.py`
gains two refusals: a non-zero L4 disagreement count, and an L4 walk that
stops short of claim:stage4_derived_test_tokens tokens.

**What the L6 rows do not do.** The L6 characterization stays on its entry 92
document slice (`L6_PREFIX_DOCUMENTS`). Those rows compare two different
estimands rather than gating an equivalence, so widening them is a separate
decision and is not made here.

**What this is and is not.** At both levels the headline score path is
unchanged: L4 remains the NumPy forward filter and L5 the closed-form mixture.
The recorded agreement is posterior agreement between the live Agent and the
filter over the full held-out set, which licenses attributing those scores to
the model the Agent stores. It is not score equivalence for the L5 headline
mixture estimand, and the manuscript, `docs/hierarchy.md`, the README and the
architecture documents now say so in the same words, replacing the earlier
"bounded"/"matrix conformance only" wording that contradicted entry 92.

**Family discipline.** No new `_p` label; five new `descriptive` labels wired
through `REQUIRED_CLAIMS`, with the L4 count and the L5 deviation added to
`CANONICAL_LABELS`.

## 95 — The generator samples the closed-form policy posterior; its agreement with infer_policies is recorded (2026-09-01)

**What was wrong.** README, `docs/hierarchy.md`, `docs/formalism_map.md`,
`docs/gnn_composition.md` and `docs/design/generation_problem.md` all said that
`infer_policies` selects the next letter during generation. It never has.
`generate_by_policy_selection` samples every action from `efe_policy_posterior`,
a closed form over `gamma`, `C` and `E` that is fast enough to run thousands of
steps; the only place `agent.infer_policies` was called was one unit test
comparing the two on a handful of (letter, prefix) pairs, and no ledger label
recorded that agreement. The prose therefore attributed generated text to a
pymdp primitive the shipped path does not execute, with the licensing check
living outside the evidence ledger.

**What changed.** Every surface now says the policy posterior is computed in
closed form and verified against `agent.infer_policies`, and the verification
is a recorded, refusable measurement: `scripts/record_generation_claims.py`
installs `C` and `E` on the operating-point agent for every current letter
paired with the unit test's fixed prefix set, compares `infer_policies` with the
closed form, refuses to write above its declared tolerance, and records the
worst deviation as claim:generation_policy_posterior_pymdp_max_abs_dev
(`analytic`: a deterministic comparison of two evaluations) over
claim:generation_policy_posterior_pymdp_states_checked states
(`descriptive`). Both are wired through `REQUIRED_CLAIMS`; the generation
section of the manuscript cites both tokens.

**Family discipline.** No new `_p` label and no change to any significance
family.

## 96 — Manuscript claim-strength corrections: the calibration gate, the word-state generator, online learning, the prior-only channel, and the abstract (2026-09-01)

**What was wrong.** A cross-vendor review of the manuscript against the ledger
and this log found prose that had drifted below or above what the record
supports. Five defects, all prose-only, are corrected here; no measurement,
ledger label, Lean statement or figure changes.

1. **The calibration gate (`05_discussion.md`).** The limitations subsection
   still described the captured-mass proxy as the admission gate and the
   rule-divergence check as "registered future work" after decision 93 had
   adopted the direct check. It now reports the proxy in the past tense and the
   direct dual-rule gate as adopted: qwen25 reports; gemma3 is withheld, its
   recorded divergences claim:bench_calib_rule_ece_divergence_gemma3 (expected
   calibration error) and claim:bench_calib_rule_brier_divergence_gemma3
   (Brier) both exceeding claim:bench_calib_rule_epsilon although the floor had
   admitted it at claim:bench_calib_captured_mass_gemma3 captured mass; smollm
   is withheld by both rules. The floor constant stays as provenance for the
   starved-fraction diagnostic and the floor-locating sweep, not as a rule.
2. **The word-state generator.** `05_discussion.md` said a word-identity
   hidden state "would" give the generator a reason to prefer one letter over
   another. Section 8 measured that construction (decision 78): information
   gain claim:generation_word_state_ig_empty_nats nats at the empty prefix, a
   per-action spread of the expected free energy of exactly
   claim:generation_word_state_max_g_spread_nats, and a policy posterior on
   the habit to within claim:generation_word_state_policy_habit_max_abs_dev.
   The limitation and the introduction now report the measurement, and the
   conclusion's forward-looking sentence names action-dependence as the
   missing condition and points at the budgeted-attention arm as the measured
   case where the epistemic term is the whole policy.
3. **Online learning.** `04_results.md` and `10_verification.md` announced the
   online-versus-frozen result (decision 28) as "a further positive result ...
   on a prediction registered before the measurement" with none of the
   manuscript's strength labels. Every `online_*` entry is `descriptive`,
   entry 28 is not in `data/registration_provenance.json`, the arm is outside
   the significance family, and no interval exists. The prose now says exactly
   that: descriptive status, outside the family, a point comparison with no
   interval, prediction committed before the run, not promoted.
4. **The prior-only channel.** The introduction and abstract counted the
   L6→L5 intervention among the measured receiver interventions without noting
   that no held-out clause score under the blended prior exists. The registry
   already classifies it `PRIOR_ONLY` (`src/ailm/channel_registry.py`) and the
   variable map already derives `CHANNEL_PRIOR_ONLY_COUNT_WORD`; both passages
   now use it and point at the narrative-level section.
5. **The abstract and the repeated enumeration.** The abstract had grown to
   roughly three times the length an abstract needs and told the per-level
   score-path inventory twice, with the same inventory repeated in the
   introduction, the results and the conclusion. It is cut to roughly a third
   of its length; the four bounded-search phrases the priority-claim tests pin
   are kept verbatim; the dangling "Its" antecedent is fixed; and the sentence
   "a receiver-side intervention has no sign in advance", which contradicted
   the in-session lexical prediction (entry 18) and the pre-registered
   clause-channel prediction (entry 27), now says the sign is not inferable
   from the sender and reports which predictions held. The enumerations in the
   introduction, results and conclusion are replaced by pointers to
   `10_verification.md` and `tbl:level_stack`, keeping the L4 and L5
   full-set posterior-agreement sentences of decision 94. The introduction's
   "Structure of the argument" paragraph now follows the rendered order
   (gate design before the level chapters; complexity named once, as a
   supplement), and `MANUSCRIPT_STATUS.md` records the pending
   main-text/supplement split and the intentionally vacant section prefix.

**Guards.** Existing: `tests/test_manuscript_tokens.py` and
`tests/test_docs_hydration.py` (every quantity is a resolving token; the
channel-count tokens remain in `04_results.md`),
`tests/test_priority_claim_scope.py` and `tests/test_literature_search.py`
(the abstract's bounded-search phrases), `tests/test_documentation.py`
(retired status wording). No new guard: these are wording corrections whose
truth conditions are already asserted by the tests that read the tokens.

## 97 — The manuscript states the replication's retirement; the ud_ewt_test members await a principal decision (2026-09-01)

**What was wrong.** Decision 91 retired the decision-79 confirmatory
replication by exhaustion for the `external_lexicon` members, and its own
"what changes in the manuscript" paragraph said the promissory note must then
be dropped. It was not: `04_results.md`, `08_conclusion.md`,
`01_introduction.md` and three passages of `98_supplement_statistics.md` still
described the run as "pre-registered rather than performed", as something that
"has not happened", or as pending "until that run is performed", and the
supplement said the EWT `test` file "has never been read by anything here".
The `CONFIRMATORY_REPLICATION` docstring in `src/ailm/statistics.py` said the
same. That last sentence is no longer true as written: the case-count twin
generator has read that split (measurement only; no family member has been
scored against it).

**What changes.** A literal token `DECISION_REPLICATION_RETIRED`, whose value
is this log's entry 91, joins `data/manuscript_literals.yaml` and
`_LITERAL_VARIABLE_NAMES`, with the fixture entry in
`tests/test_manuscript_variables.py`. Every passage above now states the same
facts and nothing more: the run was pre-registered, executed to its first data
touch, and retired by exhaustion for the `external_lexicon` members; the
`ud_ewt_test` members remain unscored; and their status is an open principal
decision recorded in `TODO.md`. The docstring says that no family member has
been scored against the split. This entry does not assert that the EWT
partition is burned or unburned, does not close protocol `"79"` in
`data/registration_provenance.json`, and adds no second literal token; those
wait on the principal's ruling.

**Guards.** `tests/test_manuscript_variables.py` (the literal registry must
match `_LITERAL_VARIABLE_NAMES` exactly and every decision literal must name
an existing entry), `tests/test_manuscript_tokens.py` (the token resolves in
every passage that uses it), `tests/test_documentation.py` (retired-status
phrases).

## 98 — Statistics prose bound to the ledger: benchmark power, cluster directions, Stage 5 coverage, and partition p-values (2026-09-01)

Four prose statements about statistics had drifted from the ledger. Each is
corrected, and where the statement is a direction or a relation that a
re-recorded run could flip, a test now re-derives it from the recorded values.

**Benchmark power.** `98_supplement_external_benchmarks.md` said of the
alphabetic arms "Detectability here is low even so ... the arm could not
reliably have found an effect the size of the reference share". At the census
pool that is false: the power derived from each arm's recorded
`bench_letter_az_paired_b_*` / `_c_*` counts at the reference share
claim:bench_power_reference_share and the reporting level
claim:bench_power_alpha is well above adequacy for every probe, and stays
above one half even at the corrected family threshold
claim:bench_family_alpha. The sentence now says the arms are well powered at
the reporting level and that power at the corrected threshold is lower, as it
is for every arm in the section; the "only where its own derived figure says
so" qualifier is dropped, since a contrast that fails the corrected threshold
is unresolved rather than absent whatever its power figure (the bounds
convention); and the reporting sentence names the uncorrected level
explicitly, with the two-threshold comparison of entry 30 unchanged. Guard:
`tests/test_benchmark_interpretations.py::test_the_alphabetic_arms_are_well_powered_at_the_reporting_alpha`
re-derives adequacy at both thresholds from the pair counts and asserts the
retired wording is gone, with a negative control.

**Cluster directions.** `04_results.md` stated per-arm variance-inflation
directions that were false at HEAD — it said the G2P ratio exceeds one and the
Stage 4 derived and both Stage 6 ratios fall below it, against recorded values
of claim:cluster_stage2_g2p_variance_inflation,
claim:cluster_stage4_derived_variance_inflation,
claim:cluster_stage6_logloss_variance_inflation and
claim:cluster_stage6_sign_variance_inflation. The `tbl:cluster` caption said a
ratio below one is produced by a negative intraclass correlation, while the
Stage 2B row prints its ratio beside a positive claim:cluster_stage2_g2p_icc.
That ratio is a Monte Carlo quantity from claim:cluster_resamples replicates,
its distance from one is within the bootstrap's resolution, and reseeding
moves it across one, so its sign is not interpretable. The results sentence
now says only that clustering widens some arms and tightens others; the
caption states the resolution qualification; the Stage 2B paragraph calls the
ratio indistinguishable from one. Guard: `tests/test_cluster_prose_directions.py`
locks the direction words the supplement still states — the Stage 6 sign arm
below one with a negative intraclass correlation, the Stage 4 derived arm
above one with a positive one, at least one arm on each side of one, and the
Stage 2B ratio within two Monte Carlo standard errors of one with a positive
correlation whose analytic design effect is only marginally above one — with
a planted-failure companion for each relation.

**Stage 5 coverage.** The supplement's Stage 5 paragraph printed the
classification interval without its coverage, and the harmonization paragraph
said `SHARE_CONFIDENCE` and `DIFFERENCE_CONFIDENCE` are one constant without
naming the Stage 5 exception. The Stage 5 bootstrap intervals are read at
claim:stage5_oracle_gain_ci_confidence_pct% under `STAGE5_BOOTSTRAP_CONFIDENCE`,
which decision 66 kept separate because they are percentile intervals rather
than the share interval; every Stage 5 verdict is invariant to that level. The
supplement now prints the coverage and names the exception, the constant in
`src/ailm/levels/clause.py` says so where it is defined, and entry 83 records
where its coverage comes from.

**Partition p-values.** `tbl:case_oracle_paths` printed the three oracle-path
McNemar p-values with a caption that did not say they are uncorrected and
outside the declared family; that exclusion lived only in
`EXCLUDED_FROM_FAMILY`, which no reader of the rendered manuscript sees. The
caption now says they partition the hypothesis
claim:stage4_derived_mcnemar_p already scores, are uncorrected, sit outside
the family by recorded exclusion, and are printed to show the aggregate is a
mixture; the morphology paragraph no longer says the stratum's outcome is "not
shown" while the table shows it. The `tbl:channel_factorial` caption likewise
says its diagonal p-values duplicate family members and are not counted
twice. Guard:
`tests/test_statistics.py::test_every_table_printing_an_excluded_p_value_says_so_in_its_caption`,
beside `EXCLUDED_FROM_FAMILY`, stated over the general class (any table body
printing a token for an excluded label, with the caption before or after the
table) and paired with a planted-failure companion.

## 99 — Registration provenance inventories protocols 18, 25 and 27; confirmatory means registration commit before outcome commit (2026-09-01)

**What was wrong.** `README.md`, `docs/hierarchy.md`, `08_conclusion.md` and
`09_hierarchy.md` called the L3→L2B lexical channel "pre-registered", and
entry 18 was titled "a pre-registered test", although entry 18, its recorder
`scripts/record_lexical_channel_claims.py`, the sweep and entry 19 first enter
Git history in one commit (`bc070dea6a475f238b68092f0cceafbc3ad8b548`); the
same wording was emitted from `scripts/generate_lean_ledger.py` into
`formal/AILM/LedgerData.lean`. `src/ailm/claims.py` defined `confirmatory` as
"a docs/decisions.md PRE-REGISTRATION entry", which several such headings
outside the confirmatory set contradict, and `scripts/assign_claim_status.py`
listed "preregistration-titled entries 22-23, 27, 28 and 36", wrong in both
directions (entry 23 is titled CONFIRMED; entries 25, 38 and 42 carry the
heading and were omitted). `data/registration_provenance.json` omitted three
protocols the Git chronology can classify: 18 (same commit), and 25 and 27,
whose registration commits strictly predate the first commits of their
recorders.

**What changes.** Protocols `"18"`, `"25"` and `"27"` join the provenance
registry with their registration and outcome commits and the temporal classes
the chronology supports, each with a note stating what its eligibility does
and does not mean; `tests/test_claims_status.py` extends the pinned inventory
and checks entry 18's heading. Entry 18 is retitled a PROSPECTIVE
DEVELOPMENT-SESSION SPECIFICATION with a dated correction paragraph, and its
index row follows. The public prose says "prospectively specified in-session"
for the lexical channel; the Lean ledger generator's grid docstrings say the
same and `formal/AILM/LedgerData.lean` is regenerated byte-identically. The
`claims.py` and `assign_claim_status.py` docstrings now state the operative
rule: the registration commit strictly predates the outcome commit per
`data/registration_provenance.json`, machine-checked against Git history.
`STATUS_BY_SOURCE` and every status in `data/claims.json` are unchanged:
eligibility for entries 25 and 27 is design-time, and their registered
outcomes — a primary interval spanning zero, and a predicted harm that did
not reproduce — leave their recorders at their mechanical tiers.

**Guards.** `tests/test_claims_status.py::test_registration_provenance_is_bound_to_git_history`
(the pickaxe, first-add and ancestry checks run for every protocol, the three
new ones included),
`tests/test_claims_status.py::test_same_commit_protocols_are_not_titled_preregistrations`
(entry 18's heading), `tests/test_decisions_index.py` (the retitled row and
the new rows), `tests/test_formal_isomorphism.py::test_regeneration_is_byte_identical`.

## 100 — Figures draw only what they claim: bounded gutter, contained verdicts, ledger-sourced declared gate, named withheld probes, drawn intervals, typed figure package (2026-09-02)

**What was wrong.** A figure review found seven ways the rendered figures said
something the code, the ledger, or the caption did not. `fig:hierarchy` drew a
double-headed arrow from the letter level to the document-position level under
a label that describes only the letter-to-phoneme alphabet identity — the arrow
kept the geometry of a retired "adjacent levels are coupled" claim after the
label was corrected, so the picture still asserted a level-spanning cascade
that `docs/hierarchy.md` rules out. In the same figure the Stage-column
verdicts ran past the table border and the footnote printed through the Stage
0 band, and no gate could see it: `assert_within_canvas` checks the canvas
edge and the layout sweep checks text on text, so text crossing a *border*
passed both. `fig:information_ceiling` and `fig:gate_taxonomy` drew the
original Stage 1 target from a literal typed into the figure module while the
recorder wrote the same declaration to the ledger under
`stage1_declared_gate_nats`; a re-recorded declaration left both figures
unchanged, which a live experiment confirmed. Three alt-text entries described
marks that were not on the canvas (an interval that existed only as title
text, a forest without its fragility strip, a level stack without its
interface row). The external-benchmark calibration panel dropped two
registered probes without saying so, and the taxonomy legend advertised an
UNAUDITED zone no row draws. Muted-grey notes sat inside filled bars at a
contrast no reader can use. And the typing gate excluded the whole figure
package, uncommented, since the day it was added.

**What changes.** The gutter arrow runs from the letter row to the phoneme
row only, one way, with a two-line label; the table columns are rebalanced so
every verdict ends inside the border and the footnote sits below the band.
`ailm.gates.STAGE1_DECLARED_GATE_NATS` is the single owner of the declared
Stage 1 target: the recorder writes it, and both figures read the ledger token
and refuse with "Ledger inconsistency" if the two disagree, exactly as the
Stage 2B threshold already worked. The three alt-text entries describe what
is drawn. The calibration panel names the withheld probes in-panel (decision
93) and the caption says so; the legend lists only the verdict zones some row
draws, with the UNAUDITED entry returning when a ceiling is unrecorded. The
inference-evidence notes sit in the headroom above their bars. The clause-null
figure draws the paired position-weighted gain and its interval against a
zero line in an inset, in its own unit, rather than quoting it in the title
alone. The figure package is inside the typing gate; the release test refuses a
`[tool.mypy] exclude` so the gate cannot be narrowed silently again. No
recorded value changed; the declared gate is the same number the ledger held.

**What was deferred, then done.** Wilson intervals for the prefix-window and
phone-count benchmark panels add per-stratum claims through the network
recorder; they landed the same day as entry 101.

**Guards.**
`tests/test_figures_architecture.py::TestArchitectureFigures::test_hierarchy_arrows_span_at_most_two_adjacent_rows_one_way`
and `test_hierarchy_text_stays_inside_its_column_and_the_table`, each with a
positive control that reinstates the defect;
`tests/test_figures_results.py::test_declared_gate_must_agree_with_the_code_owned_constant`
(the re-recorded declaration that used to pass silently) and
`test_gate_taxonomy_legend_names_exactly_the_drawn_zones`;
`tests/test_figure_alt_text.py::test_every_named_mark_is_drawn_by_its_builder`
with planted controls;
`tests/test_figures_levels56.py::test_stage5_figure_draws_the_paired_interval_from_the_tokens`;
`tests/test_release_orchestration.py::test_profiles_have_distinct_bounded_acceptance_surfaces`
(no mypy exclude); `tests/test_figure_layout.py` over every rebuilt figure.

## 101 — Every stratified benchmark bar carries a Wilson interval on its own denominator (2026-09-02)

**What was wrong.** `fig:external_benchmarks` mixed intervalled and
un-intervalled bars without saying why: the headline accuracy, paired
difference and stipulated-language panels carried Wilson or exact intervals,
while the letter-variant panel (the two prefix windows and the alphabetic
arm) and the phone-count panel drew bare points. The harness had every count
needed — the alphabetic restriction keeps the per-position outcomes, each
bucket keeps its gold phone positions and correct positions, and the two
prefix windows are full letter tasks that already computed their bounds — but
the recorder registered point accuracies only, and the figure could not draw
an interval the ledger did not hold.

**What changes.** The alphabetic restriction and every G2P length bucket
report Wilson bounds on their own denominators (the restricted positions; the
bucket's gold phone positions), the context arm carries the bounds its two
letter tasks already produced, and the recorder writes them for each probe
and for the comparator under `bench_letter_{short,long,az}_top1_wilson_*`
and `bench_g2p_pos_{bucket}_wilson_*`, plus `bench_g2p_bucket_{bucket}_phones`
as the bucket denominators. No request changes: the new labels are
restrictions and counts over the same recorded replies, so the exact cached
call set rederives them under the network-claim binding. `REQUIRED_CLAIMS`
lists every new label; the figure draws the two panels with error bars and
names the confidence in their titles; the caption states the denominators;
the drift test covers the new series token by token.

**Guards.** `tests/test_figures_results.py::test_external_benchmarks_values_match_the_ledger`
(the context and bucket interval rows join `_bench_token_rows`),
`tests/test_manuscript_variables.py::TestRealLedger::test_real_ledger_satisfies_every_required_claim`,
`scripts/verify_network_claim_bindings.py --recorder scripts/record_external_benchmark_claims.py`
(the replay must rederive the complete namespace, new labels included),
`tests/test_figure_layout.py` over the rebuilt figure.

## 102 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the family-grouped G2P sensitivity arm; its heterogeneity prediction is refuted (2026-09-02)

**Provenance.** Written in the session that ran the measurement, so like
decisions 73 and 74 this is a prospective in-session specification, not an
independently timestamped pre-registration: the recorder and this text enter
history together. Status exploratory by construction; nothing here is later
reclassified as confirmatory.

**Where it sits.** Decision 74's family-disjoint arm removed
morphological-neighbor leakage by assigning whole prefix families to one side
of the split, and scored the held-out side as one pool: a margin of
claim:stage2_g2p_famsplit_margin_pts points over the per-letter argmax bar on
claim:stage2_g2p_famsplit_eval_positions positions. The open question, raised
by the limitations section's "the length-matched subset is not English" point,
was whether that margin is a level-wide property or is carried by particular
families. This arm keeps the identical split, decoder, comparator and paired
test and changes only the reporting unit: from one pool to per-family groups
keyed by the same prefix the split assigns sides with. Nothing is re-fit per
family.

**Prediction (P1), registered before the run.** Heterogeneity in the sense that
more than half of the families with at least
claim:stage2_g2p_famgrp_min_positions scored positions carry a positive
margin. Half or fewer means the pooled margin is a majority-null average and
the manuscript may not describe it as level-wide.

**Controls, registered before any ledger write.** Positive: the
position-weighted pool over the per-family records must reproduce the famsplit
accuracy, bar, margin and discordant counts, or the partition is broken and
nothing is recorded. Negative: the first draft rotated whole correctness
vectors one family over, which preserves the multiset of margins under any
permutation of families and so could never fail; it was retired with that
reason before the measurement was recorded anywhere. Its replacement reverses
the pooled model and comparator correctness vectors in position order and
reassigns them to families by the same size counts, which keeps every marginal
statistic and destroys only the family-to-result pairing. Under reversal the
largest absolute per-family margin must be strictly smaller than the real
assignment's.

**Outcome: P1 refuted.** Of claim:stage2_g2p_famgrp_evaluated_families
families at or above the minimum, claim:stage2_g2p_famgrp_positive_margin_families
carry a positive margin, a share of
claim:stage2_g2p_famgrp_positive_margin_share against the registered bar of
more than one half. The famsplit margin is a majority-null average, not a
level-wide property, and `docs/manuscript/05_phoneme_level.md` says so through
tokens. The controls held: the pool reproduces the famsplit accuracy
(claim:stage2_g2p_famgrp_pool_accuracy against
claim:stage2_g2p_famsplit_accuracy) with the same discordant counts
(claim:stage2_g2p_famgrp_pool_mcnemar_b and
claim:stage2_g2p_famgrp_pool_mcnemar_c), and the reversal's largest margin
claim:stage2_g2p_famgrp_reversal_max_abs_margin sits below the real
assignment's claim:stage2_g2p_famgrp_real_max_abs_margin, so the
heterogeneity belongs to the families and not to the scoring. A refuted
prediction is a result: the arm is kept as measured, the largest families are
recorded as bounded rank rows, and their p-values are diagnostics of a
diagnostic arm, outside `SIGNIFICANCE_FAMILY` under the decision 74
exclusion.

**Guards.** `scripts/record_g2p_famsplit_claims.py` (single writer for the
`stage2_g2p_fam*` namespace) refuses to write when either control fails or
when any family's model and comparator position lists differ in length.
`tests/test_g2p_famsplit.py::test_famgrp_positive_control_pooled_reproduction_on_the_real_artifact`,
`test_the_recorded_famgrp_arm_holds_its_controls`,
`test_family_margin_summary_pools_positions_exactly`,
`test_family_margin_record_refuses_unpaired_vectors`,
`test_the_famgrp_labels_are_recorder_written`.

## 103 — Part A perplexity is also measured on a held-out word split, in closed form (2026-09-02)

**What was open.** The Part A phone bigram's perplexity
(claim:stage2_phoneme_perplexity against the context-free bar
claim:stage2_phoneme_unigram_baseline) was measured over the full dictionary
in sample (decision 6), and `05_discussion.md` defended that with a
parameter-to-data-ratio argument rather than a measurement. This entry
converts the defense into a measurement without touching the in-sample
protocol, which stays the descriptive dictionary number.

**Split rule, registered before the run.** The sorted list of
claim:cmudict_base_word_count base words in the generator's own ordering;
a word is held out when its index is a multiple of the stride
claim:stage2_phoneme_heldout_split_stride, and everything else trains. That is
the artifact's G2P `test_every` convention one level up: an evenly spaced
stride, never a prefix (decision 4). Sizes:
claim:stage2_phoneme_heldout_train_words training words,
claim:stage2_phoneme_heldout_test_words held-out words,
claim:stage2_phoneme_heldout_eval_positions scored positions. The unigram and
bigram are re-fit inside the analysis code from the pinned CMUdict snapshot
with the level's add-one smoothing and per-word prior reset, and the identical
closed form is scored on the disjoint sample.

**Predictions, registered.** Held-out perplexity sits above the in-sample
value by less than one perplexity unit; the context-free bar, re-fit on the
same training words and scored on the same positions, is still beaten. The
falsifier is a gap at or below zero or a penalty of one unit or more, recorded
as a bound either way.

**Controls.** Positive: the full dictionary fit and scored as one sample
through the split-aware closed form must reproduce the in-sample closed form
and bar to the declared relative tolerance with identical position counts,
which licenses reading the difference as a split effect rather than a
code-path difference. Negative, as first registered: phones shuffled within
each training word, predicted to lose all transferable structure. That
control fired. The within-word-shuffled fit beat the held-out bar by
claim:stage2_phoneme_heldout_shuffled_gap_nats nats, because shuffling within
a word preserves which phones co-occur in it, and bag-of-phones co-occurrence
is real information about English. The recorder refused and no label was
written. The control was redesigned before the re-run to a cross-word re-deal
(the training stream globally shuffled under a fixed seed and re-chunked at the
original word lengths), which preserves the unigram and every word length
while destroying adjacency and within-word co-occurrence, with its gap bounded
at a small declared tolerance. The fired result is kept as a measurement
rather than deleted.

**Outcome: held.** Held-out perplexity claim:stage2_phoneme_heldout_perplexity
against the held-out bar claim:stage2_phoneme_heldout_unigram_baseline; a
penalty of claim:stage2_phoneme_heldout_penalty_ppl perplexity units over the
in-sample value; a gap of claim:stage2_phoneme_heldout_gap_nats nats,
essentially the whole in-sample advantage. The re-dealt control's gap
claim:stage2_phoneme_heldout_redeal_gap_nats sits inside its bound. The
honest gap exceeds the bag-only gap, so adjacency carries information beyond
co-occurrence, and exceeds the re-dealt gap by far more than the registered
order of magnitude. `05_discussion.md` replaces the "Part A is evaluated
in-sample" open item with the measured statement through tokens.

**Guards.** `scripts/record_stage2_claims.py` (run `stage2_phoneme_gate`)
recomputes the full-dictionary sufficient statistics from the pinned snapshot
and refuses unless they equal the vendored artifact's counts exactly, asserts
split disjointness and coverage, and runs both controls at record time.
`tests/test_stage2_gate.py::test_parta_heldout_split_is_a_stride_not_a_prefix`,
`test_parta_split_scoring_reproduces_the_in_sample_closed_form`,
`test_parta_split_scoring_matches_a_sequence_walk`,
`test_parta_heldout_gap_gate_is_live_and_redeal_control_fails`,
`test_stage2_heldout_claims_are_self_consistent`. Every new label is
`descriptive`; no `_p` label and no family change.

## 104 — The wild cluster bootstrap is a recorded sensitivity arm beside every clustered interval (2026-09-02)

**What was open.** The supplement's cluster section carried an open-item
clause: with few clusters the cluster bootstrap can under-cover, and no second
construction checked the recorded verdicts. This entry adds a wild cluster
bootstrap (Cameron, Gelbach and Miller 2008, Rademacher weights at the cluster
level) as a sensitivity arm for every arm `scripts/record_cluster_claims.py`
covers: the word-clustered Stage 2 G2P arm and the document-clustered Stage 4
gold, derived and Czech arms and Stage 6 sign and log-loss arms. Sensitivity
only: the cluster bootstrap keeps primary uncertainty.

**Choices, registered before implementation.** Weights are Rademacher, plus or
minus one with equal probability, drawn independently per cluster per
replicate; no Mammen or Webb refinement. Each replicate re-forms the
observation-weighted mean from whole-cluster demeaned contributions, so the
point estimate is unchanged and only the perturbation of the demeaned cluster
sums carries uncertainty; with these weights the replicate variance equals the
cluster-robust variance of the mean exactly, so the percentile interval is a
symmetric interval on the same estimand. Replicates claim:wild_resamples and
seed claim:wild_seed are the clustered path's own (`DEFAULT_RESAMPLES`, and
`DEFAULT_SEED` spawned into its own child stream so no other procedure's draws
move); confidence claim:wild_confidence comes from the module default and the
manuscript reads `{{WILD_CONFIDENCE}}` rather than a literal. Prediction: the
wild spans-zero verdict agrees with the recorded cluster-bootstrap verdict for
every arm, because the recorded clustered and naive intervals bracket the
symmetric cluster-robust spread on every arm (the gold arm's all-zero vector
gives the degenerate point interval on both sides).

**Outcome: held.** claim:wild_verdict_agreements of claim:wild_arm_count arms
agree. The endpoints are recorded per arm under `wild_{arm}_ci_low` and
`wild_{arm}_ci_high` (for the G2P arm, claim:wild_stage2_g2p_ci_low to
claim:wild_stage2_g2p_ci_high) beside `wild_{arm}_verdict_agrees`; a
disagreement would have been recorded as a finding, never refused. The
supplement's open-item clause is replaced by a subsection and
`tbl:wild_cluster`, all through `{{WILD_*}}` and `{{CLUSTER_*}}` tokens.

**Guards.** `tests/test_clustering.py::test_wild_singleton_clusters_reproduce_the_naive_interval`
(positive control: singleton clusters recover the independence interval within
the declared endpoint and width tolerances),
`test_planted_strong_cluster_effect_widens_the_wild_interval` (negative
control: planted within-cluster dependence widens the wild interval by a
material factor),
`test_wild_replicate_variance_matches_the_cluster_robust_closed_form`,
`test_wild_verdict_tracks_the_cluster_bootstrap_verdict` (the comparator is
the cluster bootstrap, not the naive interval),
`test_wild_summary_is_deterministic_and_digest_pinned`. The `wild_*` labels
are wired through `REQUIRED_CLAIMS`, and the verdict flags are declared
boolean in `manuscript_variables.py` so the render gate does not misread them
as numbers. No `_p` label; no family change.

## 105 — The orphan gate measures its own dark share: three real consumption surfaces and a per-family threshold (2026-09-02)

**What was wrong.** `tests/test_claims_discipline.py::test_no_recorded_measurement_family_reaches_nothing`
passed a family of five or more labels as soon as any one label was
referenced, so the largest families passed while mostly dark. The dark count
itself erred in the other direction, because two real consumption paths are
invisible to a literal search: the derivation emits generated table blocks
the prose cites only through container tokens (`BENCH_CTX_TABLES`,
`PARAMETER_TABLE_ROWS` and their kin), and figure modules discover tokens by
prefix scan rather than by name. Under the prose-only surface a large
minority of the ledger's labels read as dark; most of that was the instrument,
not the ledger.

**What changes.** The consumed set is the union of three surfaces, all real:
literal `{{TOKEN}}` and `claim:` references in prose, as before; labels
explicitly read during one real `generate_manuscript_variables` call, observed
through a recording `dict` subclass whose item reads remember the key
(iteration deliberately does not count, since emitting a token per label is
not a decision to cite a measurement; the loader runs unmodified on the
validated ledger, an observation wrapper rather than a mock); and labels
reachable through the token stems figure modules name literally, by an AST
walk over string constants including f-string prefixes and prefix-scan
regexes. The stem filter is load-bearing: axis text, path fragments and family
keys interpolated into other tokens would otherwise mark whole families
consumed, and tightening it re-darkened one family the loose filter had
masked. The pass rule becomes a per-family dark-share threshold: a family of
five or more fails when more than half of it is dark. The non-vacuity control
on the count of large families is kept.

**The set-aside allowlist.** `_MEASURED_AND_SET_ASIDE_FAMILIES` names each
family that stays above the threshold on purpose, one reason per family citing
the decision that measured it: the clause-to-word alpha sweep (decisions 27,
88 and 89), the Stage 2 and Stage 4 cluster diagnostics (decisions 98, 77 and
85), the word-state generator degeneracy (decision 78), the geometric-pooling
controls (decisions 47 and 50), the adposition head-resolution census
(decisions 84 and 63), and the fleet's family-grouped and wild-cluster
families with their own reasons. An allowlist entry is a claim about the
ledger, and the new negative control fails any entry the ledger no longer
supports (family absent, below the gate size, or no longer mostly dark), so an
entry can be removed only by a decision, never outlived silently. Under a
consumption surface the three cannot see, the gate's error direction is
false-dark, which fails loudly, never false-clean. No claim was deleted;
ledger, recorders and prose are untouched.

**Guards.** `tests/test_claims_discipline.py::test_no_recorded_measurement_family_reaches_nothing`
(the threshold rule) and
`test_set_aside_allowlist_entries_still_describe_dark_families` (the
staleness control), sharing one cached gate computation per session.

## 106 — Conformance's live inputs are a typed record, not a positional signature (2026-09-02)

**What was wrong.** `check_conformance` took roughly two dozen untyped
positional parameters that every caller splatted from `describe_shapes()`. A
signature of that shape cannot be type-checked at call sites, cannot reject a
misspelled slot name (an unknown keyword vanished into silence, leaving a slot
unverified in production), and duplicated its own field documentation between
`describe_shapes` and `check_conformance`.

**What changes.** One frozen dataclass, `LiveConformanceInputs` in
`src/ailm/gnn/types.py` (the module that already owned the shape and spec
vocabulary), carries every live-model field the old signature did;
`describe_shapes()` returns it, `check_conformance(spec, live)` and
`assert_conforms(spec, live)` consume it, and `spec_agent.py` builds it. The
field set is single-sourced and machine-checkable: mypy verifies every read,
`describe_shapes` refuses to grow a field the record does not declare, and
the record's constructor rejects unknown fields, so a slot renamed in the
checker fails at construction rather than silently un-verifying. A thin
adapter keeps the legacy keyword form working for the gate and edge-path tests
that splat partial descriptions, with the old "absent means unverified"
default preserved; mixing the two forms is a `TypeError`. The migration
exposed one shadowing bug, a local rebinding of the new parameter inside the
checker, fixed with the same values and messages.

**What settled it.** A verdict dump over every shipped spec, the shapes-only
verdicts and every registered mutation is byte-identical before and after
(both dumps and their digest sit in
`scratchpad/fleet_2026-09-02/ailm-conformance-typed/`), and the legacy leg of
that dump exercises the adapter. No claim changes; no ledger label is added.

**Guards.** `tests/test_gnn_conformance.py::TestTypedLiveRecord` (record and
legacy forms agree on the real letter agent; a missing field defaults to None;
an extra field is a `TypeError`, not a dropped value; the forms cannot be
mixed; `describe_shapes` returns the record and round-trips through
`as_dict()`), and the mypy gate over `src/ailm`.

## 107 — generate_manuscript_variables is a pure orchestrator over per-section producers, held to a zero-diff oracle at every step (2026-09-02)

**What was wrong.** `generate_manuscript_variables` in
`src/ailm/manuscript_variables.py` had grown into one function of more than
two thousand lines with eighteen nested closures over seven banner-delimited
blocks. Nothing in it was wrong on the day it was read, and nothing about it
could be reviewed: a token's producer was wherever the closure that computed
it happened to sit, and the shadowing scan in `tests/test_benchmarks.py` that
guards token collisions walked one body it could not partition.

**What changes.** The function is a short orchestrator calling one
`_produce_<section>` writer per section, split at the existing banners and
sub-headers, every producer taking a parameter named exactly `variables` first
and writing its tokens by subscript on that name; no producer returns a fresh
map. Closures were disposed by what they captured: capture-free helpers
hoisted to module level verbatim; the closures over the claims map bound into
a `_ClaimReaders` helper with the formatting methods derived from them;
cross-section state carried by typed records (`_SyntheticSweep`,
`_ContextReconciliation`, `_LexicalSweep`) so execution order, on which the
strict token map and the planted-defect message order depend, is preserved
exactly. `_load_claims` keeps its name, signature and module, so the orphan
gate of decision 105 still observes it by attribute swap.

**What settled it.** Behavior preservation was the acceptance criterion,
measured rather than argued: a frozen copy of the pre-refactor module and the
working module ran on one identical scratch root after every commit of the
split, and their outputs were diffed ignoring only the build timestamp, with
zero key and zero value differences at every step. The oracle harness
synthesized minimal valid entries for labels other lanes had declared but not
yet recorded, so mid-flight ledger churn could not masquerade as a diff. Ruff
and mypy were clean at every commit. No claim changes; no label is added.

**Guards.** `tests/test_benchmarks.py::test_every_manuscript_producer_is_in_the_shadow_scan_shape`:
every `_produce_*` function takes `variables` first and contains a subscript
write on it, so the module-wide shadowing scan cannot pass vacuously over a
producer whose writes it can no longer see. The architecture package map names
the orchestrator split.

## 108 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the stipulated phone chain beside the letter chain; the fitted comparator beats every registered probe (2026-09-02)

**Provenance.** The design and predictions below were fixed in the lane before
the phone generator, the artifact or any live call existed. That is ordering
inside one development session in the sense of entry 51's status correction,
not an independently timestamped registration; every label is `descriptive`
and none earns `preregistered` or `confirmatory`.

**The stipulated structure.** A second stipulated language beside the letter
chain of `src/ailm/levels/synthetic.py`, additive to the same module. The
inventory is the phoneme level's exact stress-free ARPAbet set
(claim:bench_synthphone_inventory_size symbols); no prompt, schema, parser or
gold stream may use any other symbol. The vowel and consonant classes are
stipulated with the chain, not learned. Each transition column is the letter
arm's three-part mixture with the letter arm's weights: a designed dominant
successor in the opposite class chosen by a coprime stride, a uniform share
over the opposite class, and a uniform floor over the whole inventory, so the
order-one argmax over a well-estimated column is exactly the designed
successor. The corpus is claim:bench_synthphone_corpus_symbols sampled symbols
from seed claim:bench_synthphone_corpus_seed under the letter chain's sampler
contract, counted into the flat bigram format `build_phone_transition`
already consumes and vendored as `vendor/synthetic_phone_counts.json` under
checksum registration. Evaluation is one independently seeded trajectory per
scored position from seed claim:bench_synthphone_position_seed, over
claim:bench_synthphone_positions positions with a window of
claim:bench_synthphone_query_window phones and a sensitivity prefix of
claim:bench_synthphone_short_query_window phone, with the target-start
divergence recorded (claim:bench_synthphone_target_start_tvd).
The protocol `synthetic-phone-zero-shot-v1` carries its own cache identity so
no key can collide with the letter arm; the probe contract is the existing
exact phoneme task with no change to the prompt or the client; the
`bench_synthphone_*` namespace declares its own two families, corrected
within themselves at claim:bench_synthphone_family_alpha.

**Prediction.** The fitted comparator, an order-one argmax over the add-one
smoothed phone transition estimated from the vendored counts, beats every
registered no-demonstration probe on the phone chain as it does on the letter
chain, because the probes must infer the designed transition from a bounded
prompt while the comparator was fitted on the language. Refutation: any probe
clears the family threshold in its own favor; the comparator fails its own
positive controls; or the corpus is not the chain.

**Controls, pinned before any live call as recorder refusals.** The
comparator must score at least a wide multiple of the uniform baseline
claim:bench_synthphone_uniform_baseline on the executed positions, at least
twice a comparator fit on the transposed count table, and at least twice an
English-trained comparator built from the vendored CMUdict phone bigrams;
sampled bigram frequencies must sit within the letter arm's tolerance of the
written-down transition, and the artifact must reproduce byte for byte from
the generator. A run that fails any of these writes nothing.

**Outcome: held.** The comparator's accuracy
claim:bench_synthphone_phone_top1_ailm stands against
claim:bench_synthphone_phone_top1_gemma3,
claim:bench_synthphone_phone_top1_qwen25 and
claim:bench_synthphone_phone_top1_smollm, with paired exact p-values
claim:bench_synthphone_paired_p_gemma3,
claim:bench_synthphone_paired_p_qwen25 and
claim:bench_synthphone_paired_p_smollm against the family threshold. The
one-phone sensitivity arm drops every probe to the floor or near it
(claim:bench_synthphone_short_top1_gemma3,
claim:bench_synthphone_short_top1_qwen25,
claim:bench_synthphone_short_top1_smollm), so what the probes had was
within-query chain evidence, not the response inventory. Every scored call
was fresh (claim:bench_synthphone_scored_cache_misses misses, no hits) and
no reply was unparsable. The letter arm re-recorded its own namespace
byte-identically in the same publication unit. The supplement subsection
"The same stipulation over the phoneme inventory" in
`98_supplement_external_benchmarks.md` reports it through tokens.

**What this arm cannot establish.** It does not prove absent pretraining
exposure, for these probes or for anyone. The construction reduces
exact-artifact overlap: the scored trajectories are generated locally under a
transition written down in code, disjoint from the training draw. It cannot
establish that an external model's undisclosed corpus lacks the same
sequences, the same inventory, an alternating vowel-consonant regularity, or
a related rule, all of which an English-exposed model has prior mass on. The
inventory is the canonical ARPAbet set the probes may have met in lexical
resources; the transition over it is what is new. A stipulated order-one
chain is not English, so the result licenses a statement about transfer to
this structure by these probes under this prompt contract, and nothing about
phonotactic competence in general. The mapping-derangement control that
bounds the in-context letter arm's demonstration effect is not replicated
here; nothing in this namespace is a deranged-demonstration result.

**Guards.** `tests/test_synthetic_benchmark.py::test_phone_transition_matches_the_written_down_mixture`,
`test_phone_dominant_successor_is_the_argmax_of_every_column`,
`test_phone_successors_cross_the_class_boundary_and_do_not_collapse`,
`test_sampled_phone_frequencies_converge_to_the_designed_transition`,
`test_phone_artifact_reproduces_from_the_generator`,
`test_phone_positions_are_independent_seeded_trajectories_not_one_stride`,
`test_phone_prompt_schema_parser_and_gold_share_one_exact_inventory`,
`test_phone_comparator_beats_the_uniform_baseline_by_a_wide_margin`,
`test_english_trained_comparator_collapses_on_the_phone_stream`,
`test_transposed_phone_count_table_collapses`,
`test_phone_comparator_refusals_pass_on_the_executed_population`;
`tests/test_provenance_checksums.py` over the vendored artifact;
`scripts/verify_network_claim_bindings.py --recorder scripts/record_synthetic_benchmark_claims.py`
(one binding names both cache protocols and re-derives both namespaces from
cached replies); `REQUIRED_CLAIMS` lists the whole namespace. No `_p` label
joins the frozen family; the phone p-values sit in their own declared
families and in `EXCLUDED_FROM_FAMILY`.

## 109 — The Holm step-down comparison is machine-checked beside the Bonferroni arithmetic (2026-09-05)

**What was added.** `formal/AILM/Holm.lean` machine-checks the arithmetic
behind the manuscript's sentence that every member the fixed Bonferroni bar
admits, Holm's step-down admits too, and the converse can fail. Ten theory
theorems, nine satisfiability witnesses and four negative controls, all over
`Rat` and `Nat` in core Lean with no Mathlib dependency, no `sorry` and no
`native_decide`: Holm's rank-`i` bar `α/(m-i+1)` is never stricter than
Bonferroni's `α/m` (`bonferroni_le_holm_threshold`), is nondecreasing in the
rank over every natural because truncated subtraction saturates the bar
(`holm_threshold_monotone_in_rank`), and coincides with Bonferroni's at rank
one (`holm_rank_one_is_bonferroni`); on a sorted family the step-down walk
rejects every member the fixed bar rejects
(`holm_rejects_every_bonferroni_rejection`), and a synthetic family exhibits
the walk rejecting strictly more. `sortednessIsLoadBearing` is an unsorted pair
on which the fixed bar rejects one member and the walk none, so the sortedness
hypothesis cannot be dropped.

**What it does not prove.** Familywise error control, and Holm's stopping rule
as a probabilistic guarantee, are outside the layer for the reason entry 55's
module gives for Bonferroni: there is no probability measure in the tree. The
layer proves rejection-count arithmetic, and only that.

**Guards.** `scripts/record_formal_claims.py` re-recorded the layer's own
counts: claim:formal_theorem_count named theorems, of which
claim:formal_theory_theorem_count are theory, claim:formal_axiom_report_verdicts
axiom-report verdicts, claim:formal_axioms_beyond_core axiom beyond core, no
placeholder tactic and no `native_decide`. `tests/test_formal_isomorphism.py`
pins the Holm module, runs its synthetic witness family through the real
`holm_step_down` in `src/ailm/statistics.py` and requires the same rejection
counts and per-rank bars, and still requires `Controls/MustFail.lean` not to
compile. The docs-tree companion is `docs/formal.md`.

## 110 — The lexical channel's backoff matched the wrong context; corrected to the documented longest-suffix rule, every dependent arm re-recorded (2026-09-05)

**What was wrong.** `LexicalMessage.lookup` in `src/ailm/levels/lexical_channel.py`
backed off by truncating the joined `position|context` key from the right,
which drops the newest letters, while every training key is a context that
ends at the current letter. An unseen context therefore matched a training key
whose current letter differs from the one being decoded, and the reported
depth was the length of the key string rather than the number of context
letters. The docstring stated the longest-suffix rule the code did not
implement. On the shipped held-out split a large minority of positions
resolved to a different key than the documented rule selects, most of them to
a context that does not end at the observed letter. The cross-vendor source
audit of 2026-09-05 (`scratchpad/review_2026-09-05/forge/REPORT.md`) found it
by reading the key construction against the lookup, and it was confirmed by a
toy in which position three of `abcq` matched the key for `abc`.

**What was corrected.** The lookup now takes the position and the context
separately and tries the suffix of each length from the message's own bound
down to one, returning the number of context letters matched as the depth and
the fallback at depth zero. Every recorded value that passed through the
lookup was re-recorded through its own recorder, in order: the lexical sweep,
the channel factorial, the fitted arm, the geometric-pooling arm, the
proper-loss cells, then the statistical aggregate and the status assignment.
The pre-correction and post-correction values are preserved side by side in
`scratchpad/review_2026-09-05/lexical/claims.diff.md`; every `claim:` tag in
this log resolves to the corrected ledger.

**What the corrected measurement says.** The channel helps more, not less: the
best weight moves to claim:lexical_best_alpha with a margin of
claim:lexical_best_margin_pts points at p = claim:lexical_best_mcnemar_p, and
the highest tested weight helps by claim:lexical_high_alpha_margin_pts points
at p = claim:lexical_sweep_alpha_0_9_mcnemar_p instead of costing accuracy. The
sample argmax is interior, but the peak leads the highest weight by only
claim:lexical_peak_vs_high_alpha_pts points on
claim:lexical_peak_vs_high_mcnemar_b against claim:lexical_peak_vs_high_mcnemar_c
discordant positions (p = claim:lexical_peak_vs_high_mcnemar_p), so the sweep
records an ordering it cannot distinguish from a plateau, and the factorial's
wider grid places the same message's best predictive-site cell at weight
claim:channel_lexicon_predictive_best_alpha. Entry 18's second predicted
signature, an interior optimum as the mark of novel information, is therefore
withdrawn; the reversal entry 19 reported at the high weight was an artifact
of matching the wrong context. At the evidence site the corrected lexical
message moves accuracy by claim:channel_lexicon_evidence_best_margin_pts
points at p = claim:channel_lexicon_evidence_paired_p, a null where the
defective lookup had recorded a significant cost, so the factorial's "both
evidence-site cells harm" reading becomes "the evidence site harms the letter
message and does nothing detectable for the lexical one". The fitted arm
re-selected its configuration through the corrected message
(`max_prefix` claim:fitted_max_prefix, `min_count` claim:fitted_min_count,
smoothing claim:fitted_smoothing, exponent claim:fitted_exponent against the
grid's claim:geometric_lexicon_best_exponent) and reaches
claim:fitted_accuracy%, and the geometric arm's best exponent moved to
claim:geometric_lexicon_best_exponent.

**What changed downstream.** The lexical evidence-site member left the
fixed-threshold survivor set, which now holds claim:stat_family_survivor_count
of the claim:stat_family_size declared tests, and re-ranking the family moved
the Stage 6 sentence-sign test past its Holm bar: its step-down verdict is now
claim:stat_holm_stage6_rejected at a threshold of
claim:stat_holm_stage6_threshold, so claim:stat_holm_only_count members
survive only the step-down. That is the instability of a step-down bar the
statistics supplement describes, observed. The lexical high-weight arm, which
had been the project's one bounded null with the pairs to support an
equivalence claim, is a decided effect; `fig:null_bounds` draws it as the
accuracy panel's established contrast, for scale, and the supplement's
equivalence subsection now records that no arm in the ledger has both a near
null and the pairs to resolve it. `formal/AILM/Instances.lean`'s `LX1` still
proves that the recorded sweep's argmax is interior, on the corrected ledger;
the manuscript reads that theorem as the ordering it states and not as an
optimum.

**Guards.** `tests/test_lexical_channel.py` now carries assertions that fail on
the defective lookup: `test_the_table_keys_are_contexts_that_end_at_their_position`,
`test_an_unseen_context_falls_back_rather_than_matching_a_different_letter`,
`test_a_short_suffix_beats_a_long_non_suffix`,
`test_depth_counts_context_letters_and_not_key_characters`,
`test_the_lookup_bound_comes_from_the_message_not_from_the_caller`, and a
real-data invariant over the whole held-out split,
`test_every_matched_context_ends_at_the_current_letter`. The shape claim is
restated as what the data show:
`test_the_sweep_does_not_fall_monotonically_and_the_argmax_is_interior` with
`test_the_interior_argmax_is_not_evidence_that_over_weighting_costs_accuracy`
beside it. The previous tests could not have caught the defect: one checked
only that some prefix matched and one asserted a tautology, which is the class
of vacuous control this project's testing doctrine forbids and its own suite
had shipped. Three fail-open edges found by the same audit were closed in the
same change: the oracle's letter-bigram bar now refuses non-lowercase tokens
instead of wrapping them through negative indices, the letter and phoneme
scorers refuse an empty population instead of returning the uniform value, and
the Stage 4 context-free baseline breaks ties by name as the Stage 2 baseline
does. The manuscript-variables generator refuses a ledger whose step-down
verdict contradicts its fixed-threshold verdict and derives the survivor
sentences from the recorded lists, so no survivor name is typed in prose.

## 111 — PRE-REGISTRATION: the ud_ewt_test confirmatory replication of the four EWT family members (2026-09-05)

**Status: registered before the run exists.** This entry is the successor
registration decision 91 left room for, and it supersedes 91's disposition of
the four `ud_ewt_test` members only. Nothing here reopens the eight
`external_lexicon` members: they stay retired by exhaustion, and the four split
diagnostics stay non-rescoring by construction. The registration commit that
lands this entry and its `data/registration_provenance.json` protocol row
strictly precedes the outcome commit that records any number, which is the
operative confirmatory rule of entry 99, machine-checked against Git history.

**The principal's ruling on the generator read.** `TODO.md` carried an open
question 91 and 97 both refused to answer on their own: whether the case-count
twin generator's read of `en_ewt-ud-test.conllu` burned the population under
79's falsifier. The principal has ruled that it does not. The read is the
sanctioned identical-generator re-vendor that 79's own population rule
prescribes, performed by `scripts/generate_case_counts_test.py`, which is the
mechanical twin of the shipped generator: identical oracle, identical exclusion
rules, identical minimum-length filter, output written outside `vendor/` and
consumed by nothing. No family member was scored against it, no parameter was
fitted on it, and no threshold or endpoint was chosen after seeing it. 79's
falsifier fires on a *scoring* read that precedes the successor registration;
it does not fire on the vendor step the successor registration itself
prescribes. The population is therefore **not burned**, and the confirmatory
replication of the four `ud_ewt_test` members runs now, once.

**The population.** The whole of `en_ewt-ud-test.conllu` at the pinned
UD_English-EWT revision already recorded in `src/ailm/corpus/ud.py`
(`UD_EWT_REVISION`, `UD_EWT_TEST_URL`, `UD_EWT_TEST_RAW_SHA256`,
`UD_EWT_TEST_RAW_SIZE_BYTES`), acquired and authenticated through the project's
own fail-closed path (`ensure_ud_ewt_test_path` / `load_ud_ewt_test`), never by
hand. It is vendored through the identical generators, oracles, exclusion rules
and minimum-length filters the development artifacts use: the case artifact's
UPOS-only observation with `deprel` and `feats[Case]` withheld and its
`MIN_SEQUENCE_LEN`; the clause artifact's sentence-final punctuation typing with
`!` and unpunctuated sentences excluded and its PUNCT-disjointness assertion;
the narrative artifact's `MIN_TYPED_SENTENCES` and document-position tercile
phase. The one deliberate difference is the one 79 registered: **no stride is
applied**. The entire re-vendored corpus is the evaluation population, because
nothing is fitted on it. Document identity is the authenticated `# newdoc id`
list in first-seen order, extracted by the same helper the development
generators use, with its ordered digest carried in each artifact's provenance
block alongside the source URL, revision, sha256 and byte extent, and
`population: ud_ewt_test`.

**The four members, their endpoints, tests, sidedness and threshold.** Exactly
the four `ud_ewt_test` members of the frozen sixteen-member
`SIGNIFICANCE_FAMILY`, each keeping the endpoint its development label records
today:

- `stage4_derived_mcnemar_p` — per-token paired comparison of the
  forward-filtering case decoder against the context-free `argmax P(case|UPOS)`
  baseline, on the derived population. McNemar's exact conditional binomial,
  two-sided.
- `stage4_gold_mcnemar_p` — the same paired comparison on the gold population
  (UD `feats[Case]`, Nom/Acc/Gen only). McNemar's exact conditional binomial,
  two-sided.
- `stage5_mcnemar_p` — per-sentence paired comparison of MAP clause-type
  classification against the always-majority baseline. McNemar's exact
  conditional binomial, two-sided.
- `stage6_sign_test_p` — per-sentence sign test on the phase-conditional minus
  unigram log-loss difference. Exact two-sided sign test.

All four are judged at the frozen `BONFERRONI_ALPHA`, which is
claim:stat_familywise_alpha familywise divided by the frozen family size and
equals claim:stat_bonferroni_alpha. The family is `SIGNIFICANCE_FAMILY` exactly
as it stands at this entry; this registration forbids editing it for this run.
Holm's step-down verdict is reported beside Bonferroni, computed over the same
frozen family with these four confirmatory p-values substituted for their four
development counterparts and the other twelve left at their recorded values.

**Parameters are read, never re-fitted.** Every model parameter comes unchanged
from the existing dev-train fits: the Stage 4 emission, transition and start
built from the development case artifact's train split, the Stage 4 context-free
baseline table built from that same train split under the same name-sorted
tie-break, the Stage 5 per-type chains and type prior and the type-free
comparator built from the development clause artifact's train split, and the
Stage 6 phase emission and unigram bar built from the development narrative
artifact's train split. No re-fit, no re-selection of weights, no stride, no
smoothing change, no hyperparameter touched. The scoring functions are imported
from `ailm.levels.case`, `ailm.levels.clause` and `ailm.levels.narrative` and
from the shipped stage recorders' own comparison code, never re-implemented.

**One run, ever.** The recorder writes the sentinel claim
`confirmatory_run_completed_ud_ewt_test` first, under the ledger's single-writer
lock, and refuses to run when it already exists, because a second pass over a
held-back set un-holds it. `scripts/precheck_confirmatory_replication.py`
refuses on the same sentinel from the other side.

**Labels and statuses.** Outcomes are recorded under `confirmatory_*` labels
with epistemic status `confirmatory` for the four verdicts and their sufficient
counts, and `descriptive` for the population census and the control diagnostics.
No development-population label is overwritten, renamed or removed: the
development results stay exactly where they are, and the confirmatory rows sit
beside them.

**The predictions.** Each is a direction and a threshold crossing, stated before
the population is scored:

- **`stage4_derived_mcnemar_p`.** Direction: the decoder-favored discordant
  count exceeds the baseline-favored count, as on the development population
  (claim:stage4_derived_mcnemar_b against claim:stage4_derived_mcnemar_c).
  Threshold: the paired comparison crosses the frozen threshold. The
  confirmatory population is the whole test file rather than one stride arm of
  dev, so its discordant total should be several times the development one and
  the crossing should be easier, not harder. This is the one member of the four
  the development run left as a corrected-threshold survivor, and it is the one
  prediction a failure would most damage.
- **`stage4_gold_mcnemar_p`.** Direction: none is predicted, because the
  development result had none: b and c were both claim:stage4_gold_mcnemar_b,
  so the model and the context-free baseline agreed on every discordant
  opportunity. Threshold: **does not cross**. The gold population's observation
  alphabet is structurally degenerate, and the registered expectation is a
  discordant total small enough that no two-sided exact test could cross the
  frozen threshold even at a perfect split. If b and c are again both zero, that
  is recorded as the outcome and read as *not shown*, never as *no effect*.
- **`stage5_mcnemar_p`.** Direction: the majority-favored count is at least the
  model-favored count, as on the development population
  (claim:stage5_mcnemar_b model-favored against claim:stage5_mcnemar_c
  majority-favored). Threshold: **does not cross**. Stage 5 is the project's
  recorded non-passing level and this registration predicts it stays non-passing
  on held-back data. A crossing in the model-favored direction would be a
  genuine surprise and would be recorded as one.
- **`stage6_sign_test_p`.** Direction: positive differences exceed negative
  ones, as on the development population (claim:stage6_sign_test_positive
  against claim:stage6_sign_test_negative). Threshold: the development p-value
  claim:stage6_sign_test_p did not cross the frozen threshold and Holm did not
  rescue it. The confirmatory population is roughly five times the development
  test arm, so at an unchanged effect the p-value should fall and the crossing
  is predicted to **occur**. This is the prediction most sensitive to sample
  size and the one whose failure would say the most about the effect's size.

**Falsifiers.**

1. If any of the four members is scored with a family, endpoint, weight,
   sidedness or threshold differing from the frozen ones, its result is
   exploratory by this entry's own rule, whatever its p-value.
2. If the run is performed more than once against this population, every result
   after the first is void, and the sentinel exists to make a second run refuse
   rather than to be worked around.
3. If any model parameter is re-fitted on the confirmatory population, or a
   stride is applied to it, the run is not a replication of the shipped system
   and this entry does not cover it.
4. If the confirmatory population's document identities intersect the
   development population's, the two are not disjoint, the "held-back" claim is
   false, and the run is void. The recorder checks this by comparing document
   identity digests and per-document identities directly, and refuses to write
   on any intersection.
5. If either control misbehaves — the planted-effect positive control failing to
   fire on this population, or the label-shuffled negative control firing — the
   instrument cannot be trusted on this population and nothing is recorded.
6. If a member's prediction fails, the failure is recorded and reported as a
   failure. No member is re-selected, re-weighted, re-thresholded or dropped
   after the data are seen, and no fifth member is added.

**Execution note, disclosed rather than omitted (2026-09-05).** Between this
registration and the recorded run, `scripts/record_confirmatory_replication.py`
was exercised end to end once against an **isolated ledger** at a temporary path
via `AILM_CLAIMS_PATH`, not against `data/claims.json`. The reason is that the
sentinel makes the population unrecoverable: burning it on an untested recorder
would have been the worse discipline, and the shakedown found and fixed a real
defect (the planted-effect control built its emission over the confirmatory
population's narrower observation alphabet, so its row indices did not line up
with the scoring frame). That shakedown computed the same numbers the recorded
run computes, so this entry's author saw the outcomes before the outcome commit
landed. Two facts bound what that could have changed. First, the text of this
entry — every endpoint, threshold, direction, prediction and falsifier above —
was committed to Git at `8bf5e5f695194224a89c096103475b91f41bd76e` before the
recorder was run at all, so no prediction here can have been written or altered
against a known result without that showing in the history. Second, the recorded
run is deterministic: no RNG outside the fixed negative-control seed, no
timestamp, no sampling, so the official run reproduces the shakedown's values
exactly rather than being a second draw. Nothing in the protocol was adjusted
after the numbers were seen, and the only code change between the two runs is
the control-alphabet fix named above.

**What this entry does not claim.** It does not promote the eight
`external_lexicon` members, which stay retired by decision 91; it does not
promote the four split diagnostics, which have no analogue on a fresh
population; and it does not turn the twelve development-population verdicts into
confirmation. At most four of the sixteen family members can carry a
confirmatory status after this run, and the manuscript's honest sentence
survives: the remaining twelve are development-population inference.

## 112 — PROSPECTIVE DEVELOPMENT-SESSION SPECIFICATION: the widened-observation clause level (case roles plus surface-ordered VERB and AUX) on the document-disjoint partition (2026-09-05)

**Written before the widened artifact is regenerated**, for the reason entries 36
and 38 were written before their measurements landed: a rule for reading an
outcome is worth nothing once the outcome is visible. This entry registers the
alphabet, the population, the receiver, the comparators, the gate, the controls,
the predictions and the falsifiers of a second Level 5 arm. It does not touch the
shipped Level 5, whose verdict is recorded in entries 24, 83 and 85 and whose
consistency guard is `scripts/audit_gates.py::audit_stage5`.

**Status, stated first.** This is a *development-session specification*, not a
confirmatory registration. It has no independently timestamped prior artifact and
its registration commit is part of the same session as its outcome, so every
label it produces is recorded `descriptive` and none may be read as confirmatory.
Entry 79 freezes the multiplicity family for the replication run; nothing here
joins that family, and each p-value this arm records is documented in
`EXCLUDED_FROM_FAMILY` with its reason.

### The alphabet

The shipped Level 5 observes the derived case-role sequence — Level 4's hidden
states, exactly one level down. The widened alphabet adds the two parts of speech
that occupy a position in surface order and carry no case: `VERB` and `AUX`. That
is what lets the level see subject-auxiliary inversion, the strongest
interrogative cue in English, which the case-only projection cannot represent at
all.

The alphabet, the extraction and the chain fit are already defined once, in
`src/ailm/levels/widened.py`, and both consumers import them rather than
transcribing them. This arm adds a third consumer and no fourth definition.

### The disjointness assertion, strengthened rather than dropped

Stage 4's lesson (entry 16) applies with more force here, not less, because the
widened stream admits token classes the case-only stream excluded. Three parts,
all asserted at artifact build time rather than assumed:

1. `PUNCT` tokens emit nothing. They remain the label source — the sentence-final
   punctuation form names the type — so a `PUNCT` token contributing any
   observation would place the label inside the observation stream.
2. A `VERB` or `AUX` token contributes its UPOS tag **only**. Never its lemma,
   never its `feats`, never its surface form. A verb's form is the single richest
   correlate of interrogative mood available in this corpus, and admitting it
   would make the widening a label leak wearing an alphabet's clothes.
3. No token outside the widened alphabet may appear. The artifact loader rejects
   any symbol outside the case roles plus the two added indices, which is the
   structural guard of entry 16 restated at the new alphabet size.

The generator asserts all three and the arm's test suite plants a lemma leak and
requires the assertion to fire.

### The population

Entry 85's authenticated document partition, inherited before exclusions. The
`# newdoc id` identities are partitioned first, on the unfiltered corpus, and
type, punctuation and minimum-length filtering happen afterwards; the artifact
retains the ordered identity alphabet, its digest, the full split indices and one
document index per retained sentence, so every clustered interval below resamples
the identities the split actually used. Sentence-index striding is retired and is
not reintroduced here.

The sentence-usability rule is stated rather than inherited silently, because it
differs from the shipped arm's and that difference is the whole of the drift
between the queue item's scratchpad numbers and its re-run numbers. A sentence is
usable when its final punctuation names one of the two modeled types **and it
contributes at least the minimum number of widened symbols**. The shipped arm
requires the same minimum of *case-bearing* tokens. The widened rule therefore
admits sentences the case-only rule excludes — a sentence of one case role and one
verb, for instance — and the two arms' populations are nested but not equal. That
is a property of the widening, not a defect, and it is why the case-only arm is
retained below as a baseline rather than treated as the same measurement.

### The receiver

The shipped block-diagonal joint agent, at the new state count. Nothing about the
receiver's shape changes: one hidden factor over the joint (type × symbol) space
in type-major order, `A` deterministic so the observed symbol is the symbol
component, `B` block-diagonal with exactly zero cross-type mass, `D` the type
prior times each type's start distribution, `C` flat. Only the alphabet size
moves, and it moves in the builders' parameters rather than in a second copy of
them: the shipped builders take the alphabet as an argument whose default is the
shipped value, so the case-only path is bit-identical to what ships today and
`tests/test_stage5_gate.py` is unchanged.

### The comparators

Four, and the first three are the queue item's, all measured on the **identical**
widened stream rather than carried over from the case-only one:

- **(a) The type-free pooled chain fitted on the widened stream.** Entry 2 applied
  to the new stream: the honest bar is the strongest model that does not use
  sentence type, refitted on the data actually being scored. A bar carried over
  from a different alphabet would credit the widening with the refit.
- **(b) The sentence-length-only classifier on the widened stream.** Entry 55's
  control. Interrogatives are shorter in this corpus, so a length rule is a live
  competitor and not a strawman.
- **(c) The single first-symbol-is-`AUX` indicator.** Entry 54's control against a
  margin that is really one free parameter. If a one-bit feature matches the
  sequence model, the sequence structure is decorative and the finding is about
  auxiliary fronting, not about a clause-level generative model.
- **(d) The shipped case-only arm**, retained as the does-not-use-the-widening
  baseline. It answers the only question the other three cannot: whether the two
  added symbols bought anything at all.

### The gate

The mixture must beat the type-free pooled chain **fitted on the widened stream**
on held-out positions, judged by the same exact paired test the shipped Stage 5
uses, on the same paired sufficient counts. `audit_gates.py::audit_stage5_widened`
records the placement verdict under the typed-comparator taxonomy, separately from
and without altering `audit_stage5`.

The declared gate is registered here, before the artifact exists, precisely so it
cannot be re-selected afterwards. If the mixture does not beat the widened bar,
that is the result.

### The controls

Every control the shipped arm carries, at the new alphabet size, each with the
authority to refuse the write:

- **Spec-build bit identity.** The GNN spec's declared dimensions must equal the
  live agent's, checked against a running pymdp agent rather than a hand-typed
  tuple.
- **Decoder equivalence.** The closed-form mixture evidence must equal oracle
  `forward_filter` on the joint block-diagonal chain to numerical tolerance.
- **Swapped-blocks and cross-type-leak negative controls.** The mutation specs
  must fail conformance exactly as the shipped Level 5 mutations do. A mutation
  layer that cannot reject is a comment with a test-shaped wrapper.
- **The detectability control.** The identical harness on a synthetic corpus whose
  two types genuinely differ must classify decisively above its measured majority
  baseline and its mixture must beat its type-free chain. Without it a non-passing
  English result is indistinguishable from a dead instrument — the
  failure-to-reject trap.

The recorder writes nothing if any of these fails.

### The predictions

Transcribed from `scripts/probe_l5_widened_alphabet.py` and its recorded re-run in
`docs/research_queue.md`, as directions and relations rather than as typed
quantities. The probe writes nothing to the ledger; these are the numbers this arm
must reproduce, and the registered interval is the probe's own.

1. The type-oracle gain is **positive** and falls **inside the interval the probe
   registered** for it. A point outside that interval means the pre-computation
   was contaminated and the proposal is withdrawn rather than re-fitted.
2. Classification AUC is **above both** comparator (b), length-only, and
   comparator (c), first-symbol-is-`AUX`. Above one and not the other is a
   failure, not a partial success.
3. The mixture **beats** the widened type-free bar on held-out positions.
4. The realized capture of the type entropy is **substantially larger** on the
   widened stream than the shipped case-only arm's, which is the effect the whole
   arm exists to size.

### The falsifiers

- If the type-oracle gain falls outside the probe's registered interval, the
  pre-computation was contaminated. Withdrawn.
- If the mixture does not beat the widened type-free bar, **the widening is
  decorative and the null stands on the new stream too.** This is a recorded
  outcome, not a reason to change the bar.
- If the sequence model fails to beat comparator (b) or comparator (c), the signal
  is length or a single fronting indicator, and the clause-level mechanism is not
  what carries it.
- If any negative control passes, or the detectability control fails to fire, no
  number from this arm may be recorded at all.

No re-selection after the data are seen. The alphabet, the split, the smoothing
constant, the receiver order and the four comparators are fixed by this entry.

### The stratificational caveat, which this arm does not resolve

This arm produces the number that makes a choice informed; it does not make the
choice. A stream containing `VERB` and `AUX` symbols is no longer "Level 4's
hidden states read one level down" unless Level 4's own state space is widened to
match — which would move every Stage 4 number and contradict the stratificational
reading of entry 52. The alternative, letting Level 5 observe Level 4's states
*plus* material Level 4 does not represent, breaks the one-level-down claim
outright. Both branches are live after this measurement and neither is selected
by it.

Two further limits, stated beside the claim rather than apart from it. The arm
cannot explain the *mechanism*: it can show that the positions occupied by verbs
and auxiliaries carry the type signal, not why. Auxiliary fronting is the obvious
account and is untested here — comparator (c) bounds how much of the effect one
fronting indicator alone reproduces, which is not the same as testing the
mechanism. And the widening does not raise the cap: the entropy of the binary type
variable still bounds any message about it, so the widening moves realized capture
within that cap rather than enlarging it.

## 113 — RULE: a reply field that describes the server's own cache is not part of the answer (2026-09-06)

**What was decided.** `prompt_eval_cached_count` joins `_VOLATILE_OLLAMA_REPLY_FIELDS`
in `src/ailm/call_cache.py`, the reviewed list of reply fields that describe *when
and how* the server answered rather than *what it answered*. The list stays an
enumeration; a suffix rule that admitted every future `*_count` field was
considered and rejected, and a test pins the enumeration so that a new field is
semantic until someone shows it cannot vary with the answer. The call-store
module's own docstring asks that additions be reviewed rather than
pattern-matched, which is why this entry exists: the commit (`4450711`) records
the change, and this records the review.

**Why.** Ollama reports how many prompt tokens it served from its own KV cache
instead of re-evaluating. The field is zero the first time a prompt is seen and
positive while the server still holds that prefix, so two identical requests
issued moments apart differ in it as a matter of course. The store treated every
unrecognized reply field as semantic, so the second observation of a repeated
question was refused as a divergent reply. Two arms ask the same question twice
on purpose: the stipulated synthetic probes score a one-symbol window, so one
prompt recurs across positions (720 scheduled letter calls resolve to 464
distinct requests), and the reliability trials repeat one prompt deliberately.
Both are acquired live, so both hit the refusal on the first repeat whose prefix
was still warm. The synthetic recorder aborted this way on three separate
canonical runs, at a different key each time, with the server otherwise idle.
Isolating it was direct: the stored payload issued twice against the live server
produced replies whose only differing field was this one (0 against 64).

**What stays semantic.** The response text, the logprobs, the context, and the
token counts the measurements read (`prompt_eval_count`, `eval_count`) are all
still compared, and a genuinely different answer is still refused. Nothing in
this project reads the excluded field.

**What this changed in the record.** Nothing numerical. The full network re-bind
that followed (external, drift, synthetic, in-context; the run-scoped receipts
introduced in commit `3d6abcf`) left the ledger byte-identical: 2319 labels
before and after, no value or run id changed. The change is to what the store
refuses, not to what any recorder measures.

**Falsifier.** A field on this list that can vary with the answer. If a reviewer
shows that `prompt_eval_cached_count` can differ between two replies whose text
or logprobs differ while every other field agrees, the field returns to the
semantic set and every receipt that relied on the exclusion is stale.

## 114 — The widened-observation clause model is the canonical Level 5; the case-only model is the retained ablation (2026-09-06)

**What was decided.** The Level 5 the audit, the cover, the README, the hierarchy
guide and the manuscript call "the clause level" is now the widened-observation
clause model registered in entry 112: observation is the case role plus one
distinguished symbol each for `VERB` and `AUX`, on the document-disjoint
partition. The case-only clause model of entries 24, 83 and 85 stays in the
repository as a **retained ablation** with its record intact. Its gate is still
non-passing at `ABOVE_REFERENCE`, its privileged-label gain is still unresolved
on both document-resampled scales, and its non-passing outcome still replicated
on the held-back `ud_ewt_test` split with zero discordant pairs (entry 111).

Nothing is deleted, re-scored, re-fitted or re-thresholded by this entry. No
recorder runs differently, no ledger label changes name, and no number moves.
What changes is which of the two arms the project calls Level 5.

**Why now, and why this is not a rescue.** The two arms were measured under
their own registrations and both outcomes are on the record. Entry 112 declared
its gate before its artifact existed and its falsifier said in as many words
that a mixture failing the widened bar would mean the widening was decorative.
The mixture beat the bar, every registered prediction held, and the paired
privileged-label gain resolved away from zero at the document level. That is a
gate that could have failed and did not, which is the same standard Levels 1
through 4 are held to.

Promotion is therefore an editorial choice about which arm the paper describes
as its clause level, made after both measurements, and stated here rather than
performed silently. A reader who disagrees with the choice can read the ablation
row in the gate-audit table and reach the opposite conclusion from the same
numbers, which is the property that makes the choice reviewable.

**The two limits that ride with the canonical arm, stated wherever its verdict
is stated.** First, it is a development-session specification: registration and
outcome share one session (entry 112), so every label it produces is recorded
`descriptive` and none may be read as confirmatory. Second, it has no held-back
replication and cannot acquire one, because the `ud_ewt_test` population is
spent. Entry 111's `confirmatory_run_completed_ud_ewt_test` sentinel exists and
refuses a second pass from both sides, and a second pass over a held-back set
un-holds it. Neither limit is a defect of the measurement. Both are limits on
what may be claimed from it, and the canonical Level 5 is therefore a passing
development-session gate and not a replicated one.

**What the ablation keeps.** Its verdict row stays in the gate audit under the
explicit id `stage5_clause_case_only`, still exported and still carrying its own
consistency guard, so a run in which the case-only arm silently starts passing
or its gain resolves is a flagged change and not a quiet relabeling. Its
confirmatory replication row stays in the manuscript's confirmatory table. The
sentence recording that the replication found zero discordant pairs favoring the
model stays where it is. `stage5_mcnemar_p` remains the Stage 5 member of the
frozen sixteen-member significance family of entry 79, because that member **is**
the case-only arm, and the freeze is not reopened by a naming change.

**Labels.** `stage5_*` continues to name the case-only ablation's ledger labels
and `stage5w_*` the canonical arm's. Renaming them would rewrite the provenance
of every receipt that already cites them, for a cosmetic gain. The manuscript
tokens follow the labels, so a token whose name says `STAGE5W_` is the canonical
Level 5's number and a token whose name says `STAGE5_` is the ablation's. The
one derived quantity that had to move is the count of passing levels: its
producer in `src/ailm/manuscript_variables.py` now counts the canonical Level 5
as passing and asserts, fail-closed, that the widened arm's recorded numbers
still support that count. The abstract's sentence around that token was
rewritten, because the count no longer means what the old sentence said it
meant, but the count itself is still injected and is nowhere typed.

**The stratificational caveat is not settled by this entry.** Entry 112's
closing section stands unchanged: a stream containing `VERB` and `AUX` symbols
is not "Level 4's hidden states read one level down" unless Level 4's state
space is widened to match, and letting Level 5 observe material Level 4 does not
represent breaks the one-level-down claim outright. Promotion picks which arm is
described, not which branch of that question is taken. The manuscript states the
caveat beside the canonical arm rather than only in the discussion of the
ablation.

**Falsifier.** If the canonical arm's mixture stops beating the widened
type-free bar, if either document-resampled interval on its privileged-label
gain comes to span zero, or if either of its negative controls stops rejecting,
the promotion is withdrawn and Level 5 returns to the case-only description. The
audit's consistency check on the canonical row is the guard: it demands a
bracketed placement and a passing outcome, and a run that loses either exits
non-zero rather than printing a changed status.

## 115 — PRE-REGISTRATION: the document-clustered interval of the Level 6 improvement on the held-back ud_ewt_test population (2026-09-06)

**What is being registered, before it is computed.** The paper's primary
inferential procedure for Level 6 is the whole-document bootstrap of the mean
per-sentence log-loss improvement, phase model minus type-unigram bar, in nats.
On the development population that interval spans zero, and that is what makes
Stage 6's finding status a positive point estimate with a non-confirmatory
primary analysis (entry 26). Entry 111 replicated Stage 6's *sign test* on the
held-back `ud_ewt_test` split and it crossed the frozen threshold in the
registered direction. The primary estimand was never computed on that
population. This entry registers it, states the prediction before running it,
and fixes every free choice in advance so that none of them can be made after
seeing the answer.

**Population.** The claim:confirmatory_stage6_documents eligible held-back
documents in `data/confirmatory/ud_ewt_test_narrative_counts.json`, the same
population entry 111 scored and the same claim:confirmatory_stage6_sentences
typed sentences. No document is added, dropped or
re-terciled.

**Estimand.** The observation-weighted mean per-sentence improvement in nats,
`sum(d) / n`, identical in definition to the development `stage6_logloss` arm in
`scripts/record_cluster_claims.py`. Not the sign, not a per-document mean of
per-document means: the same functional of the same difference vector, so the
two populations' numbers are comparable without a conversion.

**Interval.** The whole-document percentile bootstrap of
`ailm.clustering.summarize_clustered_difference`, clustered on document id, at
`DEFAULT_RESAMPLES` resamples and `DEFAULT_SEED`, which are the development
arm's own constants and are recorded beside the bounds.

**Coverage, and a correction to the brief that commissioned this entry.** The
lane brief asked for a ninety-nine percent interval, on the stated
understanding that this is what the development primary uses. It is not. The
development Stage 6 clustered arm is computed at
`ailm.statistics.DIFFERENCE_CONFIDENCE`, the level entry 37 harmonized the
paired-difference and discordant-share families onto and the level this arm
records as claim:confirmatory_stage6_clustered_confidence; the only other
coverage constant in the module is `CORRECTED_CONFIDENCE`, recorded here as
claim:confirmatory_stage6_clustered_corrected_confidence. Neither is the level
the brief named, and no interval at that level exists anywhere in this project. The primary is therefore registered at `DIFFERENCE_CONFIDENCE`, and the
reason is not deference to precedent. The finding-status rule this arm feeds
compares "the development interval spans zero" against "the held-back interval
excludes zero". Two intervals at two coverages are two different quantities, and
a rule that crossed them would license a status change out of a coverage
difference rather than out of the data. `CORRECTED_CONFIDENCE` bounds are
recorded alongside as a declared sensitivity so the conservative reading is
available, and they are explicitly not the primary.

**Prediction, stated before the computation.** The interval will EXCLUDE zero,
with a lower bound close to it. The basis is arithmetic, not hope. The
development arm resamples claim:cluster_stage6_logloss_clusters documents and
records a clustered effect of claim:cluster_stage6_logloss_effect nats between
bounds claim:cluster_stage6_logloss_ci_low and
claim:cluster_stage6_logloss_ci_high. The held-back population carries
claim:confirmatory_stage6_documents documents, and entry 111 already recorded
its mean improvement as claim:confirmatory_stage6_improvement_nats nats, which
is the larger of the two effects. If the per-document dispersion is comparable
across the two populations, the half-width shrinks by about the square root of
the ratio of their document counts, a little under half. Applied to the
development half-width that gives a predicted half-width smaller than the
held-back effect it would be centered on, so the interval clears zero — and by
less than one half-width, which is why the prediction says the lower bound will
be close to it rather than comfortably clear.

**What would make that prediction wrong, and why it is not a safe bet.** The
development population is itself the demonstration that the sign and the mean
can disagree: there the clustered *sign* arm excluded zero comfortably while the
clustered *log-loss* arm spanned it. So entry 111's overwhelming held-back sign
test is not evidence for this interval, and it has deliberately not been treated
as any. The dispersion assumption is the real exposure. The development arm's
largest single document holds claim:cluster_stage6_logloss_largest_cluster of
its claim:cluster_stage6_logloss_observations scored sentences, better than a
tenth of the whole arm in one cluster, and if the held-back population is
similarly dominated by a few long documents, or more heterogeneous between
documents, the half-width will not shrink by the square-root factor and a center
at the held-back effect will not clear it. A confidence a little better than
even is the honest reading of this prediction, not a near-certainty.

**Status: descriptive, and the registration does not change that.** The
registration commit precedes the outcome commit, which is the usual condition
for a confirmatory tag, and this entry declines it anyway. Entry 111 spent this
population when it scored it, and the sentinel
`confirmatory_run_completed_ud_ewt_test` records that. An estimand registered
after a population has been opened is registered with knowledge of that
population's other results — here, that its sign test came out strongly positive
— and that knowledge is exactly what preregistration exists to exclude. The
downward default of `scripts/assign_claim_status.py` applies, the recorder writes
`descriptive`, and a test pins it there. What the registration does buy is real
but narrower: it fixes the estimand, the coverage, the seed and the prediction
before the number exists, so the number cannot be reverse-engineered into
whichever analysis flattered it.

**Why this recorder may re-run when entry 111's may not.** The one-run
discipline protects a population from being *scored* twice with parameters
chosen after the first look. That has already happened once, under entry 111,
and cannot happen again here: the models are the frozen development fits,
asserted equal array for array, and every analysis choice is fixed above.
Re-running a deterministic function of frozen inputs adds no information and
leaks none, so `scripts/record_confirmatory_stage6_clustered_claims.py` is
idempotent, is named into the orchestrator's `record_*_claims.py` glob, and runs
in the `core` phase like any other recorder.

**Controls, registered with the estimand.** A positive control centers the
held-back differences to exactly zero mean and plants the development
population's own recorded clustered effect on top; it must be detected. That is
the question that decides whether a null here would mean anything, because a
population that cannot resolve a development-sized effect produces an
instrument-ambiguous null rather than evidence of absence. A negative control
reverses the sign of a random subset of whole documents under a fixed recorded
seed, preserving every document's size and internal structure while destroying
the aggregate direction; it must not be detected. Both are deterministic given
their seeds, so a failure of either is a reproducible fact about the instrument
and not a flaky draw, and the recorder refuses to write anything on either.

**What the outcome changes.** If the interval excludes zero, Stage 6's finding
status becomes `positive_point_replicated` and the cover badge for Level 6 may
honestly read `passing`, on the conjunction of the held-back sign test crossing
the frozen threshold in the recorded direction and this interval excluding zero.
If it spans zero, the status stays `positive_point_nonconfirmatory` and the
badge stays `non-confirmatory`. The status is computed from the recorded
booleans in `scripts/audit_gates.py`, not typed, so neither outcome can be
asserted past the ledger. Either way the sentence that the development-population
primary interval spans zero stays in the manuscript, because it stays true.

**Falsifier.** If a later run of the recorder reproduces different bounds from
the same artifacts and the same seed, the arm is not deterministic and the
recorded interval is not a measurement. If either control changes verdict, the
instrument is not the one described here. If the two coverages are ever computed
from different resampling runs, the sensitivity stops being a sensitivity of
this result, and `tests/test_confirmatory_stage6_clustered.py` fails on the
containment check that catches it.

### Amendment to 115, before any ledger write (2026-09-06)

**What happened.** The recorder was run once under the specification above. It
refused, and wrote nothing: the positive control did not detect the planted
effect. Planting the development population's clustered effect,
claim:cluster_stage6_logloss_effect nats, on the centered held-back differences
produced the interval claim:confirmatory_stage6_clustered_control_positive_ci_low
to claim:confirmatory_stage6_clustered_control_positive_ci_high, which spans
zero. The ledger is unmodified and carries no
`confirmatory_stage6_clustered_*` label. This amendment is written before any
result of this arm exists on the record, which is the same footing as the
control amendment recorded for the family-disjoint G2P arm under entry 102.

**What I knew when writing this amendment, stated plainly.** The refusal printed
the control's interval, and that interval's half-width, together with the mean
improvement of claim:confirmatory_stage6_improvement_nats nats that entry 111
already recorded, implies the primary interval will exclude zero. So this amendment is written by someone who
can already infer the primary outcome. That is a real cost and it is why the
arm's status is not revisited here: it was already `descriptive` under the
original entry, for the independent reason that entry 111 spent the population,
and it stays `descriptive`. No amendment below moves this arm toward a stronger
epistemic tag, and none of them changes the estimand, the coverage, the seed, or
the resample count. What changes is the specification of the controls and of the
condition under which the recorder refuses.

**Why the original positive control was mis-specified, and how that was visible
before running it.** The control planted the development population's own
clustered effect, claim:cluster_stage6_logloss_effect nats, and treated
non-detection as instrument failure. The registered prediction above already
estimated this arm's half-width as larger than that shift. A control that plants an effect below the arm's own resolution
and then reads non-detection as a broken instrument is asking whether the arm
can see something smaller than it can see. The arithmetic that shows this is in
the registration itself; the error is mine and it is a design error, not a
result.

The deeper error is in what the control was made to gate. A positive control
establishes that a *null* is interpretable: that the instrument could have found
an effect had one been there. It says nothing about a *detection*. Refusing to
record a detection because a smaller hypothetical effect would not have been
resolved is incoherent — a detection is its own evidence that the instrument
detects. The original refusal rule applied the control to both cases alike.

**Amended control specification.**

*Minimum detectable effect.* The recorder computes the smallest constant
per-sentence shift which, planted on the centered held-back differences, yields
a clustered interval at `DIFFERENCE_CONFIDENCE` that excludes zero. It is found
by bisection on a fixed bracket to a fixed tolerance, and the bracket, the
tolerance and the resulting value are all recorded. This replaces a pass/fail at
one arbitrary planted size with the quantity that pass/fail was a proxy for, and
it is the number the limitations discussion needs.

*Positive control, two-sided, gating.* Planting the minimum detectable effect
plus one tolerance must be detected, and planting it minus one tolerance must
not be. This pins the recorded value as a measured boundary rather than an
asserted one. It cannot pass on a dead instrument, which would fail the upper
probe, nor on a hair-trigger one, which would fail the lower probe.

*Negative control, gating, unchanged.* Reversing the sign of a random subset of
whole documents under a fixed recorded seed must not be detected.

*Development-sized-effect probe, recorded, not gating.* Whether an effect the
size of the development population's clustered effect is resolved, recorded as a
boolean beside the minimum detectable effect. This is the original positive
control, kept as a measurement because it is the fact that decides whether a
null on this arm would be interpretable. It no longer refuses.

**Amended refusal rule.** The recorder refuses, and writes nothing, if the
negative control fires or if either side of the two-sided positive control
fails. It does not refuse when the development-sized probe is unresolved.
Instead, if the primary interval spans zero *and* that probe is unresolved, the
null is under-powered rather than evidence of absence, and the manuscript must
say so in those words rather than reporting a bare non-confirmatory result.

**What this does not change.** The population, the estimand, the bootstrap
procedure, the resamples, the seed, the two coverages, the descriptive status,
the prediction recorded above, and the finding-status rule that reads
`excludes_zero`. The prediction stands as written and is judged as written.

**Falsifier for the amendment itself.** If the recorded minimum detectable
effect does not reproduce from the same artifacts and seed, or if a planted
shift above it goes undetected, or below it is detected, the bisection is not
measuring a boundary and `tests/test_confirmatory_stage6_clustered.py` fails on
the two-sided probe that checks exactly this.

## 116 — RULE: a verdict word on a figure is read from a recorded boolean; the confirmatory replication gets its own figure (2026-09-07)

**The rule.** Any word on any figure in this repository that states an outcome —
PASS, FAIL, NONCONF., "excludes 1/2", "rejects", "scored", "replicated" — is
read from a boolean recorded in `data/claims.json`, or from a set the recorder
wrote, and never from the geometry drawn beside it and never from a literal in
the figure module. Where a figure draws both, the two must agree or the figure
refuses to draw. The reason is that geometry and verdict answer different
questions: an interval drawn at the marginal level can straddle a null while the
family-corrected decision excludes it, and a reader who reads the verdict off
the picture gets the uncorrected answer. A literal is worse, because it goes on
being drawn after the measurement it names has moved.

**What this entry governs, and what it does not.** It governs the figure layer
only. It does not license a figure to compute a verdict the ledger does not
hold: where no boolean is recorded, the figure states the absence rather than
inferring one. Two absences are stated on the canvas today for exactly this
reason — that an arm has no recorded cluster-robust re-analysis, and that a
measure has no paired per-item test attached to it.

**The confirmatory figure.** The `ud_ewt_test` replication registered by entry
111 and the Level 6 clustered estimand registered by entry 115 are recorded in
full and drawn nowhere. This entry commissions `fig:confirmatory_replication`,
and states what it asserts before it is generated: the four scored family
members, each with its development-population result beside its held-back
result, its discordant counts, its exact p on a log axis against the frozen
per-test threshold and its own Holm threshold, and a verdict badge per member.

**The labels that feed it.** Held-back arm, all `confirmatory_`-prefixed:
`_members_scored`, `_population`, `_frozen_alpha`, `_frozen_family_size`,
`_census_documents`, `_census_sentences`, `_run_completed_ud_ewt_test`, and per
member `_mcnemar_p` or `_sign_test_p`, `_mcnemar_discordant_n`, `_favored`,
`_opposed`, `_holm_threshold`, `_holm_rejected`, `_crosses_frozen_threshold`,
`_direction_agrees`, `_computable`, plus the per-member population counts.
Development arm: `stage4_derived_mcnemar_*`, `stage4_gold_mcnemar_*`,
`stage5_mcnemar_*`, `stage6_sign_test_*`. Controls:
`confirmatory_control_positive_*` and `confirmatory_control_negative_*`. The
Level 6 clustered pair from entry 115 is
`confirmatory_stage6_clustered_ci_low` / `_ci_high` against the development
`cluster_stage6_logloss_ci_low` / `_ci_high`.

**Three things that would make the figure misleading, and how it handles each.**
The two populations disagree on Level 6, so drawing the held-back interval alone
would hide that; both are drawn, and the development interval is not retracted.
The corrected-coverage interval spans zero while the primary does not, so an
unlabeled interval would overstate the level of the claim; each interval is
drawn with its coverage named. The held-back arm's minimum detectable effect is
coarser than the development effect it sits beside, so the figure states that
resolution rather than leaving a reader to infer that the two are comparable.

**Falsifier.** The figure refuses to draw if a member's drawn geometry
contradicts its recorded boolean — a marker left of a threshold whose crossing
flag is false, or right of one whose flag is true — if a discordant count
exceeds its population, if the count of members drawn disagrees with
`confirmatory_members_scored`, or if the positive control's p does not clear the
frozen threshold while the negative control's does. Those are the conditions
under which the picture would be telling a story the ledger no longer holds.

**Also recorded here, because they are the same rule applied to figures that
already existed.** `fig:multiplicity_forest`, `fig:gate_taxonomy`,
`fig:null_bounds`, `fig:perplexity_ladder` and `fig:channel_comparison` gained
the population and unit behind each drawn quantity, the level and clustering
unit of each interval, the exact test and p where a verdict is drawn, and
labeled threshold lines. No result changed. The set-aside allowlist entries for
the `cluster_stage2` and `cluster_stage4` families were removed in the same
work: those families are now reported on figures rather than measured and set
aside, which is precisely the condition
`test_set_aside_allowlist_entries_still_describe_dark_families` exists to force.

## 117 — RULE: a family exclusion is justified by a checked invariant, not by prose; the factorial's diagonal cells are one hypothesis under a second selection grid (2026-09-07)

**What was wrong.** The channel factorial's diagonal cells sit outside the frozen
significance family by recorded exclusion, and until now the recorded reason was
an identity. The caption of `tbl:channel_factorial`, the comment above
`SIGNIFICANCE_COUNT_LABELS` in `src/ailm/statistics.py`, and the reason strings in
`EXCLUDED_FROM_FAMILY` in `tests/test_statistics.py` all said the diagonal cells
were numerically identical to family members already declared, to every recorded
digit.

Decision 110 re-recorded the lexical arm, and half of that sentence stopped being
true.

The letter-at-evidence half still holds exactly:
`claim:channel_letter_evidence_best_alpha` equals `claim:coupling_best_alpha`,
`claim:channel_letter_evidence_paired_b` equals `claim:coupling_mcnemar_b`,
`claim:channel_letter_evidence_paired_c` equals `claim:coupling_mcnemar_c`, and
`claim:channel_letter_evidence_paired_p` equals `claim:coupling_mcnemar_p`. That
cell's argmax falls on a weight both grids carry, so the identity is stable under
either search.

The lexicon-at-predictive half does not hold. The factorial searches its own
weight grid and lands at `claim:channel_lexicon_predictive_best_alpha`, with
counts `claim:channel_lexicon_predictive_paired_b` and
`claim:channel_lexicon_predictive_paired_c` at
`claim:channel_lexicon_predictive_paired_p`. The registered sweep's grid stops
below that weight and selects `claim:lexical_best_alpha`, with counts
`claim:lexical_best_mcnemar_b` and `claim:lexical_best_mcnemar_c` at
`claim:lexical_best_mcnemar_p`. Under the defective backoff the lexical sweep
declined at high weight, so both searches happened to land together; the
corrected sweep rises to a plateau, and the two grids now select apart.

**How it survived.** No consumer evaluates the reason strings. Every consumer
reads `EXCLUDED_FROM_FAMILY` as a set of labels, so the exclusion stayed correct
while the sentence justifying it stopped being true, and no gate noticed. That is
the general defect this entry rules on, and it is the same shape as decision 99's:
a statement nothing executes decays silently.

**What was rejected, and why.** Re-recording the factorial cell over the sweep's
grid so the caption becomes true again was rejected: a measurement changed to fit
a sentence is precisely what this project's method forbids, and the factorial's
grid is the one entry 46 registered for it. Admitting the cell as a new family
member was also rejected: that moves the frozen per-test threshold decision 111
registered the held-back replication against, and it would tighten every other
threshold on the strength of a weight choice.

**The basis, restated.** The factorial's lexicon-at-predictive cell and the
sweep's registered selection are one hypothesis, not a pair of them. The
hypothesis is the lexical channel at the predictive site. Both are scored on the
same evaluated positions, `claim:channel_factorial_positions` against
`claim:lexical_eval_positions`, and against the same uncoupled anchor,
`claim:channel_factorial_anchor_accuracy` against `claim:lexical_anchor_accuracy`.
What differs is the weight grid each searched, and a weight is a selection rule
rather than a question. The family counts hypotheses, not weight choices, and its
member is the sweep's registered selection. Counting the factorial cell as well
would count one hypothesis twice. This is the basis the confirmatory rescorings
were excluded on: same hypothesis, new data.

**The rule.** An exclusion from the significance family is justified by an
invariant a test checks against the ledger, never by a sentence in a comment or a
caption. `EXCLUDED_FROM_FAMILY` carries a structured reason per excluded label
naming the invariants it rests on, and `tests/test_statistics.py` asserts them
against `data/claims.json`: equal evaluated positions, equal anchor accuracy, and
the same channel and insertion site. Where an exact identity does hold, as it
does for letter-at-evidence, the identity is asserted as well rather than
described.

**Falsifier.** If a factorial cell and its family counterpart ever score a
different count of evaluated positions, or a different anchor accuracy, or stop
naming the same channel and insertion site, they are not one hypothesis: the
exclusion is void, the family must be re-declared with the cell inside it, and
every corrected threshold recomputed. Letter-at-evidence carries a second
falsifier. If its recorded counts, weight and p-value stop matching their
counterparts exactly, the identity fails even where the shared-population
invariants still hold, and its reason must be narrowed to the shared-population
basis rather than left standing.

**No measurement changed.** No recorder ran for this entry, and no ledger value
moved. What changed is the recorded justification and the test that now checks it.

---

## 118 — The parameter census follows canonical Level 5 and distinguishes simplex capacity from fitted parameters (2026-09-07)

**Scope.** Decision 114 promoted the already-built widened-observation clause
model while preserving the previous parameter measurement. This entry adopts
the canonical model for the main parameter table and retains a separate census
of the case-only ablation. It changes descriptive array accounting only; no
evaluation population is rescored and no gate or inferential result changes.

**Measurement.** `scripts/record_parameter_claims.py` builds the complete
canonical inventory and the retained case-only agent from their training
artifacts through their shipped builders. Each arm must conform to its own GNN
declaration before any claim is written. Canonical Level 5 now has
claim:params_l5_stored stored entries, claim:params_l5_nonzero nonzero entries
and claim:params_l5_free simplex degrees of freedom. The separate ablation has
claim:params_l5_case_only_stored, claim:params_l5_case_only_nonzero and
claim:params_l5_case_only_free respectively, excluded from the canonical
totals. The `params_l5_*` headline stem now names canonical Level 5;
`params_l5_case_only_*` preserves the ablation's explicit scope. The historical
`stage5_*` measurement labels remain with the case-only arm.

**Interpretation.** The existing per-array rule counts one fewer degree of
freedom than the number of live entries in each distribution. It measures
capacity under support and normalization constraints, not the number of
parameters actually fitted by this pipeline. In particular, the fixed uniform
Level 1 start prior contributes capacity although the corpus does not fit it.
The rule does not infer constraints shared across arrays. Methods and source
documentation now state these limits and withdraw the earlier interpretation
that the total states how much the corpus decides. The counting rule itself
does not change.

**Falsifiers.** The retained case-only agent must fail against the canonical
widened declaration and conversely. The widened free count must equal the
independent closed form for its within-type transitions and full-support joint
prior; the deterministic emission contributes none. The existing sum,
orientation, support and invalid-probability controls remain in force. The
recorder checks both arms before recording either one.


## 119 — Make the in-context comparison visible and gate reference-rule crossings (2026-09-07)

**Decision.** Keep the full vertical scale of `fig:incontext_sweep`, including
the AILM comparator and uniform reference levels, and add a labeled inset in
each probe panel. Every inset redraws the same recorded treatment/control
points and Wilson intervals, uses the parent's x-limits, and shares a vertical
range derived from the union of the drawn intervals. A rectangle marks that
range on each parent. The figure title still comes from the validated family
outcomes; apparent interval overlap alone does not determine the paired-test
verdict. No measurement, population, or statistical decision changes.

**Build guarantee.** `save_figure` now also refuses a horizontal or vertical
reference rule through a rendered label. The detector recognizes Matplotlib's
standard x-axis, y-axis, and frame transforms and examines inset children. It
respects visible clipping and fully opaque text/legend backgrounds only when
they draw above the rule. It tolerates a small glyph-edge graze. Data curves,
gridlines, figure-owned text, and custom composed transforms remain outside
this bounded check; it does not replace visual review of a rendered figure.

**Why transform identity matters.** Inferring axes semantics from transformed
unit coordinates both missed vertical rules and misclassified a flat data
series whose limits were zero to one. Real fixtures reproduced both defects.
The detector now checks the declared standard transforms rather than inferring
a rule from a curve's numerical values. An independently constructed offset
transform remains an explicitly tested boundary of the detector.

**Falsifiers.** Deliberate horizontal/vertical and partial reference rules
through text must fail, including in inverted views. A clear label, an opaque
shield above the rule, and clipped-away text must pass; a transparent shield
or a rule drawn above its shield must fail. The ordinary data and gridline
controls remain accepted. Inset tests require both arms, both parent reference
levels, identical x-limits, complete interval coverage, and an outlined zoom
region. The calibration figure's translucent legend frames and unbacked
callout reproduced crossings; opaque backgrounds resolve those crossings
without changing its values, limits, arrows, or threshold locations.


## 120 — Correct the stipulated phone context interpretation without changing its measurement (2026-09-07)

**Correction to decision 108.** Its historical outcome paragraph says that
shortening the prefix lowers every probe's accuracy and identifies within-query
chain evidence as the source of the full-prefix answers. The recorded
contrasts do not support either statement. Gemma and SmolLM have lower
short-prefix point estimates, while Qwen's short-prefix estimate is higher:
claim:bench_synthphone_phone_top1_qwen25 at the full prefix versus
claim:bench_synthphone_short_top1_qwen25 at the short prefix. The paired context
p-values are claim:bench_synthphone_context_paired_p_gemma3,
claim:bench_synthphone_context_paired_p_qwen25, and
claim:bench_synthphone_context_paired_p_smollm; none clears the declared
claim:bench_synthphone_context_family_alpha threshold. This does not establish
an absence of context effects; it does not establish the asserted advantage
or causal mechanism either.

**What stands.** The comparator-versus-probe contrasts are a separate family
and retain their recorded interpretation. The arm's design was specified
prospectively within a development session, without an independently
timestamped prior registration. The supplement now uses that provenance
wording and reports the context p-values beside its qualified interpretation.
The historical decision 108 text remains above, superseded on these points.
No recorder, population, protocol, threshold, or ledger value is changed by
this correction.

**Guard.** The documentation tests compare each stated context direction and
threshold reading with the ledger. Planted direction reversals and newly
threshold-clearing contrasts must fail, so a future recording cannot silently
leave the fixed interpretation standing. The source manuscript is checked
alongside the ledger; this is a bounded consistency check, not an inference
that unmeasured prompt interventions have no effect.

---



## 121 — PRE-REGISTRATION: the case-word influence sweep — the L4→L3 bridge's causal-influence estimand, controls, and refusals (2026-09-14)



**Status: registration recorded BEFORE any run.** Protocol P3's partition 3

([`docs/extension_protocols.md`](extension_protocols.md) §4.5) sequences the

influence sweep behind partitions 1–2 and behind this registration. No

recorder has run for this entry, no `case_word_sweep_*` label exists in the

ledger yet, and the only numbers stated here are derived offline from the

frozen artifacts, with the derivation shown, in the

`_phone_comparator_refusals` pattern of

[`scripts/record_synthetic_benchmark_claims.py`](../scripts/record_synthetic_benchmark_claims.py).

The real sender's own log-gain was never consulted and is not stated here: a

registration fixes the controls, not the outcome.



**Estimand.** The per-token log-gain of the Level 3 word prior under the

case-word inlet against the bit-identical alpha = 0 anchor: for each scored

token, `log P_alpha(word) − log P_0(word)` where `P_alpha` is the mixed prior

produced by `apply_case_word_inlet` (`src/ailm/case_word_channel.py`) and

`P_0` is the untouched receiver prior. Aggregates: the mean per-token

log-gain at every grid weight; token direction counts (improved / degraded /

tied against the anchor); and the whole-document clustered bootstrap of

`ailm.levels.clause_channel.clustered_clause_word_summaries`, which expands

each token's difference across its `len(word) + 1` word-level emissions and

resamples and sign-flips whole source documents — the `clause_word_*`

recorded pattern, reused unchanged.



**Population.** The case-word artifact's `test` split

(`data/case_word/ud_ewt_case_word_counts.json`,

`ailm-case-word-counts-v1`): the document-stride test documents it

cross-validates against `vendor/ud_ewt_case_counts.json`'s recorded split

(`test_every = 5`; the artifact ships 64 such documents). A scored token is a

test row's word form, lowercased, kept when it is an exact-string match

against the Stage 3 receiver vocabulary (`vendor/brown_word_counts.json`, the

declared 2000-type intersection), in document then token order; the offline

derivation below resolves this population to 3001 scored tokens and 14449

word-level emissions. A UPOS tag outside the case model's observation

alphabet (`load_case_artifact().upos_to_idx` — twelve tags: ADJ, ADP, ADV,

DET, INTJ, NOUN, NUM, PRON, PROPN, SYM, VERB, X) has no posterior row to

marginalize, so its token stays in the stream and the **no-message policy**

fires: `D_α = D_0`, the token contributes zero gain at every alpha.



**Message and sender.** The model-sender estimand of §4.2 item 4, unchanged:

`K(word | role)` fitted only from the artifact's `sender_map` (train rows,

routes pooled as the declared inspectable choice, no smoothing, support =

the declared intersection) via `fit_sender`; the runtime message is the

posterior marginalization

`M(u) = Σ_role q̃_L4(role | u) · K(word | role)` via `build_message`, which

**excludes zero-support roles and renormalizes `q_L4` over the supported

roles**. The privileged-oracle arm remains a separately labeled diagnostic

and is not executed here. The receiver's scored quantity is the L3 word

level's own per-token arithmetic, mirroring

`scripts/record_clause_channel_claims.py` with the per-token UPOS observation

in place of the sentence type; the sweep loop lives in the recorder because

`sweep_clause_alpha`'s `ClauseMessage` cannot represent a per-token message

nor the arithmetic-free no-message path, while the grid, `mix_prior`,

`scramble_word_distribution`, and the clustered summaries are reused as-is.



**Alpha grid.** The shipped grid, reused unchanged from the clause-channel

arm: `(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)` — the 0.0 anchor plus six

distinct positive weights, so entry 9's grid guard cannot collapse.



**Controls.**



- **Bit-identical zero-alpha anchor.** At alpha = 0 the mixed prior is the

  receiver prior, bit for bit, and every anchor log-probability equals

  `log P_0(word)` exactly. The recorder refuses otherwise.

- **Anchor non-vacuity.** Some positive alpha must change the output.

- **Scrambled-word negative control.** `scramble_word_distribution` of each

  per-UPOS message — the fixed involutive word-axis reversal that preserves

  entropy, peakedness and normalization while destroying only the word

  correspondence.

- **Declared positive control.** A synthetic sender constructed to help: the

  evaluation stream's own pooled in-vocabulary unigram — the strongest

  role-free sender the receiver vocabulary admits. It is a sender swap: it

  sends wherever the model-sender arm sends and the no-message policy fires

  where it fires. It must lead the scrambled-word negative control by the

  pre-registered margin below, at every positive alpha, or the recorder

  refuses with no claims.



**Positive-control margin, derived offline from the frozen artifacts.** The

derivation runs exactly the arithmetic the recorder runs, on the frozen

`data/case_word/ud_ewt_case_word_counts.json`, `vendor/brown_word_counts.json`,

and the case level built by `build_case_agent` from

`vendor/ud_ewt_case_counts.json` (derived population): fit `K` with

`fit_sender`; marginalize per-UPOS messages through `build_message` on

`role_posterior_from_agent`'s rows; score every positive alpha's mean

per-token log-gain against the anchor for (a) the positive control and (b)

the scrambled control; register the gap (a) − (b). Per positive alpha, in

per-token nats: claim:case_word_sweep_alpha_0_1_positive_control_margin at
α = 0.1, claim:case_word_sweep_alpha_0_2_positive_control_margin at α = 0.2,
claim:case_word_sweep_alpha_0_3_positive_control_margin at α = 0.3,
claim:case_word_sweep_alpha_0_5_positive_control_margin at α = 0.5,
claim:case_word_sweep_alpha_0_7_positive_control_margin at α = 0.7,
claim:case_word_sweep_alpha_0_9_positive_control_margin at α = 0.9.

**The registered margin is the least of these,

claim:case_word_sweep_alpha_0_1_positive_control_margin nats per token (the
recorder's `POSITIVE_CONTROL_MARGIN` constant), binding at α = 0.1**;

at that weight the positive control's mean log-gain is 0.118180 and the

scrambled control's is −0.059348. The gap is a deterministic function of the

frozen artifacts and the shipped builders, so the recorder checks every live

positive-alpha gap against this constant minus 1e-6 of rounding slack. These

are pre-registered control constants, not ledger claims; the real sender's

own gains stay unrecorded here.



**Runtime posterior constant.** `q_L4(role | upos)` is fixed by the builder

(`src/ailm/case_word_channel.py`): the case level's **real one-observation

posterior from the reset `D` start** — the empirical prior reset to the

shipped start distribution at every token, the UPOS observed once,

`infer_states` performing the update — asserted row-wise identical to the

closed-form Bayes inversion

`D[role] · A[upos | role] / Σ_role' D[role'] · A[upos | role']` at

`POSTERIOR_IDENTITY_RTOL` (`1e-5`). The case level's learned transition `B`

is deliberately not chained across the receiver's tokens: per-token messages

admit no sequential dependence the shipped estimands never registered.



**Scope warnings.** The scored documents are the EWT dev population that was

already scored as Stage 4 case-assignment populations — development evidence

under a fresh registration, **never confirmatory, never "untouched"**; the

shared corpus's test file remains spent (decision 111). Cross-corpus

transfer — an EWT-estimated message applied to the Stage 3 Brown-receiver

vocabulary, lowercased exact-string match, the declared 2000-type default —

is part of the estimand, named as such, not hidden. The recorded receiver

sensitivities of the other arms (entries 9, 19, 26, 27) describe *their*

interventions and bound nothing here. The case-word registry entries are

registration-only structure, never to be narrated as an executed cascade.

Every derived label remains *agreement, not accuracy* wherever quoted.



**Status.** `exploratory`, by default and by protocol:

`scripts/assign_claim_status.py` maps the recorder accordingly. Nothing here

can promote a development-population sweep to confirmatory; that would

require a fresh population and a fresh registration.



**Refusals (recorder gates, `SystemExit`, no ledger writes).** The grid

guard (a collapsed grid cannot support a finding); Stage 3 vocabulary-order

alignment between the case-word artifact and the word artifact; the fitted

sender's supported roles against the artifact's declared `sender_support`

(the zero-support exclusion is declared, not emergent); **seam

normalization** — every mixed prior must sum to one, the contract

`mix_prior` enforces at the inlet; the **no-message policy** — a messageless

token must reproduce the anchor bit-for-bit at every alpha, the

deterministic `D_α = D_0` never an unnormalized distribution; anchor

bit-identity; anchor non-vacuity; and the positive-control margin gate

above.



**Falsifiers.** The registration is falsified, and the sweep void, if: the

alpha = 0 anchor stops being bit-identical to the receiver prior's own

log-probability (the sweep would be comparing against a different model); a

positive alpha reproduces the anchor exactly (a vacuous anchor); a

messageless token changes under mixing (the no-message policy is not live);

the inlet emits an unnormalized prior (the seam contract is broken); the

positive control fails to lead the scrambled control by the registered

margin (the receiver arithmetic or the channel is broken — the legitimate

finding is the refusal, not a reinterpretation); or any posterior row stops

matching its Bayes closed form within `POSTERIOR_IDENTITY_RTOL`

(`role_posterior_from_agent` refuses). Any result that survives is still

development evidence under this registration's scope warnings and acquires

no confirmatory force from surviving.

### Outcome, recorded after the registration (2026-09-15)

**What ran.** The recorder this entry registered,
`scripts/record_case_word_influence_claims.py` (run `case_word_influence_sweep`,
ledger commit `2eea8ba`), ran after the registration above and wrote the
`case_word_sweep_*` namespace. The "no recorder has run" status at the head of
this entry describes the moment of registration and is superseded here rather
than edited, so the record shows that the outcome was unknown when the
controls were fixed.

**Gates.** Every refusal held open: the zero-alpha anchor is bit-identical
(claim:case_word_sweep_anchor_bit_identical) and live
(claim:case_word_sweep_anchor_live); the scrambled-word control is negative at
every positive weight; and the positive control leads it by at least the
registered claim:case_word_sweep_positive_control_margin_registered nats per
token at every positive weight (claim:case_word_sweep_alpha_0_3_margin_ok at
the best weight, 1 = held).

**Result.** Over claim:case_word_sweep_scored_tokens scored tokens from
claim:case_word_sweep_eval_documents test documents
(claim:case_word_sweep_scored_emissions word-level emissions), the mean
per-token log-gain against the anchor is best at alpha =
claim:case_word_sweep_best_alpha, at claim:case_word_sweep_best_mean_loggain
nats, with claim:case_word_sweep_alpha_0_3_improved tokens improved,
claim:case_word_sweep_alpha_0_3_degraded made less probable and
claim:case_word_sweep_alpha_0_3_tied exactly unchanged; the scrambled control
sits at claim:case_word_sweep_alpha_0_3_scrambled_mean_loggain nats at that
weight. The whole-document clustered per-emission effect there is
claim:case_word_sweep_alpha_0_3_cluster_loggain_effect nats with a
claim:case_word_sweep_cluster_confidence_pct% interval of
[claim:case_word_sweep_alpha_0_3_cluster_loggain_ci_low,
claim:case_word_sweep_alpha_0_3_cluster_loggain_ci_high];
claim:case_word_sweep_alpha_0_3_cluster_loggain_positive_documents documents
favor the message and
claim:case_word_sweep_alpha_0_3_cluster_loggain_negative_documents favor the
anchor. The token-direction counts are the same at every positive weight,
because under convex mixing a token's direction does not depend on the weight,
and they disagree in sign with the mean: more tokens are made less probable
than improved while the mean gain is positive through alpha = 0.5 and negative
from alpha = 0.7. The results section reports both quantities and does not
resolve them into one.

**Status, unchanged.** `exploratory`: development population under a fresh
registration, cross-corpus transfer part of the estimand, never confirmatory.
The manuscript reports it under its own results heading; the queue's P3 item
records what remains, and no performance claim is registered.

## 122 — MEASURED: the P1 warm-up compute gate — the census budget does not clear at the measured rate without the declared efficiency options (2026-09-15)

**What ran.** The protocol P1 warm-up pilot (§2.2's compute gate prerequisite)
completed end to end on the SmolLM-1.7B upstream checkpoint:
`mlx_lm.lora` LoRA (0.088% trainable, 1.507M/1711.376M parameters, rank
default, 4 layers, batch size 1, learning rate 1e-5) over a 2000-instance
slice of the letter census stream
(`cache/fitted_probe_data/pilot/train.jsonl`, drawn from
`letter-instances.jsonl`), 20 iterations with masked-prompt loss, plus
validation on a held-out 400-instance slice. Two harness facts were
established by this run and are prerequisites for anything downstream:
the upstream checkpoint is a **base model without a chat template**, so the
`completions` dataset path of the installed trainer requires a
**passthrough chat-template tokenizer adapter**
(`cache/fitted_probe_data/tokenizer_adapter/` — a copy of the snapshot with
`tokenizer_config.json` gaining a template that renders the user and
assistant contents back-to-back with no role markers and appends nothing at
generation time, so the trained sequence is byte-exactly the scoring-time
`build_prompt("letter", …)` prompt followed by the completion, and the
mask-prompt offset is exactly the prompt length); and the checkpoint was
cached from Hugging Face at
`models--HuggingFaceTB--SmolLM-1.7B/snapshots/d7449ff7…` (the digest chain
records the snapshot identity before any fit).

**Measured telemetry (the trainer's own numbers, not estimates):** final
iteration It/sec 3.003, Tokens/sec 6.006 over loss-bearing tokens, Trained
Tokens 40 at iteration 20, **peak memory 7.271 GB**, validation loss falling
from seven-point-zero to 4.842 across the 20 iterations, adapter weights
saved to `adapters/adapters.safetensors` (the checkpoint/resume path exists
and works). The loss movement confirms the machinery consumes letter
supervision; the pilot is a throughput measurement, not a language result.

**The gate verdict.** The letter census is 5,816,696 fit instances
(`letter-instances.jsonl`, receipt-verified; the supervised-target census
reconciled exactly against `vendor/brown_counts.json`'s recorded
`bigram_tokens`; forty thousand diagnostic instances behind a 40-position
seam). At the measured batch-1 rate of 3.003 it/sec the census-times-
two-epochs budget is ≈ 3.87M seconds ≈ **45 days of wall clock**, and even
the protocol's optimistic minibatching assumptions do not bring it under
days. The gate therefore **does not clear**: the census budget cannot be
committed at the measured rate without the protocol's declared efficiency
options — block-masked packing is the only listed option with a large enough
lever, and it is permitted only with a written mask contract and a verified
equivalence check against single-instance conditioning (§2.2). Until that
unit exists, the letter fit is a scheduled long-running handoff with
checkpoint/resume, not a runnable step, and no letter-fit claim may be
recorded. The gate telemetry itself — this entry — is the recorded outcome
the protocol asks for.

**Amendment (same day): the batched re-measurement.** Rerunning the identical
20-iteration pilot at batch size 8 (mask-prompt, same adapter config, same
2000-instance slice, peak memory 8.964 GB) measured It/sec 1.632,
**Tokens/sec 25.463** over loss-bearing tokens — 4.2× the batch-1 rate, and
validation loss 7.091 → 3.957 (8 instances of gradient signal per step). The
instances-per-second rate scales the same way (1.632 × 8 ≈ 13.1 instances/s
against 3.0 at batch 1), so the census-times-two-epochs budget falls from ≈
45 days to **≈ 8–10 days of exclusive wall clock** — before the C2
deranged-supervision refit doubles it. The verdict is unchanged in kind: the
census budget remains a multi-day exclusive hold on the machine and still
requires the block-masked-packing unit (or equivalent) to become a
runnable step. Minibatching is recorded as the measured, protocol-listed
efficiency option it is — same per-instance objective, batch-level
throughput only; it does not by itself open the gate.
**Falsifier.** A measured run of the packing unit (contract + equivalence
check + throughput) that brings the census budget under the declared
feasibility threshold reopens partition 1. The pilot's telemetry is
reproducible from the same adapter and slice; a materially different
tokens/sec or peak memory on identical settings means the machine's
operating state changed and the gate must be re-measured, not quoted.

## 123 — RULE: every audited gate declares a smallest effect size of interest, and the audit reads the margin against it (2026-09-18)

**The rule.** Each of the eight audited gates in `src/ailm/gates.py` now
declares a smallest effect size of interest (SESOI) — a floor, on the gate's
own reporting scale, below which a strictly positive margin is treated as
operationally negligible rather than as a finding. `smallest_effect_size()`
supplies the declared floor per `GateId` and fails closed on a gate that
declares none (the retained case-only ablation is not a `GateId` member and
carries its recorded non-passing finding instead of a floor).
`scripts/audit_gates.py` computes each row's margin of interest in the
declared unit, prints it beside the verdict (`EFFECT … vs declared SESOI …:
clears | BELOW`), and exports it as `margin_of_interest` in
`output/reports/gate_audit.json` with the value, unit, floor, and a
`clears_sesoi` boolean. Each stage recorder writes its constant into the
ledger beside the gate's other claims (`stage1_gate_sesoi_nats`,
`stage2_phoneme_gate_sesoi_nats`, `stage2_g2p_gate_sesoi_pts`,
`stage3_word_gate_sesoi_nats`, `stage4_case_gate_sesoi_pts`,
`stage5w_gate_sesoi_nats_per_sentence`,
`stage6_narrative_gate_sesoi_nats`, `generation_gate_sesoi_tv`), and
`assign_claim_status.py` marks them `descriptive`.

**What this entry deliberately does not do.** It does not raise the outcome
bar. `evaluate_gate`'s `min_effect` parameter stays at its documented default
of `0.0` on all eight retrospective rows, for the reason its docstring
already gives: those thresholds were restated after measurement, so a
nonzero retrospective `min_effect` would be as retrospective as the
threshold — and on the widened Stage 5 and Stage 6 rows a defensible
retrospective floor would have rewritten recorded outcomes. A *new* gate
must still declare a nonzero `min_effect` before its population is scored;
that requirement is unchanged. What changes is legibility: a hairline
positive margin can no longer certify a gate silently, because the audit and
the ledger now state the declared floor next to the margin in the same
breath. No verdict, outcome, or recorded number moved in this entry; the
re-recorded gate claims reproduce their previous values exactly.

**The declared floors, and where each comes from.** Each floor is chosen
from the gate's own scale and its recorded baselines, not from a
cross-level convention:

- `stage1_letter`, claim:stage1_gate_sesoi_nats nats. Perplexity margins are
  read on the log scale (ln threshold/model); the level's whole captured
  signal over uniform is claim:stage1_gain_over_uniform_nats nats, so a
  quarter of a nat is roughly a quarter of everything the level claims.
- `stage2_phoneme`, claim:stage2_phoneme_gate_sesoi_nats nats. The recorded
  margin is roughly half a nat against a maximum gain over uniform of
  claim:stage2_phoneme_max_gain_nats nats; the same quarter-nat scale as
  Stage 1.
- `stage2_g2p`, claim:stage2_g2p_gate_sesoi_pts percentage points. The
  recorded margin is claim:stage2_g2p_margin_pts pts — the hairline that
  motivated this entry: significant only across all
  claim:channel_factorial_positions positions (McNemar p = 3.5e-6), while
  the bounded-subsample margin (claim:subsample_stride5_margin_pts pts on
  the stride-5 subset) is NOT significant. Half a point is 214 held-out
  positions; the recorded margin clears it, barely, and the audit now says
  so explicitly.
- `stage3_word`, claim:stage3_word_gate_sesoi_nats nats. The recorded gain
  over the letter-bigram bar is claim:stage3_gain_nats nats; the quarter-nat
  scale keeps the three log-scale gates on one declared floor.
- `stage4_case`, claim:stage4_case_gate_sesoi_pts percentage points. The
  recorded derived margin is ≈ 7.81 pts, but the results section records
  that one stratum carries it; two points is about 26 of the
  claim:cluster_stage4_derived_observations held-out tokens, below which the
  aggregate can move with a single stratum. The gold margin is 0.0 and
  non-passing; nothing changes there.
- `generation`, claim:generation_gate_sesoi_tv total variation. On the 0-1
  TV scale the recorded improvement is claim:generation_tv_improvement and
  the degenerate one-word control sits at claim:generation_degenerate_tv;
  below a tenth the policy generator does not separate from sampling from B.
- `stage5_clause`, claim:stage5w_gate_sesoi_nats_per_sentence nats per
  sentence. The point-perplexity margin against the refitted bar is hairline
  by construction, so the floor is placed on the scale the registration
  itself uses: the privileged-label oracle gain in nats per sentence
  (recorded claim:stage5w_oracle_gain_nats_per_sentence, 99%
  document-bootstrap low end claim:stage5w_oracle_gain_sentence_ci_low).
  Below the floor the gain would sit inside a single document's contribution
  and inside the low quarter of the registered band [0.039, 0.321].
- `stage6_narrative`, claim:stage6_narrative_gate_sesoi_nats nats per
  sentence. The clustered design's own resolvable unit sets the floor: the
  planted detectability control shifts claim:cluster_stage6_logloss_effect
  nats and the recorded minimum detectable effect is
  claim:confirmatory_stage6_clustered_min_detectable_effect_nats nats, so
  below half a percent of a nat the development point improvement is within
  document-sampling noise — which is exactly why the primary is recorded
  non-confirmatory (entry 115). The recorded point improvement of
  claim:cluster_stage6_logloss_effect nats clears this floor by a hair, and
  the audit now displays that hairline instead of letting the positive sign
  carry it.

**Falsifier.** The contract is machine-checked in both directions: a row
whose margin of interest falls below its declared floor prints `BELOW`
beside a `PASS` outcome, so the floor cannot be quietly retuned to flatter a
measurement; the export schema test pins `margin_of_interest`'s shape and
its `clears_sesoi` verdict on both branches; and the recorder-vs-constant
pairing means a change to a floor requires re-running a recorder, not
editing prose. If a future re-measurement drops any recorded margin below
its floor, that is a reportable condition of this entry, not a verdict
change: flipping an outcome requires the prospective route (a new gate
declaring its `min_effect` before measurement), which is the only route this
repository's gate contract has ever licensed.

## 124 — DECISION: re-measure the benchmark arms under the updated local gemma3:4b identity rather than restoring the old binary (2026-09-18)

**The observation.** The local Ollama `gemma3:4b` was re-pulled between the
last canonical measurement and this release pass: the model identity the call
cache binds moved from `sha256:28f24028…` to
claim:bench_synthg2p_probe_gemma3_model_digest. The old binary's blob is no
longer present on the machine, and the external-benchmark call store held
25,170 records bound to the old identity — enough that the store's stratified
audit could no longer sample a single identity-consistent stratum, so
`record_external_benchmark_claims.py` refused its pre-check and the `full`
release profile stopped at network-claim-bindings.

**The decision.** Re-measure the affected arms under the new identity. The
alternative of restoring the old binary is unavailable (the blob is gone), and
the alternative of carrying the stale records forward would break the store's
own contract: a cached reply bound to a binary that no longer exists cannot be
re-verified against the server, which is exactly what the identity binding
exists to guarantee. The 25,170 stale records are archived to
`cache/bench_calls.stale-identity-20260918-gemma3.jsonl.archive` under the
store's established precedent; the qwen2.5:3b and smollm stores are unchanged
and bound to their current identities. The aligned-G2P probe recorder already
re-measured under the new identity: every recorded probe score reproduced
exactly, and the digest metadata claim is the only ledger change — recorded
evidence, for one probe battery, that the update did not move this task
family's outputs.

**What this entry does and does not claim.** It does not claim the update is
behaviorally inert. It commits to the token protocol's answer: the re-measured
arms record fresh values under run scopes bound to the new identity, the
manuscript tokens re-derive from the ledger, and any value that moves reaches
the prose through regeneration rather than through a hand edit. If a moved
value crosses a declared gate boundary or a registered interval, that is a
reportable result of this entry, read against the SESOI floors of entry 123.

**Falsifier.** The binding validator refuses a store whose sampled records do
not match the currently resolved identities, so a future model update repeats
this situation loudly rather than silently. The archived store preserves the
old measurements byte-for-byte for exactly the comparison this entry exists to
make.

## 125 — DECISION: re-verify the warm Ollama core after the transport re-bind rather than carrying the 09-18 run ids (2026-09-19)

**The observation.** After the OpenRouter transport re-bind of the synthetic
and in-context recorders, the Ollama-side recorders (probe, external
benchmark, logprob drift) were re-run against the warm call store: 75,291
cache hits against 180 fresh calls (fresh fraction 0.24%), zero unidentified
records, and every manuscript-relevant value reproduced exactly. The external
benchmark claims re-stamped their run ids to the 2026-09-19 scopes with no
moved value. The one moved number is
claim:bench_logprob_drift_captured_mass, which moved from the 3.649e-05
bound recorded at the decision-124 epoch to its currently recorded value —
the maximum per-sample captured-mass drift between stored and freshly
measured answer-slot distributions, a staleness bound whose recorder samples
fresh by design (`post_generate_uncached`), so every run re-measures it and
both epochs sit far inside the tolerance the cache audit justifies.

**The decision.** Accept the fresh run scopes and the re-measured drift
bound. Carrying the 2026-09-18 run ids forward would make the ledger's
provenance columns name runs that no longer exist on the store's tail, and
the drift bound is exactly the quantity the re-run exists to refresh. The
moved value reached the supplement through regeneration
(`z_generate_manuscript_variables.py` → the rendered
`98_supplement_external_benchmarks.md`), not a hand edit, per entry 124's
contract.

**What this entry does and does not claim.** It claims one warm-store
re-verification of the Ollama-side recorders and one refreshed staleness
bound. It does not claim the scored OpenRouter recorders have run — their
scored calls refuse any replay and remained pending the exported
`OPENROUTER_API_KEY` (the release blocker of record in `TODO.md`). A future
re-run that moves the drift bound again is a reportable condition of this
entry, not a verdict change: the bound's meaning is "the stored replies sit
within this much of the live model", and it is re-measured on every
publication run by design.

**Falsifier.** If a re-run of the drift recorder ever produces a captured-mass
bound that the cache audit's tolerance cannot justify, that tolerance is the
thing on trial — the bound's recorder, its sample, and the audit constant
that cites it are re-examined together rather than the number being rounded
down.

## 126 — DECISION: archive the moved gemma3:4b metadata identity and re-measure its stratum rather than repinning the derived hash (2026-09-20)

**The observation.** The release network phase's external-benchmark recorder
refused its stratified store audit: sampled records were not bound to the
currently resolved model identities. The diagnosis is a metadata-identity
move, not a model change. The local `gemma3:4b`'s derived identity moved from
claim:bench_synthg2p_probe_gemma3_model_digest's decision-124 epoch value to a
new digest while the weights blob is byte-identical (mtime 2026-08-01, blob
`sha256-aeda25e6…`) and the Ollama manifest predates every recorded epoch
(2026-09-07). Under the serving Ollama 0.34.2, `/api/show` returns no `digest`
field at all, so the identity falls back to the stable-manifest hash — a hash
of the server's *presentation* of license, Modelfile, parameters, template and
model-info — and that presentation shifted with the server's serialization
while the weights did not. The proof that the model is unchanged came from
the same session: the aligned-G2P probe battery re-measured against the live
server reproduced every recorded score exactly, with the digest metadata claim
the only ledger change.

**The decision.** Follow entry 124's precedent, adjusted for what actually
moved. The stale gemma3 stratum — 26,490 records bound to the superseded
derived identity — is archived byte-for-byte to
`cache/bench_calls.stale-identity-20260921-gemma3.jsonl.archive` under the
store's established naming precedent; the qwen2.5:3b and smollm strata and
the hosted OpenRouter records resolved unchanged and stay. The affected arms
are re-measured fresh under run scopes bound to the resolved identity, and
the resulting values, whatever they are, reach the prose through the
regeneration chain. Repinning the derived hash instead would have made the
identity mechanism's meaning ("a content-bound build identity read from the
serving endpoint") name whatever the code prefers, and a code change to the
derivation function would have invalidated every stratum in the store, not
just the moved one.

**What this entry does and does not claim.** It claims one metadata-identity
epoch for gemma3:4b under an unchanged binary, and that the fresh
re-measurement is evidence about the *current* resolution, not a finding that
the model changed. It does not claim the derived identity is stable across
Ollama server updates — this entry exists because it demonstrably is not.
The binding validator refuses a store whose sampled records do not match the
currently resolved identities, so the next server-side presentation change
repeats this situation loudly rather than silently; the archived stratum
preserves the old measurements byte-for-byte for exactly that comparison.

**Falsifier.** If a future epoch moves the derived identity while the probe
battery's scores also move beyond their recorded tolerances, then the move
was a model change after all, and the archived strata are the evidence that
decides it — the re-measured arms, the archived bytes, and the live server
are examined together rather than the digest being treated as the truth.

## 127 — DECISION: per-scored-call cache namespaces, protocol v3, and the hosted comparator switched to a non-reasoning model (2026-09-21)

**The observation.** The first OpenRouter publication attempt of the stipulated
synthetic zero-shot recorder crashed after fourteen scored calls on
`CacheCorruptionError: divergent reply for existing key`. The diagnosis found
two defects, one external and one internal, each fatal on its own.

The external one is a provider-side capability change. The hosted snapshot
`z-ai/glm-5.3-flash` now reasons before answering, its deliberation is
effectively unbounded at temperature 0 on this task family's ambiguous
next-symbol prompts, and the reasoning budget consumes the whole completion
allowance: every sampled scored call under the recorded 32-token budget
returned a truncated reasoning block with a null answer. The provider states
the constraint directly — a reasoning-disable request is refused with
`"Reasoning is mandatory for this endpoint and cannot be disabled"` — and a
reasoning-token cap of 384 was ignored on scored-length prompts (reasoning
ran to the full budget on 16 of 29 sampled calls). No budget in a usable
range serves the task family; a 4096-token budget would have spent days of
wall clock re-measuring a hosted arm that is mostly truncated refusals.

The internal one is a namespace semantics gap in the re-bind. The
sensitivity arm's one-symbol prompts recur across positions — only 26
letters exist, and 120 positions are scored — so identical payloads recur
inside one run. The re-bind's own principle, already stated in the consumer
recorder's namespace docstring, is that identical payloads under one
namespace collide inside a single run and each fresh measurement becomes
impossible; the re-bind had carried that principle to the probe tag but not
to the scored position. The prior Ollama transports never exposed the gap
because a deterministic provider makes a repeated put idempotent; a
non-deterministic provider turns the collision into a corruption refusal.

**The decision.** Three changes, in one protocol epoch. First, the cache
namespace of every hosted scored call now carries the run scope, the probe
tag, and the scored call's own position slot and arm name, completing the
re-bind's principle: no two scored calls of one publication run share a
namespace, so a recurring prompt can never collide with itself and every
scored call stays live, which the fresh-call guard already demanded. Second,
the synthetic protocols bump versions (`synthetic-zero-shot-v3`,
`synthetic-incontext-v3`, `synthetic-phone-zero-shot-v2`) so records made
under the old namespace semantics cannot be cited by the new receipts; the
recorded `response_tokens` budget stays at 32, restoring the estimand the
hosted arm was specified under. Third, the hosted comparator moves from
`z-ai/glm-5.3-flash` to `meta-llama/llama-3.3-70b-instruct` — the
principal's choice among the options the diagnosis grounded, where two
non-reasoning-family candidates were verified against the exact recorded
prompt contract before the switch. The GLM arm stays in the ledger exactly
as recorded under its own run id, including its partial aligned-G2P run; no
recorded history is rewritten.

**Outcome.** The network phase re-ran end to end: five recorders bound and
verified, 806 claims over 87,594 exact call records; the stipulated
battery recorded 174 claims with zero unparsable answers on the letter arm,
and the in-context sweep recorded 256 claims over its declared position
population for all three registered probe namespaces.
The hosted arm's values moved substantially and reached the prose through
the regeneration chain; the one hand-typed directional sentence the movement
falsified — the phone arm's sensitivity-arm directions — was rewritten to
the recorded directions before the render. A transient socket timeout
aborted the sweep's first attempt mid-run; the hosted transport now retries
transient transport failures up to an attempt cap with a backoff, a retried
answer being a fresh observation by the same rule that already governs
every scored call. The sweep then completed with every declared scored call
fresh and bound, matching the recorded fresh-call count the receipt carries.

**What this entry does and does not claim.** It claims a namespace, version,
and comparator change that makes the hosted scored-call recorders
publishable again, and records the moved values as the new hosted arm's
measurements. It does not claim the GLM snapshot's behavior will not change
again, that the new comparator is stable across provider snapshots (its
identity is slug-derived, and a snapshot change under a stable slug would
be invisible to it — the first provider change under the new comparator
should be treated as the moment to bind the served snapshot's identity
more tightly), or that the moved values carry any finding beyond what the
tokens and the corrected directional sentence state.

**Falsifier.** If a future hosted snapshot change silently moves this arm's
values, the per-scored-call namespaces and the slug-derived identity will
reproduce it loudly in the binding validator only if the identity moves; a
snapshot change under an unchanged slug would not be refused. That residual
is the entry's own reportable condition: the first observed
value discontinuity under the new comparator triggers the tighter
identity-binding work rather than a silent re-measure.

## 128 — FINDING: the stipulated phone arm's one-symbol direction is not run-stable on the hosted transport; the interpretive sentence is retired (2026-09-22)

**The observation.** Two consecutive publication runs of the same registered
phone-arm population (120 positions, identical prompts, per-scored-call
namespaces, one hosted model) recorded conflicting full-versus-short
point-estimate directions for the stipulated phone arm's one-symbol
sensitivity arm: the first decision-127 publication run recorded the
one-symbol estimate *lower* than the full-prefix estimate for all three
registered probes; the re-record under the final source recorded a tie for
`gemma3` (1/120 versus 1/120) alongside lower readings for the other two
probes. The cell sizes are one to three correct answers out of 120 against a
2.56% uniform floor, so the direction of the difference is one-observation
sampling noise on a live non-deterministic transport, exactly the class of
instability entry 127's outcome paragraph anticipated for the new comparator.

**The decision.** The supplement's interpretive sentence that stated the
point-estimate direction is retired. The sensitivity arm's comparison is
stated through what is run-stable: the pre-registered decision-108 outcome
(the one-symbol arm at or below the uniform floor for every registered probe,
which both runs confirm) and the corrected paired contrasts (which no run has
resolved). The hydration guard that pinned the per-probe directions now pins
the pre-registered floor outcome and refuses the retired direction sentences,
so a future editor cannot reintroduce a hand-typed direction that the next
publication run would contradict. The floor outcome is the finding the
pre-registration actually named; the direction was prose convenience.

**What this entry does and does not claim.** It claims run-to-run
instability of a one-observation cell difference under the recorded
protocol, and nothing about the full-prefix arm's advantage: the corrected
comparator-versus-probe contrasts (which resolve at orders of magnitude
below the family threshold in both runs) are untouched by this instability.
It does not claim the direction will never stabilize at a larger population;
at the declared 120-position population it is not a quantity a reader can be
given.

**Falsifier.** A future publication run whose recorded one-symbol
directions are consistent across two consecutive runs with every probe's
point estimate clear of the floor by more than its Wilson interval would
make the direction a reportable quantity again, at which point the sentence
returns as a derived token rather than hand prose.

## 129 — DECISION: the local model identity binds to the weights blob, and every local stratum is archived and re-measured (2026-09-28)

**The observation.** Entry 126 recorded that the stable-manifest fallback — a
hash of the Ollama server's *presentation* of a model, used when `/api/show`
returns no top-level digest — moved for `gemma3:4b` under a byte-identical
weights blob. It moved again afterwards, across server restarts on the same
serving binary, so the fallback hash was restart-unstable by construction and
entry 126's falsifier-free archive-and-re-measure could only recur. The
identity function now returns the content-bearing weights-blob digest that
the Modelfile's `FROM` line resolves to, and a response carrying neither a
server digest nor such a blob degrades to no caching. The serving Ollama
returns no top-level digest for any of the three local probes, so every local
probe's identity changed at once: under the new function the warm store
replayed none of its publication keys, and the external-benchmark recorder's
stratified store audit refused the store because its sampled records were not
bound to the currently resolved identities.

**The decision.** Follow entries 124 and 126 rather than repinning. The three
superseded local strata — every local-transport record bound to a retired
manifest-hash identity — are archived byte-for-byte to
`cache/bench_calls.stale-identity-20260928-manifest.jsonl.archive`; the
hosted records, whose identity is the transport's own and did not move, stay
in the live store. The store is re-warmed and every local arm re-measured
fresh under run scopes bound to the blob identities, and the resulting
values, whatever they are, reach the prose through the regeneration chain.
Entry 126 rejected changing the derivation function on the ground that it
would invalidate every stratum; this entry accepts that cost once, because
the function it replaces had become the defect, and the blob digest is the
identity the mechanism always claimed to read.

**What this entry does and does not claim.** It claims one identity epoch
caused by a change to the identity function, not to any model: the weights
blobs are the same files entry 126 named. It does not claim the re-measured
values equal the archived ones; the local probes are sampled at their
recorded settings and the comparison is available from the archive. It does
claim that the blob identity is stable across server restarts and
serialization changes, because it is a content digest of the weights file.

**Falsifier.** A future epoch in which the blob digest moves while the file
on disk is byte-identical, or in which the re-measured scores move beyond
their recorded tolerances with the blob unchanged, would show that the blob
is not the whole identity of what the server serves (for example, a template
or parameter change applied on top of unchanged weights); the archived
strata are the evidence that decides it.

**Outcome (2026-09-29).** The warm under the blob identities reproduced every
re-measured local claim exactly, so the epoch changed keys and no values. The
full network re-bind then re-measured every arm that requires fresh calls; all
five recorders' bindings verified. Every claim that moved belongs to an
external-probe family — the stipulated synthetic and in-context arms, the
phone arm, the aligned-G2P probe summary, and the log-probability drift sample
— and no claim of the explicit system moved. The synthetic letter arm now
records unparsable replies where the preceding run recorded none; like every
fresh-scored-call quantity, that count reaches the supplement through its
token, and the run-to-run variability it shows is the one entry 128 already
records for these arms.

## 130 — FINDING: which phone-arm context contrasts clear is not run-stable; the supplement reports the run outcome (2026-09-29)

**The observation.** The entry-129 re-bind re-measured the stipulated phone
arm's fresh scored calls. The pre-registered decision-108 outcome held again:
every registered probe's one-phone sensitivity arm scored at or below the
uniform floor. The paired full-versus-short context contrast, which no
preceding publication run had resolved for any probe, cleared its declared
family threshold for the SmolLM base probe in this run, and for neither
instruct probe.

**The decision.** Report it as this run's outcome. The supplement states which
contrast clears, with every p-value hydrated from the ledger, and says in the
same sentence that the preceding run recorded none clearing, so the resolved
contrast is not presented as a stable property of that probe. The hydration
guard pins the per-probe outcome of this run and fails on any future run
whose outcome differs, forcing the sentence to be reviewed rather than left
stale. This is the same class of instability entry 128 recorded for the
one-symbol direction on these arms.

**What this entry does and does not claim.** It claims that the resolution of
this contrast varied between two consecutive publication runs of the same
registered population at the declared cell sizes. It does not claim that
context helps the SmolLM probe in general, and nothing in the body depends on
this accessory arm.

**Falsifier.** Two consecutive publication runs agreeing on which contrasts
clear, with each clearing p-value an order of magnitude inside the family
threshold, would make the per-probe resolution a reportable property.


## 131 — FINDING: an OS LAPACK update moved an analytic protocol field's last bits; float registry fields bind within a relative tolerance (2026-09-30)

**The observation.** Manuscript-variable generation refused the tree with
`claim 'bench_synth_target_start_tvd' disagrees with the executable protocol
registry: recorded 7.100576900487612e-05, expected 7.100576900481541e-05`. The
ledger, the source, the virtualenv and the interpreter were unchanged since the
last successful regeneration; the operating system was not. macOS 27.0.1 was
installed between that regeneration and this one, and
`independent_target_start_tvd` solves for the stationary law with
`np.linalg.lstsq`, which NumPy 2.3 routes through Accelerate's LAPACK on this
platform. The updated library returns values that differ in the last bits: the
letter TVD by 6.1e-17 and, once that guard was relaxed, the phone arm's
`bench_synthphone_target_start_tvd` by 1.2e-17. Both are rounding on the
probability scale, and the propagation test's own `pytest.approx(abs=1e-12)`
reconstruction still passes.

**The decision.** `_ClaimReaders.require_exact_claim` keeps integer, boolean and
string protocol fields bit-exact and binds float fields with
`math.isclose(rel_tol=1e-12, abs_tol=1e-15)`. A TVD is half a sum of differences
between O(1) probabilities, so its rounding error is absolute on that scale —
a few rounding units, summed — and a relative tolerance alone would read the
cancellation in a small TVD as disagreement. The guard's purpose is to refuse
a ledger recorded under a different protocol (another window, chain, seed or
inventory), and every such change moves the value by many orders of magnitude
more than a LAPACK rounding difference. Bit-exact float equality instead made
the guard a detector of platform library builds, which it was never meant to be.

**What this entry does and does not claim.** No recorded value changed, and no
claim was re-recorded. The rendered precision of every quantity bound through
this guard is far coarser than the tolerance. The entry does not relax any
integer, boolean or string binding, and it does not relax any comparison
outside the protocol-registry reconciliation.

**Falsifier.** A protocol change whose float field moves by less than the
tolerance would now pass this guard. No such change is reachable through the
registered protocols: their float fields are distances between distributions
that move at the fourth significant figure or earlier under any change of
window, chain or seed.


## 132 — IMPLEMENTED: a conditional connected runtime and a locally registered GUM comparison (2026-10-05)

**The implementation.** `ConnectedRuntime` now executes the GNN-built agents on
a causal schedule spanning normalized letters, position-aligned phonemes, word
identity, case, widened sentence type, and document phase. Train-only bridge
kernels pass explicit timed messages. Structured input may supply UPOS; the
raw-text path derives its own observations without reading evaluation labels.
Next-word generation calls live L1g policy inference and action selection.
Within-level temporal recursion remains distinct from an inter-level message.
Independent conditional references test the declared updates; they do not
certify a globally exact posterior. The compiled and eager Agent-call paths
were compared on complete inference traces and generation outputs; compilation
changes execution, not the schedule or model. Historical stage and intervention
results retain their original standalone scope.

**The local registration.** Before the confirmatory connected score was run,
`data/connected_gum/protocol.json` and `input_manifest.json` froze the pinned GUM
test artifact, document exclusions, estimand, channel strength and bootstrap.
The retained population has claim:connected_gum_documents previously unexamined
documents after excluding claim:connected_gum_excluded_documents used for bounded
development. The comparison fixed channel strength at claim:connected_gum_strength
and accepted null or harmful results without tuning that strength. This is a
prospective record inside this development session, not an external registration
or an independent replication.

**The measured result.** Equal-document normalized-character log loss was
claim:connected_gum_document_integrated_log_loss nats for the integrated arm and
claim:connected_gum_document_disconnected_log_loss nats for the named-channel
disconnected arm. The paired gain is claim:connected_gum_document_mean_gain
nats, with document-bootstrap interval [claim:connected_gum_document_ci_low,
claim:connected_gum_document_ci_high] at
claim:connected_gum_bootstrap_confidence_pct% coverage. This interval favors the
integrated arm on this population. The scoring stream consists of normalized
ASCII lexical letters and token spaces, not arbitrary Unicode or raw bytes.
All claim:connected_gum_oov_tokens out-of-vocabulary tokens remain in the
denominator; vocabulary coverage is claim:connected_gum_vocab_coverage_pct%.
Full-stream closed-vocabulary word loss is infinite. That limit prevents the
character result from being read as open-vocabulary word prediction success.

**Falsifier and boundary.** Replaying the frozen inputs with a changed score,
source-binding mismatch, dropped OOV token, or different channel intervention
invalidates this result. A new population or intervention requires a new
comparison; the present result licenses neither semantic document generation
nor global Bayes exactness.

## 133 — CORRECTION: hosted run names did not identify local model weights (2026-10-05)

**The defect.** The stipulated letter/phone and in-context recorders used the
hosted model slug `meta-llama/llama-3.3-70b-instruct` while retaining the
`gemma3`, `qwen25`, and `smollm` protocol keys. Earlier tables, legends and
decision prose rendered those keys using the local Gemma, Qwen and SmolLM
display names. Those names were not evidence that the correspondingly named
weights served these hosted requests. Historical entries, including the
run-instability observations, remain intact; their hosted model attribution is
corrected by this entry.

**The repair.** Hosted views now use labels ending in “run”, disclose the actual
requested model and provider route recorded by each run, and derive categorical
phone-context outcomes from the recorded corrected contrasts. Cohort keys,
prompt schedules, tested family and raw claim namespaces remain unchanged.
Each cohort consists of separate fresh calls to the same hosted model, not a
cross-model replication. The English and aligned spelling-to-pronunciation
reference arms retain their recorded local-model identities.

**Identity limit.** OpenRouter logical slug hashes do not authenticate immutable
serving weights. Capped receipts bind the exact provider request block and
payload; raw server metadata records the returned model and provider when
supplied. A missing system fingerprint is not a weight pin. Exact replay must
reconstruct the recorded provider block without consulting a later environment,
and must refuse a missing or altered block rather than silently changing the
cache identity.


## 134 — CORRECTION: hosted recorders have no local-model preflight or warm-up (2026-10-05)

**The reproduced defect.** Both hosted synthetic recorders refused a real
loopback server advertising an empty Ollama inventory before attempting any
hosted request. They also warmed local model tags and later rechecked local
weight digests, although every scored request targeted the single hosted model
disclosed in entry 133. Those operations neither warmed nor authenticated the
model being scored.

**The prospective execution repair.** Before the next paid publication run,
the two hosted recorders remove local availability, warm-up and weight-digest
queries. Their `--endpoint` now selects the actual chat transport; publication
requires the canonical OpenRouter endpoint. Their explicit policy is
`hosted-no-warmup-fresh-scored-cells-v1`, with an integer zero warm-up-call
count. Positions, seeds, prompt text, response contracts, arm order, run keys
and correction families remain unchanged. The local English and aligned-G2P
recorders retain their local model checks.

**Verification boundary.** Exact hosted payload/provider/cache identities,
source manifests, producer-to-consumer agreement, and fresh scored-call counts
remain mandatory. Real isolated producer and consumer CLI runs over a loopback
chat server exercise the complete request schedule, while a failed hosted call
must preserve the prior ledger. These are transport controls, not new model
performance observations. New paid measurements must record the repaired policy
rather than inheriting the old warm-up claim.

## 135 — PROSPECTIVE CORRECTION: explicit local loader and context capacity (2026-10-05)

**The operational defects.** The local Gemma diagnostic with the default loader
returned a server-start timeout; the same verified weight blob generated a reply
when the request explicitly disabled memory mapping. A later external-recorder
attempt still omitted the context option and started an oversized runner under
the server's model-dependent default. That attempt stalled amid host memory and
temporary-file pressure and was interrupted. These observations motivate a
fixed serving configuration; they do not establish a model-performance result
or isolate every cause of the startup failures.

**The prospective serving declaration.** Before the next scored external phase,
all local generation routes, including uncached warm-up and logprob requests,
explicitly send `use_mmap: false` and default `num_ctx: 8192`, declared by
`OLLAMA_USE_MMAP` and `OLLAMA_NUM_CTX`. Intentional explicit valid context limits
remain supported; the effective context must exceed the response budget. No
model tag, weight identity, prompt, seed, sampled position, protocol arm or
statistical computation is changed. There is no silent loader fallback or
alternate model. Request keys and reconstructed replay payloads bind the serving
options. Historical records with missing or different options remain preserved
but cannot satisfy the current external audit or drift population.

**Capacity and audit policy.** The default builder refuses a request before HTTP
when its UTF-8 byte-token upper bound, response budget and inspected-template
reserve exceed the declared context. The external receipt and warm planner bind
`utf8-byte-upper-bound-plus-template-reserve-v1` and
`current-loader-context-compatible-record-digest-seeded-v2`. Drift remains
model-stratified over the current authenticated external producer's logprob
requests. A cold current configuration is only a precondition state; standalone
and post-measurement audits still require every model/request-kind stratum.

**Evidence boundary.** The complete external request inventory fits the declared
capacity under source-derived bounds for the inspected tokenizer and single-turn
template paths. The SmolLM bound also fits its smaller native trained context.
The proof covers inspected Go rendering and native chat-template alternatives;
a runner's ChatML startup flag alone does not establish the request's rendering
path. The preserved evidence under `.omo/evidence/ollama-context/` separates the
initial failed tokenizer-alphabet assumption from the final passing byte-fallback
proof, and records real loopback HTTP positive and refusal controls. Capacity
analysis and transport controls are not fresh scored model-generation evidence.
The next measurement must use the frozen corrected source and record its actual
server replies, prompt token counts and provenance before publication.

## 136 — PROSPECTIVE CORRECTION: project-owned local benchmark endpoint (2026-10-05)

**The observed interference.** The shared local server repeatedly replaced the
external benchmark's explicit-context runner when another workload submitted
chat requests with different runner options. Those requests canceled while
loading; the next benchmark request then reloaded its own configuration. The
pinned scheduler compares runner options for the same model path and expires a
mismatched runner, so a model alias or longer keep-alive would not isolate this
conflict. The benchmark's own interrupted process was stopped; unrelated
processes and the shared server were preserved.

**The prospective endpoint declaration.** Before the next scored phase,
`AILM_OLLAMA_ENDPOINT` selects a project-owned HTTP loopback origin. An absent
variable retains the legacy default; an explicitly empty, malformed or nonlocal
value is refused without fallback. The current operational choice is
`http://127.0.0.1:15434`, using the same Ollama binary and weight store, with
`OLLAMA_GO_TEMPLATE=1` explicitly matching the observed shared-server template
setting. This endpoint isolation changes no prompt, weight, seed, population,
arm, score or statistic. It does not eliminate shared hardware contention.

**Identity and replay.** Selection is canonicalized once at process import and
propagated through producer configurations, warm planning, request/cache keys,
cache audits, drift and the controlled release environment. Publication endpoint
guards mean the selected canonical origin. External semantic replay and drift
refuse selected-versus-recorded endpoint mismatch; historical verification must
select the recorded origin in a separate process. Old shared-server records are
retained but excluded before current audit strata and their sampling digest are
formed. The new audit policy is
`current-endpoint-loader-context-record-digest-seeded-v3`; the drift sample remains
model-stratified over the current authenticated producer population. Cold-start
preconditions do not replace the strict complete-strata audit before ledger write.

**Evidence boundary.** Real loopback HTTP, file-backed cache, fresh subprocess and
controlled release-child controls exercise selection, refusal and endpoint-bound
identity. Readiness and matching model metadata for the isolated daemon are not
new scored generation evidence. Interrupted measurements remain historical
partial cache records; the next phase must run against this corrected frozen
configuration and record actual replies before publication.


## 137 — PROSPECTIVE CORRECTION: compare cached logprob candidates within their emitted positions (2026-10-06)

**The observed verifier defect.** The external recorder stopped during its
post-reboot cache validation before publishing replacement claims. Repeated
uncached diagnostic replies for the failing stored request retained exactly the
same returned JSON text. The old `_distributions_agree` nevertheless flattened
all emitted positions into a single candidate-to-logprob dictionary. A repeated
candidate at a later position overwrote its earlier value; when that candidate
dropped out of the later top-k tail on replay, the dictionary compared values
from different conditioning contexts. This produced a false drift failure even
though the correctly aligned distributions for that request remained within the
existing tolerance. The diagnostic record is preserved under
`.omo/evidence/cache-post-reboot-diagnosis/`; it is evidence about the observed
request, not proof that the whole cache reproduces.

**The correction declared before the next scored phase.** Distribution replay
compares candidates within every aligned emitted position, preserving position
count and emitted-token context. It requires usable finite evidence and at least
one shared candidate at each compared position. A missing position, different
emitted-token sequence, non-overlapping candidate evidence or genuine drift
above the existing `LOGPROB_TOLERANCE` refuses validation. Ordinary requests with
empty distribution lists on both sides remain valid under the separate exact
returned-text check. Reproducible malformed or otherwise invalid model answers
remain scoreable as incorrect answers; the replay verifier must not discard them
merely because they are not successful task responses.

This comparison covers every emitted position, including formatting positions;
the diagnostic answer-slot localization does not narrow the gate to the scored
answer. Exact full returned-text equality and the existing `0.5` log-space
tolerance are retained. This is a prospective verifier correction after an
observed failure, not a claim of prior registration or a new scientific result.
No prompt, seed, model weight, population, arm order, scoring rule, statistical
method or scientific threshold changes.

**Input preservation and the next run.** The stored call cache and claims ledger
remain preserved. No inconvenient response is deleted, replaced by a favorable
replay, or relabeled as a fresh call. Earlier source-bound receipts are not
silently rebound. After the correction and independent controls are frozen, the
next canonical phase must re-execute the required cache validation and recorder
paths, including their complete-strata checks, before publishing current claims.
Passing the diagnostic regression alone does not certify that phase.


## 138 — OPERATIONAL CONTINUATION: preserve interrupted scopes and verify the current upstream checkout (2026-10-07)

**Storage and evidence preservation.** Repeated loss of the selected SSD interrupted
publication acquisition. Execution continued in a verified recovery copy with
independent source, environment, corpus, provider-route and append-only call-store
checks. The original checkout and interrupted scopes were preserved. Budget
accounting carries forward settled costs and unresolved reservations, including
a conservative recovery reserve; changing the execution copy does not start a
new spending allowance. A canonical interrupted in-context sweep is restarted
under a new full scope, without combining partial cohorts or relabeling old calls.
Recovery readiness is permission to execute the acquisition, not publication
acceptance.

**Current GNN selection.** A fresh read-only upstream check found a newer public
GNN main commit than the previously selected snapshot. An isolated standalone
checkout at `1bc3a76eccccb2cc5ce8601770714075b3ec48db` was tested before selection.
Its parser and type trees match the earlier snapshot; actual interoperability
and deliberate mutation controls passed. The previous checkout is retained.
This verifies the notation/parser integration used by AILM, not every upstream
GNN pipeline or a new linguistic result. The final release receipt must name
and exercise the selected checkout through its supported discovery route.

**Local serving boundary.** The shared Ollama application changed during the
interruption. The isolated endpoint now runs the retained Homebrew-packaged
`0.35.1` server against the preserved model store; this is a different executable
package from the earlier application binary, not a claim of byte identity.
Version and weight metadata alone do not prove response equivalence. The final
complete-strata live cache audit must pass its unchanged text and aligned
log-probability requirements before publication acceptance. The hosted
in-context cohorts retain their recorded provider/model contract independently
of this local serving package.

**Documentation and review boundary.** The complete documentation tree is reviewed
for the connected runtime, current gate qualifications, acquisition versus
verification commands, and rendering contracts. Historical decisions and failed
results remain evidence. Authored edits and focused controls do not certify a
generated PDF or release: a clean full receipt, complete visual PDF review,
matching tested package bytes and independent final review remain required.
No remote release or change of publication destination is authorized by this
preparation record.

## 139 — OPERATIONAL SELECTION AMENDMENT: current GNN interoperability after release recovery (2026-10-09)

**Current selection.** Fresh read-only checks resolve both upstream main and the
clean canonical GNN checkout to `9255e19f951d4c9c1461514464d420bd3fc8b912`, the
GNN `4.1.0` merge dated 2026-10-08. This supersedes decision 138's selected
checkout for the next local release; its earlier selection and verification
remain historical evidence. The parser subtree changed since that selection,
while the types and type-checker trees retain identical bytes. Parser-tree
identity is therefore not asserted for the current selection.

**Bounded verification.** Actual AILM upstream interoperability and deliberate
declaration, inference and dimension mutation controls pass against this exact
checkout through the supported discovery route. The next complete release must
exercise it again and retain its commit, tree and imported-parser identities.
These controls verify AILM's notation/parser integration; they do not certify
the whole upstream GNN pipeline or establish equivalence of every parser input.

**Preserved measurement boundary.** The GNN specification, discrete pymdp
computational core and external LLM auxiliary probes remain distinct. Original
scientific receipts, recorded model/request identities, corpora, cache, claims
and numerical results are not rebound by this operational selection. The
original-source reproduction pin remains unchanged. Release recovery preserves
incomplete attempts and failures; focused controls do not replace the required
clean full receipt, current rendered QA and independent final acceptance.
No external push, publication, release or DOI change is authorized here.

## 140 — PUBLICATION METADATA ALIGNMENT: Institute release destination (2026-10-14)

**Destination and availability.** The planned public release destination is
[ActiveInferenceInstitute/Active_Inference_Language_Model](https://github.com/ActiveInferenceInstitute/Active_Inference_Language_Model).
The abstract, reproducibility statement, package URL table, citation file and
archive metadata identify that destination consistently. The private development
repository remains the actual source until publication. An empty public
repository is not an available release, and a reserved DOI is not a completed
deposit. The cover retains its draft DOI label until human manuscript acceptance;
publication and removal of that label require the separate acceptance boundary.

**Exact source boundary.** The finite publication contract now admits the package
URL table after comparing parsed configuration with the original measurement
source. Every non-URL field must remain identical. Dependencies, build backend,
Python requirement, package version and coverage floors each have deliberate
mutation controls that reject changes even when support hashes are refreshed.
This metadata alignment changes no measured claim, model, corpus or cache input.

**Operational failure preservation.** The completed software recipe and its
failed periodic catalog request remain separate receipts. The user-authorized
operational readiness amendment records that distinction after the attempt; it
does not rewrite the failed operator result or establish continuous service
availability. A successor operator avoids repeated whole-catalog polling during
maintenance, retaining full catalogs at startup and completion and checking every
recorded model identity periodically. Actual timeout, identity, retention and
terminal catalog controls remain required. Fresh source-bound rendering, package
verification and independent review are required before successor delivery.
