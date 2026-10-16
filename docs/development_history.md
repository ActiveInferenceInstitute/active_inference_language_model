# Development History

This is the historical decision-by-decision changelog preserved from
`CHANGELOG.md`, including its dated release entries. The entries below retain
the vocabulary and implementation state of their time; they are not current
API guidance, source availability or release-acceptance evidence.

For current model paths, start with [the governing hierarchy](hierarchy.md) and
[the connected runtime contract](design/connected_runtime.md). The append-only
[decision log](decisions.md) records later corrections and their chronology;
the public-facing [changelog](../CHANGELOG.md) carries the concise release record.

---

## [1.0.0] — 2026-09-22: the hosted-transport release pass (decisions 125–128)

### Changed

- The hosted scored-call comparator moves from `z-ai/glm-5.3-flash` to
  `meta-llama/llama-3.3-70b-instruct` (decision 127): the GLM snapshot now
  forces unbounded hidden reasoning that cannot be disabled, truncating every
  scored call under the recorded 32-token budget. The replacement comparator
  was verified against the exact recorded prompt contract before the switch.
  The `bench_synth_*`/`bench_ctx_*` arms are re-recorded under it through
  per-scored-call cache namespaces (run scope + probe tag + position slot +
  arm), which complete the transport re-bind for the hosted transport's
  non-deterministic replies; the stipulated protocols bump to
  `synthetic-zero-shot-v3`, `synthetic-incontext-v3`, and
  `synthetic-phone-zero-shot-v2` under the unchanged recorded token budget.

- The stipulated phone arm's one-symbol sensitivity direction is recorded as
  not run-stable on the hosted transport (decision 128): consecutive
  publication runs recorded conflicting full-versus-short directions. The
  supplement states the sensitivity arm's comparison through the
  pre-registered floor outcome and the corrected contrasts; the hydration
  guard pins the floor outcome and refuses the retired direction sentences.

### Fixed

- The OpenRouter transport retries transient transport failures up to an
  attempt cap with a backoff; a retried answer is a fresh observation, and a
  failed attempt records nothing. The reply's usage block feeds the receipt's
  prompt-token contract (decision 127).
- The local gemma3:4b identity moved a second time with the weights blob
  byte-identical — the serving Ollama 0.34.2 stopped returning a manifest
  digest, so the stable-manifest fallback hash moved. The stale 26,490-record
  stratum is archived byte-for-byte and the affected arms re-measured
  (decision 126); the identity mechanism's presentation sensitivity is now a
  recorded property, not an assumption.
- Release mechanics fit the grown recorder inventory: the `tests-full` bound,
  the network-bindings verification bound, the raw-corpus locator's
  subprocess bound, and the wheel-resource test's per-test timeout now name
  the measured costs; the runner forwards the hosted credential to its
  recorded steps; the live isolated-ledger recorder test runs at the smallest
  population the pre-registered phone-arm comparator refusals clear; and the
  documented untracked-run-log convention is actually ignored by Git.

## [1.0.0] — 2026-09-18: the OpenRouter transport, the mark-scale manuscript epoch, the main-text/supplement split, and the SESOI gate contract

### Added
- The external-model transport moves to OpenRouter: the synthetic and
  in-context benchmark recorders are wired for it, the module row joins the
  architecture map, and the GLM 5.3 Flash scored arm is recorded through it.

- The P2 stipulated aligned-G2P comparator arms are recorded offline: sixteen
  `bench_synthg2p_comparator_*` claims covering the native arm refit on the
  stipulated train pairs (99.72% position accuracy, 98.33% exact match), the
  A1 English-transfer arm (3.89%) and the A2 broken-correspondence arm
  (0.0%) as live refusal gates, and the uniform-floor control at 2.56%.
  `scripts/record_aligned_g2p_comparator_claims.py` runs the offline
  evaluator's arms against the digest-bound stipulated artifact and writes
  the claims.

- The P3 case-word influence sweep is recorded per decisions entry 121:
  `case_word_sweep_*` claims over 3001 scored tokens across the artifact's 64
  test documents, with the bit-identical zero-alpha anchor, the
  scrambled-word negative control, and the pre-registered positive control
  live; the mean per-token log-gain is best at α = 0.3 at +0.1715 nats.
  `scripts/record_case_word_influence_claims.py` runs the registered
  estimand and refuses outside its controls.

- The P1 letter full-corpus stream construction is implemented — the
  scheduled handoff from the warm-up pilot: the authenticated pinned stream,
  the streaming census reconciled to the 5,856,736 `bigram_tokens` of
  `vendor/brown_counts.json`, the diagnostic carve behind its declared seam,
  the 2.5GB SFT-format write, and the verified write-once receipt.

- The P2 zero-shot probe-arm recorder
  (`scripts/record_aligned_g2p_probe_claims.py`) is implemented and gated
  with three registered tags, 180 scored calls, and the analytic contract
  floor 1 − 1/39; the scored calls themselves are pending a quiet Ollama
  window and are not recorded.

### Changed

- The manuscript adopts the main-text/supplement split: the audit trail, the
  mis-wiring catalogs, the confirmatory per-member detail, the external-probe
  detail, and the Lean inventory move into five `98_supplement_*` files
  (audit trails, complexity, external benchmarks, Lean, statistics), each
  main-text seam keeping a verdict-level summary and a cross-reference.

- Every audited gate declares a smallest effect size of interest (decision
  123): `src/ailm/gates.py` carries the per-gate constants, `audit_gates.py`
  reports the margin against the declared floor, and the floors are recorded
  claims. No recorded verdict changed.

- The abstract is trimmed to venue length and now carries the Stage 4
  stratum caveat; funding and competing-interest statements are declared
  (none); the derived-count prose sites are token-bound
  (`CONFIRMATORY_MEMBERS_*`, `CHANNEL_SEPARATELY_MEASURED_COUNT`,
  `CHANNEL_CASE_WORD_SEAM_COUNT`); the decision-locator convention is
  documented (frozen-history locators may be typed; registered `DECISION_*`
  literals stay contract-checked); the overloaded sentences are split.

- Repository documentation refreshed against the tree: the 300 s pytest
  timeout text, the nine-spec GNN counts and the two `L5w_clause_widened`
  mutation rows, the set DOI in the rendering-pipeline table, the ISA goal
  at decisions 112/114/111/115, the `ailm` README's mermaid fence, and the
  2026-09-16/17 commits missing from this file are now recorded.

### Fixed

- `stage5w_auc_ci99_low` is restored to the widened Stage 5 recorder — an
  accidental omission whose recomputation reproduces the recorded 0.7341 —
  and the six clustered-interval metadata labels the Stage 5 recorders emit
  are recorded; the ledger sha pin follows.

- The foreign `setup.sh` (an unrelated installer committed in 6dffa84) is
  removed from the repository root.

### Measured

- Decisions entry 122: the P1 warm-up compute gate ran end to end on the
  SmolLM-1.7B upstream checkpoint at 3.003 it/sec (6.006 tok/s over
  loss-bearing tokens, peak memory 7.271 GB). At that batch-1 rate the
  census budget — 5.82M instances × 2 epochs — is ≈ 45 days of wall clock,
  so the gate does not clear: committing the census fit requires the
  protocol's block-masked-packing unit, and until it exists the letter fit
  remains a scheduled long-running handoff with checkpoint/resume, not a
  runnable step.

### Fixed

- The standalone demo renders probability fills at their actual widths and clears
  stale counts and bars for unscorable input. Model controls remain disabled until
  the self-check and initial rendering succeed; initialization errors are visible.
  Missing or invalid displayed claims are refused, and generated samples show their
  actual word count. Its reference label names the evaluation fixture, and
  precision instructions distinguish changing the control
  from generating a new sample.

- The dashboard labels its manual phase selector separately from chat's turn-derived
  phase and avoids requesting a missing favicon during page load.

- Prose claim validation checks each source file separately against one shared
  read-only ledger, preserving its inventory and controls while identifying the
  file responsible for a failure under the ordinary per-test timeout.

- Manuscript significance prose uses the ledger-derived development sign-test
  conclusion and distinguishes all held-back family outcomes. The data statement
  includes Czech and stipulated populations and separates EWT split histories.
  Reproduction guidance distinguishes fresh measurement from eligible cache
  replay, and the protected one-shot replication from ordinary recorder phases.
  Complete methods/results/supplement reviews corrected registration chronology,
  calibration comparison scope, the repetition guard's limits, and the documented
  claims-scan inventory.

- Related work now names the oracle's implemented inference algorithms, separates
  the analytic generation reduction from corpus fitting, and qualifies the
  finite-state comparison using the cited paper's state-pair construction and
  determinization limits. The manuscript status includes all open author decisions
  and the widened clause arm's development-session provenance.

- The phone-control interpretation follows the recorded direction and context
  contrasts, and identifies its prospective development-session provenance
  (decision 120).
  The open queue distinguishes the shipped phone chain from an unbuilt aligned
  pronunciation control; the historical research queue no longer presents its
  truncated verdict export as current authority.

- Layout, legend, and reference-rule assertions share one figure construction
  per discovered entry through a bounded fixture, while remaining separate
  tests with the same controls and completeness check.

- The in-context figure adds aligned zoom insets while retaining its comparator
  scale. Figure publication checks standard axes-coordinate rules against label
  extents, with explicit transform, clipping, opacity, and z-order controls.
  Calibration legend/callout backgrounds now hide the rules behind their text
  (decision 119).

- Documentation retention has an explicit plan/apply interface with dry-run
  inventory, a configurable count and age grace, protection for the current
  publication, and refusal of symlinks and unmanaged names. Normal rendering
  still preserves prior generations until cleanup is explicitly requested. Non-finite
  ages/times are rejected; apply mode refuses vanished, replaced, or refreshed
  directories instead of acting on a stale plan.
  Its full CLI round-trip test excludes site packages, enforcing the renderer's
  standard-library boundary across generation, freshness, repair, and refusal.

- The letter bigram builder is named `build_letter_transition` throughout active
  callers, GNN provenance and figure footers. `build_unigram_a` remains an alias
  to the same function, including the dashboard re-export; historical decision
  entries remain unchanged. The builder's numerical implementation is unchanged.

- The parameter census follows canonical widened Level 5 and records the retained
  case-only arm separately, excluded from canonical totals (decision 118). Both
  arms are checked against their own GNN declarations. Prose distinguishes
  live-support simplex capacity from parameters actually fitted by the pipeline.
- The required case-only gate report is explicitly included by `.gitignore`,
  matching the canonical report's publication contract.
- Release CLI controls use small real repositories and a deterministic invalid
  cache selection, avoiding unrelated dirty-tree hashing and live-server planning.
  Benchmark exports load their owning modules on access, and the warmer reads
  probe identities directly from the shared registry. The stdlib-only import
  boundary replaces an unsuccessful cold-import timeout increase.
- The confirmatory figure refuses a development interval wholly below zero as
  well as one wholly above it, preserving its stated spans-zero interpretation.

- The level-module guide distinguishes canonical widened L5 from its case-only
  ablation, describes the actual evenly spaced subsampling rule, and replaces
  obsolete standalone diagnostic numerals with recorded claims or source links.
  Coupling prose names its measured grid and the separate receiver interventions.

- Release executions use separate log directories keyed by a recorded run ID, so
  a later profile cannot overwrite the command output referenced by an earlier
  receipt. Destination checks cover the actual per-run log and scratch paths.
- Claims consumption and tag validity use the same factual Markdown inventory,
  including formal and GNN documentation. A removed-reference control proves the
  consumption scanner does not retain a label merely because its file exists.

- The Lean axiom driver is generated from the recorder's shared declaration
  inventory, with read-only freshness checks and refusal of unsupported or
  duplicate declarations. Per-module theorem totals are recorder-owned ledger
  tags. The shell writer stages Lean output and preserves the previous report
  on failure while printing the failed run's diagnostics.
- The confirmatory replication figure uses shorter print annotations and separate
  primary, corrected, and development interval rows. Each interval's coverage is
  read from its own recorded field; the caption retains the inference limits.

- Widened-L5 spec construction now uses the declared wider emission and the
  widened artifact's fitted transition and prior. A new live-Agent identity test
  binds it to the shipped builder; the existing case-only configuration remains
  independently covered. GNN documentation reflects both clause arms.
- Recorder orchestration orders the development clustered recorder before its
  held-back clustered consumer. The empty-ledger in-context control supplies the
  required producer scope, and the inverted-axis figure test compares interval
  endpoints in numeric order.
- The full release plan generates and checks the title-page cover before
  rendering. The cover's read-only check rejects missing or stale bytes;
  real-file controls and the ordinary suite detect an outdated published cover.
  Rendered-provenance test guidance now names Stage 04 as the report writer.
  Cover qualifiers clear every drawn level box, including the explicit L6
  family-corrected interval limitation; pale level boxes use dark text.
- Parameter accounting rejects negative and non-finite probability mass before
  checking normalization or counting free parameters; zero padding remains valid.
  Both canonical and ablation GNN specs remain explicitly registered.
- Gate-report consumers now require the complete canonical and case-only ablation
  inventories. Missing, empty, duplicate, malformed, or unknown rows fail rather
  than skip or pass vacuously. Temporary-file controls exercise both reports.
- The figure registry respects an explicitly configured template checkout and
  rejects duplicate manuscript filenames or labels before a caption is overwritten.
- Documentation rendering uses the shared ledger formatter without importing the
  model or benchmark runtime. Manuscript formatting keeps its existing import
  interface and precision. The renderer validates output paths before hydration
  and leaves an unchanged publication directory intact; changed content still
  goes through staged publication and post-publication verification.
  Check mode names retained Markdown without a current source as stale, including
  files in retired subdirectories. It preserves those files for explicit review;
  write mode continues to refuse an unreconciled publication inventory.
- Confirmatory figure controls have explicit field types, and floating-point
  plotting positions no longer reuse integer loop variables.
  The improvement axis now states the recorder's subtraction direction:
  type-unigram loss minus phase-model loss.
- Related work distinguishes variational free energy for selecting sampled
  acoustic segments from expected free energy for future observations in
  *Active listening*. The literature-search record carries the primary source
  and section locator; the priority-search limits remain unchanged.
- Contributor documentation now states canonical-ledger ownership, the shared
  gate classifier, qualitative palette usage, and the repeat-trial namespace
  prefixes consistently with their implementations.
- The spelling gate prunes its existing exempt directories before descending
  into them, and a real-file control preserves the declared scan scope.
- Results prose now directs the reader to the confirmatory replication figure.

### Changed

- The default pytest timeout is raised to 300s with the fleet-load evidence;
  the Ollama generate read timeout is scaled to 1800s for contended cold
  inference; and the SmolLM base-model passthrough tokenizer adapter for
  mlx-lm's completions training path ships with the warm-up pilot's setup,
  so the trained sequence is byte-exactly the scoring-time prompt followed
  by the completion.

### [1.0.0] — first public artifact release (2026-09-05)

Every dated section below this one is the development history of this
release; none was tagged. The two passes that closed it: the confirmatory
replication, the widened clause arm, and the reader-facing object (this
section), then the lexical backoff correction, the Holm layer, and the
reviewer findings (the section after it).

Decisions 111 and 112 in `docs/decisions.md` carry the record; every quantity
is a `claim:` tag there and a `{{TOKEN}}` in the manuscript, resolved from
`data/claims.json`. Nothing in this section types a measured value.

### Added — four family members scored once on the held-back UD EWT test split (decision 111)

- The registration was committed before the recorder existed, then the
  recorder scored the four EWT members of the frozen sixteen-member family
  exactly once against the whole of `en_ewt-ud-test.conllu`, vendored through
  the identical generators with no stride, every parameter read unchanged from
  the development fits, document identities checked disjoint, a planted-effect
  positive control and a label-shuffled negative control run first. The
  sentinel `confirmatory_run_completed_ud_ewt_test` makes a second pass refuse;
  the population is spent. Every registered direction and threshold prediction
  held: the Stage 4 derived comparison and the Stage 6 observed-position sign
  test cross the frozen threshold in their recorded direction, and the Stage 4
  gold comparison and the Stage 5 classification return no discordant pair and
  are reported as not shown rather than shown absent. The other twelve members
  are unchanged development-population inference.
- Manuscript: a Methods protocol subsection (`sec:confirmatory_method`), a
  Results section with one subsection per member and `tbl:confirmatory`
  (`sec:confirmatory`), and a Discussion section on what the replication does
  and does not change (`sec:confirmatory_discussion`). The seven sentences that
  said the EWT members remained unscored pending a principal decision now say
  what happened, and the abstract, README status block, level list, and
  `docs/hierarchy.md` carry the outcome. `data/confirmatory/` holds the
  population counts; `scripts/generate_confirmatory_ewt_counts.py`,
  `scripts/record_confirmatory_replication.py`, and
  `scripts/precheck_confirmatory_replication.py` build, score, and guard it;
  `tests/test_confirmatory_replication.py` covers the discipline.

### Added — the widened-observation clause arm (decision 112)

- Level 5 re-run with the observation alphabet widened from the derived case
  role alone by one distinguished symbol each for verbs and auxiliaries, the
  two parts of speech that occupy a position while carrying no case, on the
  same document-disjoint split, with the shipped Level 5 untouched and its
  verdict, guard, and gate-audit row bit-identical. Pre-registered before any
  artifact was regenerated; every registered prediction held; the arm passes
  its gate at `BRACKETED_REFERENCE` and its paired privileged-label gain and
  rank interval are resolved away from zero at document level, while its
  classification contrast is not. Recorded as `stage5w_*` at descriptive
  status because registration and outcome share one session; the arm is a
  sizing of the stratificational question, not a rescue of the shipped level.
- `src/ailm/levels/clause.py` builders parameterized on the alphabet with unchanged
  defaults; `data/widened/` holds the artifact (moved out of `vendor/`, which
  is the packaged surface); an `L5w_clause_widened` GNN spec with two
  mutations; `audit_gates.py --widened` writes a separate
  `gate_audit_widened.json` and leaves the canonical export byte-identical.
  Manuscript subsections in Methods, Results (`tbl:stage5w`), and Discussion.

### Fixed — the network-claim re-bind is re-runnable

- Three arms are deliberately live on every canonical network run (the
  repeat-agreement trials and the synthetic and in-context scored calls), and
  their cache keys were run-agnostic, so a re-run on a server that no longer
  reproduced the stored reply stopped with `CacheCorruptionError: divergent
  reply for existing key`. Each run now scopes its own cache namespaces and
  carries its run id in the digested receipt configuration, so a re-run
  appends beside the previous run and a receipt cannot claim another run's
  records. `docs/release_checklist.md` retires the known limit and
  `docs/troubleshooting.md` gains the two refusals that occurred, with the
  audit commands that distinguish a stale store from a moved server.
- The reproduction-documentation guard admits digest-bound data-resident
  generators, so an artifact under `data/` whose sha256 is recorded in
  `docs/provenance/sources.md` is documented the same way a vendored one is.

### Changed — the reader-facing object

- Abstract rewritten to the linguistic and analytical contribution: token-only,
  no citations, the six levels named by what each infers, the gate taxonomy
  stated as the analytical contribution, one sentence on the replication.
- IMRAD restructure: thirteen level and apparatus files consolidated into
  `03_methods.md`, `04_results.md`, and `05_discussion.md`, every section title
  a plain noun phrase, every label preserved (an exact multiset comparison of
  body lines found nothing dropped), and the registries, guards, and path
  citations following the move.
- Title page and cover: author block tightened on both rendering paths, a DOI
  line reserved on the archival Zenodo deposition
  (`publication.doi` in `config.yaml`, `doi` in `CITATION.cff`, and the
  orchestration guard requiring it), the contact email set to the institute
  address, page margins reduced, and the cover redrawn taller with every level's
  hidden state, observation, likelihood, dynamics, and drawn mechanism read
  from the level table at draw time, the generation-order label in its own
  gutter, and verdict badges (`passing`, `non-passing`, `non-confirmatory`)
  mapped from the recorded gate-audit pair rather than typed.
- `docs/release_checklist.md` documents where the Zenodo production token
  lives and the reserve-then-publish route, so the DOI step is not
  rediscovered at the tag.

## [Unreleased] — release-readiness pass: the lexical backoff correction, the Holm layer, and the reviewer findings (2026-09-05)

Decisions 109 and 110 in `docs/decisions.md` carry the record; every quantity
below is cited there as a `claim:` tag and resolved from `data/claims.json`.

### Fixed — the lexical channel's backoff matched the wrong context (decision 110)

- `LexicalMessage.lookup` truncated the joined `position|context` key from the
  right, so a backoff match described a context that did not end at the
  decoded letter and the reported depth was the key string's length. Found by
  the cross-vendor source audit, confirmed by reading the key construction
  against the lookup, corrected to the documented longest-suffix rule with the
  bound carried on the message. Six backoff-semantics tests that fail on the
  old reader and a real-data invariant over every held-out position replace
  the two assertions that could not have caught it.
- Every dependent arm re-recorded through its own recorder: the lexical sweep,
  the channel factorial, the fitted arm, the geometric-pooling arm, the
  proper-loss cells, the statistical aggregate, and the status assignment. The
  channel helps more at every tested weight; the high-weight reversal was an
  artifact; the interior argmax is an ordering the sweep cannot distinguish
  from a plateau; the evidence-site lexical cell is a null rather than a cost;
  the fitted configuration re-selected all four values. The pre-correction
  values are preserved in `scratchpad/review_2026-09-05/lexical/claims.diff.md`.
- Downstream: the lexical evidence-site member left the fixed-threshold
  survivor set, the re-ranked family moved the Stage 6 sentence-sign test past
  its Holm bar, and the lexical high-weight arm is a decided effect rather than
  the project's one bounded-and-equivalent null. The hierarchy section, the
  narrative level, results, limitations, conclusion, and the statistics
  supplement's equivalence subsection are rewritten to the corrected ledger;
  `fig:null_bounds` draws the arm as its accuracy panel's established contrast;
  `fig:multiplicity_forest` and `fig:channel_comparison` state their verdicts
  from the recorded booleans and the recorded argmax. Three new derived tokens
  (`STAT_HOLM_ONLY_CLAUSE`, `STAGE6_SIGN_STEPDOWN_CLAUSE`,
  `STAT_FAMILY_SURVIVOR_LIST_PROSE`) keep those sentences true whatever the
  family ordering does next, and the generator refuses a ledger whose step-down
  verdict contradicts its fixed-threshold verdict.
- Three fail-open edges from the same audit closed: the oracle's letter-bigram
  bar refuses non-lowercase tokens instead of wrapping them through negative
  indices, the empty-population behavior of the letter and phoneme scorers is a
  documented refusal with gate tests, and the Stage 4 context-free baseline
  breaks ties by name as the Stage 2 baseline does.

### Added — the Holm step-down comparison is machine-checked (decision 109)

- `formal/AILM/Holm.lean`: Holm's rank bar is never stricter than Bonferroni's,
  is nondecreasing in the rank, coincides with it at rank one, and on a sorted
  family the step-down walk rejects every member the fixed bar rejects, with a
  strict witness and a sortedness negative control. Rejection-count arithmetic
  only; no familywise-error claim. The formal counts are re-recorded, the
  isomorphism test runs the witness family through the real `holm_step_down`,
  and the negative-control file still fails to compile.

### Changed — the reader-facing spine, from the external-reviewer pass

- The abstract names the six-way gate taxonomy in its own typed vocabulary,
  states how many levels clear restated bars beside how many are qualified
  (`PASSING_VERDICT_COUNT` and its word form are derived, not typed), and no
  longer reads a signed harm as a double negative; its heading names the
  method. The conclusion labels its generalization claim as the working
  hypothesis the limitations already call it. The decisions log is introduced
  once and the private TODO file is no longer cited as an authority. The
  contributions' taxonomy sentence is repaired. The reproducibility section
  closes with availability, licensing, compute, and single-author declarations;
  funding and competing-interest statements are the author's and are queued.
  The manuscript keyword no longer indexes under a term the first page
  disclaims. Four captions describe every mark their figures draw.
- README opens on the honest scope statement with a status block under the
  title, drops the green glyphs on the two qualified levels, clones from the
  public remote, and carries no private filesystem path; the handoff note and
  the orchestrator internals moved to `AGENTS.md` and the quickstart. Every
  agent guide was re-verified against HEAD; four stale symbol references and a
  wrong claim about ledger writers were corrected; the governing hierarchy
  document's bare numbers became claim tags.
- Eleven figure defects found by looking (occluded traces, labels struck by
  their own rules, inconsistent color roles, a dropped marker, an empty axis
  span) are fixed in the generating modules with alt text corrected; the
  visual-audit backlog is queued. `docs/release_checklist.md` states the
  release-day sequence and what blocks it.

### Fixed — hygiene

- The American-spelling guard learns the `programme` stem and stops scanning
  the untracked scratchpad; wheel-resource tests derive their archive names
  from the declared version; the formalism map names the Holm comparison.

## [Unreleased] — adversarial audit and fleet arms (2026-09-02)

Decisions 96 to 108 in `docs/decisions.md` carry the record; every quantity
below is cited there as a `claim:` tag and resolved from `data/claims.json`.

### Fixed — the two claims-status writers agree (2026-09-04)

- `scripts/assign_claim_status.py` now mirrors the wild-cluster recorder's
  declared design: the two `wild_stage4_czech_ci_*` interval labels are
  exploratory (the recorder's `wild_czech_exploratory_suffixes`), the
  verdict-agreement marker stays descriptive, and the two stored statuses were
  re-assigned by the classifier. `test_claims_single_writer.py::test_rerunning
  _the_recorders_is_a_no_op` passes again after the two writers had been
  silently fighting over those two entries.

### Changed — manuscript and statistics prose bound to the ledger (decisions 96 to 99)

- Claim-strength corrections: the calibration gate reads as adopted (decision
  93) with the withheld probes named; the word-state generator limitation
  reports the measured degeneracy instead of a conditional; the online-learning
  arm is described at its `descriptive` status outside the family; the
  prior-only channel is counted as such; the abstract is cut to a third and the
  repeated per-level inventory is replaced by pointers.
- Statistics prose re-derived from recorded values: the alphabetic arms are
  reported as well powered at the reporting level; the cluster-direction
  sentences and the `tbl:cluster` caption state only what the bootstrap can
  resolve; the Stage 5 interval names its own coverage; every table that
  prints an excluded p-value says so in its caption. Each relation now has a
  test with a planted-failure companion.
- The confirmatory replication's retirement (decision 91) is stated in every
  passage that still promised the run, through a literal token whose value is
  the decision entry; the `ud_ewt_test` members remain unscored pending a
  principal decision.
- `data/registration_provenance.json` inventories protocols 18, 25 and 27;
  entry 18 is retitled a prospective in-session specification, and the
  operative rule (registration commit strictly before outcome commit,
  machine-checked against Git) is stated where `confirmatory` is defined.

### Changed — figures draw only what they claim (decisions 100 and 101)

- The hierarchy gutter arrow spans one adjacent pair, one way; verdicts and
  the footnote stay inside their table; the declared Stage 1 gate has one
  code-owned home read by the recorder and both figures; alt text names only
  drawn marks; the calibration panel names its withheld probes; the taxonomy
  legend lists only drawn zones; the Stage 5 paired gain is drawn against its
  zero line; the figure package is inside the mypy gate and the release test
  refuses an exclude.
- Every stratified benchmark bar (prefix windows, alphabetic arm, phone-count
  buckets) carries a Wilson interval on its own denominator, recorded by the
  benchmark recorder under the network-claim binding.

### Added — fleet measurement arms (decisions 102 to 104 and 108)

- Family-grouped G2P sensitivity arm over decision 74's family-disjoint split:
  per-family paired McNemar records, a pooled positive control and a reversal
  negative control (the drafted rotation control was provably vacuous and was
  retired before any write). The registered heterogeneity prediction was
  refuted: the famsplit margin is a majority-null average, not a level-wide
  property, and the phoneme section says so through tokens.
- Part A perplexity on a held-out word split, in closed form, with a
  positive control against the in-sample closed form and a cross-word re-deal
  negative control. The first registered control (within-word shuffling)
  fired and is kept as a measurement: bag-of-phones co-occurrence carries part
  of the bigram's margin. All predictions held; the limitations open item is
  replaced by the measured statement.
- Wild cluster bootstrap (Rademacher weights) recorded as a sensitivity arm
  beside every clustered interval; every wild verdict agrees with the recorded
  cluster-bootstrap verdict. New supplement subsection and `tbl:wild_cluster`.
- A stipulated phone chain beside the stipulated letter chain (decision 108):
  the same three-part mixture over the phoneme level's exact ARPAbet
  inventory, a checksummed vendored artifact, its own cache identity, and
  comparator refusals pinned before any live call (uniform floor, transposed
  counts, English-trained fit). Both stipulated arms are measured in one
  publication unit under one network-claim binding. The registered prediction
  held: the fitted comparator beats every registered probe, and the one-phone
  sensitivity arm drops every probe to the floor. The entry states what the
  arm cannot establish about pretraining exposure. New supplement subsection
  through `BENCH_SYNTHPHONE_*` tokens.

### Changed — gates and refactors (decisions 105 to 107)

- The orphan-claim gate unions three real consumption surfaces (prose
  references, labels read during a real derivation call, figure token stems)
  and fails a family of five or more when more than half of it is dark;
  families measured and set aside are allowlisted with the decision that
  measured them, under a staleness control that fails any entry the ledger no
  longer supports.
- `check_conformance` and `assert_conforms` take a typed
  `LiveConformanceInputs` record returned by `describe_shapes()`; unknown
  fields fail at construction; the legacy keyword form works through a thin
  adapter; verdicts are byte-identical across every shipped spec and mutation.
- `generate_manuscript_variables` is a pure orchestrator over per-section
  `_produce_*` writers into one `variables` map, with closures bound into
  `_ClaimReaders` and typed cross-section records; a frozen pre-refactor
  module diffed to zero differences at every commit, and a per-producer
  non-vacuity test holds the producers to the shape the shadowing scan
  assumes.

### Fixed — hygiene

- The thirty genuinely unused `# noqa` directives across `src/`, `scripts/` and
  `tests/` are retired (2026-09-05). The queued command was `ruff check --select
  RUF100`, which enables only that rule and therefore reports every directive as
  unused; run `ruff check --extend-select RUF100` so the project's own rule set
  stays in force. Lint, format and typing gates unchanged.

- Release-runner skips, test timeouts, a dead generator, stale docs and path
  resolvers from the audit list; the build timestamp derives from the last
  input-touching commit rather than HEAD, so a documentation-only commit no
  longer moves every receipt; `pyproject.toml` is the only copy of the
  coverage floor; recorders retire only the labels their source owns.

## [Unreleased] - inferactively-pymdp 1.0.3 to 1.0.4 (2026-08-29)

### Dependencies

- `pyproject.toml` pin moved `inferactively-pymdp==1.0.3` to `==1.0.4`
  (published on PyPI 2026-08-23; no GitHub release notes existed at upgrade
  time, so behavior was verified against the installed package, not a
  changelog). `uv.lock` re-resolved; `jax`/`jaxlib` remain at 0.4.38 (1.0.4
  declares `jax<0.11.1,>=0.3.4`, so the pymdp-verified JAX line still holds).
- Recorded receipts/claims/benchmarks are version-tied and were re-recorded
  against 1.0.4 (see the upgrade/evidence lane reports for the full deltas;
  pre-bump claims snapshot preserved at `handoff/claims.pre-pymdp-1.0.4.json`).

### Behavior verified against installed 1.0.4 (2026-08-29)

- The `UserWarning: A JAX array is being set as static!` emitted by every
  `Agent()` construction on 1.0.3 is resolved upstream: 1.0.4 raises no
  warning on the documented `Agent(A=..., B=..., C=..., D=...)` construction
  path. Docs (`docs/troubleshooting.md`) and the filter comment in
  `src/ailm/__init__.py` were updated to record this as a dated change.
- `update_empirical_prior` still vmaps over `B` and `compute_expected_state`
  still indexes `B_f[..., u_f]` - the JAX 0.4.38 pin rationale in
  `docs/troubleshooting.md` carries over unchanged.
- `infer_parameters` remains the learning API (`pymdp/agent.py:421`);
  `update_A`/`update_B` exist only under `pymdp.legacy`. Agent construction
  via `num_obs`/`num_states` keyword form is gone in 1.0.4 (Agent is an
  equinox module taking `A`/`B` arrays); AILM constructs from arrays and is
  unaffected.
- Documentation version literals swept to 1.0.4 across `docs/`, root
  `AGENTS.md`, and `docs/provenance/sources.md`; dated 1.0.3 mentions kept
  only as historical records. Manuscript continues to inject the version via
  the `ENV_PYMDP_VERSION` variable - no hardcoded literals.

## [Unreleased] — round-2 documentation lane (2026-08-29)

### Docs — provenance and installation-path accuracy

- `docs/provenance/sources.md` gained the missing entry for
  `vendor/en_ewt-ud-test.conllu` — vendored at the pinned revision through
  `src/ailm/corpus/ud.py` (`ensure_ud_ewt_test_path`) since 2026-08-26 but
  absent from the provenance index. Records URL, sha256
  (`fa024f43…`), byte extent (1,900,101), license, the decision-79/91
  held-back-partition disposition (unburned, unscored), and the carried
  falsifier (553c8d8).
- README Installation now points at the canonical external-drive checkout
  (`/Volumes/external_drive/Git/…`); the former `~/Documents/GitHub/projects`
  path survives only as a compatibility symlink (f3a9952).

Verified alongside: every tracked markdown link resolves (0 broken across
root docs + docs/), decisions TOC anchors 1–91 all resolve, the `--help`
/ read-only verify commands in README/AGENTS/ISA run clean
(`export_web_model.py`, `build_web_ailm.py`, `audit_gates.py`,
`generate_lean_ledger.py --check`), and vendor JSON sha256s all match
`sources.md`.

## [Unreleased] — benchmark re-record, confirmatory retirement, and the Fillmore re-run (2026-08-26)

Consolidated from the 2026-08-26 fleet lanes; the entries below 2026-08-21
predate this work. Receipt numbers are not restated here — run the current
full profile and read `output/reports/release-receipt.json` (per HANDOFF).

### Changed — external benchmark re-recorded under the v2 protocol

- Re-recorded the synthetic + in-context benchmark claims (`bench_ctx` v2,
  189 labels) from a real Ollama run and regenerated manuscript variables
  (ec8ee01, 879cf92, 0dc2038). Replay state: hydration 28/29, all
  MissingClaimErrors cleared.

### Resolved — decision 91: the confirmatory replication is retired by exhaustion

- Upstream cmusphinx/cmudict HEAD is byte-identical to the vendored pin, so the
  registered `external_lexicon` diff population is provably empty; the run
  halted before scoring anything (empty-diff stop condition) and no sentinel
  was written. The four `ud_ewt_test` members remain rescoring-eligible under a
  future successor registration; the partition stays unburned (724ac28, a5fec58).

### Measured and set aside — decision-42 Fillmore subject-selection re-run

- Executed under the registered per-token rule over authenticated UD EWT dev:
  702 scorable sentences, but the decode reach keeps the model path at nine
  predictions (coverage 0.0128), below the 420-pair entry bar —
  `enters_main_line = False`. The derived control behaves as circularity
  (0.9758) and the leftmost-role baseline scores 0.7906 at full coverage.
  Reconciled with the no-orphaned-family gate in
  `docs/design/stage4_case.md` (dd471ac).

### Fixed — release-orchestration and suite hygiene (ailm-src lane)

- Env-scrub helper + sidecar-ran builder probes unblocked the
  release-orchestration suite; recorder assert→raise hardening. Final lane
  confirmation: 5953 passed / 13 residual (all out-of-lane or
  artifact-regeneration) (879cf92, 4b8e814).
- Clear residual American-spelling offenders (b7e5df0); twin REPO_ROOT depth
  fix with byte-identical in-repo artifact (8668f1e, c344844).

### Docs — receipt drift and GPU decision

- Recast the printed 5,700 contrast as historical (5,693 receipt) vs current
  regeneration 5,643 (bfd5b4b); decision 91 index row added (a5fec58); cuda12
  footprint decision recorded — keep single combined lock, decline GPU profile
  (f648173); ledger-rewrite TODO closed (e91e2b8).

## [Unreleased] — the research program the audit asked for, attacked first (2026-08-21)

Every remaining audit-derived proposal went through the queue's adversarial
discipline before a line was built; two of five survived only in modified form,
and the modifications are what shipped. The human-validation arm was declined
by principal decision rather than left open (decision 76).

### Corrected — exact case-source identities and CEREBRUM boundary (decision 82)

- Pinned both Zenodo records and exact deposited PDFs by separate metadata and
  printed identities, filenames, extents, MD5/SHA-256 digests, and physical-page
  evidence locators in `data/reference_sources.json`. This preserves the record/
  title-page version and title/date conflicts instead of silently choosing one.
- Retired the historical Figure-13 component crosswalk: the exact CEREBRUM page
  is caption-only, while Supplement 8.1 defines the source tuple. The replacement
  is explicitly AILM-owned, gives pymdp `C`/`D`/`E` no source-tuple analogue,
  and guards against mapping source precision to pymdp `E`.
- Preserved separate core Genitive and novel Generative model roles despite the
  source's shared `[GEN]` abbreviation. Replaced the generic all-case scalar
  parameter table with source-faithful formulation metadata and an inert,
  fail-closed NOM/ACC-only scalar approximation whose defaults belong to AILM.
- Split the unscaled target-free-energy transition from the distinct
  gamma-scaled expected-free-energy transition. Exact-source snapshot tests and
  planted retired-attribution controls protect every correction above. Earlier
  Stage-0 entries below remain as development history, not current attribution.

### Corrected — Stage 5 is non-passing; its privileged-label gain is unresolved (decision 83)

- Replaced the unsupported *resolved/bounded null* and *negligible gain*
  language with a production-owned sentence-paired percentile bootstrap on the
  existing EWT split. The mean gain is
  claim:stage5_oracle_gain_nats_per_sentence nats per sentence with a
  claim:stage5_oracle_gain_ci_confidence_pct% interval from
  claim:stage5_oracle_gain_sentence_ci_low to
  claim:stage5_oracle_gain_sentence_ci_high. The companion position-weighted
  gain is claim:stage5_oracle_gain_nats_per_position nats, with interval
  [claim:stage5_oracle_gain_position_ci_low,
  claim:stage5_oracle_gain_position_ci_high]. Both span zero; no equivalence
  margin was registered. The supported result is non-passing with unresolved
  gain on this split, not absence or negligibility.
- The real Stage 5 recorder is sole owner of the point estimates, endpoints,
  confidence, claim:stage5_oracle_gain_bootstrap_resamples replicates, and seed
  claim:stage5_oracle_gain_bootstrap_seed. Fixed-seed independent numerical
  re-derivation, planted positive and swapped negative controls, malformed-input
  paths, recorder replay, manuscript-token derivation, and gate/figure refusal
  controls prevent the old overclaim from returning.

### Corrected — Stage 4's fidelity scope and token-level adposition construct (decision 84)

- The heuristic-versus-gold result is no longer called a fidelity bound on the
  full derived population. Its overlap of claim:stage4_heuristic_gold_tokens tokens
  contains claim:stage4_heuristic_gold_adposition_token_tokens token-level
  adposition-route examples. The artifact and canonical recorder now retain the
  full path composition plus head-resolution sufficient counts: among
  claim:stage4_oracle_full_path_adposition_token_tokens ADP-token assignments,
  claim:stage4_adposition_token_head_also_labeled_tokens have a separately
  labeled head and claim:stage4_adposition_token_head_role_disagreements roles
  disagree. Existing labels and gate values are preserved; a head-transfer
  oracle is explicitly a new research population, not a silent bug fix.

### Added — Level 4 scored against real case morphology (decisions 77 + outcome)

- **The margin survives on ground truth.** On UD Czech-PDTC, where case is
  marked morphologically on nominals, the same sequence construction scores
  claim:stage4_czech_accuracy% against a context-free bar of
  claim:stage4_czech_baseline% — a margin of claim:stage4_czech_margin_pts
  points over claim:stage4_czech_test_tokens held-out gold tokens, above even
  the context-free predictor fitted on the test labels themselves
  (claim:stage4_czech_cf_oracle%). Both derangement controls collapse below the
  baseline. The endpoint is named *accuracy* for the first time at this level,
  because the labels are real; the arm is exploratory, outside the frozen
  family. Prepositional Case= (government, not inflection) was pre-registered
  as excluded — decision 63's adposition lesson applied in advance.

### Added — the third EFE degeneracy, derived then measured (decision 78)

- A word-identity hidden state makes the information-gain term nonzero
  (claim:generation_word_state_ig_empty_nats nats at the empty prefix)
  and still cannot make it action-bearing: action-independent transitions make
  both EFE terms constant across actions, so the policy posterior equals the
  habit softmax to machine precision. Recorded as a third control beside the
  deterministic-transition zero and the stochastic ceiling, with a
  general-class property test; the adversarial pass derived this before the
  recorder measured it, which is why the generative word agent was not built.

### Changed — the significance family is honestly a development-population one (decision 79)

- A per-member census found **no** family member with a held-back partition
  inside the vendored artifacts; the manuscript now names the 16-test family
  "development-population inference" wherever it appears, and the confirmatory
  replication is pre-registered instead of faked: UD EWT *test* (never read by
  any script here) for the four EWT members, an external-lexicon rule for the
  eight CMUdict members, frozen endpoints held equal to the family by test.
- The proper-loss axis now covers the factorial: all cells scored as held-out
  log loss off the same posterior rows as their accuracies, sign agreeing with
  the accuracy axis on every cell, weight provenance (test- vs train-selected)
  recorded per cell.

### Fixed — the factorial recorder's removal scope was swallowing the proper-loss labels

- `record_channel_factorial_claims.py` removed every `channel_*` label before
  re-recording its own batch, but the `channel_proper_loss_*` labels belong to
  `record_channel_predictive_loss_claims.py`. The prefix-wide removal deleted
  them wholesale; the loss recorder's replay then re-created them as brand-new
  entries with no prior status to inherit, silently unclassifying all of them —
  caught by the replay no-op test (`test_rerunning_the_recorders_is_a_no_op`)
  on its first full run since the status field shipped. The removal is now
  scoped to the recorder's own labels, and a static ownership guard
  (`test_no_recorders_removal_scope_swallows_another_recorders_labels`) fails
  on any recorder whose `startswith`-scoped removal touches another recorder's
  labels, with a planted-violation non-vacuity check. No recorded value moved.

### Changed — every ledger claim carries an epistemic status (decision 80)

- Schema v2: each claim is confirmatory, exploratory, descriptive, or analytic,
  machine-readably; "confirmatory" is grep-guarded to recorders citing a
  pre-registration decision. Legacy construct mentions were swept from
  reader-visible surfaces under the recorded identifier rule (code ids and
  claim labels keep their legacy strings; history stays history).

---

## [Unreleased] — the external audit, answered in kind (2026-08-20)

A deep-research audit of the manuscript returned a major-revision verdict whose
sharpest findings were conceptual: an untyped gate taxonomy licensing
impossibility claims from exceedable comparators, an unhedged priority claim,
construct names outrunning their constructs, and two mechanism attributions
resting on comparisons that could not carry them. This entry is the response;
what was not built is pre-registered in `TODO.md` and `docs/research_queue.md`
rather than silently dropped.

### Changed — the gate taxonomy is typed, and the Lean layer proves the repair

- **Every ceiling now declares what kind of comparator it is**, and the verdict
  vocabulary splits accordingly: `UNREACHABLE` only from a bound
  (`model_class_bound` / `analytic_bound`), `ABOVE_REFERENCE` when an
  exceedable comparator (`privileged_oracle` / `reference_model`) fails the
  bar, `BRACKETED_BOUND` / `BRACKETED_REFERENCE` on the passing side.
  `classify_gate` fails closed on an untyped measured ceiling. The Stage 4
  declared gate's derived-population verdict retyped from `UNREACHABLE` to
  `ABOVE_REFERENCE` — the one recorded verdict this changes, and the audit's
  showcase case. Machine-checked: `unreachable_requires_bound` and
  `above_reference_requires_exceedable` in `formal/AILM/GateVerdict.lean`,
  with NC10 exhibiting the kind load-bearing on recorded Stage 1 data.
  Decision 75. `evaluate_gate` gained a prospective `min_effect` for gates
  declared from now on. `scripts/audit_gates.py --json` exports the typed
  audit to `output/reports/gate_audit.json`.

### Added — two mechanism controls the audit asked for, pre-registered first

- **L1 context destruction** (decision 73): rotating the trained conditionals
  off their contexts collapses held-out perplexity to
  claim:l1_context_derangement_perplexity against the unigram bar's
  claim:l1_context_unigram_perplexity, and the context-marginalized comparator
  lands within claim:l1_context_marginal_minus_unigram of the bar — so the
  bigram's gain is carried by the context assignment, and the prose claim
  "cannot beat unigram without the previous letter" is retired for the
  comparison claim the control actually supports.
- **Family-disjoint letter-to-phone split** (decision 74): whole four-character
  prefix families assigned to one side only (shared-prefix leakage
  claim:stage2_g2p_famsplit_shared_prefix_pct% against the interleave's 82.5%).
  The margin survives smaller — claim:stage2_g2p_famsplit_margin_pts points at
  exploratory exact-McNemar p = claim:stage2_g2p_famsplit_mcnemar_p — bounding
  the shipped result against morphological-neighbor transfer.

### Changed — claims and constructs sized to their evidence

- The priority claim is hedged ("to the author's knowledge") and backed by a
  documented search protocol in related work; three nearest neighbors joined
  the bibliography and the comparison (deep temporal models 2017, Active
  listening 2021, Donnarumma-Frosolone-Pezzulo 2025).
- Reader-visible construct names now match the constructs on every surface:
  the abstract opens on "a suite of explicit discrete probabilistic language
  components", L4 is a UD-derived syntactic role class at first mention, the
  verification oracle is an "implementation-independent reference decoder"
  with its shared-assumption scope stated, the transformer contrast is a
  guarantee claim rather than a categorical one, and the left-context-lookup
  finding is promoted into the abstract.
- The numeric environment is pinned as recorded claims
  (env_jax_enable_x64 = False, jax/jaxlib/pymdp versions, platform), with a
  refusal guard and a live-interpreter test.

---

## [Unreleased] — the queue, attacked; and Level 4's margin, partitioned (2026-08-18)

### Added — the partition that shows where an admitted margin lives

- **Stage 4's derived margin is a mixture and one oracle path carries it.**
  `oracle_path_strata` in `src/ailm/levels/case.py` reconstructs which of the three
  precedence paths assigned each derived label, and
  `record_stage4_claims.measure_oracle_path_strata` records the gate partitioned by
  it: eighteen labels plus the three stratum sizes. The dependency-relation stratum
  carries claim:stage4_path_deprel_margin_pts points at
  claim:stage4_path_deprel_mcnemar_p; the morphology stratum, where the label is
  real English case marking, is claim:stage4_path_morphology_margin_pts and *not
  shown*; the adposition stratum reverses at
  claim:stage4_path_adposition_margin_pts. The strata sum to
  claim:stage4_derived_test_tokens and pool back to
  claim:stage4_derived_margin_pts exactly, which is what establishes the partition
  is of the scored population. Two refusals guard it: token alignment, and the sum.
  Full reasoning in decision 63.
- **The guard against this was hollow.**
  `test_a_deprel_observation_would_be_near_perfect` promised in its docstring to
  build the circular predictor and show it scoring far above the honest model. It
  asserted `len(set(UD_DEPREL_TO_CASE.values())) >= 5` and built nothing. It now
  builds the predictor, which scores 72.81% against the honest model's
  claim:stage4_derived_agreement%, and asserts exactness on the stratum the map
  generates.

### Added — three nulls given intervals

- Two subsample arms were computing their discordant counts and discarding them
  into `binomtest`, so they reported a margin and a p-value with no interval. The
  counts are recorded (decision 64). Deliberately *not* recorded for the `full`
  arm, whose counts already have a single writer in `record_stage2_claims.py`.
- The clause-to-word channel recorded improved/degraded counts, which bound a
  *share*, while its reported effect is a *mean* over per-token log-losses.
  claim:clause_word_best_delta_sd supplies the dispersion a mean needs
  (decision 65).
- Both bounds are derived at render time from the counts. An interval stored
  beside its own inputs is a second copy of one fact, free to drift.

### Changed — one coverage level, and a label that no longer carries it

- `SHARE_CONFIDENCE` is now `DIFFERENCE_CONFIDENCE` (decision 66). Four intervals
  narrowed, all four reproduce exactly from `clopper_pearson_interval` at the new
  level, and all four excluded 0.5 before and after, so no verdict moved.
- `stat_*_ci99_*` became `stat_*_share_ci_*`. `summarize_paired_effect` also
  carried a bare `confidence: float = 0.99` that never referenced the constant.
- Scope corrected before acting: `stage5_auc_ci99_*` matches on name only. It is a
  bootstrap percentile interval with its own declared level and was left alone.

### Fixed — two audits that reported coverage they did not have

- `_VERIFY_STRATA` in `record_external_benchmark_claims.py` printed a note and
  continued when a declared stratum contributed zero entries, so a store holding
  one arm drew its whole sample from that arm and reported a pass. Measured on the
  shipped store: 6,436 entries, every one without logprobs, so the calibration arm
  its own comment calls "the arm most sensitive to numerical drift" had never been
  verified. It now fails closed.
- The `COINCIDENTAL_MATCHES` keys in `docs_hydration.py` are absolute line numbers,
  and appending to `decisions.md` shifted five of them twice. The literal check
  caught both, which is what it exists for; the failure mode it prevents is now
  documented beside the table.

### Added — the adversarial pass over the research queue

- All 33 items in `docs/research_queue.md` were attacked, one skeptic per
  direction, each defaulting to refuted. Of 48 judged items: 14 already landed, 11
  refuted, 19 survive modified, 4 survive as written. Verdicts and reasons are
  appended to that file.
- The queue's largest gap is that nothing in it addresses equal opportunity to fit
  the task, which this project names as the biggest threat to its external
  comparison.
---

## [Unreleased] — the expected free energy, measured in both directions (2026-08-18)

### Added — the decomposition result, promoted out of a docstring and a code path

- **The information-gain term is measured, not asserted.** That a deterministic
  controllable transition makes the expected free energy's information-gain term
  identically zero, so policy selection reduces to a preference-weighted softmax,
  had been written in the docstring of `efe_policy_posterior` and appeared nowhere
  in the manuscript. `expected_free_energy_terms` in `src/ailm/generation.py` now
  returns the decomposition, and `scripts/record_generation_claims.py` records it
  with two controls: claim:generation_shipped_information_gain_nats nats shipped,
  claim:generation_stochastic_control_information_gain_nats nats under a stochastic
  transition, and claim:generation_flat_likelihood_control_information_gain_nats
  nats under a flat likelihood. The two sufficient conditions for the collapse are
  therefore confirmed independently of each other. The recorder refuses to write if
  the shipped value is nonzero.
- **The paper was proposing as future work an experiment it had already run.**
  `08_generation.md` said a stochastic-transition construction "would put the
  information-gain term back and is the obvious next experiment". Budgeted
  attention at the letter-to-phone level is that construction: `_adaptive_schedule`
  in `scripts/probe_budgeted_attention.py` scores each candidate position by the
  posterior entropy it is expected to leave, which is a pure information-gain
  criterion with no preference term at all, and it was measured and recorded as a
  null (claim:budget_adaptivity_loss_pts points below the exhaustively optimal
  fixed schedule at a claim:budget_fraction_pct% budget). The two sections never
  referenced each other.

  Both regimes are now stated as one result in `01_introduction.md` Contributions,
  `08_generation.md`, and two mutually-referencing sections of `13_limitations.md`:
  each component of the expected free energy has been measured alone, in the only
  two places this architecture admits them. Where the pragmatic term acts alone,
  generation works. Where the information-gain term acts alone, it is beaten by a
  policy that performs no inference at decode time.

### Fixed — an operating point that was estimator noise

- **The generation optimum was wrong and the error was systematic.** The recorded
  best precision came from a Monte Carlo sweep whose estimator is biased *upward*:
  the sampled draws cannot cover the vocabulary, so total variation is
  overstated everywhere and more sampling would not have corrected it.
  `induced_word_distribution` computes the induced distribution exactly, since the
  generator's state is the emitted prefix alone and each word's probability is the
  product of action posteriors along its trie path. Over
  claim:generation_exact_grid_points grid points the optimum moves from
  claim:generation_sampled_best_gamma to claim:generation_best_gamma, and the
  distance from claim:generation_sampled_best_tv to claim:generation_best_tv. The
  sampled sweep is retained as a convergence check; its largest disagreement with
  the exact computation is claim:generation_mc_exact_max_abs_gap.

### Fixed — three stale or drifting descriptions

- **A gate whose verdict was predetermined is now named as such.**
  `gate_is_baseline_relative` in `src/ailm/gates.py` reports when a threshold *is*
  its baseline, which under the strict comparison makes VACUOUS unreachable and the
  verdict decided before measurement. It is deliberately not a fifth `GateVerdict`
  case: the four are fixed by theorems in `formal/AILM/GateVerdict.lean`, and this
  is a property of how a gate was constructed rather than of how it classifies.
  `scripts/audit_gates.py` prints the note beside every affected verdict.
- **Two senses of "channel" were being counted against each other.**
  `docs/hierarchy.md` says five inter-level channels were built and measured;
  `scripts/generate_cover.py` labels the cover "six levels, two channels" and draws
  the two that carry a preference vector. Both are correct under their own
  definition. The distinction is now stated once, in the authoritative file.
- **The sparse-path queue entry was itself the drift.** `TODO.md` and
  `docs/research_queue.md` asserted that the claim "pymdp has no sparse path"
  remained uncorrected in four locations. Every one already carried the narrow,
  accurate form ("reachable through pymdp's public `Agent` API"). Recorded as
  closed, because a stale queue routes the next reader to a fixed bug.

### Added — the end-to-end cascade, scoped and blocked

- An L6 to L1 top-down generation path has exactly one missing link, L4 to L3,
  which needs `P(word | case role)`. That map is **not fittable from any shipped
  artifact**: `vendor/ud_ewt_case_counts.json` records its observation as "UPOS tag
  only; deprel and feats[Case] withheld" and carries no word form, and
  `vendor/brown_clause_word_counts.json` is typed by sentence-final punctuation
  rather than by case role. The withholding is very likely deliberate, so
  re-vendoring the treebank with word forms attached to case roles is the shape of
  the case-oracle trap this project has sprung before. Recorded in `TODO.md` under
  Major with the pre-registration requirement that it be a structural rather than a
  performance result.

---

## [Unreleased] — the channel factorial: the missing half of a 2x2 (2026-08-12)

### Added — the two cells the paper said were not built

- `09_hierarchy.md` stated in its own text that "the two cells that would separate
  source from insertion point are not built", so decisions 18-19 — the
  novel-information criterion — rested on a diagonal. Decision 46 pre-registers
  the design and `src/ailm/channels.py` executes it, dispatching into the shipped
  primitives and reimplementing no filter. The executor is gated before use by
  reproducing both existing cells' per-position **hit vectors** exactly over all
  42,820 positions, not their accuracies: two decoders can agree to four decimals
  on an aggregate while disagreeing on thousands of positions in both directions.

| cell | weight | accuracy | margin | p |
|---|---:|---:|---:|---:|
| letter@evidence | 0.1 | 70.5371 | −0.264 | 1.5e-06 |
| letter@predictive | 0.1 | 70.9038 | +0.103 | 0.17 |
| lexicon@evidence | 0.1 | 70.6305 | −0.171 | 4.4e-05 |
| lexicon@predictive | 0.3 | 72.8071 | +2.006 | 5.6e-72 |

- **Both evidence-site cells significantly harm.** The lexical message buys two
  points at the predictive site and costs 0.17 at the evidence site. Same message,
  same receiver, same weight grid, opposite sign, and only the insertion point
  differs. The mechanism is legible afterwards and was not predicted in advance:
  the evidence site holds the model's only hard information about the current
  position, so softening it trades certainty for a prior however good that prior.
- Decisions 18-19 are qualified rather than withdrawn. They read the diagonal as
  evidence about message *content*; the factorial shows the insertion point
  carries at least as much, and that their correct prediction about the L3-to-L2B
  pair would have failed at the other site with the same message.

### Fixed — three defects found while integrating it

- **The recorder's weight selection.** It maximized over a grid containing zero,
  and zero weight *is* the anchor, so a harmful cell reported the anchor with 0/0
  discordant pairs and p=1. Two significant harms would have been published as
  nulls. The shipped coupling recorder had always selected over positive weights;
  the factorial now does too.
- **The family arithmetic needed judgement, not addition.** Two of the four cells
  are numerically identical to members already declared, because they *are* the
  shipped experiments the executor was gated against. Counting them again would
  tighten the threshold on a duplicate. They became documented exclusions and only
  the two new cells joined: family 11 to 13, corrected threshold 7.7e-04,
  survivors 5 to 6. The new survivor is a **harm** clearing familywise correction.
- **A figure key named for a count.** `family_of_9_alpha` was the collapsed family
  when it had nine members. The figure followed the growth to eleven correctly and
  only its test hardcoded the old number, so the failure pointed the wrong way.
  Renamed to `collapsed_family_alpha` and the test now derives from the
  declaration. Fourth instance this session of a name going stale while the value
  stayed right.

### Added — the alphabet transport, registered before measurement

- Implementing the off-diagonal cells surfaced a design choice neither diagonal
  cell needs: the evidence site consumes a distribution over letters and the
  predictive site one over phones, so both new cells must cross alphabets. Phone
  to letter is clean; letter to phone yields a *likelihood* summing to about 1.87,
  and normalizing it is what `coupling.py` says "would turn the evidence into a
  prior and silently change the model". Decision 46 was amended before any cell
  was measured to register the Bayes inversion under the model's own phone prior,
  with the normalized likelihood named as the sensitivity check the arm owes.

## [Unreleased] — red team, and the four items that blocked publication (2026-08-12)

### Fixed — three measurements that could not fail, and one self-contradiction

- **The determinism probe was tautological.** `repeat_agreement` issued the same
  request twice through the memoized path, so the second call was a replay and
  agreement was analytically 1.0 whenever the cache was on, while the supplement
  cited exactly those values as evidence the server reproduces its replies.
  Verified by counting server calls: two identical requests produced one. The
  probe now calls the uncached poster on both legs, and three guards hold it
  there, including one that reads the source because the behavior can be
  satisfied by a cache that happens to miss.
- **The scaling ladder's stated mechanism was false.** The prose said the
  comparator is "re-fit at every rung"; it is estimated once and re-scored, and
  its matrices are bit-identical across rungs. Its headline was also a
  denominator artifact: per-emission gain averages over each rung's surviving
  positions, and a smaller lexicon survives on an easier subset, so the "nearly
  flat, non-monotone" gain rises strictly by more than seventy percent once every
  rung shares the top rung's denominator — the one-denominator rule the coverage
  axis already enforced. Both series are recorded and reported. The old falsifier
  (the curves must not cross) was unfirable on nested Zipfian prefixes.
- **A resolved contrast was called a bound.** `10_verification.md` listed both
  context-window arms among results that "sit below" the equivalence minimum and
  are "a bound and nothing more", while their pair counts are above it and both
  contrasts clear the corrected threshold. The figure had already refused to draw
  them; the verification section had not been told.
- **A registered prediction was refuted and booked as confirmed.** Entry 36
  predicted these nulls would become bounded, "refuted if either clears the
  corrected threshold". Three of four cleared. The refutation is now recorded
  against entry 36 itself rather than in a later entry that framed it favorably.

### Added — the finding the audit itself surfaced

- The cache audit sampled five records under a fixed seed, forever the same five,
  all letter-arm and none carrying logprobs — so calibration records had a
  structurally zero chance of verification. Stratified and re-drawn with the
  store, it immediately failed. The cause is a real result: **greedy decoding
  reproduces the answer exactly and the distribution only approximately.** Over
  24 re-queried positions the answer agreed on all of them while logprobs drifted
  up to 0.040 in log space, 0.0017 in reconstructed probability, and two
  candidates entered or left the truncated top-*k*. Every accuracy number is
  exactly reproducible; every calibration number now carries that drift as a
  stated floor on its precision. Recorded by `record_logprob_drift_claims.py`.

### Changed — the four items that blocked publication

- **The census declares its inferential target.** The letter arm has sampling
  fraction one, so nothing there is an inference about unscored positions. The
  intervals are randomization statements under the conditional McNemar null,
  which is well defined on a complete enumeration; Wilson intervals are
  descriptive precision, not coverage over a superpopulation never defined. The
  sampling language is removed from the arm that no longer samples.
- **Two multiplicity families, argued rather than harmonised** (decision 45), at
  0.01 and 0.05, with the reason for each and two rules: no test belongs to both,
  and no benchmark result may be promoted into an AILM claim without re-judging.
- **Decision 43 corrected.** It had rewritten entry 38's "removed entirely, not
  demoted to a caveat" into "a candidate for archiving" — the caveat-demotion
  entry 38 forbade by name, smuggled onto a statistical correction's
  justification, and the only change that altered a real outcome. Entry 38's rule
  is restored with one exception argued on its merits and checked mechanically.
- **The abstract states that Stage 6 is rejected** by the repository-wide
  correction, not merely qualified within its stage.

### Corrected — scholarship in the new related-work material

- "Twenty years earlier" understated the gap to Reich by four decades. The claim
  that the roles without English morphology are Fillmore deep cases "in exactly
  that sense" holds for three of them; ablative is not in his inventory and
  vocative cannot be a deep case by his definition. A direct quotation from an
  unpaginated technical report became paraphrase, matching this bibliography's
  own rule against locators it cannot supply. The channel roster in that section
  contradicted every other section and now matches.

## [Unreleased] — the protocol v4 census, landed (2026-08-12)

### Fixed — a sentinel reached the ledger, twice

- `LETTER_CENSUS` is `-1`, a flag meaning "score the whole pool". The recorder
  wrote the *requested* value where the *scored* count belonged, so the ledger
  said the letter arm scored −1 positions and the manuscript would have printed
  it. The second leak was worse for being invisible: `-1 + 400 + 400 = 799` is a
  plausible denominator, so it produced no absurd number, only every
  invalid-answer rate inflated by the whole census. The measurement now reports
  its own resolved count and the recorder records that.
- Guarded over the class rather than the two labels. The first version asserted
  no claim may be negative, which is broad enough to catch the sentinel by its
  sign and also deletes signed deltas, lower interval bounds, and the negative
  Stage 6 ICC that is itself a finding. Restated over sentinels discovered from
  the module, so it catches the next one and names none.
- Re-recording after the fix took 22 seconds: 50,314 calls replayed, zero
  measured.

### Changed — what the full pool resolved

- Letter arm at 7,952 positions rather than 400. Paired p-values fall to 1.3e-225
  and 4.2e-91; recorded power rises to 0.999 and 0.991.
- Three of the four sensitivity contrasts now clear the corrected threshold: both
  context-window contrasts and the Gemma 3 4B prompt-format contrast. The Qwen
  2.5 3B prompt-format contrast remains a null, and at 335 discordant pairs
  against a minimum of 420 it is an *unresolvable* one, labeled as such.
- `fig:null_bounds` refused to draw rather than relabel a resolved contrast as a
  bounded null. The two context-window rows are removed from it, with the reason
  recorded where they were.
- The per-model calibration-rule divergence is recorded, and it disagrees with
  the captured-mass proxy on exactly one probe. Every signed divergence is
  negative, so the shipped min-bound rule reports a lower error than the
  observed-only rule in every case rather than in some — a one-directional bias,
  not scatter. The guard that asserted this quantity's *absence* is replaced by
  one asserting what it says; the old guard scanned a namespace the new labels
  do not use, so it would have gone on passing.
- Decision 43 applies entry 38's informativeness criterion and corrects it in the
  applying: the equivalence minimum is what a *null* needs, so applying it to
  arms that observed large effects would have archived the two most decisively
  significant arms in the section. No arm is archived.
- The word level's scaling ladder is recorded and drawn at eight rungs
  (`fig:scaling_ladder`). The gain is nearly flat and non-monotone across a
  lexicon growing by a factor of 76, which corrects a sentence claiming the
  margin narrows at scale.

## [Unreleased] — stratificational lineage, Fillmore, and a bibliography gate (2026-08-12)

### Added — the intellectual ancestry the architecture actually has

- `14_related_work.md` gains a section on stratificational linguistics. AILM's
  levels and Lamb's strata are the same idea, and naming the resemblance forces
  the disanalogy into the open: Lamb's realization is a live bidirectional path
  through a network, while AILM's decoders pass no beliefs and the five measured
  channels are the first attempts at those relations. Two of five help, one
  harms, two are near-nulls, which is a poor advertisement for assuming
  couplings rather than measuring them.
- Reich's finiteness thesis is cited for what it is: the reason not to treat "it
  is only finite-state" as a settled objection, with an explicit statement that
  this paper's results neither support nor refute it.
- Reich's node-count metric is the twenty-year-older form of this paper's
  gate-audit argument — a symbol count measures the notation rather than the
  object — and is now cited where the complexity accounting and the gate
  taxonomy make the same move.
- `07_case_level.md` grounds its deep-case framing in Fillmore. That most of the
  inventory has no English morphology is a property of the construct, not a
  defect of the corpus, and Fillmore is who established that.
- Nine bibliography entries, with the conflicts the sources disagree on recorded
  in the non-rendering comment block rather than silently resolved: the
  1967-preprint/1968-chapter split for Fillmore, the disputed issue number for
  Lamb's *Prolegomena* (omitted rather than picked), and one claim left uncited
  because its only reading was two removes from a primary source.

### Added — the guard nothing had, in both directions

- `tests/test_bibliography.py`: every citation must have an entry and every entry
  must be cited. A cited key with no entry renders as a bare marker in the PDF;
  an uncited entry outlives whatever it was added for, and one did. Includes a
  guard that the cross-reference exclusion cannot hide a real key, and a check
  that no entry carries an empty field, since the file's policy is to omit an
  unconfirmed field so its absence stays visible.

### Added — Fillmore's subject hierarchy, registered rather than reported

- `fillmore_subject` and `subject_prediction_accuracy` with thirteen tests. The
  target is `nsubj`, which L4 is forbidden to observe, so this is the level's
  first external validity check. The derived path is kept as a circularity
  control expected to be near-perfect, because `UD_DEPREL_TO_CASE` maps `nsubj`
  to NOM directly. The first scoring rule left nine scorable sentences, which is
  no sample; decision 42 registers the corrected rule before the re-run and
  keeps the number out of the manuscript.

## [Unreleased] — the word-level scaling ladder, and two gates that were satisfiable by a subset (2026-08-12)

### Added — coverage and perplexity across eight nested vocabulary sizes

- `ailm.levels.word.scaling_rungs` sweeps frequency-ranked prefixes of the full
  Brown lexicon, reporting coverage, word perplexity, the letter-bigram bar and
  the scored positions at each, in about 9 s. Coverage climbs 66.8% to 96.7%
  while perplexity worsens 3.26 to 3.89: two endpoints can be read as either
  story, and the rungs make the trade the result. `record_stage3_claims.py`
  emits `stage3_scale_*` per rung; the write waits on the census.
- Ten tests, including the positive control that a truncated lexicon without the
  evaluation restriction raises, and a check that coverage keeps one denominator
  — recomputing it against each rung's survivors would report 100% everywhere.

### Fixed — non-vacuity checks that a subset satisfied

- `test_caption_standard.py` asserted the scanner found "at least fifteen"
  captions. A floor is met by any subset, so the scanner could silently stop
  parsing half the figures and stay green — the same shape as the incident where
  a prefix heuristic swept five of nine figures and read as complete. The check
  now derives the population: every `fig:` anchor must have a parsed caption and
  the two sets must be equal, with the anchor pattern deliberately different from
  the caption pattern so the comparison cannot agree with itself.
- Added a gate requiring every referenced PNG to exist. The caption gate reads
  Markdown and the layout gate reads PNGs, so a figure present in one and absent
  from the other was checked by neither, and extracted text does not reveal a
  broken image.

## [Unreleased] — conformance sees the inference configuration (2026-08-12)

### Fixed — a correct preference vector that nothing was using would have passed

- `describe_shapes` now carries `use_utility`, `use_states_info_gain`,
  `use_param_info_gain`, `action_selection`, `policy_len`, the enumerated policy
  count and the agent's own `num_controls`, and `_check_inference_configuration`
  makes them load-bearing for any spec declaring `num_controls > 1`. Decision 41
  records the finding: an agent can carry a correct non-flat `C`, a correct `E`
  and a controllable `B` and still not minimize expected free energy over them,
  because the terms entering the policy score are flags rather than slots. No
  shape or entry changes, conformance passes, and the claim about generation is
  false.
- The checks deliberately do not apply to uncontrolled levels. Every
  comprehension level runs with a flat `C` and one action by design, which is
  what the paper says about them.
- Four named mutants, plus a guard-on-the-guard asserting `describe_shapes` still
  reports every field the mutants flip.

## [Unreleased] — long runs resume per call (2026-08-11)

### Added — a content-addressed store for model calls

- `src/ailm/call_cache.py` memoizes `benchmarks._post_generate`, the one function
  every model call passes through. Keyed on payload, endpoint, protocol version
  and the model's content **digest** from `/api/show`, never its mutable tag.
  Resume stops being a code path: rerunning the identical command replays what is
  on disk and measures only what is not. Decision 40 records why the previous
  per-model sidecar could not help, and the general form of that defect.
- An unresolvable digest yields no key, and a keyless call is measured but not
  stored, so unknown model identity degrades to no caching rather than to caching
  under a guess. The omission is counted and appears in the receipt.
- The store is append-only with one fsynced line per record, so a kill costs the
  last call and nothing earlier. A torn final line is skipped on load.
- `--verify-cache N` replays recorded payloads against the live server and exits
  non-zero on any mismatch, which is what keeps "reproducible" meaning the model
  agreed rather than that the disk did. Wired into the full release profile and
  run as a recorder precondition; deliberately not gated in GitHub CI, which has
  no model server.
- Cache provenance is recorded as ledger claims and cited in the manuscript,
  because a run that replayed everything and a run that measured everything
  otherwise emit byte-identical claims.
- Validated live against `gemma3:4b` over 100 real calls: cold 100 measured in
  72.8 s; replayed in a new process 100/0 in 0.025 s with an identical answer
  hash; SIGKILL at 28 s then resume gave 28 replayed and 72 measured with that
  same hash; a changed seed reused nothing; 15 of 15 replayed calls matched the
  server byte for byte. Both guards carry measured positive controls: corrupting
  the store fails the audit, breaking the shared seam fails the seam test.

## [Unreleased] — the external probe registry drops to two (2026-08-11)

### Changed — one probe retired, and the guards restated over the registry

- The external reference frame serves `gemma3:4b` and `qwen2.5:3b`. The third
  probe is retired from the protocol, the manuscript, the figures, the
  bibliography and the cross-reference registry. Decision 39 records why the
  retirement is not cosmetic: the declared correction family is six contrasts on
  each probe, so its size — and every Bonferroni threshold the supplement reads
  p-values against — scales with the probe count. The captured-mass gate's reject
  side is now empty, and [@sec:external_benchmarks] says the gate no longer
  discriminates in this run rather than that it passed.
- Three guards enforced the retirement by forbidding the retired model's *name*.
  Each therefore reprinted the string it existed to delete, and each would pass
  on the day a different unregistered probe appeared. All three now state the
  rule over the served registry: exact-set equality on the model map, a
  supplement guard that rejects any model-name-shaped token the registry does not
  serve, and count wordings generated from `len(PROBES)`.
- Prose carrying the retired count was restated: the derived-power block's "nine
  figures" is six (three arms on two probes), the reproducibility path pulls both
  probe models rather than three, and the calibration-floor figure title no longer
  fixes a probe count.
- Restating closure over the recorders found a live defect the ledger-side scan
  could not see: two bench p-values (`bench_letter_az_paired_p_qwen25`,
  `bench_letter_prompt_paired_p_qwen25`) were in neither the corrected family nor
  the documented exclusions, so nothing corrected them. Their counterparts on the
  other probe were documented. Both rows are now recorded, and a new guard
  generates the expected label set from each recorder's own probe list and family
  templates, so a missing row fails before the value is measured rather than after.
- `handoff/claims.*.json` deliberately still carry the retired labels. They are
  the pre-run snapshots the census diffs against, and rewriting provenance to
  match the present would destroy the evidence that the transition happened.

## [Unreleased] — decision 29 and figure/documentation review (2026-08-07)

### Added — latent-state G2P emission-learning experiment (2026-08-07)

- Decision 29 was measured without changing its pre-registration. Real pymdp
  `infer_parameters(..., learn_A=True)` updated the grapheme-to-phoneme emission
  while phone states remained unobserved; a same-lane count-based emission was
  the fixed comparator, and an observed-state control recovered that comparator
  exactly. The result was **NO IMPROVEMENT / QUALIFIED**: the latent-state lane
  moved the emission but did not improve held-out decoding. Claims are recorded
  by `scripts/record_g2p_learning_claims.py` and the outcome is appended to
  decision 29 in `docs/decisions.md`.
- Added the mock-free external benchmark harness, a fixed-seed same-position
  Gemma 3 4B reference receipt, the ledger-backed comparison figure, and
  manuscript supplement section 98.
- The final full suite passed **2006 tests**, with 22 justified skips and branch
  coverage 95.27% (2026-08-08).

## [Unreleased] — pre-registrations 27 & 28 measured (2026-08-06)

### Added — the two remaining pre-registered experiments, each gated and committed

- **Entry 27 — clause → word channel (L5→L3) measured.** Built the channel on
  a new vendored bridge corpus (`vendor/brown_clause_word_counts.json`:
  Brown sentences typed by sentence-final punctuation, restricted to the
  Stage 3 vocabulary) so `P(word | type)` can be estimated on train and
  applied to held-out tokens. The pre-registered prediction — **harm with a
  monotone decline** — was **not confirmed**: on the declared axis (held-out
  perplexity of the word prior) the message slightly improves the predictive
  (anchor `clause_word_anchor_ppl` → best `clause_word_best_ppl` at
  alpha `clause_word_best_alpha`, delta `clause_word_best_delta_ppl`) with an
  interior optimum, while the per-token paired sign test runs the other way
  (`clause_word_best_degraded_tokens` vs `clause_word_best_improved_tokens`)
  and the entropy-preserving scrambled control is far worse
  (`clause_word_scrambled_ppl_at_best`). A near-null that qualifies the first
  clause of the novel-information criterion per its own withdrawal condition;
  the headline p-value is in `EXCLUDED_FROM_FAMILY` (refuted-direction
  diagnostic, not a family survivor). New: `src/ailm/levels/clause_channel.py`,
  `scripts/generate_clause_word_counts.py`, `scripts/record_clause_channel_claims.py`,
  `tests/test_clause_word_channel.py`.
- **Entry 28 — online learning on non-stationary text CONFIRMED.** Built the
  W2 increment on a new vendored genre-drift artifact
  (`vendor/brown_online_drift.json`: government then romance letter streams,
  split learn/hold by sentence parity). The online agent that keeps pymdp
  `infer_parameters` learning across the drift beats the frozen control on
  post-drift held-out text — P1 drift real (`online_max_drift`), P2 frozen
  must lose (`online_ppl_online_post` < `online_ppl_frozen_post`, gap
  `online_frozen_vs_post_gap_ppl`), P3 re-fit (`online_ppl_refit_post`) and
  lr=1 identity all hold (`online_confirmed`). New:
  `src/ailm/levels/online_learning.py`,
  `scripts/generate_online_genre_counts.py`,
  `scripts/record_online_learning_claims.py`, `tests/test_online_learning.py`.
- Both outcomes recorded in `docs/decisions.md` (entry 27 outcome block,
  entry 28 CONFIRMED block + TOC), the manuscript sections
  (`docs/manuscript/09_hierarchy.md` five-channel framing + clause→word outcome,
  `docs/manuscript/10_verification.md` online increment), HANDOFF and TODO queues,
  `REQUIRED_CLAIMS`, the single-writer `CANONICAL_LABELS`, and
  `docs/provenance/sources.md` (both new artifacts checksummed). 179 new
  tests (1782 → 1961) keep the full suite green at branch coverage 95.15%
  (verified from the full run at the commit).

## [Unreleased] — Evidence-first review findings F1-F8 + render (2026-08-05)

### Fixed — `REVIEW_report_2026-08-05` findings F1–F8

- **F1** — decision entry 19's alpha-sweep table is now fully claim-backed:
  `scripts/record_lexical_channel_claims.py` writes per-alpha McNemar b/c/p
  labels (`lexical_sweep_alpha_*_mcnemar_b/c/p`) for every grid weight plus
  `lexical_high_alpha_accuracy`; the single-writer producer map and
  `test_lexical_claims_verified` bind them; every literal cell in the table is
  now a `claim:<label>` (no bare measured digits remain). The per-alpha
  p-values are documented as within-comparison exclusions in
  `tests/test_statistics.py`'s `EXCLUDED_FROM_FAMILY` — they are the same
  L3→L2B comparison as family member `lexical_best_mcnemar_p` at different
  weights, not new hypotheses.
- **F2** — coverage-threshold doc drift closed: every `90%` /
  `cov-fail-under=90` reference across `AGENTS.md`, `README.md`, `docs/*`,
  `src/AGENTS.md`, `tests/AGENTS.md` reconciled to the `fail_under = 95`
  literal in `pyproject.toml`.
- **F3** — README architecture paragraph rewritten to the measured
  four-channel state (decisions 9, 17, 19, 26): generation is built and gated,
  not unbuilt.
- **F4** — decision entry 4 closed-debt state: subsample comparison rows are
  `ledger_backed: True` (re-recorded by `scripts/record_subsample_claims.py`,
  run_id `subsample_bias`), the `subsample_*_margin_pts` labels exist in the
  ledger, and agreement is enforced at draw time.
- **F5** — recorder-replay acceptance test kept in the default suite and
  documented as the deliberate exception to the 60-second per-test cap in
  `docs/testing_philosophy.md`.
- **F6** — TODO.md release receipt updated to the measured state
  (1782 passed / 22 skipped / 0 failed, branch coverage 95.11%).
- **F7** — TODO.md legacy-caption allowlist count corrected 15 → 14
  (matches `tests/test_caption_standard.py` `LEGACY_CAPTIONS`).
- **F8** — `scripts/audit_gates.py` now prints a clarifying note when a
  stage's measured smoothing ceiling sits below the shipped model (Stage 4
  derived: 48.1295 < 49.1367), so "ceiling" is not misread as an upper bound.
  Informational only; verdicts and exit code unchanged.

### Changed — render

- Hyperlinks are red (`allcolors=red`), not blue, in `docs/manuscript/preamble.md`.
- A scientific cover image (example sentence + hierarchical topology, drawn in
  the manuscript's own palette by the deterministic
  `scripts/generate_cover.py`) is wired to the title page via
  `paper.cover.image` in `docs/manuscript/config.yaml`.
- Re-render: 77 pages, zero unresolved tokens; verified red links on the title
  page and in the table of contents.



### Added — `formal/`, a Lean 4 proof tree with no dependencies

- The parts of the argument that are actually mathematical are now proved rather
  than asserted: the gate taxonomy (`eq:gate_satisfies`, `eq:gate_verdict`) in
  `AILM/GateVerdict.lean`, the pre-registered sweep-shape classes of
  `sec:hierarchy` in `AILM/SweepShape.lean`, the correction half of
  `eq:filtering` in `AILM/Filtering.lean`, and the multiplicity arithmetic in
  `AILM/Bonferroni.lean`. Toolchain pinned in `formal/lean-toolchain`; the
  lakefile declares **no** `[[require]]`, so there is no Mathlib, no version
  resolution, and `lake build` is offline by construction — everything is proved
  against plain Lean 4 core (`List`, `Rat`, `Decidable`).
- Totals from a real build, recorded rather than typed:
  claim:formal_theorem_count named theorems, claim:formal_example_count witness
  examples, claim:formal_axiom_report_verdicts verdicts in
  `formal/axiom_report.txt`, claim:formal_native_decide_uses uses of
  `native_decide`, and claim:formal_axioms_beyond_core axiom beyond Lean's core
  three (`Classical.choice` — Lean core's own `Rat.add_comm` reports it, so a
  stricter policy would be unsatisfiable against core `Rat` rather than stricter
  about anything proved here). `sorryAx` and `Lean.ofReduceBool` appear nowhere;
  claim:formal_sorry_free is the recorded verdict.

### Added — the ledger bridge, so the proofs cannot drift from the measurements

- `scripts/generate_lean_ledger.py` is the **only** writer of
  `formal/AILM/LedgerData.lean`, the one module holding measured values; it
  renders recorded values as exact `Rat` literals (`parse_float=str` → `Fraction`
  → reduced `lit n d`, no floats and no tolerance anywhere). `AILM/Instances.lean`
  proves claim:formal_instance_theorem_count theorems about those literals — the
  lexical optimum really is interior, the Stage 4 gold population really supports
  no sound threshold at any bar, the declared 60% really does fail in two opposite
  directions on the two recorded populations.
- `tests/test_formal_isomorphism.py` binds it: byte-identical
  regeneration catches both a hand edit and a stale file, and a second,
  independently written conversion re-derives every emitted literal from
  `data/claims.json` so byte identity cannot pass on a consistently-wrong
  generator. Lean-dependent checks skip cleanly when `lake` is absent, naming the
  pinned toolchain; the ledger-binding checks need no toolchain and always run.
- `scripts/record_formal_claims.py` writes the `formal_*` labels from a real
  build and refuses to record unless the axiom report's entries and the
  declarations in `AILM/*.lean` are the same set, checked both ways — a report
  covering part of the tree would otherwise report clean for the rest.

### Added — the doctrine, each rule earned by a shipped defect

- Every conditional theorem carries a **satisfiability witness** (an unsatisfiable
  hypothesis compiles clean and proves nothing), every decidable predicate is
  exhibited both satisfiable and refutable, and every theorem's doc comment states
  **what would falsify it** — a theorem that restates a definition is decoration,
  and decoration would inflate the recorded count.
- `formal/Controls/MustFail.lean` **must not compile**: four false statements,
  each false for a different reason, the fourth refuted on *recorded* ledger
  values. Without a control refuted on real data, every `by decide` in
  `Instances.lean` would be an assertion that the kernel evaluated something.
- The incomplete-proof scan is word-boundary anchored and scoped to `AILM/`. Both
  were earned here: a bare substring scan flagged `stage4Gold_admits_no_sound_…`
  as an unfinished proof, and a whole-tree scan false-positives on the very file
  documenting the axiom policy. The theorem was renamed *and* the scan was fixed.

### Added — documentation

- `formal/README.md` and `docs/formal.md`: what is formalized, the Lean ↔
  equation ↔ Python ↔ test isomorphism table, elan/lake build and regeneration
  commands, and an explicit scope boundary separating *not a Lean object at this
  scope* from *empirical, and formalising it would launder a conjecture* from
  *decoration if proved* — with the one genuinely deferred item labeled as such.
  Familywise error control is named first: `Bonferroni.lean` proves the union
  bound's **arithmetic**; the step to a statement about rejection probabilities is
  assumed, not proved, and citing it otherwise would be the same
  necessary-is-not-sufficient defect `sec:gate_design` diagnoses elsewhere.
- `docs/manuscript/15_reproducibility.md` gains "A machine-checked formal layer",
  written against the four new tokens and stating the falsifier: a re-record that
  moved a sweep shape or a gate verdict changes the generated Lean and the
  `decide`-proofs stop compiling.
- `tests/test_checkout_completeness.py` now inventories `LedgerData.lean` and
  `axiom_report.txt`. The second is not redundant — every check that reads the
  axiom report is gated on `lake` being on `PATH`, so without the toolchain a
  missing report was invisible.

### Verified

- The ledger grew by the `formal_*` block and nothing else; `git diff --numstat
  data/claims.json` shows insertions only, so no existing entry was touched.
  (The before/after totals were written here as bare digits and went stale within
  the same working tree — the reason this file's numbers are claim tags or
  commands now, never typed counts.)

### Fixed — two audits of the formal layer, one Lean and one isomorphism

- **A conditional theorem with no witness.** `bonferroni_union_bound_div` (B3) —
  the *divided* form of the correction the manuscript actually writes — had no
  satisfiability witness and no instantiation anywhere in the tree, while
  `formal/README.md` and `docs/formal.md` both asserted that every conditional
  theorem carries one. `witnessFamily_bounded_by_corrected_threshold` and
  `witnessFamily_bound_div` supply the missing pair; B2a is now exercised through
  B3 rather than only stated.
- **Two `Decidable` instances exercised in one direction only.** `satisfies` had
  no refutation witness and `PeakAt` no failing instance, against a README
  claiming every decidable predicate is exhibited both satisfiable and refutable.
  Both polarities are now exhibited for all six instances.
- **`BRACKETED` meant less than it read.** `scripts/audit_gates.py` sets
  `threshold := baseline` on every audited row, so the vacuity half of the
  verdict is discharged by `passes_irrefl` unconditionally and the six stage-gate
  theorems each carry exactly one measured comparison. The reduction is now
  machine-checked (`bracketed_at_baseline_bar_iff`, G9) and disclosed in
  `AILM/Instances.lean`, `formal/README.md` and `docs/manuscript/03a_gate_design.md`.
- **A manuscript claim the ledger contradicted.** `03a_gate_design.md` said no
  recorded value sits on the $x = t$ boundary; every audited row sits exactly
  there, by construction. Replaced with the true statement and its consequence.
- **"The ceiling bounds every model in the class" is false here.** The Stage 4
  derived row's shipped agent scores above its recorded smoothing ceiling. The
  case-order argument never needed that premise — it needs the measured fact
  about the baseline — and the prose in `03a_gate_design.md`, `GateVerdict.lean`
  and the generated scope note now says so.
- **Measured numbers hand-typed inside the generated Lean.** Three gate scope
  strings in `scripts/generate_lean_ledger.py` quoted the audit's own subsample
  measurements as digits. They resolved to no ledger label, so byte-identity
  regeneration *preserved* them — they would have gone stale on the next
  re-measurement, inside the one file whose header promises nothing in it is
  typed by hand. The scope notes are qualitative now.
- **Five recorded values no proof read.** `couplingGridSize`,
  `couplingBestAccuracy`, `couplingAlpha1Accuracy`, `couplingBestAlpha` and
  `lexicalBestAlpha` were generated and inert, including the one carrying CP2's
  "full six-point grid" strength. CP4-CP6 and LX3a consume them. The lexical
  grid is now emitted (marked `[structural]`) so `lexicalPeakIndex` is checked in
  the kernel instead of trusted from Python.
- **A redundant `decide`.** `couplingRecorded_noInteriorGain` was decided
  independently while being strictly implied by
  `couplingRecorded_declines_across_recorded_points` via S1. It is a derivation
  now — decoration inflates `claim:formal_theorem_count`, which is the one number
  here meant to measure something.
- **Scanner blind spots.** `record_formal_claims.py` and
  `test_formal_isomorphism.py` globbed `AILM/*.lean` non-recursively and anchored
  to bare `^theorem`/`^instance`. A module under `AILM/Sub/`, or a `lemma`,
  `private` or attribute-prefixed declaration, would have escaped the axiom-report
  coverage check *and* the placeholder scan together. Both now `rglob`, the
  patterns cover modifiers and `lemma`, and any form the scanner cannot read is a
  refusal rather than a silent miss.
- **Counts hand-typed outside the scanned prose set.** `formal/README.md` and
  `HANDOFF.md` carried the theorem, example and verdict counts as digits;
  `_prose_files()` in `tests/test_claims_discipline.py` scanned neither. Both are
  in scope now, the digits are claim tags, and
  `formal_instance_theorem_count` / `formal_theory_theorem_count` were added so
  the theory/ledger split has a label to cite.
- **A token in a number-sensitive sentence.** `15_reproducibility.md` read
  "reaches `{{FORMAL_AXIOMS_BEYOND_CORE}}` axiom beyond" — correct at one, wrong
  the moment the count moves. Reworded so the token sits in a position no
  singular/plural agreement depends on.

## [Unreleased] — Deep-review + deep-improve pass (2026-08-04, third pass)

### Fixed — review findings across statistics, manuscript, figures

- Eight shipped `INSERT AFTER THIS LINE` editorial markers removed with their
  truncated sentence tails restored (incl. the coupling margin + p-value in
  Limitations); guard test added. Reversed left/right caption on
  `fig:wiring_comparison` corrected; `fig:composition` rescoped to the three
  Part-A models it draws; `fig:inference_cycle` caption matched to the drawing;
  stale GNN spec path fixed. Level/channel counting reconciled (six levels,
  four channels) across Hierarchy, Results, Gate Design, Reproducibility.
- Statistics disclosure: Stage 6 verdict fragility stated in both directions
  (one-sided p recorded; family-of-8 sensitivity); degenerate family members
  disclosed (survivor rate 5 of 9 testable); power column moved to a fixed
  reference share (0.55) with observed-power redundancy disclosed;
  Bonferroni-adjusted interval check recorded per effect; stage5 AUC carries a
  bootstrap 99% CI; family-closure test enforces the "every p-value declared"
  claim; 18 figure legibility/honesty fixes incl. a legend-distinguishability
  gate.

### Added — deep-improve winners (FirstPrinciples -> BeCreative -> RedTeam)

- `fig:multiplicity_forest` (`src/ailm/figures/multiplicity.py`): the four
  paired effects with exact marginal CI99 whiskers, per-row n and
  reference-share power, corrected-verdict badges from recorded booleans, and
  a Stage 6 p-value strip with all three thresholds; six positive-controlled
  refuse-to-draw guards; 12 tests.
- Pre-registered shape verdicts drawn onto `fig:channel_comparison`
  (interior-optimum vs no-interior-gain bands, verdicts recomputed at draw
  time). Population-indexed gate-verdict formalism; pi-overload resolved with
  a notation registry; the Stage 3 path-probability identity promoted to a
  labeled equation. Introduction gains a quantified verification-apparatus
  contribution and a reader's-contract subsection naming the four claim
  strengths; two-level-era residue purged from Background.

### Verified

- Receipt green: 1724 passed / 22 skipped / 0 failed, branch coverage 95.11%;
  75-page PDF, 27 figures, zero unresolved refs/tokens/scaffolding, sha256
  `20b62c8b...`.

## [Unreleased] — Visualization pass: belief dynamics, phoneme matrices, figure provenance (2026-08-04, second pass)

### Added — `fig:belief_dynamics` (`src/ailm/figures/belief_dynamics.py`)

- One UD EWT dev sentence ("impact of the war on the american people?", chosen
  as one of the few whose every word is in the Stage 3 vocabulary and every
  UPOS in the Stage 4 alphabet, re-verified at build time) traced through the
  REAL pymdp inference paths at five levels: L1 predictive prior (posterior
  omitted — identity emission makes it one-hot), L2B phone posterior
  (cross-checked against the float64 oracle; the build raises on disagreement
  beyond float32 tolerance), L3 trie word-posterior narrowing, L4 case-role
  posterior (oracle-exact), L5 clause-type posterior over the decoded case
  prefix (which stays near the prior — consistent with the Stage 5 null).
  Per-level entropy traces; on-figure diagnostics (jax version/backend, pymdp
  Agent class, both cross-check deviations, all computed at generation time).

### Added — `fig:phoneme_matrices` and figure provenance

- Level 2's learned G2P emission and phone-bigram transition, built through the
  exact shipped constructor calls, peak cells re-derived per build.
- `scripts/generate_figures.py` now logs the numeric environment, times every
  generator, and writes `output/figures/figure_provenance.json`;
  `tests/test_figure_provenance.py` (real subprocess) holds the record against
  the pyproject pins and asserts the CPU backend.

### Verified

- Full receipt green: 1672 passed / 16 skipped / 0 failed, branch coverage
  95.27%; 70-page PDF, 26 figures, zero unresolved refs/tokens. All 26 PNGs
  pass the geometric layout gate; the nine densest were re-inspected by eye.

## [Unreleased] — Acceptance receipts, pairwise-zip regression fix, manuscript reconciliation (2026-08-04)

### Fixed — self-offset `zip(..., strict=True)` defect class (19 sites)

- The 2026-08-04 hardening added `strict=True` mechanically (ruff B905) to
  every `zip`, including 16 self-offset bigram idioms `zip(seq, seq[1:], ...)`
  and 3 nested `zip(zip(A, B), zip(A[1:], B[1:]))` variants — unequal-length
  *by construction*, so every full-corpus bigram path raised `ValueError`
  while focused tests stayed green (55 failures + 175 errors on the first
  real full run). All rewritten to `itertools.pairwise`; the documented
  truncating zip in `figures/_common.py` restored to explicit `strict=False`.
- New guard: `tests/test_no_self_offset_strict_zip.py` — AST-based (flat and
  nested detectors), with positive controls and a non-empty-scan assertion.

### Changed — ledger re-recorded under the refreshed lock

- The Aug-4 `uv.lock` bump (numpy/scipy) drifted four recorded values:
  `stage5_auc` 0.7024 → 0.7037, two learning deltas → 0.0, one oracle
  epsilon 1.01e-06 → 1.06e-06. All recorders re-run in the locked env; no
  gate verdict changed; recorder re-run is a byte-level no-op again.

### Added — Stage 0 inventory claims; no hard-coded model constants in prose

- `stage0_letter_alphabet_size` (30), `stage0_word_alphabet_size` (27),
  `stage0_word_letters` (26), `stage0_coverage_floor_pct` (95, read from
  `pyproject.toml`) — derived from code with refuse-to-record guards, and the
  manuscript now token-references them everywhere the numbers appear.

### Changed — manuscript reconciled to the built L1–L6 state (81 verified edits)

- Framing sections no longer call built stages unbuilt; the universal
  novel-information "iff" is stated as an evidence-bounded working hypothesis
  (pre-registration records in `docs/decisions.md` preserved verbatim, entry
  18 carries a dated status note); the multiplicity family and exact 99%
  intervals render from `STAT_*` tokens; layout defect in the level-stack
  figure footnote fixed; the figure-vs-gate alpha agreement test rebound
  structurally to `ailm.statistics.FAMILYWISE_ALPHA`.

### Verified — full release receipt + rendered artifact (2026-08-04)

- `verify_release.py full --keep-going`: all nine steps exit 0, tracked diff
  unchanged. 1669 passed, 15 skipped, 0 failed, 0 xfailed, branch coverage
  95.21%; unbounded gate audit clean (Stage 7 boundary re-scoped UNAUDITED by
  design). 68-page PDF, 24 figures, zero `??`/`{{TOKEN}}`, sha256
  `fe97faed66399928958888917c9f9da5458bcd5216817760205e8fccdac91a2c`.
- Apple-Silicon Metal measured and documented (`docs/troubleshooting.md`):
  works via `jax-metal` + PJRT shim, float32 agreement ~1e-6, 309/309 tested
  Metal passes — but 3.6× slower on real AILM inference (tiny state spaces);
  CPU remains the only environment of record.

## [Unreleased] — The hierarchy completed: clauses (a null), narrative (confirmed), and the speech boundary

### Added — Stage 5, the clause level (decision 24)

- A two-type mixture of case-role chains over the joint (type × case) space —
  16 states, block-diagonal transition, deterministic emission — with the
  label source (sentence-final punctuation) provably disjoint from the
  observation stream (PUNCT carries no case role; asserted at build time).
  `vendor/ud_ewt_clause_counts.json` (checksummed), `ailm.levels.clause`,
  spec `L5_clause` + two mutations + registry entries + bit-identity.
- **The finding is a null, recorded with positive-result rigor**: the
  type-ORACLE ceiling gains only 0.13% over the type-free bar; the mixture is
  worse than the bar; classification recovers 0/20 held-out interrogatives
  against a 90.5% majority (AUC 0.70 — rank signal without consequence). The
  detectability control — the same harness on planted synthetic types —
  classifies at 99.5% and drops perplexity decisively, so the null is about
  English at this order, not the instrument. The declared gate ("free energy
  tracks clause length") shown VACUOUS executably. `audit_stage5` enforces
  consistency-with-null: the project's first sound-but-FAILED gate.

### Added — Stage 6, the narrative level (decisions 25–26, pre-registered)

- Document phases as position terciles over sentence-type sequences
  (`vendor/ud_ewt_narrative_counts.json`, 117/30 documents;
  `ailm.levels.narrative`; spec `L6_narrative` + mutation + registry +
  bit-identity). Predictions committed at `5415b8c` BEFORE the held-out
  measurement; all confirmed: position informs type out of sample (+0.0037
  nats, sign test p = 0.0012 — web discourse opens with questions); the
  top-down channel into the clause type prior works with a byte-exact α=0
  anchor and a distinguishable scramble; and its case-role value is bounded
  by Stage 5's 0.13% ceiling — refining the novel-information criterion:
  **novel AND consequential**.
- The measurement caught this project's own prose mis-stating the left-to-
  right transition's zero triangle (lower vs upper — decision 1's orientation
  trap, in a docstring); the recorder now refuses on a non-zero upper
  triangle.

### Added — Stage 7, the speech boundary

- `ailm.speech_boundary`: the interface contract — the boundary vocabulary IS
  the Level 2 inventory by identity, round-trip fidelity over every vendored
  CMUdict pronunciation, stress digits stripped, everything else rejected
  fail-closed. The declared "Whisper → phoneme → TTS" gate re-scoped in the
  audit: it would measure someone else's models.

### Changed

- Manuscript: new sections `07a_clause_level` and `07b_narrative_level`; the
  level-stack table rewritten to the reconciled 7-level reality; the
  unaudited-gates table converted from liability list to completed audit
  record ("not one declared threshold survived measurement"); **four stale
  blocks that predated Stages 3–4 and generation corrected** (11_results,
  12_gate_design, 13_limitations, 03_architecture — including "the system
  does not generate text", which had been false since generation shipped).
  Two new ledger-drawn figures (`clause_null`, `narrative_channel`) with
  refuse-to-draw guards. ISA: ISC-22/23/24 closed with reframes;
  stage_current 7/7.

---

## [Unreleased] — W2 mechanism + W3 figures: the agents learn, and the parameters are shown

### Added — W2, pre-registered before measured

- `docs/decisions.md` entry 22, committed at `c760357` BEFORE any update ran:
  under the letter level's identity emission, pymdp's `infer_parameters` must
  reproduce the shipped add-k counting path exactly (conjugate Dirichlet
  posterior mean), with a stated withdrawal condition. Entry 23 records the
  outcome: **confirmed** — matched priors agree to 2.98e-08 over 1,323
  positions; a mismatched prior breaks agreement by 0.153 and re-matching
  restores it; held-out perplexity identical. The lr≠1 control shows `lr_pB`
  is not a tuning knob: any value except 1 estimates nothing the ledger
  describes.
- `src/ailm/learning.py` — the agent's own `infer_states` loop collects
  beliefs (that they come out one-hot is the mechanism, so they are not built
  by hand), `infer_parameters` updates the Dirichlet, and the comparator is
  the exact shipped counting call. `scripts/record_learning_claims.py` —
  single writer for 11 `learning_*` labels plus `data/learning_agreement.npz`;
  refuses to record unless every pre-registered clause holds.
- `tests/test_learning.py` (10 tests) — including a toy anchor verifying the
  Dirichlet increments at integer resolution, and harness controls proving
  the prior, the stream, and the learning rate are each load-bearing.
- Manuscript: "Learning is counting, and it was predicted before it was
  measured" in the verification section, plus `fig:learning_agreement` and a
  new row in the verification-layers table.

### Added — W3, the parameters shown rather than argued

- `src/ailm/figures/interpretability.py` — `case_matrices` (Level 4's entire
  generative model, every cell annotated, real UPOS/case labels),
  `word_matrices` (deterministic spelling paths for three vocabulary words
  chosen at draw time, the +1 shift, the prior head), `trie_narrowing`
  (support vs prefix length over all 76,883 held-out tokens, refusing to draw
  if support ever grows). `src/ailm/figures/samples.py` — three real
  generation streams at the sweep's operating points, word-colored by
  lexicon membership, with guards binding the regenerated text to the
  recorded sweep. All embedded in the manuscript with falsification-stating
  captions.
- **Caption standard** in `docs/style_guide.md`, enforced by
  `tests/test_caption_standard.py` as a shrink-only ratchet: 15 pre-standard
  captions frozen by label; every new caption must state its falsifier; the
  scan proves its own non-vacuity. ISA criteria ISC-50..52.

---

## [Unreleased] — Computational complexity: exact counts, reconciled, in the manuscript

### Added

- `src/ailm/complexity.py` — the cost model: exact closed-form operation
  counts for the scaled alpha recursion (per category: mul/add/div/log/exp/cmp),
  an **instrumented counter** that runs the recursion on real workloads and
  tallies only what executes, the Θ(1)-per-token word-level collapse, and
  spec-priced memory (`spec_memory_bytes_theory` == live agent bytes, including
  the one-scalar policy prior pymdp materialises even for single-action
  agents — discovered when the first reconciliation missed by exactly four
  bytes per recognition level). 100%-branch negative controls: the off-by-one
  push formula, a dropped sequence, a mis-stated decode flag, and the
  implicit-`E` term's removal must all fail to reconcile.
- `scripts/record_complexity_claims.py` — single writer for 33 `complexity_*`
  labels, all deterministic (no timings — a timing cannot satisfy ISC-4's
  idempotence). Refuses to record if any count or byte disagrees. Workloads
  are the gate workloads: `complexity_l2b_positions` equals
  `stage2_g2p_eval_positions`, `complexity_l3_tokens` equals
  `stage3_eval_tokens`, enforced by test.
- `src/ailm/figures/complexity.py` — two ledger-drawn figures:
  `complexity_ops` (recorded per-position rates with the closed form
  recomputed at draw time and overlaid; refuses to draw on disagreement) and
  `complexity_memory` (the per-level byte ladder, and factored-vs-dense-joint
  scaling with the pymdp budget and the ~17k-word cap marked).
- `docs/manuscript/98_supplement_complexity.md` — new section `{#sec:complexity}` between
  hierarchy and verification: the cost model (`{#eq:step_cost}`), the
  per-level table (`{#tbl:complexity}`), both figures with falsification-
  stating captions, the P²=256× factorization price, and the generation
  Θ(V+S) asymmetry. Every number is a ledger token.
- `tests/test_complexity.py` (24 tests), decision 21, ISA criteria ISC-47..49.

---

## [Unreleased] — W1: the spec builds the agent, and the identity is proven first

### Added

- `src/ailm/spec_agent.py` — `build_agent_from_spec` constructs a live pymdp
  agent from a GNN spec alone: dimensions from `StateSpaceBlock`, counts from
  `ModelParameters`, slot roles from `ActInfOntologyAnnotation`, structural
  forms materialised directly, and every `Learned(...)` form resolved through
  `LEARNED_REGISTRY` — an explicit **default-deny** table keyed on the complete
  provenance keywords *and* the ontology role, so an unregistered generator, a
  changed estimator, or a known generator on the wrong slot all refuse to
  construct. Every resolver calls the same shipped building-block functions the
  hand builders call. The constructed agent is `assert_conforms`-checked
  against its own spec before it is returned.
- `tests/test_spec_agent.py` (27 tests) — the migration gate, run **before**
  any use of the spec path: all six shipped levels bit-identical (shapes,
  dtypes, bytes across `A`/`B`/`C`/`D`/`E`) between spec-built and
  builder-built agents; planted-perturbation, dtype and missing-slot controls
  on the comparator; the doctored A/B-swap spec refused **by role**; and the
  exact partition of the eleven mutations into eight that fail construction and
  three that construct provably different models.
- `docs/decisions.md` entry 20 and ISA criteria ISC-44..46.

### Changed

- `gnn/models/L1g_letter_generation.gnn.md` — `E` is now declared
  `Uniform(30)`, which is what the constructed agent has always carried; the
  habit is installed per step by the generation loop, exactly as `C` receives
  the top-down preference. The old `E=Learned(bigram)` line described a model
  the constructor never built — the one divergence construction caught that
  conformance could not, because no shape check binds `E`. Recorded numbers are
  untouched; the description was wrong, not the measurements. `gamma: 16.0` is
  now declared in `ModelParameters` as the construction default of record.
- `TODO.md` W1 — closed, with builder retirement **re-scoped**: the builders
  stay as the parameterized surface for negative controls, the spec path is the
  canonical zero-argument constructor, and the bit-identity gate binds them.

---

## [Unreleased] — Handoff, and the next three workstreams scoped

### Added

- `HANDOFF.md` — the cold-start prompt for a successor. Verified state, the eight
  invariants that must not be broken (each learned by shipping the defect), the
  three scoped workstreams with their reasoning and predicted traps, and the
  research question's current standing. Linked from README, ISA and TODO.
- `TODO.md` gains three scoped workstreams, each opened by a **verified** gap
  rather than an assumption:
  - **W1 — close the GNN → pymdp loop.** Confirmed: nothing in `ailm.gnn`
    constructs an agent; specs only *check* agents that hand-written builders
    create, so every level has two sources of truth. Make the spec build the
    model — and prove bit-identity with the builder path **before** retiring any
    builder, or the claims ledger gets silently rebased onto a different model.
  - **W2 — parameter learning inside pymdp.** Confirmed: `learn_A`, `learn_B`,
    `pA`, `pB` and `infer_parameters` appear nowhere outside the Stage 0 API
    smoke tests. Every matrix is counted *outside* pymdp and installed fixed —
    the agents infer, but have never learned. Prediction to pre-register: add-k
    counting *is* the conjugate posterior mean, so the two should agree; the
    online non-stationary case is the version that would be a real claim.
  - **W3 — visualisations and captions.** Per-level matrix heatmaps for L3/L4
    (only Level 1 currently *shows* that a parameter is a readable conditional
    probability), the induced trie narrowing, generation samples, the scaling
    ladder — plus a caption standard requiring every caption to state what would
    falsify its panel.

---

## [Unreleased] — Figures, and the last of the unbacked numbers

### Added — four figures for the new results

`src/ailm/figures/composition_results.py`, auto-discovered by the existing
`generate_*` convention and drawn **from the claims ledger**, so a panel cannot
drift from the measurement behind it. Each carries a guard that raises if its
own finding changes rather than silently redrawing:

- `word_miswirings` — Stage 3's four mis-wirings against the bar, on a log axis
  because the caught one is two orders of magnitude worse. Raises unless exactly
  three still clear the bar.
- `case_both_defects` — Stage 4's single threshold against both populations.
  On gold the three markers *coincide*, so the panel says so instead of drawing
  three overlapping labels: the observation alphabet has size one.
- `generation_sweep` — the U-shaped precision sweep. Raises unless the optimum
  is still interior.
- `channel_comparison` — the composition result: two channels into the same
  receiving level, one peaking in the middle and one declining monotonically.
  Raises unless they still have opposite signs.

All four are wired into their manuscript sections with captions. The figure
layout gate (`tests/test_figure_layout.py`) discovered them automatically and
caught a clipped caption and an 8pt label; both were fixed by giving the figures
room rather than shrinking type, per the build-time legibility invariant.

### Fixed — the last `ledger_backed: False` rows

`SUBSAMPLE_COMPARISON` carried three measured rows as hardcoded literals. Its own
docstring named them "the only numbers in this module not backed by a recorded
run" — an honest disclosure, and a disclosure is not a fix, because a figure and
a manuscript section rested on them.

`scripts/record_subsample_claims.py` now measures all four subsets and records
them. **The measured values reproduced the literals exactly**, so nothing printed
was ever wrong; what was missing was anything that would have noticed if it
became wrong. `subsample_bias_data` now verifies *every* row against the ledger,
and `test_every_subsample_row_is_verified_against_the_ledger` plants a drifted
value to prove the flags mean "checked" rather than "labeled".

### Changed

- `TODO.md` rewritten: completed items **removed** rather than struck through,
  since the CHANGELOG is the history and a queue that also tries to be one stops
  being scannable.
- `eq:case_free_energy` remains `IMPLEMENTED (inert)` — implemented, unit-tested,
  called by nothing, and saying so in both the formalism map and the manuscript.
  Wiring it in without a measurement that could fail would be decoration.

---

## [Unreleased] — What makes a hierarchy: a criterion, pre-registered and confirmed

### Added — the inter-level composition experiment

The project had two inter-level results with opposite signs, which is an anecdote
unless something explains both. `docs/decisions.md` entry 18 states the
**novel-information criterion** — a message improves the receiver iff it carries
information the receiver does not already have — derives a prediction, and
records that prediction **before** the measurement. Entry 19 records the outcome.

- `src/ailm/levels/lexical_channel.py`, `scripts/record_lexical_channel_claims.py`,
  `tests/test_lexical_channel.py`, manuscript section `09_hierarchy.md`.
- **The prediction was confirmed on both counts.** A lexical prefix message into
  the grapheme-to-phoneme level helps by claim:lexical_best_margin_pts points
  (p = claim:lexical_best_mcnemar_p over claim:lexical_eval_positions positions),
  several times the level's own headline margin of claim:stage2_g2p_margin_pts
  points — and the optimum is **interior**, with the effect reversing to
  claim:lexical_high_alpha_margin_pts points at weight claim:lexical_high_alpha.
- **The comparison is controlled.** Entry 9's channel sent the letter predictive
  into the *same* receiving level and fell monotonically; this one sends a
  lexical statistic and peaks in the middle. Architecture, direction and receiver
  matched; novelty the only variable.
- **The generalizable part is the shape, not the sign.** A monotone decline
  diagnoses redundancy, an interior optimum diagnoses novelty — so the sweep
  answers "should this channel exist" without committing to it first.
- Guards mirror entry 9's, because a positive result needs what a null needs: an
  exact `alpha = 0` identity anchor (the recorder refuses to write if it is not
  bit-identical), a non-vacuity check, an entropy-preserving scrambled control
  losing claim:lexical_best_vs_scrambled_pts points, a grid validator that
  refuses a collapsed sweep, and a check that the channel conditioned on a real
  prefix at claim:lexical_positions_with_prefix_pct% of positions — so a null
  could not have come from a channel with nothing to say.

### Changed

- Corrected statements in `03_architecture.md`, `13_limitations.md`,
  `14_related_work.md` and `docs/design/architecture.md` asserting that only one
  inter-level channel existed or that the levels exchange no beliefs. True when
  written; three channels are now measured.

---

## [Unreleased] — Generation, full-dictionary scaling, and BISC-3

### Added — generation, now in scope

Text generation moved from Out of Scope to Goal at the principal's direction.
The apparatus that was specified-but-unexercised — `eq:efe`, `C`, the policy
space — is now exercised.

- `src/ailm/generation.py`, `gnn/models/L1g_letter_generation.gnn.md` plus three
  mutations, `scripts/record_generation_claims.py`, `tests/test_generation.py`,
  `audit_generation()`, manuscript section `08_generation.md`,
  `docs/decisions.md` entry 17.
- **Policy selection is generation, not a metaphor.** The letter level gains a
  controllable transition with one action per symbol: choosing an action *is*
  choosing the next symbol. The learned bigram moves from the transition to the
  policy prior `E` — the agent's habit — and `C` is set top-down from the word
  level's lexicon. First non-flat `C` in the project, and the first channel where
  one level sets another's preferences.
- **The obvious metric could not be the gate.** Lexicon membership is green by
  construction: the preference floors any letter leaving the trie, so membership
  is saturated at every usable precision *and* at the degenerate one that emits a
  single word forever. Had it been the gate, the degenerate generator would have
  passed perfectly. The gate is total-variation distance to the true word
  distribution against ancestral sampling from `B`, paired with a diversity floor.
- **The optimum is interior**, so the shipped gate fails in two opposite
  directions — a property no recognition gate here has. As precision goes to
  zero the model *is* the baseline; as it grows the shipped lexicon's greedy
  prefix path collapses onto one word. That path is not generally the global
  prior mode.
- **An exact identity anchor:** with `C = 0` the generator reproduces ancestral
  sampling character-for-character under the same seed, and the recorder refuses
  to write if it does not. Plus a non-vacuity companion, an entropy-preserving
  scrambled control, and agreement with pymdp's own `infer_policies`.
- Conformance generalizes to **controllable transitions**: a declared
  `num_controls > 1` is now checked against `B`'s third axis, which
  `_canonical_dims` deliberately does not reduce.

### Added — full-dictionary and custom lexicons

- `scripts/generate_word_counts.py --vocab-size 0` builds the **unbounded**
  vocabulary: claim:stage3_full_vocab_size types at
  claim:stage3_full_coverage_pct% held-out coverage, against
  claim:stage3_coverage_pct% for the shipped bound. Vendored and checksummed as
  `vendor/brown_word_counts_full.json`; the gate stays BRACKETED at full scale.
- `--lexicon PATH` accepts a custom word list. Supplied without frequencies it
  produces a uniform word prior, which entry 14 records as a mis-wiring that
  beats the accuracy bar while being wrong — so the generator warns rather than
  accepting it silently. Evaluation always uses held-out Brown restricted to what
  the lexicon covers, never the lexicon itself.
- **The pymdp boundary is now explicit and computed.** `build_word_agent` raises
  with the arithmetic above `MAX_AGENT_VOCAB_BYTES` (~claim:stage3_max_agent_vocab
  types) rather than failing on an allocation. The closed form is exact at any
  size, so the model is measurable where the agent is not — and which of the two
  is lost is stated rather than left to be discovered.

### Fixed — BISC-3, the oldest open item

- The template repo's `validator_citations.py` swallowed a trailing `.`, so every
  sentence-final `@key.` was reported undefined and blocked the combined PDF.
  Committed upstream as `5bf4fa69a` with 8 tests. This session's render depended
  on it: the first attempt exited 0 while producing no PDF, and the validator is
  what named the cause.

### Fixed — stale absence claims

- `02_background.md`, `03_architecture.md` and `docs/formalism_map.md` asserted
  that no level exercises `eq:efe`. True when written, false now. `eq:efe` moves
  from `HONESTLY DEFERRED` to `IMPLEMENTED`; `eq:case_free_energy` stays deferred,
  because `case_conditioned_free_energy` still has no caller outside its tests.

---

## [Unreleased] — Stages 3 and 4: Words and Cases

### Fixed — a broken invariant found by running the suite

- **ISC-4 (one writer per claim label) was violated in the committed tree.**
  Fourteen labels were written by *both* a recorder script and a test, so the
  stored value depended on which ran last. The values happened to agree, which is
  exactly why the defect survived a green suite and a rendered PDF. The
  enumerated guard listed eight canonical labels and caught four of the fourteen.
  - `tests/test_stage1_gate.py`'s `test_stage1_claim_recorded` → renamed
    `test_stage1_claim_verified` and converted from writer to verifier.
  - `tests/test_stage2_gate.py::test_stage2_claims_recorded` now writes only the
    four labels naming populations no recorder measures, and verifies the rest.
  - `tests/test_claims_single_writer.py` gained
    `test_no_test_writes_a_label_it_does_not_own` — **default-deny** over the
    whole ledger against a documented `TEST_OWNED_LABELS` allow-list, plus a
    non-vacuity check that the allow-list stays a small exception and a positive
    control on the detector. Inverting the polarity removes the requirement to
    remember which labels matter.

### Added — Stage 3, the word level

- `src/ailm/levels/word.py`, `scripts/generate_word_counts.py`,
  `vendor/brown_word_counts.json` (checksummed, byte-reproducible),
  `scripts/record_stage3_claims.py`, `tests/test_stage3_gate.py`,
  `gnn/models/L3_word.gnn.md` plus four mutations, `audit_stage3()`.
- **The trie does not fit, and that is a design decision rather than a
  performance problem.** A dense pymdp transition over trie nodes is
  claim:stage3_trie_dense_gb GB at the G2P vocabulary and ~claim:stage3_trie_dense_full_tb TB
  at full CMUdict, at claim:stage3_trie_density_pct% density. The shipped level
  factorizes into **word identity × position** with one modality depending on
  both — an order of magnitude cheaper than trie nodes for the same coverage, and
  with word identity explicit rather than recoverable.
- **The trie is induced, not abandoned.** After a prefix, the posterior's support
  is exactly the vocabulary words carrying it. Checked against a live agent, with
  a companion test confirming the support narrows.
- **The declared gate was VACUOUS by claim:stage3_declared_gate_slack×** — the
  letter-bigram bar itself clears "perplexity < 500". It also named PTB, which
  this project does not have; Stage 3 is measured on the already-vendored Brown.
- **Three of four mis-wirings clear the restated bar**, including one that
  inverts the frequency ranking so the rarest words are predicted most
  confidently (claim:stage3_miswire_reversed_prior against a bar of
  claim:stage3_bigram_baseline). What pins the level is a closed-form path
  identity — the emission path's probability must equal the word's unigram
  probability, agreeing to claim:stage3_path_identity_max_abs_err nats between a
  live pymdp walk and an oracle importing neither pymdp nor jax. All three
  survivors change the prior, so all three fail it.

### Added — Stage 4, the case level

- `src/ailm/levels/case.py`, `scripts/generate_case_counts.py`,
  `vendor/ud_ewt_case_counts.json` (checksummed),
  `scripts/record_stage4_claims.py`, `tests/test_stage4_gate.py`,
  `gnn/models/L4_case.gnn.md` plus four mutations, `audit_stage4()`.
- **One declared threshold, both defects at once.** ">=60% vs the UD 8-case
  oracle" is VACUOUS on the gold-morphology population (a constant majority
  predictor scores claim:stage4_gold_majority%) and UNREACHABLE on the derived
  population (nothing at this model order passes
  claim:stage4_derived_agreement%). Which defect you observe depends only on
  which population it is scored against — the clearest evidence yet that a
  threshold without a named population is not a threshold.
- **The circularity trap.** The model observes UPOS and nothing else; `deprel`
  and `feats[Case]` are withheld because the labels are derived from them. Unlike
  every mis-wiring below this level, that one would make the model look
  *excellent*, so the guard is structural rather than numeric.
- **The shipped gold arm is an empirical null.** Every gold-morphology token in
  English EWT is a `PRON`, so the gold observation alphabet has size
  claim:stage4_gold_upos_alphabet_size. The committed split records
  claim:stage4_gold_mcnemar_b/claim:stage4_gold_mcnemar_c discordant pairs. A
  constant emission removes observation evidence but does not erase transition
  dynamics, so this is not a universal one-tag impossibility claim. No gate is
  stated there.
- **A fidelity measurement that was green by construction.** The natural way to
  compute heuristic-versus-gold agreement calls the project's case oracle, which
  consults `feats[Case]` first and therefore returns the gold label on exactly
  the tokens being compared — reading 100.0%. Bypassing the morphological
  shortcut gives the honest claim:stage4_heuristic_gold_agreement_pct%.

### Changed — the GNN layer becomes a registry

- Specs moved to `gnn/models/`, `gnn/mutations/`, `gnn/compositions/` under a
  parseable grammar `<kind><level><part>_<slug>[@<mutation>].gnn.md`, so a spec's
  hierarchy position, base model and injected defect are derivable from its name.
  `discover_models()` / `parse_model_id()` / `spec_path()` in `ailm.gnn`;
  `tests/test_gnn_registry.py` enforces the layout **default-deny**.
- **Mutations are specs, not Python fixtures.** Eight of them, each of which must
  FAIL conformance against the correct live agent.
- **Conformance generalizes to multi-factor models** without loosening the
  single-factor rules: one transition and one prior per factor bound by an
  `_f<N>` suffix, per-factor `num_hidden_states_f<N>`, and a modality's state axes
  checked in factor order so transposing two factors is a violation rather than a
  reshape. `describe_shapes` now returns the full per-factor lists, because
  reporting only slot 0 would leave later factors unverified while still
  returning "conformant".

### Changed — manuscript and docs

- New sections `06_word_level.md` and `07_case_level.md`; later sections
  renumbered so reading order matches the hierarchy. `10_gate_design.md` gains
  "Defect three: necessary is not sufficient, at scale" and "Defect four: one
  threshold, both defects at once".
- New design docs `docs/design/stage3_word.md` and `docs/design/stage4_case.md`.
- `docs/decisions.md` entries 13–16.
- **The generation contradiction is resolved in the ISA**, which is the system of
  record: generation is out of scope, and Stages 6–7 are therefore *blocked
  pending an explicit scope change* rather than merely unstarted. The roadmap
  sketch in `docs/design/architecture.md` that assumed otherwise is marked
  aspirational.

---

## [Unreleased] — Stage 2: Phonemes

### Added

- `src/ailm/levels/phoneme.py` — the phoneme level. Part A: a phone-sequence bigram
  model (`build_phone_transition`, `build_phoneme_agent`,
  `compute_phoneme_perplexity`) mirroring Stage 1. Part B: grapheme-to-phoneme
  (`build_g2p_emission`, `build_g2p_transition`, `build_g2p_start`,
  `build_g2p_agent`, `decode_phonemes`, `g2p_accuracy`,
  `per_letter_argmax_baseline`, and the position-aligned `decode_positions` /
  `baseline_positions` used for paired significance testing). This is the first
  level whose `A` is **not** the identity — letters are observed, phonemes inferred.
- `src/ailm/verification/oracle.py` — **reusable closed-form verification oracle**:
  `forward_filter`, `forward_backward`, `viterbi`, `sequence_perplexity`,
  `markov_chain_perplexity`. Pure numpy, imports no pymdp. A throwaway version of
  this is what caught the Stage 1 wiring bug; it is now a tested module so every
  level can be cross-checked the same way. 54 tests, all closed-form or
  analytically anchored.
- `vendor/cmudict_counts.json` — Stage 2 corpus artifact, verified byte-identical
  on regeneration. Phone unigrams/bigrams over claim:cmudict_base_word_count
  CMUdict base words plus the G2P length-matched pairs.
- `scripts/generate_phoneme_counts.py`, `scripts/record_stage2_claims.py`
- **`scripts/audit_gates.py`** — audits every stage gate for *vacuity* and
  *reachability*, classifying each as VACUOUS / UNREACHABLE / BRACKETED from its
  measured trivial baseline and achievable ceiling, and listing stages 3–7 as
  explicitly UNAUDITED. Both gate defects this project has shipped were found by
  doing this by hand; the script makes it re-runnable.
- `tests/test_phoneme.py`, `tests/test_stage2_gate.py`, `tests/test_oracle.py`
- `docs/design/stage2_g2p.md` — the alignment decision and its costs

### Stage 2 gate result

**(A) Phoneme sequence perplexity** — a comfortable margin:

| Comparator | Perplexity |
|---|---|
| Uniform over 39 phones | claim:stage2_phoneme_uniform_baseline |
| Context-free unigram phone frequencies | claim:stage2_phoneme_unigram_baseline |
| **Phoneme bigram agent** | **claim:stage2_phoneme_perplexity** |
| Order-1 information floor | claim:stage2_phoneme_floor_perplexity |

**(B) G2P** — a deliberately thin margin, defended statistically:

claim:stage2_g2p_accuracy% vs the context-free per-letter bar
claim:stage2_g2p_baseline_accuracy% = claim:stage2_g2p_margin_pts pts over
claim:stage2_g2p_eval_positions held-out positions. Paired McNemar
b = claim:stage2_g2p_mcnemar_b, c = claim:stage2_g2p_mcnemar_c,
p = claim:stage2_g2p_mcnemar_p.

### Changed

- **Stage 2's gate was restated — the declared one was unbuildable AND vacuous.**
  The spec read "letter→phoneme accuracy ≥60% on the English EWT oracle." UD EWT
  is a *syntactic* treebank (`lemma`/`upos`/`feats`/`deprel`) with **no phonemic
  annotation at all** — the line was a copy of Stage 4's case gate. And ≥60% is
  vacuous regardless: a context-free per-letter lookup table scores
  claim:stage2_g2p_baseline_accuracy%, so a model doing nothing hierarchical
  passes. The gate is now stated against that lookup table. Note this is the
  **inverse** of the Stage 1 defect — Stage 1's target was unreachable, Stage 2's
  was too easy — and both were caught by measuring the ceiling *and* the trivial
  baseline before accepting the number.
- **`_iter_pairs` subsamples by even spacing, not by prefix.** The original took
  `pairs[:limit]`, reasoning that a deterministic prefix is reproducible. It is —
  but reproducible is not representative. `g2p_pairs` is alphabetically sorted, so
  `test[:1200]` is **only a/b/c words**, and it reports a G2P margin of +1.691 pts
  where the full held-out set gives claim:stage2_g2p_margin_pts pts: a 3.1×
  overstatement that reads as a comfortable pass. Evenly-spaced indices span all
  26 initials at the same cost.

### Fixed

- **The G2P gate cannot be honestly computed on any fast subsample.** Paired
  McNemar reaches significance on the full claim:stage2_g2p_eval_positions
  positions (p = claim:stage2_g2p_mcnemar_p) but *not* on a representative stride
  of a third of them (+0.489 pts, p = 1.7e-02). The biased prefix looked
  significant for the wrong reason. Rather than shrink the eval or inflate the
  margin, `scripts/record_stage2_claims.py` proves the pymdp decoder and the numpy
  oracle agree on **claim:stage2_decoder_disagreements positions** over a bounded
  subset, then uses the fast oracle for full-set statistics — and **refuses to
  record** if that disagreement count is ever nonzero, since the full-set figures
  would then no longer describe the pymdp agent.

### Performance note

pymdp per-position `infer_states` does not scale to full-corpus evaluation: the
G2P held-out set alone is ~42.8k calls. Levels 3+ will need either the
oracle-equivalence pattern above or batched inference; a naive full-corpus pymdp
pass exceeded ten minutes.

---

## [Unreleased] — Stage 1: Letters

### Added

- `src/ailm/levels/letter.py` — letter-level pymdp agent. `ALPHABET` (30 symbols: a–z, space, period, apostrophe, hyphen), `normalize_text`, `train_on_corpus`, `build_unigram_a` (column-stochastic (1,30,30)), `build_letter_agent`, `compute_perplexity`, `load_brown_counts`
- `vendor/brown_counts.json` — claim:stage1_brown_bigram_types bigram types over claim:stage1_brown_bigram_tokens tokens from the NLTK Brown corpus. Regeneration is byte-identical, so the artifact is deterministic and tests need no network
- `scripts/generate_counts.py` — Brown counts generator (embedded public-domain fallback if NLTK is unavailable)
- `scripts/record_stage1_claims.py` / `scripts/record_stage0_claims.py` — derive every prose number into `data/claims.json`; nothing hand-typed
- `tests/test_letter.py` — module API surface; numeric assertions are closed-form or re-derived in-test
- `tests/test_stage1_gate.py` — Stage 1 gate with unigram baseline, uniform negative control, an explicit anti-wiring negative control, and a closed-form bigram-chain cross-check
- `tests/test_claims_discipline.py` — enforces the claims ledger against **real** repository prose (see Fixed)
- `nltk>=3.8` dependency

### Stage 1 gate result

| Comparator | Perplexity |
|---|---|
| Uniform over 30 symbols | claim:stage1_uniform_baseline |
| Context-free unigram letter frequencies | claim:stage1_unigram_baseline |
| Mis-wired variant (bigram as emission) | claim:stage1_anti_informative_perplexity |
| **Bigram agent** | **claim:stage1_letter_perplexity** |
| Order-1 information floor | claim:stage1_bigram_floor_perplexity |

The agent reaches claim:stage1_letter_bits bits/char, capturing
claim:stage1_bigram_signal_captured_pct% of the information available to any
order-1 letter model. Its inference was cross-checked against an independent
closed-form numpy bigram chain and agrees to six decimal places.

### Changed

- **Bigram statistics moved from the emission `A` to the transition `B`.** The
  Stage 1 handoff specified `A = bigram, B = identity`. That wiring is
  anti-informative rather than loudly broken: observing letter *c* yields a
  posterior over the letter that *preceded* c, and an identity `B` carries that
  backward-looking belief forward as the next-letter prediction. Measured, it
  scores claim:stage1_anti_informative_perplexity on held-out prose — and on a
  small in-sample corpus it scored *worse than the uniform 30.0 baseline* while
  still slipping under a naive `< 28.0` threshold. The correct wiring is
  `A = identity, B = bigram`. pymdp cannot catch this class of error: a bigram
  matrix is column-stochastic and passes `_validate()` in either slot, so only a
  numeric probe surfaces it. Stage 2 onward reuses this pattern — check the
  orientation numerically, not structurally.
- **Stage 1 gate restated from "≥2 nats above uniform" to "beats the
  context-free unigram baseline."** The original target is
  *information-theoretically unreachable* for any letter bigram model: the
  conditional entropy of the Brown bigram distribution caps the achievable gain
  at claim:stage1_max_bigram_gain_nats nats, and the agent realizes
  claim:stage1_gain_over_uniform_nats of it. A ≥2 nat gain requires a
  trigram-or-higher model. The unigram bar is both reachable and diagnostic:
  clearing uniform proves nothing (a context-free model does it), but beating
  unigram letter frequencies requires actually using the previous letter.
  README, AGENTS.md, and `docs/design/architecture.md` carried the false
  ">2 nats" claim and were corrected.

### Fixed

- **The claims ledger was never enforced against real prose.**
  `ClaimsLedger.scan_prose` shipped in Stage 0 and was exercised only against
  `tmp_path` fixtures, so `claim:phoneme_count_39` sat unbacked in README.md,
  AGENTS.md, and `docs/provenance/sources.md` while the whole suite stayed green —
  the Stage 0 tests recorded that label into throwaway ledgers only. Added
  `tests/test_claims_discipline.py`, which scans the actual tree and carries a
  positive control (a planted unbacked tag must be flagged) so the gate cannot
  pass by scanning nothing. Verified non-vacuous by injecting an unbacked claim
  into the real README and confirming the gate goes red.
- `ClaimsLedger._load` now repairs a structurally incomplete ledger (e.g. a `{}`
  seed) by backfilling only missing skeleton keys, and raises on unparseable JSON
  or a non-object top level rather than silently overwriting data it cannot read.

---

## [Unreleased] — Stage 0: Infrastructure

### Added

- `src/ailm/__init__.py` — package root with CEREBRUM and CCD paper references
- `src/ailm/cases.py` — `CaseRole` (8 linguistic cases), `CerebrumFunctionRole` (8 functional roles, [GEN]→[GNV] collision resolved), `CEREBRUM_PYMDP_MAPPING` (Figure 13 derivation), `CASE_FREE_ENERGY_PARAMS`, `transformation_cost`, `case_conditioned_free_energy`
- `src/ailm/claims.py` — `ClaimsLedger` and `load_ledger`; every quantitative prose claim tagged `claim:<label>` and backed by a JSON entry
- `src/ailm/control.py` — `ControlHarness` and `GateVacuousError`; every gate requires positive AND negative control
- `src/ailm/corpus/cmudict.py` — CMUdict download/parse; 39-phoneme invariant; `_strip_comment` to avoid comment-token contamination
- `src/ailm/corpus/ud.py` — UD English EWT download/parse; `case_oracle_stats` (DAT/INS/LOC/ABL derived via deprel mapping — research claim, not ground truth)
- `src/ailm/generative/smoke.py` — pymdp smoke tests: `A_dependencies` factorization, `update_empirical_prior` coupling, no `update_A`, `infer_policies` output shape
- `tests/conftest.py` — `tiny_cmudict`, `claims_ledger_tmp`, `two_factor_agent` fixtures
- `tests/test_cases.py` — case role tests, GEN collision pinned, CEREBRUM mapping
- `tests/test_claims_ledger.py` — ledger record/retrieve/persist/scan
- `tests/test_control_harness.py` — harness sensitivity, vacuous error, negative control
- `tests/test_pymdp_smoke.py` — all pymdp smoke test assertions
- `tests/test_cmudict.py` — 39-phoneme invariant, comment stripping, negative control
- `tests/test_stage0_gate.py` — Stage 0 meta-gate (all Stage 0 components)
- `docs/design/gen_collision.md` — [GEN] collision resolution
- `docs/design/cerebrum_mapping.md` — CEREBRUM→pymdp mapping derivation
- `docs/design/version_discrepancy.md` — pymdp version notes
- `docs/design/no_embeddings.md` — DisCoCat without embeddings
- `docs/design/speech_boundary.md` — DSP/pymdp boundary
- `docs/design/generation_problem.md` — generation via policy selection
- `docs/design/architecture.md` — hierarchical architecture
- `docs/provenance/sources.md` — paper provenance

### Design Decisions

- **[GEN] collision**: `CaseRole.GEN` = linguistic Genitive; `CerebrumFunctionRole.GNV` = computational Generative model role
- **Word as path**: 135k-state flat factor for words is impossible (66.9 GiB B matrix); words are paths through a letter/phoneme trie
- **A_dependencies factorization**: flat joint A at word level = 70.8 GiB; factorized = 11.9 KiB — non-negotiable
- **No embeddings (corrected)**: the implementation uses discrete pymdp kernels
  and no learned semantic vectors. Kernel composition is not a DisCoCat functor;
  `FinStoch` does not supply the compact closure that such a claim requires.
- **CEREBRUM mapping**: Figure 13 chosen as most implementable among conflicting mappings in paper
- **Speech boundary**: DSP (Whisper STT / TTS) is a pre/post-processing wrapper; pymdp starts at phoneme level
- **UD oracle**: Only Nom/Acc/Gen have morphological marking in English EWT; DAT/INS/LOC/ABL derived via deprel — research claim, not ground truth
