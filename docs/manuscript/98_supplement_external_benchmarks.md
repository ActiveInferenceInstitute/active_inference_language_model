# Supplement D: Non-Comparative Reference Probes on the Same Next-Symbol Tasks {#sec:external_benchmarks}

## Reference-probe scope and limits on capability comparisons

This section is a non-comparative accessory reference frame, confined to this
supplement exactly as [@sec:introduction] frames it: the explicit discrete model, connected evaluation and gate-audit-and-claims-ledger
method do not depend on anything measured here. It reports what externally served
zero-shot models return on the same next-symbol items the comparator is scored
on, under one declared protocol, so that a reader has a point of reference for
the units this paper reports in. The paired task-specific contrasts below are comparisons under the recorded
protocol. Here, “non-comparative” means that they do not support a general
capability ranking of the systems: the conditions such a ranking would need
are enumerated below, and several are not met. The supplement therefore reports
bounded probe readings, with the fitting, interface and population differences
kept visible.

AILM is an order-one discrete pymdp system with no web-scale pretraining and no
learned embeddings. Its parameters are estimated from the vendored Brown and
CMUdict artifacts, and each AILM number below is traceable to the corresponding
matrix construction. {{BENCH_MODEL_GEMMA3_DISPLAY}} (`{{BENCH_MODEL_GEMMA3_TAG}}`), {{BENCH_MODEL_QWEN25_DISPLAY}} (`{{BENCH_MODEL_QWEN25_TAG}}`), and
{{BENCH_MODEL_SMOLLM_DISPLAY}} (`{{BENCH_MODEL_SMOLLM_TAG}}`) are
externally served zero-shot references with fundamentally different training
histories and objectives. {{BENCH_INSTRUCT_PROBE_COUNT}} of the
{{BENCH_PROBE_COUNT}} registered reference models are instruction-tuned chat
models; the remaining {{BENCH_BASE_PROBE_COUNT_WORD}} is a non-instruct base
model queried with its raw completion template, so
even the registered probes do not share one training objective — one more
reason the section ranks nothing. We therefore expect the external models to be stronger
on raw next-symbol accuracy; that expectation is why the section is framed as a
reference frame rather than as a competitiveness claim.

## Requirements unmet by a capability comparison

Placing a fitted discrete matrix beside a prompted external model would be a
fair comparison only under conditions that have to be stated before any number
is read, because some of them hold here and some do not. They are collected here
rather than raised one at a time beside the arm each one qualifies, and the ones
that fail are why the section is labeled non-comparative rather than merely
hedged.

| Condition a fair comparison would need | Status in this section |
|---|---|
| Both systems answer the same items, under the same scoring rule, on the same denominator | Holds, with one declared exception on G2P (position-aligned letter-to-phone) alignment, which is consequently reported as two separate readings rather than as one row |
| The score measures the ability the task names | Fails on the full-inventory letter arm |
| Neither system has prior exposure to the evaluation corpus | Unverifiable for the external probes; the stipulated arm reduces exact-artifact overlap but cannot establish their training contents |
| Both systems have the same opportunity to fit the task | Does not hold in any arm as run; the in-context arm below supplies order-one transition examples from the comparator's training source and measures that bounded intervention rather than claiming equal opportunity |
| Both systems answer through the same interface | Does not hold in any arm |
| The comparison can resolve the contrasts it draws | Holds in some arms and not in others |

The first condition is the one the protocol was built to satisfy. Every system
is scored on the same positions, against the same declared inventory, under the
same treatment of invalid answers: an invalid answer is wrong, stays in the
denominator, and is never re-prompted. Each contrast is also reported as a
paired difference on identical positions rather than as a gap between two
independently sampled accuracies. The sampling, deduplication and splitting
rules this rests on are stated in the protocol below.

One quantity resists that condition, and the section's response is structural
rather than a caveat. G2P position accuracy is alignment-dependent: the
comparator is scored by strict positional alignment and each probe by
Levenshtein alignment, which are two different matching operators and therefore
two different questions. A single row carrying both would be a comparison whose
systems were not scored by the same rule, and disclosing that in a footnote does
not repair it. The arm is consequently reported as two readings — a Levenshtein
reading, which every system carries, and a strict-alignment reading, which only
the comparator carries — and neither reading is combined with the other.
G2P exact match is not alignment-dependent, since it is equality of the whole
predicted phone sequence with the gold one under either rule; it is the quantity
the paired G2P contrast is taken on.

The second condition fails on one arm, and it fails in a way the section
measures rather than assumes. On the full-inventory letter task a system's score
depends in part on whether it will emit a non-alphabetic symbol at all — the
scored inventory contains a space, and space is the most frequent symbol in the
stream — which is a fact about what a system is willing to produce rather than
about next-symbol prediction. The alphabetic-target arm below separates the two,
and both arms are scored from the same calls, so the difference between them is
attributable to the scored positions alone.

The third condition fails on the English arms, and the two sides fail it
differently. Brown and CMUdict are longstanding, widely distributed resources
and may have occurred in external-model training data; Brown's upstream NLTK
metadata separately restricts it to non-commercial use ([@sec:reproducibility]).
The benchmark is consequently not a
contamination-controlled comparison and should not be interpreted as a test of
generalization from an unseen corpus. The comparator's exposure to those
corpora is of a different kind: its parameters are estimated from them by
construction, and that exposure is declared and bounded by a training split
every scored position is held out from. Whatever exposure an external model has
is undeclared, unbounded, and unmeasurable from here, and if present it would
act in that model's favor. The stipulated-language arm below instead uses a
locally generated, non-natural corpus designed to reduce exact-corpus overlap. It
does not establish the contents of any external model's pretraining data.

The fourth condition does not hold anywhere, including that synthetic arm.
AILM's matrices are estimated from in-domain training counts, while each
external model has no offline fit and no separately labeled demonstrations. Its
query is not observation-free: it contains {{BENCH_SYNTH_QUERY_WINDOW}} symbols
from the held-out chain and can expose up to
{{BENCH_SYNTH_QUERY_CONTEXT_TRANSITIONS_MAX}} adjacent transitions. A difference
in either direction therefore confounds whatever separates the two kinds of
system with unequal fitting opportunities. The stipulated-language arm reduces
the exact-corpus-overlap concern and leaves this asymmetry intact.

The in-context arm below tests one bounded remediation: it supplies the same
kind of labeled pair the comparator's counts are estimated from, sweeps the
number supplied, and compares each treatment with a character-level
prefix-and-label-multiset-matched derangement. A failure to exploit that particular format cannot establish that
the fitted-versus-unfitted asymmetry is harmless. It reports how the registered
probes respond under the declared prompt and grid, while the broader asymmetry
remains unresolved.

The fifth condition does not hold either, and it is the one most easily mistaken
for a property of a model. The AILM comparator is a deterministic matrix rather
than a prompted system. Its answer is the argmax of a transition column, so it
has no prompt, no output format to comply with, and no sensitivity to how the
question is worded. Each external model answers a prompt written here, under a
response contract declared here. The wording, the delimiters, the named
inventory and the prefix width are free parameters on one side of the comparison
and absent on the other. They were fixed once and their effect measured rather
than tuned, which is what the prompt-format and context-window arms below
report; no search was run for a prompt that raises a probe's score.

The sixth condition holds unevenly, and the section says in which arms it holds.
Each paired contrast is corrected inside a declared family, and the
detectability of the effect a contrast could have found is reported beside it. A
contrast that fails to clear the corrected threshold at low power is unresolved
rather than shown to be absent, and that reading is stated at each such arm
rather than left to the reader.

The local external models registered for the English and aligned-pronunciation
arms are separate families, not rungs of one scale ladder. Hosted synthetic and
in-context labels instead identify repeated run cohorts of one served model,
as disclosed beside those experiments. Every pairwise difference belongs to the specific named models
and nothing more: no trend across parameter count is claimed, drawn, or implied,
and the families differ in training recipe as well as size, so even a "smaller
is worse" reading would be confounded.

Read under these conditions, the section supports statements of one shape: the
named systems answered the same items under a stated protocol, they differed by some
amount on those items, and some part of that difference the protocol cannot
attribute. It does not support a ranking of these systems as language models, a
trend across model families, a statement about text no system has seen, or a
claim that either kind of system would score as it does
here under a different prompt or a different inventory.

## Probe protocol, fixed positions, and declared seed

Every system is evaluated on the same fixed, evenly spaced positions rather than
a prefix. The protocol identifier is `{{BENCH_PROTOCOL_VERSION}}` and the seed is
{{BENCH_SEED}}. The evaluation covers {{BENCH_LETTER_POSITIONS}} letter
positions, {{BENCH_PHONEME_POSITIONS}} phoneme positions, and
{{BENCH_G2P_WORDS}} G2P words.

The letter, phoneme, and G2P arms are not the same kind of object and the difference has to be
stated before any interval below is read. The letter arm is a **census**: it
scores every distinct position in the held-out fixture, so
{{BENCH_LETTER_POSITIONS}} equals {{BENCH_LETTER_POOL_UNIQUE}} and the sampling
fraction is one. The phoneme and G2P arms are **samples**, of
{{BENCH_PHONEME_POSITIONS}} and {{BENCH_G2P_WORDS}} drawn from far larger
held-out splits.

What an interval means differs accordingly, and this section commits to one
reading. Nothing here is an inference about positions that were not scored,
because on the letter arm there are none. The intervals are *randomization*
statements, not sampling ones: the exact paired interval is conditional on the
observed number of discordant pairs and quantifies which system wins at each
discordant position under the null that the two are exchangeable there. That
null is well defined on a complete enumeration, which is why the letter arm can
carry an interval at all despite having no sampling variability. The per-group
Wilson intervals are the weaker case and are reported as descriptive precision
under the same exchangeability reading rather than as coverage statements about
a superpopulation of prose this project never defined. A reader who wants to
generalize from these numbers to English at large is doing something the design
does not license, on any arm.

Letters use a held-out prose fixture of {{BENCH_LETTER_POOL_TOTAL}} scored
positions, of which {{BENCH_LETTER_POOL_UNIQUE}} are distinct prefix-and-target
pairs. Positions are deduplicated before scoring, and every distinct pair is
then scored, so the count above is a count of distinct positions and not a draw
from them. The comparator is deterministic, and an ordinary cached reply repeats the
same stored observation. Repeated fixture positions therefore do not supply
independent evidence merely by appearing again; counting them as such could
narrow intervals without adding independent observations. Fresh service replies
can vary despite fixed requested settings, which is a separate source of
variation rather than a reason to count duplicated fixture positions as new
items. The deduplication is
retained as a guard even where the fixture contains no repeats, since it is the
sampling contract rather than a patch for one passage.

Phonemes use the CMUdict primary pronunciation stream restricted to the Stage 2
held-out word split, and the AILM phoneme comparator's bigram statistics are
recomputed from the training split alone. The vendored phone counts are pooled
over every CMUdict base word, which includes the words the task is scored on;
using them directly would fit the comparator on its own test set. G2P uses the
same Stage 2 held-out word split.

Each external model receives one zero-shot call per position at temperature zero
under a bounded JSON response contract. The prompt demands exactly one JSON
object with a single `answer` field, and Ollama constrains the answer slot with
a JSON schema (`format`) enumerating the letter alphabet or the
{{PHONEME_COUNT_39}}-symbol ARPAbet inventory. The vocabulary is therefore
enforced by the model's structured-output path, before any parser sees the text.
The letter prompt also names that inventory in words. It has to: the scored
alphabet includes a space, an apostrophe, a hyphen and a period alongside the
{{STAGE0_WORD_LETTERS}} letters, and a prompt that asks for a narrower set than
the scorer
accepts penalizes the probe for obeying it. The prompt also delimits the
sequence, so a prefix ending in a space remains visible to the model rather than
being trimmed away before the model reads the prefix. Letter and phoneme outputs
are scored as exact inventory symbols; a probe's G2P answer is scored by
Levenshtein-aligned position accuracy and by exact-sequence match, and the
comparator's is scored by both Levenshtein-aligned and strict positional
accuracy so that the alignment-dependent quantity can be read under either
rule. Any malformed JSON, extra
explanation, empty answer, or inventory violation is invalid; every invalid
answer counts as wrong, remains in the denominator, and is reported as a
diagnostic. No answer is ever re-prompted.

AILM is scored on these identical positions using its trained letter-bigram,
phoneme-bigram, and filtering G2P constructions. The external probes are not
part of the AILM generative-model stack.

This section is the one measurement in the paper that needs a live service.
`scripts/record_external_benchmark_claims.py` measures every registered model
and writes the `bench_` labels; `scripts/record_synthetic_benchmark_claims.py`
and `scripts/record_incontext_benchmark_claims.py` write the `bench_synth_`
and `bench_ctx_` labels for the stipulated-language probes below, scored
through the hosted OpenRouter transport. The external recorder requires the selected local Ollama endpoint and its
registered models. The synthetic and in-context recorders instead require the
configured hosted provider and `OPENROUTER_API_KEY`; their cohort tags do not
name local weights. The in-context recorder also authenticates its synthetic
producer receipt. Their preflight and complete-stratum checks refuse publication
when a required service, credential or producer is unavailable. Reproducing
this section therefore means pulling `{{BENCH_MODEL_GEMMA3_TAG}}`,
`{{BENCH_MODEL_QWEN25_TAG}}` and `{{BENCH_MODEL_SMOLLM_TAG}}` locally, exporting
the hosted credential, and rerunning the network phase through
`scripts/orchestrate_claim_recorders.py network` — which runs the external,
synthetic, in-context, aligned-G2P, and log-probability-drift recorders under
their single-writer lock; every other step in the
reproduction path is independent of the network.

## Letter, phoneme, and letter-to-phone probe outcomes

The table below carries the letter, phoneme, and G2P tasks, with G2P position
accuracy split into the alignment readings the arm supports. The subsections
then report the paired contrasts; letter-inventory, prompt-format, and
context-window controls; G2P length strata; a stipulated-language synthetic
corpus; its phoneme-inventory extension; the stipulated aligned
spelling-to-pronunciation comparator arm; calibration; the calibration
eligibility gate; protocol determinism; and the in-context arm.

| Task | AILM | {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_MODEL_SMOLLM_DISPLAY}} |
|---|---:|---:|---:|---:|
| Letter top-1 accuracy | {{BENCH_LETTER_TOP1_AILM}} ({{BENCH_LETTER_TOP1_WILSON_LOW_AILM}}–{{BENCH_LETTER_TOP1_WILSON_HIGH_AILM}}) | {{BENCH_LETTER_TOP1_GEMMA3}} ({{BENCH_LETTER_TOP1_WILSON_LOW_GEMMA3}}–{{BENCH_LETTER_TOP1_WILSON_HIGH_GEMMA3}}) | {{BENCH_LETTER_TOP1_QWEN25}} ({{BENCH_LETTER_TOP1_WILSON_LOW_QWEN25}}–{{BENCH_LETTER_TOP1_WILSON_HIGH_QWEN25}}) | {{BENCH_LETTER_TOP1_SMOLLM}} ({{BENCH_LETTER_TOP1_WILSON_LOW_SMOLLM}}–{{BENCH_LETTER_TOP1_WILSON_HIGH_SMOLLM}}) |
| Phoneme top-1 accuracy | {{BENCH_PHONEME_TOP1_AILM}} ({{BENCH_PHONEME_TOP1_WILSON_LOW_AILM}}–{{BENCH_PHONEME_TOP1_WILSON_HIGH_AILM}}) | {{BENCH_PHONEME_TOP1_GEMMA3}} ({{BENCH_PHONEME_TOP1_WILSON_LOW_GEMMA3}}–{{BENCH_PHONEME_TOP1_WILSON_HIGH_GEMMA3}}) | {{BENCH_PHONEME_TOP1_QWEN25}} ({{BENCH_PHONEME_TOP1_WILSON_LOW_QWEN25}}–{{BENCH_PHONEME_TOP1_WILSON_HIGH_QWEN25}}) | {{BENCH_PHONEME_TOP1_SMOLLM}} ({{BENCH_PHONEME_TOP1_WILSON_LOW_SMOLLM}}–{{BENCH_PHONEME_TOP1_WILSON_HIGH_SMOLLM}}) |
| G2P position accuracy, Levenshtein alignment | {{BENCH_G2P_POS_ACC_LEV_AILM}} (no interval recorded) | {{BENCH_G2P_POS_ACC_GEMMA3}} ({{BENCH_G2P_POS_ACC_WILSON_LOW_GEMMA3}}–{{BENCH_G2P_POS_ACC_WILSON_HIGH_GEMMA3}}) | {{BENCH_G2P_POS_ACC_QWEN25}} ({{BENCH_G2P_POS_ACC_WILSON_LOW_QWEN25}}–{{BENCH_G2P_POS_ACC_WILSON_HIGH_QWEN25}}) | {{BENCH_G2P_POS_ACC_SMOLLM}} ({{BENCH_G2P_POS_ACC_WILSON_LOW_SMOLLM}}–{{BENCH_G2P_POS_ACC_WILSON_HIGH_SMOLLM}}) |
| G2P position accuracy, strict positional alignment | {{BENCH_G2P_POS_ACC_AILM}} ({{BENCH_G2P_POS_ACC_WILSON_LOW_AILM}}–{{BENCH_G2P_POS_ACC_WILSON_HIGH_AILM}}) | not recorded | not recorded | not recorded |
| G2P exact-match rate | {{BENCH_G2P_EXACT_AILM}} ({{BENCH_G2P_EXACT_WILSON_LOW_AILM}}–{{BENCH_G2P_EXACT_WILSON_HIGH_AILM}}) | {{BENCH_G2P_EXACT_GEMMA3}} ({{BENCH_G2P_EXACT_WILSON_LOW_GEMMA3}}–{{BENCH_G2P_EXACT_WILSON_HIGH_GEMMA3}}) | {{BENCH_G2P_EXACT_QWEN25}} ({{BENCH_G2P_EXACT_WILSON_LOW_QWEN25}}–{{BENCH_G2P_EXACT_WILSON_HIGH_QWEN25}}) | {{BENCH_G2P_EXACT_SMOLLM}} ({{BENCH_G2P_EXACT_WILSON_LOW_SMOLLM}}–{{BENCH_G2P_EXACT_WILSON_HIGH_SMOLLM}}) |

The Levenshtein and strict-alignment rows of that table are incomplete, and in
opposite directions. The Levenshtein reading covers every system but carries an interval only for the
probes, because the comparator's Levenshtein score is recorded as a point
figure. The strict reading carries an interval but covers the comparator alone,
because no probe answer is scored under strict positional alignment anywhere in
the recorder. Completing either cell is a recording change rather than a
presentational one, and until it is made the alignment-dependent part of the
G2P arm has no single row a reader can rank systems in. What would close the
gap is a recorded strict-alignment position accuracy for each probe, scored on
the same gold-phone denominator the comparator uses.

The AILM comparator is a deterministic matrix-based reference, not a separately
prompted system, so its two readings can be compared with each other: they are
the same predictions scored twice. Under strict positional alignment it scores
{{BENCH_G2P_POS_ACC_AILM}} and under the probes' own alignment rule
{{BENCH_G2P_POS_ACC_LEV_AILM}}. Strict alignment is the lower of the two, so
wherever the two rules are still placed side by side — the G2P position panel of
[@fig:external_benchmarks] draws the comparator strictly and each probe by
Levenshtein — the headline comparison is the one less favorable to AILM. That
direction is why the arrangement was defensible, and it is not why the row has
been split. It was split because a row scoring one system by one matching
operator and the others by another is not one axis whichever way the disclosure
runs.

The invalid/unparsable diagnostic recorded {{BENCH_UNPARSABLE_GEMMA3}} unparsable
answers for {{BENCH_MODEL_GEMMA3_DISPLAY}}, {{BENCH_UNPARSABLE_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_UNPARSABLE_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}} out of
{{BENCH_UNPARSABLE_DENOMINATOR}} calls each (invalid rates
{{BENCH_UNPARSABLE_RATE_GEMMA3}}, {{BENCH_UNPARSABLE_RATE_QWEN25}} and
{{BENCH_UNPARSABLE_RATE_SMOLLM}}) across
the letter, phoneme, and G2P tasks. The Wilson intervals in [@fig:external_benchmarks] are
proportion intervals computed on the declared denominator; they do not drop
invalid answers.

### Paired differences and multiplicity correction

The per-group intervals above answer a different question than the paired
comparison does, so each contrast is also reported on the paired scale: the
difference in accuracy on identical positions, with an exact interval built
from the discordant pairs. That interval is *conditional on the observed number
of discordant pairs* — it is the standard conditional McNemar interval, not a
marginal interval for the accuracy difference, and it is conservative for the
same reason it is exact.

| Contrast | b/c | Paired difference (probe minus AILM) | Exact {{BOUND_CONFIDENCE_PCT}}% interval | p |
|---|---:|---:|---:|---:|
| Letter, {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_LETTER_PAIRED_B_GEMMA3}}/{{BENCH_LETTER_PAIRED_C_GEMMA3}} | {{BENCH_LETTER_DELTA_GEMMA3}} | {{BENCH_LETTER_DELTA_LOW_GEMMA3}}–{{BENCH_LETTER_DELTA_HIGH_GEMMA3}} | {{BENCH_LETTER_PAIRED_P_GEMMA3}} |
| Letter, {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_LETTER_PAIRED_B_QWEN25}}/{{BENCH_LETTER_PAIRED_C_QWEN25}} | {{BENCH_LETTER_DELTA_QWEN25}} | {{BENCH_LETTER_DELTA_LOW_QWEN25}}–{{BENCH_LETTER_DELTA_HIGH_QWEN25}} | {{BENCH_LETTER_PAIRED_P_QWEN25}} |
| Letter, {{BENCH_MODEL_SMOLLM_DISPLAY}} | {{BENCH_LETTER_PAIRED_B_SMOLLM}}/{{BENCH_LETTER_PAIRED_C_SMOLLM}} | {{BENCH_LETTER_DELTA_SMOLLM}} | {{BENCH_LETTER_DELTA_LOW_SMOLLM}}–{{BENCH_LETTER_DELTA_HIGH_SMOLLM}} | {{BENCH_LETTER_PAIRED_P_SMOLLM}} |
| Phoneme, {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_PHONEME_PAIRED_B_GEMMA3}}/{{BENCH_PHONEME_PAIRED_C_GEMMA3}} | {{BENCH_PHONEME_DELTA_GEMMA3}} | {{BENCH_PHONEME_DELTA_LOW_GEMMA3}}–{{BENCH_PHONEME_DELTA_HIGH_GEMMA3}} | {{BENCH_PHONEME_PAIRED_P_GEMMA3}} |
| Phoneme, {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_PHONEME_PAIRED_B_QWEN25}}/{{BENCH_PHONEME_PAIRED_C_QWEN25}} | {{BENCH_PHONEME_DELTA_QWEN25}} | {{BENCH_PHONEME_DELTA_LOW_QWEN25}}–{{BENCH_PHONEME_DELTA_HIGH_QWEN25}} | {{BENCH_PHONEME_PAIRED_P_QWEN25}} |
| Phoneme, {{BENCH_MODEL_SMOLLM_DISPLAY}} | {{BENCH_PHONEME_PAIRED_B_SMOLLM}}/{{BENCH_PHONEME_PAIRED_C_SMOLLM}} | {{BENCH_PHONEME_DELTA_SMOLLM}} | {{BENCH_PHONEME_DELTA_LOW_SMOLLM}}–{{BENCH_PHONEME_DELTA_HIGH_SMOLLM}} | {{BENCH_PHONEME_PAIRED_P_SMOLLM}} |
| G2P exact, {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_G2P_PAIRED_B_GEMMA3}}/{{BENCH_G2P_PAIRED_C_GEMMA3}} | {{BENCH_G2P_DELTA_GEMMA3}} | {{BENCH_G2P_DELTA_LOW_GEMMA3}}–{{BENCH_G2P_DELTA_HIGH_GEMMA3}} | {{BENCH_G2P_PAIRED_P_GEMMA3}} |
| G2P exact, {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_G2P_PAIRED_B_QWEN25}}/{{BENCH_G2P_PAIRED_C_QWEN25}} | {{BENCH_G2P_DELTA_QWEN25}} | {{BENCH_G2P_DELTA_LOW_QWEN25}}–{{BENCH_G2P_DELTA_HIGH_QWEN25}} | {{BENCH_G2P_PAIRED_P_QWEN25}} |
| G2P exact, {{BENCH_MODEL_SMOLLM_DISPLAY}} | {{BENCH_G2P_PAIRED_B_SMOLLM}}/{{BENCH_G2P_PAIRED_C_SMOLLM}} | {{BENCH_G2P_DELTA_SMOLLM}} | {{BENCH_G2P_DELTA_LOW_SMOLLM}}–{{BENCH_G2P_DELTA_HIGH_SMOLLM}} | {{BENCH_G2P_PAIRED_P_SMOLLM}} |

The G2P rows are the exact-match contrast rather than the position contrast,
which is what makes them alignment-free: exact match is equality of the whole
predicted sequence with the gold one, and the two alignment rules agree on it by
construction. No paired contrast is taken on G2P position accuracy, because the
two systems are not scored on it under the same rule.

These paired tests are not members of the repository-wide significance family;
they concern an external reference point rather than any hypothesis this project
advances. They are corrected within their own declared family of
{{BENCH_FAMILY_SIZE}} tests at a Bonferroni threshold of
{{BENCH_FAMILY_ALPHA}}, which is the level any claim of a real difference here
must clear. The membership rule is {{BENCH_FAMILY_CONTRASTS_PER_PROBE_WORD}}
contrast types on each of the {{BENCH_PROBE_COUNT_WORD}} probes:
the letter contrast, the alphabetic-only letter contrast, the prompt-format
contrast, the phoneme contrast, the G2P exact-match contrast, and the
context-window contrast. The letter, phoneme, and G2P exact-match contrasts are
the rows above; the alphabetic-only, prompt-format, and context-window contrasts
are introduced below, so the family is larger than the table it is first stated
beside.

Detectability is reported at the same fixed reference discordant share of
{{BENCH_POWER_REFERENCE_SHARE}} as elsewhere in this paper, rather than at the
observed effect, since power at the observed effect is a monotone transform of
the p-value and carries no additional evidence; unlike the repository family,
it is reported at the uncorrected {{BENCH_POWER_ALPHA}} (see below). At significance
level {{BENCH_POWER_ALPHA}} the recorded pair counts give power
{{BENCH_LETTER_POWER_GEMMA3}} for the {{BENCH_MODEL_GEMMA3_DISPLAY}} letter contrast,
{{BENCH_PHONEME_POWER_GEMMA3}} for phonemes, and {{BENCH_G2P_POWER_GEMMA3}} for
G2P exact match; the corresponding {{BENCH_MODEL_QWEN25_DISPLAY}} values are
{{BENCH_LETTER_POWER_QWEN25}}, {{BENCH_PHONEME_POWER_QWEN25}} and
{{BENCH_G2P_POWER_QWEN25}}; and the corresponding {{BENCH_MODEL_SMOLLM_DISPLAY}}
values are {{BENCH_LETTER_POWER_SMOLLM}}, {{BENCH_PHONEME_POWER_SMOLLM}} and
{{BENCH_G2P_POWER_SMOLLM}}. A contrast whose interval spans zero at a power
value well below one is unresolved at this sample size, not shown to be absent.

Power is recorded for the letter, phoneme and G2P arms and for no others. The
alphabetic, prompt-format and context-window arms have their own discordant
totals and therefore their own detectability, and no recorder writes a power
value for any of them. Power at a fixed reference share is a function of the
discordant counts alone, and those counts are recorded, so the missing figures
are derived from them rather than remeasured. The derivation uses the same
reference share of {{BENCH_POWER_REFERENCE_SHARE}} and the same significance
level {{BENCH_POWER_ALPHA}} as the recorded values, and a test applies it to the
arms that *do* carry a recorded power and requires it to reproduce each recorded
value and its rendered text exactly; agreement there is what licenses the
derived figures for the arms where no recorded value exists to check against.
That test also fails if a recorder ever starts writing those labels, so the
derivation cannot silently become a second producer of a measured quantity. The
largest of the {{BENCH_DERIVED_POWER_COUNT_WORD}} derived values is
{{BENCH_LETTER_AZ_POWER_SMOLLM}}, and every remaining derived value is no
larger; the largest sits on the base model's alphabetic arm, whose discordant
total is the largest of the {{BENCH_DERIVED_POWER_COUNT_WORD}} arms. At the census pool these arms are no longer uniformly
underpowered; a contrast that fails to clear the corrected threshold is
unresolved rather than absent whatever its power figure ([@sec:bounds]), and the
per-arm values below are what a reader checks. No power figure is carried across
from one arm to another.

Those power values are computed at {{BENCH_POWER_ALPHA}}, while the decision
rule these tests are actually judged against is the corrected family threshold
of {{BENCH_FAMILY_ALPHA}}. Power falls as the threshold tightens, so every power
figure in this section, recorded or derived, is an upper bound on the
detectability of the rule actually in force. The gap between the two is wide
rather than incidental: at the letter arm's discordance rate, a pair count that
would be well powered at {{BENCH_POWER_ALPHA}} loses a substantial part of that
power at {{BENCH_FAMILY_ALPHA}}. Both thresholds are named because a reader who
carried a power figure across from one to the other would misstate the design
substantially in whichever direction they moved.

For the letter task the pool bound was structural in the earlier revision: the
fixture then held only a few hundred distinct positions, so no selection inside
it could reach a reference-sized effect at this alpha. The fixture in this
revision is a substantially longer non-repeating public-domain passage holding
{{BENCH_LETTER_POOL_UNIQUE}} distinct positions, above the number the reference
share needs at {{BENCH_POWER_ALPHA}}. The letter contrast is therefore no longer
unresolved by construction: whether it resolves depends on the measured pair
counts and the power recorded above.

### Letter-inventory choices and non-equivalent readings

"Predict the next letter" has full-inventory and alphabetic-only readings, and
they are not equivalent. The corpus stream contains {{STAGE0_LETTER_ALPHABET_SIZE}} symbols —
the {{STAGE0_WORD_LETTERS}} alphabetic ones plus space, apostrophe, hyphen and period — and
space is its most frequent single symbol. Scoring against the full inventory
asks a probe to produce symbols a reader might not think of as letters at all;
scoring against the alphabetic ones alone discards the most frequent symbol in
the stream. Either choice can move the comparison, so both are reported, from
the same calls: the alphabetic arm is a restriction of the recorded per-position
outcomes, not a second run, and the comparator's prediction rule is unchanged
between them — only the scored positions differ.

The full-inventory and alphabetic-only arms do not agree, and the reason is not predictive skill. Of the
{{BENCH_LETTER_POSITIONS}} scored positions, {{BENCH_LETTER_NONALPHA_POSITIONS}}
have a non-alphabetic target, almost all of them a space. Accuracy on exactly
those positions is {{BENCH_LETTER_NONALPHA_ACC_AILM}} for AILM,
{{BENCH_LETTER_NONALPHA_ACC_GEMMA3}} for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_LETTER_NONALPHA_ACC_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_LETTER_NONALPHA_ACC_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}. Two of the
probes score on almost none of them and the remaining one on most, and the gap
between those readings is far larger than any accuracy difference the
alphabetic arm shows. That the probes at the floor are of both kinds — one instruct, one
base — says the floor is about emission behavior rather than instruction
tuning: a base model withholds a space when the stream has one, exactly as the
instruct one does. The prompt names the full
inventory and the response schema enumerates it, so the spread is a difference
in what each model is willing to emit, not in what it was permitted to emit.

The full-inventory letter arm therefore measures inventory compliance more than
next-symbol prediction, and its ranking should be read that way. The alphabetic
arm removes the behavior, because on those positions no system can score by
emitting a space.

On the {{BENCH_LETTER_AZ_POSITIONS}} positions whose target is alphabetic, the
systems score as follows, with each probe's discordant pairs and p-value taken
against AILM on the same positions.

| System | Top-1 on alphabetic targets [descriptive {{BOUND_CONFIDENCE_PCT}}% Wilson] | b/c vs AILM | p |
|---|---:|---:|---:|
| AILM | {{BENCH_LETTER_AZ_TOP1_AILM}} ({{BENCH_LETTER_AZ_TOP1_WILSON_LOW_AILM}}–{{BENCH_LETTER_AZ_TOP1_WILSON_HIGH_AILM}}) | — | — |
| {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_LETTER_AZ_TOP1_GEMMA3}} ({{BENCH_LETTER_AZ_TOP1_WILSON_LOW_GEMMA3}}–{{BENCH_LETTER_AZ_TOP1_WILSON_HIGH_GEMMA3}}) | {{BENCH_LETTER_AZ_PAIRED_B_GEMMA3}}/{{BENCH_LETTER_AZ_PAIRED_C_GEMMA3}} | {{BENCH_LETTER_AZ_PAIRED_P_GEMMA3}} |
| {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_LETTER_AZ_TOP1_QWEN25}} ({{BENCH_LETTER_AZ_TOP1_WILSON_LOW_QWEN25}}–{{BENCH_LETTER_AZ_TOP1_WILSON_HIGH_QWEN25}}) | {{BENCH_LETTER_AZ_PAIRED_B_QWEN25}}/{{BENCH_LETTER_AZ_PAIRED_C_QWEN25}} | {{BENCH_LETTER_AZ_PAIRED_P_QWEN25}} |
| {{BENCH_MODEL_SMOLLM_DISPLAY}} | {{BENCH_LETTER_AZ_TOP1_SMOLLM}} ({{BENCH_LETTER_AZ_TOP1_WILSON_LOW_SMOLLM}}–{{BENCH_LETTER_AZ_TOP1_WILSON_HIGH_SMOLLM}}) | {{BENCH_LETTER_AZ_PAIRED_B_SMOLLM}}/{{BENCH_LETTER_AZ_PAIRED_C_SMOLLM}} | {{BENCH_LETTER_AZ_PAIRED_P_SMOLLM}} |

Comparing these with the full-inventory row of the table above is the point of
reporting both: a letter conclusion that holds under one inventory and not under
the other is a conclusion about the inventory, and reporting only the arm that
favors either system would be a choice presented as a measurement.

Read against the declared threshold of {{BENCH_FAMILY_ALPHA}}, every
alphabetic-arm contrast clears it, and the discordant pairs run in the
comparator's favor in every one. Removing the positions where symbol
compliance can act therefore does not remove the difference: it survives on the
alphabetic targets alone. The alphabetic arms are well powered at the
reporting level {{BENCH_POWER_ALPHA}}: the derived power at the reference
share is {{BENCH_LETTER_AZ_POWER_GEMMA3}} for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_LETTER_AZ_POWER_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_LETTER_AZ_POWER_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}, so each
arm could reliably have found a reference-sized effect at that level, and the
contrasts clear the corrected threshold on effects larger than that share.
Power at the corrected threshold {{BENCH_FAMILY_ALPHA}} is lower, as it is for
every arm in this section, and a contrast that had failed to clear it here
would have been unresolved rather than absent. {{BENCH_MODEL_QWEN25_DISPLAY}}
remains the extreme case in both directions, scoring highest of the probes on
non-alphabetic targets and lowest once they are excluded, which is the
compliance behavior seen from the other side rather than a separate effect.

The letter task is the only arm here carrying an explicit prompt-sensitivity
control. Naming the inventory in the prompt and delimiting the sequence are each
necessary for the task to be answerable as scored, and each changes what the
probes emit. No letter number in this section should be treated as a stable
property of a model.

### Prompt format as a paired diagnostic, not a second accuracy headline

The corrected letter protocol makes the sequence boundary explicit, but that
choice is itself part of the measurement. The prompt-sensitivity arm holds the
positions, inventory, response schema and seed fixed while removing the angle
brackets and the explicit trailing-space notice. It is therefore not a second
accuracy headline: it is a paired diagnostic of whether the answer changes when
the formatting cue changes.

Under the delimited protocol, the external systems score their ordinary letter
accuracy; under the otherwise identical undelimited protocol, {{BENCH_MODEL_GEMMA3_DISPLAY}} scores
{{BENCH_LETTER_UNDELIMITED_TOP1_GEMMA3}} ({{BENCH_LETTER_UNDELIMITED_WILSON_LOW_GEMMA3}}–{{BENCH_LETTER_UNDELIMITED_WILSON_HIGH_GEMMA3}}) and
{{BENCH_MODEL_QWEN25_DISPLAY}} scores {{BENCH_LETTER_UNDELIMITED_TOP1_QWEN25}} ({{BENCH_LETTER_UNDELIMITED_WILSON_LOW_QWEN25}}–{{BENCH_LETTER_UNDELIMITED_WILSON_HIGH_QWEN25}}), while
{{BENCH_MODEL_SMOLLM_DISPLAY}} scores {{BENCH_LETTER_UNDELIMITED_TOP1_SMOLLM}} ({{BENCH_LETTER_UNDELIMITED_WILSON_LOW_SMOLLM}}–{{BENCH_LETTER_UNDELIMITED_WILSON_HIGH_SMOLLM}}). The paired
undelimited-minus-delimited contrasts are
{{BENCH_LETTER_PROMPT_DELTA_GEMMA3}} for {{BENCH_MODEL_GEMMA3_DISPLAY}}
({{BENCH_LETTER_PROMPT_PAIRED_B_GEMMA3}}/{{BENCH_LETTER_PROMPT_PAIRED_C_GEMMA3}},
p = {{BENCH_LETTER_PROMPT_PAIRED_P_GEMMA3}}),
{{BENCH_LETTER_PROMPT_DELTA_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}
({{BENCH_LETTER_PROMPT_PAIRED_B_QWEN25}}/{{BENCH_LETTER_PROMPT_PAIRED_C_QWEN25}},
p = {{BENCH_LETTER_PROMPT_PAIRED_P_QWEN25}}), and
{{BENCH_LETTER_PROMPT_DELTA_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}
({{BENCH_LETTER_PROMPT_PAIRED_B_SMOLLM}}/{{BENCH_LETTER_PROMPT_PAIRED_C_SMOLLM}},
p = {{BENCH_LETTER_PROMPT_PAIRED_P_SMOLLM}}). These values quantify protocol
sensitivity; they do not establish that either prompt is intrinsically more
correct. The registered probes now separate. The {{BENCH_MODEL_GEMMA3_DISPLAY}} contrast clears the
corrected family threshold of {{BENCH_FAMILY_ALPHA}}, so that probe's prompt
sensitivity is a resolved effect rather than a nominal one; the {{BENCH_MODEL_QWEN25_DISPLAY}}
and {{BENCH_MODEL_SMOLLM_DISPLAY}} contrasts do not clear it and stay
unresolved. The derived power at the
reference share is {{BENCH_LETTER_PROMPT_POWER_GEMMA3}} for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_LETTER_PROMPT_POWER_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_LETTER_PROMPT_POWER_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}.
This arm is the clearest case in the section of a contrast that the earlier
sample could not resolve. The pre-census effect estimates are not recoverable
from a versioned recorder artifact, so they are not quantified here; the
reproducible claim is that the complete recorded pool resolves the
{{BENCH_MODEL_GEMMA3_DISPLAY}} contrast and not the
{{BENCH_MODEL_QWEN25_DISPLAY}} or {{BENCH_MODEL_SMOLLM_DISPLAY}} contrasts.
Reading the earlier null as evidence that the formatting cue went unused would
have been wrong, and the interval convention of [@sec:bounds] is what kept that
reading off the page.
The answer-versus-encoding separation therefore has complementary controls: the
alphabetic/full-inventory contrast separates symbol compliance from alphabetic
prediction, and the paired prompt-format contrast separates prediction from encoding.

### Wider prefixes and remaining fit asymmetry

The letter task is the one axis where extra context is available to a
transformer and unavailable to AILM by construction: the comparator is order
one, so it conditions on the immediately preceding symbol and nothing else, and
its accuracy is identical at every window. Each probe was therefore rerun on the
same positions with the prefix widened from {{BENCH_LETTER_WINDOW_SHORT}} to
{{BENCH_LETTER_WINDOW_LONG}} characters. The protocol rejects a window that
reaches the passage's whole-string repetition period; this guards against a
fixture made by repeating the same passage. The current fixture does not repeat
as a whole, so that guard cannot fire at any available prefix length. Local
repetition can still put an earlier instance of the target after verbatim
context inside the wider window. The period check neither measures nor bounds
that source of literal answer availability.

| System | Short window | Long window | b/c | p |
|---|---:|---:|---:|---:|
| AILM | {{BENCH_LETTER_SHORT_TOP1_AILM}} | {{BENCH_LETTER_LONG_TOP1_AILM}} | — | — |
| {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_LETTER_SHORT_TOP1_GEMMA3}} | {{BENCH_LETTER_LONG_TOP1_GEMMA3}} | {{BENCH_LETTER_CTX_PAIRED_B_GEMMA3}}/{{BENCH_LETTER_CTX_PAIRED_C_GEMMA3}} | {{BENCH_LETTER_CTX_PAIRED_P_GEMMA3}} |
| {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_LETTER_SHORT_TOP1_QWEN25}} | {{BENCH_LETTER_LONG_TOP1_QWEN25}} | {{BENCH_LETTER_CTX_PAIRED_B_QWEN25}}/{{BENCH_LETTER_CTX_PAIRED_C_QWEN25}} | {{BENCH_LETTER_CTX_PAIRED_P_QWEN25}} |
| {{BENCH_MODEL_SMOLLM_DISPLAY}} | {{BENCH_LETTER_SHORT_TOP1_SMOLLM}} | {{BENCH_LETTER_LONG_TOP1_SMOLLM}} | {{BENCH_LETTER_CTX_PAIRED_B_SMOLLM}}/{{BENCH_LETTER_CTX_PAIRED_C_SMOLLM}} | {{BENCH_LETTER_CTX_PAIRED_P_SMOLLM}} |

The AILM row prints its accuracy twice so a reader can check the invariance
rather than take it on assurance. That equality is a property of an order-one
model, not a finding about context.

The {{BENCH_MODEL_GEMMA3_DISPLAY}} and {{BENCH_MODEL_QWEN25_DISPLAY}} contrasts clear
the corrected family threshold of {{BENCH_FAMILY_ALPHA}}; the
{{BENCH_MODEL_SMOLLM_DISPLAY}} contrast does not and is unresolved, with
derived power at the reference share of
{{BENCH_LETTER_CTX_POWER_GEMMA3}} for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_LETTER_CTX_POWER_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_LETTER_CTX_POWER_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}. The widened prefix changes
what these probes predict, and the direction is recorded in the table rather
than asserted here.
The comparison this licenses is narrow and worth stating as such. It establishes
that the probes are sensitive to the amount of context they are given, on
positions where AILM is invariant by construction. It does not establish that
the extra context is used *well*: a sensitivity is not an improvement, and the
accuracy columns are where a reader checks which it was.

### Letter-to-phone accuracy stratified by word length

G2P words are stratified by gold phone count into
{{BENCH_G2P_BUCKET_SHORT_WORDS}} short words
({{BENCH_G2P_BUCKET_SHORT_RANGE}}),
{{BENCH_G2P_BUCKET_MID_WORDS}} mid-length words
({{BENCH_G2P_BUCKET_MID_RANGE}}) and
{{BENCH_G2P_BUCKET_LONG_WORDS}} long words
({{BENCH_G2P_BUCKET_LONG_RANGE}}). Position accuracy by
bucket is:

| System | Short | Mid-length | Long |
|---|---:|---:|---:|
| AILM | {{BENCH_G2P_POS_SHORT_AILM}} ({{BENCH_G2P_POS_SHORT_WILSON_LOW_AILM}}–{{BENCH_G2P_POS_SHORT_WILSON_HIGH_AILM}}) | {{BENCH_G2P_POS_MID_AILM}} ({{BENCH_G2P_POS_MID_WILSON_LOW_AILM}}–{{BENCH_G2P_POS_MID_WILSON_HIGH_AILM}}) | {{BENCH_G2P_POS_LONG_AILM}} ({{BENCH_G2P_POS_LONG_WILSON_LOW_AILM}}–{{BENCH_G2P_POS_LONG_WILSON_HIGH_AILM}}) |
| {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_G2P_POS_SHORT_GEMMA3}} ({{BENCH_G2P_POS_SHORT_WILSON_LOW_GEMMA3}}–{{BENCH_G2P_POS_SHORT_WILSON_HIGH_GEMMA3}}) | {{BENCH_G2P_POS_MID_GEMMA3}} ({{BENCH_G2P_POS_MID_WILSON_LOW_GEMMA3}}–{{BENCH_G2P_POS_MID_WILSON_HIGH_GEMMA3}}) | {{BENCH_G2P_POS_LONG_GEMMA3}} ({{BENCH_G2P_POS_LONG_WILSON_LOW_GEMMA3}}–{{BENCH_G2P_POS_LONG_WILSON_HIGH_GEMMA3}}) |
| {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_G2P_POS_SHORT_QWEN25}} ({{BENCH_G2P_POS_SHORT_WILSON_LOW_QWEN25}}–{{BENCH_G2P_POS_SHORT_WILSON_HIGH_QWEN25}}) | {{BENCH_G2P_POS_MID_QWEN25}} ({{BENCH_G2P_POS_MID_WILSON_LOW_QWEN25}}–{{BENCH_G2P_POS_MID_WILSON_HIGH_QWEN25}}) | {{BENCH_G2P_POS_LONG_QWEN25}} ({{BENCH_G2P_POS_LONG_WILSON_LOW_QWEN25}}–{{BENCH_G2P_POS_LONG_WILSON_HIGH_QWEN25}}) |
| {{BENCH_MODEL_SMOLLM_DISPLAY}} | {{BENCH_G2P_POS_SHORT_SMOLLM}} ({{BENCH_G2P_POS_SHORT_WILSON_LOW_SMOLLM}}–{{BENCH_G2P_POS_SHORT_WILSON_HIGH_SMOLLM}}) | {{BENCH_G2P_POS_MID_SMOLLM}} ({{BENCH_G2P_POS_MID_WILSON_LOW_SMOLLM}}–{{BENCH_G2P_POS_MID_WILSON_HIGH_SMOLLM}}) | {{BENCH_G2P_POS_LONG_SMOLLM}} ({{BENCH_G2P_POS_LONG_WILSON_LOW_SMOLLM}}–{{BENCH_G2P_POS_LONG_WILSON_HIGH_SMOLLM}}) |

Each cell keeps the alignment rule its own pooled number uses, so the comparator
row is strict-aligned and the probe rows are Levenshtein-aligned. The table
therefore decomposes each of the two readings above by word length; it does not
put them on one scale, and a comparator cell should not be read against a probe
cell in the same column for the same reason the pooled row was split.

The pooled G2P number cannot distinguish a system that is uniformly accurate
from one that is accurate only on short transparent orthography, and the
stratification is reported so that distinction is visible. Every cell carries
the Wilson interval of its own bucket, so a column can be read for separation
and not only for order. Among the {{BENCH_PROBE_COUNT_WORD}} probes, which share one alignment
rule, the short and the mid-length columns separate all {{BENCH_PROBE_COUNT_WORD}} pairwise.
The long bucket holds far fewer words than either of the other two, and its column is
where that shows: the {{BENCH_MODEL_GEMMA3_DISPLAY}} and
{{BENCH_MODEL_QWEN25_DISPLAY}} intervals overlap there, so the long column
orders those two without separating them, while
{{BENCH_MODEL_SMOLLM_DISPLAY}} stays clear of both. A comparator cell carries
an interval on the same footing, but reading it against a probe cell in the
same column still crosses two alignment rules as well as two systems, so a
separation across those rows is not a like-for-like one.

### The stipulated-language control that reduces exact-artifact overlap

**Serving identity boundary.** These stipulated letter and phone requests use
`{{BENCH_SYNTH_OPENROUTER_MODEL}}` through the recorded provider route
`{{BENCH_SYNTH_OPENROUTER_PROVIDER}}`. Labels ending in “run” preserve the
registered cohort keys and identify separate fresh calls to that same hosted
model; they do not identify Gemma, Qwen, or SmolLM weights. The English and
aligned spelling-to-pronunciation comparisons retain their separately recorded
local-model identities. The in-context sweep below uses the same hosted model
with its own measured receipt. Requested and returned slug hashes do not
identify immutable serving weights; the observed provider and raw response
metadata are retained, and a missing system fingerprint supplies no weight pin.


The English corpora are public and may overlap an external model's training
data. The reference study therefore runs a separate probe on a locally generated,
non-natural language designed to reduce exact-artifact overlap. It does not
claim knowledge of an external model's training contents. The protocol
stipulates an order-one chain over {{BENCH_SYNTH_ALPHABET_SIZE}} symbols from
recorded corpus seed {{BENCH_SYNTH_CORPUS_SEED}}. Evaluation uses
{{BENCH_SYNTH_POSITIONS}} independently seeded trajectories under
`{{BENCH_SYNTH_POSITION_GENERATION}}`, beginning at seed
{{BENCH_SYNTH_POSITION_SEED}}. Each trajectory has
{{BENCH_SYNTH_TRAJECTORY_LENGTH}} symbols and contributes one scored target;
the target marginal's total-variation distance from stationarity is
{{BENCH_SYNTH_TARGET_START_TVD}}. AILM and every registered probe receive these
identical independent positions. A missing model row is a hard failure rather
than a partial family.

The corpus artifact is vendored and checksum-verified. The synthetic comparator
is fit only to its generated training counts. Every external probe has no
offline fit or separately labeled demonstration in this arm, but each query
carries {{BENCH_SYNTH_QUERY_WINDOW}} symbols and exposes at most
{{BENCH_SYNTH_QUERY_CONTEXT_TRANSITIONS_MAX}} within-query transitions. The
response opportunity contains exactly {{BENCH_SYNTH_RESPONSE_INVENTORY_SIZE}}
symbols, matching the stipulated alphabet. The recorded measurement protocol is
`{{BENCH_SYNTH_PROTOCOL_VERSION}}`; its cache identity is
`{{BENCH_SYNTH_CACHE_PROTOCOL}}`. Every scored request declares a context window of
{{BENCH_SYNTH_CONTEXT_WINDOW}} tokens and a
{{BENCH_SYNTH_RESPONSE_TOKENS}}-token response cap. The longest prompt has
{{BENCH_SYNTH_MAXIMUM_PROMPT_CHARACTERS}} ASCII characters in the preflight; the
server then reports an executed maximum of
{{BENCH_SYNTH_MAXIMUM_PROMPT_TOKENS}} prompt tokens. That measured count, the
response cap, and the recorded
{{BENCH_SYNTH_PROMPT_TOKEN_OVERHEAD_MARGIN}}-token safety margin fit inside the
declared context window. Publication
recording used {{BENCH_SYNTH_SCORED_CACHE_HITS}} scored cache hits and
{{BENCH_SYNTH_SCORED_CACHE_MISSES}} fresh scored calls.

| System | Top-1 [descriptive {{BOUND_CONFIDENCE_PCT}}% Wilson] | Probe − AILM [descriptive {{BOUND_CONFIDENCE_PCT}}% paired interval] | Probe-only / AILM-only | Paired p | Unparsable |
|---|---:|---:|---:|---:|---:|
{{BENCH_SYNTH_RESULTS_TABLE_ROWS}}

The uniform floor is {{BENCH_SYNTH_UNIFORM_BASELINE_DISPLAY}}. Discordance is
reported as positions correct only for the named probe followed by positions
correct only for AILM. {{BENCH_SYNTH_SCOPE_RESULT}} This is evidence about
transfer to a stipulated transition system, not a general language-capability
ranking; the synthetic task is deliberately narrow and its result should be
read alongside the calibration and held-out phoneme and G2P results.

These contrasts are corrected inside their own separately declared descriptive family of
{{BENCH_SYNTH_FAMILY_SIZE}} tests at {{BENCH_SYNTH_FAMILY_ALPHA_DISPLAY}}, not inside the
{{BENCH_FAMILY_SIZE}}-test external-reference family used above. The two are
separate experiments on separate corpora asking separate questions, so they are
corrected separately rather than pooled. The p-values reported here are to be read
against {{BENCH_SYNTH_FAMILY_ALPHA_DISPLAY}} and against no other threshold in this
section.

The query itself supplies chain evidence, so an audit-specified prefix-width arm
shortens the prefix to {{BENCH_SYNTH_SHORT_QUERY_WINDOW}} symbol, exposing at
most {{BENCH_SYNTH_SHORT_QUERY_CONTEXT_TRANSITIONS_MAX}} adjacent transitions,
while preserving every target position. It is a post-discovery descriptive
family, not a preregistered test. The hosted run makes no warm-up call under
`{{BENCH_SYNTH_WARMUP_POLICY}}`; each scored cell is requested fresh, and the full/short order follows
`{{BENCH_SYNTH_CONTEXT_ARM_ORDER}}` so arm identity is not server order.

| Probe | Short top-1 [descriptive {{BOUND_CONFIDENCE_PCT}}% Wilson] | Full top-1 | Full − short [descriptive {{BOUND_CONFIDENCE_PCT}}% paired interval] | Full-only / short-only | Paired p | Invalid short/full |
|---|---:|---:|---:|---:|---:|---:|
{{BENCH_SYNTH_CONTEXT_RESULTS_TABLE_ROWS}}

{{BENCH_SYNTH_CONTEXT_SCOPE_RESULT}} The prefix-width p-values are corrected
inside their own {{BENCH_SYNTH_CONTEXT_FAMILY_SIZE}}-member family at
{{BENCH_SYNTH_CONTEXT_FAMILY_ALPHA_DISPLAY}}. The intervals are descriptive,
not simultaneous familywise intervals. A non-separation leaves this protocol
contrast unresolved; it is not evidence that the exposed transitions have no
effect, and changing prefix width does not identify a unique causal "context
contribution."

### The same stipulation carried over to the phoneme inventory

The stipulated-control logic extends from letters to phones without new
network machinery: a second order-one chain is written down in code over the
canonical ARPAbet inventory — the phoneme level's exact
{{BENCH_SYNTHPHONE_RESPONSE_INVENTORY_SIZE}}-symbol set — with the letter
chain's mixture weights, its seeded sampler contract, and a vendored count
artifact generated by a named script from a recorded seed. The inventory
partition into {{BENCH_SYNTHPHONE_RESPONSE_INVENTORY_SIZE}} response symbols
is the phoneme task's own; the stipulated transition over it is what is new,
not the symbols. Training drew {{BENCH_SYNTHPHONE_CORPUS_SYMBOLS}} symbols
from seed {{BENCH_SYNTHPHONE_CORPUS_SEED}}; evaluation uses
{{BENCH_SYNTHPHONE_POSITIONS}} independently seeded trajectories under
`{{BENCH_SYNTHPHONE_POSITION_GENERATION}}`, beginning at seed
{{BENCH_SYNTHPHONE_POSITION_SEED}}, each trajectory
{{BENCH_SYNTHPHONE_TRAJECTORY_LENGTH}} symbols long with one scored target.
The target marginal's total-variation distance from stationarity is
{{BENCH_SYNTHPHONE_TARGET_START_TVD}}. The artifact reproduces byte-for-byte
from its generator before any claim is written, and the fitted comparator
must clear prospectively specified recorder refusals — four times the uniform
floor, twice a transposed-count fit, and twice an English-trained CMUdict
fit, all on the identical positions — before anything is recorded.

The probe contract is the existing exact phoneme task: the same prompt
builder, the {{BENCH_SYNTHPHONE_RESPONSE_INVENTORY_SIZE}}-symbol response
schema, and the same parser the external phoneme arm uses, with no changes to
the task layer. The full query exposes {{BENCH_SYNTHPHONE_QUERY_WINDOW}}
phones; the sensitivity arm reduces it to {{BENCH_SYNTHPHONE_SHORT_QUERY_WINDOW}},
with the same alternating arm order and no hosted or local warm-up under
`{{BENCH_SYNTHPHONE_WARMUP_POLICY}}`. Scored requests declare a context window of
{{BENCH_SYNTHPHONE_CONTEXT_WINDOW}} tokens, a
{{BENCH_SYNTHPHONE_RESPONSE_TOKENS}}-token response cap, and a
{{BENCH_SYNTHPHONE_PROMPT_TOKEN_OVERHEAD_MARGIN}}-token safety margin; the
longest prompt is {{BENCH_SYNTHPHONE_MAXIMUM_PROMPT_CHARACTERS}} characters
in preflight and {{BENCH_SYNTHPHONE_MAXIMUM_PROMPT_TOKENS}} prompt tokens as
server-reported. Publication recording used {{BENCH_SYNTHPHONE_SCORED_CACHE_HITS}}
scored cache hits and {{BENCH_SYNTHPHONE_SCORED_CACHE_MISSES}} fresh scored
calls. Its measurement protocol is `{{BENCH_SYNTHPHONE_PROTOCOL_VERSION}}`
over chain `{{BENCH_SYNTHPHONE_CHAIN_VERSION}}`, with cache identity
`{{BENCH_SYNTHPHONE_CACHE_PROTOCOL}}`.

| System | Top-1 [descriptive {{BOUND_CONFIDENCE_PCT}}% Wilson] | Probe − AILM [descriptive {{BOUND_CONFIDENCE_PCT}}% paired interval] | Probe-only / AILM-only | Paired p | Unparsable |
|---|---:|---:|---:|---:|---:|
{{BENCH_SYNTHPHONE_RESULTS_TABLE_ROWS}}

The prospectively specified prediction holds: the comparator fitted on the stipulated
corpus beats every registered probe, and the paired p-values clear their
{{BENCH_SYNTHPHONE_FAMILY_SIZE}}-member family threshold
{{BENCH_SYNTHPHONE_FAMILY_ALPHA_DISPLAY}} with orders of magnitude to spare. The
one-phone sensitivity arm scores
{{BENCH_SYNTHPHONE_SHORT_TOP1_GEMMA3}}, {{BENCH_SYNTHPHONE_SHORT_TOP1_QWEN25}},
and {{BENCH_SYNTHPHONE_SHORT_TOP1_SMOLLM}} respectively — at or below the
uniform floor of one in {{BENCH_SYNTHPHONE_RESPONSE_INVENTORY_SIZE}}, the
recorded finding that decision 108 pre-registered. The paired full-versus-short context results for this publication run are:
{{BENCH_SYNTHPHONE_CONTEXT_RESULT}} Paired p-values are
{{BENCH_SYNTHPHONE_CONTEXT_PAIRED_P_GEMMA3}},
{{BENCH_SYNTHPHONE_CONTEXT_PAIRED_P_QWEN25}}, and
{{BENCH_SYNTHPHONE_CONTEXT_PAIRED_P_SMOLLM}}, respectively. Which contrasts
clear is itself not run-stable at these cell sizes: the preceding publication
run of the same registered population recorded no contrast clearing
(decision 130). Any resolved contrast is therefore reported as that run's
outcome, not as a stable property of its cohort or the served model.
Previous one-observation changes on the live hosted transport have produced
conflicting full-versus-short point-estimate directions across consecutive
publication runs of the same registered population (decision 128). This section
therefore states the sensitivity comparison through the pre-registered floor
outcome and the validated corrected contrasts, without assuming a direction or
a resolved contrast for a future run.
The comparator-versus-probe contrasts are corrected inside their separately declared
{{BENCH_SYNTHPHONE_FAMILY_SIZE}}-member descriptive family at
{{BENCH_SYNTHPHONE_FAMILY_ALPHA_DISPLAY}}, with the sensitivity contrasts corrected
inside their own {{BENCH_SYNTHPHONE_CONTEXT_FAMILY_SIZE}}-member family at
{{BENCH_SYNTHPHONE_CONTEXT_FAMILY_ALPHA_DISPLAY}}; neither family pools with the
letter arm's. Decision 108 specifies the design prospectively within the same
development session, without an independently timestamped prior registration;
the findings remain descriptive.

This arm does not establish absent pretraining exposure. It reduces
exact-artifact overlap: the scored trajectories are generated locally under a
transition written down in code, disjoint from the training draw. It cannot
establish that an external model's undisclosed pretraining corpus lacks the
same sequences, a similar phone inventory, an alternating vowel/consonant
regularity, or a related transition rule — all generic properties an
English-exposed model has prior mass on. The result licenses a statement
about transfer to this stipulated structure by these registered probes under
this prompt contract, and nothing about general phonotactic competence or
about what any model saw in training.

### A stipulated aligned spelling-to-pronunciation correspondence (design B)

The letter arm and the phoneme arm above each stipulate an order-one
continuation chain. Protocol P2's
design B ([Protocol P2 aligned correspondence]({{PROTOCOL_P2_ALIGNED_CORRESPONDENCE_URL}})) stipulates something the pair-structured
tasks never had: a **per-letter correspondence** `T` from the letter inventory
to a fixed subset of the {{PHONEME_COUNT_39}}-symbol phone inventory, drawn by
a named seeded generator and frozen, together with train and test
spelling/pronunciation splits generated under it, into the vendored artifact
`vendor/synthetic_aligned_g2p_counts.json` (schema
`ailm-synthetic-aligned-g2p-v1`, digest
`{{BENCH_SYNTHG2P_COMPARATOR_ARTIFACT_SHA256}}`). The correspondence is
recorded in the artifact and is deliberately **not** an English
letter-to-phone reading: the inventories are the task's own, the stipulated
correspondence is new. Design B's erratum withdrew the generator-owned
"nothing to leak" reasoning, so the design states what it is instead of what
it is not: a stipulated correspondence that reduces exact-artifact overlap and
is never known to be free of exposure — the artifact makes no claim that the
probes' pretraining contents are known or absent, and unknown pretraining
exposure remains the standing limitation of the whole stipulated family.

The scored population is the artifact's test split:
{{BENCH_SYNTHG2P_COMPARATOR_TEST_WORD_COUNT}} test words,
{{BENCH_SYNTHG2P_COMPARATOR_TEST_POSITION_COUNT}} scored positions under the
correspondence's per-letter 1:1 alignment. The comparator arm is recorded
offline, before any probe call, and the engine is the shipped G2P stack refit
on the stipulated train pairs — `build_g2p_emission`,
`build_g2p_transition` through the train-only pronunciation bigrams, and
`build_g2p_start`, decoded by the shipped readout with default additive smoothing:
{{MODEL_DEFAULT_EMISSION_SMOOTHING}} for the emission and
{{MODEL_DEFAULT_SMOOTHING}} for transitions and the initial-state prior. The closed-form `T` lookup is a **separately named baseline**:
the trivial map is exactly right on `T`-generated data by construction, so it
is reported as a comparison and never presented as the shipped engine, and
the recorder's recorded tie verdict — the native engine
{{BENCH_SYNTHG2P_COMPARATOR_NATIVE_TIE_WORD}} the trivial map's predictions —
is what the panel states rather than a verdict re-derived in prose.

```{=latex}
\csname @ifclassloaded\endcsname{beamer}{}{\Needspace{12\baselineskip}}
```

| Arm | Position accuracy [descriptive {{BOUND_CONFIDENCE_PCT}}% Wilson] | Exact-word accuracy [descriptive {{BOUND_CONFIDENCE_PCT}}% Wilson] |
|---|---:|---:|
| Native stipulated fit | {{BENCH_SYNTHG2P_COMPARATOR_NATIVE_POSITION_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_NATIVE_POSITION_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_NATIVE_POSITION_WILSON_HIGH_PCT}}) | {{BENCH_SYNTHG2P_COMPARATOR_NATIVE_EXACT_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_NATIVE_EXACT_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_NATIVE_EXACT_WILSON_HIGH_PCT}}) |
| `T` lookup (the trivial map) | {{BENCH_SYNTHG2P_COMPARATOR_T_LOOKUP_POSITION_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_T_LOOKUP_POSITION_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_T_LOOKUP_POSITION_WILSON_HIGH_PCT}}) | {{BENCH_SYNTHG2P_COMPARATOR_T_LOOKUP_EXACT_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_T_LOOKUP_EXACT_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_T_LOOKUP_EXACT_WILSON_HIGH_PCT}}) |
| Context-free pronunciation bigram | {{BENCH_SYNTHG2P_COMPARATOR_CONTEXT_FREE_POSITION_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_CONTEXT_FREE_POSITION_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_CONTEXT_FREE_POSITION_WILSON_HIGH_PCT}}) | {{BENCH_SYNTHG2P_COMPARATOR_CONTEXT_FREE_EXACT_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_CONTEXT_FREE_EXACT_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_CONTEXT_FREE_EXACT_WILSON_HIGH_PCT}}) |
| A1: fit on English CMUdict pairs | {{BENCH_SYNTHG2P_COMPARATOR_A1_ENGLISH_POSITION_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_A1_ENGLISH_POSITION_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_A1_ENGLISH_POSITION_WILSON_HIGH_PCT}}) | {{BENCH_SYNTHG2P_COMPARATOR_A1_ENGLISH_EXACT_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_A1_ENGLISH_EXACT_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_A1_ENGLISH_EXACT_WILSON_HIGH_PCT}}) |
| A2: fit on the broken correspondence | {{BENCH_SYNTHG2P_COMPARATOR_A2_BROKEN_POSITION_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_A2_BROKEN_POSITION_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_A2_BROKEN_POSITION_WILSON_HIGH_PCT}}) | {{BENCH_SYNTHG2P_COMPARATOR_A2_BROKEN_EXACT_PCT}} ({{BENCH_SYNTHG2P_COMPARATOR_A2_BROKEN_EXACT_WILSON_LOW_PCT}}–{{BENCH_SYNTHG2P_COMPARATOR_A2_BROKEN_EXACT_WILSON_HIGH_PCT}}) |

The intervals in the table and in [@fig:aligned_g2p_arms] are derived at
render time from each arm's recorded accuracy and the recorded test
denominator it was scored on — a Wilson interval on its own denominator per
the decision-101 convention stated with [@fig:g2p_accuracy]. The uniform floor is
{{BENCH_SYNTHG2P_COMPARATOR_UNIFORM_POSITION_FLOOR_PCT}} per position
(`1/{{PHONEME_COUNT_39}}`) and
{{BENCH_SYNTHG2P_COMPARATOR_UNIFORM_EXACT_FLOOR_PCT}} per whole word, an
analytic reference the native bar must clear on both readings (A3) and not a
measured arm, so it is drawn as a rule rather than as a bar.

The A1 and A2 rows are **attribution controls**, and their design is a
refusal, not a ranking. A1 refits the same engine on English CMUdict pairs
and must not meet or clear the native bar on the identical stipulated
positions; a fired A1 would mean the stipulated signal is not separable from
English transfer, an attribution ambiguity reported by refusal and never
silently retried. A2 refits on a shape-valid broken correspondence — a
complete seeded derangement of `T`'s phone codomain — and must likewise fail
below the native bar. Both controls sit far below the native fit on both
readings in the recorded run, and the uniform floor sits far below both
controls, so the recorded evaluation shows the native bar holding above
every gate it was pre-registered to clear. The context-free
pronunciation-bigram comparator is recorded descriptively: it is the
pronunciation-only bar, not a spelling arm, and it never gates anything.

![Position accuracy of the P2 stipulated aligned spelling-to-pronunciation comparator arms, each bar the recorded position accuracy of one arm with a {{BOUND_CONFIDENCE_PCT}}% Wilson interval derived at render time on the arm's own denominator ({{BENCH_SYNTHG2P_COMPARATOR_TEST_POSITION_COUNT}} scored positions of {{BENCH_SYNTHG2P_COMPARATOR_TEST_WORD_COUNT}} test words). The dashed rule is the analytic uniform floor (1 of {{PHONEME_COUNT_39}}), a reference and not a measured arm, so it carries no interval. The panel title's tie statement is the recorder's recorded boolean rendered through the derived tie-word token, and no other verdict is drawn. The caption is falsified by a bar whose plotted value disagrees with its recorded token, by an interval not derived from the recorded denominator beside its own accuracy, by a floor rule drawn as a bar, or by any control drawn at or above the native bar while the recorded geometry that cleared the pre-registered refusal still holds.](../../output/figures/aligned_g2p.png){#fig:aligned_g2p_arms width=75%}

**Closure, and what stays open.** The queue item this protocol answered asked
for an aligned pronunciation design; the recorded evaluation — artifact,
attribution controls, and comparator — closes the design's artifact-and-comparator
portion, and the zero-shot probe arms are recorded beside it under run
`{{BENCH_SYNTHG2P_PROBE_ARTIFACT_SHA256_RUN}}`. The {{BENCH_MODEL_GEMMA3_DISPLAY}} and
{{BENCH_MODEL_QWEN25_DISPLAY}} arms scored {{BENCH_SYNTHG2P_PROBE_GEMMA3_WORDS_SCORED}} and
{{BENCH_SYNTHG2P_PROBE_QWEN25_WORDS_SCORED}} test words, at
{{BENCH_SYNTHG2P_PROBE_GEMMA3_POSITION_ACCURACY_PCT}} and
{{BENCH_SYNTHG2P_PROBE_QWEN25_POSITION_ACCURACY_PCT}} position accuracy and
{{BENCH_SYNTHG2P_PROBE_GEMMA3_EXACT_ACCURACY_PCT}} and
{{BENCH_SYNTHG2P_PROBE_QWEN25_EXACT_ACCURACY_PCT}} exact-word accuracy, with
contract compliance {{BENCH_SYNTHG2P_PROBE_GEMMA3_CONTRACT_COMPLIANCE_PCT}} and
{{BENCH_SYNTHG2P_PROBE_QWEN25_CONTRACT_COMPLIANCE_PCT}}; the {{BENCH_MODEL_SMOLLM_DISPLAY}} probe
is recorded as a contract refusal rather than a score. Both scoring arms sit far
below the analytic contract floor
({{BENCH_SYNTHG2P_PROBE_CONTRACT_FLOOR_PCT}}), which is the design's
pre-registered expectation for zero-shot models against a stipulated
correspondence, so the native-bar comparison closes by refusal and is
reported as that refusal, not as a ranking. What is closed closes
the way every arm in this section closes: unknown pretraining exposure
remains the stipulated family's standing limitation, and the arm licenses a
statement about transfer to this stipulated correspondence and nothing about
an English reading.

### Giving the probe order-one transition examples, a bounded reduction of the fit asymmetry

This sweep requests `{{BENCH_CTX_OPENROUTER_MODEL}}` through the recorded
provider route `{{BENCH_CTX_OPENROUTER_PROVIDER}}`. Each named panel is a
separate run cohort of that same hosted model.

The fourth condition above does not hold: AILM is fitted to
{{BENCH_CTX_CORPUS_CHARS}} generated symbols, while each probe starts without an
offline fit or separately labeled demonstrations. This arm measures one bounded
attempt to reduce that asymmetry; it cannot by itself remove it.

The probe is given `k` labeled continuation pairs drawn from the same corpus
AILM's counts are estimated from. Each prefix contains exactly
{{BENCH_CTX_EXAMPLE_WINDOW}} symbol, so each pair supplies one order-one
transition—the comparator's sufficient statistic—and contains
{{BENCH_CTX_EXAMPLE_INTERNAL_TRANSITIONS_PER_WINDOW}} unrelated internal
transitions. At the headline rung the prompt supplies
{{BENCH_CTX_HEADLINE_TOTAL_TRANSITIONS}} labeled transitions and retains
{{BENCH_CTX_HEADLINE_INTERNAL_TRANSITIONS}} within-prefix transitions. Draws
follow `{{BENCH_CTX_EXAMPLE_DRAW_GENERATION}}`: each is deterministically keyed
by the example seed, `k`, and the scored prefix (never its gold target), and
samples the separate fixed training stream. A set is accepted only when its
labels admit a complete derangement; otherwise that position is redrawn, up to
{{BENCH_CTX_DERANGEABLE_DRAW_ATTEMPTS}} attempts. This conditions the realized
example distribution. The exact seeded diagnostics below are injected from the
claims ledger and independently reconstructed from the frozen protocol:

{{BENCH_CTX_DRAW_DIAGNOSTICS_TABLE}}

The grid is {{BENCH_CTX_K_GRID}}.
The probe is not given
AILM's fitted column. Supplying the
comparator's own answer would cap the probe at the comparator by construction,
put the prediction on the first line of its own prompt, and produce a large
number that measures nothing; that option is recorded as rejected rather than
pending. The chain and artifact are generated locally, reducing exact-artifact
recall as an explanation without establishing the external model's complete
pretraining contents or behavior.
Each query still carries {{BENCH_CTX_QUERY_WINDOW}} symbols and up to
{{BENCH_CTX_QUERY_CONTEXT_TRANSITIONS_MAX}} adjacent transitions under an exact
{{BENCH_CTX_RESPONSE_INVENTORY_SIZE}}-symbol response contract.

Each `k` runs twice. The control shows the same examples with their symbols
permuted so that no example keeps its own — a derangement, so the displayed
prefixes, the symbol multiset and the prompt length are all preserved exactly
at the character level, while only the prefix-to-symbol *mapping* is destroyed.
No claim is made that a tokenizer assigns identical token counts after that
permutation. Without it a lift cannot
be told apart from the probe being pulled toward the alphabet by the sight of
plausible symbols. The hosted transport uses no warm-up under
`{{BENCH_CTX_WARMUP_POLICY}}`; all scored cells are fresh and all arms are interleaved under
`{{BENCH_CTX_ARM_ORDER}}` so a complete arm is never synonymous with one period
of the server run.

{{BENCH_CTX_MAPPING_CONTROL_RESULT}} This data-level positive control consumes
the exact transition tuples used to serialize the prompts and shows that the
stimulus/control pairing contains detectable mapping information. It does not
prove that a particular tokenizer or prompt interface conveys that mapping to a
probe. It prevents probe-level non-separation from being misread as proof that
the underlying stimulus or control was vacuous.

{{BENCH_CTX_TABLES}}

Accuracy cells include descriptive, non-simultaneous
{{BOUND_CONFIDENCE_PCT}}% Wilson intervals. Invalid
structured replies remain in each denominator and are reported beside the two
prompt conditions. P-values are rounded to three significant digits for
display; exact values remain in the claims ledger.

Scored over {{BENCH_CTX_POSITIONS}} independent evaluation trajectories under
`{{BENCH_CTX_POSITION_GENERATION}}` (base seed
{{BENCH_CTX_POSITION_SEED}}, trajectory length {{BENCH_CTX_TRAJECTORY_LENGTH}},
target-start TVD {{BENCH_CTX_TARGET_START_TVD}}) against a uniform floor of
{{BENCH_CTX_UNIFORM_BASELINE_DISPLAY}} and an AILM comparator at
{{BENCH_CTX_TOP1_AILM_DISPLAY}}, with every external-probe contrast p-value
corrected inside its declared family of {{BENCH_CTX_FAMILY_SIZE}} at
{{BENCH_CTX_FAMILY_ALPHA_DISPLAY}}.
Every scored request declares a context window of {{BENCH_CTX_CONTEXT_WINDOW}} tokens
and a {{BENCH_CTX_RESPONSE_TOKENS}}-token response cap. The longest prompt has
{{BENCH_CTX_MAXIMUM_PROMPT_CHARACTERS}} ASCII characters in the preflight; the
server reports at most {{BENCH_CTX_MAXIMUM_PROMPT_TOKENS}} executed prompt
tokens. That measured count, the response cap, and the
{{BENCH_CTX_PROMPT_TOKEN_OVERHEAD_MARGIN}}-token safety margin fit inside the
declared context window. The canonical
run recorded {{BENCH_CTX_SCORED_CACHE_HITS}} scored cache hits and
{{BENCH_CTX_SCORED_CACHE_MISSES}} fresh scored calls.

{{BENCH_CTX_FAMILY_RESULT}} The sweep itself is drawn in
[@fig:incontext_sweep].

![Separate run cohorts of `{{BENCH_CTX_OPENROUTER_MODEL}}` through provider route `{{BENCH_CTX_OPENROUTER_PROVIDER}}`, scored against the number of labeled single-transition continuation examples supplied, with the character-level multiset-matched derangement control built from the same examples. The comparator and uniform floor are reference levels, not series. Each full-scale panel retains both arms and reference levels; its inset enlarges the same treatment/control intervals on a shared narrower vertical scale, with the enlarged region outlined in the parent. The title is generated from the validated family outcomes rather than typed from an earlier run. This panel would be falsified by a plotted cell not matching its recorded token, by a rung or registered probe omitted from the figure, by a treatment/control prompt mismatch beyond the derangement, or by a title that disagrees with the corrected paired contrasts.](../../output/figures/incontext_sweep.png){#fig:incontext_sweep width=88%}

In the article PDF, enlarged views of [@fig:incontext_sweep] retain both
percent-accuracy axes and reference lines. Blue circles show labeled transitions;
orange squares show the matched derangement. The overview retains the shared
legend and generated outcome headline.

```{=latex}
\begingroup
\csname @ifclassloaded\endcsname{beamer}{}{
\noindent\begin{minipage}{\linewidth}
\centering
\textit{Blue circles: shown transitions; orange squares: deranged.}\par\smallskip
\includegraphics[viewport=0bp 0bp 383.4bp 318.6bp,clip,width=0.86\linewidth,alt={Enlarged gemma3 run panel from the preceding in-context sweep; blue circles are labeled transitions and orange squares are deranged examples. Full scale, reference lines, and interval inset are retained.}]{../../output/figures/incontext_sweep.png}
\end{minipage}\par\medskip
\noindent\begin{minipage}{\linewidth}
\centering
\textit{Blue circles: shown transitions; orange squares: deranged.}\par\smallskip
\includegraphics[viewport=383.4bp 0bp 754.2bp 318.6bp,clip,width=0.86\linewidth,alt={Enlarged qwen25 run panel from the preceding in-context sweep; blue circles are labeled transitions and orange squares are deranged examples. Full scale, reference lines, and interval inset are retained.}]{../../output/figures/incontext_sweep.png}
\end{minipage}\par\medskip
\noindent\begin{minipage}{\linewidth}
\centering
\textit{Blue circles: shown transitions; orange squares: deranged.}\par\smallskip
\includegraphics[viewport=754.2bp 0bp 1137.6bp 318.6bp,clip,width=0.86\linewidth,alt={Enlarged smollm run panel from the preceding in-context sweep; blue circles are labeled transitions and orange squares are deranged examples. Full scale, reference lines, and interval inset are retained.}]{../../output/figures/incontext_sweep.png}
\end{minipage}\par\medskip
}
\endgroup
```

{{BENCH_CTX_HEADLINE_RESULT}} This bounds performance under the declared maximum
`k`, prompt, and derangement; it does not demonstrate equal training opportunity
and does not exclude fitting asymmetry as an explanation of the remaining
margin.

Scope, because a result this shaped invites over-reading in both directions.
The {{BENCH_PROBE_COUNT_WORD}} registered run cohorts of the same hosted model were tested on the
stipulated order-one chain under the recorded prompt and grid ending at
{{BENCH_CTX_HEADLINE_K}} examples. Agreement across these run cohorts is
still not a general claim: a different family of
model, a longer context, or a chain with a different marginal could each behave
differently, and nothing here is evidence about in-context learning in general.

### Calibration scores and limits of interpretation

On the letter task both kinds of system put a distribution over the same
declared inventory, so the same scoring rules can be computed for both. That is
not enough to make the resulting numbers commensurable, and this subsection does
not treat them as though it were. The two distributions are objects of different
kinds. The comparator's is its own exact transition column. Each probe's is a
reconstruction assembled here from a truncated candidate list — filtered to the
scored inventory, completed for the symbols the truncation hid, and renormalized
by the procedure set out in the next paragraph. A probe's Brier and negative
log likelihood below are therefore proper scores of that reconstructed inventory
distribution rather than of the external model's constrained-output
distribution, while the comparator's are scores of the column it actually uses.
Nothing in the table ranks one kind of object against the other.

Expected calibration error uses {{BENCH_CALIB_BINS}} equal-width bins over
{{BENCH_CALIB_POSITIONS}} positions, an evenly spaced stride subsample of the letter
census, so the table's Top-1 column is a subsample figure and differs from the
census accuracy above. Calibration is reported only for a system
whose own answer-slot candidates survive the direct dual-rule check: the two
completion rules are both applied to that system's recorded distributions, and
a system whose reported calibration numbers move by more than
{{BENCH_CALIB_RULE_EPSILON}} between them is withheld from the table. That gate
is the one decision 93 adopted, replacing the captured-mass floor the protocol
originally declared; the floor is retained as provenance and the subsection
after this one measures it. Under the adopted gate two of the
{{BENCH_PROBE_COUNT_WORD}} registered probes are withheld, and their withheld
figures are still recorded in the ledger — a withheld calibration is an audited
one, not an absent one.

AILM's distribution is the exact bigram column it takes its argmax of. Each
probe's distribution is read from the answer-slot token logprobs of the same
schema-constrained call the accuracy is scored from — restricting the read to a
declared answer slot is the same move that lets a language model's stated
probabilities be well calibrated on a closed-set question in the first place
[@kadavath2022language], and the protocol here inherits that precondition
rather than establishing it fresh. Because each probe's
answer-slot distribution is reconstructed from a truncated top-k list rather
than observed in full, the external calibration figures are explicitly
exploratory: no full-distribution reading or missing-mass bound is computed, and
the captured-mass floor of the next subsection exists precisely to measure how
much the truncation decides. [@vaicenavicien2019evaluating],
[@kumar2019calibration], and [@nixon2019measuring] make the general point that
binned calibration estimates on reconstructed distributions carry
completion-rule sensitivity; this protocol treats those references as design
guidance for the floor rather than as computed checks here. Two properties of that list
matter. The server reports at most {{BENCH_CALIB_TOP_LOGPROBS}} candidates for a
{{STAGE0_LETTER_ALPHABET_SIZE}}-symbol inventory, so it is truncated. And what
it returns is the raw next-token distribution rather than the schema-masked one,
so it also carries multi-character tokens the response contract could never
emit. That mass is discarded rather than redistributed, since it belongs to no
inventory symbol, and the remainder is renormalized. An inventory symbol absent
from the truncated list fell below every reported one, so it is assigned that
smallest reported probability as an upper bound and the distribution
renormalized again.

The share genuinely observed is itself recorded —
{{BENCH_CALIB_CAPTURED_MASS_GEMMA3}} for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_CALIB_CAPTURED_MASS_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_CALIB_CAPTURED_MASS_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}} on average — because the
completion is an assumption and its weight should be visible. The answer slot
was unlocatable on {{BENCH_CALIB_UNLOCATED_GEMMA3}} positions for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_CALIB_UNLOCATED_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_CALIB_UNLOCATED_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}. Those are the only positions
scored from an entirely assumed distribution; elsewhere the assumed part is the
uncaptured mass recorded above. The captured
share is not a property of the inventory alone: a model whose probability mass
sits mostly on multi-symbol tokens the grammar forbids yields a low share
whatever the inventory size, which is why the share is recorded per system.

That share is not always concentrated. Where it is spread, most positions are
near-fully observed and a minority near-empty, and the mean then sits between
two clusters and matches neither; where every position is well observed the mean
is simply the answer. A mean alone can describe neither of those two cases, so
both summaries are reported. The median captured share is
{{BENCH_CALIB_CAPTURED_MEDIAN_GEMMA3}} for {{BENCH_MODEL_GEMMA3_DISPLAY}} against an individually
starved fraction of {{BENCH_CALIB_STARVED_FRACTION_GEMMA3}},
{{BENCH_CALIB_CAPTURED_MEDIAN_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}} against
{{BENCH_CALIB_STARVED_FRACTION_QWEN25}}, and
{{BENCH_CALIB_CAPTURED_MEDIAN_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}} against
{{BENCH_CALIB_STARVED_FRACTION_SMOLLM}}: {{BENCH_MODEL_GEMMA3_DISPLAY}} is the spread case,
{{BENCH_MODEL_QWEN25_DISPLAY}} the concentrated one, and
{{BENCH_MODEL_SMOLLM_DISPLAY}} the starved one — nearly every position scored
from an assumed distribution rather than a reported one. The retired floor was applied to the mean, which is the
stricter summary whenever the two differ, and both are recorded so a reader can
see how much of each figure rests on mostly-reconstructed positions. An absent
inventory symbol is completed with the smallest probability the server did
report, never with a share of the discarded mass; that bound cannot exceed any
observed candidate, so it perturbs the distribution less than spreading the
discarded mass would.

The size of that advantage is not something the method guarantees. The
gap between these numbers and an observed-only renormalization grows with the
starved fraction, and the registered probes do not sit between the near-clean
and near-starved extremes — {{BENCH_MODEL_GEMMA3_DISPLAY}} and
{{BENCH_MODEL_QWEN25_DISPLAY}} sit near the clean end and
{{BENCH_MODEL_SMOLLM_DISPLAY}} at the starved one. Above the declared floor
{{BENCH_MODEL_QWEN25_DISPLAY}} is close to rule-insensitive and
{{BENCH_MODEL_GEMMA3_DISPLAY}} is not; that is an observation about
this data, not a guarantee. What
the floor rests on is the narrower fact that below it no completion rule
recovers a distribution that was almost entirely unobserved, so the reported
quantity would be the rule rather than the model. Where between those two
extremes the transition actually falls is a measurable question, and the next
subsection answers it offline, on distributions whose truncation can be swept.

The class each probe is scored on is the answer it returned, not the argmax of
the reconstructed distribution. The two can differ. The reported candidates are
the raw pre-mask distribution, while generation is grammar-constrained, so the
constrained decoder can be pushed off its raw argmax. On this run they agree on
{{BENCH_CALIB_ARGMAX_AGREEMENT_GEMMA3}},
{{BENCH_CALIB_ARGMAX_AGREEMENT_QWEN25}} and
{{BENCH_CALIB_ARGMAX_AGREEMENT_SMOLLM}} of positions for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_MODEL_QWEN25_DISPLAY}} and {{BENCH_MODEL_SMOLLM_DISPLAY}}. That is near-total, and it is not guaranteed: agreement is a property
of the decode rather than of the method — one more reason the gate, not the
table, decides what appears here. Where the two differ, scoring the argmax would report
a calibration accuracy the benchmark never recorded. The parsed answer is used
instead, and the agreement rate is reported beside it rather than assumed.

Expected calibration error, Brier score, and negative log likelihood are
reported per system, all computed over the full distribution rather than the top symbol alone. Expected calibration error is the
binned gap between stated confidence and realized accuracy; the Brier score is
the multiclass form $\sum_k (p_k - y_k)^2$; and the negative log likelihood is
the mean of $-\log p$ on the gold symbol. The latter two are strictly proper
[@gneiting2007strictly] as
rules applied to whatever distribution is handed to them, which is why a
top-symbol Brier variant is not used: it can rank a system that is uniformly
wrong at low confidence above one that is often right at moderate confidence.
Propriety here is a property of the rule and of the argument it is given, and
for a probe that argument is the reconstruction. A strictly proper rule is
minimized in expectation by reporting one's own beliefs, and that incentive
attaches to whatever produced the scored vector; since the scored vector is
assembled here rather than emitted by the external system, the score carries no
strictly-proper interpretation with respect to that system. It remains a proper
score of the reconstruction, which is the quantity the table reports.
Both proper rules are reported rather than one, because they are not redundant:
under a change to how an unreported symbol is completed they can move in
opposite directions on the same system, so a single rule would present a
reconstruction choice as a property of the model.

| System | Top-1 | Mean confidence | ECE | Multiclass Brier | NLL |
|---|---:|---:|---:|---:|---:|
{{BENCH_CALIB_TABLE_ROWS}}

The first three numeric columns are percentages. Proper scores are rounded to
three decimal places for display; their exact values remain in the claims
ledger.

A system the gate withholds has no row in that table, and the reason is the
same specific misreading the captured-mass analysis located, now measured
directly. A reconstruction the completion rule dominates flattens toward the
smallest reported probability; flattening lowers the stated confidence, and on
a system that is usually wrong a lowered confidence *reduces* the measured
calibration error. A rule-dependent system would therefore tend to appear as
the best-calibrated row in the table, and marking such a row unreliable does
not stop a reader ranking it. {{BENCH_MODEL_GEMMA3_DISPLAY}} is that system on
this run: its two rules move its expected calibration error by
{{BENCH_CALIB_RULE_ECE_DIVERGENCE_GEMMA3}} and its Brier score by
{{BENCH_CALIB_RULE_BRIER_DIVERGENCE_GEMMA3}}, both beyond the
{{BENCH_CALIB_RULE_EPSILON}} tolerance, so it has no row.
{{BENCH_MODEL_SMOLLM_DISPLAY}} is withheld by the same gate, and by larger
margins — {{BENCH_CALIB_RULE_ECE_DIVERGENCE_SMOLLM}} in expected calibration
error and {{BENCH_CALIB_RULE_BRIER_DIVERGENCE_SMOLLM}} in Brier — and its
recorded-but-unpublished expected calibration error is the *lowest* among the
probes' recorded figures, which is the flattening trap above working exactly as
described: the most reconstructed probe distribution would have appeared the
best calibrated among them.
Both systems' figures remain in the ledger, and the gate's verdict is
recomputable from them — recording nothing was never what stopped the
misreading; measuring the rule was.

Calibration is a distinct axis from accuracy rather than a restatement of it: a
system whose distribution were one-hot at every position would post an expected
calibration error of exactly one minus its accuracy, and the measured values
above do not coincide with that. The mean-confidence column is what separates
the two — a system may state a confidence far from the rate at which it is
right, in either direction, at any accuracy.

### Captured-mass floor and calibration scope

The captured-mass floor was the gate the calibration table originally applied,
and it was argued rather than located. The declared value {{BENCH_FLOOR_DECLARED_FLOOR}}
follows from a sound premise — below some captured mass the completion rule
rather than the model decides the reported number — but nothing measured where
that mass is. The live arm cannot supply the answer: its probes sit at
{{BENCH_CALIB_CAPTURED_MASS_GEMMA3}}, {{BENCH_CALIB_CAPTURED_MASS_QWEN25}} and
{{BENCH_CALIB_CAPTURED_MASS_SMOLLM}}
captured mass, so almost the entire range in which the transition could fall is
unsampled. This subsection is a measurement of the project's own gate rather
than of any model.

The measurement needs distributions whose truncation behavior can be swept, and
it needs them without a model call, so it uses AILM's own letter transition
columns: real next-symbol distributions over the same
{{BENCH_FLOOR_INVENTORY_SIZE}}-symbol inventory the benchmark scores, already the
comparator the calibration arm reports. Protocol
`{{BENCH_FLOOR_PROTOCOL_VERSION}}` at seed {{BENCH_FLOOR_SEED}} scores
{{BENCH_FLOOR_POSITIONS}} positions with {{BENCH_FLOOR_BINS}} bins, matching the
sample size and binning of the live arm. At each of
{{BENCH_FLOOR_GRID_POINTS}} target captured masses from {{BENCH_FLOOR_GRID_LOW}}
to {{BENCH_FLOOR_GRID_HIGH}} in steps of {{BENCH_FLOOR_GRID_STEP}}, every column
is truncated to the smallest top-k reaching that target — the truncation the
logprob cap imposes — and reconstructed twice: once under the shipped min-bound
rule, read out of the benchmark's own reconstruction path rather than restated,
and once under an observed-only rule that renormalizes the survivors and gives an
absent symbol nothing. The measured quantity is the gap between the two rules,
not either rule's fidelity to the untruncated column, because that gap is
precisely what "the completion rule rather than the model decides the number"
means. The whole sweep is offline; it issues no request and reaches no service.

The located floor is the smallest captured mass above which neither expected
calibration error nor the Brier score again disagrees between the two rules by
more than {{BENCH_FLOOR_EPSILON}}. It is {{BENCH_FLOOR_LOCATED_CAPTURED}}, which
is {{BENCH_FLOOR_LOCATED_RATIO}} times the declared
{{BENCH_FLOOR_DECLARED_FLOOR}}. The declared floor is therefore not a
conservative choice: it sits below the located one by that factor, admitting
systems at a captured mass where the two rules still disagree.
Repeating the sweep at {{BENCH_FLOOR_CHECK_POSITIONS}} positions puts the floor
at {{BENCH_FLOOR_CHECK_LOCATED_CAPTURED}}, so the location is not an artifact of
the sample size a binned statistic is computed at.

At the declared floor itself the two rules have not converged. The grid point
whose achieved mass is nearest it captures {{BENCH_FLOOR_ATDECLARED_CAPTURED}},
and there the reconstructions differ by
{{BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE}} in expected calibration error and
{{BENCH_FLOOR_ATDECLARED_BRIER_DIVERGENCE}} in Brier score, while sitting
{{BENCH_FLOOR_ATDECLARED_RULE_TV}} apart in total variation. AILM's whole
recorded expected calibration error is {{BENCH_CALIB_ECE_AILM}}, so a
rule-induced disagreement of {{BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE}} is a large
fraction of the effect this axis reports. A system admitted at the gate could
have a substantial part of its published calibration number decided by a
reconstruction choice rather than by its own distribution.

Raising the gate to the located floor is not available through this endpoint.
Reaching {{BENCH_FLOOR_LOCATED_CAPTURED}} captured mass on these columns needs
{{BENCH_FLOOR_LOCATED_K}} of {{BENCH_FLOOR_INVENTORY_SIZE}} candidates on
average, against a server cap of {{BENCH_FLOOR_CAP}}, and
{{BENCH_FLOOR_LOCATED_OVER_CAP_FRACTION}} of positions need more than the cap
allows. The declared floor by contrast needs {{BENCH_FLOOR_ATDECLARED_K}}
candidates, comfortably inside the cap, which is part of why it was easy to
declare and never tested. The located floor is consequently unreachable in
principle here rather than merely unmet by the registered probes, and the
constraint is the API's, not the models'. What would falsify that: an endpoint
returning all {{BENCH_FLOOR_INVENTORY_SIZE}} candidates, on which the located
floor becomes an ordinary requirement.

The locator has one property worth reporting on its own, because a plausible
alternative would have produced a wrong answer that looked like a confirmation.
Each metric's divergence first drops below {{BENCH_FLOOR_EPSILON}} far below the
located floor: at {{BENCH_FLOOR_NAIVE_ECE_CAPTURED}} captured mass for expected
calibration error and {{BENCH_FLOOR_NAIVE_BRIER_CAPTURED}} for Brier, both just
below the declared {{BENCH_FLOOR_DECLARED_FLOOR}}, while the two reconstructions
are still {{BENCH_FLOOR_NAIVE_ECE_RULE_TV}} apart in total variation and so
agree on almost nothing. The two rules bias a metric
in opposite directions, min-bound flattening the distribution and observed-only
sharpening it, and each signed difference passes through zero exactly once
({{BENCH_FLOOR_ECE_SIGN_CHANGES}} sign change for expected calibration error,
{{BENCH_FLOOR_BRIER_SIGN_CHANGES}} for Brier). A locator that took the first
crossing would have landed on those two values, each of which makes the declared
floor look conservative, from a sweep in which the completion rule is still
deciding almost the whole distribution. The recorded locator requires a sustained
crossing instead — the smallest point above which the divergence never again
exceeds the threshold — and that requirement is load-bearing rather than
cosmetic: the metric divergences are not monotone in captured mass, rising on
{{BENCH_FLOOR_ECE_INCREASES}} grid steps for expected calibration error and
{{BENCH_FLOOR_BRIER_INCREASES}} for Brier, even though the total variation
between the two reconstructions rises on {{BENCH_FLOOR_TV_INCREASES}}. The
distributional gap closes monotonically; its image in either metric does not.

[@fig:calibration_floor] puts the whole sweep on one set of axes: the two signed
differences with the distributional gap they cross zero inside, the same
divergences against the threshold on a log axis, the size of the rule-decided
part of a published number, the candidate cost of reaching the located floor,
and a captured-mass ruler carrying both floors beside the measured probes.

![**What to see: a calibration floor an external benchmark declared by argument, not measurement, is relocated here by direct computation — it sits far from the declared value, showing the original floor was not conservative.** Locating the captured-mass floor offline. Population: AILM's own order-one letter transition columns at {{BENCH_FLOOR_POSITIONS}} deterministically subsampled held-out letter positions (seed {{BENCH_FLOOR_SEED}}, {{BENCH_FLOOR_BINS}} bins), each truncated to the smallest top-k reaching each of {{BENCH_FLOOR_GRID_POINTS}} target captured masses from {{BENCH_FLOOR_GRID_LOW}} to {{BENCH_FLOOR_GRID_HIGH}} over the {{BENCH_FLOOR_INVENTORY_SIZE}}-symbol inventory and reconstructed under both completion rules; no model call and no network. Top left: each metric's signed rule difference against mean captured mass, with the rule-versus-rule total variation on the right axis. Every grid point is drawn as its own marker rather than smoothed. Top right: the same two divergences in absolute value on a log axis against the declared threshold {{BENCH_FLOOR_EPSILON}}, with the span a first-crossing locator would have certified shaded. Middle: candidates per position needed to reach each captured mass, against the server cap of {{BENCH_FLOOR_CAP}}, with the share of positions individually over that cap on the right axis. Bottom: the declared floor {{BENCH_FLOOR_DECLARED_FLOOR}}, the located floor {{BENCH_FLOOR_LOCATED_CAPTURED}}, both first crossings and every measured probe on one captured-mass axis. (At the declared floor the two completion rules differ by {{BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE}} in expected calibration error, against AILM's own recorded {{BENCH_CALIB_ECE_AILM}} — stated here in prose rather than as a fifth panel; see the paragraph above.) The per-grid-point series is not in the ledger, so the figure recomputes the sweep from the recorder's own offline functions on the recorder's own recorded parameters, and refuses to draw if any re-derived summary contradicts its recorded claim. Any of the following would falsify this panel: a located floor, first crossing, identity value, candidate cost or divergence that disagrees with its recorded token; a signed gap crossing zero a number of times other than the recorded sign-change count; a divergence drawn as a smooth interpolation rather than at the swept grid points; a probe marker for a system the live calibration arm did not record; or the identity point at total capture failing to reduce both completion rules to the same column.](../../output/figures/calibration_floor.png){#fig:calibration_floor width=98%}

The sweep also carries an identity control, and it is what makes the rest of
these numbers a measurement of the shipped code. At the top of the grid the
truncation removes nothing, so both rules must reduce to the untruncated column:
they differ there by {{BENCH_FLOOR_IDENTITY_RULE_TV}} in total variation against
a declared tolerance of {{BENCH_FLOOR_IDENTITY_TOLERANCE}}, and the figures at
that point are {{BENCH_FLOOR_IDENTITY_ECE}} and {{BENCH_FLOOR_IDENTITY_BRIER}} —
the same expected calibration error and Brier score the calibration table above
records for AILM, {{BENCH_CALIB_ECE_AILM}} and {{BENCH_CALIB_BRIER_AILM}}. The
sweep is therefore exercising the benchmark's own reconstruction and scoring
path, not a reimplementation that happens to resemble it. Either pair failing to
coincide would falsify that claim, and the recorder writes no claim at all if
the two rules disagree beyond the tolerance at the top of the grid.

What this does and does not say about the reported calibration numbers turns on
whose distributions were swept. The ground truth here is AILM's column shape, and
column shape is exactly what the result depends on. AILM's mean confidence is
{{BENCH_CALIB_CONFIDENCE_AILM}}, against {{BENCH_CALIB_CONFIDENCE_GEMMA3}} for
{{BENCH_MODEL_GEMMA3_DISPLAY}}, {{BENCH_CALIB_CONFIDENCE_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_CALIB_CONFIDENCE_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}: the probes'
reported distributions are far more peaked, so a truncation reaching a given
captured mass leaves them far less structure to disagree about, and the two
rules have correspondingly less room to diverge. The sweep states a divergence
at a captured mass for AILM-shaped distributions, and that statement does not
transfer to a peaked one. Nothing here shows that the calibration figures
reported above are wrong.

The conclusion is narrower, and it is about the gate rather than about the
numbers it admitted. Captured mass is a *proxy* for rule-dependence, and on at
least one realistic distribution family the proxy is not conservative at the
value the protocol declares. {{BENCH_MODEL_GEMMA3_DISPLAY}}'s {{BENCH_CALIB_CAPTURED_MASS_GEMMA3}}
sits above the declared floor and below the located one, so one of the
registered systems is inside the band this sweep flags; {{BENCH_MODEL_QWEN25_DISPLAY}}'s
{{BENCH_CALIB_CAPTURED_MASS_QWEN25}} sits above both; and
{{BENCH_MODEL_SMOLLM_DISPLAY}}'s {{BENCH_CALIB_CAPTURED_MASS_SMOLLM}} sits
below the declared floor itself, so the floor alone would have excluded it —
and the direct check excludes it on its own divergence regardless. It is
{{BENCH_MODEL_GEMMA3_DISPLAY}} on which the two verdicts disagree: the floor
admits what the divergence check withholds.

What would actually license a probe's calibration number is not its captured mass
but the divergence measured on that probe's own recorded candidates under the two
rules — the same quantity this sweep measures on AILM's columns, applied to each
probe's answer-slot list instead. That quantity is now recorded. Under the two
completion rules the probes' calibration numbers move by
{{BENCH_CALIB_RULE_ECE_DIVERGENCE_GEMMA3}} in ECE and
{{BENCH_CALIB_RULE_BRIER_DIVERGENCE_GEMMA3}} in Brier for {{BENCH_MODEL_GEMMA3_DISPLAY}}, by
{{BENCH_CALIB_RULE_ECE_DIVERGENCE_QWEN25}} and
{{BENCH_CALIB_RULE_BRIER_DIVERGENCE_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and by
{{BENCH_CALIB_RULE_ECE_DIVERGENCE_SMOLLM}} and
{{BENCH_CALIB_RULE_BRIER_DIVERGENCE_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}, against a
tolerance of {{BENCH_CALIB_RULE_EPSILON}}. {{BENCH_MODEL_QWEN25_DISPLAY}} is inside that
tolerance on both metrics and {{BENCH_MODEL_GEMMA3_DISPLAY}} and
{{BENCH_MODEL_SMOLLM_DISPLAY}} are outside it on both, which is the direct
measurement the captured-mass proxy was standing in for — and the two verdicts
disagree on exactly one system. SmolLM is the flattening trap demonstrated
rather than argued: its recorded-but-unpublished expected calibration error is
the lowest among the probes' recorded figures, and publishing it would have credited the
completion rule's own flattening to the model.

That disagreement was the finding, and it is now the gate. The proxy admitted
both probes; the direct check showed one of them to be rule-dependent enough to
matter at the declared tolerance. Both signed divergences were negative for
every probe, so the shipped min-bound rule reports a *lower* error than the
observed-only rule in every case rather than in some — the reconstruction is
not simply noisy around the observed value, it is biased in one direction. The
gate change — gating on the dual-rule divergence exceeding {{BENCH_CALIB_RULE_EPSILON}} rather than on captured mass falling
below {{BENCH_FLOOR_DECLARED_FLOOR}} — was deferred so that a version of the
paper comparing both rules would exist; the record under version
{{META_VERSION}} is that version, and the
gate was adopted on its evidence (decision 93). The calibration table above
already reports under the adopted gate: {{BENCH_MODEL_GEMMA3_DISPLAY}} and
{{BENCH_MODEL_SMOLLM_DISPLAY}} are withheld and
{{BENCH_MODEL_QWEN25_DISPLAY}} reports. `CALIBRATION_MASS_FLOOR` stands in the
code at {{BENCH_CALIB_MASS_FLOOR}} as provenance only; it no longer decides
anything.

### Registered greedy-decoding stability diagnostic

Decoding is greedy at temperature zero, so the sampling seed cannot reach the
output and the expected run-to-run spread is exactly zero. That is a prediction
about the protocol, and it was measured rather than assumed.

The prediction holds for the *answer* and not for the *distribution*, and the two
are separate claims that this section previously ran together. Re-issuing a
recorded request against the live server reproduces the reply string exactly; it
reproduces the answer-slot logprobs only approximately. Over
{{BENCH_LOGPROB_DRIFT_POSITIONS}} re-queried positions the answer agreed on a
fraction {{BENCH_LOGPROB_DRIFT_ANSWER_AGREEMENT}} of them, while the logprobs
drifted by up to {{BENCH_LOGPROB_DRIFT_MAX}} in log space,
{{BENCH_LOGPROB_DRIFT_PROBABILITY}} in reconstructed probability and
{{BENCH_LOGPROB_DRIFT_CAPTURED_MASS}} in captured mass, with
{{BENCH_LOGPROB_DRIFT_MEMBERSHIP_CHANGES}} candidate tokens entering or leaving
the truncated top-*k* lists entirely.
Every accuracy number from the locally served arms of this section is therefore
exactly reproducible (the hosted arms are not run-stable; decisions 128 and 130), and
every calibration number carries that drift as a floor on its precision — ECE,
Brier and captured mass are computed from the quantity that moves. The cache
audit checks the two separately for the same reason: the answer must match
byte-for-byte, the distribution within a declared tolerance. The stability and
determinism arms draw their own stride subsamples from the same letter pool
rather than reusing the scored positions, so they replicate the protocol
independently rather than rerunning the table above.

Rerunning the letter task over {{BENCH_ENSEMBLE_POSITIONS}} positions at two
neighboring seeds gives:

| System | First seed | Second seed | Spread |
|---|---:|---:|---:|
| {{BENCH_MODEL_GEMMA3_DISPLAY}} | {{BENCH_RUN_LETTER_TOP1_GEMMA3_SEED1}} | {{BENCH_RUN_LETTER_TOP1_GEMMA3_SEED2}} | {{BENCH_RUN_LETTER_SPREAD_GEMMA3}} |
| {{BENCH_MODEL_QWEN25_DISPLAY}} | {{BENCH_RUN_LETTER_TOP1_QWEN25_SEED1}} | {{BENCH_RUN_LETTER_TOP1_QWEN25_SEED2}} | {{BENCH_RUN_LETTER_SPREAD_QWEN25}} |
| {{BENCH_MODEL_SMOLLM_DISPLAY}} | {{BENCH_RUN_LETTER_TOP1_SMOLLM_SEED1}} | {{BENCH_RUN_LETTER_TOP1_SMOLLM_SEED2}} | {{BENCH_RUN_LETTER_SPREAD_SMOLLM}} |

Issuing the identical request twice at the identical seed reproduces the reply
string on a fraction {{BENCH_DETERMINISM_AGREEMENT_GEMMA3}} of
{{BENCH_DETERMINISM_POSITIONS}} positions for {{BENCH_MODEL_GEMMA3_DISPLAY}},
{{BENCH_DETERMINISM_AGREEMENT_QWEN25}} for {{BENCH_MODEL_QWEN25_DISPLAY}}, and
{{BENCH_DETERMINISM_AGREEMENT_SMOLLM}} for {{BENCH_MODEL_SMOLLM_DISPLAY}}. Reproducibility holds
under matched server state; a model load or an intervening workload is a change
in that state, which is why the protocol issues one discarded warm-up call
before any position is scored. These are determinism replications and not
estimates of sampling variance; there is no sampling variance to estimate under
greedy decoding.

![**What to see: AILM and several externally served zero-shot models are scored on the same next-symbol positions, but this is a reference frame rather than a leaderboard — no panel here ranks the systems against each other.** Same-position reference probes; no panel is a ranking. The top row shows accuracy on the letter, phoneme, and G2P tasks with {{BOUND_CONFIDENCE_PCT}}% Wilson proportion intervals on the declared denominator ({{BENCH_LETTER_POSITIONS}} letter positions, {{BENCH_PHONEME_POSITIONS}} phoneme positions, and for G2P the aligned phone positions of {{BENCH_G2P_WORDS}} words rather than the word count itself); paired probe-minus-AILM differences with exact conditional intervals, where the G2P group is the exact-match contrast rather than the position accuracy; and letter accuracy at the {{BENCH_LETTER_WINDOW_SHORT}}- and {{BENCH_LETTER_WINDOW_LONG}}-character prefix windows and on alphabetic-gold positions. The bottom row includes separate run cohorts of `{{BENCH_SYNTH_OPENROUTER_MODEL}}` through provider route `{{BENCH_SYNTH_OPENROUTER_PROVIDER}}` on the stipulated letter and phone chains; these labels do not identify the local weights shown in the other panels. The stipulated panel shows each bar group against its own uniform floor ({{BENCH_SYNTH_UNIFORM_BASELINE_DISPLAY}} letters, {{BENCH_SYNTHPHONE_UNIFORM_BASELINE_DISPLAY}} phones). The other bottom panels show letter-task expected calibration error and Brier score scaled by their respective theoretical maxima for the systems the direct dual-rule gate admits, with the withheld probes named in-panel rather than drawn (decision 93), and G2P position accuracy stratified by gold phone count. The prefix-window and phone-count panels carry Wilson intervals on their own denominators: the scored positions of each letter arm, and the gold phone positions of each bucket ({{BENCH_G2P_BUCKET_SHORT_PHONES}}, {{BENCH_G2P_BUCKET_MID_PHONES}}, and {{BENCH_G2P_BUCKET_LONG_PHONES}}). The G2P position panels are not single-rule readings: the comparator uses strict positional alignment and each probe uses Levenshtein alignment, which is why the section reports separate readings. Invalid answers are scored wrong and never dropped, and there is no top-k series because the endpoint returns a single answer per call. Any of the following would falsify this panel: a bar not matching the recorded tokens once the calibration panel's worst-case scaling is undone; an interval not matching the paired or Wilson tokens; an AILM context bar that differs between the prefix windows; a sampled-position count that has changed; a figure that silently includes a model the recorders did not register; a withheld system drawn as a calibration bar; a G2P position bar drawn from an alignment token other than the one this caption assigns its system; or a line connecting the external models as a trend.](../../output/figures/external_benchmarks.png){#fig:external_benchmarks width=98%}

In the article PDF, enlarged views reproduce the panels of
[@fig:external_benchmarks] from the same image, grouped to retain their
complete headlines, axes, intervals, and legends. The global identity headline
and complete caption remain in the overview. The stipulated-chain panel uses
hosted run cohorts; the other panels use the locally served model identities
named in their legends.

```{=latex}
\begingroup
\csname @ifclassloaded\endcsname{beamer}{}{
\begin{landscape}
\centering
\textbf{Same-position accuracy and paired probe-minus-AILM differences}\par\medskip
\includegraphics[viewport=0bp 351.72bp 799.2bp 655.92bp,clip,width=0.98\linewidth,alt={Enlarged same-position accuracy and paired probe-minus-ailm differences panels from the preceding external reference-probe overview, retaining original headlines, axes, intervals, and legends.}]{../../output/figures/external_benchmarks.png}
\clearpage
\textbf{Position accuracy by word length and letter-task calibration}\par\medskip
\includegraphics[viewport=0bp 0bp 799.2bp 342.72bp,clip,width=0.98\linewidth,alt={Enlarged position accuracy by word length and letter-task calibration panels from the preceding external reference-probe overview, retaining original headlines, axes, intervals, and legends.}]{../../output/figures/external_benchmarks.png}
\clearpage
\textbf{Letter-task variants and hosted stipulated-language runs}\par\medskip
\includegraphics[viewport=799.2bp 351.72bp 1202.4bp 655.92bp,clip,width=0.49\linewidth,alt={Enlarged letter-task variants panel from the preceding external reference-probe overview, retaining its original headline, axes, intervals, and legend.}]{../../output/figures/external_benchmarks.png}
\includegraphics[viewport=799.2bp 0bp 1202.4bp 346.32bp,clip,width=0.49\linewidth,alt={Enlarged hosted stipulated-letter and phone runs panel from the preceding external reference-probe overview, retaining its original headline, axes, intervals, and legend.}]{../../output/figures/external_benchmarks.png}
\end{landscape}
}
\endgroup
```

## External reference probes and same-position comparators

The finding and the reason it holds are not separable, so both are stated here
rather than quoted out of the main text. Over the
{{BENCH_LETTER_POSITIONS}}-position letter census the comparator scores
{{BENCH_LETTER_TOP1_AILM}} against {{BENCH_LETTER_TOP1_GEMMA3}},
{{BENCH_LETTER_TOP1_QWEN25}} and {{BENCH_LETTER_TOP1_SMOLLM}}. Much of that margin is *inventory compliance*, not
prediction, in a direction that differs by probe: the scored alphabet contains a space, space is the most frequent
symbol in the stream, and a system's score therefore depends in part on whether
it will emit a non-alphabetic symbol at all, which is a fact about what a model
is willing to produce. On the {{BENCH_LETTER_AZ_POSITIONS}} positions whose
target is alphabetic, where no system can score by emitting a space, the
comparator falls to {{BENCH_LETTER_AZ_TOP1_AILM}} and {{BENCH_MODEL_QWEN25_DISPLAY}} falls further
than it does, to {{BENCH_LETTER_AZ_TOP1_QWEN25}}. The contrast survives that
restriction, which is the result worth having; the raw census margin is not, and
a reader who takes the first number without the second has the arithmetic and
not the finding.

## Scope of the external reference evidence

What follows restates the conditions at the head of this section against what
the arms returned.

This measurement does not certify AILM competitiveness, compare perplexity
across different tokenizations, or claim that any external model registered here
is a generative model of the same kind. On the letter task it establishes less
than the headline numbers suggest: the full-inventory comparison is dominated by
whether a probe will emit a non-alphabetic symbol at all, and the task is
sensitive to how the prompt names the inventory. The separation does survive on
alphabetic targets alone, where that behavior cannot act, and every alphabetic
contrast clears the corrected family threshold. The phoneme and G2P
contrasts do not share the compliance weakness — both are scored on a held-out
split with no comparable compliance channel.

The alphabetic, prompt-format, and context-window controls state their detectability rather than leaving it open.
Power for the alphabetic, prompt-format and context-window arms is derived from
the discordant counts those arms already record, using the reference share and
significance level the recorded arms use, and the derivation is checked against
the arms that carry a recorded power. The largest of those figures is
{{BENCH_LETTER_AZ_POWER_SMOLLM}}.
The census changed what these controls support, and in the direction that is
easiest to under-report. At the sampled size every letter control was underpowered and
every contrast in them was unresolved; over the full pool the alphabetic arm's
contrasts clear the corrected threshold on effects larger than the reference
share, the instruct probes' context-window contrasts clear it, and the
{{BENCH_MODEL_GEMMA3_DISPLAY}} prompt-format contrast clears it. What remain unresolved
are the {{BENCH_MODEL_QWEN25_DISPLAY}} and {{BENCH_MODEL_SMOLLM_DISPLAY}}
prompt-format contrasts and the {{BENCH_MODEL_SMOLLM_DISPLAY}} context-window
contrast. The reading that survives is
narrower and firmer than the old one: the instruct probes are demonstrably
sensitive to prefix width and, for one of them, to prompt formatting, on
positions where AILM is invariant by construction. Sensitivity is not skill, and the accuracy columns
rather than the p-values are where a reader sees which it was.

{{BENCH_SYNTH_SCOPE_RESULT}} The locally generated stipulated artifact reduces
the exact-overlap threat but does not establish any external model's training
contents. A stipulated order-one chain is not English, so the complete
registered-probe measurement narrows one exposure concern without resolving the
English-corpus limitation.

Calibration is not a commensurable axis, and the section no longer treats it as
one. The comparator is scored on its own exact transition column; each
probe is scored on a reconstruction of its answer-slot distribution, truncated by
the server's candidate cap, filtered to the scored inventory, completed for the
symbols the cap hid, and renormalized. Brier and the negative log likelihood are
proper scores of that reconstruction and carry no strictly-proper interpretation
with respect to the external system, because the vector they score is not the
one that system emits. What the calibration arm reports is how the endpoint's
answer-slot report behaves once assembled into an inventory distribution, beside
a comparator whose distribution needed no assembly.

The gate deciding which systems that arm reports is itself measured here, and it
does not come out well.
Located offline against AILM's own letter columns, the mass above which
the two completion rules stop deciding the reported number is
{{BENCH_FLOOR_LOCATED_CAPTURED}}, some {{BENCH_FLOOR_LOCATED_RATIO}} times the
declared {{BENCH_FLOOR_DECLARED_FLOOR}}, and at the declared value the rules
still differ by {{BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE}} in expected
calibration error against an AILM total of {{BENCH_CALIB_ECE_AILM}}. The
declared floor is not conservative on that distribution family, and the located
one cannot be adopted, since reaching it needs more candidates than the server
reports. This does not show that the calibration numbers above are wrong: the
swept distributions are AILM-shaped and the probes' are far more peaked, so the
sweep's divergence at a given captured mass is not a prediction for them. What
it shows is that captured mass is a proxy for rule-dependence and an uncalibrated
one. The arm is therefore no longer gated on it: decision 93 adopts the direct
per-model rule divergence as the measurability decision, and
{{BENCH_CALIB_WITHHELD_COUNT_WORD}} probes' calibration ({{BENCH_CALIB_WITHHELD_LIST}})
is withheld on exactly that evidence.

What the section does is place a mechanistic discrete model beside zero-shot
external references under identical next-symbol units, on a letter
pool of {{BENCH_LETTER_POOL_UNIQUE}} distinct positions drawn from a longer
non-repeating held-out passage. The Brown and CMUdict arms do not control prior
exposure: those corpora are public and may have occurred in external-model
training data. The stipulated arm reduces exact-artifact overlap, but does not
prove absence from or independence of pretraining. Contrasts whose paired
interval spans zero are unresolved at this sample size even so.

## What a fair comparison with a prompted transformer would require

[@sec:external_benchmarks] scores AILM against {{BENCH_PROBE_COUNT_WORD}} externally served zero-shot
models — {{BENCH_MODEL_GEMMA3_DISPLAY}} [@gemma3_2025], {{BENCH_MODEL_QWEN25_DISPLAY}} [@qwen25_2024], and {{BENCH_MODEL_SMOLLM_DISPLAY}} [@smollm_2024] — on identical
next-symbol positions, and calls itself a
reference frame rather than a competitiveness claim. That disclaimer is load
bearing, so it needs an argument rather than a sentence.

Start with how the mechanistic comparator and the external systems differ. AILM's letter comparator is one
transition matrix over a {{STAGE0_LETTER_ALPHABET_SIZE}}-symbol inventory,
estimated by counting {{STAGE1_BROWN_BIGRAM_TOKENS}} adjacent pairs in a frozen
corpus; its prediction is the argmax of a column that exists whether or not
anyone asks for it. The probes are decoder-only transformers
[@vaswani2017attention], pretrained on corpora whose size and composition are
described only in summary, instruction-tuned afterwards except for the base
model {{BENCH_MODEL_SMOLLM_DISPLAY}}, and then queried
zero-shot in the manner @brown2020language established as an evaluation mode.
Four things differ at once — parameter count, training data, training objective,
and the interface through which a prediction is obtained — and the first two are
not separately adjustable in any case, since the compute-optimal frontier
[@hoffmann2022training] makes capability a joint function of both. "The same
size" and "the same data" are not two independent knobs one could hold fixed
while turning the other.

The interface is the confound the letter arm measures most directly. A probe's
answer is a token sequence produced in response to an instruction, restricted
here to an enumerated inventory by the server's structured-output path
[@willard2023efficient]. That restriction is what closes one failure channel:
the invalid-answer rates recorded for the probes are
{{BENCH_UNPARSABLE_RATE_GEMMA3}}, {{BENCH_UNPARSABLE_RATE_QWEN25}} and
{{BENCH_UNPARSABLE_RATE_SMOLLM}} — zero for the instruct probes and non-zero
for the base model — and every invalid answer counts as wrong and stays in the
denominator, so the comparison never rewards malformed output. What it leaves open is which legal symbol a model is willing
to emit. Of the {{BENCH_LETTER_POSITIONS}} scored letter positions,
{{BENCH_LETTER_NONALPHA_POSITIONS}} have a non-alphabetic target, and the
probes' accuracy on exactly those positions varies between them far more than
their accuracy on the alphabetic ones does ([@sec:external_benchmarks]). A
prompted model's prior over answer strings is a property of the model and the
instruction jointly rather than of the task [@zhao2021calibrate], so a
full-inventory letter score is partly a compliance score. That is why the
alphabetic-only arm over {{BENCH_LETTER_AZ_POSITIONS}} positions is reported
beside it, and why the separation surviving there is the result that carries
weight.

Prompt wording is a second interface effect, and it is not a nuisance that
careful phrasing removes. Changes that leave a task's content untouched can move
a prompted model's accuracy by more than the differences such comparisons
usually report, which is the case @sclar2024quantifying makes and quantifies
across formats. The benchmark measures one such perturbation on identical
positions, holding the inventory, schema and seed fixed, and over the full pool
that arm resolves for {{BENCH_MODEL_GEMMA3_DISPLAY}} but not
{{BENCH_MODEL_QWEN25_DISPLAY}} or {{BENCH_MODEL_SMOLLM_DISPLAY}}. The honest reading is unchanged and
now has a measurement behind it: every letter number here is a point drawn from a
distribution over plausible prompt formats that was never sampled, and the single
control run says how far the tested perturbation moves the answer, not how wide
the space is. Its decisiveness for {{BENCH_MODEL_GEMMA3_DISPLAY}} is a
reason to treat the unsampled space as a live threat rather than a caveat.

Contamination is uncontrolled on the English letter, phoneme, and G2P tasks and
cannot be resolved by assertion. Brown [@francis1979brown] and CMUdict [@cmudict] are longstanding,
widely distributed resources; that fact is relevant to possible training-data
exposure and does not override Brown's upstream non-commercial-use terms
([@sec:reproducibility]). No probe's training corpus is documented at the
granularity that would let anyone check for their presence — the difficulty
@dodge2021documenting demonstrates even for a corpus that *is* released — and
overlap is a property of a benchmark and a model together rather than something
a model report could settle once [@sainz2023nlp]. The synthetic arm is the
control, and it is narrow on purpose: a stipulated order-one chain over
{{BENCH_SYNTH_ALPHABET_SIZE}} symbols generated from a recorded seed, scored on
{{BENCH_SYNTH_POSITIONS}} independently generated evaluation trajectories. It
reduces exact-artifact overlap without certifying pretraining contents or
removing the other opportunity and interface differences. It preserves every registered run cohort while using a single hosted model,
but a stipulated chain is not English, so it licenses a
statement about transfer to a declared transition system and not a statement
about the English arms.

What would have to be equalized for a competitiveness claim is therefore the
following set of conditions, each of which is either unavailable or changes the object being
compared. Both systems would need the same training corpus and the same amount
of it, which for these probes is neither published nor reproducible. They would
need the same training objective, which for a fitted transition matrix and an
instruction-tuned transformer is not a shared quantity. They would need the same
interface — either both prompted, which a matrix cannot be, or both scored from
a distribution over the same declared inventory, which is available only where a
server reports enough candidates. Every arm would need independently verified
training-corpus non-exposure, which no arm here establishes. And the comparison would
need either enough discordant pairs to resolve the effects it reports, or a
declared equivalence bound where it cannot. The benchmark does not equalize
training-corpus identity and quantity, training objective, interface, or
unseen-corpus coverage, and it does not establish the inferential-resolution
condition.

One axis survives that list, and it is the reason the calibration arm exists.
Where both systems place a distribution over the same declared inventory they
are emitting the same kind of object, and accuracy is not the only thing worth
reading off it — the multi-metric position @liang2023holistic argues for, with
calibration as a first-class axis rather than an appendix to accuracy. Even
there the footing is not level: AILM's distribution is the exact column it takes
its argmax of, while a probe's is reconstructed from at most
{{BENCH_CALIB_TOP_LOGPROBS}} reported candidates over a
{{STAGE0_LETTER_ALPHABET_SIZE}}-symbol inventory, so part of it is a completion
rule rather than a model. [@sec:external_benchmarks] measures how much of a
published calibration number that rule can decide, found the gate that admitted
systems to the arm was a proxy that is not conservative where it was set, and
now gates the arm on the measured divergence itself. The consequence for this
section is sharper than a caveat. Calibration was the
axis this comparison had the best claim to share, and it does not survive
inspection either: the score is computed on a reconstruction rather than on the
distribution the external system actually reports, so no axis here is
commensurable in the strict sense the word implies.

The claim the benchmark supports is consequently narrow and stated as such: on
these positions, under this protocol, with these prompts, these are the numbers.
No arm of it orders the participating systems as language models, and this manuscript
makes no such ordering anywhere.
