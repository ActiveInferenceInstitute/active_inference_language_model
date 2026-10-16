# Extension protocols and implementation status

Status: **implementation granted and partially landed; P1's fitted probe
is still unmeasured.** P1's offline data/identity foundation, P2's artifact and offline
evaluator, and P3's artifact, registry entries, and L3 inlet/executor are
implemented and fixture-tested in this tree (per-partition state in [§5](#5-ownership-grant-boundaries-and-what-this-milestone-did-not-do)).
Measurement has since followed for two of the three: P3's case-word
influence sweep is scored (decision 121), P2's offline comparator arms and
zero-shot probe arms are recorded, and P1's warm-up compute gate is measured
(decision 122), while P1's fit pipeline itself has not run. Those values live
in `data/claims.json` and are reported in the manuscript, not here; the
sections below are the protocols as proposed. This file is deliberately *outside*
[`design/`](design/) — that directory holds adopted past-tense decision
records, and the measured arms remain governed by their own sequenced
registrations. No experiment may be marked complete on the strength of this
file. Proposal-era present/future tense in the detailed protocol sections is
historical; use the current status below and each actual recorder receipt when
planning work.

The connected runtime is now implemented separately from the original P3 seam.
Its raw-text path derives a delayed lexical UPOS observation from word beliefs;
P3’s earlier fixed-UPOS role-to-word influence estimand remains distinct. The
connected GUM comparison uses a frozen population and fixed channel setting,
not a retroactive claim that P3’s development sweep improved a whole pipeline.
See [the connected contract](design/connected_runtime.md) and
[the GUM data scope](reference/data_statement.md#universal-dependencies-english-gum--connected-evaluation).

The local registry described in §1 is not the identity of today’s hosted
synthetic/in-context model. Those recorders disclose the configured hosted model
and retain cohort suffixes for protocol accounting. Current loader, context,
endpoint and budget contracts live in the
[benchmark package guide](../src/ailm/benchmarks/README.md); the dated metadata
below cannot override them.

It responds to the three open Major items in [`../TODO.md`](../TODO.md):

| TODO item | Protocol here | Short form |
|---|---|---|
| "Equal opportunity to fit the task" | [§2](#2-protocol-p1--a-fitted-external-probe-arm) | Fit an actual registered probe's parameters offline on the task |
| "Extend the stipulated-artifact design to aligned pronunciation" | [§3](#3-protocol-p2--stipulated-aligned-spelling-to-pronunciation-design-b) | Pair-structured stipulated artifact, distinct from the shipped phone-chain arm |
| "The end-to-end top-down cascade is blocked by data provenance" | [§4](#4-protocol-p3--the-l4l3-pword--case-role-bridge-account) | The TODO-required label/split/observation account, built *before* any artifact |

Design discipline: cite code by `path::symbol`; every quantitative statement is
either a `claim:`-tagged ledger reference, a vendored-artifact field named in
inline code, a protocol constant, or an explicitly labeled estimate. No new
comparator, calibration relabeling, prompt tweak, or in-context example is
presented as a substitute for a fitted arm anywhere below.

Sources actually inspected while drafting (small official metadata/config/license
reads only; no weights downloaded, no inference run): local `ollama show` output
for the three registered tags; `https://huggingface.co/api/models/HuggingFaceTB/SmolLM-1.7B`
and `https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct` (model cards,
config lineage, licenses); the official `mlx-lm` fine-tuning documentation
(`https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/LORA.md` and the
repo `README.md`); `https://docs.ollama.com/` for the import path; plus every
repository file cited below. A `gemma3` HF checkpoint could not be read without
license acceptance; that limitation is recorded where it matters
([§2.8](#28-external-authorization-and-resource-requirements)).

---

## 1. The three registered probes, verified identity (evidence)

`src/ailm/benchmarks/constants.py::SERVED_PROBES` (lines 395–409) registers
exactly three systems, validated at import by `validate_served_probes`. Identity
facts below were read from the local Ollama registry (`ollama show <tag>`) on
2026-09-08; they predate the metadata-identity epochs of decisions 124-126, in
which the derived digests moved while the weights blobs did not — re-read the
live registry at implementation time and record the then-current digests; the
upstream lineage is traced from official metadata, and every
row states what is verified versus what must be pinned at implementation time.

| Registered key | Ollama tag | `ollama show` (verified locally) | Upstream lineage | License (as reported) | Fit backend family |
|---|---|---|---|---|---|
| `gemma3` | `gemma3:4b` | `gemma3` arch; completion **and vision** capabilities; Gemma Terms of Use | Instruct/multimodal line; exact HF checkpoint gated behind Gemma license acceptance | Gemma Terms of Use | **Unresolved** — `gemma3` is not in `mlx-lm`'s documented LoRA family list; see options in §2.8 |
| `qwen25` | `qwen2.5:3b` | `qwen2` arch; completion + tools; ships the Qwen system prompt → **instruct** lineage; Qwen RESEARCH LICENSE AGREEMENT text | `Qwen/Qwen2.5-3B-Instruct` (HF card: `qwen-research` license, base model `Qwen/Qwen2.5-3B`, `qwen2` architecture) — **verified via HF API** | Qwen research license: non-commercial research use; derivative redistribution per license terms | `mlx-lm` documented LoRA family (`Qwen2`) |
| `smollm` | `smollm:1.7b-base-v0.2-q4_K_M` | `llama` arch; **completion-only** (base model, no chat template); Apache License 2.0 | SmolLM 1.7B **base** line: `HuggingFaceTB/SmolLM-1.7B` (HF: `LlamaForCausalLM`, `llama` model type, Apache-2.0, ungated; card reports full-precision footprint `3422.76 MB`) — **verified via HF API**. The `v0.2` tag suffix denotes the refreshed base checkpoints of that same line | Apache-2.0 | `mlx-lm` documented LoRA family (`Llama`) |

Fidelity notes the protocol fixes, not optional caveats:

- The registered SmolLM probe is the **v0.2 base** checkpoint of the original
  SmolLM line — *not* a SmolLM2 model. The locally served
  `smollm2:135m-instruct-q4_K_S` blob is a different model family and must never
  stand in for the registered probe. A different model may be added explicitly
  as a new identity; it cannot silently substitute for a registered one.
- The registered Qwen probe is the **instruct** checkpoint (system prompt present
  in the served metadata), and the registered Gemma probe is **multimodal
  instruct** (vision capability present). Neither maps to a base/`-pt` text
  checkpoint by assumption.
- The served blobs are `Q4_K_M` quantizations. Those GGUF blobs are not the
  fit/merge target (foreign container format); the fit target is the upstream
  checkpoint of the registered lineage in the trainer's own precision, and
  QLoRA applies only to a quantized *MLX/HF-format* base, never to the served
  GGUF. Lineage is established by the digest chain (§2.3) and operational
  behavior by predeclared output checks (§2.6); the historical served weights
  keep their own source-revision uncertainty, and no agreement check removes it.
- Exact upstream repo revisions (commit SHAs on HF) are **not pinned yet** and
  are an implementation-time requirement, recorded as digests before any fit.

---

## 2. Protocol P1 — a fitted external probe arm

### 2.1 What exists today (the measured baseline this protocol extends)

The external benchmark (`src/ailm/benchmarks/`) already runs the three probes
and AILM's own decoders on identical next-symbol tasks, positions, prompts, and
scoring:

- Letters: comparator fit = `vendor/brown_counts.json` (schema
  `ailm-brown-bigram-counts-v2`, loaded by
  `src/ailm/levels/letter.py::load_brown_counts`); evaluation = the frozen
  non-repeating folklore passage (`src/ailm/benchmarks/constants.py::LETTER_TEXT`)
  deduplicated by `src/ailm/benchmarks/corpora.py::unique_letter_positions` and
  scored as a census (`constants.py::LETTER_CENSUS`, `claim:bench_letter_positions`
  in the ledger).
- Phonemes: comparator fit = the train side of the stride split
  (`src/ailm/levels/phoneme.py::g2p_split`, `test_every` = `5` over the vendored
  `vendor/cmudict_counts.json::g2p_pairs` list); evaluation = next-phone
  positions on the test side (`claim:bench_phoneme_positions`).
- G2P: same train/test stride; evaluation words from the test side
  (`claim:bench_g2p_words`), scored by exact match and the two alignment
  readings the supplement declares.
- Calls: one HTTP shape for every arm —
  `src/ailm/benchmarks/ollama.py::_generate_payload` fixes `temperature: 0`, a
  fixed `seed`, `num_predict`, the strict JSON `format` schema, and a
  content-addressed cache keyed by prompt and resolved model digest
  (`ollama.py::resolved_model_identity`). Prompts are fixed strings from
  `src/ailm/benchmarks/prompts.py::build_prompt` with a declared inventory
  phrase; an invalid answer is wrong, stays in the denominator, and is never
  re-prompted. One uncached warm-up precedes scoring (`harness.py` line 361).

The supplement's fairness table
([`manuscript/98_supplement_external_benchmarks.md`](manuscript/98_supplement_external_benchmarks.md),
"What a comparison would need") records that the condition **"Both systems have
the same opportunity to fit the task"** holds in no arm as run. The in-context
arm supplies labeled transitions through the prompt and measures that bounded
intervention; the fitted-probe arm is the first design in which the condition
could hold, which is exactly what the TODO demands.

### 2.2 What gets fitted — and the explicit list of what does not qualify

[]{#protocol-p1-fit-evaluation}

What is fitted: **low-rank adapters on the upstream checkpoints of the
registered lineages**, trained offline on the task's training material, then
fused and scored through the same server contract. Concretely, per task:

| Task | Fit material (train side only) | SFT format | Evaluated on |
|---|---|---|---|
| Letter | The normalized Brown stream regenerated from the pinned NLTK `brown.zip` recorded in `vendor/brown_counts.json` `provenance.raw_source` (revision `550b6625…`, `source_sha256` recorded) under the artifact's recorded normalization string; instances per the deterministic policy below | The exact `build_prompt("letter", …)` format, completions-style, gold = the stream's next character | The frozen census positions on `LETTER_TEXT` (folklore passage — a different corpus than Brown, so eval-position disjointness holds by construction and is digest-asserted) |
| Phoneme | Phone streams of the train pairs from `g2p_split`; instances = (prefix window, next phone) in the exact phoneme prompt format | Same, phoneme task | The frozen test-side positions (`claim:bench_phoneme_positions` population) |
| G2P | Train pairs (word, gold phone array) from the same stride split | `completions` format: `{"prompt": build_prompt("g2p", word), "completion": <JSON gold>}` | The frozen test-side words (`claim:bench_g2p_words` population) |

**Deterministic training-instance policy (declared before any evaluation):**

- **Objective: streaming exact census.** The fitting objective is the full
  census of the declared fit material, computed by a streaming pass — no
  subsampling, no thinning for resource convenience. Letters: one supervised
  target per stream position at stride one (target = the stream's next
  character, context = the scored prefix window), so supervised-target
  exposure equals the comparator's count fit exactly — the vendored artifact's
  `provenance.bigram_tokens` field is the per-pass target count. Phonemes:
  one target per phone position inside each train pronunciation, prefix from
  the word's start (the scored phoneme task conditions on the whole prefix —
  `corpora.py::phoneme_positions` builds `phones[:i]` prefixes), reset per
  pair — exactly the `phone_statistics` per-pair contract. G2P: one instance
  per train pair.
- **Exposure is two different quantities, both declared.** Supervised-target
  parity is exact (above). Conditioning-token exposure is **not** equal and is
  not claimed to be: each letter instance re-pays its window plus the prompt
  template, putting conditioning tokens per letter pass roughly two orders of
  magnitude above the count fit's input (an estimate from artifact-field
  arithmetic; the exact per-run number is measured at the warm-up gate below
  and recorded in the receipt). The census objective stands anyway: it is the
  only objective that preserves the declared "same training material" claim.
- **Efficiency options, each with its objective/attention assumptions
  (proposed, none adopted by default):** streaming and prompt-caching — no
  objective change, per-instance conditioning unchanged; minibatching — same
  per-instance objective, batch-level throughput only; **block-masked
  packing** — several instances per sequence under per-instance attention
  masks that reproduce each instance's exact conditioning, permitted only with
  a written mask contract and a verified equivalence check against
  single-instance conditioning; weighted grouping of byte-identical duplicate
  instances — same objective under exact-duplicate weighting. **Unmasked
  packing is forbidden**: cross-instance attention would inject exactly the
  excluded in-context signal.
- **Rejected remedy, recorded so it stays rejected:** uniform stride-d
  subsampling of the census. It does **not** preserve bigram relative
  frequencies and is not automatically unbiased — an alternating stream
  sampled at stride two sees only same-symbol successors and loses every
  alternation (the direct aliasing counterexample, implemented as a checkable
  fixture in `src/ailm/benchmarks/fitted/census.py`). Striding is sampling;
  the full census stands.
- **Compute gate (before any full fit).** A measured trainer warm-up
  throughput pilot and a checkpoint/resume resource check are recorded before
  the census-times-epochs budget is committed; the token budget is declared
  from the measured rate, and the fit proceeds only when the gate passes. No
  throughput or duration figure is claimed in this document — every such
  number in the review was an unmeasured estimate, and receipts record only
  measured telemetry.
- **Diagnostic separation (declared seam).** The diagnostic carve is
  positional, taken **before** any shuffle: a contiguous tail block of
  nonoverlapping target positions behind a declared seam gap of at least the
  window width, so no conditioning window reaches across the seam; skipped gap
  positions belong to neither set and are counted in the receipt. Every
  diagnostic contrast runs against the **experimental native comparator refit
  on exactly the fit positions** — never against the full-corpus counts, which
  remain the shipped reference for scored arms only. The fit-set census is
  preserved whole in the data records and receipts. If a run skips the carve,
  the diagnostic is named an in-sample training diagnostic and is never called
  held-out validation.

This is a genuine offline parameter fit on the task: gradient updates to learned
parameters, frozen and digested before any scored call. It is **not** any of the
following, each excluded by the TODO's framing and recorded here as a refusal
boundary, not a gap:

- in-context demonstrations (the existing `bench_ctx_*` sweep arm);
- hand prompt tuning (no prompt is authored, searched, or edited for score);
- calibration-only relabeling or output-rule changes (the response contract,
  parser, and inventory are byte-identical to the shipped arms);
- supplying AILM's fitted column (the permanently rejected circularity —
  [`../TODO.md`](../TODO.md) "Rejected, with reasons");
- a new classical comparator standing in for the probe (the S2 erratum's Design A
  is a different work item and closes nothing here).

### 2.3 Arms, artifact identity, and the registry boundary

Because the fit target is the upstream checkpoint (in the trainer's precision),
not the served GGUF blob, each registered probe family gains a declared arm set:

| Arm | Identity | Role |
|---|---|---|
| Registered (existing) | `SERVED_PROBES` tag, `Q4_K_M` | Published reference; untouched |
| Upstream-unfitted control | Upstream checkpoint → fused → converted GGUF → local Ollama tag | The treatment's paired control through the identical export pipeline |
| Upstream-fitted treatment | Upstream checkpoint + LoRA adapter fused → converted GGUF → local Ollama tag | The fitted arm |

**Registry boundary.** The fitted arms do **not** enter `SERVED_PROBES`. That
registry is the single identity source for every historical bench family, and
adding to it would silently rewrite live surfaces:

- Recorders derive their model set from it — `MODELS = dict(PROBE_MODEL_TAGS)`
  at `scripts/record_external_benchmark_claims.py:166`,
  `scripts/record_incontext_benchmark_claims.py:151`, and
  `scripts/record_synthetic_benchmark_claims.py:140` — so every future run of
  any recorder would expand its arm set and its expected-call/receipt
  arithmetic.
- Manuscript hydration fails closed on registry change
  (`src/ailm/manuscript_variables.py:2286-2293`) and iterates `SERVED_PROBES`
  demanding per-probe claims in every historical family (lines 2466, 2632,
  2949, 3400, 3434, 3474, 3565, 3613, 3687) — a fitted probe without
  `bench_synth_*`/`bench_ctx_*`/calibration claims would break the whole
  render.
- Figures derive inventories from the registry
  (`src/ailm/figures/benchmarks.py:31-36`;
  `figures/incontext_sweep.py:128-132` asserts an exact match), and the
  instruct/base classification (`constants.py::BASE_MODEL_PROBE_KEYS`) would
  need extending.
- The package's own rule (`src/ailm/benchmarks/AGENTS.md:94-96`) makes adding a
  probe a whole-surface change: a `ProbeSpec`, warmed cache, re-recorded
  dependent claims, and a supplement update in one change.

Repair: a **separate experimental registry** `FITTED_PROBES`, defined in
`src/ailm/benchmarks/fitted/identity.py` with its own import-time validator
(the `validate_served_probes` dedup discipline on a new namespace), consumed
only by the fitted recorder, its tests, and explicitly listed consumers.
Historical `SERVED_PROBES`, its populations, and its namespaces are untouched;
**no historical namespace is rewritten** (the earlier "nothing is re-recorded"
claim was false at the registry surface and is withdrawn). Cache identity still
binds every scored call to the fitted weights via `resolved_model_identity` —
distinct local tags give distinct digests and distinct keys.

Identity requirements (all implementation-time, all digested):

1. Fitted-registry entries (above) — never `SERVED_PROBES` entries.
2. Upstream revision pin: HF repo id + commit sha256 recorded *before* the fit,
   in a new provenance record under the vendored-artifact discipline
   ([`provenance/sources.md`](provenance/sources.md) entry with digests; see
   [`provenance/AGENTS.md`](provenance/AGENTS.md) for the regeneration contract).
3. Adapter artifact: config (rank, layers, schedule) + learned weights +
   training-loss curve, written once, digested, never overwritten
   (`synthetic_protocol.py`'s frozen-contract discipline is the template).
4. Merged/converted artifact digest and the local Ollama tag it was imported
   under (official import path: a Modelfile whose `FROM` names the converted
   GGUF — see `https://docs.ollama.com/`), recorded so the cache identity of
   every scored call binds to the fitted weights, exactly as
   `resolved_model_identity` binds registered calls today.
5. A new protocol namespace (`fitted-probe-v1`, the pattern of
   `CALIBRATION_FLOOR_PROTOCOL`) with its own cache protocols, so no key can
   collide with any existing arm; repeat-trial scoping reuses
   `repeat_trial_protocols` with a fresh run scope.
6. **The digest chain establishes lineage**: upstream revision digest →
   adapter digest → fused-model digest → converted-GGUF digest → served-tag
   identity (`resolved_model_identity`), recorded as one receipt before any
   scored call. Lineage rests on this chain; behavioral agreement checks only
   validate operation (§2.6).
7. **Precision and quantization choices (proposed, not run).** Primary: LoRA on
   the bf16 MLX upstream, fuse at full precision, convert with `llama.cpp`,
   quantize to `Q4_K_M`. Rationale: the identical conversion/quantization
   pipeline applied to treatment and control makes quantization a controlled
   common factor of the specified paired comparison, and keeps fitted numbers
   in the registered arms' quantization family. A `Q8_0` sensitivity rung is
   optional. QLoRA (fitting against an MLX-quantized base) is the declared
   fallback; it changes the fused artifact's identity (quantized base +
   adapter) and would be recorded as such. Matched quantization supports the
   paired comparison; it does not establish identity with the historical
   served weights, does not remove transformation effects, and the
   fit-by-quantization interaction (the adapter is absorbed before
   quantization) stays declared.

### 2.4 What is and is not equalized (declared, per condition)

| Condition (the supplement's six) | Status under P1 |
|---|---|
| Same items, scoring rule, denominator | Holds as shipped; unchanged |
| The score measures the ability the task names | **Still fails on the full-inventory letter arm** (willingness to emit a non-alphabetic symbol is not next-symbol ability); the letter fit inherits this, and both letter readings stay reported separately |
| Neither system has prior exposure to the evaluation corpus | Unchanged: unverifiable for the probes; letter-arm corpus disjointness holds by construction, the CMUdict stride split by code |
| Same opportunity to fit the task | **Holds by construction for the first time**: both systems receive offline parameter estimation on the same training material under the census policy above. Declared residual: the estimator *classes* still differ (count-estimated matrix vs gradient-fitted adapter) — the condition is equalized, the compared objects remain different estimator kinds |
| Same interface | **Not equal**: AILM's comparator arms are native in-process decoders (the harness's letter/phoneme/G2P comparator paths — no prompt, no HTTP, no output format), while every probe arm answers the JSON/temperature-`0` HTTP contract. Within the probe side, the fitted and unfitted arms share the interface exactly. Condition five stays non-equal |
| Resolvable contrasts | Declared per arm; the fitted contrasts get their own declared family |

Two protocol constants that are **not** fairness conditions, labeled as such:

- **Optimization budget — frozen-declared, not an equalization.** The probe fit
  gets a fixed epoch/iteration budget (policy above) with selection only on
  inner-train validation. AILM's count fits have no budget dial, so there is
  nothing to equalize; this bounds the fitted arm's compute.
- **Quantization matching — a paired-comparison control, not identity**
  (§2.3 item 7).

The honest summary: P1 equalizes *supervision and fitting opportunity*;
interface, capacity, prior exposure, and the measure-the-named-ability
condition remain failed exactly as the supplement records them. No claim is
made that every fairness condition is now satisfied — the fourth condition
moves from "holds nowhere" to "holds in this arm, with declared residuals",
and the comparison remains a reference frame, not a competitiveness claim.

### 2.5 Separation, freeze, and measurement order

1. **Freeze evaluation** first: positions, prompts, parser, scorer are already
   frozen constants; record their digests in the run receipt.
2. **Build fit artifacts** (train-only instances, schema below); record sha256
   + provenance (raw-source identity, generator path, seed) before any fit.
3. **Select** hyperparameters (rank, iters, learning rate, batch) on an
   inner-train validation split carved from the train material (the contiguous
   diagnostic tail and seam defined in §2.2, disjoint from fitting instances); never read a scored population during
   selection.
4. **Fit** and freeze: adapter digest, config digest, loss curve, and the
   exact trainer command recorded.
5. **Fuse, convert, quantize, serve**: fused GGUF digest; `ollama create` tag;
   `resolved_model_identity` digest recorded before any scored call.
6. **Score once, in declared order**: the unfitted-upstream control arm first,
   then the fitted arm, interleaved under a declared arm-order policy (the
   `alternating-full-short-by-position-v1` pattern); one uncached warm-up per
   model per arm before scoring; repeat trials under a fresh run scope with the
   existing `repeat_trial_protocols` machinery.
7. **Record** under the new namespace with the same wholesale-replacement,
   single-writer-lock, refuse-partial-run discipline as
   `scripts/record_external_benchmark_claims.py`; bind receipts through
   `scripts/verify_network_claim_bindings.py`.

### 2.6 Controls (gate liveness, leakage, lineage, legitimate outcomes)

The project's own rule: no gate without a live positive *and* negative control
(`GateVacuousError` discipline in [`../docs/gate_design.md`](gate_design.md)
and [`testing_philosophy.md`](testing_philosophy.md)). P1's controls, with
tolerances, seeds, and margins fixed before fitting:

- **C1 — fit-consumption check (declared tolerance and seeds).** On the
  diagnostic slice (never scored positions), the fitted arm is compared with
  its unfitted twin and with the **named empirical reference: the experimental
  native comparator's own accuracy on the same slice**, refit on the fit
  positions per the §2.2 seam. That reference is an empirical comparator,
  **not a mathematical ceiling**; the saturation diagnosis reads the unfitted
  twin against it, never against an idealized bound. Outcomes, all declared
  before fitting: a declared-margin lift over both = supervision consumed;
  **no lift where the unfitted twin is already at or near the empirical
  reference = saturation** (the base already solves the task at this slice — a
  reportable finding, not dead instrumentation); no lift without saturation =
  the fit did not consume its supervision (instrumentation suspect → block).
  The diagnostic is computed before any scored call.
- **C2 — shuffled-supervision control (a floor, not a stop sign).** A refit at
  the same budget on deranged supervision (pair labels deranged; stream labels
  permuted with a declared seed). The permutation preserves the label
  marginal, so a refit can legitimately lift via **marginal/format learning**;
  C2 therefore *measures that floor* rather than expecting zero. The C1 lift
  is attributed to supervision only when it exceeds the C2 floor by the
  declared margin; a C1 lift at or below the C2 floor is recorded as
  marginal/format learning — a legitimate outcome, never a filter that
  silently keeps only successful runs.
- **C3 — export-pipeline operational validation (not lineage proof).** Lineage
  rests on the §2.3 digest chain. C3 runs the predeclared output checks —
  response-contract compliance, warm-up agreement, repeat-trial agreement,
  prompt-token ceilings — and scores the unfitted-upstream arm on the same
  frozen positions as the registered served arm under a **pre-declared
  tolerance**. That agreement supports the specified paired comparison; it is
  not parameter identity, does not remove the historical served weights'
  source-revision uncertainty, and does not exclude every
  quantization-transformation effect. Divergence beyond the tolerance refuses
  the run (conversion defect); agreement is never narrated as lineage proof.
- **C4 — structural leakage guards (tests, not prose).** (a) No scored-eval
  material in any fit artifact: for letters, the folklore passage and the
  Stage 1 fixture are not Brown — asserted by corpus identity plus a digest
  check that the SFT file contains none of their normalized text; for CMUdict,
  the stride split is enforced by `g2p_split` and asserted on the SFT file.
  (b) No AILM output in any training file: no comparator predictions, no fitted
  columns, no level outputs — a source-level test binding the generator to the
  declared input list. (c) No `deprel`/`feats[Case]` anywhere in P1 (it has no
  role here; the guard mirrors the Stage 4 trap statement).
- **C5 — optional, cheap:** a broken-supervision fit (letters: every target
  replaced by the transposed bigram successor — the Stage 1 matrix-transpose
  defect class pinned at `tests/test_stage1_gate.py:335`) must behave like
  C2's floor rather than exceed it — the same refusal *shape* as
  `scripts/record_synthetic_benchmark_claims.py:200`
  (`_phone_comparator_refusals`), whose thresholds were derived from that
  artifact's own offline arithmetic, not transferred.

Failure criteria for the protocol as a whole — **blocks only**: a C4
leakage-guard failure; a response-contract or receipt failure; the backend
unavailable on the target architecture; C1 no-lift *without* the saturation
diagnosis. **Reportable, non-blocking outcomes** (all recorded, never filtered
post hoc): supervision-attributed lift; no-lift with saturation; marginal/
format learning (C1 lift at or below the C2 floor); fitted arm at or below
unfitted on the scored positions after C1 resolved — the Stage 2 lesson that a
fitting arm can fail the bar it was built for.

### 2.7 Expected files, dependencies, feasibility (estimates)

Original proposed file layout (not a current module inventory). The implemented
foundation is indexed in [its package README](../src/ailm/benchmarks/fitted/README.md);
this table preserves the proposed remaining trainer/export boundaries:

| Proposed file | Owns |
|---|---|
| `src/ailm/benchmarks/fitted/__init__.py` | Package surface, registry access (mirrors the package's lazy re-export discipline) |
| `src/ailm/benchmarks/fitted/identity.py` | Upstream pin, adapter/merge/quantization digests, and the separate `FITTED_PROBES` registry with its own validator |
| `src/ailm/benchmarks/fitted/data.py` | SFT artifact builders (train-side only), digests, leakage-guard hooks |
| `src/ailm/benchmarks/fitted/driver.py` | Trainer invocation (declarative config, recorded command), validation split, freeze checks |
| `src/ailm/benchmarks/fitted/export.py` | Fuse → convert → quantize → local `ollama create`, digest binding |
| `scripts/generate_fitted_probe_data.py` | Writes the vendored SFT artifacts under the provenance discipline |
| `scripts/record_fitted_probe_claims.py` | The recorder: wholesale namespace, single-writer lock, receipts, refused partial runs |
| `tests/test_fitted_probe_protocol.py` | Offline tests: registry coherence, digest checks, C1/C2 logic on fixtures, no-network invariant |

The import contract is respected: the new package touches the *probe* side
only, imports no pymdp, and adds nothing to any level build or gate
([`architecture.md`](architecture.md) — the probe package is already outside the
pymdp stack by design).

Original proposed fitting dependencies. Availability and versions must be
verified in the separate trainer environment before use; this historical list
is not a claim about the current AILM virtualenv:

- `mlx` + `mlx-lm[train]` — the official Apple-silicon trainer. Official docs
  (`mlx_lm/LORA.md`): LoRA/DORA/full fine-tune via `mlx_lm.lora --model …
  --train --data <dir with train.jsonl> --iters N`; `completions`
  (`{"prompt", "completion"}`) and `text` formats with `--mask-prompt`;
  QLoRA when the base is quantized; fusion via `mlx_lm.fuse`, whose
  `--export-gguf` is documented for **Mistral/Mixtral/Llama-style fp16 only**.
  Supported fine-tune families per the docs: Mistral, Llama, Phi2, Mixtral,
  **Qwen2**, Gemma, OLMo, MiniCPM, InternLM2 — the registered SmolLM (llama
  arch) and Qwen (qwen2 arch) are in-family; **gemma3 is not**.
- `llama.cpp` conversion/quantization binaries (`convert_hf_to_gguf.py`,
  `llama-quantize`), pinned to a recorded version, for the qwen2 export path
  (fuse to safetensors, convert, quantize) and for any quantization step on the
  llama-style path beyond what `mlx_lm.fuse --export-gguf` emits.
- Ollama Modelfile import (official `https://docs.ollama.com/`) to serve the
  converted models under local tags.
- No `datasets`/Hugging Face loader dependency is needed: fit material is local
  (pinned Brown zip re-fetch and the vendored CMUdict artifact), delivered as
  local `.jsonl` files.

Feasibility — **estimates, explicitly not measured**: the machine is a Mac with
the `hw.memsize` recorded in this session's inventory (twenty-four gibibytes of
unified memory). The SmolLM 1.7B bf16 checkpoint's full-precision footprint is
card-reported at `3422.76 MB` (inline, model-card fact); the qwen2 3B checkpoint
is roughly twice that by parameter count (estimate); both LoRA fits are expected
to sit comfortably inside unified memory with the official trainer's documented
memory controls (`--batch-size`, `--num-layers`, `--grad-checkpoint`,
`--grad-accumulation-steps`). Disk space is ample (the drive reports large free
capacity). **No wall-clock, token-throughput, or cost estimate is claimed
anywhere**; the protocol instead fixes a declared optimization budget before
fitting and records the trainer's own loss telemetry as the receipt.

### 2.8 External authorization and resource requirements

Distinguished honestly, per the milestone instruction not to invent
authorization theater for routine public fetches:

- **Routine public downloads (no new authorization):** the pinned NLTK
  `brown.zip` re-fetch (already the project's own corpus-practice —
  `src/ailm/corpus/brown.py` authenticates bytes against the recorded
  `source_sha256` before use), and the un-gated HF checkpoints
  (`HuggingFaceTB/SmolLM-1.7B`, `Qwen/Qwen2.5-3B-Instruct`). Same class as the
  existing CMUdict/UD download paths.
- **A genuine external requirement:** `google/gemma-3-4b-it` is license-gated on
  HF — fitting that registered line's upstream requires accepting the Gemma
  terms under an authenticated account. This is the one real
  authorization/account action in the protocol, and it is contingent (below).
- **License constraints to declare in any resulting manuscript:** Qwen research
  license (non-commercial research; derivative redistribution per license
  terms); Apache-2.0 for SmolLM (unrestricted); Gemma terms if that arm ever
  runs.
- **What still cannot be done from here at any price:** equalizing the probes'
  *pretraining* exposure or contents. Declared, per condition three.

### 2.9 Success and failure criteria

- **Blocks (recorder refuses, no claims):** a C4 structural leakage-guard
  failure; a response-contract or receipt failure; the backend unavailable on
  the target architecture; C1 no-lift *without* the saturation diagnosis.
  Each block has a named refusal, in the style of the existing synthetic
  recorder's pre-registration refusals.
- **Reportable outcomes, never filters:** supervision-attributed lift;
  no-lift with saturation; marginal/format learning (C1 at or below the C2
  floor); fitted arm at or below unfitted on the scored positions. No outcome
  class is dropped post hoc; every declared outcome is recorded.
- Pass = declared controls resolved, every declared outcome recorded, and the
  scored eval arms reported under their family correction with ledger tags
  written by the new recorder.

### 2.10 Ordered implementation partitions (ownership boundaries)

| # | Partition | Grant needed | Test/failure criterion |
|---|---|---|---|
| 0 | Fit-artifact generators + identity module + offline tests (no model calls, no network) | source/tests | **Implemented and fixture-tested in this tree (E1R2: `src/ailm/benchmarks/fitted/` identity/census/data/receipt + thin CLI)**; digest-stable regeneration; C4 structural tests pass offline |
| 1 | SmolLM letter fit end-to-end (fit, fuse, convert, serve, score, record) | network fetch (public), local serving, training, recorder | **Prerequisites in order:** measured warm-up throughput pilot + checkpoint/resume resource gate (§2.2), then controls per §2.6; receipt verifies; ledger namespace complete or the run refuses. Full-corpus stream preparation runs under its own scheduled handoff |
| 2 | Phoneme + G2P fits on the same skeleton | same | Same, per task |
| 3 | Qwen arm (qwen2 export path via llama.cpp) | same | Same |
| 4 | Gemma arm | **only with the gated-license acceptance resolved** | Same; blocked until then |
| 5 | Manuscript/supplement integration | manuscript grant | Not in this milestone |

Serialization boundary: the fitted registry's frozen contracts, the recorder
receipt schema, and (for P2) the synthetic protocol registry are shared
surfaces — one writer per change, per the benchmarks package's ownership table.
`SERVED_PROBES` and every historical namespace are outside this protocol's
write surface.

---

## 3. Protocol P2 — stipulated aligned spelling-to-pronunciation (design B)

[]{#protocol-p2-aligned-correspondence}

A pair-structured stipulated artifact, structurally distinct from the shipped
order-one continuation chains (evidence: the existing arms are single-symbol
next-symbol tasks over a bigram chain — `src/ailm/levels/synthetic.py`, its two
`SYNTHETIC_*_CHAIN_VERSION` constants, and the flat bigram-count artifacts
`vendor/synthetic_counts.json` / `vendor/synthetic_phone_counts.json`; the pairwise generator and aligned artifact were absent when this protocol was
drafted; they now exist under the P2 implementation described above).

### 3.1 The stipulated correspondence and population

- A **stipulated per-letter correspondence** `T`: the letter inventory → a
  fixed subset of the canonical ARPAbet inventory (claim:phoneme_count_39
  symbols), drawn by a named seeded generator and written into the artifact.
  Deliberately *not* an English letter-to-phone reading; the exact table is
  recorded, and "the inventories are met, the stipulated correspondence is
  new" is the design B analogue of the phone arm's framing under decision 108.
- Spelling words are **finite constructed objects**, not open-ended draws:
  each word is a fixed-length window of `L` letters (proposed `L = 6` — long
  enough to be word-like, short enough to bound the collision space and keep
  prompts tiny) cut from an independent stipulated letter-chain trajectory,
  one word per trajectory under consecutive integer seeds (the existing
  held-out trajectory pattern). The chain alphabet is exactly the 26 lowercase
  letters with no separators, so normalization is the identity on the
  alphabet. **Seed-domain separation alone proves nothing about disjointness**
  — the disjointness check is set-theoretic: within-split duplicates and
  cross-split collisions are asserted absent at generation, with a bounded
  deterministic resample (proposed cap: 64 attempts per word, the
  `derangeable_draw_attempts` precedent) and a `SystemExit`, no-claims refusal
  if any collision survives; the artifact records both word-set digests and
  the recorder re-asserts disjointness before any call. Each word's
  pronunciation is `T` applied per position; train words come from the corpus
  seed domain, evaluation trajectories from the held-out seed domain.
- Comparator (AILM side): a newly fitted per-letter map decode from *stipulated
  train pairs* — exact closed form per position (the same "closed-form path
  identity" style the project already prefers), beside a context-free
  bigram-over-pronunciations comparator fitted on the same train pairs.

### 3.2 Artifact schema (proposed)

`vendor/synthetic_aligned_g2p_counts.json` (schema name proposed
`ailm-synthetic-aligned-g2p-v1`), written by a new
`scripts/generate_synthetic_aligned_g2p.py`:

```
schema, provenance {
  chain_version, correspondence_version, corpus_seed, held_out_seed,
  word_length_L, resample_attempts_cap,
  generator_path, letter_inventory_source, phone_inventory_source,
  train_words_digest, test_words_digest, correspondence_table_digest
},
correspondence {letter -> phone, complete over the letter inventory},
splits {train: [(spelling, phone_string)], test: [(spelling, phone_string)]}
```

Registered in [`provenance/sources.md`](provenance/sources.md) with its sha256
and enforced by `tests/test_provenance_checksums.py`, exactly like the two
existing synthetic artifacts (`32b00fa2…` and `319b3926…` are the current
precedents recorded there).

### 3.3 Fit/evaluation boundaries and the adversarial checks

The design-B erratum (artifact locator:
`output/reports/omp-dispatch-20260908/evidence/scholar-S2-primary-evidence/ERRATUM-S2.md`)
withdrew Design B's "generator-owned, so no labels exist to leak" reasoning as
too strong: generator ownership does not certify absence of leakage or of
unknown pretraining exposure (`src/ailm/levels/synthetic.py`'s own docstring
says the same for the whole stipulated family). The checks below are
therefore *structural*. **Closure statement:** completing this protocol —
artifact, comparator controls, and the recorded evaluation — can close the
TODO's aligned-pronunciation design item; unknown pretraining exposure remains
the standing limitation of the stipulated family, and the artifact is never
called contamination-free.

- **L1 (prompt purity):** the task prompt for this arm is a pure function of the
  spelling string and the declared inventory phrase — a frozen prompt builder
  with a structural test asserting no correspondence-table constant, no English
  G2P hint, and no gold phone string can enter any prompt (mirrors the exact
  purity contract of `src/ailm/benchmarks/prompts.py::build_prompt`).
- **L2 (gold isolation):** evaluation gold strings exist only in the scorer;
  the response contract is the existing strict JSON array parser
  (`parse_g2p_answer`-shaped); the test suite pins that the scorer's gold is
  never serialized into any request payload (replay the recorder's payload set
  offline and assert).
- **L3 (train/eval word disjointness):** the §3.1 set-disjointness assertions
  (within-split duplicates, cross-split collisions) re-asserted by the
  recorder before any call — not seed-domain separation alone.
- **A1 — English-transfer refusal:** a comparator fitted on English CMUdict
  pairs (the vendored `g2p_pairs`) must fail to clear the stipulated arm's bar
  on the identical positions. **No threshold is carried over:** the phone
  arm's numbers came from offline arithmetic on its own artifact
  (`scripts/record_synthetic_benchmark_claims.py:200`,
  `_phone_comparator_refusals`); this arm's numeric gates are derived the same
  way — pre-registered arithmetic on the frozen artifact at finalization time,
  before any probe call. A fired A1 means the stipulated signal is not
  separable from English transfer — an attribution ambiguity and a legitimate
  finding, not a silently retried configuration.
- **A2 — broken-correspondence control:** a **shape-valid broken map**
  `T' = σ∘T`, where σ is a declared seeded permutation of `T`'s phone codomain
  (complete, deterministic, same codomain — well-defined for a rectangular
  letter→phone map, where "transpose" is not). A comparator fitted on `T'`
  must not clear the bar on `T`-generated data, under the same pre-registered
  arithmetic as A1; a fired A2 is the same legitimate attribution finding.
- **A3 — uniform floor** on the same positions.
- **A4 — mapping-control (data-level positive control):** the exact
  (spelling→phone) tuples serialized into any demonstration prompts must
  separate treatment from a character-level multiset-matched derangement — the
  `bench_ctx_*` mapping-control pattern — so a probe-level non-separation
  cannot be misread as a vacuous stimulus.

### 3.4 Arms, protocol, claims

- Comparator arm recorded offline first, with A1–A3 as recorder refusals
  (gate not live → `SystemExit`, no claims — the refusal pattern of
  `scripts/record_synthetic_benchmark_claims.py:200`).
- **Named model arm and dataflow.** The AILM-side model for this arm is the
  shipped G2P engine refit on the stipulated train pairs:
  `src/ailm/levels/phoneme.py::build_g2p_emission` accepts arbitrary aligned
  `(word, phones)` pairs, and stipulated words satisfy its letter/phone 1:1
  alignment restriction by construction; decoding reuses the benchmark
  comparator path. The shipped Stage-3 trie-constrained word agent is **not**
  instantiated on this artifact, and the closed-form `T` lookup is a
  **separately named baseline** — the trivial map is never presented as the
  shipped G2P. The decode path is the shipped three-matrix filter:
  `build_g2p_emission` (add-k over letters, its own emission smoothing dial) +
  `build_g2p_transition` (train-only pronunciation bigrams through
  `build_phone_transition`, add-k) + `build_g2p_start` (first-phone unigram,
  add-k), read out by `_ailm_g2p_decode` — per-letter Bayesian filtering with
  argmax readout, whole-word conditioning (no window truncation), default
  add-1 Laplace throughout (`ailm.config.DEFAULT_LEVEL_CONFIG`). The P2
  context-free pronunciation-bigram comparator follows the same smoothing
  contract (`build_phone_transition`: an unseen bigram lands at
  `k/(n + k·39)`; empty prefix falls back to the unigram argmax;
  `phone_statistics` resets per pair). The phoneme SFT window is tied to the
  scored English phoneme task: full prefix from the word's start
  (`phoneme_positions` builds `phones[:i]`), per-pair reset — not the
  synthetic arm's shortened query window. Dataflow: train pairs → engine fit
  (all three matrices) plus the context-free bigram fit; test spellings →
  model inputs only; gold phone strings → scorer only; optional
  demonstrations → drawn exclusively from train pairs, keyed by declared
  per-position seeds (the `bench_ctx_*` draw pattern), never test gold;
  prompts carry no table constants (L1) and no gold (L2).
- Probe arms (optional, Ollama-slot discipline as shipped): zero-shot
  spelling→pronunciation under the existing 39-symbol response inventory
  (claim:bench_synthphone_inventory_size names the same inventory), plus an
  explicitly optional demonstration sweep with the A4 derangement control.
- A new frozen protocol dataclass beside `SyntheticZeroShotProtocol` with its
  own version string, run id, and cache protocol — the registry's distinctness
  requirement (`synthetic_protocol.py::_validate_protocol_registry`) extended
  to a fourth member; a new `bench_synthg2p_*` label namespace with its own two
  declared families; recorder arm entry extending the expected-call arithmetic;
  receipt re-derivation extended to the new namespace.
- **At the drafting milestone, nothing was recorded.** The above specifies the
  proposed schema; the status at the top of this document records subsequent
  implementation and measurement. Recording is an execution-ordered action, not an approval gate: the
  arm records only after the artifact finalizes and Root assigns the recorder
  ownership; the standing authorization covers the measurement itself once the
  protocol is sound.

### 3.5 Success/failure and cost

Success: stipulated-alignment comparator clears A1–A3 on identical positions
(mechanism works under a true correspondence); probe arms reported descriptively
under declared families. Failure: any refusal fires, or the comparator fails
under the condition it was built for — reported, not repaired.
Cost: one generator + one artifact + recorder arm (offline); probe arms need
the Ollama slot under existing discipline. No new network machinery.

---

## 4. Protocol P3 — the L4→L3 `P(word | case role)` bridge account

[]{#protocol-p3-bridge}

The TODO is explicit that the map is not fittable from any shipped artifact and
that a word-form extension needs an explicit label-construction / split /
observation account **before** anything is built. This section builds that
account as it stood before implementation. The artifact, executor and influence
sweep subsequently landed; this section preserves their declared rationale and
must not be read as today’s implementation inventory.

### 4.1 The gap, stated from the shipped artifacts

- `vendor/ud_ewt_case_counts.json` (schema `ailm-case-counts-v2`): populations
  are `[UPOS, caseRole]` pairs with **no word form anywhere**; its
  `provenance.observation` records "UPOS tag only; deprel and feats[Case]
  withheld"; document-stride split over authenticated `# newdoc` ids with a
  recorded `document_identity_sha256`.
- `vendor/brown_clause_word_counts.json` (schema
  `ailm-clause-word-counts-v3`): carries words but types them by
  sentence-final punctuation — an L5→L3 route, not L4→L3.
- Raw word-form-bearing EWT dev is available locally and authenticated:
  `vendor/en_ewt-ud-dev.conllu` digest
  `39239e0a60db3ae68f4b7036189f11b6692741d10ff8240dd91f74f2760d90f8` (verified
  on disk this session against the recorded `source_sha256`), CC BY-SA 4.0,
  pinned revision `4a4d77f5…`. **The EWT test file's bytes are spent** —
  decision 111 marks `en_ewt-ud-test.conllu` `confirmatory_run_completed_ud_ewt_test`;
  no new result may be scored against those bytes, and this protocol reads the
  dev file only.

### 4.2 The account (what the TODO requires before any artifact)

1. **Label provenance, per token.** Every token row of the proposed artifact
   carries `(word_form, upos, case_label_or_null, label_source)` where
   `label_source ∈ CASE_ORACLE_PATHS` (`src/ailm/corpus/ud.py:76`):
   `morphology` (from `feats[Case]`), `deprel`
   (`src/ailm/cases.py::UD_DEPREL_TO_CASE`), or `adposition_token`
   (`src/ailm/cases.py::ADPOSITION_TO_CASE`, labeling the adposition token
   itself — no head-transfer rule, per
   [`../src/ailm/corpus/AGENTS.md`](../src/ailm/corpus/AGENTS.md)). Assignment
   reuses `src/ailm/corpus/ud.py::case_oracle_assignment` unchanged. Derived
   labels remain *agreement, not accuracy* wherever quoted.
2. **Split.** The same document-stride split construction as the existing case
   artifact (`test_every = 5` over authenticated newdoc ids, dev corpus only),
   with the new artifact's `split_indices` cross-validated against
   `vendor/ud_ewt_case_counts.json`'s recorded split so the populations are
   the same documents. The resulting artifact is therefore the **same
   development population in a new representation**: those dev documents were
   already scored as Stage 4 case-assignment populations, so any new result on
   them is development evidence under a fresh registration — never
   confirmatory, never "untouched". The shared corpus's *test file* remains
   untouched and spent (decision 111).
3. **Model-observation account.** The L4 model's observation is unchanged —
   UPOS only (`src/ailm/levels/case.py::FORBIDDEN_OBSERVATIONS`,
   `frozenset({"deprel", "feats", "case"})`). The oracle case labels are
   **sender-side inputs** for fitting only, recorded per token with
   `label_source`, and structurally unavailable to any model
   observation (the artifact's `observation` provenance string states this, and
   the loader refuses an artifact that drops it — the pattern the existing case
   artifact already enforces at `scripts/generate_case_counts.py`).
4. **The estimand and the executable seam (model-sender).** The bridge's
   message is built from `K(word | role)` — a role-conditioned word
   distribution fitted **only** from the declared training oracle-label routes
   (train rows, `label_source` recorded per token). At runtime, for a receiver
   state whose L4 observation is UPOS `u`, the message is the posterior
   marginalization
   `M(u) = Σ_role q_L4(role | u) · K(word | role)`,
   where `q_L4(role | u)` is the case level's own inferred role posterior
   given the observed UPOS. Its exact form is a **named constant fixed at P3
   partition-3 finalization, coordinated with the P3 builder**: a static
   per-UPOS Bayes inversion of the learned emission, or the sequential
   filtered posterior read from the live L4 filter — one is chosen and
   recorded in the registry entry; the account here does not depend on which.
   This runtime
   marginalization is legitimate and is explicitly distinct from *fitting* `K`
   on predicted roles (the rejected different estimand `P(word | predicted
   role)` — the design-C substitution; erratum locator as in §3.3). The
   message is consumed by an **explicit new L3 inlet/executor** — a new seam
   that reads the L4 posterior marginal; no shipped channel reads any case
   posterior today, and the registry entry below is registration-only
   structure, never narrated as an executed cascade. A **privileged-oracle
   arm** — the token's oracle role selecting `K(word | role)` directly at
   runtime — is a separately labeled diagnostic arm only, never the bridge
   estimand.
5. **Forbidden predictors, restated structurally.** No `deprel`/`feats[Case]`
   observation to any model level (guard above, plus the new artifact simply
   does not carry those fields as *observables*); the target word never
   conditions the role posterior (the case level's input space is unchanged);
   the message flows role→word, never word→role.
6. **Vocabulary intersection, mixing math, zero-support and no-message
   policies.** Sender corpus = UD EWT dev; receiver vocabulary = the Stage 3
   default vocabulary, fixed at two thousand types (principal's decision — not
   to be raised). Normalization rule (declared, not post-hoc): EWT word forms
   are lowercased and matched by **exact string** against the Stage 3
   vocabulary's lowercase type keys (from `vendor/brown_word_counts.json`); no
   other folding or stemming. The message's support is the intersection.
   Mixing math, stated exactly: `mix_prior`
   (`src/ailm/levels/clause_channel.py:529`) computes
   `(1−α)·prior + α·message` **without renormalization**. With zero message
   mass at out-of-intersection indices, every `α > 0` shrinks the prior at
   those indices by exactly `(1−α)` (e.g., prior `0.28` at α=`0.2` becomes
   `0.224`) while intersection words are boosted; the total stays one. This
   per-token effective-α heterogeneity is **part of the declared
   intervention** — the message cannot speak about words it has no data for —
   and policy (i), zero message mass outside the intersection, is adopted with
   that corrected reading (the earlier "receiver prior unchanged" claim was
   false and is withdrawn). The executor adds a **normalization assertion at
   the seam**: a message whose masses do not sum to one is refused
   (`mix_prior` today accepts one silently). **Zero-support policy:** roles
   with no training words are excluded from the marginalization and `q_L4`
   renormalized over the supported roles; if no role has support for an
   observed UPOS, the no-message case fires with the deterministic policy
   `D_α = D_0` (skip the insertion) — never an unnormalized distribution.
   Cross-corpus transfer (an EWT-estimated message applied to a Brown-receiver
   population) is part of the declared estimand, named as such, not hidden.
7. **Reuse, not re-derivation, of the shipped machinery.** The convex mixing
   with the bit-identical alpha=0 anchor, the sweep grid guard, the word-axis
   scrambled negative control, and the document-clustered paired summary are
   `src/ailm/levels/clause_channel.py::mix_prior`,
   `::sweep_clause_alpha`, `::scramble_word_distribution`, and
   `::clustered_clause_word_summaries` respectively — used as-is. Two new
   registry identities, both additive entries in
   `src/ailm/channel_registry.py`, neither a new model level nor a
   clause-channel edit: the **model-sender bridge entry** (registration-only
   structure; a model-level sender cannot claim `PRIVILEGED_ORACLE` —
   `channel_registry.py:130-131` refuses it) and a **separately labeled
   diagnostic entry** for the privileged-oracle arm (sender "case-oracle
   (train routes, dev)", receiver L3, insertion word prior `D`,
   `comprehension_only`). `K` pools the declared routes on train rows; the
   artifact reports **per-route coverage** (only NOM/ACC/GEN have a
   morphology route; the deprel and adposition routes are derived — their
   agreement is agreement, not accuracy) and per-route agreement with the
   pooled label, so the pooling is a declared, inspectable choice. A
   Brown-side sender alternative exists (EWT-derived `build_upos_lexicon` tag
   projection over Brown forms) but adds an uncontrolled tag-quality layer and
   changes the estimand to a lexicon-projected one; it is listed under
   decisions, not adopted.

### 4.3 The three-layer separation the TODO demands

- **Structural reachability** (registrable now): typed registry entries plus a
  test asserting the L6→L5→L4→L3→L1 path is type-correct end to end **at the
  registry level** — registration-only, explicitly not an executed cascade
  (the executor is partition 3 below); no performance claim.
- **Causal influence** (a separate, later registration): alpha sweep with the
  bit-identical zero-alpha anchor, the scrambled-word negative control, a
  **declared positive control** (a synthetic sender constructed to help must
  produce log-gain exceeding the scrambled control by a pre-registered margin
  derived from the frozen artifact's offline arithmetic — the
  `_phone_comparator_refusals` derivation pattern), and the whole-document
  clustered bootstrap — the existing clause-channel arm's
  recorded pattern (`clause_word_*` claims), with its scope warning honored:
  the recorded receiver sensitivities describe *their* interventions and bound
  nothing about a combined cascade.
- **Performance** (never claimed here): would need its own population,
  comparator, controls, and uncertainty analysis under a fresh registration —
  default `descriptive` status for anything computed on the development
  population, per `scripts/assign_claim_status.py`'s mechanical assignment.

### 4.4 Artifact schema sketch (proposed, not built)

```
schema "ailm-case-word-counts-v1" (proposed)
provenance {
  corpus "UD English EWT dev (en_ewt-ud-dev.conllu)", license "CC BY-SA 4.0",
  source_revision, source_sha256, source_size_bytes,
  observation "UPOS only for any model observation; oracle case labels are
               sender-side fitting inputs, recorded per token with
               label_source",
  split "document-index stride over authenticated # newdoc id values (dev)",
  test_every, document_identity_sha256
}
populations { train | test } rows: [word_form, upos, case_label_or_null,
                                    label_source]
sender_map  { role -> {word_form: count} }   (train rows only; per-route
             coverage and per-route agreement fields alongside)
vocab       mapping to the Stage 3 receiver vocabulary
digests     {populations, sender_map, vocabulary}
```

### 4.5 Implementation partitions and decisions

| # | Partition | Ownership boundary | Criterion |
|---|---|---|---|
| 1 | `scripts/generate_case_word_counts.py` + artifact + offline tests | new files only; `vendor/` provenance entry | Byte-stable regeneration; observation string enforced; per-route coverage/agreement fields present; no test bytes read |
| 2 | Registry identities + registry-level structural path test | additive entries in `channel_registry.py`; no level edits | Model-sender entry and separately labeled oracle-diagnostic entry both type-correct; comprehension-only |
| 3 | L3 inlet/executor + message builder + alpha sweep recorder (exploratory status by default) | new executor module; reuses `clause_channel.py` unchanged | Normalization assertion at the seam; zero-support and no-message policies enforced; anchor identity, scramble + declared positive controls live; clustered summary declared |

Resolved as proposed (rationale inline above): sender = model-sender `K`
fitted from train routes, runtime message = `q_L4` marginalization (oracle
sender = diagnostic arm only); OOV = zero message mass with the corrected
shrink semantics declared; no-message = deterministic `D_α = D_0`; vocabulary
normalization = lowercase exact-string match. Execution-ordering items, not
approval gates: the influence sweep registers only after partitions 1–2 land
and its estimand registration is written; the posterior-constant choice
(§4.2 item 4) is fixed with the P3 builder at partition-3 finalization.

---

## 5. Ownership, grant boundaries, and what this milestone did not do

At the original drafting milestone, the written scope was this file and its index row in
[`README.md`](README.md). Subsequent implementation and measurement were assigned separately. The
partition table records those boundaries; it is not a new approval requirement
for work already authorized in the active session. No Git writes, no stashes, no environment mutation, no
production recorder run, no training or download occurred. The dirty main work
at `01b868b` is untouched.

**Faithful vs unresolved, per original TODO item:**

| Item | Covered here (faithfully) | Still unresolved |
|---|---|---|
| Equal opportunity to fit | A complete, source-grounded fit protocol over the registered probe lineages with equalization statements, controls, separation, and partitions; the identity/data foundation is implemented and fixture-tested (E1R2) | The fit itself, the `FITTED_PROBES` registry population and consumers, any measurement; the gemma3 gated-license question |
| Aligned pronunciation | A pair-structured stipulated design with leakage/adversarial checks that never claim contamination-freedom | ~~The generator, artifact, recorder arm, any optional probe arm~~ — the generator and committed artifact ship, and the offline evaluator runs the three controls; offline and zero-shot probe arms have recorded evidence; new optional arms remain separately specified work |
| L4→L3 bridge | The full label/split/observation account built *before* an artifact, with the model-sender estimand (`K` from train routes, `q_L4` marginalization at runtime), corrected mixing math, and the dev population declared development evidence in a new representation | ~~The artifact, the registry entries, the L3 inlet/executor~~ — all ship; the registered development influence sweep is recorded under decision 121; the connected runtime and GUM comparison are separate evidence |

The three Principal-decision TODO items (inline rationale, declarations,
main-text split) are explicitly out of scope here and remain the author's.

## 6. Verification performed for this document (read-only)

- Local probe identity: `ollama list`, `ollama show` on all three registered
  tags (arch/params/context/capabilities/license quoted in §1). No inference,
  no generation, no downloads.
- HF metadata reads (public API/card pages only): SmolLM-1.7B (llama arch,
  Apache-2.0, ungated), Qwen2.5-3B-Instruct (qwen-research license, instruct,
  base-model lineage). The gemma3 HF repo was not read (gated); stated as such.
- Vendor artifact schemas and digests re-checked on disk; the EWT dev conllu
  digest matched its recorded `source_sha256`.
- All `claim:` tags used in this file were checked against
  [`../data/claims.json`](../data/claims.json) before writing; no tag here is
  an illustration, and no measured number is stated without either a verified
  tag, an artifact field in inline code, or an explicit estimate marker.
- Revision E1R (this file's second state): applied every supported finding
  P1-1..P1-7, P2-1..P2-4, and P3-1..P3-4 from the independent protocol audit
  (`output/reports/omp-dispatch-20260908/evidence/extensions-E1-independent-review.txt`),
  plus its citation nits; `mix_prior` (`clause_channel.py:529`) and the three
  `MODELS = dict(PROBE_MODEL_TAGS)` recorder sites re-verified at their cited
  lines during the repair.
- Revision E1R2 (this file's third state): P1-8 corrected without the
  review's faulty remedy — stride-d subsampling rejected as biased (the
  aliasing counterexample is implemented as a fixture in
  `src/ailm/benchmarks/fitted/census.py`), the streaming exact census kept
  with the supervised-target versus conditioning-exposure distinction made
  explicit; efficiency options declared with their objective/attention
  assumptions; a measured warm-up throughput + checkpoint/resume compute gate
  added before any full fit; the diagnostic seam declared (positional carve
  before shuffle, seam gap at least the window width, experimental comparator
  refit on exactly the fit positions); C1's reference named as the empirical
  comparator on the same slice, not a ceiling; approval-gate phrasing
  replaced by execution-ordering in §3.4, §4.5, and §5; the three-matrix G2P
  decode path and the P2 smoothing/unseen policy stated; the P3
  posterior-constant choice named for partition-3 finalization. The fitted
  data/identity foundation (`src/ailm/benchmarks/fitted/`) is implemented and
  fixture-tested in this tree.