# TODO — the open queue

Only open work lives here. Completed work is in `CHANGELOG.md`; every scientific
outcome, including every negative and null result, is in `docs/decisions.md` and
stays there permanently. An item that is finished is deleted from this file
rather than ticked, because a queue of ticked boxes stops being a queue.

Sizing below is by *blast radius*, not by effort: **minor** touches one file and
no recorded number, **medium** changes a measurement or a guarantee, **major**
changes what the paper claims or what the artifact is.

## Minor

- [ ] At the next render, change the abstract's closing sentence from "The
  planned release repository is" to "The release repository is": the public
  `v1.0.0` tag exists. The edit is deferred rather than applied because
  `tests/test_manuscript_crossrefs.py` requires the tracked rendered
  provenance to match the authored sources, so a manuscript wording change
  ships only with a template Stage 03/04 render.
- [ ] Complete the whole-suite cold-start verification under fleet load. Bounded
  release fixtures, lazy imports, shared figure construction, and scoped
  full-model timeout allowances now have focused passing checks. The complete
  release run must still verify their integration. Record import and filesystem
  stalls separately from executed-test cost; a timeout remains a failed check
  and must not be converted into a skip to obtain a passing receipt.

## Medium

- [ ] **Narrow the source bindings to what each measurement executes.** Three
  bindings currently freeze bytes that the bound measurement never runs, so an
  unrelated edit forces a re-measurement or a reviewed derivative: the
  network-claim binding receipts hash the whole `src/ailm` tree, so any source
  change requires fresh hosted calls; the retained connected-GUM report's
  execution inventory binds seven modules plus every non-URL field of
  `pyproject.toml`, so a lint-rule change there needs a scoring replay; and the
  publication derivative pins two reviewed test fixtures byte-for-byte. The
  proposal: bind each recorder to its executed import closure (resolved by
  import, not by directory), bind the connected report to the modules its
  replay actually loads plus the dependency and build fields of `pyproject.toml`
  rather than the whole file, and let the derivative pin fixtures by their
  parsed test inventory rather than their bytes. Each is an evidence-rule change
  that needs its own decision entry and a re-bind on the workstation, and the
  existing receipts stay valid until then.

## Major

- [ ] **Equal opportunity to fit the task.** The supplement's fairness table
  records this condition as holding nowhere. Protocol P1
  (`docs/extension_protocols.md` §2) is written and its offline data/identity
  foundation is implemented and fixture-tested: the streaming exact-census
  objective, the diagnostic carve behind a declared seam, write-once receipts,
  the separate `FITTED_PROBES` registry, and the stride-aliasing refutation
  (`src/ailm/benchmarks/fitted/`, `scripts/generate_fitted_probe_data.py`).
  The warm-up compute gate is measured (decision 122: 3.003 it/s at batch
  size 1, 6.006 tok/s over loss-bearing tokens, peak 7.271 GB), and its
  verdict is that the census budget — 5.82M instances × 2 epochs ≈ 45 days
  of wall clock — does not clear at that rate: committing the census fit
  requires the protocol's block-masked-packing unit. What remains is the
  fit pipeline itself — the LoRA fit with its C1–C5 controls behind that
  gate, fuse/convert/serve, the fitted recorder and its claims — all
  requiring the external model infrastructure the protocol assigns to its
  own implementation grant
  (upstream checkpoint downloads, the mlx-lm/llama.cpp toolchain, local
  serving), plus the gemma3 gated-license question.

  The historical proposals in `docs/research_queue.md` do not supply this
  fitted-probe design. That queue is exploratory history, and its truncated
  adversarial export does not establish current task status.

- [ ] **Extend the stipulated-artifact design to aligned pronunciation.**
  Protocol P2 design B is implemented: the stipulated pair-structured
  artifact ships with digest-bound provenance
  (`vendor/synthetic_aligned_g2p_counts.json`,
  `scripts/generate_synthetic_aligned_g2p.py`), the native comparator is the
  shipped G2P engine refit on the stipulated train pairs, and the offline
  evaluator runs the English-transfer, broken-correspondence, and
  uniform-floor controls (`scripts/evaluate_synthetic_aligned_g2p.py`).
  The comparator arm is recorded offline (`bench_synthg2p_comparator_*`:
  the native arm at 99.72% position accuracy and 98.33% exact match, the
  A1 English-transfer refusal gate at 3.89%, the A2 broken-correspondence
  refusal gate at 0.0%, the uniform-floor control at 2.56%), and the
  zero-shot probe arms are implemented, gated, and recorded
  (`scripts/record_aligned_g2p_probe_claims.py`; three registered tags, the
  analytic contract floor 1 − 1/39; run `synthetic_aligned_g2p_probe_v1`:
  gemma3 and qwen2.5 scored 60 test words each at near-zero accuracy with
  full contract compliance and no exact matches, smollm recorded as a
  contract refusal; both scoring arms sit below the analytic contract floor,
  which closes the comparison by refusal rather than ranking). The artifact
  is never called contamination-free.

- [ ] **The end-to-end top-down cascade: the L4 → L3 account is built and
  the influence sweep is recorded; any performance claim is not.**
  Protocol P3's account ships: the label-construction/split/observation
  artifact (`data/case_word/`, `src/ailm/corpus/case_word.py`), the
  registry entries,
  and the L3 inlet/executor (`src/ailm/case_word_channel.py`), with the
  model-sender estimand fixed — `K` fitted from train routes, `q_L4`
  marginalized at runtime, corrected convex mixing, OOV and no-message
  policies declared. The influence sweep is recorded per decisions entry
  121 (`case_word_sweep_*`: 3001 scored tokens over the artifact's 64 test
  documents, best α = 0.3 at a mean per-token log-gain of +0.1715 nats,
  against the bit-identical zero-alpha anchor with the scrambled-word
  negative control and the pre-registered positive control live). What
  remains, execution-sequenced: any performance claim still needs its own
  specified population, comparator, controls, and uncertainty analysis.
  *Cost large, value uncertain, and the uncertainty is the point.*

## Principal decisions (these block "done"; not the agent's to make)

- [ ] How much rationale stays inline in code.
## Rejected, with reasons, so they are not rebuilt

Kept permanently. A rejection with its reasoning is cheaper to read than the
rebuild it prevents, and this section has already stopped one proposal from
returning in a new costume.

- **The morphemic level.** The unanalysable-word tax is computable before anything
  is written and is roughly 93% of the entire distance from the shipped word level
  to the best closed-vocabulary word model — paid on the head of the distribution
  while every hoped-for gain arrives on the tail. Its dominance criterion is
  unsatisfiable at two of seven declared grid points, so its falsifier fires
  regardless of whether morphology works.
- **Supplying AILM's fitted column to the probes.** This is the case-oracle trap in
  a new costume, with the property that makes that trap the sharpest one this
  project has sprung: it produces an impressive number rather than a degraded one.
  The probe would be scored against the very system whose prediction is line 1 of
  its prompt, so its upper bound is AILM, and the proposed leak guard *certifies*
  the circularity rather than catching it.
- **The GNN composition ontology.** There is no live pymdp object for a channel, so
  conformance would have nothing to read the new roles off. Build the executor, not
  the ontology it cannot check.
- **Raising the Stage 3 default vocabulary.** Principal's decision: the default
  stays at 2,000 types. The scaling ladder already publishes every rung through a
  closed form that is exact at any vocabulary size, so no science is bought by
  raising it and a full re-record would be spent.
- **Taking a Mathlib dependency for the Lean layer.** Principal's decision: the
  offline dependency-free build is kept, and the `Real`-requiring items are
  *declined* rather than deferred. See `formal/README.md`.
- **Eleven items from `docs/research_queue.md`**, refuted under adversarial review
  and listed there with the reason each fails.

## Levels 3 to 6 and the hierarchy — see docs/research_queue.md

That file preserves historical proposals and inline completion notes. Its
adversarial export is retired because the rows and supporting evidence were
truncated; it does not establish current completion counts or a ranked work
queue. Follow its opening corrections, the governing hierarchy, and the decision
record before turning a historical proposal into new work.
