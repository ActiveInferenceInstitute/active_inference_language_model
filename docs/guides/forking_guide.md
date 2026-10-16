# Forking Guide: How to Add a Level

The prospective procedure for building Stage N of the hierarchy. It is written
as a sequence because the order is load-bearing: **the gate is audited before
the level is built**, and every step after that assumes the previous one
produced a number rather than an intention. The historical Stage 1-4 build did
not consistently follow this order; [`decisions.md`](../decisions.md) records the
retroactive repairs.

The canonical widened-observation Stage 5 arm passes its development-session
gate and has no held-back replication; the retained case-only ablation remains
non-passing. Stage 6's document-clustered primary interval spans zero on its
development population and excludes zero on the separately registered held-back
population at the declared, but not family-corrected, coverage. Stage 7 remains a
speech boundary contract. The [governing hierarchy](../hierarchy.md) identifies
each scorer and the separate connected runtime; these outcomes are not evidence
of fluent or speech-ready language generation.

**License note for contributors.** A new level's code lands under `src/`,
which is **MIT**; its manuscript prose, figures, and any authored data (a new
`docs/design/stageN_*.md`, a new fixture) land under **CC BY-SA 4.0**. A
vendored corpus you add in Step 1 keeps whatever terms its upstream imposes —
it is not automatically MIT or CC BY-SA just because it lives in this
repository. See [`../LICENSE`](../../LICENSE) for the exact per-component split.

- [The order matters](#the-order-matters)
- [Step 0 — audit the gate BEFORE building](#step-0--audit-the-gate-before-building)
- [Step 1 — vendor a deterministic corpus artifact](#step-1--vendor-a-deterministic-corpus-artifact)
- [Step 2 — implement the level](#step-2--implement-the-level)
- [Step 3 — wire the oracle cross-check](#step-3--wire-the-oracle-cross-check)
- [Step 4 — controls](#step-4--controls)
- [Step 5 — record the claims](#step-5--record-the-claims)
- [Step 6 — figures](#step-6--figures)
- [Step 7 — manuscript and prose](#step-7--manuscript-and-prose)
- [Step 8 — close the stage](#step-8--close-the-stage)
- [Registration checklist](#registration-checklist)

---

## The order matters

Two orderings are available and only one is honest.

- **Build, measure, then set the threshold.** The threshold becomes a description
  of what was built. Every gate passes. Nothing is learned.
- **Audit, then build.** The threshold is committed before the result exists, so
  the gate can fail — and when it does, the failure is informative.

The second order is also what surfaces an *unbuildable* gate before you spend a
week failing it. Stage 1's target of claim:stage1_declared_gate_nats nats over uniform was unreachable by
claim:stage1_max_bigram_gain_nats nats' worth of ceiling; that was discoverable in
an afternoon, before any agent existed.

---

## Step 0 — audit the gate BEFORE building

Work through the pre-build checklist in
[`gate_design.md`](../gate_design.md#pre-build-checklist). The
irreducible core:

1. **Check the corpus carries the annotation the metric needs.** List the fields.
   Stage 2's declared gate scored a phoneme task on a syntactic treebank with no
   phonemic field — a copy-paste from Stage 4's case gate that survived because
   nobody looked at the columns.
2. **Measure the trivial baseline** — the strongest model that lacks the mechanism
   this level claims. If it already clears the declared threshold, the gate is
   **VACUOUS**; restate it against the baseline.
3. **Measure and type the strongest comparator.** A conditional-entropy or
   analytic bound licenses `UNREACHABLE` when it cannot clear the threshold. A
   strictly more powerful but exceedable decode licenses only
   `ABOVE_REFERENCE` in the same geometry. Record the comparator type before
   interpreting the verdict.
4. **Estimate the expected margin.** If it will be thin, choose the paired
   statistic and its no-effect control now, not after a disappointing inequality.

Then make the audit re-runnable. Add an `audit_stageN()` to
[`../scripts/audit_gates.py`](../../scripts/audit_gates.py) returning the same dict
shape the existing ones return:

Use `scripts/audit_gates.py::audit_stage3` as the executable schema reference,
not as a template for its measured values. It compares word paths against the
letter-bigram bar on identical emitted positions and names the held-out word
oracle as `ComparatorType.PRIVILEGED_ORACLE`. Supply the new gate’s registered
`GateId`, comparator type, direction, threshold, and `sesoi_margin`, alongside
the baseline/model/comparator fields consumed by `gate_audit_row`. The historical
PTB target in that function is a retained failed specification, not a new gate.

Add it to the `audits` list in `main()` and **delete the stage's row from
`UNAUDITED_STAGES`**. Removing that row is what makes the audit's own coverage
honest; leaving it produces a report that lists the stage as unchecked while
checking it.

```bash
.venv/bin/python scripts/audit_gates.py
```

Do not proceed while the verdict is anything other than `BRACKETED_BOUND` or
`BRACKETED_REFERENCE`. The other complete enum members are `VACUOUS`,
`UNREACHABLE`, `ABOVE_REFERENCE`, and `UNAUDITED`.

---

## Step 1 — vendor a deterministic corpus artifact

Tests must not touch the network. Follow the precedent set by
`vendor/brown_counts.json` and `vendor/cmudict_counts.json`:

- A generator script under `scripts/` with a `_schema` tag in the payload, so a
  stale artifact fails loudly rather than misindexing every matrix built from it.
- **Sorted keys, no RNG, no timestamps** — the artifact must be byte-identical on
  regeneration.
- Choose the **independent sampling unit before the assignment rule**. If
  observations share a document, lexical family, speaker, or other dependence
  unit, keep that whole unit on one side. Only then apply a deterministic stable
  hash or a unit-level stratification, and record the unit definition, assignment
  rule, and resulting disjointness checks.
- Use a **positional** split only when an explicit leakage analysis establishes
  that ordering cannot encode the target and that related observations cannot
  straddle the split. “No seed required” proves reproducibility, not independence.
- A guard that raises rather than vendoring out-of-inventory symbols. The phoneme
  generator refuses any phone outside `CANONICAL_ARPABET`.
- The sha256 documented in [`provenance/sources.md`](../provenance/sources.md) **and
  enforced** by [`../tests/test_provenance_checksums.py`](../../tests/test_provenance_checksums.py).
  A checksum in prose that is never compared is worse than none.
- Every **derivation choice that biases the data**, written down. The CMUdict
  entry records primary-pronunciation selection, stress stripping,
  length-matching at claim:stage2_g2p_length_matched_pct% of the dictionary, the
  shipped positional interleave and its measured family leakage, and the separate
  family-disjoint robustness arm. Each choice narrows what the level's numbers are
  a claim about.

```bash
.venv/bin/python scripts/generate_word_counts.py
.venv/bin/python scripts/generate_word_counts.py   # run twice; diff must be empty
.venv/bin/python -m pytest tests/test_provenance_checksums.py -q
```

---

## Step 2 — implement the level

Create `src/ailm/levels/<name>.py` following
[`../src/ailm/levels/phoneme.py`](../../src/ailm/levels/phoneme.py), which is the
better model of the two because its `A` is not the identity.

**Wiring, non-negotiable:** sequence dynamics go in the transition
`B[0, next, current, 0]`; the emission goes in `A[0, obs, state]`. The inverse is
anti-informative — observing a symbol yields a posterior over the symbol that
*preceded* it — and pymdp validates it anyway, because a bigram matrix is
column-stochastic in either slot. That defect measured
claim:stage1_anti_informative_perplexity against claim:stage1_letter_perplexity
for correct wiring, and on a small in-sample corpus scored worse than uniform
while still slipping under a naive threshold.

The module docstring is not documentation-after-the-fact; it is where the level's
reasoning lives. Include, in this order:

1. The factors, modalities, and exact matrix shapes.
2. Where the corpus statistics live, and the **measured** cost of the wrong slot.
3. The comparator and why it is not uniform.
4. **Which numbers are not interchangeable** — smoothed versus unsmoothed floors,
   all-positions versus transitions-only, subset versus full corpus. A gate
   comparing the wrong pair can never pass, and this section is what prevents
   someone writing it.
5. The numerical policy: float width and the tolerance it implies.

Boundaries: no `infrastructure` import, no hardcoded measured numbers, complete
type annotations, `from __future__ import annotations`. See
[`style_guide.md`](../method/style_guide.md#layer-boundaries).

---

## Step 3 — wire the oracle cross-check

Extend [`../src/ailm/verification/oracle.py`](../../src/ailm/verification/oracle.py)
rather than writing a scratch script. The existing surface —
`forward_filter`, `forward_backward`, `viterbi`, `sequence_perplexity`,
`markov_chain_perplexity` — already covers single-factor discrete levels; a level
with new structure needs a new closed form there.

Three constraints:

1. **It must not import pymdp or jax.** Independence is the whole value; an
   auditor that shares code with its subject inherits the flaw it audits for.
2. **Its conventions must match the pinned ones** in the module docstring, since a
   transpose in the oracle silently invalidates every cross-check built on it.
3. **Test the oracle against closed forms, not against pymdp.** Anchors that are
   derivable on paper — a uniform transition gives exactly the inventory size, a
   deterministic chain gives exactly 1 — plus agreement with a literal
   step-by-step walk on a hand-built corpus. Without the anchors the two
   implementations can be self-consistently wrong; without the walk a closed form
   can be self-consistently wrong alone.

Then add `test_stageN_pymdp_matches_oracle`. Compare with a justified numerical
tolerance and match the installed likelihood-floor and soft-observation
conventions. Check complete distributions when the claim concerns predictive
loss; equality of argmax labels alone cannot establish loss agreement.

**If eager inference is too slow, preserve the registered population.** The
connected runtime caches compilation of the actual pymdp calls and checks eager
versus compiled event predictions, posteriors, resets and policy actions. A
NumPy substitution is a separate method choice: the historical G2P recorder
uses bounded decision agreement before computing full-set statistics with its
oracle, and refuses recording when that agreement fails. A bounded check does
not prove global equivalence on every possible input or posterior value.
See
[`gate_design.md`](../gate_design.md#when-the-full-set-is-too-slow-oracle-equivalence)
and `verify_decoder_equivalence` in
[`../scripts/record_stage2_claims.py`](../../scripts/record_stage2_claims.py).

---

### Connecting a level to the live runtime

A builder and a GNN declaration do not establish a working connection. Extend
[the composition contract](../design/connected_runtime.md) with the sender,
receiver insertion point, observation availability, event clock and reset rule.
Expose pre-observation predictions before computing loss. Test missing evidence,
disabled and shuffled messages, document resets, and leakage from future tokens.
Keep the independent conditional oracle and the actual pymdp call path; a
registry-only route is not an executed cascade.

## Step 4 — controls

Create `tests/test_stageN_gate.py`. It needs all of the following, and the
negative controls must be *observed* to fail before the gate is declared live.

**Positive control** — the gate itself, against the context-free baseline.

**Negative controls, one per mis-wiring the level makes available.** Enumerate
them explicitly: which matrices are learned, and which could be transposed,
time-reversed, mis-normalized, or replaced by uniform? Level 2's set was:

| Mis-wiring | Caught by "beats the baseline"? |
|---|---|
| time-reversed `B` | yes |
| uniform, information-free `A` | yes |
| `A` normalized over the wrong axis, then renormalized so pymdp validates it | **no** |

**Therefore: at least one guard that is not a baseline comparison.**

- A structural invariant with the **axis named** — a transposed matrix still sums
  to 1 along *some* axis, so an unnamed axis is a green check on broken code. Plus
  an explicit index assertion that a lopsided count lands in
  `B[0, next, current, 0]` and not its transpose.
- Numerical agreement with the oracle (Step 3).

**Harness wiring**, so vacuity is an exception rather than a judgement call:

```python
from ailm.control import ControlHarness

harness = ControlHarness("stage3_word_gate")
harness.assert_gate_is_live(good_value=model, bad_value=mis_wired, direction="lower_is_better")
harness.verify_negative_control_fails(
    metric_fn=perplexity_of, broken_input=mis_wired_agent,
    threshold=baseline, direction="lower_is_better",
)
```

**If the margin is thin**, add the paired statistic *and* its no-effect control —
the same procedure run on two identical predictors must come back
non-significant. Without that control the significance test is itself unfalsified.

**If the gate sweeps or enumerates a set**, assert the set is non-empty *and*
complete against an independent source. Both failure modes read exactly like a
clean pass; that is how four of nine figures once escaped the layout sweep.

**Pin the historical findings** so they cannot silently return, the way
`test_stage2_two_nat_target_is_provably_unreachable` does.

---

## Step 5 — record the claims

Create `scripts/record_stageN_claims.py`. Every number that will appear in prose
is derived here; nothing is hand-typed anywhere else.

```python
ledger.record(
    "stage3_word_perplexity",
    measured,
    run_id="stage3_word_gate",
    source="scripts/record_stage3_claims.py",
)
```

- `run_id` and `source` are mandatory — an entry without them cannot be
  re-derived, and `test_every_ledger_entry_has_provenance` fails on it.
- Record the **eval-set size** as its own claim. A margin without an *n* is not a
  result.
- Record the **ceiling**, even when the architecture deliberately does not claim
  it, so the unclaimed headroom is visible rather than hidden.
- Record the **subsample** figure alongside the full-set one when the two differ
  materially. That contrast is the subsample-bias evidence, and it is why
  claim:stage2_phoneme_subset_perplexity sits in the ledger next to
  claim:stage2_phoneme_perplexity.
- Refuse to record at all if an equivalence or sanity precondition fails.

Then add the labels the manuscript needs to `REQUIRED_CLAIMS` in
[`../src/ailm/required_claims.py`](../../src/ailm/required_claims.py) —
including any the derived-quantity block reads, or the failure surfaces later as
a bare `KeyError` instead of a named missing label.

```bash
.venv/bin/python scripts/record_stage3_claims.py
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
```

---

## Step 6 — figures

Add generators under [`../src/ailm/figures/`](../../src/ailm/figures), one module
per topic, returning `matplotlib.figure.Figure` objects and knowing nothing about
the pipeline.

- **No hardcoded measured numbers.** Read from
  `output/data/manuscript_variables.json` via the `_common` helpers; a missing
  token raises `MissingVariableError` rather than drawing a placeholder.
- **Deterministic**: fixed seeds, sorted iteration, no timestamp in the image, so
  regenerating twice is byte-identical.
- **Legible by construction**: `save_figure` refuses type below
  `ailm.viz_config.MIN_FONT_PT`. There is no clamp — give the figure room or
  shorten the text.
- Name the builder with a recognized prefix (`generate_`, `build_`, `plot_`,
  `make_`, `figure_`) and take **no required arguments**, or it silently escapes
  the layout sweep in
  [`../tests/test_figure_layout.py`](../../tests/test_figure_layout.py).

```bash
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --list
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_figure_layout.py -q
```

Register the new label in
[`syntax_guide.md`](../syntax_guide.md#figure-label-registry).

---

## Step 7 — manuscript and prose

- **Manuscript**: reference numbers as `{{TOKEN}}` (the ledger label uppercased).
  A missing claim fails the render, which is the intended behavior — an
  unresolved token in a rendered document reads as a fabricated result.
- **`docs/`, `README.md`, `AGENTS.md`, `CHANGELOG.md`, docstrings**: reference
  numbers as `claim:<label>`. Authored tags remain literal, and
  `test_no_unbacked_prose_claims` scans them. The generated `output/docs/` mirror
  hydrates recorded values; edit the authored source, never that mirror.
- **Say what the numbers are a claim about.** Name the subset and what it costs;
  distinguish agreement from accuracy; say which guard actually established the
  result, since beating the baseline is necessary and not sufficient. Full rules
  in [`style_guide.md`](../method/style_guide.md#honesty-rules).
- **If a gate was restated, write it as a correction** — original wording, why it
  was wrong, what replaced it — in [`../CHANGELOG.md`](../../CHANGELOG.md) and in
  every document that carried the old claim. Both existing restatements are
  written this way, in
  [`../AGENTS.md`](../../AGENTS.md) and
  [`design/architecture.md`](../design/architecture.md), and a correction that lives
  only in the changelog leaves the wrong statement in the document a reader
  opens.
- **Add a `docs/design/stageN_*.md`** for any decision that costs something — an
  alignment restriction, a filtering-versus-smoothing choice, a heuristic oracle.
  [`design/stage2_g2p.md`](../design/stage2_g2p.md) is the template: the decision,
  the alternative, the cost stated plainly, and what it forbids downstream.

---

## Step 8 — close the stage

Flip the status only when the gate passes with its controls live:

- The stage row in [`../AGENTS.md`](../../AGENTS.md)'s gate table and
  [`../README.md`](../../README.md)'s roadmap.
- The `UNAUDITED_STAGES` row in
  [`../scripts/audit_gates.py`](../../scripts/audit_gates.py) (deleted in Step 0).
- The current-stage line in [`../AGENTS.md`](../../AGENTS.md).
- A `[Unreleased] — Stage N` section in [`../CHANGELOG.md`](../../CHANGELOG.md) with
  Added / Changed / Fixed, and the gate result table using `claim:` tags.
- The **honest scope statement** in [`../README.md`](../../README.md): what this
  stage does *not* yet do.

Then run the full verification checklist in
[`agent_instructions.md`](agent_instructions.md#rule-5-verification-checklist).

---

## Registration checklist

A new level touches more places than the level file. Missing one of these leaves
the tree internally inconsistent in a way no single test catches.

- [ ] `src/ailm/levels/<name>.py` with the five-section module docstring
- [ ] `vendor/<artifact>.json` + generator + `_schema` tag, byte-identical on regeneration
- [ ] sha256 in [`provenance/sources.md`](../provenance/sources.md), enforced by `tests/test_provenance_checksums.py`
- [ ] derivation choices and their biases documented in the same entry
- [ ] `src/ailm/verification/oracle.py` extended if the level has new structure
- [ ] `tests/test_<name>.py` — module API surface, closed-form numeric assertions
- [ ] `tests/test_stageN_gate.py` — positive control, one negative control per mis-wiring, a non-comparison guard, harness wiring, paired statistic if the margin is thin
- [ ] `scripts/record_stageN_claims.py` — every prose number, with `run_id` and `source`
- [ ] eval-set size, ceiling, and any materially different subsample figure recorded as claims
- [ ] `REQUIRED_CLAIMS` extended in `src/ailm/required_claims.py`
- [ ] `audit_stageN()` added to `scripts/audit_gates.py`, `UNAUDITED_STAGES` row removed
- [ ] figures added, registered in [`syntax_guide.md`](../syntax_guide.md), passing the layout sweep
- [ ] `docs/design/stageN_*.md` for any decision with a cost
- [ ] gate tables updated in `AGENTS.md` and `README.md`
- [ ] `CHANGELOG.md` section, with any restatement written as a correction
- [ ] honest scope statement updated in `README.md`
- [ ] full verification checklist green

---

## Cross-references

- [`gate_design.md`](../gate_design.md) — Step 0 in full, plus the per-stage hazards for 3–7
- [`testing_philosophy.md`](../testing_philosophy.md) — Step 4 in full
- [`syntax_guide.md`](../syntax_guide.md) — Steps 5–7 mechanics
- [`style_guide.md`](../method/style_guide.md) — prose and layer boundaries
- [`agent_instructions.md`](agent_instructions.md) — invariants, commands, verification checklist
- [`design/architecture.md`](../design/architecture.md) — where each level sits and what its matrices are
