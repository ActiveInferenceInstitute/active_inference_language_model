# Handoff — Active Inference Language Model

> **ARCHIVED HISTORICAL SNAPSHOT (2026-08-21).** This file predates the completed
> external-audit response. It is retained for provenance, not as a current
> handoff or verification receipt. Start with `README.md`, `AGENTS.md`, the
> current `docs/decisions.md`, and `output/reports/release-receipt.json`; the
> `CHANGELOG.md` entry "the external audit, answered in kind (2026-08-20)"
> explains the transition from this snapshot.

At the snapshot date, the protocol v4 census had landed and
`handoff/claims.post-v4-census.json` was the recorded ledger. Its diff against
`handoff/claims.pre-v4-census-restart.json` caught the sentinel leaks described
in the CHANGELOG. Neither that lock state nor that ledger copy is current-state
evidence.

Do not infer that recorded results, generated outputs, locks, or tokens are
current from this file. Re-run the current acceptance surfaces below.

## Read this before touching anything

Model calls are memoized per call, content-addressed on payload, endpoint,
protocol version and the model's content **digest** (decision 40). Two
consequences matter in practice. A long run is safe to halt at any point, and
rerunning the identical command replays what is on disk. At the snapshot date,
the sentinel correction replayed the store rather than re-measuring it; consult
the cache receipt for current counts and timings.

- `--verify-cache N` re-measures a sample against the live server and is what
  keeps "reproducible" a statement about the model rather than about the disk.
  Inspect the current recorder and receipt for its precondition count.
- Use the marker exclusions documented by the current release profile when
  running a bounded local suite. The acceptance run owns the recorder-replay
  policy and timeout configuration; this snapshot does not. Re-run an isolated
  timed-out file before diagnosing code, and never raise the clock merely to
  make a broad run green.
- The pre-run ledgers are backed up under `handoff/`. Diff against them before
  believing any recorded value moved for the reason you assume.

## What this project actually is

A language model whose parameters are interpretable conditional probabilities
in explicit `pymdp` generative models, designed level by level and evaluated with
tests that can fail. Recorded outcomes include a bounded null, and the historical
build did not consistently follow the prospective stage order.

The capability cost is accepted openly. **The contribution is as much the
verification apparatus as the models.** If you optimize for better numbers at the
expense of that apparatus, you have not advanced the project — you have replaced
it with a worse one that happens to score higher.

## Historical state summary — not a verification receipt

| | |
|---|---|
| Levels built | L1–L6 have built model objects and recorded outcomes. Stage 5's gate is non-passing and its document-clustered paired-sentence privileged-label gain remains unresolved; Stage 6 has a positive point estimate whose document-clustered primary interval spans zero and is non-confirmatory. Stage 7 is a tested boundary contract, not a model gate. |
| Generation | built; policy selection with a real action set and top-down `C` |
| Audited gates | `scripts/audit_gates.py` is authoritative. Stage 5 is expected `ABOVE_REFERENCE`: both the mixture and fitted privileged reference remain on the failing side, both whole-document-bootstrap gain intervals span zero, and the planted-effect control stays live; a point pass, comparator crossing, or resolved interval is a changed finding. |
| Spec → model loop | closed: `ailm.spec_agent` builds every level from its GNN spec, **bit-identical** to the builders (decision 20) |
| Complexity | every level's cost is an exact, reconciled operation/byte count; manuscript section `09a` (decision 21) |
| Tests | No count is frozen here. Run the current full profile and inspect `output/reports/release-receipt.json`, including skips and coverage. |
| Manuscript | The tracked combined PDF is regenerable output; hydrate, render, inspect the PDF visually, and check extracted cross-references/tokens before publication. No page count or Downloads copy in this snapshot is authoritative. |
| Ledger | Run the claim-tag, manuscript-token, status, and single-writer gates against the current tree; this snapshot is not evidence that they are green. |
| Unaudited | Stage 7 speech only — re-scoped: the declared gate measures Whisper/TTS; the boundary interface contract is tested (`tests/test_speech_boundary.py`) |
| Gate classifier | one source of truth: `src/ailm/gates.py` (`GateVerdict`, `classify_gate`), imported by BOTH `scripts/audit_gates.py` and `src/ailm/figures/gates.py` — resolves the `handoff/reports/NEVILLE_PROXY_REVIEW.md` L5 two-copies drift |
| Statistics | `src/ailm/statistics.py::SIGNIFICANCE_FAMILY` is the source of truth: claim:stat_family_size members, claim:stat_family_survivor_count fixed-Bonferroni survivors, and claim:stat_holm_survivor_count Holm survivors. Named, machine-checked exclusions carry no threshold-crossing inference. |
| Formal layer | `formal/` uses a pinned, dependency-free Lean toolchain. Current counts are claim-backed: claim:formal_theorem_count named theorems and claim:formal_example_count witness examples, with claim:formal_theory_theorem_count theory theorems, claim:formal_instance_theorem_count ledger-instance theorems, and claim:formal_axioms_beyond_core axiom beyond Lean's core set. `formal/AILM/LedgerData.lean` is generated from `data/claims.json` and held by a byte-identity test; see `docs/formal.md` and re-run the build before relying on it. |
| Release verification | `scripts/verify_release.py` (quick/full profiles) emits `output/reports/release-receipt.json`; CI at `.github/workflows/ci.yml` |

Environment note: the suite is JAX-version-sensitive. Use the project's pinned
lock and `.venv`; `pyproject.toml` and `uv.lock`, not this historical snapshot,
own the current version. See `docs/troubleshooting.md` § "JAX version changes
float results or breaks pymdp" before changing the backend.

Run `scripts/audit_gates.py` and the full release profile before relying on any
current-state assertion. The historical statements above are provenance, not a
guarantee.

---

## The invariants. Break these and the project stops meaning anything

These are not style preferences. Each was learned by shipping the defect.

1. **A gate is BRACKETED only when the trivial baseline FAILS it and the achievable
   ceiling CLEARS it.** Measure both before accepting any threshold. This project
   has shipped a gate nothing could pass, a gate anything could pass, and a gate
   that was *both at once* depending on which population it was scored against.
2. **Beating the baseline is necessary, never sufficient.** Mis-wirings at both
   the phoneme and word levels cleared their bars while being wrong, including a
   word prior whose frequency ranking was inverted. Every gate needs a guard
   that is not a comparison — an invariant, an oracle agreement, or an
   enumerated mis-wiring that must fail.
3. **Recorders write claims; tests verify them.** One writer per label,
   default-deny enforced in `tests/test_claims_single_writer.py`. Multiple labels
   were once dual-written and the values *happened to agree*, which is exactly
   why it survived a green suite and a rendered PDF.
4. **Before adopting a metric, build the configuration that should fail it.** If
   you cannot construct a failing case, it is a description, not a gate.
   Generation's first metric was completely saturated, including on a generator that
   emitted one word forever.
5. **The model must never observe the fields its labels are derived from.** This
   is the only defect class here that makes a model look *excellent*, so there is
   no numeric signature and the guard has to be structural.
6. **A null needs the same controls as a positive**, and so does a
   pre-registered prediction. Identity anchor, non-vacuity check,
   entropy-preserving scramble.
7. **Verify the artifact, not the exit code.** A render has exited 0 while
   producing no PDF at all. Extract the text and count `??` and `{{TOKEN}}`.
8. **Never delete a claim tag or a sentence to make a gate green.** Re-record
   from the run that produced the value, or change the finding on purpose.

Corollary you will need on day one: when a test fails because a *finding*
changed, update the finding and its decision entry. When it fails because the
code broke, fix the code. Telling these apart is most of the work.

---

## Your three workstreams

Fully scoped in [`TODO.md`](TODO.md); the reasoning is here.

### W1 — Close the GNN → pymdp loop *(DONE, 2026-07-30)*

Built as scoped: `ailm.spec_agent.build_agent_from_spec` constructs every
shipped level from its spec alone behind the default-deny `LEARNED_REGISTRY`;
every registered level is bit-identical to its builder, proven **before** any use
of the spec path; mutations fail construction or construct provably different
models. The one divergence the exercise caught — the L1g spec's `E`
misdescribing the constructed agent — is decision 20. Builder retirement was
**re-scoped, not done**: the builders are the parameterized surface the
negative controls inject through, so retiring them would remove a guard. What
remains of W1 sits in `TODO.md`.

A fourth workstream landed the same day: the **complexity section**
(`docs/manuscript/98_supplement_complexity.md`, decision 21) — every level's cost as an exact
operation/byte count, reconciled against instrumented execution, never a
timing.

### W2 — Parameter learning inside pymdp *(historical scope; later outcomes recorded)*

The equivalence increment landed under pre-registration — the prediction was
committed (`c760357`, decision 22) before any update ran, then verified on the
development stream
(decision 23): under the letter level's identity emission, pymdp's
`infer_parameters` reproduces the shipped add-k counting exactly, the
mismatched-prior and lr≠1 controls prove the comparison can fail, and the
learned-vs-counted perplexity gate became the identity it had to be. Learned
matrices write only `learning_*` labels; the predicted two-writers trap never
got a chance.

The separate online-learning experiment was subsequently pre-registered and
measured (decision 28): on the planted Brown genre shift, the continuing learner
beats its frozen control on the development stream. That scoped result does not
make learned matrices the shipped parameterization or establish an
untouched-population confirmation. `learn_A` at G2P is a different experiment:
its states are unobserved, so conjugate agreement is not the expected outcome.

### W3 — More visualisations, with captions that carry the argument

At the snapshot date, the figure set had falsification-stating captions, and
`tests/test_caption_standard.py` binds the rule absolutely: the legacy allowlist
reached empty and was removed rather than left as a dormant exception mechanism.
Its non-vacuity check is now exact coverage — every `fig:` anchor in the
manuscript must have a caption the scanner parsed, and the two sets must be
equal — rather than the old "at least fifteen", which any subset satisfied. A
companion gate requires every referenced PNG to exist, because the caption gate
reads Markdown and the layout gate reads PNGs, so a figure in one and not the
other was checked by neither.

The scaling ladder's remaining work moved to `TODO.md`: the sweep is built and
tested, and only the ledger write, the figure and one prose correction are left.

Two mechanical notes that survive completion. The layout gate checks type size
and clipping but **not overlap** — look at the rendered PNGs (a dark-on-dark
annotation bug shipped past the gate and was caught only by eye). And an
artist drawn outside the axis limits still expands `bbox_inches="tight"`: one
bar at 255 on a linear axis produced a 37,000-pixel canvas.

---

## The research question, and where it stands

*What makes these models hierarchical, intelligent and relevant?*

There is a **working criterion with a pre-registered development-population
result** ([`docs/decisions.md`](docs/decisions.md) entries 18–19). It is not an
untouched-population confirmation:

> Working hypothesis, bounded by the independently measured channels: an inter-level
> message improves the receiver when it carries information the
> receiver does not already have. Its optimal weight is **interior** when the
> message is novel and **zero** when it is redundant — so the *shape* of a weight
> sweep diagnoses novelty independently of the sign of the effect.

The measured seams disagree in sign. L1→L2B harms, L3→L1g generation helps,
and L3→L2B helps by `claim:lexical_best_margin_pts` points at
p = `claim:lexical_best_mcnemar_p` with the predicted interior optimum. That
lexical prediction was recorded before measurement on the development
population. L6→L5 changes the receiver's `D` prior but its primary recognition
analysis is non-confirmatory; L5→L3 is a near-null whose predicted harm did not
reproduce. These are independent interventions, not a connected cascade.

The later pre-registrations have also been measured. Entry 27's clause → word
channel did not reproduce the predicted harm, qualifying the first clause of the
criterion. Entry 28's online-vs-frozen experiment met its registered clauses on
the development stream: the continuing learner beats its frozen control after
the planted genre shift. That verifies the scoped mechanism; under the current
claim-strength vocabulary it is not ordinary confirmation on untouched data.
Each outcome and its controls remain recorded in its decision entry.

**Keep pre-registering.** Write the prediction, and what would withdraw the
principle, into `docs/decisions.md` *before* the measurement runs. A principle
only ever fitted to results already in hand explains nothing, and post-hoc
rationalisation is indistinguishable from insight in the write-up.

---

## How to work here

```bash
cd projects/working/active_inference_language_model
.venv/bin/python scripts/orchestrate_claim_recorders.py core
.venv/bin/python scripts/orchestrate_claim_recorders.py network
.venv/bin/python scripts/orchestrate_claim_recorders.py aggregate

# the Lean layer: regenerate the ledger bridge FIRST, then re-record from a real build
.venv/bin/python scripts/generate_lean_ledger.py
(cd formal && lake build && ./Scripts/axiom_report.sh)
.venv/bin/python -m pytest tests/test_formal_isomorphism.py::test_the_negative_control_does_not_compile -v --no-cov
.venv/bin/python scripts/orchestrate_claim_recorders.py formal
.venv/bin/python scripts/assign_claim_status.py
.venv/bin/python scripts/audit_gates.py --json

# every ledger consumer follows every writer
.venv/bin/python scripts/z_generate_manuscript_variables.py
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py
.venv/bin/python -m pytest tests/ -q                       # authoritative count in the release receipt
.venv/bin/python -I -S scripts/verify_release.py full --keep-going # canonical acceptance receipt

# render from the TEMPLATE repo root, not from here:
cd /path/to/template
UV_NO_CACHE=1 uv run --locked python scripts/pipeline/stage_03_render.py \
    --project working/active_inference_language_model
```

Then **verify the PDF, not the exit code**: `pdftotext`, count `??` and
`{{TOKEN}}`, confirm the figures are present.

Commits go to this sidecar repository. The one exception on record is the
citation-validator fix in the template repo (`5bf4fa69a`), which is committed but
**not pushed** — that remote is public and pushing is the principal's call.

## Open decisions that are the principal's, not yours

- Which Stage 3 vocabulary ships as the default. All three configurations work;
  raising it changes every Stage 3 number.
- Whether to push the template-repo fix.
- How much rationale stays inline in the code.

Ask rather than assume. Each changes published numbers or a public repository.
