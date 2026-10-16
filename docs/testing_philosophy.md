# Testing Philosophy

What a passing test suite in this project is allowed to mean, and the specific
ways it has failed to mean anything.

The short version: green is not evidence. A test proves something only if it
could have failed for the reason you care about, and this project has shipped
tests that could not — a claims scanner that only ever ran on synthetic input, a
legibility check that measured font size while labels sat on top of each other,
a wiring bug that pymdp validated in either orientation. Each section below is
one of those, generalized.

- [No mocks](#no-mocks)
- [Every gate needs a positive AND a negative control](#every-gate-needs-a-positive-and-a-negative-control)
- [GateVacuousError](#gatevacuouserror)
- [A gate that runs only on synthetic input constrains nothing](#a-gate-that-runs-only-on-synthetic-input-constrains-nothing)
- [The oracle cross-check pattern](#the-oracle-cross-check-pattern)
- [Beating the baseline is necessary, not sufficient](#beating-the-baseline-is-necessary-not-sufficient)
- [Structural invariants must name the axis](#structural-invariants-must-name-the-axis)
- [Legibility is a build-time invariant](#legibility-is-a-build-time-invariant)
- [Determinism and float width](#determinism-and-float-width)
- [Running the suite](#running-the-suite)
- [Checklist before adding a test](#checklist-before-adding-a-test)

---

## No mocks

`MagicMock`, `unittest.mock`, `mocker.patch`, and any other framework that
substitutes a fake for a real dependency are prohibited. This is a hard invariant
(see [`../AGENTS.md`](../AGENTS.md)), not a preference.

The reason is specific to what AILM claims. The project's entire thesis is that
the generative engine is **real pymdp** — a mocked `Agent` would let the suite
stay green while the claim became false, which is precisely the failure this
project is built to avoid. The same argument applies to corpora: a stubbed
CMUdict would let a phoneme inventory drift away from
claim:phoneme_count_39 without any test noticing.

What replaces mocks:

| Need | Pattern used here |
|---|---|
| Corpus data without network access | Vendored deterministic artifacts (`vendor/brown_counts.json`, `vendor/cmudict_counts.json`), regeneration verified byte-identical, checksums enforced by [`../tests/test_provenance_checksums.py`](../tests/test_provenance_checksums.py) |
| A small, inspectable corpus | Hand-built count dicts (`_counts_from_sequences` in [`../tests/test_phoneme_full_corpus.py`](../tests/test_phoneme_full_corpus.py)) whose exact positions are known on paper |
| Checking pymdp's arithmetic | An independent numpy oracle, never a stubbed return value |
| Files and directories | Real `tmp_path` files |
| Figures | Real matplotlib renders, inspected in display coordinates |

A checksum documented in prose and never compared against the artifact is worse
than no checksum, because it licenses the belief that the input was pinned. That
is why the provenance hashes in [`provenance/sources.md`](provenance/sources.md)
are enforced by a test rather than trusted.

---

## Every gate needs a positive AND a negative control

A gate asserts that a metric on a good model lands on the right side of a
threshold. That assertion is meaningless unless the **same metric on a broken
model** lands on the wrong side.

- **Positive control** — the gate passes with the correct model. (The gate
  itself.)
- **Negative control** — the same measurement, run on a deliberately broken
  input or a deliberately mis-wired agent, must **fail**.

[`../src/ailm/control.py`](../src/ailm/control.py) makes this explicit:

```python
from ailm.control import ControlHarness

harness = ControlHarness("stage2_g2p_gate")

# 1. The metric can tell good from bad at all.
harness.assert_gate_is_live(
    good_value=model_accuracy,
    bad_value=mis_wired_accuracy,
    direction="higher_is_better",
)

# 2. The broken input is actually caught by the threshold in use.
harness.verify_negative_control_fails(
    metric_fn=accuracy_of,
    broken_input=mis_wired_agent,
    threshold=baseline_accuracy,
    direction="higher_is_better",
)
```

The two calls check different things and neither substitutes for the other.
`assert_gate_is_live` checks that the *metric* discriminates; a metric that
returns the same value for both is vacuous regardless of threshold.
`verify_negative_control_fails` checks that the *threshold* discriminates; a
sensitive metric with a badly placed threshold still accepts a broken model. That
second failure is exactly Stage 2's original copy of the
claim:stage4_declared_gate% Stage 4 threshold.

Live negative controls in the suite, as a template for new levels:

| Test | Broken thing | Must fail because |
|---|---|---|
| `test_stage1_negative_control_uniform_model` | uniform transition | no learned dynamics |
| `test_stage1_unigram_gate_rejects_the_anti_informative_wiring` | bigram in `A`, identity `B` | backward-looking prediction |
| `test_stage2_phoneme_negative_control_uniform_transition` | uniform `B` | no sequence structure |
| `test_stage2_g2p_negative_control_uniform_emission` | uniform `A` | information-free emission |
| `test_stage2_g2p_negative_control_anti_informative_transition` | time-reversed `B` | predicts the *previous* phone |
| `test_stage2_mcnemar_can_return_no_effect` | two identical predictors | a significance test that always says "significant" proves nothing |
| `test_stage2_two_nat_target_is_provably_unreachable` | the original Stage 1-style threshold | pins the UNREACHABLE finding so it cannot silently return |

`test_stage2_mcnemar_can_return_no_effect` deserves emphasis. The negative
control for a *statistical test* is not a broken model — it is a case with no
real effect, which must come back non-significant. Without it, the significance
machinery itself is unfalsified.

---

## GateVacuousError

`GateVacuousError` ([`../src/ailm/control.py`](../src/ailm/control.py)) is raised
when the negative control **passes** — the broken model was not detected. It is
deliberately an exception and not a warning, because the correct response is to
block the stage, not to note a concern.

It fires in two situations:

1. `check_metric_is_sensitive(good, bad, ...)` where `good == bad`. The metric
   cannot distinguish the two models at all.
2. `verify_negative_control_fails(...)` where the broken input's metric is *not*
   worse than the threshold. The gate would accept a broken model.

The error message names the gate, both values, the threshold, and the direction,
so the diagnosis does not require reading source.

**When it fires, fix the metric or the threshold — never the gate wording.**
Weakening the assertion to make `GateVacuousError` go away converts a detected
defect into an undetected one. `Stage N` is blocked until the control fails as
it should.

---

## A gate that runs only on synthetic input constrains nothing

This is the sharpest lesson in the project's history, and it cost a full stage.

`ClaimsLedger.scan_prose` shipped in Stage 0 with tests. Every one of those tests
built a `tmp_path` ledger, wrote a synthetic markdown file, and asserted the
scanner behaved. All green. Meanwhile `claim:phoneme_count_39` sat **unbacked in
the real README.md, AGENTS.md, and `docs/provenance/sources.md`** for an entire
stage: the Stage 0 tests recorded that label into throwaway ledgers only, so the
real ledger never had it.

The scanner worked. The discipline it was supposed to enforce did not exist,
because nothing ever pointed the scanner at the shipped tree.

The fix is [`../tests/test_claims_discipline.py`](../tests/test_claims_discipline.py),
and its structure is the pattern to copy:

| Test | Role |
|---|---|
| `test_prose_file_set_is_non_empty` | the scanned set must be non-trivially sized — an empty glob would pass every check below |
| `test_prose_set_includes_the_top_level_docs` | the specific reader-facing files must be *in* scope, not just "some files" |
| `test_scanner_detects_an_injected_unbacked_claim` | **positive control** — a planted fake tag must be flagged, so the real assertion is load-bearing |
| `test_scanner_accepts_a_backed_claim` | **negative control** — a backed tag must not be flagged, or the gate would be unusable on correct prose |
| `test_no_unbacked_prose_claims` | the actual gate, over `README.md`, `AGENTS.md`, `CHANGELOG.md`, `docs/**/*.md`, and `src/**/*.py` |
| `test_ledger_is_not_empty` | an empty ledger would make the scan trivially pass |
| `test_every_ledger_entry_has_provenance` | a recorded value with no `run_id`/`source` cannot be re-derived, so it is not evidence |
| `test_phoneme_count_claim_is_backed_and_correct` | binds the ledger value to `len(CANONICAL_ARPABET)` rather than to the literal, so drifting code fails here instead of quietly disagreeing with prose |

Non-vacuity was verified the only way it can be: by injecting an unbacked claim
into the real README and confirming the gate went red.

**Generalize it.** Any gate that scans, sweeps, or enumerates has two failure
modes beyond "the check is wrong": the scanned set can be empty, and it can be
incomplete. Both read exactly like a clean pass. So every sweep needs:

1. an assertion that the set is non-empty,
2. an assertion that the set is **complete** — bound to something independent,
   not to the sweep's own discovery,
3. a positive control proving the check can fire.

`tests/` is deliberately excluded from the prose scan: test files carry
illustrative tags — a `claim:` prefix on a fixture label such as `alpha_claim` —
that are test data, not assertions about the project. (For the same reason, a
document explaining the protocol must not spell out an example tag in full: the
scanner cannot tell an illustration from an assertion, so an invented label in
prose turns the gate red. Write the prefix and the label separately, as here.)

---

## The oracle cross-check pattern

pymdp validates normalization. It does not validate meaning. A bigram matrix is
column-stochastic whether it sits in the emission `A` or the transition `B`, so
`_validate()` accepts the anti-informative wiring without complaint — the defect
that measured claim:stage1_anti_informative_perplexity against
claim:stage1_letter_perplexity for correct wiring.

What caught it was a hand-rolled numpy Markov chain that disagreed with pymdp's
perplexity materially. That instrument was a scratch script.
It is now [`../src/ailm/verification/oracle.py`](../src/ailm/verification/oracle.py):
`forward_filter`, `forward_backward`, `viterbi`, `sequence_perplexity`,
`markov_chain_perplexity` — a tested module so every level can be checked the
same way instead of the instrument being rebuilt ad hoc.

Three rules govern it:

1. **The oracle must never import pymdp or jax.** Its entire value is being an
   independent implementation. An auditor that shares code with the thing it
   audits inherits the flaw it audits for. This is enforceable by grep and worth
   grepping for.
2. **Its conventions are pinned in the module docstring**, because a transpose in
   the oracle silently invalidates every cross-check that uses it. `emission[i,
   j] = P(obs=i | state=j)`, `transition[i, j] = P(next=i | cur=j)`, both
   column-stochastic; `start` is the distribution over `state_0` *before* any
   observation, so the predictive distribution at `t = 0` is `start` itself.
3. **The oracle is itself tested against closed forms**, not against pymdp.
   [`../tests/test_oracle.py`](../tests/test_oracle.py) anchors it on analytically
   derivable cases: a uniform transition must give exactly the inventory size, a
   deterministic chain exactly 1. Otherwise the two implementations could be
   self-consistently wrong together.

The cross-check appears at both levels — `test_stage1_matches_closed_form_bigram_chain`,
`test_stage2_phoneme_pymdp_matches_oracle`, `test_stage2_g2p_pymdp_matches_oracle` —
and additionally as a bounded *equivalence check* supporting the stated
performance substitution. See
[`gate_design.md`](gate_design.md#when-the-full-set-is-too-slow-oracle-equivalence):
require claim:stage2_decoder_disagreements position disagreements on the declared
bounded subset, then use the fast oracle for full-set statistics, and **refuse to record**
if that count is ever nonzero.

A failed equivalence check must block the substitution. Passing bounded label
agreement is not a proof of equality on all inputs or of predictive-loss equality.
The connected runtime instead compiles the actual live Agent calls and checks
complete eager/compiled predictions and posteriors, including reset and policy
controls; its full GUM measurement stays on that live-agent path.

---

## Conditional correctness and connected performance

The independent `_exact_marginals` oracle in `tests/test_connected.py` checks each
receiver update conditional on the supplied messages. The checked convention
includes pymdp’s expected-log soft evidence and likelihood floor. A uniform
pseudo-observation is not interchangeable with absent evidence; disabled
bottom-up channels must preserve the prior before temporal advance.

The conditional and compiled-execution controls in `tests/test_connected.py`
exercise real agents, adversarial messages, pre-observation predictions and
reset behavior. They establish implementation properties, not improved language
performance. The separately registered GUM recorder measures the fixed connected
configuration against all named channels disabled, retaining the shared raw
lexical UPOS encoder and explicit OOV scope. Its complete execution inventory is
authenticated by `ailm.verification.connected_provenance`; accepting a truncated
but rehashed inventory would invalidate that boundary.

## Beating the baseline is necessary, not sufficient

The Level 2 lesson, and what it means for test design specifically: `assert
model > baseline` is a necessary condition, not a sufficient one, because a
mis-wired variant can still clear the bar. See
[`gate_design.md#beating-the-baseline-is-necessary-not-sufficient`](gate_design.md#beating-the-baseline-is-necessary-not-sufficient)
for the full derivation — the three mis-wirings, which ones "beats the
baseline" catches, and the rule that follows: every level needs at least one
guard that is not a comparison against a baseline.

---

## Structural invariants must name the axis

A transposed matrix still sums to 1 along *some* axis. So
`assert np.allclose(M.sum(axis=0), 1)` can be a green check on a matrix that is
wired backwards.

Two assertions are needed, and both are in
[`../tests/test_stage2_gate.py`](../tests/test_stage2_gate.py):

1. **An explicit index assertion.** A deliberately lopsided count must land in
   `B[0, next, current, 0]` and *not* its transpose. Build a count dict with one
   dominant bigram and assert the position of the mass, not just its
   normalization.
2. **Column-stochasticity along the specific axis** — pymdp's `axis=1`, which
   indexes the produced variable. Naming the axis is the whole assertion.

`test_stage2_g2p_emission_orientation_is_pinned` also asserts that `A[0, :, p]`
sums to 1 *before* any renormalization, which is what catches the mis-normalized
variant above.

Related: `test_stage2_gate_constants_are_derived_not_typed` and
`test_stage1_gate_constants_are_consistent` guard the gate itself against a
hardcoded threshold that has drifted from the measurement it was supposed to
describe.

---

## Legibility is a build-time invariant

Figures are tested, for the same reason prose numbers are: a figure is a claim.

`ailm.figures._common.assert_legible` enforces a minimum font size
(`ailm.viz_config.MIN_FONT_PT`) and refuses to write a figure that violates it.
There is **no clamp** — an overflowing label fails the build via
`IllegibleFigureError` or `FigureOverflowError` rather than shrinking to
illegibility. A clamp is not a floor.

Font size alone is a partially vacuous legibility gate, though: a 10 pt label is
not legible if another 10 pt label sits on top of it.
[`../tests/test_figure_layout.py`](../tests/test_figure_layout.py) closes that
hole by measuring rendered **display coordinates** — what a reader sees rather
than what the code intended — and checking for overlapping and clipped text.

It also demonstrates the sweep-completeness rule from above, having been caught
by it:

- `test_at_least_one_figure_builder_was_discovered` — the sweep is non-empty.
- `test_discovery_covers_every_shipped_figure` — the discovered builder count is
  bound to the PNGs actually on disk. The name-prefix heuristic originally
  omitted `generate_` and silently excluded four of nine figures, **including two
  with confirmed defects**, while reading as a clean sweep. Non-emptiness passed;
  completeness is what catches this.
- `TestChecksCanFail` — positive controls for each detector (stacked labels must
  be flagged, a label wider than the canvas must be flagged) and negative
  controls (a clean figure must not be flagged).

Figures must also be deterministic: fixed seeds, sorted iteration, no timestamp
in the image, a pinned PNG `Software` field, so regenerating twice is
byte-identical. And no generator hardcodes a measured number — values come from
the vendored artifacts or from `output/data/manuscript_variables.json`, which is
generated from the claims ledger. A figure with a typed-in number is the same
defect as prose with a typed-in number, with a longer feedback loop.

---

## Determinism and float width

- **Vendored artifacts are byte-identical on regeneration** (keys sorted, no RNG,
  no timestamps), which is what allows the tests to run with no network and
  allows a checksum to be enforced rather than recorded.
- **JAX defaults to float32.** Exact analytic anchors land within roughly 1e-7
  relative: the uniform-transition case measures 38.999998, not 39. Compare with
  a tolerance, never for equality. `FLOAT32_REL = 1e-6` in
  [`../tests/test_phoneme_full_corpus.py`](../tests/test_phoneme_full_corpus.py)
  is the documented bound, with the reasoning attached: asserting tighter than
  the arithmetic can deliver produces a test that fails on correct code.
- **Splits are positional, not seeded.** The G2P train/test split is by sorted
  index (every *n*th pair to test), so it is reproducible from the artifact with
  no seed to record — and `test_stage2_g2p_split_is_disjoint_and_positional`
  pins that.
- **Reproducible is not representative.** A deterministic prefix over
  alphabetically sorted data is perfectly reproducible and badly biased. See
  [`gate_design.md`](gate_design.md#defect-3--reproducible-is-not-representative).

---

## Running the suite

The canonical interpreter is the project's own `.venv`:

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# Required ignored interface artifacts before the full suite
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py
.venv/bin/python -m pytest tests/ -q

# With the coverage gate (fail_under in pyproject.toml, claim:stage0_coverage_floor_pct%; pytest-cov applies it whenever --cov is passed)
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing

# One gate
.venv/bin/python -m pytest tests/test_stage2_gate.py -v

# One test
.venv/bin/python -m pytest tests/test_claims_discipline.py::test_no_unbacked_prose_claims -v

# Figures need the headless backend
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_figure_layout.py -q
```

Never use bare `uv run pytest` in this sidecar: it can resolve to a system
pytest with the wrong Python and the wrong pymdp. See
[`troubleshooting.md`](troubleshooting.md#bare-uv-run-pytest-resolves-to-the-wrong-python-and-a-different-pymdp).

**An exit code alone is not evidence of executed tests.** Confirm a nonzero
executed-test count and the coverage result when requested. Pytest normally
returns a distinct status for no tests collected, while collection-only mode
can succeed without executing any test. The release runner checks execution
counts and rejects skips or expected failures in its publication pytest steps.

The per-test default is `timeout = 300` in `pyproject.toml`; bounded integration
and replay tests declare their own limits. Diagnose import, compilation and
execution costs before changing a limit. A full-corpus recorder belongs in its
registered measurement workflow, not inside a short unit test.

The one deliberate exception is `test_rerunning_the_recorders_is_a_no_op` in
`tests/test_claims_single_writer.py`. It re-runs the `record_*_claims.py` scripts
as subprocesses against an isolated ledger (re-measuring Stage 2, the complexity
reconciliation, and a real `lake build`), so it takes longer than a minute even on
a quiet machine, and the parent blocks in `subprocess.run` so the in-process
signal timer may not provide a useful child diagnostic. The test’s own timeout
and the `recorder_replay` marker are declared beside it; consult that source
rather than copying a runtime budget into this guide. It is an **acceptance replay**, not a unit test:
proving the stored ledger already equals what the recorders produce is exactly
the acceptance this repository exists to keep, so it stays in the default suite.
When the suite appears to stall at its start, it is this test re-measuring — poll
the log rather than assuming it hung.

The default is not a universal runtime promise. A test that
spawns an interpreter (a script run through `subprocess`, a `python -O` probe, a
disposable-venv wheel smoke) or that runs a bounded live-pymdp walk carries an
explicit `@pytest.mark.timeout(...)` or a module-level `pytestmark`, sized for a
cold jax import under load, and its inner `subprocess.run(..., timeout=...)` stays
strictly below that mark so the child is the one that reports a hang, with its
output attached, rather than pytest-timeout killing the parent blind. Changing a timeout is not a numerical validation. Preserve the registered
population and method; use the connected compiled path or a separately justified
oracle substitution where the protocol supports it.

**Network-producing recorders are excluded from hermetic recorder replay.**
`discover_network_recorders` in `ailm.replay_provenance` derives that boundary
from actual local and hosted call reachability. The current producers are the
external, aligned-G2P, synthetic, in-context and logprob-drift recorders. The
classification control checks hosted producers as well as local ones and retains
offline counterexamples, so adding an import cannot silently exclude everything.
Use `scripts/orchestrate_claim_recorders.py network --list` to inspect the live
phase inventory without making calls.

These recorders remain under the single-writer and semantic receipt contracts.
Canonical synthetic and in-context scored calls require fresh hosted replies;
local audit and drift checks also have live-call requirements. Offline replay
validates stored evidence where allowed; it does not repeat those observations.
An interrupted canonical fresh-call run cannot be completed by relabeling cached
cells. Preserve its scope and follow [the release procedure](method/release_checklist.md)
for a new authorized run.

---

## Checklist before adding a test

- [ ] No mocking framework anywhere in the file.
- [ ] Every numeric assertion is closed-form, re-derived in-test, or read from
      the claims ledger — not a literal copied from a previous run's output.
- [ ] Float comparisons use a tolerance with a documented reason.
- [ ] If it is a gate: a negative control exists and has been observed to fail.
- [ ] If it is a gate: at least one guard is not a baseline comparison.
- [ ] If it sweeps a set: the set is asserted non-empty *and* complete against an
      independent source.
- [ ] If it checks a detector: a positive control proves the detector can fire.
- [ ] If it asserts a matrix invariant: the axis is named.
- [ ] Corpus input comes from a vendored artifact, not the network.
- [ ] The test was observed to fail before the fix, or was made to fail by
      injection afterwards. An assertion never seen red is an assumption.

---

## Cross-references

- [`gate_design.md`](gate_design.md) — the VACUOUS / UNREACHABLE /
  ABOVE_REFERENCE / UNAUDITED / BRACKETED_BOUND / BRACKETED_REFERENCE taxonomy
  and the pre-build checklist
- [`syntax_guide.md`](syntax_guide.md) — the claims ledger and `{{TOKEN}}` protocol
- [`troubleshooting.md`](troubleshooting.md) — interpreter, timeout, and figure failure modes
- [`design/stage2_g2p.md`](design/stage2_g2p.md) — the guards that pin Level 2's orientation
- [`../src/ailm/control.py`](../src/ailm/control.py) — `ControlHarness`, `GateVacuousError`
- [`../src/ailm/verification/oracle.py`](../src/ailm/verification/oracle.py) — the independent instrument

The `quick` release profile is the portable checkout check. Tests marked
`publication_context` require retained private history/raw-source proof, and
`template_context` requires the configured renderer or staged auxiliaries.
Their assertions remain mandatory in `full`; the quick exclusion is not a
publication-acceptance result. The authenticated connected-GUM report is shipped
as a retained input so portable report consumers need no original private history.
