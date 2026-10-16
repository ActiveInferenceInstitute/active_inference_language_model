# AGENTS.md — data/

## Ownership boundary

`data/claims.json` is **owned by the recorders**, not by this directory and not by any
agent's editor. The write path is exactly:

```
src/ailm/levels/  +  vendor/*.json
        │
        ▼
scripts/record_*_claims.py   ──►  ledger.record(...)  ──►  data/claims.json
```

The stage recorders are `scripts/record_stage0_claims.py` through
`scripts/record_stage6_claims.py`; the remaining `record_*_claims.py` recorders cover
the channel, generation, learning, benchmark, statistical, and formal estimands.
`scripts/orchestrate_claim_recorders.py` runs the set in dependency order and
[`../scripts/README.md`](../scripts/README.md) states that order.

There is no test-side exception any more. Gate tests verify the ledger against
their own re-measurement and never write it
(`tests/test_stage2_gate.py::test_stage2_claims_recorded` is verify-only, and no
entry in the committed ledger carries a `tests/` source). An ordinary test that
wrote the project ledger would make the stored value a race between writers.

Nothing else writes `claims.json`. The directory does hold other evidence records —
`figure_alt_text.yaml`, `learning_agreement.npz`, `literature_search.json`,
`manuscript_literals.yaml`, `reference_sources.json`, the `case_word/` and
`confirmatory/` artifact directories, and
`registration_provenance.json` — and each has exactly one writer or validator, listed
in the table in [`README.md`](README.md). Adding a file here means deciding what its
provenance discipline is and naming the code that owns it.

## Must not change without an explicit decision

**Never hand-edit `claims.json`.** This is the hardest rule in the repository, and it
has no exceptions — not for a rounding, not to add a label some document already
mentions, not to make a red gate green. Editing it manufactures a `value` with a
`run_id` that never happened, which converts the ledger from evidence into
transcription. The whole apparatus exists because hand-typed numbers have drifted from
measured ones in this codebase before.

If prose and ledger disagree, **the prose is wrong**. Re-run the recorder and fix the
sentence.

**Do not delete a claim to silence a scanner failure.** `test_no_unbacked_prose_claims`
reports a label found in prose with no entry. The fix is to record it. Deleting the
prose tag instead removes the evidence requirement and leaves the assertion standing —
which is precisely the failure mode this project spent a stage in.

**Do not change `_schema` (`ailm-claims-v2`) or the entry shape** — three required
fields (`value`, `run_id`, `source`) plus the optional epistemic `status`
(`confirmatory` / `exploratory` / `descriptive` / `analytic`, see
`scripts/assign_claim_status.py` for the assignment rule and README.md here for
what each means).
`_load()` backfills a missing skeleton key but raises on unparseable JSON or a
non-object top level, on purpose: the file may hold data this code cannot interpret,
and silently replacing it would discard recorded claims. `_save()` writes
`json.dumps(..., indent=2)` — keep that formatting so diffs stay reviewable.

**Do not add a `value` that is not JSON-serializable**, and do not store a numpy scalar.
Cast to `float` / `int` in the recorder.

**Do not add a second mapping from label to manuscript token.** The token is the label
uppercased. A lookup table is a second thing to drift.

**Do not add a claim to `REQUIRED_CLAIMS`** in `src/ailm/required_claims.py`
without recording it first — a render will then fail hard, which is the intended
direction of failure but blocks the manuscript until the measurement exists.

## Local gotchas

**Derived tokens are not claims.** `manuscript_variables.py` emits
`STAGE1_UNIGRAM_MARGIN`, `STAGE1_CEILING_SHORTFALL_RATIO`,
`STAGE2_MCNEMAR_TOTAL_DISCORDANT`, `STAGE2_PHONEME_UNIGRAM_MARGIN`, the `META_*`
provenance block, and a `<LABEL>_RUN` / `<LABEL>_SOURCE` pair for every claim. All are
valid `{{TOKEN}}`s. **None** has a ledger entry, so a `claim:` tag naming one of them
in a `.md` file fails the discipline gate. Check
`claims["..."]` before writing a `claim:` tag; check
`output/data/manuscript_variables.json` before writing a `{{TOKEN}}`.

**`output/data/manuscript_variables.json` can be staler than the ledger.** It is
generated and disposable; it is tracked only as a stable publication surface (see
`docs/reference/output_conventions.md`), never authored. If a figure or a manuscript number
disagrees with `claims.json`, regenerate the token map — do not reconcile by editing
either file.

**`load_ledger(base_dir)` appends `data/claims.json` itself.** Pass the project root,
not the `data/` directory.

**`ClaimsLedger.record` saves on every call.** A recorder that raises partway through
leaves the ledger partially updated. Recorders are therefore idempotent — re-running
one is the correct recovery, not restoring from git.

**A rounding change is a claim change.** Values are stored at the precision the
recorder rounded to (`11.9215`, `70.801`, `3.468e-06`). Changing a `round()` in a
recorder alters every downstream `{{TOKEN}}` and every figure axis label. Re-run
`z_generate_manuscript_variables.py` and the figure generators in the same change.

**Multiple recorders and multiple concurrent sessions write this one file.** If two
agents run recorders at once the last writer wins for overlapping labels. Re-run the
full recorder sequence rather than assuming the file is consistent.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# Structure, provenance, prose backing
.venv/bin/python -m pytest \
  tests/test_claims_discipline.py tests/test_claims_ledger.py -q --no-cov

# Every entry carries run_id and source, and the count is non-zero
.venv/bin/python -c "
import json
c = json.load(open('data/claims.json'))['claims']
bad = [k for k, v in c.items() if not v.get('run_id') or not v.get('source')]
assert c, 'ledger is empty'
assert not bad, f'missing provenance: {bad}'
print(len(c), 'claims, all with provenance')
"

# The ledger must survive a round trip through the recorders unchanged
cp data/claims.json /tmp/ailm_claims_before.json
.venv/bin/python scripts/record_stage0_claims.py
.venv/bin/python scripts/record_stage1_claims.py
.venv/bin/python scripts/record_stage2_claims.py
diff /tmp/ailm_claims_before.json data/claims.json && echo "recorders are idempotent"
```

A non-empty diff from that last check is meaningful: either a measurement moved (which
must be explained and propagated to every `{{TOKEN}}` and figure) or the ledger had
been hand-edited.
