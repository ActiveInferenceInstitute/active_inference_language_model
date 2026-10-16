# AGENTS.md — tests/

## Ownership boundary

`tests/` owns **whether a claim is allowed to stand**. It does not own the claim's
value: the owning recorders write `data/claims.json`; tests verify the canonical
ledger and write test ledgers only under `tmp_path`. A test must never record a
measurement into the canonical ledger.

Tests do not own the implementation. If a test needs a helper that computes something
non-trivially, the helper belongs in `src/ailm/` — with one deliberate exception:
**independent re-derivations stay in the test file**. `_independent_emission` in
`test_stage2_gate.py` rebuilds `A[l, p] = P(letter l | phone p)` in plain numpy on
purpose. Moving it into `src/` would make it share code with what it audits, and it
would then inherit the flaw it audits for.

## May change freely

- Add tests. Add a case to an existing class. Tighten a tolerance.
- Add a fixture to `conftest.py`, offline and deterministic.
- Add a new gate file for a new stage — with a positive control, a negative control,
  and a corresponding `audit_*` in `scripts/audit_gates.py`.

## Must not change without an explicit decision

**No mocks. Ever.** No `MagicMock`, `unittest.mock`, `mocker.patch`, and no
`monkeypatch.setattr` replacing a dependency. Use real pymdp, the vendored corpus
JSON, `tmp_path`, inline CoNLL-U strings, and the `tiny_cmudict` fixture. In
`test_pymdp_smoke.py` this is not stylistic: a mock has every attribute, so
`hasattr(agent, "update_empirical_prior")` would pass against a pymdp that lacks it,
and the *absence* check `not hasattr(agent, "update_A")` could not be demonstrated at
all.

The one permitted use of `monkeypatch` is **environment isolation**, not dependency
replacement: `test_manuscript_variables.py` sets `SOURCE_DATE_EPOCH` to prove the build
stamp is reproducible. Setting an environment variable does not substitute the code
under test; patching a module attribute does.

**Do not delete a negative or positive control.** The suite is full of assertions
whose only job is to prove another assertion is load-bearing. Named ones:

| Control | Without it |
| --- | --- |
| `test_cmudict.py::TestNegativeControl` (3 tests) | the 39-phone invariant is unfalsifiable; a 54-phone inventory still builds valid matrices |
| `test_stage1_gate.py` anti-informative control | the declared `< 28.0` threshold admits the mis-wiring at claim:stage1_anti_informative_perplexity, so uniform alone does not detect it |
| `test_figures_architecture.py::TestAssertLegible::test_tiny_font_figure_is_rejected` | every legibility assertion in the suite becomes vacuously true, since `save_figure` calls `assert_legible` on every figure |
| `test_figure_layout.py` overlap/clip positive controls | a geometric assertion that never fires licenses the belief layout was verified |
| `test_figure_layout.py::test_discovery_covers_every_shipped_figure` | the prefix heuristic under-collects and a partial sweep reads exactly like a full one — this happened, omitting four of nine figures |
| `test_oracle.py` transposition control | an identity transition is column-stochastic either way, so an identity-only test cannot detect a transpose |
| `test_claims_discipline.py::test_scanner_detects_an_injected_unbacked_claim` | `assert unbacked == []` also passes when the file glob returns nothing |
| `test_provenance_checksums.py::test_provenance_document_records_at_least_one_checksum` | the drift checks pass for free when no hash is documented |
| `test_control_harness.py` | if the harness cannot detect a broken model, every gate that uses it is vacuous |

**Do not narrow `test_claims_discipline.py`'s scan set.** It globs root `README.md`,
`AGENTS.md`, `CHANGELOG.md`, `docs/**/*.md`, and `src/**/*.py`. `_prose_files()`
returning an empty or shrunken list is the exact failure this module exists to
prevent — `scan_prose` was tested only against `tmp_path` fixtures for an entire
stage while `claim:phoneme_count_39` sat unbacked in the shipped README.
`tests/` stays excluded because its illustrative `claim:` tags (`alpha_claim` and
friends) are fixtures, not assertions about the project.

**Never fix a claims-discipline failure by deleting the tag from prose.** Record the
measurement with the matching `scripts/record_stage*_claims.py`. Deleting the tag
removes the evidence requirement and leaves the sentence.

**Do not pin a number copied from a run.** Every numeric assertion is an exact closed
form or re-derived in-test from the same inputs. Specifically: the unigram comparator
is rebuilt from `phone_unigrams` inside the test rather than pinned, and the trained
bigram figure is never compared against claim:stage2_phoneme_floor_perplexity — that
is an unsmoothed, transitions-only conditional-entropy floor, and asserting an
all-positions add-1-smoothed perplexity equals it is unsatisfiable by construction.
The gate's comparable bound is
claim:stage2_phoneme_all_position_floor_perplexity, which includes the same word
starts and transition positions as the reported model.

**Do not weaken a gate to the crippled comparator.** Beating uniform
(claim:stage1_uniform_baseline, claim:stage2_phoneme_uniform_baseline) is not the
gate; beating the context-free bar is. And do not restore the "≥2 nats over uniform"
wording — the conditional entropy of the training distribution caps the achievable
gain at claim:stage1_max_bigram_gain_nats nats, so it is unreachable at order 1.

**Do not use `limit=` to produce a gate statistic.** Every subsample of the
position-aligned letter-to-phone (G2P) eval
is underpowered. Significance is only reached on all claim:stage2_g2p_eval_positions
positions (p = claim:stage2_g2p_mcnemar_p). `limit=` is for smoke checks.

**Do not compare aggregate accuracies to establish the G2P margin.** It is
claim:stage2_g2p_margin_pts points. The gate pairs the decoders position-for-position
through `decode_positions` / `baseline_positions` and runs McNemar on the discordant
pairs; the unpaired comparison is indistinguishable from noise.

## Local gotchas

**Always `.venv/bin/python -m pytest`.** A bare `pytest`, or `uv run pytest` in this
sidecar, can resolve to a system interpreter with a different Python and a different
pymdp — which silently changes every number the suite asserts.

**`--no-cov` only matters once a command already asks for coverage.** `addopts` is
`--tb=short -v`, so a bare focused run never invokes coverage and the `fail_under`
under `[tool.coverage.report]` cannot fire. Adding `--no-cov` to a command that passes
`--cov` is what keeps a single-file run from failing on the floor rather than on the
test. Run the full `--cov=ailm` suite (the pyproject floor applies) before declaring anything done.

**60-second per-test timeout** (`pytest-timeout`, signal method). `decode_positions`
runs at ~4.3 ms/position, so a full-split pymdp pass would blow it. That constraint is
*why* full-set statistics go through `ailm.verification.oracle` after
`verify_decoder_equivalence` proves zero disagreements.

`test_pymdp_smoke.py` has a scoped 300-second limit for real dependency imports
and initial JAX compilation. Fresh-process failures reproduced the default-limit
overrun during matplotlib import; a separately bounded startup diagnostic passed
all smoke checks. This allowance does not change the ordinary suite default or
the smoke assertions, including the malformed-dependency negative control.

**`MPLBACKEND=Agg` for anything touching figures.** `_common.py` calls
`matplotlib.use("Agg", force=False)`, but a prior import can win. Set it in the
environment and keep the pytest command direct so its exit status cannot be
replaced by an output filter's status.

**Gate files skip rather than fail when a vendored artifact is absent.**
`test_stage1_gate.py` skips without `vendor/brown_counts.json`;
`test_stage2_gate.py` without `vendor/cmudict_counts.json`; the discipline tests
without `data/claims.json`. `test_checkout_completeness.py` is what stops that from
reading as a green suite: it fails loudly when any artifact a gate depends on is
missing, so a partial checkout cannot skip its way to a pass. A vendored or
generated-and-committed artifact added for a new stage must be added to that file's
`REQUIRED_ARTIFACTS` in the same change — a gate that skips on an artifact absent from
that inventory is the exact defect the file exists to catch.

**Audit-report consumers fail on absent evidence.** Cover, hierarchy, and Stage 6
export checks use `conftest.py::gate_audit_rows`. It requires both committed
`output/reports/gate_audit*.json` reports, their complete registered gate sets,
unique identities, and valid verdict/outcome words. Missing or empty reports
must not become skips or empty-loop successes. The real-file controls in
`test_checkout_completeness.py` exercise those refusals.

**`test_stage2_gate.py` is ~1400 lines and `test_phoneme.py` ~2100.** They are long
because they pin orientation three independent ways. Do not split them for tidiness
without keeping each control adjacent to the assertion it protects.

**Float32.** JAX's default. Exact anchors land at ~2e-7 relative — the
uniform-transition case measures 38.999998, not 39.0. Compare with a tolerance.

**Purge `__pycache__` after reverting code.** Stale `.pyc` files with colliding mtimes
have executed reverted code in this project's history:
`find . -name __pycache__ -type d -prune -exec rm -rf {} +`.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# The enforced gate
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing

# Confirm nothing silently skipped
.venv/bin/python -m pytest tests/ -q --no-cov -rs

# The no-mocks invariant (frameworks and dependency replacement).
# Scoped to *.py — these two Markdown files name the forbidden symbols in prose.
! rg -n "MagicMock|unittest\.mock|mocker\.patch|monkeypatch\.setattr" tests/*.py

# Prose discipline
.venv/bin/python -m pytest tests/test_claims_discipline.py -q --no-cov
```
