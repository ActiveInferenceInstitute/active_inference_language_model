# AGENTS.md — src/ailm/generative/

## Ownership boundary

`generative/` owns **probes of the pymdp API surface** — assertions about what the
installed library provides. It does not own any AILM model.

The name is aspirational: the package docstring reserves it for future
generation-side machinery (policy selection over next-symbol commitments, the G(π)
cascade from a document-position (tercile) C vector — `narrative` in code — down
to a letter). Today it contains only
`smoke.py`. Level models live in `ailm/levels/`, and that is where a `word.py` or a
`clause.py` belongs — **not** here, even though those are "generative models" in the
active-inference sense.

## May change freely

- Add a probe for another pymdp facility the architecture starts depending on. Pattern:
  a `verify_*(agent) -> dict[str, Any]` that catches its own exceptions, plus an entry
  in `run_all_smoke_tests()`, plus a line in that function's `checks` list.
- Extend a probe's returned dict with more diagnostic keys.
- Add a fixture agent alongside `make_two_factor_agent` for a shape the probes cannot
  currently exercise.

## Must not change without an explicit decision

**Do not mock pymdp, monkeypatch it, or stub an array.** A `MagicMock` satisfies every
`hasattr`, so a mocked probe would pass against a pymdp that lacks the method
entirely — the probe would assert nothing while looking green. This is the one
directory where the no-mocks rule is not a style preference but the thing being
measured.

**Do not delete `verify_no_update_a_method`, and do not invert its polarity.** It
asserts `update_A` and `update_B` are **absent** and `infer_parameters` is present.
An absence check is exactly what a mock cannot demonstrate, and it is what stops
someone writing learning code against an API that does not exist.

**Do not add a fallback to production code to match the tolerance here.**
`verify_empirical_prior_coupling` deliberately tries both the
`update_empirical_prior(action, qs)` and `update_empirical_prior(qs)` signatures
because they differed between pymdp 1.0.1 and 1.0.3 and reports which one worked (the 1.0.3 form is also the 1.0.4 form).
`ailm/levels/` calls the 1.0.3/1.0.4 form directly, on purpose: a silent
signature-negotiating wrapper there would let a version mismatch produce numbers
nobody could reproduce.

**Do not unpin `inferactively-pymdp==1.0.4`** without re-running every stage gate and
re-recording every affected claim. There is no second backend, so a version bump is a
model change.

**Do not let a probe raise.** Every `verify_*` swallows its own exceptions and returns
a failure dict, so one broken check does not hide the state of the rest. The *test*
raises; the probe reports.

## Local gotchas

**`run_all_smoke_tests()` returns a truthy dict even when everything failed.** Read
`results["all_passed"]`. And `all_passed` is derived from an explicit `checks` list —
`factorization_works`, `method_exists`, `not has_update_A`, `has_infer_parameters`,
`infer_policies_output.success` — not from every `success` key present. Adding a probe
without adding it to `checks` produces a probe that can never fail the gate.

**`make_agent` failure short-circuits.** If the `Agent` constructor raises,
`run_all_smoke_tests` returns immediately with only `make_agent` and `all_passed`
populated. Downstream code indexing `results["a_dependencies_shape"]` will `KeyError`
rather than see a failure — use `.get()` when reporting.

**Shapes are hardcoded to the fixture.** `verify_a_dependencies_shape` checks for
`(1, 5, 4)` and `(1, 4, 3)`, matching `make_two_factor_agent`'s `n_states = [4, 3]`,
`n_obs = [5, 4]`. Change the fixture and this probe silently starts reporting
`factorization_works: False`. Change both together.

**pymdp 1.0.3/1.0.4 requires `jnp`, not `np`.** Passing numpy arrays to `Agent` fails or
converts silently depending on the call. Every tensor also carries a leading batch
dimension even at `batch_size=1`.

**pymdp emits `JAX array is being set as static` on every `Agent()`.** Benign. Filter
it at a noisy call site if needed (see `scripts/audit_gates.py`); never globally.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

.venv/bin/python -m pytest tests/test_pymdp_smoke.py tests/test_stage0_gate.py -q --no-cov

# Direct probe run, with the aggregate the Stage 0 gate reads
.venv/bin/python -c "
from ailm.generative.smoke import run_all_smoke_tests
r = run_all_smoke_tests()
for k, v in r.items():
    if isinstance(v, dict): print(k, '->', v)
print('all_passed:', r['all_passed'])
raise SystemExit(0 if r['all_passed'] else 1)
"

# The installed version must match the pin
.venv/bin/python -c "
import importlib.metadata as m
v = m.version('inferactively-pymdp'); print(v)
assert v == '1.0.4', f'pymdp {v} != pinned 1.0.4 — re-run every stage gate'
"
```
