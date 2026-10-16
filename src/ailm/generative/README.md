# ailm/generative/ — pymdp API surface verification

One module: `smoke.py`. It answers a single question, against the pymdp actually
installed: **does this version have the API the AILM architecture is built on?**

pymdp is not an optional backend here — it is the entire generative engine. There is
no fallback path, so an API change upstream is an architecture-level failure, not a
dependency warning. This module makes that failure arrive at Stage 0 instead of at
Stage 5.

## What it verifies

| Function | Claim under test | Why it matters |
| --- | --- | --- |
| `make_two_factor_agent()` | a minimal 2-factor / 2-modality `Agent` constructs with `A_dependencies=[[0],[1]]` | the factorization primitive the whole hierarchy rests on |
| `verify_a_dependencies_shape(agent)` | `agent.A[m]` is **3-D** `(batch, n_obs_m, n_states_for_m)`, not 4-D+ | a flat joint `A` at the word level is physically impossible to store |
| `verify_empirical_prior_coupling(agent)` | `update_empirical_prior` exists and is callable | it is the second half of the filtering recursion — a level's own posterior pushed through its own `B` to form its next prior. Within-level and temporal, never a channel between levels; see `docs/hierarchy.md` |
| `verify_no_update_a_method(agent)` | `update_A` / `update_B` do **not** exist; `infer_parameters` does | learning goes through `infer_parameters`; code written against `update_A` targets an API that isn't there |
| `verify_infer_policies_output(agent)` | `infer_policies(qs)` returns a 2-tuple `(q_pi, neg_efe)` | generation is policy selection over next-symbol commitments scored by G(π) |
| `run_all_smoke_tests()` | all of the above, aggregated into `all_passed` | the Stage 0 gate reads this one boolean |

## Quick start

```python
from ailm.generative.smoke import run_all_smoke_tests

results = run_all_smoke_tests()
assert results["all_passed"], results
print(results["a_dependencies_shape"]["A0_shape"])   # (1, 5, 4)
print(results["empirical_prior_coupling"]["call_signature"])
```

## Real pymdp, no mocks

Every function here constructs and calls a live `pymdp.agent.Agent`. Mocking the
thing you are verifying the existence of is a contradiction: a `MagicMock` has every
attribute, so `hasattr(agent, "update_empirical_prior")` would pass against any
pymdp version, including one that does not have it. That is why
`verify_no_update_a_method` is a *negative* existence check — it asserts an attribute
is **absent**, which a mock can never demonstrate.

pymdp (1.0.3, re-verified under the 1.0.4 pin) requires JAX arrays, so the fixtures are `jnp`, and every tensor carries a
leading batch dimension even at `batch_size=1`.

## The two things a newcomer gets wrong here

**1. `run_all_smoke_tests()` never raises.** Each sub-check catches its own exception
and returns `{"success": False, "error": ...}` so one failure does not mask the
others. Reading a truthy return value as success is therefore wrong — the result is a
dict, and dicts are truthy. Read `results["all_passed"]`, and note that
`all_passed` is computed from a specific list of checks, not from every key: a
sub-result can carry `success: True` while the check that actually matters
(`factorization_works`, or `has_update_A` being `False`) fails.

**2. `verify_empirical_prior_coupling` tries two call signatures** —
`update_empirical_prior(action, qs)` and `update_empirical_prior(qs)` — because the
signature moved between pymdp 1.0.1 and 1.0.3 (the 1.0.3 form is also the 1.0.4 form). It reports which one worked in
`call_signature`. That tolerance belongs here, in the probe. Production code in
`ailm/levels/` calls the 1.0.3/1.0.4 form directly and must not acquire a fallback: silently
accepting either signature there would let a version mismatch reach a published
number.

## Verification

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -m pytest tests/test_pymdp_smoke.py tests/test_stage0_gate.py -q --no-cov

.venv/bin/python -c "
from ailm.generative.smoke import run_all_smoke_tests
r = run_all_smoke_tests()
print('all_passed:', r['all_passed'])
for k, v in r.items():
    if isinstance(v, dict): print(' ', k, v)
"
```

## More information

[`AGENTS.md`](AGENTS.md). Why factorization is mandatory:
`../../../docs/design/architecture.md`.
