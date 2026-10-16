# AGENTS.md — src/ailm/verification/

## The rule that overrides convenience

**Nothing in this directory may import `pymdp` or `jax`, at module level or inside a
function.** Not "prefer not to" — may not. The oracle's only value is being an
implementation that cannot inherit the flaw it audits for. A lazy import inside a
helper couples it just as completely as a top-level one, and does so invisibly.

If a task seems to require pymdp here, the task belongs in `ailm/levels/` or in a
test that imports *both* and compares them.

The corollary is that this module cannot be refactored to "share code" with
`ailm/levels/`. A shared matrix builder would mean a transposition bug appears
identically on both sides of the comparison and the cross-check goes green.

## Ownership boundary

`verification/` owns closed-form reference implementations and source/input
identity checks. `connected_provenance.py` supplies the canonical execution
inventory shared by the connected recorder and figure; it uses the standard
library and performs no inference. When changing that inventory, require a fresh
measurement binding rather than relabeling an earlier report. It does not own:

- anything that builds a pymdp `Agent` → `ailm/levels/`
- the claims those cross-checks license → `data/claims.json`, via `scripts/`
- gate liveness (positive/negative controls) → `ailm/control.py`

The oracle is *the instrument*, not *the gate*. It reports a number; a test decides
whether the number is acceptable.

## May change freely

- Add another closed-form algorithm (Baum–Welch, posterior decoding, a second-order
  chain) in pure numpy, with analytically-derived tests.
- Tighten a validator or improve an error message. The validators are the reason a
  transposed input fails loudly instead of producing a plausible wrong number.
- Extend `__all__` for a new public function.

## Must not change without an explicit decision

**Do not change the orientation conventions.** `emission[i, j] = P(obs=i | state=j)`,
`transition[i, j] = P(next=i | cur=j)`, both column-stochastic down axis 0; `start` is
the distribution over `state_0` *before* any observation, so the predictive
distribution at `t = 0` is `start` itself. Every cross-check in the repository
translates pymdp's layout into these, and a transpose here silently invalidates all of
them while leaving both implementations self-consistent.

**Do not relax `_validate_column_stochastic` or `_STOCHASTIC_TOL`.** The mis-wirings
this project has actually shipped were all column-stochastic *somewhere*; the
validator's job is to insist they be column-stochastic in the right place.

**Do not calibrate a test against a run of the code.** Every expected value in
`tests/test_oracle.py` is derived on paper. Replacing one with a captured output turns
the oracle into a regression fixture for itself, which is exactly the failure mode it
was built to escape.

**Do not remove the analytic anchors or the transposition control.** Uniform
everything must give exactly `n`; a deterministic chain observed along its own
trajectory must give exactly 1; the orientation test must fail if predictive mass
lands on the row instead of the column. An identity transition is column-stochastic
either way, which is precisely why an identity-only test cannot detect a transpose.

**Do not switch to float32.** This module is the float64 reference that the float32
JAX path is compared against at ~2e-7 relative tolerance. Matching dtypes would
remove the independence in a second dimension.

## Local gotchas

**The dependency boundary is automated.**
`tests/test_oracle_independence.py` imports the oracle in a fresh isolated
interpreter and fails if `pymdp`, `jax`, or `jaxlib` appears in `sys.modules`.
Its planted-leak positive control proves the detector itself can fail. Keep the
one-liner below as a useful manual probe, not as the only enforcement.

**`forward_backward` calls `forward_filter`, which re-validates.** That is
intentional redundancy, not waste. Do not thread a `validate=False` flag through to
"optimize" it.

**Zero-likelihood fallbacks warn rather than raise.** `forward_filter` falls back to
the predictive belief and `forward_backward` to the filtered belief, after charging
the `1e-300` floor to the evidence. If you convert either to an exception, long
real-corpus runs that currently complete will start aborting.

**`viterbi` returns the joint-MAP path**, which is not the sequence of per-timestep
argmaxes of `forward_backward`. `decode_phonemes` in `ailm/levels/phoneme.py` uses
*filtering* argmaxes, a third thing again. Comparing the wrong pair produces a
spurious "disagreement" and would void a claim that is actually fine.

**`identity_emission` rejects `bool`.** `isinstance(True, int)` is `True` in Python,
so the check is explicit. Do not simplify it.

## Verification for this directory

```bash
cd /path/to/active_inference_language_model

# The independence invariant — run this on every change here
.venv/bin/python -c "
import sys, ailm.verification.oracle
leaked = sorted({m.split('.')[0] for m in sys.modules} & {'pymdp','jax','jaxlib'})
assert not leaked, f'oracle pulled in {leaked} — it is no longer independent'
print('oracle is independent')
"

# Closed-form tests
.venv/bin/python -m pytest tests/test_oracle.py -q --no-cov

# The cross-checks that consume the oracle
.venv/bin/python -m pytest \
  tests/test_phoneme.py tests/test_phoneme_full_corpus.py tests/test_stage2_gate.py \
  -q --no-cov
```
