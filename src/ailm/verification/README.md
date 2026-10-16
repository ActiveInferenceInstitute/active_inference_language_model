# ailm/verification/ — the independent instrument

`oracle.py` implements a closed-form HMM in pure numpy. Its independent
implementation lets it detect a flaw in the model it audits.
`connected_provenance.py` separately provides the standard-library execution
inventory shared by the connected recorder and figure. This provenance check
performs no inference and is not a numerical oracle.

> **`oracle.py` must never import `pymdp` or `jax`.** The numerical oracle remains
> independent of the runtime.

## Why it exists

Stage 1 shipped a bug pymdp could not see: the learned bigram matrix was placed in
the emission `A` with an identity `B`. That wiring is anti-informative rather than
loudly broken — observing symbol `c` yields a posterior over the symbol that
*preceded* `c`. A bigram matrix is column-stochastic, so `pymdp._validate()` accepts
it in either slot. Measured, the defect scored
claim:stage1_anti_informative_perplexity against claim:stage1_letter_perplexity for
correct wiring, and on a small in-sample corpus scored *worse than uniform* while
still slipping under a naive threshold.

What caught it was a hand-rolled numpy Markov chain that disagreed with pymdp by
roughly 15 perplexity units. That instrument was a throwaway scratch script. This
module is that instrument promoted to a tested, first-class dependency, so every
level from Stage 2 onward can be cross-checked **numerically** rather than
structurally.

It has a second job that matters just as much: it is fast. `decode_positions` costs
milliseconds per position through pymdp, so the full G2P test split —
claim:stage2_g2p_eval_positions positions — is minutes of `infer_states` calls. The resolution was not to shrink
the eval set — it was to prove the pymdp decoder and this oracle agree on
claim:stage2_decoder_disagreements positions, then compute full-set statistics through
the oracle.

## API

| Function | Returns |
| --- | --- |
| `canonical_execution_binding(root)` | Complete relative source/input paths and SHA-256 digests, from `connected_provenance.py` |
| `forward_filter(emission, transition, start, obs)` | `(posteriors, log_evidence)` — the scaled alpha recursion |
| `forward_backward(...)` | smoothed `P(state_t \| obs_0..obs_{T-1})` |
| `viterbi(...)` | the joint-MAP state path, in log space |
| `sequence_perplexity(...)` | `exp(-log_evidence / T)` |
| `identity_emission(n)` | the `(n, n)` identity — a directly observed state |
| `markov_chain_perplexity(transition, start, obs)` | perplexity of a directly observed chain |

## Pinned conventions — a transpose here silently invalidates every cross-check

```
emission    (n_obs, n_states)      emission[i, j] = P(obs = i    | state = j)
transition  (n_states, n_states)   transition[i, j] = P(next = i | cur = j)
start       (n_states,)            distribution over state_0, BEFORE any observation
```

All three are column-stochastic down axis 0 — the axis indexing the *produced*
variable. This matches the pymdp layout once the leading batch dimension is stripped:
`A[0]` is `(n_obs, n_states)` and `B[0, :, :, 0]` is `(n_next, n_current)`, both
validated along pymdp's `axis=1`, which is this module's `axis=0`.

Because `start` is the distribution over `state_0` before any observation, the
predictive distribution at `t = 0` is `start` itself — the transition is **not**
applied first.

## Quick start

```python
import numpy as np
from ailm.verification.oracle import identity_emission, markov_chain_perplexity

n = 4
uniform = np.full((n, n), 1.0 / n)
# Analytic anchor: uniform transition + uniform start + identity emission => exactly n
print(markov_chain_perplexity(uniform, np.full(n, 1.0 / n), [0, 1, 2, 3, 0]))
```

## The two invariants a newcomer would otherwise violate

**1. No pymdp, no jax — not even inside a function, not even lazily.** A local import
still couples the instrument to the instrument under test. The numerical references
use float64 numpy; the provenance inventory uses the standard library.

**2. Every expected value in this module's tests is derived analytically or by hand
arithmetic written out in a comment.** No number in `tests/test_oracle.py` was copied
from a run of the code under test. The oracle exists in order to *disagree* with
pymdp, so it cannot be calibrated against anything but a closed form. The anchors are
stated in that file's docstring: uniform-everything gives exactly `n` for any
observation sequence; a deterministic chain observed along its own trajectory gives
exactly 1; a worked 2×2 forward pass; and a transposition control proving predictive
mass lands on the column rather than the row.

## Numerical policy

Forward and backward messages are renormalized at every timestep and the log-evidence
is accumulated from the normalizers, so a 5000-symbol sequence cannot underflow.
Viterbi runs entirely in log space. Probabilities are clipped to `1e-300` before any
`log`, keeping it finite at about −690.8 nats. Zero-likelihood observations fall back
to the predictive belief with a warning, after the floor penalty has already been
charged to the evidence.

## Verification

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -m pytest tests/test_oracle.py -q --no-cov

# The independence invariant itself
.venv/bin/python -c "
import sys, ailm.verification.oracle
leaked = sorted({m.split('.')[0] for m in sys.modules} & {'pymdp','jax','jaxlib'})
assert not leaked, f'oracle pulled in {leaked} — it is no longer independent'
print('oracle is independent')
"
```

## More information

[`AGENTS.md`](AGENTS.md). Cross-check call sites: `tests/test_phoneme.py`,
`tests/test_stage2_gate.py`, `scripts/record_stage2_claims.py`.
