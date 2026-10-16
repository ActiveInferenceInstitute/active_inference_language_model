# AGENTS.md — src/ailm/benchmarks/

Operating rules for the external reference-probe package. The quick reference is
[`README.md`](README.md); this file is the ownership boundary and the invariants.

## Overview

This package measures local Ollama reference models and hosted OpenRouter
stipulated-language runs on declared next-symbol tasks. Hosted synthetic and
in-context arms use `HOSTED_SYNTHETIC_MODEL`; inherited registry suffixes are run
cohort labels and must not be rendered as local model identities. It exists so the manuscript's
reference numbers are measured here rather than quoted from a paper that used a
different task, a different population, and a different scoring rule.

It is a *probe*, not a component. AILM's generative model remains entirely pymdp;
nothing under this package participates in a level build, a stage gate, or a
comprehension path.

## Ownership boundary

| Artifact | Owner |
| --- | --- |
| Probe identity — ledger key, Ollama tag, display name | `constants.py::SERVED_PROBES`, validated at import time |
| The protocol version stamped into every call | `constants.py::PROTOCOL_VERSION` |
| The memoized call store | `cache/bench_calls.jsonl`, written through `ailm.call_cache`; ignored by Git, machine-specific |
| Recorded benchmark claims | `scripts/record_external_benchmark_claims.py`, `record_incontext_benchmark_claims.py`, `record_synthetic_benchmark_claims.py`, `record_calibration_floor_claims.py`, `record_logprob_drift_claims.py`, `record_aligned_g2p_comparator_claims.py` |
| Cache-to-claim bindings | `cache/network_claim_bindings/`, verified by `scripts/verify_network_claim_bindings.py` |
| The frozen synthetic execution contracts | `synthetic_protocol.py` |
| The stipulated chain the synthetic arms run on | `ailm.levels.synthetic` — not this package |
| The stipulated aligned spelling-to-pronunciation correspondence, seeded word generation, splits, and artifact build/validation (protocol P2) | `aligned_g2p.py` |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **No mocks, anywhere** | Project-wide invariant. A probe result must come from a real call or from a cached real call, never a fabricated response |
| **No test makes a network call** | Offline tests read the cache or skip. A test that reaches the server turns a hosted run into a coin flip |
| **`SERVED_PROBES` owns local identities and cohort keys** | Duplicate keys, tags, or display names would collapse two served systems onto one ledger namespace, one cache identity, or one legend entry — refused at import |
| **Accuracy arms and the calibration arm keep different names** | The calibration comparator is scored on its exact predictive column while the probe is scored on a *reconstruction* from truncated top-k mass. Calling both "calibration" would assert a comparison the data does not support |
| **A synthetic protocol is frozen once recorded** | `synthetic_protocol.py` values are read by recorders *and* by manuscript hydration; editing one without re-recording makes a stale ledger self-consistent and wrong |
| **The call cache is an input, not an output** | It lives in `cache/`, is gitignored, and is machine-specific. Warming it needs a live server. Never regenerate it to make a check pass |
| **Every network-excluded recorder carries a source/configuration/model/call receipt** | The release profile refuses publication acceptance without them |

## File organization

| File | Concern |
| --- | --- |
| `__init__.py` | Lazy re-export surface, preserving owner-object identity and static types; registry access stays stdlib-only |
| `constants.py` | Seeds, windows, buckets, `OLLAMA_URL`, `PROTOCOL_VERSION`, `SERVED_PROBES` and its validator |
| `corpora.py` | Evaluation positions drawn from the vendored corpora |
| `prompts.py` | Position sampling and the exact prompt text |
| `ollama.py` | HTTP transport, model-identity resolution, the content-addressed call cache |
| `comparators.py` | AILM's own letter / phoneme / G2P decoders on the same positions |
| `calibration.py` | Truncated-mass reconstruction, completion rules, the divergence floor |
| `scoring.py` | Clopper-Pearson intervals, paired exact tests, proper scores |
| `harness.py` | Replicates, repeat agreement, per-bucket metrics |
| `synthetic_protocol.py` | The no-demonstration probe and the explicit-demonstration sweep contracts |
| `aligned_g2p.py` | The stipulated aligned spelling-to-pronunciation correspondence, seeded word generation, splits, and artifact build/validation (protocol P2) |

The package-layout list in `__init__.py`'s docstring follows this table.

## Configuration

| Variable | Effect |
| --- | --- |
| `AILM_CALL_CACHE_PATH` | Overrides the call-store location; unset uses `cache/bench_calls.jsonl` |
| `AILM_OLLAMA_ENDPOINT` | Validated loopback origin, selected once at import as `OLLAMA_URL`; absent retains the legacy default. See [README.md](README.md) for selection and replay rules. |

## Usage

```python
from ailm.benchmarks import PROBE_MODEL_TAGS, ollama_available

tag = PROBE_MODEL_TAGS["gemma3"]
if ollama_available(tag):
    ...  # a warming or recording run may proceed
```

## Testing

Run from the project root, with the project's own interpreter — never `uv run`
inside this sidecar (see [`../../../docs/quickstart.md`](../../../docs/guides/quickstart.md)).

```bash
.venv/bin/python -m pytest tests/test_benchmarks.py -q --no-cov
.venv/bin/python -m pytest tests/test_benchmark_imports.py -q --no-cov
.venv/bin/python -m pytest tests/test_benchmark_interpretations.py -q --no-cov
.venv/bin/python -m pytest tests/test_calibration_floor.py -q --no-cov
.venv/bin/python -m pytest tests/test_incontext_benchmark.py -q --no-cov
.venv/bin/python -m pytest tests/test_ollama_client_contracts.py -q --no-cov
.venv/bin/python -m pytest tests/test_synthetic_benchmark.py -q --no-cov
.venv/bin/python -m pytest tests/test_synthetic_claim_registry.py -q --no-cov
.venv/bin/python -m pytest tests/test_external_benchmark_provenance.py -q --no-cov
```

## Best practices

- Adding a probe means adding a `ProbeSpec` to `SERVED_PROBES`, warming its cache
  entries, re-recording every dependent claim, and updating the supplement — in one
  change. A half-added probe shows up as a legend entry with no data behind it.
- Report a probe number with the arm that produced it. A bare accuracy from this
  package is not comparable to a level's headline score; the tasks are matched but
  the populations and the scoring rules are stated per arm.
- Never widen a bucket or a window to make an interval exclude zero. Both are frozen
  protocol values.

## See Also

- [`README.md`](README.md) — quick reference for this package
- [`../../../docs/manuscript/98_supplement_external_benchmarks.md`](../../../docs/manuscript/98_supplement_external_benchmarks.md) — how these results are reported and bounded
- [`../../../docs/output_conventions.md`](../../../docs/reference/output_conventions.md) — why `cache/` is an input and `output/` is not
- [`../../../scripts/AGENTS.md`](../../../scripts/AGENTS.md) — the recorders that write these claims
- [`../../../AGENTS.md`](../../../AGENTS.md) — the no-neural-network invariant this package sits beside
