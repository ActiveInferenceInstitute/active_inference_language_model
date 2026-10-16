# ailm/benchmarks/ — mock-free external reference probes

Scores local Ollama reference models and hosted OpenRouter stipulated-language
runs on declared next-symbol tasks. The synthetic letter/phone and in-context
recorders use `HOSTED_SYNTHETIC_MODEL`; their inherited registry keys label run
cohorts, not the correspondingly named local weights. Receipts bind the exact
provider route and payload, while hosted model slug hashes do not pin serving weights.
Those hosted-only recorders have no Ollama prerequisite or warm-up; their
`--endpoint` selects the chat endpoint and publication uses canonical OpenRouter.

> **Ollama is an external probe only.** No neural model is part of AILM's pymdp
> stack, and nothing in this package makes AILM a neural language model. The
> project invariant in [`../../../AGENTS.md`](../../../AGENTS.md) is unchanged.

## Quick start

Requires a running Ollama server, or a warmed call cache. Run from the project root.

```python
from ailm.benchmarks import PROBE_DISPLAY_NAMES, PROBE_MODEL_TAGS, ollama_available

print(PROBE_MODEL_TAGS)     # {'gemma3': 'gemma3:4b', 'qwen25': ..., 'smollm': ...}
print(PROBE_DISPLAY_NAMES)  # ledger suffix -> reader-facing name
print(ollama_available(PROBE_MODEL_TAGS["gemma3"]))  # False with no server running
```

`ollama_available` takes the model tag; it answers "is *this* probe served here?",
not "is a server up?".

Constants and probe identities are available without importing the numerical
runtime. The package loads each function from its owner when first accessed,
preserving the existing API and function identity. Missing dependencies still
raise when that function's module is requested.

## Key features

- **Same object, same score.** An accuracy arm scores probe and comparator on what
  each actually answered. The calibration arm is deliberately named differently
  (`reconstructed_calibration_metrics`) because the endpoint returns a truncated
  candidate list, so the probe is scored on a *reconstruction* of a predictive
  column while the comparator is scored on its exact one.
- **One registry for probe identity.** `SERVED_PROBES` holds the ledger key, the
  mutable Ollama tag, and the display name together, validated at import time, so a
  local recorder and its figures agree. Hosted-only renderers use `hosted_cohort_label`
  for the retained suffixes and disclose the actual hosted model separately.
- **Content-addressed call cache.** Every request is memoized in `cache/` keyed by
  prompt and model digest, so a recorded result is replayable without the server.
  Override the location with `AILM_CALL_CACHE_PATH`.
- **Frozen synthetic protocols.** `synthetic_protocol.py` holds the immutable
  execution contracts for the stipulated-language arms, so a stale ledger cannot
  become authoritative by being internally self-consistent.

## Local serving choice and drift population

Local scored generation and uncached warmup explicitly send
`options.use_mmap: false` and the fixed default `options.num_ctx` declared in
`constants.py`. Intentional explicit valid context limits remain supported. The request
builder also owns the logprob payloads and reconstructed publication requests,
so the loader choice is part of the actual request and its content-addressed
cache key. Requests that omitted either field, used another context, or enabled memory mapping remain
historical records and cannot satisfy a new request. Model tags, weight-blob
identity verification, prompts, seeds and scoring rules retain their contracts;
there is no alternate-endpoint or automatic loader fallback.

The external recorder's precondition and post-measurement cache audit also
select only requests with the declared loader and default context choices. Their sampling seed mixes
the digest of that eligible population, so unrelated historical rows cannot
change which compatible requests are audited. The policy is recorded as
`current-endpoint-loader-context-record-digest-seeded-v3` in the external receipt's
configuration. No compatible rows can be treated as a cold configuration before
measurement, explicitly without claiming an audit pass; standalone verification
and the strict check before ledger publication still require every declared
model/request-kind stratum. Historical rows remain in the store.

Before new drift measurements under this serving choice, the drift recorder's
eligible population is restricted to logprob keys bound by the current external
benchmark producer receipt. It authenticates that receipt against current
source, configuration, claims, cached records and semantic request reconstruction.
Every eligible request must declare the fixed loader and default context choices. Unrelated cache
cohorts cannot enlarge the population, and an insufficient model stratum refuses
measurement. The model-stratified sample size, seed and answer-slot comparison
remain the registered ones. Drift receipts bind the exact producer receipt as
well as their eligible-record digest; a changed producer requires a new drift
measurement. This is a prospective eligibility correction for the new serving
configuration, not a reinterpretation of historical drift results.

## Files

| File | What it owns |
| --- | --- |
| `__init__.py` | Lazy re-exports with static types and the accuracy-vs-calibration naming rationale |
| `constants.py` | Seeds, window sizes, bucket definitions, `OLLAMA_URL`, `PROTOCOL_VERSION`, `SERVED_PROBES` |
| `corpora.py` | Drawing evaluation positions out of the vendored corpora |
| `prompts.py` | Position sampling and the exact prompt strings handed to a model |
| `ollama.py` | Transport to the local server plus the content-addressed call cache |
| `comparators.py` | AILM's own decoders, scored on the same positions |
| `calibration.py` | Reconstructing a distribution from truncated top-k mass, and the floor that creates |
| `scoring.py` | Interval estimates, paired tests, and proper scores |
| `harness.py` | Running a whole benchmark: replicates, agreement, per-bucket metrics |
| `synthetic_protocol.py` | Frozen contracts for the no-demonstration probe and the demonstration sweep |
| `aligned_g2p.py` | The stipulated aligned spelling-to-pronunciation correspondence, seeded word generation, splits, and artifact build/validation (protocol P2) |

The `fitted/` subpackage (fitting datasets, stream census, identity digests,
receipt) carries its own [`README.md`](fitted/README.md) and
[`AGENTS.md`](fitted/AGENTS.md); the fitted-probe arm's protocol and recorders
are described there and in their owning lane's handoff.

## Common commands

Run from the project root with the project's own interpreter.

```bash
# Offline: everything that does not need a server
.venv/bin/python -m pytest tests/test_benchmarks.py -q --no-cov
.venv/bin/python -m pytest tests/test_benchmark_imports.py -q --no-cov
.venv/bin/python -m pytest tests/test_calibration_floor.py -q --no-cov
.venv/bin/python -m pytest tests/test_synthetic_benchmark.py -q --no-cov
.venv/bin/python -m pytest tests/test_ollama_client_contracts.py -q --no-cov

# Provenance of the recorded external-benchmark claims
.venv/bin/python -m pytest tests/test_external_benchmark_provenance.py -q --no-cov

# Warm the call cache against a live server (slow; needs Ollama)
.venv/bin/python scripts/warm_call_cache.py
```

## More information

See [`AGENTS.md`](AGENTS.md) for the ownership boundary and the invariants, and
[`../../../docs/manuscript/98_supplement_external_benchmarks.md`](../../../docs/manuscript/98_supplement_external_benchmarks.md)
for how these numbers are reported and bounded.

Default-context requests are refused before HTTP if the source-derived UTF-8
byte-token bound, requested response and template reserve exceed the declared
capacity. The multiplier, reserve and versioned `context_budget_policy` are
configuration constants; the external receipt and warm planner require their
exact declaration. This conservative guard is scoped to the registered local
models and inspected serving templates, not arbitrary replacement templates.
The full publication request plan is checked against this capacity before fresh
measurement; actual server prompt token counts remain part of the receipt.

Local endpoint selection uses `AILM_OLLAMA_ENDPOINT`. An absent variable retains
`http://localhost:11434`; an explicit value must satisfy the existing HTTP
loopback-origin contract. Empty values, remote hosts, credentials, paths, queries
and fragments are refused before a request, without falling back. Selection is
canonicalized once at process import: export it before launching each recorder,
warmer, verifier or release process. Changing an environment mapping after import
does not retarget that running process.

The selected endpoint defines the publication endpoint for existing CLI guards
and is preserved in controlled release children and recorded in their environment
receipt. Request keys, producer configurations and reconstructed requests bind
that origin. External semantic replay and drift refuse a recorded endpoint that
differs from the selected configuration; verifying historical evidence requires
launching a separate process with its recorded endpoint selected. An origin
change is a new execution configuration, not permission to rebind old records.
Cache audits filter authenticated endpoint identities before forming the current
population, its sampling digest or any model/request-kind stratum. Historical
records from another origin remain intact and contribute no current coverage.
A dedicated server can isolate scheduling, but shares hardware resources; changing
only a model alias on the same server does not isolate its runner.
