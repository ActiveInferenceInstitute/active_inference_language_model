# AGENTS.md — src/ailm/benchmarks/fitted/

Operating rules for the fitted-probe data/identity foundation. The quick
reference is [`README.md`](README.md); this file is the ownership boundary.

## Ownership boundary

| Artifact | Owner |
| --- | --- |
| Fitted-probe identity — `FITTED_PROBES`, `FittedProbeSpec`, lineage chains | `identity.py`; validated against but never merged into `SERVED_PROBES` |
| Streaming census + the stride-d aliasing refutation | `census.py` |
| Dataset builders for the declared fitting sources | `data.py` (level helpers imported lazily, at call time), including `letter_stream_from_raw` — the raw stream through the recorded letter normalizer that `scripts/generate_fitted_probe_data.py`'s letter path consumes |
| Receipts and the trainer checkpoint contract | `receipt.py`; written once, verified by digest recomputation |
| The thin data-prep CLI | `scripts/generate_fitted_probe_data.py` |
| Foundation tests | `tests/test_fitted_probe_foundation.py` |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **No mocks** | Project-wide. Receipts are verified by recomputing sha256 over real bytes |
| **`FITTED_PROBES` never feeds `SERVED_PROBES`** | The registered registry is the identity source for every historical bench family; expanding it would rewrite recorders, manuscript hydration, and figures |
| **Lineage is digests, never behavior** | Agreement-with-a-served-model is a protocol control with a pre-declared tolerance; it can never substitute for the digest chain |
| **The census is never thinned** | Fit-set thinning breaks the declared equal-fit claim; only the diagnostic slice may carry a declared stride cap |
| **Receipts are write-once and reconciling** | `bigram_tokens == supervised_targets` and the separation sum are enforced at construction; overwrites are refused |
| **No model download or training in this package** | Fitting belongs to a later unit with its own compute gate (measured warm-up throughput + checkpoint/resume) |

## Shared surfaces — coordinate before editing

`identity.py` reads `ailm.benchmarks.constants` (read-only). The receipt
schema is a contract for the future trainer: changing a field name is a
breaking change for that consumer and must be recorded in
`docs/extension_protocols.md` §2.3 in the same change.
