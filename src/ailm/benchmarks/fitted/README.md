# ailm/benchmarks/fitted/ — fitted-probe data/identity foundation

The separate experimental surface for the fitted external-probe arm
([`docs/extension_protocols.md`](../../../../docs/extension_protocols.md) §2).
Nothing here enters [`../constants.py::SERVED_PROBES`](../constants.py),
touches any historical bench namespace, downloads a model, or trains.

## What is implemented

| Module | Owns |
| --- | --- |
| `identity.py` | The strict fitted-model identity schema: `FITTED_PROBES` (separate registry, empty until a fit registers), `validate_fitted_probes`, `build_lineage_chain` (five ordered digest stages), digest helpers. Lineage is digests only — no behavioral-agreement field exists |
| `census.py` | `StreamCensus` — streaming exact-census counting with per-occurrence context accounting that always reconciles; `stride_aliasing_demo` — the checkable refutation that stride-d subsampling preserves bigram frequencies |
| `data.py` | Deterministic dataset builders for the declared letter/phoneme/G2P fitting sources: instance construction, positional diagnostic carve (seam gap, contiguous tail), the experimental comparator's refit counts over the fit region, strict 1:1 G2P instances, scored-task-shaped phoneme streams, and `letter_stream_from_raw` — raw corpus text through the letter level's recorded normalizer, consumed by the letter path of `scripts/generate_fitted_probe_data.py` |
| `receipt.py` | `DataReceipt` (immutable, arithmetic-reconciling, write-once) with real digest re-verification via `verify_receipt`, and `CheckpointSpec` — the contract a later isolated trainer binds its adapter run to |

Thin CLI: [`scripts/generate_fitted_probe_data.py`](../../../../scripts/generate_fitted_probe_data.py)
(`--dataset letter|phoneme|g2p`, declared `--run-scope`, write-once outputs,
self-verifying receipts). Full-corpus letter preparation is gated behind
`--allow-full-corpus` and refused in normal runs — it is scheduled work.

## What is NOT here (yet)

No trainer, no model download, no weights, no tokenizer inspection, no
behavioral-agreement claims. The fitting objective, its controls (C1–C5), and
the serving path live in the protocol document and its later units.

## Testing

```bash
.venv/bin/python -m pytest tests/test_fitted_probe_foundation.py -q --no-cov
```
