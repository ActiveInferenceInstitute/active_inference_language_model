# ailm/figures/ — manuscript figures

Every figure in the AILM manuscript, drawn with matplotlib and nothing else — no
diagram tool, no network, no external asset. Generators return
`matplotlib.figure.Figure` objects (or write and return a `Path`) and know nothing
about the pipeline; `scripts/generate_figures.py` discovers and calls them.

## Modules

Every module here, with the output stems it writes. The stems are the filenames
under `output/figures/` and, with `fig:` prefixed, the labels the manuscript
references. The authoritative filename/label/generator projection is
`output/figures/figure_registry.json`, checked against
[`../../../docs/output_inventory.md`](../../../docs/reference/output_inventory.md) and
[`../../../docs/syntax_guide.md`](../../../docs/syntax_guide.md) by
`tests/test_documentation.py`; this table names modules, not that projection.

| Module | Output stems | Argument carried |
| --- | --- | --- |
| `_common.py` | — | deterministic writing, the legibility gate, token loading |
| `architecture.py` | `hierarchy`, `inference_cycle` | the level stack, and where `A` and `B` enter the per-symbol pymdp cycle |
| `matrices.py` | `letter_transition`, `phoneme_matrices`, `wiring_comparison` | the Brown letter-bigram matrix, the Stage 2 matrices, and the measured cost of putting a learned matrix in the wrong slot |
| `results.py` | `perplexity_ladder`, `information_ceiling`, `g2p_accuracy` | uniform vs unigram vs model vs floor; the unreachable Stage 1 target; the thin G2P margin and its paired statistic |
| `gates.py` | `gate_taxonomy`, `subsample_bias` | typed-comparator VACUOUS / UNREACHABLE / ABOVE_REFERENCE / UNAUDITED / BRACKETED_BOUND / BRACKETED_REFERENCE; prefix vs stride vs full |
| `inference_evidence.py` | `inference_evidence` | per-level evidence status of the L4/L5/L6 scores: live pymdp inference vs numpy/closed-form/conditional paths, all values from tokens |
| `interpretability.py` | `case_matrices`, `trie_narrowing`, `word_matrices` | the named factors a reader can actually read off the matrices |
| `confirmatory.py` | `confirmatory_replication` | the held-back replication of the four registered members, its controls, and Level 6's clustered estimand on both populations |
| `connected.py` | `connected_core` | runtime-derived named channels and the locally frozen GUM paired character-loss comparison, authenticated against typed claims and the exact report |
| `composition.py` | `composition`, `gnn_pipeline` | how the specs compose, and what the GNN path does |
| `composition_results.py` | `case_both_defects`, `channel_comparison`, `generation_sweep`, `word_miswirings` | the measured cross-level interventions and the mis-wiring outcomes |
| `levels56.py` | `clause_null`, `narrative_channel` | the Stage 5 null and the L6→L5 intervention |
| `belief_dynamics.py` | `belief_dynamics` | how a posterior moves across a sequence |
| `learning.py` | `learning_agreement` | the recorded learning-agreement arrays from `data/learning_agreement.npz` |
| `complexity.py` | `complexity_memory`, `complexity_ops` | the operation and byte accounting of Supplement A |
| `scaling_ladder.py` | `scaling_ladder` | how the per-level state spaces scale |
| `multiplicity.py` | `multiplicity_forest` | the corrected family behind the multiplicity table |
| `null_bounds.py` | `null_bounds` | the bounds each null admits |
| `benchmarks.py` | `external_benchmarks` | the external reference probes of Supplement B |
| `calibration_floor.py` | `calibration_floor` | the divergence floor truncated top-k mass creates |
| `incontext_sweep.py` | `incontext_sweep` | the stipulated-language demonstration sweep |
| `aligned_g2p.py` | `aligned_g2p` | the P2 stipulated aligned spelling-to-pronunciation comparator arms, Wilson intervals derived at render time on the recorded denominators |
| `case_word_sweep.py` | `case_word_sweep` | the decision-121 case-word bridge influence sweep across the shipped mixing grid |
| `samples.py` | `generation_samples` | what the L3→L1g rollout actually emits |

## Quick start

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only perplexity_ladder
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --only results --list
```

```python
from ailm.figures._common import save_figure
from ailm.figures.results import build_perplexity_ladder_figure, load_variables

fig = build_perplexity_ladder_figure(load_variables())
print(save_figure(fig, "perplexity_ladder"))
```

Import from the concrete module. `ailm/figures/__init__.py` is a **partial** barrel:
it re-exports `architecture` and `matrices` but not `results` or `gates`, so
`from ailm.figures import generate_perplexity_ladder` raises `ImportError`.

## No figure contains a typed number

Values come from `output/data/manuscript_variables.json` — the same token map the
manuscript renders from — via `load_manuscript_variables` / `token_float` /
`token_int`, or they are computed from a vendored corpus artifact at generation time.
A missing token raises `MissingVariableError` rather than substituting a plausible
default. A perplexity typed into plotting code is exactly as fabricated as one typed
into prose, and considerably harder to notice.

The connected figure reads its dedicated claim family through the manuscript's
typed reader and authenticates the frozen per-document report. Its mean and
interval must reproduce from those document summaries, and its complete source
inventory and ordered population must match the canonical provenance helpers.

`results.py` and `gates.py` go further and re-derive identities that must hold
*between* separately recorded claims: the ladder must be monotone,
`log(uniform perplexity)` must equal the model's cross-entropy plus its recorded
gain, and the recorded G2P margin must equal model minus baseline. A violation means
two claims came from different runs, and the generator raises rather than drawing a
story the ledger no longer supports.

## The provenance receipt is the publication commit marker

`scripts/generate_figures.py` writes `figure_provenance.json` schema
`ailm-figure-provenance-v3`. The receipt is portable—paths are project-relative or
PNG basenames—and binds the bounded rendering surface to the published claim:

- the exact numeric environment (Python, JAX/JAXlib backend and devices, pymdp,
  NumPy, and matplotlib);
- `output/data/manuscript_variables.json` by its canonical path and SHA-256;
- each discovered generator's `src/ailm/figures/<module>.py` path and SHA-256;
- a sorted, exact inventory of every `src/ailm/**/*.py` source, so local or
  transitive computation imports cannot escape the receipt;
- the shared orchestrator, descriptor-safe atomic writer/layout helpers, and style
  contract (`scripts/generate_figures.py`, `scripts/_artifact_io.py`, `_common.py`,
  and `viz_config.py`) by path and SHA-256;
- all bounded `data/` and `vendor/` JSON/NPZ/YAML inputs plus every GNN model and
  mutation spec, by exact inventory and SHA-256;
- every generated PNG basename and SHA-256.

A selective `--only` build authenticates the complete existing receipt, including
untouched PNGs, before creating a staging directory. A changed token map, code or
data/spec inventory, source digest, mutated or missing PNG, orphan PNG, malformed
receipt, special-file substitution, or different numeric environment fails closed
and leaves the publication directory unchanged. JSON inputs are opened once with
`O_NOFOLLOW|O_NONBLOCK`, authenticated as regular by `fstat`, and parsed and hashed
from that same descriptor; a FIFO cannot block validation and a pathname swap cannot
substitute different parsed bytes.

Every selected generator renders into a temporary sibling of `output/figures/`.
Only after all generators succeed and the input/source snapshots are rechecked does
the orchestrator fsync and atomically replace each PNG. It publishes the receipt
last, as the commit marker, and only then prints final paths to stdout. A failed
generator therefore publishes neither partial PNGs nor a receipt, and a manifest
collector can never mistake a staging path for a published artifact. The individual
replacements are atomic; the whole set is not a batch filesystem transaction. If a
process or disk fails between replacements, the retained old receipt intentionally
fails to authenticate the mixed set on the next selective build or registry read,
which requires a full regeneration before reuse.

## The two invariants a newcomer would otherwise violate

### 1. Determinism is byte-level, and matplotlib fights it by default

`save_figure` pins dpi, pins `pad_inches`, and overrides the PNG `Software` metadata
field with the constant `"ailm"` — matplotlib otherwise stamps its own version in
there, so the same plot produces different bytes after a patch upgrade.
`viz_config.apply_style()` pins `font.sans-serif` to bundled DejaVu Sans, because a
substituted system font changes glyph widths, which changes the `tight` bounding box,
which changes the image. No generator writes a timestamp, uses an RNG without a fixed
seed, or iterates an unsorted collection.

Regenerating a figure twice must produce identical bytes. `tests/test_figures_*.py`
checks that by regenerating and comparing.

### 2. Legibility is a build-time invariant, and `assert_legible` only checks font size

`save_figure` calls `assert_legible`, which walks every visible, non-empty text artist
on the produced figure and raises `IllegibleFigureError` for anything below
`viz_config.MIN_FONT_PT`. It also refuses a figure with *no* visible text, because a
legibility check that passes on an empty canvas is not a check.

**There is deliberately no clamp.** A clamp that shrinks type to make it fit converts
an overflowing label into an unreadable one and reports success.

But size is only one of three ways a figure becomes unreadable. The other two —
**text drawn on top of other text**, and **text pushed outside the canvas** — are
invisible to `assert_legible`, because a 10 pt label is not legible when another 10 pt
label sits on it. Those are covered separately by `tests/test_figure_layout.py`, which
works in rendered display coordinates and carries positive controls proving each check
can fire.

That geometry sweep discovers builders by name prefix (`build_`, `make_`, `plot_`,
`figure_`, `generate_`) and only picks up **zero-argument** callables, so a new figure
whose builder takes a required positional argument is silently excluded. The module
guards against exactly that with `test_discovery_covers_every_shipped_figure`, which
compares the shipped PNG stems on disk against the discovered builder stems as sets,
so a missing figure, an extra builder, or two callables claiming one stem each fail
the sweep — the prefix list once omitted `generate_`, which quietly excluded four of
the nine figures then shipped, including two with confirmed defects, and read as a
clean sweep.

## Verification

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
MPLBACKEND=Agg .venv/bin/python -m pytest \
  tests/test_figures_architecture.py tests/test_figures_results.py \
  tests/test_figure_layout.py -q --no-cov

# Regenerate everything and confirm byte-identity across two runs
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
shasum -a 256 output/figures/*.png > /tmp/ailm_figs_a.txt
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
shasum -a 256 output/figures/*.png > /tmp/ailm_figs_b.txt
diff /tmp/ailm_figs_a.txt /tmp/ailm_figs_b.txt
```

## More information

[`AGENTS.md`](AGENTS.md). Palette, type scale and the legibility floor:
[`../viz_config.py`](../viz_config.py).

The in-context sweep retains its full comparator axis and adds a labeled zoom
of the treatment/control intervals in each panel. All panels share the zoom's
vertical range, and each parent outlines the enlarged region. The same token
values draw both views.

`save_figure` also rejects standard axes-coordinate reference rules crossing
labels. Fully opaque label/legend backgrounds drawn above the rule are respected.
This check covers the standard horizontal, vertical, and frame transforms; it
does not infer reference-rule semantics from data curves or custom composed
transforms. See the neighboring agent guide for the precise scope and controls.
