# AGENTS.md — src/

Layer contract for the `src/` tree. Per-subpackage rules live in the nested
`AGENTS.md` files; this file states what applies to *everything* under `src/`.

## Ownership boundary

`src/` is the only place business logic may live. Concretely:

| Concern | Where it belongs |
| --- | --- |
| A generative model, matrix builder, metric, decoder | `src/ailm/levels/` |
| An independent numerical check | `src/ailm/verification/` |
| A corpus parser | `src/ailm/corpus/` |
| A figure | `src/ailm/figures/` |
| Reading a file, printing a path, parsing argv, exiting non-zero | `scripts/` |
| A number that appears in prose | `data/claims.json`, written by a recorder in `scripts/` |

Shared reusable computation belongs in `src/`; bounded recorder-specific
estimands may stay in their owning script with direct tests. An agent
adding I/O orchestration or CLI plumbing to `src/` is doing the inverse. The one
exception is deliberate and narrow: modules here may *read* the vendored corpus
artifacts and the generated token map, because that is how they avoid hardcoding
measurements.

## May change freely

- Add a module or subpackage under `src/ailm/`, with tests in `tests/` in the same
  change. New code arriving without a test lowers the coverage ratio against a
  `fail_under` coverage gate in `pyproject.toml` (claim:stage0_coverage_floor_pct%), so an untested addition fails the suite rather than
  merely being untidy.
- Extend a docstring. These docstrings are the project's design record — several are
  longer than the code they document, on purpose. Do not compress them.
- Tighten a validation or an assertion message.

## Must not change without an explicit decision

- **The `infrastructure` prohibition.** No module under `src/` may import
  `infrastructure.*`. If a task appears to need it, the work belongs in
  `scripts/z_generate_manuscript_variables.py`.
- **`ailm.verification` must not import `pymdp` or `jax`** — see
  [`ailm/verification/AGENTS.md`](ailm/verification/AGENTS.md). This is the single
  most load-bearing import rule in the repository.
- **Which matrix carries the learned statistics.** Sequence dynamics live in the
  transition `B`; the emission is `A`. Reversing them is anti-informative rather
  than loudly broken, and `pymdp._validate()` accepts a column-stochastic matrix in
  either slot. See [`ailm/levels/AGENTS.md`](ailm/levels/AGENTS.md).
- **Package layout.** `src/ailm` is named in three places in `pyproject.toml`
  (wheel packages, pytest `pythonpath`, coverage `source`). Renaming or adding a
  sibling top-level package means editing all three.

## Local gotchas

**The `src.ailm` import does not exist.** This is a src-layout project. Import
`ailm.levels.letter`, never `src.ailm.levels.letter`. Nothing in the tree uses the
`src.` prefix; adding it works only by accident of `sys.path` and breaks under the
installed wheel.

**`ailm.figures` is a partial barrel.** `ailm/figures/__init__.py` re-exports
`architecture` and `matrices` but *not* `results` or `gates`, so
`from ailm.figures import generate_perplexity_ladder` raises `ImportError` while
`from ailm.figures.results import generate_perplexity_ladder` works. The figure
pipeline does not notice because `scripts/generate_figures.py` discovers generators
with `pkgutil` rather than reading the barrel. Import from the concrete module.

`ailm.levels.letter.load_brown_counts` reports the real generator,
`scripts/generate_counts.py`, when the vendored Brown artifact is absent.

**Docstring prose is scanned.** `tests/test_claims_discipline.py` globs
`src/**/*.py` and extracts every `claim:<label>` tag. A tag with no entry in
`data/claims.json` fails the suite. Never fix that failure by deleting the tag;
record the measurement with the matching `scripts/record_stage*_claims.py`.

**Historical pymdp warning.** The `JAX array is being set as static` warning
was observed before the pinned 1.0.4 line; it is not a current expected warning.
Existing narrow call-site filters remain (see
`scripts/audit_gates.py`). Do not silence it globally — a blanket
`warnings.filterwarnings("ignore")` in library code would hide real ones.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

# Layer contract: no infrastructure import anywhere under src/
! grep -rn "^\s*\(from\|import\)\s\+infrastructure" src/

# Full suite plus the coverage gate that governs this tree
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing

# Prose discipline over src/ docstrings
.venv/bin/python -m pytest tests/test_claims_discipline.py -q --no-cov
```

Never `uv run pytest` in this sidecar — it can resolve to a system pytest with the
wrong Python and the wrong pymdp. Always `.venv/bin/python -m pytest`.
