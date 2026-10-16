# src/ — the importable AILM package root

`src/` holds exactly one thing: the `ailm` package. Nothing else belongs here — no
scripts, no notebooks, no data, no generated output. It is a src-layout Python
package, which means the import name is `ailm`, never `src.ailm`.

```
src/
└── ailm/          # the whole project library — see ailm/README.md
```

## Quick start

The package is installed editable into the project virtualenv, so imports work from
any working directory:

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -c "import ailm; print(ailm.__version__, ailm.__file__)"
```

```python
from ailm.levels.letter import build_letter_agent, build_letter_transition, load_brown_counts
from ailm.verification.oracle import markov_chain_perplexity

agent = build_letter_agent(build_letter_transition(load_brown_counts()))
```

## How `src/` is wired into the project

Three declarations in `pyproject.toml` all point here, and they must stay in step:

| Declaration | Value | Consequence if it drifts |
| --- | --- | --- |
| `[tool.hatch.build.targets.wheel] packages` | `["src/ailm"]` | a new top-level package under `src/` would not be shipped in the wheel |
| `[tool.pytest.ini_options] pythonpath` | `["src"]` | tests would import an installed copy instead of the working tree |
| `[tool.coverage.run] source` | `["src/ailm"]` | code added outside `ailm/` would be invisible to the branch-coverage floor (`claim:stage0_coverage_floor_pct%`, `[tool.coverage.report] fail_under` in `pyproject.toml`) |

## The two invariants a newcomer breaks first

**1. `src/` never imports `infrastructure`.** AILM renders through the template
repo's pipeline, but the library must not know that. Every module here is importable
from a bare checkout with only the `pyproject.toml` dependencies installed —
`cycler`, `inferactively-pymdp`, `jax`, `jaxlib`, `matplotlib`, `nltk`, `numpy`,
`pyyaml`, `requests`, and `scipy` (that list is `[project].dependencies`; treat
`pyproject.toml` as authoritative). The only place `infrastructure` may be imported is
`scripts/z_generate_manuscript_variables.py`, which exists to bridge the two.
`src/ailm/manuscript_variables.py` reads `SOURCE_DATE_EPOCH` from the environment
directly rather than borrowing the template's helper for exactly this reason.

**2. `src/` never contains a measured number as a literal.** Values come from
`data/claims.json` (via `ailm.claims`), from `output/data/manuscript_variables.json`
(via `ailm.figures._common.load_manuscript_variables`), or from the vendored corpus
artifacts. A number typed into plotting code is as fabricated as a number typed into
prose and considerably harder to notice. Prose in a docstring that states a
measurement must carry a `claim:<label>` tag — `tests/test_claims_discipline.py`
scans every `.py` file under `src/` and fails on a label with no ledger entry.

## Verification

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root
.venv/bin/python -m pytest tests/ -q --no-cov
.venv/bin/python -m pytest tests/ --cov=ailm --cov-report=term-missing
```

## More information

Per-module detail: [`ailm/README.md`](ailm/README.md). Agent boundaries and local
gotchas: [`AGENTS.md`](AGENTS.md).
