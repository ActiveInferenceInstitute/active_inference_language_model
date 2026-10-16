# Contributing

This is a research repository with a hard evidentiary standard, not a
general-purpose open-source project soliciting drive-by pull requests. Reading
[`AGENTS.md`](AGENTS.md) before opening one will save both of us time — it is
the actual operating contract this repository holds contributors (and
AI agents) to, and this file only summarizes it.

## Before you start

1. Read [`AGENTS.md`](AGENTS.md) in full. It covers the level architecture,
   the stage-gate protocol, the claims-ledger discipline, and the things this
   project will never do (no transformers, no learned embeddings, no
   `update_A`/`update_B`).
2. Read [`docs/guides/forking_guide.md`](docs/guides/forking_guide.md) if you are adding a
   new level or extending an existing one — it is a checklist, not prose.
3. Check [`docs/decisions.md`](docs/decisions.md) before re-proposing
   something that was already tried and measured; the rationale for why it
   was rejected or restated is recorded there, not just the outcome.

## Environment

Install with [`uv`](https://docs.astral.sh/uv/), not bare `pip`:

```bash
uv sync
```

Run tests against the project's own interpreter — never `uv run pytest` in
this repository, which can resolve to a different pymdp than the one the
gates were measured against:

```bash
.venv/bin/python -m pytest tests/ -v --cov=ailm --cov-report=term-missing
```

Lint and format with Ruff:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

## The no-mocks policy

`MagicMock`, `unittest.mock`, `mocker.patch`, and any other fake-substitution
framework are prohibited in this repository's tests. The project's central
claim is that the generative engine is real `pymdp`; a mocked `Agent` would
let the suite stay green while that claim became false. See
[`docs/testing_philosophy.md`](docs/testing_philosophy.md) for what replaces
mocks (real fixtures, small real corpora, oracle cross-checks).

## The claims-ledger discipline

Every quantitative claim in prose (README, AGENTS.md, docs/, the manuscript)
must resolve to a `claim:<label>` tag backed by an entry in
`data/claims.json`, written by a recording script under `scripts/`. Never
type a measured number directly into prose — the claims-discipline test
suite (`tests/test_claims_discipline.py`) rejects it. See
[`docs/method/style_guide.md`](docs/method/style_guide.md) for the tag syntax.

## Every gate needs a positive and a negative control

A stage gate that only ever ran on synthetic input, or that never had a
mis-wired variant to catch, constrains nothing. `ailm.control.ControlHarness`
and `GateVacuousError` exist to make a vacuous gate fail loudly instead of
passing silently. See [`docs/gate_design.md`](docs/gate_design.md).

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For anything
security-relevant, see [`SECURITY.md`](SECURITY.md) instead of opening a
public issue.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
