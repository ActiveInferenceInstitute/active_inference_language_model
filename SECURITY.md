# Security Policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

Email **daniel@activeinference.institute** with:

- a description of the issue and its potential impact,
- steps to reproduce it (a minimal example is ideal), and
- the version or commit you tested against.

You should expect an acknowledgment within a few business days. This is a
research repository maintained by one person alongside teaching and research
duties, so please be patient with turnaround time; a real report is never
ignored.

## Scope

AILM is a research codebase (discrete probabilistic language models built on
`pymdp`) with no network service, no user authentication, and no production
deployment. Realistic concerns are things like:

- unsafe deserialization or path handling in the data-loading and
  vendoring scripts (`scripts/`, `src/ailm/corpus/`),
- supply-chain issues in a pinned dependency (see `pyproject.toml` and
  `uv.lock`), or
- a vulnerability in a vendored or archived artifact under `vendor/`.

Please do not use this channel for general bug reports (open a regular issue
instead — see [`CONTRIBUTING.md`](CONTRIBUTING.md)) or for questions about the
research method or its results (those are welcome as a normal issue or a
discussion).

## Disclosure

We ask for a reasonable amount of time to investigate and address a report
before any public disclosure. Credit will be given in the fix's changelog
entry unless you ask to remain anonymous.
