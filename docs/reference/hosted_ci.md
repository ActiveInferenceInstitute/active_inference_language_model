# Hosted CI

What `.github/workflows/ci.yml` runs, which command owns each check row, and
what the hosted jobs deliberately do not do. Every row here is a profile of
[`../../scripts/verify_release.py`](../../scripts/verify_release.py), so the
command in a job and the command in [`../guides/quickstart.md`](../guides/quickstart.md)
are the same string; the workflow never restates a step the runner already
defines.

## The gate is local

GitHub Actions is billing-locked for this organization. A hosted run is a
convenience, not the gate: the gate is the same arm run on a prepared
workstation, and [`_generated/gate_arms.md`](../_generated/gate_arms.md) records
the measured wall time of each arm on the recording host. A green hosted job that
no local run reproduced is not evidence.

## Check rows

| Check row | Job key | Arm (profile) | What it owns |
| --- | --- | --- | --- |
| `lint` | `lint` | `lint` | `ruff format --check` and `ruff check` over `src`, `scripts`, `tests` |
| `types` | `types` | `types` | `mypy` over `src/ailm` and the typed release scripts |
| `docs` | `docs` | `docs` | The documentation index and its relative links, then the hydrated mirror |
| `quick` | `quick` | `quick` | Locked environment, generated web artifacts, the portable test suite, and the bounded gate audit |
| `measurement gate` | `measurement` | `measurement` | The `measurement_gate` files, serially, under their own watchdog |
| `contract gate` | `contract` | `contract` | The `contract_gate` files (wheel builds, real-socket retries), serially |
| `gnn conformance (pinned)` | `gnn-interop` | `gnn` | The pinned upstream GNN release and `tests/test_gnn_conformance.py` under `AILM_REQUIRE_GNN_INTEROP=1` |

Each arm runs in exactly one job, and no job repeats another's command: format,
lint and typing are the `lint` and `types` rows only, the documentation tests are
the `docs` row only, and the `quick`, `measurement` and `contract` profiles carry
no static step. `full` does re-run them, because publication acceptance is one
receipt over one tree rather than a set of independent rows. The environment all of them need is defined once, in
[`../../.github/actions/setup-ailm/action.yml`](../../.github/actions/setup-ailm/action.yml):
the pinned uv and Python, the pinned uv on the release verifier's search path, the
optional Homebrew tools an arm shells out to, and the locked sync. Only the `quick`
row installs Pandoc and ripgrep, because only its test selection runs the
destination proofs and the formal vacuity control that need them.

## Pinned actions

The workflow and the composite action pin every third-party action to a full
commit, and `tests/test_release_orchestration.py` refuses a floating tag, an
unapproved action, a checkout that persists credentials, a job-level permission
override, or a `${{ github.event.* }}` expression inside a `run:` block.

| Action | Revision |
| --- | --- |
| `actions/checkout` | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| `astral-sh/setup-uv` | `68559e99cc807c1eb7e9238fd43a15bba460eec2` |
| `actions/upload-artifact` | `cf430e030ddbb5b0abf93d22962f4752f3646cd9` |
| `./.github/actions/setup-ailm` | the workflow's own commit; a local action has no separate revision |

The organization composite
`ActiveInferenceInstitute/.github/.github/actions/setup-python-uv@7159e67315330aac8e414aebae0ad1e999a41448`
is **not** used here. This repository's setup lives in the local composite action
above, which additionally exposes the pinned uv to the release verifier and takes
the extra and tool inputs each arm needs. Recording that fact is the point: a
reader comparing repositories should not assume a shared setup action is in play.

## GNN conformance

The canonical pin is [`../../gnn/upstream.json`](../../gnn/upstream.json), schema
`gnn-importer-pin/v1`: repository
`ActiveInferenceInstitute/Generalized_Notation_Notation`, tag `v5.0.3`, commit
`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`. The `gnn` arm materializes that tag
with `scripts/fetch_gnn_upstream.py` into the Git-ignored
`refs/gnn-upstream/v5.0.3`, refuses a checkout whose HEAD, tag ref or source bytes
disagree with the pin, and then runs `scripts/verify_gnn_release.py` followed by
`tests/test_gnn_conformance.py`.

[`../../.github/GNN.md`](../../.github/GNN.md) is the pointer that records what the
conformance step parses: the nine specs under `gnn/models/`, the declarations it
requires to be read and checked, and the one recorded `π`/`pi` rename that
separates AILM's reader from the upstream Markdown parser.

## What the hosted workflow does not run

The `full` profile is publication acceptance, and it is not a hosted arm: it
consumes the ignored machine-local external-call cache, rechecks a sample against
a reachable Ollama server serving every declared probe, requires a clean
`docxology/template` checkout routed to this sidecar, and needs Lean, TeX, `qpdf`,
Pandoc and the template validator. A hosted job without those inputs can only fail
or skip, and neither result is publication evidence. Run it from a prepared
release workstation as [`../method/release_checklist.md`](../method/release_checklist.md)
describes.
