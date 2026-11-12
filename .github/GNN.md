# GNN conformance

`gnn conformance (pinned)` is job `gnn-interop` in `.github/workflows/ci.yml`
and reads `gnn/upstream.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`
(`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`). The job clones that tag, requires
its HEAD to equal the pinned commit, syncs the `gnn-interop` extra, and runs
`scripts/verify_gnn_release.py` followed by
`.venv/bin/python -m pytest tests/test_gnn_conformance.py -q --no-cov` under
`AILM_REQUIRE_GNN_INTEROP=1`, so a missing pin or parser fails instead of
skipping.

The suite parses the nine specs under `gnn/models/*.gnn.md`, binds each to a
live pymdp agent built from the vendored artifacts, and requires the declared
shapes and roles, and every declaration kind present, to be read and checked.
It also parses the same nine files with the pinned upstream Markdown parser and
compares the two readers, one recorded `π`/`pi` rename aside.
`scripts/verify_gnn_release.py` checks the checkout's `src/` files against
their Git blobs. It does not scan the rest of the repository for `*.gnn.md`.
