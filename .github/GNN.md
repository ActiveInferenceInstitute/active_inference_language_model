# GNN conformance

`gnn conformance (pinned)` is job `gnn-interop` in `.github/workflows/ci.yml`
and reads `gnn/upstream.json`, which pins
`ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
(`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`). The job materializes that
release's `src/` into the ignored `refs/gnn-upstream/checkout`, requires its HEAD
to equal the pinned commit, syncs the `gnn-interop` extra, and runs the `gnn`
arm: `python -m gnn.importers check --repo .` with the checkout's `src` on
`PYTHONPATH`, then `scripts/verify_gnn_release.py`, then
`.venv/bin/python -m pytest tests/test_gnn_conformance.py -q --no-cov` under
`AILM_REQUIRE_GNN_INTEROP=1`, so a missing pin or parser fails instead of
skipping.

The reader that parses the pin, the pointer and every `*.gnn.md` is
`gnn.importers` from the pinned release — the same module
`docs/workflows/gnn-fep.md` in `Generalized_Notation_Notation` binds every
importer to. It shipped in `v5.1.0`, so this repository's interim pin reader,
which served `v5.0.3` (`8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`), was retired in
the same change that advanced the pin.

The suite parses the nine specs under `gnn/models/*.gnn.md`, binds each to a
live pymdp agent built from the vendored artifacts, and requires the declared
shapes and roles, and every declaration kind present, to be read and checked.
It also parses the same nine files with the pinned upstream Markdown parser and
compares the two readers, one recorded `π`/`pi` rename aside.
`scripts/verify_gnn_release.py` checks the checkout's `src/` files against
their Git blobs. It does not scan the rest of the repository for `*.gnn.md`.
