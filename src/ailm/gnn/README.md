# ailm/gnn/ — the declarative-spec conformance checker

Binds AILM's built pymdp models to the GeneralizedNotationNotation (GNN) specs under
[`../../../gnn/models/`](../../../gnn/models). It is the third independent guard on
matrix-slot orientation, alongside the numpy oracle in
[`../verification/`](../verification/README.md) and the measured baseline comparisons.

> **This package never imports `pymdp` or `jax`.** It reads a live agent through
> `numpy.asarray` and the public `A`/`B`/`C`/`D` attributes only. A checker that
> shared code with the thing it audits would inherit the flaw it audits for.

## Quick start

```python
from ailm.gnn import assert_conforms, describe_shapes, discover_models, load_spec
from ailm.levels.letter import build_letter_agent, build_letter_transition, load_brown_counts

agent = build_letter_agent(build_letter_transition(load_brown_counts()))
assert_conforms(load_spec("L1_letter"), describe_shapes(agent))

for model_id, path in discover_models().items():
    print(model_id, path)
```

`load_spec`, `spec_path`, and `discover_models` take no root argument: `spec_dir()`
walks up from the package until it finds the checkout containing both `gnn/` and
`src/`.

## Key features

- **Structural forms, not learned values.** `InitialParameterization` declares
  `Identity(n)`, `Uniform(n)`, `Zeros(n)`, or `Learned(k=v, ...)`. An inlined matrix
  literal is rejected: it would be a copy of a derived artifact, not a specification
  of it.
- **Catches the A/B swap that shapes cannot.** At the letter level the correct and
  the transposed wiring have identical dimensions, so only the declared role and
  structural form discriminate.
- **Stricter than upstream GNN in three named places** — unknown `##` sections are
  errors, matrix literals are refused, and declared variables must be in bijection
  with ontology annotations.
- **Structure only.** Conformance never certifies that a probability is *right*;
  the oracle cross-check and the gate tests do that.

## Files

| File | What it owns |
| --- | --- |
| `__init__.py` | Re-exports the whole public API; the design rationale lives in its docstring |
| `types.py` | `GnnSpec`, `GnnConnection`, `LiveConformanceInputs`, `GnnSpecError`, `GnnConformanceError`, `GNN_SECTIONS`, `MATRIX_ROLES` |
| `parsing.py` | GNN source text to a `GnnSpec` — no live object anywhere in reach |
| `shapes.py` | `describe_shapes` / `live_inputs`: a built agent's canonical dimensions as a typed `LiveConformanceInputs`, without judging them |
| `declarations.py` | Checks a spec against itself; no agent needed |
| `live.py` | Checks a built agent against the spec that declares it |
| `conformance.py` | `check_conformance` / `assert_conforms` — orchestration only |
| `registry.py` | `discover_models`, `load_spec`, `parse_model_id`, `spec_path` over `gnn/models/` |

Import order is linear with no cycles: `types` → (`parsing`, `shapes`) →
`declarations` → `live` → (`conformance`, `registry`).

## Common commands

Run from the project root with the project's own interpreter.

```bash
# The conformance checks themselves
.venv/bin/python -m pytest tests/test_gnn_conformance.py -q --no-cov

# Registry naming grammar and spec discovery
.venv/bin/python -m pytest tests/test_gnn_registry.py -q --no-cov

# Error paths that a happy-path run never reaches
.venv/bin/python -m pytest tests/test_gnn_edge_paths.py -q --no-cov
```

## More information

See [`AGENTS.md`](AGENTS.md) for the ownership boundary and the invariants.
The enforced subset and the explicit non-goals are written out in
[`../../../gnn/SPEC.md`](../../../gnn/SPEC.md); composition and conformance in the
wider project are described in
[`../../../docs/gnn_composition.md`](../../../docs/gnn_composition.md).
