# AGENTS.md — src/ailm/gnn/

Operating rules for the GNN conformance package. The quick reference is
[`README.md`](README.md); this file is the ownership boundary, the invariants, and
what breaks when one of them is relaxed.

## Overview

This package parses the GeneralizedNotationNotation specs that live under
[`../../../gnn/`](../../../gnn/README.md) and checks them against the pymdp agents
AILM builds. Two families of check, kept on opposite sides of a deliberate seam:
declaration checks that need only the spec, and live checks that need the built
object. `conformance.py` runs both and reports what disagreed.

The spec does not generate the builder. Both are written by hand — that
independence is the only reason running the check is worth anything.

## Ownership boundary

| Artifact | Owner |
| --- | --- |
| The spec files themselves | [`../../../gnn/models/`](../../../gnn/models) and `../../../gnn/mutations/`; this package only reads them |
| The enforced subset and its non-goals | [`../../../gnn/SPEC.md`](../../../gnn/SPEC.md) |
| The naming grammar for spec identifiers | `registry.py` (`SPEC_KINDS`, `SPEC_SUFFIX`, `parse_model_id`) |
| The public API surface | `__init__.py`'s `__all__` — every other module's exports reach callers through it |
| Composition semantics in the wider project | [`../../../docs/gnn_composition.md`](../../../docs/gnn_composition.md) |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **No `pymdp`, no `jax` import — not even lazily, not inside a function** | A checker coupled to the thing it audits inherits the flaw it audits for. Live arrays arrive via `numpy.asarray` over public attributes |
| **No import of the upstream GNN pipeline** | Same reason, plus it would drag an unrelated dependency tree into a stdlib+numpy module |
| **`InitialParameterization` accepts structural forms only** | `Identity(n)`, `Uniform(n)`, `Zeros(n)`, `Learned(k=v, ...)`. An inlined matrix literal duplicates a vendored artifact and will drift away from it |
| **An unrecognized `##` section is an error** | Treating it as an extension would silently drop a whole block on a typo |
| **Declared variables and ontology annotations are in bijection** | A declared-but-unannotated slot is a slot whose semantic role nothing pins — which is exactly the historical bug class |
| **Malformed input raises `GnnSpecError` naming the line** | Nothing returns a partial spec |
| **Conformance never certifies a probability** | It reports structure. The numeric guard is `ailm.verification.oracle` plus the gate tests |
| **Import order stays acyclic** | `types` → (`parsing`, `shapes`) → `declarations` → `live` → (`conformance`, `registry`) |

## File organization

| File | Concern |
| --- | --- |
| `__init__.py` | Re-export surface plus the package's design rationale |
| `types.py` | Section names, regexes, dataclasses, and the two error types |
| `parsing.py` | `parse_gnn_spec`, `parse_gnn_string`, `parse_learned_arguments` |
| `shapes.py` | `describe_shapes`, `live_inputs`, and the batch-axis normalization |
| `declarations.py` | Spec-only coherence: factor sizes, connections, inference block, combination rules |
| `live.py` | Built-object checks: array shapes, stochastic orientation, declared precision |
| `conformance.py` | `check_conformance`, `assert_conforms` — collection and raising, nothing else |
| `registry.py` | `spec_dir`, `spec_path`, `parse_model_id`, `discover_models`, `load_spec` |

Underscore-prefixed helpers are re-exported from `__init__.py` because
`tests/test_gnn_conformance.py` exercises individual checks in isolation. They are
internal and stay out of `__all__`; do not treat them as a stable API.

## Usage

```python
from ailm.gnn import assert_conforms, describe_shapes, load_spec
from ailm.levels.letter import build_letter_agent, build_letter_transition, load_brown_counts

agent = build_letter_agent(build_letter_transition(load_brown_counts()))
assert_conforms(load_spec("L1_letter"), describe_shapes(agent))
```

`check_conformance` returns the report instead of raising, which is what you want
when you are looking at several disagreements at once:

```python
from ailm.gnn import check_conformance, describe_shapes, load_spec

report = check_conformance(load_spec("L1_letter"), describe_shapes(agent))
print(report)  # empty when the model and its declaration agree
```

## Local gotchas

**`spec_dir()` uses installed `_gnn/` resources or walks up in a source checkout.**
Canonical specs ship in the wheel; deliberate mutation fixtures remain in the
repository. The historical fixed ancestor hop count was wrong
the moment this module moved one level deeper during the package split, and it
failed in the least useful way — `src/gnn/models/...` is a plausible-looking path,
so every caller reported "no such file" for a spec sitting where it always had been.

**Shapes alone cannot catch the historical A/B swap.** At the letter level the
correct and the swapped wiring have identical dimensions. The structural forms in
`InitialParameterization` are what discriminate. Never weaken that block to a
dimension check.

**A mutation spec is a first-class artifact, not a fixture.** Files named
`<model>@<mutation>.gnn.md` live in `../../../gnn/mutations/` and are discovered by
the same registry. Adding one to the wrong subdirectory is refused rather than
inferred.

## Testing

Run from the project root, with the project's own interpreter — never `uv run`
inside this sidecar (see [`../../../docs/quickstart.md`](../../../docs/guides/quickstart.md)).

```bash
.venv/bin/python -m pytest tests/test_gnn_conformance.py -q --no-cov
.venv/bin/python -m pytest tests/test_gnn_registry.py -q --no-cov
.venv/bin/python -m pytest tests/test_gnn_edge_paths.py -q --no-cov

# The independence invariant itself
.venv/bin/python -c "
import sys, ailm.gnn
leaked = sorted({m.split('.')[0] for m in sys.modules} & {'pymdp', 'jax', 'jaxlib'})
assert not leaked, f'ailm.gnn pulled in {leaked} — it is no longer independent'
print('ailm.gnn is independent')
"
```

## Best practices

- Add a check to `declarations.py` when it can be decided from the spec alone;
  add it to `live.py` only when it genuinely needs the built object.
- Name the offending variable or connection in every failure message. A
  conformance report that says only "mismatch" costs more than it saves.
- When a check would need a probability value to decide, it belongs in the oracle
  cross-check or a gate test, not here.

## See Also

- [`README.md`](README.md) — quick reference for this package
- [`../../../gnn/SPEC.md`](../../../gnn/SPEC.md) — the enforced subset and non-goals
- [`../../../gnn/AGENTS.md`](../../../gnn/AGENTS.md) — rules for the spec files themselves
- [`../verification/AGENTS.md`](../verification/AGENTS.md) — the independent numeric instrument
- [`../../../docs/architecture.md`](../../../docs/architecture.md) — the package map and import contract
