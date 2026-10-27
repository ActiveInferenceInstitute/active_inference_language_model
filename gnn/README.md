# AILM in GNN

This directory holds one **GeneralizedNotationNotation (GNN)** spec per shipped
AILM generative model, the deliberate mis-wirings each model admits, and the
contract governing them.

```
gnn/
  README.md  SPEC.md  AGENTS.md      documentation -- plain .md, never parsed
  models/        L<n>[<part>]_<slug>.gnn.md            the canonical level models
  mutations/     L<n>[<part>]_<slug>@<mutation>.gnn.md deliberate defects, must FAIL
  compositions/  X<n>_<slug>.gnn.md                    specs spanning several levels
```

## What GNN is

GNN is the Active Inference Institute's notation for writing an Active Inference
generative model down as structured Markdown: a fixed sequence of `##` sections
declaring the state space, the connections between variables, the
parameterization, the time structure, and — the part AILM cares most about — an
ontology block naming the *semantic role* of every matrix. Upstream ships a
25-step pipeline that renders a spec to PyMDP, RxInfer and ActiveInference.jl.

Upstream: <https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation>

## Why AILM carries specs

Not for documentation. For a guard.

The ontology block states each matrix's role declaratively:

```
A=RecognitionMatrix
B=TransitionMatrix
D=Prior
```

That is exactly the invariant AILM's worst shipped bug violated. Stage 1 put the
learned letter-bigram table in the emission `A` with an identity `B`. The failure
is quiet, not loud: observing a symbol yields a posterior over the symbol that
*preceded* it, and an identity `B` carries that backward-looking belief forward as
the next-symbol prediction. pymdp cannot see it — a bigram matrix is
column-stochastic and passes validation in either slot — and the two wirings have
*identical matrix shapes*.

AILM already had two guards on wiring orientation:

1. `src/ailm/verification/oracle.py` — an independent closed-form numpy HMM that
   never imports pymdp, so a disagreement in perplexity is evidence.
2. Measured comparison against a context-free baseline. Necessary but **not
   sufficient**: a G2P emission normalized over the wrong axis and then
   renormalized so pymdp still validates it cleared the baseline while being
   wrong.

The GNN spec plus `src/ailm/gnn/` is the **third** guard, and unlike the second
it is not merely necessary-but-insufficient: it pins each slot by declared name
and checks the declaration against the live pymdp object. A matrix in the wrong
slot becomes a named violation instead of a slightly different number.

## Upstream release

[`upstream.json`](upstream.json) pins the public
[GNN v5.0.1 release](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v5.0.1)
to commit `e54813b4197df36f7ae0054d5e79f3ad2d5db0ff`. This is an optional parser
interoperability dependency, separate from AILM's own strict reader and model.
Use an isolated checkout so ongoing upstream development cannot silently change
the parser used for release verification:

```bash
git clone --depth 1 --branch v5.0.1 https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation.git ../ailm-gnn-v5.0.1
export AILM_GNN_UPSTREAM="$(cd ../ailm-gnn-v5.0.1 && pwd)"
uv sync --locked --extra gnn-interop
.venv/bin/python scripts/verify_gnn_release.py
.venv/bin/python -m pytest tests/test_gnn_conformance.py -q --no-cov
```

The standalone verifier and upstream test fixture refuse a different commit,
retargeted tag, missing or altered source, and unexpected source files. The full
release runner forwards the explicit selection and records its identity check.
Changing the pin requires actual parser interoperability and mutation controls;
it changes no historical scientific result. AILM's Lean layer remains local and
dependency-free; FEP_Formal is not required by this integration.

## The naming grammar

```
<kind><level><part>_<slug>[@<mutation>]
```

| Field | Meaning |
|---|---|
| `kind` | `L` for a level model, `X` for a composition |
| `level` | Hierarchy position; for a composition, the lowest level spanned |
| `part` | Sub-model discriminator within a level (`a`, `b`, ...), or empty |
| `slug` | What the model is *of*, in `snake_case` |
| `mutation` | The named defect this spec injects; absent on a canonical model |

`L1_letter`, `L2b_grapheme_to_phoneme`, `L3_word@uniform_prior`,
`X1_letter_to_phoneme`.

The grammar exists so a spec's identity is **derivable rather than recorded**. A
hand-maintained index beside the specs would be a second source of truth, and the
copy is the one that drifts. Sorting by `(level, part, mutation)` gives hierarchy
order with each mutation immediately after its base model, so enumerating,
switching, composing and mutating become directory operations.

A name that does not parse is **refused, never defaulted** -- the same reason
`SPEC.md` rejects a symbolic dimension. An identity guessed from an unparseable
name would bind a conformance check to the wrong model.

```python
from ailm.gnn import discover_models, load_spec, parse_model_id, spec_path

discover_models()                    # {GnnModelId: Path}, in hierarchy order
load_spec("L3_word@frozen_position") # load a mutation that exists on disk
parse_model_id("L3_word@x").base     # -> L3_word
```

`tests/test_gnn_registry.py` enforces the layout **default-deny**: every
`.gnn.md` must parse, must sit in the directory its kind demands, and every
mutation's base must exist. A plain `.md` inside a spec directory is a failure
rather than a skip -- a spec that stops being discovered stops being checked, and
nothing else would notice.

## Mutations are specs, not fixtures

A negative control is a *model*, so it is written down as one. Each file under
`mutations/` is a complete spec that must **fail** conformance against the
correct live agent; a mutation that passes means the check does not constrain
what it claims to.

| Mutation | Targets |
|---|---|
| `L1g_letter_generation@habit_as_transition` | removes the action-indexed transition, leaving the generator unable to change symbol |
| `L1g_letter_generation@learned_emission` | moves learned bigram content into the identity emission slot |
| `L1g_letter_generation@no_action_set` | collapses the policy space to one action |
| `L3_word@frozen_position` | the position transition declared as the identity -- a model stuck on the first letter |
| `L3_word@drifting_word` | the word transition declared learned rather than the identity |
| `L3_word@transposed_factors` | the emission's two state axes declared in the wrong order |
| `L3_word@unsized_factor` | a factor left unsized, so nothing pins it |
| `L4_case@identity_emission` | the UPOS tag declared to *be* the case role |
| `L4_case@frozen_transition` | a case role that never changes along a sentence |
| `L4_case@uninformed_prior` | the sentence-initial distribution declared uniform |
| `L4_case@swapped_alphabets` | the emission indexed `[case, UPOS]` rather than `[UPOS, case]` |
| `L5_clause@identity_joint_transition` | replaces the type-specific case chains with a frozen joint state |
| `L5_clause@single_type` | removes one sentence-type component from the declared joint model |
| `L6_narrative@uniform_phase_prior` | replaces the opening-tercile point mass with a uniform prior |
| `L5w_clause_widened@identity_joint_transition` | the 20-state joint widened transition declared as the identity, preserving the alphabet but removing every learned within-type chain |
| `L5w_clause_widened@single_type` | removes one sentence-type component from the declared widened joint model |

## The shipped specs

| File | Level | Hidden factor | Observation modality | A | B | D |
|---|---|---|---|---|---|---|
| [`models/L1_letter.gnn.md`](models/L1_letter.gnn.md) | L1 letter | 30 letters | 30 letters | identity | Brown letter bigram | uniform |
| [`models/L1g_letter_generation.gnn.md`](models/L1g_letter_generation.gnn.md) | L1g controllable letter generation | 30 letters | 30 letters | identity | action-indexed deterministic transition | uniform; also declares policy prior `E` |
| [`models/L2a_phoneme_sequence.gnn.md`](models/L2a_phoneme_sequence.gnn.md) | L2A phoneme sequence | 39 phones | 39 phones | identity | CMUdict phone bigram | phone unigram |
| [`models/L2b_grapheme_to_phoneme.gnn.md`](models/L2b_grapheme_to_phoneme.gnn.md) | L2B grapheme-to-phoneme | 39 phones | **30 letters** | **learned 30x39 emission** | phone bigram, TRAIN split only | first-phone-of-word |
| [`models/L3_word.gnn.md`](models/L3_word.gnn.md) | L3 word | **word identity x position** | 27 letters + END | deterministic spelling | identity + `+1` shift | **learned word unigram** |
| [`models/L4_case.gnn.md`](models/L4_case.gnn.md) | L4 case | 8 case roles | 12 UPOS tags | **learned P(UPOS \| case)** | case-role transition | sentence-initial case |
| [`models/L5_clause.gnn.md`](models/L5_clause.gnn.md) | retained L5 case-only ablation | joint sentence type × case role | case role | deterministic case read-off | type-persistent blocks of fitted case chains | type prior × first-case-given-type |
| [`models/L5w_clause_widened.gnn.md`](models/L5w_clause_widened.gnn.md) | canonical L5 widened clause | sentence type × widened symbol (20 states) | case roles plus VERB and AUX (10 symbols) | deterministic symbol read-off | type-persistent blocks of fitted widened-symbol chains | type prior × first-symbol-given-type |
| [`models/L6_narrative.gnn.md`](models/L6_narrative.gnn.md) | L6 document position | document-position tercile | sentence type | fitted P(type \| tercile) | left-to-right tercile transition | opening-tercile point mass |

These are nine unmutated model configurations for a six-level linguistic stack:
L2 has two comprehension models, L1g is the separate generation configuration,
and Level 5 retains its case-only ablation beside the canonical widened model.
Stage 7 is a speech-boundary contract with no GNN Agent.

`ailm.parameters.GNN_SPEC_FILES` binds the canonical parameter table to the
widened Level 5 spec. The case-only ablation and the Level 1 generation variant
stay in `GNN_AUXILIARY_SPECS`; the parameter recorder also checks and reports the
case-only census separately, outside the canonical totals (decision 118).

The structural inventory does not turn every recorded outcome into a pass.
The canonical widened Stage 5 gate passes as a development-session
specification, without held-back replication; the retained case-only ablation
is non-passing and its paired privileged-label gain remains unresolved. The
privileged type-label arm is a receiver-sensitivity reference, not a hard bound
or evidence of equivalence. Stage 6's registered observed-position score has a
positive point estimate. Its document-clustered primary interval spans zero on
the development population and excludes zero on the held-back population under
decision 115; the family-corrected interval still spans zero. A
reversed-tercile falsifier can make mean loss worse while retaining a nominally
favorable sentence sign, so that sign result remains secondary. The separate
descriptive live-Agent score is a different sequential estimand. The L6→L5
intervention reads the fitted type distribution from L6 `A` and changes the
receiving clause Agent's joint `D`; L6 `C` stays flat.

The widened spec explicitly records `n_symbols=10` for the shared structural
emission builder. Its transition and prior resolve through the widened artifact;
the case-only artifact has a separate cache entry. `test_spec_agent.py` compares
each spec-built configuration with its shipped builder byte for byte, including
both clause arms. Shape conformance alone cannot establish this identity.
Changing or omitting that alphabet argument is rejected by the construction
registry's complete provenance key. This is a construction check; the independent
live conformance checker does not reconstruct the source estimator.

`models/L2b_grapheme_to_phoneme.gnn.md` is the interesting one. It is the first AILM level whose `A` is a
genuine emission rather than the identity, and its observation alphabet is a
*different* alphabet of a *different size* from its state space — 30 letters over
39 phones. That asymmetry is the whole content of Stage 2 Part B, and the spec
makes it explicit in both `StateSpaceBlock` and `ModelParameters`.

Learned matrices are **not inlined**. A 30x30 bigram table is a derived artifact,
not a specification; a copy inside the spec would be a second source of truth that
drifts. `InitialParameterization` instead declares the structural form (identity,
uniform, zeros) or names the generator and the vendored artifact that reproduce
the learned entries byte-identically:

- `scripts/generate_counts.py` -> `vendor/brown_counts.json`
- `scripts/generate_phoneme_counts.py` -> `vendor/cmudict_counts.json`
- `scripts/generate_word_counts.py` -> `vendor/brown_word_counts.json`
- `scripts/generate_case_counts.py` -> `vendor/ud_ewt_case_counts.json`
- `scripts/generate_clause_counts.py` -> `vendor/ud_ewt_clause_counts.json`
- `scripts/generate_narrative_counts.py` -> `vendor/ud_ewt_narrative_counts.json`

`models/L3_word.gnn.md` is the second interesting one, for a different reason. It
is the first AILM model with **more than one hidden factor** -- word identity and
position, with a single letter modality depending on both through
`A_dependencies` -- and the first at which the learned quantity sits in `D`
rather than `B`, because both of its transitions are structural constants. That
relocation moves the mis-wiring risk with it, which is why its mutations target
the prior rather than the transition.

For a model declaring `num_factors > 1` the single-factor rules **generalize
rather than relax**: one transition and one prior per factor, each bound by an
`_f<N>` suffix on its variable name, one recognition matrix per modality,
per-factor sizes as `num_hidden_states_f<N>`, and a modality's state axes checked
in factor order -- so transposing two factors of different sizes is a violation,
not a reshape.

## Running the conformance check

```bash
.venv/bin/python -m pytest tests/test_gnn_conformance.py -q
```

That builds each agent for real from the vendored artifacts, reads its shapes and
matrices off the live `pymdp.agent.Agent`, and asserts conformance for all nine
canonical specs. Coverage is default-deny: the suite requires its covered names
to equal the canonical files on disk, with no exclusion list. It also runs
positive controls — genuinely mis-wired agents, contradictory specs, altered
dimensions, roles and structural forms — each of which must be *rejected*, with
the test asserting on the reported violation text.

From Python:

```python
from ailm.gnn import assert_conforms, describe_shapes, load_spec
from ailm.levels.letter import build_letter_agent, build_letter_transition, load_brown_counts

agent = build_letter_agent(build_letter_transition(load_brown_counts()))
assert_conforms(load_spec("L1_letter"), describe_shapes(agent))
```

`describe_shapes` returns the typed `LiveConformanceInputs` record, which is passed
positionally — it is not a mapping, so unpacking it with `**` raises `TypeError`.

`src/ailm/gnn/` is stdlib plus numpy. It never imports the GNN pipeline, pymdp,
or jax — a checker sharing code with what it audits would inherit the flaw it
audits for.

The separate construction gate runs with:

```bash
.venv/bin/python -m pytest tests/test_spec_agent.py -q
```

`ailm.spec_agent.build_agent_from_spec` resolves each declared structural form
and each complete `Learned(...)` provenance through an explicit default-deny
registry, constructs a real Agent, and then checks conformance. The test above
requires each of the nine results to be bit-identical to the corresponding
hand-written builder in shape, dtype and bytes. This closes the spec→Agent path
without making the independent `ailm.gnn` checker import construction code.

## Upstream interoperability

`tests/test_gnn_conformance.py::TestUpstreamInteroperability` parses all nine
canonical specs with the real upstream Markdown parser and requires that both
readers extract the *same* model — same variables and dimensions, same ontology,
same `ModelParameters`, with the one documented `pi`→`π` normalization in L1g.
It skips cleanly when no checkout is present; point
`AILM_GNN_UPSTREAM` at one to enable it. "Expressed in GNN" is therefore
demonstrated, not asserted.

## Read next

- [`../docs/gnn_composition.md`](../docs/gnn_composition.md) — the long form:
  what each section contributes, how a spec becomes a running agent, what one
  step of inference computes and which matrix enters where, and — stated
  precisely, because it is easy to overclaim — in what sense the levels
  *compose*. They share alphabets and a section grammar; no message passes by
  default. Five explicit channels were measured independently, with mixed
  outcomes, and none forms a connected cascade.
- [`AGENTS.md`](AGENTS.md) — the contract. Read before editing any spec.
- [`SPEC.md`](SPEC.md) — exactly what the checker enforces, and exactly what it
  does not. Conformance is **not** validation of the learned numbers.
