# SPEC.md — the AILM-side subset of GNN, and its limits

What `src/ailm/gnn/` reads, what it enforces against a live pymdp agent, and —
in equal detail — what it does **not** check. The second list matters more than
the first: mistaking conformance for validation would make this directory a
liability rather than a guard.

---

## 1. Sections

`parse_gnn_spec` accepts exactly the canonical thirteen, in canonical order:

`GNNSection`, `GNNVersionAndFlags`, `ModelName`, `ModelAnnotation`,
`StateSpaceBlock`, `Connections`, `InitialParameterization`, `Equations`,
`Time`, `ActInfOntologyAnnotation`, `ModelParameters`, `Footer`, `Signature`.

- Anything else is a **parse error**, not an "extension". Upstream stores an
  unknown section as an extension; AILM refuses, because a misspelled section name
  would otherwise silently drop the whole block — `## ModelParamaters` would take
  the dimensions with it and the conformance check would go quiet.
- A duplicate section is a parse error.
- `ModelName`, `StateSpaceBlock` and `ActInfOntologyAnnotation` are **required**.
  Without the ontology block there is nothing to pin, which is the whole point.
- Every section's raw text is retained on `GnnSpec.sections`. Nothing the checker
  declines to interpret is thrown away.

## 2. What is parsed

| Section | Parsed into | Enforced at parse time |
|---|---|---|
| `ModelName` | `spec.model_name` | non-empty |
| `StateSpaceBlock` | `spec.variables`: `name -> (dims, dtype)` | `name[d,...,type=t]` form; positive integer dims only; no duplicate names; no symbolic dimensions |
| `Connections` | `spec.connections` | `>` (directed) or `-` (undirected); both endpoints declared |
| `InitialParameterization` | `spec.parameterization`: `name -> (form, args)` | only `Identity(n)`, `Uniform(n)`, `Zeros(n)`, `Learned(k=v, ...)`; the variable must be declared; `Learned` must carry arguments |
| `ActInfOntologyAnnotation` | `spec.ontology`: `name -> term` | `Variable=Term`; in **bijection** with `spec.variables` — nothing declared may be un-annotated and nothing annotated may be undeclared |
| `ModelParameters` | `spec.parameters` | `key: value` or `key = value`; no duplicate keys; ints/floats coerced |
| `GNNSection`, `GNNVersionAndFlags`, `ModelAnnotation`, `Equations`, `Time`, `Footer`, `Signature` | raw text only | nothing |

Symbolic dimensions are rejected on purpose. Upstream defaults an unparseable
dimension to `1`; a checker cannot bind that to a real matrix, and silently
comparing against `1` would be worse than refusing.

Rejecting inlined matrix literals is also on purpose. Learned entries are a
derived artifact reproduced byte-identically by a generator. A copy of them inside
the spec is a second source of truth, and the copy is the one that will drift.

## 3. What conformance enforces against a LIVE agent

`describe_shapes(agent)` reads the public `A`/`B`/`C`/`D` lists, `E`, policy
precision, and inference configuration off a real
`pymdp.agent.Agent`. `check_conformance` then returns **every** violation
(`assert_conforms` raises with the complete list — failing on the first problem
only would force a one-at-a-time repair loop):

1. `ModelParameters` carries positive-integer `num_hidden_states` and `num_obs`.
   Factorized models instead declare `num_hidden_states_f<N>` for every factor.
2. Exactly one variable is annotated `RecognitionMatrix`, exactly one
   `TransitionMatrix`, exactly one `Prior`. Two variables claiming a role pin
   nothing, so ambiguity is a violation. Factorized models bind a transition
   and prior to each factor through `_f<N>` suffixes; modality suffixes bind
   recognition matrices. L3's emission axes follow the declared factor order.
3. A shape was **supplied** for every role-carrying variable the spec declares.
   An unverified slot is reported, never silently skipped.
4. Declared dims equal live dims, after shape normalization (§4).
5. Spec-internal agreement: the `RecognitionMatrix` is declared
   `num_obs` x `num_hidden_states`, the `TransitionMatrix`
   `num_hidden_states` square, the `Prior` a `num_hidden_states` vector, the
   `LogPreferenceVector` a `num_obs` vector. A spec contradicting itself fails
   even against a correct agent.
6. The live `TransitionMatrix` is square in its state dims, and that dimension
   equals `num_hidden_states`.
7. The live `RecognitionMatrix` is `num_obs` x `num_hidden_states`. **This is
   what rejects a transposed emission**, and it is why L2B's 30-by-39 asymmetry is
   the easiest of the three levels to pin.
8. `num_factors` / `num_modalities` agree with `len(agent.B)` / `len(agent.A)`.
9. Each `InitialParameterization` structural form holds for the supplied live array:
   `Identity(n)` is the identity, `Uniform(n)` is `1/n` everywhere, `Zeros(n)` is
   zero, and a `Learned(...)` square matrix is **not** the identity. Non-finite
   entries are rejected before these comparisons. A shape-only description
   cannot establish any of these value-dependent properties.
10. Declared `Learned` normalization constrains the axis-0 or whole-array sum;
    an unqualified Laplace estimator requires strictly positive entries.
11. Belief, observation and policy vector dimensions and dtypes agree with their
    roles; the `Time` block names the declared time index; recognized combination
    rules are checked against the executable rule vocabulary.
12. A nonempty connection graph leaves no non-time variable orphaned, directs
    priors into hidden states, and agrees with `num_controls` on whether a policy
    reaches a transition. The hidden-state neighbors of the emission agree with
    its live state-axis count.
13. Controlled models retain the non-singleton action axis of `B`, size `E` and
    the policy vector against `num_controls`, and compare supplied inference
    flags, policy count and live controls. Disabling utility or state information
    gain is a violation. Declared `gamma` must be positive and finite and agree
    with the supplied finite live policy precision.

### Why check 9 exists — the one place shapes are provably insufficient

On the letter and phoneme-sequence levels the correct wiring and the historical
anti-informative swap have **identical shapes**: `A` is `(1, 30, 30)` and `B` is
`(1, 30, 30, 1)` whether the bigram table sits in the emission or the transition.
A purely dimensional checker passes the swap. `tests/test_gnn_conformance.py`
contains `test_shape_only_check_cannot_catch_the_swap`, which asserts exactly
that, so this document cannot go stale silently.

Check 9 closes it by treating identity/uniformity as **structure**, which they
are: an identity emission asserts that the hidden state is observed directly, a
uniform prior asserts that the model starts uninformed, and a matrix declared
learned-from-a-corpus that turns out to be the identity carries no learned
content at all. None of those is a claim about a learned probability.

Tolerance is `1e-5`, because JAX runs float32 and exact structural anchors land
within about `2e-7` relative. Do not widen it to absorb a failure.

## 4. Shape normalization, and what it discards

Two reductions, applied symmetrically:

- A leading axis of size 1 is stripped from a **live** shape. pymdp carries a
  batch axis that GNN has no notion of: `A` is `(1, n_obs, n_states)` in the agent
  and `A[n_obs, n_states]` in the spec.
- Trailing axes of size 1 are dropped from **both** sides, so the spec's
  `B[39,39,1]` matches the agent's `(1, 39, 39, 1)` and the spec's column vector
  `D[39,1]` matches the agent's `(1, 39)`.

Non-singleton action axes survive this reduction. L1g is controlled, and the
checker explicitly compares its transition action axis, policy prior and policy
vector with `num_controls`. Supplied live control counts and policy counts are
checked separately. Dropping a singleton axis alone does not prove an action
space is correct; these additional checks provide that guard. Arbitrary
multifactor control layouts are not licensed by the scalar `num_controls`
contract.

## 5. What conformance does NOT check

Read this list as seriously as §3.

- **Matrix values.** Beyond the structural forms of check 9 above, no learned probability is
  examined. The bigram entries, the emission entries and the prior weights are
  never compared to anything. A spec-conformant model can be numerically wrong in
  every entry.
- **Unspecified distribution properties.** Normalization and Laplace positivity
  are checked when declared in `Learned(...)`; this is not universal stochastic
  validation. A renormalized wrong matrix can satisfy those declarations while
  encoding the wrong conditional distribution.
- **Equations.** The `Equations` block is retained verbatim and never interpreted.
  It documents the recursion; it does not constrain the implementation.
- **Temporal execution.** The time-index declaration is checked, but the
  `Dynamic`/`Discrete`/`ModelTimeHorizon` prose does not execute a schedule.
  The prior-reset policy each spec describes in its `Time` block —
  per-text for L1, per-word for L2A and L2B — is enforced by the level's own tests,
  not here.
- **Complete graph semantics.** The checks in §3 cover only the named graph
  invariants, not every possible edge or its runtime use. They do not establish
  cross-agent messaging, message timing, shared-variable consistency or a joint
  posterior. Those require an executable composition contract and numerical
  controls outside this independent checker.
- **Provenance strings, beyond existence.** `Learned(generator=..., artifact=...)`
  is parsed, required to be non-empty, and
  `test_learned_provenance_paths_exist_on_disk` asserts the named files are
  actually present — a spec citing a deleted generator would otherwise read as
  reproducible. Whether the generator in fact rebuilds the artifact
  byte-identically is checked by `tests/test_provenance_checksums.py`, not here,
  and nothing here confirms that the artifact is the one the live matrix was
  built from.
- **`ModelAnnotation` prose,** including its `claim:` labels. Those are backed by
  `data/claims.json` and enforced by `tests/test_claims_discipline.py`.
- **Whether the model is any good.** Nothing here measures perplexity, accuracy,
  or a margin over a baseline. Conformance is orthogonal to the stage gates.
- **Omitted live inputs.** Some checks, including controlled inference flags and
  structural values, only run when those live inputs are supplied. Use
  `describe_shapes(agent)` to carry the complete supported public attributes;
  a hand-built shape-only record provides narrower evidence.

## 6. Where the real evidence lives

| Question | Answered by |
|---|---|
| Is each matrix in the slot its role names? | this checker |
| Do the dimensions match the code? | this checker |
| Are the learned probabilities right? | `src/ailm/verification/oracle.py` cross-check, `tests/test_oracle.py` |
| Does the model beat a non-crippled baseline, significantly? | `tests/test_stage1_gate.py`, `tests/test_stage2_gate.py` |
| Can the gate fail at all? | `src/ailm/control.py` (`GateVacuousError`) and the negative controls in the gate tests |
| Do the vendored artifacts reproduce? | `tests/test_provenance_checksums.py` |
| Is a number in prose backed by a measured run? | `data/claims.json`, `tests/test_claims_discipline.py` |

Three independent guards now cover wiring orientation: the numpy oracle
(numerical, shares no code with pymdp), the measured baseline comparison
(necessary but not sufficient), and this conformance layer (declarative, pins the
slot by name). None of the three subsumes the others, and none of them is a
substitute for the gates.

## 7. Upstream interoperability

`TestUpstreamInteroperability` in `tests/test_gnn_conformance.py` imports the
real upstream Markdown parser from `AILM_GNN_UPSTREAM`. It checks every canonical
model for parse success without errors or warnings and compares model names,
variable dimensions, ontology roles and model parameters. The explicit `pi` to
`π` rename is the only accepted variable-name difference. The runtime package
never imports that parser or its pipeline.

An explicit `AILM_GNN_UPSTREAM` overrides discovery. Otherwise the tests first
look under the checkout's sibling namespace at
`PROJECT_ROOT.parents[2] / ActiveInferenceInstitute/Public/Generalized_Notation_Notation`,
then the legacy home-directory checkout, then the temporary development probe.
Only a candidate containing `src/gnn/parsers/markdown_parser.py` as a file is
usable. This ordering selects the canonical GitVault checkout before an older
home-directory copy; its current Git identity still needs verification.

Run the checks against a checkout whose Git identity has been recorded:

```bash
AILM_GNN_UPSTREAM="$GNN_CHECKOUT" .venv/bin/python -m pytest \
  tests/test_gnn_conformance.py::TestUpstreamInteroperability -q --no-cov
```

An absent checkout skips in ordinary development and fails when
`AILM_RELEASE_MANDATORY_ENVIRONMENT=1`. Parser interoperability does not certify
upstream rendering, execution, or equivalence of generated Agents; AILM's
spec-built Agent identity tests are a separate contract.
