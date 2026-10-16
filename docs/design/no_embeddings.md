# No Embeddings: the Categorical Boundary

## What standard DisCoCat provides

DisCoCat maps a compact pregroup grammar into a compact-closed linear category,
usually `FVect` or `FinHilb`. Word meanings are tensors; grammatical cups and
caps become contractions; the monoidal functor preserves the equations that make
the reduction compositional.

Vector spaces do not by themselves imply learned embeddings. A hand-specified
linear semantics is possible. AILM's narrower invariant is operational: none of
its pymdp model paths uses a learned semantic vector, transformer or neural
network, and the repository does not implement a DisCoCat meaning functor.
External neural-model benchmark clients are separate reference instruments;
they do not participate in AILM inference.

## What AILM actually implements

AILM uses finite state spaces and conditional-probability arrays. Under the
project's convention, each source state occupies a column:

- `A[m][:, s]` is `P(observation | state=s)` and each such column sums to one;
- `B[f][:, s, action]` is `P(next_state | current_state=s, action)` and each such
  column sums to one;
- ordinary matrix multiplication composes compatible stochastic kernels.

These kernels are morphisms in `FinStoch` (up to the explicit column-oriented
array convention). `FinStoch` is a symmetric monoidal Markov category, not a
compact-closed category. In particular, its causal normalization and discard
structure do not supply the nontrivial cups and caps required to interpret a
pregroup reduction while satisfying the snake equations.

Consequently, AILM does **not** claim a functor `Preg → FinStoch`, does not equate
pregroup contraction with pymdp policy selection, and does not claim that a
product of `B` matrices preserves DisCoCat semantics. Those statements appeared
in an earlier version of this document and were incorrect.

## Relationship to the cited case paper

*Compositional Approaches to Linguistic Case for Cognitive Modeling* develops its
DisCoCat meaning functor into `FVect`/`FinHilb`; it does not define a `FinStoch`
functor. AILM draws on that work for case-typed categorical motivation and its
belief-update discussion. The executable model uses only the independently
specified pymdp state spaces, likelihoods, transitions, priors, and explicitly
tested receiver interventions and [connected conditional messages](connected_runtime.md).
That runtime composition does not supply a categorical meaning functor.

No reported AILM result depends on a categorical semantic functor. Calling the
implemented kernel products “stochastic composition” is accurate; calling them
“DisCoCat” is not.

## What a future bridge would have to prove

There are at least two coherent research directions, neither implemented here:

- define a compact-closed semantics in an ambient matrix category such as
  non-negative matrices, where cups and caps need not be normalized stochastic
  maps, then specify and test the normalization boundary into pymdp kernels; or
- stay inside a Markov category and formulate a different stochastic language
  semantics without claiming that it preserves pregroup compact closure.

Either route needs an explicit object map, morphism map, tensorator, cup/cap
interpretation where applicable, and executable identity tests before the
repository can call it a functorial language semantics.

## Sources

- Coecke, Sadrzadeh, and Clark (2010), *Mathematical Foundations for a
  Compositional Distributional Model of Meaning* — the standard DisCoCat
  construction.
- Friedman (2026), Zenodo-record title *Compositional Approaches to Linguistic
  Case for Cognitive Modeling*, exact record `10.5281/zenodo.19695260` — the
  deposited PDF instead prints *Cognitive Diagrams: Reviewing Categorical
  Accounts of Linguistic Case* (2026-04-22) and no version. Its meaning functor
  is in the `FVect`/`FinHilb` setting; exact artifact identity is pinned in
  `data/reference_sources.json`.
- Fong and Spivak (2019), *An Invitation to Applied Category Theory* — stochastic
  maps and applied monoidal categories.
