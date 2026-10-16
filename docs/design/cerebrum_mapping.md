# CEREBRUM source tuple and AILM's pymdp heuristic

This document replaces the retired claim that CEREBRUM Figure 13 supplies a
pymdp crosswalk. The exact deposited PDF does not contain that crosswalk.
`data/reference_sources.json` pins the Zenodo record identity, deposited-file
digest and extent, printed identity, and every physical-page locator used here.

## What the exact deposited PDF says

The Zenodo record labels `{{CEREBRUM_PDF_FILENAME}}` as version {{CEREBRUM_ZENODO_VERSION}},
published {{CEREBRUM_ZENODO_DATE}}. The title page of those exact bytes prints Version {{CEREBRUM_PRINTED_VERSION}}
({{CEREBRUM_PRINTED_DATE}}). Both are source facts; neither is silently promoted over the
other.

Physical page {{CEREBRUM_ACTIVE_INFERENCE_FIGURE_PAGES}} contains the caption "[Figure {{CEREBRUM_ACTIVE_INFERENCE_FIGURE}}]({{CEREBRUM_ACTIVE_INFERENCE_FIGURE_URL}}): Active Inference Integration
Framework." It contains no diagram, equation, symbol table, or component
crosswalk. A caption cannot be used as evidence for the mapping formerly
attributed to it.

The actual symbol inventory appears in [Supplement {{CEREBRUM_POMDP_SUPPLEMENT}}]({{CEREBRUM_POMDP_SUPPLEMENT_URL}}) on physical page {{CEREBRUM_POMDP_SUPPLEMENT_PAGES}}:

| Source symbol | Meaning in CEREBRUM |
|---|---|
| `S` | hidden state space |
| `A` | action space |
| `T` | state transition dynamics |
| `Ω` | observation space |
| `O` | likelihood mapping from states to observations |
| `F` | free-energy functional |
| `π` | precision parameters |

Those meanings matter because the same letters mean different things in pymdp.
CEREBRUM `A` is not pymdp likelihood `A`; CEREBRUM `F` is variational free
energy rather than pymdp expected-free-energy `G`; and CEREBRUM `π` is a set of
precision parameters rather than pymdp policy-prior `E`.

## The project-owned heuristic

`AILM_PYMDP_HEURISTIC_CROSSWALK` in `src/ailm/cases.py` records an engineering
interpretation, not a quotation or derivation from a CEREBRUM figure:

| CEREBRUM element | AILM/pymdp interpretation | Strength of relation |
|---|---|---|
| `S` | hidden-state factor domains | shared domain concept, not a matrix identity |
| `A` | action and control-factor domains | shared domain concept; explicitly not pymdp `A` |
| `T` | pymdp transition arrays `B` | structural correspondence |
| `Ω` | observation-modality domains | shared domain concept, not a matrix identity |
| `O` | pymdp likelihood arrays `A` | structural correspondence |
| `F` | the VFE objective used in state inference | conceptual role; explicitly not pymdp `G` |
| `π` | precision hyperparameters | conceptual role; explicitly not pymdp `E` |

pymdp preference `C`, initial prior `D`, and policy habit `E` have no direct
analogue in the CEREBRUM [Supplement {{CEREBRUM_POMDP_SUPPLEMENT}}]({{CEREBRUM_POMDP_SUPPLEMENT_URL}}) tuple. They are AILM/pymdp model
components and must not be reverse-attributed to the source.

## Alternatives considered

1. **Keep the Figure-13 attribution.** Rejected because the exact page is
   caption-only. The old dictionary also invented `Omega_O` and `O_pref`, mapped
   VFE `F` to EFE `G`, and mapped precision `π` to habit `E`.
2. **Publish no correspondence at all.** This is defensible and remains the
   fallback if a future implementation needs a stronger equality than the source
   warrants. It is not required for the limited structural facts that a
   transition kernel occupies pymdp `B` and a likelihood occupies pymdp `A`.
3. **Retain a named, validated project heuristic.** Chosen. The constant's name,
   type, validation function, documentation, and tests all mark its epistemic
   status and forbid the retired symbol conflations.

## Two different transformation equations

[Supplement {{CEREBRUM_TRANSFORMATION_SUPPLEMENT}}]({{CEREBRUM_TRANSFORMATION_SUPPLEMENT_URL}}) on physical page {{CEREBRUM_TRANSFORMATION_SUPPLEMENT_PAGES}} defines transformation-specific free
energy as target-case VFE plus an **unscaled** directed KL divergence. [Supplement {{CEREBRUM_EFE_TRANSITION_SUPPLEMENT}}]({{CEREBRUM_EFE_TRANSITION_SUPPLEMENT_URL}}) on physical page {{CEREBRUM_EFE_TRANSITION_SUPPLEMENT_PAGES}} separately adds a `γ`-scaled directed KL term to an
expected-free-energy transition objective. `transformation_cost` implements the
former complete sum; `expected_free_energy_transition` implements the latter.
Their tests use the same nonzero KL example and require the results to differ
when `γ` is not the neutral scale.

## Core Genitive versus novel Generative

CEREBRUM [Table {{CEREBRUM_CORE_FUNCTION_TABLE}}]({{CEREBRUM_CORE_FUNCTION_TABLE_URL}}) (physical page {{CEREBRUM_CORE_FUNCTION_TABLE_PAGES}}) and
[Supplement {{CEREBRUM_CASE_SUPPLEMENT}}, core Genitive]({{CEREBRUM_GENITIVE_CASE_URL}}) (physical page {{CEREBRUM_GENITIVE_CASE_PAGES}})
define a core Genitive model role: relation, context, possession, and parameter
source. [Supplement {{CEREBRUM_GENERATIVE_SUPPLEMENT}}]({{CEREBRUM_GENERATIVE_SUPPLEMENT_URL}}) (physical pages {{CEREBRUM_GENERATIVE_SUPPLEMENT_PAGES}}) separately introduces a novel
Generative case. The source abbreviates both `[GEN]`.

AILM therefore represents a core `CORE_GENITIVE` and a separate
`NOVEL_GENERATIVE`. Their serialized values contain `core:GEN` and `novel:GEN`,
respectively. `CEREBRUM_FUNCTION_ABBREVIATIONS` retains the shared source label,
and `validate_cerebrum_function_roles` refuses an inventory that drops either
role or collapses their serialization.

## Per-case free energy

[Supplement {{CEREBRUM_CASE_SUPPLEMENT}}]({{CEREBRUM_CASE_SUPPLEMENT_URL}}) does not assign an `alpha, beta` scalar pair to every core case.
Only Nominative has a likelihood weight inequality and only Accusative has a
complexity weight inequality. The other core cases introduce distinct auxiliary
latent variables and precision relationships:

| Case | Source-specific structure |
|---|---|
| GEN | relationship variable `r` |
| DAT | goal variable `g` |
| INS | action variable `a` |
| LOC | environment variable `e` |
| ABL | origin variable (disambiguated as `o_origin` in AILM) |
| VOC | communication variable `c` |

`CEREBRUM_CASE_FORMULATIONS` records those distinctions.
`AILM_SCALAR_CASE_WEIGHTS` contains project-owned numerical defaults only for
NOM and ACC. `case_conditioned_free_energy` fails closed for the augmented cases
until a function accepts and uses their additional latent state. The scaffold
remains unit-tested but inert in Level 4's measured decoder.

## Case oracle boundary

The source ontology does not validate AILM's English UD label heuristic.
`UD_DEPREL_TO_CASE` and `ADPOSITION_TO_CASE` are project research decisions;
agreement with their output is agreement, never grammatical-case accuracy. The
Level 4 model observes UPOS only and is structurally forbidden from observing
the `deprel` and morphology fields used to construct evaluation labels.
