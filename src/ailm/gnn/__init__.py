"""GNN conformance — bind AILM's pymdp models to a declarative spec.

What this module is for
-----------------------
GeneralizedNotationNotation (GNN) is the Active Inference Institute's notation
for writing an Active Inference generative model down as structured Markdown.
AILM carries one GNN spec per shipped level under ``gnn/``. The value of doing
so is not documentation: it is that the ontology block states the *semantic role*
of every matrix by name::

    A=RecognitionMatrix
    B=TransitionMatrix
    D=Prior

That is exactly the invariant AILM's worst shipped bug violated. Stage 1 put the
learned letter bigram table in the emission ``A`` with an identity ``B``, which
is anti-informative rather than loudly broken — observing a symbol yields a
posterior over the symbol that *preceded* it, and an identity ``B`` carries that
backward-looking belief forward as the next-symbol prediction. It measured
claim:stage1_anti_informative_perplexity against
claim:stage1_letter_perplexity for correct wiring. pymdp cannot see it: a bigram
matrix is column-stochastic and passes ``_validate()`` in either slot.

So this module is AILM's **third independent guard** on wiring orientation,
alongside the numpy oracle (``ailm.verification.oracle``) and the measured
comparison against a context-free baseline. Unlike the baseline guard it is not
merely necessary-but-insufficient: it pins each slot by declared *name*, so a
matrix in the wrong slot is a named violation rather than a slightly worse number.

What it deliberately does NOT do
--------------------------------
Conformance checks **structure and declared roles**. It does not verify the
learned probabilities — the oracle cross-check and the gate tests do that, and
nothing here should ever be reported as evidence that a number is right. The
enforced subset and the explicit non-goals are written out in ``gnn/SPEC.md``.

One nuance matters, because a purely dimensional check would be vacuous against
the historical bug: for the letter level the correct and the swapped wiring have
*identical shapes* — ``A`` is ``(1, 30, 30)`` and ``B`` is ``(1, 30, 30, 1)``
either way. Shapes alone therefore cannot catch it. What catches it is the
``InitialParameterization`` block, which declares each matrix's *structural* form
— ``Identity(30)`` for a directly-observed state, ``Uniform(30)`` for an
uninformed prior, ``Learned(...)`` for anything sourced from a vendored corpus
artifact. Those forms are structure, not learned values: an identity emission
means "the hidden state is observed directly", and a ``Learned`` transition that
turns out to be the identity carries no dynamics at all. Both are checkable
against a live agent, and both fire on the A/B swap.

No upstream dependency
----------------------
This module is stdlib plus numpy. It never imports the GNN pipeline (25 steps,
its own dependency tree) and never imports pymdp or jax — a conformance checker
that shared code with the thing it audits would inherit the flaw it audits for.
``describe_shapes`` reads a live ``pymdp.agent.Agent`` through ``numpy.asarray``
and the public ``A``/``B``/``C``/``D`` attributes only.

The AILM subset of GNN
----------------------
``parse_gnn_spec`` accepts the canonical section set and section order, and is
deliberately *stricter* than upstream in three places:

1. An unrecognized ``##`` section is an error, not an "extension". A typo in a
   section name would otherwise silently drop the whole block.
2. ``InitialParameterization`` accepts only structural forms
   (``Identity(n)``, ``Uniform(n)``, ``Zeros(n)``, ``Learned(k=v, ...)``) and
   rejects an inlined matrix literal. A 30x30 bigram table is a derived
   artifact; a spec that inlines it is a copy of the data, not a specification
   of it, and it will drift.
3. Declared variables and ontology annotations must be in bijection. Anything
   declared but un-annotated is a slot whose role nothing pins.

Malformed input raises :class:`GnnSpecError` naming the offending line. Nothing
returns a partial spec.

## Package layout

Split from a single 2,405-line module along the section boundaries its own
author had already drawn. The public API is unchanged and re-exported here, so
every existing import keeps working.

* ``ailm.gnn.types`` - errors, section names, the parsed-spec dataclasses.
* ``ailm.gnn.parsing`` - GNN source text to a ``GnnSpec``. Never touches a live
  agent, so a spec can be read and inspected with no pymdp in reach.
* ``ailm.gnn.shapes`` - a live agent's canonical dimensions, without judging
  them.
* ``ailm.gnn.declarations`` - what a spec says about itself, checked without
  building anything. A spec can be internally incoherent, and none of that needs
  a live agent to detect.
* ``ailm.gnn.live`` - a built agent against the spec that declares it. Every
  check here needs the object: the arrays pymdp holds, their shapes, their
  stochastic orientation. These are the ways a declaration and a model disagree
  while each looks individually reasonable.
* ``ailm.gnn.conformance`` - the entry point, and nothing else. It collects both
  families of check and turns a non-empty report into a raise.
* ``ailm.gnn.registry`` - which specs exist under ``gnn/models/``, and where.

``conformance`` was one file of 1,260 lines holding all three concerns. The
split runs along a seam that was already in the code and invisible in it: checks
needing only the spec, checks needing the built object, and the orchestration
over both.

The dependency order is linear with no cycles: ``types``, then ``parsing`` and
``shapes``, then ``declarations``, then ``live``, then ``conformance`` and
``registry``.
"""

from __future__ import annotations

# ruff: noqa: F401  -- this module exists to re-export; nothing here is used locally.
from ailm.gnn.conformance import assert_conforms, check_conformance
from ailm.gnn.declarations import (
    COMBINATION_RULES,
    _check_combination_rule,
    _check_inference_configuration,
)
from ailm.gnn.parsing import parse_gnn_spec, parse_gnn_string, parse_learned_arguments
from ailm.gnn.registry import (
    SPEC_KINDS,
    SPEC_SUFFIX,
    GnnModelId,
    discover_models,
    load_spec,
    parse_model_id,
    spec_dir,
    spec_path,
)
from ailm.gnn.shapes import describe_shapes, live_inputs
from ailm.gnn.types import (
    _VECTOR_ROLE_SIZES,
    GNN_SECTIONS,
    MATRIX_ROLES,
    GnnConformanceError,
    GnnConnection,
    GnnSpec,
    GnnSpecError,
    LiveConformanceInputs,
)

#: The public API is unchanged from the single-module version. The
#: underscore-prefixed names below are re-exported because
#: ``tests/test_gnn_conformance.py`` reaches for them directly to test
#: individual checks in isolation; they are internal and stay out of ``__all__``.
__all__ = [
    "COMBINATION_RULES",
    "GNN_SECTIONS",
    "MATRIX_ROLES",
    "SPEC_KINDS",
    "SPEC_SUFFIX",
    "GnnConformanceError",
    "GnnConnection",
    "GnnModelId",
    "GnnSpec",
    "GnnSpecError",
    "LiveConformanceInputs",
    "assert_conforms",
    "check_conformance",
    "describe_shapes",
    "live_inputs",
    "discover_models",
    "load_spec",
    "parse_gnn_spec",
    "parse_gnn_string",
    "parse_learned_arguments",
    "parse_model_id",
    "spec_dir",
    "spec_path",
]
