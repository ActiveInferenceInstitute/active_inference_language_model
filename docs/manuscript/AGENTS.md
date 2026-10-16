---
title: "Manuscript directory: active_inference_language_model"
type: "manuscript_guide"
---

# Manuscript agent contract — AILM

Project-wide agent rules are in [`../../AGENTS.md`](../../AGENTS.md). The docs-wide operating rules are in [`../AGENTS.md`](../AGENTS.md). This file
governs editing inside `docs/manuscript/` only. Read [`SYNTAX.md`](SYNTAX.md) for the live
cross-reference registry and the full token list; this file is the set of rules
that, when broken, produce a rendered PDF that is wrong rather than ugly.

## The five rules that matter

1. **Use registered token producers for quantitative values.** Measured inputs
   come from `data/claims.json`; validated producers also supply derived values,
   declared protocol settings and registered metadata/reference fields. Use their
   double-brace tokens in numbered sections. If a needed measured value has no
   recorded producer, record it through its owning workflow or omit the
   unsupported precision. Mathematical notation and structural literals must
   remain within the source contract; do not invent a result or copy one from
   an unrelated run.
2. **Never write `\cite{}`, `\ref{}`, or `\eqref{}`.** Use `[@bibkey]`,
   `[@sec:label]`, `[@eq:label]`, `[@fig:label]`, `[@tbl:label]`. The bracket forms
   render through every output path; the LaTeX macros render through one.
3. **Never hand-number a definition or theorem.** Use the `amsthm` environments
   `preamble.md` declares and let LaTeX number them.
4. **Never add a citation you cannot verify.** `references.bib` omits unconfirmed
   fields deliberately. A fabricated page range or DOI is worse than a missing one
   because it is not visibly missing.
5. **Never embed a figure twice.** One `![…]{#fig:x}` per label, in the section
   that owns it. [`SYNTAX.md`](SYNTAX.md) records which section that is.

## The token pipeline

```text
data/claims.json + data/manuscript_literals.yaml + data/reference_sources.json
  → src/ailm/manuscript_variables/__init__.py::generate_manuscript_variables()
  → src/ailm/manuscript_contract.py (source validation)
  → scripts/z_generate_manuscript_variables.py
  → output/data/manuscript_variables.json
  → infrastructure.rendering.manuscript_injection.write_resolved_manuscript_tree()
  → output/manuscript/*.md        ← the renderer reads THESE, not docs/manuscript/*.md
  → src/ailm/manuscript_contract.py (exact-output validation)
  → scripts/pipeline/stage_03_render.py
```

Measured-token naming is mechanical: the ledger label uppercased. Search metadata
and decision-log locators are declared with substantive reasons under the v2
schema in `data/manuscript_literals.yaml`. External source versions, extents, and
locators are read directly from the duplicate-refusing reference snapshot;
executable protocol values are read from their code registries. Every path enters
a strict map that rejects duplicate producers.

Adding a number to the manuscript:

1. Record it from the run that measured it —
   `ledger.record("<label>", value, run_id=..., source=__file__)` inside the
   relevant `scripts/record_stage*_claims.py`.
2. If prose depends on it structurally, add the label to `REQUIRED_CLAIMS` in
   `src/ailm/required_claims.py` so a missing entry fails the render with a
   named label instead of a bare `KeyError`.
3. Reference it in prose as the uppercased label wrapped in double braces.
4. Regenerate. The command's post-write contract checks the complete resolved
   tree and exits nonzero if any marker remains:

```bash
.venv/bin/python scripts/z_generate_manuscript_variables.py
```

An unresolved token that reaches a PDF reads as a fabricated result. The generic
template writer warns and preserves unknown tokens, so AILM wraps it with a
fail-closed pre/post contract. Unknown uppercase tokens; lowercase, spaced, or
hyphenated brace markers; empty replacements; duplicate producers; missing or
extra output files; and output drift all make the command exit nonzero.

## The fenced-preamble trap

`preamble.md` must keep its LaTeX inside the ```` ```latex ```` fence. An unfenced
preamble is **silently dropped** by
`infrastructure/rendering/latex_utils.py`: the render succeeds, and the output has
no `amsthm` environments, no `cleveref`, no `natbib`, and no error message. If
definitions stop being numbered or citations render as `[?]`, check the fence
before checking anything else.

Two related preamble constraints:

- Load order is load-bearing — `hyperref` before `cleveref`; preserve the tested preamble order.
- No `\newcommand`. Custom macros live only in `header-includes`, which the HTML
  and EPUB paths never consult, so a macro silently produces blank math there.
  Write matrix notation out in full.

## Files excluded from rendering

`AGENTS.md`, `MANUSCRIPT_STATUS.md`, `README.md`, and `SYNTAX.md` are listed in
`infrastructure.rendering.manuscript_injection.EXCLUDED_DOC_FILENAMES`. They are not
token-substituted and are not copied to `output/manuscript/`, which is why they can
show literal token examples. `preamble.md` is refreshed into the injected tree as
the LaTeX preamble rather than assembled as a section. Every other `*.md` in this
directory **is** a section of the paper and will be assembled in lexicographic
order — do not park drafts or notes here.

## Claims-discipline boundary

Manuscript token validation and repository prose scanning are complementary
checks. The repository scanner includes manuscript Markdown as part of its
recursive docs inventory; it is not an exclusion from tag validity or the
conservative bare-number scan:

| Where | Mechanism | Enforced by |
|---|---|---|
| Rendered manuscript sections | double-brace tokens resolved and checked at render, plus repository prose scanning | `src/ailm/manuscript_variables/`, `tests/test_manuscript_variables.py` |
| `README.md`, `AGENTS.md`, `CHANGELOG.md`, `docs/`, `src/` docstrings | inline `claim:<label>` tags scanned against the ledger | `tests/test_claims_discipline.py` |

Use double-brace tokens for measured values in rendered manuscript sections.
A `claim:<label>` tag would remain literal in the PDF, although the repository
scanner also checks any real tag it finds. The excluded control guides use
backed claim tags where needed because they are not token-substituted.

## Editing workflow

1. Classify the change. Editorial corrections must describe the existing
   implementation and evidence; they do not require a new scientific run. If a
   model or measurement changes, its owner changes and validates that source first.
2. When measurement inputs changed, re-record the affected claims: `.venv/bin/python scripts/record_stage2_claims.py` (and the
   stage 0 / stage 1 recorders if their inputs moved).
3. Hydrate the manuscript variables and resolved tree. This must precede figure
   generation because figures consume the generated token map:
   `.venv/bin/python scripts/z_generate_manuscript_variables.py`
4. Regenerate figures — their numbers come from that token map, so a stale PNG
   and a fresh table will disagree:
   `MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py`
5. Re-audit the gates: `.venv/bin/python scripts/audit_gates.py`. Exit code matters.
6. Render, then **read the rendered PDF**. A single pass leaves `??` cross-references;
   confirm figures, tables, equations, and citations all resolved, and look at the
   theorem-like blocks specifically — mangled math in a `definition` block does not
   show up in extracted text.
7. Update [`SYNTAX.md`](SYNTAX.md) if you added or moved a label, using the
   re-derivation commands at the top of that file rather than editing rows by hand.

## Honesty constraints specific to this manuscript

This paper's subject is partly its own defects, which makes overstatement both
easier and more damaging.

- A **quoted declared threshold** goes in backticks with its recorded token,
  preserving the historical wording after substitution. Quotation does not
  exempt a mutable protocol value from injection. Mathematical constants,
  symbol indices, level identifiers, source locators, and executable syntax
  retain their structural meaning; they are not measured result literals.
- Every display equation has a stable semantic `eq:` label and is numbered by
  the renderer. Keep displays outside raw theorem environments so PDF and HTML
  receive the same math. Source validation rejects unnumbered or manually
  numbered displays.
- Never describe a boundary-only stage as a generative model. Stages 1–6 have
  built model objects and recorded outcomes; Stage 5's canonical arm passes a
  gate registered before its artifact existed and carries no held-back
  replication, while the case-only arm it displaced is a retained ablation that
  is non-passing with an
  unresolved paired privileged-label gain, and Stage
  6's development-population document-clustered primary is non-confirmatory; its
  separately registered held-back primary excludes zero at the development
  coverage but spans zero at the family-corrected coverage. Stage 7 is a tested
  speech interface contract rather than a shipped pymdp language level. Preserve
  these distinctions in [@tbl:unaudited_gates] and do not turn a recorded outcome,
  agreement result, or interface-only contract into a metric pass or accuracy
  claim.
- Never report agreement with the Universal Dependencies case oracle as accuracy.
  The oracle is a derivation from `deprel`, not ground truth.
- Never report the forward–backward or Viterbi ceiling as the level's result. The
  architecture filters; the ceilings are headroom it does not claim.
- Do not assert a direction of inequality between two tokens whose values come from
  different recorders. If prose needs "X is lower than Y", confirm both claims were
  recorded by the same run, or state the two quantities without ranking them.

## Current evidence scopes

Keep historical component gates and individually measured interventions separate
from the connected GUM comparison. The latter evaluates the fixed conditional
schedule against disabling all named channels while retaining the same raw-text
encoder. Preserve local-registration scope, normalized-character units, vocabulary
coverage and full-stream closed-vocabulary word loss. Conditional reference checks
do not establish a globally exact posterior.

Hosted synthetic and in-context probe tags name run cohorts of the served model
identified by their receipts; they do not identify separate local weights. Keep
those cohort labels distinct from the local external-model benchmark. Use the
validated outcome tokens when fresh benchmark results can change the prose.

After editing, compare token, citation and labeled-object inventories against the
starting sources and explain any removal. Preserve mathematical definitions and
scientific qualifiers unless the correction is supported by the implementation
or recorded evidence. Generated figures and cover bytes belong to their
generators. A source review is complete only when the focused source checks pass;
publication readiness additionally requires fresh rendering and visual inspection.
