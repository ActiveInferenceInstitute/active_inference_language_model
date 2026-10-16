# `docs/design/` — one decision per file

The scientific rationale behind AILM's models: why each level is shaped the way it
is, and what the rejected alternative was. The **code** architecture is a different
document, [`../architecture.md`](../architecture.md); confusing the two produces two
files that each look authoritative and disagree.

The governing description of the level structure is
[`../hierarchy.md`](../hierarchy.md). Where a design file disagrees with it,
`hierarchy.md` wins.

## Quick start

Reading order for someone new to the models:

1. [`../hierarchy.md`](../hierarchy.md) — current model paths and the scope of each result
2. [`connected_runtime.md`](connected_runtime.md) — the connected API/CLI, causal schedule and evidence limits
3. [`architecture.md`](architecture.md) — per-level A/B/C/D and the retained standalone designs
4. the per-stage file for whichever level you are about to change

## The files

| File | The decision it records |
|---|---|
| [`connected_runtime.md`](connected_runtime.md) | The causal connected runtime and its conditional inference contract, with explicit evidence limits |
| [`architecture.md`](architecture.md) | The level hierarchy, the per-level A/B/C/D, the scale arithmetic |
| [`stage2_g2p.md`](stage2_g2p.md) | The one-to-one letter/phone alignment restriction, and what it biases |
| [`stage3_word.md`](stage3_word.md) | Why the word level factorizes into identity × position rather than trie nodes, what the closed vocabulary costs, and why three of five mis-wirings still clear the bar |
| [`stage4_case.md`](stage4_case.md) | Why the case level observes only UPOS (the circularity trap), and why one declared threshold is vacuous on gold labels and above-reference on derived ones |
| [`cerebrum_mapping.md`](cerebrum_mapping.md) | The exact CEREBRUM source tuple, Figure 13's caption-only boundary, and AILM's explicitly project-owned pymdp crosswalk |
| [`no_embeddings.md`](no_embeddings.md) | Why kernel composition is not a DisCoCat meaning functor, and what a valid bridge would require |
| [`generation_problem.md`](generation_problem.md) | Generation via policy selection under expected free energy — the implemented design and its analysis |
| [`speech_boundary.md`](speech_boundary.md) | Where DSP ends and pymdp begins |
| [`power_and_ci.md`](power_and_ci.md) | Power analysis and exact confidence intervals for the paired tests |

## Key rules

- **State the alternative.** A file that only describes what was built is a
  description, not a decision record.
- **Cite code by path**, down to the function: `src/ailm/levels/phoneme.py::build_g2p_emission`.
- **Measured numbers use a `claim:` tag** backed by [`../../data/claims.json`](../../data/claims.json).
  These files are inside the claims-discipline scan.
- **Relative links only**, and every one must resolve — a parameterized test checks
  every document under `docs/`.

## Common commands

Run from the project root, with the project's own interpreter.

```bash
.venv/bin/python -m pytest tests/test_claims_discipline.py -q --no-cov
.venv/bin/python -m pytest tests/test_documentation.py -q --no-cov
```

## More information

See [`AGENTS.md`](AGENTS.md) for the rules in full, [`../AGENTS.md`](../AGENTS.md)
for how `docs/` is gated, and [`../README.md`](../README.md) for the full index.
