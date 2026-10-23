# Style Guide

Prose and code conventions for AILM. The honesty rules come first because they
are the ones with a test behind them; everything after is craft.

- [Honesty rules](#honesty-rules)
- [Prose conventions](#prose-conventions)
- [Naming](#naming)
- [Docstrings carry the load-bearing decisions](#docstrings-carry-the-load-bearing-decisions)
- [Error messages](#error-messages)
- [Code conventions](#code-conventions)
- [Layer boundaries](#layer-boundaries)

---

## Honesty rules

### 1. No bare numerals

Every quantitative statement resolves from the claims ledger. In `docs/manuscript/`
that means a `{{TOKEN}}`; in `docs/`, `README.md`, `AGENTS.md`, `CHANGELOG.md`,
and `src/` docstrings it means a `claim:<label>` tag. See
[`syntax_guide.md`](../syntax_guide.md) for the mechanics.

| | |
|---|---|
| **BAD** | The letter agent reaches 11.92 perplexity against a 19.96 unigram baseline. |
| **GOOD** | The letter agent reaches claim:stage1_letter_perplexity against a claim:stage1_unigram_baseline unigram baseline. |

Two exceptions, both narrow:

- **Thresholds quoted from a specification** are not measurements. "The gate
  demanded ≥2 nats over uniform" is a quotation of a spec, and the point of the
  sentence is that the spec was wrong. Quote it, then immediately give the
  measured ceiling as a claim.
- **Structural constants defined in code** — the number of levels, the alphabet
  composition — are better written as a reference to the symbol (`N_PHONES`,
  `ALPHABET`) than as a digit, so that a drifting constant is caught by a reader
  looking at one place.

If a number has no ledger entry and is not one of those two cases: **do not
invent one and do not write a literal.** Record it from the run that produced it,
or say explicitly that it is a missing claim.

### 2. Report what was measured, not what was hoped

State the metric, the evaluation set, and the size — every time.

| | |
|---|---|
| **BAD** | G2P accuracy is about 71%. |
| **GOOD** | The full-set NumPy forward filter reaches claim:stage2_g2p_accuracy% over claim:stage2_g2p_eval_positions after zero disagreement with the pymdp Agent on a bounded subset, against claim:stage2_g2p_baseline_accuracy% for the context-free per-letter `argmax`. |

A number without its *n* is not a result. A margin without its comparator is not
a finding.

### 3. Name the subset, and name what it costs

The G2P evaluation uses only length-matched pairs —
claim:stage2_g2p_length_matched_pct% of CMUdict, biased toward short,
transparent-orthography words. Every accuracy figure in this project is a claim
about that subset, not about English.

| | |
|---|---|
| **BAD** | The model achieves competitive G2P accuracy. |
| **GOOD** | On the length-matched subset (claim:stage2_g2p_length_matched_pct% of CMUdict, biased toward transparent orthography), the model reaches claim:stage2_g2p_accuracy%. Comparison with published G2P systems would be invalid without matching this subset. |

### 4. Distinguish agreement from accuracy

Where the reference is itself a research claim, agreement with it is
**agreement**, not accuracy. The 8-case UD oracle is the standing example: only
NOM/ACC/GEN are morphologically marked in English; DAT/INS/LOC/ABL are derived
via `deprel` mapping — a principled heuristic, not ground truth. Stage 4 prose
must say "agreement with the oracle" and must not say "case assignment accuracy".

### 5. Distinguish necessary from sufficient

"Beats the baseline" is necessary and not sufficient: one of Level 2's three
mis-wirings cleared the context-free bar while being wrong. Prose that reports
only the comparison overstates what the comparison established. Say which guard
did the work.

| | |
|---|---|
| **BAD** | The emission is verified because the agent beats the baseline. |
| **GOOD** | Beating the context-free bar rules out two of the three mis-wirings tested; the third clears it. What pins the emission is the column-sum invariant on `A` and agreement with the closed-form oracle. |

### 6. Never launder a defect into a feature

Every gate restatement or stage re-scope in this project is written as a
**correction**, with the original wording, why it was wrong, and what replaced
it. Do not rewrite history into a design decision that was made deliberately
when it was not. In particular, the original Stage 1-4 gates were repaired after
their levels had already been built or measured; later preregistration does not
retroactively make that sequence prospective.

| | |
|---|---|
| **BAD** | The gate is stated against the unigram baseline, which is the natural comparator. |
| **GOOD** | The gate was originally specified as "≥2 nats over uniform". That is unreachable at this model order — the ceiling is claim:stage1_max_bigram_gain_nats nats — so it was restated against the context-free unigram baseline. |

### 7. State the current project state honestly

Name the arm and population before its verdict. The canonical widened Stage 5
arm and its retained case-only ablation differ; Stage 6’s development and
held-back document analyses differ. The [hierarchy](../hierarchy.md) owns those
current distinctions. The connected runtime is implemented and separately
measured, but conditional oracle agreement is not a global posterior proof,
finite in-vocabulary word loss is not full-stream word loss, and a bounded
policy-generated word is not speech readiness. A roadmap row must distinguish
implementation, measured evidence and a proposed extension.

### 8. Ceilings and unclaimed headroom stay visible

The forward–backward and Viterbi ceilings (claim:stage2_g2p_smoothing_ceiling%
and claim:stage2_g2p_viterbi_ceiling%) are recorded precisely because the
filtering architecture deliberately does not claim them. Reporting only the
figure that flatters the gate, without the ceiling, hides the gap.

---

## Prose conventions

- **Markdown, not HTML.** Use HTML only for what markdown lacks (`<details>`).
  Never XML-style tags.
- **Lead with the finding**, not with the framework that produced it.
- **Paragraphs do the work.** Bullets are for genuine lists — parallel items of
  the same kind — not for chopping an argument into fragments.
- **Active voice for methods.** "The recorder derives the margin", not "the
  margin is derived".
- **One idea per paragraph.** If a paragraph needs "additionally", it is two.
- **No "In summary" / "In conclusion"** at section ends. If the section needed a
  summary, it was too long.
- **Concrete paths, not descriptions.** "the phoneme level" is a gesture;
  [`../src/ailm/levels/phoneme.py`](../../src/ailm/levels/phoneme.py) is a
  reference. Links are relative to the containing Markdown file, and every link must
  resolve.
- **Name the test, not the concept.** "This is verified" is unfalsifiable;
  "`test_stage2_g2p_emission_orientation_is_pinned` asserts the column sums"
  can be checked.
- **Varied rhythm.** Short sentences for findings, longer ones for mechanism.

---

## Naming

**Claim labels.** `stage<N>_<subject>_<what>`, lowercase with underscores, unit
suffix wherever the unit is not obvious: `_nats`, `_bits`, `_pts`, `_pct`,
`_positions`, `_perplexity`, `_accuracy`. `stage2_g2p_margin_pts` is unambiguous;
`stage2_g2p_margin` is not, and the ambiguity propagates into prose.

**Matrices.** Follow the pymdp letters and say which slot: "the emission `A`",
"the transition `B`", "the preference vector `C`", "the initial prior `D`". Never
"the matrix" — the entire Stage 1 defect was about which slot held which matrix.

**Case roles.** `CaseRole.GEN` is the linguistic Genitive. CEREBRUM's model-role
inventory separately contains `CerebrumFunctionRole.CORE_GENITIVE` and
`CerebrumFunctionRole.NOVEL_GENERATIVE`; the source abbreviates both `[GEN]`.
Prose must name the scoped enum member, never infer equivalence from a bare
abbreviation or row position.

**Levels versus stages.** A *level* is a place in the hierarchy (Level 2 =
phonemes). A *stage* is a build step with a gate (Stage 2 = the phoneme level's
gate). L1–L6 are the linguistic levels; Stage 0 is infrastructure and Stage 7
is an external speech boundary. Stage 2 covers the L2A phoneme-sequence and
L2B aligned-decoding paths. Document-position terciles belong to L6. Use the
[governing level table](../hierarchy.md) when naming a path.

---

## Docstrings carry the load-bearing decisions

`src/**/*.py` is inside the claims-scanner scope, and the module docstrings are
where the reasoning behind each level lives — not in comments beside the code.
The pattern established by
[`../src/ailm/levels/letter.py`](../../src/ailm/levels/letter.py) and
[`../src/ailm/levels/phoneme.py`](../../src/ailm/levels/phoneme.py):

1. **Architecture** — the factors, modalities, and exact matrix shapes.
2. **Where the corpus statistics live** — which slot, and the measured cost of
   the wrong slot.
3. **What the comparator is and why** — including why uniform is not it.
4. **Which numbers are not interchangeable** — smoothed versus unsmoothed floors,
   all-positions versus transitions-only, subset versus full corpus. This section
   exists because a gate asserting the wrong pair of them can never pass.
5. **Numerical policy** — float width and the tolerance it implies.

Function docstrings use Google style with `Args:`, `Returns:`, `Raises:`. Raises
is not optional: an undocumented exception is an undocumented control-flow
branch.

---

## Error messages

Every raise names the offending value and, where possible, the fix. The failures
this project cares about are found by reading error text, so error text is
interface.

```python
# BAD
raise ValueError("bad matrix")

# GOOD (the pattern in src/ailm/verification/oracle.py)
raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
```

For a gate or ledger failure, the message must also say what to do:

```python
raise MissingClaimError(
    f"The manuscript requires {len(missing)} claim(s) the ledger does not have: "
    f"{sorted(missing)}. Record them from the run that produced the value — "
    "never hand-write the number into prose."
)
```

`GateVacuousError` messages name the gate, both values, the threshold, and the
direction, so a vacuous gate is diagnosable without opening the source.

---

## Code conventions

- **Python 3.11+ union syntax**: `dict[str, int] | None`, never
  `Optional[Dict[str, int]]`.
- **`from __future__ import annotations`** at the top of every module.
- **Complete type annotations** on every public function.
- **Module-level `__all__`** where the public surface matters — and re-exports
  are deliberate, documented choices (the phoneme level re-exports the letter
  level's indexing because the G2P emission is indexed by letter, and callers
  need the same indexing).
- **Fixed seeds and sorted iteration** anywhere randomness or dict order could
  reach an output. Vendored artifacts and figures must be byte-identical on
  regeneration.
- **`MPLBACKEND=Agg`** for anything touching matplotlib.
- **Never suppress a failing step.** No `2>/dev/null || true` around a generator,
  a recorder, or a validator: a swallowed exit code turns a broken regenerate
  into a silent no-op. If a step is genuinely best-effort, say why in a comment.

---

## Layer boundaries

| Layer | May import | Must not import |
|---|---|---|
| [`../src/ailm/verification/oracle.py`](../../src/ailm/verification/oracle.py) | `numpy` | **`pymdp`, `jax`** — independence is its entire value |
| `src/ailm/levels/` | `pymdp`, `jax`, `numpy`, `ailm.corpus`, `ailm.levels` | `infrastructure.*`, anything in `scripts/` |
| `src/ailm/figures/` | `matplotlib`, `ailm.viz_config`, the vendored artifacts, the token map | `infrastructure.*`; and **no hardcoded measured numbers** |
| `src/ailm/` generally | stdlib, declared third-party deps | `infrastructure.*` — `src/` stays standalone |
| `scripts/*.py` | declared project dependencies; `infrastructure.*` only behind an explicit root resolution | duplicated reusable business logic; prefer orchestration and generated-artifact I/O |

The preferred `scripts/` pattern is a thin orchestrator: reusable computation
lives in `src/ailm/`, and a script wires it, writes files, and reports paths.
The audited tree still contains bounded recorder-specific estimands in stage and
specialized `record_*_claims.py` writers. Those functions must remain directly
testable and single-writer-owned; new shared computation belongs in `src/`, and
existing recorder-local computation should migrate when it is reused rather
than being duplicated.

---

## The figure caption standard

Adopted 2026-07-30, enforced by `tests/test_caption_standard.py`. Every
manuscript figure caption states three things:

1. **What is plotted** — the quantity and its source (the builder call, the
   ledger artifact, the recorder).
2. **Over what population, and how many** — the workload the panel describes,
   with its *n* as a `{{TOKEN}}` where the ledger records one.
3. **What would falsify the panel** — the concrete observation (a marker off a
   bar, a point off the identity line, support that grows with prefix length)
   that would refute the claim the figure carries. A caption that cannot name
   its falsifier is describing decoration.

The enforcement is absolute: every caption the manuscript's `fig:` anchors
name must comply, and `tests/test_caption_standard.py` requires exact coverage
between those anchors and the captions it parsed. The rule began as a ratchet
over an allowlist of captions predating the standard; that allowlist reached
empty and was removed rather than kept as a dormant exception, so a new figure
complies from its first commit and there is no list to add a label to.
The mechanical proxy for clause 3 is the presence of a falsification statement
(`falsif…` / `refuses to draw` / `would … fail`); the reviewer's judgement is
still required for clauses 1 and 2, which resist mechanical checking.

---

## Cross-references

- [`syntax_guide.md`](../syntax_guide.md) — `{{TOKEN}}` and `claim:` mechanics, crossref syntax
- [`testing_philosophy.md`](../testing_philosophy.md) — the tests that enforce the honesty rules
- [`gate_design.md`](../gate_design.md) — why rules 2, 3, and 5 exist
- [`agent_instructions.md`](../guides/agent_instructions.md) — the operational contract
