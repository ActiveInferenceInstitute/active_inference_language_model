# Stage 2 — The Grapheme-to-Phoneme Level

Why the G2P task is scoped the way it is, what that scoping costs, and why the
gate is defended with a significance test instead of a bare inequality.

## The alignment problem

CMUdict gives `WORD → phone sequence`, but the two are not the same length:

```
CAT      → K AE T          3 letters, 3 phones   (1:1)
THROUGH  → TH R UW         7 letters, 3 phones   (7:3)
X        → EH K S          1 letter,  3 phones   (1:3)
```

To learn `P(letter | phoneme)` we need to know *which* letter emitted *which*
phoneme. Equal lengths permit a deterministic position-to-position pairing;
they do not supply independently annotated phonological alignment. Unequal
lengths require an additional alignment model or an explicitly different task.

**Decision: use only the length-matched subset.** No EM, no heuristic alignment.

A learned alignment would need its own convergence diagnostics and held-out
likelihood check; it cannot be introduced as an invisible preprocessing change
to this gate. The implemented restriction instead makes the pairing rule
explicit and reproducible. Position matching is a task definition, not proof
that every paired letter and phone corresponds linguistically.

## What that costs — stated plainly

The length-matched subset is **claim:stage2_g2p_length_matched_pct% of CMUdict**
(claim:stage2_g2p_train_pairs train + claim:stage2_g2p_test_pairs test pairs).
Roughly three quarters of the dictionary is discarded.

This subset is **not a random sample of English**. Length-matched words are
biased toward:

- shorter words (fewer positions, fewer chances to mismatch)
- transparent orthography (`cat`, `stop`, `plant`) over opaque (`through`, `knight`)
- words without silent letters, digraphs, or vowel clusters

So the accuracy figures here are **not** a claim about G2P on English generally.
They are a claim about G2P on the transparent-orthography subset, which is an
easier problem. Any comparison with published G2P systems would be invalid
without matching this subset and pairing rule. Stage 3 instead models spelling
paths over a bounded word vocabulary; it does not silently solve variable-length
pronunciation alignment. The connected runtime likewise reports position-aligned
phone posteriors, not a general speech or pronunciation-length model.

## The gate is a small margin, and that is expected

| Method | Position accuracy | Uses phoneme context? |
|---|---|---|
| Majority phoneme | weak scale marker; not used as the gate bar | no |
| Per-letter `argmax P(phone\|letter)` | claim:stage2_g2p_baseline_accuracy | no |
| **Full-set NumPy forward filter after bounded zero disagreement with pymdp** | **claim:stage2_g2p_accuracy** | yes, left context only |
| Forward-backward smoothing reference | claim:stage2_g2p_smoothing_ceiling% | yes, both directions |
| Viterbi (MAP path) reference | claim:stage2_g2p_viterbi_ceiling% | yes, both directions |

On a transparent-orthography subset, the letter alone is *nearly sufficient* —
which is exactly why the context-free baseline is already strong. The headroom
for context is small, and forward filtering captures only part of it because
pymdp's `infer_states` is a filtering step: the posterior at position *t*
conditions on observations up to *t* and never revisits them. Smoothing and
Viterbi both see the whole word and score higher on the recorded population.

**Consequence for the gate:** the honest margin for this filtering model is
claim:stage2_g2p_margin_pts percentage points. A bare
`assert agent > baseline` on that margin is not evidence —
it is within the range that a different tie-break rule or smoothing constant
could produce. So the gate requires a **paired McNemar test** over
claim:stage2_g2p_eval_positions aligned positions
(p = claim:stage2_g2p_mcnemar_p), plus a control in which the same procedure is
run on two identical predictors and must come back *non*-significant. Without
that second control the significance test itself is unfalsified.

## Why filtering rather than smoothing

Filtering is what a pymdp `Agent` does natively, one `infer_states` call per
observation. The canonical full-set statistic runs the same recurrence in the
independent NumPy decoder because a full pymdp walk is impractical; the recorder
first requires zero position-level disagreement on a bounded subset. No shipped
Level 3 path consumes Level 2's posterior. A future streaming channel could do
so, but it would be a new measured seam rather than a property of this gate.
Reporting a smoothed number would flatter the gate while describing a different
whole-word computation.

The smoothing and Viterbi figures are recorded as privileged whole-word
references, so the gap is visible rather than hidden. Their ledger labels say
`ceiling`, but they are not hard bounds on a same-information model. If a
future stage needs that additional performance, the honest move is to state that
it uses whole-word smoothing and gate it as a separate, offline capability.

## Orientation is pinned, not assumed

Stage 1 shipped with the bigram matrix in the emission slot — anti-informative,
and invisible to pymdp's validator because the matrix is column-stochastic either
way. Stage 2 has two learned matrices (`A` emission and `B` transition) and so
twice the room for that error.

Guards, all in `tests/test_stage2_gate.py` and `tests/test_phoneme.py`:

- an explicit index assertion that a lopsided bigram count lands in
  `B[0, next, current, 0]` and not its transpose
- column-stochasticity asserted along **axis 1** specifically, since a transposed
  matrix still sums to 1 along *some* axis
- a swapped/transposed-wiring agent that must FAIL the baseline bar
- pymdp cross-checked against `ailm.verification.oracle`, an independent NumPy
  implementation, with the enforced tolerance stated in the test

That last one is the instrument that caught the Stage 1 bug. It is now a tested
module rather than a scratch script, precisely so every level can be checked the
same way.

## Cross-references

- [`architecture.md`](architecture.md) — Level 2 in the full hierarchy
- `src/ailm/levels/phoneme.py` — implementation
- `src/ailm/verification/oracle.py` — the independent cross-check
- `scripts/generate_phoneme_counts.py` — how the vendored artifact is derived
- `docs/provenance/sources.md` — CMUdict provenance and artifact checksum
