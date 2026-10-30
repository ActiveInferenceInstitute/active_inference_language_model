# Supplement E: Machine-Checked Lean Proofs and Their Scope {#sec:lean}

[@sec:reproducibility] states what the machine-checked layer proves and the one
thing it cannot: that any recorded number is *correct*. This supplement carries
the layer's full argument — how the generated ledger module binds the proofs to
the recorded measurements, what a falsifying re-record looks like, and where the
deliberate scope boundaries lie — kept separate from the body because that
argument needs the Lean definitions themselves, not only their conclusion.

## Lean proof coverage and implementation boundaries

Selected mathematical claims are stated and proved in Lean 4 under `formal/`: the gate taxonomy of [@eq:gate_satisfies]
and [@eq:gate_verdict], the pre-registered sweep-shape predicates of
[@sec:hierarchy], both halves of [@eq:filtering], and the Bonferroni
arithmetic behind [@tbl:multiplicity_family]. The layer ships
164 named theorems and 34 witness
examples. No proof in it is incomplete: the recorded `formal_sorry_free` verdict
is true, established by a word-boundary scan *and* by
`#print axioms` on every named theorem, because an unfinished Lean proof raises a
warning rather than an error and a zero exit code alone would not catch one. The
number of axioms it reaches beyond Lean's `propext` and `Quot.sound` is
1: `Classical.choice`, which core's own `Rat.add_comm`
already depends on, so a stricter policy would be unsatisfiable against core
rationals rather than stricter about anything proved here. `native_decide` is
used nowhere, so no proof trades kernel checking for compiler trust. The
toolchain is pinned in `formal/lean-toolchain`, and the Lake project declares no
dependencies, not even Mathlib, so `lake build` runs offline from a clean
checkout once the pinned Lean toolchain is installed.

The theorems that touch measured values are proved about
`formal/AILM/LedgerData.lean`, the one module holding recorded numbers, and that
module is **generated** from `data/claims.json` by
`scripts/generate_lean_ledger.py` rather than written.
`tests/test_formal_isomorphism.py` requires a fresh generation to reproduce the
committed file byte for byte, and separately re-derives every emitted literal
from the ledger through an independently written conversion, so byte identity
cannot pass on a generator that converts consistently wrongly. The machine-checked
layer therefore cannot drift from the recorded measurements in either direction:
a hand edit fails byte identity, and a re-record nobody regenerated fails it too.

The falsifier is concrete, and it is the reason this layer is a check rather than
a decoration. Every ledger-facing theorem that fixes a threshold closes by
`decide`, so the kernel — not a tolerance — evaluates the recorded values; the
two that instead quantify over *every* threshold or *every* bounded model are
derived from the general theorems, which is what makes them stronger than a
check at one bar. A re-record that moved a channel
sweep's argmax off its interior point, or moved a baseline or a ceiling across a
declared threshold, changes `LedgerData.lean` on regeneration and the
corresponding proof stops compiling. What the layer does **not** establish is
that any recorded number is *correct*: Lean checks that the values have the
logical properties [@sec:results] and `scripts/audit_gates.py` assert of them, not
that they were measured over the right positions, and no theorem here should ever
be cited as evidence that a measurement is right. `formal/README.md` names the
rest of the deliberate scope boundary, familywise error control first among them:
`AILM/Bonferroni.lean` proves the *arithmetic* of the union bound, and the step
from that arithmetic to a statement about rejection probabilities is assumed, not
proved.

The connected runtime's conditional-update checks and GUM comparison do not
acquire a theorem of global posterior correctness from this layer. The Lean
results retain the specific statements and measurement bindings listed above.
