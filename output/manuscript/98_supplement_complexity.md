# Supplement C: Exact Computational and Memory Costs {#sec:complexity}

AILM's interpretability is usually priced in accuracy, the capability cost that
[@sec:limitations] accepts openly. Its other price is what each level costs to
*run*, in scalar operations and in bytes — priced here, in a supplement, because
[@sec:limitations] needs only that conclusion while checking it needs the exact
counts and the instrumented reconciliation below.
Asymptotic statements are easy to assert and hard to falsify, so the project's
complexity claims are held to the same standard as its accuracy claims: every
big-O below is backed by an exact closed-form count, and every closed form is
reconciled against an **instrumented execution** of the shipped recursion on the
shipped workload. The reconciliation is exact — one disagreeing operation, or
one disagreeing byte, and the recorder refuses to write the claim.

Every figure below is a property of two specific choices — the state spaces AILM
enumerates, and the dense arrays `pymdp` requires to hold them — and generalizes
no further. A model can be explicit and symbolic without being dense, and
[@sec:complexity_collapse] gives this project's own numbers for the difference,
so none of what follows should be read as the price of interpretability as such.

These are historical component workloads, not a census of the complete
connected runtime. Additional bridge and observation-encoder arrays, message
transformations and compiled execution overhead are outside the totals below.

Nothing in this section is a timing. Operation counts are exact for the stated
recursion and workload; parameter bytes additionally depend on the declared
dtypes and dense representation. These quantities can be derived by independent
routes and required to agree. Deterministic replay of this census does not
imply that fresh externally served measurements must repeat unchanged.

## The component filtering cost model and its exact closed form {#sec:cost_model}

The instrumented component cost model uses the scaled alpha recursion, the two halves of
[@eq:filtering]. One scored position over $S$ hidden states costs a likelihood
row product, a normalization, and, at every position except the last of a
sequence, a transition push:

$$
\underbrace{S}_{\text{likelihood } \times}
\;+\;
\underbrace{(S-1)}_{\text{normalizer } +}
\;+\;
\underbrace{S}_{\text{normalize } \div}
\;+\;
\underbrace{1}_{\log Z_t}
\;+\;
\underbrace{S^2 + S(S-1)}_{\text{push } \mathbf{B}q \;(t < T)}
$$ {#eq:step_cost}

Every term names the matrix that causes it. The first is one entry of the
$\mathbf{A}$ row for the observed symbol against one entry of the predictive
prior, per state. The next three are the normalization of that product and the
logarithm of its normalizer, which is the position's contribution to the log
evidence; they touch no matrix at all. The last is the entire $\mathbf{B}$
product, one multiply per transition entry and one add per accumulation. The
prior $\mathbf{D}$ is copied once per sequence and is free, and $\mathbf{C}$ and
the policy prior are never read during recognition.

Collected, the closed form is compact enough to state exactly. A position that
carries a push costs $2S^2 + 2S$ scalar operations; the last position of a
sequence, which pushes nothing, costs $3S$; a decoding level adds $S-1$
comparisons at every position. Every scored position therefore lies between
$3S$ and $2S^2 + 3S - 1$, and a corpus of $n$ sequences totaling $T$ positions
costs $T\,S + (T-n)\,S^2$ multiplications, $\Theta(T\!\cdot\!S^2)$ overall
with one push withheld per sequence, plus $T(S-1)$ comparisons when the level
decodes — the standard per-step cost of the forward algorithm's trellis
recursion over $S$ hidden states, exact here because [@eq:filtering] is
computed rather than approximated [@rabiner1989tutorial]. The push is not one term among several: $2S^2 - S$ exceeds $3S$ for
every $S > 2$, and their ratio is $(2S-1)/3$, so above a two-state inventory the
$\mathbf{B}$ product is essentially the whole bill and grows without bound
relative to everything else. Filtering is quadratic because transition is
quadratic; nothing else in [@eq:step_cost] is.

The $T-n$ term is the easiest to state wrongly, and it is what the
instrumentation checks: the counter in `ailm.complexity` increments beside each
statement of the recursion as it executes, so a formula with the off-by-one (a
push after the final observation) fails the reconciliation, and
`tests/test_complexity.py` carries that wrong formula as a negative control that
must fail.

| level | hidden states $S$ | workload (sequences · positions) | ops / position | live agent memory |
|---|---|---|---|---|
| letters (L1) | 30 | 1 · 1323 | 1858.7 | 7444 B |
| phonemes (L2A) | 39 | 32,420 · 215,349 | 2667.9 | 12,484 B |
| G2P decode (L2B) | 39 | 6484 · 42,820 | 2703.3 | 11,044 B |
| words (L3) | 2000 × 21 | 78,241 tokens | 2 per token | 20,545,960 B |
| cases (L4, decode) | 8 | 229 · 1319 | 130.2 | 724 B |
| generation (L1g) | 30, one action per symbol | — | $\Theta(V + S)$ per emitted letter | 111,960 B |

: Exact per-level costs over each level's own gate workload — the identical
populations the accuracy numbers of [@sec:results] are stated over, enforced by
`tests/test_complexity.py`. Every operation count is an instrumented-execution
total that the closed form of [@eq:step_cost] must reproduce exactly; every
memory figure is the live agent's parameter bytes, priced by its GNN
specification. {#tbl:complexity}

The recorded rates sit below $2S^2 + 2S$ by exactly the fraction of positions
that withhold a push, which is why the short-sequence levels sit further below
their ceiling than the letter level does: the phoneme and G2P workloads are
lexicon words, so one position in every word is word-final and pushes nothing,
while the letter workload is a single passage that withholds one push in total.

![Per-symbol cost, decomposed and then scaled. Left: each filtering level's
recorded rate split into the terms of [@eq:step_cost] — the $\mathbf{A}$ row
with its normalization at $3S$ per position, the $\mathbf{B}$ push at
$2S^2 - S$ amortized over the positions that carry one, and the decode argmax at
$S-1$. The three segments are recomputed from the workload-shape tokens when the
figure is drawn and must sum to the recorded bar; a split that does not sum to
its own total refuses to draw. Middle: the same recorded rates against state
count, inside the envelope the recursion itself sets: $3S$ where a position
withholds its push, $2S^2 + 2S$ where it carries one, and up to $S-1$ more where
the level decodes. A recorded rate outside $[3S,\, 2S^2 + 3S - 1]$ is
unproducible by any workload shape and refuses to draw; the word level's diamond
lies far below the band because determinism removed the recursion, not because
the band is wrong. Right: what context length costs each mechanism, each
system's per-symbol cost divided by its own context-free term, so no constant of
either system enters. AILM's line is flat by construction — no term of
[@eq:step_cost] mentions how much text precedes the scored symbol — and a
dependence on it appearing there would falsify the
panel.](../../output/figures/complexity_ops.png){#fig:complexity_ops width=98%}

## Factored storage and the remaining joint-emission cost {#sec:complexity_factored}

A level with $F$ hidden factors of sizes $S_1,\dots,S_F$ has a joint state space
of size $\prod_f S_f$, and [@eq:filtering] is defined over that joint however the
factors are stored. Substituting that joint size for $S$ in the closed form, one position of
an unfactored joint costs $2(\prod_f S_f)^2 + 2\prod_f S_f$ operations and its
dense transition holds $(\prod_f S_f)^2$ entries. The product is what makes both
explode, and it does so where a sum would be affordable: adding a modest factor
multiplies the state space rather than extending it.

Factoring the transition changes the second of those quantities and not the
first. With one transition per factor the stored transition drops from
$(\prod_f S_f)^2$ to $\sum_f S_f^2$, while the emission stays dense over the
joint. One closed form then prices every level of [@tbl:complexity] from its
declared shape alone, given $N_o$ observations, $N_u$ control states, and
four bytes per float32 entry, as in [@eq:agent_memory]:

$$
4\left(
\underbrace{N_o \prod_f S_f}_{\mathbf{A}}
\;+\;
\underbrace{N_u \sum_f S_f^2}_{\mathbf{B}}
\;+\;
\underbrace{N_o}_{\mathbf{C}}
\;+\;
\underbrace{\sum_f S_f}_{\mathbf{D}}
\;+\;
\underbrace{N_u}_{\mathbf{E}}
\right)
\text{ bytes}
$$ {#eq:agent_memory}

The last term is the policy prior `pymdp` materializes for every agent, one
scalar at the single-action recognition levels; it was found because the first
reconciliation missed by exactly four bytes per level, and dropping it is a
tested negative control. [@fig:complexity_memory] overlays this form on each
level's measured bytes, and it reproduces all six of them exactly.

The product survives linearly in the emission, which is dense over the joint at
any factorization, and is squared only in an unfactored transition, which is the
term Level 3 deletes. Which term dominates afterwards is a question about the factor
sizes rather than about the architecture. At the word level the vocabulary
factor dwarfs the position factor, so the $V^2$ identity over word states is the
binding cost while the emission's $N_o V P$ entries stay linear in $V$. It is
the identity, the least informative matrix in the level, that sets the
vocabulary cap.

## Determinism collapses the largest state space to a lookup {#sec:complexity_collapse}

The one level that escapes $\Theta(S^2)$ is the one with by far the largest
state space. Level 3's hidden state factorizes into word identity
(2000 states) and position (21
states), and every factor except the word prior is deterministic — the identity
over word identity, the +1 shift over position, the spelling-table emission. The
path identity of [@sec:word_level] ($P(\text{path of } w) = \mathbf{D}_{f0}[w]$)
is therefore also a *complexity* statement: scoring a token is one prior lookup,
one logarithm and one accumulate, 2 operations
per token over all 78,241 held-out tokens, at any vocabulary
size. The unfactored joint would have cost $2(VP)^2 + 2VP$ operations
per position; the measured cost is 2, and the gap
is the whole payoff.

Memory is where the factorization is priced. A dense transition over the
unfactored joint — one hidden factor over (word, position) pairs — would need
$(V\!\cdot\!P)^2$ float32 entries: 7.056 GB at the
shipped configuration. The factored transitions need $V^2 + P^2$ entries:
16 MB. The ratio is
441 — exactly $P^2$, up to the negligible
$P^2$ term in the denominator.

That exponent belongs to the dense encoding rather than to the decision to
factor: the arithmetic is correct, and its obvious reading is not. Both transitions are
deterministic maps: from each state exactly one successor is reachable, so the
joint holds 42,000 structurally nonzero entries and
the factored pair holds 2021. Priced that way the
ratio is 20.78, which tends to $P$ as $V$
grows. Under any storage that does not materialize structural zeros the
factorization buys a factor of $P$; the second factor of $P$ is bought by
storing the arrays densely. Crediting it to the factorization would state as a
property of hierarchical models what is a property of this implementation's
representation, and the representation is not free to change through `pymdp`'s
public `Agent` API. What remains is the $V^2$ identity over
word states, which pymdp materializes densely; that term sets the instantiation
cap of roughly 17,320 words ([@sec:word_level]), while the
closed-form scorer, being $\Theta(1)$ per token, is exact at any size — the
complete length-eligible training vocabulary of 38,007 types
would need 5.86 GB as a live agent but adds no additional
lookup cost per token to the closed-form scorer.

The trie the design document asked for is the third pricing, and it is the one
that does not fit at all. Words as paths through a letter trie put the trie's
nodes in the state space: over the length-matched G2P vocabulary that is
78,398 nodes joined by 78,397 edges, so a
dense transition over them is 24.6 GB at float32 and
0.0013% occupied. That occupancy is not a surprise to be
measured but a consequence to be predicted: a tree has one parent per node, so a
dense $\text{nodes} \times \text{nodes}$ array is occupied at essentially
$1/\text{nodes}$ and the rest of it is structurally zero. At full CMUdict the
same encoding is roughly 0.37 TB. No sparse path is
reachable through `pymdp`'s public `Agent` API: the sparse contraction
machinery exists in its JAX backend and the transition contraction is already
sparse-capable, but `Agent` rejects sparse arrays at construction and does not
preserve sparsity across the filtering loop. This is therefore not a constant to
optimize away; it is where the dense representation stops being a
representation *through the interface this project uses*. The trie is not
discarded but induced: after a prefix, the posterior over the word factor is
supported exactly on the words sharing it, which is the trie node
([@sec:word_level]), and the induced form costs the $V^2$ of the factored
transition rather than the 24.6 GB of the dense node
encoding.

That same occupancy bounds what the number licenses. It prices a dense array,
not a trie and not explicitness: stored by its edges, the identical automaton
holds 78,397 entries against the
$\text{nodes} \times \text{nodes}$ cells the dense form materializes. Sparse edge
storage is the ordinary representation for weighted finite-state machinery of
this shape, and is why that tradition carries explicit symbolic models no dense
transition could hold. The obstacle recorded here is that `pymdp` is a
dense-array interface; the decision is a choice of *host*, not a finding that
explicit structure is inherently expensive.

![Memory, exactly. Left: each live agent's parameter bytes — the sum of every
array in $\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}, \mathbf{E}$ —
spanning 724 B for the
8-state case level to 20,545,960 B
for the word level, with the shape closed form of [@sec:complexity_factored]
overlaid as an open marker. Each value is predicted to the byte from the level's
GNN specification, and a marker off its bar — or a recorder disagreeing with the
overlay by one byte — refuses to draw. Right: Level 3 transition memory against
vocabulary size, log-log. The dense unfactored joint grows as $V^2P^2$ and
crosses the pymdp budget before it reaches the shipped vocabulary of
2000 words; the factored form grows as $V^2$, puts the shipped
configuration at 16 MB, and meets the same
budget at the recorded cap of about 17,320 words. The
trie-node encoding is drawn as the level it sits at, recomputed from its
recorded node count: a node count and a size claim that stop agreeing would
falsify it, as would a measured point off either curve or a cap off the
curve–budget intersection.](../../output/figures/complexity_memory.png){#fig:complexity_memory width=98%}

## Computational costs and fixed historical context {#sec:complexity_versus}

The systems of [@sec:external_benchmarks] can be compared with AILM on cost in
one respect only: which quantity each one's per-symbol arithmetic scales with.
AILM's scales with the size of an enumerated state space and with nothing else. [@eq:step_cost] has no term in the amount of text already seen, because an
order-one model carries all of it in a belief of fixed size; scoring the
thousandth symbol of a passage costs exactly what scoring the second cost. A
decoder-only transformer [@vaswani2017attention] is the other way round. Per
generated symbol, with a cache, each layer spends $\Theta(d^2)$ on its
projections and feed-forward block, which is context-free, and $\Theta(Ld)$ on
attending to $L$ cached positions, which is not; over a whole sequence the
attention term is $\Theta(L^2d)$. Width is paid for quadratically, context
linearly per symbol, and the output layer pays $\Theta(d|\mathcal{V}|)$ over a
token vocabulary $\mathcal{V}$ — linear in the inventory, where AILM's inventory
cost is quadratic. The right panel of
[@fig:complexity_ops] draws the shape of that difference in the only units in
which it is well posed: each system's per-symbol cost divided by its own
context-free term, which needs no constant from either.

Two things follow, and a third does not. The first is that AILM's cost is
context-free and its state space is not: doubling the alphabet quadruples the
bill under the dense transition `pymdp` requires, and the state space must be
enumerated in advance, which is the same constraint that makes the model
interpretable. The second is that a
transformer's cost is inventory-free in the sense that matters here: it never
enumerates a hidden state, so its arithmetic is independent of how many distinct
things the world might be in, and it pays instead for the width that lets it
represent them implicitly. What does not follow is any statement about value for
money. A comparison of per-symbol operations between an order-one filter over a
small enumerated inventory and an instruction-tuned transformer pretrained on a
web-scale corpus is a statement about mechanism: it says which quantity each
system's arithmetic scales with, and it says nothing about what either system
can do. [@sec:external_benchmarks] lists what would have to be equalized before a capability
comparison meant anything, and none of it is equalized here.

Constant factors and units deserve the same honesty. The counts in
[@tbl:complexity] are scalar operations, one per array element the recursion
touches, which is the natural currency for a model whose arrays are small
enough to state exactly. Transformer costs are conventionally counted in
floating-point operations inside dense matrix products, executed as fused
multiply-adds on wide vector units at throughputs no scalar count anticipates.
An operation here and a FLOP there are not the same unit, a ratio of counts is
not a ratio of times, and this section deliberately reports neither system's
seconds. The counts here also exclude everything outside the recursion —
interpreter dispatch, `pymdp`'s array bookkeeping, JAX tracing — none of it
counted, and any of it larger than the recursion itself at these sizes. A level
costing 130.2 operations per position does not run
in 130.2 operations' worth of time, and nothing
here claims it does.

Finally, measured and derived quantities are kept distinct. Every per-level
operation total and every memory figure in [@tbl:complexity] is measured: an
instrumented execution of the shipped recursion, and a live agent's actual array
bytes. Every
asymptotic form, every closed form, and both envelopes drawn in
[@fig:complexity_ops] are derived, and each is required to reproduce the
measurement exactly. The transformer side of this subsection is derived alone:
it follows from the published architecture, no transformer was profiled for it,
and no number in it is a claim in the ledger.

## Lexicon-scan cost per generated letter {#sec:complexity_generation}

Generation ([@sec:generation]) adds one term the recognition levels do not
have. Selecting each emitted letter evaluates the top-down preference, a scan
of the 2000-word lexicon for the current prefix at $\Theta(V)$,
and then a policy posterior over the 30 actions at
$\Theta(S)$, so a letter costs $\Theta(V + S)$ and the lexicon scan dominates at
every shipped configuration. This is the honest asymmetry of the architecture:
recognition is quadratic in a level's *own* alphabet, while generation is linear
in the vocabulary of the level *above*. The top-down channel is what costs, and
it costs at every emitted symbol. An action set is also what makes generation
the memory outlier among the letter-sized levels: with one control state per
emitted symbol, $N_u = S$, the $N_u \sum_f S_f^2$ term of
[@sec:complexity_factored] becomes $S^3$, which is almost the whole of
111,960 B. A trie index over the vocabulary would reduce
the scan to the prefix's own support; it is not built, because no measured claim
depends on generation throughput.

Two properties are worth stating plainly. First, every level priced here runs in
kilobytes to megabytes, from the 8-state case level's
724 B to the word level's
20,545,960 B ([@fig:complexity_memory]), so what
interpretability costs at these sizes is accuracy rather than compute. Where
those bytes sit in a given machine's memory hierarchy is not measured here and is
not claimed: cache capacities are hardware-specific, the platform string
recorded in [@sec:reproducibility] names no cache hierarchy, and a byte count on
its own licenses no statement about residency or speed. The priced set is the one [@tbl:complexity] lists: the four
filtering levels L1, L2A, L2B and L4, the word level L3, and the generation
agent L1g. `scripts/record_complexity_claims.py` instruments no others, so the
clause and document-position (`narrative` in code) levels carry no operation
count and no memory figure here,
and none should be read off the ones that do. Second, every number in this
section is refutable by re-execution: the instrumented counter, the closed
forms, and the spec-derived memory prices are three independent derivations
bound together by exact equality, and `scripts/record_complexity_claims.py`
re-derives all of them on every run.
