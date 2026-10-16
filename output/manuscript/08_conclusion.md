# Established Findings and the Next Independent Tests {#sec:conclusion}

AILM now combines explicit GNN-declared components with a live conditional
message schedule. The connected GUM result, the historical component experiments
and the gate-audit method support different conclusions and retain their own
populations, comparators and limitations.

The six modeled levels are built and their outcomes recorded;
one of the six verdicts is qualified.
The clause level clears a gate that was registered before its artifact existed,
on an observation alphabet widened past the case roles, and its privileged-label
gain resolves away from zero under a paired whole-document bootstrap. That gate
is a development-session specification and has no held-back replication, because
the held-back partition was already spent. The case-only clause model it
displaced is retained as an ablation: its gate is non-passing, its
privileged-label gain is
unresolved on both reported scales under the same bootstrap, and its
detectability control recovers a larger planted effect but supplies no
equivalence claim (decision 83). The document-position (tercile) level —
`narrative` in code — has a positive point estimate in a direction timestamped before its measurement, but its
document-clustered primary interval spans zero and is non-confirmatory
(decisions 25--26). The same estimand, registered separately under decision
115 and carried to the held-back split,
gives 0.008288 nats with an
interval that excludes zero at the development coverage and spans zero at the
family-corrected one, on a population whose own resolution,
0.005726 nats, is coarser
than the development-sized effect ([@sec:stage6_heldback_clustered]); the
development result is not retracted by it. The level's sentence-sign statistic
is secondary and nominally
positive; it crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold, the fixed threshold being the one
7 of 16 declared tests cross.
[@sec:results] holds the letter and phoneme measurements and points to where each
later level's measurements are recorded. All level builders produce ordinary pymdp agents,
but among the historical stage scores only L1's headline filters live; the later levels' exact or NumPy
score paths and the live-Agent checks that license them are inventoried in
[@tbl:level_stack] and [@sec:verification]. Two of those checks are exhaustive:
L5 uses an exact closed-form mixture with a live-Agent
posterior-agreement check on the full held-out set that is not score
equivalence, and L4 uses a NumPy filter that the live Agent reproduces at every
held-out position.

The historical stack first established representational hierarchy: L2B infers
over a variable L1 represents, observing the alphabet L1's states are drawn
from. The historical standalone decoders exchange no beliefs. The
seven registered receiver interventions are built on that
composition: five measured separately, the case-word bridge measured as an
exploratory prior-only sweep on the development population, and its
privileged-oracle diagnostic the one registered seam that records no
estimand. The measured ones disagree in sign. The L3→L2B lexical channel and L1→L2B coupling enter the
same letter-to-phone receiver but differ in sender: the lexical channel helps
(+3.333 pts at *p* = 3.756e-102), the
coupling harms (moving accuracy -0.2639 pts at *p* = 1.547e-06). L3→L1g, the lexicon-to-letter preference used by
the recorded generation run, helps without a paired significance test behind
it. The independent L6→L5 and privileged punctuation-type-oracle→L3
interventions are less clean and stay on
the record as such. L6→L5 moves the
case-only clause model's `D` prior against a tercile-scrambled control, in a
receiver whose privileged-label gain is unresolved. The independent oracle→L3 word-prior intervention is a corrected
helpful result on both directional summaries that falsified its predicted harm
([@sec:hierarchy]). It is not evidence about an L5 output channel. These
experiments were not executed as a connected cascade.
The subsequent connected runtime executes explicit conditional messages and
live next-word policy selection. Its locally frozen GUM comparison spans
22 previously unexamined documents after excluding
8 development documents. The mean paired
normalized-character log-loss gain is 0.4420177562251402 nats
with interval [0.42010385279024465, 0.46325671494396264],
indicating lower integrated loss. Full-stream word loss remains
infinite because out-of-vocabulary tokens receive zero word probability. These are results for a fixed conditional
schedule and normalization, not a globally exact joint posterior or external
replication ([@sec:connected_gum]).

Comprehension filtering alone does not distinguish the system from classical
Bayesian state estimation. Every modeled
level has a flat `C` and a single action in comprehension, so no policy is scored
and no expected free energy is computed. L1's headline and the bounded L2
verification paths execute live filtering; that recursion is the forward
algorithm of a hidden Markov model, while the L2 full-set and later headline
scores use the exact or NumPy paths just enumerated. This is precisely what lets
a plain-NumPy HMM serve as the implementation-independent reference decoder at
the checked levels ([@sec:related]). The additional policy-selection mechanism is exercised in generation, where the letter transition is made controllable and the
lexicon supplies the only non-flat preference vector in the system, so the
emitted symbol is selected by minimizing expected free energy over policies
([@sec:generation]). Even that one place is narrower than it sounds, and saying
so is a result rather than a concession: because the controllable transition is
deterministic, the information-gain term is identically zero and policy
selection reduces analytically to a preference-weighted softmax over the habit
([@eq:policy_reduction]). The measurement is reported with a control that
restores the term to its analytic ceiling, so the claim is that the formalism
contributes nothing beyond that softmax *here*, not that it could not
elsewhere. A transition that is both stochastic and action-dependent would put
the term back and let it bear on the choice; the uniform-transition control
fixes the ceiling such a design works within, and the word-state construction
of [@sec:generation] shows that uncertainty without action-dependence buys
nothing. The budgeted-attention arm ([@sec:limitations]) already measures the
case where that term is the whole policy. The gate-audit findings that follow concern these filtering models and the
thresholds written for them; they are not a verdict on active inference as a
general framework.

The more transferable results are the gate-audit method and the negative
findings. A load-bearing modeling decision, which slot the learned matrix
occupies, was wrong in a way the library could not detect. `pymdp` validates
more than normalization: it checks array shapes against declared factor
dependencies, index bounds, and control consistency. What no structural check
reaches is semantics. A column-stochastic bigram is column-stochastic whichever
slot it occupies, and nothing in the library knows whether an axis was intended
to mean *next given current* or its transpose. The
wrong wiring was not loudly broken but *quietly anti-informative*, and it
slipped under a threshold that had never been checked against either a trivial
baseline or an achievable ceiling. Then the same class of oversight produced
two opposite gate defects: one threshold no correct implementation could ever
reach, one that a lookup table already cleared. The detection method is the
same in both directions, and it is cheap: measure what a model doing none of
the work scores, measure what the data allows, and accept a threshold only when
the first fails it and the second clears it.

That check is expected to generalize past this project and past active
inference, a working hypothesis in the sense of [@sec:introduction]: its
evidence is every declared gate of one repository, written and audited by one
author, and nothing outside it has yet been tested. It would apply to any
staged or gated build whose progress is defined by a numeric criterion:
accuracy floors in a test suite, latency budgets, coverage thresholds,
benchmark targets in a research program. The taxonomy of [@sec:gate_design]
needs only three measured quantities, two of which can usually be obtained
before the system exists, and the checklist there is written to be run before
the build rather than after the result. `scripts/audit_gates.py` is what makes
it a re-runnable artifact instead of something someone happened to do twice;
the same shape would fit any repository with declared thresholds.

The check was then turned on a threshold of our own, which is the harder half of
the same lesson. The benchmark originally refused to report a calibration number when too
little of a model's reported distribution was observed, and that floor was
declared on an argument rather than measured. Measuring where it belongs put it
at 1.9826100924184078 times the declared value, and at a mass the
serving interface cannot supply for 0.395 of
positions: the floor was not conservative, and no reachable gate of that shape
would be ([@sec:external_benchmarks]). The instructive part is how nearly the
measurement endorsed the wrong answer. Both metrics' rule disagreement crosses
zero once, essentially at the declared floor, while the two reconstructions are
still far apart; a locator that took the first crossing would have confirmed
0.5 by a coincidence of opposite biases. A threshold
that survives its own audit is not thereby correct, and the audit needs its own
negative control.

One reporting change follows from the same discipline. Every interval-spanning
paired result in this project is stated as an interval rather than as an absence,
because "no effect was found" is weaker than the data support and hides the
distinction that matters: no near-null arm in the plotted paired-bound roster
produced enough discordant pairs to support an equivalence claim, so each
bounds its effect and says nothing about absence ([@sec:bounds]). The one arm
that once did, the lexical channel at its highest tested weight, became a
decided effect when its backoff defect was corrected (decision
110), and the roster now holds bounds only. A
bounded effect and an unresolved one can read alike in a table; separating them
is the reporting result.

The project's other negative result is about its own headline architecture. An
inter-level message, L1's predictive over letters folded into L2B's
observation likelihood, was built rather than assumed, and it moved accuracy
by -0.2639 points against the uncoupled decode, a loss
significant at paired McNemar *p* = 1.547e-06. The channel works;
it simply carries nothing L2B does not already observe. Reporting that is the
point: the mechanism had been asserted in this project's own documentation
before anyone measured it, which is the same failure as an unfalsified
threshold, one layer up. By contrast, the L3→L2B lexical channel was
prospectively specified in the same development session, built, and measured
as helpful (3.333
pts), so the novel-information criterion (decisions 18--19) is falsifiable, and
on that pair — same receiving level, opposite senders — its prediction matched
the development-population result. Its next test went the other way. The privileged
punctuation-type-oracle→L3 word-prior intervention, which the criterion itself
named as its cheapest
falsification and predicted would harm, instead moved the word prior's
held-out perplexity slightly in the helpful direction throughout the tested
positive grid; the best tested weight is its high endpoint, not a located
optimum (3.268023 → 3.263007, decision 27 and
[@sec:clause_word_channel]); the token-direction tally agrees, with
51,364 improved against
22,649 made less probable. The predicted harm was
not supported, so the criterion's first clause is qualified on that channel.
One prospectively specified supporting result and one corrected result that
falsifies the criterion's prediction leave a criterion under test, not an
established one.

Three subsidiary lessons carried real cost here. A reproducible evaluation
subsample is not a representative one, and an unbiased subsample of the same
size can be too small to establish the effect it correctly measures. The
resolution was to remove the constraint that forced subsampling: require an
implementation-independent oracle and the pymdp decoder to agree on the bounded
comparison subset (0 disagreements), then compute
full-set statistics. A verification gate that runs only on synthetic fixtures constrains
nothing about the shipped artifact: the claims scanner had never been pointed
at real prose, and an unbacked claim sat in the repository's own README for an
entire stage while every test passed. A failure signature that catches the
mis-wirings you thought of is necessary but not sufficient: an emission
normalized along the wrong axis cleared the baseline comparison while being
wrong, and only a column-sum invariant and numerical agreement with the oracle
pinned it down.

The staged-gate method's own description was inaccurate, which is why
[@sec:introduction] states the qualification alongside the claim-strength labels
rather than leaving it to this section. The original Stage 1--4 gates were
restated after those levels had been built or measured because their thresholds
were vacuous, unreachable, or named the wrong population
([@tbl:gate_audit], [@tbl:unaudited_gates]). Later declarations were also
re-scoped during audit. Stage 6's direction was registered before its held-out
measurement, but the development population and analysis family were reused and
its clustered primary remains non-confirmatory. The repaired gates and controls
are more rigorous, but they do not turn the historical process into sequential
validation, and no multiplicity correction repairs population reuse. A rerun of
the frozen family against a population never inspected while the gates were
written was pre-registered, executed to its first data touch, and retired by
exhaustion for the `external_lexicon` members (decision
91); the `ud_ewt_test` members were then scored exactly once against the held-back EWT
test split under decision 111
([@sec:confirmatory]): two clear the frozen per-test bar in their recorded direction, two yield no
discordant pair, and the other twelve members of the family remain
development-population inference. The supplements that follow — the gate-audit
trails and the confirmatory replication, the channel factorial, operation
and byte counts, the registered external probes read as a reference frame
rather than a competitiveness claim, the machine-checked Lean layer, and the
statistical apparatus the paired results rest on ([@sec:audit_trails],
[@sec:channel_factorial], [@sec:complexity], [@sec:external_benchmarks],
[@sec:lean], [@sec:statistics]) — carry the detail behind the claims and
qualifications stated here. Naming
the defect is not the same as fixing it: the same honesty the method claims
for its measurements is owed for its own process.

What travels beyond this one repository is the pair the introduction opened
with: an explicit discrete generative language system whose probabilities,
preferences, habits and policy precision are inspectable, and a
gate-audit-and-claims-ledger method for holding
a staged build to its own thresholds. Neither depends on active inference to
be useful, and neither depends on this corpus, this language, or this
author's own gates having been the ones checked. The next test of either is a
build this project did not run: a second staged system, audited by someone
other than its builder.

What comes after this paper follows the same three commitments the abstract
opens with — a generative model declared in Generalized Notation Notation, an
Active Inference formalism that turns that declaration into filtering and
policy selection, and a grain of linguistic description the declaration is
built along — rather than abandoning any of them for scale. That combination
is what makes the open work in [@sec:future] proposable in the terms this
paper already uses: a new level, a new language, or a new cross-implementation
render is another GNN declaration and another conformance check, not a
different kind of artifact; a globally normalized hierarchy would require a new joint specification and
fresh comparisons beyond the conditional connected runtime measured here; and a composition with a large language model, in either direction,
is answerable to the same discipline that kept this project's own fitted-probe
proposal from certifying its own circularity. None of that is claimed to have
happened. It is what the method this paper reports is for.
