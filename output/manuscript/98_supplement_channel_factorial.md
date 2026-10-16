# Supplement B: Channel Factorial and Proper-Loss Re-Scoring {#sec:channel_factorial}

[@sec:hierarchy] reports the measured cross-level interventions and the
novel-information criterion that organizes their disagreeing signs. Two of
those channels — the letter-predictive and lexical-prefix messages into the
letter-to-phone decoder — were swept at one insertion point each, which
confounds *which message* was sent with *where* it entered the receiver. This
supplement carries the exploratory extension built to separate the two
confounds: a factorial crossing message source with insertion point under a
shared executor, a control that varies the combination rule itself, a fitted
variant that replaces four hand-set values with measured ones, and a
re-scoring of every cell on the receiver's own proper predictive loss rather
than on accuracy alone. None of it revises [@sec:hierarchy]'s reported signs;
it explains why the two swept channels disagreed and how much of that
disagreement is attributable to the mixing rule rather than to the message.

These are historical receiver experiments. They do not decompose the later
connected GUM result into per-channel gains: that comparison switches its named
channels together under a different population and scoring schedule.

## The channel factorial separates message source from insertion point

The sweeps above differ in message source *and* in insertion point, so the
gap between them attributes to neither. The crossed combinations were built with
one executor dispatching into the same shipped primitives, gated before use by
requiring it to reproduce both existing cells' per-position hit vectors exactly
rather than their accuracies, over all 42,820
positions. Each cell reports its best *engaged* weight, never zero, because zero
weight is the uncoupled anchor and a harmful channel maximized over a grid
containing zero would report the anchor and appear to do nothing.
[@tbl:channel_factorial] gives the complete factorial.

**Every weight in this section is selected on the positions its p-value is
computed over, and that is a defect in the inferential reading rather than in
the measurement.** Each cell's reported weight is the argmax of a grid scored on
the held-out split, and the accompanying paired test is then computed on that
same split at that same weight. Optimizing a criterion on a finite sample and
reporting the optimized value against it biases the estimate optimistically, by
an amount that grows with the grid and with the noise in the criterion, and no
multiplicity correction repairs it: Bonferroni bounds a declared family of
tests, and this is a selection made before any of them was declared. The
margins are therefore upper-leaning, and the p-values are smaller than a
selection-aware procedure would return.

The exception is the fitted arm below, which is the one place the separation is
made: its 81-point grid is scored against an inner split of
the *training* set, the held-out positions are scored once after selection
closes, and the recorder refuses to run if any selection word appears in the
evaluation half. Its gain on the selection split
(1.5958 points) and on the held-out set
(1.4736) being close is the evidence that the fit transferred
rather than memorized. Reading the two together is the honest comparison
available here: the sweep arms show what a channel can reach when its weight is
chosen with hindsight, and the fitted arm shows what one reaches when it is not.

| Message source | Insertion point | Weight | Accuracy | Margin | Bound at 95% | *b/c* | *p* |
|---|---|---:|---:|---:|---:|---:|---:|
| Letter (L1) | Evidence | 0.1 | 70.5371 | -0.2639 | [-0.3693, -0.1556] | 217/330 | 1.547e-06 |
| Letter (L1) | Predictive | 0.1 | 70.9038 | 0.1028 | [-0.0414, 0.2463] | 502/458 | 0.1652 |
| Lexicon (L3) | Evidence | 0.2 | 70.8034 | 0.0023 | [-0.108, 0.1126] | 280/279 | 1 |
| Lexicon (L3) | Predictive | 1 | 74.1453 | 3.3442 | [3.0216, 3.6631] | 3333/1901 | 3.598e-88 |

: The channel factorial, with every cell compared against the uncoupled anchor of 70.801% on 42,820 positions. Margins are percentage points; a negative margin is a harm. Each margin carries an interval on the same scale, derived at render time from that cell's own discordant counts and the shared denominator, so the size of an effect is readable beside the question of whether it cleared a threshold — and the letter-at-predictive and lexicon-at-evidence cells are the ones whose intervals span zero and whose *p* does not reach it. The intervals are marginal rather than simultaneous across the factorial. The letter-at-evidence and lexicon-at-predictive diagonal cells are the sweeps reported above and exist here to gate the executor: it reproduces their per-position hit vectors exactly, so any disagreement would mean the factorial was measuring a reimplementation rather than the channels. The design is falsified if a cell's recomputed margin disagrees with its recorded pair, if either diagonal cell fails to match the shipped decoder position for position, or if the zero-weight outcomes are not byte-identical to each other and to the uncoupled decode. The letter-at-evidence and lexicon-at-predictive diagonal cells restate the coupling and lexical hypotheses reported above, searched over this table's own weight grid on the same positions against the same anchor, so they sit outside the declared significance family by recorded exclusion: the family counts hypotheses rather than weight choices. Where a diagonal cell's selected weight lies on both grids it still reproduces its family member exactly, which is the case at 0.1 for letter-at-evidence. Where the grids reach different weights it does not: lexicon-at-predictive is reported here at 1 and above at 0.7, and the counts and *p*-values differ accordingly. {#tbl:channel_factorial}

**The evidence site harms the letter message and does nothing detectable for
the lexical one.** The lexical message buys
3.3442 points at the predictive site
and moves accuracy by 0.0023 at the
evidence site (*p* = 1), which resolves
neither way. Same message, same receiver, same weight grid: a gain at one site
and nothing detectable at the other, and the only thing that differs is where
it enters. Before the backoff correction of decision
110 that cell recorded a significant cost and this
paragraph read "both evidence-site cells harm"; the corrected cell is the null
the table now shows. The letter message at the predictive site is the other
cell that resolves neither way. Every cell of this table is also re-scored on
the receiver's proper predictive loss in [@tbl:channel_proper_loss] — the axis the
working-hypothesis box of [@sec:proper_loss_axis] declares, on which the accuracies here are the secondary
readout — and the axes are compared cell for cell there.

The decomposition follows. Message source is worth
0.2662 points at the evidence site and
3.2415 at the predictive one; insertion
point is worth 0.3667 for the letter source and
3.3419 for the lexical one. The interaction is
2.9752 points against a diagonal gap of
3.6081, so most of what the two sweeps appeared to say
about message content is carried by where the message was inserted.

The mechanism is legible after the fact and was not predicted here in advance,
which is why it was worth measuring. The evidence site mixes a message into the
observation likelihood, which holds the model's only hard information about the
current position; softening it trades certain evidence for a prior, and that
trade is bad however good the prior. The predictive site mixes into the state
prior, which is where a prior belongs.

That reading is correct and narrower than it first appears, and the narrowing is
worth stating because it was found by asking what the factorial held fixed. Every
factorial cell shares one *combination rule*: linear mixing. Under geometric pooling —
a product of experts [@hinton2002training], and the rule the underlying library uses when it combines
modalities — the evidence and predictive sites are provably the same computation, since multiplication
is associative and the intermediate normalization cancels. Measured on the shipped
matrices at an exponent of 0.5 the two sites agree to
9.99e-16 with
0 argmax disagreements anywhere. So what this table establishes is not that the insertion point governs a
message's sign, but that *linear mixing makes the insertion point govern it*: the
rule trades away a certainty in proportion to the weight, and no message repays
that trade. A factorial cannot see the factor every one of its cells holds fixed,
and naming that factor is the more transferable result. The consequence for the working hypothesis
of [@sec:hierarchy] is a qualification rather than a retraction: the novel-information
criterion predicted the L3→L2B pair correctly, and the factorial shows the
prediction would have found nothing at the other insertion point with the same
message.

### Combination-rule controls {#sec:pooling_rule}

Varying the rule settles the question the two earlier readings could not, and it
does so through a control rather than through a better number. Under geometric
pooling a *uniform* message is inert by construction, since a constant raised to
any power cancels in normalization. Measured, it returns the anchor to every
recorded digit at every exponent on zero discordant pairs, a margin of
0 points. Under linear mixing the same uniform
message moves posteriors materially, which is why any linear result mixes
information with regularization and the two can be separated only by argument.

With dilution structurally excluded, the remaining effects are content. The
lexical message gains 5.3573 points at exponent
1, with a 95% bound of
[5.1321, 5.5741]
points on 3704 discordant pairs
(*p* = 0 as computed; the exact value lies below
double-precision range); the
scrambled control loses at every exponent, widening to
-7.9659 points at full weight, so the
gain is correspondence to the phone inventory and not smoothing. And the letter
message *still harms*, moving accuracy -0.1868 points at its best
exponent — bounded at [-0.3177,
-0.0545] points, an interval entirely below zero, so
the harm is small but its direction is not in doubt — and
-2.1579 at full weight,
monotonically worse the more it is weighted. Monotone degradation under a rule
that cannot dilute is content rather than arithmetic.

That is the working hypothesis's original claim, established at last on evidence
rather than on a plausible story: the letter message carries nothing the
letter-to-phone level does not already observe. Reaching it took one
reinterpretation that was wrong about the site, one derivation showing the site
effect was itself a rule artifact, and a third rule whose control can tell
dilution from content.

### Train-fitted channel parameters {#sec:fitted_channel}

Four numbers governing this gain were nobody's measurement: the message's
`max_prefix`, `min_count` and `smoothing`, all set by hand, and the pooling
exponent read off a declared grid. They are now fitted, on a grid of
81 points selected against an inner split of the training
set — 20,748 pairs to fit and 5188 to
select on — with the matrices for selection built from the fit half alone. The
held-out set is scored once, after selection closes, and the recorder refuses to
run if any selection word appears in it. Choosing a value on the positions the
headline is reported on is the likeliest way an arm like this manufactures a gain,
so the guard is code rather than intention.

The fitted configuration scores 76.0766% against the hand-chosen
74.603% and the uncoupled 70.801%, a
paired gain of 1.4736 points, 95% bound
[1.3281, 1.6121], on
941/310 discordant pairs
(*p* = 3.172e-74) and 5.2756 points above the
uncoupled decode. That the gain on the selection split
(1.5958 points) and on the held-out set (1.4736) are
close is the evidence that the fit transferred rather than memorized; both are
reported because a large selection gain beside a small held-out one is the
opposite finding and would be invisible if only the second were shown.

4 of the 4 values moved. The
selection set the longest conditioned prefix at 4 letters,
admitted prefixes seen at least 2 times, set the pseudocount
at 0.5, and set the pooling exponent at 0.75
against the grid-selected 1 — which is to say
it trusts sparse short contexts more, and the longest contexts and the full
message weight less, than the hand-chosen configuration did. The fitted
configuration is *not* adopted as the shipped default here: doing so in the same
revision that first measures it would leave no version of this paper in which the
declared and the measured configuration can be compared, which is the reason
[@sec:gate_design] gives for deferring the calibration gate.

On accuracy the geometric rule leads — it peaks at
76.1583% against linear's
74.1453% — but both weights were selected
on the evaluation positions, so nothing here recommends changing the shipped
decoder, and it is unchanged. What the rule buys is not
performance but the ability to say which of two explanations a measurement
supports.

## Factorial cells under proper predictive loss

One cell is measured on both axes over a full weight grid. The
letter-at-evidence cell of
[@tbl:channel_factorial] is re-scored as the receiver's proper predictive loss,
$-\log q_t(y_t)$ at the gold phone, over the same
42,820 held-out positions
(6484 words) that produced its accuracy. The uncoupled
anchor costs 0.927068 nats per position, and no
engaged weight improves on it. Writing the realized gain as anchor minus
weighted loss, so that a positive number would be an improvement, the best of
the 4 non-zero weights scores
-0.035174 nats at weight
0.05 and the worst
-0.454083: every one of them is a cost. The decline is monotone in the
weight, and the accuracy on the identical positions declines monotonically too
(-0.0981 points at weight
0.05, a weight on this re-scoring grid that the
factorial grid does not contain,
against the uncoupled 70.801%). The two axes
agree in sign and in shape class. For this cell the diagnostic is a property of
the channel and not of the statistic — which is a result about the diagnostic,
established by a measurement that could have gone the other way.

The oracle-sender arm says why the decline is structural rather than incidental,
and it supplies the hypothesis box's own *sensitivity* term. Replace the L1
message with a perfect one — a point mass on the letter actually at that
position — and the receiver's loss does not move at all: the spread across every
weight is 0 nats, exactly, because the
mixture $(1-\alpha)\,\delta_{x_t} + \alpha\,\delta_{x_t}$ is $\delta_{x_t}$ for
every $\alpha$. The receiver's sensitivity to the variable this message carries
is 0: it already observes the letter, so
no message about the letter can be worth anything to it, however well
calibrated. With the information term at zero, nothing remains on the positive
side of the decomposition of [@sec:hierarchy]: a message about an already-observed variable
cannot contribute conditional information, so the only outcomes available were
"no effect" and the mismatch term going the wrong way, which is harm.
Mixing a distribution over a variable the receiver observes directly can only
spread the evidence, and the sweep is the size of that cost.

The remaining linear cells of [@tbl:channel_factorial], and the geometric and
fitted non-diluting lexical arms, are now on the same axis. Each is re-scored as held-out
per-position log loss over the identical 42,820
positions, with the channel engaged at its recorded weight — a point
re-scoring, not a sweep, so it speaks to each cell's *sign* on the declared
axis and not to its shape class, which only the letter-at-evidence cell has on
both axes. The loss is the primary column of [@tbl:channel_proper_loss] and the
accuracy the secondary one, because the loss is the axis the hypothesis box
declares and the accuracy is the statistic the rest of this section reports.
For every linear-sweep cell the engaged weight was selected on the same held-out
split the loss is reported on, so these loss numbers inherit, unchanged, the
selection caveat stated at the head of [@sec:channel_factorial] for the
accuracies, and the ledger records each of them with an explicit exploratory
status. The fitted arm's weight came from an inner split of the training set
(decision 49's leak guard), and it is the one row whose weight the evaluation
set did not choose.

| Cell | Rule | Weight | Loss (nats) | Gain over anchor (nats) | Accuracy (%) | Weight chosen on |
|---|---|---:|---:|---:|---:|---|
| Letter at evidence | linear | 0.1 | 1.001154 | -0.074085 | 70.5371 | held-out split |
| Letter at predictive | linear | 0.1 | 0.920421 | 0.006647 | 70.9038 | held-out split |
| Lexicon at evidence | linear | 0.2 | 0.960078 | -0.03301 | 70.8034 | held-out split |
| Lexicon at predictive | linear | 1 | 0.842033 | 0.085036 | 74.1453 | held-out split |
| Lexicon, geometric | geometric | 1 | 0.832156 | 0.094912 | 76.1583 | held-out split |
| Lexicon, fitted | geometric | 0.75 | 0.810529 | 0.116539 | 76.0766 | inner train split |

: The channel cells on the receiver's proper loss, all against the uncoupled
anchor of 0.927068 nats per position on
42,820 positions. The gain column is anchor minus
loss, so a positive number is an improvement; loss is the primary axis here and
accuracy the secondary. Every row's loss and accuracy are read off the same
posterior rows of the same recursion that produced that cell's recorded hit
vector, and the recorder refuses to write if any row's accuracy fails to
reproduce its recorded accuracy cell, if the position count moves, or if the
three anchor recursions (linear at either site at weight zero, geometric at
exponent zero) disagree by any amount — all three claim to *be* the uncoupled
decode. Rows whose weight is marked "held-out split" carry test-selected
weights and are exploratory; only the fitted row's weight was fixed without
sight of these positions. The design is falsified if a row's recorded gain
disagrees with its anchor-minus-loss arithmetic, or if a row's sign disagrees
between this table and its accuracy cell without the text saying so.
{#tbl:channel_proper_loss}

**Every decided sign agrees across both axes, and the undecided cells are
named.** The lexical message helps at the predictive site on both axes,
under the linear rule and the geometric rule and the fitted configuration; the
letter message harms at the evidence site on both axes. The cells that
resolve neither way on accuracy lean in opposite directions on the loss axis,
and neither lean is a result: the letter-at-predictive cell leans the same
small positive way (0.006647
nats beside +0.1028 points), and the
lexicon-at-evidence cell leans to a small cost
(-0.03301 nats beside
0.0023 points), which the text records,
as the table's caption requires, rather than resolves. A point re-scoring at a
test-selected weight attaches no test, so nothing here promotes either. The
research-queue falsifier — a sign disagreement between decided cells that
would make the section's diagnostic a property of the argmax statistic — fires
on none of the reported configurations. On this axis the fitted geometric arm
leads the linear lexicon-at-predictive cell
(0.116539 against
0.085036 nats), as it does on
accuracy, and it is the one row whose weight the evaluation set did not choose,
so its lead is the comparison here that a selection caveat does not qualify.

What this extension does not do is also worth separating. It is not the
measurement [@sec:results] names as missing: $I(Y; M \mid X)$ is a property of
the message independent of any insertion rule, and every row is still an
end-to-end intervention at a fixed combination rule. It attaches no inferential
apparatus — these are descriptive re-scorings of decided and undecided cells
alike, and no p-value moves. And it reaches only the channels that share the
G2P receiver: the punctuation-type-oracle→word-prior intervention is already scored on its own held-out
log loss ([@sec:clause_word_channel]), while the generation channel (scored on
total-variation distance and lexicon membership) and the L6→L5 prior intervention
remain off this axis and would each need their own construction to join it.
