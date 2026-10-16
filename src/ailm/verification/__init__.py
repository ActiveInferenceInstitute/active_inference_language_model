"""Independent verification instruments for AILM.

oracle: pure-numpy HMM (forward filtering, forward-backward smoothing, Viterbi,
        perplexity). Deliberately imports **neither pymdp nor jax** — its whole
        value is being an implementation that cannot inherit the flaw it audits
        for. A hand-rolled version of this caught the Stage 1 anti-informative
        wiring bug (bigram matrix in the emission A instead of the transition B),
        which pymdp validated happily in either slot.

Use it to cross-check any pymdp level numerically against a closed form, never
structurally.
"""

from __future__ import annotations

from ailm.verification.oracle import (
    forward_backward,
    forward_filter,
    identity_emission,
    markov_chain_perplexity,
    sequence_perplexity,
    viterbi,
)

__all__ = [
    "forward_backward",
    "forward_filter",
    "identity_emission",
    "markov_chain_perplexity",
    "sequence_perplexity",
    "viterbi",
]
