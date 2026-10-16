"""Belief dynamics: one sentence traced through the real inference stack.

``fig:belief_dynamics``
    A single UD EWT dev sentence — chosen at module level, verified at build
    time against every artifact it must exist in — pushed through the *shipped*
    inference paths at five levels, with the per-position belief state drawn as
    a heatmap column per position:

    - **L1 letters**: the pymdp letter agent's *predictive prior* per position.
      The posterior is deliberately not drawn: the letter emission is the
      identity, so the posterior collapses onto the observed letter and a
      posterior panel would be a featureless one-hot stripe. The prior is where
      the learned bigram dynamics live.
    - **L2B grapheme→phoneme**: the pymdp G2P agent's phone posterior per
      letter of one sentence word, the same filtering loop
      :func:`ailm.levels.phoneme.decode_phonemes` argmaxes over — and the
      module's cross-check: the float64 numpy oracle
      (:func:`ailm.verification.oracle.forward_filter`) must reproduce these
      posteriors to float32 tolerance or the figure refuses to draw.
    - **L3 words**: the trie-constrained word posterior narrowing letter by
      letter, the same loop :func:`ailm.levels.word.posterior_word_support`
      thresholds.
    - **L4 case roles**: the forward-filtered role posterior per token
      (numpy — the shipped Stage 4 decoder is numpy), cross-checked against
      the oracle at float64 exactness.
    - **L5 clause type**: ``P(interrogative | prefix)`` as the case-role
      prefix grows, through :func:`ailm.levels.clause.sequence_log_evidence`.

    A bottom panel overlays each level's per-position belief entropy (nats),
    which is the narrowing dynamics stated quantitatively.

Every matrix comes from the shipped builders; every posterior from the shipped
inference loops; no number in this module is typed in. The diagnostics block —
jax version and backend, the pymdp Agent class, both cross-check deviations —
is computed at generation time and drawn onto the figure.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from ailm.figures._common import save_figure
from ailm.viz_config import FONTS, PALETTE, SEQUENTIAL_CMAP, apply_style

__all__ = [
    "SENTENCE",
    "belief_dynamics_data",
    "build_belief_dynamics_figure",
    "generate_belief_dynamics",
]

#: The traced sentence: UD EWT dev "impact of the war on the american people?",
#: an interrogative. Chosen because it is one of the few dev sentences whose
#: every word is in the Stage 3 vocabulary AND whose every UPOS tag is in the
#: Stage 4 observation alphabet — ``belief_dynamics_data`` re-verifies both at
#: build time and raises rather than substituting.
SENTENCE: tuple[tuple[str, str], ...] = (
    ("impact", "NOUN"),
    ("of", "ADP"),
    ("the", "DET"),
    ("war", "NOUN"),
    ("on", "ADP"),
    ("the", "DET"),
    ("american", "ADJ"),
    ("people", "NOUN"),
)

#: The sentence is punctuated as a question; L5's panel shows the interrogative
#: posterior drifting below the declarative-heavy prior as this prefix grows.
#: This is one illustrative trajectory, not evidence for Stage 5's aggregate
#: non-passing/unresolved outcome.
SENTENCE_IS_INTERROGATIVE: bool = True

_CROSSCHECK_TOL: float = 1e-4  # float32 pymdp vs float64 oracle


@lru_cache(maxsize=1)
def _letter_setup() -> tuple[Any, Any]:
    from ailm.levels.letter import build_letter_agent, build_letter_transition, load_brown_counts

    a = build_letter_transition(load_brown_counts())
    return build_letter_agent(a), a


@lru_cache(maxsize=1)
def _g2p_setup() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    from ailm.levels.phoneme import (
        build_g2p_agent,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        g2p_split,
        load_phoneme_counts,
    )

    counts = load_phoneme_counts()
    train, _test = g2p_split(counts)
    emission = build_g2p_emission(train)
    transition = build_g2p_transition(train)
    start = build_g2p_start(train)
    agent = build_g2p_agent(emission, transition, start)
    pair_map = {word: phones for word, phones in train}
    return (
        agent,
        np.asarray(emission, dtype=np.float64)[0],
        np.asarray(transition, dtype=np.float64)[0, :, :, 0],
        np.asarray(start, dtype=np.float64)[0],
        pair_map,
    )


@lru_cache(maxsize=1)
def _word_setup() -> tuple[Any, Any]:
    from ailm.levels.word import build_word_agent, load_word_artifact

    artifact = load_word_artifact()
    return build_word_agent(artifact), artifact


def _letter_prior_track(text: str) -> np.ndarray:
    """Predictive prior per position from the real pymdp letter agent.

    Mirrors the loop in :func:`ailm.levels.letter.compute_perplexity` — same
    agent, same ``infer_states`` / ``update_empirical_prior`` calls — but keeps
    the prior at each position instead of only the scalar likelihood.
    """
    import jax.numpy as jnp

    from ailm.levels.letter import ALPHA_TO_IDX, N_SYMBOLS, normalize_text

    agent, _a = _letter_setup()
    chars = [c for c in normalize_text(text) if c in ALPHA_TO_IDX]
    empirical_prior = [jnp.ones((1, N_SYMBOLS)) / N_SYMBOLS]
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    priors = np.empty((N_SYMBOLS, len(chars)), dtype=np.float64)
    for t, char in enumerate(chars):
        priors[:, t] = np.asarray(empirical_prior[0]).ravel()
        qs = agent.infer_states([jnp.array([ALPHA_TO_IDX[char]])], empirical_prior)
        empirical_prior = agent.update_empirical_prior(action, qs)
    return priors


def _g2p_posterior_track(word: str) -> tuple[np.ndarray, float]:
    """Phone posterior per letter from the real pymdp G2P agent, cross-checked.

    The pymdp filtering loop is the one :func:`decode_phonemes` argmaxes over;
    the float64 oracle recomputes the same posteriors independently and the
    maximum absolute disagreement is returned. A disagreement beyond float32
    tolerance raises — the figure must never show beliefs the shipped decoder
    does not hold.
    """
    import jax.numpy as jnp

    from ailm.levels.letter import ALPHA_TO_IDX
    from ailm.levels.phoneme import N_PHONES
    from ailm.verification.oracle import forward_filter

    agent, emission, transition, start, _pairs = _g2p_setup()
    letters = list(word)
    empirical_prior = [jnp.asarray(agent.D[0])]
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    posteriors = np.empty((N_PHONES, len(letters)), dtype=np.float64)
    for t, letter in enumerate(letters):
        qs = agent.infer_states([jnp.array([ALPHA_TO_IDX[letter]])], empirical_prior)
        posteriors[:, t] = np.asarray(qs[0]).ravel()
        empirical_prior = agent.update_empirical_prior(action, qs)

    obs = [ALPHA_TO_IDX[letter] for letter in letters]
    oracle_posteriors, _log_evidence = forward_filter(emission, transition, start, obs)
    deviation = float(np.max(np.abs(oracle_posteriors.T - posteriors)))
    if deviation > _CROSSCHECK_TOL:
        raise ValueError(
            f"pymdp and the float64 oracle disagree on the G2P posteriors for "
            f"{word!r}: max |delta| = {deviation:.3e} > {_CROSSCHECK_TOL:.0e}. "
            "The figure refuses to draw beliefs the shipped decoder does not hold."
        )
    return posteriors, deviation


def _word_posterior_track(word: str) -> tuple[np.ndarray, Any]:
    """Word-identity posterior per observed letter from the real pymdp agent.

    The loop is :func:`ailm.levels.word.posterior_word_support`'s, kept per
    position instead of thresholded at the end.
    """
    import jax.numpy as jnp

    agent, artifact = _word_setup()
    position_start = np.zeros(artifact.n_positions, dtype=np.float32)
    position_start[0] = 1.0
    prior = [
        jnp.array(np.asarray(agent.D[0]).reshape(1, -1)),
        jnp.array(position_start[np.newaxis]),
    ]
    track = np.empty((len(artifact.vocab), len(word)), dtype=np.float64)
    for t, letter in enumerate(word):
        qs = agent.infer_states([jnp.array([ord(letter) - 97])], prior)
        track[:, t] = np.asarray(qs[0]).ravel()
        prior = agent.update_empirical_prior(jnp.array([[0, 0]]), qs)
    return track, artifact


def _case_posterior_track(
    upos_sequence: list[str],
) -> tuple[np.ndarray, list[int], float]:
    """Role posterior per token via the shipped Stage 4 filter, oracle-checked.

    The recursion is exactly :func:`ailm.levels.case.decode_case_positions`'s;
    the oracle recomputes it in one call and must agree to float64 exactness,
    because both sides are float64 numpy.
    """
    from ailm.levels.case import (
        build_case_emission,
        build_case_start,
        build_case_transition,
        load_case_artifact,
    )
    from ailm.verification.oracle import forward_filter

    artifact = load_case_artifact()
    emission = build_case_emission(artifact)
    transition = build_case_transition(artifact)
    start = build_case_start(artifact)
    observations = [artifact.upos_to_idx[u] for u in upos_sequence]

    beliefs = np.empty((emission.shape[1], len(observations)), dtype=np.float64)
    belief = start * emission[observations[0], :]
    belief = belief / belief.sum()
    beliefs[:, 0] = belief
    for position in range(1, len(observations)):
        belief = (transition @ belief) * emission[observations[position], :]
        belief = belief / belief.sum()
        beliefs[:, position] = belief

    oracle_posteriors, _log_evidence = forward_filter(emission, transition, start, observations)
    deviation = float(np.max(np.abs(oracle_posteriors.T - beliefs)))
    if deviation > 1e-10:
        raise ValueError(
            f"the Stage 4 filter and the oracle disagree: max |delta| = "
            f"{deviation:.3e}. Both are float64; any disagreement is a bug."
        )
    roles = [int(np.argmax(beliefs[:, t])) for t in range(beliefs.shape[1])]
    return beliefs, roles, deviation


def _clause_type_track(case_sequence: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """``P(type | prefix)`` for growing prefixes, plus the static type prior."""
    from ailm.levels.clause import build_type_models, load_clause_artifact, sequence_log_evidence

    models = build_type_models(load_clause_artifact().train)
    track = np.empty((len(models["type_prior"]), len(case_sequence)), dtype=np.float64)
    for t in range(len(case_sequence)):
        _evidence, posterior = sequence_log_evidence(case_sequence[: t + 1], models)
        track[:, t] = posterior
    return track, np.asarray(models["type_prior"], dtype=np.float64)


def _entropy(track: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of each column of a distributions-by-position track."""
    clipped = np.clip(track, 1e-300, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=0)


def belief_dynamics_data() -> dict[str, Any]:
    """Run the sentence through all five shipped inference paths.

    Returns:
        Everything the figure draws: the per-level belief tracks, the words the
        panels focus on, both cross-check deviations, and the runtime
        diagnostics (jax version/backend, pymdp Agent class).

    Raises:
        ValueError: If the sentence stops satisfying any artifact it depends
            on (vocabulary, UPOS alphabet, G2P pair set), or if a cross-check
            fails. Substituting a different sentence silently would change
            what every panel shows.
    """
    import jax

    words = [word for word, _upos in SENTENCE]
    upos = [tag for _word, tag in SENTENCE]

    _agent, word_artifact = _word_setup()
    missing = [w for w in words if w not in set(word_artifact.vocab)]
    if missing:
        raise ValueError(f"sentence words {missing} are not in the Stage 3 vocabulary")
    from ailm.levels.case import load_case_artifact

    case_artifact = load_case_artifact()
    bad_tags = [u for u in upos if u not in case_artifact.upos_to_idx]
    if bad_tags:
        raise ValueError(f"sentence UPOS {bad_tags} are not in the Stage 4 alphabet")

    _g2p_agent, _e, _t, _s, pair_map = _g2p_setup()
    g2p_word = next((w for w in words if w in pair_map), None)
    if g2p_word is None:
        raise ValueError("no sentence word is in the length-matched G2P training set")
    focus_word = max(words, key=len)

    letter_track = _letter_prior_track(" ".join(words))
    g2p_track, g2p_deviation = _g2p_posterior_track(g2p_word)
    word_track, _artifact = _word_posterior_track(focus_word)
    case_track, case_roles, case_deviation = _case_posterior_track(upos)
    clause_track, type_prior = _clause_type_track(case_roles)

    from pymdp.agent import Agent as PymdpAgent

    return {
        "words": words,
        "upos": upos,
        "g2p_word": g2p_word,
        "focus_word": focus_word,
        "letter_track": letter_track,
        "g2p_track": g2p_track,
        "word_track": word_track,
        "word_vocab": list(word_artifact.vocab),
        "case_track": case_track,
        "case_roles": case_roles,
        "clause_track": clause_track,
        "type_prior": type_prior,
        "g2p_crosscheck_max_abs_diff": g2p_deviation,
        "case_crosscheck_max_abs_diff": case_deviation,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "pymdp_agent_class": f"{PymdpAgent.__module__}.{PymdpAgent.__name__}",
    }


def _heat(ax: Any, track: np.ndarray, vmin: float = 1e-4) -> Any:
    return ax.imshow(
        track,
        cmap=plt.get_cmap(SEQUENTIAL_CMAP).with_extremes(under="white"),
        norm=LogNorm(vmin=vmin, vmax=1.0),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )


def build_belief_dynamics_figure() -> Figure:
    """Build ``fig:belief_dynamics``. The caller owns the figure."""
    from ailm.levels.case import CASES
    from ailm.levels.clause import CLAUSE_TYPES
    from ailm.levels.letter import ALPHA_LIST, normalize_text
    from ailm.levels.phoneme import PHONES

    apply_style()
    data = belief_dynamics_data()
    sentence_text = " ".join(data["words"])
    chars = [c for c in normalize_text(sentence_text)]

    fig = plt.figure(figsize=(15.0, 12.6))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.35, 1.0, 0.75),
        width_ratios=(1.0, 1.0, 1.0),
        hspace=0.52,
        wspace=0.42,
        left=0.065,
        right=0.92,
        top=0.925,
        bottom=0.105,
    )

    # ── L1: predictive prior over the whole sentence ──────────────────────────
    ax1 = fig.add_subplot(grid[0, :])
    heatmap = _heat(ax1, data["letter_track"])
    # Label only every 2nd symbol: 30 labels at this panel height are vertically
    # crushed. The odd-index ticks stay as unlabeled minor ticks so every row
    # keeps a tick mark.
    even = range(0, len(ALPHA_LIST), 2)
    ax1.set_yticks(list(even))
    ax1.set_yticklabels(
        ["␣" if ALPHA_LIST[i] == " " else ALPHA_LIST[i] for i in even],
        fontsize=FONTS["tick_label"] - 1,
    )
    ax1.set_yticks(range(1, len(ALPHA_LIST), 2), minor=True)
    ax1.set_xticks(range(len(chars)))
    ax1.set_xticklabels(["␣" if c == " " else c for c in chars], fontsize=FONTS["tick_label"])
    ax1.set_xlabel("observed letter at position $t$", fontsize=FONTS["axis_label"])
    ax1.set_ylabel("next-letter prior", fontsize=FONTS["axis_label"])
    # The terminal '?' is punctuation outside the letter alphabet, so it is not
    # among the drawn positions; the suffix says so whenever the suptitle shows it.
    q_suffix = (
        " (final '?' outside the letter alphabet, not scored)" if SENTENCE_IS_INTERROGATIVE else ""
    )
    ax1.set_title(
        f"L1 — pymdp letter agent, predictive prior $P(s_t \\mid o_{{<t}})$ "
        f"over {len(chars)} positions\n"
        f"(posterior omitted: identity emission makes it one-hot){q_suffix}",
        fontsize=FONTS["title"],
    )

    # ── L2B: phone posterior for the G2P word ─────────────────────────────────
    ax2 = fig.add_subplot(grid[1, 0])
    g2p_track = data["g2p_track"]
    keep = np.where(g2p_track.max(axis=1) > 0.05)[0]
    rest = np.clip(1.0 - g2p_track[keep].sum(axis=0), 0.0, 1.0)
    shown = np.vstack([g2p_track[keep], rest[np.newaxis, :]])
    _heat(ax2, shown)
    ax2.set_yticks(range(len(keep) + 1))
    ax2.set_yticklabels([PHONES[i] for i in keep] + ["(rest)"], fontsize=FONTS["tick_label"])
    ax2.set_xticks(range(len(data["g2p_word"])))
    ax2.set_xticklabels(list(data["g2p_word"]), fontsize=FONTS["tick_label"])
    ax2.set_title(
        f"L2B — letter-to-phone posterior,\n'{data['g2p_word']}' (pymdp, oracle-checked)",
        fontsize=FONTS["title"],
    )
    ax2.set_ylabel("phone (mass > 0.05)", fontsize=FONTS["axis_label"])

    # ── L3: word posterior narrowing ──────────────────────────────────────────
    ax3 = fig.add_subplot(grid[1, 1])
    word_track = data["word_track"]
    vocab = data["word_vocab"]
    top = np.argsort(-word_track.max(axis=1))[:12]
    rest_mass = np.clip(1.0 - word_track[top].sum(axis=0), 0.0, 1.0)
    shown_words = np.vstack([word_track[top], rest_mass[np.newaxis, :]])
    _heat(ax3, shown_words)
    ax3.set_yticks(range(len(top) + 1))
    ax3.set_yticklabels([vocab[i] for i in top] + ["(rest of vocab)"], fontsize=FONTS["tick_label"])
    ax3.set_xticks(range(len(data["focus_word"])))
    ax3.set_xticklabels(list(data["focus_word"]), fontsize=FONTS["tick_label"])
    ax3.set_title(
        f"L3 — trie word posterior,\n'{data['focus_word']}' letter by letter (pymdp)",
        fontsize=FONTS["title"],
    )

    # ── L4: case-role posterior per token ─────────────────────────────────────
    ax4 = fig.add_subplot(grid[1, 2])
    _heat(ax4, data["case_track"])
    ax4.set_yticks(range(len(CASES)))
    ax4.set_yticklabels([role.name for role in CASES], fontsize=FONTS["tick_label"])
    ax4.set_xticks(range(len(data["words"])))
    ax4.set_xticklabels(
        [f"{w} · {u}" for w, u in zip(data["words"], data["upos"], strict=True)],
        fontsize=FONTS["tick_label"] - 1,
        rotation=90,
    )
    ax4.set_title(
        "L4 — case-role posterior per token\n(Stage 4 filter, oracle-exact)",
        fontsize=FONTS["title"],
    )
    fig.colorbar(
        heatmap,
        ax=[ax1, ax2, ax3, ax4],
        fraction=0.02,
        pad=0.03,
        label="posterior probability",
    )

    # ── L5: clause-type posterior on growing prefixes ─────────────────────────
    ax5 = fig.add_subplot(grid[2, 0:2])
    clause_track = data["clause_track"]
    interrogative = CLAUSE_TYPES.index("interrogative")
    positions = np.arange(1, clause_track.shape[1] + 1)
    ax5.plot(
        positions,
        clause_track[interrogative],
        marker="o",
        color=PALETTE["MODEL"],
        linewidth=2.0,
        label="P(interrogative | case prefix)",
    )
    ax5.axhline(
        float(data["type_prior"][interrogative]),
        linestyle="--",
        color=PALETTE["INK"],
        linewidth=1.2,
        label="static type prior",
    )
    ax5.set_xticks(positions)
    ax5.set_xticklabels(data["words"], fontsize=FONTS["tick_label"])
    # A tight ceiling: the posterior and prior both live well under 0.15, and a
    # 0..0.5 range wasted four fifths of the panel on empty space. The data-driven
    # branch keeps the curve inside the frame if the measurement ever grows.
    peak = float(clause_track[interrogative].max())
    ax5.set_ylim(0.0, 0.15 if peak <= 0.12 else peak * 1.25)
    ax5.set_ylabel("P(interrogative)", fontsize=FONTS["axis_label"])
    ax5.set_title(
        "L5 — clause-type posterior as the decoded case prefix grows",
        fontsize=FONTS["title"],
    )
    # Lower left: with the tight y ceiling the curve and the prior rule both run
    # along the top of the panel, so an upper-left legend would sit on the data.
    ax5.legend(fontsize=FONTS["annotation"], loc="lower left")

    # ── Entropy traces ────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(grid[2, 2])
    # Five visually distinct hues (PALETTE["LEARNED"] aliases PALETTE["MODEL"],
    # so using both would draw two traces in the same blue). L4 is additionally
    # dashed so no two traces share both color and linestyle.
    tracks = (
        ("L1 prior", data["letter_track"], PALETTE["MUTED"], "-"),
        ("L2B phones", data["g2p_track"], PALETTE["MODEL"], "-"),
        ("L3 words", data["word_track"], PALETTE["ACCENT"], "-"),
        ("L4 roles", data["case_track"], PALETTE["UNIGRAM"], "--"),
        ("L5 type", data["clause_track"], PALETTE["INK"], "-"),
    )
    peak_entropy = 0.0
    for label, track, color, linestyle in tracks:
        entropy = _entropy(track)
        peak_entropy = max(peak_entropy, float(entropy.max()))
        x = np.linspace(0.0, 1.0, len(entropy))
        ax6.plot(x, entropy, label=label, color=color, linestyle=linestyle, linewidth=1.6)
    ax6.set_xlabel("fraction of sequence", fontsize=FONTS["axis_label"])
    ax6.set_ylabel("belief entropy (nats)", fontsize=FONTS["axis_label"])
    ax6.set_title("Entropy per position", fontsize=FONTS["title"])
    # Headroom for the legend, and two columns so it stays short. The L1 prior
    # oscillates between 1.7 and 3.0 across the whole right half of this panel,
    # so an opaque single-column legend at upper right hid the last third of
    # that trace — the defect the L5 panel above already avoids by moving its
    # own legend. It went unnoticed here because an occluded series does not
    # collide with anything; it simply stops appearing.
    ax6.set_ylim(0.0, peak_entropy * 1.42)
    ax6.legend(fontsize=FONTS["annotation"] - 1, loc="upper right", ncol=2, framealpha=1.0)

    terminal = "?" if SENTENCE_IS_INTERROGATIVE else ""
    fig.suptitle(
        f"One sentence through the stack: '{sentence_text}{terminal}'",
        fontsize=FONTS["title"] + 2,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.018,
        f"Computed at generation time on jax {data['jax_version']} "
        f"({data['jax_backend']}) via {data['pymdp_agent_class']}.\n"
        f"Cross-checks: pymdp vs float64 oracle max |Δ| = "
        f"{data['g2p_crosscheck_max_abs_diff']:.1e} (G2P), "
        f"{data['case_crosscheck_max_abs_diff']:.1e} (case filter); "
        "the build raises rather than drawing if either exceeds tolerance.",
        fontsize=FONTS["annotation"],
        va="bottom",
        ha="left",
        color=PALETTE["MUTED"],
    )
    return fig


def generate_belief_dynamics(output_dir: Any = None) -> Any:
    """Generate and write ``belief_dynamics.png``; returns the written path."""
    fig = build_belief_dynamics_figure()
    try:
        return save_figure(fig, "belief_dynamics", output_dir)
    finally:
        plt.close(fig)
