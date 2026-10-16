#!/usr/bin/env python3
"""Generate the AILM cover image: an example sentence, its path through the six
level models, and the historical stage topology with measured inter-level
interventions. The connected runtime has its own comparison in the main text.

The topology panel draws exactly the structure the manuscript claims and no more.
An earlier version drew an observation arrow and a preference arrow between every
adjacent pair of levels — ten couplings, under the header "hierarchical generative
model" — while the body states in four places that the shipped decoders exchange
no beliefs and that measured channels exist. A title-page graphic asserting
what the paper refutes is the first thing a reader sees, so the panel is
generated from the edge sets the body names and gated by
``tests/test_cover_honesty.py``.

A later revision mislabeled the family again in a quieter way: it drew two of
the {{CHANNEL_COUNT}} measured receiver interventions under the header
"measured preference channels" — but the L6→L5 intervention moves the clause
level's joint ``D`` prior and the manuscript states in as many words that it
does not use the generator's C mechanism ([@sec:narrative_level]). The
panel now draws all five measured interventions, names each one's mechanism in
the manuscript's own vocabulary (``A`` mixing, predictive mixing, ``C``
preference, ``D`` prior), names its outcome in words, and carries the abstract's
"measured one at a time" qualifier. Hue stays neutral (ACCENT, the
annotation role); outcomes are carried by words, never by color.

Two further defects are fixed here, both of them ways a compressed cover
misleads rather than merely disappoints:

*Abbreviated verdicts.* The gate outcomes were drawn as ``FAIL`` and
``NONCONF.``. Neither is the manuscript's word, and the first is actively
wrong about what happened: the clause level's model was built and audited, and
what did not clear its declared bar is the gate. The badges are drawn from the
body's own register, and the two outcomes that need a qualifier carry it beside
them rather than in a caption the title page does not have. Since decision 114
the clause badge reads ``passing`` and its qualifier is what a reader needs to
size that pass: the arm was registered and answered inside one session, and it
has no held-back replication because decision 111 spent that population.

*A cover that showed outputs without their models.* The left panel drew one
example instance per level and nothing about the model that produced it. It now
states, for every level, the hidden state, the observation modality, where the
likelihood and the dynamics come from, and which mechanism produced the drawn
example. Those four facts are read from the ``LevelRow`` table in
``src/ailm/figures/architecture.py`` at draw time, not copied here, so the cover
cannot outlive the structure it summarizes; ``tests/test_cover_honesty.py``
also checks the tags and names against that same table.

This title-page graphic reports model structure and gate verdicts without
measured numeric values. Those qualitative claims are checked against the same
sources as the body. It is kept in the manuscript's palette and rendered
deterministically; ``--check`` refuses a missing or stale published cover.

Usage:
    MPLBACKEND=Agg    .venv/bin/python scripts/generate_cover.py
    MPLBACKEND=Agg    .venv/bin/python scripts/generate_cover.py --out docs/manuscript/cover.png
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The release runner executes this file through runpy under an isolated
# interpreter, which puts neither the scripts directory nor src/ on sys.path.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

#: Seams where the upper level's observation alphabet IS the lower level's hidden
#: alphabet, as `(lower, upper)`. A type identity fixed when the matrices are
#: built — nothing flows along it at run time — which is why the cover draws
#: these as plain lines and never as arrows. Named in
#: `docs/manuscript/03_methods.md`; the L2→L3 and L3→L4 seams are deliberately
#: absent because Level 3 reads its own within-word alphabet and Level 4 reads
#: UPOS tags from the treebank.
#:
#: L4→L5 left this tuple with decision 114. The canonical Level 5 is now the
#: widened-observation clause model, whose alphabet is the eight case roles
#: *plus* `VERB` and `AUX`: it strictly contains Level 4's hidden alphabet
#: rather than equaling it. That containment is the whole of decision 112's
#: stratificational caveat, and a plain line here would assert on the title page
#: the identity the discussion spends a section denying. The seam still holds
#: for the retained case-only ablation, which is not what the cover draws.
ALPHABET_SEAMS: tuple[tuple[str, str], ...] = (
    ("L1", "L2"),
    ("L5", "L6"),
)

#: All five measured receiver interventions ([@fig:composition],
#: [@sec:hierarchy]), as `(mechanism, source, receiver, mechanism in words,
#: outcome in words)`. The mechanism names are the manuscript's own: the harmful
#: L1→L2B coupling folds into the observation likelihood (``A``) over phones; the
#: helpful L3→L2B message mixes into the phone predictive; the L3→L1g generation
#: seam sets the letter level's ``C`` from the word level's lexicon; the L6→L5 and
#: oracle→L3 interventions move a ``D`` prior (and the oracle arm is privileged,
#: not an L5 output).
#:
#: The last two fields exist because the earlier cover compressed each
#: intervention into two or three words beside its arrow, which is enough room to
#: name a mechanism OR an outcome but not both. They are spelled out in the panel's
#: legend instead, so no reader has to decode "harms (A)".
INTERVENTIONS: tuple[tuple[str, str, str, str, str], ...] = (
    ("A", "L1", "L2", "folds into the phone observation likelihood A", "harms"),
    ("predictive", "L3", "L2", "mixes into the phone predictive", "helps"),
    ("C", "L3", "L1", "sets the letter generator's C preference", "helps"),
    ("D", "L6", "L5", "moves the clause level's joint D prior", "prior only"),
    ("D", "oracle", "L3", "privileged source, moves the word D prior", "helps"),
)

#: Gate outcome per built level, as drawn in [@fig:hierarchy] and summarized in
#: [@sec:results]: Stages 1-5 pass their declared gates (L2 carries two gates,
#: 2A and 2B, both passing); Stage 6's primary document-clustered analysis is
#: non-confirmatory. Words, not hues, so nothing here asserts an encoding.
#:
#: L5 reads `passing` since decision 114, which made the widened-observation
#: clause model the canonical Level 5. That arm's gate was registered before its
#: artifact existed and cleared its refitted bar, so the badge is the same word
#: the other cleared gates carry. It cannot be read on its own, though, which is
#: what its qualifier below is for: the retained case-only ablation is still
#: non-passing, and that record lives in the gate audit and the body, not here.
#:
#: The words are the body's own. `FAIL` was the worse of the two abbreviations
#: they replaced: the clause model was built and audited, and it was the gate
#: that did not pass, not the level that failed to exist.
GATE_OUTCOMES: dict[str, str] = {
    "L6": "passing",
    "L5": "passing",
    "L4": "passing",
    "L3": "passing",
    "L2": "passing",
    "L1": "passing",
}

#: The qualifier each badge cannot be read correctly without, in the root
#: README's own wording. A bare `passing` on L5 invites the reading that the
#: clause gate carries the same evidential weight as Stages 1 through 4; it does
#: not, because decision 112 registered and answered it inside one session and
#: decision 111 had already spent the held-back population, so it can never
#: acquire a replication.
#:
#: L6 reads `passing` since decision 115, and its qualifier is what stops that
#: word from erasing the result it replaced. The development population's
#: clustered interval still spans zero and is not retracted; what excludes zero
#: is the same estimand on the held-back population at the primary coverage.
#: Its family-corrected interval still spans zero. Both populations belong on the
#: title page, because a reader who sees only the second will not know that the
#: two populations disagree. Levels with nothing to qualify are absent rather
#: than carrying an empty string, so a missing qualifier is a KeyError and never
#: a blank line.
GATE_QUALIFIERS: dict[str, str] = {
    "L5": "development-session gate,\nno held-back replication",
    "L6": (
        "positive point estimate,\ndevelopment interval spans zero,\n"
        "held-back interval does not;\nfamily-corrected interval spans zero"
    ),
}

#: The built levels, top (document scale) to bottom (letters), matching the
#: LevelRow table in ``src/ailm/figures/architecture.py``. Level 7 (Speech) is
#: the external DSP boundary, not a built level, and is deliberately absent.
LEVEL_BLOCKS: tuple[tuple[str, str], ...] = (
    ("L6", "Doc-position"),
    ("L5", "Clause"),
    ("L4", "Case"),
    ("L3", "Word"),
    ("L2", "Phoneme"),
    ("L1", "Letter"),
)

#: One example instance per level, the running sentence's own decomposition. At
#: L4 the syntactic heuristic assigns a role to the subject noun only (`nsubj`
#: maps to NOM; determiners and the verb get none), and at L5 the instance is the
#: sentence type with the widened observation stream that decision 114 made
#: canonical: the subject's case role, then the `VERB` symbol contributed by
#: `sat`. The determiner contributes nothing at either level. An earlier row,
#: "the:NOM cat:NOM sat:V", assigned roles the heuristic never assigns — `V` is
#: not one of the eight case roles at all, and the widened alphabet's `VERB` is
#: a distinguished tenth symbol rather than a role the read-off produces.
EXAMPLES: dict[str, str] = {
    "L6": "sentence type: declarative",
    "L5": "declarative · NOM VERB",
    "L4": "cat:NOM",
    "L3": "the | cat | sat",
    "L2": "/DH AH/ /K AE T/ /S AE T/",
    "L1": "t h e   c a t   s a t",
}

#: The mechanism that produces the drawn example, which is NOT the same question
#: as where the level's statistics live: the L4 row shows the role
#: ``UD_DEPREL_TO_CASE`` assigns to the subject noun, while the level's own
#: likelihood is a learned P(UPOS | case). A cover that showed the output without
#: naming the producing mechanism would advertise an instance with no account of
#: where it came from.
EXAMPLE_SOURCES: dict[str, str] = {
    "L6": "position tercile, counted over the document",
    "L5": "per-type widened-symbol bigram",
    "L4": "UD_DEPREL_TO_CASE",
    "L3": "bounded lexicon",
    "L2": "CMUdict letter-to-phone",
    "L1": "Brown bigram",
}

#: Field labels for the per-level fact rows, in draw order. ``likelihood A`` and
#: ``dynamics B`` are the manuscript's names for the two matrices, so the cover
#: introduces the reader to the vocabulary the body then uses throughout.
FACT_FIELDS: tuple[str, ...] = (
    "hidden state",
    "observation",
    "likelihood A",
    "dynamics B",
    "example from",
)


def _frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _level_facts() -> dict[str, tuple[str, ...]]:
    """Return the four model facts per built level, read from the figure layer.

    The values are the ``LevelRow`` fields the level-stack figure draws, with
    their embedded newlines flattened for a single-line cover row. Reading them
    rather than restating them is the point: the cover and [@fig:hierarchy] then
    cannot disagree about what a level's hidden state or likelihood is.

    Returns:
        Level tag to ``(hidden, observation, likelihood, dynamics)``, for the
        built levels only. Level 7 is the external DSP boundary and is excluded
        the same way it is excluded from :data:`LEVEL_BLOCKS`.
    """
    from ailm.figures.architecture import LEVELS

    def flat(value: str) -> str:
        return " ".join(value.split())

    return {
        row.level: (
            flat(row.hidden),
            flat(row.observation),
            flat(row.emission),
            flat(row.transition),
        )
        for row in LEVELS
        if row.built
    }


def build_cover():
    """Draw the cover and return the figure without saving or closing it.

    Split out of :func:`main` so ``tests/test_cover_honesty.py`` can measure
    every string's rendered extent against the axes that owns it, and can check
    that the rotated order labels clear the level boxes. Measuring the saved PNG
    cannot do either: once matplotlib has drawn a string over a box or under the
    neighboring axes, the pixels no longer say which text it was or which artist
    it collided with.

    Returns:
        The matplotlib figure. The caller owns it and must close it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgb
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from ailm.viz_config import FONTS, PALETTE, apply_style

    apply_style()

    facts = _level_facts()

    # ── Figure: a tall banner in the manuscript's palette ─────────────────────
    # Taller than the original 6.2in: five fact rows per level need vertical
    # room, and the alternative — shrinking the type — would put the cover below
    # the legibility floor every other figure in the paper is held to.
    fig = plt.figure(figsize=(13.0, 11.5), dpi=200, facecolor=PALETTE["BACKGROUND"])
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.30, 1.0],
        wspace=0.05,
        left=0.040,
        right=0.985,
        top=0.955,
        bottom=0.035,
    )

    # ══ LEFT PANEL — the example sentence, decomposed level by level ═══════════
    axs = fig.add_subplot(gs[0, 0])
    _frame(axs)

    axs.text(
        0.5,
        0.995,
        "one sentence, six explicit levels",
        transform=axs.transAxes,
        ha="center",
        va="top",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        color=PALETTE["INK"],
    )

    # The order arrows and their rotated labels live in a gutter of their own,
    # left of every box. In the shipped cover the generation label sat at 0.112
    # with the boxes starting at 0.135, and a rotated string's bounding box is
    # about one line height wide, so the label ran onto the box edge on the
    # first page a reader sees. The gutter is now explicit, and
    # `tests/test_cover_honesty.py` measures label extents against box extents
    # rather than trusting the arithmetic in this comment.
    RECOGNITION_ARROW_X = 0.018
    RECOGNITION_LABEL_X = 0.030
    GENERATION_ARROW_X = 0.056
    GENERATION_LABEL_X = 0.068
    BOX_LEFT = 0.105
    # Wide enough for the longest tag-and-name pair at the title size. At 0.245
    # "L6  Doc-position" measured wider than its own box and printed over both
    # rounded edges.
    BOX_WIDTH = 0.285
    EXAMPLE_X = 0.405
    # The field-label column is right-aligned, so its LEFT edge moves with the
    # longest label. At 0.190 the widest label reached back to 0.066 and ran
    # into the generation-order gutter at 0.068 — the same class of collision as
    # the one being fixed above, one column over. Held at 0.235, and gated.
    FACT_LABEL_X = 0.235
    FACT_VALUE_X = 0.245

    LEVEL_TOP = 0.928
    LEVEL_BOTTOM = 0.068
    band = (LEVEL_TOP - LEVEL_BOTTOM) / len(LEVEL_BLOCKS)
    BOX_HEIGHT = 0.034
    FACT_STEP = 0.0190

    for index, (tag, name) in enumerate(LEVEL_BLOCKS):
        y_top = LEVEL_TOP - index * band
        box = FancyBboxPatch(
            (BOX_LEFT, y_top - BOX_HEIGHT),
            BOX_WIDTH,
            BOX_HEIGHT,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            facecolor=PALETTE["BACKGROUND"],
            edgecolor=PALETTE["INK"],
            linewidth=1.1,
        )
        # Tagged so the overlap gate can find the boxes by role rather than by
        # guessing which patches on the panel are level boxes.
        box.set_gid("level-box")
        axs.add_patch(box)
        axs.text(
            BOX_LEFT + BOX_WIDTH / 2,
            y_top - BOX_HEIGHT / 2,
            f"{tag}  {name}",
            transform=axs.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["title"],
            fontweight="bold",
            color=PALETTE["INK"],
        )
        axs.text(
            EXAMPLE_X,
            y_top - BOX_HEIGHT / 2,
            EXAMPLES[tag],
            transform=axs.transAxes,
            ha="left",
            va="center",
            fontsize=FONTS["axis_label"],
            color=PALETTE["INK"],
        )

        hidden, observation, likelihood, dynamics = facts[tag]
        values = (hidden, observation, likelihood, dynamics, EXAMPLE_SOURCES[tag])
        for row, (field, value) in enumerate(zip(FACT_FIELDS, values, strict=True)):
            y = y_top - BOX_HEIGHT - 0.013 - row * FACT_STEP
            axs.text(
                FACT_LABEL_X,
                y,
                field,
                transform=axs.transAxes,
                ha="right",
                va="center",
                fontsize=FONTS["cell"],
                color=PALETTE["MUTED"],
            )
            axs.text(
                FACT_VALUE_X,
                y,
                value,
                transform=axs.transAxes,
                ha="left",
                va="center",
                fontsize=FONTS["annotation"],
                color=PALETTE["INK"],
            )

    # Recognition path (bottom-up) and generation path (top-down) arrows.
    # Recognition flows letters -> document scale (upward); generation flows
    # document scale -> letters (downward).
    axs.annotate(
        "",
        xy=(RECOGNITION_ARROW_X, LEVEL_TOP - 0.01),
        xytext=(RECOGNITION_ARROW_X, LEVEL_BOTTOM + 0.02),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["MODEL"], lw=2.2, shrinkA=0, shrinkB=0),
    )
    recognition_label = axs.text(
        RECOGNITION_LABEL_X,
        0.48,
        "recognition order (bottom-up)",
        transform=axs.transAxes,
        rotation=90,
        ha="left",
        va="center",
        fontsize=FONTS["footnote"],
        color=PALETTE["MODEL"],
        fontweight="bold",
    )
    recognition_label.set_gid("direction-label")
    axs.annotate(
        "",
        xy=(GENERATION_ARROW_X, LEVEL_BOTTOM + 0.02),
        xytext=(GENERATION_ARROW_X, LEVEL_TOP - 0.01),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["ANTI"], lw=2.2, shrinkA=0, shrinkB=0),
    )
    generation_label = axs.text(
        GENERATION_LABEL_X,
        0.48,
        "generation order (top-down)",
        transform=axs.transAxes,
        rotation=90,
        ha="left",
        va="center",
        fontsize=FONTS["footnote"],
        color=PALETTE["ANTI"],
        fontweight="bold",
    )
    generation_label.set_gid("direction-label")

    axs.text(
        0.5,
        0.018,
        "named probabilities, preferences, habits and precision\n"
        "historical standalone examples; connected runtime in text",
        transform=axs.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONTS["subtitle"],
        style="italic",
        color=PALETTE["MUTED"],
    )

    # ══ RIGHT PANEL — the model topology ══════════════════════════════════════
    axr = fig.add_subplot(gs[0, 1])
    _frame(axr)

    # Two lines: at 17pt the single line measured 1.42x the panel width and the
    # layout gate clipped it mid-word. Wrapping keeps the same words inside.
    axr.text(
        0.5,
        0.995,
        "historical level gates\nand measured interventions",
        transform=axr.transAxes,
        ha="center",
        va="top",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        color=PALETTE["INK"],
    )

    # Level blocks, top (document scale) to bottom (letters).
    # One hue, ramped by depth. The previous version drew each level in a
    # different member of the semantic palette -- Word in FLOOR, Phoneme in
    # UNIGRAM -- so the panel asserted an encoding it did not have, and asserted
    # it in names that mean something else in every other figure. Six hues that
    # stand for nothing are worse than none, because a reader is entitled to
    # assume a color difference marks a difference.
    #
    # Depth is the one ordering the panel genuinely shows, so lightness carries
    # it and hue is reserved for structure; the intervention arrows and the
    # gate outcomes carry words, not hues.
    #
    # The ramp is derived from PALETTE keys only: MODEL (the shipped agent's
    # blue) is mixed toward BACKGROUND by depth, so the panel stays in the
    # manuscript's palette and no raw hex lives in this script. Mixing in
    # RGB, not alpha: alpha would blend against whatever sits behind the
    # block, while the mixed color is the block's own.
    def _depth_color(fraction: float) -> str:
        """Mix MODEL toward BACKGROUND; fraction 0 = pure MODEL, 1 = BACKGROUND."""
        model = to_rgb(PALETTE["MODEL"])
        background = to_rgb(PALETTE["BACKGROUND"])
        mixed = (
            model_channel + (background_channel - model_channel) * fraction
            for model_channel, background_channel in zip(model, background, strict=True)
        )
        return to_hex(tuple(mixed))  # type: ignore[arg-type]

    # The ramp stops short of BACKGROUND so all six level boxes remain visible.
    levels = [
        (tag, name, _depth_color(0.75 * i / (len(LEVEL_BLOCKS) - 1)))
        for i, (tag, name) in enumerate(LEVEL_BLOCKS)
    ]
    n = len(levels)
    # The stack sits high so the band below it can carry the structural summary
    # and the full intervention legend.
    BLOCK_LEFT = 0.105
    BLOCK_WIDTH = 0.620
    top = 0.905
    bottom = 0.400
    stack_band = (top - bottom) / n
    # Leave room for gate qualifiers, including rounded-box padding.
    # Half-band boxes let its final line cross the next level's top border.
    h = stack_band * 0.25
    ys = [top - i * stack_band - stack_band / 2 for i in range(n)]
    # L6 has both population and coverage qualifications. Its extra line needs
    # more space than the other rows; the header leaves this room above the stack.
    ys[0] += 0.025

    for (tag, name, color), y in zip(levels, ys, strict=True):
        # White text loses contrast on the pale part of the depth ramp.
        label_color = PALETTE["BACKGROUND"] if tag == "L6" else PALETTE["INK"]
        axr.add_patch(
            FancyBboxPatch(
                (BLOCK_LEFT, y - h / 2),
                BLOCK_WIDTH,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.014",
                facecolor=color,
                edgecolor=PALETTE["INK"],
                linewidth=1.2,
                alpha=0.92,
            )
        )
        axr.text(
            0.145,
            y,
            tag,
            transform=axr.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["axis_label"],
            fontweight="bold",
            color=label_color,
        )
        axr.text(
            # Left of where the longest verdict word starts once it is
            # right-aligned in the block: "Doc-position" and "non-confirmatory"
            # are the widest pair and they met at 0.235.
            0.205,
            y,
            name,
            transform=axr.transAxes,
            ha="left",
            va="center",
            fontsize=FONTS["axis_label"],
            fontweight="bold",
            color=label_color,
        )
        # Gate outcome, right-aligned inside the block. Matches the audited
        # verdicts drawn in [@fig:hierarchy]; see GATE_OUTCOMES for the source.
        axr.text(
            BLOCK_LEFT + BLOCK_WIDTH - 0.012,
            y,
            GATE_OUTCOMES[tag],
            transform=axr.transAxes,
            ha="right",
            va="center",
            fontsize=FONTS["cell"],
            fontweight="bold",
            color=label_color,
        )
        if tag in GATE_QUALIFIERS:
            # Under the block, not inside it: the qualifier is what the outcome
            # means, and a reader who stops at the badge has still not been
            # told something false.
            axr.text(
                BLOCK_LEFT + BLOCK_WIDTH - 0.012,
                y - h / 2 - 0.010,
                GATE_QUALIFIERS[tag],
                transform=axr.transAxes,
                ha="right",
                va="top",
                fontsize=FONTS["cell"],
                style="italic",
                color=PALETTE["MUTED"],
            )

    index_of = {tag: position for position, (tag, _, _) in enumerate(levels)}

    def _y(tag: str) -> float:
        return ys[index_of[tag]]

    # Two edge sets, and only two. ALPHABET_SEAMS are places where the upper
    # level's observation alphabet *is* the lower level's hidden alphabet — a
    # type identity fixed when the matrices are built, with nothing flowing
    # along it at run time, so it is drawn as a plain line and never as an
    # arrow. INTERVENTIONS are the five measured receiver interventions, each
    # drawn as an arrow whose outcome is named beside it and whose mechanism is
    # spelled out in the legend below.
    #
    # Anything drawn between two other levels would be a claim the manuscript
    # spends [@sec:architecture] and [@sec:limitations] denying.
    SEAM_X = 0.082
    for lower_tag, upper_tag in ALPHABET_SEAMS:
        lower, upper = ys[index_of[lower_tag]], ys[index_of[upper_tag]]
        # A headless FancyArrowPatch rather than `axr.plot`: the level blocks are
        # added with `add_patch` in DATA coordinates, and the axes' data limits
        # are the default unit square only because nothing autoscales them.
        # `plot` would autoscale and silently rescale every block on the panel.
        axr.add_patch(
            FancyArrowPatch(
                (SEAM_X, lower),
                (SEAM_X, upper),
                arrowstyle="-",
                color=PALETTE["MODEL"],
                lw=2.2,
            )
        )
        axr.text(
            SEAM_X,
            (upper + lower) / 2,
            "o",
            transform=axr.transAxes,
            ha="center",
            va="center",
            fontsize=FONTS["subtitle"],
            color=PALETTE["MODEL"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )

    # The five measured interventions. ACCENT is the palette's annotation role —
    # a neutral structural hue, so no arrow's color asserts an outcome; the
    # outcome is written beside each arrow. The two interventions whose receiver
    # is the L2B composition receiver sit on the right rail; the C and D
    # channels between modeled levels sit on the left rail, so no two arrows
    # share a vertical span and every label has an unambiguous arrow.
    RECEIVER_X = 0.775
    CHANNEL_X = 0.042
    for _mech, src, rcv, _mechanism_words, outcome in INTERVENTIONS:
        if src == "oracle":
            # The oracle arm's source is not a built level; it is drawn
            # as a dashed external node after this loop.
            continue
        if rcv == "L2" and index_of[src] > index_of[rcv]:
            # Source below the receiver (L1 -> L2B): message travels up, head at
            # the receiver.
            y_start, y_end = _y(src) + 0.005, _y(rcv) - 0.005
        elif rcv == "L2":
            # Source above the receiver (L3 -> L2B): message travels down, head
            # at the receiver; tail anchored just below L3's block edge so the
            # oracle arrow's horizontal lane stays clear.
            y_start, y_end = _y(src) - 0.018, _y(rcv) + 0.005
        else:
            # C / D channels run top-down between modeled levels on the left
            # rail, head at the lower level.
            y_start, y_end = _y(src), _y(rcv)
        x = RECEIVER_X if rcv == "L2" else CHANNEL_X
        axr.add_patch(
            FancyArrowPatch(
                (x, y_start),
                (x, y_end),
                arrowstyle="-|>",
                mutation_scale=16,
                color=PALETTE["ACCENT"],
                lw=2.2,
            )
        )
        if rcv == "L2":
            # Anchor beside the arrow's lower reach (near the receiver) rather
            # than the span midpoint: the L3->L2 label otherwise sits at L3's
            # height, where the oracle arrow's head lane passes.
            anchor_y = y_end + 0.010 if index_of[src] < index_of[rcv] else y_start - 0.014
            axr.text(
                RECEIVER_X + 0.015,
                anchor_y,
                outcome,
                transform=axr.transAxes,
                ha="left",
                va="center",
                fontsize=FONTS["cell"],
                color=PALETTE["ACCENT"],
            )
        else:
            # Beside its arrow on the outside of the rail: rotated, in the clear
            # strip left of CHANNEL_X, so the label cannot be read onto a seam.
            axr.text(
                0.014,
                (y_start + y_end) / 2,
                outcome,
                transform=axr.transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=FONTS["cell"],
                color=PALETTE["ACCENT"],
            )

    # The privileged oracle arm: its source is an external punctuation-type
    # oracle, not an L5 output — the manuscript is explicit that attributing it
    # to L5 would falsify [@fig:composition]. Dashed box = privileged source.
    oracle_y = _y("L3")
    axr.add_patch(
        FancyBboxPatch(
            (0.822, oracle_y - 0.030),
            0.166,
            0.060,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            facecolor="white",
            edgecolor=PALETTE["INK"],
            linewidth=1.0,
            linestyle="--",
        )
    )
    axr.text(
        0.905,
        oracle_y,
        "type oracle\n(privileged)",
        transform=axr.transAxes,
        ha="center",
        va="center",
        fontsize=FONTS["cell"],
        color=PALETTE["INK"],
    )
    axr.add_patch(
        FancyArrowPatch(
            (0.820, oracle_y),
            (BLOCK_LEFT + BLOCK_WIDTH + 0.006, oracle_y),
            arrowstyle="-|>",
            mutation_scale=14,
            color=PALETTE["ACCENT"],
            lw=2.0,
        )
    )

    # Structural summary, a claim the architecture section makes
    # ([@sec:architecture], [@fig:gnn_pipeline]) — not a measured value.
    axr.text(
        0.5,
        0.345,
        "every level: GNN-declared model, conformance-checked\n"
        "generative-model inference with named score paths\n"
        "connected runtime and GUM comparison: main text",
        transform=axr.transAxes,
        ha="center",
        va="top",
        fontsize=FONTS["subtitle"],
        style="italic",
        color=PALETTE["MUTED"],
    )

    # The intervention legend: every measured intervention, its mechanism and
    # its outcome, in full words. This is the half of the panel the arrows
    # cannot carry — an arrow can show that L1 reaches L2, but not that it does
    # so through the observation likelihood, nor that doing so hurts.
    axr.text(
        0.005,
        0.262,
        "historical interventions, measured separately",
        transform=axr.transAxes,
        ha="left",
        va="top",
        fontsize=FONTS["cell"],
        fontweight="bold",
        color=PALETTE["ACCENT"],
    )
    legend_y = 0.222
    for _mech, src, rcv, mechanism_words, outcome in INTERVENTIONS:
        axr.text(
            0.005,
            legend_y,
            f"{src} → {rcv}",
            transform=axr.transAxes,
            ha="left",
            va="center",
            fontsize=FONTS["cell"],
            fontweight="bold",
            color=PALETTE["ACCENT"],
        )
        axr.text(
            # Clear of the widest source-to-receiver pair, which is the oracle
            # arm's; at 0.150 "oracle → L3" ran under its own mechanism text.
            0.185,
            legend_y,
            f"{mechanism_words} · {outcome}",
            transform=axr.transAxes,
            ha="left",
            va="center",
            fontsize=FONTS["cell"],
            color=PALETTE["INK"],
        )
        legend_y -= 0.032

    axr.text(
        0.005,
        0.030,
        "shared alphabet  o   (a type identity, nothing flows along it)",
        transform=axr.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONTS["cell"],
        color=PALETTE["MODEL"],
    )

    return fig


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "manuscript" / "cover.png"))
    parser.add_argument("--check", action="store_true", help="verify cover bytes without writing")
    args = parser.parse_args(argv)

    from _artifact_io import (
        AuthenticatedFileError,
        atomic_write_bytes,
        read_authenticated_regular_file,
    )

    from ailm.figures._common import assert_legible

    fig = build_cover()
    try:
        # A failed layout must not replace the published cover.
        assert_legible(fig)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "AILM"},
        )
        expected = buffer.getvalue()
    finally:
        plt.close(fig)

    out = Path(args.out)
    if args.check:
        try:
            actual = read_authenticated_regular_file(out, description="published cover").content
        except (FileNotFoundError, AuthenticatedFileError) as error:
            print(f"cover check failed: {error}")
            return 1
        if actual != expected:
            print(f"cover is stale: {out}; run scripts/generate_cover.py")
            return 1
        print(f"cover is current: {out}")
        return 0
    atomic_write_bytes(out, expected)
    print(f"cover written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
