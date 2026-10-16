"""Architecture figures: the AILM level stack and the per-symbol inference cycle.

Two figures, drawn entirely with matplotlib patches and text — no diagram tool, no
network, no external asset. Both are pure functions of the code's own constants
(:data:`ailm.levels.letter.N_SYMBOLS`, :data:`ailm.levels.phoneme.N_PHONES`,
``len(CaseRole)``), so a change to the alphabet or the phone inventory shows up in
the figure instead of leaving a stale number behind.

``fig:hierarchy``
    One row per level, naming the hidden factor, the observation modality, what
    ``A`` is, what ``B`` carries, and the stage that builds it. The
    identity-versus-learned status of ``A`` is encoded three ways — badge glyph,
    cell tint, and an explicit legend — because Level 2 is the first level whose
    ``A`` is a real emission, and that transition is the point where the model
    starts doing inference rather than reading its state back off the observation.

``fig:inference_cycle``
    The four pymdp calls that make up one symbol step, with ``A`` and ``B``
    labeled at the exact call each enters, plus the predictive read-off that
    perplexity is accumulated from. This is the figure that makes the orientation
    argument legible: once you can see that ``A`` is consulted at
    ``infer_states`` and ``B`` at ``update_empirical_prior``, swapping them is
    visibly a claim about *which variable the posterior is over*, not a
    stylistic choice.

Neither figure contains a measured number, so neither depends on the claims
ledger. The measured cost of the mis-wiring lives in ``fig:wiring_comparison``
(:mod:`ailm.figures.matrices`), which loads it from the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ailm.cases import CaseRole
from ailm.figures._common import save_figure
from ailm.gates import GateId, GateVerdict
from ailm.levels.letter import N_SYMBOLS
from ailm.levels.phoneme import N_PHONES
from ailm.levels.widened import N_WIDENED_JOINT, N_WIDENED_SYMBOLS
from ailm.viz_config import FIGSIZES, FONTS, PALETTE, apply_style

__all__ = [
    "LEVELS",
    "LevelRow",
    "generate_hierarchy_figure",
    "generate_inference_cycle_figure",
    "write_architecture_figures",
]


# ── Level inventory ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LevelRow:
    """One row of the level-stack figure.

    Attributes:
        level: Generative-level identifier, e.g. ``"L1"``. Stages 4 and 5 share
            ``"L4"`` because they build the same level.
        name: Level name as the roadmap calls it.
        hidden: The hidden state factor.
        observation: The observation modality.
        emission: What ``A`` is at this level.
        transition: What ``B`` carries at this level.
        stage: Stage that builds it, e.g. ``"Stage 1"``.
        built: Whether that stage's model is constructed and audited. Outcome is
            reported separately because a built model may have a non-passing or
            non-confirmatory primary analysis.
        a_kind: One of ``"identity"``, ``"learned"``, ``"planned"``,
            ``"external"`` — drives the badge, the tint, and the legend entry.
        gate_id: Which audited gate this row's Stage-column badge reports, or
            ``None`` for a row that is not an audited gate. Naming it is what
            makes the typed badge checkable: ``tests/test_figures_architecture``
            reads ``output/reports/gate_audit.json`` and requires the drawn
            verdict to be the one the re-runnable audit computes for this id.
            Two stages audit more than one gate — Stage 2 audits the phoneme
            perplexity gate and the G2P accuracy gate, Stage 5 audits the clause
            gate and its retained case-only ablation — and the row draws exactly
            one of them, so the id says which.
        gate_note: Short name of that gate, drawn beside the stage so a reader
            sees which of the stage's gates the badge belongs to without
            reading the code. Empty for a stage that audits only one gate;
            ``tests/test_figures_architecture.py`` requires it wherever a stage
            audits more than one. ``stage`` itself cannot carry this: it is
            asserted equal to "Stage 1" through "Stage 7" exactly, and the cover
            keys off the level rather than the stage.
    """

    level: str
    name: str
    hidden: str
    observation: str
    emission: str
    transition: str
    stage: str
    built: bool
    a_kind: str
    status: str = ""
    gate_id: GateId | str | None = None
    gate_note: str = ""


def _status(verdict: GateVerdict, outcome: str) -> str:
    """A Stage-column status: an audited placement verdict and its outcome.

    The verdict half is a :class:`ailm.gates.GateVerdict` member, not a string
    typed into the row, so a row cannot draw a verdict the shared gate vocabulary
    does not define. It is also no longer compared to the audit by a reader
    alone: each row names the ``gate_id`` whose badge it draws, and
    ``tests/test_figures_architecture.py`` requires the drawn verdict to equal
    the one ``scripts/audit_gates.py`` writes to
    ``output/reports/gate_audit.json`` for that id.

    The outcome half is still typed, and the residual gap is narrower but real.
    Two rows deliberately state something finer than the audit's binary outcome.
    Stage 5 says PASS (DEV-SESSION) and Stage 6 says NONCONF., where the audit
    says PASS in both cases: a gate can pass while its registration and its
    outcome share one session with no held-back population left (decision 114),
    or while its primary analysis is non-confirmatory. Both are qualifications
    on a pass, not contradictions of one. So the test binds the verdict exactly
    and binds the outcome only in the direction that cannot be a qualification —
    a row that says PASS must correspond to an audit row that says PASS.

    This sentence described Stage 5 as drawing its retained case-only ablation,
    which decision 114 had already demoted; the row beneath now draws the
    canonical widened arm and names its ``gate_id``.

    The two-line layout is deliberate: the typed verdict names are long enough
    that a one-line "VERDICT / OUTCOME" collides with the neighboring
    annotation, and the legibility check treats that as a build failure.
    """
    return f"{verdict}\n{outcome}"


#: Bottom-to-top, which is also build order. Symbol counts are read from the code
#: rather than written here, so the figure cannot drift from the alphabet. The
#: Stage-column verdicts are typed — see :func:`_status` for exactly how far that
#: is bound and how far it is not.
LEVELS: tuple[LevelRow, ...] = (
    LevelRow(
        level="L1",
        name="Letter",
        hidden=f"current letter\n({N_SYMBOLS} symbols)",
        observation=f"current letter\n({N_SYMBOLS} symbols)",
        emission="identity — the observation IS the state",
        transition="Brown letter bigram\nP(next | current)",
        stage="Stage 1",
        built=True,
        a_kind="identity",
        status=_status(GateVerdict.BRACKETED_BOUND, "PASS"),
    ),
    LevelRow(
        level="L2",
        name="Phoneme",
        hidden=f"current phone\n({N_PHONES} ARPAbet)",
        observation=f"current letter\n({N_SYMBOLS} symbols)",
        emission=(f"learned P(letter | phone) — a real\n{N_SYMBOLS} x {N_PHONES} emission"),
        transition="CMUdict phone bigram\nP(next | current)",
        stage="Stage 2",
        built=True,
        a_kind="learned",
        status=_status(GateVerdict.BRACKETED_REFERENCE, "PASS"),
        # Stage 2 audits two gates, and this badge is the G2P accuracy one. The
        # phoneme-perplexity gate is audited separately as BRACKETED_BOUND /
        # PASS, so a bare "Stage 2" label left the reader unable to tell which
        # of the two the verdict belonged to — the same ambiguity the Stage 5
        # row carried until it drew a failing ablation under a passing gate's
        # name. Naming the id is what makes the badge checkable against the
        # audit rather than merely plausible.
        gate_id=GateId.STAGE2_G2P,
        gate_note="G2P gate",
    ),
    LevelRow(
        level="L3",
        name="Word",
        hidden="word identity x position\n(bounded lexicon)",
        observation="current letter\n(a-z + END)",
        emission="structural: the lexicon's spelling,\nA[letter, word, position]",
        transition="identity x the +1 shift\n(both structural)",
        stage="Stage 3",
        built=True,
        a_kind="structural",
        status=_status(GateVerdict.BRACKETED_REFERENCE, "PASS"),
    ),
    LevelRow(
        level="L4",
        name="Case",
        hidden=f"case role\n({len(CaseRole)} roles)",
        observation="UPOS tag\n(labels' sources withheld)",
        emission="learned P(UPOS | case)",
        transition="case-role bigram\nover a sentence",
        stage="Stage 4",
        built=True,
        a_kind="learned",
        status=_status(GateVerdict.BRACKETED_REFERENCE, "PASS"),
    ),
    LevelRow(
        # Decision 114 made the widened-observation clause model the canonical
        # Level 5 and kept the case-only model as a retained ablation. Every
        # field below described the ablation, so the row drew a failing badge for
        # a gate `scripts/audit_gates.py` records under GateId.STAGE5_CLAUSE as
        # BRACKETED_REFERENCE / PASS, and the caption says this column carries
        # each gate's audited outcome. The ablation's own row keeps its verdict
        # in the gate audit under `stage5_clause_case_only`; it is not drawn here.
        #
        # The observation is the point a reader is most likely to get wrong. The
        # widened alphabet strictly CONTAINS Level 4's eight case roles — it adds
        # a distinguished symbol for VERB and one for AUX — so this is not the
        # alphabet identity the L1/L2 seam is. Saying "shared L4 alphabet" here
        # asserted an identity the canonical arm does not have.
        level="L5",
        name="Clause",
        hidden=f"sentence type x symbol\n(joint, {N_WIDENED_JOINT} states)",
        observation=f"case role, VERB or AUX\n({N_WIDENED_SYMBOLS} corpus labels)",
        emission="structural: deterministic\nsymbol read-off",
        transition="type-persistent blocks of\nwidened-symbol bigrams",
        stage="Stage 5",
        built=True,
        a_kind="structural",
        # The qualifier is not decoration. Decision 114 requires both limits
        # stated wherever this verdict is stated, and a bare PASS here would read
        # as the same kind of pass Levels 1 through 4 carry. The footnote below
        # spells out what DEV-SESSION abbreviates.
        status=_status(GateVerdict.BRACKETED_REFERENCE, "PASS (DEV-SESSION)"),
        # Stage 5 audits two gates. Naming the canonical one is what stops this
        # row from silently drifting back onto the ablation's verdict, which is
        # how it came to draw a FAIL badge for a passing gate in the first place.
        gate_id=GateId.STAGE5_CLAUSE,
        # Stage 5 audits two gates for the same reason Stage 2 does. The retained
        # case-only ablation is drawn as its own row in fig:gate_taxonomy at its
        # measured position, not here: this table is one row per level, and an
        # ablation is a second gate on a level rather than a second level.
        gate_note="clause gate",
    ),
    LevelRow(
        level="L6",
        name="Doc-position",
        hidden="document position\n(tercile of the document)",
        observation="sentence-type corpus labels\n(shared L5 alphabet)",
        emission="learned P(type | tercile)",
        transition="left-to-right\ntercile advance",
        stage="Stage 6",
        built=True,
        a_kind="learned",
        status=_status(GateVerdict.BRACKETED_REFERENCE, "NONCONF."),
    ),
    LevelRow(
        level="L7",
        name="Speech",
        hidden=f"current phone\n({N_PHONES} ARPAbet)",
        observation="acoustic frame\n(Whisper in, TTS out)",
        emission="outside pymdp — the DSP boundary,\nnot a learned matrix",
        transition="reuses the Level 2\nphone bigram",
        stage="Stage 7",
        built=False,
        a_kind="external",
        status="contract \u2713",
    ),
)

#: Column headers and their fractional extent within the table's x-span.
_COLUMNS: tuple[tuple[str, float, float], ...] = (
    ("Level", 0.000, 0.114),
    ("Hidden state\nfactor", 0.114, 0.272),
    ("Observation\nmodality", 0.272, 0.440),
    ("A — emission", 0.440, 0.694),
    ("B — transition\n(sequence stats)", 0.694, 0.836),
    ("Stage", 0.836, 1.000),
)

#: Badge style and legend wording per ``a_kind``. Distinctions never rest on hue
#: alone: each entry differs in fill, edge style, and hatch as well as color.
_A_KIND_STYLE: dict[str, dict[str, object]] = {
    "identity": {
        "facecolor": "white",
        "edgecolor": PALETTE["IDENTITY"],
        "hatch": "///",
        "linestyle": "-",
        "tint": PALETTE["IDENTITY"],
        "label": "A = I — identity: the observation is the state",
    },
    "learned": {
        "facecolor": PALETTE["LEARNED"],
        "edgecolor": PALETTE["INK"],
        "hatch": "",
        "linestyle": "-",
        "tint": PALETTE["LEARNED"],
        "label": "A learned — a genuine emission matrix",
    },
    "structural": {
        "facecolor": "white",
        "edgecolor": PALETTE["ACCENT"],
        "hatch": "..",
        "linestyle": "-",
        "tint": PALETTE["ACCENT"],
        "tint_alpha": 0.10,
        "label": "A structural — deterministic, read off the artifact",
    },
    "external": {
        "facecolor": PALETTE["PANEL"],
        "edgecolor": PALETTE["MUTED"],
        "hatch": "xxx",
        "linestyle": ":",
        "tint": PALETTE["MUTED"],
        "tint_alpha": 0.07,
        "label": "A outside pymdp — the DSP boundary",
    },
}

# ── Hierarchy layout constants (axes fraction, 0..1 in both directions) ──────
_TABLE_X0, _TABLE_X1 = 0.055, 0.800
_ROW_H, _ROW_GAP = 0.096, 0.008
_ROW_BASE = 0.124
_HEADER_Y0, _HEADER_Y1 = 0.852, 0.891
#: The gutter notes sit clear of the table's right border, and their text ends
#: just inside the canvas; there is little slack to move them further right.
#: Every Stage-column verdict must end inside the table border, which
#: ``tests/test_figures_architecture.py`` measures on the rendered figure.
_NOTE_X = 0.852
_LEGEND_Y = (0.958, 0.922)
_STAGE0_Y0, _STAGE0_H = 0.068, 0.046
#: Badge side length in x-fraction; the y side is scaled by the figure aspect so
#: the badge renders square rather than as a stretched rectangle.
_BADGE_W = 0.0095


def _row_bounds(index: int) -> tuple[float, float]:
    """Return ``(y0, y1)`` for row ``index``, counting 0 from the bottom."""
    y0 = _ROW_BASE + index * (_ROW_H + _ROW_GAP)
    return y0, y0 + _ROW_H


def _col_x(fraction: float) -> float:
    """Map a column fraction to an axes-fraction x coordinate."""
    return _TABLE_X0 + fraction * (_TABLE_X1 - _TABLE_X0)


def generate_hierarchy_figure(
    *,
    columns: tuple[tuple[str, float, float], ...] = _COLUMNS,
    stage0_y0: float = _STAGE0_Y0,
) -> Figure:
    """Build ``fig:hierarchy`` — the AILM level stack.

    Args:
        columns: Column header/fractional-extent geometry for the table.
            Defaults to the real :data:`_COLUMNS`; tests that need to prove
            ``_hierarchy_containment_violations`` can actually fail pass a
            perturbed table here instead of monkeypatching the module
            constant.
        stage0_y0: Y-origin (axes fraction) of the Stage-0 band. Defaults to
            the real :data:`_STAGE0_Y0`, overridable for the same reason.

    Returns:
        A matplotlib Figure. The caller owns it and should pass it to
        :func:`ailm.figures._common.save_figure` and then close it.
    """
    apply_style()
    width, height = FIGSIZES["table"]
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    badge_h = _BADGE_W * width / height  # square in physical inches

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(
        0.0,
        0.995,
        "AILM level stack: what each level infers, and where the learned statistics live",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["INK"],
    )

    # ── Legend for the A column, two rows of two ─────────────────────────────
    for slot, kind in enumerate(("identity", "learned", "structural", "external")):
        style = _A_KIND_STYLE[kind]
        lx = 0.0 if slot % 2 == 0 else 0.40
        ly = _LEGEND_Y[0] if slot < 2 else _LEGEND_Y[1]
        ax.add_patch(
            Rectangle(
                (lx, ly - badge_h / 2),
                _BADGE_W,
                badge_h,
                facecolor=str(style["facecolor"]),
                edgecolor=str(style["edgecolor"]),
                hatch=str(style["hatch"]) or None,
                linestyle=str(style["linestyle"]),
                linewidth=1.3,
                zorder=3,
            )
        )
        ax.text(
            lx + _BADGE_W + 0.008,
            ly,
            str(style["label"]),
            fontsize=FONTS["legend"],
            va="center",
            ha="left",
            color=PALETTE["INK"],
        )

    # ── Column headers ───────────────────────────────────────────────────────
    ax.add_patch(
        Rectangle(
            (_TABLE_X0, _HEADER_Y0),
            _TABLE_X1 - _TABLE_X0,
            _HEADER_Y1 - _HEADER_Y0,
            facecolor=PALETTE["INK"],
            edgecolor="none",
            zorder=1,
        )
    )
    for header, f0, _f1 in columns:
        ax.text(
            _col_x(f0) + 0.006,
            (_HEADER_Y0 + _HEADER_Y1) / 2,
            header,
            fontsize=FONTS["tick_label"],
            fontweight="bold",
            va="center",
            ha="left",
            color="white",
            linespacing=1.25,
            zorder=2,
        )

    # ── Rows ─────────────────────────────────────────────────────────────────
    for index, row in enumerate(LEVELS):
        y0, y1 = _row_bounds(index)
        y_mid = (y0 + y1) / 2
        style = _A_KIND_STYLE[row.a_kind]

        # Row plate. Built levels get a solid edge; unbuilt ones a dashed edge,
        # so "not yet measured" is visible without reading the Stage column.
        ax.add_patch(
            Rectangle(
                (_TABLE_X0, y0),
                _TABLE_X1 - _TABLE_X0,
                _ROW_H,
                facecolor="white" if row.built else PALETTE["PANEL"],
                edgecolor=PALETTE["INK"] if row.built else PALETTE["MUTED"],
                linestyle="-" if row.built else "--",
                linewidth=1.1,
                zorder=1,
            )
        )

        # A-cell tint — the second, redundant encoding of the A distinction.
        a_f0, a_f1 = columns[3][1], columns[3][2]
        ax.add_patch(
            Rectangle(
                (_col_x(a_f0), y0),
                _col_x(a_f1) - _col_x(a_f0),
                _ROW_H,
                facecolor=str(style["tint"]),
                edgecolor="none",
                alpha=float(style.get("tint_alpha", 0.18)),  # type: ignore[arg-type]
                zorder=1.5,
            )
        )

        # Level id + name.
        ax.text(
            _col_x(0.0) + 0.006,
            y_mid + 0.018,
            row.level,
            fontsize=FONTS["cell"],
            fontweight="bold",
            va="center",
            ha="left",
            color=PALETTE["MUTED"],
            zorder=3,
        )
        ax.text(
            _col_x(0.0) + 0.006,
            y_mid - 0.014,
            row.name,
            fontsize=FONTS["axis_label"],
            fontweight="bold",
            va="center",
            ha="left",
            color=PALETTE["INK"] if row.built else PALETTE["MUTED"],
            zorder=3,
        )

        # Hidden factor / observation modality.
        for text, (_h, f0, _f1) in ((row.hidden, columns[1]), (row.observation, columns[2])):
            ax.text(
                _col_x(f0) + 0.006,
                y_mid,
                text,
                fontsize=FONTS["cell"],
                va="center",
                ha="left",
                color=PALETTE["INK"] if row.built else PALETTE["MUTED"],
                linespacing=1.35,
                zorder=3,
            )

        # A cell: badge then description.
        ax.add_patch(
            Rectangle(
                (_col_x(a_f0) + 0.006, y_mid - badge_h / 2),
                _BADGE_W,
                badge_h,
                facecolor=str(style["facecolor"]),
                edgecolor=str(style["edgecolor"]),
                hatch=str(style["hatch"]) or None,
                linestyle=str(style["linestyle"]),
                linewidth=1.3,
                zorder=3,
            )
        )
        ax.text(
            _col_x(a_f0) + 0.006 + _BADGE_W + 0.007,
            y_mid,
            row.emission,
            fontsize=FONTS["cell"],
            va="center",
            ha="left",
            color=PALETTE["INK"] if row.built else PALETTE["MUTED"],
            linespacing=1.35,
            zorder=3,
        )

        # B cell.
        ax.text(
            _col_x(columns[4][1]) + 0.006,
            y_mid,
            row.transition,
            fontsize=FONTS["cell"],
            va="center",
            ha="left",
            color=PALETTE["INK"] if row.built else PALETTE["MUTED"],
            linespacing=1.35,
            zorder=3,
        )

        # Stage cell. The gate note rides on the stage line rather than becoming
        # a fourth line: the row is 0.096 high and already carries the stage, the
        # verdict and the outcome, and a fourth line prints through the band
        # below it.
        ax.text(
            _col_x(columns[5][1]) + 0.006,
            y_mid + 0.016,
            f"{row.stage} · {row.gate_note}" if row.gate_note else row.stage,
            fontsize=FONTS["cell"],
            fontweight="bold",
            va="center",
            ha="left",
            color=PALETTE["INK"] if row.built else PALETTE["MUTED"],
            zorder=3,
        )
        ax.text(
            _col_x(columns[5][1]) + 0.006,
            y_mid - 0.016,
            row.status,
            fontsize=FONTS["cell"],
            va="center",
            ha="left",
            color=(
                PALETTE["ANTI"]
                if "FAIL" in row.status or "NONCONF" in row.status
                else PALETTE["MUTED"]
                if not row.built
                else PALETTE["FLOOR"]
            ),
            zorder=3,
        )

    # ── Highlight rule on the first level whose A is a real emission ─────────
    phoneme_index = next(i for i, r in enumerate(LEVELS) if r.a_kind == "learned")
    hy0, _hy1 = _row_bounds(phoneme_index)
    ax.add_patch(
        Rectangle(
            (_TABLE_X0 - 0.006, hy0),
            0.005,
            _ROW_H,
            facecolor=PALETTE["ACCENT"],
            edgecolor="none",
            zorder=3,
        )
    )

    # ── Right-gutter notes, each anchored to the row it describes ────────────
    notes = (
        (
            0,
            "Corpus statistics live in B,\n"
            "never in A. Both slots accept\n"
            "a column-stochastic matrix,\n"
            "so pymdp's _validate()\n"
            "accepts either one.",
            True,
        ),
        (
            phoneme_index,
            "Level 2 is the first level\n"
            "whose A is a genuine emission\n"
            "— letters observed, phonemes\n"
            "inferred, not read straight off\n"
            "an identity A.",
            True,
        ),
    )
    for row_index, note, with_arrow in notes:
        ny0, ny1 = _row_bounds(row_index)
        ny_mid = (ny0 + ny1) / 2
        ax.text(
            _NOTE_X + 0.014,
            ny_mid,
            note,
            fontsize=FONTS["cell"],
            va="center",
            ha="left",
            color=PALETTE["INK"],
            linespacing=1.35,
        )
        if with_arrow:
            ax.add_patch(
                FancyArrowPatch(
                    (_NOTE_X + 0.010, ny_mid),
                    (_TABLE_X1 + 0.002, ny_mid),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.2,
                    color=PALETTE["ACCENT"],
                    shrinkA=0,
                    shrinkB=0,
                )
            )

    # ── Alphabet-identity gutter: L1 to L2 only ─────────────────────────────
    # The arrow spans exactly the two rows its label describes and points one
    # way, because the relation is directional: L2's observation modality is
    # L1's alphabet. It once ran from L1 to the top modeled level under the label
    # "update_empirical_prior couples adjacent levels" — a claim no code
    # supported. That call appears only inside each level's own temporal
    # recursion; there is no agent-to-agent message pass. The coupling was then
    # built and measured (src/ailm/levels/coupling.py) and it made accuracy
    # significantly WORSE: at position t the G2P level already observes letter
    # t, while the L1 predictive depends only on letters 0..t-1, so mixing it in
    # dilutes exact evidence with a strictly less informative guess. See
    # claim:coupling_margin_pts and claim:coupling_supported. The label was
    # corrected first; the geometry now matches the label.
    letter_mid = sum(_row_bounds(0)) / 2
    phoneme_mid = sum(_row_bounds(phoneme_index)) / 2
    ax.add_patch(
        FancyArrowPatch(
            (0.040, letter_mid),
            (0.040, phoneme_mid),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=PALETTE["MODEL"],
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        0.021,
        (letter_mid + phoneme_mid) / 2,
        "L2 observes\nL1's alphabet",
        fontsize=FONTS["cell"],
        rotation=90,
        va="center",
        ha="center",
        color=PALETTE["MODEL"],
        linespacing=1.3,
    )

    # ── Stage 0 foundation ───────────────────────────────────────────────────
    ax.add_patch(
        Rectangle(
            (_TABLE_X0, stage0_y0),
            _TABLE_X1 - _TABLE_X0,
            _STAGE0_H,
            facecolor=PALETTE["FLOOR"],
            edgecolor="none",
            alpha=0.20,
            zorder=1,
        )
    )
    ax.text(
        _TABLE_X0 + 0.008,
        stage0_y0 + _STAGE0_H / 2,
        "Stage 0 — infrastructure: claims ledger, control harness, corpus "
        "parsers, verified pymdp API surface     ✓ passed",
        fontsize=FONTS["cell"],
        fontweight="bold",
        va="center",
        ha="left",
        color=PALETTE["INK"],
        zorder=3,
    )

    # ── Footnotes ────────────────────────────────────────────────────────────
    ax.text(
        0.0,
        0.006,
        "Audited gates use measured comparators and live controls. "
        "BRACKETED_BOUND/_REFERENCE names threshold placement and comparator "
        "strength; PASS/FAIL/NONCONF. names the point outcome.\n"
        # Three lines and no more: the block grows upward from a fixed baseline
        # and the Stage 0 band sits 0.068 above it, so a fourth line prints
        # through the band. That is the defect decision 100 removed from this
        # figure, and assert_within_canvas cannot see it — only the containment
        # sweep in tests/test_figures_architecture.py does.
        "Stage 5 is the widened clause model (decision 114); its DEV-SESSION pass is "
        "unreplicated. Stage 6's registered point is positive, but its "
        "document-clustered interval spans zero; the reversed-position control shows "
        "sentence signs are direction-insensitive.\n"
        "L7 is a dashed interface, not a model: Whisper/TTS attach at the L2 phoneme "
        "boundary; tests enforce inventory identity, round-trip behavior, and "
        "fail-closed rejection.",
        fontsize=FONTS["cell"],
        va="bottom",
        ha="left",
        color=PALETTE["MUTED"],
        linespacing=1.4,
    )

    return fig


# ── Inference cycle ──────────────────────────────────────────────────────────


def _rounded_box(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    linestyle: str = "-",
    linewidth: float = 1.6,
) -> None:
    """Draw a rounded panel centerd on ``(cx, cy)`` in axes-fraction units."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.014",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=2,
        )
    )


def _elbow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    angle_a: float,
    angle_b: float,
    color: str,
    label: str = "",
    label_xy: tuple[float, float] | None = None,
) -> None:
    """Draw a right-angled arrow from ``start`` to ``end`` with an optional label."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.8,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"angle,angleA={angle_a},angleB={angle_b},rad=8",
            zorder=3,
        )
    )
    if label and label_xy is not None:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            fontsize=FONTS["annotation"],
            va="center",
            ha="center",
            color=color,
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
        )


def generate_inference_cycle_figure() -> Figure:
    """Build ``fig:inference_cycle`` — one pymdp step, with A and B located.

    Returns:
        A matplotlib Figure. The caller owns it.
    """
    apply_style()
    width, height = FIGSIZES["wide"]
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.0,
        0.995,
        "One symbol, one cycle: where A and B each enter a pymdp step",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["INK"],
    )

    # Box geometry: (center_x, center_y, width, height).
    prior = (0.115, 0.550, 0.210, 0.155)
    infer = (0.345, 0.790, 0.300, 0.150)
    post = (0.575, 0.550, 0.210, 0.155)
    update = (0.345, 0.310, 0.340, 0.150)
    readoff = (0.128, 0.135, 0.225, 0.150)
    ppl = (0.440, 0.135, 0.280, 0.120)

    for cx, cy, w, h in (prior, post):
        _rounded_box(ax, cx, cy, w, h, facecolor="white", edgecolor=PALETTE["INK"])
    _rounded_box(ax, *infer, facecolor="white", edgecolor=PALETTE["MODEL"], linewidth=2.2)
    _rounded_box(ax, *update, facecolor="white", edgecolor=PALETTE["FLOOR"], linewidth=2.2)
    for cx, cy, w, h in (readoff, ppl):
        _rounded_box(
            ax,
            cx,
            cy,
            w,
            h,
            facecolor=PALETTE["PANEL"],
            edgecolor=PALETTE["MUTED"],
            linestyle="--",
            linewidth=1.4,
        )

    def _stack(
        center: tuple[float, float, float, float],
        lines: list[tuple[str, float, str, str]],
    ) -> None:
        """Place vertically stacked, individually styled lines inside a box."""
        cx, cy = center[0], center[1]
        step = 0.040
        offset = (len(lines) - 1) / 2 * step
        for i, (text, size, color, weight) in enumerate(lines):
            ax.text(
                cx,
                cy + offset - i * step,
                text,
                fontsize=size,
                fontweight=weight,
                va="center",
                ha="center",
                color=color,
                zorder=4,
            )

    # Mathtext lines are set at ``math`` rather than ``annotation`` size: matplotlib
    # renders subscripts one level down, so a subscript on a 10pt symbol lands near
    # 7pt. assert_legible sees the artist's declared size, not the rendered
    # subscript, so the floor is honoured here by setting the base size high enough
    # that the subscripts themselves stay readable.
    math_pt = FONTS["axis_label"]
    _stack(
        prior,
        [
            ("empirical prior", FONTS["title"], PALETTE["INK"], "bold"),
            (r"$\rho_t = P(s_t \mid o_{1..t-1})$", math_pt, PALETTE["INK"], "normal"),
        ],
    )
    _stack(
        infer,
        [
            ("agent.infer_states($o_t$, $\\rho_t$)", FONTS["title"], PALETTE["INK"], "bold"),
            (r"$q(s_t) \propto A[o_t, :] \times \rho_t$", math_pt, PALETTE["INK"], "normal"),
            ("A — the emission — enters here", FONTS["annotation"], PALETTE["MODEL"], "bold"),
        ],
    )
    _stack(
        post,
        [
            ("posterior", FONTS["title"], PALETTE["INK"], "bold"),
            (r"$q(s_t) = P(s_t \mid o_{1..t})$", math_pt, PALETTE["INK"], "normal"),
        ],
    )
    _stack(
        update,
        [
            ("agent.update_empirical_prior($a$, $q$)", FONTS["title"], PALETTE["INK"], "bold"),
            (r"$\rho_{t+1} = B[:, :, a]\, q(s_t)$", math_pt, PALETTE["INK"], "normal"),
            ("B — the transition — enters here", FONTS["annotation"], PALETTE["FLOOR"], "bold"),
        ],
    )
    _stack(
        readoff,
        [
            ("predictive read-off", FONTS["title"], PALETTE["INK"], "bold"),
            (r"$P(o_t \mid \rho_t)$", math_pt, PALETTE["INK"], "normal"),
            (r"$= \sum_j A[o_t, j]\, \rho_t[j]$", math_pt, PALETTE["INK"], "normal"),
        ],
    )
    _stack(
        ppl,
        [
            ("perplexity", FONTS["title"], PALETTE["INK"], "bold"),
            (
                # No \frac: matplotlib nests fraction numerals two size levels down,
                # which puts them well below the legibility floor.
                r"$\exp(-\langle \log P(o_t \mid \rho_t)\rangle_t)$",
                math_pt,
                PALETTE["INK"],
                "normal",
            ),
        ],
    )

    # ── The cycle ────────────────────────────────────────────────────────────
    _elbow(
        ax,
        (prior[0], prior[1] + prior[3] / 2),
        (infer[0] - infer[2] / 2, infer[1]),
        angle_a=90,
        angle_b=180,
        color=PALETTE["INK"],
    )
    _elbow(
        ax,
        (infer[0] + infer[2] / 2, infer[1]),
        (post[0], post[1] + post[3] / 2),
        angle_a=0,
        angle_b=90,
        color=PALETTE["INK"],
    )
    _elbow(
        ax,
        (post[0], post[1] - post[3] / 2),
        (update[0] + update[2] / 2, update[1]),
        angle_a=-90,
        angle_b=0,
        color=PALETTE["INK"],
    )
    _elbow(
        ax,
        (update[0] - update[2] / 2, update[1]),
        (prior[0], prior[1] - prior[3] / 2),
        angle_a=180,
        angle_b=-90,
        color=PALETTE["INK"],
        label=r"$\rho_{t+1}$  becomes  $\rho_t$",
        label_xy=(0.190, 0.420),
    )

    # ── Observation entering the cycle ───────────────────────────────────────
    ax.add_patch(
        FancyArrowPatch(
            (infer[0], 0.935),
            (infer[0], infer[1] + infer[3] / 2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.8,
            color=PALETTE["ACCENT"],
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )
    ax.text(
        infer[0] + 0.014,
        0.925,
        r"$o_t$ — the observed symbol",
        fontsize=FONTS["axis_label"],
        va="center",
        ha="left",
        color=PALETTE["ACCENT"],
    )

    # ── Read-off tap: uses the prior, does not consume it ────────────────────
    ax.add_patch(
        FancyArrowPatch(
            (0.048, prior[1] - prior[3] / 2),
            (0.048, readoff[1] + readoff[3] / 2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            linestyle="--",
            color=PALETTE["MUTED"],
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )
    ax.text(
        0.040,
        (prior[1] - prior[3] / 2 + readoff[1] + readoff[3] / 2) / 2,
        "scored, not consumed",
        fontsize=FONTS["annotation"],
        rotation=90,
        va="center",
        ha="center",
        color=PALETTE["MUTED"],
    )
    ax.add_patch(
        FancyArrowPatch(
            (readoff[0] + readoff[2] / 2, readoff[1]),
            (ppl[0] - ppl[2] / 2, readoff[1]),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            linestyle="--",
            color=PALETTE["MUTED"],
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )

    # ── The orientation callout ──────────────────────────────────────────────
    callout_x0, callout_x1 = 0.705, 0.982
    _rounded_box(
        ax,
        (callout_x0 + callout_x1) / 2,
        0.520,
        callout_x1 - callout_x0,
        0.640,
        facecolor=PALETTE["PANEL"],
        edgecolor=PALETTE["ANTI"],
        linewidth=2.0,
    )
    ax.text(
        callout_x0 + 0.018,
        0.815,
        "Why the orientation is load-bearing",
        fontsize=FONTS["axis_label"],
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["ANTI"],
        zorder=4,
    )
    # Deliberately free of mathtext: an inline subscript would render below the
    # legibility floor at this body size, and the paragraph reads fine without it.
    ax.text(
        callout_x0 + 0.018,
        0.770,
        "The learned bigram belongs in B, the slot\n"
        "consulted by update_empirical_prior.\n"
        "\n"
        "Put it in A with an identity B and nothing\n"
        "raises. infer_states then returns a\n"
        "posterior over the symbol that PRECEDED\n"
        "the observed one, and the identity B\n"
        "carries that backward-looking belief\n"
        "forward as the next-symbol prediction.\n"
        "The model is anti-informative, not broken.\n"
        "\n"
        "pymdp cannot catch it: _validate() checks\n"
        "normalization only, and a bigram matrix is\n"
        "column-stochastic in either slot.\n"
        "\n"
        "The measured cost of the swap is in the\n"
        "wiring-comparison figure.",
        fontsize=FONTS["annotation"],
        va="top",
        ha="left",
        color=PALETTE["INK"],
        linespacing=1.45,
        zorder=4,
    )

    # ── Footnote ─────────────────────────────────────────────────────────────
    ax.text(
        0.0,
        0.012,
        "Filtering, not smoothing: position t sees observations 1..t only. The "
        "read-off is taken from the prior before the update, so the probability "
        "scored at each position is a genuine one-step-ahead prediction.",
        fontsize=FONTS["cell"],
        va="bottom",
        ha="left",
        color=PALETTE["MUTED"],
    )

    return fig


# ── Orchestration hook ───────────────────────────────────────────────────────


def write_architecture_figures(output_dir: Path | None = None) -> list[Path]:
    """Generate and write both architecture figures.

    Args:
        output_dir: Destination directory. Defaults to
            :func:`ailm.figures._common.default_output_dir`.

    Returns:
        Paths written, in generation order.
    """
    paths: list[Path] = []
    for name, builder in (
        ("hierarchy", generate_hierarchy_figure),
        ("inference_cycle", generate_inference_cycle_figure),
    ):
        fig = builder()
        try:
            paths.append(save_figure(fig, name, output_dir))
        finally:
            plt.close(fig)
    return paths
