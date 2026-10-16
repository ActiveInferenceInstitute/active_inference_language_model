"""The title-page graphic may not assert structure the manuscript denies.

The cover is the first thing a reader sees, and it sat outside every honesty
gate this project has: `tests/test_caption_standard.py` derives its population
from `fig:` anchors in the manuscript, and the cover has no anchor. Its own
docstring's disclaimer — "not a claim-bearing manuscript figure, so it carries
no ledger-backed values" — excuses absent numbers, not a present structural
claim.

It carried one. Under the header "hierarchical generative model" it drew a
bottom-up observation arrow and a top-down preference arrow between *every*
adjacent pair of the six levels: ten couplings. Two exist. The manuscript states
the opposite in four places, including a limitations subsection titled "The
shipped decoders exchange no beliefs".

Two invariants are gated here. The **edge sets** must match the structure the
body names: three shared-alphabet seams, which are type identities with nothing
flowing along them, and two measured preference channels. And the **layout**
must keep every string inside its panel — the note column had been silently
pushing the example strings under the second axes, where matplotlib clipped
them mid-word on the title page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from generate_cover import (  # noqa: E402
    ALPHABET_SEAMS,
    GATE_OUTCOMES,
    GATE_QUALIFIERS,
    INTERVENTIONS,
    LEVEL_BLOCKS,
)

#: The seams `docs/manuscript/03_methods.md` names as type IDENTITIES, as
#: (lower, upper): letters into the letter-to-phone model, sentence types into
#: the narrative level. The L2→L3 and L3→L4 seams are deliberately absent —
#: Level 3 reads its own within-word alphabet and Level 4 reads UPOS tags from
#: the treebank.
_DECLARED_SEAMS = frozenset({("L1", "L2"), ("L5", "L6")})

#: L4→L5 is the seam that must NOT be drawn as an identity since decision 114.
#: The canonical Level 5 observes the eight case roles plus `VERB` and `AUX`, so
#: its alphabet strictly contains Level 4's hidden alphabet. Containment is not
#: identity, and the difference is the whole of decision 112's stratificational
#: caveat, which the discussion spends a section on. Stated as its own constant
#: rather than by absence, because a seam that silently reappears is exactly the
#: defect this module exists to catch.
_CONTAINMENT_NOT_IDENTITY_SEAM = ("L4", "L5")

#: The five measured receiver interventions ([@fig:composition],
#: [@sec:hierarchy]), as (mechanism, source, receiver). The mechanism names are
#: the manuscript's own: the harmful L1->L2B coupling folds into the observation
#: likelihood over phones; the helpful L3->L2B message mixes into the phone
#: predictive; the L3->L1g generation seam sets the letter level's ``C``; the
#: L6->L5 and oracle->L3 interventions move a ``D`` prior. The oracle arm's
#: source is a privileged external oracle, NOT an L5 output — attributing it to
#: L5 would falsify [@fig:composition].
_DECLARED_INTERVENTIONS = frozenset(
    {
        ("A", "L1", "L2"),
        ("predictive", "L3", "L2"),
        ("C", "L3", "L1"),
        ("D", "L6", "L5"),
        ("D", "oracle", "L3"),
    }
)

#: The outcome words the panel is allowed to print, in the body's own register.
#: `FAIL` and `NONCONF.` were the abbreviations these replaced. `FAIL` was the
#: worse of the two: the clause model was built and audited, and it is the gate
#: that does not pass, not the level that failed to exist.
_COVER_VERDICT_WORDS = frozenset({"passing", "non-passing", "non-confirmatory"})

#: Gate outcome per built level, as recorded in the audited gate report and
#: drawn in [@fig:hierarchy]. Level 5 reads `passing` since decision 114
#: promoted the widened-observation clause arm, whose registered gate cleared
#: its refitted bar; the case-only arm it displaced keeps its non-passing record
#: in the ablation export and the body. Level 6 reads `passing` since decision
#: 115 carried its primary document-clustered estimand to the held-back split,
#: where the interval excludes zero; the development population's interval still
#: spans zero, which is what that level's qualifier says.
_DECLARED_OUTCOMES = {
    "L6": "passing",
    "L5": "passing",
    "L4": "passing",
    "L3": "passing",
    "L2": "passing",
    "L1": "passing",
}

#: What the gate audit's own two fields mean on a title page, per gate id. The
#: pair matters, and Stage 6 is why: its audited ``outcome`` has always been
#: ``PASS``, and the ``finding_status`` beside it is the whole of the
#: difference between a hedged result and a replicated one. A cover that read
#: only the outcome column would have printed ``passing`` beside Level 6 for as
#: long as the level has existed, including throughout the period when the
#: primary analysis was non-confirmatory.
#:
#: Both Stage 6 statuses are kept, not just the current one. The status is
#: computed from ledger booleans (decision 115), so a re-recorded arm can move
#: it back, and a map that had dropped the retired pair would fail on a
#: withdrawal instead of describing it.
_AUDITED_PAIR_TO_COVER_WORD = {
    ("PASS", None): "passing",
    ("FAIL", "nonpassing_oracle_gain_unresolved"): "non-passing",
    ("PASS", "positive_point_nonconfirmatory"): "non-confirmatory",
    ("PASS", "positive_point_replicated"): "passing",
}

#: Which built level each audited gate id speaks for. ``stage2_phoneme`` and
#: ``generation`` have no level block of their own — the first is Level 2's
#: other gate and the second is a cross-level generation gate — so they are
#: checked for consistency and not mapped onto a badge.
_GATE_TO_LEVEL = {
    "stage1_letter": "L1",
    "stage2_g2p": "L2",
    "stage3_word": "L3",
    "stage4_case": "L4",
    "stage5_clause": "L5",
    "stage6_narrative": "L6",
}


def test_the_cover_draws_exactly_the_declared_seams() -> None:
    assert frozenset(ALPHABET_SEAMS) == _DECLARED_SEAMS, (
        "the cover's alphabet seams no longer match the ones the architecture "
        "section names as identities; a seam drawn between two other levels "
        "asserts a type identity that does not exist"
    )


def test_the_cover_does_not_draw_the_case_to_clause_seam_as_an_identity() -> None:
    """The canonical Level 5's alphabet CONTAINS Level 4's; it does not equal it.

    Decision 114 promoted the widened-observation clause model, whose alphabet
    is the eight case roles plus `VERB` and `AUX`. Drawing a plain identity line
    from L4 to L5 would assert on the title page the reading that
    [@sec:widened_caveat] denies, and the title page is where a reader starts.
    """
    from ailm.cases import CaseRole
    from ailm.levels.widened import N_WIDENED_SYMBOLS, WIDENED_SYMBOL_LABELS

    assert _CONTAINMENT_NOT_IDENTITY_SEAM not in frozenset(ALPHABET_SEAMS), (
        "the cover draws L4→L5 as an alphabet identity again. The canonical "
        "Level 5 observes symbols Level 4 has no hidden state for."
    )
    # NON-VACUITY: the seam is wrong because the alphabets genuinely differ, not
    # because someone deleted a line. A widening that collapsed back to the case
    # roles would make the check above a rule with nothing behind it.
    role_names = tuple(role.name for role in CaseRole)
    assert N_WIDENED_SYMBOLS > len(role_names)
    assert set(role_names) < set(WIDENED_SYMBOL_LABELS)


def test_the_cover_draws_exactly_the_declared_interventions() -> None:
    """All five measured receiver interventions, each with a manuscript mechanism."""
    assert len(INTERVENTIONS) == len(_DECLARED_INTERVENTIONS), (
        "the cover must draw every measured receiver intervention the "
        "manuscript reports ([@sec:hierarchy]); omitting one quietly narrows "
        "the paper's central measured result"
    )
    drawn = frozenset((mech, src, rcv) for mech, src, rcv, _words, _outcome in INTERVENTIONS)
    assert drawn == _DECLARED_INTERVENTIONS, (
        "the cover's intervention set no longer matches the five [@sec:hierarchy] "
        "reports; every extra arrow is a coupling claim the body refutes, every "
        "missing one hides a measurement"
    )


def test_the_preferred_channel_mechanisms_are_labeled() -> None:
    """The C/D split the manuscript insists on: L6->L5 moves D, not C."""
    mechanisms = {
        src: mech for mech, src, _rcv, _words, _outcome in INTERVENTIONS if src in {"L6", "L3"}
    }
    assert mechanisms["L6"] == "D", (
        "the manuscript states the tercile intervention 'does not use the "
        "generator's C mechanism' — it moves the clause level's joint D prior; "
        "labeling it C would redraw a mechanism the body denies"
    )
    assert mechanisms["L3"] == "C", (
        "the L3->L1g seam is the only place one level sets another's C "
        "([@sec:generation]); drawing it as D erases the one genuine "
        "preference seam"
    )


def test_the_oracle_arm_is_not_drawn_as_an_l5_output() -> None:
    """[@fig:composition] is falsified by attributing the oracle arm to L5."""
    oracle = [entry for entry in INTERVENTIONS if entry[1] == "oracle"]
    assert oracle and oracle[0][2] == "L3", (
        "the privileged punctuation-type-oracle intervention acts on L3's word "
        "prior and is not an L5-output channel ([@sec:hierarchy])"
    )


def test_gate_outcomes_match_the_audited_report(gate_audit_rows) -> None:
    """The outcome words on the cover agree with the audited gate verdicts.

    Reads the generated gate audit itself, so a re-audit that flips an outcome
    fails this test until the cover is regenerated — the cover may not
    outlive the evidence it summarizes.
    """
    audited = {
        gate_id: (row["outcome"], row.get("finding_status"))
        for gate_id, row in gate_audit_rows.items()
    }
    assert set(GATE_OUTCOMES) == {"L1", "L2", "L3", "L4", "L5", "L6"}
    assert GATE_OUTCOMES == _DECLARED_OUTCOMES

    for gate_id, pair in audited.items():
        assert pair in _AUDITED_PAIR_TO_COVER_WORD, (
            f"gate {gate_id} was audited as {pair}, which no cover word stands "
            "for. A re-audit that produces a new outcome or finding must be "
            "given a word here before the cover can claim to summarize it."
        )
        level = _GATE_TO_LEVEL.get(gate_id)
        if level is None:
            # stage2_phoneme and generation have no badge of their own; they
            # still have to be a recognized pair, checked just above.
            continue
        assert GATE_OUTCOMES[level] == _AUDITED_PAIR_TO_COVER_WORD[pair], (
            f"{level}'s badge says {GATE_OUTCOMES[level]!r} but {gate_id} was "
            f"audited as {pair}. The cover may not outlive the evidence."
        )

    assert set(_GATE_TO_LEVEL.values()) == set(GATE_OUTCOMES), (
        "every level block must be backed by an audited gate id"
    )


def test_the_level_blocks_name_the_six_built_levels() -> None:
    assert [tag for tag, _name in LEVEL_BLOCKS] == ["L6", "L5", "L4", "L3", "L2", "L1"]
    names = dict(LEVEL_BLOCKS)
    assert names == {
        "L6": "Doc-position",
        "L5": "Clause",
        "L4": "Case",
        "L3": "Word",
        "L2": "Phoneme",
        "L1": "Letter",
    }


def test_the_all_adjacent_pairs_drawing_would_fail_this_gate() -> None:
    """NEGATIVE CONTROL: the exact defect this file exists to prevent.

    Not hypothetical — this is the edge set the cover shipped with.
    """
    order = ("L6", "L5", "L4", "L3", "L2", "L1")
    all_pairs_up = frozenset((order[i], order[i - 1]) for i in range(1, len(order)))
    all_pairs_down = frozenset((order[i - 1], order[i]) for i in range(1, len(order)))
    assert all_pairs_up != _DECLARED_SEAMS
    adjacent_pairs = frozenset(
        (src, rcv) for _mech, src, rcv, _words, _outcome in INTERVENTIONS if src != "oracle"
    )
    assert all_pairs_down != adjacent_pairs
    assert len(all_pairs_down) == 5, "five couplings per direction, ten in total"


def test_the_header_does_not_claim_a_hierarchical_generative_model() -> None:
    source = (_ROOT / "scripts" / "generate_cover.py").read_text(encoding="utf-8")
    # The word may appear in the explanatory comment; it may not be a drawn string.
    drawn = [
        line
        for line in source.splitlines()
        if '"' in line
        and "hierarchical generative model" in line.lower()
        and not line.lstrip().startswith("#")
    ]
    assert not drawn, (
        f"the cover draws the phrase the manuscript denies: {drawn}. "
        "[@sec:architecture] and [@sec:limitations] state that the shipped "
        "decoders exchange no beliefs."
    )


def test_the_cover_does_not_call_level_six_narrative() -> None:
    """The cover shipped the reading [@sec:narrative_level] denies in as many words.

    L6's hidden state is the document-position tercile, and the section says
    plainly that "the tercile is a position, not a narrative phase". The cover
    labeled the level *Narrative* and annotated it *phase: opening*, on the first
    page a reader sees, which is the strongest placement any claim in the paper
    gets. The example beside it went further and advertised tense and topic, and
    no level of this system models either.

    The rule is stated over the class rather than the retired strings: any drawn
    string that names a narrative construct fails, so a different word for the
    same overclaim is caught too.
    """
    import re

    source = (_ROOT / "scripts" / "generate_cover.py").read_text(encoding="utf-8")
    forbidden = re.compile(r"narrative|\bphase\b|\btopic\b|\btense\b|\bpast\b", re.I)
    drawn = [
        line.strip()
        for line in source.splitlines()
        if '"' in line
        and not line.lstrip().startswith("#")
        and forbidden.search(re.sub(r"^[^\"]*", "", line))
    ]
    assert not drawn, (
        f"the cover draws a narrative construct L6 does not have: {drawn}. "
        "L6's state is a document-position tercile."
    )


@pytest.mark.parametrize("panel", ["left", "right"])
def test_no_string_overflows_its_panel(panel: str) -> None:
    """Every text stays inside the axes that owns it.

    POSITIVE CONTROL for the layout half. Rendered rather than reasoned about:
    the previous layout satisfied every arithmetic constraint its own comment
    described and still ran the example column off the panel, because the
    comment reasoned about the note column and not about the panel edge.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from generate_cover import build_cover

    figure = build_cover()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes = sorted(figure.axes, key=lambda a: a.get_position().x0)
    ax = axes[0] if panel == "left" else axes[-1]
    box = ax.get_window_extent()

    overflowing = []
    for text in ax.texts:
        if not text.get_text().strip():
            continue
        extent = text.get_window_extent(renderer)
        if extent.x0 < box.x0 - 1.0 or extent.x1 > box.x1 + 1.0:
            overflowing.append(text.get_text())
    plt.close(figure)

    assert not overflowing, (
        f"{panel} panel strings run outside their axes: {overflowing}. On the "
        "title page matplotlib draws them under the neighboring axes and clips "
        "them mid-word."
    )


# ── Style-source invariants ──────────────────────────────────────────────────
#
# The cover draws from ``src/ailm/viz_config.py`` like every other figure: the
# palette keys carry the meanings, the FONTS scale carries the sizes. A raw hex
# or a raw ``fontsize=`` in the generator silently opts the title page out of
# that contract.


def _drawn_lines(source: str) -> list[str]:
    """Lines of the generator that execute (not comments) and contain a string."""
    return [
        line for line in source.splitlines() if '"' in line and not line.lstrip().startswith("#")
    ]


def test_the_generator_uses_palette_keys_and_no_raw_hex() -> None:
    """POSITIVE CONTROL: the shipped generator passes."""
    source = (_ROOT / "scripts" / "generate_cover.py").read_text(encoding="utf-8")
    import re

    drawn = [line for line in _drawn_lines(source) if re.search(r'"#[0-9a-fA-F]{6}"', line)]
    assert not drawn, (
        f"the cover hard-codes colors outside PALETTE: {drawn}. Colors come "
        "from src/ailm/viz_config.py, where keys name meanings; a raw hex on "
        "the title page is unaccountable to the palette contract."
    )


def test_the_palette_hex_detector_catches_a_raw_hex() -> None:
    """NEGATIVE CONTROL: a known-wrong line is flagged by the same detector."""
    import re

    wrong = 'axr.text(0.5, 0.5, "x", color="#123ABC")  # drawn, not a comment'
    assert re.search(r'"#[0-9a-fA-F]{6}"', wrong)
    flagged = [line for line in _drawn_lines(wrong) if re.search(r'"#[0-9a-fA-F]{6}"', line)]
    assert flagged == [wrong]


def test_the_generator_uses_font_roles_and_no_raw_fontsize() -> None:
    """POSITIVE CONTROL: every drawn size is a FONTS role or a named constant."""
    source = (_ROOT / "scripts" / "generate_cover.py").read_text(encoding="utf-8")
    import re

    drawn = [line for line in _drawn_lines(source) if re.search(r"fontsize=[0-9.]", line)]
    assert not drawn, (
        f"the cover hard-codes font sizes outside FONTS: {drawn}. Sizes come "
        "from the shared type scale so the title page cannot drift from the "
        "figures it introduces."
    )


def test_the_fontsize_detector_catches_a_raw_size() -> None:
    """NEGATIVE CONTROL: a known-wrong line is flagged by the same detector."""
    import re

    wrong = 'axs.text(0.5, 0.5, "x", fontsize=13)'
    flagged = [line for line in _drawn_lines(wrong) if re.search(r"fontsize=[0-9.]", line)]
    assert flagged == [wrong]


def test_example_strings_carry_their_producing_heuristics() -> None:
    """Every example instance names the mechanism that produces it.

    The cover is a claim surface: the L4 row shows the case role the
    ``UD_DEPREL_TO_CASE`` heuristic assigns, and the row must say so, or the
    title page advertises an output with no account of where it came from.
    """
    source = (_ROOT / "scripts" / "generate_cover.py").read_text(encoding="utf-8")
    assert "UD_DEPREL_TO_CASE" in source, (
        "the L4 example string must name its producing heuristic "
        "(UD_DEPREL_TO_CASE), not just show its output"
    )
    # And the heuristic really assigns what the example shows: nsubj -> NOM.
    sys.path.insert(0, str(_ROOT / "src"))
    from ailm.cases import UD_DEPREL_TO_CASE, CaseRole

    assert UD_DEPREL_TO_CASE["nsubj"] is CaseRole.NOM, (
        "the cover shows cat:NOM as the L4 instance; the producing table "
        "must still map the subject deprel to Nominative"
    )


def test_no_verdict_badge_is_an_abbreviation() -> None:
    """The badges carry the body's words, not compressions of them.

    Stated over the class rather than over the two retired strings: `FAIL` and
    `NONCONF.` are gone, but a future `NONCONF` or `UNRES.` would be the same
    defect, and a rule naming the old strings would reprint what it exists to
    remove. The cover is the one page a reader meets before any definition, so a
    badge it cannot expand is a badge it cannot defend.
    """
    for level, word in GATE_OUTCOMES.items():
        assert word == word.lower(), f"{level}'s badge {word!r} is shouted, not written"
        assert not word.endswith("."), f"{level}'s badge {word!r} is truncated"
        assert " " not in word.strip(), (
            f"{level}'s badge {word!r} should be one word or hyphenation"
        )
        assert word in _COVER_VERDICT_WORDS, (
            f"{level}'s badge {word!r} is not one of the manuscript's outcome words"
        )


def test_the_verdict_words_the_cover_prints_are_the_readmes_own() -> None:
    """POSITIVE CONTROL: every badge word appears verbatim in the README.

    Not a style preference. If the cover and the README describe the same
    results with different words, a reader has to work out whether they are the
    same finding, and the cover is where they start.

    Derived from the badges rather than typed, so a promotion or a withdrawal
    that changes which words the cover prints changes what this checks. Typed,
    it would have gone on demanding `non-passing` of the README after decision
    114 stopped printing it, and stopped demanding whatever replaced it.
    """
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    printed = set(GATE_OUTCOMES.values())
    assert printed, "no badge words to check; this control is vacuous"
    for word in sorted(printed):
        assert word in readme, f"the cover prints {word!r}; the README does not"


def test_the_two_qualified_outcomes_carry_the_qualifier_they_need() -> None:
    """A bare outcome word misleads on exactly these two levels.

    Both now read `passing`, and both are qualified for the same underlying
    reason: what the badge licenses is narrower than what Stages 1 through 4
    carry, and the title page is where a reader forms that impression.

    L5's arm was registered and answered inside one development session, and it
    can never acquire a held-back replication because decision 111 spent that
    population.

    L6's arm did reach that population, and its badge changed with decision 115
    when the primary estimand excluded zero there. The qualifier carries what
    the promotion does not erase: the development population's own clustered
    interval still spans zero, the two populations disagree, and a reader who
    sees only the held-back half has been told half the result.
    """
    assert set(GATE_QUALIFIERS) == {"L5", "L6"}
    for level, qualifier in GATE_QUALIFIERS.items():
        # A qualifier limits what the badge licenses. One that restates or
        # contradicts an outcome word is a second verdict smuggled in beside the
        # first, which is the defect the badge vocabulary exists to prevent.
        for word in _COVER_VERDICT_WORDS:
            assert word not in qualifier, (
                f"{level}'s qualifier {qualifier!r} contains the outcome word "
                f"{word!r}; a qualifier bounds a badge, it does not restate one"
            )
    assert "development-session" in GATE_QUALIFIERS["L5"]
    assert "no held-back replication" in GATE_QUALIFIERS["L5"]
    assert "positive point estimate" in GATE_QUALIFIERS["L6"]
    # Both halves, not just the flattering one. A qualifier that named only the
    # held-back interval would let the promotion quietly retire the development
    # result instead of standing beside it.
    assert "development interval spans zero" in GATE_QUALIFIERS["L6"]
    assert "held-back interval does not" in GATE_QUALIFIERS["L6"]
    assert "family-corrected interval spans zero" in GATE_QUALIFIERS["L6"]

    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "gain unresolved" in readme
    assert "interval spans zero" in readme


def test_the_cover_reads_its_level_facts_from_the_architecture_table() -> None:
    """The per-level facts are the figure layer's, not a second copy of them.

    The cover now states each level's hidden state, observation, likelihood and
    dynamics. Restating those here would create a second place for them to be
    true, and the two would drift the first time a level's `A` changed. They are
    read from the same `LevelRow` table `fig:hierarchy` draws.
    """
    sys.path.insert(0, str(_ROOT / "src"))
    from generate_cover import _level_facts

    from ailm.figures.architecture import LEVELS

    facts = _level_facts()
    built = {row.level: row for row in LEVELS if row.built}

    assert set(facts) == set(built)
    assert set(facts) == {tag for tag, _name in LEVEL_BLOCKS}
    assert "L7" not in facts, (
        "Level 7 is the external DSP boundary, not a built level; a cover row "
        "for it would advertise a model that does not exist"
    )
    for tag, row in built.items():
        hidden, observation, likelihood, dynamics = facts[tag]
        assert hidden == " ".join(row.hidden.split())
        assert observation == " ".join(row.observation.split())
        assert likelihood == " ".join(row.emission.split())
        assert dynamics == " ".join(row.transition.split())


def test_the_level_blocks_agree_with_the_architecture_table() -> None:
    """Tags and names come from the same table as the facts beside them."""
    sys.path.insert(0, str(_ROOT / "src"))
    from ailm.figures.architecture import LEVELS

    built = [(row.level, row.name) for row in LEVELS if row.built]
    # The cover draws top-down; the table is ordered bottom-up.
    assert list(LEVEL_BLOCKS) == list(reversed(built))


def test_every_intervention_names_a_mechanism_and_an_outcome_in_words() -> None:
    """Two or three words beside an arrow cannot carry both, so neither is there.

    The shipped cover labeled the harmful coupling `harms (A)`. That is a
    mechanism a reader has to already know to decode, next to an outcome with no
    stated estimand. Each intervention now carries a spelled-out mechanism and a
    spelled-out outcome, drawn in the panel's legend.
    """
    outcomes = set()
    for _mech, src, rcv, words, outcome in INTERVENTIONS:
        assert len(words.split()) >= 4, (
            f"the {src}->{rcv} mechanism is still abbreviated: {words!r}"
        )
        assert outcome and outcome == outcome.lower()
        assert "(" not in outcome, (
            f"the {src}->{rcv} outcome {outcome!r} hides a mechanism in parentheses"
        )
        outcomes.add(outcome)
    assert outcomes == {"harms", "helps", "prior only"}, (
        "the drawn outcomes must stay the three the manuscript reports; a new "
        "word here is a new claim about a measured intervention"
    )


def test_the_order_labels_clear_every_box_and_label_on_their_panel() -> None:
    """POSITIVE CONTROL: the rotated gutter labels collide with nothing.

    The shipped cover drew "generation order (top-down)" at x=0.112 with the
    level boxes starting at 0.135. A rotated string's bounding box is about one
    line height wide, so the label printed onto the box edges on the first page
    a reader sees. Reasoning about the arithmetic is what produced that defect;
    this measures the rendered extents instead, against every box AND every
    other string on the panel, because the second attempt at the fix cleared the
    boxes and landed on the field-label column instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from generate_cover import build_cover

    figure = build_cover()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panel = sorted(figure.axes, key=lambda a: a.get_position().x0)[0]

    labels = [text for text in panel.texts if text.get_gid() == "direction-label"]
    boxes = [patch for patch in panel.patches if patch.get_gid() == "level-box"]
    assert len(labels) == 2, "both order labels must be tagged or this gate is vacuous"
    assert len(boxes) == len(LEVEL_BLOCKS), "every level box must be tagged"

    obstacles = [
        (f"level box {index}", box.get_window_extent(renderer)) for index, box in enumerate(boxes)
    ]
    obstacles += [
        (repr(text.get_text()), text.get_window_extent(renderer))
        for text in panel.texts
        if text.get_gid() != "direction-label" and text.get_text().strip()
    ]

    collisions = [
        (label.get_text(), name)
        for label in labels
        for name, extent in obstacles
        if label.get_window_extent(renderer).overlaps(extent)
    ]
    plt.close(figure)

    assert not collisions, (
        f"the rotated order labels overlap panel content: {collisions}. On the "
        "title page they print on top of it."
    )


def test_the_overlap_detector_reports_a_known_collision() -> None:
    """NEGATIVE CONTROL: the same `overlaps` call flags two boxes that do."""
    from matplotlib.transforms import Bbox

    assert Bbox([[0.0, 0.0], [10.0, 10.0]]).overlaps(Bbox([[5.0, 5.0], [15.0, 15.0]]))
    assert not Bbox([[0.0, 0.0], [10.0, 10.0]]).overlaps(Bbox([[11.0, 11.0], [15.0, 15.0]]))


def test_gate_qualifiers_do_not_cross_level_boxes() -> None:
    """Measure qualifiers against drawn boxes, including a planted collision."""
    import matplotlib.pyplot as plt
    from generate_cover import build_cover
    from matplotlib.patches import FancyBboxPatch

    figure = build_cover()
    try:
        panel = sorted(figure.axes, key=lambda axes: axes.get_position().x0)[-1]
        labels = [text for text in panel.texts if text.get_text() in GATE_QUALIFIERS.values()]
        boxes = [patch for patch in panel.patches if isinstance(patch, FancyBboxPatch)]
        assert len(labels) == len(GATE_QUALIFIERS)
        assert len(boxes) >= len(LEVEL_BLOCKS)

        def collisions() -> list[str]:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            return [
                text.get_text()
                for text in labels
                for box in boxes
                if text.get_window_extent(renderer).overlaps(box.get_window_extent(renderer))
            ]

        assert collisions() == []
        labels[0].set_position((0.5, boxes[0].get_y() + boxes[0].get_height()))
        assert collisions(), "a qualifier planted across a box did not trip the detector"
    finally:
        plt.close(figure)


def test_the_cover_states_the_inference_and_declaration_chain() -> None:
    """The structural summary on the panel matches what the architecture section claims."""
    source = (_ROOT / "scripts" / "generate_cover.py").read_text(encoding="utf-8")
    assert "GNN-declared model" in source and "conformance-checked" in source, (
        "the cover must state that every level is GNN-declared and "
        "conformance-checked ([@sec:architecture], [@fig:gnn_pipeline])"
    )
    assert "generative-model inference" in source, (
        "the engine phrasing must stay at the evidence level: "
        "'generative-model inference' until live pymdp per-level evidence "
        "lands; 'pymdp free-energy inference' would overclaim the cover "
        "before the manuscript's own sections say it"
    )


@pytest.mark.timeout(300)
def test_cover_publication_checks_real_missing_current_and_stale_files(tmp_path: Path) -> None:
    """Check mode must reject missing/tampered covers without repairing them."""
    from generate_cover import main

    cover = tmp_path / "cover.png"
    args = ["--out", str(cover)]
    assert main([*args, "--check"]) == 1
    assert not cover.exists()

    assert main(args) == 0
    current = cover.read_bytes()
    assert current.startswith(b"\x89PNG\r\n\x1a\n")
    assert main([*args, "--check"]) == 0
    assert cover.read_bytes() == current

    stale = current + b"stale cover control"
    cover.write_bytes(stale)
    assert main([*args, "--check"]) == 1
    assert cover.read_bytes() == stale
    assert main(args) == 0
    assert cover.read_bytes() == current


@pytest.mark.timeout(300)
def test_published_cover_matches_its_generator() -> None:
    """The ordinary suite must detect a cover left over from an earlier source."""
    from generate_cover import main

    assert main(["--check"]) == 0
