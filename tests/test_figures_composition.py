"""Tests for the composition figures.

No mocks. Every test builds a real matplotlib figure, and the writing tests write
a real PNG to a real temporary directory and read the bytes back.

The load-bearing test in this file is not a layout test. Layout is covered by
``tests/test_figure_layout.py``, which sweeps every discovered builder. What is
covered *here* and nowhere else is **content**: ``fig:composition`` exists to say
that the AILM levels do not exchange beliefs, and a figure that drifted into
implying they do would still render perfectly, still pass the legibility floor,
and still pass the overlap sweep. :func:`implied_message_passing` scans the
rendered text for the vocabulary of inter-agent messaging and requires every
occurrence to be a denial, and
:meth:`TestCompositionIsHonest.test_the_scanner_catches_an_injected_claim` is the
positive control proving that scan can fail — it plants an offending line on the
real figure, not on a fixture.

The scanner is deliberately lexical, which makes it a necessary condition and not
a sufficient one. It cannot decide whether a *drawing* implies a message; the
plain, arrowhead-free connector in the figure is the answer to that, and the
reason is stated at the call site in ``ailm.figures.composition``.
"""

from __future__ import annotations

import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text

from ailm.figures._common import (
    MissingVariableError,
    assert_legible,
    load_manuscript_variables,
    project_root,
    save_figure,
    token_float,
)
from ailm.figures.composition import (
    COUPLING_TOKENS,
    NO_MESSAGE_MARKER,
    generate_composition_figure,
    generate_gnn_pipeline_figure,
    write_composition_figures,
)
from ailm.gnn import load_spec
from ailm.levels.letter import N_SYMBOLS
from ailm.levels.phoneme import N_PHONES

warnings.filterwarnings("ignore", message=".*JAX array is being set as static.*")

#: Below this a PNG is a blank canvas, not a figure. The real ones land in the
#: hundreds of kilobytes.
_MIN_PNG_BYTES = 20_000

#: The vocabulary of agent-to-agent messaging. A line of figure text containing
#: any of these is making a claim about inter-level communication.
MESSAGING_WORDS: tuple[str, ...] = (
    "message",
    "messages",
    "messaging",
    "passes",
    "passing",
    "sends",
    "exchange",
    "exchanges",
)

#: Tokens that turn such a line into a denial. Matched on word boundaries — a
#: bare substring test would accept "notation" as a negation of "not".
NEGATION_WORDS: tuple[str, ...] = (
    "no",
    "not",
    "never",
    "none",
    "nothing",
    "without",
)

_MESSAGING_RE = re.compile(r"\b(?:" + "|".join(MESSAGING_WORDS) + r")\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:" + "|".join(NEGATION_WORDS) + r")\b", re.IGNORECASE)


def figure_lines(fig: Figure) -> list[str]:
    """Every visible line of text on ``fig``, one entry per rendered line.

    Multi-line artists are split, because negation binds to a sentence: a block
    whose first line asserts a channel and whose fourth line denies one would
    pass a whole-artist check while reading as a claim.
    """
    lines: list[str] = []
    for artist in fig.findobj(Text):
        if not artist.get_visible():
            continue
        for line in artist.get_text().split("\n"):
            if line.strip():
                lines.append(line.strip())
    return lines


def implied_message_passing(fig: Figure) -> list[str]:
    """Return every line that names inter-agent messaging without denying it.

    Args:
        fig: The figure to scan.

    Returns:
        Offending lines. Empty means no line on the figure asserts that the
        levels communicate.
    """
    offenders: list[str] = []
    for line in figure_lines(fig):
        if not _MESSAGING_RE.search(line):
            continue
        if _NEGATION_RE.search(line) or "✗" in line:
            continue
        offenders.append(line)
    return offenders


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """The generated manuscript token map."""
    return load_manuscript_variables()


# ── The figures build, and they are legible ──────────────────────────────────


class TestBuildersReturnFigures:
    """Both generators hand back a real, legible Figure."""

    def test_gnn_pipeline_returns_a_legible_figure(self) -> None:
        fig = generate_gnn_pipeline_figure()
        try:
            assert isinstance(fig, Figure)
            assert_legible(fig)
        finally:
            plt.close(fig)

    def test_composition_returns_a_legible_figure(self) -> None:
        fig = generate_composition_figure()
        try:
            assert isinstance(fig, Figure)
            assert_legible(fig)
        finally:
            plt.close(fig)


# ── Writing and determinism ──────────────────────────────────────────────────


class TestWriting:
    """Real PNGs, written and read back."""

    @pytest.mark.parametrize(
        "name,builder",
        [
            ("gnn_pipeline", generate_gnn_pipeline_figure),
            ("composition", generate_composition_figure),
        ],
    )
    def test_writes_a_nontrivial_png(self, tmp_path, name, builder) -> None:
        fig = builder()
        try:
            path = save_figure(fig, name, tmp_path)
        finally:
            plt.close(fig)
        assert path.name == f"{name}.png"
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size > _MIN_PNG_BYTES, (
            f"{name}.png is {path.stat().st_size} bytes — that is a blank canvas, not a figure."
        )

    @pytest.mark.parametrize(
        "builder",
        [generate_gnn_pipeline_figure, generate_composition_figure],
        ids=["gnn_pipeline", "composition"],
    )
    def test_two_runs_produce_identical_bytes(self, tmp_path, builder) -> None:
        """Regeneration must be byte-identical, or "up to date" is not checkable."""
        written = []
        for run in ("first", "second"):
            fig = builder()
            try:
                written.append(save_figure(fig, run, tmp_path).read_bytes())
            finally:
                plt.close(fig)
        assert written[0] == written[1], (
            f"{builder.__name__} is not deterministic: "
            f"{len(written[0])} vs {len(written[1])} bytes."
        )

    def test_the_byte_comparison_can_distinguish_two_figures(self, tmp_path) -> None:
        """POSITIVE CONTROL for the determinism check.

        If ``save_figure`` wrote a constant, the two tests above would pass while
        proving nothing about determinism.
        """
        first = generate_gnn_pipeline_figure()
        try:
            a = save_figure(first, "a", tmp_path).read_bytes()
        finally:
            plt.close(first)
        second = generate_composition_figure()
        try:
            b = save_figure(second, "b", tmp_path).read_bytes()
        finally:
            plt.close(second)
        assert a != b

    def test_write_composition_figures_writes_both(self, tmp_path) -> None:
        paths = write_composition_figures(tmp_path)
        assert [p.name for p in paths] == ["composition.png", "gnn_pipeline.png"]
        for path in paths:
            assert path.stat().st_size > _MIN_PNG_BYTES

    def test_write_composition_figures_closes_its_figures(self, tmp_path) -> None:
        """A generator that leaks figures exhausts memory over a full render."""
        plt.close("all")
        write_composition_figures(tmp_path)
        assert plt.get_fignums() == []


# ── THE CONTENT GATE: the figure must not imply message passing ──────────────


class TestCompositionIsHonest:
    """What keeps fig:composition honest as it gets edited.

    A layout sweep cannot see this class of defect at all: a figure asserting
    that L1 hands its belief to L2B would render beautifully.
    """

    def test_the_scanner_catches_an_injected_claim(self) -> None:
        """POSITIVE CONTROL — required, and planted on the production figure.

        Without it, ``implied_message_passing(fig) == []`` would also pass if the
        regex stopped matching or the text walk returned nothing.
        """
        fig = generate_composition_figure()
        try:
            assert implied_message_passing(fig) == [], (
                "the real figure must be clean before the control is planted"
            )
            fig.text(0.5, 0.5, "L1 passes its predictive belief up to L2B")
            offenders = implied_message_passing(fig)
            assert offenders == ["L1 passes its predictive belief up to L2B"], (
                f"the scanner missed a planted claim; it reported {offenders}"
            )
        finally:
            plt.close(fig)

    def test_the_scanner_catches_a_claim_buried_in_a_multiline_block(self) -> None:
        """POSITIVE CONTROL — negation on another line must not launder a claim."""
        fig = generate_composition_figure()
        try:
            fig.text(
                0.5,
                0.5,
                "No coupling is enabled.\nL2B sends its posterior back to L1.",
            )
            assert implied_message_passing(fig) == ["L2B sends its posterior back to L1."]
        finally:
            plt.close(fig)

    def test_the_scanner_accepts_a_denial(self) -> None:
        """NEGATIVE CONTROL — the scanner must not reject honest wording.

        Otherwise the only way to satisfy it would be to delete the marker, which
        is the opposite of what it is for.
        """
        fig = generate_composition_figure()
        try:
            fig.text(0.5, 0.5, "There is no message channel between the agents.")
            assert implied_message_passing(fig) == []
        finally:
            plt.close(fig)

    def test_negation_matching_is_word_bounded(self) -> None:
        """A word merely *containing* a negation must not launder a claim.

        ``"notation"`` contains ``"not"``; a substring test would accept
        "the notation passes beliefs between levels" as a denial.
        """
        fig = generate_composition_figure()
        try:
            fig.text(0.5, 0.5, "the notation passes beliefs between levels")
            assert implied_message_passing(fig) == ["the notation passes beliefs between levels"]
        finally:
            plt.close(fig)

    def test_composition_figure_implies_no_message_passing(self) -> None:
        """THE GATE. Every messaging word on the figure must be a denial."""
        fig = generate_composition_figure()
        try:
            offenders = implied_message_passing(fig)
            assert offenders == [], (
                "fig:composition contains text a reader would take as "
                f"agent-to-agent inference: {offenders}. The levels do not "
                "exchange beliefs — update_empirical_prior is used only for "
                "within-level temporal recursion, and the one channel that was "
                "built measured harmful. See docs/decisions.md entry 9."
            )
        finally:
            plt.close(fig)

    def test_the_no_coupling_marker_is_drawn(self) -> None:
        """The absence must be a visible element, not merely an omission."""
        fig = generate_composition_figure()
        try:
            assert NO_MESSAGE_MARKER in figure_lines(fig), (
                f"{NO_MESSAGE_MARKER!r} is not drawn on fig:composition. A reader "
                "who only looks at the figure would have nothing telling them the "
                "levels never communicate."
            )
        finally:
            plt.close(fig)

    def test_the_marker_itself_denies_rather_than_asserts(self) -> None:
        """Guards the marker against being rewritten into a claim.

        The marker is the one string on the figure that is *required* to contain
        a messaging word, so it is the one place where a careless edit could turn
        the figure's headline into the opposite of its meaning while every other
        test stayed green.
        """
        assert _MESSAGING_RE.search(NO_MESSAGE_MARKER), (
            "the marker no longer mentions messaging at all, so it no longer denies anything"
        )
        assert _NEGATION_RE.search(NO_MESSAGE_MARKER), (
            f"{NO_MESSAGE_MARKER!r} names messaging without denying it"
        )

    def test_the_pipeline_figure_is_also_clean(self) -> None:
        """The same gate, applied to the sibling figure."""
        fig = generate_gnn_pipeline_figure()
        try:
            assert implied_message_passing(fig) == []
        finally:
            plt.close(fig)


# ── fig:composition reports the ledger, not a remembered number ──────────────


class TestCompositionUsesMeasuredValues:
    """The coupling null is printed from the claims ledger."""

    def test_every_coupling_token_is_printed_verbatim(self, variables) -> None:
        """The figure must print the same strings the manuscript prints."""
        fig = generate_composition_figure()
        try:
            blob = " | ".join(figure_lines(fig))
            for token in COUPLING_TOKENS:
                assert variables[token] in blob, (
                    f"{token} = {variables[token]!r} does not appear in fig:composition."
                )
        finally:
            plt.close(fig)

    def test_a_missing_measurement_fails_the_figure(self) -> None:
        """A figure that made up a number would be worse than one that failed."""
        with pytest.raises(MissingVariableError, match="do not put the number"):
            generate_composition_figure({"SOMETHING_ELSE": "1.0"})

    def test_the_panel_calls_it_a_loss_and_the_ledger_agrees(self, variables) -> None:
        """The substantive claim, checked against the ledger rather than drawn.

        The panel says the coupling cost accuracy. If the recorded margin ever
        turned positive the figure would still render, still print a ledger
        value, and still say "a LOSS" — and be false.
        """
        margin = token_float(variables, "COUPLING_MARGIN_PTS")
        assert margin < 0.0, (
            f"COUPLING_MARGIN_PTS is {margin}, which is a gain. fig:composition "
            "describes the coupling as harmful; re-derive the panel before "
            "shipping it."
        )
        best = token_float(variables, "COUPLING_BEST_ACCURACY")
        uncoupled = token_float(variables, "COUPLING_ALPHA0_ACCURACY")
        assert best < uncoupled, (
            f"the best coupled decode ({best}) beats the uncoupled anchor "
            f"({uncoupled}); the panel's ordering is wrong."
        )
        assert token_float(variables, "COUPLING_SUPPORTED") == 0.0, (
            "the recorded support flag says the coupling IS supported, which contradicts the panel."
        )


# ── fig:composition agrees with the code and the spec files ──────────────────


class TestCompositionAgreesWithTheCode:
    """A figure that contradicts the repository is a defect that renders fine."""

    def test_every_named_spec_file_exists(self) -> None:
        """Each card names its spec; a renamed spec must fail here, not mislead."""
        fig = generate_composition_figure()
        try:
            named = [line for line in figure_lines(fig) if line.startswith("gnn/")]
        finally:
            plt.close(fig)
        assert len(named) == 3, f"expected one spec file per card, got {named}"
        for relative in named:
            assert (project_root() / relative).is_file(), (
                f"fig:composition names {relative}, which is not on disk."
            )

    def test_alphabet_sizes_come_from_the_code(self) -> None:
        """A hardcoded 30 or 39 would drift when an inventory changes."""
        fig = generate_composition_figure()
        try:
            blob = " | ".join(figure_lines(fig))
        finally:
            plt.close(fig)
        assert f"letter ({N_SYMBOLS} symbols)" in blob
        assert f"phone ({N_PHONES} ARPAbet)" in blob
        assert f"({N_SYMBOLS}, {N_PHONES}) learned emission" in blob

    def test_only_l2b_is_marked_as_a_learned_emission(self) -> None:
        """The distinction the figure exists to draw must be drawn once."""
        fig = generate_composition_figure()
        try:
            lines = figure_lines(fig)
        finally:
            plt.close(fig)
        identity = [line for line in lines if line.startswith("A = I")]
        learned = [line for line in lines if line.startswith("A learned")]
        assert len(identity) == 2, identity
        assert len(learned) == 1, learned


# ── fig:gnn_pipeline reads the spec rather than restating it ─────────────────


class TestPipelineReadsTheSpec:
    """The declared dimensions on the panel come out of the shipped spec file."""

    def test_declared_dimensions_match_the_g2p_spec(self) -> None:
        spec = load_spec("L2b_grapheme_to_phoneme")
        fig = generate_gnn_pipeline_figure()
        try:
            blob = " | ".join(figure_lines(fig))
        finally:
            plt.close(fig)
        for name in ("A", "B", "C", "D"):
            declared = ",".join(str(d) for d in spec.variable_dims(name))
            assert f"{name}[{declared}]" in blob, (
                f"the panel does not show {name}[{declared}] as gnn/models/L2b_grapheme_to_phoneme.gnn.md "
                f"declares it."
            )
        assert f"num_hidden_states: {spec.parameters['num_hidden_states']}" in blob
        assert f"num_obs: {spec.parameters['num_obs']}" in blob

    def test_a_different_spec_produces_a_different_panel(self) -> None:
        """POSITIVE CONTROL — proves the dims are read, not typed in.

        If the panel restated the numbers as literals, passing a spec with a
        different state space would leave the text unchanged.
        """
        letter = load_spec("L1_letter")
        fig = generate_gnn_pipeline_figure(letter)
        try:
            blob = " | ".join(figure_lines(fig))
        finally:
            plt.close(fig)
        declared = ",".join(str(d) for d in letter.variable_dims("A"))
        assert f"A[{declared}]" in blob, (
            f"passing the letter spec did not change the declared emission dims; "
            f"expected A[{declared}]."
        )
        assert "A[30,39]" not in blob, (
            "the panel still shows the G2P dimensions after being handed a "
            "different spec, so it is not reading the spec at all."
        )

    def test_the_panel_names_the_three_conformance_entry_points(self) -> None:
        """The closing arrow is the figure's point; it must name real functions."""
        fig = generate_gnn_pipeline_figure()
        try:
            blob = " | ".join(figure_lines(fig))
        finally:
            plt.close(fig)
        for symbol in ("describe_shapes", "check_conformance", "assert_conforms"):
            assert symbol in blob, f"{symbol} is not named on fig:gnn_pipeline"

    def test_the_named_entry_points_are_importable(self) -> None:
        """A named function that does not exist would make the figure fiction."""
        import ailm.gnn as gnn_module

        for symbol in ("describe_shapes", "check_conformance", "assert_conforms"):
            assert callable(getattr(gnn_module, symbol)), symbol
