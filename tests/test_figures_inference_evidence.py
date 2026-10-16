"""Evidence controls for the inference-evidence figure.

The figure claims, per upper level, exactly how its recorded score was
produced: live pymdp inference, a numpy path checked against pymdp at every
held-out position, a closed form whose live posterior agrees with the numpy
filter on the full held-out set, or a conditional table that is not inference
at all. These tests bind each claim to recorded tokens and prove every guard
can fire:

- positive: the current ledger draws, and the drawn text names the right
  evidence kind per level;
- negative: a drifted token that would falsify a caption claim is refused;
- determinism: two regenerations are byte-identical;
- layout: legibility floor and canvas containment on the produced figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Text

from ailm.figures._common import MissingVariableError
from ailm.figures.inference_evidence import (
    build_inference_evidence_figure,
    generate_inference_evidence,
    inference_evidence_data,
)
from ailm.manuscript_variables import generate_manuscript_variables

#: The module fixture runs the full ledger-to-token path (git-history epochs,
#: claim scans) whose wall time balloons under concurrent fleets; the file
#: carries its own budget instead of the repository default.
pytestmark = pytest.mark.timeout(900)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """Exercise the current ledger-to-token path directly, without dirtying git.

    ``generate_manuscript_variables`` stamps a fresh ``META_BUILD_TIMESTAMP``,
    so calling it on the real project root rewrites the tracked
    ``output/data/manuscript_variables.json`` and invalidates the provenance
    digest for every later test in the same session. Building into a temp root
    keeps that live-path coverage while the tracked map stays byte-stable; the
    tokens that figure code reads are ledger-derived and identical in both.
    """
    staged = tmp_project_root()
    return generate_manuscript_variables(staged)


def tmp_project_root() -> Path:
    """A project root whose tracked artifacts live in a temp directory."""
    import shutil

    staged = PROJECT_ROOT / "scratchpad" / "_p2_vartest_root"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for rel in ("data", "docs", "vendor"):
        source = PROJECT_ROOT / rel
        if source.exists():
            (staged / rel).mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, staged / rel / item.name)
                else:
                    shutil.copytree(item, staged / rel / item.name)
    return staged


def _with(variables: dict[str, str], token: str, value: str) -> dict[str, str]:
    changed = dict(variables)
    changed[token] = value
    return changed


def _figure_text(figure) -> str:
    return "\n".join(text.get_text() for text in figure.findobj(match=Text))


def test_data_reads_every_claim_from_tokens(variables) -> None:
    data = inference_evidence_data(variables)
    # Values agree with the tracked generated map, within its rounding.
    assert data["l4_agreement_pct"] == pytest.approx(50.72, abs=1e-3)
    assert data["l6_live_filter_improvement_nats"] == pytest.approx(0.0043, abs=1e-4)
    # The recorded L5 equivalence evidence (decision 92).
    assert data["l5_pymdp_type_argmax_disagreements"] == 0.0
    assert data["l5_pymdp_prefix_sentences"] > 0
    # The two L6 deltas are distinct recorded estimands.
    assert (
        abs(data["l6_registered_improvement_nats"] - data["l6_live_filter_improvement_nats"]) > 1e-6
    )


def test_figure_names_the_correct_evidence_kind_per_level(variables) -> None:
    figure = build_inference_evidence_figure(variables)
    try:
        text = _figure_text(figure)
        assert "agrees at every held-out position" in text, "L4 evidence kind missing"
        assert "argmax disagreements" in text, "L5 recorded equivalence evidence missing"
        assert "max joint-posterior deviation" in text, "L5 posterior tolerance missing"
        assert "posterior agreement, not score equivalence" in text, "L5 caveat missing"
        assert "the registered one is not inference" in text, "L6 evidence kind missing"
        assert "pymdp filter" in text, "the live pymdp L6 score must be named"
        # Honesty wording: nothing implies the headline L5/L6 scores run live
        # pymdp inference.
        assert "closed-form mixture" in text
    finally:
        plt.close(figure)


def test_missing_token_raises_missing_variable_error(variables) -> None:
    """A missing token fails loudly as MissingVariableError, never a default."""
    drifted = _with(variables, "STAGE5_TYPEFREE_PERPLEXITY", "")
    del drifted["STAGE5_TYPEFREE_PERPLEXITY"]
    with pytest.raises(MissingVariableError):
        inference_evidence_data(drifted)


def test_drifted_live_filter_delta_is_refused(variables) -> None:
    """NEGATIVE CONTROL: a live-filter delta inconsistent with its score."""
    with pytest.raises(ValueError, match="recorded improvement disagree"):
        inference_evidence_data(_with(variables, "STAGE6_LIVE_FILTER_IMPROVEMENT_NATS", "0.5"))


def test_drifted_registered_delta_collapsing_onto_live_is_refused(variables) -> None:
    """NEGATIVE CONTROL: the two L6 estimands must stay distinct."""
    with pytest.raises(ValueError, match="separate-estimand annotation"):
        inference_evidence_data(_with(variables, "STAGE6_LOGLOSS_IMPROVEMENT_NATS", "0.0043"))


def test_drifted_stage4_agreement_outside_majority_is_refused(variables) -> None:
    """NEGATIVE CONTROL: the headline agreement claim has a supported range."""
    with pytest.raises(ValueError, match="outside \\[50, 100\\]"):
        inference_evidence_data(_with(variables, "STAGE4_DERIVED_AGREEMENT", "40.0"))


def test_drifted_l5_disagreement_count_is_refused(variables) -> None:
    """NEGATIVE CONTROL: a non-zero recorded L5 disagreement breaks equivalence."""
    with pytest.raises(ValueError, match="type-argmax disagreement count is non-zero"):
        inference_evidence_data(_with(variables, "STAGE5_PYMDP_TYPE_ARGMAX_DISAGREEMENTS", "1"))


def test_drifted_l4_disagreement_count_is_refused(variables) -> None:
    """NEGATIVE CONTROL: a non-zero recorded L4 disagreement breaks the annotation."""
    with pytest.raises(ValueError, match="L4 pymdp argmax disagreement count is non-zero"):
        inference_evidence_data(_with(variables, "STAGE4_PYMDP_ARGMAX_DISAGREEMENTS", "1"))


def test_l4_comparison_that_stops_short_of_the_population_is_refused(variables) -> None:
    """NEGATIVE CONTROL: 'every held-out position' must cover the scored tokens."""
    walked = str(int(float(variables["STAGE4_DERIVED_TEST_TOKENS"])) - 1)
    with pytest.raises(ValueError, match="does not cover the scored population"):
        inference_evidence_data(_with(variables, "STAGE4_PYMDP_POSITIONS", walked))


def test_current_ledger_does_not_trigger_refusal(variables) -> None:
    """Negative control on the guards: the true ledger must draw."""
    assert inference_evidence_data(variables)["l4_test_tokens"] > 0


def test_regeneration_is_byte_identical(variables, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    path_a = Path(generate_inference_evidence(output_dir=first, variables=variables))
    path_b = Path(generate_inference_evidence(output_dir=second, variables=variables))
    assert path_a.name == "inference_evidence.png"
    assert path_a.read_bytes() == path_b.read_bytes(), (
        "two regenerations produced different bytes; something non-deterministic entered the figure"
    )


def test_no_text_below_the_legibility_floor(variables) -> None:
    figure = build_inference_evidence_figure(variables)
    try:
        offenders = [
            (artist.get_text()[:40], artist.get_fontsize())
            for artist in figure.findobj(match=Text)
            if artist.get_text().strip() and artist.get_fontsize() < 9.0
        ]
        assert not offenders, f"text below the 9pt floor: {offenders[:5]}"
    finally:
        plt.close(figure)


def test_no_text_escapes_the_canvas(variables) -> None:
    from ailm.figures._common import assert_within_canvas

    figure = build_inference_evidence_figure(variables)
    try:
        assert_within_canvas(figure)
    finally:
        plt.close(figure)
