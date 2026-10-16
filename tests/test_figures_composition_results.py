"""Tests for the Stage 3/4/generation/channel figures and the scaling ladder.

``ailm.figures.composition_results`` and ``ailm.figures.scaling_ladder`` were the
only manuscript-figure modules with no direct test file: the generic discovery
sweep in ``tests/test_figure_layout.py`` checks layout for every discovered
figure, but nothing verified that these panels' data functions agree with the
ledger tokens they claim to read, or that their internal cross-checks (the
mis-wiring survivor count, the two-population verdicts, the U-shaped sweep, the
shared-denominator channel gains) actually fire when the ledger is wrong.

Same contract as ``tests/test_figures_results.py``: real token maps, real
figures, no mocks, and a negative control for every cross-check.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ailm.figures import composition_results as comp
from ailm.figures import scaling_ladder
from ailm.figures._common import (
    MissingVariableError,
    assert_within_canvas,
    load_manuscript_variables,
)
from ailm.viz_config import MIN_FONT_PT


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    """The real manuscript token map."""
    return load_manuscript_variables()


def _texts(fig) -> list:
    return [
        artist
        for artist in fig.findobj(plt.Text)
        if isinstance(artist, plt.Text) and artist.get_text().strip()
    ]


# == 1. DATA FUNCTIONS AGREE WITH THE LEDGER ==================================


def test_word_miswirings_rows_come_from_the_ledger(variables) -> None:
    data = comp.word_miswirings_data(variables)
    assert data["bar"] == float(variables["STAGE3_BIGRAM_BASELINE"])
    assert data["rows"][0] == (
        "shipped model",
        float(variables["STAGE3_WORD_PERPLEXITY"]),
        comp.COLOR_MODEL,
    )
    assert data["survivors"] == int(variables["STAGE3_MISWIRINGS_CLEARING_BAR"])
    names = [name for name, _value, _color in data["rows"][1:]]
    assert len(names) == len(set(names))
    assert all(names), "a mis-wiring row lost its display name"


def test_case_both_defects_reads_both_populations(variables) -> None:
    data = comp.case_both_defects_data(variables)
    by_name = {population["name"]: population for population in data["populations"]}
    for name, prefix in (
        ("gold morphology", "STAGE4_GOLD"),
        ("derived oracle", "STAGE4_DERIVED"),
    ):
        panel = by_name[name]
        assert panel["n"] == int(variables[f"{prefix}_TEST_TOKENS"])
        assert panel["majority"] == float(variables[f"{prefix}_MAJORITY"])
        assert panel["baseline"] == float(variables[f"{prefix}_BASELINE"])
        assert panel["model"] == float(variables[f"{prefix}_AGREEMENT"])
    assert data["threshold"] == float(variables["STAGE4_DECLARED_GATE"])


def test_generation_sweep_reads_all_three_operating_points(variables) -> None:
    data = comp.generation_sweep_data(variables)
    assert data["baseline"] == float(variables["GENERATION_BASELINE_TV"])
    gammas = {
        float(variables["GENERATION_LOW_GAMMA"]),
        float(variables["GENERATION_BEST_GAMMA"]),
        float(variables["GENERATION_DEGENERATE_GAMMA"]),
    }
    assert len(gammas) == 3, "two operating points collapsed onto one gamma"
    # The panel's claim: U-shaped — the best point beats both endpoints.
    assert data["best"][1] < data["low"][1]
    assert data["best"][1] < data["high"][1]


def test_channel_comparison_bests_come_from_the_ledger(variables) -> None:
    data = comp.channel_comparison_data(variables)
    assert data["harmful_alpha"] == float(variables["COUPLING_BEST_ALPHA"])
    assert data["harmful_margin"] == float(variables["COUPLING_MARGIN_PTS"])
    assert data["lexical"]["best_alpha"] == float(variables["LEXICAL_BEST_ALPHA"])
    assert data["lexical"]["deltas"] == [
        float(variables[f"LEXICAL_SWEEP_ALPHA_{suffix}_DELTA"])
        for _alpha, suffix in comp._LEXICAL_SWEEP_GRID
    ]


def test_every_sweep_point_carries_an_interval_from_its_own_counts(variables) -> None:
    """A shape argument made of bare point estimates states no precision.

    The interval is re-derived through the recorder's own function from the
    counts already in the ledger, so it needs no new measurement — but it does
    need the right denominator, which is what the point-estimate check below is
    for.
    """
    data = comp.channel_comparison_data(variables)
    lexical = data["lexical"]
    grid = comp._LEXICAL_SWEEP_GRID
    assert len(lexical["lows"]) == len(lexical["highs"]) == len(grid)
    assert data["positions"] == int(variables["LEXICAL_EVAL_POSITIONS"].replace(",", ""))
    for index, (_alpha, suffix) in enumerate(grid):
        b = int(variables[f"LEXICAL_SWEEP_ALPHA_{suffix}_MCNEMAR_B"].replace(",", ""))
        c = int(variables[f"LEXICAL_SWEEP_ALPHA_{suffix}_MCNEMAR_C"].replace(",", ""))
        assert lexical["discordants"][index] == b + c
        assert lexical["lows"][index] <= lexical["deltas"][index] <= lexical["highs"][index]
        assert lexical["p_values"][index] == float(
            variables[f"LEXICAL_SWEEP_ALPHA_{suffix}_MCNEMAR_P"]
        )
    # The two largest weights are the pair the caption calls indistinguishable,
    # and the drawing has to be able to show that rather than only assert it.
    assert lexical["lows"][-1] < lexical["highs"][-2]


def test_a_wrong_sweep_denominator_refuses_to_draw(variables) -> None:
    """Positive control: the counts alone cannot reveal a wrong denominator."""
    broken = dict(variables)
    broken["LEXICAL_EVAL_POSITIONS"] = str(
        int(variables["LEXICAL_EVAL_POSITIONS"].replace(",", "")) // 2
    )
    with pytest.raises(ValueError, match="the counts do not reproduce"):
        comp.channel_comparison_data(broken)


def test_sweep_counts_over_the_population_refuse_to_draw(variables) -> None:
    broken = dict(variables)
    broken["LEXICAL_EVAL_POSITIONS"] = "10"
    with pytest.raises(ValueError, match="came from different runs"):
        comp.channel_comparison_data(broken)


# == 2. NEGATIVE CONTROLS FOR EVERY CROSS-CHECK ===============================


def test_miswirings_rejects_a_survivor_count_that_no_longer_holds(variables) -> None:
    """NEGATIVE CONTROL — drift the recorded survivor count; the figure must refuse."""
    broken = dict(variables)
    broken["STAGE3_MISWIRINGS_CLEARING_BAR"] = "0"
    with pytest.raises(ValueError, match="bar-clearing mis-wirings"):
        comp.word_miswirings_data(broken)


def test_miswirings_rejects_a_missing_bar_token(variables) -> None:
    broken = dict(variables)
    broken.pop("STAGE3_BIGRAM_BASELINE")
    with pytest.raises(MissingVariableError):
        comp.word_miswirings_data(broken)


def test_case_both_defects_rejects_a_threshold_that_no_longer_straddles(
    variables,
) -> None:
    """NEGATIVE CONTROL — the straddle cross-check must fire.

    The panel's claim is that the declared gate sits ABOVE the derived model and
    BELOW the gold majority. Perturb the derived model upward until the straddle
    breaks; the figure must refuse rather than redraw the two-defect finding.
    """
    broken = dict(variables)
    broken["STAGE4_DERIVED_AGREEMENT"] = str(float(variables["STAGE4_DECLARED_GATE"]) + 1.0)
    with pytest.raises(ValueError, match="no longer sits above"):
        comp.case_both_defects_data(broken)


def test_generation_sweep_rejects_a_non_u_shape(variables) -> None:
    """NEGATIVE CONTROL — a best point worse than an endpoint must fail the figure."""
    broken = dict(variables)
    broken["GENERATION_BEST_TV"] = "0.99"
    with pytest.raises(ValueError, match="no longer interior"):
        comp.generation_sweep_data(broken)


def test_channel_comparison_rejects_a_flipped_gain_sign(variables) -> None:
    """NEGATIVE CONTROL — the harmful/helpful sign convention is the panel's claim."""
    broken = dict(variables)
    broken["COUPLING_MARGIN_PTS"] = "5.0"
    with pytest.raises(ValueError, match="no longer have opposite signs"):
        comp.channel_comparison_data(broken)


def test_channel_comparison_rejects_a_missing_sweep_rung(variables) -> None:
    broken = dict(variables)
    for name in list(broken):
        if name.startswith("LEXICAL_SWEEP_ALPHA_") and name.endswith("_DELTA"):
            del broken[name]
            break
    with pytest.raises(MissingVariableError):
        comp.channel_comparison_data(broken)


def test_scaling_ladder_rejects_a_single_rung(variables) -> None:
    """NEGATIVE CONTROL — one point is not a ladder (the function's own contract)."""
    broken = {
        name: value
        for name, value in variables.items()
        if not (name.startswith("STAGE3_SCALE_") and name.endswith("_COVERAGE_PCT"))
        or name == "STAGE3_SCALE_2_COVERAGE_PCT"
    }
    with pytest.raises(ValueError, match="ladder needs at least two"):
        scaling_ladder.scaling_ladder_data(broken)


def test_scaling_ladder_values_match_the_ledger(variables) -> None:
    data = scaling_ladder.scaling_ladder_data(variables)
    sizes = sorted(
        int(name.split("_")[2])
        for name in variables
        if name.startswith("STAGE3_SCALE_") and name.endswith("_COVERAGE_PCT")
    )
    assert data["sizes"] == [float(size) for size in sizes]
    assert data["coverage"] == [
        float(variables[f"STAGE3_SCALE_{size}_COVERAGE_PCT"]) for size in sizes
    ]
    assert data["cap"] == [float(variables["STAGE3_MAX_AGENT_VOCAB"].replace(",", ""))]


def test_scaling_ladder_rejects_a_missing_rung_token(variables) -> None:
    broken = dict(variables)
    rung = next(
        name
        for name in broken
        if name.startswith("STAGE3_SCALE_") and name.endswith("_WORD_PERPLEXITY")
    )
    del broken[rung]
    with pytest.raises(MissingVariableError):
        scaling_ladder.scaling_ladder_data(broken)


# == 3. THE FIGURES BUILD LEGIBLY AND INSIDE THE CANVAS =======================


@pytest.mark.parametrize(
    "build",
    [
        comp.generate_word_miswirings,
        comp.generate_case_both_defects,
        comp.generate_generation_sweep,
        comp.generate_channel_comparison,
        scaling_ladder.generate_scaling_ladder,
    ],
    ids=[
        "word_miswirings",
        "case_both_defects",
        "generation_sweep",
        "channel_comparison",
        "scaling_ladder",
    ],
)
def test_figure_builds_legibly_inside_the_canvas(build, variables) -> None:
    fig = build(variables)
    try:
        assert_within_canvas(fig)
        offenders = [
            (artist.get_text()[:40], artist.get_fontsize())
            for artist in _texts(fig)
            if artist.get_fontsize() < MIN_FONT_PT
        ]
        assert not offenders, f"{build.__name__}: text below MIN_FONT_PT: {offenders[:5]}"
    finally:
        plt.close(fig)


# == 4. GATE-AUDIT INVENTORY CLOSURE (scripts/audit_gates.py) =================
#
# scripts/audit_gates.py hard-codes its audit list in main() and guards it with
# an equality check against AUDITED_GATE_IDS — but nothing pinned that the
# script's main() calls one audit per registered gate, nor that the script's
# UNAUDITED_STAGES disclosure stays in sync with the registry.


def test_audit_gates_main_audits_every_registered_gate_exactly_once() -> None:
    """Import scripts/audit_gates.py and inspect main's audit-call inventory."""
    import ast
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "ailm_audit_gates_ast", root / "scripts" / "audit_gates.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The main() body must call one audit_* function per registered gate.
    source = (root / "scripts" / "audit_gates.py").read_text()
    tree = ast.parse(source)
    main_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    audit_calls = [
        node.func.id
        for node in ast.walk(main_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("audit_")
    ]
    # One stage has two sub-audits (stage2a/2b), so >= len(AUDITED_GATE_IDS)
    # calls, each named in the module, each called exactly once.
    assert len(audit_calls) == len(set(audit_calls))
    for name in audit_calls:
        assert hasattr(module, name), f"main() calls {name} which the module does not define"
    assert len(audit_calls) >= len(module.AUDITED_GATE_IDS)

    # The UNAUDITED_STAGES disclosure must exist and reference a real doc path.
    for stage, reason in module.UNAUDITED_STAGES:
        assert stage and reason
        assert "docs/" in reason, f"unaudited stage {stage!r} lacks a doc pointer"


def test_unaudited_disclosure_names_no_registered_gate() -> None:
    """A stage both registered and 'unaudited' would be a contradiction."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "ailm_audit_gates_ast2", root / "scripts" / "audit_gates.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registered = {gate.value for gate in module.AUDITED_GATE_IDS}
    for stage, _reason in module.UNAUDITED_STAGES:
        assert not any(gate in stage for gate in registered), (
            f"UNAUDITED_STAGES names registered gate in {stage!r}"
        )
