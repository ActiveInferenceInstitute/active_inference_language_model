"""Parameter accounting: the three counts, and the orientation guard.

A parameter count is the number a reader will quote, and it is unusually easy
to get wrong in a way that still looks right. Three distinct wrong answers are
available for this model:

* the dense entry count, which reports Stage 3 as a 4.8-million-parameter level
  when its word transition is a 2,000-entry identity padded out to
  ``(2000, 2000)``;
* the non-zero count, which still charges a one-hot emission for every entry it
  has, though a lookup table has no freedom;
* any of the above taken off a mis-oriented array, which produces a plausible
  number from summing the wrong axis.

Each test below pairs the property with a control that must not fire.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ailm.parameters import (
    GNN_AUXILIARY_SPECS,
    GNN_SPEC_FILES,
    NORMALIZED_AXIS,
    ArrayParameters,
    check_level_against_gnn,
    check_model_against_gnn,
    count_array,
    count_case_only_clause,
    count_level,
    count_model,
    gnn_declared_sizes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# Full-hierarchy integration tests below allow bounded cold JAX startup on
# the external drive (a measured setup exceeded 300 seconds under fleet load).
# Array-level controls retain the default test timeout; no timeout becomes a skip.
@pytest.fixture(scope="module")
def model():
    """The whole hierarchy, built through the shipped level builders."""
    return count_model()


# ── The counting rule itself ─────────────────────────────────────────────────


def test_a_deterministic_column_has_no_free_parameters():
    """POSITIVE: a one-hot array is a lookup table, not a fit.

    NEGATIVE CONTROL: a uniform array over the same shape must report the full
    ``n - 1`` per distribution, or the rule is just returning zero.
    """
    onehot = np.zeros((1, 4, 3))
    onehot[0, 0, :] = 1.0
    deterministic = count_array("A[0]", onehot, "A")
    assert deterministic.free == 0
    assert deterministic.nonzero == 3
    assert deterministic.stored == 12
    assert deterministic.deterministic == 3
    assert deterministic.distributions == 3

    uniform = count_array("A[0]", np.full((1, 4, 3), 0.25), "A")
    assert uniform.free == 3 * (4 - 1)
    assert uniform.deterministic == 0


@pytest.mark.parametrize("kind", tuple(NORMALIZED_AXIS))
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((-0.1, 1.1), "nonnegative"),
        ((-0.2, -0.8), "nonnegative"),
        ((float("nan"), 0.5), "finite"),
        ((float("inf"), 0.5), "finite"),
        ((float("-inf"), 0.5), "finite"),
    ],
)
def test_invalid_probability_mass_cannot_be_counted_as_parameters(kind, values, message):
    # Negative mass can sum to one; a negative or NaN sum can also evade the
    # occupied-column test. None of those arrays is a probability distribution.
    array = np.array(values).reshape(1, 2, 1)
    with pytest.raises(ValueError, match=message):
        count_array("invalid", array, kind)


def test_free_parameters_ignore_structural_zeros():
    """A column with two live entries carries one free parameter, not three.

    This is the whole reason ``nonzero`` and ``free`` are separate columns: the
    simplex constraint removes one degree of freedom per distribution, and the
    structural zeros were never degrees of freedom to begin with.
    """
    sparse = np.zeros((1, 4, 2))
    sparse[0, 0, 0] = 0.6
    sparse[0, 1, 0] = 0.4
    sparse[0, 2, 1] = 1.0
    counted = count_array("B[0]", sparse, "B")
    assert counted.stored == 8
    assert counted.nonzero == 3
    assert counted.free == 1  # (2 - 1) from the first column, (1 - 1) from the second


def test_a_misoriented_array_is_rejected():
    """NEGATIVE CONTROL for the orientation trap.

    A row-stochastic array handed in as column-stochastic sums to something
    other than one along the declared axis. Without this check the count would
    come back looking entirely reasonable.
    """
    row_stochastic = np.zeros((1, 3, 3))
    row_stochastic[0, 0, :] = [0.5, 0.3, 0.2]
    row_stochastic[0, 1, :] = [0.1, 0.8, 0.1]
    row_stochastic[0, 2, :] = [0.2, 0.2, 0.6]
    with pytest.raises(ValueError, match="does not sum to 1"):
        count_array("B[0]", row_stochastic, "B")

    # POSITIVE: its transpose IS column-stochastic and must be accepted, so the
    # guard is rejecting the orientation rather than the array.
    column_stochastic = row_stochastic.transpose(0, 2, 1)
    assert count_array("B[0]", column_stochastic, "B").distributions == 3


def test_unknown_kinds_and_flat_arrays_are_rejected():
    with pytest.raises(ValueError, match="unknown array kind"):
        count_array("C[0]", np.ones((1, 3)), "C")
    with pytest.raises(ValueError, match="no axis"):
        count_array("D[0]", np.ones(3), "D")


def test_all_zero_slices_are_tolerated_not_counted():
    """An unreachable slice is padding, not a mis-orientation."""
    padded = np.zeros((1, 3, 2))
    padded[0, :, 0] = [0.2, 0.3, 0.5]
    counted = count_array("B[0]", padded, "B")
    assert counted.distributions == 1
    assert counted.free == 2


# ── The model as built ───────────────────────────────────────────────────────


@pytest.mark.timeout(600)
def test_every_level_is_counted(model):
    """All seven level models appear, and each normalizes on its declared axis.

    Reaching this assertion at all means every array passed the stochasticity
    check inside ``count_array`` — the orientation of the shipped model is
    verified by the count succeeding, not by a separate assertion.
    """
    names = [level.level for level in model.levels]
    assert names == [
        "L1_letter",
        "L2A_phoneme",
        "L2B_g2p",
        "L3_word",
        "L4_case",
        "L5_clause",
        "L6_narrative",
    ]
    assert set(NORMALIZED_AXIS) == {"A", "B", "D"}


@pytest.mark.timeout(600)
def test_totals_are_the_sum_of_the_levels(model):
    """The headline number must be the parts, not a separately maintained one."""
    assert model.stored == sum(level.stored for level in model.levels)
    assert model.nonzero == sum(level.nonzero for level in model.levels)
    assert model.free == sum(level.free for level in model.levels)
    for level in model.levels:
        assert level.stored == sum(a.stored for a in level.arrays)
        assert level.free == sum(a.free for a in level.arrays)


@pytest.mark.timeout(600)
def test_the_three_counts_are_ordered(model):
    """free <= nonzero <= stored, at every level and overall.

    Any inversion means one of the three is computed against a different array
    than the others.
    """
    assert model.free <= model.nonzero <= model.stored
    for level in model.levels:
        assert level.free <= level.nonzero <= level.stored, level.level


@pytest.mark.timeout(600)
def test_the_word_level_is_overwhelmingly_structure(model):
    """Stage 3's dense volume is the claim's whole point.

    POSITIVE: its stored count dwarfs its free count. NEGATIVE CONTROL: a level
    with no sparse structure (the G2P emission) must NOT show the same gap, or
    the test is measuring an artifact of the counting rather than of Stage 3.
    """
    word = model.by_level()["L3_word"]
    assert word.stored > 1_000_000
    assert word.free < 3_000
    assert word.nonzero / word.stored < 0.01

    g2p = model.by_level()["L2B_g2p"]
    assert g2p.nonzero == g2p.stored, "the G2P level is dense; the control is not a control"


@pytest.mark.timeout(600)
def test_preferences_are_excluded_from_the_parameter_counts(model):
    """C is a preference, not a fitted parameter.

    It is measured so the exclusion is visible rather than silent, and it must
    not be inside any of the three totals.
    """
    assert model.preference_entries > 0
    array_names = {a.name[0] for level in model.levels for a in level.arrays}
    assert array_names <= {"A", "B", "D"}, f"a preference array was counted: {array_names}"


@pytest.mark.timeout(600)
def test_the_counts_are_stable_across_two_builds():
    """Deterministic: nothing in the count depends on run order or sampling."""
    first, second = count_model(), count_model()
    assert (first.stored, first.nonzero, first.free) == (
        second.stored,
        second.nonzero,
        second.free,
    )


def test_array_parameters_density_is_a_ratio():
    empty = ArrayParameters("X", (0,), 0, 0, 0, 0, 0)
    assert empty.density == 0.0
    assert count_array("D[0]", np.full((1, 5), 0.2), "D").density == 1.0


def test_count_level_accepts_an_agent_without_preferences():
    """An agent with no C must count cleanly rather than raise."""

    class _Bare:
        A = [np.full((1, 3, 2), 1 / 3)]
        B = [np.full((1, 2, 2, 1), 0.5)]
        D = [np.full((1, 2), 0.5)]

    level = count_level("bare", _Bare())
    assert level.preference_entries == 0
    assert level.free == 3 * 2 - 2 + (2 - 1) * 2 + 1


# ── THE GNN BINDING ──────────────────────────────────────────────────────────


@pytest.mark.timeout(600)
def test_every_level_names_the_gnn_file_that_declares_it(model):
    """The registry must cover every built level, and every shipped spec.

    "Which file declares this level" needs exactly one answer, and it needs to
    be machine-readable rather than prose: the manuscript points readers at
    `gnn/models/`, and a level whose spec is missing or misnamed would send
    them to a file describing a different model.
    """
    for level in model.levels:
        assert level.level in GNN_SPEC_FILES, f"{level.level} declares no GNN spec"
        spec = REPO_ROOT / GNN_SPEC_FILES[level.level]
        assert spec.exists(), f"{level.level} points at a missing spec: {spec}"
        assert spec.suffixes[-2:] == [".gnn", ".md"], f"{spec} is not a .gnn.md file"

    # Bijection with what is on disk: an orphaned spec is a model the paper
    # describes and nothing builds.
    on_disk = {
        str(path.relative_to(REPO_ROOT)) for path in (REPO_ROOT / "gnn" / "models").glob("*.gnn.md")
    }
    registered = set(GNN_SPEC_FILES.values()) | set(GNN_AUXILIARY_SPECS.values())
    assert on_disk == registered, (
        f"registry and gnn/models/ disagree; only on disk: {sorted(on_disk - registered)}, "
        f"only registered: {sorted(registered - on_disk)}"
    )


@pytest.mark.timeout(600)
def test_the_built_model_conforms_to_its_gnn_declarations(model):
    """Every declared array size must equal the built one, exactly.

    The specs and the builders are written by hand, independently — the spec
    does not generate the builder. Nothing at runtime notices them disagreeing:
    the agent runs, the gate reports a number, and the file the manuscript
    points at as the level's definition describes a different model than the
    one measured.
    """
    problems = check_model_against_gnn(model)
    assert problems == [], "GNN declarations disagree with the built model:\n" + "\n".join(problems)


@pytest.mark.timeout(600)
def test_canonical_clause_census_uses_widening_and_retains_the_case_only_census(model):
    """The old arm must fail canonical conformance, not silently fill its row."""
    from ailm.claims import load_ledger
    from ailm.levels.clause import N_TYPES
    from ailm.levels.widened import N_WIDENED_SYMBOLS

    canonical = model.by_level()["L5_clause"]
    ablation = count_case_only_clause()
    canonical_spec = REPO_ROOT / GNN_SPEC_FILES[canonical.level]
    ablation_spec = REPO_ROOT / GNN_AUXILIARY_SPECS[ablation.level]
    assert check_level_against_gnn(canonical, canonical_spec) == []
    assert check_level_against_gnn(ablation, ablation_spec) == []
    assert check_level_against_gnn(ablation, canonical_spec)
    assert check_level_against_gnn(canonical, ablation_spec)
    # Full within-type transition supports and a full joint prior; the read-off
    # emission is deterministic. This closed form is independent of count_array.
    symbols = N_WIDENED_SYMBOLS
    assert canonical.free == N_TYPES * symbols * (symbols - 1) + N_TYPES * symbols - 1
    assert canonical.stored > ablation.stored
    assert canonical.free > ablation.free
    ledger = load_ledger(REPO_ROOT)
    for attribute in ("stored", "nonzero", "free"):
        entry = ledger.get(f"params_l5_case_only_{attribute}")
        assert entry is not None
        assert entry["value"] == getattr(ablation, attribute)


@pytest.mark.timeout(600)
def test_the_conformance_check_fires_on_a_planted_disagreement(model, tmp_path):
    """NEGATIVE CONTROL: a spec that declares the wrong size must be caught.

    Without this, ``check_level_against_gnn`` returning ``[]`` proves nothing —
    a function that always returns an empty list would pass the test above.
    """
    level = model.by_level()["L4_case"]
    original = (REPO_ROOT / GNN_SPEC_FILES["L4_case"]).read_text(encoding="utf-8")

    # Same file, one dimension changed: the case transition becomes 9x9.
    planted = tmp_path / "L4_case_planted.gnn.md"
    assert "B[8,8,1,type=float]" in original
    planted.write_text(
        original.replace("B[8,8,1,type=float]", "B[9,9,1,type=float]"), encoding="utf-8"
    )

    problems = check_level_against_gnn(level, planted)
    assert problems, "a mis-declared transition size was not detected"
    assert any("B[0]" in p and "81" in p for p in problems), problems

    # POSITIVE: the unmodified spec still conforms, so the detector is
    # responding to the plant rather than rejecting everything.
    assert check_level_against_gnn(level, REPO_ROOT / GNN_SPEC_FILES["L4_case"]) == []


@pytest.mark.timeout(600)
def test_multi_factor_arrays_map_onto_their_gnn_names(model):
    """`B_f0`/`B_f1` in a spec are `B[0]`/`B[1]` in the built agent.

    Level 3 is the only multi-factor level, and a naive mapping that only
    understood the single-factor `B` name would silently skip its second factor
    — comparing nothing and reporting conformance.
    """
    declared = gnn_declared_sizes(REPO_ROOT / GNN_SPEC_FILES["L3_word"])
    assert {"A[0]", "B[0]", "B[1]", "D[0]", "D[1]"} <= set(declared)
    assert declared["B[0]"] == 2000 * 2000 * 1
    # 20 letter slots + 1 END slot = 21 positions (the vendored word artifact's
    # provenance pins max_word_len=20; the spec declares 21 states per factor).
    assert declared["B[1]"] == 21 * 21 * 1

    built = {a.name: a.stored for a in model.by_level()["L3_word"].arrays}
    assert built["B[1]"] == declared["B[1]"], "the second factor was not compared"


def test_non_generative_gnn_variables_are_not_compared():
    """Hidden-state and observation variables are not parameters.

    A spec declares `s_t`, `o_t` and `C` alongside `A`/`B`/`D`. Treating those
    as arrays to compare would report a discrepancy for every level.
    """
    declared = gnn_declared_sizes(REPO_ROOT / GNN_SPEC_FILES["L4_case"])
    assert set(declared) == {"A[0]", "B[0]", "D[0]"}
    assert not any(key.startswith(("s_", "o_", "C")) for key in declared)


def test_every_level_section_states_its_gnn_file():
    """Each level subsection must name the spec that declares it, and name it correctly.

    Prose and registry are two hand-maintained lists of the same fact. A
    manuscript path that drifts from the registry sends a reader to a file
    describing a different model, and nothing else in the build would notice:
    the path is prose, not a token, so hydration cannot check it.

    Every level is constructed in the Methods section, so all six declarations
    live in one file; the count assertion is what keeps a lost declaration from
    passing because a sibling level still carries the phrase.
    """
    methods = (REPO_ROOT / "docs" / "manuscript" / "03_methods.md").read_text(encoding="utf-8")
    declared = (
        "L1_letter",
        "L2A_phoneme",
        "L2B_g2p",
        "L3_word",
        "L4_case",
        "L5_clause",
        "L6_narrative",
    )
    for key in declared:
        spec_path = GNN_SPEC_FILES[key]
        assert spec_path in methods, f"03_methods.md does not name {spec_path}"

    # One bolded declaration sentence per level subsection, Level 2 naming both
    # of its models in the single sentence its subsection carries. The plain
    # phrase also appears in the tbl:parameters caption, so the bolded form is
    # what the count is taken over.
    assert methods.count("is declared as a GNN file.**") == len(declared) - 1, (
        "the Methods section no longer carries one GNN declaration per level; a "
        "level's spec pointer was dropped or duplicated"
    )


def test_every_gnn_path_in_the_manuscript_exists():
    """A path the paper prints must resolve, whoever wrote it.

    Written over the general class — every `gnn/models/...` string anywhere in
    the manuscript — rather than over the paths this change added, so a stale
    reference introduced later is caught by the same check.
    """
    referenced: set[str] = set()
    for path in sorted((REPO_ROOT / "docs" / "manuscript").glob("*.md")):
        referenced |= set(
            re.findall(r"gnn/models/[A-Za-z0-9_]+\.gnn\.md", path.read_text(encoding="utf-8"))
        )
    assert referenced, "the manuscript names no GNN model files at all"
    missing = sorted(ref for ref in referenced if not (REPO_ROOT / ref).exists())
    assert not missing, f"the manuscript points at GNN files that do not exist: {missing}"

    known = set(GNN_SPEC_FILES.values()) | set(GNN_AUXILIARY_SPECS.values())
    unregistered = sorted(referenced - known)
    assert not unregistered, f"the manuscript names unregistered specs: {unregistered}"


def test_every_gnn_path_in_the_docs_exists():
    """A GNN path the docs print must resolve, in docs/ as well as docs/manuscript/.

    The manuscript already had this guard; `docs/` did not, and two references
    to `gnn/ailm_word_level.md` survived the reorganization that moved the specs
    under `gnn/models/` with level-prefixed names. Nothing noticed: a path in
    prose is not a token, so hydration cannot check it, and the link checker
    only follows `.md` links written as markdown links rather than inline code.

    Stated over the general class — every `gnn/...` path anywhere under `docs/`
    or in the root reference files — so the next rename is caught the same way.
    """
    roots = sorted((REPO_ROOT / "docs").rglob("*.md"))
    roots += [REPO_ROOT / name for name in ("README.md", "AGENTS.md", "CHANGELOG.md")]
    roots += sorted((REPO_ROOT / "gnn").glob("*.md"))

    referenced: dict[str, str] = {}
    for path in roots:
        if not path.exists():
            continue
        for match in re.finditer(
            r"gnn/[A-Za-z0-9_/.-]*\.gnn\.md", path.read_text(encoding="utf-8")
        ):
            referenced[match.group(0)] = str(path.relative_to(REPO_ROOT))

    assert referenced, "no GNN spec paths are referenced in the docs at all"
    missing = sorted(
        f"{where}: {ref}" for ref, where in referenced.items() if not (REPO_ROOT / ref).exists()
    )
    assert not missing, "docs point at GNN files that do not exist:\n" + "\n".join(missing)

    known = set(GNN_SPEC_FILES.values()) | set(GNN_AUXILIARY_SPECS.values())
    unregistered = sorted(ref for ref in referenced if ref not in known)
    assert not unregistered, f"docs name unregistered specs: {unregistered}"
