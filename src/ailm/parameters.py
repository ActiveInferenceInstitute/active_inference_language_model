"""Parameter accounting for every level of the hierarchy.

Three numbers per array, because one number would be misleading in three
different directions:

**stored** — entries in the dense array as pymdp holds it. This is what memory
costs (``ailm.complexity`` prices it in bytes) and it is the number a reader
expects when they ask "how big is this model". Taken alone it badly overstates
the model: Stage 3's word transition is a ``(2000, 2000)`` array with 2,000
non-zero entries, because a word does not change into another word while its
letters are emitted.

**nonzero** — entries the model can actually place mass on. Everything else is
structural zero, fixed by the trie and the lexicon rather than estimated from
anything. This is the honest size of the object.

**free** — simplex degrees of freedom on the array's live support. Every distribution in a normalized
array loses one degree of freedom to the simplex constraint, so a column with
$k$ non-zero entries carries $k - 1$ free parameters, and a deterministic column
($k = 1$) carries none. A one-hot emission is a lookup table written as a
matrix: it has entries, it has no freedom. This counts array capacity, not the
parameters actually estimated by the shipped fitting procedure. In particular,
the builder-fixed uniform Level 1 start prior contributes simplex capacity even
though the corpus does not fit it. Constraints shared across different arrays
are not inferred by this per-array rule.

The gap between the three is the point. Most of this model's dense volume is
structure. None of these columns alone is a directly comparable count of
trainable neural-model parameters or a measure of corpus information learned.

``C`` is excluded from all three and reported separately. It is a preference
vector, not a probability: it does not normalize, nothing estimates it from a
corpus, and folding a preference into a parameter count would make the model
look larger by an amount that carries no information about the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: Repository root, for resolving the GNN spec paths in the registry below.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

__all__ = [
    "GNN_AUXILIARY_SPECS",
    "GNN_SPEC_FILES",
    "NORMALIZED_AXIS",
    "ArrayParameters",
    "LevelParameters",
    "ModelParameters",
    "check_level_against_gnn",
    "check_model_against_gnn",
    "count_array",
    "count_case_only_clause",
    "count_level",
    "count_model",
    "gnn_declared_sizes",
]

#: The axis each pymdp array normalizes over. ``A`` is column-stochastic over
#: observations, ``B`` over the NEXT state, ``D`` over the state it priors.
#: Every one is asserted at count time rather than trusted: the orientation trap
#: (summing the wrong axis and getting a plausible number) is the single most
#: repeated defect in this codebase's history, and a parameter count read off a
#: mis-oriented array is wrong without being obviously wrong.
NORMALIZED_AXIS: dict[str, int] = {"A": 1, "B": 1, "D": 1}

#: Tolerance for the stochasticity assertion. Counts are built from float32
#: pymdp arrays, so exact equality to 1.0 is not available.
_SIMPLEX_TOLERANCE = 2e-3


@dataclass(frozen=True)
class ArrayParameters:
    """Parameter accounting for one pymdp array.

    Attributes:
        name: Array label, e.g. ``"B[0]"``.
        shape: Dense shape as stored.
        stored: Dense entry count.
        nonzero: Entries that are not a structural zero.
        free: Simplex degrees of freedom on the live support, whether fitted or fixed.
        distributions: Number of normalized distributions in the array.
        deterministic: Distributions with exactly one non-zero entry, which
            contribute no freedom at all.
    """

    name: str
    shape: tuple[int, ...]
    stored: int
    nonzero: int
    free: int
    distributions: int
    deterministic: int

    @property
    def density(self) -> float:
        """Fraction of stored entries that are non-zero."""
        return self.nonzero / self.stored if self.stored else 0.0


@dataclass(frozen=True)
class LevelParameters:
    """Parameter accounting for one level's generative model."""

    level: str
    arrays: tuple[ArrayParameters, ...]
    preference_entries: int

    @property
    def stored(self) -> int:
        """Total parameter entries stored across this level A/B/D arrays."""
        return sum(a.stored for a in self.arrays)

    @property
    def nonzero(self) -> int:
        """Total non-zero entries across this level A/B/D arrays."""
        return sum(a.nonzero for a in self.arrays)

    @property
    def free(self) -> int:
        """Total free (non-degenerate) parameters across this level arrays."""
        return sum(a.free for a in self.arrays)


@dataclass(frozen=True)
class ModelParameters:
    """The whole hierarchy, level by level."""

    levels: tuple[LevelParameters, ...]

    @property
    def stored(self) -> int:
        """Total stored parameter entries summed over every level."""
        return sum(level.stored for level in self.levels)

    @property
    def nonzero(self) -> int:
        """Total non-zero entries summed over every level."""
        return sum(level.nonzero for level in self.levels)

    @property
    def free(self) -> int:
        """Total free parameters summed over every level."""
        return sum(level.free for level in self.levels)

    @property
    def preference_entries(self) -> int:
        """C-matrix entry count summed over levels (measured, not a parameter)."""
        return sum(level.preference_entries for level in self.levels)

    def by_level(self) -> dict[str, LevelParameters]:
        """Map each level name to its LevelParameters record."""
        return {level.level: level for level in self.levels}


def count_array(name: str, array: NDArray[np.floating] | object, kind: str) -> ArrayParameters:
    """Count one array's stored, non-zero, and free parameters.

    Args:
        name: Label for the array, e.g. ``"A[0]"``.
        array: The dense array, in pymdp's stored orientation.
        kind: ``"A"``, ``"B"`` or ``"D"`` — selects the normalized axis.

    Returns:
        The three counts plus the distribution census.

    Raises:
        ValueError: If ``kind`` is unknown, or the array contains non-finite or
            negative probabilities, or does not normalize along the axis its
            kind declares. A parameter count taken off a
            mis-oriented array is silently wrong, so the orientation is
            checked here rather than assumed.
    """
    if kind not in NORMALIZED_AXIS:
        raise ValueError(f"unknown array kind {kind!r}; expected one of {sorted(NORMALIZED_AXIS)}")

    dense = np.asarray(array, dtype=np.float64)
    axis = NORMALIZED_AXIS[kind]
    if dense.ndim <= axis:
        raise ValueError(f"{name}: shape {dense.shape} has no axis {axis} to normalize over")
    if not np.isfinite(dense).all():
        raise ValueError(f"{name}: probabilities must be finite")
    if (dense < 0.0).any():
        raise ValueError(f"{name}: probabilities must be nonnegative")

    sums = dense.sum(axis=axis)
    # A distribution that is entirely zero is not a mis-oriented array; it is an
    # unreachable slice (a control factor pymdp pads out, say). Those are
    # excluded from the check and contribute nothing to any count.
    occupied = sums > 0.0
    if occupied.any() and not np.allclose(sums[occupied], 1.0, atol=_SIMPLEX_TOLERANCE):
        worst = float(np.max(np.abs(sums[occupied] - 1.0)))
        raise ValueError(
            f"{name}: axis {axis} does not sum to 1 (worst deviation {worst:.3e}). "
            f"Either the array is mis-oriented or it is not a distribution, and a "
            f"free-parameter count would be meaningless either way."
        )

    nonzero_mask = dense != 0.0
    per_distribution = nonzero_mask.sum(axis=axis)
    live = per_distribution[per_distribution > 0]

    return ArrayParameters(
        name=name,
        shape=tuple(int(n) for n in dense.shape),
        stored=int(dense.size),
        nonzero=int(nonzero_mask.sum()),
        # One degree of freedom per distribution goes to the simplex constraint.
        free=int(np.sum(live - 1)),
        distributions=int(live.size),
        deterministic=int(np.sum(live == 1)),
    )


def count_level(level: str, agent: object) -> LevelParameters:
    """Count every generative array on a built pymdp agent.

    ``C`` is measured but excluded from the parameter totals; see the module
    docstring for why a preference is not a parameter.
    """
    arrays: list[ArrayParameters] = []
    for kind in ("A", "B", "D"):
        factors = getattr(agent, kind, None)
        if factors is None:
            continue
        for index, factor in enumerate(factors):
            arrays.append(count_array(f"{kind}[{index}]", factor, kind))

    preference = getattr(agent, "C", None)
    preference_entries = (
        sum(int(np.asarray(c).size) for c in preference) if preference is not None else 0
    )
    return LevelParameters(level=level, arrays=tuple(arrays), preference_entries=preference_entries)


def count_model() -> ModelParameters:
    """Build every level from its vendored artifact and count the whole model.

    Every level is constructed through its own shipped builder, so the counts
    describe the model the rest of the pipeline runs rather than a
    reimplementation of it that could drift.
    """
    import jax.numpy as jnp

    from ailm.levels.case import build_case_agent, load_case_artifact
    from ailm.levels.clause import build_type_models
    from ailm.levels.letter import build_letter_agent, build_letter_transition, load_brown_counts
    from ailm.levels.narrative import build_narrative_agent, load_narrative_artifact
    from ailm.levels.phoneme import (
        build_g2p_agent,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        build_phone_transition,
        build_phoneme_agent,
        g2p_split,
        load_phoneme_counts,
    )
    from ailm.levels.widened import (
        N_WIDENED_SYMBOLS,
        build_widened_agent,
        load_widened_clause_artifact,
    )
    from ailm.levels.word import build_word_agent, load_word_artifact

    phone_counts = load_phoneme_counts()
    g2p_train, _ = g2p_split(phone_counts)
    widened = load_widened_clause_artifact()
    widened_models = build_type_models(widened.train, n_symbols=N_WIDENED_SYMBOLS)

    levels = (
        count_level(
            "L1_letter",
            build_letter_agent(jnp.asarray(build_letter_transition(load_brown_counts()))),
        ),
        count_level(
            "L2A_phoneme",
            build_phoneme_agent(jnp.asarray(build_phone_transition(phone_counts))),
        ),
        count_level(
            "L2B_g2p",
            build_g2p_agent(
                build_g2p_emission(g2p_train),
                build_g2p_transition(g2p_train),
                build_g2p_start(g2p_train),
            ),
        ),
        count_level("L3_word", build_word_agent(load_word_artifact())),
        count_level("L4_case", build_case_agent(load_case_artifact())),
        count_level("L5_clause", build_widened_agent(widened_models)),
        count_level("L6_narrative", build_narrative_agent(load_narrative_artifact())),
    )
    return ModelParameters(levels=levels)


def count_case_only_clause() -> LevelParameters:
    """Count the retained ablation separately from the canonical hierarchy.

    The same array-capacity rule applies to both arms. This function only
    builds the already fitted agent; it does not evaluate any population.
    """
    from ailm.levels.clause import build_clause_agent, load_clause_artifact

    return count_level("L5_case_only", build_clause_agent(load_clause_artifact()))


# ── THE GNN BINDING ───────────────────────────────────────────────────────────

#: Every level model and the GNN file that declares it. This is the registry the
#: manuscript, the conformance tests and the parameter accounting all read, so
#: "which file declares this level" has exactly one answer. A level added
#: without an entry here fails ``tests/test_parameters.py`` rather than quietly
#: going undeclared.
GNN_SPEC_FILES: dict[str, str] = {
    "L1_letter": "gnn/models/L1_letter.gnn.md",
    "L2A_phoneme": "gnn/models/L2a_phoneme_sequence.gnn.md",
    "L2B_g2p": "gnn/models/L2b_grapheme_to_phoneme.gnn.md",
    "L3_word": "gnn/models/L3_word.gnn.md",
    "L4_case": "gnn/models/L4_case.gnn.md",
    "L5_clause": "gnn/models/L5w_clause_widened.gnn.md",
    "L6_narrative": "gnn/models/L6_narrative.gnn.md",
}

#: Specs outside the canonical parameter total: the Level 1 generation variant
#: and the retained case-only clause ablation. The latter has its own recorded
#: counts, so promoting widened Level 5 never erases the old arm's census.
GNN_AUXILIARY_SPECS: dict[str, str] = {
    "L1g_letter_generation": "gnn/models/L1g_letter_generation.gnn.md",
    "L5_case_only": "gnn/models/L5_clause.gnn.md",
}
#: The auxiliary specs' declared sizes are deliberately not re-checked here
#: against a built agent: that binding lives in the conformance suite, which
#: pins ``L1g_letter_generation`` to a live build with planted-defect negative
#: controls (tests/test_gnn_conformance.py). The census counts parameters;
#: the conformance suite binds shapes — two independent readers, not two
#: copies of one check.


def _array_key(declared_name: str) -> str | None:
    """Map a GNN variable name onto a counted array key.

    GNN writes a single-factor array as ``B`` and a multi-factor one as ``B_f0``,
    ``B_f1``; the counter labels them ``B[0]``, ``B[1]``. Anything that is not a
    generative array (hidden-state or observation variables, preferences)
    returns None and is not compared.
    """
    if declared_name in ("A", "B", "D"):
        return f"{declared_name}[0]"
    for kind in ("A", "B", "D"):
        prefix = f"{kind}_f"
        if declared_name.startswith(prefix):
            suffix = declared_name[len(prefix) :]
            if suffix.isdigit():
                return f"{kind}[{int(suffix)}]"
    return None


def gnn_declared_sizes(spec_path: str | Path) -> dict[str, int]:
    """Dense entry count per generative array, as the GNN file declares it.

    Args:
        spec_path: Path to a ``.gnn.md`` model file.

    Returns:
        ``{"B[0]": 4000000, ...}`` — the product of each declared dimension
        tuple, which is the number of entries the array must hold.
    """
    from ailm.gnn import parse_gnn_spec

    spec = parse_gnn_spec(Path(spec_path))
    sizes: dict[str, int] = {}
    for name, (dims, _dtype) in spec.variables.items():
        key = _array_key(name)
        if key is None:
            continue
        sizes[key] = math.prod(dims)
    return sizes


def check_level_against_gnn(level: LevelParameters, spec_path: str | Path) -> list[str]:
    """Discrepancies between a level's built arrays and its GNN declaration.

    The declaration and the builders are written by hand, independently — the
    spec does not generate the builders. Nothing at runtime would notice them
    disagreeing on a size: the agent would run, the gate would report a number,
    and the file the manuscript points at as the level's definition would
    describe a different model than the one measured.

    Args:
        level: Counted parameters for one level.
        spec_path: The GNN file that declares it.

    Returns:
        Human-readable discrepancies, empty when the two agree exactly.
    """
    declared = gnn_declared_sizes(spec_path)
    built = {array.name: array.stored for array in level.arrays}

    problems: list[str] = []
    for key in sorted(set(declared) | set(built)):
        if key not in declared:
            problems.append(f"{level.level}: {key} is built but not declared in {spec_path}")
        elif key not in built:
            problems.append(f"{level.level}: {key} is declared in {spec_path} but not built")
        elif declared[key] != built[key]:
            problems.append(
                f"{level.level}: {key} declared {declared[key]:,} entries, built {built[key]:,}"
            )
    return problems


def check_model_against_gnn(model: ModelParameters | None = None) -> list[str]:
    """Every level checked against its declaration, plus registry completeness.

    Returns:
        Every discrepancy found, empty when the whole model conforms.
    """
    counted = model if model is not None else count_model()
    problems: list[str] = []

    for level in counted.levels:
        spec_path = GNN_SPEC_FILES.get(level.level)
        if spec_path is None:
            problems.append(f"{level.level} has no entry in GNN_SPEC_FILES")
            continue
        resolved = _REPO_ROOT / spec_path
        if not resolved.exists():
            problems.append(f"{level.level}: declared spec {spec_path} does not exist")
            continue
        problems.extend(check_level_against_gnn(level, resolved))

    # The registry must account for every shipped spec, not just the ones a
    # level happens to reference: an orphaned .gnn.md is a model the paper
    # describes and nothing builds.
    on_disk = {
        str(path.relative_to(_REPO_ROOT))
        for path in sorted((_REPO_ROOT / "gnn" / "models").glob("*.gnn.md"))
    }
    registered = set(GNN_SPEC_FILES.values()) | set(GNN_AUXILIARY_SPECS.values())
    for orphan in sorted(on_disk - registered):
        problems.append(f"{orphan} exists but no level registers it")
    for missing in sorted(registered - on_disk):
        problems.append(f"{missing} is registered but absent from gnn/models/")
    return problems
