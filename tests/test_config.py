"""Every knob in :mod:`ailm.config` must be CONSUMED, not merely accepted.

Why this file is shaped the way it is
-------------------------------------
A configuration parameter that is plumbed but ignored is worse than no parameter
at all: it invites a sensitivity claim that the code does not support. "The
builders accept a ``config``" is a type-checking fact. "Changing ``smoothing``
changes the matrix, in the direction the arithmetic predicts" is the fact that
licenses a robustness statement. So every knob below is checked three ways:

1. **Consumption** — the artifact built at the default and at a clearly
   different value must DIFFER. If they do not, the knob is decorative.
2. **Direction and exact value** — the change must match the closed form. With
   pseudocount ``k`` over ``V`` symbols and count ``c`` in a column of observed
   total ``n``, the probability is ``(c + k) / (n + kV)``; an unseen cell is
   therefore exactly ``k / (n + kV)`` and ``KL(column || uniform)`` is strictly
   decreasing in ``k``. Those are re-derived here from the counts, never copied
   from a previous run.
3. **Default equivalence** — the builders called with no config must reproduce
   the matrices the hardcoded ``np.ones(...)`` implementations produced, bit for
   bit. That is the regression anchor for the whole change.

Bit-identity is pinned by **stored digests** rather than by eye. The digests in
:data:`_SYNTHETIC_DIGESTS` and :data:`_VENDOR_DIGESTS` were captured from the
pre-change code at git ``28423c9`` — before any ``config=`` parameter existed —
with exactly the helper below::

    def _digest(arr):
        a = np.asarray(arr)
        h = hashlib.sha256()
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
        return h.hexdigest()

``test_reference_digests_are_discriminating`` is the non-vacuity control for
them: a matrix built at a *different* pseudocount must NOT match the stored
digest. Without it, a digest table could pass because the helper had gone
constant.

Directionality caveats kept explicit
------------------------------------
Held-out perplexity is *not* monotone in ``k`` — it has an interior optimum, and
on the eval set stated in ``HELD_OUT_TEXT`` it moves from 10.32 at ``k = 1`` to
10.32 at ``k = 10``, i.e. by less than the third decimal. Only the extremes are
asserted (over-smoothing must be worse; ``k = 1e9`` must reach the uniform
ceiling), because a two-point ordering at moderate ``k`` would be noise dressed
as a finding. The same caution applies to the G2P decode: at
``emission_smoothing = 100`` accuracy barely moves, so the end-to-end assertion
is made only where the emission has genuinely been flattened.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ailm.config import DEFAULT_LEVEL_CONFIG, LevelConfig
from ailm.levels.letter import (
    ALPHA_TO_IDX,
    N_SYMBOLS,
    build_letter_agent,
    build_letter_transition,
    compute_perplexity,
    load_brown_counts,
)
from ailm.levels.phoneme import (
    N_PHONES,
    PHONE_TO_IDX,
    build_g2p_agent,
    build_g2p_emission,
    build_g2p_start,
    build_g2p_transition,
    build_phone_transition,
    build_phone_unigram,
    g2p_accuracy,
    g2p_split,
    load_phoneme_counts,
)

# ── PATHS AND SKIP GUARDS ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"
BROWN_COUNTS_FILE = VENDOR_DIR / "brown_counts.json"
CMUDICT_COUNTS_FILE = VENDOR_DIR / "cmudict_counts.json"

_SKIP_NO_BROWN = pytest.mark.skipif(
    not BROWN_COUNTS_FILE.exists(),
    reason=(
        f"Brown corpus counts not at {BROWN_COUNTS_FILE}. "
        "Run: .venv/bin/python scripts/generate_counts.py"
    ),
)
_SKIP_NO_CMUDICT = pytest.mark.skipif(
    not CMUDICT_COUNTS_FILE.exists(),
    reason=(
        f"CMUdict counts not at {CMUDICT_COUNTS_FILE}. "
        "Run: .venv/bin/python scripts/generate_phoneme_counts.py"
    ),
)

# ── SYNTHETIC INPUTS ──────────────────────────────────────────────────────────
# Small, hand-written, and vendor-independent, so the default-equivalence anchor
# and the closed-form checks still run in a checkout with no corpora vendored.
# These literals are load-bearing: the stored digests below were computed from
# exactly these inputs.
SYN_LETTER_BIGRAMS: dict[str, int] = {"a,b": 3, "b,a": 5, "a,a": 1, ".,a": 2, "q,z": 7}
SYN_PHONE_BIGRAMS: dict[str, int] = {"AA,B": 4, "B,AA": 9, "T,S": 2, "Z,ZH": 1}
SYN_PHONE_UNIGRAMS: dict[str, int] = {"AA": 10, "B": 3, "ZH": 1}
SYN_G2P_PAIRS: list[tuple[str, list[str]]] = [
    ("cat", ["K", "AE", "T"]),
    ("dog", ["D", "AO", "G"]),
    ("bee", ["B", "IY", "IY"]),
]

#: Held-out evaluation text for the end-to-end perplexity checks. Public-domain
#: nursery rhymes and a pangram, 187 characters after normalization. It is NOT
#: part of any training corpus: the letter model is fitted only on the vendored
#: Brown bigram counts. Stated here so the eval set is explicit at the
#: assertion, rather than inherited from another module.
HELD_OUT_TEXT = (
    "the cat sat on the mat and the rat ran from the flat "
    "she sells sea shells by the sea shore "
    "the quick brown fox jumps over the lazy dog "
    "mary had a little lamb its fleece was white as snow "
)

# ── STORED PRE-CHANGE REFERENCE DIGESTS ───────────────────────────────────────
# Captured from git 28423c9, before ailm.config existed, with the helper quoted
# in this module's docstring. A mismatch means the default build changed — fix
# the builder, never these constants.
_SYNTHETIC_DIGESTS: dict[str, str] = {
    "letter_a": "f1b7b45d328271a93da9e3487ba52c0c1190428135e7edd243416d9c72470c48",
    "phone_transition": "0e6dba36ddc4f02860b85ff4295af07c4a46b3affc5b141bf69a74fb265d4e54",
    "phone_unigram": "56885a1e863f27cbe728bcad0fc123edb7fac0398284c084863ef261546cac04",
    "g2p_emission": "24d76679a04d06f76f39e2a8727e6d323fedb7c94b43a42438e5c051d90fffe0",
    "g2p_transition": "b72460c6220c69bc593b7b91a29e936b156ab67ced6afd7c0e6726f668130423",
    "g2p_start": "a84400c1b6209eaca598a8fed3606f7765ebb52611211387f9507bc58cc1f6a6",
}
_VENDOR_DIGESTS: dict[str, str] = {
    "letter_a": "009a67a28ee5e6b2d26a5657a19fffca97f6c7deb473e8b9f61a7d73e58af57a",
    "phone_transition": "8594f5f368a6e5f44f186cc1b777d6a0f162b3b75680361dc95a8160f57c7685",
    "phone_unigram": "53ea82e63c9739cdd47153a091ffa7b8e48e4ff339b48cc8e5b77e5ea1d5f74b",
    "g2p_emission": "d42e4ef82c64cb185551542aa68be40457211aa86e073d7151de5ce8cef4a228",
    "g2p_transition": "44be89ef1835f0ba826060c0c9698dd0b1014965c5235d62bad520a900c58488",
    "g2p_start": "d897f99b14c80c2c20f638ab6a61ae7f6d9a232fe77ec18b3eaf186f2d3f9604",
}

#: Pseudocount grid used by the monotonicity sweeps. Spans five orders of
#: magnitude either side of the default so the direction of change is far larger
#: than float32 noise.
SMOOTHING_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


# ── HELPERS ───────────────────────────────────────────────────────────────────


def _digest(arr) -> str:
    """SHA-256 over dtype, shape and raw bytes of an array.

    Identical to the helper used to capture the stored references, so a match is
    bit-identity and not merely numerical closeness.
    """
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _add_k(count: float, column_total: float, k: float, n_symbols: int) -> float:
    """The closed-form smoothed probability ``(c + k) / (n + kV)``."""
    return (count + k) / (column_total + k * n_symbols)


def _column_totals_from_bigrams(bigrams: dict) -> dict[str, float]:
    """Observed total per conditioning symbol, i.e. the ``n`` of ``(c + k)/(n + kV)``."""
    totals: dict[str, float] = {}
    for key, cnt in bigrams.items():
        prev, _sep, _nxt = key.partition(",")
        totals[prev] = totals.get(prev, 0.0) + float(cnt)
    return totals


def _mean_kl_from_uniform(matrix: np.ndarray) -> float:
    """Mean over columns of ``KL(column || uniform)`` in nats.

    ``KL(p || u) = sum_i p_i log(p_i * V)`` when ``u`` is uniform over ``V``.
    Computed in float64 from the float32 matrix, so the value carries the
    matrix's own precision and no more.
    """
    n_rows = matrix.shape[0]
    per_column = [
        float(
            np.sum(
                matrix[:, j].astype(np.float64) * np.log(matrix[:, j].astype(np.float64) * n_rows)
            )
        )
        for j in range(matrix.shape[1])
    ]
    return float(np.mean(per_column))


def _build_all_synthetic(config: LevelConfig | None) -> dict[str, np.ndarray]:
    """Build every configurable artifact from the synthetic inputs.

    ``config=None`` exercises the no-argument path that must stay byte-identical
    to the pre-change code.
    """
    if config is None:
        return {
            "letter_a": np.asarray(build_letter_transition(SYN_LETTER_BIGRAMS)),
            "phone_transition": np.asarray(build_phone_transition(SYN_PHONE_BIGRAMS)),
            "phone_unigram": np.asarray(build_phone_unigram(SYN_PHONE_UNIGRAMS)),
            "g2p_emission": np.asarray(build_g2p_emission(SYN_G2P_PAIRS)),
            "g2p_transition": np.asarray(build_g2p_transition(SYN_G2P_PAIRS)),
            "g2p_start": np.asarray(build_g2p_start(SYN_G2P_PAIRS)),
        }
    return {
        "letter_a": np.asarray(build_letter_transition(SYN_LETTER_BIGRAMS, config=config)),
        "phone_transition": np.asarray(build_phone_transition(SYN_PHONE_BIGRAMS, config=config)),
        "phone_unigram": np.asarray(build_phone_unigram(SYN_PHONE_UNIGRAMS, config=config)),
        "g2p_emission": np.asarray(build_g2p_emission(SYN_G2P_PAIRS, config=config)),
        "g2p_transition": np.asarray(build_g2p_transition(SYN_G2P_PAIRS, config=config)),
        "g2p_start": np.asarray(build_g2p_start(SYN_G2P_PAIRS, config=config)),
    }


def _build_all_vendor(config: LevelConfig | None) -> dict[str, np.ndarray]:
    """Build every configurable artifact from the vendored corpora."""
    brown = load_brown_counts()
    counts = load_phoneme_counts()
    train, _test = g2p_split(counts) if config is None else g2p_split(counts, config=config)
    if config is None:
        return {
            "letter_a": np.asarray(build_letter_transition(brown)),
            "phone_transition": np.asarray(build_phone_transition(counts["phone_bigrams"])),
            "phone_unigram": np.asarray(build_phone_unigram(counts["phone_unigrams"])),
            "g2p_emission": np.asarray(build_g2p_emission(train)),
            "g2p_transition": np.asarray(build_g2p_transition(train)),
            "g2p_start": np.asarray(build_g2p_start(train)),
        }
    return {
        "letter_a": np.asarray(build_letter_transition(brown, config=config)),
        "phone_transition": np.asarray(
            build_phone_transition(counts["phone_bigrams"], config=config)
        ),
        "phone_unigram": np.asarray(build_phone_unigram(counts["phone_unigrams"], config=config)),
        "g2p_emission": np.asarray(build_g2p_emission(train, config=config)),
        "g2p_transition": np.asarray(build_g2p_transition(train, config=config)),
        "g2p_start": np.asarray(build_g2p_start(train, config=config)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. THE CONFIG OBJECT ITSELF
# ══════════════════════════════════════════════════════════════════════════════


def test_default_config_is_add_one_laplace_with_the_artifact_stride():
    """The documented default must be the behavior the project already had."""
    assert DEFAULT_LEVEL_CONFIG.smoothing == 1.0
    assert DEFAULT_LEVEL_CONFIG.emission_smoothing is None
    assert DEFAULT_LEVEL_CONFIG.test_every is None
    assert DEFAULT_LEVEL_CONFIG.resolved_emission_smoothing == 1.0
    assert LevelConfig() == DEFAULT_LEVEL_CONFIG


def test_resolved_emission_smoothing_falls_back_to_smoothing():
    """With no emission-specific value, the emission follows ``smoothing``."""
    assert LevelConfig(smoothing=4.5).resolved_emission_smoothing == 4.5


def test_resolved_emission_smoothing_overrides_smoothing():
    """When both are set, the emission-specific value wins for the emission."""
    cfg = LevelConfig(smoothing=4.5, emission_smoothing=0.25)
    assert cfg.resolved_emission_smoothing == 0.25
    assert cfg.smoothing == 4.5


def test_config_is_frozen():
    """A builder must not be able to mutate the caller's configuration."""
    cfg = LevelConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.smoothing = 2.0  # type: ignore[misc]


def test_config_is_hashable_and_compares_by_value():
    """A sweep keys results by configuration, so value semantics are required."""
    assert LevelConfig(smoothing=2.0) == LevelConfig(smoothing=2.0)
    assert LevelConfig(smoothing=2.0) != LevelConfig(smoothing=3.0)
    assert len({LevelConfig(smoothing=2.0), LevelConfig(smoothing=2.0)}) == 1


def test_integer_pseudocounts_are_accepted_as_reals():
    """``smoothing=3`` is the same configuration as ``smoothing=3.0``."""
    assert LevelConfig(smoothing=3).resolved_emission_smoothing == 3.0


# ── REJECTED-VALUE CONTROLS ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad",
    [0, 0.0, -1, -1.0, -0.5, float("nan"), float("inf"), float("-inf")],
)
def test_rejected_smoothing_values(bad):
    """A zero, negative or non-finite pseudocount must raise, naming the field.

    ``0`` restores unsmoothed MLE tables, so one unseen bigram in held-out text
    makes the perplexity infinite; a negative pseudocount can produce negative
    "probabilities" that still normalize and still pass pymdp's ``_validate()``;
    ``nan``/``inf`` poison every cell. All four failures are quiet downstream,
    which is why they are refused at construction.
    """
    with pytest.raises(ValueError) as exc:
        LevelConfig(smoothing=bad)
    message = str(exc.value)
    assert "smoothing" in message, f"error does not name the field: {message}"
    assert repr(bad) in message, f"error does not name the value: {message}"


@pytest.mark.parametrize("bad", ["1.0", None, True, [1.0]])
def test_rejected_smoothing_types(bad):
    """A non-real pseudocount must raise rather than being coerced."""
    with pytest.raises(ValueError, match="smoothing"):
        LevelConfig(smoothing=bad)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_rejected_emission_smoothing_values(bad):
    """The emission pseudocount is validated on exactly the same terms."""
    with pytest.raises(ValueError) as exc:
        LevelConfig(emission_smoothing=bad)
    message = str(exc.value)
    assert "emission_smoothing" in message
    assert repr(bad) in message


@pytest.mark.parametrize("bad", [1, 0, -3])
def test_rejected_test_every_values(bad):
    """A stride below 2 must raise: at 1 the TRAIN split is empty."""
    with pytest.raises(ValueError) as exc:
        LevelConfig(test_every=bad)
    message = str(exc.value)
    assert "test_every" in message
    assert repr(bad) in message


@pytest.mark.parametrize("bad", [2.5, "5", True, 2.0])
def test_rejected_test_every_types(bad):
    """``i % test_every == 0`` is only defined for integers."""
    with pytest.raises(ValueError, match="test_every"):
        LevelConfig(test_every=bad)


def test_minimum_valid_test_every_is_accepted():
    """2 is the smallest stride that leaves a non-empty TRAIN split."""
    assert LevelConfig(test_every=2).test_every == 2


def test_nan_is_rejected_before_the_sign_test():
    """Regression guard for the classic fail-closed hole: ``nan <= 0`` is False.

    A validator written only as ``if value <= 0: raise`` accepts NaN. This
    asserts the NaN branch fires with its own message rather than the
    positivity message, which is the observable difference between the two
    implementations.
    """
    with pytest.raises(ValueError, match="finite"):
        LevelConfig(smoothing=float("nan"))


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFAULT EQUIVALENCE — the bit-identical regression anchor
# ══════════════════════════════════════════════════════════════════════════════


def test_synthetic_default_digests_match_the_prechange_reference():
    """Every builder, called with no config, is byte-identical to git 28423c9."""
    built = _build_all_synthetic(None)
    assert set(built) == set(_SYNTHETIC_DIGESTS), "anchor set drifted from builder set"
    mismatches = {
        name: (_digest(arr), _SYNTHETIC_DIGESTS[name])
        for name, arr in built.items()
        if _digest(arr) != _SYNTHETIC_DIGESTS[name]
    }
    assert mismatches == {}, (
        f"Default build is no longer bit-identical to the pre-config code: "
        f"{mismatches}. The keyword-only config parameters must not change any "
        f"default value. Fix the builder, not this table."
    )


@_SKIP_NO_BROWN
@_SKIP_NO_CMUDICT
def test_vendor_default_digests_match_the_prechange_reference():
    """The same anchor on the real corpora, which is what the manuscript cites."""
    built = _build_all_vendor(None)
    mismatches = {
        name: (_digest(arr), _VENDOR_DIGESTS[name])
        for name, arr in built.items()
        if _digest(arr) != _VENDOR_DIGESTS[name]
    }
    assert mismatches == {}, (
        f"Default build from the vendored corpora drifted: {mismatches}. Either a "
        f"builder default changed or a vendored artifact was regenerated."
    )


def test_reference_digests_are_discriminating():
    """NON-VACUITY CONTROL — a different pseudocount must NOT match the anchor.

    Without this, the digest tables above could pass because ``_digest`` had
    become constant or because the builders had stopped consuming their inputs.
    """
    off_default = _build_all_synthetic(LevelConfig(smoothing=2.0, test_every=2))
    for name, arr in off_default.items():
        assert _digest(arr) != _SYNTHETIC_DIGESTS[name], (
            f"{name} built at smoothing=2.0 has the same digest as the add-1 "
            f"reference — the digest is not discriminating, so the anchor above "
            f"proves nothing."
        )


def test_explicit_default_config_equals_no_config():
    """Passing ``DEFAULT_LEVEL_CONFIG`` and passing nothing are the same call."""
    implicit = _build_all_synthetic(None)
    explicit = _build_all_synthetic(DEFAULT_LEVEL_CONFIG)
    for name in implicit:
        assert np.array_equal(implicit[name], explicit[name]), (
            f"{name} differs between config=None and config=DEFAULT_LEVEL_CONFIG"
        )


def test_unit_smoothing_config_equals_no_config():
    """A freshly constructed ``LevelConfig(smoothing=1.0)`` is also the default."""
    implicit = _build_all_synthetic(None)
    unit = _build_all_synthetic(LevelConfig(smoothing=1.0))
    for name in implicit:
        assert np.array_equal(implicit[name], unit[name]), f"{name} differs at k=1.0"


def test_default_letter_matrix_matches_an_independent_add_one_oracle():
    """The stored digest is cross-checked by re-deriving add-1 from scratch.

    An independent implementation of the original arithmetic — ``np.ones`` plus
    the counts, normalized down the columns — must reproduce the default build
    exactly. This is what makes the digest table interpretable rather than
    magic: if both this and the digest fail, the builder changed; if only the
    digest fails, the reference was captured wrongly.
    """
    expected = np.ones((N_SYMBOLS, N_SYMBOLS), dtype=np.float64)
    for key, cnt in SYN_LETTER_BIGRAMS.items():
        prev, _sep, curr = key.partition(",")
        expected[ALPHA_TO_IDX[curr], ALPHA_TO_IDX[prev]] += float(cnt)
    expected = expected / expected.sum(axis=0, keepdims=True)

    built = np.asarray(build_letter_transition(SYN_LETTER_BIGRAMS))
    assert built.shape == (1, N_SYMBOLS, N_SYMBOLS)
    # float32 round-trip of the same float64 arithmetic, so cast the oracle the
    # same way the builder does rather than loosening the comparison.
    assert np.array_equal(built[0], expected.astype(built.dtype)), (
        "Default build disagrees with an independent add-1 Laplace derivation."
    )


@_SKIP_NO_CMUDICT
def test_default_g2p_split_is_still_the_artifact_split():
    """``test_every=None`` must reproduce the artifact-defined split exactly."""
    counts = load_phoneme_counts()
    artifact_stride = counts["test_every"]
    train_a, test_a = g2p_split(counts)
    train_b, test_b = g2p_split(counts, config=DEFAULT_LEVEL_CONFIG)
    train_c, test_c = g2p_split(counts, config=LevelConfig(test_every=artifact_stride))
    assert train_a == train_b == train_c
    assert test_a == test_b == test_c
    assert len(test_a) > 0 and len(train_a) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. KNOB: smoothing — consumed by every count table
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "name",
    [
        "letter_a",
        "phone_transition",
        "phone_unigram",
        "g2p_emission",
        "g2p_transition",
        "g2p_start",
    ],
)
def test_every_artifact_changes_when_smoothing_changes(name):
    """CONSUMPTION PROOF — one test per artifact, so a silent gap cannot hide.

    ``build_g2p_transition`` and ``build_g2p_start`` only pass here if they
    forward the config to their delegates, which is exactly the plumbing that is
    easy to leave out.
    """
    default = _build_all_synthetic(None)[name]
    changed = _build_all_synthetic(LevelConfig(smoothing=50.0))[name]
    assert default.shape == changed.shape
    assert not np.array_equal(default, changed), (
        f"{name} is identical at smoothing=1.0 and smoothing=50.0 — the knob is "
        f"accepted but not consumed."
    )
    # Still a valid distribution at the new value. Every one of these artifacts
    # is normalized along axis 1 — that is the axis pymdp validates, whether the
    # array is a (1, V, V) emission, a (1, V, V, 1) transition, or a (1, V) prior.
    np.testing.assert_allclose(
        changed.sum(axis=1).ravel(),
        1.0,
        rtol=1e-6,
        err_msg=f"{name} stopped being normalized at smoothing=50.0",
    )
    assert changed.min() > 0.0, f"{name} has a zero cell at smoothing=50.0"


@pytest.mark.parametrize("k", SMOOTHING_GRID)
def test_unseen_letter_bigram_cell_is_exactly_k_over_n_plus_kv(k):
    """DIRECTION, EXACT — an unseen cell must equal ``k / (n + kV)``.

    The value is derived inside the test from the same counts the builder saw.
    """
    totals = _column_totals_from_bigrams(SYN_LETTER_BIGRAMS)
    observed = {tuple(key.partition(",")[::2]) for key in SYN_LETTER_BIGRAMS}
    # 'a' is a conditioning symbol with observed mass; ('a','c') was never seen.
    assert ("a", "c") not in observed, "test premise broken: ('a','c') is not unseen"
    n = totals["a"]
    assert n > 0.0

    matrix = np.asarray(
        build_letter_transition(SYN_LETTER_BIGRAMS, config=LevelConfig(smoothing=k))
    )[0]
    got = float(matrix[ALPHA_TO_IDX["c"], ALPHA_TO_IDX["a"]])
    expected = _add_k(0.0, n, k, N_SYMBOLS)
    assert got == pytest.approx(expected, rel=1e-6), (
        f"unseen cell at k={k}: got {got!r}, closed form k/(n + kV) = {expected!r} "
        f"with n={n}, V={N_SYMBOLS}"
    )


@pytest.mark.parametrize("k", SMOOTHING_GRID)
def test_seen_letter_bigram_cell_matches_the_add_k_formula(k):
    """DIRECTION, EXACT — a seen cell must equal ``(c + k) / (n + kV)``."""
    totals = _column_totals_from_bigrams(SYN_LETTER_BIGRAMS)
    matrix = np.asarray(
        build_letter_transition(SYN_LETTER_BIGRAMS, config=LevelConfig(smoothing=k))
    )[0]
    got = float(matrix[ALPHA_TO_IDX["b"], ALPHA_TO_IDX["a"]])
    expected = _add_k(float(SYN_LETTER_BIGRAMS["a,b"]), totals["a"], k, N_SYMBOLS)
    assert got == pytest.approx(expected, rel=1e-6)


@_SKIP_NO_BROWN
def test_unseen_brown_bigram_cell_is_exactly_k_over_n_plus_kv():
    """The same closed form on the real corpus, where ``n`` is ~10^6.

    Uses the first genuinely unseen pair in sorted order, and asserts that at
    least one exists so the check cannot pass by finding nothing to test.
    """
    brown = load_brown_counts()
    observed = {tuple(key.partition(",")[::2]) for key in brown}
    totals = _column_totals_from_bigrams(brown)
    unseen = [
        (prev, curr)
        for prev in sorted(ALPHA_TO_IDX)
        for curr in sorted(ALPHA_TO_IDX)
        if (prev, curr) not in observed and totals.get(prev, 0.0) > 0.0
    ]
    assert unseen, "no unseen Brown bigram exists — this check would be vacuous"
    prev, curr = unseen[0]
    n = totals[prev]

    for k in (0.01, 1.0, 100.0):
        matrix = np.asarray(build_letter_transition(brown, config=LevelConfig(smoothing=k)))[0]
        got = float(matrix[ALPHA_TO_IDX[curr], ALPHA_TO_IDX[prev]])
        expected = _add_k(0.0, n, k, N_SYMBOLS)
        assert got == pytest.approx(expected, rel=1e-6), (
            f"unseen Brown cell ({prev!r},{curr!r}) at k={k}: got {got!r}, "
            f"expected {expected!r} (n={n})"
        )


@_SKIP_NO_BROWN
def test_letter_kl_from_uniform_decreases_monotonically_with_smoothing():
    """DIRECTION — larger ``k`` moves every column toward uniform, monotonically.

    Derivation, so the assertion is not a remembered shape: the smoothed column
    is a convex mixture ``p_k = lam * p_mle + (1 - lam) * u`` with
    ``lam = n / (n + kV)`` strictly decreasing in ``k``. ``h(lam) = KL(p_lam || u)``
    is convex with ``h(0) = 0`` and ``h'(0) = sum_i (p_i - u_i) = 0``, so ``h`` is
    non-decreasing in ``lam`` — hence non-increasing in ``k``, and strictly
    decreasing whenever the MLE column differs from uniform, which it does on
    Brown counts.
    """
    brown = load_brown_counts()
    kls = [
        _mean_kl_from_uniform(
            np.asarray(build_letter_transition(brown, config=LevelConfig(smoothing=k)))[0]
        )
        for k in SMOOTHING_GRID
    ]
    for (k_lo, kl_lo), (k_hi, kl_hi) in pairwise(zip(SMOOTHING_GRID, kls, strict=True)):
        assert kl_hi < kl_lo, (
            f"KL from uniform did not decrease from k={k_lo} ({kl_lo:.10f}) to "
            f"k={k_hi} ({kl_hi:.10f}). Larger pseudocounts must flatten the "
            f"columns."
        )
    assert kls[0] > 0.0, "the k -> 0 end must retain corpus structure"
    assert kls[-1] < kls[0] / 2.0, f"the k -> inf end barely moved: {kls[-1]:.6f} vs {kls[0]:.6f}"


@_SKIP_NO_BROWN
def test_huge_smoothing_drives_the_letter_columns_to_uniform():
    """LIMIT — as ``k -> inf`` every cell must approach ``1 / V``."""
    brown = load_brown_counts()
    matrix = np.asarray(build_letter_transition(brown, config=LevelConfig(smoothing=1e12)))[0]
    np.testing.assert_allclose(matrix, 1.0 / N_SYMBOLS, rtol=1e-4)


def test_unseen_phone_transition_cell_matches_the_add_k_formula():
    """The transition family consumes ``smoothing`` with the same closed form."""
    totals = _column_totals_from_bigrams(SYN_PHONE_BIGRAMS)
    for k in (0.5, 1.0, 25.0):
        matrix = np.asarray(
            build_phone_transition(SYN_PHONE_BIGRAMS, config=LevelConfig(smoothing=k))
        )[0, :, :, 0]
        # ('AA','T') is unseen; 'AA' has observed mass from ('AA','B').
        got = float(matrix[PHONE_TO_IDX["T"], PHONE_TO_IDX["AA"]])
        assert got == pytest.approx(_add_k(0.0, totals["AA"], k, N_PHONES), rel=1e-6)
        seen = float(matrix[PHONE_TO_IDX["B"], PHONE_TO_IDX["AA"]])
        assert seen == pytest.approx(
            _add_k(float(SYN_PHONE_BIGRAMS["AA,B"]), totals["AA"], k, N_PHONES), rel=1e-6
        )


def test_phone_unigram_matches_the_add_k_formula():
    """The start-prior family consumes ``smoothing`` with the same closed form."""
    total = float(sum(SYN_PHONE_UNIGRAMS.values()))
    for k in (0.5, 1.0, 25.0):
        prior = np.asarray(
            build_phone_unigram(SYN_PHONE_UNIGRAMS, config=LevelConfig(smoothing=k))
        ).ravel()
        got = float(prior[PHONE_TO_IDX["AA"]])
        assert got == pytest.approx(
            _add_k(float(SYN_PHONE_UNIGRAMS["AA"]), total, k, N_PHONES), rel=1e-6
        )
        unseen = float(prior[PHONE_TO_IDX["T"]])
        assert unseen == pytest.approx(_add_k(0.0, total, k, N_PHONES), rel=1e-6)


def test_g2p_emission_cell_matches_the_add_k_formula():
    """The emission is normalized over LETTERS, so ``V`` here is 30, not 39."""
    # SYN_G2P_PAIRS aligns 1:1, so phone 'T' is observed exactly once, with 't'.
    for k in (0.5, 1.0, 25.0):
        emission = np.asarray(build_g2p_emission(SYN_G2P_PAIRS, config=LevelConfig(smoothing=k)))[0]
        got = float(emission[ALPHA_TO_IDX["t"], PHONE_TO_IDX["T"]])
        assert got == pytest.approx(_add_k(1.0, 1.0, k, N_SYMBOLS), rel=1e-6)
        unseen = float(emission[ALPHA_TO_IDX["z"], PHONE_TO_IDX["T"]])
        assert unseen == pytest.approx(_add_k(0.0, 1.0, k, N_SYMBOLS), rel=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 4. KNOB: emission_smoothing — consumed by the G2P emission ALONE
# ══════════════════════════════════════════════════════════════════════════════


def test_emission_smoothing_moves_the_emission():
    """CONSUMPTION PROOF for the second pseudocount."""
    default = np.asarray(build_g2p_emission(SYN_G2P_PAIRS))
    changed = np.asarray(
        build_g2p_emission(SYN_G2P_PAIRS, config=LevelConfig(emission_smoothing=30.0))
    )
    assert not np.array_equal(default, changed), (
        "emission_smoothing is accepted but not consumed by build_g2p_emission"
    )


def test_emission_smoothing_leaves_the_transition_and_start_untouched():
    """SEPARATION PROOF — the two pseudocounts are distinct knobs, not aliases.

    If ``build_g2p_transition`` read ``resolved_emission_smoothing`` this would
    fail, which is the mistake that would make the pair of knobs a lie.
    """
    cfg = LevelConfig(emission_smoothing=30.0)
    assert np.array_equal(
        np.asarray(build_g2p_transition(SYN_G2P_PAIRS)),
        np.asarray(build_g2p_transition(SYN_G2P_PAIRS, config=cfg)),
    ), "emission_smoothing leaked into the transition"
    assert np.array_equal(
        np.asarray(build_g2p_start(SYN_G2P_PAIRS)),
        np.asarray(build_g2p_start(SYN_G2P_PAIRS, config=cfg)),
    ), "emission_smoothing leaked into the start prior"
    assert np.array_equal(
        np.asarray(build_phone_transition(SYN_PHONE_BIGRAMS)),
        np.asarray(build_phone_transition(SYN_PHONE_BIGRAMS, config=cfg)),
    ), "emission_smoothing leaked into build_phone_transition"


def test_emission_smoothing_equals_smoothing_when_it_is_the_only_dial_set():
    """``emission_smoothing=k`` and ``smoothing=k`` must build the same emission."""
    via_emission = np.asarray(
        build_g2p_emission(SYN_G2P_PAIRS, config=LevelConfig(emission_smoothing=7.0))
    )
    via_smoothing = np.asarray(build_g2p_emission(SYN_G2P_PAIRS, config=LevelConfig(smoothing=7.0)))
    assert np.array_equal(via_emission, via_smoothing)


@_SKIP_NO_CMUDICT
def test_larger_emission_smoothing_flattens_the_emission_monotonically():
    """DIRECTION — the emission columns approach uniform over letters as k grows.

    Measured on the real TRAIN split rather than :data:`SYN_G2P_PAIRS`. The
    synthetic table has one observation per aligned pair, so by ``k = 1000`` its
    columns are uniform to within float32 resolution (mean KL ~4e-8) and the
    sweep stops being *measurable* even though it is still mathematically
    monotone. Asserting monotonicity there would be asserting rounding noise.
    """
    train, _test = g2p_split(load_phoneme_counts())
    kls = [
        _mean_kl_from_uniform(
            np.asarray(build_g2p_emission(train, config=LevelConfig(emission_smoothing=k)))[0]
        )
        for k in SMOOTHING_GRID
    ]
    for (k_lo, kl_lo), (k_hi, kl_hi) in pairwise(zip(SMOOTHING_GRID, kls, strict=True)):
        assert kl_hi < kl_lo, (
            f"emission KL from uniform did not decrease from k={k_lo} "
            f"({kl_lo:.10f}) to k={k_hi} ({kl_hi:.10f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. KNOB: test_every — consumed by the split AND by everything downstream
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_CMUDICT
@pytest.mark.parametrize("stride", [2, 3, 5, 13, 50])
def test_test_every_sets_the_split_sizes_exactly(stride):
    """CONSUMPTION + EXACT — held-out size is ``ceil(N / stride)`` by the rule itself.

    ``i`` is TEST iff ``i % stride == 0``, so the count of test indices in
    ``range(N)`` is ``(N + stride - 1) // stride``. Derived from the rule, not
    remembered from a run.
    """
    counts = load_phoneme_counts()
    total = len(counts["g2p_pairs"])
    train, test = g2p_split(counts, config=LevelConfig(test_every=stride))
    assert len(test) == (total + stride - 1) // stride
    assert len(train) + len(test) == total, "the split stopped being a partition"
    assert len(train) > 0


@_SKIP_NO_CMUDICT
def test_held_out_fraction_decreases_monotonically_with_test_every():
    """DIRECTION — the held-out fraction is ~1/stride, so it must shrink."""
    counts = load_phoneme_counts()
    total = len(counts["g2p_pairs"])
    strides = [2, 3, 5, 13, 50]
    fractions = [
        len(g2p_split(counts, config=LevelConfig(test_every=s))[1]) / total for s in strides
    ]
    for (s_lo, f_lo), (s_hi, f_hi) in pairwise(zip(strides, fractions, strict=True)):
        assert f_hi < f_lo, (
            f"held-out fraction did not shrink from stride {s_lo} ({f_lo:.6f}) to "
            f"stride {s_hi} ({f_hi:.6f})"
        )
    assert fractions[0] == pytest.approx(0.5, abs=1e-4)


@_SKIP_NO_CMUDICT
def test_test_every_selects_exactly_the_stride_indices():
    """The rule must be positional and must not re-sort the artifact list."""
    counts = load_phoneme_counts()
    pairs = counts["g2p_pairs"]
    stride = 7
    _train, test = g2p_split(counts, config=LevelConfig(test_every=stride))
    expected_first_words = [str(pairs[i][0]) for i in range(0, min(len(pairs), stride * 5), stride)]
    assert [word for word, _phones in test[:5]] == expected_first_words[:5]


@_SKIP_NO_CMUDICT
def test_test_every_flows_into_the_trained_matrices():
    """CONSUMPTION, DOWNSTREAM — a different split must train a different model.

    Changing list lengths is not enough: the knob only means something if the
    matrices built from the TRAIN split move with it.
    """
    counts = load_phoneme_counts()
    default_train, _ = g2p_split(counts)
    other_train, _ = g2p_split(counts, config=LevelConfig(test_every=2))
    assert len(other_train) != len(default_train)
    assert not np.array_equal(
        np.asarray(build_g2p_emission(default_train)),
        np.asarray(build_g2p_emission(other_train)),
    ), "the emission is unchanged by a different train split"
    assert not np.array_equal(
        np.asarray(build_g2p_transition(default_train)),
        np.asarray(build_g2p_transition(other_train)),
    ), "the transition is unchanged by a different train split"


@_SKIP_NO_CMUDICT
def test_a_bad_artifact_stride_still_raises_even_with_an_override():
    """The override must not launder a malformed provenance field.

    ``test_every`` in the artifact is provenance for the shipped split. A config
    override changes the experiment; it must not make a broken artifact look
    fine.
    """
    counts = dict(load_phoneme_counts())
    counts["test_every"] = 0
    with pytest.raises(ValueError, match="test_every"):
        g2p_split(counts, config=LevelConfig(test_every=4))


# ══════════════════════════════════════════════════════════════════════════════
# 6. END-TO-END EFFECT — the knob must reach the reported metric
# ══════════════════════════════════════════════════════════════════════════════


@_SKIP_NO_BROWN
def test_oversmoothing_worsens_held_out_letter_perplexity():
    """END-TO-END — over-smoothing discards learned structure, so PP must rise.

    Eval set: :data:`HELD_OUT_TEXT` — 187 characters of public-domain nursery
    rhymes plus a pangram, never used to fit the model (the model is fitted only
    on the vendored Brown bigram counts). Both agents are scored on exactly that
    string, so the comparison is paired by construction.

    Only the extremes are compared. Held-out perplexity has an interior optimum
    in ``k`` and moves by well under 0.1 between k=1 and k=10 on this eval set;
    asserting an ordering there would be asserting noise.
    """
    brown = load_brown_counts()
    default_agent = build_letter_agent(build_letter_transition(brown))
    oversmoothed_agent = build_letter_agent(
        build_letter_transition(brown, config=LevelConfig(smoothing=1e6))
    )
    default_pp = compute_perplexity(default_agent, HELD_OUT_TEXT)
    oversmoothed_pp = compute_perplexity(oversmoothed_agent, HELD_OUT_TEXT)

    assert math.isfinite(default_pp) and math.isfinite(oversmoothed_pp)
    assert default_pp < oversmoothed_pp, (
        f"Over-smoothing did not hurt: k=1 gave {default_pp:.4f} and k=1e6 gave "
        f"{oversmoothed_pp:.4f} on HELD_OUT_TEXT. Either the pseudocount is not "
        f"reaching the agent or the metric is not reading the matrix."
    )
    assert default_pp < float(N_SYMBOLS), (
        f"the default model must still beat the uniform bar of {N_SYMBOLS}"
    )


@_SKIP_NO_BROWN
def test_extreme_smoothing_reaches_the_uniform_perplexity_ceiling():
    """END-TO-END LIMIT — at ``k = 1e9`` the model must score the uniform value.

    Eval set: :data:`HELD_OUT_TEXT`. This pins the far end of the dial to an
    analytic anchor (``exp(log 30) = 30``) rather than to a measured number, so
    it cannot drift.
    """
    brown = load_brown_counts()
    agent = build_letter_agent(build_letter_transition(brown, config=LevelConfig(smoothing=1e9)))
    perplexity = compute_perplexity(agent, HELD_OUT_TEXT)
    assert perplexity == pytest.approx(float(N_SYMBOLS), abs=0.05), (
        f"k=1e9 should be indistinguishable from uniform ({N_SYMBOLS}), got {perplexity:.6f}"
    )


@_SKIP_NO_CMUDICT
def test_flattening_the_emission_collapses_g2p_accuracy():
    """END-TO-END for ``emission_smoothing`` — an uninformative A must destroy it.

    Eval set: the first 100 pairs of the held-out G2P test split taken by the
    module's evenly spaced stride (``g2p_accuracy(..., limit=100)``), which spans
    the alphabet rather than a prefix of it. Both agents see exactly the same
    positions in the same order, and only the emission differs between them.

    The extreme is used deliberately: at ``emission_smoothing = 100`` accuracy on
    this eval set moves by under 0.2 points, so a moderate value proves nothing
    end-to-end even though the matrix has demonstrably changed.
    """
    counts = load_phoneme_counts()
    train, test = g2p_split(counts)
    transition = build_g2p_transition(train)
    start = build_g2p_start(train)

    trained = build_g2p_agent(build_g2p_emission(train), transition, start)
    flattened = build_g2p_agent(
        build_g2p_emission(train, config=LevelConfig(emission_smoothing=1e6)),
        transition,
        start,
    )
    trained_acc = g2p_accuracy(trained, test, limit=100)
    flattened_acc = g2p_accuracy(flattened, test, limit=100)

    assert trained_acc > 3.0 * flattened_acc, (
        f"Flattening the emission barely changed the decode: {trained_acc:.4f} -> "
        f"{flattened_acc:.4f}. emission_smoothing is not reaching the decoder."
    )
    assert flattened_acc < 0.30, (
        f"an emission with pseudocount 1e6 is essentially uniform over letters, so "
        f"the decode should collapse; got {flattened_acc:.4f}"
    )
