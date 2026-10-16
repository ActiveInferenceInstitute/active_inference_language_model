"""Where the external benchmark's captured-mass floor actually belongs.

One figure, four panels, all drawn from a single offline sweep of AILM's own
letter transition columns — the same sweep
``scripts/record_calibration_floor_claims.py`` ran to locate the floor.

What each panel is for
----------------------
``signed`` (top left)
    Each metric's *signed* rule difference (min-bound minus observed-only)
    against captured mass, with the rule-versus-rule total variation on the
    right axis. This is the panel the whole figure exists for: both signed
    differences pass through zero once, close to the declared floor, while the
    two reconstructions are still nearly disjoint. The grid points are drawn as
    markers rather than smoothed, because the non-monotonicity between them is
    the reason a first-crossing locator is unsound and a smooth curve would hide
    it.
``absolute`` (top right)
    The same two divergences in absolute value on a log axis, against the
    declared negligibility threshold. The notch each metric cuts below the
    threshold long before it stays there is the trap, and the shaded band is the
    span a first-crossing rule would have wrongly certified.
``reachable`` (middle, full width)
    Candidates per position needed to reach a given captured mass, against the
    server's ``top_logprobs`` cap. The located floor sits where a large share of
    positions need more candidates than the endpoint will return, which is why
    it cannot simply be adopted.
``ruler`` (bottom strip)
    The declared floor, the located floor, the two first crossings and the three
    measured probes on one captured-mass axis, with the admitted-but-divergent
    band between the two floors shaded.

A fifth panel — the rule-decided share of a published expected calibration
error at the declared floor — was cut from the figure (kept only in prose,
where the same number already appears: see ``98_supplement_external_
benchmarks.md`` and ``05_discussion.md``). Five panels of dense annotation was
the highest-clutter figure in the manuscript's figure set, and that panel's
one number was the least load-bearing of the five: stating it once in text
carries it exactly as well and leaves the other four panels more legible. The
underlying computation (``calibration_floor_data``) still derives and
falsifies that quantity; only the bar-chart panel is gone.

Provenance and the falsifier
----------------------------
The per-grid-point series is not in the ledger — only summary statistics are —
so the sweep is recomputed here at draw time from the same
:mod:`ailm.benchmarks` functions the recorder called, on parameters read from the
ledger rather than typed. That is reproducible because the sweep is
deterministic and offline: it makes no model call, draws no random number, and
reads one vendored corpus artifact.

Recomputation without a check would be a second, unaudited measurement, so
:func:`calibration_floor_data` refuses to draw if any quantity it recomputes
contradicts a recorded summary claim — the identity control, the located floor
and its candidate cost, both first crossings, the row at the declared floor,
the extremes, and the recorded monotonicity and sign-change counts. This is the
``complexity_ops`` / ``multiplicity_forest`` pattern: a disagreement is a finding
about the recorder or the shipped rule, not something to redraw around.

Nothing here runs a recorder and nothing here writes the ledger.

Every generator returns an unwritten ``Figure``; ``scripts/generate_figures.py``
discovers them by the ``generate_`` prefix and handles saving.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ailm.benchmarks import (
    LETTER_TEXT,
    calibration_floor_grid,
    completion_rule_sweep,
    divergence_increases,
    divergence_sign_changes,
    fidelity_crossover,
    locate_divergence_floor,
    subsample_positions,
    unique_letter_positions,
)
from ailm.benchmarks.constants import PROBE_DISPLAY_NAMES
from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.figures._common import (
    token_float,
    token_int,
)
from ailm.levels.letter import load_brown_counts
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = [
    "build_calibration_floor_figure",
    "calibration_floor_data",
    "generate_calibration_floor",
]

#: Semantic color roles for this figure, fixed once so the five panels agree.
#: ``FLOOR`` is the palette's "bound" hue and carries the located floor here;
#: ``GATE`` is the palette's declared-threshold hue and carries the declared one,
#: which is exactly the distinction the figure is about.
COLOR_ECE = PALETTE["MODEL"]  # the shipped min-bound rule, and the ECE axis
COLOR_BRIER = PALETTE["UNIGRAM"]  # the Brier axis
COLOR_OBSERVED = PALETTE["ANTI"]  # the observed-only alternative rule
COLOR_TV = PALETTE["ACCENT"]  # distance between the two reconstructions
COLOR_TRAP = PALETTE["HIGHLIGHT"]  # the first crossing and the band it certifies
COLOR_DECLARED = PALETTE["GATE"]  # the declared captured-mass floor
COLOR_LOCATED = PALETTE["FLOOR"]  # the located captured-mass floor
COLOR_INK = PALETTE["INK"]
COLOR_MUTED = PALETTE["MUTED"]

#: Relative tolerance for holding a recomputed quantity against its ledger token.
#: ``manuscript_variables`` renders six significant figures, so the recorded value
#: can differ from the recomputed one by at most 5e-7 relatively. 1e-5 absorbs
#: that with two orders of margin and is still far tighter than any real
#: disagreement: the smallest genuine difference anywhere on this grid — the
#: located floor moving by one grid step — is a relative change of ~1e-2.
_ROUNDING_REL_TOL: float = 1e-5

#: The externally probed systems, ordered by measured captured mass, for
#: the ruler strip. Each captured mass is read from the live calibration arm's
#: own recorded claim; none of them is recomputed here.
_PROBES: tuple[tuple[str, str], ...] = (
    (PROBE_DISPLAY_NAMES["gemma3"], "BENCH_CALIB_CAPTURED_MASS_GEMMA3"),
    (PROBE_DISPLAY_NAMES["qwen25"], "BENCH_CALIB_CAPTURED_MASS_QWEN25"),
    (PROBE_DISPLAY_NAMES["smollm"], "BENCH_CALIB_CAPTURED_MASS_SMOLLM"),
)


@lru_cache(maxsize=2)
def _sweep_rows(
    positions_n: int,
    seed: int,
    bins: int,
    cap: int,
    grid: tuple[float, ...],
) -> tuple[dict[str, Any], ...]:
    """Re-run the recorder's offline sweep, memoized per parameter set.

    Every argument is passed in from a ledger token rather than defaulted, so
    the drawn grid is the recorded grid and a protocol change cannot leave the
    figure sweeping something else. The corpus is the vendored count artifact the
    recorder read.

    Memoized because the layout, legend and discovery gates each build every
    figure, and re-running an identical deterministic sweep four times buys
    nothing.
    """
    counts = load_brown_counts()
    pool, _unused = unique_letter_positions(LETTER_TEXT)
    positions = subsample_positions(pool, positions_n, seed)
    return tuple(completion_rule_sweep(counts, positions, grid=grid, bins=bins, cap=cap))


def _nearest_to_declared(
    rows: tuple[dict[str, Any], ...], declared: float
) -> tuple[int, dict[str, Any]]:
    """Index and row whose achieved captured mass sits closest to the declared floor.

    Mirrors ``_nearest_to_declared`` in the recorder, including its tie-breaking:
    ``min`` keeps the first of two rows at equal distance, and several grid points
    share a captured mass because *k* is an integer. Picking the row by requested
    target instead would name a point above the floor and quietly overstate how
    far the rules have converged there.
    """
    index = min(range(len(rows)), key=lambda i: abs(float(rows[i]["captured_mass"]) - declared))
    return index, rows[index]


def _sign_change_indices(values: list[float]) -> list[int]:
    """Indices at which a signed series changes sign, ignoring exact zeros.

    The same convention as :func:`ailm.benchmarks.divergence_sign_changes`, so the
    markers this figure draws are the crossings the ledger counted rather than a
    second, differently-defined set of them.
    """
    found: list[int] = []
    previous: int | None = None
    for index, value in enumerate(values):
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            continue
        if previous is not None and sign != previous:
            found.append(index)
        previous = sign
    return found


def _disagreements(
    checks: list[tuple[str, float, float]],
) -> list[str]:
    """Return a human-readable line for every recomputed value off its recorded one."""
    offenders = []
    for name, computed, recorded in checks:
        if recorded == 0.0:
            agrees = computed == 0.0
        else:
            agrees = abs(computed - recorded) <= _ROUNDING_REL_TOL * abs(recorded)
        if not agrees:
            offenders.append(f"{name}: recomputed {computed!r} vs recorded {recorded!r}")
    return offenders


def calibration_floor_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """The floor sweep, recomputed and held against every recorded summary claim.

    The ledger carries summary statistics of this sweep, not its per-grid-point
    series, so the series is recomputed here from the recorder's own functions on
    the recorder's own recorded parameters. Recomputation is only honest if it can
    be caught disagreeing, which is what the check below is: every recorded
    summary of the sweep is re-derived from the recomputed rows and compared.

    Raises:
        ValueError: If any re-derived quantity contradicts its recorded claim.
            The recorded values came from an execution of the same code on the
            same inputs, so a disagreement means the sweep, the shipped
            completion rule, or the recorded protocol has changed — a finding to
            investigate, not a figure to redraw. Drawing anyway would publish a
            panel that silently disagrees with the prose beside it.
        MissingVariableError: If a token this figure needs is not in the ledger.
    """
    v = _v(variables)

    declared = token_float(v, "BENCH_FLOOR_DECLARED_FLOOR")
    epsilon = token_float(v, "BENCH_FLOOR_EPSILON")
    cap = token_int(v, "BENCH_FLOOR_CAP")
    inventory = token_int(v, "BENCH_FLOOR_INVENTORY_SIZE")
    grid = calibration_floor_grid(
        token_float(v, "BENCH_FLOOR_GRID_LOW"),
        token_float(v, "BENCH_FLOOR_GRID_HIGH"),
        token_float(v, "BENCH_FLOOR_GRID_STEP"),
    )
    rows = _sweep_rows(
        token_int(v, "BENCH_FLOOR_POSITIONS"),
        token_int(v, "BENCH_FLOOR_SEED"),
        token_int(v, "BENCH_FLOOR_BINS"),
        cap,
        grid,
    )

    listed = list(rows)
    identity = rows[-1]
    joint = locate_divergence_floor(listed, "joint_divergence", epsilon)
    ece = locate_divergence_floor(listed, "ece_divergence", epsilon)
    brier = locate_divergence_floor(listed, "brier_divergence", epsilon)
    located = rows[int(joint["located_index"])]
    at_declared_index, at_declared = _nearest_to_declared(rows, declared)
    crossover = fidelity_crossover(listed)

    # ── The falsifier. Every recorded summary of this sweep, re-derived. ──────
    numeric: list[tuple[str, float, float]] = [
        ("grid low", float(rows[0]["target_mass"]), token_float(v, "BENCH_FLOOR_GRID_LOW")),
        ("grid high", float(identity["target_mass"]), token_float(v, "BENCH_FLOOR_GRID_HIGH")),
        # The identity control: at full capture truncation removes nothing, so
        # these are the live calibration arm's own AILM numbers.
        (
            "identity ECE",
            float(identity["min_bound_ece"]),
            token_float(v, "BENCH_FLOOR_IDENTITY_ECE"),
        ),
        (
            "identity Brier",
            float(identity["min_bound_brier"]),
            token_float(v, "BENCH_FLOOR_IDENTITY_BRIER"),
        ),
        (
            "largest rule TV",
            max(float(row["rule_tv"]) for row in rows),
            token_float(v, "BENCH_FLOOR_MAX_RULE_TV"),
        ),
        (
            "largest joint divergence",
            float(joint["max_divergence"]),
            token_float(v, "BENCH_FLOOR_MAX_JOINT_DIVERGENCE"),
        ),
        # The located floor and what reaching it costs.
        (
            "located target",
            float(located["target_mass"]),
            token_float(v, "BENCH_FLOOR_LOCATED_TARGET"),
        ),
        (
            "located captured mass",
            float(located["captured_mass"]),
            token_float(v, "BENCH_FLOOR_LOCATED_CAPTURED"),
        ),
        ("located mean k", float(located["mean_k"]), token_float(v, "BENCH_FLOOR_LOCATED_K")),
        (
            "located over-cap fraction",
            float(located["over_cap_fraction"]),
            token_float(v, "BENCH_FLOOR_LOCATED_OVER_CAP_FRACTION"),
        ),
        (
            "located rule TV",
            float(located["rule_tv"]),
            token_float(v, "BENCH_FLOOR_LOCATED_RULE_TV"),
        ),
        (
            "located / declared ratio",
            float(located["captured_mass"]) / declared,
            token_float(v, "BENCH_FLOOR_LOCATED_RATIO"),
        ),
        (
            "ECE located captured mass",
            float(ece["located_captured"]),
            token_float(v, "BENCH_FLOOR_LOCATED_ECE_CAPTURED"),
        ),
        (
            "Brier located captured mass",
            float(brier["located_captured"]),
            token_float(v, "BENCH_FLOOR_LOCATED_BRIER_CAPTURED"),
        ),
        # The trap: each metric's first crossing, and how far apart the two
        # reconstructions still are there.
        (
            "ECE first crossing",
            float(ece["naive_captured"]),
            token_float(v, "BENCH_FLOOR_NAIVE_ECE_CAPTURED"),
        ),
        (
            "ECE first crossing rule TV",
            float(rows[int(ece["naive_index"])]["rule_tv"]),
            token_float(v, "BENCH_FLOOR_NAIVE_ECE_RULE_TV"),
        ),
        (
            "Brier first crossing",
            float(brier["naive_captured"]),
            token_float(v, "BENCH_FLOOR_NAIVE_BRIER_CAPTURED"),
        ),
        # The row that stands in for a system sitting exactly at the gate.
        (
            "declared-floor captured mass",
            float(at_declared["captured_mass"]),
            token_float(v, "BENCH_FLOOR_ATDECLARED_CAPTURED"),
        ),
        (
            "declared-floor mean k",
            float(at_declared["mean_k"]),
            token_float(v, "BENCH_FLOOR_ATDECLARED_K"),
        ),
        (
            "declared-floor ECE divergence",
            float(at_declared["ece_divergence"]),
            token_float(v, "BENCH_FLOOR_ATDECLARED_ECE_DIVERGENCE"),
        ),
        (
            "declared-floor Brier divergence",
            float(at_declared["brier_divergence"]),
            token_float(v, "BENCH_FLOOR_ATDECLARED_BRIER_DIVERGENCE"),
        ),
        (
            "declared-floor rule TV",
            float(at_declared["rule_tv"]),
            token_float(v, "BENCH_FLOOR_ATDECLARED_RULE_TV"),
        ),
    ]
    offenders = _disagreements(numeric)

    # The identity control is held against the recorded *tolerance* rather than
    # against the recorded value. Both numbers are a log/exp round-trip residual
    # a few units in the last place wide, and the token map renders them to three
    # significant figures, so comparing them digit for digit would test the
    # renderer. What the control actually asserts is that truncation at full mass
    # removes nothing and the two rules therefore reduce to the same column.
    identity_tv = float(identity["rule_tv"])
    identity_tolerance = token_float(v, "BENCH_FLOOR_IDENTITY_TOLERANCE")
    if identity_tv > identity_tolerance:
        offenders.append(
            f"identity rule TV: recomputed {identity_tv!r}, above the recorded "
            f"tolerance {identity_tolerance!r} — truncation at full mass is no "
            f"longer the identity"
        )

    integral: list[tuple[str, int, int]] = [
        ("grid points", len(rows), token_int(v, "BENCH_FLOOR_GRID_POINTS")),
        (
            "ECE divergence rises",
            divergence_increases(listed, "ece_divergence"),
            token_int(v, "BENCH_FLOOR_ECE_INCREASES"),
        ),
        (
            "Brier divergence rises",
            divergence_increases(listed, "brier_divergence"),
            token_int(v, "BENCH_FLOOR_BRIER_INCREASES"),
        ),
        (
            "rule TV rises",
            divergence_increases(listed, "rule_tv"),
            token_int(v, "BENCH_FLOOR_TV_INCREASES"),
        ),
        (
            "ECE sign changes",
            divergence_sign_changes(listed, "ece_signed"),
            token_int(v, "BENCH_FLOOR_ECE_SIGN_CHANGES"),
        ),
        (
            "Brier sign changes",
            divergence_sign_changes(listed, "brier_signed"),
            token_int(v, "BENCH_FLOOR_BRIER_SIGN_CHANGES"),
        ),
        (
            "fidelity crossover exists",
            int(crossover is not None),
            token_int(v, "BENCH_FLOOR_FIDELITY_CROSSOVER_EXISTS"),
        ),
    ]
    offenders += [
        f"{name}: recomputed {computed} vs recorded {recorded}"
        for name, computed, recorded in integral
        if computed != recorded
    ]

    # The crossing markers this figure draws must be the crossings the ledger
    # counted. Drawing a different number of them would put a claim on the axis
    # that the recorded sign-change count denies.
    ece_crossings = _sign_change_indices([float(row["ece_signed"]) for row in rows])
    brier_crossings = _sign_change_indices([float(row["brier_signed"]) for row in rows])
    for label, drawn, token in (
        ("ECE", ece_crossings, "BENCH_FLOOR_ECE_SIGN_CHANGES"),
        ("Brier", brier_crossings, "BENCH_FLOOR_BRIER_SIGN_CHANGES"),
    ):
        if len(drawn) != token_int(v, token):
            offenders.append(
                f"{label} zero crossings drawn: {len(drawn)} vs recorded {token_int(v, token)}"
            )

    if offenders:
        listing = "; ".join(offenders)
        raise ValueError(
            f"the recomputed captured-mass floor sweep contradicts the ledger in "
            f"{len(offenders)} place(s): {listing}. This figure recomputes the sweep "
            f"from the same offline functions the recorder used, so a disagreement "
            f"means the sweep, a completion rule, or the recorded protocol has "
            f"changed — re-run scripts/record_calibration_floor_claims.py and find "
            f"out which. Refusing to draw."
        )

    return {
        "rows": listed,
        "captured": [float(row["captured_mass"]) for row in rows],
        "declared": declared,
        "epsilon": epsilon,
        "cap": cap,
        "inventory": inventory,
        "positions": token_int(v, "BENCH_FLOOR_POSITIONS"),
        "grid_points": len(rows),
        "located": located,
        "located_index": int(joint["located_index"]),
        "at_declared": at_declared,
        "at_declared_index": at_declared_index,
        "ece_first_crossing": rows[int(ece["naive_index"])],
        "brier_first_crossing": rows[int(brier["naive_index"])],
        "ece_crossings": ece_crossings,
        "brier_crossings": brier_crossings,
        "ailm_ece": token_float(v, "BENCH_CALIB_ECE_AILM"),
        "probes": [(name, token_float(v, token)) for name, token in _PROBES],
    }


# ── Panels ───────────────────────────────────────────────────────────────────


def _floor_markers(ax, data: dict[str, Any], *, label: bool) -> None:
    """Draw the declared and located floors as vertical rules on one panel."""
    declared = data["declared"]
    located = float(data["located"]["captured_mass"])
    ax.axvline(
        declared,
        color=COLOR_DECLARED,
        linestyle="--",
        linewidth=1.6,
        label=f"declared floor {declared:.2f}" if label else "_nolegend_",
    )
    ax.axvline(
        located,
        color=COLOR_LOCATED,
        linestyle="-.",
        linewidth=1.6,
        label=f"located floor {located:.3f}" if label else "_nolegend_",
    )


def _draw_signed(ax, data: dict[str, Any]) -> None:
    """Signed rule difference per metric, with the distributional gap behind it."""
    rows = data["rows"]
    captured = data["captured"]
    ax.axhline(0.0, color=COLOR_INK, linewidth=1.0)
    _floor_markers(ax, data, label=True)

    ax.plot(
        captured,
        [float(row["ece_signed"]) for row in rows],
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        color=COLOR_ECE,
        label="ECE: min-bound - observed-only",
    )
    ax.plot(
        captured,
        [float(row["brier_signed"]) for row in rows],
        marker="s",
        markersize=3.2,
        linewidth=1.1,
        color=COLOR_BRIER,
        label="Brier: min-bound - observed-only",
    )
    for index in data["ece_crossings"] + data["brier_crossings"]:
        ax.plot(
            captured[index],
            0.0,
            marker="D",
            markersize=9,
            markerfacecolor=COLOR_TRAP,
            markeredgecolor=COLOR_INK,
            linestyle="none",
            zorder=5,
        )

    # Total variation shares the panel because the coincidence is the finding:
    # the metric gaps vanish exactly where the distributions have not converged
    # at all. On separate panels a reader has to hold one curve in memory to see
    # it. The right axis is scaled so both series clear the upper third, which is
    # where the legend and the callout live.
    gap_ax = ax.twinx()
    gap_ax.plot(
        captured,
        [float(row["rule_tv"]) for row in rows],
        marker="^",
        markersize=3.0,
        linewidth=1.1,
        color=COLOR_TV,
        label="total variation between the two reconstructions",
    )
    gap_ax.set_ylim(0.0, 1.90)
    gap_ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    gap_ax.set_ylabel("Total variation", fontsize=FONTS["axis_label"], color=COLOR_TV)
    gap_ax.tick_params(axis="y", colors=COLOR_TV, labelsize=FONTS["tick_label"])
    gap_ax.spines["right"].set_visible(True)
    gap_ax.spines["right"].set_color(COLOR_TV)

    crossing = data["ece_first_crossing"]
    ax.annotate(
        f"both signed gaps cross zero once, here —\n"
        f"where the two reconstructions are still\n"
        f"{float(crossing['rule_tv']):.2f} apart in total variation",
        xy=(float(crossing["captured_mass"]), 0.0),
        # The band under both signed curves and under the descending total
        # variation is the only region of this panel with no data in it.
        xytext=(0.28, 0.045),
        textcoords="axes fraction",
        # The declared-floor rule passes behind this callout. An opaque card
        # keeps the words readable while the arrow still points at the crossing.
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0},
        fontsize=FONTS["annotation"],
        color=COLOR_INK,
        arrowprops={"arrowstyle": "->", "color": COLOR_INK, "linewidth": 1.2},
    )

    ax.set_xlim(0.22, 1.03)
    ax.set_ylim(-0.47, 0.62)
    ax.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0, 0.1])
    ax.set_title(
        "Signed rule difference: the crossing that traps a naive locator",
        fontsize=FONTS["title"],
    )
    ax.set_xlabel("Mean captured mass", fontsize=FONTS["axis_label"])
    ax.set_ylabel("Metric under min-bound - under observed-only", fontsize=FONTS["axis_label"])
    handles, labels = ax.get_legend_handles_labels()
    gap_handles, gap_labels = gap_ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            marker="D",
            markersize=8,
            linestyle="none",
            markerfacecolor=COLOR_TRAP,
            markeredgecolor=COLOR_INK,
        )
    )
    labels.append("zero crossing")
    ax.legend(
        handles + gap_handles,
        labels + gap_labels,
        fontsize=FONTS["legend"],
        loc="upper left",
        framealpha=1.0,
    )


def _draw_absolute(ax, data: dict[str, Any]) -> None:
    """Absolute divergence on a log axis, against the declared threshold."""
    rows = data["rows"]
    captured = data["captured"]
    epsilon = data["epsilon"]
    first = float(data["ece_first_crossing"]["captured_mass"])
    located = float(data["located"]["captured_mass"])

    ax.axvspan(
        first,
        located,
        color=COLOR_TRAP,
        alpha=0.30,
        label="crossed once, then diverged again",
    )
    _floor_markers(ax, data, label=False)
    ax.axhline(
        epsilon,
        color=COLOR_INK,
        linestyle=":",
        linewidth=1.6,
        label=f"declared threshold {epsilon:g}",
    )
    ax.semilogy(
        captured,
        [float(row["ece_divergence"]) for row in rows],
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        color=COLOR_ECE,
        label="|ECE divergence|",
    )
    ax.semilogy(
        captured,
        [float(row["brier_divergence"]) for row in rows],
        marker="s",
        markersize=3.2,
        linewidth=1.1,
        color=COLOR_BRIER,
        label="|Brier divergence|",
    )

    ax.set_xlim(0.22, 1.03)
    # The floor of the axis is far below the deepest notch on purpose: it gives
    # the legend a home under the data rather than over it, and it shows how far
    # a genuinely converged point would have to fall.
    ax.set_ylim(1e-6, 3.0)
    ax.set_title(
        "Each metric dips below the threshold long before it stays there",
        fontsize=FONTS["title"],
    )
    ax.set_xlabel("Mean captured mass", fontsize=FONTS["axis_label"])
    ax.set_ylabel("Absolute rule divergence", fontsize=FONTS["axis_label"])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="v", markersize=7, linestyle="none", color=COLOR_MUTED))
    labels.append("identity point (mass 1.00): exactly 0, off a log axis")
    ax.legend(handles, labels, fontsize=FONTS["legend"], loc="lower left", framealpha=1.0)


def _draw_reachable(ax, data: dict[str, Any]) -> None:
    """Candidates per position needed, against the endpoint's reporting cap."""
    rows = data["rows"]
    captured = data["captured"]
    cap = data["cap"]
    located = data["located"]

    _floor_markers(ax, data, label=False)
    ax.plot(
        captured,
        [float(row["mean_k"]) for row in rows],
        marker="o",
        markersize=3.2,
        linewidth=1.1,
        color=COLOR_ECE,
        label="mean candidates needed per position",
    )
    ax.axhline(
        float(cap),
        color=COLOR_DECLARED,
        linestyle=":",
        linewidth=1.8,
        label=f"server candidate cap ({cap})",
    )
    ax.axhline(
        float(data["inventory"]),
        color=COLOR_MUTED,
        linestyle=(0, (1, 3)),
        linewidth=1.8,
        label=f"scored inventory ({data['inventory']} symbols)",
    )

    # A mean under the cap is not reachability: the share of individual positions
    # that need more than the cap is what decides whether the endpoint can be
    # asked for this floor, so it is drawn rather than left to the caption.
    over_ax = ax.twinx()
    over_ax.fill_between(
        captured,
        [float(row["over_cap_fraction"]) for row in rows],
        step="pre",
        color=COLOR_OBSERVED,
        alpha=0.30,
        linewidth=0.0,
        label="share of positions needing more than the cap",
    )
    over_ax.set_ylim(0.0, 1.85)
    over_ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    over_ax.set_ylabel(
        "Share of positions over the cap", fontsize=FONTS["axis_label"], color=COLOR_OBSERVED
    )
    over_ax.tick_params(axis="y", colors=COLOR_OBSERVED, labelsize=FONTS["tick_label"])
    over_ax.spines["right"].set_visible(True)
    over_ax.spines["right"].set_color(COLOR_OBSERVED)

    ax.annotate(
        f"the located floor needs {float(located['mean_k']):.1f} candidates on\n"
        f"average and puts {float(located['over_cap_fraction']):.1%} of positions\n"
        f"over the cap",
        xy=(float(located["captured_mass"]), float(located["mean_k"])),
        xytext=(0.30, 0.52),
        textcoords="axes fraction",
        fontsize=FONTS["annotation"],
        color=COLOR_INK,
        arrowprops={"arrowstyle": "->", "color": COLOR_INK, "linewidth": 1.2},
        # The dotted server-cap rule this callout is about ran through its last
        # line, "over the cap". The patch masks the rule behind the glyphs.
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": "none",
        },
        zorder=7,
    )

    ax.set_xlim(0.22, 1.03)
    ax.set_ylim(0.0, 40.0)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_title(
        "Reaching the located floor costs more candidates than the endpoint returns",
        fontsize=FONTS["title"],
    )
    ax.set_xlabel("Mean captured mass", fontsize=FONTS["axis_label"])
    ax.set_ylabel("Candidates per position", fontsize=FONTS["axis_label"])
    handles, labels = ax.get_legend_handles_labels()
    over_handles, over_labels = over_ax.get_legend_handles_labels()
    ax.legend(
        handles + over_handles,
        labels + over_labels,
        fontsize=FONTS["legend"],
        loc="upper left",
        framealpha=1.0,
    )


def _draw_ruler(ax, data: dict[str, Any]) -> None:
    """One captured-mass axis carrying both floors, both crossings and the probes.

    No legend: every mark on the strip is labeled where it sits, which is the
    only arrangement that fits without a legend box landing on the band or on the
    probe nearest total capture.
    """
    declared = data["declared"]
    located = float(data["located"]["captured_mass"])
    ax.axvspan(declared, located, color=COLOR_TRAP, alpha=0.30)
    ax.axhline(0.0, color=COLOR_MUTED, linewidth=1.2)
    _floor_markers(ax, data, label=False)

    ax.text(
        (declared + located) / 2.0,
        1.02,
        f"between the declared floor ({declared:.2f}) and the located floor ({located:.3f}):\n"
        f"admitted by the gate, and the band a first-crossing locator would have certified",
        ha="center",
        va="bottom",
        fontsize=FONTS["annotation"],
        color=COLOR_INK,
    )
    # Every label on this number line is anchored at the captured mass it
    # reports, so a label naming a floor sits on that floor's own rule and the
    # labels either side of the band straddle it: the declared rule was drawn
    # through "declared floor 0.50" and "first Brier crossing 0.496", the located
    # rule through "located floor 0.991" and the Qwen probe. The x anchor is the
    # datum and cannot move, so each patch masks the rule behind its glyphs — the
    # remedy the subsample panel already uses. A rule crossing a word is
    # invisible to a text-versus-text overlap sweep.
    ax.text(
        declared,
        -1.06,
        f"declared floor {declared:.2f}",
        ha="center",
        va="bottom",
        fontsize=FONTS["annotation"],
        color=COLOR_DECLARED,
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": "none",
        },
        zorder=7,
    )
    ax.text(
        located,
        -1.06,
        f"located floor {located:.3f}",
        ha="right",
        va="bottom",
        fontsize=FONTS["annotation"],
        color=COLOR_LOCATED,
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": "none",
        },
        zorder=7,
    )

    for label, key, offset, align in (
        ("first ECE crossing", "ece_first_crossing", 0.42, "right"),
        ("first Brier crossing", "brier_first_crossing", -0.52, "left"),
    ):
        mass = float(data[key]["captured_mass"])
        ax.plot(
            mass,
            0.0,
            marker="D",
            markersize=9,
            markerfacecolor=COLOR_TRAP,
            markeredgecolor=COLOR_INK,
            linestyle="none",
            zorder=5,
        )
        ax.annotate(
            f"{label} {mass:.3f}",
            xy=(mass, 0.0),
            xytext=(mass, offset),
            fontsize=FONTS["annotation"],
            ha=align,
            va="bottom" if offset > 0 else "top",
            color=COLOR_INK,
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
            zorder=7,
        )

    # Placement is derived from the probe count rather than written out per probe.
    # Hardcoding one row per probe made the panel break the moment a model was
    # removed from the benchmark: the label positions, not the data, carried the
    # assumption that there were exactly three.
    _placements = [(0.42, "left"), (-0.52, "center"), (0.42, "right")]
    for index, (name, mass) in enumerate(data["probes"]):
        height, align = _placements[index % len(_placements)]
        ax.plot(
            mass,
            0.0,
            marker="o",
            markersize=10,
            markerfacecolor=COLOR_ECE,
            markeredgecolor=COLOR_INK,
            linestyle="none",
            zorder=6,
        )
        ax.annotate(
            f"{name}, measured probe, {mass:.3f}",
            xy=(mass, 0.0),
            xytext=(mass, height),
            fontsize=FONTS["annotation"],
            ha=align,
            va="bottom" if height > 0 else "top",
            color=COLOR_INK,
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
            zorder=7,
        )

    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-1.55, 1.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(
        "The gate, the located floor, and where the probes sit",
        fontsize=FONTS["title"],
    )
    ax.set_xlabel("Mean captured answer-slot mass", fontsize=FONTS["axis_label"])


def build_calibration_floor_figure(variables: dict[str, str] | None = None) -> Figure:
    """Assemble the four-panel captured-mass floor figure.

    The gridspec margins are set explicitly rather than by ``tight_layout``:
    two of the four panels carry a twin axis, which ``tight_layout`` cannot
    measure and warns about, and an unmeasured auto-layout is not a layout.
    The reachable panel spans the full width on its own row rather than
    sharing one with a fifth panel, which is the room it needs for its own
    twin axis, callout and legend — the fifth panel this figure used to carry
    (the rule-decided share of a published ECE) is dropped to make that room;
    its one number is already stated in prose next to this figure's caption.
    """
    apply_style()
    data = calibration_floor_data(variables)

    fig = plt.figure(figsize=(18.0, 14.5))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 0.95, 0.55],
        hspace=0.32,
        wspace=0.26,
        left=0.055,
        right=0.945,
        top=0.935,
        bottom=0.045,
    )
    _draw_signed(fig.add_subplot(grid[0, 0]), data)
    _draw_absolute(fig.add_subplot(grid[0, 1]), data)
    _draw_reachable(fig.add_subplot(grid[1, :]), data)
    _draw_ruler(fig.add_subplot(grid[2, :]), data)

    fig.suptitle(
        f"Locating the captured-mass floor offline: declared {data['declared']:.2f}, "
        f"located {float(data['located']['captured_mass']):.3f} "
        f"({data['positions']} AILM letter positions, {data['grid_points']} grid points)",
        fontsize=FONTS["suptitle"],
    )
    return fig


def generate_calibration_floor(variables: dict[str, str] | None = None) -> Figure:
    """Generate the captured-mass floor figure; the pipeline owns saving."""
    return build_calibration_floor_figure(variables)
