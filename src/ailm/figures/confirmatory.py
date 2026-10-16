"""The confirmatory replication, drawn: development beside held-back.

Decision 111 registered a confirmatory replication of the four significance-family
members that map to the ``ud_ewt_test`` population, and decision 115 registered
the Level 6 document-clustered estimand on that same held-back split. Both were
recorded in full and drawn nowhere; decision 116 commissions this figure and
fixes what it asserts.

``confirmatory_replication`` (top panel)
    Each scored member's exact two-sided p on a log axis, development population
    beside held-back, against the frozen per-test threshold and that member's own
    Holm step-down threshold. A verdict badge per member, read from the recorded
    crossing and Holm booleans, never from which side of a line the marker
    happens to land on. The registered positive and negative controls are drawn
    on the same axis beneath a rule, because a replication whose controls are not
    visible is a claim the reader cannot audit. A p below ``_AXIS_FLOOR`` is
    compressed to the floor and marked with a small left-pointing triangle rather
    than left to stretch the shared log axis across dozens of extra decades for
    one or two points while the frozen threshold, both Holm thresholds and every
    other p — all within nine orders of magnitude of 1 — are squeezed into a
    sliver on the right; the exact value is written small beside the clamped
    marker instead, so the compression costs nothing the panel used to say.

``confirmatory_replication`` (middle panel)
    The discordant counts each population produced, favored against opposed, with
    the evaluated population and its unit beside them. Rows with no discordant
    pair retain their zero counts; the top panel identifies their comparisons
    as not computable rather than treating a p of one as a measured near-null.

``confirmatory_replication`` (bottom panel)
    Level 6's document-clustered mean improvement on both populations. Three
    things would make this panel misleading if left out, and each is drawn: the
    two populations disagree, so the development interval is drawn beside the
    held-back one and is not retracted; the corrected-coverage interval spans
    zero while the primary does not, so both are drawn with their coverage named;
    and the held-back arm's minimum detectable effect is coarser than the
    development effect it sits beside, so that resolution is drawn as its own
    mark rather than left to be inferred.

Every number comes from the ledger through the token map. Every verdict word is
read from a recorded boolean, per decision 116, and the data function refuses to
draw when a drawn geometry and a recorded boolean disagree.

Every generator returns an unwritten ``Figure``; ``scripts/generate_figures.py``
discovers them by the ``generate_`` prefix and handles saving.
"""

from __future__ import annotations

from typing import Any, NamedTuple, NotRequired, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ailm.figures._common import (
    MissingVariableError,
    token_float,
    token_int,
)
from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.statistics import (
    CONFIRMATORY_REPLICATION,
    FAMILYWISE_ALPHA,
    SIGNIFICANCE_FAMILY,
)
from ailm.viz_config import FIGSIZES, FONTS, PALETTE, apply_style

__all__ = [
    "confirmatory_replication_data",
    "generate_confirmatory_replication",
]

COLOR_DEV = PALETTE["UNIGRAM"]  # the development population
COLOR_HELD = PALETTE["MODEL"]  # the held-back population
COLOR_INK = PALETTE["INK"]
COLOR_MUTED = PALETTE["MUTED"]
COLOR_GATE = PALETTE["GATE"]  # the frozen per-test threshold
COLOR_ACCENT = PALETTE["ACCENT"]  # Holm's per-member threshold
COLOR_FLOOR = PALETTE["FLOOR"]  # the arm's own resolution

#: The held-back population the protocol registers for these members. Method
#: structure from :data:`ailm.statistics.CONFIRMATORY_REPLICATION`, not a
#: measurement.
_HELD_BACK_POPULATION: str = "ud_ewt_test"


class _ReplicationControl(TypedDict):
    """Recorded control inputs, with a seed only for the shuffled control."""

    label: str
    p: float
    favored: int
    opposed: int
    seed: NotRequired[int]


class _Member(NamedTuple):
    """One replicated family member: where each population's numbers live.

    Every field names a token or a declared family member. None of them is a
    measured value; the values are read at draw time.
    """

    label: str
    family_label: str
    unit: str
    dev_p: str
    dev_favored: str
    dev_opposed: str
    dev_population: str
    dev_documents: str
    held: str
    held_population: str
    held_documents: str


_MEMBERS: tuple[_Member, ...] = (
    _Member(
        "L4 derived cases",
        "stage4_derived_mcnemar_p",
        "tokens",
        "STAGE4_DERIVED_MCNEMAR_P",
        "STAGE4_DERIVED_MCNEMAR_B",
        "STAGE4_DERIVED_MCNEMAR_C",
        "STAGE4_DERIVED_TEST_TOKENS",
        "STAGE4_DERIVED_TEST_DOCUMENTS",
        "CONFIRMATORY_STAGE4_DERIVED_MCNEMAR",
        "CONFIRMATORY_STAGE4_DERIVED_POSITIONS",
        "CONFIRMATORY_STAGE4_DERIVED_DOCUMENTS",
    ),
    _Member(
        "L4 gold morphology",
        "stage4_gold_mcnemar_p",
        "tokens",
        "STAGE4_GOLD_MCNEMAR_P",
        "STAGE4_GOLD_MCNEMAR_B",
        "STAGE4_GOLD_MCNEMAR_C",
        "STAGE4_GOLD_TEST_TOKENS",
        "STAGE4_GOLD_TEST_DOCUMENTS",
        "CONFIRMATORY_STAGE4_GOLD_MCNEMAR",
        "CONFIRMATORY_STAGE4_GOLD_POSITIONS",
        "CONFIRMATORY_STAGE4_GOLD_DOCUMENTS",
    ),
    _Member(
        "L5 case-only clauses",
        "stage5_mcnemar_p",
        "sentences",
        "STAGE5_MCNEMAR_P",
        "STAGE5_MCNEMAR_B",
        "STAGE5_MCNEMAR_C",
        "STAGE5_TEST_SENTENCES",
        "STAGE5_TEST_DOCUMENTS",
        "CONFIRMATORY_STAGE5_MCNEMAR",
        "CONFIRMATORY_STAGE5_SENTENCES",
        "CONFIRMATORY_STAGE5_DOCUMENTS",
    ),
    _Member(
        "L6 document position",
        "stage6_sign_test_p",
        "sentences",
        "STAGE6_SIGN_TEST_P",
        "STAGE6_SIGN_TEST_POSITIVE",
        "STAGE6_SIGN_TEST_NEGATIVE",
        "STAGE6_TEST_SENTENCES",
        "STAGE6_TEST_DOCUMENTS",
        "CONFIRMATORY_STAGE6_SIGN_TEST",
        "CONFIRMATORY_STAGE6_SENTENCES",
        "CONFIRMATORY_STAGE6_DOCUMENTS",
    ),
)

# A member the protocol does not map to the held-back population would be drawn
# under a replication it was never registered for. Fail at import.
_REGISTERED = {
    member
    for member, disposition in CONFIRMATORY_REPLICATION
    if disposition == _HELD_BACK_POPULATION
}
if {member.family_label for member in _MEMBERS} != _REGISTERED:
    raise RuntimeError(
        "the drawn members must be exactly the family members the confirmatory "
        f"protocol maps to {_HELD_BACK_POPULATION}; drawn "
        f"{sorted(m.family_label for m in _MEMBERS)}, registered {sorted(_REGISTERED)}"
    )


def _token_bool(variables: dict[str, str], token: str) -> bool:
    """Read a boolean token, rejecting anything but the two rendered forms."""
    if token not in variables:
        raise MissingVariableError(f"Token {token!r} is not in the manuscript variable map.")
    raw = variables[token].strip().lower()
    if raw not in ("true", "false"):
        raise MissingVariableError(f"Token {token!r} has non-boolean value {variables[token]!r}.")
    return raw == "true"


def _token_direction(variables: dict[str, str], token: str) -> str | None:
    """Read a direction flag that is genuinely three-valued.

    A member with no discordant pair has no direction to agree or disagree with,
    and the recorder writes ``undefined`` rather than forcing a false. Collapsing
    that to ``False`` would draw "direction disagrees" on an arm where nothing
    was measured.
    """
    if token not in variables:
        raise MissingVariableError(f"Token {token!r} is not in the manuscript variable map.")
    raw = variables[token].strip().lower()
    if raw == "undefined":
        return None
    if raw not in ("true", "false"):
        raise MissingVariableError(
            f"Token {token!r} is neither a boolean nor 'undefined': {variables[token]!r}."
        )
    return raw


def confirmatory_replication_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """The replication rows, cross-checked against the recorded booleans.

    Raises:
        MissingVariableError: If a required token is absent or malformed.
        ValueError: If the frozen threshold is not the familywise level over the
            frozen family size, or that size has drifted from the executable
            declaration; if a member's drawn p-versus-threshold geometry
            contradicts its recorded crossing or Holm boolean; if a computability
            flag disagrees with whether any discordant pair exists; if favored
            and opposed do not sum to the recorded discordant total, or that
            total exceeds the evaluated population; if the number of drawn
            members disagrees with the recorded count of scored members; if a
            registered control lands on the wrong side of the frozen threshold;
            or if a recorded exclusion boolean disagrees with the interval it
            describes.
    """
    v = _v(variables)

    frozen_alpha = token_float(v, "CONFIRMATORY_FROZEN_ALPHA")
    frozen_family_size = token_int(v, "CONFIRMATORY_FROZEN_FAMILY_SIZE")
    if frozen_family_size != len(SIGNIFICANCE_FAMILY):
        raise ValueError(
            f"the replication froze a family of {frozen_family_size} but "
            f"SIGNIFICANCE_FAMILY declares {len(SIGNIFICANCE_FAMILY)}; the frozen "
            "threshold no longer corresponds to the family it corrects."
        )
    expected_alpha = FAMILYWISE_ALPHA / frozen_family_size
    if abs(frozen_alpha - expected_alpha) > expected_alpha * 1e-2:
        raise ValueError(
            f"the frozen per-test alpha {frozen_alpha} is not the familywise "
            f"{FAMILYWISE_ALPHA} over the frozen family size {frozen_family_size}."
        )
    if not _token_bool(v, f"CONFIRMATORY_RUN_COMPLETED_{_HELD_BACK_POPULATION.upper()}"):
        raise ValueError(
            f"the {_HELD_BACK_POPULATION} run is not recorded as completed; refusing to "
            "draw a replication that did not happen."
        )

    rows: list[dict[str, Any]] = []
    for member in _MEMBERS:
        held: dict[str, Any] = {
            "p": token_float(v, f"{member.held}_P"),
            "favored": token_int(v, f"{member.held}_FAVORED"),
            "opposed": token_int(v, f"{member.held}_OPPOSED"),
            "discordant": token_int(v, f"{member.held}_DISCORDANT_N"),
            "population": token_int(v, member.held_population),
            "documents": token_int(v, member.held_documents),
            "holm_threshold": token_float(v, f"{member.held}_HOLM_THRESHOLD"),
            "holm_rejected": _token_bool(v, f"{member.held}_HOLM_REJECTED"),
            "holm_rank": token_int(v, f"{member.held}_HOLM_RANK"),
            "crosses": _token_bool(v, f"{member.held}_CROSSES_FROZEN_THRESHOLD"),
            "computable": _token_bool(v, f"{member.held}_COMPUTABLE"),
            "direction": _token_direction(v, f"{member.held}_DIRECTION_AGREES"),
        }
        development: dict[str, Any] = {
            "p": token_float(v, member.dev_p),
            "favored": token_int(v, member.dev_favored),
            "opposed": token_int(v, member.dev_opposed),
            "population": token_int(v, member.dev_population),
            "documents": token_int(v, member.dev_documents),
        }
        development["discordant"] = development["favored"] + development["opposed"]

        for name, arm in (("held-back", held), ("development", development)):
            if arm["favored"] + arm["opposed"] != arm["discordant"]:
                raise ValueError(
                    f"{member.label} ({name}): {arm['favored']} favored and "
                    f"{arm['opposed']} opposed do not sum to the recorded discordant "
                    f"total {arm['discordant']}; the counts came from different runs."
                )
            if arm["discordant"] > arm["population"]:
                raise ValueError(
                    f"{member.label} ({name}): {arm['discordant']} discordant pairs over "
                    f"{arm['population']} evaluated {member.unit} is impossible."
                )

        # Every verdict word this row draws, bound to the geometry beside it.
        if held["computable"] != (held["discordant"] > 0):
            raise ValueError(
                f"{member.label}: the computability flag is {held['computable']} but the "
                f"arm produced {held['discordant']} discordant pairs. A conditional test "
                "is computable exactly when a discordant pair exists."
            )
        if (held["p"] < frozen_alpha) != held["crosses"]:
            raise ValueError(
                f"{member.label}: the drawn geometry (held-back p {held['p']} against the "
                f"frozen alpha {frozen_alpha}) contradicts the recorded crossing flag "
                f"{held['crosses']}. Refusing to draw a crossing the ledger denies."
            )
        if (held["p"] < held["holm_threshold"]) != held["holm_rejected"]:
            raise ValueError(
                f"{member.label}: the drawn geometry (held-back p {held['p']} against its "
                f"Holm threshold {held['holm_threshold']}) contradicts the recorded Holm "
                f"verdict {held['holm_rejected']}."
            )
        if held["crosses"] and held["direction"] != "true":
            raise ValueError(
                f"{member.label}: the member is recorded as crossing the frozen threshold "
                f"while its direction flag is {held['direction']!r}. A crossing in the "
                "unregistered direction is not a replication and must not be badged as one."
            )
        rows.append({"member": member, "development": development, "held": held})

    scored = token_int(v, "CONFIRMATORY_MEMBERS_SCORED")
    if len(rows) != scored:
        raise ValueError(
            f"{len(rows)} members are drawn but the ledger records {scored} scored; "
            "the figure and the recorder disagree about what was replicated."
        )

    # ── The registered controls ─────────────────────────────────────────────
    positive: _ReplicationControl = {
        "label": "positive control",
        "p": token_float(v, "CONFIRMATORY_CONTROL_POSITIVE_P"),
        "favored": token_int(v, "CONFIRMATORY_CONTROL_POSITIVE_FAVORED"),
        "opposed": token_int(v, "CONFIRMATORY_CONTROL_POSITIVE_OPPOSED"),
    }
    negative: _ReplicationControl = {
        "label": "negative control",
        "p": token_float(v, "CONFIRMATORY_CONTROL_NEGATIVE_P"),
        "favored": token_int(v, "CONFIRMATORY_CONTROL_NEGATIVE_FAVORED"),
        "opposed": token_int(v, "CONFIRMATORY_CONTROL_NEGATIVE_OPPOSED"),
        "seed": token_int(v, "CONFIRMATORY_CONTROL_NEGATIVE_SEED"),
    }
    if not positive["p"] < frozen_alpha:
        raise ValueError(
            f"the positive control's p {positive['p']} does not clear the frozen alpha "
            f"{frozen_alpha}. The replication apparatus cannot detect an effect it was "
            "built to detect; nothing else on this figure can be trusted."
        )
    if negative["p"] < frozen_alpha:
        raise ValueError(
            f"the negative control's p {negative['p']} clears the frozen alpha "
            f"{frozen_alpha}. The apparatus reports an effect where none was planted."
        )

    # ── Level 6's clustered estimand on both populations (decision 115) ──────
    clustered = {
        "held_effect": token_float(v, "CONFIRMATORY_STAGE6_CLUSTERED_MEAN_IMPROVEMENT_NATS"),
        "held_low": token_float(v, "CONFIRMATORY_STAGE6_CLUSTERED_CI_LOW"),
        "held_high": token_float(v, "CONFIRMATORY_STAGE6_CLUSTERED_CI_HIGH"),
        "held_excludes_zero": _token_bool(v, "CONFIRMATORY_STAGE6_CLUSTERED_EXCLUDES_ZERO"),
        "corrected_low": token_float(v, "CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CI_LOW"),
        "corrected_high": token_float(v, "CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CI_HIGH"),
        "corrected_excludes_zero": _token_bool(
            v, "CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_EXCLUDES_ZERO"
        ),
        "min_detectable": token_float(
            v, "CONFIRMATORY_STAGE6_CLUSTERED_MIN_DETECTABLE_EFFECT_NATS"
        ),
        "development_effect_resolved": _token_bool(
            v, "CONFIRMATORY_STAGE6_CLUSTERED_DEVELOPMENT_EFFECT_RESOLVED"
        ),
        "held_documents": token_int(v, "CONFIRMATORY_STAGE6_CLUSTERED_DOCUMENTS"),
        "held_positive_documents": token_int(v, "CONFIRMATORY_STAGE6_CLUSTERED_POSITIVE_DOCUMENTS"),
        "held_negative_documents": token_int(v, "CONFIRMATORY_STAGE6_CLUSTERED_NEGATIVE_DOCUMENTS"),
        "dev_effect": token_float(v, "CLUSTER_STAGE6_LOGLOSS_EFFECT"),
        "dev_low": token_float(v, "CLUSTER_STAGE6_LOGLOSS_CI_LOW"),
        "dev_high": token_float(v, "CLUSTER_STAGE6_LOGLOSS_CI_HIGH"),
        "dev_documents": token_int(v, "CLUSTER_STAGE6_LOGLOSS_CLUSTERS"),
        "confidence": token_float(v, "CLUSTER_CONFIDENCE"),
        "held_confidence": token_float(v, "CONFIRMATORY_STAGE6_CLUSTERED_CONFIDENCE"),
        "corrected_confidence": token_float(
            v, "CONFIRMATORY_STAGE6_CLUSTERED_CORRECTED_CONFIDENCE"
        ),
    }
    for name, low, flag in (
        ("primary", clustered["held_low"], clustered["held_excludes_zero"]),
        ("corrected", clustered["corrected_low"], clustered["corrected_excludes_zero"]),
    ):
        if (low > 0.0) != flag:
            raise ValueError(
                f"Level 6 clustered {name} interval: the recorded excludes-zero flag is "
                f"{flag} but the drawn lower bound is {low}. Refusing to draw an exclusion "
                "the interval does not show."
            )
    if clustered["corrected_low"] > clustered["held_low"]:
        raise ValueError(
            "the corrected-coverage interval's lower bound "
            f"{clustered['corrected_low']} is above the primary's "
            f"{clustered['held_low']}, which wider coverage makes impossible."
        )
    # The awkward fact the panel exists to state plainly: the held-back arm's own
    # resolution is coarser than the development effect it sits beside.
    resolved = clustered["dev_effect"] >= clustered["min_detectable"]
    if resolved != clustered["development_effect_resolved"]:
        raise ValueError(
            f"the development effect {clustered['dev_effect']} against the held-back "
            f"arm's minimum detectable effect {clustered['min_detectable']} gives "
            f"resolved = {resolved}, but the ledger records "
            f"{clustered['development_effect_resolved']}."
        )
    if clustered["dev_low"] > 0.0 or clustered["dev_high"] < 0.0:
        raise ValueError(
            "the development clustered interval no longer spans zero; the two "
            "populations no longer disagree in the stated way and this panel's subject has changed."
        )

    return {
        "rows": rows,
        "frozen_alpha": frozen_alpha,
        "frozen_family_size": frozen_family_size,
        "population": v["CONFIRMATORY_POPULATION"].strip(),
        "census_documents": token_int(v, "CONFIRMATORY_CENSUS_DOCUMENTS"),
        "census_sentences": token_int(v, "CONFIRMATORY_CENSUS_SENTENCES"),
        "controls": (positive, negative),
        "clustered": clustered,
        "registration_decision": token_int(v, "CONFIRMATORY_REGISTRATION_DECISION"),
    }


def _p_floor(values: list[float]) -> float:
    """Left edge for the log p-axis, a decade below the smallest drawn p."""
    smallest = min(value for value in values if value > 0.0)
    return smallest / 12.0


#: Below this, a p is compressed to the axis edge rather than left to stretch the
#: log axis across dozens of extra decades for one or two points. On the live
#: ledger the frozen threshold, every Holm threshold, every development p and
#: the negative control all sit within nine orders of magnitude of 1, while the
#: positive control and two held-back replications sit 20-70 orders further
#: left — decades the reader gains nothing from and that leave the crowded
#: cluster near 1 unreadable. A point below the floor is drawn at the floor with
#: a distinguishing marker, and its exact value is written into the same badge
#: text (or control label) a reader already looks at for that row, so nothing
#: the panel would otherwise say is lost to the compression.
_AXIS_FLOOR: float = 1e-16


def _clamp_to_floor(value: float, floor: float = _AXIS_FLOOR) -> tuple[float, bool]:
    """Plotted x-position for *value*, and whether it was compressed to the floor."""
    if value < floor:
        return floor, True
    return value, False


def _format_off_scale(value: float) -> str:
    """Render a compressed p in the mantissa-times-ten-to-the-power form."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}×10{_superscript(int(exponent))}"


_SUPERSCRIPT_DIGITS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _superscript(exponent: int) -> str:
    return str(exponent).translate(_SUPERSCRIPT_DIGITS)


def generate_confirmatory_replication(variables: dict[str, str] | None = None) -> Figure:
    """The held-back replication of the four registered members, with controls."""
    apply_style()
    data = confirmatory_replication_data(variables)
    rows = data["rows"]
    positive, negative = data["controls"]
    clustered = data["clustered"]
    alpha = data["frozen_alpha"]

    fig, (top, middle, bottom) = plt.subplots(
        3,
        1,
        figsize=FIGSIZES["confirmatory"],
        gridspec_kw={"height_ratios": (3.4, 2.6, 2.2)},
    )

    # ── Top: exact p, development beside held-back ──────────────────────────
    every_p = [row["held"]["p"] for row in rows]
    every_p += [row["development"]["p"] for row in rows]
    every_p += [positive["p"], negative["p"]]
    # The left edge follows whatever is actually drawn, not the smallest exact p:
    # a value below the floor is compressed to the floor itself (see
    # ``_clamp_to_floor``), so the axis only has to reach the floor plus margin
    # rather than out to a p that may be 70 orders of magnitude smaller.
    plotted = [_clamp_to_floor(value)[0] for value in every_p]
    left = min(_p_floor(plotted), _AXIS_FLOOR / 3.0)
    ys = list(range(len(rows) - 1, -1, -1))
    control_ys = [-1.15, -1.85]

    top.set_xscale("log")
    top.axvline(alpha, color=COLOR_GATE, linewidth=2.0, zorder=2)
    top.axvline(_AXIS_FLOOR, color=COLOR_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    row_labels: list[str] = []
    for y, row in zip(ys, rows, strict=True):
        held, development = row["held"], row["development"]
        dev_x, dev_clamped = _clamp_to_floor(development["p"])
        held_x, held_clamped = _clamp_to_floor(held["p"])
        top.scatter([dev_x], [y], marker="s", s=80, color=COLOR_DEV, zorder=4)
        top.scatter([held_x], [y], marker="o", s=100, color=COLOR_HELD, zorder=5)
        # An off-scale point gets a second, smaller left-pointing marker behind
        # it — the reader's cue that this dot sits at the compression floor, not
        # at its true position — rather than a different color or shape for the
        # dot itself, which would break the held/development color-and-shape
        # legend every other row still uses. The exact value it hides is
        # written into that row's own y-tick label (below), which has empty
        # margin on both sides; a text annotation pinned beside the marker
        # itself runs into the neighboring row's badge, one data-unit away.
        for x, clamped, color in (
            (dev_x, dev_clamped, COLOR_DEV),
            (held_x, held_clamped, COLOR_HELD),
        ):
            if clamped:
                top.scatter(
                    [x],
                    [y],
                    marker="<",
                    s=55,
                    color=color,
                    edgecolors=COLOR_INK,
                    linewidths=0.9,
                    zorder=6,
                )
        if held_clamped:
            row_labels.append(f"{row['member'].label}\n(held p≈{_format_off_scale(held['p'])})")
        elif dev_clamped:
            row_labels.append(
                f"{row['member'].label}\n(dev p≈{_format_off_scale(development['p'])})"
            )
        else:
            row_labels.append(row["member"].label)
        # Holm's threshold is per member, so it is a tick on that member's row
        # rather than a rule across a panel where it means something different
        # on every line.
        top.scatter(
            [held["holm_threshold"]],
            [y],
            marker="|",
            s=340,
            color=COLOR_ACCENT,
            linewidths=2.4,
            zorder=3,
        )
        if not held["computable"]:
            badge = "not computable: no discordant pair"
        elif held["crosses"] and held["holm_rejected"]:
            badge = "replicated: crosses the frozen threshold and Holm's"
        elif held["crosses"]:
            badge = "crosses the frozen threshold, not Holm's"
        elif held["holm_rejected"]:
            badge = "crosses Holm's threshold, not the frozen one"
        else:
            badge = "does not cross either threshold"
        # Left-aligned above the row. Right-aligned, these ran into the frozen
        # threshold rule and its label, which sit at the right of a log p-axis
        # by construction; a verdict word struck through by the very rule it
        # reports on is the defect this panel would be least able to afford.
        top.text(
            0.012,
            y + 0.26,
            badge,
            transform=top.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=FONTS["tick_label"],
            color=COLOR_INK if held["computable"] else COLOR_MUTED,
            bbox={
                "boxstyle": "square,pad=0.1",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
            zorder=7,
        )
    top.axhline(-0.62, color=COLOR_MUTED, linewidth=0.9, linestyle=":", zorder=1)
    control_labels = []
    for control_y, control in zip(control_ys, (positive, negative), strict=True):
        control_x, control_clamped = _clamp_to_floor(control["p"])
        top.scatter(
            [control_x],
            [control_y],
            marker="D",
            s=85,
            facecolors="white",
            edgecolors=COLOR_INK,
            linewidths=1.8,
            zorder=5,
        )
        if control_clamped:
            top.scatter(
                [control_x],
                [control_y],
                marker="<",
                s=55,
                color=COLOR_INK,
                zorder=6,
            )
            control_labels.append(
                f"{control['label']} (p ≈ {_format_off_scale(control['p'])}, off-scale)"
            )
        else:
            control_labels.append(control["label"])
    top.set_yticks([*ys, *control_ys])
    top.set_yticklabels(
        row_labels + control_labels,
        fontsize=FONTS["tick_label"],
    )
    top.set_ylim(-3.85, len(rows) + 0.15)
    # Leave display space beyond p=1 so its marker and the control are not clipped.
    top.set_xlim(left, 100.0)
    top.set_xlabel(
        "exact two-sided p (log scale; left of the dashed line: off-scale)",
        fontsize=FONTS["tick_label"],
    )
    top.set_title(
        f"Held-back replication on {data['population']}: "
        f"{data['census_documents']:,} documents, {data['census_sentences']:,} sentences\n"
        f"(registered in decision {data['registration_decision']})",
        fontsize=FONTS["subtitle"],
        fontweight="bold",
        pad=34,
    )
    top.text(
        1.0,
        1.02,
        f"frozen per-test α = {alpha:g} (familywise over {data['frozen_family_size']})",
        transform=top.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONTS["tick_label"],
        color=COLOR_GATE,
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": "none",
        },
        zorder=6,
    )
    top.legend(
        handles=[
            Line2D([], [], color=COLOR_HELD, marker="o", linestyle="None", label="held-back p"),
            Line2D([], [], color=COLOR_DEV, marker="s", linestyle="None", label="development p"),
            Line2D(
                [],
                [],
                color=COLOR_ACCENT,
                marker="|",
                linestyle="None",
                markersize=11,
                label="this member's Holm threshold",
            ),
            Line2D([], [], color=COLOR_GATE, linestyle="-", label="frozen per-test threshold"),
            Line2D(
                [],
                [],
                color=COLOR_INK,
                marker="D",
                markerfacecolor="white",
                linestyle="None",
                label="registered control",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.005, 0.005),
        fontsize=FONTS["tick_label"],
        # Opaque rather than frameless: the compressed axis (see ``_AXIS_FLOOR``)
        # pulls the frozen-threshold rule leftward into the legend's own text
        # band, so the legend needs a shield against a rule running through it.
        frameon=True,
        facecolor=PALETTE["BACKGROUND"],
        edgecolor="none",
        ncol=2,
    )
    top.spines[["top", "right"]].set_visible(False)

    # ── Middle: the discordant counts each population produced ──────────────
    height = 0.34
    for y, row in zip(ys, rows, strict=True):
        member, held, development = row["member"], row["held"], row["development"]
        for offset, arm, color in (
            (height * 0.62, held, COLOR_HELD),
            (-height * 0.62, development, COLOR_DEV),
        ):
            middle.barh(
                y + offset,
                arm["favored"],
                height=height,
                color=color,
                edgecolor=COLOR_INK,
                linewidth=0.8,
                zorder=3,
            )
            middle.barh(
                y + offset,
                -arm["opposed"],
                height=height,
                color=color,
                alpha=0.45,
                edgecolor=COLOR_INK,
                linewidth=0.8,
                zorder=3,
            )
        census = (
            f"held-back: {held['discordant']:,}/{held['population']:,} {member.unit}, "
            f"{held['documents']:,} docs\n"
            f"development: {development['discordant']:,}/{development['population']:,} "
            f"{member.unit}, {development['documents']:,} docs"
        )
        middle.text(
            1.0,
            y,
            census,
            transform=middle.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FONTS["tick_label"],
            linespacing=1.2,
            color=COLOR_INK if held["computable"] else COLOR_MUTED,
        )
    middle.axvline(0.0, color=COLOR_INK, linewidth=1.2, zorder=2)
    middle.set_yticks(ys)
    middle.set_yticklabels([row["member"].label for row in rows], fontsize=FONTS["tick_label"])
    middle.set_ylim(-0.7, len(rows) + 0.1)
    widest = max(max(r["held"]["favored"], r["development"]["favored"]) for r in rows)
    middle.set_xlim(-widest * 1.15, widest * 3.5)
    middle.set_xlabel(
        "discordant pairs (favored right; opposed left)",
        fontsize=FONTS["tick_label"],
    )
    middle.set_title(
        "Discordant pairs and evaluated populations", fontsize=FONTS["subtitle"], fontweight="bold"
    )
    middle.legend(
        handles=[
            Line2D([], [], color=COLOR_HELD, linewidth=8, label="held-back"),
            Line2D([], [], color=COLOR_DEV, linewidth=8, label="development"),
        ],
        loc="upper center",
        fontsize=FONTS["tick_label"],
        frameon=False,
        ncol=2,
    )
    middle.spines[["top", "right"]].set_visible(False)

    # Separate rows make the primary and corrected intervals readable at print
    # width. Both held-back rows describe the same estimate and population.
    interval_rows = (
        (2.0, "held", "held_low", "held_high", COLOR_HELD),
        (1.0, "held", "corrected_low", "corrected_high", COLOR_HELD),
        (0.0, "dev", "dev_low", "dev_high", COLOR_DEV),
    )
    for interval_y, prefix, low_key, high_key, color in interval_rows:
        point = clustered[f"{prefix}_effect"]
        bottom.errorbar(
            [point],
            [interval_y],
            xerr=[[point - clustered[low_key]], [clustered[high_key] - point]],
            fmt="o",
            color=color,
            elinewidth=2.0,
            capsize=4,
            zorder=4,
        )
    bottom.axvline(0.0, color=COLOR_INK, linestyle="--", linewidth=1.0, zorder=1)
    bottom.axvline(clustered["min_detectable"], color=COLOR_FLOOR, linewidth=1.0, zorder=1)
    held_pct = clustered["held_confidence"] * 100.0
    corrected_pct = clustered["corrected_confidence"] * 100.0
    dev_pct = clustered["confidence"] * 100.0
    bottom.set_yticks([2.0, 1.0, 0.0])
    bottom.set_yticklabels(
        [
            f"held-back, primary {held_pct:g}%\n{clustered['held_documents']:,} documents",
            f"held-back, corrected {corrected_pct:g}%\n{clustered['held_documents']:,} documents",
            f"development, primary {dev_pct:g}%\n{clustered['dev_documents']:,} documents",
        ],
        fontsize=FONTS["tick_label"],
    )
    bottom.set_ylim(-0.6, 2.8)
    bottom.set_xlabel(
        "mean per-sentence improvement (nats):\ntype-unigram minus phase-model log loss",
        fontsize=FONTS["tick_label"],
    )
    held_verdict = "excludes zero" if clustered["held_excludes_zero"] else "spans zero"
    corrected_verdict = "excludes zero" if clustered["corrected_excludes_zero"] else "spans zero"
    bottom.set_title(
        f"L6: held-back primary interval {held_verdict};\n"
        f"held-back corrected {corrected_verdict}; development spans zero",
        fontsize=FONTS["subtitle"],
        fontweight="bold",
        pad=36,
    )
    bottom.text(
        0.5,
        1.03,
        f"minimum detectable effect: {clustered['min_detectable']:g} nats",
        transform=bottom.transAxes,
        ha="center",
        va="bottom",
        color=COLOR_FLOOR,
        fontsize=FONTS["tick_label"],
    )
    bottom.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(h_pad=2.4)
    return fig
