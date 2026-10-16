"""Every null in this paper drawn as a bound, not as an absence.

One figure, three panels, all read from the claims ledger and the derived
interval tokens so nothing here can drift from the measurements.

``null_bounds`` (top panel)
    The accuracy-scale nulls as intervals around zero, in percentage points:
    Stage 5 clause-type classification against the majority baseline, with the
    Stage 3 lexical channel at the high mixing weight drawn beside it as the
    panel's established contrast, for scale. That lexical row was itself a
    bounded null until its backoff lookup was corrected (decision 110). The two
    context-window arms were here until the census resolved both contrasts; the
    null row set is whatever is still a null, and the figure refuses to draw a
    resolved contrast as a null rather than relabeling one. Each row prints its discordant pair count and
    its recorded p value. A row whose interval spans
    zero is a bounded effect, not a demonstrated absence, and its width is how
    large the effect is still allowed to be.
``null_bounds`` (middle panel)
    The two directional-split results, on the same percentage-point scale but a
    different measurement: the share of items falling the probe's way rather
    than an accuracy gap. Stage 6's document-position sign test is the paper's
    near-null — it clears Holm's step-down threshold but not the fixed one — and the
    Stage 2B G2P comparison is drawn beside it over its discordant positions as
    the one row here that *does* clear the frozen family threshold. That row is
    the ruler: it shows what an established directional effect looks like on
    this axis, so the reader can see that the near-null is bounded rather than
    merely unlabeled. The punctuation-type-oracle intervention is deliberately
    absent: its tokens are nested in Brown documents, so its uncertainty is the
    whole-document bootstrap reported in the hierarchy section, not an
    independent-token paired interval.
``null_bounds`` (bottom panel)
    The smallest effect the study could have resolved, drawn on the scale where
    the project actually declared one. Each row's discordant pair count against
    the two analytic reachability thresholds from
    ``ailm.statistics.minimum_discordant_pairs_for_equivalence``: the fewest
    pairs from which an equivalence claim is reachable *at all*, at the interval
    convention's level and at the family-corrected level. Rows left of the first
    line could not have supported an equivalence claim however their pairs had
    fallen, which is the difference between "we bounded this effect" and "we
    showed this effect is negligible". Three mutually exclusive states are drawn:
    established, equivalent, and bounded-only. Color carries the same meaning it
    carries in the panels above — vermillion is the established row wherever it
    appears — because a hue that means "established" in one panel and "not
    equivalent" in the next is a legend the reader has to relearn mid-figure.

Why the resolution threshold is not drawn on the interval axes
--------------------------------------------------------------
It is tempting to mark the smallest-effect-of-interest as a vertical band on the
top two panels. ``ailm.statistics.equivalence_region`` explains why that would be
wrong here, and the defect is worth naming: the project's declared
smallest-effect-of-interest is a *share*, ``REFERENCE_SHARE``, and converting it
to the difference scale requires multiplying by the arm's own observed
discordance rate. A threshold that depends on the data is no longer
pre-declared, and the dependence runs backwards — the fewer discordant pairs an
arm produced, the smaller its converted threshold, so equivalence would be
easiest to claim exactly where the evidence is thinnest. Every arm on the top
two panels would then acquire its own private band, and the arm with two
discordant pairs would get the narrowest one. The bottom panel puts the same
question on the share scale, where one threshold serves every row.

Provenance and the falsifier
----------------------------
Nothing here measures anything. Every interval is a derived token built in
``src/ailm/manuscript_variables.py`` from recorded discordant counts, and every
threshold is analytic. Drawing derived numbers without a check would make the
figure a second, unaudited derivation, so :func:`null_bounds_data` refuses to
draw when a re-derived value contradicts the ledger — the ``multiplicity_forest``
and ``calibration_floor`` pattern. A disagreement is a finding about the recorder
or the derivation, never something to redraw around.

The status check carries its own positive control. The predicate "this row's
recorded p clears the corrected threshold" is exercised in both directions on
every run: three rows must not clear it and the contrast row must, so a predicate
that silently stopped discriminating — a token read that always returned the
same value, an alpha comparison inverted — fails on the contrast row instead of
reading as a clean sweep.

Nothing here runs a recorder and nothing here writes the ledger.

Every generator returns an unwritten ``Figure``; ``scripts/generate_figures.py``
discovers them by the ``generate_`` prefix and handles saving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ailm.figures._common import (
    resolve_variables as _v,
)
from ailm.figures._common import (
    token_float,
    token_int,
)
from ailm.statistics import (
    BONFERRONI_ALPHA,
    CORRECTED_CONFIDENCE,
    DIFFERENCE_CONFIDENCE,
    REFERENCE_SHARE,
    equivalence_region,
    minimum_discordant_pairs_for_equivalence,
    share_interval_supports_equivalence,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = [
    "generate_null_bounds",
    "null_bounds_data",
]

COLOR_BOUND = PALETTE["MODEL"]  # a bounded, not-established effect
COLOR_CONTRAST = PALETTE["ANTI"]  # the one established row, drawn for scale
COLOR_INK = PALETTE["INK"]
COLOR_MUTED = PALETTE["MUTED"]
COLOR_GATE = PALETTE["GATE"]  # the corrected-level reachability threshold
COLOR_EQUIVALENT = PALETTE["FLOOR"]  # a row that reached an equivalence claim
COLOR_BAND = PALETTE["HIGHLIGHT"]  # the span where no equivalence is reachable

#: The null on every difference-scale axis in this figure. A structural constant
#: of the paired comparison — zero difference between probe and comparator — not
#: a measured number, so it is not a ledger token.
NULL_DIFFERENCE: float = 0.0

#: Floor on any single-token tolerance, as a fraction of the compared value.
#: The rendered decimals give the real tolerance; this only keeps a comparison
#: against a value with many printed decimals from being tighter than float
#: arithmetic can honour after the count division and the percentage rescale.
_RELATIVE_FLOOR: float = 1e-9

#: Relative tolerance for holding a *convention constant* against its executable
#: definition. Deliberately not :func:`_rendered_tolerance`: that reads the
#: precision off the token being checked, so a drifted constant printed with one
#: decimal would grant itself half a unit of slack — a reference share of 0.6
#: would pass as 0.55, which is the exact defect this comparison exists to catch.
#: Both sides are known to full precision here, so the only legitimate
#: disagreement is the renderer's six significant figures: 5e-7 relatively, which
#: 1e-5 absorbs with two orders of margin while staying far below any real drift.
_CONSTANT_REL_TOL: float = 1e-5


@dataclass(frozen=True, slots=True)
class _Row:
    """One drawn result: where its numbers come from and what it claims.

    Attributes:
        label: Row label as drawn.
        panel: ``"accuracy"`` for an accuracy difference, ``"share"`` for a net
            directional split. The two are separate panels because they are
            different measurements, not because their magnitudes differ.
        point: Token holding the point estimate.
        low: Token holding the lower interval endpoint.
        high: Token holding the upper interval endpoint.
        b: Recorded count of pairs falling the probe's way.
        c: Recorded count of pairs falling the comparator's way.
        n: Recorded denominator of the difference — total paired items, not the
            discordant subset.
        p: Recorded p value for this comparison.
        unit: What one item of ``n`` is. A bound stated over an unnamed unit
            cannot be judged: "no effect over 187" means one thing over sentences
            and another over documents, and this figure's whole subject is how
            much a null does and does not rule out.
        cluster_prefix: Token prefix of the recorded cluster-robust re-analysis
            of this arm, or ``None`` where none was recorded. The drawn interval
            is the independence-assuming one either way; naming the cluster arm
            beside it says whether that assumption was ever checked, and naming
            its absence says it was not.
        token_is_points: Whether ``point``/``low``/``high`` are already in
            percentage points. The derived tokens are written on the scale of
            whatever recorded quantity sits beside them in prose, so the scale
            differs per arm and must be declared rather than guessed.
        established: Whether this row's recorded p is expected to clear the
            family-corrected threshold. Exactly one row sets this, and it is the
            positive control for the status check.
        every_item_discordant: Whether the arm's denominator equals its
            discordant count. True for the two directional splits, where the
            quantity is a net share over all items rather than an accuracy gap;
            asserting it catches a denominator token swapped in from another arm.
        independent_point: Optional ``(minuend, subtrahend)`` tokens whose
            difference reproduces the point estimate from separately recorded
            quantities. The denominator is the one input the counts cannot check
            themselves, so where a second recorded route to the same number
            exists it is used.
    """

    label: str
    panel: str
    point: str
    low: str
    high: str
    b: str
    c: str
    n: str
    p: str
    token_is_points: bool
    unit: str = "paired items"
    cluster_prefix: str | None = None
    established: bool = False
    every_item_discordant: bool = False
    independent_point: tuple[str, str] | None = None


#: The four drawn rows. Three are reported as null or near-null; the fourth is
#: the established contrast that keeps the status check honest.
#:
#: Directions follow the recorder's own convention in every case: ``b`` is the
#: count in the direction the arm's claim names, ``c`` the other way, so the
#: point estimate ``(b - c) / n`` carries the sign the prose carries.
_ROWS: tuple[_Row, ...] = (
    _Row(
        label="Stage 5 clause type vs majority baseline",
        panel="accuracy",
        point="STAGE5_CLASSIFICATION_DELTA_PTS",
        low="STAGE5_CLASSIFICATION_DELTA_LOW_PTS",
        high="STAGE5_CLASSIFICATION_DELTA_HIGH_PTS",
        b="STAGE5_MCNEMAR_B",
        c="STAGE5_MCNEMAR_C",
        n="STAGE5_TEST_SENTENCES",
        p="STAGE5_MCNEMAR_P",
        token_is_points=True,
        unit="sentences",
        # Two recorded accuracies pin the same difference without going through
        # the discordant counts at all, so a wrong denominator — the failure the
        # counts are blind to — cannot pass here.
        independent_point=("STAGE5_CLASSIFICATION_PCT", "STAGE5_MAJORITY_BASELINE_PCT"),
    ),
    _Row(
        label="Stage 3 lexical channel, mixing weight 0.9 (established)",
        panel="accuracy",
        # Recorded, not derived: the ledger holds this arm's margin directly, so
        # the drawn point estimate and the drawn interval come from independent
        # routes and a disagreement between them is visible.
        #
        # This row was the figure's one bounded-and-equivalent null until the
        # lexical channel's backoff lookup was corrected (decision 110). Under
        # the documented longest-suffix rule the same arm is a decided effect
        # at the corrected threshold, so it is drawn as the accuracy panel's
        # established contrast, for scale, exactly as the Stage 2 contrast is
        # drawn in the share panel. Dropping it instead would have left the
        # accuracy panel with a single row and the status check with nothing
        # to exercise its positive branch on there.
        point="LEXICAL_HIGH_ALPHA_MARGIN_PTS",
        low="LEXICAL_HIGH_ALPHA_DELTA_LOW_PTS",
        high="LEXICAL_HIGH_ALPHA_DELTA_HIGH_PTS",
        b="LEXICAL_SWEEP_ALPHA_0_9_MCNEMAR_B",
        c="LEXICAL_SWEEP_ALPHA_0_9_MCNEMAR_C",
        n="LEXICAL_EVAL_POSITIONS",
        p="LEXICAL_SWEEP_ALPHA_0_9_MCNEMAR_P",
        token_is_points=True,
        unit="letter positions",
        established=True,
    ),
    # The two context-window rows were here and have been removed, not silently:
    # this figure is the inventory of the project's NULLS, and over the census
    # pool both contrasts became established. The figure refused to draw them
    # rather than relabeling them, which is how the change was noticed. A row
    # here is a claim that an effect was bounded and not shown; an established
    # effect belongs in the accuracy tables, and leaving it in a bounds panel
    # would report a resolved contrast as an unresolved one.
    #
    # The prompt-format contrast on Qwen 2.5 3B is the arm that did stay a null,
    # and it is deliberately NOT added here in its place: at its discordant
    # total it is below the minimum an equivalence claim needs, so it is an
    # unresolvable null rather than a bounded one, and this figure's shaded
    # region is exactly the line between those two readings.
    _Row(
        label="Stage 6 document position, sign test",
        panel="share",
        point="STAGE6_SIGN_NET_SHARE",
        low="STAGE6_SIGN_NET_SHARE_LOW",
        high="STAGE6_SIGN_NET_SHARE_HIGH",
        b="STAGE6_SIGN_TEST_POSITIVE",
        c="STAGE6_SIGN_TEST_NEGATIVE",
        n="STAGE6_TEST_SENTENCES",
        p="STAGE6_SIGN_TEST_P",
        token_is_points=False,
        unit="sentences",
        cluster_prefix="CLUSTER_STAGE6_SIGN",
        every_item_discordant=True,
    ),
    _Row(
        label="Stage 2 G2P vs per-letter bar (established)",
        panel="share",
        point="STAGE2_G2P_DISCORDANT_NET_SHARE",
        low="STAGE2_G2P_DISCORDANT_NET_SHARE_LOW",
        high="STAGE2_G2P_DISCORDANT_NET_SHARE_HIGH",
        b="STAGE2_G2P_MCNEMAR_B",
        c="STAGE2_G2P_MCNEMAR_C",
        n="STAT_STAGE2_DISCORDANT_N",
        p="STAGE2_G2P_MCNEMAR_P",
        token_is_points=False,
        unit="letter-position pairs",
        cluster_prefix="CLUSTER_STAGE2_G2P",
        established=True,
        every_item_discordant=True,
    ),
)


def _rendered_tolerance(variables: dict[str, str], token: str, value: float) -> float:
    """Half a unit in the last decimal place the renderer actually printed.

    A fixed tolerance is wrong in both directions here. The derived interval
    tokens are rounded to six decimals, so a slack tolerance would accept a
    genuinely different number; the recorded ``lexical_high_alpha_margin_pts`` is
    rounded to three, so a tight one would reject the correct number for having
    been printed shortly. Reading the precision off the rendered string makes
    each comparison exactly as strict as the value it is comparing.

    Args:
        variables: Mapping from :func:`load_manuscript_variables`.
        token: Token whose printed precision sets the tolerance.
        value: The parsed value, used only for the relative floor.

    Returns:
        Absolute tolerance, in the token's own units.
    """
    raw = variables[token].replace(",", "").strip().rstrip("%")
    if "e" in raw.lower():
        # Scientific notation prints significant figures rather than decimals, so
        # the last place is relative. Six significant figures is what the
        # renderer emits, and half a unit in the sixth is 5e-6 relatively.
        return max(abs(value) * 5e-6, _RELATIVE_FLOOR)
    decimals = len(raw.partition(".")[2])
    return max(0.5 * 10.0**-decimals, abs(value) * _RELATIVE_FLOOR)


def null_bounds_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Drawn rows and thresholds, each cross-checked against the ledger.

    Every check below is a refuse-to-draw falsifier rather than a correction.
    The intervals are derived at render time from recorded counts and the
    thresholds are analytic, so a disagreement means the recorder, the
    derivation in ``src/ailm/manuscript_variables.py``, or
    ``ailm.statistics`` has changed — a finding to investigate, never something
    to paper over in a figure.

    Returns:
        ``rows`` (one dict per drawn result, in percentage points), plus the
        convention constants and the two analytic reachability thresholds.

    Raises:
        MissingVariableError: If a token the figure draws is absent or is not a
            number.
        ValueError: If a convention token has drifted from its executable
            constant in ``ailm.statistics``; if a drawn interval is inverted or
            fails to contain its own point estimate; if a point estimate
            recomputed from the row's recorded counts and denominator disagrees
            with the drawn token beyond the renderer's own rounding; if a row's
            recorded p contradicts the established/not-established status the
            figure draws it with; if a row declared to have every item
            discordant no longer does; if the two reachability thresholds are
            not ordered by strictness; if a row's recomputed equivalence verdict
            contradicts the recorded ``LEXICAL_HIGH_ALPHA_EQUIVALENT`` token or
            is claimed below the analytic pair-count minimum; or if no
            established contrast row is present, which would leave the status
            check with nothing to exercise its positive branch on.
    """
    v = _v(variables)

    # ── Convention constants must still be the executable ones ───────────────
    # These four tokens are the manuscript's statement of what the intervals
    # mean. If prose and code disagree about the coverage, every interval below
    # is labeled with a level it was not built at.
    def _matches_constant(drawn: float, declared: float) -> bool:
        """Whether a token is the renderer's rounding of an executable constant."""
        return abs(drawn - declared) <= max(abs(declared) * _CONSTANT_REL_TOL, 1e-12)

    confidence_pct = token_float(v, "BOUND_CONFIDENCE_PCT")
    corrected_pct = token_float(v, "BOUND_CORRECTED_CONFIDENCE_PCT")
    if not _matches_constant(confidence_pct, DIFFERENCE_CONFIDENCE * 100.0):
        raise ValueError(
            f"the ledger's bound confidence {confidence_pct}% is not the declared "
            f"{DIFFERENCE_CONFIDENCE * 100.0}%; re-generate the manuscript variables."
        )
    if not _matches_constant(corrected_pct, CORRECTED_CONFIDENCE * 100.0):
        raise ValueError(
            f"the ledger's corrected confidence {corrected_pct}% is not the declared "
            f"{CORRECTED_CONFIDENCE * 100.0}%; re-generate the manuscript variables."
        )
    reference_share = token_float(v, "EQUIVALENCE_REFERENCE_SHARE")
    if not _matches_constant(reference_share, REFERENCE_SHARE):
        raise ValueError(
            f"the ledger's reference share {reference_share} is not the declared "
            f"{REFERENCE_SHARE}; the equivalence region drawn below would be the "
            f"wrong region."
        )
    # Rounded comparison, not equality: the region is ``1 - share`` computed in
    # binary, so the declared 0.55 gives 0.44999999999999996 in code and 0.45
    # once the renderer rounds it. An equality check here rejects a correct
    # ledger for a representation artifact and says nothing about drift.
    region_low = token_float(v, "EQUIVALENCE_REGION_LOW")
    region_high = token_float(v, "EQUIVALENCE_REGION_HIGH")
    code_region_low, code_region_high = equivalence_region()
    if not (
        _matches_constant(region_low, code_region_low)
        and _matches_constant(region_high, code_region_high)
    ):
        raise ValueError(
            f"the ledger's equivalence region [{region_low}, {region_high}] is not the "
            f"declared [{code_region_low}, {code_region_high}]; re-generate the "
            f"manuscript variables."
        )

    # ── The two analytic reachability thresholds ─────────────────────────────
    minimum_pairs = token_int(v, "EQUIVALENCE_MIN_DISCORDANT_PAIRS")
    minimum_pairs_corrected = token_int(v, "EQUIVALENCE_MIN_DISCORDANT_PAIRS_CORRECTED")
    code_minimum = minimum_discordant_pairs_for_equivalence()
    code_minimum_corrected = minimum_discordant_pairs_for_equivalence(
        confidence=CORRECTED_CONFIDENCE
    )
    if minimum_pairs != code_minimum or minimum_pairs_corrected != code_minimum_corrected:
        raise ValueError(
            f"the drawn reachability thresholds ({minimum_pairs}, "
            f"{minimum_pairs_corrected}) are not the ones "
            f"minimum_discordant_pairs_for_equivalence returns ({code_minimum}, "
            f"{code_minimum_corrected}); re-generate the manuscript variables."
        )
    # Structural, not measured: a stricter level needs a narrower interval and
    # therefore more pairs. Drawing them the other way round would tell the
    # reader that correcting for multiplicity makes equivalence easier. The
    # check above already pins both tokens to the code, so this one constrains
    # the code path — a search that started returning a disordered pair would
    # otherwise propagate silently into both tokens and agree with itself.
    if not minimum_pairs_corrected > minimum_pairs:
        raise ValueError(
            f"the corrected reachability threshold {minimum_pairs_corrected} is not "
            f"above the uncorrected {minimum_pairs}. A stricter level cannot need "
            f"fewer pairs; one of the two was computed at the wrong confidence."
        )

    rows: list[dict[str, Any]] = []
    for row in _ROWS:
        b = token_int(v, row.b)
        c = token_int(v, row.c)
        n = token_int(v, row.n)
        if n <= 0:
            raise ValueError(f"{row.label}: denominator token {row.n} is {n}, not positive.")
        discordant = b + c
        if discordant > n:
            raise ValueError(
                f"{row.label}: {discordant} discordant pairs exceed the {n} paired "
                f"items in {row.n}. The denominator belongs to a different arm."
            )
        if row.every_item_discordant and discordant != n:
            raise ValueError(
                f"{row.label}: this row is drawn as a net directional share over all "
                f"{n} items, which requires every item to be discordant, but "
                f"{discordant} are. The quantity is an accuracy gap now and the row "
                f"label is wrong."
            )

        # Units. Each derived token was written on the scale of the recorded
        # quantity it sits beside in prose, so the arms differ; the comparison
        # happens in the token's own units and only the drawing is rescaled.
        unit_factor = 100.0 if row.token_is_points else 1.0
        points_per_unit = 1.0 if row.token_is_points else 100.0

        point = token_float(v, row.point)
        low = token_float(v, row.low)
        high = token_float(v, row.high)
        if not low <= high:
            raise ValueError(
                f"{row.label}: the drawn interval [{low}, {high}] is inverted; the "
                f"low and high tokens are paired the wrong way round."
            )
        if not low <= point <= high:
            raise ValueError(
                f"{row.label}: the point estimate {point} falls outside its own "
                f"interval [{low}, {high}]. The point and the interval belong to "
                f"different arms; refusing to draw a whisker that does not bracket "
                f"its own marker."
            )

        # Recompute the point estimate from this row's own recorded counts and
        # denominator. The realiztic authoring defect in a table this wide is a
        # token from the neighboring arm — gemma3's interval beside qwen25's
        # estimate — and that survives every check that reads one arm at a time.
        recomputed = (b - c) / n * unit_factor
        tolerance = _rendered_tolerance(v, row.point, point)
        if abs(recomputed - point) > tolerance:
            raise ValueError(
                f"{row.label}: the point estimate recomputed from the recorded counts "
                f"({b} - {c}) / {n} is {recomputed!r} but the drawn token {row.point} "
                f"says {point!r}, a gap wider than the {tolerance!r} the renderer's "
                f"own rounding allows. The tokens and the counts have diverged."
            )

        if row.independent_point is not None:
            # A second recorded route to the same difference, bypassing the
            # counts entirely. Without it a plausible wrong denominator would
            # rescale the whole row and every count-based check would still pass.
            minuend_token, subtrahend_token = row.independent_point
            minuend = token_float(v, minuend_token)
            subtrahend = token_float(v, subtrahend_token)
            independent = minuend - subtrahend
            slack = _rendered_tolerance(v, minuend_token, minuend) + _rendered_tolerance(
                v, subtrahend_token, subtrahend
            )
            if abs(independent - point) > slack:
                raise ValueError(
                    f"{row.label}: {minuend_token} - {subtrahend_token} is "
                    f"{independent!r} but the drawn point estimate is {point!r}. Two "
                    f"recorded routes to one difference disagree, so the denominator "
                    f"{row.n} is not the one those accuracies were measured over."
                )

        # Status. The corrected threshold is the code constant, not the rounded
        # token, because the decision is the code's; the token is checked
        # against it as a rounding rather than used to decide.
        p_value = token_float(v, row.p)
        clears = p_value < BONFERRONI_ALPHA
        if clears != row.established:
            raise ValueError(
                f"{row.label}: recorded p = {p_value} against the corrected threshold "
                f"{BONFERRONI_ALPHA} makes this row "
                f"{'established' if clears else 'not established'}, but the figure "
                f"draws it as {'established' if row.established else 'not established'}. "
                f"Refusing to present it under the wrong heading."
            )

        # Equivalence is a share-scale question, so it is recomputed on the share
        # scale from the raw counts, through the same predicate the derived
        # LEXICAL_HIGH_ALPHA_EQUIVALENT token was built with.
        equivalent = discordant > 0 and share_interval_supports_equivalence(
            b, c, confidence=CORRECTED_CONFIDENCE
        )
        # Exclusivity first: it is the more fundamental contradiction, and
        # reporting the pair-count mismatch ahead of it would name a symptom.
        if equivalent and row.established:
            raise ValueError(
                f"{row.label}: drawn as both established and equivalent. Equivalence "
                f"puts the whole corrected share interval inside "
                f"[{region_low}, {region_high}], which contains the null share, so an "
                f"equivalent arm cannot also exclude it. The two verdicts are computed "
                f"from different quantities and one of them is about another arm."
            )
        if row.point == "LEXICAL_HIGH_ALPHA_MARGIN_PTS":
            # The only arm with a recorded equivalence verdict. Holding the
            # recomputation against it keeps the predicate honest for the six
            # rows that have no such token to check against.
            recorded_equivalent = bool(token_int(v, "LEXICAL_HIGH_ALPHA_EQUIVALENT"))
            if equivalent != recorded_equivalent:
                raise ValueError(
                    f"{row.label}: the equivalence verdict recomputed from "
                    f"({b}, {c}) is {equivalent} but LEXICAL_HIGH_ALPHA_EQUIVALENT "
                    f"records {recorded_equivalent}; re-generate the manuscript "
                    f"variables and find out which is wrong."
                )
        if equivalent and discordant < minimum_pairs_corrected:
            raise ValueError(
                f"{row.label}: an equivalence claim is drawn on {discordant} discordant "
                f"pairs, below the {minimum_pairs_corrected} pairs from which one is "
                f"reachable at all. The predicate and the analytic minimum disagree."
            )

        # The dependence structure the drawn interval assumes, named either way.
        cluster: dict[str, Any] | None = None
        if row.cluster_prefix is not None:
            cluster_count = token_int(v, f"{row.cluster_prefix}_CLUSTERS")
            cluster_observations = token_int(v, f"{row.cluster_prefix}_OBSERVATIONS")
            if not 0 < cluster_count <= cluster_observations:
                raise ValueError(
                    f"{row.label}: {cluster_count} clusters over {cluster_observations} "
                    "observations is not a partition of the arm's population."
                )
            cluster = {
                "unit": v[f"{row.cluster_prefix}_CLUSTER_UNIT"].strip(),
                "count": cluster_count,
                "observations": cluster_observations,
                "confidence": token_float(v, "CLUSTER_CONFIDENCE"),
            }
        rows.append(
            {
                "label": row.label,
                "panel": row.panel,
                "point_pts": point * points_per_unit,
                "low_pts": low * points_per_unit,
                "high_pts": high * points_per_unit,
                "discordant": discordant,
                "n": n,
                "unit": row.unit,
                "cluster": cluster,
                "every_item_discordant": row.every_item_discordant,
                "p": p_value,
                "established": row.established,
                "equivalent": equivalent,
            }
        )

    # A status check that only ever sees not-established rows would pass with a
    # broken comparison. The contrast row is what exercises the other branch, so
    # its absence is a defect in the figure rather than a smaller figure.
    if not any(r["established"] for r in rows):
        raise ValueError(
            "no established row is drawn, so the established/not-established check "
            "never exercises its positive branch and a broken comparison would read "
            "as a clean sweep. Restore the contrast row."
        )

    return {
        "rows": rows,
        "confidence_pct": confidence_pct,
        "corrected_confidence_pct": corrected_pct,
        "reference_share": reference_share,
        "region_low": region_low,
        "region_high": region_high,
        "minimum_pairs": minimum_pairs,
        "minimum_pairs_corrected": minimum_pairs_corrected,
        "corrected_alpha": BONFERRONI_ALPHA,
    }


def _format_p(value: float) -> str:
    """Render a recorded p value, naming an underflow rather than hiding it."""
    if value == 0.0:
        # The exact binomial tail underflowed to zero at this pair count. Printing
        # "p = 0" without saying so invites a reader to treat it as an exact
        # zero probability, and printing an invented "< 1e-300" would be a
        # hand-typed number standing in for a recorded one.
        return "p = 0 (underflow)"
    if value < 0.001:
        return f"p = {value:.1e}"
    return f"p = {value:g}"


def _draw_interval_panel(
    ax,
    rows: list[dict[str, Any]],
    *,
    title: str,
    xlabel: str,
    data_limits: tuple[float, float],
    annotation_x: float,
    right_limit: float,
) -> None:
    """Draw one interval panel: intervals around zero, one row each.

    Args:
        ax: Target axes.
        rows: Row dicts from :func:`null_bounds_data`, top row first.
        title: Panel title.
        xlabel: Panel x-axis label.
        data_limits: ``(left, right)`` of the region the intervals occupy.
        annotation_x: Where the per-row ``n``/``p`` annotation starts.
        right_limit: Right edge of the axes, reserving the annotation column.
    """
    ys = list(range(len(rows) - 1, -1, -1))
    for y, row in zip(ys, rows, strict=True):
        color = COLOR_CONTRAST if row["established"] else COLOR_BOUND
        ax.errorbar(
            [row["point_pts"]],
            [y],
            xerr=[[row["point_pts"] - row["low_pts"]], [row["high_pts"] - row["point_pts"]]],
            fmt="o" if not row["established"] else "s",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            markersize=8,
            zorder=3,
        )
        cluster = row["cluster"]
        if cluster is None:
            clustering = "no cluster-robust arm recorded"
        else:
            clustering = (
                f"clustered: {cluster['count']:,} {cluster['unit']}s, "
                f"{cluster['confidence'] * 100:g}%"
            )
        # Where every paired item is discordant, "n discordant of n" reads as a
        # coincidence rather than as the definition of the measure, so the two
        # cases are worded differently.
        if row["every_item_discordant"]:
            census = f"all {row['n']:,} {row['unit']} discordant"
        else:
            census = f"{row['discordant']:,} of {row['n']:,} {row['unit']} discordant"
        ax.text(
            annotation_x,
            y,
            f"{census}\n{_format_p(row['p'])}\n{clustering}",
            va="center",
            ha="left",
            fontsize=FONTS["cell"],
            linespacing=1.5,
            color=COLOR_MUTED if not row["established"] else COLOR_CONTRAST,
        )
    ax.axvline(NULL_DIFFERENCE, color=COLOR_INK, linestyle="--", linewidth=1.4, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=FONTS["footnote"])
    ax.set_xlim(data_limits[0], right_limit)
    # Headroom above the top row, so the legend has somewhere to sit that is not
    # on top of a row's annotation. The annotation column spans the full height
    # of the drawn rows, so a legend anywhere on the right without this band
    # overprints the last row it is meant to explain.
    ax.set_ylim(-0.85, len(rows) + 0.45)
    ax.set_xlabel(xlabel, fontsize=FONTS["tick_label"])
    ax.set_title(title, fontsize=FONTS["subtitle"], fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    # The reserved annotation column is not data. Ticking it would invite the
    # reader to read an interval that reaches into it, and none does.
    ax.set_xticks([t for t in ax.get_xticks() if data_limits[0] <= t <= data_limits[1]])


def generate_null_bounds(variables: dict[str, str] | None = None) -> Figure:
    """Bounds plot of every null and near-null, with the resolution floor."""
    apply_style()
    data = null_bounds_data(variables)
    accuracy_rows = [r for r in data["rows"] if r["panel"] == "accuracy"]
    share_rows = [r for r in data["rows"] if r["panel"] == "share"]

    fig, (top, middle, bottom) = plt.subplots(
        3,
        1,
        # Width: the accuracy panel's y-tick labels name the lexical row as the
        # established contrast (decision 110), and the longer label narrowed the
        # axes under tight_layout until the centered x-label and the bottom
        # title overran the right edge. Room, not smaller type.
        # Height: each row's annotation is three lines — the census with its
        # unit, the exact p, and whether a cluster-robust arm exists — so the
        # share panel's two rows need roughly half again the band the old
        # one-line annotation fitted in, and the legend needs a lane clear of
        # the top row's block.
        figsize=(11.4, 12.2),
        gridspec_kw={"height_ratios": (4.4, 3.6, 5.0)},
    )

    confidence = data["confidence_pct"]

    # ── Top: the accuracy-scale nulls ────────────────────────────────────────
    _draw_interval_panel(
        top,
        accuracy_rows,
        title=(
            f"{len(accuracy_rows)} accuracy-scale null"
            f"{'s' if len(accuracy_rows) != 1 else ''}, "
            "each bounded rather than absent"
        ),
        xlabel=(
            f"accuracy difference, probe minus comparator (percentage points; "
            f"whiskers: {confidence:g}% paired interval)"
        ),
        data_limits=(-4.6, 4.8),
        annotation_x=5.4,
        right_limit=12.2,
    )
    top.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLOR_BOUND,
                marker="o",
                linestyle="-",
                label=f"bounded effect ± {confidence:g}% paired interval",
            ),
            Line2D([], [], color=COLOR_INK, linestyle="--", label="no difference (structural 0)"),
        ],
        loc="upper right",
        fontsize=FONTS["cell"],
        frameon=False,
    )

    # ── Middle: the directional splits, with the established contrast ────────
    _draw_interval_panel(
        middle,
        share_rows,
        title="Two directional splits: one bounded near-null, one established",
        xlabel=(
            f"net share falling the probe's way, in percentage points "
            f"(whiskers: {confidence:g}% paired interval)"
        ),
        data_limits=(-24.0, 40.0),
        annotation_x=43.0,
        right_limit=92.0,
    )
    middle.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLOR_BOUND,
                marker="o",
                linestyle="-",
                label=f"below the fixed corrected α = {data['corrected_alpha']:.3g}",
            ),
            Line2D(
                [],
                [],
                color=COLOR_CONTRAST,
                marker="s",
                linestyle="-",
                label="established, drawn for scale",
            ),
        ],
        loc="upper right",
        fontsize=FONTS["cell"],
        frameon=False,
    )

    # ── Bottom: what it would have taken to say more than "bounded" ──────────
    all_rows = accuracy_rows + share_rows
    ys = list(range(len(all_rows) - 1, -1, -1))
    minimum = data["minimum_pairs"]
    minimum_corrected = data["minimum_pairs_corrected"]
    # The left limit was fixed at 1.2, one notch above the smallest count an arm
    # can produce, so the Stage 5 row's single discordant pair fell outside the
    # view and matplotlib dropped its marker without a word: the row rendered as
    # a label with no datum, which reads as a measurement that failed rather than
    # as the one-pair measurement it is. The limit now follows the data down.
    smallest_pairs = min(row["discordant"] for row in all_rows)
    left_limit, right_limit = min(0.7, smallest_pairs * 0.7), 3.0e5
    bottom.set_xscale("log")
    bottom.set_xlim(left_limit, right_limit)
    bottom.axvspan(left_limit, minimum, color=COLOR_BAND, alpha=0.30, zorder=0)
    bottom.axvline(minimum, color=COLOR_MUTED, linestyle=":", linewidth=1.6, zorder=1)
    bottom.axvline(minimum_corrected, color=COLOR_GATE, linestyle="-", linewidth=1.4, zorder=1)

    def _log_fraction(value: float) -> float:
        """Where ``value`` falls across the drawn log axis, in ``[0, 1]``."""
        import math

        span = math.log10(right_limit) - math.log10(left_limit)
        return (math.log10(value) - math.log10(left_limit)) / span

    for y, row in zip(ys, all_rows, strict=True):
        # Color keeps one meaning across all three panels: vermillion is the
        # established row wherever it appears, blue is a bounded-only row. Green
        # is the third mutually exclusive state — an arm whose interval fits
        # inside the equivalence region — which the guard above proves cannot
        # coincide with established. Reusing blue for "not equivalent" here would
        # have given one color two meanings between panels.
        color = COLOR_CONTRAST if row["established"] else COLOR_BOUND
        marker = "s" if row["established"] else "o"
        if row["equivalent"]:
            color, marker = COLOR_EQUIVALENT, "D"
        bottom.scatter([row["discordant"]], [y], color=color, s=64, zorder=3, marker=marker)
        if row["equivalent"]:
            badge = f"equivalent at {data['corrected_confidence_pct']:g}%"
        elif row["established"]:
            # "Interval too wide" would be the wrong reading for this row: it has
            # the pairs and a narrow interval, and misses the equivalence region
            # because the effect is real and large, not because it is unresolved.
            badge = "established effect, not an equivalence"
        elif row["discordant"] < minimum:
            badge = "no equivalence reachable"
        else:
            badge = "pairs suffice, interval too wide"
        # The badge sits under its own marker and runs away from the nearer edge.
        # A fixed annotation column cannot work on this axis: the pair counts span
        # five orders of magnitude, so any column left of the widest marker would
        # be drawn through by the narrower ones.
        past_middle = _log_fraction(row["discordant"]) > 0.5
        bottom.text(
            row["discordant"] * (0.86 if past_middle else 1.16),
            y - 0.30,
            f"{row['discordant']:,} · {badge}",
            va="top",
            ha="right" if past_middle else "left",
            fontsize=FONTS["cell"],
            color=COLOR_MUTED if (color is COLOR_BOUND) else color,
            # A badge anchored under its own marker necessarily spans one or both
            # design thresholds, which were drawn straight through the words "no
            # equivalence" and "not an equivalence". The x anchor carries the pair
            # count and cannot move, so the patch masks the rule behind the glyphs
            # — the remedy the subsample panel already uses for its p-labels. A
            # rule crossing a word is invisible to a text-versus-text sweep.
            bbox={
                "boxstyle": "square,pad=0.12",
                "facecolor": PALETTE["BACKGROUND"],
                "edgecolor": "none",
            },
            zorder=2,
        )
    bottom.set_yticks(ys)
    bottom.set_yticklabels([row["label"] for row in all_rows], fontsize=FONTS["footnote"])
    # Headroom for the two threshold labels, which belong above every row rather
    # than beside one of them: they are properties of the study's design, not of
    # any arm.
    bottom.set_ylim(-0.95, len(all_rows) + 0.15)
    bottom.text(
        minimum * 0.86,
        len(all_rows) - 0.45,
        f"{minimum:,} pairs\nneeded ({confidence:g}%)",
        ha="right",
        va="center",
        fontsize=FONTS["cell"],
        color=COLOR_MUTED,
    )
    bottom.text(
        minimum_corrected * 1.16,
        len(all_rows) - 0.45,
        f"{minimum_corrected:,} pairs\nneeded ({data['corrected_confidence_pct']:g}%)",
        ha="left",
        va="center",
        fontsize=FONTS["cell"],
        color=COLOR_GATE,
    )
    bottom.text(
        left_limit * 1.4,
        -0.72,
        "shaded: no equivalence claim reachable at any outcome",
        ha="left",
        va="center",
        fontsize=FONTS["cell"],
        color=COLOR_MUTED,
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": "none",
        },
        zorder=2,
    )
    bottom.set_xlabel("discordant pairs the arm produced (log scale)", fontsize=FONTS["tick_label"])
    bottom.set_title(
        f"The smallest effect the study could resolve, on the scale where one was "
        f"declared:\nequivalence needs the share interval inside "
        f"[{data['region_low']:g}, {data['region_high']:g}]",
        fontsize=FONTS["subtitle"],
        fontweight="bold",
    )
    bottom.spines[["top", "right"]].set_visible(False)

    # h_pad rather than a gridspec hspace: a fixed hspace makes the axes
    # incompatible with tight_layout, which then warns and ignores it.
    fig.tight_layout(h_pad=2.4)
    return fig
