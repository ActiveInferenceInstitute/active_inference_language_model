"""Forest plot of the multiplicity family, with the Stage 6 fragility strip.

One figure, two panels, both drawn from the claims ledger so nothing here can
drift from the measurements:

``multiplicity_forest`` (top panels)
    The four primary paired effects as a horizontal forest plot beside the
    statistics each row is read under. The forest draws the favored share of
    discordant pairs with two nested whiskers — the exact *marginal*
    Clopper–Pearson interval and, behind it, the family-corrected interval at
    ``1 - BONFERRONI_ALPHA`` whose exclusion of one half *is* the verdict — plus
    the McNemar null at the structural constant 1/2 (a constant of the test, not
    a measured number) and the fixed power reference share as a separately
    labeled line. The statistics column states, per row: the discordant ``n``
    and the population it is drawn from with its unit; whether a cluster-robust
    re-analysis covers the same population, and if so its clustering unit,
    cluster count and level, or explicitly that the interval is not
    cluster-adjusted; the exact power at the reference share; the corrected
    verdict badge; the Holm verdict; and which held-back population is
    registered to re-score the row, and whether that run has happened. Every
    verdict word is read from a recorded boolean, never inferred from the
    geometry beside it, and the panel footer states the survivor set — how many
    of the family survive Bonferroni, how many survive Holm, and how many Holm
    retains that Bonferroni does not.
``multiplicity_forest`` (bottom strip)
    The Stage 6 sign test's threshold-crossing geometry on a log p-axis: the
    two-sided family p and the one-sided pre-registered p against the
    familywise threshold, the Bonferroni per-test threshold, and the
    collapsed-family relaxed threshold (the familywise level over the family size
    that remains once the three nested subsample re-analyzes count as one — the
    second disclosed fragility direction), with the band where the corrected
    verdict is lost shaded. Its size and its label are both derived from
    ``SIGNIFICANCE_FAMILY``, never typed.

The falsifier is drawn as structure, in the ``complexity_ops`` pattern: the
data function re-derives the corrected Clopper–Pearson verdict from the
recorded *raw discordant counts* — never by inverting the rounded favored
ratio — through the same ``ailm.statistics`` code path the recorder used, and
refuses to draw if any re-derived verdict disagrees with the ledger's recorded
booleans, if a drawn marginal whisker straddles 1/2 while its corrected verdict
claims exclusion, or if the drawn p-threshold geometry contradicts the recorded
familywise flag.

Every generator returns an unwritten ``Figure``; ``scripts/generate_figures.py``
discovers them by the ``generate_`` prefix and handles saving.
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

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
    BONFERRONI_ALPHA,
    CONFIRMATORY_POPULATIONS,
    CONFIRMATORY_REPLICATION,
    FAMILYWISE_ALPHA,
    SHARE_CONFIDENCE,
    SIGNIFICANCE_FAMILY,
    clopper_pearson_interval,
    summarize_paired_effect,
)
from ailm.viz_config import FONTS, PALETTE, apply_style

__all__ = [
    "generate_multiplicity_forest",
    "multiplicity_forest_data",
]

COLOR_MODEL = PALETTE["MODEL"]
COLOR_ACCENT = PALETTE["ACCENT"]
COLOR_INK = PALETTE["INK"]
COLOR_MUTED = PALETTE["MUTED"]
COLOR_BAND = PALETTE["HIGHLIGHT"]

#: The McNemar / sign-test null share. A structural constant of the method —
#: under the null the favored side of a discordant pair is a fair coin — not a
#: measured number, so it is not a ledger token.
NULL_SHARE: float = 0.5


class _Effect(NamedTuple):
    """One forest row: what is compared, over what population, and under which
    correction.

    Every field is *method structure* — which test, over which unit, as which
    member of the frozen family — and none of it is a measurement, so it belongs
    in code beside the family declaration rather than in the claims ledger. The
    quantities these fields name are all read from tokens at draw time.

    Attributes:
        label: The row's y-axis label.
        prefix: Ledger token prefix for the ``STAT_*`` summary of this effect.
        favored_token: Raw discordant count on the side the claim names.
        opposed_token: Raw discordant count on the other side.
        family_label: This effect's member name in
            :data:`ailm.statistics.SIGNIFICANCE_FAMILY`, which is how the row is
            matched to the recorded survivor sets and to the confirmatory
            replication protocol.
        population_token: Token holding the evaluated population the discordant
            pairs are drawn from — the denominator a reader needs to size ``n``.
        unit: What that population is counted in. A discordant count with no
            unit is not a sample size, it is a number.
        cluster_prefix: Token prefix of the recorded cluster-robust re-analysis
            covering the same comparison, or ``None`` where no cluster arm was
            recorded and the interval is therefore not cluster-adjusted. Stating
            the absence is the point: an interval whose clustering is unstated
            reads as though independence had been established.
    """

    label: str
    prefix: str
    favored_token: str
    opposed_token: str
    family_label: str
    population_token: str
    unit: str
    cluster_prefix: str | None


#: The four primary paired effects. For the coupling row the measured effect is
#: *harm*, so ``c`` (uncoupled decoder correct) is the scientifically favored
#: side — same convention as the recorder.
_EFFECTS: tuple[_Effect, ...] = (
    _Effect(
        "Stage 2 G2P vs lookup table",
        "STAGE2",
        "STAGE2_G2P_MCNEMAR_B",
        "STAGE2_G2P_MCNEMAR_C",
        "stage2_g2p_mcnemar_p",
        "STAGE2_G2P_EVAL_POSITIONS",
        "positions",
        "CLUSTER_STAGE2_G2P",
    ),
    _Effect(
        "Stage 4 case vs UPOS bar",
        "STAGE4",
        "STAGE4_DERIVED_MCNEMAR_B",
        "STAGE4_DERIVED_MCNEMAR_C",
        "stage4_derived_mcnemar_p",
        "STAGE4_DERIVED_TEST_TOKENS",
        "tokens",
        "CLUSTER_STAGE4_DERIVED",
    ),
    _Effect(
        "Lexical channel vs anchor",
        "LEXICAL",
        "LEXICAL_BEST_MCNEMAR_B",
        "LEXICAL_BEST_MCNEMAR_C",
        "lexical_best_mcnemar_p",
        "LEXICAL_EVAL_POSITIONS",
        "positions",
        None,
    ),
    _Effect(
        "Coupling harm (uncoupled favored)",
        "COUPLING_HARM",
        "COUPLING_MCNEMAR_C",
        "COUPLING_MCNEMAR_B",
        "coupling_mcnemar_p",
        "COUPLING_EVAL_POSITIONS",
        "positions",
        None,
    ),
)

# A row that names a family member the declaration does not have would be drawn
# with a survivor verdict read from a set it cannot be in — silently "retained
# under Holm" for every row, forever. Fail at import instead.
if not {effect.family_label for effect in _EFFECTS} <= set(SIGNIFICANCE_FAMILY):
    raise RuntimeError(
        "every forest row must name a member of SIGNIFICANCE_FAMILY; "
        f"{sorted({e.family_label for e in _EFFECTS} - set(SIGNIFICANCE_FAMILY))} is not one."
    )

#: The confirmatory replication protocol as a lookup. Frozen in
#: :mod:`ailm.statistics`; read here so a row can say which held-back population
#: is registered to re-score it.
_DISPOSITIONS: dict[str, str] = dict(CONFIRMATORY_REPLICATION)

#: Tolerance for comparing a re-derived statistic against its ledger token. The
#: recorder rounds to six decimals before recording, so agreement must hold to
#: within one unit in the sixth decimal plus platform ulp noise in scipy's
#: incomplete-beta inversion.
_ROUNDING_TOL: float = 2e-6


def _token_bool(variables: dict[str, str], token: str) -> bool:
    """Read a boolean token, rejecting anything but the two rendered forms."""
    if token not in variables:
        raise MissingVariableError(f"Token {token!r} is not in the manuscript variable map.")
    raw = variables[token].strip().lower()
    if raw not in ("true", "false"):
        raise MissingVariableError(f"Token {token!r} has non-boolean value {variables[token]!r}.")
    return raw == "true"


def _token_text(variables: dict[str, str], token: str) -> str:
    """Read a non-numeric token, such as a recorded clustering unit."""
    if token not in variables:
        raise MissingVariableError(f"Token {token!r} is not in the manuscript variable map.")
    text = variables[token].strip()
    if not text:
        raise MissingVariableError(f"Token {token!r} is empty.")
    return text


def _token_labels(variables: dict[str, str], token: str, count_token: str) -> list[str]:
    """Read a recorded JSON list of family-member labels, checked both ways.

    The recorder writes the list and its length as separate claims. Reading only
    one of them lets a drawn "7 of 16 survive" disagree with the seven labels
    beside it, so both are read and required to agree, and every label must be a
    declared family member.

    Raises:
        MissingVariableError: If either token is absent, or the list token does
            not parse as a JSON array of strings.
        ValueError: If the list length disagrees with the recorded count, or a
            label is not in :data:`ailm.statistics.SIGNIFICANCE_FAMILY`.
    """
    raw = _token_text(variables, token)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise MissingVariableError(f"Token {token!r} is not JSON: {raw!r}.") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise MissingVariableError(f"Token {token!r} is not a JSON array of strings: {raw!r}.")
    recorded_count = token_int(variables, count_token)
    if len(parsed) != recorded_count:
        raise ValueError(
            f"{token} lists {len(parsed)} labels but {count_token} records "
            f"{recorded_count}. The survivor set and its count came from different "
            "runs; re-run the statistical recorder."
        )
    unknown = sorted(set(parsed) - set(SIGNIFICANCE_FAMILY))
    if unknown:
        raise ValueError(
            f"{token} names {unknown}, which are not members of SIGNIFICANCE_FAMILY. "
            "The recorded survivor set and the executable family have diverged."
        )
    return sorted(parsed)


def _population_is_scored(variables: dict[str, str], population: str) -> bool:
    """Whether the confirmatory run against ``population`` has been completed.

    A held-back population counts as scored only when the recorder wrote a
    completion flag for it. Absence is *not* scored — which is why this reads a
    missing token as ``False`` rather than raising: the registered
    ``external_lexicon`` arm has no completion claim precisely because it has not
    been run, and the figure has to be able to say so.
    """
    token = f"CONFIRMATORY_RUN_COMPLETED_{population.upper()}"
    return token in variables and variables[token].strip().lower() == "true"


def multiplicity_forest_data(variables: dict[str, str] | None = None) -> dict[str, Any]:
    """Ledger rows for the forest plot, cross-checked against a re-derivation.

    Every check below is a refuse-to-draw falsifier, not a redraw: the recorded
    values came from ``scripts/record_statistical_claims.py``, so a disagreement
    means the statistics module, the family declaration, or the ledger changed —
    a finding to investigate, never something to paper over in a figure.

    Raises:
        ValueError: If the executable family declaration has drifted from the
            recorded family metadata; if the corrected Clopper–Pearson verdict
            re-derived from the raw discordant counts (like-for-like, through
            ``summarize_paired_effect`` and ``clopper_pearson_interval`` at the
            code-constant Bonferroni alpha) disagrees with a recorded
            ``*_CORRECTED_CI_EXCLUDES_HALF`` boolean; if a re-derived marginal
            whisker or power disagrees with its drawn token beyond recorder
            rounding; if a recorded corrected verdict claims exclusion while the
            drawn marginal whisker straddles 1/2 (corrected level is stricter,
            so corrected-excludes must imply marginal-excludes); or if the Stage
            6 strip's drawn threshold geometry contradicts the recorded
            familywise flag or either disclosed fragility direction.
    """
    v = _v(variables)

    # The executable family declaration is the source of truth the recorder
    # used; the drawn metadata must still be that declaration.
    family_size = token_int(v, "STAT_FAMILY_SIZE")
    familywise_alpha = token_float(v, "STAT_FAMILYWISE_ALPHA")
    bonferroni_alpha = token_float(v, "STAT_BONFERRONI_ALPHA")
    if family_size != len(SIGNIFICANCE_FAMILY):
        raise ValueError(
            f"the ledger records a family of {family_size} but SIGNIFICANCE_FAMILY "
            f"declares {len(SIGNIFICANCE_FAMILY)}; re-run the statistical recorder."
        )
    if familywise_alpha != FAMILYWISE_ALPHA:
        raise ValueError(
            f"the ledger's familywise alpha {familywise_alpha} is not the declared "
            f"{FAMILYWISE_ALPHA}; re-run the statistical recorder."
        )
    # The token is the recorder's presentation rounding of the code constant.
    if abs(bonferroni_alpha - BONFERRONI_ALPHA) > BONFERRONI_ALPHA * 1e-2:
        raise ValueError(
            f"the ledger's Bonferroni alpha {bonferroni_alpha} is not a rounding of "
            f"the declared {BONFERRONI_ALPHA}; re-run the statistical recorder."
        )

    # The recorded survivor sets, which are what "corrected" means in the drawn
    # badges. Reading them here lets each row's badge be checked against the
    # family decision the recorder actually made, rather than against a rule
    # this module re-implements.
    bonferroni_survivors = _token_labels(v, "STAT_FAMILY_SURVIVORS", "STAT_FAMILY_SURVIVOR_COUNT")
    holm_survivors = _token_labels(v, "STAT_HOLM_SURVIVORS", "STAT_HOLM_SURVIVOR_COUNT")
    if not set(bonferroni_survivors) <= set(holm_survivors):
        raise ValueError(
            "the recorded Bonferroni survivors are not a subset of the Holm survivors "
            f"({sorted(set(bonferroni_survivors) - set(holm_survivors))} is Bonferroni-only), "
            "which Holm's step-down construction makes impossible. Refusing to draw."
        )
    holm_only = sorted(set(holm_survivors) - set(bonferroni_survivors))
    recorded_holm_only = token_int(v, "STAT_HOLM_ONLY_COUNT")
    if len(holm_only) != recorded_holm_only:
        raise ValueError(
            f"the two recorded survivor sets differ on {len(holm_only)} members but "
            f"STAT_HOLM_ONLY_COUNT records {recorded_holm_only}; re-run the recorder."
        )

    # The confirmatory protocol's completed populations, cross-checked against
    # the recorded count of scored members. A figure that marks a row "held back,
    # scored" from a completion flag the recorder never counted is drawing a
    # replication that did not happen.
    scored_members = sorted(
        member
        for member, disposition in CONFIRMATORY_REPLICATION
        if disposition in CONFIRMATORY_POPULATIONS and _population_is_scored(v, disposition)
    )
    recorded_scored = token_int(v, "CONFIRMATORY_MEMBERS_SCORED")
    if len(scored_members) != recorded_scored:
        raise ValueError(
            f"the protocol maps {len(scored_members)} family members to a completed "
            f"confirmatory population but CONFIRMATORY_MEMBERS_SCORED records "
            f"{recorded_scored}. Refusing to mark a replication the ledger does not count."
        )

    rows: list[dict[str, Any]] = []
    for effect in _EFFECTS:
        label = effect.label
        prefix = effect.prefix
        favored = token_int(v, effect.favored_token)
        opposed = token_int(v, effect.opposed_token)
        # Like-for-like re-derivation from the RAW counts, through the exact
        # code path the recorder used. The rounded favored ratio is never
        # inverted: a value rounded for presentation cannot be inverted back
        # into the counts that produced it.
        summary = summarize_paired_effect(favored, opposed)
        corrected_low, corrected_high = clopper_pearson_interval(
            favored, summary.n, confidence=1.0 - BONFERRONI_ALPHA
        )
        recomputed_excludes = corrected_low > NULL_SHARE
        ledger_excludes = _token_bool(v, f"STAT_{prefix}_CORRECTED_CI_EXCLUDES_HALF")
        if recomputed_excludes != ledger_excludes:
            raise ValueError(
                f"{label}: the corrected Clopper–Pearson interval recomputed from the "
                f"raw counts ({favored}/{summary.n}) gives excludes-1/2 = "
                f"{recomputed_excludes}, but the ledger records {ledger_excludes}. "
                f"Refusing to draw a verdict the counts no longer support."
            )

        ratio = token_float(v, f"STAT_{prefix}_FAVORED_RATIO")
        ci_low = token_float(v, f"STAT_{prefix}_SHARE_CI_LOW")
        ci_high = token_float(v, f"STAT_{prefix}_SHARE_CI_HIGH")
        power = token_float(v, f"STAT_{prefix}_POWER_AT_REFERENCE")
        n = token_int(v, f"STAT_{prefix}_DISCORDANT_N")
        derived = {
            "favored ratio": (summary.ratio, ratio),
            "share CI low": (summary.ci_low, ci_low),
            "share CI high": (summary.ci_high, ci_high),
            "power at reference": (summary.power_at_reference, power),
            "discordant n": (float(summary.n), float(n)),
        }
        for name, (recomputed, drawn) in derived.items():
            if abs(recomputed - drawn) > _ROUNDING_TOL:
                raise ValueError(
                    f"{label}: {name} recomputed from the raw counts is {recomputed!r} "
                    f"but the drawn token says {drawn!r}. The ledger and the statistics "
                    f"module have diverged; re-run the recorder and find out which is wrong."
                )
        # Implication direction only, on the drawn geometry: the corrected
        # interval is wider than the marginal one, so corrected-excludes
        # must imply that the visible whisker clears 1/2.
        if ledger_excludes and ci_low <= NULL_SHARE:
            raise ValueError(
                f"{label}: the ledger says the corrected interval excludes 1/2 but the "
                f"drawn marginal whisker [{ci_low}, {ci_high}] straddles it. The drawing "
                f"would contradict the recorded verdict; refusing to draw."
            )

        # ── What the n is an n OF ────────────────────────────────────────────
        # A discordant count without its evaluated population is not a sample
        # size. More discordant pairs than evaluated units is arithmetically
        # impossible and means the two claims describe different populations.
        population = token_int(v, effect.population_token)
        if n > population:
            raise ValueError(
                f"{label}: {n} discordant pairs over {population} evaluated "
                f"{effect.unit} is impossible. {effect.population_token} and "
                f"STAT_{prefix}_DISCORDANT_N describe different populations."
            )

        # ── The clustering the interval does or does not carry ───────────────
        cluster: dict[str, Any] | None = None
        if effect.cluster_prefix is not None:
            cluster_count = token_int(v, f"{effect.cluster_prefix}_CLUSTERS")
            cluster_observations = token_int(v, f"{effect.cluster_prefix}_OBSERVATIONS")
            if cluster_observations != population:
                raise ValueError(
                    f"{label}: the cluster arm covers {cluster_observations} observations "
                    f"but the paired test covers {population} {effect.unit}. A "
                    "cluster-robust re-analysis of a different population is not this "
                    "row's clustering; refusing to name it as such."
                )
            if not 0 < cluster_count <= cluster_observations:
                raise ValueError(
                    f"{label}: {cluster_count} clusters over {cluster_observations} "
                    "observations is not a partition of the evaluated population."
                )
            cluster = {
                "unit": _token_text(v, f"{effect.cluster_prefix}_CLUSTER_UNIT"),
                "count": cluster_count,
                "confidence": token_float(v, "CLUSTER_CONFIDENCE"),
            }

        # ── Which correction retains this row, and what replicates it ────────
        # The exact conditional test and its Clopper–Pearson interval are duals:
        # the corrected interval excludes 1/2 exactly when the exact p clears the
        # per-test alpha the interval's level was built from. So the badge this
        # figure draws and the survivor set the recorder wrote must agree, and a
        # disagreement means one of the two was computed from stale counts.
        bonferroni_survivor = effect.family_label in bonferroni_survivors
        if ledger_excludes != bonferroni_survivor:
            raise ValueError(
                f"{label}: the recorded corrected interval says excludes-1/2 = "
                f"{ledger_excludes}, but the recorded Bonferroni survivor set "
                f"{'contains' if bonferroni_survivor else 'omits'} "
                f"{effect.family_label}. The interval and the p-value decision are duals "
                "and cannot disagree; re-run the statistical recorder."
            )
        disposition = _DISPOSITIONS[effect.family_label]
        rows.append(
            {
                "label": label,
                "ratio": ratio,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "corrected_low": corrected_low,
                "corrected_high": corrected_high,
                "population": population,
                "unit": effect.unit,
                "cluster": cluster,
                "family_label": effect.family_label,
                "bonferroni_survivor": bonferroni_survivor,
                "holm_survivor": effect.family_label in holm_survivors,
                "disposition": disposition,
                "disposition_scored": disposition in CONFIRMATORY_POPULATIONS
                and _population_is_scored(v, disposition),
                "disposition_rescorable": disposition in CONFIRMATORY_POPULATIONS,
                "n": n,
                "power": power,
                "corrected_excludes": ledger_excludes,
            }
        )

    # ── Stage 6 fragility strip ──────────────────────────────────────────────
    two_sided_p = token_float(v, "STAGE6_SIGN_TEST_P")
    one_sided_p = token_float(v, "STAT_STAGE6_SIGN_TEST_ONE_SIDED_P")
    confirmed = _token_bool(v, "STAT_STAGE6_FAMILYWISE_CONFIRMED")
    # The second disclosed fragility direction (10_verification.md): counting the
    # three nested subsample re-analyzes as one member. Collapsing three members
    # of an eleven-member family into one leaves NINE, not eight — the earlier
    # figure divided by eight, which made the relaxed threshold looser than the
    # collapse rule allows and turned a non-crossing into a crossing. Derived in
    # code from the familywise constant; not a measurement.
    # Derived from the family definition rather than typed, so the collapse rule
    # cannot drift away from SIGNIFICANCE_FAMILY if a member is added or removed.
    nested_subsample_members = 3
    collapsed_family_size = len(SIGNIFICANCE_FAMILY) - (nested_subsample_members - 1)
    # Named for what it is, not for how many members it had when it was written.
    # The key was ``family_of_9_alpha`` until the channel factorial grew the family
    # from 11 to 13 and the collapsed size from 9 to 11; the value tracked the
    # declaration correctly and only the NAME went stale, which is the quietest
    # way for a figure to start lying about what it draws.
    collapsed_family_alpha = FAMILYWISE_ALPHA / collapsed_family_size

    # The strip's verdict must be readable off the drawn geometry: the marker
    # is left of the Bonferroni line exactly when the flag says confirmed.
    if (two_sided_p < bonferroni_alpha) != confirmed:
        raise ValueError(
            f"the drawn Stage 6 geometry (two-sided p {two_sided_p} vs Bonferroni "
            f"alpha {bonferroni_alpha}) contradicts the recorded familywise flag "
            f"{confirmed}. Refusing to draw a crossing the ledger denies."
        )
    # The strip exists to show the two disclosed fragility crossings and the
    # nominal significance they qualify. If any of the three stops holding, the
    # picture's story is wrong and the manuscript's disclosure needs rewriting
    # first — refuse to draw rather than illustrate a stale claim.
    if not one_sided_p < bonferroni_alpha:
        raise ValueError(
            f"the one-sided p {one_sided_p} no longer crosses the Bonferroni alpha "
            f"{bonferroni_alpha}: the disclosed sidedness fragility has changed."
        )
    # Note the direction. At the correct collapsed size the two-sided p does NOT
    # cross, so the family-size relaxation does not rescue Stage 6 and only the
    # sidedness relaxation does. If that ever changes, the disclosure in
    # 10_verification.md is wrong and must be rewritten before this figure is
    # redrawn — so the guard fires on a crossing rather than on its absence.
    if two_sided_p < collapsed_family_alpha:
        raise ValueError(
            f"the two-sided p {two_sided_p} now crosses the "
            f"family-of-{collapsed_family_size} alpha "
            f"{collapsed_family_alpha}: the disclosed family-size fragility has changed "
            f"direction and the manuscript says it does not cross."
        )
    if not two_sided_p < familywise_alpha:
        raise ValueError(
            f"the two-sided p {two_sided_p} is not even nominally significant at "
            f"{familywise_alpha}: the Stage 6 result itself has changed."
        )

    # Holm's threshold for this member. It is the line that decides the verdict
    # the strip is about, so a strip drawn without it shows the p-value losing
    # against every threshold except the one it wins.
    holm_alpha = token_float(v, "STAT_HOLM_STAGE6_THRESHOLD")
    holm_rejected = _token_bool(v, "STAT_HOLM_STAGE6_REJECTED")
    if (two_sided_p < holm_alpha) != holm_rejected:
        raise ValueError(
            f"the drawn Stage 6 geometry (two-sided p {two_sided_p} vs Holm "
            f"threshold {holm_alpha}) contradicts the recorded Holm verdict "
            f"{holm_rejected}. Refusing to draw a crossing the ledger denies."
        )
    if not bonferroni_alpha <= holm_alpha:
        raise ValueError(
            f"Holm's threshold {holm_alpha} is below Bonferroni's {bonferroni_alpha}, "
            "which is impossible: Holm's smallest threshold is Bonferroni's."
        )

    # The strip's own unit and n. A p-value on a log axis with no sample size
    # beside it is the commonest way a fragility picture overstates its case.
    stage6_positive = token_int(v, "STAGE6_SIGN_TEST_POSITIVE")
    stage6_negative = token_int(v, "STAGE6_SIGN_TEST_NEGATIVE")
    stage6_discordant = token_int(v, "STAGE6_SIGN_DISCORDANT")
    stage6_sentences = token_int(v, "STAGE6_TEST_SENTENCES")
    stage6_documents = token_int(v, "STAGE6_TEST_DOCUMENTS")
    if stage6_positive + stage6_negative != stage6_discordant:
        raise ValueError(
            f"the Stage 6 sign test records {stage6_positive} favored and "
            f"{stage6_negative} opposed but a discordant total of {stage6_discordant}; "
            "the counts and the total came from different runs."
        )
    if stage6_discordant > stage6_sentences:
        raise ValueError(
            f"{stage6_discordant} discordant sign-test pairs over "
            f"{stage6_sentences} evaluated sentences is impossible."
        )

    return {
        "rows": rows,
        "family_size": family_size,
        "bonferroni_survivors": bonferroni_survivors,
        "holm_survivors": holm_survivors,
        "holm_only": holm_only,
        "scored_members": scored_members,
        "corrected_confidence": 1.0 - BONFERRONI_ALPHA,
        "familywise_alpha": familywise_alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "holm_alpha": holm_alpha,
        "holm_rejected": holm_rejected,
        "collapsed_family_alpha": collapsed_family_alpha,
        "collapsed_family_size": collapsed_family_size,
        "reference_share": token_float(v, "STAT_REFERENCE_SHARE"),
        "stage6": {
            "two_sided_p": two_sided_p,
            "one_sided_p": one_sided_p,
            "familywise_confirmed": confirmed,
            "discordant": stage6_discordant,
            "sentences": stage6_sentences,
            "documents": stage6_documents,
        },
    }


def generate_multiplicity_forest(variables: dict[str, str] | None = None) -> Figure:
    """Forest plot of the four paired effects plus the Stage 6 fragility strip."""
    apply_style()
    data = multiplicity_forest_data(variables)
    rows = data["rows"]

    # Three panels: the forest, the statistics column that reads off it, and the
    # Stage 6 strip beneath both. The statistics live in their own axes rather
    # than inside the data range because a forest row now carries five facts —
    # population, interval level, clustering, corrected verdict, replication
    # status — and text that long inside the plot either overprints the whiskers
    # or squeezes the scale until the intervals stop being comparable.
    corrected_pct = data["corrected_confidence"] * 100.0
    fig = plt.figure(figsize=(15.4, 8.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.22, 1.0), height_ratios=(2.6, 1.0))
    ax = fig.add_subplot(grid[0, 0])
    stats = fig.add_subplot(grid[0, 1])
    strip = fig.add_subplot(grid[1, :])

    # ── Top left: the forest ─────────────────────────────────────────────────
    ys = list(range(len(rows) - 1, -1, -1))
    # The corrected interval is drawn first and wider: it is the one the verdict
    # badge is read from, and a figure that shows only the marginal whisker
    # invites the reader to judge a corrected claim against an uncorrected
    # picture. Re-derived from the recorded raw counts through the same
    # ``ailm.statistics`` path the recorder used, and refused above if its
    # exclusion of 1/2 disagrees with the recorded verdict.
    ax.errorbar(
        [row["ratio"] for row in rows],
        ys,
        xerr=[
            [row["ratio"] - row["corrected_low"] for row in rows],
            [row["corrected_high"] - row["ratio"] for row in rows],
        ],
        fmt="none",
        ecolor=COLOR_MODEL,
        elinewidth=1.1,
        capsize=8,
        alpha=0.55,
        zorder=2,
    )
    ax.errorbar(
        [row["ratio"] for row in rows],
        ys,
        xerr=[
            [row["ratio"] - row["ci_low"] for row in rows],
            [row["ci_high"] - row["ratio"] for row in rows],
        ],
        fmt="o",
        color=COLOR_MODEL,
        ecolor=COLOR_MODEL,
        elinewidth=2.6,
        capsize=4,
        markersize=8,
        zorder=3,
    )
    ax.axvline(NULL_SHARE, color=COLOR_INK, linestyle="--", linewidth=1.4, zorder=1)
    ax.axvline(data["reference_share"], color=COLOR_ACCENT, linestyle=":", linewidth=1.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=FONTS["tick_label"])
    ax.set_xlim(0.42, 0.96)
    ax.set_ylim(-1.05, 4.15)
    ax.set_xlabel(
        "favored share of discordant pairs",
        fontsize=FONTS["tick_label"],
    )
    ax.set_title(
        "Four paired effects on the discordant-share scale",
        fontsize=FONTS["subtitle"],
        fontweight="bold",
    )
    # The survivor set, stated. "All four corrected intervals exclude one half"
    # is only meaningful next to how many of the whole family do, and the count
    # of members Holm retains that Bonferroni does not is the standard reason to
    # doubt a Bonferroni-only reading.
    holm_only_clause = (
        "no member is retained by Holm alone"
        if not data["holm_only"]
        else f"{len(data['holm_only'])} retained by Holm alone"
    )
    ax.text(
        0.425,
        -0.62,
        f"family: {data['family_size']} tests · familywise α = "
        f"{data['familywise_alpha']:g} · Bonferroni per-test α = {data['bonferroni_alpha']:g}"
        f"\nsurvive correction: {len(data['bonferroni_survivors'])} of "
        f"{data['family_size']} at Bonferroni, {len(data['holm_survivors'])} of "
        f"{data['family_size']} under Holm · {holm_only_clause}"
        f"\nheld-back replication scored so far: {len(data['scored_members'])} of "
        f"{data['family_size']} members",
        va="center",
        ha="left",
        fontsize=FONTS["cell"],
        color=COLOR_MUTED,
        # This note starts left of the structural null and runs most of the panel
        # width, so both reference rules were drawn through it — the null through
        # "16 tests", the power reference through "familywise". It belongs below
        # the rows and spanning the axis, so the patch masks the rules behind the
        # glyphs rather than moving the note somewhere worse. A rule crossing a
        # word is invisible to a text-versus-text overlap sweep.
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": "none",
        },
        zorder=5,
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLOR_MODEL,
                marker="o",
                linestyle="-",
                linewidth=2.6,
                label=f"favored share ± exact marginal {SHARE_CONFIDENCE * 100:g}% CI",
            ),
            Line2D(
                [],
                [],
                color=COLOR_MODEL,
                linestyle="-",
                linewidth=1.1,
                alpha=0.55,
                label=f"family-corrected {corrected_pct:.4g}% CI (= 1 − Bonferroni α)",
            ),
            Line2D([], [], color=COLOR_INK, linestyle="--", label="McNemar null (structural 1/2)"),
            Line2D(
                [],
                [],
                color=COLOR_ACCENT,
                linestyle=":",
                label=f"power reference share ({data['reference_share']:g})",
            ),
        ],
        loc="upper right",
        fontsize=FONTS["cell"],
        frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)

    # ── Top right: the statistics each row is read under ─────────────────────
    stats.set_axis_off()
    stats.set_xlim(0.0, 1.0)
    stats.set_ylim(*ax.get_ylim())
    _COL_LEFT = 0.01
    _COL_RIGHT = 0.53
    stats.text(
        _COL_LEFT,
        3.80,
        "population and precision",
        ha="left",
        va="center",
        fontsize=FONTS["cell"],
        fontweight="bold",
    )
    stats.text(
        _COL_RIGHT,
        3.80,
        "corrected verdict and replication",
        ha="left",
        va="center",
        fontsize=FONTS["cell"],
        fontweight="bold",
    )
    for y, row in zip(ys, rows, strict=True):
        cluster = row["cluster"]
        if cluster is None:
            clustering = (
                f"not cluster-adjusted: exact conditional test\nover {row['unit']}, "
                "no cluster arm recorded"
            )
        else:
            clustering = (
                f"cluster-robust arm: {cluster['count']:,} {cluster['unit']} clusters\n"
                f"at {cluster['confidence'] * 100:g}% over the same {row['unit']}"
            )
        stats.text(
            _COL_LEFT,
            y,
            f"{row['n']:,} discordant pairs of {row['population']:,} {row['unit']}\n"
            f"{clustering}\n"
            f"exact power {row['power']:.2f} at reference share "
            f"{data['reference_share']:g}",
            ha="left",
            va="center",
            fontsize=FONTS["cell"],
            linespacing=1.5,
        )
        # Verdict words, each read from a recorded boolean rather than from the
        # geometry beside it.
        badge = "excludes 1/2" if row["corrected_excludes"] else "includes 1/2"
        holm = "rejects" if row["holm_survivor"] else "retains the null"
        if not row["disposition_rescorable"]:
            replication = f"held back: {row['disposition']}, not re-scorable"
        elif row["disposition_scored"]:
            replication = f"held back: {row['disposition']}, scored"
        else:
            replication = f"held back: {row['disposition']}, not yet scored"
        stats.text(
            _COL_RIGHT,
            y,
            f"corrected {corrected_pct:.4g}% CI {badge}\n"
            f"Holm at familywise {data['familywise_alpha']:g}: {holm}\n"
            f"{replication}",
            ha="left",
            va="center",
            fontsize=FONTS["cell"],
            linespacing=1.5,
            color=COLOR_INK if row["corrected_excludes"] else COLOR_MUTED,
        )

    # ── Bottom: the Stage 6 fragility strip ──────────────────────────────────
    stage6 = data["stage6"]
    bon = data["bonferroni_alpha"]
    fam = data["familywise_alpha"]
    collapsed_alpha = data["collapsed_family_alpha"]
    collapsed = data["collapsed_family_size"]
    strip.set_xscale("log")
    strip.set_xlim(2.2e-4, 2.4e-2)
    strip.set_ylim(0.0, 1.12)
    strip.set_yticks([])
    strip.axvspan(bon, data["holm_alpha"], color=COLOR_BAND, alpha=0.30, zorder=0)
    strip.axvline(bon, color=COLOR_INK, linestyle="-", linewidth=1.4, zorder=1)
    strip.axvline(fam, color=COLOR_INK, linestyle="--", linewidth=1.4, zorder=1)
    strip.axvline(collapsed_alpha, color=COLOR_MUTED, linestyle=":", linewidth=1.6, zorder=1)
    # Holm's threshold for this member, drawn in the accent hue because it is the
    # one line the two-sided p clears. The shaded band now runs from Bonferroni
    # to Holm: that is the interval in which the verdict depends on which of two
    # equally valid familywise corrections is used.
    holm_alpha = data["holm_alpha"]
    strip.axvline(holm_alpha, color=COLOR_ACCENT, linestyle="-", linewidth=1.8, zorder=2)
    strip.scatter([stage6["two_sided_p"]], [0.40], color=COLOR_MODEL, s=70, zorder=3, marker="o")
    strip.scatter(
        [stage6["one_sided_p"]],
        [0.40],
        facecolors="white",
        edgecolors=COLOR_ACCENT,
        linewidths=2.0,
        s=70,
        zorder=3,
        marker="D",
    )
    strip.legend(
        handles=[
            Line2D([], [], color=COLOR_MODEL, marker="o", linestyle="None", label="two-sided p"),
            Line2D(
                [],
                [],
                color=COLOR_ACCENT,
                marker="D",
                markerfacecolor="white",
                linestyle="None",
                label="one-sided p",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        fontsize=FONTS["cell"],
        frameon=False,
    )
    strip.text(bon * 0.95, 0.90, f"Bonferroni α = {bon:g}", ha="right", fontsize=FONTS["cell"])
    strip.text(fam * 0.95, 0.90, f"familywise α = {fam:g}", ha="right", fontsize=FONTS["cell"])
    strip.text(
        collapsed_alpha * 1.06,
        0.76,
        f"family-of-{collapsed} α = {collapsed_alpha:g}",
        ha="left",
        fontsize=FONTS["cell"],
        color=COLOR_MUTED,
        # Anchored just right of its own rule, this label is wide enough to reach
        # the Holm rule, which struck through the digits of the alpha it prints.
        # The anchor says which threshold the label names, so it cannot move.
        bbox={
            "boxstyle": "square,pad=0.15",
            "facecolor": PALETTE["BACKGROUND"],
            "edgecolor": "none",
        },
        zorder=5,
    )
    strip.text(
        stage6["one_sided_p"] * 0.94,
        0.10,
        f"one-sided p = {stage6['one_sided_p']:g}\n(pre-registered direction)",
        ha="right",
        va="bottom",
        fontsize=FONTS["cell"],
        color=COLOR_ACCENT,
    )
    strip.text(
        stage6["two_sided_p"] * 1.07,
        0.18,
        f"two-sided p = {stage6['two_sided_p']:g}",
        ha="left",
        va="bottom",
        fontsize=FONTS["cell"],
        color=COLOR_MODEL,
    )
    strip.text(
        holm_alpha * 1.06,
        0.60,
        f"Holm α = {holm_alpha:g}",
        ha="left",
        fontsize=FONTS["cell"],
        fontweight="bold",
        color=COLOR_ACCENT,
    )
    strip.set_xlabel(
        f"p value, log scale · exact sign test over {stage6['discordant']:,} discordant "
        f"pairs of {stage6['sentences']:,} evaluated sentences in "
        f"{stage6['documents']:,} documents",
        fontsize=FONTS["tick_label"],
    )
    # The title states the recorded verdicts rather than a remembered outcome:
    # the Holm bar moves whenever another family member is re-recorded, and
    # this test crossed it before the lexical backoff correction (decision 110)
    # and does not after it.
    if stage6["familywise_confirmed"]:
        strip_title = "Stage 6 sign test: clears both the fixed Bonferroni threshold and Holm's"
    elif data["holm_rejected"]:
        strip_title = "Stage 6 sign test: clears Holm's threshold, not the fixed Bonferroni one"
    else:
        strip_title = (
            "Stage 6 sign test: clears neither Holm's threshold nor the fixed Bonferroni one"
        )
    strip.set_title(strip_title, fontsize=FONTS["subtitle"], fontweight="bold")
    strip.spines[["top", "right", "left"]].set_visible(False)

    # h_pad rather than a gridspec hspace: a fixed hspace makes the axes
    # incompatible with tight_layout, which then warns and ignores it.
    fig.tight_layout(h_pad=2.6, w_pad=0.6)
    return fig
