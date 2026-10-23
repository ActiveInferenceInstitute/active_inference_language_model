"""Ledger tokens, derived quantities, paper metadata, and the top-level orchestrator."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import (
    Any,
)

from ailm.benchmarks.constants import (
    BASE_MODEL_PROBE_KEYS,
    PROBE_DISPLAY_NAMES,
    PROBE_MODEL_TAGS,
    SERVED_PROBE_COUNT,
    SERVED_PROBES,
)
from ailm.evaluation import STAGE1_EVALUATION
from ailm.gates import HISTORICAL_STAGE2_G2P_THRESHOLD_PCT
from ailm.statistics import (
    BONFERRONI_ALPHA,
    CONFIRMATORY_REPLICATION,
    RETROSPECTIVE_POSTHOC_MEMBERS,
    SIGNIFICANCE_COUNT_LABELS,
    SUBSAMPLE_LARGER_STRIDE,
    SUBSAMPLE_MATCHED_STRIDE,
    exact_two_sided_rejection_region,
)
from ailm.value_formatting import (
    ManuscriptVariableError,
)
from ailm.value_formatting import (
    format_ledger_value as _format_value,
)

from ._claims import _ClaimReaders
from ._formatting import (
    _FAMILY_MEMBER_PHRASES,
    _build_timestamp,
    _count_word,
    _join_prose,
    _ordinal_word,
)
from ._io import (
    DIGEST_SHORT_LENGTH,
    _StrictVariableMap,
)
from ._stats import (
    AUTHORED_MANUSCRIPT_PROBE_KEYS,
    DERIVED_POWER_ARM_PREFIXES,
    _whole,
)
from .architecture import (
    _produce_architecture_tallies,
    _produce_calibration_rows,
    _produce_fitted_and_stage5,
    _produce_lexical_sweep,
)
from .connected import _produce_connected_gum
from .context import (
    _produce_context_protocol,
    _produce_context_results,
)
from .literature import (
    _load_claims,
    _load_config,
    _load_literal_variables,
    _load_literature_variables,
    _load_reference_variables,
    _load_rendering_variables,
    _vendor_metadata,
)
from .null_bounds import (
    _produce_derived_arm_power,
    _produce_null_bound_conventions,
    _produce_null_bounds,
)
from .scaling import _produce_stage3_scaling_ladder
from .synthetic import (
    _g2p_bucket_variables,
    _produce_aligned_g2p_comparator,
    _produce_synthetic_context_results,
    _produce_synthetic_phone_results,
    _produce_synthetic_results,
)


def _produce_ledger_tokens(
    variables: _StrictVariableMap, claims: Mapping[str, Any], project_root: Path
) -> None:
    """Every ledger label uppercased with its provenance companions, then the
    audited metadata registries, the frozen method configuration, the G2P
    bucket registry, and the Stage 1 evaluation fixture.
    """
    for label, entry in claims.items():
        variables[label.upper()] = _format_value(entry.get("value"))
        # Provenance companions so the manuscript can cite where a number came
        # from without a second lookup table.
        variables[f"{label.upper()}_RUN"] = str(entry.get("run_id", ""))
        variables[f"{label.upper()}_SOURCE"] = str(entry.get("source", ""))

    # Literature-search metadata is injected from the auditable search record,
    # including its incompleteness/priority policy, rather than a free literal.
    variables.add_many(
        _load_literature_variables(project_root), producer="literature-search metadata"
    )

    # Declared non-measurement variables use their YAML key directly.  They are
    # inserted after claims through the same strict map, so they cannot shadow a
    # recorded value or one of its provenance companions.
    variables.add_many(_load_literal_variables(project_root), producer="literal registry")

    # Exact source identity and PDF locators come from the deposited-source
    # snapshot itself, never from a second literal registry.
    variables.add_many(
        _load_reference_variables(project_root), producer="reference-source metadata"
    )

    # Frozen method configuration is injected from the same executable
    # constants used by the recorder and figure.  These are not measurements and
    # therefore do not belong in the claims ledger, but they must never be typed
    # independently into manuscript prose.
    variables["SUBSAMPLE_MATCHED_STRIDE"] = str(SUBSAMPLE_MATCHED_STRIDE)
    variables["SUBSAMPLE_LARGER_STRIDE"] = str(SUBSAMPLE_LARGER_STRIDE)
    variables["HISTORICAL_STAGE2_GATE_PCT"] = _format_value(HISTORICAL_STAGE2_G2P_THRESHOLD_PCT)
    variables["DIGEST_SHORT_LENGTH"] = str(DIGEST_SHORT_LENGTH)
    variables.add_many(_g2p_bucket_variables(), producer="G2P bucket registry")

    # The Stage 1 evaluation text is a versioned protocol fixture, not a test
    # literal. Its disclosure tokens come from the same authenticated object
    # that every scorer and recorder consumes.
    stage1_fixture = STAGE1_EVALUATION.metadata()
    variables["STAGE1_EVALUATION_PROTOCOL"] = str(stage1_fixture["protocol"])
    variables["STAGE1_EVALUATION_SHA256"] = str(stage1_fixture["sha256"])
    variables["STAGE1_EVALUATION_SHA256_SHORT"] = str(stage1_fixture["sha256"])[
        :DIGEST_SHORT_LENGTH
    ]
    variables["STAGE1_EVALUATION_SIZE_BYTES"] = str(stage1_fixture["size_bytes"])
    variables["STAGE1_EVALUATION_REPETITIONS"] = str(stage1_fixture["repetitions"])
    variables["STAGE1_EVALUATION_SOURCE"] = str(stage1_fixture["source"])
    variables["STAGE1_EVALUATION_LICENSE"] = str(stage1_fixture["license"])


def _produce_derived_quantities(variables: _StrictVariableMap, readers: _ClaimReaders) -> None:
    """Derived quantities the prose needs but the ledger stores in parts: the
    recorded G2P split protocol's ordinal words, the structurally unrejectable
    significance members, the unigram margins, and the case-path headroom
    captures.
    """
    num = readers.num

    from ailm.levels.phoneme import G2P_FAMILY_PREFIX_LENGTH, G2P_FAMILY_STRIDE

    _split_stride = _whole("stage2_g2p_split_stride", num("stage2_g2p_split_stride"))
    _family_prefix_length = _whole(
        "stage2_g2p_famsplit_prefix_length", num("stage2_g2p_famsplit_prefix_length")
    )
    _family_stride = _whole(
        "stage2_g2p_famsplit_family_stride", num("stage2_g2p_famsplit_family_stride")
    )
    if (_family_prefix_length, _family_stride) != (
        G2P_FAMILY_PREFIX_LENGTH,
        G2P_FAMILY_STRIDE,
    ):
        raise ManuscriptVariableError(
            "recorded G2P family-split protocol disagrees with the split implementation: "
            f"ledger={(_family_prefix_length, _family_stride)}, "
            f"code={(G2P_FAMILY_PREFIX_LENGTH, G2P_FAMILY_STRIDE)}"
        )
    variables["STAGE2_G2P_SPLIT_STRIDE_ORDINAL"] = _ordinal_word(_split_stride)
    variables["STAGE2_G2P_FAMSPLIT_FAMILY_STRIDE_ORDINAL"] = _ordinal_word(_family_stride)

    _posthoc_count = len(RETROSPECTIVE_POSTHOC_MEMBERS)
    variables["STAT_POSTHOC_MEMBER_COUNT"] = str(_posthoc_count)
    variables["STAT_POSTHOC_MEMBER_COUNT_WORD"] = _count_word(_posthoc_count)
    _unrejectable_members: list[str] = []
    for _p_label, _favored_label, _opposed_label in SIGNIFICANCE_COUNT_LABELS:
        _discordant = _whole(_favored_label, num(_favored_label)) + _whole(
            _opposed_label, num(_opposed_label)
        )
        if _discordant == 0:
            _unrejectable_members.append(_p_label)
            continue
        _lower, _upper = exact_two_sided_rejection_region(_discordant, BONFERRONI_ALPHA)
        if _lower < 0 and _upper > _discordant:
            _unrejectable_members.append(_p_label)
    _unrejectable_count = len(_unrejectable_members)
    variables["STAT_STRUCTURALLY_UNREJECTABLE_COUNT"] = str(_unrejectable_count)
    variables["STAT_STRUCTURALLY_UNREJECTABLE_COUNT_WORD"] = _count_word(_unrejectable_count)

    # The confirmatory replication's member accounting. The scored count is a
    # recorded ledger value; the lexicon and split-diagnostic composition comes
    # from the frozen CONFIRMATORY_REPLICATION registry, cross-checked against
    # the recorded family size so the prose cannot describe a family the
    # registry no longer maps. The unrescored complement is derived here too,
    # so the introduction, discussion, and conclusion restate one registry
    # instead of three hand-typed tallies.
    _scored_member_count = _whole("confirmatory_members_scored", num("confirmatory_members_scored"))
    _recorded_family_size = _whole(
        "confirmatory_frozen_family_size", num("confirmatory_frozen_family_size")
    )
    _lexicon_member_count = sum(
        1 for _label, _disposition in CONFIRMATORY_REPLICATION if _disposition == "external_lexicon"
    )
    _split_diagnostic_count = sum(
        1 for _label, _disposition in CONFIRMATORY_REPLICATION if _disposition == "split_diagnostic"
    )
    if (
        _scored_member_count + _lexicon_member_count + _split_diagnostic_count
        != _recorded_family_size
    ):
        raise ManuscriptVariableError(
            "the confirmatory member composition disagrees with the recorded "
            f"family size: scored={_scored_member_count}, "
            f"lexicon={_lexicon_member_count}, "
            f"split diagnostics={_split_diagnostic_count}, "
            f"recorded family size={_recorded_family_size}"
        )
    variables["CONFIRMATORY_MEMBERS_SCORED_WORD"] = _count_word(_scored_member_count)
    _unrescored_member_count = _recorded_family_size - _scored_member_count
    variables["CONFIRMATORY_MEMBERS_UNRESCORED"] = str(_unrescored_member_count)
    variables["CONFIRMATORY_MEMBERS_UNRESCORED_WORD"] = _count_word(_unrescored_member_count)
    variables["CONFIRMATORY_MEMBERS_LEXICON_COUNT"] = str(_lexicon_member_count)
    variables["CONFIRMATORY_MEMBERS_LEXICON_COUNT_WORD"] = _count_word(_lexicon_member_count)
    variables["CONFIRMATORY_MEMBERS_SPLIT_DIAGNOSTIC_COUNT"] = str(_split_diagnostic_count)
    variables["CONFIRMATORY_MEMBERS_SPLIT_DIAGNOSTIC_COUNT_WORD"] = _count_word(
        _split_diagnostic_count
    )
    # The step-down-only survivor count is a recorded ledger value; its word
    # form exists so the introduction can name it at a clause start without a
    # numeral.
    _holm_only_count = _whole("stat_holm_only_count", num("stat_holm_only_count"))
    variables["STAT_HOLM_ONLY_COUNT_WORD"] = _count_word(_holm_only_count)
    # The step-down-only survivors are a recorded list; the prose names them
    # by what they test, so the clause is derived here and fails closed on a
    # member this map cannot name. A step-down verdict depends on where every
    # other member fell, so this set moves when an unrelated arm is re-recorded.
    _holm_only_survivors = readers.raw("stat_holm_only_survivors")
    if not isinstance(_holm_only_survivors, list) or len(_holm_only_survivors) != _holm_only_count:
        raise ManuscriptVariableError(
            "stat_holm_only_survivors disagrees with stat_holm_only_count: "
            f"{_holm_only_survivors!r} against {_holm_only_count}"
        )
    _named = []
    for _member in _holm_only_survivors:
        if _member not in _FAMILY_MEMBER_PHRASES:
            raise ManuscriptVariableError(
                f"no prose phrase is declared for family member {_member!r}; "
                "extend _FAMILY_MEMBER_PHRASES deliberately"
            )
        _named.append(_FAMILY_MEMBER_PHRASES[_member])
    if _holm_only_count == 0:
        variables["STAT_HOLM_ONLY_CLAUSE"] = "none does"
    else:
        _verb = "does" if _holm_only_count == 1 else "do"
        variables["STAT_HOLM_ONLY_CLAUSE"] = (
            f"{_count_word(_holm_only_count)} {_verb}: {_join_prose(_named)}"
        )
    # The fixed-threshold survivor set, as prose, so the supplement never types
    # the member names again.
    _family_survivors = readers.raw("stat_family_survivors")
    if not isinstance(_family_survivors, list) or not _family_survivors:
        raise ManuscriptVariableError("stat_family_survivors must be a non-empty list")
    variables["STAT_FAMILY_SURVIVOR_LIST_PROSE"] = _join_prose(
        [f"`{_label}`" for _label in _family_survivors]
    )
    # The Stage 6 sign test against its two corrections, as one predicate. Holm
    # dominates the fixed bar, so a fixed-threshold survivor that the step-down
    # refuses is an impossible ledger and is refused rather than rendered.
    _holm_rejected = readers.raw("stat_holm_stage6_rejected")
    _fixed_confirmed = readers.raw("stat_stage6_familywise_confirmed")
    for _name, _flag in (
        ("stat_holm_stage6_rejected", _holm_rejected),
        ("stat_stage6_familywise_confirmed", _fixed_confirmed),
    ):
        if not isinstance(_flag, bool):
            raise ManuscriptVariableError(f"{_name} must be a recorded boolean, got {_flag!r}")
    if _fixed_confirmed and not _holm_rejected:
        raise ManuscriptVariableError(
            "stat_stage6_familywise_confirmed is true while stat_holm_stage6_rejected is "
            "false; Holm admits every fixed-threshold survivor, so this ledger is inconsistent"
        )
    variables["STAGE6_SIGN_STEPDOWN_CLAUSE"] = {
        (
            True,
            True,
        ): "crosses both the fixed Bonferroni threshold and its Holm step-down threshold",
        (
            True,
            False,
        ): "crosses its Holm step-down threshold but not the fixed Bonferroni threshold",
        (
            False,
            False,
        ): "crosses neither the fixed Bonferroni threshold nor its Holm step-down threshold",
    }[(_holm_rejected, _fixed_confirmed)]

    variables["STAGE1_UNIGRAM_MARGIN"] = (
        f"{num('stage1_unigram_baseline') - num('stage1_letter_perplexity'):.4f}"
    )
    variables["STAGE2_PHONEME_UNIGRAM_MARGIN"] = (
        f"{num('stage2_phoneme_unigram_baseline') - num('stage2_phoneme_perplexity'):.4f}"
    )
    # Both operands come from the ledger: a literal here would put a hand-typed
    # threshold inside the token layer.
    variables["STAGE1_CEILING_SHORTFALL_RATIO"] = (
        f"{num('stage1_declared_gate_nats') / num('stage1_max_bigram_gain_nats'):.2f}"
    )
    variables["STAGE2_MCNEMAR_TOTAL_DISCORDANT"] = str(
        int(num("stage2_g2p_mcnemar_b") + num("stage2_g2p_mcnemar_c"))
    )
    # The peeking arm's gain over the runnable policy. Typed as "+3.90 points"
    # in the limitations section until this existed, which made it the one
    # measured number in the manuscript that no re-record would ever update.
    variables["BUDGET_PEEKING_GAIN_PTS"] = (
        f"{num('budget_peeking_accuracy') - num('budget_fixed_accuracy'):.2f}"
    )

    # The case-path discussion used to spell these measured ratios as “roughly
    # a fifth” and “roughly a twenty-fifth”. Derive the exact percentages from
    # the same baseline, agreement, and margin claims as the adjacent table so a
    # recorder replay cannot leave the prose's fractions behind.
    def stage4_path_headroom_captured(prefix: str) -> float:
        baseline_label = f"stage4_path_{prefix}_baseline"
        agreement_label = f"stage4_path_{prefix}_agreement"
        margin_label = f"stage4_path_{prefix}_margin_pts"
        baseline = num(baseline_label)
        agreement = num(agreement_label)
        margin = num(margin_label)
        if not 0.0 <= baseline < 100.0 or not 0.0 <= agreement <= 100.0:
            raise ManuscriptVariableError(
                f"{baseline_label}/{agreement_label} must be percentages with "
                f"baseline below 100, got {baseline!r}/{agreement!r}"
            )
        if not math.isclose(agreement - baseline, margin, rel_tol=0.0, abs_tol=0.011):
            raise ManuscriptVariableError(
                f"{margin_label} must equal agreement minus baseline within the "
                f"ledger's two-decimal precision, got {margin!r} versus "
                f"{agreement - baseline!r}"
            )
        return 100.0 * margin / (100.0 - baseline)

    variables["STAGE4_PATH_DEPREL_HEADROOM_CAPTURED_PCT"] = (
        f"{stage4_path_headroom_captured('deprel'):.2f}"
    )
    variables["STAGE4_PATH_MORPHOLOGY_HEADROOM_CAPTURED_PCT"] = (
        f"{stage4_path_headroom_captured('morphology'):.2f}"
    )


def _produce_counted_claims(
    variables: _StrictVariableMap, claims: Mapping[str, Any], readers: _ClaimReaders
) -> None:
    """The counts the paper had stated in words, now counted from the ledger:
    Stage 3 mis-wirings against their registry, the served-probe inventory, and
    the benchmark family's per-probe contrast arithmetic.
    """
    num = readers.num

    #: Level 3 mis-wirings that clear the letter-bigram bar. The registry owns
    #: both finite measured arms and the analytic-infinite end-of-word arm, so
    #: no consumer reconstructs the architecture with ``len(labels) + 1``.
    from ailm.levels.word import validate_stage3_miswirings

    _stage3_miswirings = validate_stage3_miswirings()

    _registered_miswire_labels = {
        arm.claim_label for arm in _stage3_miswirings if arm.claim_label is not None
    }
    _ledger_miswire_labels = {label for label in claims if label.startswith("stage3_miswire_")}
    if _ledger_miswire_labels != _registered_miswire_labels:
        raise ManuscriptVariableError(
            "Stage 3 miswiring claims must exactly match the canonical registry; "
            f"ledger={sorted(_ledger_miswire_labels)}, "
            f"registry={sorted(_registered_miswire_labels)}"
        )
    _miswire = {label: num(label) for label in _registered_miswire_labels}
    if _miswire and "stage3_bigram_baseline" in claims:
        _bar = num("stage3_bigram_baseline")
        _clearing = sum(1 for value in _miswire.values() if value < _bar)
        _finite = sum(arm.has_finite_score for arm in _stage3_miswirings)
        _total = len(_stage3_miswirings)
        variables["STAGE3_MISWIRINGS_CLEARING_BAR"] = str(_clearing)
        variables["STAGE3_MISWIRINGS_CLEARING_BAR_WORD"] = _count_word(_clearing)
        variables["STAGE3_MISWIRINGS_FINITE"] = str(_finite)
        variables["STAGE3_MISWIRINGS_FINITE_WORD"] = _count_word(_finite)
        variables["STAGE3_MISWIRINGS_TOTAL"] = str(_total)
        variables["STAGE3_MISWIRINGS_TOTAL_WORD"] = _count_word(_total)
        # How far the worst mis-wiring sits above the bar, which is the reason
        # the figure uses a log axis. The paper said two orders; it is not two.
        variables["STAGE3_MISWIRE_TIME_REVERSED_ORDERS"] = (
            f"{math.log10(_miswire['stage3_miswire_time_reversed'] / _bar):.1f}"
        )
        variables["STAGE3_MISWIRE_TIME_REVERSED_RATIO"] = (
            f"{_miswire['stage3_miswire_time_reversed'] / _bar:.0f}"
        )

    #: Contrasts the external benchmark declares per served probe. The probe
    #: count is code-owned (`MODELS` in the benchmark recorder), so adding a
    #: model changes this sentence rather than leaving it stale.
    _registered_probe_keys = tuple(probe.key for probe in SERVED_PROBES)
    if _registered_probe_keys != AUTHORED_MANUSCRIPT_PROBE_KEYS:
        raise ManuscriptVariableError(
            "the served-probe registry changed; expand the authored benchmark "
            "claim families, citations, tables, and interpretation guards before "
            f"rendering (authored {AUTHORED_MANUSCRIPT_PROBE_KEYS}, registered "
            f"{_registered_probe_keys})"
        )
    _registered_probe_keys_set = set(_registered_probe_keys)
    _base_keys = BASE_MODEL_PROBE_KEYS & _registered_probe_keys_set
    _instruct_count = SERVED_PROBE_COUNT - len(_base_keys)
    if not (BASE_MODEL_PROBE_KEYS and _registered_probe_keys_set >= BASE_MODEL_PROBE_KEYS):
        raise ManuscriptVariableError(
            "BASE_MODEL_PROBE_KEYS must be a nonempty subset of the served "
            "probe registry; the inventory sentence cannot be derived "
            f"(base {sorted(BASE_MODEL_PROBE_KEYS)}, registered "
            f"{sorted(_registered_probe_keys_set)})"
        )
    variables["BENCH_PROBE_COUNT"] = str(SERVED_PROBE_COUNT)
    variables["BENCH_PROBE_COUNT_WORD"] = _count_word(SERVED_PROBE_COUNT)
    variables["BENCH_INSTRUCT_PROBE_COUNT"] = str(_instruct_count)
    variables["BENCH_INSTRUCT_PROBE_COUNT_WORD"] = _count_word(_instruct_count)
    variables["BENCH_BASE_PROBE_COUNT_WORD"] = _count_word(len(_base_keys))
    _measurable_flags = {
        _key: bool(num(f"bench_calib_measurable_{_key}")) for _key in _registered_probe_keys_set
    }
    _withheld = sorted(_key for _key, _ok in _measurable_flags.items() if not _ok)
    variables["BENCH_CALIB_WITHHELD_COUNT"] = str(len(_withheld))
    variables["BENCH_CALIB_WITHHELD_COUNT_WORD"] = _count_word(len(_withheld))
    variables["BENCH_CALIB_WITHHELD_LIST"] = ", ".join(
        PROBE_DISPLAY_NAMES[_key] for _key in _withheld
    )
    variables["BENCH_MODEL_GEMMA3_DISPLAY"] = PROBE_DISPLAY_NAMES["gemma3"]
    variables["BENCH_MODEL_QWEN25_DISPLAY"] = PROBE_DISPLAY_NAMES["qwen25"]
    variables["BENCH_MODEL_SMOLLM_DISPLAY"] = PROBE_DISPLAY_NAMES["smollm"]
    variables["BENCH_MODEL_GEMMA3_TAG"] = PROBE_MODEL_TAGS["gemma3"]
    variables["BENCH_MODEL_QWEN25_TAG"] = PROBE_MODEL_TAGS["qwen25"]
    variables["BENCH_MODEL_SMOLLM_TAG"] = PROBE_MODEL_TAGS["smollm"]
    _family_size = _whole("bench_family_size", num("bench_family_size"))
    _per_probe, _remainder = divmod(_family_size, SERVED_PROBE_COUNT)
    if _remainder:
        raise ManuscriptVariableError(
            "bench_family_size must divide exactly across the served probe registry; "
            f"got {_family_size} contrasts and {SERVED_PROBE_COUNT} probes"
        )
    variables["BENCH_FAMILY_CONTRASTS_PER_PROBE"] = str(_per_probe)
    variables["BENCH_FAMILY_CONTRASTS_PER_PROBE_WORD"] = _count_word(_per_probe)
    _derived_power_count = len(DERIVED_POWER_ARM_PREFIXES) * SERVED_PROBE_COUNT
    variables["BENCH_DERIVED_POWER_COUNT"] = str(_derived_power_count)
    variables["BENCH_DERIVED_POWER_COUNT_WORD"] = _count_word(_derived_power_count)


def _produce_paper_metadata(variables: _StrictVariableMap, config: Mapping[str, Any]) -> None:
    """Title, version, author list, and keywords from the reviewable config."""
    paper = config["paper"]
    variables["META_TITLE"] = str(paper["title"])
    variables["META_VERSION"] = str(paper["version"])

    authors = config["authors"]
    variables["META_AUTHOR_COUNT"] = str(len(authors))
    variables["META_AUTHORS"] = "; ".join(
        str(a.get("name", "")) for a in authors if isinstance(a, dict)
    )
    variables["META_KEYWORDS"] = ", ".join(str(k) for k in config["keywords"])


def _produce_build_provenance(
    variables: _StrictVariableMap, claims: Mapping[str, Any], project_root: Path
) -> None:
    """Build stamp, claim count, and vendored-artifact checksums."""
    variables["META_BUILD_TIMESTAMP"] = _build_timestamp()
    variables["META_CLAIM_COUNT"] = str(len(claims))
    for _vendor_name, _vendor_value in _vendor_metadata(project_root).items():
        variables[_vendor_name] = _vendor_value


def generate_manuscript_variables(
    project_root: Path,
    *,
    claims_loader: Callable[[Path], dict[str, Any]] = _load_claims,
) -> dict[str, str]:
    """Build the full ``{{TOKEN}}`` map for the AILM manuscript.

    Args:
        project_root: The AILM project root (the directory holding ``docs/manuscript/``).
        claims_loader: The function that loads the claims ledger. Defaults to
            the real :func:`_load_claims`; callers that need to observe which
            labels a derivation reads (without replacing the loader as a
            module attribute) pass an observing wrapper here instead.

    Returns:
        Flat mapping of UPPERCASE token name to string value.

    Raises:
        MissingClaimError: If the ledger is absent or lacks a required claim.
    """
    claims = claims_loader(project_root)
    config = _load_config(project_root)

    variables: _StrictVariableMap = _StrictVariableMap()
    readers = _ClaimReaders(claims)

    # ── Ledger-derived tokens: label.upper() ─────────────────────────────────
    _produce_ledger_tokens(variables, claims, project_root)

    # ── Derived quantities the prose needs but the ledger stores in parts ────
    _produce_derived_quantities(variables, readers)

    # ── Counts the paper stated in words, now counted ─────────────────────────
    # Each of these was a spelled-out number: "six of the eight cells", "Three
    # of the five mis-wirings", "two orders of magnitude", "two of the six
    # verdicts". A spelled count is exactly as hard-coded as a digit and no
    # numeral sweep can see it, which is how two of them came to be wrong.
    # Every one below is counted from claims already in the ledger, so none is
    # a new measurement and none can drift from the values beside it.

    _produce_counted_claims(variables, claims, readers)

    synthetic = _produce_synthetic_results(variables, readers)
    _produce_synthetic_context_results(variables, readers, synthetic)
    _produce_synthetic_phone_results(variables, readers)
    _produce_aligned_g2p_comparator(variables, readers)

    context = _produce_context_protocol(variables, claims, readers)
    _produce_context_results(variables, claims, readers, context)

    _produce_calibration_rows(variables, readers)
    _produce_connected_gum(variables, readers)

    lexical = _produce_lexical_sweep(variables, claims, readers)
    _produce_fitted_and_stage5(variables, claims, readers)
    # The null-bounds section still consumes the high-alpha arm's sweep state
    # until it becomes its own producer.
    _lexical_prefix_by_alpha = lexical.prefix_by_alpha
    _lexical_metrics = lexical.metrics

    _produce_architecture_tallies(variables, claims, readers)

    # ── Detectability for the benchmark arms the recorder left without power ──
    _produce_derived_arm_power(variables, readers)

    # ── Bounds on the nulls (the convention lives in ailm.statistics) ────────
    # Each null below is reported as "no effect" and the data support more than
    # that: an interval bounding how large the effect can be. Every input is
    # recorded, so these are derivations, not measurements.
    #

    _produce_null_bound_conventions(variables, readers)
    _produce_stage3_scaling_ladder(variables, claims, readers)

    _produce_null_bounds(variables, readers, lexical)
    # ── Paper metadata ───────────────────────────────────────────────────────
    _produce_paper_metadata(variables, config)
    variables.add_many(_load_rendering_variables(project_root), producer="rendering configuration")

    _produce_build_provenance(variables, claims, project_root)

    return dict(variables)
