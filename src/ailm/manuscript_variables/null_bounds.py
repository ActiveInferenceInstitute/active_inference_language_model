"""Derived-power and null-bound producers."""

from __future__ import annotations

from ailm.statistics import (
    CORRECTED_CONFIDENCE,
    DIFFERENCE_CONFIDENCE,
    SHARE_CONFIDENCE,
    equivalence_region,
    share_interval_supports_equivalence,
)
from ailm.value_formatting import (
    ManuscriptVariableError,
)
from ailm.value_formatting import (
    format_ledger_value as _format_value,
)

from ._claims import (
    _ClaimReaders,
    _LexicalSweep,
    require_close,
)
from ._io import _StrictVariableMap
from ._stats import (
    DERIVED_BOUND_ARMS,
    SUBSAMPLE_BOUND_ARMS,
    _whole,
    paired_difference_bound_from_counts,
    paired_power_from_counts,
)


def _produce_derived_arm_power(variables: _StrictVariableMap, readers: _ClaimReaders) -> None:
    """Detectability for the benchmark arms the recorder left without power:
    retrospective power derived from the recorded discordant counts.
    """
    num = readers.num

    # The letter, phoneme and G2P arms have recorded ``bench_*_power_*`` claims.
    # These three arms do not, so the supplement could report their contrast but
    # not whether the pair count could have found it. Power is a pure function of
    # b and c, both recorded, so it is derived here instead of remeasured.
    #
    # Each key is written as a literal so the shadowing scan in
    # ``tests/test_benchmarks.py::test_no_derived_token_shadows_a_recorded_claim``
    # can read it. A loop over dynamic keys would be invisible to that scan.
    def arm_power(prefix: str, model: str) -> str:
        """Derived power for one benchmark arm and model, formatted for prose."""
        return _format_value(
            paired_power_from_counts(
                num(f"{prefix}_paired_b_{model}"), num(f"{prefix}_paired_c_{model}")
            )
        )

    variables["BENCH_LETTER_AZ_POWER_GEMMA3"] = arm_power("bench_letter_az", "gemma3")
    variables["BENCH_LETTER_AZ_POWER_QWEN25"] = arm_power("bench_letter_az", "qwen25")
    variables["BENCH_LETTER_AZ_POWER_SMOLLM"] = arm_power("bench_letter_az", "smollm")
    variables["BENCH_LETTER_PROMPT_POWER_GEMMA3"] = arm_power("bench_letter_prompt", "gemma3")
    variables["BENCH_LETTER_PROMPT_POWER_QWEN25"] = arm_power("bench_letter_prompt", "qwen25")
    variables["BENCH_LETTER_PROMPT_POWER_SMOLLM"] = arm_power("bench_letter_prompt", "smollm")
    variables["BENCH_LETTER_CTX_POWER_GEMMA3"] = arm_power("bench_letter_ctx", "gemma3")
    variables["BENCH_LETTER_CTX_POWER_QWEN25"] = arm_power("bench_letter_ctx", "qwen25")
    variables["BENCH_LETTER_CTX_POWER_SMOLLM"] = arm_power("bench_letter_ctx", "smollm")


def _produce_null_bound_conventions(variables: _StrictVariableMap, readers: _ClaimReaders) -> None:
    """The null-bounds block's convention constants: the sampled generation
    population, the interval coverages, and the equivalence region.
    """
    num = readers.num

    # Convention constants. The manuscript states the coverage of every interval
    # it prints; hand-typing "95%" beside a derived bound is how a level and its
    # interval drift apart.
    # Total sampled word tokens behind the generation sweep's membership and
    # diversity columns. Derived rather than recorded because it is the product
    # of two recorded counts, and typing it beside them is how the three drift.
    variables["GENERATION_SAMPLED_TOKENS"] = _format_value(
        _whole("generation_words_per_run", num("generation_words_per_run"))
        * _whole("generation_seeds", num("generation_seeds"))
    )

    variables["BOUND_CONFIDENCE_PCT"] = _format_value(round(DIFFERENCE_CONFIDENCE * 100, 4))
    # Both scales now read at one coverage (decision 37). The rationale that
    # stood here for keeping them apart is what that decision overturned: the
    # levels sat backwards on their roles, stricter on primary effects already
    # judged at the corrected threshold and looser on the nulls where the
    # interval is the whole claim. The original warning still holds and is why
    # both are computed rather than typed: two levels one of which is
    # hand-typed is how a manuscript prints a coverage its intervals do not
    # have.
    # Harmonized onto DIFFERENCE_CONFIDENCE by decision 37, so this now renders
    # the same number as BOUND_CONFIDENCE_PCT. The token is kept, and kept
    # distinct, because the two SCALES stay distinct even when the level does
    # not: a share lives in [0, 1] and a paired difference in [-1, 1]. Collapsing
    # the tokens would let a future divergence go unnoticed in the prose.
    variables["SHARE_CONFIDENCE_PCT"] = _format_value(round(SHARE_CONFIDENCE * 100, 4))
    variables["BOUND_CORRECTED_CONFIDENCE_PCT"] = _format_value(
        round(CORRECTED_CONFIDENCE * 100, 4)
    )
    _region_low, _region_high = equivalence_region()
    variables["EQUIVALENCE_REGION_LOW"] = _format_value(round(_region_low, 6))
    variables["EQUIVALENCE_REGION_HIGH"] = _format_value(round(_region_high, 6))


def _produce_null_bounds(
    variables: _StrictVariableMap, readers: _ClaimReaders, lexical: _LexicalSweep
) -> None:
    """The bounds on the nulls: a paired interval for every arm the recorder
    left without one, derived from recorded counts, with each point estimate
    checked against its separately recorded counterpart.
    """
    num = readers.num
    _lexical_prefix_by_alpha = lexical.prefix_by_alpha
    _lexical_metrics = lexical.metrics

    # The denominator is the one input the counts cannot check themselves, so
    # every arm here has its point estimate ``(b - c) / n`` verified against a
    # separately recorded quantity in
    # ``tests/test_statistics.py::TestDerivedPairedBound``. Without that check a
    # plausible-looking wrong ``n`` would rescale every bound in this block and
    # nothing would report it.
    def bound(b_label: str, c_label: str, n_label: str, *, scale: float = 1.0) -> tuple[str, str]:
        """Formatted ``(low, high)`` for one arm, on the scale its neighbors use."""
        low, high = paired_difference_bound_from_counts(num(b_label), num(c_label), num(n_label))
        return _format_value(round(low * scale, 6)), _format_value(round(high * scale, 6))

    def discordant(b_label: str, c_label: str) -> str:
        """Pair count the bound rests on, so prose can weigh the bound's width."""
        return str(_whole("b", num(b_label)) + _whole("c", num(c_label)))

    # Stage 5's privileged-label gain is a signed relative change, not a share.
    # Keep its ledger-owned token, but reconcile it to the two recorded
    # perplexities at the recorder's three-decimal precision so a plausible
    # mutation cannot render beside unchanged operands.
    _stage5_typefree = num("stage5_typefree_perplexity")
    _stage5_type_oracle = num("stage5_type_oracle_perplexity")
    if _stage5_typefree <= 0.0:
        raise ManuscriptVariableError("stage5_typefree_perplexity must be positive")
    require_close(
        "stage5_privileged_label_gain_pct",
        num("stage5_privileged_label_gain_pct"),
        round(100.0 * (_stage5_typefree - _stage5_type_oracle) / _stage5_typefree, 3),
    )

    # Stage 5, clause-type classification against the majority baseline. The
    # point tie (p = 0.5) rests on two discordant pairs, which is why the bound
    # is nearly as wide as the quantity itself. Reported in percentage
    # points to match ``stage5_classification_pct`` beside it.
    _stage5_low, _stage5_high = bound(
        "stage5_mcnemar_b", "stage5_mcnemar_c", "stage5_test_sentences", scale=100.0
    )
    variables["STAGE5_CLASSIFICATION_DELTA_PTS"] = _format_value(
        round(
            100.0
            * (num("stage5_mcnemar_b") - num("stage5_mcnemar_c"))
            / num("stage5_test_sentences"),
            6,
        )
    )
    variables["STAGE5_CLASSIFICATION_DELTA_LOW_PTS"] = _stage5_low
    variables["STAGE5_CLASSIFICATION_DELTA_HIGH_PTS"] = _stage5_high
    variables["STAGE5_CLASSIFICATION_DISCORDANT"] = discordant(
        "stage5_mcnemar_b", "stage5_mcnemar_c"
    )

    # The channel arms and the two geometric-pooling arms. Each of these was
    # reported as a margin and a p-value and nothing else, so a reader could see
    # whether an effect reached a threshold but not how large it could be — the
    # ordering the report calls out, effect size before p-value.
    #
    # Every arm's point estimate (b - c) / n is checked against its separately
    # recorded ``*_best_margin_pts`` in
    # ``tests/test_statistics.py::TestDerivedPairedBound``. That check is the
    # only thing standing between a plausible wrong denominator and seven
    # silently rescaled intervals, since the counts cannot verify n themselves.
    for _arm, _position_claim in DERIVED_BOUND_ARMS:
        # Four decimals, matching the ``*_best_margin_pts`` these sit beside. A
        # bound printed to more places than the estimate it brackets reads as
        # the more precise of the two, which is backwards.
        _low, _high = paired_difference_bound_from_counts(
            num(f"{_arm}_paired_b"),
            num(f"{_arm}_paired_c"),
            num(_position_claim),
        )
        variables[f"{_arm.upper()}_DELTA_LOW_PTS"] = _format_value(round(_low * 100.0, 4))
        variables[f"{_arm.upper()}_DELTA_HIGH_PTS"] = _format_value(round(_high * 100.0, 4))
        variables[f"{_arm.upper()}_DISCORDANT"] = discordant(f"{_arm}_paired_b", f"{_arm}_paired_c")

    # The two subsample arms, on the same interval and the same scale. They keep
    # the ``_mcnemar_`` naming of their own family rather than the ``_paired_``
    # of the loop above, so they are emitted here instead of being renamed into
    # it: a label rename is a re-record, and these counts were added precisely so
    # that no re-record would be needed to bound them.
    #
    # The bound is DERIVED, never recorded. The counts are the measurement and
    # have one writer; an interval stored beside them would be a second copy of
    # the same fact, free to drift the first time either was recomputed.
    for _sub in SUBSAMPLE_BOUND_ARMS:
        _low, _high = paired_difference_bound_from_counts(
            num(f"{_sub}_mcnemar_b"), num(f"{_sub}_mcnemar_c"), num(f"{_sub}_positions")
        )
        variables[f"{_sub.upper()}_DELTA_LOW_PTS"] = _format_value(round(_low * 100.0, 4))
        variables[f"{_sub.upper()}_DELTA_HIGH_PTS"] = _format_value(round(_high * 100.0, 4))
        variables[f"{_sub.upper()}_DISCORDANT"] = discordant(
            f"{_sub}_mcnemar_b", f"{_sub}_mcnemar_c"
        )

    # Stage 6, the narrative-phase sign test. Its p-value misses the corrected
    # threshold, so the manuscript may not call the effect established; the
    # interval says how large it can be. Every test sentence is discordant here,
    # so the difference is a net directional share rather than an accuracy gap.
    _stage6_low, _stage6_high = bound(
        "stage6_sign_test_positive", "stage6_sign_test_negative", "stage6_test_sentences"
    )
    variables["STAGE6_SIGN_NET_SHARE"] = _format_value(
        round(
            (num("stage6_sign_test_positive") - num("stage6_sign_test_negative"))
            / num("stage6_test_sentences"),
            6,
        )
    )
    variables["STAGE6_SIGN_NET_SHARE_LOW"] = _stage6_low
    variables["STAGE6_SIGN_NET_SHARE_HIGH"] = _stage6_high
    variables["STAGE6_SIGN_DISCORDANT"] = discordant(
        "stage6_sign_test_positive", "stage6_sign_test_negative"
    )

    # Stage 2B's model-versus-per-letter-baseline direction among the
    # discordant positions.  This is the established contrast used as the live
    # positive control in fig:null_bounds.  Conditioning on the discordant set
    # puts it on the same net-share axis as the Stage 6 sign split without
    # pretending the remaining tied positions supplied pair information.
    _stage2_low, _stage2_high = bound(
        "stage2_g2p_mcnemar_b",
        "stage2_g2p_mcnemar_c",
        "stat_stage2_discordant_n",
    )
    variables["STAGE2_G2P_DISCORDANT_NET_SHARE"] = _format_value(
        round(
            (num("stage2_g2p_mcnemar_b") - num("stage2_g2p_mcnemar_c"))
            / num("stat_stage2_discordant_n"),
            6,
        )
    )
    variables["STAGE2_G2P_DISCORDANT_NET_SHARE_LOW"] = _stage2_low
    variables["STAGE2_G2P_DISCORDANT_NET_SHARE_HIGH"] = _stage2_high

    # Stage 3's high-mixing-weight row. The only null in the plotted paired-bound roster with
    # enough discordant pairs to say more than "bounded": its share interval
    # sits inside the equivalence region even at the corrected level, so the
    # harm is smaller than the smallest effect the project declared
    # interesting. Points, to match ``lexical_high_alpha_margin_pts``.
    _high_alpha = num("lexical_high_alpha")
    if _high_alpha != max(_lexical_prefix_by_alpha):
        raise ManuscriptVariableError(
            "lexical_high_alpha must equal the maximum recorded sweep coordinate; "
            f"got {_high_alpha} and {tuple(sorted(_lexical_prefix_by_alpha))}"
        )
    _high_prefix = _lexical_prefix_by_alpha[_high_alpha]
    _high_metrics = _lexical_metrics[_high_alpha]
    require_close(
        "lexical_high_alpha_accuracy",
        num("lexical_high_alpha_accuracy"),
        float(_high_metrics["accuracy"]),
    )
    require_close(
        "lexical_high_alpha_margin_pts",
        num("lexical_high_alpha_margin_pts"),
        float(_high_metrics["delta"]),
    )
    _high_b = f"{_high_prefix}_mcnemar_b"
    _high_c = f"{_high_prefix}_mcnemar_c"
    _lex9_low, _lex9_high = bound(
        _high_b,
        _high_c,
        "lexical_eval_positions",
        scale=100.0,
    )
    variables["LEXICAL_HIGH_ALPHA_DELTA_LOW_PTS"] = _lex9_low
    variables["LEXICAL_HIGH_ALPHA_DELTA_HIGH_PTS"] = _lex9_high
    variables["LEXICAL_HIGH_ALPHA_DISCORDANT"] = discordant(_high_b, _high_c)
    variables["LEXICAL_HIGH_ALPHA_EQUIVALENT"] = str(
        int(
            share_interval_supports_equivalence(
                _whole("b", num(_high_b)),
                _whole("c", num(_high_c)),
                confidence=CORRECTED_CONFIDENCE,
            )
        )
    )

    # The context-window arm: no measurable difference between the
    # short and the long window. Fractions, to match the recorded
    # ``bench_*_delta_*`` claims beside them. Direction is long minus short,
    # fixed by the recorder's ``paired_discordance(long, short)``.
    def ctx_bound(model: str) -> tuple[str, str]:
        return bound(
            f"bench_letter_ctx_paired_b_{model}",
            f"bench_letter_ctx_paired_c_{model}",
            "bench_letter_positions",
        )

    def ctx_delta(model: str) -> str:
        return _format_value(
            round(
                (
                    num(f"bench_letter_ctx_paired_b_{model}")
                    - num(f"bench_letter_ctx_paired_c_{model}")
                )
                / num("bench_letter_positions"),
                6,
            )
        )

    # Literal keys, so the shadowing scan in
    # ``tests/test_benchmarks.py::test_no_derived_token_shadows_a_recorded_claim``
    # can read them. A loop over dynamic keys would be invisible to that scan.
    variables["BENCH_LETTER_CTX_DELTA_GEMMA3"] = ctx_delta("gemma3")
    variables["BENCH_LETTER_CTX_DELTA_QWEN25"] = ctx_delta("qwen25")
    variables["BENCH_LETTER_CTX_DELTA_SMOLLM"] = ctx_delta("smollm")
    variables["BENCH_LETTER_CTX_DELTA_LOW_GEMMA3"] = ctx_bound("gemma3")[0]
    variables["BENCH_LETTER_CTX_DELTA_HIGH_GEMMA3"] = ctx_bound("gemma3")[1]
    variables["BENCH_LETTER_CTX_DELTA_LOW_QWEN25"] = ctx_bound("qwen25")[0]
    variables["BENCH_LETTER_CTX_DELTA_HIGH_QWEN25"] = ctx_bound("qwen25")[1]
    variables["BENCH_LETTER_CTX_DELTA_LOW_SMOLLM"] = ctx_bound("smollm")[0]
    variables["BENCH_LETTER_CTX_DELTA_HIGH_SMOLLM"] = ctx_bound("smollm")[1]
    variables["BENCH_LETTER_CTX_DISCORDANT_GEMMA3"] = discordant(
        "bench_letter_ctx_paired_b_gemma3", "bench_letter_ctx_paired_c_gemma3"
    )
    variables["BENCH_LETTER_CTX_DISCORDANT_QWEN25"] = discordant(
        "bench_letter_ctx_paired_b_qwen25", "bench_letter_ctx_paired_c_qwen25"
    )
    variables["BENCH_LETTER_CTX_DISCORDANT_SMOLLM"] = discordant(
        "bench_letter_ctx_paired_b_smollm", "bench_letter_ctx_paired_c_smollm"
    )
