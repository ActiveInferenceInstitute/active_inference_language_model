"""The generator's reconciliation checks each refuse one planted ledger defect.

Every token the manuscript prints is recomputed or bounded from its recorded
operands before rendering, so a stale or impossible value fails the build
instead of appearing beside unchanged neighbors. Each case below takes the
complete fixture ledger the generator accepts, plants exactly one defect, and
matches the message of the check that owns it. A refusal from a different
check would fail the match, so every case pins one specific guard.

Defects that the protocol-registry binding, a recomputation check, or the
load-time domain validation intercept first are deliberately absent: the
later guards behind them cannot be reached through the ledger.
"""

from __future__ import annotations

import copy

import pytest

from ailm.manuscript_variables import (
    ManuscriptVariableError,
    generate_manuscript_variables,
)
from tests.test_manuscript_variables import _full_claims, _write_project


def _set(label: str, value):
    def mutate(claims: dict) -> None:
        claims[label]["value"] = value

    return mutate


def _shift(label: str, delta):
    def mutate(claims: dict) -> None:
        claims[label]["value"] = claims[label]["value"] + delta

    return mutate


def _copy_as(source_label: str, new_label: str):
    def mutate(claims: dict) -> None:
        claims[new_label] = copy.deepcopy(claims[source_label])

    return mutate


def _rebind_source(label: str):
    def mutate(claims: dict) -> None:
        claims[label]["source"] = "scripts/somewhere_else.py"

    return mutate


CASES = {
    "famsplit-protocol": (
        _shift("stage2_g2p_famsplit_prefix_length", 1),
        "family-split protocol disagrees",
    ),
    "path-headroom-domain": (_set("stage4_path_deprel_baseline", 100.0), "must be percentages"),
    "miswire-registry": (
        _copy_as("stage3_miswire_uniform_prior", "stage3_miswire_bogus"),
        "miswiring claims must exactly match",
    ),
    "probability-domain": (_set("bench_synth_family_alpha", 1.5), r"must lie in \[0, 1\]"),
    "single-writer": (_rebind_source("bench_synth_positions"), "are not all bound to"),
    "synth-cache-hits": (_set("bench_synth_scored_cache_hits", 1), "zero cache hits"),
    "synth-prompt-chars": (
        _set("bench_synth_maximum_prompt_characters", 0),
        "synthetic maximum prompt character count must be positive",
    ),
    "synth-prompt-tokens": (
        _set("bench_synth_maximum_prompt_tokens", 10**9),
        "synthetic maximum prompt does not fit",
    ),
    "synth-family-size": (_set("bench_synth_family_size", 99), "one paired contrast for every"),
    "synth-unparsable": (
        _set("bench_synth_unparsable_gemma3", 10**9),
        "successes plus unparsable responses",
    ),
    "synth-context-family": (
        _set("bench_synth_context_family_size", 99),
        "context-sensitivity family",
    ),
    "synth-arm-order": (
        _shift("bench_synth_context_order_full_first_gemma3", 1),
        "not the registered alternating",
    ),
    "synth-short-unparsable": (
        _set("bench_synth_short_unparsable_gemma3", 10**9),
        "short-prefix successes plus unparsable",
    ),
    "ctx-cache-hits": (
        _set("bench_ctx_scored_cache_hits", 1),
        "in-context schedule must record zero cache hits",
    ),
    "ctx-prompt-chars": (
        _set("bench_ctx_maximum_prompt_characters", 0),
        "in-context maximum prompt character count",
    ),
    "ctx-prompt-tokens": (
        _set("bench_ctx_maximum_prompt_tokens", 10**9),
        "in-context maximum prompt does not fit",
    ),
    "ctx-mapping-family": (_set("bench_ctx_mapping_control_family_size", 2), "one-test family"),
    "lexical-parse": (
        _copy_as("lexical_sweep_alpha_0_1_accuracy", "lexical_sweep_alpha_x_accuracy"),
        "cannot parse lexical sweep coordinates",
    ),
    "lexical-positions": (
        _set("lexical_eval_positions", 0),
        "lexical_eval_positions must be positive",
    ),
    "scale-field": (
        _copy_as("stage3_scale_500_gain_nats", "stage3_scale_500_bogus"),
        "unknown Stage 3 scaling field",
    ),
    "scale-rung": (
        _copy_as("stage3_scale_500_gain_nats", "stage3_scale_7_gain_nats"),
        "disagree with SCALING_RUNGS",
    ),
    "scale-count": (_set("stage3_scale_rungs", 99), "stage3_scale_rungs is inconsistent"),
    "scale-top-positions": (
        _set("stage3_scale_38007_eval_positions", 0),
        "top Stage 3 scaling rung must score positions",
    ),
    "scale-rung-positions": (
        _set("stage3_scale_500_eval_positions", 0),
        "must have positive positions and perplexities",
    ),
    "scale-monotone": (_set("stage3_scale_500_coverage_pct", 99.0), "must rise strictly"),
    "oov-gap": (_set("stage3_oov_raw_gap_nats", 0.0), "must be positive to define a relative"),
}


def test_the_fixture_ledger_itself_renders(tmp_path) -> None:
    """The positive control: without a planted defect, every case below is vacuous."""
    root = _write_project(tmp_path, _full_claims())
    assert generate_manuscript_variables(root)


@pytest.mark.parametrize(("mutate", "message"), list(CASES.values()), ids=list(CASES))
def test_one_planted_ledger_defect_is_refused_by_its_owning_check(
    tmp_path, mutate, message
) -> None:
    claims = copy.deepcopy(_full_claims())
    mutate(claims)
    root = _write_project(tmp_path, claims)
    with pytest.raises(ManuscriptVariableError, match=message):
        generate_manuscript_variables(root)
