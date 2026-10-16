"""Registry closure for the synthetic zero-shot and in-context claim families.

These checks exercise only deterministic claim constructors.  They never build
the thousands of cache records used by a publication replay and never contact a
served model.
"""

from __future__ import annotations

import importlib
from typing import cast

import matplotlib.pyplot as plt
import pytest

from ailm.benchmarks.constants import (
    SERVED_PROBES,
    ProbeSpec,
    validate_served_probes,
)
from ailm.benchmarks.synthetic_protocol import SYNTHETIC_INCONTEXT_PROTOCOL
from ailm.figures.benchmarks import _model_presentation
from ailm.figures.incontext_sweep import _probe_titles, generate_incontext_sweep
from ailm.required_claims import (
    REQUIRED_CLAIMS,
    synthetic_benchmark_required_claims,
    synthetic_phone_benchmark_required_claims,
)


def _zero_recorder_namespace(arm_key: str, probe_keys: tuple[str, ...]) -> set[str]:
    """Derive one stipulated arm's namespace through the recorder's own helpers.

    Both arms run the same mechanics off one ``_ZeroShotArm`` record, so the
    namespace has to be derived per arm rather than assumed to be the letter
    arm's — otherwise the phone arm's labels are required by the guard and
    produced by nothing this test can see.
    """
    recorder = importlib.import_module("scripts.record_synthetic_benchmark_claims")
    (arm,) = (candidate for candidate in recorder.ARMS if candidate.key == arm_key)
    ailm_flags = [True, False, True, False]
    values = recorder._base_claim_values(
        arm,
        total=len(ailm_flags),
        ailm_flags=ailm_flags,
        maximum_prompt_characters=40,
        maximum_prompt_tokens=20,
        cache_hits=0,
        cache_misses=8,
    )
    for key in probe_keys:
        values.update(
            recorder._probe_claim_values(
                arm,
                name=key,
                reference_flags=[True, True, False, False],
                short_flags=[False, True, False, False],
                ailm_flags=ailm_flags,
                unparsable=0,
                short_unparsable=1,
                full_first=2,
                short_first=2,
            )
        )
    if arm == recorder.LETTER_ARM:
        values.update(
            bench_synth_openrouter_model=recorder.OPENROUTER_MODEL,
            bench_synth_openrouter_provider="offline-fixture-route",
        )
    return set(values)


def _context_recorder_namespace(probe_keys: tuple[str, ...]) -> set[str]:
    recorder = importlib.import_module("scripts.record_incontext_benchmark_claims")
    total = 4
    ailm_flags = [True, True, False, False]
    arms = {
        arm: ([False] * total if arm == "k0" else [True, False, False, False])
        for arm, _k, _deranged in recorder._arm_specs()
    }
    unparsable = dict.fromkeys(arms, 0)
    values: dict[str, object] = {}
    for key in probe_keys:
        values.update(recorder._claim_values(key, arms, unparsable, ailm_flags, total))

    # These model-independent values are constructed once around the per-probe
    # helper in both the live recorder and its semantic replay.  The draw helper
    # is executed on a small deterministic pool so its rung-derived namespace is
    # tested without constructing any model-call cache.
    letters = "abcdefghijklmnopqrstuvwxyz"
    positions = [("abc", "d"), ("efg", "h")]
    pool = [
        (f"{letters[(index // 26) % 26]}{letters[index % 26]}", letters[index % 26])
        for index in range(26 * 8)
    ]
    values.update(recorder._draw_diagnostic_claim_values(positions, pool))
    values.update(
        {
            "bench_ctx_openrouter_model": recorder.OPENROUTER_MODEL,
            "bench_ctx_openrouter_provider": "offline-fixture-route",
            "bench_ctx_mapping_control_top1_treatment": 1.0,
            "bench_ctx_mapping_control_top1_deranged": 0.0,
            "bench_ctx_mapping_control_paired_b": 2,
            "bench_ctx_mapping_control_paired_c": 0,
            "bench_ctx_mapping_control_paired_p": 0.5,
            "bench_ctx_mapping_control_family_size": 1,
            "bench_ctx_mapping_control_family_alpha": 0.05,
            "bench_ctx_maximum_prompt_characters": 100,
            "bench_ctx_maximum_prompt_tokens": 50,
            "bench_ctx_scored_cache_hits": 0,
            "bench_ctx_scored_cache_misses": 1,
        }
    )
    return set(values)


def test_registry_derived_claims_equal_both_recorder_namespaces_exactly() -> None:
    """A recorder label can be neither unguarded nor invented by the guard."""

    probe_keys = tuple(probe.key for probe in SERVED_PROBES)
    required = set(
        synthetic_benchmark_required_claims(
            probe_keys,
            SYNTHETIC_INCONTEXT_PROTOCOL.k_grid,
            SYNTHETIC_INCONTEXT_PROTOCOL.headline_k,
        )
    )
    zero_required = {label for label in required if label.startswith("bench_synth_")}
    context_required = {label for label in required if label.startswith("bench_ctx_")}
    assert zero_required == _zero_recorder_namespace("letter", probe_keys)
    assert context_required == _context_recorder_namespace(probe_keys)


def test_registry_derived_claims_equal_the_phone_recorder_namespace_exactly() -> None:
    """The phone arm is registry-closed on the same terms as the letter arm.

    Its labels reach ``REQUIRED_CLAIMS`` through a registry function of its own,
    so without this the guard could require a phone label no recorder produces —
    or miss one it does — and every other closure check would still pass.
    """

    probe_keys = tuple(probe.key for probe in SERVED_PROBES)
    required = set(synthetic_phone_benchmark_required_claims(probe_keys))
    assert required == _zero_recorder_namespace("phone", probe_keys)


def test_required_claims_contains_exactly_the_registry_derived_synthetic_family() -> None:
    """No filtered hand-maintained benchmark block may survive beside the registry."""

    probe_keys = tuple(sorted(probe.key for probe in SERVED_PROBES))
    expected = set(
        synthetic_benchmark_required_claims(
            probe_keys,
            SYNTHETIC_INCONTEXT_PROTOCOL.k_grid,
            SYNTHETIC_INCONTEXT_PROTOCOL.headline_k,
        )
    )
    actual = {
        label for label in REQUIRED_CLAIMS if label.startswith(("bench_synth_", "bench_ctx_"))
    }
    assert actual == expected


def test_third_probe_expands_every_probe_owned_claim_shape() -> None:
    base_probes = ("alpha", "beta")
    expanded_probes = (*base_probes, "gamma")
    base = set(synthetic_benchmark_required_claims(base_probes, (0, 4), 4))
    expanded = set(synthetic_benchmark_required_claims(expanded_probes, (0, 4), 4))

    zero_stems = (
        "letter_top1",
        "letter_top1_wilson_low",
        "letter_top1_wilson_high",
        "paired_b",
        "paired_c",
        "paired_p",
        "unparsable",
        "unparsable_rate",
        "short_top1",
        "short_wilson_low",
        "short_wilson_high",
        "short_unparsable",
        "short_unparsable_rate",
        "context_paired_b",
        "context_paired_c",
        "context_paired_p",
        "context_order_full_first",
        "context_order_short_first",
    )
    expected = {f"bench_synth_{stem}_gamma" for stem in zero_stems}
    expected.update(
        f"bench_ctx_{stem}_{arm}_gamma"
        for arm in ("k0", "k4", "k4_deranged")
        for stem in ("top1", "top1_wilson_low", "top1_wilson_high", "unparsable")
    )
    expected.update(
        f"bench_ctx_k4_vs_{comparison}_paired_{stat}_gamma"
        for comparison in ("zero", "deranged")
        for stat in ("b", "c", "p")
    )
    expected.update(
        f"bench_ctx_k4_vs_ailm_{stem}_gamma"
        for stem in ("paired_b", "paired_c", "paired_p", "delta", "delta_low", "delta_high")
    )
    expected.add("bench_ctx_gap_closed_ratio_gamma")
    assert expanded - base == expected


def test_extra_rung_expands_diagnostics_arms_and_both_contrasts() -> None:
    probes = ("alpha", "beta")
    base = set(synthetic_benchmark_required_claims(probes, (0, 4), 4))
    expanded = set(synthetic_benchmark_required_claims(probes, (0, 4, 16), 4))

    expected = {
        f"bench_ctx_draw_{stem}_k16"
        for stem in ("retries", "retried_positions", "max_attempts_used")
    }
    expected.update(
        f"bench_ctx_{stem}_{arm}_{probe}"
        for probe in probes
        for arm in ("k16", "k16_deranged")
        for stem in ("top1", "top1_wilson_low", "top1_wilson_high", "unparsable")
    )
    expected.update(
        f"bench_ctx_k16_vs_{comparison}_paired_{stat}_{probe}"
        for probe in probes
        for comparison in ("zero", "deranged")
        for stat in ("b", "c", "p")
    )
    assert expanded - base == expected


@pytest.mark.parametrize(
    ("probes", "message"),
    [
        ((), "at least one"),
        ((ProbeSpec("", "tag:1", "Display"),), "key must be nonempty"),
        ((ProbeSpec("key", "", "Display"),), "tag must be nonempty"),
        ((ProbeSpec("key", "tag:1", ""),), "display_name must be nonempty"),
        ((ProbeSpec(" key", "tag:1", "Display"),), "surrounding whitespace"),
        (
            (ProbeSpec("same", "tag:1", "One"), ProbeSpec("same", "tag:2", "Two")),
            "key values must be unique",
        ),
        (
            (ProbeSpec("one", "same", "One"), ProbeSpec("two", "same", "Two")),
            "tag values must be unique",
        ),
        (
            (ProbeSpec("one", "tag:1", "Same"), ProbeSpec("two", "tag:2", "same")),
            "display_name values must be unique",
        ),
    ],
)
def test_served_probe_registry_fails_fast_on_invalid_identity(
    probes: tuple[ProbeSpec, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_served_probes(probes)


def test_served_probe_registry_rejects_non_specs() -> None:
    with pytest.raises(TypeError, match="must be ProbeSpec"):
        validate_served_probes((cast(ProbeSpec, "not-a-probe"),))


def test_figure_presentation_expands_from_probe_specs() -> None:
    probes = (
        ProbeSpec("alpha", "alpha:1", "Alpha Model"),
        ProbeSpec("beta", "beta:1", "Beta Model"),
        ProbeSpec("gamma", "gamma:1", "Gamma Model"),
    )
    labels, colors = _model_presentation(probes)
    assert labels == {
        "ailm": "AILM",
        "alpha": "Alpha Model",
        "beta": "Beta Model",
        "gamma": "Gamma Model",
    }
    assert list(colors) == ["ailm", "alpha", "beta", "gamma"]
    assert len(set(colors.values())) == len(colors)
    assert _probe_titles(probes) == {
        "ALPHA": "alpha run",
        "BETA": "beta run",
        "GAMMA": "gamma run",
    }


def test_incontext_panel_titles_use_hosted_run_names() -> None:
    tokens = {
        "BENCH_CTX_FIGURE_TITLE": "Registry title check",
        "BENCH_CTX_TOP1_AILM": "0.75",
        "BENCH_CTX_UNIFORM_BASELINE": "0.04",
        "BENCH_CTX_POSITIONS": "4",
    }
    for probe in SERVED_PROBES:
        suffix = probe.key.upper()
        for index, k in enumerate(SYNTHETIC_INCONTEXT_PROTOCOL.k_grid):
            value = 0.1 + index / 20
            tokens[f"BENCH_CTX_TOP1_K{k}_{suffix}"] = str(value)
            tokens[f"BENCH_CTX_TOP1_WILSON_LOW_K{k}_{suffix}"] = str(value / 2)
            tokens[f"BENCH_CTX_TOP1_WILSON_HIGH_K{k}_{suffix}"] = str(min(1.0, value * 1.5))
            if k:
                tokens[f"BENCH_CTX_TOP1_K{k}_DERANGED_{suffix}"] = str(value / 2)
                tokens[f"BENCH_CTX_TOP1_WILSON_LOW_K{k}_DERANGED_{suffix}"] = str(value / 4)
                tokens[f"BENCH_CTX_TOP1_WILSON_HIGH_K{k}_DERANGED_{suffix}"] = str(value * 0.75)

    figure = generate_incontext_sweep(tokens)
    try:
        assert [axis.get_title() for axis in figure.axes] == [
            f"{probe.key} run" for probe in SERVED_PROBES
        ]
    finally:
        plt.close(figure)
