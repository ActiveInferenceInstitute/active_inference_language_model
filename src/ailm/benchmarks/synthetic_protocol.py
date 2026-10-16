"""Frozen execution contracts for the stipulated-language benchmark arms.

The chain itself is owned by :mod:`ailm.levels.synthetic`.  This module owns
the two measurement opportunities built around it: the no-demonstration probe
and the explicit-demonstration sweep.  Recorders and manuscript hydration both
read these immutable values, so a stale ledger cannot become authoritative by
being internally self-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass

from ailm.benchmarks.aligned_g2p import (
    ALIGNED_G2P_ARTIFACT_NAME,
    ALIGNED_G2P_BROKEN_MAP_SEED,
    ALIGNED_G2P_CORRESPONDENCE_SEED,
    ALIGNED_G2P_HELD_OUT_SEED_BASE,
    ALIGNED_G2P_RESAMPLE_ATTEMPTS_CAP,
    ALIGNED_G2P_SCHEMA,
    ALIGNED_G2P_TEST_WORDS,
    ALIGNED_G2P_TRAIN_SEED_BASE,
    ALIGNED_G2P_TRAIN_WORDS,
    ALIGNED_G2P_WORD_LENGTH,
    CORRESPONDENCE_VERSION,
)
from ailm.benchmarks.constants import LETTER_WINDOW_SHORT, PROTOCOL_VERSION
from ailm.levels.synthetic import (
    DEFAULT_HELD_OUT_SEED,
    DEFAULT_PHONE_HELD_OUT_SEED,
    SYNTHETIC_CHAIN_VERSION,
    SYNTHETIC_LETTERS,
    SYNTHETIC_PHONE_CHAIN_VERSION,
    SYNTHETIC_PHONES,
)


@dataclass(frozen=True)
class SyntheticAlignedG2pProtocol:
    """The aligned spelling-to-pronunciation arm's frozen execution contract.

    Protocol P2 of ``docs/extension_protocols.md`` §3: a pair-structured
    stipulated artifact over a seeded per-letter correspondence, structurally
    distinct from the order-one continuation chains.  Nothing is recorded
    under this contract by this module's own milestone; the offline evaluator
    owns the arm until a recorder grant exists.
    """

    version: str
    run_id: str
    source: str
    chain_version: str
    correspondence_version: str
    artifact_schema: str
    artifact_path: str
    word_length: int
    resample_attempts_cap: int
    train_words: int
    test_words: int
    train_seed_base: int
    held_out_seed_base: int
    correspondence_seed: int
    broken_map_seed: int
    response_inventory: tuple[str, ...]
    warmup_policy: str
    arm_order: str
    cache_protocol: str
    #: The zero-shot probe arms' execution contract (protocol §3.4's optional
    #: probe-arm bullet, granted to its recorder): recorded beside the offline
    #: comparator contract above without touching any of its fields, so the
    #: comparator run's provenance stays exactly as it was written.
    probe_run_id: str
    probe_source: str
    probe_response_tokens: int
    probe_context_window: int
    probe_warmup_policy: str
    probe_arm_order: str
    probe_cache_protocol: str


SYNTHETIC_ALIGNED_G2P_PROTOCOL = SyntheticAlignedG2pProtocol(
    version="synthetic-aligned-g2p-v1",
    run_id="synthetic_aligned_g2p_v1",
    source="scripts/evaluate_synthetic_aligned_g2p.py",
    chain_version=SYNTHETIC_CHAIN_VERSION,
    correspondence_version=CORRESPONDENCE_VERSION,
    artifact_schema=ALIGNED_G2P_SCHEMA,
    artifact_path=f"vendor/{ALIGNED_G2P_ARTIFACT_NAME}",
    word_length=ALIGNED_G2P_WORD_LENGTH,
    resample_attempts_cap=ALIGNED_G2P_RESAMPLE_ATTEMPTS_CAP,
    train_words=ALIGNED_G2P_TRAIN_WORDS,
    test_words=ALIGNED_G2P_TEST_WORDS,
    train_seed_base=ALIGNED_G2P_TRAIN_SEED_BASE,
    held_out_seed_base=ALIGNED_G2P_HELD_OUT_SEED_BASE,
    correspondence_seed=ALIGNED_G2P_CORRESPONDENCE_SEED,
    broken_map_seed=ALIGNED_G2P_BROKEN_MAP_SEED,
    response_inventory=SYNTHETIC_PHONES,
    warmup_policy="offline-no-model-calls-v1",
    arm_order="offline-comparator-arms-only-v1",
    cache_protocol=f"{PROTOCOL_VERSION}:synthetic-aligned-g2p-v1:offline-comparator-arms-only-v1",
    probe_run_id="synthetic_aligned_g2p_probe_v1",
    probe_source="scripts/record_aligned_g2p_probe_claims.py",
    # The shipped zero-shot arms' response budget and context window: the g2p
    # prompt is a single short word, so the declared budget is headroom, not a
    # bound (the smollm 2048-token server limit is not reached in practice).
    probe_response_tokens=32,
    probe_context_window=8192,
    probe_warmup_policy="one-uncached-warmup-per-probe-v1",
    probe_arm_order="comparator-arms-then-probe-arms-v1",
    probe_cache_protocol=f"{PROTOCOL_VERSION}:synthetic-aligned-g2p-v1:probe-arms-v1",
)


@dataclass(frozen=True)
class SyntheticZeroShotProtocol:
    """The exact no-demonstration measurement and its sensitivity control."""

    version: str
    run_id: str
    source: str
    chain_version: str
    positions: int
    position_seed: int
    position_generation: str
    query_window: int
    short_query_window: int
    response_inventory: tuple[str, ...]
    context_window: int
    response_tokens: int
    prompt_token_overhead_margin: int
    warmup_policy: str
    arm_order: str
    cache_protocol: str

    @property
    def query_context_transitions_max(self) -> int:
        """Maximum adjacent transitions exposed by the full query prefix."""

        return self.query_window - 1

    @property
    def short_query_context_transitions_max(self) -> int:
        """Maximum adjacent transitions exposed by the sensitivity prefix."""

        return self.short_query_window - 1


@dataclass(frozen=True)
class SyntheticInContextProtocol:
    """The exact explicit-demonstration sweep consumed by the manuscript."""

    version: str
    run_id: str
    source: str
    chain_version: str
    positions: int
    position_seed: int
    position_generation: str
    query_window: int
    example_window: int
    response_inventory: tuple[str, ...]
    context_window: int
    response_tokens: int
    prompt_token_overhead_margin: int
    example_seed: int
    example_draw_generation: str
    k_grid: tuple[int, ...]
    headline_k: int
    derangeable_draw_attempts: int
    warmup_policy: str
    arm_order: str
    cache_protocol: str

    @property
    def example_internal_transitions_per_window(self) -> int:
        """Chain transitions retained inside every treatment/control example."""

        return self.example_window - 1

    @property
    def headline_internal_transitions(self) -> int:
        """Maximum shared within-prefix transitions at the headline rung."""

        return self.headline_k * self.example_internal_transitions_per_window

    @property
    def headline_total_transitions(self) -> int:
        """Internal plus labeled terminal transitions shown at the headline rung."""

        return self.headline_k * self.example_window


SYNTHETIC_ZERO_SHOT_PROTOCOL = SyntheticZeroShotProtocol(
    version="synthetic-zero-shot-v3",
    run_id="synthetic_letter_probe_v3",
    source="scripts/record_synthetic_benchmark_claims.py",
    chain_version=SYNTHETIC_CHAIN_VERSION,
    positions=120,
    position_seed=DEFAULT_HELD_OUT_SEED,
    position_generation="independent-seeded-chains-v1",
    query_window=LETTER_WINDOW_SHORT,
    short_query_window=1,
    response_inventory=SYNTHETIC_LETTERS,
    context_window=8192,
    response_tokens=32,
    prompt_token_overhead_margin=256,
    warmup_policy="hosted-no-warmup-fresh-scored-cells-v1",
    arm_order="alternating-full-short-by-position-v1",
    cache_protocol=(
        f"{PROTOCOL_VERSION}:synthetic-zero-shot-v3:alternating-full-short-by-position-v1"
    ),
)
SYNTHETIC_INCONTEXT_PROTOCOL = SyntheticInContextProtocol(
    version="synthetic-incontext-v3",
    run_id="incontext_letter_probe_v3",
    source="scripts/record_incontext_benchmark_claims.py",
    chain_version=SYNTHETIC_CHAIN_VERSION,
    positions=400,
    position_seed=DEFAULT_HELD_OUT_SEED,
    position_generation="independent-seeded-chains-v1",
    query_window=LETTER_WINDOW_SHORT,
    # The comparator is order-1.  Each demonstration therefore exposes exactly
    # its sufficient statistic: previous symbol -> next symbol.  Longer prefixes
    # would retain unrelated chain transitions in the deranged arm and make k
    # understate the amount of evidence shown.
    example_window=1,
    response_inventory=SYNTHETIC_LETTERS,
    context_window=8192,
    response_tokens=32,
    prompt_token_overhead_margin=256,
    example_seed=20260813,
    example_draw_generation="per-position-derived-seed-v1",
    k_grid=(0, 4, 16, 48, 128),
    headline_k=128,
    derangeable_draw_attempts=64,
    warmup_policy="hosted-no-warmup-fresh-scored-cells-v1",
    arm_order="rotating-arms-by-position-v1",
    cache_protocol=(f"{PROTOCOL_VERSION}:synthetic-incontext-v3:rotating-arms-by-position-v1"),
)


SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL = SyntheticZeroShotProtocol(
    version="synthetic-phone-zero-shot-v2",
    run_id="synthetic_phone_probe_v2",
    source="scripts/record_synthetic_benchmark_claims.py",
    chain_version=SYNTHETIC_PHONE_CHAIN_VERSION,
    positions=120,
    position_seed=DEFAULT_PHONE_HELD_OUT_SEED,
    position_generation="independent-seeded-chains-v1",
    # Same query shape as the letter arm: the full arm exposes a bounded
    # phone prefix; the sensitivity arm reduces it to one phone. The
    # comparator is order-1, so the asymmetry this measures is the probe's
    # access to within-query chain evidence.
    query_window=LETTER_WINDOW_SHORT,
    short_query_window=1,
    response_inventory=SYNTHETIC_PHONES,
    context_window=8192,
    response_tokens=32,
    prompt_token_overhead_margin=256,
    warmup_policy="hosted-no-warmup-fresh-scored-cells-v1",
    arm_order="alternating-full-short-by-position-v1",
    cache_protocol=(
        f"{PROTOCOL_VERSION}:synthetic-phone-zero-shot-v2:alternating-full-short-by-position-v1"
    ),
)


def _validate_protocol_registry() -> None:
    """Reject an incoherent code registry at import, before any model calls."""
    zero = SYNTHETIC_ZERO_SHOT_PROTOCOL
    context = SYNTHETIC_INCONTEXT_PROTOCOL
    phone = SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL
    if zero.response_inventory != SYNTHETIC_LETTERS:
        raise RuntimeError("synthetic no-demonstration response inventory drifted from the chain")
    if context.response_inventory != SYNTHETIC_LETTERS:
        raise RuntimeError("synthetic in-context response inventory drifted from the chain")
    if phone.response_inventory != SYNTHETIC_PHONES:
        raise RuntimeError("synthetic phone response inventory drifted from the chain")
    if phone.chain_version != SYNTHETIC_PHONE_CHAIN_VERSION:
        raise RuntimeError("synthetic phone protocol names another chain than the level owns")
    if zero.chain_version != context.chain_version:
        raise RuntimeError("synthetic benchmark arms name different chain protocols")
    if zero.position_seed != context.position_seed:
        raise RuntimeError("synthetic benchmark arms name different position seeds")
    if zero.position_generation != context.position_generation:
        raise RuntimeError("synthetic benchmark arms name different position generators")
    if zero.query_window != context.query_window:
        raise RuntimeError("synthetic benchmark arms name different scored query windows")
    if not 0 < zero.short_query_window < zero.query_window:
        raise RuntimeError("synthetic sensitivity window must be positive and shorter than full")
    if zero.positions <= 0 or context.positions <= 0:
        raise RuntimeError("synthetic benchmark populations must be positive")
    if (
        zero.context_window <= zero.response_tokens
        or context.context_window <= context.response_tokens
    ):
        raise RuntimeError("synthetic context windows must leave room for a response")
    if zero.response_tokens <= 0 or context.response_tokens <= 0:
        raise RuntimeError("synthetic response-token limits must be positive")
    if zero.prompt_token_overhead_margin <= 0 or context.prompt_token_overhead_margin <= 0:
        raise RuntimeError("synthetic prompt-token overhead margins must be positive")
    if (
        not context.k_grid
        or context.k_grid[0] != 0
        or any(
            current <= previous
            for previous, current in zip(context.k_grid[:-1], context.k_grid[1:], strict=True)
        )
    ):
        raise RuntimeError("synthetic in-context grid must start at zero and increase")
    if context.headline_k != context.k_grid[-1]:
        raise RuntimeError("synthetic in-context headline must be the final declared rung")
    if context.derangeable_draw_attempts <= 0:
        raise RuntimeError("synthetic derangement draw budget must be positive")
    if not context.example_draw_generation:
        raise RuntimeError("synthetic example draws must name their seed derivation")
    if context.example_window <= 0:
        raise RuntimeError("synthetic example window must be positive")
    if not zero.warmup_policy or not context.warmup_policy:
        raise RuntimeError("synthetic benchmark arms must declare a warm-up policy")
    if not zero.arm_order or not context.arm_order:
        raise RuntimeError("synthetic benchmark arms must declare an arm-order policy")
    if not zero.cache_protocol or not context.cache_protocol:
        raise RuntimeError("synthetic benchmark arms must declare a cache protocol")
    if zero.cache_protocol == context.cache_protocol:
        raise RuntimeError("distinct synthetic measurements must not share cache keys")
    if (
        phone.context_window <= phone.response_tokens
        or phone.response_tokens <= 0
        or phone.prompt_token_overhead_margin <= 0
        or phone.positions <= 0
        or not 0 < phone.short_query_window < phone.query_window
    ):
        raise RuntimeError("synthetic phone protocol population and windows are incoherent")
    if not phone.warmup_policy or not phone.arm_order or not phone.cache_protocol:
        raise RuntimeError("synthetic phone protocol must declare its execution policy")
    if len({zero.cache_protocol, context.cache_protocol, phone.cache_protocol}) != 3:
        raise RuntimeError("distinct synthetic measurements must not share cache keys")
    g2p = SYNTHETIC_ALIGNED_G2P_PROTOCOL
    if g2p.response_inventory != SYNTHETIC_PHONES:
        raise RuntimeError("aligned g2p response inventory drifted from the canonical 39")
    if g2p.chain_version != SYNTHETIC_CHAIN_VERSION:
        raise RuntimeError("aligned g2p protocol names another chain than the level owns")
    if g2p.word_length < 2 or g2p.resample_attempts_cap <= 0:
        raise RuntimeError("aligned g2p word length and resample budget are incoherent")
    if g2p.train_words <= 0 or g2p.test_words <= 0:
        raise RuntimeError("aligned g2p populations must be positive")
    if g2p.held_out_seed_base < g2p.train_seed_base + g2p.train_words:
        raise RuntimeError("aligned g2p held-out seed domain overlaps the train domain")
    if not g2p.correspondence_version or not g2p.artifact_schema or not g2p.artifact_path:
        raise RuntimeError("aligned g2p protocol must name its correspondence, schema and artifact")
    if not g2p.warmup_policy or not g2p.arm_order or not g2p.cache_protocol:
        raise RuntimeError("aligned g2p protocol must declare its execution policy")
    if (
        len(
            {
                zero.cache_protocol,
                context.cache_protocol,
                phone.cache_protocol,
                g2p.cache_protocol,
                g2p.probe_cache_protocol,
            }
        )
        != 5
    ):
        raise RuntimeError("distinct synthetic measurements must not share cache keys")
    if g2p.probe_response_tokens <= 0 or g2p.probe_context_window <= g2p.probe_response_tokens:
        raise RuntimeError("aligned g2p probe windows are incoherent")
    if not g2p.probe_run_id or not g2p.probe_source:
        raise RuntimeError("aligned g2p probe arms must name their run and recorder")
    if not g2p.probe_warmup_policy or not g2p.probe_arm_order or not g2p.probe_cache_protocol:
        raise RuntimeError("aligned g2p probe arms must declare their execution policy")


_validate_protocol_registry()

__all__ = [
    "SYNTHETIC_ALIGNED_G2P_PROTOCOL",
    "SYNTHETIC_INCONTEXT_PROTOCOL",
    "SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL",
    "SYNTHETIC_ZERO_SHOT_PROTOCOL",
    "SyntheticAlignedG2pProtocol",
    "SyntheticInContextProtocol",
    "SyntheticZeroShotProtocol",
]
