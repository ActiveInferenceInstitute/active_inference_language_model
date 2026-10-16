"""Strict fitted-model identity and provenance schema for the fitted-probe arm.

Separate experimental registry: entries here never enter
``ailm.benchmarks.constants.SERVED_PROBES`` — that registry is the single
identity source for every historical bench family (recorders, manuscript
hydration, figures), and adding to it would silently rewrite live surfaces.
This module owns its own namespace, its own validator, and its own consumers.

Two boundaries this module enforces by construction:

- **Lineage is digests, not behavior.** A lineage chain binds upstream
  revision, adapter, fused model, converted artifact, and served-tag identity
  as sha256 digests. No field here records or implies behavioral agreement
  with any served model; agreement checks live in the protocol's controls and
  can never substitute for the chain.
- **No model is claimed inspected.** Digest fields are declarations supplied
  by the implementer at fit time; nothing in this module fetches weights,
  tokenizers, or model cards.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping, Tuple

#: A digest is always spelled ``sha256:<64 lowercase hex>``.
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

#: The declared fitting tasks, in canonical order.
TASKS: tuple[str, ...] = ("g2p", "letter", "phoneme")

#: The ordered lineage stages of one fitted model.
LINEAGE_STAGES: tuple[str, ...] = ("upstream", "adapter", "fused", "converted", "served")


class FittedIdentityError(ValueError):
    """Raised for any malformed fitted-probe identity or lineage chain."""


def _require_digest(value: str, what: str) -> str:
    if not isinstance(value, str) or not DIGEST_PATTERN.match(value):
        raise FittedIdentityError(f"{what} must be 'sha256:<64 hex>', got {value!r}")
    return value


def sha256_digest(data: bytes) -> str:
    """Return the canonical ``sha256:<hex>`` digest of ``data``."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the canonical digest of ``text`` encoded as UTF-8."""
    return sha256_digest(text.encode("utf-8"))


@dataclass(frozen=True)
class FittedProbeSpec:
    """Identity of one fitted probe arm.

    ``key`` names the fitted ledger namespace; ``lineage_family`` must be a
    *registered* probe key (the fitted arm descends from that line); digests
    are filled at implementation time and validated here whenever present.
    ``served_tag`` is the local Ollama tag the converted artifact is imported
    under, and must differ from every registered probe's tag.
    """

    key: str
    lineage_family: str
    display_name: str
    tasks: Tuple[str, ...]
    upstream_repo: str
    upstream_revision: str
    precision: str
    served_tag: str
    adapter_digest: str = ""
    fused_digest: str = ""
    converted_digest: str = ""


def validate_fitted_probes(
    probes: tuple[FittedProbeSpec, ...],
    *,
    registered_keys: frozenset[str],
    registered_tags: frozenset[str],
) -> tuple[FittedProbeSpec, ...]:
    """Validate the fitted registry against the registered probe surface.

    Refuses empty or blank identity fields, unknown lineage families, tasks
    outside the declared set, malformed digests, duplicate keys/display
    names/served tags, and any served tag that would collide with a
    registered probe tag. An empty registry is valid: entries appear only
    when a fit actually produces one.
    """
    seen_keys: set[str] = set()
    seen_display: set[str] = set()
    seen_tags: set[str] = set()
    for index, probe in enumerate(probes):
        if not isinstance(probe, FittedProbeSpec):
            raise FittedIdentityError(
                f"FITTED_PROBES[{index}] must be FittedProbeSpec, got {type(probe).__name__}"
            )
        for simple in (
            "key",
            "lineage_family",
            "display_name",
            "upstream_repo",
            "precision",
            "served_tag",
        ):
            value = getattr(probe, simple)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise FittedIdentityError(
                    f"FITTED_PROBES[{index}].{simple} must be nonblank with no surrounding whitespace"
                )
        if probe.key in seen_keys:
            raise FittedIdentityError(f"duplicate fitted key {probe.key!r}")
        seen_keys.add(probe.key)
        if probe.display_name.casefold() in seen_display:
            raise FittedIdentityError(f"duplicate fitted display name {probe.display_name!r}")
        seen_display.add(probe.display_name.casefold())
        if probe.served_tag in seen_tags:
            raise FittedIdentityError(f"duplicate fitted served tag {probe.served_tag!r}")
        seen_tags.add(probe.served_tag)
        if probe.served_tag in registered_tags:
            raise FittedIdentityError(
                f"fitted served tag {probe.served_tag!r} collides with a registered probe tag"
            )
        if probe.lineage_family not in registered_keys:
            raise FittedIdentityError(
                f"lineage_family {probe.lineage_family!r} is not a registered probe key"
            )
        if not probe.tasks or tuple(probe.tasks) != tuple(
            t for t in TASKS if t in set(probe.tasks)
        ):
            raise FittedIdentityError(
                f"tasks must be a nonempty ordered subset of {TASKS}, got {probe.tasks!r}"
            )
        for digest_field in (
            "upstream_revision",
            "adapter_digest",
            "fused_digest",
            "converted_digest",
        ):
            value = getattr(probe, digest_field)
            if value:
                _require_digest(value, f"FITTED_PROBES[{index}].{digest_field}")
    return tuple(probes)


def _registered_surface() -> tuple[frozenset[str], frozenset[str]]:
    """The registered probe keys and tags, read without expanding them."""
    from ailm.benchmarks.constants import PROBE_MODEL_TAGS, SERVED_PROBES

    return (
        frozenset(probe.key for probe in SERVED_PROBES),
        frozenset(PROBE_MODEL_TAGS.values()),
    )


def validate_current_registry() -> tuple[FittedProbeSpec, ...]:
    """Validate the module-level registry against the live registered surface."""
    keys, tags = _registered_surface()
    return validate_fitted_probes(FITTED_PROBES, registered_keys=keys, registered_tags=tags)


#: The separate experimental registry. Empty by construction until a fit
#: registers an identity; never a ``SERVED_PROBES`` entry.
FITTED_PROBES: tuple[FittedProbeSpec, ...] = ()


def build_lineage_chain(stages: Mapping[str, str]) -> Tuple[Tuple[str, str], ...]:
    """Build the ordered lineage chain from stage digests.

    Requires exactly the five declared stages, in order, each carrying a
    well-formed digest. The chain is a digest binding only — it establishes
    lineage and makes no behavioral claim about any served or unfitted model.
    """
    missing = [stage for stage in LINEAGE_STAGES if stage not in stages]
    if missing:
        raise FittedIdentityError(f"lineage chain is missing stages: {missing}")
    extra = sorted(set(stages) - set(LINEAGE_STAGES))
    if extra:
        raise FittedIdentityError(f"lineage chain has unknown stages: {extra}")
    chain: list[tuple[str, str]] = []
    for stage in LINEAGE_STAGES:
        chain.append((stage, _require_digest(stages[stage], f"lineage stage {stage!r}")))
    return tuple(chain)
