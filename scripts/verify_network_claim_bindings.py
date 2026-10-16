#!/usr/bin/env python3
"""Validate provenance receipts for every network-excluded claim recorder."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ailm.call_cache import CallCache, default_cache_path  # noqa: E402
from ailm.replay_provenance import (  # noqa: E402
    NetworkClaimBindingError,
    declared_openrouter_provider,
    declared_run_scope,
    discover_network_recorders,
    validate_all_network_claim_bindings,
    validate_network_claim_binding,
)


def _load_recorder(path: Path) -> ModuleType:
    name = f"_ailm_replay_config_{path.stem}"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise NetworkClaimBindingError(f"cannot import network recorder configuration: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def expected_configurations(
    root: Path = PROJECT_ROOT,
    recorders: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Read the canonical publication configuration of each selected recorder.

    ``recorders`` defaults to the whole discovered set. It is narrowed when the
    caller is validating a subset, and that matters mid-phase: the network phase
    interleaves each recorder with its own checkpoint, so at the second
    checkpoint two recorders have been re-run and two have not. Building a
    configuration for a recorder whose receipt is still the previous run's would
    fail the checkpoint on a recorder it was not asked about.
    """

    configurations: dict[str, dict[str, object]] = {}
    for relative in discover_network_recorders(root) if recorders is None else recorders:
        module = _load_recorder(root / relative)
        provider = getattr(module, "publication_replay_configuration", None)
        if not callable(provider):
            raise NetworkClaimBindingError(
                f"network recorder {relative} has no publication_replay_configuration()"
            )
        scope = declared_run_scope(root, relative)
        route = declared_openrouter_provider(root, relative)
        try:
            if route is not None:
                configuration = provider(run_scope=scope, openrouter_provider=route)
            else:
                configuration = provider() if scope is None else provider(run_scope=scope)
        except TypeError as error:
            # A run-scoped recorder whose receipt states no scope has a receipt
            # from before its namespaces carried one, or no receipt at all.
            # Either way the receipt cannot be validated and the recorder has to
            # be re-run; say that rather than reporting a signature mismatch.
            raise NetworkClaimBindingError(
                f"network recorder {relative} scopes its cache namespaces by run, but its "
                f"receipt declares no run scope; re-run the recorder to write a current "
                f"receipt ({error})"
            ) from error
        if not isinstance(configuration, dict):
            raise NetworkClaimBindingError(
                f"network recorder {relative} returned a non-object configuration"
            )
        configurations[relative] = configuration
    return configurations


def _selected_recorders(root: Path, requested: list[str]) -> tuple[str, ...]:
    """Resolve an exact optional checkpoint subset against live discovery."""

    discovered = discover_network_recorders(root)
    if not requested:
        return discovered
    if len(requested) != len(set(requested)):
        raise NetworkClaimBindingError("--recorder values must be unique")
    unknown = sorted(set(requested) - set(discovered))
    if unknown:
        raise NetworkClaimBindingError(
            f"--recorder does not name a discovered network recorder: {unknown}"
        )
    return tuple(requested)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate network claim bindings and semantic measurement replay."
    )
    parser.add_argument(
        "--recorder",
        action="append",
        default=[],
        metavar="SCRIPTS/RECORD_*_CLAIMS.PY",
        help=(
            "validate only this dynamically discovered recorder; repeat for an exact "
            "checkpoint subset (default: validate the complete recorder set)"
        ),
    )
    return parser


def _validate_semantic_measurements(
    root: Path,
    validated: dict[str, dict[str, object]],
    store: CallCache,
) -> None:
    """Require every network exclusion to have an executable semantic proof."""

    for relative, receipt in validated.items():
        verifier = getattr(
            _load_recorder(root / relative),
            "validate_publication_measurement",
            None,
        )
        if receipt.get("measurement") is None:
            raise NetworkClaimBindingError(
                f"network recorder {relative} has no semantic measurement evidence"
            )
        if not callable(verifier):
            raise NetworkClaimBindingError(
                f"network recorder {relative} has no semantic measurement validator"
            )
        verifier(receipt, store)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selected = _selected_recorders(PROJECT_ROOT, args.recorder)
        configurations = expected_configurations(PROJECT_ROOT, selected)
        if args.recorder:
            validated = {
                relative: validate_network_claim_binding(
                    project_root=PROJECT_ROOT,
                    recorder=relative,
                    expected_configuration=configurations[relative],
                )
                for relative in selected
            }
        else:
            validated = validate_all_network_claim_bindings(
                project_root=PROJECT_ROOT,
                expected_configurations=configurations,
            )
        store = CallCache(default_cache_path(PROJECT_ROOT))
        _validate_semantic_measurements(PROJECT_ROOT, validated, store)
    except (OSError, ValueError) as error:
        print(f"network claim binding verification failed: {error}", file=sys.stderr)
        return 1
    total_claims = sum(int(receipt["claims"]["count"]) for receipt in validated.values())
    total_calls = sum(int(receipt["cache"]["record_count"]) for receipt in validated.values())
    print(
        f"network claim bindings verified: {len(validated)} recorder(s), "
        f"{total_claims} claim(s), {total_calls} exact v2 call record(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
