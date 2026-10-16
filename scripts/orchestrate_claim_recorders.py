#!/usr/bin/env python3
"""Run the complete dynamically discovered claims-recorder inventory by phase.

The repository has enough recorders that a hand-copied command list cannot stay
complete. This orchestrator derives the inventory from ``record_*_claims.py``,
partitions it by actual network reachability, and orders each phase with a small
dependency graph.  The graph records only producer -> consumer edges; it is not
a second recorder inventory, so a newly added recorder is still discovered and
run automatically:

``core``
    Every offline recorder except the formal and aggregate writers.
``network``
    Every recorder discovered to call the local model server, followed by the
    independent binding validator.
``aggregate``
    The statistical aggregate recorder, after every contributing phase.
``formal``
    The formal-claims recorder, after the caller has regenerated and rebuilt
    the Lean ledger bridge from the aggregate-complete claims ledger.

Each phase is fail-fast. A failed recorder leaves later recorders unexecuted;
rerunning the phase is safe because recorders replace their own ledger namespace
under the shared single-writer lock.
"""

from __future__ import annotations

import argparse
import heapq
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
FORMAL_RECORDER = "scripts/record_formal_claims.py"
AGGREGATE_RECORDER = "scripts/record_statistical_claims.py"
NETWORK_VALIDATOR = "scripts/verify_network_claim_bindings.py"
PHASES = ("core", "network", "aggregate", "formal")

# Only real data dependencies belong here.  This deliberately is not a list of
# recorders: discovery below owns that inventory.  The edges close two failure
# modes that lexical ordering cannot: a fresh ledger missing a producer, and a
# populated ledger letting a consumer validate stale values before the producer
# overwrites them later in the same run.
RECORDER_ORDER_EDGES: tuple[tuple[str, str], ...] = (
    # Ledger-backed population and anchor cross-checks.
    ("scripts/record_stage1_claims.py", "scripts/record_l1_context_control_claims.py"),
    ("scripts/record_stage2_claims.py", "scripts/record_coupling_claims.py"),
    ("scripts/record_stage2_claims.py", "scripts/record_g2p_famsplit_claims.py"),
    ("scripts/record_stage2_claims.py", "scripts/record_cluster_claims.py"),
    ("scripts/record_stage4_claims.py", "scripts/record_cluster_claims.py"),
    ("scripts/record_czech_case_claims.py", "scripts/record_cluster_claims.py"),
    ("scripts/record_stage6_claims.py", "scripts/record_cluster_claims.py"),
    (
        "scripts/record_cluster_claims.py",
        "scripts/record_confirmatory_stage6_clustered_claims.py",
    ),
    ("scripts/record_generation_claims.py", "scripts/record_generation_word_state_claims.py"),
    # The proper-loss probe authenticates both the primary coupling anchor and
    # the factorial cell it names before writing its own result.
    ("scripts/record_coupling_claims.py", "scripts/record_channel_predictive_loss_claims.py"),
    (
        "scripts/record_channel_factorial_claims.py",
        "scripts/record_channel_predictive_loss_claims.py",
    ),
    (
        "scripts/record_geometric_pooling_claims.py",
        "scripts/record_channel_predictive_loss_claims.py",
    ),
    ("scripts/record_fitted_channel_claims.py", "scripts/record_channel_predictive_loss_claims.py"),
    # Network/cache producers.  The log-probability drift sample is drawn from
    # the external benchmark's authenticated calls; the in-context arm must
    # reproduce the zero-shot synthetic arm on its recorded population.
    ("scripts/record_external_benchmark_claims.py", "scripts/record_logprob_drift_claims.py"),
    (
        "scripts/record_synthetic_benchmark_claims.py",
        "scripts/record_incontext_benchmark_claims.py",
    ),
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ailm.replay_provenance import discover_network_recorders  # noqa: E402


def discover_recorders(project_root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Return every claims recorder in deterministic discovery order."""

    recorders = sorted(
        path.relative_to(project_root).as_posix()
        for path in (project_root / "scripts").glob("record_*_claims.py")
    )
    return tuple(recorders)


def stable_topological_order(
    nodes: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> tuple[str, ...]:
    """Return a lexical-stable topological order and reject a broken graph.

    Every edge endpoint must be present.  Silently discarding an edge when a
    recorder was renamed would recreate the stale-producer defect this graph is
    meant to prevent.  A min-heap makes unrelated recorders deterministic while
    allowing the dependency edges to override lexical order.
    """

    inventory = tuple(nodes)
    if len(inventory) != len(set(inventory)):
        raise ValueError("topological inventory contains duplicates")
    known = set(inventory)
    successors: dict[str, set[str]] = {node: set() for node in inventory}
    indegree = dict.fromkeys(inventory, 0)
    normalized_edges = tuple(edges)
    for producer, consumer in normalized_edges:
        missing = sorted({producer, consumer} - known)
        if missing:
            raise ValueError(f"recorder dependency names missing endpoints: {missing}")
        if consumer not in successors[producer]:
            successors[producer].add(consumer)
            indegree[consumer] += 1

    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        ordered.append(node)
        for successor in sorted(successors[node]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if len(ordered) != len(inventory):
        cyclic = sorted(node for node, degree in indegree.items() if degree > 0)
        raise ValueError(f"claims-recorder dependency graph contains a cycle: {cyclic}")
    return tuple(ordered)


def partition_recorders(
    recorders: Iterable[str],
    network_recorders: Iterable[str],
    dependencies: Iterable[tuple[str, str]] = RECORDER_ORDER_EDGES,
) -> dict[str, tuple[str, ...]]:
    """Partition and dependency-order one exact recorder inventory."""

    inventory = tuple(recorders)
    if len(set(inventory)) != len(inventory):
        raise ValueError("claims-recorder inventory contains duplicates")
    known = set(inventory)
    network = set(network_recorders)
    unknown_network = sorted(network - known)
    if unknown_network:
        raise ValueError(f"network discovery returned unknown recorders: {unknown_network}")
    missing_special = [path for path in (FORMAL_RECORDER, AGGREGATE_RECORDER) if path not in known]
    if missing_special:
        raise ValueError(f"required order-sensitive recorders are missing: {missing_special}")

    special = {FORMAL_RECORDER, AGGREGATE_RECORDER}
    phase_members = {
        "core": tuple(path for path in inventory if path not in network | special),
        "network": tuple(path for path in inventory if path in network),
        "aggregate": (AGGREGATE_RECORDER,),
        "formal": (FORMAL_RECORDER,),
    }
    owner = {path: phase for phase, paths in phase_members.items() for path in paths}
    phase_rank = {phase: index for index, phase in enumerate(PHASES)}
    dependency_edges = tuple(dependencies)
    for producer, consumer in dependency_edges:
        missing = sorted({producer, consumer} - known)
        if missing:
            raise ValueError(f"recorder dependency names missing endpoints: {missing}")
        if phase_rank[owner[producer]] > phase_rank[owner[consumer]]:
            raise ValueError(
                "recorder dependency contradicts phase order: "
                f"{producer} ({owner[producer]}) -> {consumer} ({owner[consumer]})"
            )

    phases = {
        phase: stable_topological_order(
            paths,
            (
                (producer, consumer)
                for producer, consumer in dependency_edges
                if owner[producer] == phase and owner[consumer] == phase
            ),
        )
        for phase, paths in phase_members.items()
    }
    flattened = [path for phase in PHASES for path in phases[phase]]
    if len(flattened) != len(set(flattened)) or set(flattened) != known:
        raise ValueError("claims-recorder phase partition is not an exact disjoint cover")
    return phases


def recorder_phases(project_root: Path = PROJECT_ROOT) -> dict[str, tuple[str, ...]]:
    """Return the live phase map from production discovery."""

    return partition_recorders(
        discover_recorders(project_root),
        discover_network_recorders(project_root),
    )


def run_commands(
    commands: Sequence[str | Sequence[str]],
    *,
    project_root: Path = PROJECT_ROOT,
    python: Path = VENV_PYTHON,
) -> int:
    """Run repository-script commands sequentially and stop at first failure."""

    for command in commands:
        parts = (command,) if isinstance(command, str) else tuple(command)
        if not parts or any(not isinstance(part, str) or not part for part in parts):
            raise ValueError(f"invalid repository command: {command!r}")
        relative, *arguments = parts
        print(f"==> {' '.join(parts)}", flush=True)
        result = subprocess.run(
            [str(python), str(project_root / relative), *arguments],
            cwd=project_root,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"claims phase stopped: {relative} exited {result.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return result.returncode
    return 0


def network_phase_commands(recorders: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Interleave each network producer with its exact semantic checkpoint.

    The final unscoped validator proves that discovery, configuration, and
    receipt coverage are complete as a set.  Per-recorder checkpoints serve a
    different purpose: no consumer runs after its producer left an invalid
    semantic receipt.
    """

    commands: list[tuple[str, ...]] = []
    for recorder in recorders:
        commands.append((recorder,))
        commands.append((NETWORK_VALIDATOR, "--recorder", recorder))
    commands.append((NETWORK_VALIDATOR,))
    return tuple(commands)


def _project_interpreter_is_active() -> bool:
    """Whether this process came from the sidecar's exact virtual environment."""

    return VENV_PYTHON.exists() and Path(sys.executable).resolve() == VENV_PYTHON.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the phase's discovered commands without executing them",
    )
    args = parser.parse_args(argv)

    phases = recorder_phases()
    commands: Sequence[str | Sequence[str]] = phases[args.phase]
    if args.phase == "network":
        commands = network_phase_commands(phases[args.phase])
    if args.list:
        print(
            "\n".join(
                command if isinstance(command, str) else " ".join(command) for command in commands
            )
        )
        return 0
    if not _project_interpreter_is_active():
        parser.error(f"run with the exact project interpreter: {VENV_PYTHON}")
    return run_commands(commands)


if __name__ == "__main__":
    raise SystemExit(main())
