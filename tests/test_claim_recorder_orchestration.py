"""Exact, fail-fast orchestration for the dynamically discovered recorders."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.orchestrate_claim_recorders import (
    AGGREGATE_RECORDER,
    FORMAL_RECORDER,
    NETWORK_VALIDATOR,
    PHASES,
    RECORDER_ORDER_EDGES,
    discover_recorders,
    network_phase_commands,
    partition_recorders,
    recorder_phases,
    run_commands,
    stable_topological_order,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "orchestrate_claim_recorders.py"
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def test_phase_partition_is_an_exact_dynamic_cover() -> None:
    discovered = discover_recorders()
    phases = recorder_phases()
    flattened = [path for phase in PHASES for path in phases[phase]]

    assert set(flattened) == set(discovered)
    assert len(flattened) == len(set(flattened))
    assert phases["formal"] == (FORMAL_RECORDER,)
    assert phases["aggregate"] == (AGGREGATE_RECORDER,)
    assert NETWORK_VALIDATOR not in flattened

    planted = "scripts/record_planted_future_claims.py"
    expanded = partition_recorders((*discovered, planted), phases["network"])
    assert planted in expanded["core"]


def test_every_producer_precedes_its_consumer_in_the_live_phase_map() -> None:
    phases = recorder_phases()
    positions = {
        path: (PHASES.index(phase), index)
        for phase, paths in phases.items()
        for index, path in enumerate(paths)
    }

    for producer, consumer in RECORDER_ORDER_EDGES:
        assert positions[producer] < positions[consumer], f"{producer} must precede {consumer}"

    assert phases["network"].index("scripts/record_synthetic_benchmark_claims.py") < phases[
        "network"
    ].index("scripts/record_incontext_benchmark_claims.py")
    assert phases["core"].index("scripts/record_stage1_claims.py") < phases["core"].index(
        "scripts/record_l1_context_control_claims.py"
    )


def test_predictive_loss_complete_claim_contract_has_graph_edges() -> None:
    from scripts.record_channel_predictive_loss_claims import REQUIRED_CLAIMS

    claims = json.loads((PROJECT_ROOT / "data" / "claims.json").read_text(encoding="utf-8"))[
        "claims"
    ]
    consumer = "scripts/record_channel_predictive_loss_claims.py"
    producers = {claims[label]["source"] for label in REQUIRED_CLAIMS}

    assert {(producer, consumer) for producer in producers} <= set(RECORDER_ORDER_EDGES)


def test_every_static_cross_recorder_claim_reference_has_an_order_path() -> None:
    """Close dependency discovery against the ledger's actual ownership map."""

    claims = json.loads((PROJECT_ROOT / "data" / "claims.json").read_text(encoding="utf-8"))[
        "claims"
    ]
    phases = recorder_phases()
    owner = {path: phase for phase, paths in phases.items() for path in paths}
    successors: dict[str, set[str]] = {path: set() for path in owner}
    for producer, consumer in RECORDER_ORDER_EDGES:
        successors[producer].add(consumer)

    def reachable(producer: str, consumer: str) -> bool:
        frontier = [producer]
        visited: set[str] = set()
        while frontier:
            node = frontier.pop()
            if node == consumer:
                return True
            if node not in visited:
                visited.add(node)
                frontier.extend(successors[node] - visited)
        return False

    missing_paths: list[tuple[str, str, str]] = []
    for consumer in discover_recorders():
        tree = ast.parse((PROJECT_ROOT / consumer).read_text(encoding="utf-8"))
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for label in literals & claims.keys():
            producer = str(claims[label].get("source", ""))
            if producer == consumer or producer not in owner:
                continue
            if owner[producer] == owner[consumer]:
                if not reachable(producer, consumer):
                    missing_paths.append((producer, consumer, label))
            elif PHASES.index(owner[producer]) >= PHASES.index(owner[consumer]):
                missing_paths.append((producer, consumer, label))

    assert missing_paths == []


def test_topological_order_overrides_lexical_order_and_detects_mutations() -> None:
    # The consumer sorts first: this is the exact shape that made alphabetical
    # orchestration accept a stale populated ledger and fail a fresh one.
    producer = "scripts/record_z_producer_claims.py"
    consumer = "scripts/record_a_consumer_claims.py"
    assert stable_topological_order((consumer, producer), ((producer, consumer),)) == (
        producer,
        consumer,
    )

    with pytest.raises(ValueError, match="cycle"):
        stable_topological_order(
            (producer, consumer),
            ((producer, consumer), (consumer, producer)),
        )
    with pytest.raises(ValueError, match="missing endpoints"):
        stable_topological_order((consumer,), ((producer, consumer),))


def test_partition_refuses_unknown_network_and_duplicate_recorders() -> None:
    discovered = discover_recorders()
    with pytest.raises(ValueError, match="unknown recorders"):
        partition_recorders(discovered, ("scripts/record_missing_claims.py",))
    with pytest.raises(ValueError, match="duplicates"):
        partition_recorders((*discovered, discovered[0]), ())

    reversed_phase_edge = ((FORMAL_RECORDER, discovered[0]),)
    with pytest.raises(ValueError, match="contradicts phase order"):
        partition_recorders(discovered, recorder_phases()["network"], reversed_phase_edge)


def test_list_mode_uses_the_real_discovery_without_running_recorders() -> None:
    result = subprocess.run(
        [str(PROJECT_PYTHON), str(SCRIPT), "network", "--list"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    listed = tuple(result.stdout.splitlines())
    expected = tuple(
        " ".join(command) for command in network_phase_commands(recorder_phases()["network"])
    )
    assert listed == expected
    assert listed[-1] == NETWORK_VALIDATOR
    for recorder in recorder_phases()["network"]:
        producer = listed.index(recorder)
        checkpoint = listed.index(f"{NETWORK_VALIDATOR} --recorder {recorder}")
        assert checkpoint == producer + 1


def test_network_checkpoint_plan_rejects_vacuity_and_keeps_final_set_validation() -> None:
    recorders = ("scripts/record_first_claims.py", "scripts/record_second_claims.py")
    commands = network_phase_commands(recorders)

    assert commands == (
        (recorders[0],),
        (NETWORK_VALIDATOR, "--recorder", recorders[0]),
        (recorders[1],),
        (NETWORK_VALIDATOR, "--recorder", recorders[1]),
        (NETWORK_VALIDATOR,),
    )
    planted = tuple(command for command in commands if "--recorder" not in command)
    assert planted != commands, "negative control failed to remove producer checkpoints"


@pytest.mark.timeout(600)
def test_real_reversed_dependency_fails_on_empty_ledger_then_ordered_pair_passes(
    tmp_path: Path,
) -> None:
    """Exercise a real producer/consumer pair, not only graph bookkeeping."""

    ledger = tmp_path / "claims.json"
    env = {**os.environ, "AILM_CLAIMS_PATH": str(ledger.resolve())}
    producer = "scripts/record_stage1_claims.py"
    consumer = "scripts/record_l1_context_control_claims.py"

    reversed_run = subprocess.run(
        [str(PROJECT_PYTHON), str(PROJECT_ROOT / consumer)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )
    assert reversed_run.returncode != 0
    assert "run scripts/record_stage1_claims.py first" in reversed_run.stderr
    assert json.loads(ledger.read_text(encoding="utf-8"))["claims"] == {}

    for recorder in stable_topological_order(
        (consumer, producer),
        ((producer, consumer),),
    ):
        completed = subprocess.run(
            [str(PROJECT_PYTHON), str(PROJECT_ROOT / recorder)],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=240,
        )
        assert completed.returncode == 0, completed.stderr

    claims = json.loads(ledger.read_text(encoding="utf-8"))["claims"]
    assert "stage1_letter_perplexity" in claims
    assert "l1_context_control_confirmed" in claims


@pytest.mark.parametrize(
    ("command", "diagnostic"),
    (
        (
            [str(PROJECT_PYTHON), "scripts/record_coupling_claims.py"],
            "run scripts/record_stage2_claims.py first",
        ),
        (
            [str(PROJECT_PYTHON), "scripts/record_g2p_famsplit_claims.py"],
            "run scripts/record_stage2_claims.py first",
        ),
        (
            [str(PROJECT_PYTHON), "scripts/record_channel_predictive_loss_claims.py"],
            "run the coupling and factorial recorders before this one",
        ),
        (
            [str(PROJECT_PYTHON), "scripts/record_generation_word_state_claims.py"],
            "run scripts/record_generation_claims.py first",
        ),
        (
            [str(PROJECT_PYTHON), "scripts/record_cluster_claims.py"],
            "owning recorders must run first",
        ),
        (
            [
                str(PROJECT_PYTHON),
                "-c",
                (
                    "from scripts.record_incontext_benchmark_claims import "
                    "_check_zero_shot_reproduces; "
                    "_check_zero_shot_reproduces('gemma3:4b', 'gemma3', "
                    "producer_scope='missing-producer-control')"
                ),
            ],
            "run scripts/record_synthetic_benchmark_claims.py first",
        ),
    ),
)
@pytest.mark.timeout(480)
def test_cross_recorder_guards_refuse_an_empty_ledger_before_measurement(
    tmp_path: Path,
    command: list[str],
    diagnostic: str,
) -> None:
    ledger = tmp_path / "claims.json"
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={**os.environ, "AILM_CLAIMS_PATH": str(ledger.resolve())},
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
    )

    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert json.loads(ledger.read_text(encoding="utf-8"))["claims"] == {}


def test_execution_stops_before_a_later_recorder_after_real_failure(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.txt"
    first = tmp_path / "first.py"
    failing = tmp_path / "failing.py"
    forbidden = tmp_path / "forbidden.py"
    first.write_text(
        f"from pathlib import Path\nPath({str(receipt)!r}).write_text('first\\n')\n",
        encoding="utf-8",
    )
    failing.write_text("raise SystemExit(7)\n", encoding="utf-8")
    forbidden.write_text(
        f"from pathlib import Path\nPath({str(receipt)!r}).write_text('forbidden\\n')\n",
        encoding="utf-8",
    )

    status = run_commands(
        (first.name, failing.name, forbidden.name),
        project_root=tmp_path,
        python=Path(sys.executable),
    )

    assert status == 7
    assert receipt.read_text(encoding="utf-8") == "first\n"
