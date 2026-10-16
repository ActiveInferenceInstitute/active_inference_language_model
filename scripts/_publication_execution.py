"""Durable subprocess execution and two-source publication orchestration."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ailm._publication_attestation import ArtistProof, AttestationData, ControlProof, SelfProof
    from ailm._publication_types import DerivativeContract, ExecutionRecord

if not __package__:
    __package__ = "scripts"
    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

from ._publication_artists import internal_artists as internal_artists
from ._publication_bootstrap import PROJECT_ROOT as PROJECT_ROOT
from ._publication_bootstrap import audited_startup as audited_startup
from ._publication_bootstrap import configure_imports as configure_imports
from ._publication_bootstrap import dependency_state as dependency_state
from ._publication_bootstrap import install_policy as install_policy
from ._publication_bootstrap import internal_dependencies as internal_dependencies
from ._publication_bootstrap import internal_original as internal_original
from ._publication_bootstrap import origins as origins
from ._publication_bootstrap import sanitized_environment as sanitized_environment
from ._publication_bootstrap import write_report as write_report
from ._publication_controls import internal_predicate as internal_predicate
from ._publication_controls import internal_transport as internal_transport
from ._publication_snapshot import materialize as materialize
from ._publication_snapshot import mutation_tree as mutation_tree
from ._publication_snapshot import snapshot_inventory as snapshot_inventory


def execution(
    root: Path,
    work: Path,
    name: str,
    internal: str,
    arguments: list[str],
    endpoint: str,
    *,
    timeout: int = 1800,
) -> ExecutionRecord:
    from ailm.publication_derivative import canonical_bytes, digest, read_json

    report_path = work / (name + ".json")
    # Each internal signature places report at a documented index.
    report_indexes = {
        "internal-original": 3,
        "internal-artists": 3,
        "internal-transport": 1,
        "internal-predicate": 4,
    }
    argv_arguments = list(arguments)
    argv_arguments[report_indexes[internal]] = str(report_path)
    python = root / ".venv/bin/python"
    bootstrap = root / "scripts/_publication_execution.py"
    argv = [str(python), "-I", "-S", "-B", str(bootstrap), internal] + argv_arguments
    environment = sanitized_environment(endpoint, work)
    execution_cwd = (
        Path(argv_arguments[0])
        if internal in {"internal-original", "internal-artists", "internal-predicate"}
        else work
    )
    completed = subprocess.run(
        argv,
        cwd=execution_cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        close_fds=True,
        timeout=timeout,
    )
    (work / (name + ".stdout")).write_text(completed.stdout)
    (work / (name + ".stderr")).write_text(completed.stderr)
    if not report_path.is_file():
        raise ValueError(
            f"{name} produced no typed execution report: exit={completed.returncode}, stderr={completed.stderr}"
        )
    report = read_json(report_path)
    result: ExecutionRecord = {
        "argv": argv,
        "cwd": str(execution_cwd),
        "environment": environment,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": report,
        "report_sha256": digest(canonical_bytes(report)),
    }
    write_report(work / (name + ".execution.json"), result)
    return result


def run_derivative(
    contract: DerivativeContract,
    root: Path,
    mode: str,
    output: Path,
    contract_path: Path,
    *,
    candidate: bool = False,
) -> AttestationData:
    from ailm._publication_types import json_map, json_text
    from ailm.publication_derivative import ATTESTATION_SCHEMA, capture_derivative_state, require

    require(mode in {"offline", "live"}, "invalid derivative mode")
    require(not output.exists(), "attestation output already exists; select a fresh path")
    initial = capture_derivative_state(contract, root)
    work = Path(tempfile.mkdtemp(prefix="publication-" + mode + "-", dir=contract.snapshot_parent))
    dependency_before = dependency_state(root)
    materialization = materialize(contract, root, work / "original")
    snapshot = Path(materialization["snapshot"])
    site = json_text(dependency_before["site_packages"])
    endpoint = contract.endpoint

    def run(name: str, internal: str, arguments: list[str]) -> ExecutionRecord:
        return execution(root, work, name, internal, arguments, endpoint)

    semantic: ExecutionRecord | None = None
    live: ExecutionRecord | None = None
    controls: ControlProof | None = None
    artists: ArtistProof | None = None
    selfproof: SelfProof | None = None
    if mode == "offline":
        semantic = run(
            "original-semantic", "internal-original", [str(snapshot), site, "offline", "", endpoint]
        )
        require(
            semantic["exit_code"] == 0, f"original semantic verifier failed: {semantic['stderr']}"
        )
        baseline = run("transport-baseline", "internal-transport", ["baseline", "", endpoint])
        denied = {
            name: run("transport-" + name, "internal-transport", [name, "", endpoint])
            for name in ("connect", "connect_ex", "dns", "udp", "spawn")
        }
        require(
            baseline["exit_code"] == 0 and all(x["exit_code"] == 0 for x in denied.values()),
            "transport controls failed",
        )
        controls = {"baseline": baseline, "denied": denied}
        tokens = work / "artist-tokens.json"
        shutil.copyfile(root / contract.tokens_path, tokens, follow_symlinks=False)
        from ailm.publication_derivative import file_hash

        require(file_hash(tokens) == contract.tokens_sha256, "artist token copy drift")
        artist_runs = {
            name: run(
                "artists-" + name,
                "internal-artists",
                [str(source), site, str(tokens), "", endpoint, variant],
            )
            for name, source, variant in (
                ("original", snapshot, "normal"),
                ("current", root, "normal"),
                ("mutation", root, "mutation"),
            )
        }
        original, current, mutation = (
            artist_runs[name] for name in ("original", "current", "mutation")
        )
        require(
            all(x["exit_code"] == 0 for x in (original, current, mutation)),
            "artist execution failed",
        )
        original_artists = json_map(original["report"]["artists"])
        current_artists = json_map(current["report"]["artists"])
        mutated_artists = json_map(mutation["report"]["artists"])
        require(
            original_artists == current_artists
            and all(
                original_artists[name] != mutated_artists[name]
                for name in ("case_word_sweep", "external_benchmarks", "multiplicity_forest")
            ),
            "artist equality/negative control failed",
        )
        artists = {"original": original, "current": current, "mutation": mutation, "equal": True}
        strict = run(
            "strict-current", "internal-original", [str(root), site, "strict-current", "", endpoint]
        )
        require(
            strict["exit_code"] != 0
            and "stale" in strict["stderr"]
            and "source" in strict["stderr"],
            "strict current source refusal was not reproduced",
        )
        kind = "candidate" if candidate else "reviewed"

        def predicate(name: str, source: Path, contract_input: Path) -> ExecutionRecord:
            return run(
                name,
                "internal-predicate",
                [str(source), kind, str(contract_input), contract.snapshot_parent, ""],
            )

        good = predicate("predicate-current", root, contract_path)
        mutation_root = work / "numerical-mutation"
        mutation_tree(contract, root, mutation_root)
        mutant_declaration = (
            mutation_root / "data/publication_derivative.json" if candidate else contract_path
        )
        bad = predicate("predicate-numerical-mutation", mutation_root, mutant_declaration)
        require(
            good["exit_code"] == 0
            and bad["exit_code"] == 1
            and good["report"].get("valid") is True
            and bad["report"].get("valid") is False,
            "four-outcome selfproof failed",
        )
        selfproof = {
            "original": semantic,
            "strict_current": strict,
            "derivative": good,
            "numerical_mutation": bad,
        }
    else:
        live = run(
            "original-live", "internal-original", [str(snapshot), site, "live", "", endpoint]
        )
        require(
            live["exit_code"] == 0 and "across 6 of 6 declared stratum(s)" in live["stdout"],
            f"original live cache audit failed: {live['stdout']} {live['stderr']}",
        )
    materialization["after"] = snapshot_inventory(snapshot)
    require(
        materialization["before"] == materialization["after"],
        "original snapshot changed during execution",
    )
    dependency_after = dependency_state(root)
    require(dependency_before == dependency_after, "dependency drift during execution")
    final = capture_derivative_state(contract, root)
    require(initial == final, "publication/input/evidence drift during execution")
    attestation: AttestationData = {
        "schema": ATTESTATION_SCHEMA,
        "output_path": str(output.absolute()),
        "run_directory": str(work),
        "mode": mode,
        "accepted": contract.reviewed,
        "measurement_reused": True,
        "fresh_hosted_calls": 0,
        "contract_sha256": contract.contract_sha256,
        "declaration_sha256": contract.declaration_sha256,
        "measurement_commit": contract.measurement_commit,
        "measurement_tree": contract.measurement_tree,
        "publication_commit": contract.publication_commit,
        "publication_tree": contract.publication_tree,
        "bootstrap_sha256": contract.support["scripts/_publication_execution.py"],
        "original_verifier_sha256": contract.original_inventory[
            "scripts/verify_network_claim_bindings.py"
        ].sha256,
        "initial_state": asdict(initial),
        "final_state": asdict(final),
        "materialization": materialization,
        "dependency_before": dependency_before,
        "dependency_after": dependency_after,
        "semantic": semantic,
        "transport_controls": controls,
        "artists": artists,
        "selfproof": selfproof,
        "live": live,
        "final_qa": "pending-parent-independent-all-page-review",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(output, attestation)
    return attestation


def main() -> int:
    actions = {
        "internal-dependencies": internal_dependencies,
        "internal-original": internal_original,
        "internal-artists": internal_artists,
        "internal-transport": internal_transport,
        "internal-predicate": internal_predicate,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        raise ValueError("internal execution mode required")
    return actions[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
