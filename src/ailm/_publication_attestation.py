"""Bounded publication derivative attestation responsibility."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

from . import _publication_types as t
from ._publication_inventory import execution_snapshot_inventory, file_hash, read_json
from ._publication_proof import (
    live_serving_identity,
    original_execution_identity,
    retained_execution,
)


class ControlProof(TypedDict):
    baseline: t.ExecutionRecord
    denied: dict[str, t.ExecutionRecord]


class ArtistProof(TypedDict):
    original: t.ExecutionRecord
    current: t.ExecutionRecord
    mutation: t.ExecutionRecord
    equal: bool


class SelfProof(TypedDict):
    original: t.ExecutionRecord
    strict_current: t.ExecutionRecord
    derivative: t.ExecutionRecord
    numerical_mutation: t.ExecutionRecord


class AttestationData(TypedDict):
    schema: str
    output_path: str
    run_directory: str
    mode: str
    accepted: bool
    measurement_reused: bool
    fresh_hosted_calls: int
    contract_sha256: str
    declaration_sha256: str
    measurement_commit: str
    measurement_tree: str
    publication_commit: str
    publication_tree: str
    bootstrap_sha256: str
    original_verifier_sha256: str
    initial_state: t.JSONMap
    final_state: t.JSONMap
    materialization: t.Materialization
    dependency_before: t.JSONMap
    dependency_after: t.JSONMap
    semantic: t.ExecutionRecord | None
    transport_controls: ControlProof | None
    artists: ArtistProof | None
    selfproof: SelfProof | None
    live: t.ExecutionRecord | None
    final_qa: str


def validate_attestation(
    path: Path, contract: t.DerivativeContract, state: t.DerivativeState, mode: str
) -> t.AttestationResult:
    """Revalidate proof bytes and state; a candidate can never certify a release."""
    errors: list[str] = []
    sha = ""
    try:
        sha = file_hash(path)
        row = t.exact_keys(
            read_json(path),
            set(AttestationData.__annotations__),
            "attestation",
        )
        t.require(
            row["output_path"] == str(path.absolute()),
            "attestation output path does not bind fresh execution",
        )
        run_directory = Path(t.json_text(row["run_directory"]))
        t.require(
            run_directory.is_absolute()
            and run_directory.parent == Path(contract.snapshot_parent)
            and run_directory.name.startswith("publication-" + mode + "-"),
            "run directory outside controlled destination",
        )

        def execution_proof(value: t.JSONValue, expected_exit: int | None = 0) -> t.ExecutionRecord:
            return retained_execution(value, run_directory, expected_exit)

        t.require(
            row["schema"] == t.ATTESTATION_SCHEMA
            and row["mode"] == mode
            and mode in {"offline", "live"},
            "wrong attestation mode/schema",
        )
        t.require(
            row["measurement_reused"] is True
            and type(row["fresh_hosted_calls"]) is int
            and row["fresh_hosted_calls"] == 0,
            "measurement/call boundary invalid",
        )
        for name in (
            "contract_sha256",
            "declaration_sha256",
            "measurement_commit",
            "measurement_tree",
            "publication_commit",
            "publication_tree",
        ):
            t.require(row[name] == getattr(contract, name), f"attestation {name} mismatch")
        t.require(
            row["bootstrap_sha256"] == contract.support["scripts/_publication_execution.py"],
            "bootstrap bytes unbound",
        )
        t.require(
            row["original_verifier_sha256"]
            == contract.original_inventory["scripts/verify_network_claim_bindings.py"].sha256,
            "original verifier unbound",
        )
        t.require(
            row["initial_state"]
            == row["final_state"]
            == json.loads(t.canonical_bytes(asdict(state))),
            "stale/drifted attestation state",
        )
        t.require(
            row["dependency_before"] == row["dependency_after"]
            and isinstance(row["dependency_before"], dict)
            and row["dependency_before"].get("integrity_ok") is True,
            "dependency integrity/drift",
        )
        materialization = t.exact_keys(
            row["materialization"],
            {"snapshot", "pre_overlay", "before", "after"},
            "materialization",
        )
        t.require(
            materialization["pre_overlay"]
            == json.loads(
                t.canonical_bytes({p: asdict(i) for p, i in contract.original_inventory.items()})
            ),
            "incomplete pinned tree materialization",
        )
        t.require(
            materialization["before"] == materialization["after"]
            and isinstance(materialization["before"], dict)
            and bool(materialization["before"]),
            "snapshot drift/empty inventory",
        )
        snapshot = Path(t.json_text(materialization["snapshot"]))
        t.require(
            snapshot == run_directory / "original", "snapshot is not fresh materialized source"
        )
        t.require(
            execution_snapshot_inventory(snapshot) == materialization["after"],
            "actual retained snapshot differs",
        )
        expected_files = {
            p: i.sha256 for p, i in contract.original_inventory.items()
        } | contract.inputs
        t.require(
            set(t.json_map(materialization["before"])) == set(expected_files),
            "extra/missing execution files",
        )
        for name, value in t.json_map(materialization["before"]).items():
            t.require(
                t.json_map(value).get("sha256") == expected_files[name]
                and t.json_map(value).get("permissions")
                == (
                    "0555"
                    if contract.original_inventory.get(name, t.FileIdentity("", 0, "100644")).mode
                    == "100755"
                    else "0444"
                ),
                f"snapshot identity/permissions mismatch: {name}",
            )
        if mode == "offline":
            semantic = execution_proof(row["semantic"])
            original_execution_identity(semantic, snapshot, contract, "offline")
            expected = f"network claim bindings verified: {len(contract.receipts)} recorder(s), {sum(x.claims_count for x in contract.receipts)} claim(s), {sum(x.calls_count for x in contract.receipts)} exact v2 call record(s)"
            t.require(
                expected in semantic["stdout"] and semantic["report"].get("policy") == "offline",
                "missing complete semantic evidence",
            )
            controls = t.exact_keys(
                row["transport_controls"], {"baseline", "denied"}, "transport controls"
            )
            baseline = execution_proof(controls["baseline"])
            t.require(
                baseline["report"].get("reached") is True, "transport positive control not live"
            )
            t.require(
                set(t.json_map(controls["denied"]))
                == {"connect", "connect_ex", "dns", "udp", "spawn"},
                "incomplete transport controls",
            )
            for name, control in t.json_map(controls["denied"]).items():
                proof = execution_proof(control)
                t.require(
                    proof["report"].get("denied") is True
                    and proof["report"].get("operation") == name
                    and proof["report"].get("reached") is False,
                    "transport denial control failed",
                )
            artists = t.exact_keys(
                row["artists"], {"original", "current", "mutation", "equal"}, "artist proof"
            )
            a, b = execution_proof(artists["original"]), execution_proof(artists["current"])
            c = execution_proof(artists["mutation"])
            t.require(
                artists["equal"] is True
                and a["report"]["artists"] == b["report"]["artists"]
                and a["report"]["artists"] != c["report"]["artists"],
                "numeric artist proof/control failed",
            )
            t.require(
                a["report"]["tokens_sha256"]
                == b["report"]["tokens_sha256"]
                == c["report"]["tokens_sha256"]
                == contract.tokens_sha256,
                "artist tokens unbound",
            )
            selfproof = t.exact_keys(
                row["selfproof"],
                {"original", "strict_current", "derivative", "numerical_mutation"},
                "four-outcome selfproof",
            )
            t.require(
                selfproof["original"] == row["semantic"],
                "selfproof original not actual semantic execution",
            )
            strict = execution_proof(selfproof["strict_current"], None)
            t.require(
                strict["exit_code"] != 0
                and "source" in strict["stderr"]
                and "stale" in strict["stderr"],
                "strict current did not fail on source binding",
            )
            for name, outcome in (("derivative", True), ("numerical_mutation", False)):
                case = execution_proof(selfproof[name], 0 if outcome else 1)
                t.require(
                    case["report"].get("valid") is outcome, "derivative selfproof outcome wrong"
                )
            t.require(row["live"] is None, "offline result falsely contains live proof")
        else:
            t.require(
                all(
                    row[k] is None
                    for k in ("semantic", "transport_controls", "artists", "selfproof")
                ),
                "live proof mixes offline claims",
            )
            live = execution_proof(row["live"])
            t.require(
                live["report"].get("policy") == "live"
                and live["report"].get("endpoint") == contract.endpoint,
                "live endpoint/mode mismatch",
            )
            original_execution_identity(live, snapshot, contract, "live")
            live_serving_identity(live, snapshot, contract)
            t.require(
                "--verify-cache" in t.json_strings(live["report"].get("actual_argv", []))
                and "25" in t.json_strings(live["report"].get("actual_argv", []))
                and "across 6 of 6 declared stratum(s)" in live["stdout"],
                "missing actual complete-strata live audit",
            )
        t.require(
            row["final_qa"] == "pending-parent-independent-all-page-review", "invented final QA"
        )
        t.require(row["accepted"] is contract.reviewed, "acceptance/review mismatch")
        t.require(
            contract.reviewed, "candidate development evidence cannot satisfy release acceptance"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return t.AttestationResult(
        not errors, mode, sha, contract.contract_sha256, state.fingerprint, tuple(errors)
    )
