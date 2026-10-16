"""Bounded publication derivative proof responsibility."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import _publication_types as t
from ._publication_inventory import (
    file_hash,
    file_identity,
    original_literal,
    read_json,
    regular_bytes,
)
from ._publication_types import (
    DerivativeContract,
    ExecutionRecord,
    T,
    canonical_bytes,
    digest,
    exact_keys,
    hash_string,
    json_integer,
    json_map,
    json_strings,
    json_text,
    require,
)


def _execution(row: T, expected_exit: int | None = 0) -> ExecutionRecord:
    fields = exact_keys(
        row,
        {"argv", "cwd", "environment", "exit_code", "stdout", "stderr", "report", "report_sha256"},
        "execution",
    )
    argv = json_strings(fields["argv"])
    require(len(argv) > 3 and all(argv), "empty execution argv")
    require(all(flag in argv for flag in ("-I", "-S", "-B")), "execution not isolated")
    status = json_integer(fields["exit_code"])
    require(expected_exit is None or status == expected_exit, "execution exit mismatch")
    stdout, stderr = json_text(fields["stdout"]), json_text(fields["stderr"])
    require(bool(stdout or stderr), "empty execution evidence")
    report = json_map(fields["report"])
    report_hash = hash_string(fields["report_sha256"])
    require(
        bool(report) and digest(canonical_bytes(report)) == report_hash, "execution report mismatch"
    )
    environment = {k: json_text(v) for k, v in json_map(fields["environment"]).items()}
    return {
        "argv": argv,
        "cwd": json_text(fields["cwd"]),
        "environment": environment,
        "exit_code": status,
        "stdout": stdout,
        "stderr": stderr,
        "report": report,
        "report_sha256": report_hash,
    }


def retained_execution(
    row: T, run_directory: Path, expected_exit: int | None = 0
) -> ExecutionRecord:
    """Authenticate the actual subordinate report and durable stdout/stderr rows."""
    fields = _execution(row, expected_exit)
    argv = fields["argv"]
    require(
        len(argv) > 6 and Path(argv[4]).name == "_publication_execution.py",
        "untrusted execution bootstrap",
    )
    indexes = {
        "internal-original": 9,
        "internal-artists": 9,
        "internal-transport": 7,
        "internal-predicate": 10,
    }
    require(argv[5] in indexes and len(argv) > indexes[argv[5]], "invalid retained execution argv")
    report_path = Path(argv[indexes[argv[5]]])
    require(
        report_path.is_absolute() and report_path.parent == run_directory,
        "execution report outside fresh run",
    )
    require(read_json(report_path) == fields["report"], "retained execution report bytes differ")
    stem = report_path.with_suffix("")
    require(
        regular_bytes(Path(str(stem) + ".stdout")).decode() == fields["stdout"],
        "retained stdout differs",
    )
    require(
        regular_bytes(Path(str(stem) + ".stderr")).decode() == fields["stderr"],
        "retained stderr differs",
    )
    require(read_json(Path(str(stem) + ".execution.json")) == fields, "retained invocation differs")
    return fields


def original_execution_identity(
    proof: ExecutionRecord, snapshot: Path, contract: DerivativeContract, mode: str
) -> None:
    report = proof["report"]
    require(
        report.get("source") == str(snapshot)
        and report.get("policy") == mode
        and report.get("endpoint") == contract.endpoint,
        "original execution root/policy mismatch",
    )
    target = (
        "scripts/record_external_benchmark_claims.py"
        if mode == "live"
        else "scripts/verify_network_claim_bindings.py"
    )
    require(
        report.get("verifier_sha256") == contract.original_inventory[target].sha256,
        "original execution verifier mismatch",
    )
    require(
        report.get("actual_argv")
        == [str(snapshot / target), *(["--verify-cache", "25"] if mode == "live" else [])],
        "original verifier argv narrowed or altered",
    )
    require(
        report.get("flags") == {"isolated": 1, "no_site": 1, "dont_write_bytecode": 1},
        "original execution flags mismatch",
    )
    module_origins = json_map(report.get("module_origins"))
    require(isinstance(module_origins, dict) and bool(module_origins), "empty module origins")
    for value in module_origins.values():
        origin = Path(json_text(value))
        require(
            snapshot in origin.parents
            and origin.relative_to(snapshot).as_posix() in contract.original_inventory,
            "project module origin not pinned",
        )
        require(
            file_hash(origin)
            == contract.original_inventory[origin.relative_to(snapshot).as_posix()].sha256,
            "project module origin bytes changed",
        )


def live_serving_identity(
    proof: ExecutionRecord, snapshot: Path, contract: DerivativeContract
) -> None:
    serving = exact_keys(
        proof["report"].get("serving"),
        {"model_identities", "strata", "checked", "mismatches", "unreachable", "return_code"},
        "actual live serving evidence",
    )
    identities = {k: json_text(v) for k, v in json_map(serving["model_identities"]).items()}
    external = next(
        x for x in contract.receipts if x.recorder.endswith("record_external_benchmark_claims.py")
    )
    configuration = json_map(read_json(snapshot / external.path)["configuration"])
    require(
        set(identities) == set(json_strings(configuration["model_tags"]))
        and all(identities.values()),
        "live serving identities incomplete",
    )
    strata = json_strings(serving["strata"])
    require(
        len(strata) == len(set(strata)) == 6 and json_integer(serving["checked"]) >= 6,
        "live complete-strata serving evidence absent",
    )
    require(
        all(json_integer(serving[k]) == 0 for k in ("mismatches", "unreachable", "return_code")),
        "actual live serving audit failed",
    )


def archival_identity_evidence(root: Path, contract: DerivativeContract) -> dict[str, str]:
    evidence = {}
    for name, artifact in contract.archival_artifacts.items():
        require(
            name == "vendor/staging_ud_czech/cs_pdtc-ud-test.conllu",
            "unknown archival artifact path",
        )
        require(
            file_identity(root / name) == artifact.identity and artifact.identity.mode == "100644",
            "archival file identity changed",
        )
        proof_path = root / artifact.proof_path
        require(file_hash(proof_path) == artifact.proof_sha256, "archival proof digest changed")
        proof = exact_keys(
            read_json(proof_path),
            {
                "schema",
                "role",
                "path",
                "bytes",
                "sha256",
                "git_blob_sha1",
                "source_revision",
                "source_url",
                "tree_response_sha256",
                "tree_response_path",
                "upstream_entry",
                "code_boundary",
            },
            "archival proof",
        )
        revision = json_text(
            original_literal(
                root, contract, "src/ailm/corpus/ud_czech.py", "UD_CZECH_PDTC_REVISION"
            )
        )
        require(
            proof["schema"] == "ailm-retained-archival-corpus-proof-v1"
            and proof["role"]
            == "retained nonconsumed upstream test artifact; no scoring or measurement input"
            and proof["path"] == name,
            "archival role/path/schema mismatch",
        )
        require(
            proof["sha256"] == artifact.identity.sha256
            and json_integer(proof["bytes"]) == artifact.identity.bytes
            and proof["source_revision"] == revision,
            "archival original revision/identity mismatch",
        )
        response_path = Path(json_text(proof["tree_response_path"]))
        require(
            response_path.is_absolute()
            and file_hash(response_path) == hash_string(proof["tree_response_sha256"]),
            "archival retained API response digest mismatch",
        )
        response = read_json(response_path)
        require(
            response.get("truncated") is False
            and response.get("sha") == revision
            and proof["source_url"]
            == "https://api.github.com/repos/UniversalDependencies/UD_Czech-PDTC/git/trees/"
            + revision,
            "archival retained API revision/truncation mismatch",
        )
        matches = [
            json_map(row)
            for row in t.json_list(response["tree"])
            if json_map(row).get("path") == Path(name).name
        ]
        expected = {
            "path": Path(name).name,
            "mode": "100644",
            "type": "blob",
            "sha": hash_string(proof["git_blob_sha1"], 40),
            "size": artifact.identity.bytes,
            "url": "https://api.github.com/repos/UniversalDependencies/UD_Czech-PDTC/git/blobs/"
            + json_text(proof["git_blob_sha1"]),
        }
        require(
            matches == [expected] and proof["upstream_entry"] == expected,
            "archival retained API entry missing/duplicate/changed",
        )
        content = regular_bytes(root / name)
        require(
            hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            == proof["git_blob_sha1"],
            "archival Git blob bytes mismatch",
        )
        evidence[artifact.proof_path] = artifact.proof_sha256
        evidence[str(response_path)] = json_text(proof["tree_response_sha256"])
    return evidence


def raw_identities(root: Path, contract: DerivativeContract) -> dict[str, t.FileIdentity]:
    """Expose original-source raw identity capture for independent controls."""
    from .publication_derivative import original_raw_identities

    return original_raw_identities(root, contract)
