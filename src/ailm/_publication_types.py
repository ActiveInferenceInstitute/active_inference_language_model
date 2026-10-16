"""Bounded publication derivative types responsibility."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TypeAlias, TypedDict, TypeVar

T = TypeVar("T")
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONMap: TypeAlias = dict[str, JSONValue]


class ExecutionRecord(TypedDict):
    argv: list[str]
    cwd: str
    environment: dict[str, str]
    exit_code: int
    stdout: str
    stderr: str
    report: JSONMap
    report_sha256: str


class SnapshotFile(TypedDict):
    sha256: str
    bytes: int
    permissions: str


SnapshotInventory: TypeAlias = dict[str, SnapshotFile]


class FileIdentityData(TypedDict):
    sha256: str
    bytes: int
    mode: str


class Materialization(TypedDict):
    snapshot: str
    pre_overlay: dict[str, FileIdentityData]
    before: SnapshotInventory
    after: SnapshotInventory


class PublicationDelta(TypedDict):
    original: str
    current: str


DECLARATION_SCHEMA = "ailm-publication-derivative-declaration-v1"

CONTRACT_SCHEMA = "ailm-publication-derivative-contract-v1"

STATE_SCHEMA = "ailm-publication-derivative-state-v1"

ATTESTATION_SCHEMA = "ailm-publication-derivative-attestation-v1"

SUPPORT_PATHS = (
    ".gitignore",
    ".zenodo.json",
    "CITATION.cff",
    "pyproject.toml",
    "tests/test_availability_disclosure.py",
    "src/ailm/manuscript_contract.py",
    "src/ailm/_publication_types.py",
    "src/ailm/_publication_inventory.py",
    "src/ailm/_publication_contract.py",
    "src/ailm/_publication_proof.py",
    "src/ailm/_publication_attestation.py",
    "scripts/_publication_bootstrap.py",
    "scripts/_publication_artists.py",
    "scripts/_publication_controls.py",
    "scripts/_publication_snapshot.py",
    "src/ailm/publication_derivative.py",
    "scripts/verify_publication_derivative.py",
    "scripts/_publication_execution.py",
    "tests/test_publication_derivative.py",
    "scripts/verify_release.py",
    "tests/test_release_orchestration.py",
    "data/publication_derivative.json",
    "data/publication_sources.json",
    "data/README.md",
    "docs/publication_derivative.md",
    "docs/architecture.md",
    "docs/README.md",
    "scripts/README.md",
    # R13: exact reviewed interface and metadata support; original model stays pinned.
    "src/ailm/dashboard/server.py",
    "src/ailm/dashboard/README.md",
    "src/ailm/dashboard/static/index.html",
    "src/ailm/dashboard/static/dashboard.css",
    "src/ailm/dashboard/static/js/main.js",
    "src/ailm/dashboard/static/js/chat.js",
    "tests/test_dashboard.py",
    "tests/test_documentation.py",
    "src/ailm/manuscript_variables/literature.py",
    "src/ailm/manuscript_variables/_core.py",
    "scripts/z_generate_manuscript_variables.py",
    "src/ailm/figures/_common.py",
    "tests/test_figure_export_dpi.py",
    "tests/test_figures_connected.py",
    "src/ailm/figures/connected.py",
    "tests/test_connected_publication_metadata.py",
    "tests/test_figures_multiplicity.py",
    "tests/test_manuscript_variables.py",
    "tests/test_manuscript_tokens.py",
    "tests/test_reference_source_fidelity.py",
    "docs/manuscript/config.yaml",
    "docs/manuscript/SYNTAX.md",
    "docs/manuscript/00_abstract.md",
    "docs/manuscript/references.bib",
    "scripts/build_web_ailm.py",
    # R15: exact reference presentation and portable supporting-doc support.
    "docs/AGENTS.md",
    "scripts/z_render_docs.py",
    "src/ailm/docs_hydration.py",
    "src/ailm/manuscript_variables/_exceptions.py",
    "src/ailm/manuscript_variables/_io.py",
    "src/ailm/reference_presentation.py",
    "tests/test_reference_docs.py",
    "src/ailm/supporting_package.py",
    "tests/test_supporting_package.py",
    "tests/test_american_spelling.py",
    # R17: independently reviewed current-guidance repairs, each hash-bound.
    "docs/design/connected_runtime.md",
    "CHANGELOG.md",
    "docs/formal.md",
    "docs/manuscript/MANUSCRIPT_STATUS.md",
    "docs/manuscript/AGENTS.md",
    "docs/guides/faq.md",
    "docs/guides/forking_guide.md",
    "docs/troubleshooting.md",
    "docs/gate_design.md",
    "docs/reference/glossary.md",
    "docs/method/release_checklist.md",
    "docs/guides/quickstart.md",
    "docs/reference/output_conventions.md",
    "docs/reference/output_inventory.md",
    "docs/syntax_guide.md",
    "docs/reference/notation.md",
    "src/ailm/levels/README.md",
    "src/ailm/README.md",
    "src/AGENTS.md",
    "docs/decisions.md",
    # R19: exact reviewed manual rendering guidance.
    "docs/method/rendering_pipeline.md",
    # R21: exact reviewed decision-index oracle repair; numerical sources stay pinned.
    "tests/test_decisions_index.py",
    # R23: finite manuscript presentation and authenticated dashboard capture inputs.
    "data/figure_alt_text.yaml",
    "data/dashboard_capture/analysis.png",
    "data/dashboard_capture/receipt.json",
    "docs/manuscript/98_supplement_channel_factorial.md",
    "docs/manuscript/98_supplement_complexity.md",
    "docs/manuscript/98_supplement_lean.md",
    "docs/manuscript/98_supplement_statistics.md",
    "docs/manuscript/98_supplement_audit_trails.md",
    "scripts/generate_figures.py",
    "src/ailm/manuscript_variables/architecture.py",
    "src/ailm/figures/dashboard_observation.py",
    "tests/test_figure_provenance.py",
    "tests/test_manuscript_crossrefs.py",
    "tests/test_figures_dashboard_observation.py",
    "output/figures/dashboard_observation.png",
    "tests/conftest.py",
    "tests/test_checkout_completeness.py",
    "tests/test_claims_status.py",
    "tests/README.md",
    "docs/testing_philosophy.md",
    "output/reports/connected_gum.json",
)

PUBLICATION_PATHS = (
    "scripts/warm_call_cache.py",  # Only the exact printed rendering route may change.
    "docs/manuscript/01_introduction.md",
    "docs/manuscript/02_background.md",
    "docs/manuscript/05_discussion.md",
    "docs/manuscript/06_related_work.md",
    "docs/manuscript/08_conclusion.md",
    "src/ailm/figures/case_word_sweep.py",
    "src/ailm/figures/incontext_sweep.py",
    "src/ailm/figures/results.py",
    "src/ailm/figures/benchmarks.py",
    "src/ailm/figures/multiplicity.py",
    "src/ailm/manuscript_variables/context.py",
    "docs/manuscript/03_methods.md",
    "docs/manuscript/04_results.md",
    "docs/manuscript/07_reproducibility.md",
    "docs/manuscript/preamble.md",
    "docs/manuscript/98_supplement_external_benchmarks.md",
    "docs/manuscript/98_supplement_statistics.md",
    "docs/manuscript/98_supplement_audit_trails.md",
    "README.md",
    "docs/design/cerebrum_mapping.md",
    "docs/extension_protocols.md",
    "docs/provenance/sources.md",
    "docs/reference/formalism_map.md",
)

SCAN_EXCLUDED = frozenset(
    {
        ".git",
        ".venv",
        "refs",
        "scratchpad",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

IMPLEMENTATION_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".sh",
        ".bash",
        ".lean",
        ".so",
        ".dylib",
        ".dll",
        ".wasm",
    }
)


class DerivativeError(ValueError):
    """An exact identity or required execution proof could not be established."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    sha256: str
    bytes: int
    mode: str


@dataclass(frozen=True, slots=True)
class ReceiptIdentity:
    path: str
    recorder: str
    labels: tuple[str, ...]
    run_ids: tuple[str, ...]
    claims_count: int
    calls_count: int


@dataclass(frozen=True, slots=True)
class ArchivalArtifact:
    identity: FileIdentity
    proof_path: str
    proof_sha256: str


@dataclass(frozen=True, slots=True)
class DerivativeContract:
    schema: str
    contract_sha256: str
    reviewed: bool
    declaration_sha256: str
    measurement_commit: str
    measurement_tree: str
    publication_commit: str
    publication_tree: str
    snapshot_parent: str
    support: dict[str, str]
    review_path: str
    review_sha256: str
    original_inventory: dict[str, FileIdentity]
    publication_delta: dict[str, PublicationDelta]
    inputs: dict[str, str]
    receipts: tuple[ReceiptIdentity, ...]
    worker_evidence: dict[str, str]
    tokens_path: str
    tokens_sha256: str
    baseline_untracked: dict[str, FileIdentity]
    endpoint: str
    archival_artifacts: dict[str, ArchivalArtifact] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DerivativeState:
    schema: str
    contract_sha256: str
    measurement_commit: str
    measurement_tree: str
    current_commit: str
    current_tree: str
    inventory: dict[str, FileIdentity]
    outputs: dict[str, FileIdentity]
    inputs: dict[str, str]
    support: dict[str, str]
    evidence: dict[str, str]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class AttestationResult:
    accepted: bool
    mode: str
    attestation_sha256: str
    contract_sha256: str
    state_fingerprint: str
    errors: tuple[str, ...]


def canonical_bytes(value: T) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DerivativeError(message)


def exact_keys(value: T, keys: set[str], description: str) -> JSONMap:
    try:
        fields = json_map(value)
    except DerivativeError as error:
        raise DerivativeError(f"{description}: {error}") from error
    require(set(fields) == keys, f"{description}: missing/unknown fields")
    return fields


def hash_string(value: JSONValue, length: int = 64) -> str:
    text = json_text(value)
    require(re.fullmatch(rf"[0-9a-f]{{{length}}}", text) is not None, "invalid digest")
    return text


def relative_path(value: JSONValue) -> str:
    text = json_text(value)
    require(bool(text), "empty path")
    path = PurePosixPath(text)
    require(
        not path.is_absolute()
        and path.as_posix() == text
        and not any(x in {".", ".."} for x in path.parts),
        f"unsafe relative path: {text}",
    )
    return text


def json_value(value: T) -> JSONValue:
    match value:
        case None | bool() | int() | str():
            return value
        case float():
            require(math.isfinite(value), "nonfinite JSON number")
            return value
        case list():
            return [json_value(item) for item in value]
        case dict():
            require(all(isinstance(k, str) for k in value), "JSON object key must be text")
            return {str(k): json_value(v) for k, v in value.items()}
        case _:
            raise DerivativeError("value is not JSON")


def json_map(value: T) -> JSONMap:
    normalized = json_value(value)
    if not isinstance(normalized, dict):
        raise DerivativeError("JSON object required")
    return normalized


def json_text(value: JSONValue) -> str:
    if not isinstance(value, str):
        raise DerivativeError("JSON text required")
    return value


def json_integer(value: JSONValue) -> int:
    if type(value) is not int:
        raise DerivativeError("JSON integer required")
    return value


def json_list(value: JSONValue) -> list[JSONValue]:
    if not isinstance(value, list):
        raise DerivativeError("JSON list required")
    return value


def json_strings(value: JSONValue) -> list[str]:
    return [json_text(x) for x in json_list(value)]


def parse_inventory(value: JSONValue) -> dict[str, FileIdentity]:
    value = json_map(value)
    require(bool(value), "empty inventory")
    result = {}
    for path, row in value.items():
        relative_path(path)
        fields = exact_keys(row, {"sha256", "bytes", "mode"}, "file identity")
        require(
            type(fields["bytes"]) is int
            and fields["bytes"] >= 0
            and fields["mode"] in {"100644", "100755"},
            "invalid file identity",
        )
        result[path] = FileIdentity(
            hash_string(fields["sha256"]),
            json_integer(fields["bytes"]),
            json_text(fields["mode"]),
        )
    return result


def parse_hashes(value: JSONValue, *, relative: bool = True) -> dict[str, str]:
    value = json_map(value)
    require(bool(value), "empty hash set")
    return {(relative_path(k) if relative else str(k)): hash_string(v) for k, v in value.items()}


def parse_archival_artifacts(value: JSONValue) -> dict[str, ArchivalArtifact]:
    result = {}
    for name, item in json_map(value).items():
        require(
            name == "vendor/staging_ud_czech/cs_pdtc-ud-test.conllu",
            "unknown archival artifact path",
        )
        row = exact_keys(
            item, {"sha256", "bytes", "mode", "proof_path", "proof_sha256"}, "archival declaration"
        )
        identity = FileIdentity(
            hash_string(row["sha256"]), json_integer(row["bytes"]), json_text(row["mode"])
        )
        require(identity.bytes > 0 and identity.mode == "100644", "invalid archival size/mode")
        proof_path = relative_path(row["proof_path"])
        require(
            proof_path == ".omo/evidence/publication-derivative/czech-archival-proof.json",
            "unknown archival proof path",
        )
        result[name] = ArchivalArtifact(identity, proof_path, hash_string(row["proof_sha256"]))
    return result
