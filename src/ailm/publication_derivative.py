"""Stable public facade for exact two-source publication identities and proofs."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from pathlib import Path
from typing import Final

from . import _publication_types as t
from ._publication_attestation import validate_attestation as validate_attestation
from ._publication_contract import _declaration as _declaration
from ._publication_contract import load_candidate_contract as load_candidate_contract
from ._publication_contract import load_contract as load_contract
from ._publication_inventory import execution_snapshot_inventory as execution_snapshot_inventory
from ._publication_inventory import file_hash as file_hash
from ._publication_inventory import file_identity as file_identity
from ._publication_inventory import git_bytes as git_bytes
from ._publication_inventory import git_command as git_command
from ._publication_inventory import git_inventory as git_inventory
from ._publication_inventory import original_literal as original_literal
from ._publication_inventory import parse_hashes as parse_hashes
from ._publication_inventory import parse_inventory as parse_inventory
from ._publication_inventory import publication_source_bytes as publication_source_bytes
from ._publication_inventory import read_json as read_json
from ._publication_inventory import regular_bytes as regular_bytes
from ._publication_inventory import scan_untracked_implementation as scan_untracked_implementation
from ._publication_inventory import trusted_git as trusted_git
from ._publication_inventory import (
    validate_package_publication_urls as validate_package_publication_urls,
)
from ._publication_proof import _execution as _execution
from ._publication_proof import archival_identity_evidence as archival_identity_evidence
from ._publication_proof import original_execution_identity as original_execution_identity
from ._publication_proof import retained_execution as retained_execution
from ._publication_types import ATTESTATION_SCHEMA as ATTESTATION_SCHEMA
from ._publication_types import CONTRACT_SCHEMA as CONTRACT_SCHEMA
from ._publication_types import DECLARATION_SCHEMA as DECLARATION_SCHEMA
from ._publication_types import IMPLEMENTATION_SUFFIXES as IMPLEMENTATION_SUFFIXES
from ._publication_types import PUBLICATION_PATHS as PUBLICATION_PATHS
from ._publication_types import SCAN_EXCLUDED as SCAN_EXCLUDED
from ._publication_types import STATE_SCHEMA as STATE_SCHEMA
from ._publication_types import SUPPORT_PATHS as SUPPORT_PATHS
from ._publication_types import AttestationResult as AttestationResult
from ._publication_types import DerivativeContract as DerivativeContract
from ._publication_types import DerivativeError as DerivativeError
from ._publication_types import DerivativeState as DerivativeState
from ._publication_types import FileIdentity as FileIdentity
from ._publication_types import ReceiptIdentity as ReceiptIdentity
from ._publication_types import canonical_bytes as canonical_bytes
from ._publication_types import digest as digest
from ._publication_types import exact_keys as exact_keys
from ._publication_types import hash_string as hash_string
from ._publication_types import json_integer, json_map, json_strings, json_text
from ._publication_types import relative_path as relative_path
from ._publication_types import require as require

_REVIEWED_TEST_FIXTURE_DELTAS: Final = {
    "tests/test_claims_single_writer.py": (
        FileIdentity(
            "0cb3a5a649fd99a7d71f06049944b6dc6d2ad5f8c015c7ec3dc19c82ff02fa6d", 43935, "100644"
        ),
        FileIdentity(
            "662b9230b7d07569f1a5f49d3013b2702daf3c191accb0ebb2e3d4eeaf97abf1", 55589, "100644"
        ),
    ),
    "tests/test_docs_hydration.py": (
        FileIdentity(
            "520bc2997078ce7e580f13eaea35c77dc9ad162c38fc8a1ce3e7f9c300fc2484", 57020, "100644"
        ),
        FileIdentity(
            "ca2c5755046799c1821a8077b29898ca1de7c9ed211ef3875d4eede1edc9fbef", 58610, "100644"
        ),
    ),
}


def _validate_reviewed_test_fixture_delta(
    name: str, original: FileIdentity, current: FileIdentity
) -> None:
    """Accept only the reviewed fixture pairs, including their original identities."""
    require(
        _REVIEWED_TEST_FIXTURE_DELTAS.get(name) == (original, current),
        f"reviewed test fixture identity differs: {name}",
    )


def transformed_figure_source(name: str, original: bytes) -> bytes:
    """Exhaustive approved source substitutions, never a path-only exemption."""
    if name == "scripts/warm_call_cache.py":
        acceptance = b'        "  .venv/bin/python -I -S scripts/verify_release.py full\\n"\n'
        require(original.count(acceptance) == 1, "printed acceptance boundary ambiguous")
        original = original.replace(acceptance, b"")
        header = b'        "\\nWith the store warm, the rest of the rebuild is:\\n"\n'
        setup = b'        "  # Set AILM_TEMPLATE_ROOT to the selected template checkout first.\\n"\n        \'  export TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT"\\n\'\n'
        require(original.count(header) == 1, "printed provider boundary ambiguous")
        original = original.replace(header, header + setup)
        renderer = b'        "  .venv/bin/python scripts/z_render_docs.py\\n"\n'
        companions = b'        "  .venv/bin/python scripts/z_render_docs.py --supporting-protocols\\n"\n        "  .venv/bin/python scripts/z_render_docs.py --check\\n"\n'
        require(original.count(renderer) == 1, "printed companion boundary ambiguous")
        original = original.replace(renderer, companions)
        old = b'        "  uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_inference_language_model\\n"\n'
        new = b'        "  # From the AILM checkout, resolve and check the individual HTML exports:\\n"\n        "  .venv/bin/python scripts/z_render_docs.py --publication-references\\n"\n        "  .venv/bin/python scripts/z_render_docs.py --check --supporting-protocols\\n"\n        "  # Back in the template checkout:\\n"\n        "  uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_inference_language_model\\n"\n        "  # Finally, from the AILM checkout, run acceptance over the completed tree:\\n"\n        "  .venv/bin/python -I -S scripts/verify_release.py full\\n"\n'
    elif name.endswith("case_word_sweep.py"):
        old = b'ax.legend(fontsize=FONTS["legend"], loc="upper left")'
        new = b'ax.legend(fontsize=FONTS["legend"], loc="lower left")'
    elif name == "src/ailm/figures/benchmarks.py":
        old = b'delta_ax.legend(fontsize=FONTS["legend"], loc="lower right", framealpha=0.95)'
        new = b'delta_ax.legend(fontsize=FONTS["legend"], loc="upper right", framealpha=0.95)'
        bucket_old = b'bucket_ax.legend(fontsize=FONTS["legend"], loc="upper left")'
        bucket_new = b'bucket_ax.legend(fontsize=FONTS["legend"], loc="upper right")'
        require(original.count(bucket_old) == 1, "word-length legend transformation ambiguous")
        original = original.replace(bucket_old, bucket_new)
    elif name == "src/ailm/figures/multiplicity.py":
        old = b"f\"not cluster-adjusted: exact conditional\\ntest over {row['unit']}, \""
        new = b"f\"not cluster-adjusted: exact conditional test\\nover {row['unit']}, \""
    elif name == "docs/manuscript/98_supplement_statistics.md":
        path_old = b"src/ailm/manuscript_variables.py"
        path_new = b"src/ailm/manuscript_variables/__init__.py"
        require(original.count(path_old) == 1, "statistics module locator ambiguous")
        original = original.replace(path_old, path_new)
        old = b"\n### The forest plot's geometry, and where the exact values are recorded\n"
        new = (
            b"\n```{=latex}\n\\csname @ifclassloaded\\endcsname{beamer}{}{\\Needspace{36\\baselineskip}}\n```\n"
            + old
        )
    elif name == "docs/manuscript/98_supplement_audit_trails.md":
        old = b"\n### Level 3: the declared gate states a condition the bar itself already clears {#sec:word_gate}\n"
        new = (
            b"\n```{=latex}\n\\csname @ifclassloaded\\endcsname{beamer}{}{\\Needspace{32\\baselineskip}}\n```\n"
            + old
        )
    elif name == "src/ailm/manuscript_variables/context.py":
        old = b'            f"**{hosted_cohort_label(_probe.key)}**",'
        new = (
            b'            "```{=latex}\\n\\\\csname @ifclassloaded\\\\endcsname{beamer}{}{\\\\Needspace{8\\\\baselineskip} }\\n```\\n",\n'
            + old
            + b'\n            "",\n            "```{=latex}\\n\\\\csname @ifclassloaded\\\\endcsname{beamer}{}{\\\\vspace{0.5\\\\baselineskip} }\\n```\\n",'
        )
        old += b'\n            "",'
    elif name.endswith("results.py"):
        annotation_old = b"            value + span * 0.22,"
        annotation_new = b"            value + (interval_highs[x] if x < 2 else 0.0) + span * 0.22,"
        require(original.count(annotation_old) == 1, "results annotation transform ambiguous")
        original = original.replace(annotation_old, annotation_new)
        old = b'"forward\xe2\x80\x93backward\\nsmoothing ceiling",'
        new = b'"forward\xe2\x80\x93backward\\nsmoothing reference",'
    elif name.endswith("incontext_sweep.py"):
        annotation_old = (
            b"        inset.set_xticks([rungs[0], rungs[len(rungs) // 2], rungs[-1]])\n"
        )
        annotation_new = b'        inset.set_xticks([rungs[0], rungs[len(rungs) // 2], rungs[-1]])\n        inset.tick_params(axis="x", pad=-15)\n'
        require(
            original.count(annotation_old) == 1, "incontext_sweep annotation transform ambiguous"
        )
        original = original.replace(annotation_old, annotation_new)
        old = b'        axis.set_xlabel("Labeled one-transition examples supplied (k)")\n'
        new = (
            old
            + b'        axis.set_ylabel("Letter accuracy (%)")\n        axis.tick_params(axis="y", labelleft=True)\n'
        )
        require(
            original.count(b'    axes[0].set_ylabel("Letter accuracy (%)")\n') == 1,
            "shared label transformation ambiguous",
        )
        original = original.replace(b'    axes[0].set_ylabel("Letter accuracy (%)")\n', b"")
    else:
        raise DerivativeError(f"unapproved figure transform: {name}")
    require(original.count(old) == 1, f"figure transformation not unique: {name}")
    return original.replace(old, new)


def validate_receipt_identities(contract: DerivativeContract, root: Path) -> None:
    actual_paths = {
        x.relative_to(root).as_posix()
        for x in (root / "cache/network_claim_bindings").glob("*.json")
    }
    require(actual_paths == {x.path for x in contract.receipts}, "missing/extra receipt")
    local_endpoints = set()
    for expected in contract.receipts:
        row = read_json(root / expected.path)
        require(row.get("recorder") == expected.recorder, "wrong recorder identity")
        claims = json_map(row.get("claims", {}))
        entries = json_map(claims.get("entries", {}))
        require(
            tuple(sorted(entries)) == expected.labels
            and claims.get("count") == expected.claims_count
            and json_map(row.get("cache", {})).get("record_count") == expected.calls_count,
            "receipt claim/call set mismatch",
        )
        require(
            tuple(sorted({json_text(json_map(entry)["run_id"]) for entry in entries.values()}))
            == expected.run_ids,
            "receipt run IDs differ",
        )
        require(
            tuple(sorted(json_strings(claims.get("labels", [])))) == expected.labels,
            "receipt labels mismatch",
        )
        endpoint = json_map(row.get("configuration", {})).get("endpoint")
        if isinstance(endpoint, str) and endpoint.startswith("http:"):
            local_endpoints.add(endpoint)
    require(local_endpoints == {contract.endpoint}, "authenticated local origins disagree")


def capture_derivative_state(contract: DerivativeContract, project_root: Path) -> DerivativeState:
    root = project_root.absolute()
    original = git_inventory(root, contract.measurement_commit)
    require(original == contract.original_inventory, "pinned tree changed")
    current_commit = git_bytes(root, "rev-parse", "HEAD").decode().strip()
    current_tree = git_bytes(root, "rev-parse", "HEAD^{tree}").decode().strip()
    tracked = {x.decode() for x in git_bytes(root, "ls-files", "-z").split(b"\0") if x}
    allowed = set(original) | set(SUPPORT_PATHS)
    require(not (tracked - allowed), f"unexpected tracked paths: {sorted(tracked - allowed)}")
    require(
        not (set(original) - tracked), f"deleted tracked paths: {sorted(set(original) - tracked)}"
    )
    validate_package_publication_urls(root, contract.measurement_commit)
    inventory = {}
    outputs = {}
    for name in sorted(tracked | {p for p in SUPPORT_PATHS if (root / p).exists()}):
        identity = file_identity(root / name)
        if name in SUPPORT_PATHS:
            require(identity.sha256 == contract.support.get(name), f"support bytes changed: {name}")
        if name.startswith("output/"):
            outputs[name] = identity
            continue
        inventory[name] = identity
        if name in SUPPORT_PATHS:
            continue
        elif name in _REVIEWED_TEST_FIXTURE_DELTAS and identity != original[name]:
            _validate_reviewed_test_fixture_delta(name, original[name], identity)
        elif name in contract.publication_delta:
            require(
                identity.sha256 == contract.publication_delta[name]["current"]
                and identity.mode == original[name].mode,
                f"publication delta differs: {name}",
            )
            if name.startswith("src/") or name in {
                "docs/manuscript/98_supplement_statistics.md",
                "docs/manuscript/98_supplement_audit_trails.md",
                "scripts/warm_call_cache.py",
            }:
                source = git_bytes(root, "show", contract.measurement_commit + ":" + name)
                require(
                    regular_bytes(root / name) == transformed_figure_source(name, source),
                    f"nonstyling figure delta: {name}",
                )
        else:
            require(identity == original[name], f"numerical/source/input identity changed: {name}")
    raw_inputs = original_raw_identities(root, contract)
    archival_evidence = archival_identity_evidence(root, contract)
    untracked = scan_untracked_implementation(root, tracked)
    for name, identity in {
        **raw_inputs,
        **{p: a.identity for p, a in contract.archival_artifacts.items()},
    }.items():
        require(
            untracked.get(name) == identity, "authenticated extra input absent or altered: " + name
        )
        inventory[name] = untracked.pop(name)

    require(
        not any(
            p.startswith(("src/", "scripts/", "vendor/", "gnn/", "formal/", "data/", "tests/"))
            for p in untracked
        ),
        "untracked numerical/source/input implementation is never grandfathered",
    )
    require(
        untracked == contract.baseline_untracked,
        f"untracked implementation drift: {sorted(set(untracked) ^ set(contract.baseline_untracked)) or 'changed bytes'}",
    )
    inventory.update(untracked)
    inputs = {p: file_hash(root / p) for p in contract.inputs}
    require(inputs == contract.inputs, "original input bytes changed")
    validate_receipt_identities(contract, root)
    evidence = {p: file_hash(root / p) for p in contract.worker_evidence}
    require(evidence == contract.worker_evidence, "worker supporting evidence changed")
    evidence.update(archival_evidence)
    git_tool = trusted_git()
    evidence[str(git_tool)] = file_hash(git_tool)
    if contract.reviewed:
        require(
            file_hash(Path(contract.review_path)) == contract.review_sha256,
            "independent review changed",
        )
        evidence[contract.review_path] = contract.review_sha256
        frozen = git_inventory(root, contract.publication_commit)
        require(
            git_bytes(root, "rev-parse", contract.publication_commit + "^{tree}").decode().strip()
            == contract.publication_tree,
            "publication freeze changed",
        )
        require(
            {p: i for p, i in frozen.items() if not p.startswith("output/")}
            == {p: i for p, i in inventory.items() if p in tracked},
            "current nonoutput source differs from reviewed freeze",
        )
    state = DerivativeState(
        STATE_SCHEMA,
        contract.contract_sha256,
        contract.measurement_commit,
        contract.measurement_tree,
        current_commit,
        current_tree,
        inventory,
        outputs,
        inputs,
        contract.support,
        evidence,
        "",
    )
    values = asdict(state)
    values.pop("fingerprint")
    return replace(state, fingerprint=digest(canonical_bytes(values)))


def original_raw_identities(root: Path, contract: DerivativeContract) -> dict[str, t.FileIdentity]:
    modules = ("src/ailm/corpus/cmudict.py", "src/ailm/corpus/ud.py", "src/ailm/corpus/ud_czech.py")
    if not any(p in contract.original_inventory for p in modules):
        return {}
    require(
        all(p in contract.original_inventory for p in modules),
        "incomplete original raw source pins",
    )
    pins = {
        path: t.FileIdentity(
            hash_string(original_literal(root, contract, module, prefix + "_SHA256")),
            json_integer(original_literal(root, contract, module, prefix + "_SIZE_BYTES")),
            "100644",
        )
        for path, module, prefix in (
            ("vendor/cmudict.dict", modules[0], "CMUDICT_RAW"),
            ("vendor/en_ewt-ud-dev.conllu", modules[1], "UD_EWT_RAW"),
        )
    }
    source = git_bytes(root, "show", contract.measurement_commit + ":" + modules[2])
    require(
        digest(source) == contract.original_inventory[modules[2]].sha256,
        "original Czech source identity mismatch",
    )
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "UDCzechShard"
        ):
            data = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}
            require(
                data["filename"] in {"cs_pdtc-ud-dev.conllu", "cs_pdtc-ud-train-ft.conllu"},
                "unexpected original raw shard extraction",
            )
            name = "vendor/staging_ud_czech/" + data["filename"]
            require(name not in pins, "duplicate original raw shard")
            pins[name] = t.FileIdentity(
                hash_string(data["sha256"]), json_integer(data["size_bytes"]), "100644"
            )
    require(len(pins) == 4, "incomplete original raw shard extraction")
    for name, identity in pins.items():
        require(
            file_identity(root / name) == identity, "original raw input identity changed: " + name
        )
    return pins
