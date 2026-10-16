"""Strict declaration loading and externally reviewed source freeze binding."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from . import _publication_types as t
from ._publication_inventory import (
    file_hash,
    git_bytes,
    git_inventory,
    parse_hashes,
    parse_inventory,
    read_json,
    regular_bytes,
    validate_package_publication_urls,
)
from ._publication_types import parse_archival_artifacts as parse_archival_artifacts


def _declaration(
    path: Path, root: Path, *, reviewed: bool, external: t.JSONMap | None, parent: Path
) -> t.DerivativeContract:
    fields = t.exact_keys(
        read_json(path),
        {
            "schema",
            "measurement_commit",
            "measurement_tree",
            "original_inventory",
            "publication_delta",
            "inputs",
            "receipts",
            "worker_evidence",
            "tokens_path",
            "tokens_sha256",
            "baseline_untracked",
            "endpoint",
            "archival_artifacts",
        },
        "declaration",
    )
    t.require(fields["schema"] == t.DECLARATION_SCHEMA, "wrong declaration schema")
    measurement_commit = t.hash_string(fields["measurement_commit"], 40)
    measurement_tree = t.hash_string(fields["measurement_tree"], 40)
    validate_package_publication_urls(root, measurement_commit)
    inventory = parse_inventory(fields["original_inventory"])
    t.require(
        inventory == git_inventory(root, measurement_commit),
        "declaration pinned inventory differs from Git",
    )
    t.require(
        git_bytes(root, "rev-parse", measurement_commit + "^{tree}").decode().strip()
        == measurement_tree,
        "wrong measurement tree",
    )
    raw_delta = t.json_map(fields["publication_delta"])
    t.require(
        set(raw_delta) == set(t.PUBLICATION_PATHS),
        "publication delta is not the approved finite set",
    )
    delta: dict[str, t.PublicationDelta] = {}
    for name, value in raw_delta.items():
        row = t.exact_keys(value, {"original", "current"}, "publication delta")
        original = t.hash_string(row["original"])
        t.require(original == inventory[name].sha256, f"original delta mismatch: {name}")
        delta[name] = {"original": original, "current": t.hash_string(row["current"])}
    receipts = []
    for value in t.json_list(fields["receipts"]):
        row = t.exact_keys(
            value,
            {"path", "recorder", "labels", "run_ids", "claims_count", "calls_count"},
            "receipt identity",
        )
        labels, run_ids = t.json_strings(row["labels"]), t.json_strings(row["run_ids"])
        for values in (labels, run_ids):
            t.require(
                bool(values) and all(values) and values == sorted(set(values)),
                "invalid labels/run_ids set",
            )
        claims_count, calls_count = (
            t.json_integer(row["claims_count"]),
            t.json_integer(row["calls_count"]),
        )
        t.require(
            claims_count == len(labels) and calls_count > 0, "empty/inconsistent receipt counts"
        )
        recorder, receipt_path = t.relative_path(row["recorder"]), t.relative_path(row["path"])
        t.require(
            receipt_path == "cache/network_claim_bindings/" + Path(recorder).stem + ".json",
            "recorder receipt path mismatch",
        )
        receipts.append(
            t.ReceiptIdentity(
                receipt_path, recorder, tuple(labels), tuple(run_ids), claims_count, calls_count
            )
        )
    t.require(
        len(receipts) == 5
        and len({x.recorder for x in receipts}) == 5
        and len({x.path for x in receipts}) == 5,
        "complete five-recorder set required",
    )
    labels_flat = [label for receipt in receipts for label in receipt.labels]
    t.require(len(labels_flat) == len(set(labels_flat)), "overlapping recorder claim sets")
    inputs = parse_hashes(fields["inputs"])
    t.require(
        set(inputs) == {"data/claims.json", "cache/bench_calls.jsonl", *(x.path for x in receipts)},
        "invalid input set",
    )
    t.require(
        inputs["data/claims.json"] == inventory["data/claims.json"].sha256,
        "claims must remain pinned",
    )
    tokens = t.relative_path(fields["tokens_path"])
    endpoint = t.json_text(fields["endpoint"])
    url = urlsplit(endpoint)
    t.require(
        url.scheme == "http"
        and url.hostname == "127.0.0.1"
        and url.port is not None
        and endpoint == f"http://127.0.0.1:{url.port}",
        "endpoint must be canonical IPv4 loopback",
    )
    baseline = t.json_map(fields["baseline_untracked"])
    baseline_inventory = parse_inventory(baseline) if baseline else {}
    for name in baseline_inventory:
        t.require(
            name.startswith((".omo/evidence/", ".omo/teams/", "output/reports/"))
            or name
            in {
                "output/architecture-map/render.js",
                "output/architecture-map/snapshot.js",
                "output/architecture-map/version.js",
            },
            f"untracked allowance is not historical non-model evidence: {name}",
        )
    support = (
        parse_hashes(external["support"])
        if external
        else {x: file_hash(root / x) for x in t.SUPPORT_PATHS if (root / x).is_file()}
    )
    if external:
        t.require(
            set(support) == set(t.SUPPORT_PATHS), "review must bind every finite support path"
        )
        t.require(
            support["data/publication_derivative.json"] == file_hash(path),
            "declaration/support mismatch",
        )
    commit = (
        t.hash_string(external["publication_commit"], 40)
        if external
        else git_bytes(root, "rev-parse", "HEAD").decode().strip()
    )
    tree = (
        t.hash_string(external["publication_tree"], 40)
        if external
        else git_bytes(root, "rev-parse", "HEAD^{tree}").decode().strip()
    )
    t.require(
        parent.is_absolute() and parent.is_dir() and not parent.is_symlink(),
        "snapshot parent must be existing regular directory",
    )
    worker = parse_hashes(fields["worker_evidence"])
    tokens_sha256 = t.hash_string(fields["tokens_sha256"])
    t.require(worker.get(tokens) == tokens_sha256, "token evidence mismatch")
    archives = t.parse_archival_artifacts(fields["archival_artifacts"])
    return t.DerivativeContract(
        t.CONTRACT_SCHEMA,
        t.digest(t.canonical_bytes(external)) if external else t.digest(regular_bytes(path)),
        reviewed,
        file_hash(path),
        measurement_commit,
        measurement_tree,
        commit,
        tree,
        str(parent),
        support,
        t.json_text(external["review_path"]) if external else "",
        t.hash_string(external["review_sha256"]) if external else "",
        inventory,
        delta,
        inputs,
        tuple(receipts),
        worker,
        tokens,
        tokens_sha256,
        baseline_inventory,
        endpoint,
        archives,
    )


def load_candidate_contract(
    declaration: Path, project_root: Path, snapshot_parent: Path
) -> t.DerivativeContract:
    """Development-only execution. It cannot produce a release acceptance."""
    return _declaration(
        declaration, project_root, reviewed=False, external=None, parent=snapshot_parent
    )


def load_contract(path: Path, project_root: Path) -> t.DerivativeContract:
    fields = t.exact_keys(
        read_json(path),
        {
            "schema",
            "declaration_path",
            "declaration_sha256",
            "publication_commit",
            "publication_tree",
            "support",
            "review_path",
            "review_sha256",
            "snapshot_parent",
        },
        "external reviewed contract",
    )
    t.require(fields["schema"] == t.CONTRACT_SCHEMA, "wrong contract schema")
    t.require(
        path.absolute() != project_root.absolute()
        and project_root.absolute() not in path.absolute().parents,
        "reviewed contract must be external to project",
    )
    declaration = project_root / t.relative_path(fields["declaration_path"])
    t.require(
        file_hash(declaration) == t.hash_string(fields["declaration_sha256"]),
        "declaration digest mismatch",
    )
    review = Path(t.json_text(fields["review_path"]))
    t.require(
        review.is_absolute()
        and file_hash(review) == t.hash_string(fields["review_sha256"])
        and bool(regular_bytes(review).strip()),
        "missing/changed independent review",
    )
    commit = t.hash_string(fields["publication_commit"], 40)
    t.require(
        git_bytes(project_root, "rev-parse", commit + "^{tree}").decode().strip()
        == fields["publication_tree"],
        "source freeze tree mismatch",
    )
    git_bytes(project_root, "merge-base", "--is-ancestor", commit, "HEAD")
    contract = _declaration(
        declaration,
        project_root,
        reviewed=True,
        external=fields,
        parent=Path(t.json_text(fields["snapshot_parent"])),
    )
    frozen = git_inventory(project_root, commit)
    for name, expected in contract.support.items():
        t.require(
            name in frozen and frozen[name].sha256 == expected,
            f"support not in reviewed source freeze: {name}",
        )
    return contract
