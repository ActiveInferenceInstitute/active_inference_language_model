"""Real-file refusal controls for the bounded publication derivative.

Small Git object trees test identities; canonical semantic replay is a separate
parent-run obligation. No test replaces a recorder or dependency implementation.
"""

from __future__ import annotations

import ast
import json
import os
import py_compile
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, assert_never

import pytest

from ailm import _publication_inventory as inventory
from ailm import _publication_proof as proof
from ailm import _publication_types as types
from ailm import publication_derivative as derivative

ROOT = Path(__file__).resolve().parents[1]

_REVIEWED_FIXTURE_PATHS = (
    "tests/test_claims_single_writer.py",
    "tests/test_docs_hydration.py",
)


@pytest.mark.parametrize("name", _REVIEWED_FIXTURE_PATHS)
def test_exact_reviewed_fixture_pair_is_accepted(name: str) -> None:
    declaration = json.loads((ROOT / "data/publication_derivative.json").read_bytes())
    original = derivative.FileIdentity(**declaration["original_inventory"][name])
    derivative._validate_reviewed_test_fixture_delta(
        name, original, derivative.file_identity(ROOT / name)
    )


@pytest.mark.parametrize("name", _REVIEWED_FIXTURE_PATHS)
@pytest.mark.parametrize(
    "mutation", ("original_hash", "original_size", "original_mode", "content", "mode", "path")
)
def test_reviewed_fixture_pair_refuses_other_identities(
    tmp_path: Path,
    name: str,
    mutation: Literal["original_hash", "original_size", "original_mode", "content", "mode", "path"],
) -> None:
    declaration = json.loads((ROOT / "data/publication_derivative.json").read_bytes())
    original = derivative.FileIdentity(**declaration["original_inventory"][name])
    candidate = tmp_path / "candidate.py"
    candidate.write_bytes((ROOT / name).read_bytes())
    candidate.chmod(0o644)
    candidate_name = name
    match mutation:
        case "original_hash":
            original = replace(original, sha256="0" * 64)
        case "original_size":
            original = replace(original, bytes=original.bytes + 1)
        case "original_mode":
            original = replace(original, mode="100755")
        case "content":
            candidate.write_bytes(candidate.read_bytes() + b"\n")
        case "mode":
            candidate.chmod(0o755)
        case "path":
            candidate_name = "tests/unreviewed.py"
        case unreachable:
            assert_never(unreachable)
    with pytest.raises(derivative.DerivativeError, match="reviewed test fixture identity differs"):
        derivative._validate_reviewed_test_fixture_delta(
            candidate_name, original, derivative.file_identity(candidate)
        )


_CROP_PATH = "docs/manuscript/98_supplement_external_benchmarks.md"
_CROP_REPLACEMENTS = (
    (b"0bp 0bp 403.2bp 318.6bp", b"0bp 0bp 383.4bp 318.6bp"),
    (b"403.2bp 0bp 766.8bp 318.6bp", b"383.4bp 0bp 754.2bp 318.6bp"),
    (b"766.8bp 0bp 1137.6bp 318.6bp", b"754.2bp 0bp 1137.6bp 318.6bp"),
)


def _crop_candidate(original: bytes) -> bytes:
    candidate = original
    for before, after in _CROP_REPLACEMENTS:
        assert candidate.count(before) == 1
        candidate = candidate.replace(before, after)
    return candidate


@pytest.mark.parametrize("name", _REVIEWED_FIXTURE_PATHS)
def test_capture_refuses_forged_fixture_original_even_with_fresh_support_hash(project, name: str):
    root, contract = project
    path = root / name
    path.write_bytes((ROOT / name).read_bytes())
    forged_support = {**contract.support, name: derivative.file_hash(path)}
    with pytest.raises(derivative.DerivativeError, match="reviewed test fixture identity differs"):
        derivative.capture_derivative_state(replace(contract, support=forged_support), root)


def _write(root: Path, name: str, content: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    return subprocess.run(
        ["git", "-C", str(root), *args],
        input=input,
        env=environment,
        capture_output=True,
        check=True,
    ).stdout


def _pin(root: Path) -> tuple[str, str]:
    """Create genuine fixture objects, without committing the shared checkout."""
    _git(root, "add", ".")
    return _pin_index(root)


def _pin_index(root: Path) -> tuple[str, str]:
    tree = _git(root, "write-tree").decode().strip()
    raw = (
        f"tree {tree}\nauthor Fixture <fixture@example.invalid> 0 +0000\n"
        "committer Fixture <fixture@example.invalid> 0 +0000\n\ncontrol fixture\n"
    ).encode()
    commit = _git(root, "hash-object", "-t", "commit", "-w", "--stdin", input=raw).decode().strip()
    # Detached HEAD names a real commit object in this temporary repository.
    (root / ".git/HEAD").write_text(commit + "\n")
    return commit, tree


@pytest.fixture
def project(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "core.fsmonitor", "false")
    source = {
        "src/ailm/model.py": b"VALUE = 0.25\n",
        "tests/test_claims_single_writer.py": b"unchanged fixture original\n",
        "tests/test_docs_hydration.py": b"unchanged fixture original\n",
        "vendor/raw/sample.conllu": b"1\tword\tword\tNOUN\t_\t_\t0\troot\t_\t_\n",
        "vendor/counts.json": b'{"count": 2}\n',
        "gnn/models/L1.gnn.md": b"Model: letters\n",
        "pyproject.toml": b"[project]\nname='control'\n",
        "uv.lock": b"version = 1\n",
        "scripts/verify_network_claim_bindings.py": b'"""Inventory fixture; never executed as semantic proof."""\n',
        "data/claims.json": b'{"claims":{"probe":{"value":0.25,"run_id":"original"}}}\n',
        "output/figure.png": b"fixture output bytes\n",
        ".gitignore": b"ignored/\n",
    }
    for name in derivative.PUBLICATION_PATHS:
        if name.endswith("case_word_sweep.py"):
            source[name] = b'VALUE = 0.25\nax.legend(fontsize=FONTS["legend"], loc="upper left")\n'
        elif name == "src/ailm/figures/benchmarks.py":
            source[name] = (
                b'VALUE = 0.25\ndelta_ax.legend(fontsize=FONTS["legend"], loc="lower right", framealpha=0.95)\n'
                b'bucket_ax.legend(fontsize=FONTS["legend"], loc="upper left")\n'
            )
        elif name == "src/ailm/figures/multiplicity.py":
            source[name] = (
                b"VALUE = 0.25\n"
                b"f\"not cluster-adjusted: exact conditional\\ntest over {row['unit']}, \"\n"
            )
        elif name == "src/ailm/manuscript_variables/context.py":
            source[name] = (
                b'VALUE = 0.25\n            f"**{hosted_cohort_label(_probe.key)}**",\n            "",\n'
            )
        elif name.endswith("results.py"):
            source[name] = (
                b'VALUE = 0.25\n"forward\xe2\x80\x93backward\\nsmoothing ceiling",\n'
                b"            value + span * 0.22,\n"
            )
        elif name.endswith("incontext_sweep.py"):
            source[name] = (
                b'VALUE = 0.25\n        axis.set_xlabel("Labeled one-transition examples supplied (k)")\n'
                b'    axes[0].set_ylabel("Letter accuracy (%)")\n'
                b"        inset.set_xticks([rungs[0], rungs[len(rungs) // 2], rungs[-1]])\n"
            )
        elif name == "docs/manuscript/98_supplement_statistics.md":
            source[name] = (
                b"VALUE = 0.25\nsrc/ailm/manuscript_variables.py\n"
                + b"\n### The forest plot's geometry, and where the exact values are recorded\n"
            )
        elif name == "docs/manuscript/98_supplement_audit_trails.md":
            source[name] = (
                b"VALUE = 0.25\n"
                + b"\n### Level 3: the declared gate states a condition the bar itself already clears {#sec:word_gate}\n"
            )
        elif name == "scripts/warm_call_cache.py":
            source[name] = (
                b"VALUE = 0.25\n"
                + b'        "\\nWith the store warm, the rest of the rebuild is:\\n"\n        "  .venv/bin/python scripts/z_render_docs.py\\n"\n        "  .venv/bin/python -I -S scripts/verify_release.py full\\n"\n        "  uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_inference_language_model\\n"\n'
            )
        elif name == _CROP_PATH:
            source[name] = b"{{TOKEN}} [@fig:incontext_sweep] claim:original\n" + b"".join(
                b"\\includegraphics[viewport="
                + before
                + b",clip,width=0.86\\linewidth]{../../output/figures/incontext_sweep.png}\n"
                for before, _ in _CROP_REPLACEMENTS
            )
        else:
            source[name] = b"Original publication prose.\n"
    for name, content in source.items():
        _write(root, name, content)
    receipts = []
    for n in range(5):
        recorder = f"scripts/record_control_{n}_claims.py"
        _write(root, recorder, b'"""Inventory recorder; not a semantic substitute."""\n')
        receipts.append(
            derivative.ReceiptIdentity(
                f"cache/network_claim_bindings/record_control_{n}_claims.json",
                recorder,
                (f"control_{n}",),
                (f"run_{n}",),
                1,
                1,
            )
        )
    commit, tree = _pin(root)
    original = derivative.git_inventory(root, commit)
    delta = {}
    for name in derivative.PUBLICATION_PATHS:
        content = source[name]
        new = (
            derivative.transformed_figure_source(name, content)
            if name.startswith("src/")
            or name
            in {
                "docs/manuscript/98_supplement_statistics.md",
                "docs/manuscript/98_supplement_audit_trails.md",
                "scripts/warm_call_cache.py",
            }
            else _crop_candidate(content)
            if name == _CROP_PATH
            else content + b"Style.\n"
        )
        _write(root, name, new)
        delta[name] = {"original": derivative.digest(content), "current": derivative.digest(new)}
    _write(root, "cache/bench_calls.jsonl", b'{"inventory_only":true}\n')
    for receipt in receipts:
        payload = {
            "recorder": receipt.recorder,
            "claims": {
                "count": 1,
                "labels": list(receipt.labels),
                "entries": {receipt.labels[0]: {"run_id": receipt.run_ids[0]}},
            },
            "cache": {"record_count": 1},
            "configuration": {"endpoint": "http://127.0.0.1:15434"},
        }
        _write(root, receipt.path, json.dumps(payload).encode())
    tokens = _write(root, "evidence/tokens.json", b'{"TOKEN":"original"}\n')
    support = {}
    for name in derivative.SUPPORT_PATHS:
        if name == "data/publication_derivative.json":
            continue
        # Publication/support overlap retains its real transformed numeric input.
        # Replacing it with a dummy would make the numeric mutation vacuous.
        if name not in derivative.PUBLICATION_PATHS and name != "pyproject.toml":
            _write(root, name, b"finite support fixture\n")
        support[name] = derivative.file_hash(root / name)
    inputs = {
        name: derivative.file_hash(root / name)
        for name in ["data/claims.json", "cache/bench_calls.jsonl", *(r.path for r in receipts)]
    }
    contract = derivative.DerivativeContract(
        derivative.CONTRACT_SCHEMA,
        "a" * 64,
        False,
        "b" * 64,
        commit,
        tree,
        commit,
        tree,
        str(tmp_path),
        support,
        "",
        "",
        original,
        delta,
        inputs,
        tuple(receipts),
        {"evidence/tokens.json": derivative.file_hash(tokens)},
        "evidence/tokens.json",
        derivative.file_hash(tokens),
        {},
        "http://127.0.0.1:15434",
    )
    return root, contract


def test_small_real_git_inventory_positive(project):
    root, contract = project
    state = derivative.capture_derivative_state(contract, root)
    assert state.measurement_commit == contract.measurement_commit
    assert state.inventory["src/ailm/model.py"] == contract.original_inventory["src/ailm/model.py"]
    assert state.outputs["output/figure.png"].sha256 == derivative.file_hash(
        root / "output/figure.png"
    )
    assert state == derivative.capture_derivative_state(contract, root)


@pytest.fixture
def large_batch_git_tree(tmp_path):
    root = tmp_path / "large-batch"
    root.mkdir()
    _git(root, "init", "--quiet")
    blobs = (bytes(range(256)) * 256 + b"\x00end", b"B" * 65537)
    oids = [_git(root, "hash-object", "-w", "--stdin", input=b).strip() for b in blobs]
    entries = [
        b"100644 blob " + oids[index % 2] + f"\tfile-{index:04d}\n".encode()
        for index in range(2048)
    ]
    tree = _git(root, "mktree", input=b"".join(entries)).decode().strip()
    commit = (
        _git(
            root,
            "hash-object",
            "-t",
            "commit",
            "-w",
            "--stdin",
            input=(
                f"tree {tree}\nauthor Fixture <fixture@example.invalid> 0 +0000\n"
                "committer Fixture <fixture@example.invalid> 0 +0000\n\nlarge batch\n"
            ).encode(),
        )
        .decode()
        .strip()
    )
    requests = tmp_path / "batch-requests"
    requests.write_bytes(b"".join(oids[index % 2] + b"\n" for index in range(2048)))
    assert requests.stat().st_size > 65536
    expected = {
        f"file-{index:04d}": asdict(
            types.FileIdentity(types.digest(blobs[index % 2]), len(blobs[index % 2]), "100644")
        )
        for index in range(2048)
    }
    return root, commit, requests, expected


def test_git_batch_large_input_and_output_authenticate_every_blob(large_batch_git_tree):
    root, commit, _, expected = large_batch_git_tree
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from pathlib import Path; from dataclasses import asdict; "
            "from ailm._publication_inventory import git_inventory; "
            "print(json.dumps({p:asdict(v) for p,v in "
            "git_inventory(Path(sys.argv[1]),sys.argv[2]).items()}))",
            str(root),
            commit,
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert json.loads(child.stdout) == expected


def test_git_batch_write_before_read_negative_control_blocks(large_batch_git_tree):
    root, _, requests, _ = large_batch_git_tree
    # This deliberately unsafe real transport demonstrates the reciprocal-pipe
    # hazard. It is not a claim that CPython's communicate always reproduces it.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """import signal, subprocess, sys
from pathlib import Path
def deadline(signum, frame):
    raise TimeoutError('reciprocal pipe write blocked')
signal.signal(signal.SIGALRM, deadline)
p = subprocess.Popen([sys.argv[1], '-C', sys.argv[2], 'cat-file', '--batch'],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    signal.alarm(1)
    try:
        p.stdin.write(Path(sys.argv[3]).read_bytes() * 16)
        p.stdin.flush()
    except TimeoutError as error:
        print(str(error), flush=True)
    else:
        raise AssertionError('unsafe transport did not block')
finally:
    signal.alarm(0)
    p.kill()
    p.wait(timeout=5)
    try:
        p.stdin.close()
    except BrokenPipeError:
        pass
    p.stdout.close()
    p.stderr.close()
""",
            str(inventory.trusted_git()),
            str(root),
            str(requests),
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert child.stdout == b"reciprocal pipe write blocked\n"


def test_git_command_failure_is_not_hidden_by_regular_file_input(tmp_path):
    root = tmp_path / "git-failure"
    root.mkdir()
    _git(root, "init", "--quiet")
    with pytest.raises(types.DerivativeError, match="Git cat-file --not-a-real-option failed"):
        inventory.git_command(root, "cat-file", "--not-a-real-option", input_bytes=b"invalid\n")


@pytest.mark.parametrize(
    "name",
    [
        "src/ailm/model.py",
        "vendor/raw/sample.conllu",
        "vendor/counts.json",
        "gnn/models/L1.gnn.md",
        "pyproject.toml",
        "uv.lock",
        "data/claims.json",
        "scripts/record_control_0_claims.py",
    ],
)
def test_numerical_source_and_input_mutation_refused(project, name):
    root, contract = project
    derivative.capture_derivative_state(contract, root)
    path = root / name
    path.write_bytes(path.read_bytes().replace(b"original", b"forged-run") + b"changed\n")
    refusal = "package configuration is invalid" if name == "pyproject.toml" else "identity changed"
    with pytest.raises(derivative.DerivativeError, match=refusal):
        derivative.capture_derivative_state(contract, root)


def test_package_publication_url_delta_accepts_only_url_fields(project):
    root, contract = project
    path = root / "pyproject.toml"
    path.write_text(
        path.read_text() + "\n[project.urls]\nRepository='https://example.org/release'\n"
    )
    support = {**contract.support, "pyproject.toml": derivative.file_hash(path)}
    state = derivative.capture_derivative_state(replace(contract, support=support), root)
    assert state.inventory["pyproject.toml"].sha256 == support["pyproject.toml"]


@pytest.mark.parametrize(
    "configuration",
    [
        "dependencies=['jax==999']\n",
        "requires-python='<3.11'\n",
        "version='99.0.0'\n",
        "\n[build-system]\nrequires=['untrusted-builder']\nbuild-backend='untrusted'\n",
        "\n[tool.pytest.ini_options]\naddopts='--cov-fail-under=0'\n",
    ],
)
def test_package_non_url_delta_refused_even_with_new_support_hash(project, configuration):
    root, contract = project
    path = root / "pyproject.toml"
    path.write_text(path.read_text() + "\n" + configuration)
    support = {**contract.support, "pyproject.toml": derivative.file_hash(path)}
    with pytest.raises(derivative.DerivativeError, match="outside publication URLs"):
        derivative.capture_derivative_state(replace(contract, support=support), root)


@pytest.mark.parametrize(
    "name",
    [
        p
        for p in derivative.PUBLICATION_PATHS
        if p.startswith("src/")
        or p
        in {
            "docs/manuscript/98_supplement_statistics.md",
            "docs/manuscript/98_supplement_audit_trails.md",
            "scripts/warm_call_cache.py",
        }
    ],
)
def test_hidden_numeric_figure_edit_refused_even_with_updated_declared_hash(project, name):
    root, contract = project
    derivative.capture_derivative_state(contract, root)
    path = root / name
    before = path.read_bytes()
    assert b"0.25" in before
    after = before.replace(b"0.25", b"0.75")
    assert after != before
    path.write_bytes(after)
    delta = {p: dict(row) for p, row in contract.publication_delta.items()}
    delta[name]["current"] = derivative.file_hash(path)
    refusal = (
        "support bytes changed" if name in derivative.SUPPORT_PATHS else "nonstyling figure delta"
    )
    with pytest.raises(derivative.DerivativeError, match=refusal):
        derivative.capture_derivative_state(replace(contract, publication_delta=delta), root)


@pytest.mark.parametrize(
    "operation",
    [
        "added",
        "deleted",
        "mode",
        "symlink",
        "ancestor_symlink",
        "ignored",
        "output_implementation",
        "evidence_implementation",
    ],
)
def test_inventory_structure_mutation_refused(project, tmp_path, operation):
    root, contract = project
    derivative.capture_derivative_state(contract, root)
    if operation == "added":
        _write(root, "src/ailm/new.py", b"NUMERIC = 1\n")
        _git(root, "add", "src/ailm/new.py")
    elif operation == "deleted":
        (root / "src/ailm/model.py").unlink()
        _git(root, "add", "-u", "src/ailm/model.py")
    elif operation == "mode":
        (root / "src/ailm/model.py").chmod(0o755)
    elif operation == "symlink":
        outside = _write(tmp_path, "outside.py", b"VALUE = 0.25\n")
        (root / "src/ailm/model.py").unlink()
        (root / "src/ailm/model.py").symlink_to(outside)
    elif operation == "ancestor_symlink":
        outside = tmp_path / "escape"
        outside.mkdir()
        (root / "ignored").symlink_to(outside, target_is_directory=True)
    else:
        prefix = {
            "ignored": "ignored",
            "output_implementation": "output",
            "evidence_implementation": "evidence",
        }[operation]
        _write(root, f"{prefix}/hidden.py", b"NUMERIC = 1\n")
    with pytest.raises(derivative.DerivativeError):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize(
    "mutation", ["wrong_pin", "wrong_inventory", "wrong_input", "wrong_worker", "changed_cache"]
)
def test_wrong_pin_and_bound_input_digests_refused(project, mutation):
    root, contract = project
    derivative.capture_derivative_state(contract, root)
    if mutation == "wrong_pin":
        contract = replace(contract, measurement_commit="f" * 40)
    elif mutation == "wrong_inventory":
        inventory = dict(contract.original_inventory)
        inventory["src/ailm/model.py"] = replace(inventory["src/ailm/model.py"], sha256="f" * 64)
        contract = replace(contract, original_inventory=inventory)
    elif mutation == "wrong_input":
        contract = replace(
            contract, inputs={**contract.inputs, "cache/bench_calls.jsonl": "f" * 64}
        )
    elif mutation == "wrong_worker":
        (root / "evidence/tokens.json").write_text('{"TOKEN":"altered"}')
    else:
        (root / "cache/bench_calls.jsonl").write_bytes(b"corrupt cache\n")
    with pytest.raises(derivative.DerivativeError):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "producer", "run_id", "label", "count", "malformed"]
)
def test_receipt_identity_set_is_not_vacuous(project, mutation):
    root, contract = project
    derivative.validate_receipt_identities(contract, root)
    path = root / contract.receipts[0].path
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        _write(root, "cache/network_claim_bindings/extra.json", b"{}")
    elif mutation == "malformed":
        path.write_bytes(b"{invalid")
    else:
        row = json.loads(path.read_bytes())
        if mutation == "producer":
            row["recorder"] = contract.receipts[1].recorder
        elif mutation == "run_id":
            row["claims"]["entries"]["control_0"]["run_id"] = "new-run"
        elif mutation == "label":
            row["claims"]["labels"] = ["another"]
        else:
            row["cache"]["record_count"] = 0
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        derivative.validate_receipt_identities(contract, root)


def test_before_after_state_refuses_real_subprocess_mutation(project):
    root, contract = project
    before = derivative.capture_derivative_state(contract, root)
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('drift')",
            str(root / "vendor/counts.json"),
        ],
        check=True,
        capture_output=True,
    )
    assert before.inputs == contract.inputs
    with pytest.raises(derivative.DerivativeError, match="identity changed"):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize(
    "content", [b"{}", b"", b"[]", b"null", b'{"schema":1,"schema":2}', b'{"x":NaN}']
)
def test_missing_empty_malformed_attestation_never_accepted(project, tmp_path, content):
    root, contract = project
    state = derivative.capture_derivative_state(contract, root)
    proof = tmp_path / "attestation.json"
    proof.write_bytes(content)
    result = derivative.validate_attestation(proof, contract, state, "offline")
    assert result.accepted is False
    assert result.errors


def test_missing_attestation_never_accepted(project, tmp_path):
    root, contract = project
    state = derivative.capture_derivative_state(contract, root)
    result = derivative.validate_attestation(tmp_path / "missing.json", contract, state, "offline")
    assert result.accepted is False
    assert result.errors


def _declaration(root, contract):
    fields = {
        "schema": derivative.DECLARATION_SCHEMA,
        "measurement_commit": contract.measurement_commit,
        "measurement_tree": contract.measurement_tree,
        "original_inventory": {p: asdict(i) for p, i in contract.original_inventory.items()},
        "publication_delta": contract.publication_delta,
        "inputs": contract.inputs,
        "receipts": [asdict(r) for r in contract.receipts],
        "worker_evidence": contract.worker_evidence,
        "tokens_path": contract.tokens_path,
        "tokens_sha256": contract.tokens_sha256,
        "baseline_untracked": {},
        "archival_artifacts": {},
        "endpoint": contract.endpoint,
    }
    return _write(root, "data/publication_derivative.json", json.dumps(fields).encode())


def test_candidate_declaration_is_typed_and_never_reviewed(project, tmp_path):
    root, contract = project
    declaration = _declaration(root, contract)
    parsed = derivative.load_candidate_contract(declaration, root, tmp_path)
    assert isinstance(parsed, derivative.DerivativeContract)
    assert parsed.reviewed is False
    assert parsed.receipts == contract.receipts
    assert parsed.original_inventory == contract.original_inventory
    derivative.capture_derivative_state(parsed, root)


@pytest.mark.parametrize("mutation", ["none", "original", "crop", "prose", "missing", "extra"])
def test_named_crop_delta_uses_real_git_and_exact_bound_bytes(project, tmp_path, mutation):
    root, contract = project
    original = derivative.git_bytes(root, "show", contract.measurement_commit + ":" + _CROP_PATH)
    assert (root / _CROP_PATH).read_bytes() == _crop_candidate(original)
    declaration = _declaration(root, contract)
    row = json.loads(declaration.read_text())
    if mutation == "original":
        row["publication_delta"][_CROP_PATH]["original"] = "f" * 64
    elif mutation == "missing":
        del row["publication_delta"][_CROP_PATH]
    elif mutation == "extra":
        row["publication_delta"]["docs/manuscript/undeclared.md"] = dict(
            row["publication_delta"][_CROP_PATH]
        )
    declaration.write_text(json.dumps(row))
    if mutation in {"original", "missing", "extra"}:
        message = "original delta mismatch" if mutation == "original" else "approved finite set"
        with pytest.raises(derivative.DerivativeError, match=message):
            derivative.load_candidate_contract(declaration, root, tmp_path)
        return
    parsed = derivative.load_candidate_contract(declaration, root, tmp_path)
    derivative.capture_derivative_state(parsed, root)
    if mutation == "none":
        return
    path = root / _CROP_PATH
    before = path.read_bytes()
    after = (
        before.replace(b"383.4bp", b"380.0bp")
        if mutation == "crop"
        else before + b"Forged prose.\n"
    )
    assert after != before
    path.write_bytes(after)
    with pytest.raises(derivative.DerivativeError, match="publication delta differs"):
        derivative.capture_derivative_state(parsed, root)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "missing",
        "wrong_schema",
        "wrong_tree",
        "wrong_pin",
        "empty_inventory",
        "empty_inputs",
        "missing_recorder",
        "extra_recorder",
        "duplicate_recorder",
        "empty_labels",
        "empty_run_ids",
        "wrong_claims_count",
        "empty_calls",
        "boolean_count",
        "wrong_claims_pin",
        "remote_endpoint",
        "hostname_endpoint",
        "path_escape",
        "wrong_tokens",
    ],
)
def test_declaration_refuses_incomplete_and_inconsistent_identities(project, tmp_path, mutation):
    root, contract = project
    declaration = _declaration(root, contract)
    derivative.load_candidate_contract(declaration, root, tmp_path)
    row = json.loads(declaration.read_bytes())
    if mutation == "unknown":
        row["free_allowlist"] = ["src/ailm/model.py"]
    elif mutation == "missing":
        row.pop("measurement_tree")
    elif mutation == "wrong_schema":
        row["schema"] = "invented"
    elif mutation == "wrong_tree":
        row["measurement_tree"] = "f" * 40
    elif mutation == "wrong_pin":
        row["measurement_commit"] = "f" * 40
    elif mutation == "empty_inventory":
        row["original_inventory"] = {}
    elif mutation == "empty_inputs":
        row["inputs"] = {}
    elif mutation == "missing_recorder":
        row["receipts"].pop()
    elif mutation == "extra_recorder":
        row["receipts"].append(dict(row["receipts"][0]))
    elif mutation == "duplicate_recorder":
        row["receipts"][0]["recorder"] = row["receipts"][1]["recorder"]
    elif mutation == "empty_labels":
        row["receipts"][0]["labels"] = []
    elif mutation == "empty_run_ids":
        row["receipts"][0]["run_ids"] = []
    elif mutation == "wrong_claims_count":
        row["receipts"][0]["claims_count"] = 2
    elif mutation == "empty_calls":
        row["receipts"][0]["calls_count"] = 0
    elif mutation == "boolean_count":
        row["receipts"][0]["calls_count"] = True
    elif mutation == "wrong_claims_pin":
        row["inputs"]["data/claims.json"] = "f" * 64
    elif mutation == "remote_endpoint":
        row["endpoint"] = "http://192.0.2.1:15434"
    elif mutation == "hostname_endpoint":
        row["endpoint"] = "http://localhost:15434"
    elif mutation == "path_escape":
        row["tokens_path"] = "../outside.json"
    else:
        row["tokens_sha256"] = "f" * 64
    declaration.write_text(json.dumps(row))
    with pytest.raises(derivative.DerivativeError):
        derivative.load_candidate_contract(declaration, root, tmp_path)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "a//b", "", "a/./b"])
def test_relative_paths_cannot_escape_or_alias(path):
    assert derivative.relative_path("a/b") == "a/b"
    with pytest.raises(derivative.DerivativeError):
        derivative.relative_path(path)


@pytest.mark.parametrize(
    "name",
    [
        p
        for p in derivative.PUBLICATION_PATHS
        if p.startswith("src/")
        or p
        in {
            "docs/manuscript/98_supplement_statistics.md",
            "docs/manuscript/98_supplement_audit_trails.md",
            "scripts/warm_call_cache.py",
        }
    ],
)
def test_transformation_requires_exactly_one_original_literal(project, name):
    root, contract = project
    original = derivative.git_bytes(root, "show", contract.measurement_commit + ":" + name)
    assert derivative.transformed_figure_source(name, original) == (root / name).read_bytes()
    with pytest.raises(derivative.DerivativeError):
        derivative.transformed_figure_source(name, original + original)
    with pytest.raises(derivative.DerivativeError):
        derivative.transformed_figure_source(name, b"NUMERIC = 1\n")


def _actual_execution(tmp_path):
    report = {"control": "real subprocess", "valid": True}
    argv = [sys.executable, "-I", "-S", "-B", "-c", "print('real control executed')"]
    environment = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        argv, cwd=tmp_path, env=environment, capture_output=True, text=True, check=True
    )
    return {
        "argv": argv,
        "cwd": str(tmp_path),
        "environment": environment,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report": report,
        "report_sha256": derivative.digest(derivative.canonical_bytes(report)),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "argv",
        "isolation",
        "nonzero",
        "empty_output",
        "empty_report",
        "forged_hash",
        "unknown",
        "missing",
    ],
)
def test_execution_evidence_has_real_positive_and_refusal_controls(tmp_path, mutation):
    row = _actual_execution(tmp_path)
    assert derivative._execution(row) == row
    if mutation == "argv":
        row["argv"] = []
    elif mutation == "isolation":
        row["argv"].remove("-I")
    elif mutation == "nonzero":
        row["exit_code"] = 1
    elif mutation == "empty_output":
        row["stdout"] = row["stderr"] = ""
    elif mutation == "empty_report":
        row["report"] = {}
    elif mutation == "forged_hash":
        row["report_sha256"] = "f" * 64
    elif mutation == "unknown":
        row["invented"] = True
    else:
        row.pop("stderr")
    with pytest.raises(derivative.DerivativeError):
        derivative._execution(row)


def _incomplete_attestation(contract, state, root, output_path):
    """Real inventory, deliberately absent semantic proof; never a fake success."""
    run_directory = root.parent / "publication-offline-controls"
    run_directory.mkdir()
    materialization = _execution_module().materialize(contract, root, run_directory / "original")
    materialization["after"] = materialization["before"]
    return {
        "schema": derivative.ATTESTATION_SCHEMA,
        "output_path": str(output_path.absolute()),
        "run_directory": str(run_directory),
        "mode": "offline",
        "accepted": False,
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
        "initial_state": json.loads(derivative.canonical_bytes(asdict(state))),
        "final_state": json.loads(derivative.canonical_bytes(asdict(state))),
        "materialization": materialization,
        "dependency_before": {"integrity_ok": True},
        "dependency_after": {"integrity_ok": True},
        "semantic": None,
        "transport_controls": None,
        "artists": None,
        "selfproof": None,
        "live": None,
        "final_qa": "pending-parent-independent-all-page-review",
    }


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("stale", "stale/drifted"),
        ("dependency_drift", "dependency integrity"),
        ("snapshot_drift", "snapshot drift"),
        ("extra_snapshot", "actual retained snapshot"),
        ("wrong_bootstrap", "bootstrap bytes"),
        ("wrong_verifier", "original verifier"),
        ("fresh_calls", "measurement/call"),
        ("wrong_pin", "measurement_commit mismatch"),
        ("wrong_mode", "mode/schema"),
        ("forged_success", "execution"),
        ("nonzero_semantic", "exit mismatch"),
        ("empty_semantic", "empty execution evidence"),
    ],
)
def test_attestation_identity_and_nonexecution_forgery_refused(
    project, tmp_path, mutation, expected
):
    root, contract = project
    state = derivative.capture_derivative_state(contract, root)
    path = tmp_path / "proof.json"
    row = _incomplete_attestation(contract, state, root, path)
    path.write_text(json.dumps(row))
    baseline = derivative.validate_attestation(path, contract, state, "offline")
    assert not baseline.accepted and "execution" in baseline.errors[0]
    if mutation == "stale":
        row["final_state"]["fingerprint"] = "f" * 64
    elif mutation == "dependency_drift":
        row["dependency_after"] = {"integrity_ok": False}
    elif mutation == "snapshot_drift":
        row["materialization"]["after"] = {}
    elif mutation == "extra_snapshot":
        row["materialization"]["before"] = {
            **row["materialization"]["before"],
            "escape.py": {"sha256": "f" * 64},
        }
        row["materialization"]["after"] = row["materialization"]["before"]
    elif mutation == "wrong_bootstrap":
        row["bootstrap_sha256"] = "f" * 64
    elif mutation == "wrong_verifier":
        row["original_verifier_sha256"] = "f" * 64
    elif mutation == "fresh_calls":
        row["fresh_hosted_calls"] = 1
    elif mutation == "wrong_pin":
        row["measurement_commit"] = "f" * 40
    elif mutation == "wrong_mode":
        row["mode"] = "live"
    elif mutation == "forged_success":
        row["accepted"] = True
    else:
        row["semantic"] = _actual_execution(tmp_path)
        if mutation == "nonzero_semantic":
            row["semantic"]["exit_code"] = 1
        else:
            row["semantic"]["stdout"] = row["semantic"]["stderr"] = ""
    path.write_text(json.dumps(row))
    result = derivative.validate_attestation(path, contract, state, "offline")
    assert result.accepted is False
    assert any(expected in error for error in result.errors), result.errors


def _execution_module():
    # Use the same production module identity the infrastructure gate measures.
    from scripts import _publication_execution

    return _publication_execution


@pytest.mark.parametrize("code", ["'original CLI refusal'", "0", "7", "None"])
def test_original_worker_preserves_system_exit_status_and_message(tmp_path, code):
    source = tmp_path / "original"
    target = _write(
        source, "scripts/verify_network_claim_bindings.py", f"raise SystemExit({code})\n".encode()
    )
    report = tmp_path / "original-report.json"
    site = (
        ROOT
        / ".venv/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    result = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-I",
            "-S",
            "-B",
            str(ROOT / "scripts/_publication_execution.py"),
            "internal-original",
            str(source),
            str(site),
            "offline",
            str(report),
            "http://127.0.0.1:15434",
        ],
        cwd=source,
        env=_execution_module().sanitized_environment("http://127.0.0.1:15434", tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected = 1 if code.startswith("'") else 0 if code == "None" else int(code)
    assert result.returncode == expected, result.stderr
    assert result.stderr.strip() == ("original CLI refusal" if expected == 1 else "")
    recorded = json.loads(report.read_text())
    assert recorded["exit_code"] == expected
    assert str(target) in recorded["module_origins"].values()


@pytest.mark.parametrize("operation", ["baseline", "connect", "connect_ex", "dns", "udp", "spawn"])
def test_actual_transport_subprocess_positive_and_denial_controls(tmp_path, operation):
    module = _execution_module()
    row = module.execution(
        ROOT,
        tmp_path,
        operation,
        "internal-transport",
        [operation, "unused", "http://127.0.0.1:15434"],
        "http://127.0.0.1:15434",
        timeout=30,
    )
    assert derivative._execution(row) == row
    assert row["exit_code"] == 0, row["stderr"]
    assert row["report"]["operation"] == operation
    assert row["report"]["valid"] is True
    assert row["report"]["reached"] is (operation == "baseline")
    assert row["report"]["denied"] is (operation != "baseline")


def test_environment_sanitization_ignores_original_poison(monkeypatch, tmp_path):
    module = _execution_module()
    endpoint = "http://127.0.0.1:15434"
    before = module.sanitized_environment(endpoint, tmp_path)
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "OPENROUTER_API_KEY",
        "AILM_OLLAMA_ENDPOINT",
        "AILM_CLAIMS_PATH",
        "AILM_CACHE_PATH",
        "AILM_OPENROUTER_BUDGET_FILE",
        "GNN_ROOT",
        "HOME",
        "PATH",
    ):
        monkeypatch.setenv(key, "contaminated-control-value")
    after = module.sanitized_environment(endpoint, tmp_path)
    assert before == after
    assert after["AILM_OLLAMA_ENDPOINT"] == endpoint
    assert "contaminated-control-value" not in after.values()
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        "import os,json; print(json.dumps(dict(os.environ)))",
    ]
    result = subprocess.run(
        argv, env=after, cwd=tmp_path, capture_output=True, text=True, check=True
    )
    observed = json.loads(result.stdout)
    assert all(observed[key] == value for key, value in after.items())
    assert "PYTHONPATH" not in observed and "OPENROUTER_API_KEY" not in observed


@pytest.mark.parametrize("mutation", ["byte", "extra", "mode", "symlink", "directory_mode"])
def test_snapshot_before_after_mutations_are_real_files(project, tmp_path, mutation):
    root, contract = project
    module = _execution_module()
    destination = tmp_path / "snapshot"
    row = module.materialize(contract, root, destination)
    before = module.snapshot_inventory(destination)
    assert before == row["before"]
    assert row["pre_overlay"] == {p: asdict(i) for p, i in contract.original_inventory.items()}
    assert set(before) == set(contract.original_inventory) | set(contract.inputs)
    target = destination / "src/ailm/model.py"
    if mutation == "byte":
        target.chmod(0o644)
        target.write_bytes(b"VALUE = 0.75\n")
        target.chmod(0o444)
    elif mutation == "extra":
        destination.chmod(0o755)
        _write(destination, "extra.py", b"NUMERIC = 1\n").chmod(0o444)
        destination.chmod(0o555)
    elif mutation == "mode":
        target.chmod(0o644)
    elif mutation == "directory_mode":
        destination.chmod(0o755)
    else:
        target.parent.chmod(0o755)
        target.unlink()
        target.symlink_to(root / "src/ailm/model.py")
        target.parent.chmod(0o555)
    if mutation in {"symlink", "directory_mode"}:
        with pytest.raises(derivative.DerivativeError):
            module.snapshot_inventory(destination)
    else:
        assert module.snapshot_inventory(destination) != before
    # Permission changes leave the failed snapshot readable for evidence.


def test_original_import_origins_and_current_source_denial(tmp_path):
    """Import actual project source in isolation, never a recorder stand-in."""
    source = tmp_path / "original"
    original_init = derivative.publication_source_bytes(
        ROOT, "9f8842d93c9d3c463589fb6fae45087e7b2048a5", "src/ailm/__init__.py"
    )
    _write(source, "src/ailm/__init__.py", original_init)
    code = (
        "import importlib.util,sys,json,runpy; from pathlib import Path; "
        "spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "source=Path(sys.argv[2]); trusted=Path(sys.argv[3]); "
        "executed=module.install_policy('offline','http://127.0.0.1:15434',source,trusted); "
        "module.configure_imports(source,trusted,Path(sys.argv[4])); "
        "import ailm; print(json.dumps(module.origins(source,executed))); "
        "runpy.run_path(str(trusted/'src/ailm/__init__.py'))"
    )
    site = (
        ROOT
        / ".venv/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    environment = _execution_module().sanitized_environment("http://127.0.0.1:15434", tmp_path)
    environment["PYTHONPATH"] = str(ROOT / "src")
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        code,
        str(ROOT / "scripts/_publication_execution.py"),
        str(source),
        str(ROOT),
        str(site),
    ]
    result = subprocess.run(argv, env=environment, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    origins = json.loads(result.stdout)
    assert origins["ailm"] == str(source / "src/ailm/__init__.py")
    assert all(str(source) in origin for origin in origins.values())
    assert "current project execution in original process" in result.stderr


@pytest.mark.parametrize(
    "operation", ["gethostbyname", "gethostbyaddr", "getnameinfo", "system", "posix_spawn"]
)
def test_additional_dns_and_process_escape_paths_denied_in_real_child(tmp_path, operation):
    code = """
import importlib.util,json,os,socket,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path(sys.argv[2]); module.install_policy('offline','http://127.0.0.1:15434',root,root)
operation=sys.argv[3]
try:
    if operation=='gethostbyname': socket.gethostbyname('localhost')
    elif operation=='gethostbyaddr': socket.gethostbyaddr('127.0.0.1')
    elif operation=='getnameinfo': socket.getnameinfo(('127.0.0.1',12345),0)
    elif operation=='system': os.system('exit 0')
    else: os.posix_spawn(sys.executable,[sys.executable,'-I','-S','-B','-c','pass'],{})
except PermissionError as error:
    print(json.dumps({'denied':True,'error':str(error)}))
else:
    raise RuntimeError('escape operation unexpectedly allowed')
"""
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        code,
        str(ROOT / "scripts/_publication_execution.py"),
        str(ROOT),
        operation,
    ]
    result = subprocess.run(
        argv,
        cwd=tmp_path,
        env=_execution_module().sanitized_environment("http://127.0.0.1:15434", tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["denied"] is True and "policy denies" in report["error"]


def test_live_policy_permits_only_authenticated_ephemeral_loopback(tmp_path):
    code = """
import importlib.util,json,socket,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path(sys.argv[2]);server=socket.socket();server.bind(('127.0.0.1',0));server.listen();server.settimeout(1)
port=server.getsockname()[1]
module.install_policy('live',f'http://127.0.0.1:{port}',root,root)
client=socket.socket();client.connect(('127.0.0.1',port));connection,_=server.accept()
assert socket.getaddrinfo('127.0.0.1',port)
denied=0
for address in [('127.0.0.1',port+1),('192.0.2.1',port)]:
    try: client.connect(address)
    except PermissionError: denied+=1
    else: raise RuntimeError('unapproved destination allowed')
try: socket.getaddrinfo('127.0.0.1',port+1)
except PermissionError: denied+=1
else: raise RuntimeError('unapproved DNS port allowed')
connection.close();client.close();server.close();print(json.dumps({'reached':True,'denied':denied}))
"""
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-I",
        "-S",
        "-B",
        "-c",
        code,
        str(ROOT / "scripts/_publication_execution.py"),
        str(ROOT),
    ]
    result = subprocess.run(
        argv,
        cwd=tmp_path,
        env=_execution_module().sanitized_environment("http://127.0.0.1:15434", tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"reached": True, "denied": 3}


@pytest.mark.parametrize(
    "mutation", ["stdout", "stderr", "report", "invocation", "forged_row", "stale_run"]
)
def test_retained_execution_authenticates_actual_production_subprocess_files(tmp_path, mutation):
    module = _execution_module()
    row = module.execution(
        ROOT,
        tmp_path,
        "baseline",
        "internal-transport",
        ["baseline", "unused", "http://127.0.0.1:15434"],
        "http://127.0.0.1:15434",
        timeout=30,
    )
    assert derivative.retained_execution(row, tmp_path) == row
    if mutation == "stdout":
        (tmp_path / "baseline.stdout").write_text("forged success")
    elif mutation == "stderr":
        (tmp_path / "baseline.stderr").write_text("hidden failure")
    elif mutation == "report":
        (tmp_path / "baseline.json").write_text("{}")
    elif mutation == "invocation":
        (tmp_path / "baseline.execution.json").write_text("{}")
    elif mutation == "forged_row":
        row = {**row, "report": {**row["report"], "reached": False}}
        row["report_sha256"] = derivative.digest(derivative.canonical_bytes(row["report"]))
    else:
        wrong = tmp_path / "stale"
        wrong.mkdir()
        with pytest.raises(derivative.DerivativeError, match="outside fresh run"):
            derivative.retained_execution(row, wrong)
        return
    with pytest.raises(derivative.DerivativeError):
        derivative.retained_execution(row, tmp_path)


def test_pinned_git_symlink_is_refused_without_following_escape(tmp_path):
    root = tmp_path / "git-symlink"
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "core.fsmonitor", "false")
    outside = _write(tmp_path, "outside.txt", b"escape bytes")
    (root / "model.py").symlink_to(outside)
    commit, _ = _pin(root)
    with pytest.raises(derivative.DerivativeError, match="nonregular pinned Git entry"):
        derivative.git_inventory(root, commit)


def test_materialization_refuses_corrupt_input_and_preserves_failed_tree(project, tmp_path):
    root, contract = project
    (root / "cache/bench_calls.jsonl").write_bytes(b"corrupt cache bytes")
    failed = tmp_path / "failed-snapshot"
    with pytest.raises(derivative.DerivativeError, match="overlay hash mismatch"):
        _execution_module().materialize(contract, root, failed)
    assert failed.is_dir()
    assert (failed / "cache/bench_calls.jsonl").read_bytes() == b"corrupt cache bytes"
    assert (failed / "src/ailm/model.py").read_bytes() == b"VALUE = 0.25\n"


RAW_PATHS = (
    "vendor/cmudict.dict",
    "vendor/en_ewt-ud-dev.conllu",
    "vendor/staging_ud_czech/cs_pdtc-ud-dev.conllu",
    "vendor/staging_ud_czech/cs_pdtc-ud-train-ft.conllu",
)
CORPUS_SOURCES = (
    "src/ailm/corpus/cmudict.py",
    "src/ailm/corpus/ud.py",
    "src/ailm/corpus/ud_czech.py",
)
ARCHIVAL_PATH = "vendor/staging_ud_czech/cs_pdtc-ud-test.conllu"
ARCHIVAL_PROOF = ".omo/evidence/publication-derivative/czech-archival-proof.json"


def _add_pinned_sources(root, contract, sources):
    for name in sources:
        _write(root, name, (ROOT / name).read_bytes())
    _git(root, "add", *sources)
    commit, tree = _pin_index(root)
    return replace(
        contract,
        measurement_commit=commit,
        measurement_tree=tree,
        publication_commit=commit,
        publication_tree=tree,
        original_inventory=derivative.git_inventory(root, commit),
    )


@pytest.fixture
def raw_project(project):
    root, contract = project
    contract = _add_pinned_sources(root, contract, CORPUS_SOURCES)
    for name in RAW_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return root, contract


@pytest.mark.parametrize("name", RAW_PATHS)
@pytest.mark.parametrize("operation", ["sha", "size", "mode", "symlink"])
@pytest.mark.publication_context
def test_v5_original_raw_exact_identity_controls(raw_project, name, operation):
    root, contract = raw_project
    identities = proof.raw_identities(root, contract)
    assert set(identities) == set(RAW_PATHS)
    state = derivative.capture_derivative_state(contract, root)
    assert all(state.inventory[p] == identities[p] for p in RAW_PATHS)
    path = root / name
    if operation == "sha":
        with path.open("r+b") as stream:
            first = stream.read(1)
            stream.seek(0)
            stream.write(bytes([first[0] ^ 1]))
    elif operation == "size":
        with path.open("ab") as stream:
            stream.write(b"\n")
    elif operation == "mode":
        path.chmod(0o755)
    else:
        original = path.with_suffix(".retained")
        path.rename(original)
        path.symlink_to(original)
    with pytest.raises(
        derivative.DerivativeError, match="original raw input identity|not regular|symlink"
    ):
        proof.raw_identities(root, contract)


@pytest.mark.publication_context
def test_v5_unknown_raw_is_not_grandfathered(raw_project):
    root, contract = raw_project
    derivative.capture_derivative_state(contract, root)
    _write(root, "vendor/staging_ud_czech/unapproved.conllu", b"unknown raw input\n")
    with pytest.raises(derivative.DerivativeError, match="untracked numerical/source/input"):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize(
    "content,expected",
    [
        (b"PIN = 'original'\nPIN = 'duplicate'\n", "missing/duplicate"),
        (b"OTHER = 'original'\n", "missing/duplicate"),
        (b"PIN: str = 'original'\n", "missing/duplicate"),
        (b"def f():\n    PIN = 'original'\n", "missing/duplicate"),
        (b"PIN = str('original')\n", "literal"),
        (b"PIN = (\n", "syntax|literal|parse"),
    ],
)
def test_v5_original_literal_wrong_ast_refused(project, content, expected):
    root, contract = project
    name = "src/ailm/literal_control.py"
    _write(root, name, b"PIN = 'original'\n")
    _git(root, "add", name)
    commit, tree = _pin_index(root)
    contract = replace(
        contract,
        measurement_commit=commit,
        measurement_tree=tree,
        original_inventory=derivative.git_inventory(root, commit),
    )
    assert proof.original_literal(root, contract, name, "PIN") == "original"
    _write(root, name, content)
    _git(root, "add", name)
    commit, tree = _pin_index(root)
    mutated = replace(
        contract,
        measurement_commit=commit,
        measurement_tree=tree,
        original_inventory=derivative.git_inventory(root, commit),
    )
    with pytest.raises(derivative.DerivativeError, match=expected):
        proof.original_literal(root, mutated, name, "PIN")


def test_v5_original_literal_authenticates_git_not_worktree(project):
    root, contract = project
    contract = _add_pinned_sources(root, contract, CORPUS_SOURCES)
    name = CORPUS_SOURCES[0]
    value = proof.original_literal(root, contract, name, "CMUDICT_RAW_SHA256")
    _write(root, name, b"CMUDICT_RAW_SHA256 = 'forged'\n")
    assert proof.original_literal(root, contract, name, "CMUDICT_RAW_SHA256") == value
    identities = dict(contract.original_inventory)
    identities[name] = replace(identities[name], sha256="f" * 64)
    with pytest.raises(derivative.DerivativeError, match="source literal identity"):
        proof.original_literal(
            root, replace(contract, original_inventory=identities), name, "CMUDICT_RAW_SHA256"
        )


def _install_archival(root, contract, temporary):
    path = root / ARCHIVAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / ARCHIVAL_PATH, path)
    retained = json.loads((ROOT / ARCHIVAL_PROOF).read_bytes())
    response = temporary / "retained-api-response.json"
    shutil.copyfile(Path(retained["tree_response_path"]), response)
    retained["tree_response_path"] = str(response)
    _write(root, ARCHIVAL_PROOF, json.dumps(retained).encode())
    artifact = types.ArchivalArtifact(
        derivative.file_identity(path), ARCHIVAL_PROOF, derivative.file_hash(root / ARCHIVAL_PROOF)
    )
    return replace(contract, archival_artifacts={ARCHIVAL_PATH: artifact}), retained, response


@pytest.fixture
def archival_project(project, tmp_path):
    root, contract = project
    contract = _add_pinned_sources(root, contract, (CORPUS_SOURCES[2],))
    contract, retained, response = _install_archival(root, contract, tmp_path)
    evidence = proof.archival_identity_evidence(root, contract)
    assert evidence == {
        ARCHIVAL_PROOF: derivative.file_hash(root / ARCHIVAL_PROOF),
        str(response): derivative.file_hash(response),
    }
    return root, contract, retained, response


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("schema", "role/path/schema"),
        ("role", "role/path/schema"),
        ("path", "role/path/schema"),
        ("unknown", "missing/unknown"),
        ("missing", "missing/unknown"),
        ("revision", "revision/identity"),
        ("size", "revision/identity"),
        ("sha", "revision/identity"),
        ("blob", "entry missing/duplicate/changed"),
        ("url", "revision/truncation"),
        ("response_digest", "response digest"),
        ("truncated", "revision/truncation"),
        ("response_revision", "revision/truncation"),
        ("duplicate_entry", "entry missing/duplicate/changed"),
        ("entry_size", "entry missing/duplicate/changed"),
        ("entry_mode", "entry missing/duplicate/changed"),
        ("entry_type", "entry missing/duplicate/changed"),
        ("entry_path", "entry missing/duplicate/changed"),
        ("proof_tamper", "proof digest"),
        ("response_tamper", "response digest"),
        ("file_tamper", "file identity"),
        ("file_mode", "file identity"),
        ("file_symlink", "not regular|symlink"),
    ],
)
@pytest.mark.publication_context
def test_v5_archival_authentication_controls(archival_project, operation, expected):
    root, contract, retained, response = archival_project
    # Fixture has already authenticated honest retained upstream bytes before mutation.
    response_data = json.loads(response.read_bytes())
    if operation in {"file_tamper", "file_mode", "file_symlink"}:
        path = root / ARCHIVAL_PATH
        if operation == "file_mode":
            path.chmod(0o755)
        elif operation == "file_symlink":
            renamed = path.with_suffix(".retained")
            path.rename(renamed)
            path.symlink_to(renamed)
        else:
            with path.open("ab") as stream:
                stream.write(b"drift")
    elif operation == "proof_tamper":
        with (root / ARCHIVAL_PROOF).open("ab") as stream:
            stream.write(b" ")
    elif operation == "response_tamper":
        with response.open("ab") as stream:
            stream.write(b" ")
    else:
        if operation.startswith("entry_") or operation == "duplicate_entry":
            entry = next(x for x in response_data["tree"] if x["path"] == Path(ARCHIVAL_PATH).name)
            if operation == "duplicate_entry":
                response_data["tree"].append(dict(entry))
            else:
                key = operation.removeprefix("entry_")
                entry[key] = {"size": 1, "mode": "100755", "type": "tree", "path": "wrong.conllu"}[
                    key
                ]
        elif operation == "truncated":
            response_data["truncated"] = True
        elif operation == "response_revision":
            response_data["sha"] = "f" * 40
        elif operation == "unknown":
            retained["unreviewed"] = True
        elif operation == "missing":
            retained.pop("code_boundary")
        else:
            key, value = {
                "schema": ("schema", "wrong"),
                "role": ("role", "raw replay input"),
                "path": ("path", "vendor/wrong.conllu"),
                "revision": ("source_revision", "f" * 40),
                "size": ("bytes", 1),
                "sha": ("sha256", "f" * 64),
                "blob": ("git_blob_sha1", "f" * 40),
                "url": ("source_url", "https://example.invalid"),
                "response_digest": ("tree_response_sha256", "f" * 64),
            }[operation]
            retained[key] = value
        if operation.startswith("entry_") or operation in {
            "duplicate_entry",
            "truncated",
            "response_revision",
        }:
            response.write_text(json.dumps(response_data))
            retained["tree_response_sha256"] = derivative.file_hash(response)
        _write(root, ARCHIVAL_PROOF, json.dumps(retained).encode())
        artifact = replace(
            contract.archival_artifacts[ARCHIVAL_PATH],
            proof_sha256=derivative.file_hash(root / ARCHIVAL_PROOF),
        )
        contract = replace(contract, archival_artifacts={ARCHIVAL_PATH: artifact})
    with pytest.raises(derivative.DerivativeError, match=expected):
        proof.archival_identity_evidence(root, contract)


@pytest.mark.parametrize(
    "operation", ["unknown_path", "unknown_field", "missing_field", "size", "mode", "proof_path"]
)
@pytest.mark.publication_context
def test_v5_archival_declaration_strict_schema(archival_project, operation):
    _, contract, _, _ = archival_project
    artifact = contract.archival_artifacts[ARCHIVAL_PATH]
    row = {
        **asdict(artifact.identity),
        "proof_path": artifact.proof_path,
        "proof_sha256": artifact.proof_sha256,
    }
    assert types.parse_archival_artifacts({ARCHIVAL_PATH: row}) == contract.archival_artifacts
    if operation == "unknown_path":
        data = {"vendor/other.conllu": row}
    else:
        data = {ARCHIVAL_PATH: row}
        if operation == "unknown_field":
            row["free_allowance"] = True
        elif operation == "missing_field":
            row.pop("proof_sha256")
        elif operation == "size":
            row["bytes"] = 0
        elif operation == "mode":
            row["mode"] = "100755"
        else:
            row["proof_path"] = "other-proof.json"
    with pytest.raises(derivative.DerivativeError, match="archival"):
        types.parse_archival_artifacts(data)


@pytest.mark.publication_context
def test_v5_actual_archival_omission_and_before_after_drift(raw_project, tmp_path):
    root, contract = raw_project
    contract, _, response = _install_archival(root, contract, tmp_path)
    initial = derivative.capture_derivative_state(contract, root)
    assert initial.inventory[ARCHIVAL_PATH] == contract.archival_artifacts[ARCHIVAL_PATH].identity
    assert initial.evidence[str(response)] == derivative.file_hash(response)
    assert ARCHIVAL_PATH not in contract.inputs
    assert ARCHIVAL_PATH not in proof.raw_identities(root, contract)
    with pytest.raises(derivative.DerivativeError, match="untracked numerical/source/input"):
        derivative.capture_derivative_state(replace(contract, archival_artifacts={}), root)
    with response.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(derivative.DerivativeError, match="response digest"):
        derivative.capture_derivative_state(contract, root)


LEAN_PRODUCTS = (
    "formal/.lake/build/ir/AILM/Control.c",
    "formal/.lake/build/ir/AILM/Control.c.hash",
    "formal/.lake/build/ir/AILM/Control.setup.json",
    "formal/.lake/build/lib/lean/AILM/Control.ilean",
    "formal/.lake/build/lib/lean/AILM/Control.olean",
    "formal/.lake/build/lib/lean/AILM/Control.ilean.hash",
    "formal/.lake/build/lib/lean/AILM/Control.olean.hash",
    "formal/.lake/build/lib/lean/AILM/Control.trace",
)


@pytest.mark.parametrize("name", LEAN_PRODUCTS)
def test_v5_exact_lean_product_regular_counterpart(tmp_path, name):
    tracked = {"formal/AILM/Control.lean"}
    _write(tmp_path, "formal/AILM/Control.lean", b"def control : Nat := 1\n")
    product = _write(tmp_path, name, b"compiler product control\n")
    assert inventory.generated_lean_product(name, tracked)
    assert inventory.scan_untracked_implementation(tmp_path, tracked) == {}
    product.rename(product.with_suffix(product.suffix + ".retained"))
    product.symlink_to(product.with_suffix(product.suffix + ".retained"))
    with pytest.raises(derivative.DerivativeError, match="not regular|symlink"):
        inventory.scan_untracked_implementation(tmp_path, tracked)


@pytest.mark.parametrize(
    "name",
    [
        "formal/.lake/build/ir/AILM/Control.py",
        "formal/.lake/build/ir/AILM/Other.c",
        "formal/.lake/build/lib/lean/AILM/Control.lean",
        "formal/.lake/build/lib/lean/AILM/Control.py",
        "formal/.lake/build/elsewhere/AILM/Control.olean",
        "formal/.lake/build/ir/AILM/Control.olean",
        "formal/.lake/build/lib/lean/AILM/Control.c",
        "formal/.lake/build/ir/AILM/Control.c.bak",
        "formal/.lake/build/ir/AILM//Control.c",
        "formal/.lake/build/ir/AILM/../Control.c",
    ],
)
def test_v5_lean_hidden_source_unrelated_root_suffix_and_module_refused(tmp_path, name):
    tracked = {"formal/AILM/Control.lean"}
    assert inventory.generated_lean_product(LEAN_PRODUCTS[0], tracked)
    assert not inventory.generated_lean_product(name, tracked)
    if "//" in name or ".." in Path(name).parts:
        return  # Invalid logical paths cannot be materialized as distinct filesystem names.
    path = _write(tmp_path, name, b"unapproved implementation\n")
    assert path.relative_to(tmp_path).as_posix() in inventory.scan_untracked_implementation(
        tmp_path, tracked
    )


def test_v5_trusted_git_ignores_poisoned_path_config_and_loader(project, tmp_path):
    root, contract = project
    trusted = inventory.trusted_git()
    assert trusted.is_absolute() and trusted.is_file() and not trusted.is_symlink()
    assert (
        derivative.git_inventory(root, contract.measurement_commit) == contract.original_inventory
    )
    marker = tmp_path / "poison-executed"
    poison = _write(
        tmp_path, "poison/git", f"#!/bin/sh\nprintf poison > '{marker}'\nexit 91\n".encode()
    )
    poison.chmod(0o755)
    config = _write(
        tmp_path, "hostile.gitconfig", b"[core]\n fsmonitor = invalid-fsmonitor-command\n"
    )
    environment = dict(os.environ)
    contaminants = dict(
        PATH=str(poison.parent),
        GIT_CONFIG_GLOBAL=str(config),
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="core.fsmonitor",
        GIT_CONFIG_VALUE_0=str(poison),
        GIT_DIR=str(tmp_path / "wrong-git"),
        DYLD_INSERT_LIBRARIES=str(tmp_path / "missing.dylib"),
        LD_PRELOAD=str(tmp_path / "missing.so"),
    )
    environment.update(contaminants)
    baseline = subprocess.run([str(poison)], env=environment, capture_output=True, check=False)
    assert baseline.returncode == 91 and marker.read_text() == "poison"
    marker.unlink()
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            "import os, sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
            "from ailm._publication_inventory import git_inventory, git_bytes, trusted_git; "
            f"os.environ.update({contaminants!r}); "
            f"assert str(trusted_git()) == {str(trusted)!r}; "
            f"assert len(git_inventory(Path({str(root)!r}), {contract.measurement_commit!r})) == {len(contract.original_inventory)}; "
            f"assert git_bytes(Path({str(root)!r}), 'config', '--global', '--list') == b''; "
            "print('trusted-original-inventory')",
        ],
        env={
            k: v for k, v in environment.items() if k not in {"DYLD_INSERT_LIBRARIES", "LD_PRELOAD"}
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "trusted-original-inventory"
    assert not marker.exists()


@pytest.mark.parametrize(
    "operation",
    [
        "missing_module",
        "wrong_sha_literal",
        "wrong_size_literal",
        "wrong_shard",
        "duplicate_shard",
        "missing_shard",
    ],
)
@pytest.mark.publication_context
def test_v5_raw_original_source_extraction_controls(raw_project, operation):
    root, contract = raw_project
    assert set(proof.raw_identities(root, contract)) == set(RAW_PATHS)
    if operation == "missing_module":
        identities = dict(contract.original_inventory)
        identities.pop(CORPUS_SOURCES[0])
        mutated = replace(contract, original_inventory=identities)
        expected = "incomplete original raw source"
    else:
        name = (
            CORPUS_SOURCES[0]
            if operation.startswith("wrong_") and operation.endswith("literal")
            else CORPUS_SOURCES[2]
        )
        content = (root / name).read_text()
        if operation == "wrong_sha_literal":
            import ast

            module = ast.parse(content)
            literal = next(
                node.value.value
                for node in module.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "CMUDICT_RAW_SHA256" for t in node.targets
                )
            )
            content = content.replace(literal, "f" * 64)
            expected = "original raw input identity"
        elif operation == "wrong_size_literal":
            content = content.replace("3_618_488", "3_618_489")
            expected = "original raw input identity"
        elif operation == "wrong_shard":
            content = content.replace(
                'filename="cs_pdtc-ud-dev.conllu"', 'filename="unapproved.conllu"'
            )
            expected = "shard"
        elif operation == "duplicate_shard":
            content = content.replace(
                'filename="cs_pdtc-ud-dev.conllu"', 'filename="cs_pdtc-ud-train-ft.conllu"'
            )
            expected = "duplicate.*shard"
        else:
            content = content.replace("UDCzechShard(", "UnrecognizedShard(")
            expected = "incomplete.*shard|shard.*missing"
        _write(root, name, content.encode())
        _git(root, "add", name)
        commit, tree = _pin_index(root)
        mutated = replace(
            contract,
            measurement_commit=commit,
            measurement_tree=tree,
            original_inventory=derivative.git_inventory(root, commit),
        )
    with pytest.raises(derivative.DerivativeError, match=expected):
        proof.raw_identities(root, mutated)


def test_v5_missing_archival_schema_field_refused(project, tmp_path):
    root, contract = project
    declaration = _declaration(root, contract)
    derivative.load_candidate_contract(declaration, root, tmp_path)
    data = json.loads(declaration.read_bytes())
    data.pop("archival_artifacts")
    declaration.write_text(json.dumps(data))
    with pytest.raises(derivative.DerivativeError, match="missing/unknown"):
        derivative.load_candidate_contract(declaration, root, tmp_path)


@pytest.mark.publication_context
def test_v5_archival_blob_framing_is_independent_of_declared_api(archival_project):
    root, contract, retained, response = archival_project
    forged_blob = "f" * 40
    retained["git_blob_sha1"] = forged_blob
    retained["upstream_entry"]["sha"] = forged_blob
    retained["upstream_entry"]["url"] = (
        "https://api.github.com/repos/UniversalDependencies/UD_Czech-PDTC/git/blobs/" + forged_blob
    )
    data = json.loads(response.read_bytes())
    entry = next(x for x in data["tree"] if x["path"] == Path(ARCHIVAL_PATH).name)
    entry.update(retained["upstream_entry"])
    response.write_text(json.dumps(data))
    retained["tree_response_sha256"] = derivative.file_hash(response)
    _write(root, ARCHIVAL_PROOF, json.dumps(retained).encode())
    artifact = replace(
        contract.archival_artifacts[ARCHIVAL_PATH],
        proof_sha256=derivative.file_hash(root / ARCHIVAL_PROOF),
    )
    with pytest.raises(derivative.DerivativeError, match="Git blob bytes mismatch"):
        proof.archival_identity_evidence(
            root, replace(contract, archival_artifacts={ARCHIVAL_PATH: artifact})
        )


@pytest.mark.publication_context
def test_v5_archival_and_raw_files_never_become_semantic_overlays(raw_project, tmp_path):
    root, contract = raw_project
    contract, _, _ = _install_archival(root, contract, tmp_path)
    derivative.capture_derivative_state(contract, root)
    destination = tmp_path / "historical-semantic-source"
    materialized = _execution_module().materialize(contract, root, destination)
    assert not any((destination / name).exists() for name in (*RAW_PATHS, ARCHIVAL_PATH))
    assert set(materialized["pre_overlay"]) == set(contract.original_inventory)


@pytest.mark.parametrize("operation", ["executable", "ancestor_symlink"])
def test_v5_lean_product_mode_and_ancestor_controls(tmp_path, operation):
    tracked = {"formal/AILM/Control.lean"}
    name = LEAN_PRODUCTS[0]
    path = _write(tmp_path, name, b"compiler product control\n")
    assert inventory.scan_untracked_implementation(tmp_path, tracked) == {}
    if operation == "executable":
        path.chmod(0o755)
        expected = "Lean product executable"
    else:
        module = path.parent
        moved = tmp_path / "retained-module"
        module.rename(moved)
        module.symlink_to(moved, target_is_directory=True)
        expected = "directory symlink"
    with pytest.raises(derivative.DerivativeError, match=expected):
        inventory.scan_untracked_implementation(tmp_path, tracked)


@pytest.mark.parametrize(
    "operation", ["missing_response", "relative_response", "duplicate_json", "ancestor_symlink"]
)
@pytest.mark.publication_context
def test_v5_archival_missing_duplicate_and_escape_controls(archival_project, tmp_path, operation):
    root, contract, retained, response = archival_project
    if operation == "missing_response":
        response.rename(response.with_suffix(".retained"))
        with pytest.raises(FileNotFoundError, match="retained-api-response"):
            proof.archival_identity_evidence(root, contract)
        return
    if operation == "ancestor_symlink":
        folder = (root / ARCHIVAL_PATH).parent
        moved = tmp_path / "retained-archival-directory"
        folder.rename(moved)
        folder.symlink_to(moved, target_is_directory=True)
        with pytest.raises(derivative.DerivativeError, match="symlink"):
            proof.archival_identity_evidence(root, contract)
        return
    if operation == "relative_response":
        retained["tree_response_path"] = "relative-api-response.json"
        content = json.dumps(retained).encode()
        expected = "response digest"
    else:
        content = (
            json.dumps(retained)
            .replace('"schema":', '"schema": "duplicate", "schema":', 1)
            .encode()
        )
        expected = "duplicate JSON field"
    _write(root, ARCHIVAL_PROOF, content)
    artifact = replace(
        contract.archival_artifacts[ARCHIVAL_PATH],
        proof_sha256=derivative.file_hash(root / ARCHIVAL_PROOF),
    )
    with pytest.raises(derivative.DerivativeError, match=expected):
        proof.archival_identity_evidence(
            root, replace(contract, archival_artifacts={ARCHIVAL_PATH: artifact})
        )


@pytest.mark.parametrize(
    "base", ["src/ailm", "scripts", "vendor", "gnn", "data", "tests", "formal"]
)
@pytest.mark.parametrize(
    "excluded",
    [
        "refs",
        "scratchpad",
        ".venv",
        ".git",
        "cache",
        "tooling",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    ],
)
def test_r1_ignored_nested_directory_addition_refused(project, base, excluded):
    root, contract = project
    baseline = derivative.capture_derivative_state(contract, root)
    name = f"{base}/{excluded}/hidden_control.py"
    _git(
        root,
        "config",
        "core.excludesFile",
        str(_write(root, "evidence/ignore-r1", (excluded + "/\n").encode())),
    )
    assert derivative.capture_derivative_state(contract, root) == baseline
    path = _write(root, name, b"VALUE = 0.75\n")
    assert _git(root, "check-ignore", "--no-index", name).decode().strip() == name
    assert path.is_file()
    with pytest.raises(derivative.DerivativeError, match="untracked numerical/source/input"):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize(
    "name",
    [
        "refs",
        "scratchpad",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "src/ailm/refs",
        "scripts/.venv",
        "vendor/.git",
        "gnn/cache",
        "data/tooling",
        "tests/__pycache__",
        "formal/scratchpad",
    ],
)
@pytest.mark.parametrize("kind", ["directory", "file", "dangling"])
def test_r1_excluded_name_symlink_refused(project, tmp_path, name, kind):
    root, contract = project
    derivative.capture_derivative_state(contract, root)
    target = tmp_path / "outside"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_bytes(b"VALUE = 0.75\n")
    link = root / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(derivative.DerivativeError, match="directory symlink|symlink refused"):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize(
    "operation",
    [
        "unchanged",
        "source",
        "input",
        "executable",
        "symlink",
        "directory_symlink",
        "unknown_pyc",
        "bad_header",
        "different_code",
        "trailing",
    ],
)
def test_r1_generated_bytecode_exact_positive_and_refusals(project, tmp_path, operation):
    root, contract = project
    baseline = derivative.capture_derivative_state(contract, root)
    source = root / "src/ailm/model.py"
    product = Path(py_compile.compile(str(source), doraise=True))
    assert derivative.capture_derivative_state(contract, root) == baseline
    if operation == "unchanged":
        return
    expected = "untracked numerical/source/input"
    if operation in {"source", "input", "executable", "unknown_pyc"}:
        name = {
            "source": "hidden.py",
            "input": "model.json",
            "executable": "tool",
            "unknown_pyc": "unbound.pyc",
        }[operation]
        path = _write(root, "src/ailm/__pycache__/" + name, b"unapproved input\n")
        if operation == "executable":
            path.chmod(0o755)
    elif operation == "symlink":
        (product.parent / "hidden.json").symlink_to(source)
        expected = "symlink refused"
    elif operation == "directory_symlink":
        (product.parent / "refs").symlink_to(tmp_path, target_is_directory=True)
        expected = "directory symlink"
    else:
        expected = "generated bytecode"
        if operation == "bad_header":
            product.write_bytes(b"invalid bytecode")
        elif operation == "trailing":
            product.write_bytes(product.read_bytes() + b"input payload")
        else:
            other = _write(tmp_path, "other.py", b"VALUE = 0.75\n")
            py_compile.compile(str(other), cfile=str(product), doraise=True)
    with pytest.raises(derivative.DerivativeError, match=expected):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize("name", sorted(types.SCAN_EXCLUDED - {".git"}))
def test_r1_exact_root_environment_directories_remain_excluded(project, name):
    root, contract = project
    baseline = derivative.capture_derivative_state(contract, root)
    _write(root, name + "/environment.py", b"local tool environment\n")
    assert derivative.capture_derivative_state(contract, root) == baseline


@pytest.mark.parametrize("kind", ["directory", "file", "dangling"])
def test_r1_root_git_symlink_checked_before_pruning(project, tmp_path, kind):
    root, _ = project
    tracked = {p.decode() for p in inventory.git_bytes(root, "ls-files", "-z").split(b"\0") if p}
    baseline = inventory.scan_untracked_implementation(root, tracked)
    metadata = root / ".git"
    metadata.rename(tmp_path / "retained-git")
    target = tmp_path / "target"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_bytes(b"control\n")
    assert inventory.scan_untracked_implementation(root, tracked) == baseline
    metadata.symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(derivative.DerivativeError, match="directory symlink|symlink refused"):
        inventory.scan_untracked_implementation(root, tracked)


@pytest.mark.parametrize("mutation", ["bytes", "mode", "deleted", "extra"])
def test_exact_archived_demo_hook_identity_refuses_drift(project, mutation):
    root, contract = project
    relative = ".omo/evidence/reference/.git/hooks/pre-commit.sample"
    hook = _write(root, relative, b"#!/bin/sh\nexit 0\n")
    hook.chmod(0o755)
    bound = replace(contract, baseline_untracked={relative: derivative.file_identity(hook)})
    derivative.capture_derivative_state(bound, root)
    if mutation == "bytes":
        hook.write_bytes(b"#!/bin/sh\nexit 1\n")
    elif mutation == "mode":
        hook.chmod(0o644)
    elif mutation == "deleted":
        hook.unlink()
    else:
        extra = _write(root, ".omo/evidence/reference/.git/hooks/additional.sample", b"exit 0\n")
        extra.chmod(0o755)
    with pytest.raises(derivative.DerivativeError, match="untracked implementation drift"):
        derivative.capture_derivative_state(bound, root)


@pytest.mark.parametrize("name", ["case_word_sweep", "incontext_sweep", "results"])
@pytest.mark.parametrize("mutation", ["bytes", "mode", "deleted", "extra", "renamed", "symlink"])
@pytest.mark.publication_context
def test_exact_original_worker_bytecode_remains_archival_and_refuses_drift(
    project, tmp_path, name, mutation
):
    root, contract = project
    folder = ".omo/evidence/figure-display-fixes/original-source/"
    relative = folder + "__pycache__/" + name + ".cpython-312.pyc"
    source_relative = folder + name + ".py"
    source = _write(root, source_relative, (ROOT / source_relative).read_bytes())
    archive = _write(root, relative, (ROOT / relative).read_bytes())
    source.chmod(0o644)
    archive.chmod(0o644)
    baseline = {p: derivative.file_identity(root / p) for p in (relative, source_relative)}
    bound = replace(contract, baseline_untracked=baseline)
    derivative.capture_derivative_state(bound, root)
    assert not inventory.generated_bytecode_product(root, archive, set(contract.original_inventory))
    if mutation == "bytes":
        archive.write_bytes(archive.read_bytes() + b"changed archive")
    elif mutation == "mode":
        archive.chmod(0o755)
    elif mutation == "deleted":
        archive.unlink()
    elif mutation == "renamed":
        archive.rename(archive.with_name("renamed.cpython-312.pyc"))
    elif mutation == "symlink":
        retained = tmp_path / "retained-worker-bytecode"
        archive.rename(retained)
        archive.symlink_to(retained)
    else:
        _write(root, folder + "__pycache__/extra.cpython-312.pyc", b"unexpected bytecode")
    with pytest.raises(derivative.DerivativeError, match="untracked implementation drift|symlink"):
        derivative.capture_derivative_state(bound, root)


@pytest.mark.parametrize("name", derivative.SUPPORT_PATHS)
def test_each_review_bound_support_file_mutation_is_refused(project, name):
    """Interface/metadata admission is exact reviewed bytes, never a directory exemption."""
    root, contract = project
    if name == "data/publication_derivative.json":
        declaration = _declaration(root, contract)
        contract = replace(
            contract, support={**contract.support, name: derivative.file_hash(declaration)}
        )
    derivative.capture_derivative_state(contract, root)
    path = root / name
    path.write_bytes(path.read_bytes() + b"unauthorized support mutation\n")
    refusal = (
        "package configuration is invalid" if name == "pyproject.toml" else "support bytes changed"
    )
    with pytest.raises(derivative.DerivativeError, match=refusal):
        derivative.capture_derivative_state(contract, root)


@pytest.fixture
def reviewed_dashboard_project(project, tmp_path: Path) -> tuple[Path, Path]:
    root, candidate = project
    name = "output/figures/dashboard_observation.png"
    _write(root, name, (ROOT / name).read_bytes())
    declaration = _declaration(root, candidate)
    support = {path: derivative.file_hash(root / path) for path in derivative.SUPPORT_PATHS}
    _git(root, "add", *derivative.SUPPORT_PATHS, *derivative.PUBLICATION_PATHS)
    commit, tree = _pin_index(root)
    review = _write(tmp_path, "review.md", b"Finite fixture source review.\n")
    fields = {
        "schema": derivative.CONTRACT_SCHEMA,
        "declaration_path": str(declaration.relative_to(root)),
        "declaration_sha256": derivative.file_hash(declaration),
        "publication_commit": commit,
        "publication_tree": tree,
        "support": support,
        "review_path": str(review),
        "review_sha256": derivative.file_hash(review),
        "snapshot_parent": str(tmp_path),
    }
    return root, _write(tmp_path, "reviewed-contract.json", json.dumps(fields).encode())


def test_reviewed_dashboard_figure_remains_output_when_admitted(reviewed_dashboard_project):
    root, path = reviewed_dashboard_project
    contract = derivative.load_contract(path, root)

    state = derivative.capture_derivative_state(contract, root)

    name = "output/figures/dashboard_observation.png"
    assert name not in state.inventory
    assert state.outputs[name].sha256 == contract.support[name]
    assert state.outputs[name].sha256 == derivative.file_hash(ROOT / name)


@pytest.mark.parametrize("missing", [False, True])
def test_dashboard_admission_refuses_wrong_or_absent_reviewed_support(
    reviewed_dashboard_project, missing: bool
):
    root, path = reviewed_dashboard_project
    fields = json.loads(path.read_text())
    name = "output/figures/dashboard_observation.png"
    if missing:
        fields["support"].pop(name)
    else:
        fields["support"][name] = "f" * 64
    path.write_text(json.dumps(fields))

    with pytest.raises(
        derivative.DerivativeError,
        match="review must bind every finite support path|support not in reviewed source freeze",
    ):
        derivative.load_contract(path, root)


def test_dashboard_admission_does_not_allow_an_unregistered_tracked_neighbor(
    reviewed_dashboard_project,
):
    root, path = reviewed_dashboard_project
    contract = derivative.load_contract(path, root)
    name = "output/figures/dashboard_observation_unregistered.png"
    _write(root, name, (ROOT / "output/figures/dashboard_observation.png").read_bytes())
    _git(root, "add", name)

    with pytest.raises(derivative.DerivativeError, match="unexpected tracked paths"):
        derivative.capture_derivative_state(contract, root)


@pytest.mark.parametrize(
    "mutation",
    [b"BUDGET = 51\n", b"REQUEST = 'changed'\n", b"CACHE = 'different'\n", b"run_acquisition()\n"],
)
def test_warming_operational_mutation_refused_even_with_rebound_hash(project, mutation):
    root, contract = project
    name = "scripts/warm_call_cache.py"
    derivative.capture_derivative_state(contract, root)
    path = root / name
    path.write_bytes(path.read_bytes() + mutation)
    delta = {p: dict(row) for p, row in contract.publication_delta.items()}
    delta[name]["current"] = derivative.file_hash(path)
    with pytest.raises(derivative.DerivativeError, match="nonstyling figure delta"):
        derivative.capture_derivative_state(replace(contract, publication_delta=delta), root)


@pytest.mark.parametrize("boundary", ["header", "renderer"])
@pytest.mark.parametrize("operation", ["missing", "duplicate"])
def test_warming_help_boundary_requires_one_original_git_literal(project, boundary, operation):
    root, contract = project
    name = "scripts/warm_call_cache.py"
    original = derivative.git_bytes(root, "show", contract.measurement_commit + ":" + name)
    literal = (
        b'        "\\nWith the store warm, the rest of the rebuild is:\\n"\n'
        if boundary == "header"
        else b'        "  .venv/bin/python scripts/z_render_docs.py\\n"\n'
    )
    assert original.count(literal) == 1
    altered = original.replace(literal, b"" if operation == "missing" else literal + literal)
    with pytest.raises(derivative.DerivativeError, match="boundary ambiguous"):
        derivative.transformed_figure_source(name, altered)


def test_actual_warming_finite_transform_is_formatter_clean_and_runtime_identical(tmp_path):
    """Authenticate actual Git bytes, not a smaller print-only stand-in."""
    name = "scripts/warm_call_cache.py"
    original = derivative.publication_source_bytes(
        ROOT, "9f8842d93c9d3c463589fb6fae45087e7b2048a5", name
    )
    before_format = derivative.publication_source_bytes(
        ROOT, "6b1f24c7300a24d50e82a9b3b8a92f5c7d1bca8e", name
    )
    current = (ROOT / name).read_bytes()
    assert (
        derivative.digest(original)
        == "e52a9a189788b0d7aa02318ab92c46fa0557d3bdef2e40edf8e4e482857e7a24"
    )
    assert derivative.transformed_figure_source(name, original) == current
    assert ast.dump(ast.parse(before_format)) == ast.dump(ast.parse(current))
    original_tree, current_tree = ast.parse(original), ast.parse(current)
    original_tree.body = [
        n
        for n in original_tree.body
        if not (isinstance(n, ast.FunctionDef) and n.name == "next_steps")
    ]
    current_tree.body = [
        n
        for n in current_tree.body
        if not (isinstance(n, ast.FunctionDef) and n.name == "next_steps")
    ]
    assert ast.dump(original_tree) == ast.dump(current_tree)
    for label, content, expected in [
        ("current", current, 0),
        ("old-quote-negative", before_format, 1),
    ]:
        path = _write(tmp_path, label + ".py", content)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "ruff",
                "format",
                "--check",
                "--diff",
                "--config",
                str(ROOT / "pyproject.toml"),
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        if expected:
            assert "TEMPLATE_REPO_ROOT" in result.stdout and "would be reformatted" in result.stderr
