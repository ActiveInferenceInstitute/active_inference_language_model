"""Atomic publication contract shared by deterministic artifact writers."""

from __future__ import annotations

import ast
import json
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from scripts._artifact_io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    open_authenticated_regular_file,
    read_strict_json_regular_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATORS = (
    "generate_case_counts.py",
    "generate_clause_counts.py",
    "generate_narrative_counts.py",
    "generate_phoneme_counts.py",
    "generate_synthetic_counts.py",
)

STABLE_PUBLISHERS = {
    "audit_gates.py": "atomic_write_json(",
    "build_web_ailm.py": "atomic_write_text(",
    "export_web_model.py": "atomic_write_text(",
    "generate_lean_ledger.py": "atomic_write_text(",
    "generate_pymdp_sparse_patch.py": "atomic_write_text(",
    "record_external_benchmark_claims.py": "atomic_write_text(",
    "z_render_docs.py": "atomic_write_text(",
}


def _direct_write_calls(source: str) -> list[str]:
    """Locate common direct publication APIs in a script's syntax tree."""
    calls: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"write_text", "write_bytes"}:
            calls.append(f"{node.func.attr}:{node.lineno}")
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode_node = node.args[1] if len(node.args) >= 2 else None
            for keyword in node.keywords:
                if keyword.arg == "mode":
                    mode_node = keyword.value
            call_kind = "open"
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "open":
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id == "os":
                flags_node = node.args[1] if len(node.args) >= 2 else None
                write_flag_names = {
                    "O_WRONLY",
                    "O_RDWR",
                    "O_CREAT",
                    "O_TRUNC",
                    "O_APPEND",
                }
                names = (
                    {
                        child.attr
                        for child in ast.walk(flags_node)
                        if isinstance(child, ast.Attribute)
                    }
                    if flags_node is not None
                    else set()
                )
                dynamic = flags_node is None or any(
                    isinstance(child, ast.Name) and child.id != "os"
                    for child in ast.walk(flags_node)
                )
                if names & write_flag_names:
                    calls.append(f"os.open-write:{node.lineno}")
                elif dynamic:
                    calls.append(f"os.open-dynamic:{node.lineno}")
                continue
            mode_node = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg == "mode":
                    mode_node = keyword.value
            call_kind = "path.open"
        else:
            continue

        if mode_node is None:
            continue
        if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
            if any(marker in mode_node.value for marker in "wax+"):
                calls.append(f"{call_kind}-{mode_node.value}:{node.lineno}")
        else:
            calls.append(f"{call_kind}-dynamic:{node.lineno}")
    return calls


def _reachable_calls(source: str) -> list[ast.Call]:
    """Return calls reachable from module execution through local functions."""
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def expression_calls(node: ast.AST | None) -> list[ast.Call]:
        if node is None:
            return []
        return [child for child in ast.walk(node) if isinstance(child, ast.Call)]

    def block_calls(statements: list[ast.stmt]) -> list[ast.Call]:
        found: list[ast.Call] = []
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.If):
                found.extend(expression_calls(statement.test))
                if isinstance(statement.test, ast.Constant):
                    branch = statement.body if bool(statement.test.value) else statement.orelse
                    found.extend(block_calls(branch))
                else:
                    found.extend(block_calls(statement.body))
                    found.extend(block_calls(statement.orelse))
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                found.extend(expression_calls(statement.iter))
                found.extend(block_calls(statement.body))
                found.extend(block_calls(statement.orelse))
            elif isinstance(statement, ast.While):
                found.extend(expression_calls(statement.test))
                if not (
                    isinstance(statement.test, ast.Constant) and not bool(statement.test.value)
                ):
                    found.extend(block_calls(statement.body))
                found.extend(block_calls(statement.orelse))
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                for item in statement.items:
                    found.extend(expression_calls(item.context_expr))
                    found.extend(expression_calls(item.optional_vars))
                found.extend(block_calls(statement.body))
            elif isinstance(statement, ast.Try):
                found.extend(block_calls(statement.body))
                for handler in statement.handlers:
                    found.extend(expression_calls(handler.type))
                    found.extend(block_calls(handler.body))
                found.extend(block_calls(statement.orelse))
                found.extend(block_calls(statement.finalbody))
            elif isinstance(statement, ast.Match):
                found.extend(expression_calls(statement.subject))
                for case in statement.cases:
                    found.extend(expression_calls(case.guard))
                    found.extend(block_calls(case.body))
            else:
                found.extend(expression_calls(statement))
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                break
        return found

    reachable = block_calls(tree.body)
    scanned_functions: set[str] = set()
    index = 0
    while index < len(reachable):
        call = reachable[index]
        index += 1
        if not isinstance(call.func, ast.Name):
            continue
        function_name = call.func.id
        if function_name in functions and function_name not in scanned_functions:
            scanned_functions.add(function_name)
            reachable.extend(block_calls(functions[function_name].body))
    return reachable


def _non_atomic_writer_defects(source: str, primitive: str = "atomic_write_json") -> list[str]:
    defects: list[str] = []
    calls = [
        node
        for node in _reachable_calls(source)
        if (
            (isinstance(node.func, ast.Name) and node.func.id == primitive)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == primitive)
        )
    ]
    if not calls:
        defects.append(f"missing reachable {primitive}")
    defects.extend(f"direct artifact write {call}" for call in _direct_write_calls(source))
    return defects


def test_all_canonical_json_generators_publish_through_the_atomic_writer() -> None:
    for name in GENERATORS:
        source = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert _non_atomic_writer_defects(source) == [], name

    # NEGATIVE CONTROL: the historical direct writer must be detected.
    planted = "path.write_text(json.dumps(payload))"
    assert _non_atomic_writer_defects(planted) == [
        "missing reachable atomic_write_json",
        "direct artifact write write_text:1",
    ]


def test_stable_publication_surfaces_use_an_atomic_publication_primitive() -> None:
    for name, primitive in STABLE_PUBLISHERS.items():
        source = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert _non_atomic_writer_defects(source, primitive.removesuffix("(")) == [], name

    planted = (
        "if False:\n    atomic_write_text(target, rendered)\nopen(target, 'w').write('partial')\n"
    )
    assert _non_atomic_writer_defects(planted, "atomic_write_text") == [
        "missing reachable atomic_write_text",
        "direct artifact write open-w:3",
    ]

    uncalled_decoy = (
        "def decoy():\n    atomic_write_text(target, rendered)\n"
        "def main():\n    target.open('w').write('partial')\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    assert _non_atomic_writer_defects(uncalled_decoy, "atomic_write_text") == [
        "missing reachable atomic_write_text",
        "direct artifact write path.open-w:4",
    ]

    variable_mode = (
        "def main(mode):\n"
        "    atomic_write_text(other, rendered)\n"
        "    open(target, mode).write('partial')\n"
        "if __name__ == '__main__':\n    main(mode)\n"
    )
    assert _non_atomic_writer_defects(variable_mode, "atomic_write_text") == [
        "direct artifact write open-dynamic:3"
    ]


def test_canonical_generator_reaches_the_atomic_writer_in_a_real_process(tmp_path) -> None:
    """Static reachability is backed by one real canonical publication path."""
    target = tmp_path / "initially-absent" / "nested" / "synthetic.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_synthetic_counts.py"),
            "--output",
            str(target),
            "--chars",
            "1000",
            "--seed",
            "7",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload and sum(payload.values()) == 999
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_serialization_failure_preserves_the_prior_artifact_and_leaves_no_scratch(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    sentinel = b'{"prior":"valid"}\n'
    target.write_bytes(sentinel)

    with pytest.raises(TypeError):
        atomic_write_json(target, {"not_json": object()}, indent=2)

    assert target.read_bytes() == sentinel
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_writer_emits_the_declared_serialization_exactly(tmp_path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    payload = {"b": [2, 1], "a": "value"}
    content = atomic_write_json(target, payload, sort_keys=True, separators=(",", ":"))
    assert content == b'{"a":"value","b":[2,1]}\n'
    assert target.read_bytes() == content
    assert json.loads(content) == payload
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_byte_and_text_writers_publish_exact_content_and_public_mode(tmp_path) -> None:
    binary = tmp_path / "new" / "artifact.bin"
    text = tmp_path / "new" / "artifact.txt"

    assert atomic_write_bytes(binary, b"\x00exact\xff") == b"\x00exact\xff"
    assert (
        atomic_write_text(text, "exact \N{GREEK SMALL LETTER ALPHA}\n")
        == "exact \N{GREEK SMALL LETTER ALPHA}\n".encode()
    )

    assert binary.read_bytes() == b"\x00exact\xff"
    assert text.read_text(encoding="utf-8") == "exact \N{GREEK SMALL LETTER ALPHA}\n"
    assert stat.S_IMODE(binary.stat().st_mode) == 0o644
    assert stat.S_IMODE(text.stat().st_mode) == 0o644


def test_authenticated_json_bytes_cannot_be_substituted_after_descriptor_open(tmp_path) -> None:
    """The bytes parsed and hashed stay coupled to the authenticated descriptor."""
    source = tmp_path / "receipt.json"
    opened_name = tmp_path / "opened-receipt.json"
    replacement = tmp_path / "replacement.json"
    source.write_text('{"which":"authenticated"}\n', encoding="utf-8")
    replacement.write_text('{"which":"substituted"}\n', encoding="utf-8")

    with open_authenticated_regular_file(source, description="test receipt") as handle:
        source.rename(opened_name)
        source.symlink_to(replacement)
        authenticated_content = handle.read()

    assert json.loads(authenticated_content) == {"which": "authenticated"}
    assert source.read_text(encoding="utf-8") == '{"which":"substituted"}\n'

    # The high-level strict reader goes through the same non-following open and
    # therefore refuses the replacement path rather than parsing different bytes.
    with pytest.raises(ValueError, match="regular non-symlink file"):
        read_strict_json_regular_file(source, description="test receipt")


def test_atomic_writer_preserves_existing_regular_file_mode(tmp_path) -> None:
    target = tmp_path / "private.json"
    target.write_text("prior", encoding="utf-8")
    target.chmod(0o600)

    atomic_write_json(target, {"replacement": True})

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_atomic_writer_refuses_a_symlink_publication_directory(tmp_path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        atomic_write_bytes(linked / "artifact.bin", b"must not escape")

    assert not (real / "artifact.bin").exists()


def test_atomic_writer_names_the_symlinked_ancestor(tmp_path) -> None:
    """The refusal must say which component is the link and what to pass instead.

    The bare kernel refusal was ``NotADirectoryError('tmp')``: a caller publishing
    under macOS ``/tmp`` or ``/var`` was told nothing about the symlink or about
    ``os.path.realpath``. The error stays an ``OSError`` so existing handlers keep
    catching it, and nothing is created beneath the link's real target.
    """
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError) as raised:
        atomic_write_bytes(linked / "nested" / "artifact.bin", b"must not escape")

    message = str(raised.value)
    assert isinstance(raised.value, NotADirectoryError)
    assert "'linked'" in message, message
    assert "os.path.realpath()" in message, message
    assert not (real / "nested").exists()
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["linked", "real"]


def test_parent_path_swap_cannot_redirect_or_substitute_publication(tmp_path) -> None:
    """Real race control for the directory-FD anchoring contract."""
    publication = tmp_path / "publication"
    publication.mkdir()
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    moved_publication = tmp_path / "anchored-publication"
    content = b"I" * (64 * 1024 * 1024)
    outcome: list[BaseException | bytes] = []

    def write() -> None:
        try:
            outcome.append(atomic_write_bytes(publication / "artifact.bin", content))
        except BaseException as error:  # surfaced below with its original type
            outcome.append(error)

    thread = threading.Thread(target=write)
    thread.start()
    deadline = time.monotonic() + 10
    scratches: list[Path] = []
    while time.monotonic() < deadline:
        scratches = list(publication.glob(".artifact.bin.*.tmp"))
        if scratches:
            break
        time.sleep(0.001)
    assert scratches, "writer never exposed its real scratch; the race control was not exercised"

    publication.rename(moved_publication)
    publication.symlink_to(attacker, target_is_directory=True)
    # Plant the historical attack shape: the same basename under the pathname's
    # new referent. A pathname-based replace would publish these bytes.
    (attacker / scratches[0].name).write_bytes(b"ATTACKER")
    thread.join(timeout=20)

    assert not thread.is_alive()
    assert outcome == [content]
    assert (moved_publication / "artifact.bin").read_bytes() == content
    assert not (attacker / "artifact.bin").exists()
    assert (attacker / scratches[0].name).read_bytes() == b"ATTACKER"
    assert list(moved_publication.glob(".artifact.bin.*.tmp")) == []


def test_absent_nested_parent_swap_cannot_redirect_publication(tmp_path) -> None:
    """Real race control for creation before the final parent exists.

    Waiting for the first descendant proves the writer has opened the new root
    and still has many descriptor-relative components left to create.  The old
    ``mkdir(parents=True); open(path)`` implementation follows the replacement
    symlink during that remainder and writes into ``attacker``.
    """
    publication = tmp_path / "publication"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    moved_publication = tmp_path / "anchored-publication"
    components = tuple(f"d{index:03d}" for index in range(120))
    relative_parent = Path(*components)
    target = publication / relative_parent / "artifact.bin"
    content = b"descriptor-anchored absent-parent publication\n"
    outcome: list[BaseException | bytes] = []

    def write() -> None:
        try:
            outcome.append(atomic_write_bytes(target, content))
        except BaseException as error:  # surfaced below with its original type
            outcome.append(error)

    thread = threading.Thread(target=write)
    thread.start()
    deadline = time.monotonic() + 10
    first_descendant = publication / components[0]
    while time.monotonic() < deadline and not first_descendant.is_dir():
        time.sleep(0.0001)
    assert first_descendant.is_dir(), (
        "writer did not expose the newly created ancestor; the race was not exercised"
    )

    publication.rename(moved_publication)
    publication.symlink_to(attacker, target_is_directory=True)
    thread.join(timeout=20)

    assert not thread.is_alive()
    assert outcome == [content]
    assert (moved_publication / relative_parent / "artifact.bin").read_bytes() == content
    assert not (attacker / relative_parent / "artifact.bin").exists()
    assert list((moved_publication / relative_parent).glob(".artifact.bin.*.tmp")) == []


@pytest.mark.parametrize(
    ("writer", "content"),
    ((atomic_write_bytes, "not bytes"), (atomic_write_text, b"not text")),
)
def test_atomic_primitive_rejects_the_wrong_content_type_without_side_effects(
    tmp_path, writer, content
) -> None:
    target = tmp_path / "absent" / "artifact"
    with pytest.raises(TypeError):
        writer(target, content)
    assert not target.parent.exists()
