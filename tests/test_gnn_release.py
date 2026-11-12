"""Real Git and filesystem controls for upstream release selection and custody."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from ailm.verification.upstream_gnn import (
    UpstreamRelease,
    UpstreamReleaseError,
    default_upstream_root,
    load_importer_pin,
    selected_upstream_root,
    upstream_candidates,
    verify_release_checkout,
)


def _git(root: Path, *args: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(
        ["git", "-C", str(root), *args],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


@pytest.fixture
def release(tmp_path: Path) -> tuple[Path, UpstreamRelease]:
    root = tmp_path / "upstream"
    parser = root / "src/gnn/parsers/markdown_parser.py"
    parser.parent.mkdir(parents=True)
    parser.write_text("VALUE = 'released'\n")
    (root / "src/gnn/__init__.py").write_text("")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "AILM release control")
    _git(root, "config", "user.email", "release@example.invalid")
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "release")
    _git(root, "tag", "v1.2.3")
    return root, UpstreamRelease(
        "https://github.com/example/upstream", "v1.2.3", _git(root, "rev-parse", "HEAD")
    )


def test_matching_release_source_produces_a_stable_identity(release):
    root, pin = release
    identity = verify_release_checkout(root, pin)
    assert identity["commit"] == pin.commit
    assert identity["tag"] == pin.tag
    assert identity["source_files"] == 2
    assert len(str(identity["source_sha256"])) == 64
    assert verify_release_checkout(root, pin) == identity


def test_newer_head_is_refused_even_with_the_release_tag_present(release):
    root, pin = release
    _git(root, "commit", "--allow-empty", "-qm", "development")
    with pytest.raises(UpstreamReleaseError, match="HEAD .* differs"):
        verify_release_checkout(root, pin)


def test_tag_pointing_elsewhere_is_refused(release):
    root, pin = release
    _git(root, "commit", "--allow-empty", "-qm", "development")
    altered = replace(pin, commit=_git(root, "rev-parse", "HEAD"))
    with pytest.raises(UpstreamReleaseError, match="tag .* differs"):
        verify_release_checkout(root, altered)


@pytest.mark.parametrize("flag", [None, "--assume-unchanged", "--skip-worktree"])
def test_modified_source_is_refused_despite_index_flags(release, flag):
    root, pin = release
    relative = "src/gnn/parsers/markdown_parser.py"
    if flag is not None:
        _git(root, "update-index", flag, relative)
    (root / relative).write_text("VALUE = 'modified'\n")
    with pytest.raises(UpstreamReleaseError, match="modified upstream source"):
        verify_release_checkout(root, pin)


def test_git_replacement_objects_cannot_redefine_the_pinned_source(release):
    root, pin = release
    parser = root / "src/gnn/parsers/markdown_parser.py"
    parser.write_text("VALUE = 'replacement'\n")
    _git(root, "add", "src")
    _git(root, "commit", "-qm", "replacement")
    replacement = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "HEAD", pin.commit)
    _git(root, "replace", pin.commit, replacement)
    with pytest.raises(UpstreamReleaseError, match="modified upstream source"):
        verify_release_checkout(root, pin)


@pytest.mark.parametrize("kind", ["missing", "extra", "extra-bytecode", "symlink", "fifo"])
def test_changed_source_inventory_is_refused(release, tmp_path, kind):
    root, pin = release
    parser = root / "src/gnn/parsers/markdown_parser.py"
    if kind == "extra":
        (parser.parent / "injected.py").write_text("VALUE = 'injected'\n")
    elif kind == "extra-bytecode":
        (parser.parent / "injected.pyc").write_bytes(b"unversioned sourceless module")
    else:
        content = parser.read_bytes()
        parser.unlink()
        if kind == "symlink":
            target = tmp_path / "outside.py"
            target.write_bytes(content)
            parser.symlink_to(target)
        elif kind == "fifo":
            os.mkfifo(parser)
    with pytest.raises(UpstreamReleaseError):
        verify_release_checkout(root, pin)


def test_unversioned_bytecode_cache_does_not_change_source_identity(release):
    root, pin = release
    before = verify_release_checkout(root, pin)
    cache = root / "src/gnn/__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"cache")
    assert verify_release_checkout(root, pin) == before


def test_symlinked_source_directory_is_refused(release, tmp_path):
    root, pin = release
    source = root / "src"
    moved = tmp_path / "moved-source"
    source.rename(moved)
    source.symlink_to(moved, target_is_directory=True)
    with pytest.raises(UpstreamReleaseError, match="symlinked upstream source"):
        verify_release_checkout(root, pin)


def test_missing_checkout_is_a_refusal(release, tmp_path):
    _, pin = release
    with pytest.raises(UpstreamReleaseError, match="cannot inspect"):
        verify_release_checkout(tmp_path / "absent", pin)


def test_release_without_parser_is_refused(release):
    root, pin = release
    _git(root, "rm", "src/gnn/parsers/markdown_parser.py")
    _git(root, "commit", "-qm", "remove parser")
    _git(root, "tag", "v1.2.4")
    altered = replace(pin, tag="v1.2.4", commit=_git(root, "rev-parse", "HEAD"))
    with pytest.raises(UpstreamReleaseError, match="has no Markdown parser"):
        verify_release_checkout(root, altered)


def test_the_shipped_pin_is_read_by_the_released_reader():
    """The pin's reader is the one the pinned release ships, not a copy here."""

    project = Path(__file__).resolve().parent.parent
    checkout = default_upstream_root(upstream_candidates(project))
    if not (checkout / "src/gnn/importers.py").is_file():
        pytest.skip(f"no checkout at {checkout} ships gnn.importers")
    pin = load_importer_pin(project / "gnn/upstream.json", checkout_root=checkout)
    assert pin.repository.endswith("Generalized_Notation_Notation")
    assert re.fullmatch(r"v\d+\.\d+\.\d+", pin.tag)
    assert re.fullmatch(r"[0-9a-f]{40}", pin.commit)


def test_a_checkout_without_the_reader_is_refused(tmp_path):
    project = Path(__file__).resolve().parent.parent
    checkout = tmp_path / "checkout"
    (checkout / "src/gnn").mkdir(parents=True)
    with pytest.raises(UpstreamReleaseError, match="does not ship gnn.importers"):
        load_importer_pin(project / "gnn/upstream.json", checkout_root=checkout)


def test_a_checkout_without_sources_is_refused(tmp_path):
    project = Path(__file__).resolve().parent.parent
    with pytest.raises(UpstreamReleaseError, match="has no src directory"):
        load_importer_pin(project / "gnn/upstream.json", checkout_root=tmp_path / "absent")


def test_a_pin_without_a_tag_cannot_be_verified(tmp_path):
    """The released reader accepts a tagless pin; custody cannot verify one."""

    project = Path(__file__).resolve().parent.parent
    checkout = default_upstream_root(upstream_candidates(project))
    if not (checkout / "src/gnn/importers.py").is_file():
        pytest.skip(f"no checkout at {checkout} ships gnn.importers")
    path = tmp_path / "pin.json"
    path.write_text(json.dumps({"commit": "0" * 40}), encoding="utf-8")
    with pytest.raises(UpstreamReleaseError, match="records no release tag"):
        load_importer_pin(path, checkout_root=checkout)


def test_explicit_upstream_selection_is_final(tmp_path, monkeypatch):
    project = tmp_path / "project"
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("AILM_GNN_UPSTREAM", str(chosen))
    assert selected_upstream_root(project) == chosen


def test_empty_explicit_upstream_selection_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("AILM_GNN_UPSTREAM", " ")
    with pytest.raises(UpstreamReleaseError, match="must not be empty"):
        selected_upstream_root(tmp_path)
