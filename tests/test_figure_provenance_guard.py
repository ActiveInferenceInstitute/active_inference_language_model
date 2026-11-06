"""The pre-push guard refuses a commit whose figure provenance is stale.

``scripts/verify_figure_provenance.py`` reads the v3 receipt and the source blobs
from one commit. These tests exercise it against the real repository head and
against a throwaway repository whose source moves without the receipt — a
positive control, a digest mismatch, and an inventory mismatch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUARD = PROJECT_ROOT / "scripts" / "verify_figure_provenance.py"
RECEIPT_PATH = "output/figures/figure_provenance.json"


def _run(root: Path, commit: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-S", str(GUARD), "--project-root", str(root), "--commit", commit],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=120,
    )


def _repository(root: Path) -> Path:
    (root / "src" / "ailm").mkdir(parents=True)
    (root / "output" / "figures").mkdir(parents=True)
    module = root / "src" / "ailm" / "mod.py"
    module.write_text("x = 1\n", encoding="utf-8")
    receipt = {
        "schema": "ailm-figure-provenance-v3",
        "code_dependencies": [
            {
                "path": "src/ailm/mod.py",
                "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            }
        ],
    }
    (root / RECEIPT_PATH).write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    _git(root, "init", "-q")
    # Local identity and signing config: the controls must not depend on whatever
    # the developer or the hosted runner has configured globally.
    _git(root, "config", "user.name", "AILM provenance guard")
    _git(root, "config", "user.email", "ailm@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "receipt and source agree")
    return module


def test_guard_accepts_the_repository_head() -> None:
    """Positive control on the real checkout: HEAD's receipt matches HEAD's sources."""
    receipt = json.loads((PROJECT_ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))
    recorded = receipt["code_dependencies"]

    result = _run(PROJECT_ROOT, "HEAD")

    assert result.returncode == 0, result.stderr
    assert f"{len(recorded)} AILM code dependencies match" in result.stdout


def test_guard_refuses_a_source_edit_and_a_deleted_dependency(tmp_path: Path) -> None:
    """NEGATIVE CONTROLS: a digest change and an inventory change both refuse."""
    root = tmp_path / "repo"
    module = _repository(root)

    accepted = _run(root, "HEAD")
    assert accepted.returncode == 0, accepted.stderr
    assert "1 AILM code dependencies match" in accepted.stdout

    module.write_text("x = 2\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "source moves without the receipt")
    stale_digest = _run(root, "HEAD")
    assert stale_digest.returncode == 1
    assert "src/ailm/mod.py" in stale_digest.stderr
    assert "digest changed" in stale_digest.stderr

    module.unlink()
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "dependency disappears without the receipt")
    stale_inventory = _run(root, "HEAD")
    assert stale_inventory.returncode == 1
    assert "inventory changed" in stale_inventory.stderr
    assert "recorded but absent" in stale_inventory.stderr
