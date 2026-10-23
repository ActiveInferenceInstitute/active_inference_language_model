"""The capture receipt the tool assembles is the one the figure generator validates."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ailm.dashboard import capture
from ailm.figures import dashboard_observation

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture_dashboard.py"


def _png(width: int = 3, height: int = 2) -> tuple[bytes, bytes]:
    """A tiny opaque raster and its decoded RGBA buffer."""
    pixels = np.arange(width * height * 4, dtype=np.uint8).reshape(height, width, 4)
    pixels[..., 3] = 255
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buffer, format="PNG")
    return buffer.getvalue(), pixels.tobytes()


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    for source in capture.CAPTURE_SOURCE_PATHS:
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    return tmp_path


def _assets(root: Path, *, prior: bytes | None = None) -> list[capture.ServedAsset]:
    return [
        capture.served_asset(root, route, (root / source).read_bytes(), 200, prior_blob=prior)
        for route, source in capture.SERVED_ROUTES.items()
    ]


def _receipt(root: Path, assets, png: bytes, rgba: bytes, width: int = 3, height: int = 2):
    return capture.build_receipt(
        captured_at="2026-10-23T00:00:00+00:00",
        source_commit="0" * 40,
        url="http://127.0.0.1:8080/",
        input_text="The ant followed the trail.",
        reply="and the and",
        viewport=(900, 1050),
        image=capture.image_identity(png, rgba, width, height),
        served_assets=assets,
        initial_sha256=hashlib.sha256(b"initial").hexdigest(),
        responsive_sha256=hashlib.sha256(png).hexdigest(),
        browser={"engine": "chromium", "version": "test"},
        procedure="scripts/capture_dashboard.py",
    )


def test_the_figure_module_validates_against_the_capture_module_table() -> None:
    assert dashboard_observation.CAPTURE_SOURCE_PATHS is capture.CAPTURE_SOURCE_PATHS
    assert set(capture.SERVED_ROUTES.values()) == capture.CAPTURE_SOURCE_PATHS
    assert len(capture.SERVED_ROUTES) == len(capture.CAPTURE_SOURCE_PATHS)


def test_the_retained_receipt_covers_exactly_the_served_route_table() -> None:
    receipt = json.loads((ROOT / capture.CAPTURE_DIRECTORY / capture.RECEIPT_NAME).read_text())
    routes = {asset["route"]: asset["source"] for asset in receipt["served_assets"]}
    assert routes == dict(capture.SERVED_ROUTES)
    assert receipt["scope"] == capture.SCOPE_STATEMENT
    assert receipt["transform"] == capture.TRANSFORM_STATEMENT


def test_an_assembled_receipt_is_accepted_by_the_figure_validator(source_tree: Path) -> None:
    png, rgba = _png()
    receipt = _receipt(source_tree, _assets(source_tree), png, rgba)
    capture.write_capture(source_tree / capture.CAPTURE_DIRECTORY, png, receipt)
    pixels = dashboard_observation.dashboard_capture_pixels(source_tree)
    assert pixels.shape == (2, 3, 4)
    assert pixels.tobytes() == rgba
    written = json.loads((source_tree / capture.CAPTURE_DIRECTORY / "receipt.json").read_text())
    assert written["schema"] == capture.RECEIPT_SCHEMA
    assert written["image"]["path"] == capture.IMAGE_NAME
    assert written["procedure"] == "scripts/capture_dashboard.py"


def test_a_served_body_must_equal_the_source_file(source_tree: Path) -> None:
    route, source = "/dashboard.css", capture.SERVED_ROUTES["/dashboard.css"]
    body = (source_tree / source).read_bytes()
    bound = capture.served_asset(source_tree, route, body, 200, prior_blob=body)
    assert bound.exact_prior_main_blob is True
    assert bound.bytes == len(body)
    assert (
        capture.served_asset(source_tree, route, body, 200, prior_blob=None).exact_prior_main_blob
        is False
    )
    assert (
        capture.served_asset(
            source_tree, route, body, 200, prior_blob=body + b"x"
        ).exact_prior_main_blob
        is False
    )
    with pytest.raises(capture.CaptureError, match="differ from"):
        capture.served_asset(source_tree, route, body + b"\n", 200, prior_blob=None)
    with pytest.raises(capture.CaptureError, match="not 200"):
        capture.served_asset(source_tree, route, body, 304, prior_blob=None)
    with pytest.raises(capture.CaptureError, match="not part of the captured surface"):
        capture.served_asset(source_tree, "/favicon.ico", b"", 200, prior_blob=None)


def test_a_receipt_must_cover_the_surface_exactly(source_tree: Path) -> None:
    png, rgba = _png()
    assets = _assets(source_tree)
    with pytest.raises(capture.CaptureError, match="missing=\\['output/web/ailm.html'\\]"):
        _receipt(source_tree, [a for a in assets if a.route != "/explorer/"], png, rgba)
    with pytest.raises(capture.CaptureError, match="duplicate="):
        _receipt(source_tree, [*assets, assets[0]], png, rgba)
    with pytest.raises(capture.CaptureError, match="viewport must be positive"):
        capture.build_receipt(
            captured_at="",
            source_commit="",
            url="",
            input_text="",
            reply="",
            viewport=(0, 10),
            image=capture.image_identity(png, rgba, 3, 2),
            served_assets=assets,
            initial_sha256="",
            responsive_sha256="",
            browser={},
            procedure="",
        )


def test_the_pixel_digest_binds_decoded_pixels_not_encoding() -> None:
    png, rgba = _png()
    identity = capture.image_identity(png, rgba, 3, 2)
    with Image.open(io.BytesIO(png)) as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=0)
    recoded = buffer.getvalue()
    assert recoded != png
    again = capture.image_identity(recoded, rgba, 3, 2)
    assert again.rgba_sha256 == identity.rgba_sha256
    assert again.sha256 != identity.sha256
    with pytest.raises(capture.CaptureError, match="not 3x2x4"):
        capture.image_identity(png, rgba[:-1], 3, 2)


def test_prior_blob_lookup_reads_the_commit_not_the_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp_path)}

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            env={**env, "PATH": "/usr/bin:/bin"},
            stdin=subprocess.DEVNULL,
        ).stdout.strip()

    git("init", "-q")
    (repo / "tracked.txt").write_bytes(b"committed\n")
    git("add", "tracked.txt")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "one")
    head = capture.git_head(repo)
    assert head == git("rev-parse", "HEAD")
    assert capture.git_blob(repo, head, "tracked.txt") == b"committed\n"
    (repo / "tracked.txt").write_bytes(b"edited\n")
    assert capture.git_blob(repo, head, "tracked.txt") == b"committed\n"
    (repo / "untracked.txt").write_bytes(b"new\n")
    assert capture.git_blob(repo, head, "untracked.txt") is None
    with pytest.raises(capture.CaptureError, match="no HEAD commit"):
        capture.git_head(tmp_path)


def test_write_capture_replaces_both_files_atomically(tmp_path: Path) -> None:
    png, rgba = _png()
    directory = tmp_path / "capture"
    capture.write_capture(directory, png, {"schema": capture.RECEIPT_SCHEMA})
    assert (directory / capture.IMAGE_NAME).read_bytes() == png
    assert json.loads((directory / capture.RECEIPT_NAME).read_text()) == {
        "schema": capture.RECEIPT_SCHEMA
    }
    assert sorted(p.name for p in directory.iterdir()) == [capture.IMAGE_NAME, capture.RECEIPT_NAME]


def test_the_capture_tool_is_a_thin_orchestrator_over_the_capture_module() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from ailm.dashboard.capture import" in source
    assert "playwright" in source and "sync_playwright" in source
    # The browser and imaging dependencies stay out of the project environment:
    # they are imported where they are used, never at module import.
    module_level = source.split("def parse_args", 1)[0]
    assert "from PIL" not in module_level
    assert "from playwright" not in module_level
    assert "sha256(" not in source.replace("sha256_hex(", "")
