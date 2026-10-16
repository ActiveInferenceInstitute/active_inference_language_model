"""Real capture files bind the dashboard illustration to observed browser pixels."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from ailm.figures.dashboard_observation import (
    DashboardCaptureError,
    dashboard_capture_pixels,
    generate_dashboard_observation_figure,
)
from scripts.generate_figures import _computation_input_paths

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def capture_root(tmp_path):
    directory = tmp_path / "data/dashboard_capture"
    shutil.copytree(ROOT / "data/dashboard_capture", directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    for asset in receipt["served_assets"]:
        target = tmp_path / asset["source"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / asset["source"], target)
    return tmp_path


def test_capture_pixels_equal_the_original_browser_raster():
    with Image.open(ROOT / "data/dashboard_capture/analysis.png") as image:
        expected = np.array(image.convert("RGBA"))
    actual = dashboard_capture_pixels(ROOT)
    np.testing.assert_array_equal(actual, expected)


def test_figure_frames_the_complete_original_pixels():
    figure = generate_dashboard_observation_figure()
    try:
        assert len(figure.axes) == 1
        np.testing.assert_array_equal(
            figure.axes[0].images[0].get_array(), dashboard_capture_pixels(ROOT)
        )
        assert figure.axes[0].images[0].get_interpolation() == "none"
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "defect",
    [
        "bytes",
        "pixels",
        "dimensions",
        "source",
        "empty",
        "duplicate",
        "path",
        "schema",
        "asset-path",
        "asset-traversal",
        "altered-pixel",
        "missing-one",
        "extra-static",
    ],
)
def test_altered_capture_or_source_is_refused(capture_root, defect):
    receipt_path = capture_root / "data/dashboard_capture/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    match defect:
        case "bytes":
            image_path = capture_root / "data/dashboard_capture/analysis.png"
            image_path.write_bytes(image_path.read_bytes() + b"changed")
        case "pixels":
            receipt["image"]["rgba_sha256"] = "0" * 64
        case "dimensions":
            receipt["image"]["width"] += 1
        case "source":
            source_path = capture_root / receipt["served_assets"][0]["source"]
            source_path.write_bytes(source_path.read_bytes() + b"changed")
        case "empty":
            receipt["served_assets"] = []
        case "duplicate":
            receipt["served_assets"].append(receipt["served_assets"][0])
        case "path":
            receipt["image"]["path"] = "../outside.png"
        case "asset-path":
            receipt["served_assets"][0]["source"] = "../outside.html"
        case "asset-traversal":
            receipt["served_assets"][0]["source"] = "src/ailm/dashboard/static/../outside.html"
        case "altered-pixel":
            image_path = capture_root / "data/dashboard_capture/analysis.png"
            with Image.open(image_path) as original:
                changed = original.convert("RGBA")
            changed.putpixel((0, 0), (255, 0, 0, 255))
            changed.save(image_path)
            receipt["image"]["sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        case "missing-one":
            receipt["served_assets"].pop()
        case "extra-static":
            relative = Path("src/ailm/dashboard/static/README.md")
            shutil.copyfile(ROOT / relative, capture_root / relative)
            receipt["served_assets"].append(
                {
                    "source": relative.as_posix(),
                    "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                }
            )
        case "schema":
            receipt["schema"] = "unknown"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(DashboardCaptureError):
        dashboard_capture_pixels(capture_root)


def test_capture_png_and_receipt_are_in_figure_input_provenance():
    paths = _computation_input_paths(ROOT)
    assert Path("data/dashboard_capture/analysis.png") in paths
    assert Path("data/dashboard_capture/receipt.json") in paths
    receipt = json.loads((ROOT / "data/dashboard_capture/receipt.json").read_text())
    assert all(Path(asset["source"]) in paths for asset in receipt["served_assets"])
    assert (
        hashlib.sha256((ROOT / "data/dashboard_capture/analysis.png").read_bytes()).hexdigest()
        == receipt["image"]["sha256"]
    )
