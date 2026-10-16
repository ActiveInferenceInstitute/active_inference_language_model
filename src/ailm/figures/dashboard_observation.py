"""Present an authenticated browser capture without reconstructing its interface.

The raster is observational evidence of a warm local interaction, not a new
scientific measurement. Pixel contents stay separate from the vector title.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

from ailm.figures._common import project_root
from ailm.reference_presentation import _unique_json_object
from ailm.viz_config import FIGSIZES, FONTS, apply_style

CAPTURE_SOURCE_PATHS: Final[frozenset[str]] = frozenset(
    {
        "src/ailm/dashboard/static/index.html",
        "src/ailm/dashboard/static/dashboard.css",
        "output/web/ailm.html",
        *(
            "src/ailm/dashboard/static/js/" + name + ".js"
            for name in (
                "matrices",
                "config",
                "speech",
                "main",
                "stt",
                "state",
                "status",
                "chat",
                "inference",
            )
        ),
    }
)
"""
The exact surface authenticated by the capture receipt; changing that surface
requires recapture rather than accepting a shortened or enlarged receipt.
"""


class DashboardCaptureError(ValueError):
    """The screenshot, source assets, or capture metadata cannot be authenticated."""


def dashboard_capture_pixels(root: Path | None = None) -> NDArray[np.uint8]:
    """Read the exact captured raster only after byte, pixel, and source checks."""
    base = project_root() if root is None else root
    directory = base / "data" / "dashboard_capture"
    receipt = json.loads(
        (directory / "receipt.json").read_text(), object_pairs_hook=_unique_json_object
    )
    if receipt.get("schema") != "ailm-dashboard-capture-v1":
        raise DashboardCaptureError("unsupported dashboard capture schema")
    capture = receipt["image"]
    if capture["path"] != "analysis.png":
        raise DashboardCaptureError("unexpected dashboard capture image path")
    content = (directory / "analysis.png").read_bytes()
    if hashlib.sha256(content).hexdigest() != capture["sha256"]:
        raise DashboardCaptureError("dashboard capture byte digest differs")
    with Image.open(io.BytesIO(content)) as image:
        pixels = np.array(image.convert("RGBA"), dtype=np.uint8)
    if pixels.shape != (capture["height"], capture["width"], 4):
        raise DashboardCaptureError("dashboard capture dimensions differ")
    if hashlib.sha256(pixels.tobytes()).hexdigest() != capture["rgba_sha256"]:
        raise DashboardCaptureError("dashboard capture pixel digest differs")
    assets = receipt["served_assets"]
    names = [asset["source"] for asset in assets]
    if not names or len(names) != len(set(names)):
        raise DashboardCaptureError("missing or duplicate dashboard source bindings")
    if set(names) != CAPTURE_SOURCE_PATHS:
        raise DashboardCaptureError("incomplete or unexpected dashboard source bindings")
    for asset in assets:
        relative = Path(asset["source"])
        if hashlib.sha256((base / relative).read_bytes()).hexdigest() != asset["sha256"]:
            raise DashboardCaptureError(f"dashboard captured source differs: {relative}")
    return pixels


def generate_dashboard_observation_figure() -> Figure:
    """Frame the authentic screenshot with a title; never draw substitute UI."""
    pixels = dashboard_capture_pixels()
    apply_style()
    width = FIGSIZES["square"][0]
    height = width * pixels.shape[0] / pixels.shape[1]
    figure = plt.figure(figsize=(width, height + 0.5))
    axis = figure.add_axes((0, 0, 1, height / (height + 0.5)))
    axis.imshow(pixels, interpolation="none")
    axis.set_axis_off()
    figure.text(
        0.5,
        0.985,
        "Observed dashboard: analysis and lexical reply",
        ha="center",
        va="top",
        fontsize=FONTS["title"],
    )
    return figure
