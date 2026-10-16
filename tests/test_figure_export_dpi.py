"""Real PNG metadata pins physical type size separately from raster resolution."""

from __future__ import annotations

import struct
import zlib

import matplotlib.pyplot as plt
import pytest

from ailm.figures._common import (
    _PAD_INCHES,
    ExportFigure,
    effective_on_page_dpi,
    on_page_point_size,
    save_figure,
)
from ailm.viz_config import DPI, apply_style


def _page(root):
    manuscript = root / "docs/manuscript"
    manuscript.mkdir(parents=True)
    (manuscript / "preamble.md").write_text("letterpaper, margin=1in\n")
    (manuscript / "04_results.md").write_text(
        "![Density control](../../output/figures/density.png){#fig:density width=100%}\n"
    )


def _figure(export_dpi):
    apply_style()
    if export_dpi is None:
        figure = plt.figure(figsize=(6.4, 3), dpi=72)
    else:
        figure = plt.figure(figsize=(6.4, 3), FigureClass=ExportFigure, export_dpi=export_dpi)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.text(0.5, 0.5, "Unchanged physical type", ha="center", fontsize=10)
    return figure


def test_opt_in_density_improves_resolution_without_halving_printed_type(tmp_path):
    metrics = []
    for label, density in (("default", None), ("higher", 400)):
        root = tmp_path / label
        _page(root)
        figure = _figure(density)
        try:
            path = save_figure(figure, "density", root / "output/figures")
            content = path.read_bytes()
            start = content.index(b"pHYs") + 4
            x, y, unit = struct.unpack_from(">IIB", content, start)
            assert unit == 1 and x == y
            if density is None:
                assert x * 0.0254 >= DPI
            else:
                assert x * 0.0254 == pytest.approx(density, abs=0.01)
            metrics.append(
                (on_page_point_size("density", 10, root), effective_on_page_dpi("density", root))
            )
        finally:
            plt.close(figure)
    assert metrics[0][1] >= 300
    assert metrics[1][1] >= 300
    assert metrics[1][1] > metrics[0][1]
    assert metrics[1][0] == pytest.approx(metrics[0][0], rel=0.002)


@pytest.mark.parametrize("density", [False, 0, -1, 200.5, "400"])
def test_invalid_export_density_refuses(density):
    with pytest.raises(ValueError, match="positive integer"):
        ExportFigure(export_dpi=density)


@pytest.mark.parametrize(
    "defect",
    [
        "signature",
        "truncated",
        "checksum",
        "missing",
        "duplicate",
        "unit",
        "zero",
        "length",
        "empty",
        "header",
        "end",
    ],
)
def test_missing_or_corrupt_density_does_not_fall_back_to_global_dpi(tmp_path, defect):
    _page(tmp_path)
    figure = _figure(400)
    try:
        path = save_figure(figure, "density", tmp_path / "output/figures")
    finally:
        plt.close(figure)
    content = path.read_bytes()
    start = content.index(b"pHYs") - 4
    end = start + 21

    def chunk(kind, body):
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    if defect == "signature":
        content = b"not a PNG"
    elif defect == "truncated":
        content = content[:20]
    elif defect == "checksum":
        content = content[: start + 8] + bytes([content[start + 8] ^ 1]) + content[start + 9 :]
    elif defect == "missing":
        content = content[:start] + content[end:]
    elif defect == "duplicate":
        content = content[:end] + content[start:end] + content[end:]
    elif defect in {"unit", "zero", "length"}:
        body = struct.pack(
            ">IIB", 0 if defect == "zero" else 15748, 15748, 0 if defect == "unit" else 1
        )
        if defect == "length":
            body = body[:-1]
        content = content[:start] + chunk(b"pHYs", body) + content[end:]
    elif defect == "empty":
        content = content[:8] + chunk(b"IHDR", b"\0\0\0\0" + content[20:29]) + content[33:]
    elif defect == "header":
        content = content[:33] + content[8:33] + content[33:]
    else:
        content = content[:-12]
    path.write_bytes(content)
    with pytest.raises(ValueError):
        on_page_point_size("density", project=tmp_path)


@pytest.mark.parametrize(
    "geometry",
    [
        r"\geometry{left=0.4in,right=0.7in,top=0.5in,bottom=0.6in}",
        r"\geometry{margin=0.5in,right=0.6in}",
    ],
)
def test_asymmetric_geometry_uses_actual_text_width(tmp_path, geometry):
    _page(tmp_path)
    (tmp_path / "docs/manuscript/preamble.md").write_text("letterpaper\n" + geometry)
    figure = _figure(400)
    try:
        path = save_figure(figure, "density", tmp_path / "output/figures")
    finally:
        plt.close(figure)
    content = path.read_bytes()
    width = struct.unpack_from(">I", content, 16)[0]
    density = struct.unpack_from(">I", content, content.index(b"pHYs") + 4)[0] * 0.0254
    assert on_page_point_size("density", 10, tmp_path) == pytest.approx(
        10 * (8.5 - 1.1) / (width / density)
    )


def test_tokenized_geometry_uses_validated_configuration(tmp_path):
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "docs/manuscript"
    _page(tmp_path)
    for name in ("preamble.md", "config.yaml"):
        (tmp_path / "docs/manuscript" / name).write_bytes((source / name).read_bytes())
    figure = _figure(400)
    try:
        path = save_figure(figure, "density", tmp_path / "output/figures")
    finally:
        plt.close(figure)
    content = path.read_bytes()
    width = struct.unpack_from(">I", content, 16)[0]
    density = struct.unpack_from(">I", content, content.index(b"pHYs") + 4)[0] * 0.0254
    assert on_page_point_size("density", 10, tmp_path) == pytest.approx(
        10 * (8.5 - 1.1) / (width / density)
    )


@pytest.mark.parametrize(
    "geometry",
    [
        r"\geometry{left=invalid,right=0.7in}",
        r"\geometry{left=4in,right=5in}",
        r"\geometry{top=0.5in,bottom=0.6in}",
        r"\geometry{left=0.4in}",
    ],
)
def test_malformed_geometry_cannot_use_default_margin(tmp_path, geometry):
    _page(tmp_path)
    (tmp_path / "docs/manuscript/preamble.md").write_text(geometry)
    figure = _figure(400)
    try:
        with pytest.raises(ValueError, match="geometry"):
            save_figure(figure, "density", tmp_path / "output/figures")
        assert not (tmp_path / "output/figures/density.png").exists()
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "canvas,rectangle", [((8, 4), (0.2, 0.1, 0.55, 0.7)), ((3.2, 2), (0, 0, 1, 1))]
)
def test_vector_export_meets_print_dpi_after_tight_crop(tmp_path, canvas, rectangle):
    _page(tmp_path)
    apply_style()
    figure = plt.figure(figsize=canvas, dpi=72)
    axis = figure.add_axes(rectangle)
    axis.axis("off")
    axis.text(0.5, 0.5, "Real cropped figure", ha="center", fontsize=10)
    raw = tmp_path / "raw.png"
    try:
        # Negative control reproduces the old central export at actual baseline DPI.
        figure.savefig(raw, dpi=DPI, bbox_inches="tight", pad_inches=_PAD_INCHES)
        raw_pixels = struct.unpack_from(">I", raw.read_bytes(), 16)[0]
        assert raw_pixels / 6.5 < 300
        original = (
            figure.dpi,
            tuple(figure.get_size_inches()),
            axis.texts[0].get_fontsize(),
            axis.texts[0].get_position(),
        )
        final = save_figure(figure, "density", tmp_path / "output/figures")
        final_pixels = struct.unpack_from(">I", final.read_bytes(), 16)[0]
        assert final_pixels / 6.5 >= 300
        assert final_pixels > raw_pixels
        assert effective_on_page_dpi("density", tmp_path) >= 300
        assert (
            figure.dpi,
            tuple(figure.get_size_inches()),
            axis.texts[0].get_fontsize(),
            axis.texts[0].get_position(),
        ) == original
    finally:
        plt.close(figure)


def test_ignored_export_density_cannot_publish_an_underresolved_raster(tmp_path):
    class FixedDensityFigure(plt.Figure):
        def savefig(self, *args, **kwargs):
            kwargs["dpi"] = DPI  # Deliberately broken renderer ignores requested density.
            return super().savefig(*args, **kwargs)

    _page(tmp_path)
    target = tmp_path / "output/figures/density.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"retained publication bytes")
    figure = plt.figure(figsize=(3.2, 2), FigureClass=FixedDensityFigure)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.axis("off")
    axis.text(0.5, 0.5, "Real underresolved control", ha="center", fontsize=10)
    try:
        with pytest.raises(RuntimeError, match="cannot meet 300 dpi"):
            save_figure(figure, "density", target.parent)
        assert target.read_bytes() == b"retained publication bytes"
        assert list(target.parent.glob(".density.png.*.tmp.png")) == []
    finally:
        plt.close(figure)


def test_adaptive_png_export_preserves_vector_outputs_and_numeric_artists(tmp_path):
    _page(tmp_path)
    figure = _figure(None)
    figure.set_size_inches(3.2, 2)
    axis = figure.axes[0]
    axis.errorbar([0.25, 0.5, 0.75], [0.3, 0.6, 0.4], yerr=[0.1, 0.05, 0.08], fmt="o")

    def artists():
        return {
            "canvas": (figure.dpi, tuple(figure.get_size_inches())),
            "lines": [
                (line.get_xdata().tolist(), line.get_ydata().tolist()) for line in axis.lines
            ],
            "intervals": [
                [segment.tolist() for segment in item.get_segments()] for item in axis.collections
            ],
            "text": [
                (text.get_text(), text.get_fontsize(), text.get_position()) for text in axis.texts
            ],
        }

    try:
        before_svg = tmp_path / "before.svg"
        after_svg = tmp_path / "after.svg"
        before_pdf = tmp_path / "before.pdf"
        after_pdf = tmp_path / "after.pdf"
        figure.savefig(before_svg, metadata={"Date": None})
        figure.savefig(before_pdf, metadata={"CreationDate": None, "ModDate": None})
        before = artists()
        save_figure(figure, "density", tmp_path / "output/figures")
        assert effective_on_page_dpi("density", tmp_path) >= 300
        assert artists() == before
        figure.savefig(after_svg, metadata={"Date": None})
        figure.savefig(after_pdf, metadata={"CreationDate": None, "ModDate": None})
        assert after_svg.read_bytes() == before_svg.read_bytes()
        assert after_pdf.read_bytes() == before_pdf.read_bytes()
    finally:
        plt.close(figure)
