"""Stdlib-first bounded publication artists helpers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes._base import _AxesBase
    from numpy.typing import ArrayLike

    from ailm._publication_types import JSONMap, JSONValue

from ._publication_bootstrap import (
    PROJECT_ROOT,
    audited_startup,
    configure_imports,
    install_policy,
    origins,
    write_report,
)


def internal_artists(arguments: list[str]) -> int:
    source, site_arg, token_arg, report_arg, endpoint, mutation = arguments
    root = Path(source).absolute()
    site, startup = audited_startup(PROJECT_ROOT)
    if site != Path(site_arg).absolute() or mutation not in {"normal", "mutation"}:
        raise ValueError("invalid artist roots/mutation")
    executed = install_policy("offline", endpoint, root, PROJECT_ROOT)
    configure_imports(root, PROJECT_ROOT, site)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection

    from ailm.figures import results
    from ailm.figures.benchmarks import build_external_benchmarks_figure
    from ailm.figures.case_word_sweep import build_case_word_sweep_figure
    from ailm.figures.incontext_sweep import generate_incontext_sweep
    from ailm.figures.multiplicity import generate_multiplicity_forest

    content = Path(token_arg).read_bytes()
    tokens = json.loads(content)
    figures = {
        "case_word_sweep": build_case_word_sweep_figure(tokens),
        "g2p_accuracy": results.build_g2p_accuracy_figure(tokens)[0],
        "incontext_sweep": generate_incontext_sweep(tokens),
        "external_benchmarks": build_external_benchmarks_figure(tokens),
        "multiplicity_forest": generate_multiplicity_forest(tokens),
    }
    if mutation == "mutation":
        axis = figures["case_word_sweep"].axes[0]
        if not axis.lines:
            raise ValueError("artist negative control has no line")
        axis.lines[0].set_ydata(np.asarray(axis.lines[0].get_ydata()) + 1.0)
        delta_axis = figures["external_benchmarks"].axes[1]
        if not delta_axis.collections:
            raise ValueError("external interval negative control has no collection")
        intervals = delta_axis.collections[-1]
        if not isinstance(intervals, LineCollection):
            raise ValueError("external interval negative control is not a line collection")
        segments = [np.asarray(path.vertices).copy() for path in intervals.get_paths()]
        if len(segments) != 3 or segments[1].shape != (2, 2):
            raise ValueError("external phoneme interval negative control has unexpected geometry")
        segments[1][0, 1] += 0.01
        intervals.set_segments(segments)
        forest_axis = figures["multiplicity_forest"].axes[0]
        if not forest_axis.collections:
            raise ValueError("multiplicity interval negative control has no collection")
        forest_intervals = forest_axis.collections[0]
        if not isinstance(forest_intervals, LineCollection):
            raise ValueError("multiplicity interval negative control is not a line collection")
        forest_segments = [
            np.asarray(path.vertices).copy() for path in forest_intervals.get_paths()
        ]
        if len(forest_segments) != 4 or any(segment.shape != (2, 2) for segment in forest_segments):
            raise ValueError("multiplicity interval negative control has unexpected geometry")
        forest_segments[1][0, 0] += 0.01
        forest_intervals.set_segments(forest_segments)

    def values(value: ArrayLike | None) -> JSONValue:
        if value is None:
            return None
        array = np.asanyarray(value)
        if isinstance(array, np.ma.MaskedArray):
            array = array.filled(np.nan)

        def finite(item: JSONValue) -> JSONValue:
            if isinstance(item, list):
                return [finite(x) for x in item]
            if isinstance(item, float) and not np.isfinite(item):
                return {"nonfinite": str(item)}
            return item

        return finite(array.tolist())

    def axis_values(axis: _AxesBase) -> JSONMap:
        return {
            "lines": [{"x": values(x.get_xdata()), "y": values(x.get_ydata())} for x in axis.lines],
            "collections": [
                {
                    "offsets": values(x.get_offsets()),
                    "array": values(x.get_array()),
                    "paths": [
                        {"vertices": values(p.vertices), "codes": values(p.codes)}
                        for p in x.get_paths()
                    ],
                }
                for x in axis.collections
            ],
            "patches": [
                {
                    "vertices": values(x.get_path().vertices),
                    "codes": values(x.get_path().codes),
                    "bbox": values(x.get_bbox().bounds) if hasattr(x, "get_bbox") else None,
                }
                for x in axis.patches
            ],
            "images": [
                {
                    "array": values(x.get_array()),
                    "extent": values(x.get_extent()),
                    "clim": values(x.get_clim()),
                }
                for x in axis.images
            ],
            "xlim": values(axis.get_xlim()),
            "ylim": values(axis.get_ylim()),
            "children": [axis_values(x) for x in axis.child_axes],
        }

    artists = {}
    for name, figure in figures.items():
        figure.canvas.draw()
        artists[name] = [axis_values(x) for x in figure.axes]
        plt.close(figure)
    report = {
        "artists": artists,
        "tokens_sha256": hashlib.sha256(content).hexdigest(),
        "module_origins": origins(root, executed),
        "sys_path": list(sys.path),
        "dependency_startup": startup,
        "mutation": mutation,
        "policy": "offline",
    }
    write_report(Path(report_arg), report)
    print("numeric artists captured: " + ", ".join(sorted(artists)), flush=True)
    return 0
