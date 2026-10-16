"""Figure generators for the AILM manuscript.

One module per topic, a shared ``_common`` for deterministic writing and the
legibility gate, and :mod:`ailm.viz_config` for the palette and type scale. The
generators return :class:`matplotlib.figure.Figure` objects and know nothing about
the pipeline; a thin orchestrator in ``scripts/`` calls them and prints the paths.

Two properties are enforced rather than encouraged:

- **Determinism.** Fixed seeds where randomness would otherwise appear, sorted
  iteration, no timestamp in the image, and a pinned PNG ``Software`` field, so
  regenerating a figure twice yields byte-identical output.
- **Legibility.** ``save_figure`` refuses to write a figure containing type below
  :data:`ailm.viz_config.MIN_FONT_PT`. There is no clamp — an overflowing label
  fails the build instead of shrinking to illegibility.

No generator hardcodes a measured number. Values come from the vendored corpus
artifacts or from ``output/data/manuscript_variables.json``, which is generated
from the claims ledger.

Nothing here imports ``infrastructure``: ``src/`` stays standalone.
"""

from __future__ import annotations

from ailm.figures._common import (
    FigureOverflowError,
    IllegibleFigureError,
    MissingVariableError,
    assert_legible,
    assert_within_canvas,
    default_output_dir,
    load_manuscript_variables,
    project_root,
    rendered_texts,
    save_figure,
    text_extent,
    token_float,
    token_int,
)
from ailm.figures.architecture import (
    LEVELS,
    LevelRow,
    generate_hierarchy_figure,
    generate_inference_cycle_figure,
    write_architecture_figures,
)
from ailm.figures.matrices import (
    MARKED_BIGRAMS,
    generate_letter_transition_figure,
    generate_wiring_comparison_figure,
    letter_transition_matrix,
    write_matrix_figures,
)

__all__ = [
    "LEVELS",
    "MARKED_BIGRAMS",
    "FigureOverflowError",
    "IllegibleFigureError",
    "LevelRow",
    "MissingVariableError",
    "assert_legible",
    "assert_within_canvas",
    "default_output_dir",
    "generate_hierarchy_figure",
    "generate_inference_cycle_figure",
    "generate_letter_transition_figure",
    "generate_wiring_comparison_figure",
    "letter_transition_matrix",
    "load_manuscript_variables",
    "project_root",
    "rendered_texts",
    "save_figure",
    "text_extent",
    "token_float",
    "token_int",
    "write_architecture_figures",
    "write_matrix_figures",
]
