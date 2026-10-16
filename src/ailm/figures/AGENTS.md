# AGENTS.md — src/ailm/figures/

## Ownership boundary

`figures/` owns **what a figure looks like** and **the geometry that makes it
readable**. It does not own:

- the numbers on the axes → `data/claims.json` → `output/data/manuscript_variables.json`
- the palette, type scale, canvas sizes, `MIN_FONT_PT`, `DPI` → `ailm/viz_config.py`
- discovery, ordering, argv, stdout, exit codes → `scripts/generate_figures.py`
- where the PNGs land → `output/figures/`, which is disposable and gitignored

A generator computes layout and returns a `Figure` (or writes via `save_figure` and
returns a `Path`). It does not print, does not parse arguments, and does not decide
which figures the build needs.

## May change freely

- Add a figure: a `generate_<stem>() -> Path` or `generate_<stem>_figure() -> Figure`
  in a non-underscore module. `scripts/generate_figures.py` finds it by `pkgutil`
  discovery — there is no registry to update, which is the point.
- Adjust layout constants, panel arrangement, annotation wording.
- Add a ledger cross-check inside a `*_data()` function.

## Must not change without an explicit decision

**Do not type a measured number into this package.** Read it from the token map via
`load_manuscript_variables` / `token_float` / `token_int`, or compute it from a
vendored artifact. `MissingVariableError` on an absent token is correct behavior —
do not add a default, a `try/except`, or an `or 0.0`.

**Do not weaken `assert_legible`, and do not add a clamp.** A runtime clamp that
shrinks text to fit is not a floor; it produces an unreadable panel and reports
success. Lowering `MIN_FONT_PT` to make a figure pass inverts the invariant.
`FONTS["cell"]` sits exactly on the floor on purpose, so any future shrink is caught
by the gate rather than by a squinting reader.

**Do not remove the empty-figure branch of `assert_legible`.** A legibility check that
passes on a canvas with no text is vacuous, and a generator that silently stopped
emitting labels would sail through it.

**Do not remove the PNG `Software` metadata override or the font pin.** matplotlib
stamps its own version into PNG text metadata, and a substituted system font changes
glyph widths, hence the `tight` bbox, hence the bytes. Both are why "the figures are
up to date" is a checkable claim.

**Do not remove a ledger cross-check in `results.py` / `gates.py`.** The ladder
monotonicity check, the `log(uniform) == cross-entropy + gain` identity, and the
`margin == model - baseline` identity each catch two claims recorded in different
runs. `tests/test_figures_results.py` feeds every one of them a deliberately
inconsistent token map, so none is decoration — deleting a check would silently
delete its test's subject.

**Do not color-code by red versus green alone.** The palette is Okabe–Ito and its
keys name meanings (`MODEL`, `UNIGRAM`, `UNIFORM`, `FLOOR`, `ANTI`), not appearances.
The two comparators that sit closest in the argument — correct wiring and the
anti-informative wiring — are additionally separated by fill pattern and an explicit
text label. Keep all three encodings.

**Keep classification in `ailm.gates`.** `_classify` in the figure module
delegates to `ailm.gates.classify_gate`, and `scripts/audit_gates.py` uses the same
semantics through `evaluate_gate`. Change that shared implementation and its
controls when the gate contract changes; do not restore a second classifier in
either consumer.

## Local gotchas

**`assert_legible` checks font *size* only.** It cannot see overlapping text or text
clipped outside the canvas, and both are more common failure modes than tiny type.
Those live in `tests/test_figure_layout.py`, which renders and compares display-space
bounding boxes with `OVERLAP_TOLERANCE = 0.22` and `CLIP_TOLERANCE_PT = 2.0`.

**That sweep only collects zero-argument builders**, matched by the name prefixes
`build_` / `make_` / `plot_` / `figure_` / `generate_`. A builder with a required
positional parameter is skipped. Give new builders defaults for every argument, or the
figure escapes the geometry gate while the size gate still passes. The prefix list
once omitted `generate_` entirely, silently excluding four of nine figures — two of
which had confirmed defects — and reading as a clean sweep; that is why
`test_discovery_covers_every_shipped_figure` compares the shipped PNG stems on
disk against the discovered builder stems as sets, so a missing figure, an extra
builder, or two callables claiming one stem each fail the sweep — a property the
older count comparison could not guarantee. Do not relax that completeness
assertion to a non-emptiness check.

**`ailm/figures/__init__.py` is a partial barrel.** It re-exports `architecture` and
`matrices` but not `results` or `gates`. Import from the concrete module. Do not
"fix" the barrel by re-exporting everything without checking for name collisions
between modules' `generate_*` functions.

**Discovery filters on `__module__`.** `scripts/generate_figures.py` skips a
`generate_*` attribute whose `__module__` is not the module it was found in, so a
re-exported generator is not run twice. A generator defined in one module and
imported into another therefore *will not* be discovered from the second — keep
generators where they are defined.

**Module-level `write_*_figures` aggregates are intentionally not discovered.** They
would re-generate figures the script already produced individually and cannot be
selected with `--only`.

**Determinism is not just about seeds.** A `set` iteration, a `dict` built from an
unsorted glob, or a `datetime.now()` in an annotation each break byte-identity without
changing the visible image. The regeneration diff in the verification block is the
only reliable detector.

**`project_root()` honours `AILM_PROJECT_ROOT`.** The renderer may invoke a generator
from a different working directory. Do not replace it with a relative walk-up.

**Kill stray renderer processes after a long figure session** if you have been
invoking any external diagram tool; nothing in this package spawns one, but the
project's history includes leaked Chrome/mmdc processes from adjacent tooling.

## Verification for this directory

```bash
cd "$(git rev-parse --show-toplevel)"   # the repository root

.venv/bin/python scripts/z_generate_manuscript_variables.py --print-only
MPLBACKEND=Agg .venv/bin/python -m pytest \
  tests/test_figures_architecture.py tests/test_figures_results.py \
  tests/test_figure_layout.py -q --no-cov

# Byte-identical regeneration
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
shasum -a 256 output/figures/*.png > /tmp/ailm_figs_a.txt
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
shasum -a 256 output/figures/*.png > /tmp/ailm_figs_b.txt
diff /tmp/ailm_figs_a.txt /tmp/ailm_figs_b.txt

# Every discovered generator, with its output stem
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py --list
```

If `output/data/manuscript_variables.json` is absent, every token-reading generator
raises `MissingVariableError`. Run
`.venv/bin/python scripts/z_generate_manuscript_variables.py` first — that is the
correct fix, not a default value.

**Reference-rule crossings are checked at save time.** `assert_no_text_crosses_rules`
uses rendered glyph extents for horizontal/vertical rules drawn with Matplotlib's
standard axes transforms, including inset children. An opaque text or legend
background protects a label only when it draws above the rule. Data curves,
gridlines, figure-owned text, and custom composed transforms are outside this
bounded check; their geometry still requires visual inspection. Controls cover
horizontal/vertical rules, partial spans, inverted views, opacity and z-order,
clipping, and the unit-limit data/axes distinction.
