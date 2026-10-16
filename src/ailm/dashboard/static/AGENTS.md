# AGENTS.md — src/ailm/dashboard/static/

Operating rules for the dashboard's browser tree. Quick reference:
[`README.md`](README.md). The dashboard as a whole is documented in
[`../README.md`](../README.md), which this file does not restate.

## Overview

Three concerns, three places: `index.html` is markup, `dashboard.css` is style, and
`js/` is behavior. They were one file. One file holding three languages cannot be
reviewed a concern at a time, and a change to any of them showed up as a diff
against all three.

Getting the `<style>` block out was only most of the job — dozens of `style=`
attributes stayed behind, the same three button backgrounds transcribed across
eight buttons. An attribute is CSS in the markup that does not announce itself,
cannot be reused, and cannot be overridden without `!important`. A test refuses
the next one.

## File organization

| Path | Concern |
|:-----|:--------|
| `index.html` | Structure and inline handler names only |
| `dashboard.css` | Every rule, including named classes for what used to be attributes |
| `js/` | Native ES modules, entered from `main.js` (see [`js/AGENTS.md`](js/AGENTS.md)) |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **No inline `<style>`, `<script>` body, or `style=` attribute** | Enforced by `tests/test_dashboard.py`. This is the rule the directory exists to make possible |
| **Every class the markup names is defined in `dashboard.css`** | A class with no rule is a silent no-op that looks intentional |
| **A variant rule out-specifies its base** | The base is `button.btn-primary` (specificity 0,1,1); a bare `.btn-stop` (0,1,0) loses regardless of source order and the stop button renders default green. No DOM test can see it — the classes are present, the markup is right, the page is the wrong color |
| **No external assets and no network calls** | Nothing loads from a CDN, a font host, or anywhere but the local server |
| **No raw TeX** | There is no math renderer; `$...$` renders literally. Use Unicode |
| **No measured number in the page** | Level tags describe what a panel shows, never what a run measured |
| **No modal dialog, and no heading that titles two elements** | Both are asserted by the structural tests |

## Configuration

Nothing here is configured by hand. `js/config.js` fetches `GET /api/config` and
applies the server's own bounds to the controls at load, so no limit is written
twice. A reply-length input once carried a hand-transcribed `max="200"` beside a
speech cap declared as a `const`; a limit written twice is a limit that will
disagree with itself, and the browser copy is the one nobody tests.

## Testing

Run from the project root, with the project's own interpreter.

```bash
.venv/bin/python -m pytest tests/test_dashboard.py -q --no-cov
```

That file asserts the structural rules above plus the module graph in `js/`.
Structural tests cannot see a degraded model, which is why every behavioral test in
this area carries a negative control.

## Best practices

- Add a class, then a rule, then use it. A `style=` attribute is never the shortcut
  it looks like.
- When a control's behavior changes, change the handler name in the markup and the
  exported function together — `main.js` attaches every named handler to `window`
  in one explicit list, and a test resolves each one.
- Verify a visual change in a real browser, not only in the test. The specificity
  defect above passed every DOM assertion.

## See Also

- [`README.md`](README.md) — quick reference for this directory
- [`js/AGENTS.md`](js/AGENTS.md) — the module graph and its rules
- [`../AGENTS.md`](../AGENTS.md) — dashboard operating rules
- [`../README.md`](../README.md) — the full dashboard guide
