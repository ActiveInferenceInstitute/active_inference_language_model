# dashboard/static/ — the browser tree

Everything the dashboard serves to a browser: markup, one stylesheet, and native ES
modules. No build step, no bundler, no external assets, no network calls beyond the
local server.

## Quick start

```bash
# From the project root, with the project's own interpreter
.venv/bin/python -m ailm.dashboard        # then open http://127.0.0.1:8080/
```

Edit a file here and reload. Static assets are served `no-store`, so the browser
cannot hand you the pre-edit copy.

## Files

| Path | What it is |
|:-----|:-----------|
| `index.html` | Markup only — no `<style>`, no `<script>` body, no `style=` attribute |
| `dashboard.css` | Every style, including the button variants and chart labels that were once inline attributes |
| `js/` | The ES modules, entered from `js/main.js` — see [`js/README.md`](js/README.md) |

## Key rules

- **Markup carries no styling and no logic.** `tests/test_dashboard.py` fails on an
  inline `<style>`, a `<script>` body, or a `style=` attribute.
- **Every class the markup names must exist in `dashboard.css`**, and a variant rule
  must out-specify its base (`button.btn-stop`, not `.btn-stop`).
- **No raw TeX.** There is no math renderer and no external assets, so `$...$`
  renders as literal source. Use Unicode.
- **No measured number in the page.** A manuscript figure typed here is one that no
  `{{TOKEN}}` hydration would ever reach.

## Common commands

```bash
.venv/bin/python -m pytest tests/test_dashboard.py -q --no-cov
```

## More information

See [`AGENTS.md`](AGENTS.md) for the rules in full and
[`../README.md`](../README.md) for the dashboard as a whole.
