# dashboard/static/js/ — native ES modules

The dashboard's client behavior. No bundler, no transpiler, no dependencies: the
browser loads these modules directly from the local server.

## Quick start

```bash
# From the project root, with the project's own interpreter
.venv/bin/python -m ailm.dashboard        # then open http://127.0.0.1:8080/
```

`main.js` is the only entry point. Every other module is reachable from it, and a
test fails on an orphan.

## Modules

| File | What it owns |
|:-----|:-------------|
| `main.js` | Entry point: config load, lifecycle, and the explicit list of handlers attached to `window` |
| `config.js` | Fetches `GET /api/config` and applies the server's bounds to the controls |
| `state.js` | Cross-module state, so no module owns another's variables |
| `status.js` | The shared status region |
| `speech.js` | Speech output — the only module permitted to queue an utterance |
| `stt.js` | Speech recognition for the inspector input |
| `matrices.js` | Generative matrices drawn as canvas heatmaps |
| `inference.js` | `POST /api/infer`, the timeline, and every analysis panel |
| `chat.js` | The bidirectional transcript |

## Key rules

- **`main.js` is the only entry.** No orphan modules; the module graph is asserted.
- **No module exceeds 400 lines.** Split it instead.
- **Shared state lives in `state.js`** and is never redeclared elsewhere.
- **Only `speech.js` calls `speechSynthesis.speak`,** and `speak()` unconditionally
  calls `stopSpeaking()` first.
- **Inline `onclick` handlers are deliberate.** ES modules do not populate the
  global scope, so `main.js` attaches each named function to `window` explicitly;
  a test resolves every handler the markup names.

## Common commands

```bash
.venv/bin/python -m pytest tests/test_dashboard.py -q --no-cov
```

## More information

See [`AGENTS.md`](AGENTS.md) for the rules in full,
[`../AGENTS.md`](../AGENTS.md) for the browser tree, and
[`../../README.md`](../../README.md) for the dashboard as a whole.
