# AGENTS.md — src/ailm/dashboard/static/js/

Operating rules for the dashboard's client modules. Quick reference:
[`README.md`](README.md). The full dashboard guide is
[`../../README.md`](../../README.md).

## Overview

Native ES modules, loaded directly by the browser from the local server. There is
no build step and no dependency to install, which is why a change here is verified
by reloading the page rather than by rebuilding anything.

The module boundaries follow the concerns the page actually has — configuration,
shared state, status, speech out, speech in, matrices, inference, chat — and the
test suite holds them: the graph must have `main.js` as its only entry, no orphans,
and no module over 400 lines.

## File organization

| File | Concern |
|:-----|:--------|
| `main.js` | Config load, lifecycle wiring, and the single explicit list of `window`-attached handlers |
| `config.js` | `GET /api/config` → control bounds |
| `state.js` | Cross-module state; the only place a shared variable is declared |
| `status.js` | The shared status region |
| `speech.js` | Speech synthesis: `speak`, `stopSpeaking`, the watchdog, the boundary hooks |
| `stt.js` | Speech recognition for the inspector input |
| `matrices.js` | Canvas heatmaps of the generative matrices |
| `inference.js` | `POST /api/infer`, the timeline, the analysis panels |
| `chat.js` | The bidirectional transcript |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **`main.js` is the only entry, and there are no orphans** | An unreachable module is code nobody runs and everybody maintains |
| **No module exceeds 400 lines** | Asserted directly; the whole tree exists because one file held three languages |
| **Shared state is declared only in `state.js`** | A second declaration makes two modules disagree about the same value with no error |
| **Only `speech.js` queues an utterance** | `speechSynthesis` is a queue, not a channel: `cancel()` clears only what is pending at that instant, so a second queueing path leaves it draining after the turn ends. The interface looks idle and the machine is still talking |
| **`speak()` calls `stopSpeaking()` first, unconditionally** | And `stopSpeaking` cancels twice with a `resume()` between, because one `cancel()` can leave a paused item on a Chrome queue to resume later |
| **Speech stops at every attention boundary** | A new chat turn, the microphone opening (the page would otherwise hear itself), a hidden tab, and unload |
| **`interrupted` and `canceled` are not errors** | They are what our own stop produces. Anything else is reported |
| **Every inline handler the markup names is exported and attached** | A function that stops being exported must fail a test, not become a dead button |
| **Building the timeline renders it** | `setupTimeline` once left step 0 to its caller: the inspector path called it, the chat path did not, and a turn showed the new sentence's characters above the previous turn's posteriors. A timeline not showing a step is not in a valid state |
| **No limit is transcribed from the server** | `config.js` applies the served object. The browser copy of a hand-written limit is the one nobody tests |

## Usage

```bash
# From the project root
.venv/bin/python -m ailm.dashboard
```

Static assets are served `no-store`, so an edit here is live on reload. That header
is not a convenience: with normal caching, a page verified after a change runs the
code from before it, a fix looks broken, a break looks fixed, and no test can see
it because the stale copy lives in the browser while every test reads the file.

## Testing

Run from the project root, with the project's own interpreter.

```bash
.venv/bin/python -m pytest tests/test_dashboard.py -q --no-cov
```

The structural assertions — module graph, line bounds, handler resolution, state
declarations — live there alongside the behavioral tests. Structural checks alone
passed throughout a period when the Level 5 classifier ranked below the
majority-class baseline, so every behavioral test carries a negative control.

## Best practices

- Put new behavior in the module that already owns the concern; add a module only
  when a genuinely new one appears, and wire it from `main.js` in the same change.
- Splitting a module that approaches the line bound is the intended response.
  Raising the bound is not.
- Verify in a real browser. Several defects in this tree's history passed every DOM
  assertion and were visible only on the page.

## See Also

- [`README.md`](README.md) — quick reference for these modules
- [`../AGENTS.md`](../AGENTS.md) — markup and stylesheet rules
- [`../../AGENTS.md`](../../AGENTS.md) — dashboard operating rules
- [`../../README.md`](../../README.md) — the full dashboard guide, including the API and speech lifecycle
