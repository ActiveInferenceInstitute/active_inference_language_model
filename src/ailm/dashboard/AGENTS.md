# AGENTS.md — src/ailm/dashboard/

Operating rules for the interactive dashboard. [`README.md`](README.md) is the full
guide — the two browser interfaces, every configuration key, the four endpoints, the speech
lifecycle, and the invariants worth not breaking. This file is the short version an
agent needs before touching anything here, and it does not restate the README.

## Overview

A single-page local inspector and bidirectional chat over the Level 1–6 analyses,
served by a loopback-only Python HTTP server. It is a **demonstrator**: no
manuscript number comes from it, it holds no `claim:` tag, and it has no entry in
`data/claims.json`.

The [connected runtime's Python API and CLI](../../../docs/design/connected_runtime.md)
are separate entry points. This browser dashboard retains independent component
inspection and the separate L3→L1g reply rollout.

## Ownership boundary

| Concern | Owner |
| --- | --- |
| Level builders and matrices | `ailm.levels.*` — `engine.py` re-exports the *same function objects*, it does not reimplement them |
| The widened alphabet (`VERB_SYMBOL`, `AUX_SYMBOL`, `N_WIDENED_SYMBOLS`) | `ailm.levels.widened`; this engine imports it and may never redeclare it |
| The part-of-speech lexicon | `ailm.corpus.ud.build_upos_lexicon` |
| Every tunable and its bound | `config.py`; `GET /api/config` serves the same object the server enforces against |
| The browser interface inventory | `DECLARED_INTERFACES` in `tests/test_interface_inventory.py` |
| The generated standalone page | `scripts/export_web_model.py` then `scripts/build_web_ailm.py` — not this package |

## Invariants (never violate)

| Invariant | Why |
| --- | --- |
| **No third browser interface** | The two browser surfaces are declared and distinguished; the inventory test refuses an undeclared browser surface. The connected Python API and CLI are outside this browser inventory |
| **No measured number printed on the page** | A manuscript figure typed into an interface is a number no `{{TOKEN}}` hydration would ever reach. A Level 5 tag once carried the probe's gold-case AUC beside a panel that infers case |
| **No inline `<style>`, `<script>` body, or `style=` attribute in `index.html`** | Enforced by `tests/test_dashboard.py`; CSS in markup cannot be reused or overridden without `!important` |
| **Only `js/speech.js` queues an utterance, and `speak()` calls `stopSpeaking()` first** | `speechSynthesis` is a queue; two turns in quick succession leave it draining after the turn is over |
| **The server binds loopback only, and refuses anything else** | Explicit enforcement, not convention |
| **Static assets are served `no-store`** | A cached stylesheet or module makes a verified page run pre-change code, invisibly to every test |
| **A CSS variant must out-specify its base rule** | The base is `button.btn-primary`; a bare `.btn-stop` loses regardless of source order |
| **Nothing here backs a claim** | Do not cite a dashboard reading as support for a level result — its Level 5 panel runs on *inferred* case roles where the probe uses gold ones |

## File organization

| File | Concern |
| --- | --- |
| `__init__.py` | Lazy public API; returns the original server or engine function on demand |
| `__main__.py` | `python -m ailm.dashboard [port] [host]`; holds no defaults of its own |
| `config.py` | `DashboardConfig` — frozen, env-overridable, validating on construction |
| `engine.py` | Level 1–6 input analyses, the separate L3→L1g rollout, `converse()` |
| `server.py` | The HTTP API, the static tree, the credential and rate limits |
| `capture.py` | The manuscript capture receipt — served-route table and raster bindings; `scripts/capture_dashboard.py` assembles it, `ailm.figures.dashboard_observation` validates it |
| `static/` | The browser tree — see [`static/AGENTS.md`](static/AGENTS.md) |

## Usage

Run from the project root, with the project's own interpreter.

```bash
.venv/bin/python -m ailm.dashboard              # http://127.0.0.1:8080/
.venv/bin/python -m ailm.dashboard 9000         # a different port
AILM_DASHBOARD_PORT=9000 .venv/bin/python -m ailm.dashboard
```

Precedence is argument, then environment, then the shipped default. Run the
package entry point for that configuration contract. Static assets and
configuration do not import the numerical engine; computation requests load it
through the default `DashboardServices` adapters.

## Testing

```bash
.venv/bin/python -m pytest tests/test_dashboard.py -q --no-cov
.venv/bin/python -m pytest tests/test_widened.py -q --no-cov
.venv/bin/python -m pytest tests/test_interface_inventory.py -q --no-cov
```

Structural checks alone are not enough here, and this package is the proof: they
passed throughout a period when the Level 5 classifier ranked below the
majority-class baseline. Every behavioral test added since carries a negative
control — a variant that must *not* fire.

## See Also

- [`README.md`](README.md) — the full guide, including the API and the configuration table
- [`static/AGENTS.md`](static/AGENTS.md) — rules for the browser tree
- [`../../../docs/README.md`](../../../docs/README.md) — the documentation index that reaches this artifact
- [`../levels/AGENTS.md`](../levels/AGENTS.md) — the builders this engine re-exports
- [`../../../AGENTS.md`](../../../AGENTS.md) — project invariants
