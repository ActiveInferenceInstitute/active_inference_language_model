# AILM Interactive Dashboard

A single-page inspector and bidirectional chat over the Level 1–6 analyses.
Type or speak a sentence to inspect what each implemented level reports. This
is not an end-to-end L1–L6 reply cascade: only the L3→L1g policy rollout
produces a reply. L2, L4, L5, and L6 analyze the input for inspection, and their
posteriors do not feed that rollout.

The connected runtime has a separate [Python API and CLI](../../../docs/design/connected_runtime.md).
This dashboard retains the component inspection and separate reply rollout
described above.

## One dashboard URL, with an optional server-free export

Open the dashboard URL once. **Analysis & chat** is the default view;
**Matrix explorer** embeds the generated standalone explorer at `/explorer/`
inside the same page. Switching views stops speech. A missing export reports
its build commands in the view and can be retried after generation. The server
serves only that exact generated HTML; it does not expose the output directory.

There are exactly two browser artifacts in this inventory. The connected
Python API and CLI do not add a browser surface.
Both read the *same* builders out of `ailm.levels.*` — `engine.build_letter_transition`
**is** `ailm.levels.letter.build_letter_transition`, the same function object, and
`tests/test_interface_inventory.py` fails if that stops being true.

| | this dashboard | `output/web/ailm.html` |
|:--|:--|:--|
| Levels | 1 through 6 | letter, generation, the Level 4 partition |
| Chat | yes, bidirectional, with speech in and out | no |
| Needs | a local Python server | nothing; opens from `file://` |
| Inference runs in | `engine.py` | JavaScript, over the exported matrices |
| Role | **the canonical browser interface** for independent component inspection and development | **a generated, ignored local demo** for exploration and QA; not a tracked or distributed publication artifact |

Use this dashboard URL for both views; no separate file tab is needed.
Before opening Matrix explorer, generate its inputs from the checkout root:

```bash
.venv/bin/python scripts/export_web_model.py
.venv/bin/python scripts/build_web_ailm.py
```

These commands write artifacts and do not open a browser. The standalone file
remains useful when Python is unavailable. An installed package without the
checkout's generated output still supports Analysis & chat and explains why
Matrix explorer is unavailable.

Use this dashboard when you want to inspect the full local analysis surface.
Generate `ailm.html` when a local, server-free smoke demo is useful; rebuild it
rather than treating it as a tracked deliverable.

The standalone page reimplements the letter recursion in JavaScript, which is the
one real hazard in having two: it could drift into being a second model that
agrees with nothing. Two things hold it: its first act is to recompute
`stage1_letter_perplexity` from its own inlined matrix and refuse to render if
that disagrees with the ledger, and the inventory test pins its alphabet and its
transition to the shipped ones. Rebuild it with
`scripts/export_web_model.py` then `scripts/build_web_ailm.py`, in that order,
after any re-record.

**Do not add a third browser interface.** If a new browser surface is warranted, declare it in
`DECLARED_INTERFACES` with what distinguishes it; the test refuses an undeclared
browser surface. Keep its scope distinct from the connected runtime's Python API and CLI.

```bash
.venv/bin/python -m ailm.dashboard              # http://127.0.0.1:8080/
.venv/bin/python -m ailm.dashboard 9000         # a different port
AILM_DASHBOARD_PORT=9000 \
    .venv/bin/python -m ailm.dashboard          # same, from the environment
AILM_DASHBOARD_MAX_SPOKEN_CHARS=120 \
    .venv/bin/python -m ailm.dashboard          # override any setting
```

Precedence is argument, then environment, then the shipped default. The entry
point holds no defaults of its own: it used to carry `DEFAULT_PORT = 8080` and
pass it to the server unconditionally, so `AILM_DASHBOARD_PORT` was read,
validated, and discarded before the socket opened. Nothing raised — the server
came up on 8080 while the operator believed they had moved it.

Run the package entry point, `python -m ailm.dashboard`, to apply configuration
validation and bind precedence. The package resolves its public API lazily;
static assets and configuration can be served before a computation endpoint
loads the numerical engine.

The server module is not an entry point: `python -m ailm.dashboard.server`
refuses with guidance instead of opening a second, unvalidated launch path.

No build step, no bundler, no external assets, no network calls beyond the local
server. The browser loads native ES modules directly.

---

## Layout

| File | What it is |
|:-----|:-----------|
| `config.py` | Every tunable, one default each, env-overridable |
| `engine.py` | Level 1–6 input analyses, the separate L3→L1g rollout, `converse()` |
| `server.py` | The HTTP API and static tree |
| `capture.py` | The manuscript capture receipt: the served-route table and the digests that bind a browser raster to the static tree |
| `__main__.py` | `python -m ailm.dashboard [port] [host]` |
| `static/index.html` | Markup only — no `<style>`, no `<script>` body, no `style=` attribute |
| `static/dashboard.css` | Styles |
| `static/js/main.js` | Entry point: config, lifecycle, global handler wiring |
| `static/js/config.js` | Fetches `/api/config`, applies bounds to controls |
| `static/js/state.js` | Cross-module state, so no module owns another's variables |
| `static/js/status.js` | The shared status region |
| `static/js/speech.js` | Speech output: one utterance, always terminated |
| `static/js/stt.js` | Speech recognition for the inspector input |
| `static/js/matrices.js` | Generative matrices as canvas heatmaps |
| `static/js/inference.js` | `/api/infer`, the timeline, every analysis panel |
| `static/js/chat.js` | The bidirectional transcript |

`index.html` previously held its markup, CSS, and JavaScript inline. One file
holding three languages cannot be reviewed a concern at a time, and a change to
any of them showed up as a diff against all three.

Getting the `<style>` block out was only most of the job: 48 `style=` attributes
stayed behind — the same three button backgrounds transcribed across eight
buttons, the same muted twelve-pixel label at every chart. An attribute is CSS in
the markup that does not announce itself, cannot be reused, and cannot be
overridden without `!important`. They now live in `dashboard.css` as named
classes, and a test refuses the next one.

`tests/test_dashboard.py` asserts the markup carries no inline `<style>`,
`<script>` body **or** `style=` attribute, that every class the markup names is
defined in the stylesheet, that every module is reachable from `main.js` with no
orphans, and that no module exceeds 400 lines.

**Why inline `onclick` handlers, and how they resolve.** ES modules do not
populate the global scope, so `main.js` attaches every function the markup names
to `window`, explicitly and in one list. That is a deliberate trade: inline
handlers keep a control's behavior readable beside the control, where a block of
`addEventListener` calls would move that information somewhere the reader has to
go find. A test checks every inline handler resolves, so a function that stops
being exported fails a test rather than becoming a dead button.

---

## Configuration

`config.py` holds every tunable. Each field reads `AILM_DASHBOARD_<FIELD>` from
the environment; a value that does not parse is **refused**, not defaulted,
because a typo in a limit is exactly the case where a quiet fallback hides the
mistake.

| Setting | Default | What it bounds |
|:--------|--------:|:---------------|
| `host` | `127.0.0.1` | Bind address. Explicit loopback names and addresses only |
| `port` | `8080` | Listen port |
| `max_request_bytes` | `65536` | Largest POST body; an unbounded client-declared length is an allocation |
| `max_concurrent_requests` | `4` | Maximum simultaneous request-handler threads |
| `max_requests_per_minute` | `300` | Rolling authenticated API request ceiling per local client |
| `request_timeout_seconds` | `30` | Socket timeout for one client |
| `max_continuation_words` | `200` | Rollout ceiling; the rollout runs inside the request handler |
| `default_reply_words` | `8` | Reply length the interface opens on |
| `max_history_turns` | `50` | Turns retained; the transcript is replayed every request |
| `max_spoken_chars` | `400` | Longest utterance; a long rollout outlives its turn |
| `speech_watchdog_ms` | `30000` | Force-clear of the synthesis queue |
| `default_narrative_alpha` | `0.5` | Starting L6→L5 channel weight |
| `alpha_grid` | `0, 0.05, 0.1, 0.25, 0.5` | Sweep weights; `0.0` required as the identity anchor |

`DashboardConfig` is frozen and validates on construction: a port outside
1–65535, a non-positive limit, a default reply longer than the ceiling, an alpha
outside `[0, 1]`, a grid missing `0.0`, or a grid out of order all raise.

**Nothing is written twice.** `GET /api/config` serves the same object the
server enforces requests against, and `js/config.js` applies it to the controls
at load. Previously the reply-length input carried `max="200"` and the speech
cap was a `const` in a script block, both transcribed by hand from constants
they had no link to. A limit written twice is a limit that will disagree with
itself, and the browser copy is the one nobody tests.

---

## API

| Endpoint | Body | Returns |
|:---------|:-----|:--------|
| `GET /api/config` | — | The public settings above |
| `GET /api/static_matrices` | — | `A_letter`, `A_g2p`, `B_g2p`, `D_g2p`, `A_case`, `B_case`, `D_case`, `clause_prior` |
| `POST /api/infer` | `{text, n_words?, narrative_phase?, narrative_alpha?}` | Full hierarchy for one sentence |
| `POST /api/chat` | `{message, history?, n_words?, narrative_phase?, narrative_alpha?}` | `reply`, `read_as`, `hierarchy`, `turn_index` |

Bodies over `max_request_bytes` return 413; a malformed `Content-Length`, a
non-JSON body or a JSON array returns 400; unknown endpoints 404. Numeric fields
that do not parse fall back to their configured defaults rather than raising,
since a chat client can send anything.

For `/api/chat`, omitting `narrative_phase` is the interesting case: the engine
derives it from the turn index, which is the whole reason Level 6 means anything
in a conversation.

---

## What this artifact is, and is not

**A demonstrator, not evidence.** Nothing here backs a manuscript number. The
dashboard has no entry in `data/claims.json` and no `claim:` tag anywhere,
deliberately: every quantitative claim in the manuscript comes from a recorder
under `scripts/record_*_claims.py` running on a declared split, and the dashboard
runs on whatever was typed.

**The reply is a rollout, not an answer.** It is produced only by the L3→L1g
generation path: Level 3's lexicon preference against Level 1g's letter habit,
resumed from your text. The L2 phoneme, L4 case, L5 clause, and L6 narrative
panels analyze the same input but do not condition the reply. They are adjacent
diagnostics, not evidence of an end-to-end cascade. The model has no semantics,
no retrieval, and nothing that rewards
answering a question — a few thousand free parameters emitting English letter
statistics shaped by a 2,000-word lexicon. Expect word salad. The line under
each reply is the part worth reading: it is what the level stack inferred about
*your* turn, counted from the first turn as the header counts, with any
part-of-speech tag the lexicon had to default reported rather than hidden. The
interface says so on the page, and a test asserts that disclaimer stays there.

**Its Level 5 panel is a harder setting than the Level 5 claim:**

| | `scripts/probe_l5_widened_alphabet.py` | this dashboard |
|---|---|---|
| Case roles | **oracle** — `case_derived` from gold UD | **inferred** — argmax of the L4 posterior |
| Part of speech | gold UD `upos` | MLE lexicon over UD EWT dev, NOUN on miss |
| Type ranking (AUC) | 0.93 held out, printed by the probe | materially lower; floor asserted in tests |

The probe asks *given true syntactic case roles (UD-derived), does surface
order carry sentence type*.
The dashboard asks *can the stack recover type end to end from raw text*. The
gap is the cost of Level 4 inference, not a defect to tune away. Do not cite a
dashboard reading as support for the Level 5 result.

---

## Level 6 is an input, not an inference

A narrative phase is a document-position tercile. A single sentence has no
position, so in the inspector the phase is a **control**: a dropdown beside an α
slider. The panel shows `D'(α) = (1−α)·P_clause + α·P(type | phase)` across all
three phases at the selected α, against the phase-reversed scramble.

In **chat** the document-position input is real. A turn in a transcript does
have a position, so with `narrative_phase` unset the engine derives it from the
turn index and the displayed L6→L5 analysis moves as the exchange progresses.
That intervention changes the clause-analysis prior; it still does not enter
the separate L3→L1g reply policy.

Two properties are visible rather than asserted, both pinned by tests:

- **α = 0 is the exact identity anchor.** Every phase and the control return the
  unmodified clause prior, bit for bit.
- **The scramble fixes the middle phase.** Reversal on three elements has a fixed
  point, so `middle` is *necessarily* identical between message and control, for
  any corpus. The control is informative at `opening` and `closing` only.

---

## Speech output stops when the turn does

`speechSynthesis` is a queue, not a channel. `speak()` appends and `cancel()`
clears only what is pending at that instant, so two turns in quick succession —
or one turn speaking from two paths — leave the queue draining after the turn is
over. The interface looks idle and the machine is still talking.

`js/speech.js` is the only module permitted to queue an utterance, and `speak()`
calls `stopSpeaking()` first, unconditionally. `stopSpeaking` cancels twice with
a `resume()` between, because a single `cancel()` can leave a paused item on a
Chrome queue to resume later. Speech also terminates at every boundary where the
page stops being what the reader is attending to: a new chat turn, the
microphone opening (the page would otherwise hear itself), a hidden tab, and
unload. A **Stop speaking** button and a live indicator sit in the chat header,
beside **Export**, which downloads the transcript as the page holds it: every
turn's text and, for each reply, what the analysis read the preceding user turn
as (`read_as`, the turn index and the rollout seed). It is a local JSON
download built from the page's own transcript store; it sends nothing to the
server, and a test holds it to that.

`interrupted` and `canceled` are what our own stop produces, so they are not
reported as errors; anything else is.

The server is local-only by enforcement, not convention. Non-loopback bind
addresses are refused. Loading the static entry page issues a random per-process
credential as an HTTP-only, strict same-site cookie; every `/api/*` route requires
that credential, and POST routes additionally require `application/json` and an
exact same-loopback-origin browser request. API responses never send wildcard
CORS headers. Command-line API clients can set
`AILM_DASHBOARD_API_TOKEN` to a long random value and present it as
`Authorization: Bearer <token>`. A bounded worker semaphore, rolling request
limit, and per-socket timeout keep a stalled or retrying client from creating an
unbounded thread or request queue. Unexpected engine exceptions are logged with
detail and returned only as a generic 500 response.

**What is verified, and what is not.** Driving three turns in real Chrome with
the synthesis queue instrumented shows one `speak` per turn and an empty queue
afterwards, nothing speaking, the stop control correctly disabled. The *audible*
mid-utterance stop cannot be driven from automation: Chrome refuses synthesis
without a user gesture and returns `not-allowed`, so no utterance starts under a
scripted click. That path is the same `stopSpeaking()` every other boundary
calls, checked by reading the code rather than by hearing it.

---

## Invariants worth not breaking

- **The widened alphabet is shared, not re-declared.** `VERB_SYMBOL`,
  `AUX_SYMBOL`, `N_WIDENED_SYMBOLS`, the extraction and the Markov fit all live
  in `ailm.levels.widened`; this engine and
  `scripts/probe_l5_widened_alphabet.py` import them. `tests/test_widened.py`
  parses both consumers and fails on any redeclaration, so the check is "is
  there still only one" rather than "do the copies still agree" — a matching
  second copy is a divergence that has not happened yet.
- **The part-of-speech lexicon comes from `ailm.corpus.ud`.** The dashboard asks
  `build_upos_lexicon` rather than counting UD tags itself, so the tag it infers
  from is the tag anything else asking the same question gets.
- **Resolve the part of speech before any case-emission fallback.** `AUX` has no
  column in the case emission. Defaulting an unmapped tag *first* rewrites every
  auxiliary to `NOUN`, makes symbol 9 unreachable, and deletes
  subject-auxiliary inversion without any error surfacing.
- **Punctuation contributes no symbol.** The chains are fitted with UD `PUNCT`
  skipped; inference must skip them too.
- **A defaulted tag is reported, never hidden.** `upos_source` distinguishes
  `lexicon` from `fallback`, and the interface marks the difference.
- **Bounds are enforced at the engine and server, not the caller.** Rollout
  length, phase, α and history all arrive off the wire.
- **No raw TeX in the page.** There is no math renderer and no external assets,
  so `$...$` renders as literal source. Use Unicode.
- **A button variant must out-specify the base rule.** The base is
  `button.btn-primary` (specificity 0,1,1), so a bare `.btn-stop` (0,1,0) loses
  regardless of source order and the button renders the default green. This is
  not hypothetical: it is what happened the moment the colors left their inline
  attributes, and no DOM test could see it — the classes were present, the markup
  was right, the page was the wrong color. Qualify variants with `button`.
- **Static assets are served `no-store`.** `SimpleHTTPRequestHandler` honors
  `If-Modified-Since`, which is correct for a file server and wrong here: the
  browser keeps a stylesheet or module across edits, so a page verified after a
  change runs the code from before it. A fix then looks broken and a break looks
  fixed, and no test can see it, because the stale copy lives in the browser
  while every test reads the file. Measured live — an edited stylesheet served
  byte-identical to disk while the loaded CSSOM held the previous rules, through
  a reload and a cache-busted URL, since the query string busts the document and
  not its subresources.
- **Building the timeline renders it.** `setupTimeline` left step 0 to its
  caller; the inspector path called it and the chat path did not, so a turn
  showed the new sentence's characters above the previous turn's posteriors. A
  timeline not showing a step is not in a valid state, so the two are one
  operation and no caller can do half of it.
- **The page prints no measured number of its own.** The Level 5 tag read
  "AUC 0.93" — the probe's figure, on gold case roles, printed beside a panel
  that infers them. A manuscript number typed into an interface, where no
  `{{TOKEN}}` hydration would ever reach it.

---

## The manuscript figure is a capture of this page

The manuscript's dashboard figure (`fig:dashboard_observation`) is a real
screenshot of this page after one chat turn, retained under
`data/dashboard_capture/` with a receipt that binds the raster's bytes and
decoded pixels to a digest of every asset the server handed the browser:
`index.html`, `dashboard.css`, each module under `js/`, and the explorer
export. The figure generator refuses the raster when any binding fails, so an
edit to the static tree is also a request for a new capture, never for an
edited receipt.

`scripts/capture_dashboard.py` records that pair. It runs under a scratch
interpreter with Playwright and Pillow (the project `.venv` stays locked to
the release lock file), launches the dashboard with the project interpreter,
drives the locally installed Chrome headless through one turn with speech
off, fetches every served route and refuses any whose bytes differ from the
file on disk, and finally has the project interpreter validate what it wrote:

```bash
python3 -m venv /tmp/ailm-capture && /tmp/ailm-capture/bin/pip install playwright pillow
/tmp/ailm-capture/bin/python -I scripts/capture_dashboard.py
MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py
```

The receipt names the commit the sources were compared with and flags, per
asset, whether the served bytes equal that commit's blob; a capture taken
from an edited working tree flags those files false until the commit that
carries both. Rasters are not byte-reproducible across browser builds or font
stacks, so the receipt records the browser build, scale and viewport: the
binding is to this capture, and the procedure is what a reader repeats.

## Tests

`tests/test_dashboard.py` plus `tests/test_widened.py` on the shared alphabet,
and `tests/test_dashboard_capture.py` for the capture receipt.
The structural checks came first and passed throughout a period when the Level 5 classifier ranked *below* the
majority-class baseline: shape assertions cannot see a degraded model. Every
behavioral test added since carries a negative control — a variant that must
**not** fire — so a test cannot pass by asserting something true of any model.

They cover the level behavior, the chat turn model, the speech lifecycle, the
configuration surface and its refusals, the entry point's precedence chain
(argument over environment over default), and the page's structure: no inline
CSS or JS, no `style=` attribute, no class the stylesheet does not define, no
variant rule the base out-specifies, no modal dialog, no heading that titles two
elements, no measured number in a level tag, no orphan modules, no module over
400 lines, no dead element or handler references, and no shared state
redeclared outside `state.js`.

The launch boundary is tested in fresh interpreters: importing the entry
point leaves numpy, jax and pymdp out of `sys.modules`; the public API
resolves the original function objects; static, credential and configuration
routes serve before any numerical import; the service adapters forward to
the engine unchanged; and the server module refuses direct execution.
