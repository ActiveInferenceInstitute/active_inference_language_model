# Configuration reference

Every knob the shipped code reads, in one place: the Python-level level
hyperparameters, then every `AILM_*` environment variable, grouped by the
surface that consumes it. Each row names the reader so the behavior can be
checked in the source rather than taken from this page. Nothing here changes a
recorded number: the measured results are bound to the defaults, and a changed
knob is a new measurement that needs its own recorder run.

Two rules hold across every reader unless a row says otherwise:

- A value that does not parse, or that violates the reader's bound, is refused
  with a named error. No reader substitutes a default for a bad value, because
  a silently corrected typo is indistinguishable from a deliberate setting.
- Paths must be absolute. Recorder isolation and the call-store protocol refuse
  a relative path so that behavior cannot depend on a subprocess working
  directory.

## Level hyperparameters (`ailm.config`)

[`src/ailm/config.py`](../../src/ailm/config.py) owns the tunable
hyperparameters of the letter and phoneme builders as the immutable
`LevelConfig` dataclass. There is no environment variable for these: a
configuration is constructed in Python and passed to a builder, and
`DEFAULT_LEVEL_CONFIG` is what every builder uses when none is passed. Its
matrices are byte-identical to the historical hardcoded implementations, which
`tests/test_config.py` pins by digest.

| Field | Default | Meaning | Refused when |
|---|---|---|---|
| `smoothing` | `1.0` | Add-k pseudocount for every count table (letter bigrams, phone bigrams, phone unigrams, G2P start prior); `(c + k) / (n + kV)` per cell | not finite or not strictly positive |
| `emission_smoothing` | `None` | Separate pseudocount for the G2P letter-given-phone emission; `None` means use `smoothing` | not finite or not strictly positive when set |
| `test_every` | `None` | Held-out stride for the G2P split (index `i` is test iff `i % test_every == 0`); `None` means the stride recorded in the vendored artifact | not an integer, or below `2` |

The module docstring derives why the pseudocount is a monotone dial between the
corpus column and the uniform column; `tests/test_config.py` asserts that
monotonicity rather than restating it.

## Library runtime

Read by the installed package and by any script that imports it.

| Variable | Reader | Meaning | Unset |
|---|---|---|---|
| `AILM_CLAIMS_PATH` | `src/ailm/claims.py` (`load_ledger`), `src/ailm/ledger_lock.py`, `src/ailm/replay_provenance.py` | Sends claim writes to an isolated ledger instead of the tracked `data/claims.json`; the single-writer lock moves beside that ledger, and a non-default ledger is treated as non-publication, so network-claim bindings refuse to be written against it | `data/claims.json` under the project root; a relative path is refused |
| `AILM_CALL_CACHE_PATH` | `src/ailm/benchmarks/ollama.py`, `scripts/warm_call_cache.py` | Process-local location of the memoized external-call store when a runner enables the cache; an explicit `path` argument wins, and publication recorders refuse any store other than the canonical one | the canonical `cache/bench_calls.jsonl`; a relative path is refused, and the warmer treats a blank value as unset |
| `AILM_OLLAMA_ENDPOINT` | `src/ailm/benchmarks/constants.py` | Local Ollama loopback origin, resolved and validated once at import so requests, configuration and replay share one origin; the release verifier records it in the receipt | `http://localhost:11434`; an empty or non-loopback value is an error, never a fallback |
| `AILM_OPENROUTER_BUDGET_FILE` | `src/ailm/benchmarks/openrouter_budget.py` | Path of the shared JSON budget ledger (`ailm-openrouter-budget-v1`) that pins provider and pricing and reserves the worst-case charge before each hosted request | no budget is enforced only when both budget variables are unset; one without the other is refused |
| `AILM_OPENROUTER_BUDGET_LIMIT_USD` | `src/ailm/benchmarks/openrouter_budget.py` | Independently configured spending cap; must be a finite, positive decimal equal to the ledger's own limit | see the row above |
| `AILM_CORPUS_CACHE_DIR` | `src/ailm/resources.py` | Destination for raw corpus downloads (CMUdict, UD EWT) | a source checkout keeps `vendor/`; an installed wheel uses the platform's per-user cache directory; an empty or relative value is refused |
| `AILM_BROWN_SOURCE_ZIP` | `src/ailm/corpus/brown.py` | Candidate path of the pinned NLTK Brown archive, searched before the NLTK data roots and then digest-verified | only the NLTK data roots are searched; when nothing verifies, the error names this variable |
| `AILM_PROJECT_ROOT` | `src/ailm/figures/_common.py` | Overrides the project root holding `output/`, `vendor/` and `docs/manuscript/` when a figure generator runs from another working directory; the path is resolved but not required to be absolute | walk up from the figures package |

## Recorders and scripts

| Variable | Reader | Meaning | Unset |
|---|---|---|---|
| `AILM_CONFIRMATORY_ROOT` | `scripts/record_confirmatory_replication.py` | Fixture root whose `data/confirmatory` is scored instead of production; must be absolute and must come with `AILM_CLAIMS_PATH` and the opt-in below, otherwise the run is refused | the production root |
| `AILM_CONFIRMATORY_ALTERNATE_LEDGER` | `scripts/record_confirmatory_replication.py` | Explicit opt-in to fixture-mode shakedown, so an isolated run never starts by accident | only the exact value `1` enables it |
| `AILM_PATCHED_PYMDP_ROOT` | `scripts/probe_pymdp_sparse.py` (child worker) | Authenticated staged package root the patched sparse worker must import `pymdp` from; the parent process clears any inherited value and sets its own | an empty value makes the worker refuse to start |
| `TEMPLATE_REPO_ROOT` | `scripts/verify_release.py`, `scripts/z_generate_manuscript_variables.py`, `scripts/generate_figure_registry.py` | The `docxology/template` checkout that provides the `infrastructure` rendering package; an explicit value is final and never silently replaced by a fallback | the scripts search the documented candidate locations and refuse loudly when none holds `infrastructure/` |

The guides write `AILM_TEMPLATE_ROOT` as a shell variable for the operator's
own chosen checkout and pass it as `TEMPLATE_REPO_ROOT`; the code reads only
the latter. The other `AILM_*` names that appear in command examples
(`AILM_INPUT_JSON`, `AILM_RELEASE_RECEIPT`, `AILM_PUBLICATION_CONTRACT`,
`AILM_RELEASE_VERSION`, and similar) are likewise shell placeholders for the
reader's own values, not variables the code reads.

## Dashboard

Every field of `DashboardConfig` in
[`src/ailm/dashboard/config.py`](../../src/ailm/dashboard/config.py) reads
`AILM_DASHBOARD_<FIELD>`, uppercased. A blank value keeps the default; a value
that does not parse exits the launcher with a named error. The dashboard
README documents each field and the security model in full; the names are:

| Variable | Default |
|---|---|
| `AILM_DASHBOARD_HOST` | `127.0.0.1` (loopback names and addresses only) |
| `AILM_DASHBOARD_PORT` | `8080` |
| `AILM_DASHBOARD_MAX_REQUEST_BYTES` | `64 * 1024` |
| `AILM_DASHBOARD_MAX_CONTINUATION_WORDS` | `200` |
| `AILM_DASHBOARD_DEFAULT_REPLY_WORDS` | `8` (never above the continuation ceiling) |
| `AILM_DASHBOARD_MAX_HISTORY_TURNS` | `50` |
| `AILM_DASHBOARD_MAX_SPOKEN_CHARS` | `400` |
| `AILM_DASHBOARD_SPEECH_WATCHDOG_MS` | `30_000` |
| `AILM_DASHBOARD_DEFAULT_NARRATIVE_ALPHA` | `0.5` |
| `AILM_DASHBOARD_ALPHA_GRID` | `0.0, 0.05, 0.1, 0.25, 0.5` (ascending, within the unit interval, must include `0.0`) |
| `AILM_DASHBOARD_MAX_CONCURRENT_REQUESTS` | `4` |
| `AILM_DASHBOARD_MAX_REQUESTS_PER_MINUTE` | `300` |
| `AILM_DASHBOARD_REQUEST_TIMEOUT_SECONDS` | `30.0` |
| `AILM_DASHBOARD_API_TOKEN` | a random per-process token; an explicit value must be at least `32` bytes and is never echoed by the configuration endpoint |

The launcher's positional arguments (port, then host) override the
corresponding variables. See
[`src/ailm/dashboard/README.md`](../../src/ailm/dashboard/README.md).

## Release verifier and test opt-ins

The release runner starts every step with an explicit environment allowlist
(`RELEASE_ENV_ALLOWLIST` in [`scripts/verify_release.py`](../../scripts/verify_release.py))
and sets the following for its own child pytest steps. They are read only by
tests, so setting them by hand is how a developer opts a single test run into a
behavior the quick profile deselects.

| Variable | Set by the runner for | Read in | Effect |
|---|---|---|---|
| `AILM_RUN_RAW_CORPUS_REGENERATION=1` | the raw-corpus reproducibility step | `tests/test_provenance_checksums.py` | replay the vendored artifacts from the ignored raw corpora instead of skipping |
| `AILM_RUN_LIVE_OLLAMA_TESTS=1` | the full test step | `tests/test_benchmarks.py`, `tests/test_synthetic_benchmark.py` | let the opted-in tests reach the configured local Ollama endpoint; availability is checked inside each test body, never at collection |
| `AILM_RELEASE_MANDATORY_ENVIRONMENT=1` | the full test step | `tests/test_synthetic_benchmark.py`, `tests/test_gnn_conformance.py` | turn an environment gap (unserved probe, absent upstream GNN checkout) into a failure that names it, because the full profile forbids skips |
| `AILM_PUBLICATION_DERIVATIVE_CONTRACT` | the derivative route only | `tests/test_release_orchestration.py` | the reviewed external derivative contract under test |
| `AILM_PUBLICATION_DERIVATIVE_LIVE_PROOF` | the derivative route only | `tests/test_release_orchestration.py` | the live local proof the derivative audit produced |
| `AILM_GNN_UPSTREAM` | nothing (not forwarded by the runner) | `tests/test_gnn_conformance.py` | a direct test invocation's upstream GNN checkout; the full profile discovers its checkout through the test's candidate list instead |

The runner also strips any `AILM_OPENROUTER_BUDGET_*` value from its child
environment, so a release run can never make a hosted acquisition under a
budget inherited from the shell.

## Formal layer

`formal/Scripts/axiom_report.sh` reads two shell variables of its own:
`AILM_LAKE` (default `lake`) names the Lake executable, and `AILM_PYTHON`
(default `../.venv/bin/python`) names the interpreter that generates the axiom
script. Both exist so the release verifier and the formal tests can pin the
exact tools they audited.

## Where the rest lives

- Dependency pins, the per-test timeout, the coverage floor and the pytest
  markers: [`pyproject.toml`](../../pyproject.toml).
- The release-only uv configuration: [`release-uv.toml`](../../release-uv.toml).
- Paper metadata and render policy: [`docs/manuscript/config.yaml`](../manuscript/config.yaml).
- The quick and full profiles, what each needs, and the interpreter rule:
  [`guides/quickstart.md`](../guides/quickstart.md) and
  [`method/release_checklist.md`](../method/release_checklist.md).
