#!/usr/bin/env python3
"""Warm the memoized external-model call store, then say what to run next.

## Why this is a script rather than a paragraph

The `full` release profile enforces the `pyproject.toml` coverage floor, and roughly a point
and a half of that coverage lives in code that only executes when the memoized
calls are on disk. On a cold store the total lands near 91.87% with no failing
test naming the cause, so the first clean rebuild lost real time to a symptom
with exactly one source and no signpost.

Warming is therefore a documented prerequisite. It is also a few hours of model
calls with four ways to get it subtly wrong — wrong server, missing model, the
real ledger as the write target, or a resume that silently starts from nothing —
so it gets a script that checks all four before spending the first hour.

## What it does

1. Refuses to start unless the Ollama server answers and serves every declared
   probe. A run that fails on the third model after two hours is a run that
   should have failed in the first second.
2. Points ``AILM_CLAIMS_PATH`` at an isolated copy of the ledger, so a
   re-measurement cannot overwrite a recorded claim. Whether a value moved is
   the question worth asking, not one to answer silently in place.
3. Resumes. The store is content-addressed per call, so every answer already
   held is reused and only the remainder is measured.
4. Reports the delta against the recorded ledger when it finishes, and prints
   the exact commands for the rest of the rebuild.

Usage:
    .venv/bin/python scripts/warm_call_cache.py            # preflight, then warm
    .venv/bin/python scripts/warm_call_cache.py --check    # preflight only
    .venv/bin/python scripts/warm_call_cache.py --audit-strata-only
        # populate only the model/request-kind cells required by cache audit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RECORDER = PROJECT_ROOT / "scripts" / "record_external_benchmark_claims.py"
LEDGER = PROJECT_ROOT / "data" / "claims.json"
ISOLATED_LEDGER = PROJECT_ROOT / "cache" / "warming_claims.json"

OLLAMA_ENDPOINT = "http://localhost:11434"

AUDIT_STRATUM_CALIBRATION_N = 25

_MODEL_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}")


def _validated_ledger_snapshot(content: bytes, *, description: str) -> dict[str, object]:
    """Decode one exact v2 claims-ledger snapshot without repairing it on disk."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{description} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{description} has non-finite JSON constant {value}")

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{description} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} root must be an object")
    if payload.get("_schema") != "ailm-claims-v2":
        raise ValueError(
            f"{description} schema must be 'ailm-claims-v2', got {payload.get('_schema')!r}"
        )
    if not isinstance(payload.get("_description"), str) or not payload["_description"].strip():
        raise ValueError(f"{description} needs a non-empty _description")
    claims = payload.get("claims")
    if not isinstance(claims, dict):
        raise ValueError(f"{description} claims must be an object")

    from ailm.claims import ClaimsLedger

    for label, entry in claims.items():
        ClaimsLedger._validate_entry(label, entry)
    return payload


def _read_validated_ledger(path: Path, *, description: str) -> bytes:
    """Read and validate one ordinary claims-ledger file from the same bytes."""

    from _artifact_io import read_authenticated_regular_file

    content = read_authenticated_regular_file(path, description=description).content
    _validated_ledger_snapshot(content, description=f"{description} at {path}")
    return content


def _declared_probes() -> dict[str, str]:
    # Probe identity belongs to the shared registry. Importing the recorder just
    # to read its copy also imports every numerical estimator before preflight.
    from ailm.benchmarks.constants import PROBE_MODEL_TAGS

    return dict(PROBE_MODEL_TAGS)


def _cache_path() -> Path:
    from ailm.call_cache import default_cache_path

    return default_cache_path(PROJECT_ROOT)


def _warm_inventory(path: Path | None = None) -> int:
    """Return the number of validated current-schema records."""
    from ailm.call_cache import CallCache

    path = _cache_path() if path is None else path
    if not path.exists():
        return 0
    return len(CallCache(path))


def _publication_configuration() -> dict[str, object]:
    """Load and validate the recorder-owned publication protocol.

    The warmer must not carry a second workload constant.  Its denominator is
    the union of the exact request keys implied by this configuration and the
    benchmark fixtures below.  An unknown configuration field is refused rather
    than guessed at: a new protocol arm needs a corresponding planner change
    before progress can be reported truthfully.
    """
    import record_external_benchmark_claims as recorder

    # The warmer plans keys, it does not record a run, so it asks for the
    # configuration under the recorder's planning scope rather than inventing a
    # run id of its own. The repeat-trial arm contributes no warmable key either
    # way, and the planning scope cannot pass the recorder's publication scope
    # check, so a configuration built here can never be mistaken for a receipt's.
    configuration = recorder.publication_replay_configuration(run_scope=recorder.PLANNING_RUN_SCOPE)
    root_fields = {"call_protocol", "endpoint", "model_tags", "parameters", "schema"}
    if not isinstance(configuration, dict) or set(configuration) != root_fields:
        raise ValueError(
            "publication replay configuration fields changed; update the warm-key planner"
        )
    if configuration["schema"] != "ailm-network-recorder-configuration-v1":
        raise ValueError("publication replay configuration schema is not supported")

    parameters = configuration["parameters"]
    parameter_fields = {
        "measurement_protocol",
        "logical_models",
        "letter_n",
        "phoneme_n",
        "g2p_n",
        "calibration_n",
        "ensemble_n",
        "repeat_n",
        "seed",
        "run_scope",
        "cache_protocols",
        "cache_audit_sampling",
        "local_context_window",
        "context_budget_policy",
    }
    if not isinstance(parameters, dict) or set(parameters) != parameter_fields:
        raise ValueError("publication benchmark parameters changed; update the warm-key planner")
    if parameters["cache_audit_sampling"] != recorder.CACHE_AUDIT_SAMPLING:
        raise ValueError("publication cache audit sampling policy differs from the recorder")
    if (
        type(parameters["local_context_window"]) is not int
        or parameters["local_context_window"] != recorder.OLLAMA_NUM_CTX
    ):
        raise ValueError("publication local context window differs from the recorder")
    if parameters["context_budget_policy"] != recorder.OLLAMA_CONTEXT_BUDGET_POLICY:
        raise ValueError("publication context budget policy differs from the recorder")

    protocol = configuration["call_protocol"]
    endpoint = configuration["endpoint"]
    model_tags = configuration["model_tags"]
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("publication call protocol must be a non-empty string")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("publication endpoint must be a non-empty string")
    if (
        not isinstance(model_tags, list)
        or not model_tags
        or any(not isinstance(tag, str) or not tag for tag in model_tags)
        or len(model_tags) != len(set(model_tags))
    ):
        raise ValueError("publication model tags must be unique non-empty strings")

    declared = _declared_probes()
    if sorted(model_tags) != sorted(declared.values()):
        raise ValueError("publication model tags disagree with the recorder registry")
    # Purpose-bound cache protocols are recorder-owned (the two repeat-trial
    # records are written but never read through the memo).  Validate against
    # the recorder's own derivation rather than carrying a second copy here;
    # these protocols contribute no warmable key either way.
    cache_protocols = parameters["cache_protocols"]
    if (
        not isinstance(cache_protocols, list)
        or any(not isinstance(item, str) or not item for item in cache_protocols)
        or len(cache_protocols) != len(set(cache_protocols))
        or cache_protocols != sorted(cache_protocols)
        or cache_protocols != list(recorder._cache_protocols(recorder.PLANNING_RUN_SCOPE))
    ):
        raise ValueError("publication cache protocols disagree with the recorder registry")
    if parameters["run_scope"] != recorder.PLANNING_RUN_SCOPE:
        raise ValueError("publication configuration did not honor the planning run scope")
    if parameters["logical_models"] != sorted(declared):
        raise ValueError("publication logical models disagree with the recorder registry")
    if parameters["measurement_protocol"] != protocol:
        raise ValueError("measurement protocol disagrees with the call protocol")

    from ailm.benchmarks import OLLAMA_URL, PROTOCOL_VERSION

    if protocol != PROTOCOL_VERSION:
        raise ValueError("publication call protocol disagrees with the benchmark package")
    if endpoint.rstrip("/") != OLLAMA_URL.rstrip("/"):
        raise ValueError("publication endpoint disagrees with the benchmark package")
    return dict(configuration)


def _protocol_count(
    parameters: dict[str, object], name: str, *, sentinel: int | None = None
) -> int:
    """Read one exact integer protocol coordinate without accepting booleans."""
    value = parameters.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"publication parameter {name} must be an integer")
    if value < 0 and value != sentinel:
        raise ValueError(f"publication parameter {name} has unsupported value {value}")
    return value


def _publication_payloads(
    model_tag: str, configuration: dict[str, object]
) -> tuple[dict[str, object], ...]:
    """Plan every memoized request made by one publication benchmark model.

    This is a set-of-keys protocol, not a sum-of-sample-sizes protocol.  The
    short and long arms share some requests, phoneme positions can repeat a
    prompt, and calibration changes the payload with log-probability fields.
    Returning the concrete payloads lets :class:`CacheRequest` perform the same
    canonical key construction as the live transport.

    The repeat-agreement arm is deliberately absent: it calls
    ``post_generate_uncached`` twice per position precisely so the memo cannot
    force agreement.  Its sample is still resolved below so an invalid protocol
    coordinate fails before a percentage is printed.
    """
    from ailm.benchmarks import (
        LETTER_CENSUS,
        LETTER_TEXT,
        LETTER_WINDOW_LONG,
        LETTER_WINDOW_SHORT,
        TOP_LOGPROBS,
        phoneme_positions,
        unique_letter_positions,
    )
    from ailm.benchmarks.constants import SEED_OFFSETS
    from ailm.benchmarks.ollama import _generate_payload
    from ailm.benchmarks.prompts import (
        build_prompt,
        build_undelimited_letter_prompt,
        subsample_positions,
    )
    from ailm.levels.phoneme import g2p_split, load_phoneme_counts

    raw_parameters = configuration.get("parameters")
    if not isinstance(raw_parameters, dict):
        raise ValueError("publication configuration parameters are not an object")
    letter_n = _protocol_count(raw_parameters, "letter_n", sentinel=LETTER_CENSUS)
    phoneme_n = _protocol_count(raw_parameters, "phoneme_n")
    g2p_n = _protocol_count(raw_parameters, "g2p_n")
    calibration_n = _protocol_count(raw_parameters, "calibration_n")
    ensemble_n = _protocol_count(raw_parameters, "ensemble_n")
    repeat_n = _protocol_count(raw_parameters, "repeat_n")
    seed = _protocol_count(raw_parameters, "seed")

    pool_short, pool_long = unique_letter_positions(
        LETTER_TEXT,
        window=LETTER_WINDOW_SHORT,
        long_window=LETTER_WINDOW_LONG,
    )
    resolved_letter_n = len(pool_short) if letter_n == LETTER_CENSUS else letter_n
    letters = subsample_positions(pool_short, resolved_letter_n, seed)
    letters_long = subsample_positions(pool_long, resolved_letter_n, seed)
    if [gold for _prefix, gold in letters] != [gold for _prefix, gold in letters_long]:
        raise ValueError("publication short and long letter positions are not aligned")

    counts = load_phoneme_counts()
    _train_pairs, test_pairs = g2p_split(counts)
    phones = subsample_positions(phoneme_positions(counts, test_pairs), phoneme_n, seed)
    words = subsample_positions(test_pairs, g2p_n, seed)
    calibration = subsample_positions(pool_short, calibration_n, seed)
    ensemble = subsample_positions(pool_short, ensemble_n, seed)
    # Validate the uncached arm's declared sample even though it contributes no
    # memo key.  Otherwise a malformed repeat coordinate could still receive a
    # confident progress percentage.
    subsample_positions(pool_short, repeat_n, seed)

    payloads: list[dict[str, object]] = [
        _generate_payload(model_tag, build_prompt("letter", "the"), "letter", 32, seed)
    ]
    payloads.extend(
        _generate_payload(model_tag, build_prompt("letter", prefix), "letter", 32, seed)
        for prefix, _gold in letters
    )
    payloads.extend(
        _generate_payload(model_tag, build_prompt("letter", prefix), "letter", 32, seed)
        for prefix, _gold in letters_long
    )
    payloads.extend(
        _generate_payload(
            model_tag,
            build_undelimited_letter_prompt(prefix),
            "letter",
            32,
            seed,
        )
        for prefix, _gold in letters
    )
    payloads.extend(
        _generate_payload(model_tag, build_prompt("phoneme", prefix), "phoneme", 32, seed)
        for prefix, _gold in phones
    )
    payloads.extend(
        _generate_payload(
            model_tag,
            build_prompt("g2p", word),
            "g2p",
            max(16, len(gold) * 8 + 4),
            seed,
        )
        for word, gold in words
    )
    for prefix, _gold in calibration:
        payloads.append(
            {
                **_generate_payload(
                    model_tag,
                    build_prompt("letter", prefix),
                    "letter",
                    32,
                    seed,
                ),
                "logprobs": True,
                "top_logprobs": TOP_LOGPROBS,
            }
        )
    for offset in SEED_OFFSETS:
        payloads.extend(
            _generate_payload(
                model_tag,
                build_prompt("letter", prefix),
                "letter",
                32,
                seed + offset,
            )
            for prefix, _gold in ensemble
        )
    return tuple(payloads)


def _publication_call_keys(*, identities: dict[str, str] | None = None) -> frozenset[str]:
    """Return the exact identity-bound cache-key union for publication replay."""
    from ailm.benchmarks import resolved_model_identity
    from ailm.call_cache import CacheRequest

    configuration = _publication_configuration()
    model_tags = configuration["model_tags"]
    endpoint = configuration["endpoint"]
    protocol = configuration["call_protocol"]
    if (
        not isinstance(model_tags, list)
        or not isinstance(endpoint, str)
        or not isinstance(protocol, str)
    ):
        raise ValueError("validated publication configuration changed type")

    if identities is not None and set(identities) != set(model_tags):
        raise ValueError("provided model identities do not cover the publication registry exactly")
    resolved: dict[str, str] = {}
    for tag in model_tags:
        identity = (
            resolved_model_identity(tag, endpoint=endpoint)
            if identities is None
            else identities[tag]
        )
        if not isinstance(identity, str) or _MODEL_IDENTITY.fullmatch(identity) is None:
            raise ValueError(f"immutable model identity for {tag} must be a full sha256 digest")
        resolved[tag] = identity
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("registered model tags resolved to duplicate immutable identities")

    keys = {
        CacheRequest(
            payload=payload,
            endpoint=endpoint,
            protocol=protocol,
            identity=resolved[tag],
        ).key
        for tag in model_tags
        for payload in _publication_payloads(tag, configuration)
    }
    if not keys:
        raise ValueError("publication warm-key planner produced no calls")
    return frozenset(keys)


def _publication_inventory(
    expected_keys: frozenset[str], path: Path | None = None
) -> tuple[int, int, int]:
    """Return ``covered, required, validated total`` for exact replay keys."""
    from ailm.call_cache import CallCache

    if not expected_keys:
        raise ValueError("publication key inventory cannot be empty")
    path = _cache_path() if path is None else path
    if not path.exists():
        return 0, len(expected_keys), 0
    cache = CallCache(path)
    stored_keys = {key for key, _payload, _value in cache.entries()}
    return (
        len(expected_keys & stored_keys),
        len(expected_keys),
        len(cache),
    )


def preflight() -> int:
    """Check everything that can fail fast, before anything that takes hours.

    Returns:
        0 when the run may proceed, 1 otherwise.
    """
    problems: list[str] = []

    if not VENV_PYTHON.is_file() or not os.access(VENV_PYTHON, os.X_OK):
        problems.append(f"the project interpreter is missing or not executable: {VENV_PYTHON}")
    else:
        interpreter = subprocess.run(
            [str(VENV_PYTHON), "--version"],
            capture_output=True,
            check=False,
            timeout=10,
        )
        if interpreter.returncode != 0:
            problems.append(
                f"the project interpreter cannot execute (exit {interpreter.returncode})"
            )

    try:
        _read_validated_ledger(LEDGER, description="source claims ledger for warming")
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        problems.append(f"the source claims ledger is not publishable: {error}")

    cache_override = os.environ.get("AILM_CALL_CACHE_PATH", "").strip()
    if cache_override:
        override_path = Path(cache_override)
        if not override_path.is_absolute() or override_path.resolve() != _cache_path().resolve():
            problems.append(
                "AILM_CALL_CACHE_PATH selects a different store than the canonical "
                f"warming target {_cache_path()}"
            )

    # Local state failures are cheaper and more deterministic than a model-server
    # probe. Refuse now so malformed input cannot launch even the preflight call.
    if problems:
        print("Cannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    try:
        from ailm.loopback_http import LoopbackTransportError, loopback_json_request

        body = loopback_json_request(OLLAMA_ENDPOINT, "/api/tags", timeout=5)
        if not isinstance(body, dict) or not isinstance(body.get("models"), list):
            raise ValueError("/api/tags did not return a models array")
        served = {
            entry["name"]
            for entry in body["models"]
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        }
    except (OSError, KeyError, ValueError, LoopbackTransportError) as error:
        problems.append(
            f"no Ollama server answering at {OLLAMA_ENDPOINT}/api/tags ({error}). Start it with "
            f"`ollama serve` in another shell."
        )
        served = set()

    if served:
        missing = sorted(tag for tag in _declared_probes().values() if tag not in served)
        if missing:
            problems.append(
                "the server does not have every declared probe: "
                + ", ".join(f"`ollama pull {tag}`" for tag in missing)
            )

    if problems:
        print("Cannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    try:
        expected_keys = _publication_call_keys()
        covered, required, trusted = _publication_inventory(expected_keys)
    except (KeyError, OSError, TypeError, ValueError) as error:
        problems.append(f"the exact publication call inventory cannot be established: {error}")
        covered = required = trusted = 0
    if problems:
        print("Cannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    percent = 100.0 * covered / required
    print("Ready to warm.")
    print(f"  server        : {len(served)} model(s) served, every declared probe present")
    print(f"  store         : {_cache_path().relative_to(PROJECT_ROOT)}")
    print(f"  exact replay  : {covered:,} / {required:,} publication keys ({percent:.1f}%)")
    print(f"  validated v2  : {trusted:,} total calls across all protocols and identities")
    print(
        f"  writes to     : {ISOLATED_LEDGER.relative_to(PROJECT_ROOT)} (the real ledger is untouched)"
    )
    return 0


def report_ledger_delta(recorded_ledger: bytes, measured_ledger: bytes | None) -> None:
    """Say whether any re-measured value differs from what is recorded.

    This is the entire scientific interest of a warming run. A value that moved
    is a finding; a run that overwrote it in place would be a loss.
    """
    if measured_ledger is None:
        print("\nNo isolated ledger was written, so nothing to compare.")
        return

    recorded = json.loads(recorded_ledger)["claims"]
    measured = json.loads(measured_ledger)["claims"]
    moved = external_claim_delta(recorded, measured)
    if not moved:
        print("\nEvery re-measured claim reproduced its recorded value exactly.")
        return
    print(f"\n{len(moved)} claim(s) differ from the ledger. This is a finding, not a fix:")
    for label, (was, now) in list(moved.items())[:20]:
        print(f"  {label}: recorded {was} -> re-measured {now}")
    print(
        "\nNothing has been written to data/claims.json. Decide deliberately whether "
        "the recorded value or the new measurement is the one to keep."
    )


def external_claim_delta(
    recorded: dict[str, dict[str, object]],
    measured: dict[str, dict[str, object]],
) -> dict[str, tuple[object | None, object | None]]:
    """Return value changes only in the recorder this warm actually ran.

    An isolated run may overlap an offline recorder pass against the real
    ledger. Comparing the whole files would then misreport an unrelated core
    update as a newly measured external-model finding. Ownership is checked on
    either side so a label retired by the external recorder remains visible.
    """

    def owned(entry: dict[str, object] | None) -> bool:
        return entry is not None and Path(str(entry.get("source", ""))).name == RECORDER.name

    return {
        label: (recorded.get(label, {}).get("value"), measured.get(label, {}).get("value"))
        for label in sorted(set(recorded) | set(measured))
        if (owned(recorded.get(label)) or owned(measured.get(label)))
        and recorded.get(label, {}).get("value") != measured.get(label, {}).get("value")
    }


def next_steps() -> None:
    print(
        "\nWith the store warm, the rest of the rebuild is:\n"
        "  # Set AILM_TEMPLATE_ROOT to the selected template checkout first.\n"
        '  export TEMPLATE_REPO_ROOT="$AILM_TEMPLATE_ROOT"\n'
        "  # Dynamic discovery, dependency order, then exact-call binding validation.\n"
        "  .venv/bin/python scripts/orchestrate_claim_recorders.py network\n"
        "  # Regenerate in place through each writer's staging/atomic-publication path.\n"
        "  .venv/bin/python scripts/z_generate_manuscript_variables.py --print-only\n"
        "  .venv/bin/python scripts/generate_figures.py\n"
        "  .venv/bin/python scripts/generate_figure_registry.py\n"
        "  .venv/bin/python scripts/generate_pymdp_sparse_patch.py\n"
        "  .venv/bin/python scripts/z_render_docs.py --supporting-protocols\n"
        "  .venv/bin/python scripts/z_render_docs.py --check\n"
        "  # then, from the template checkout, using its locked uv environment:\n"
        "  uv run --locked python scripts/pipeline/stage_03_render.py --project working/active_inference_language_model\n"
        "  # From the AILM checkout, resolve and check the individual HTML exports:\n"
        "  .venv/bin/python scripts/z_render_docs.py --publication-references\n"
        "  .venv/bin/python scripts/z_render_docs.py --check --supporting-protocols\n"
        "  # Back in the template checkout:\n"
        "  uv run --locked python scripts/pipeline/stage_04_validate.py --project working/active_inference_language_model\n"
        "  # Finally, from the AILM checkout, run acceptance over the completed tree:\n"
        "  .venv/bin/python -I -S scripts/verify_release.py full\n"
        "\nNo output-tree deletion is required: each generator validates and replaces "
        "only the artifacts it owns. The call store remains an input under cache/."
    )


def _recorder_command(*, audit_strata_only: bool) -> list[str]:
    """Build the recorder invocation; audit repair must never reuse sidecars."""
    command = [str(VENV_PYTHON), str(RECORDER)]
    if not audit_strata_only:
        return command
    command.extend(sorted(_declared_probes()))
    command.extend(
        [
            "--ignore-sidecars",
            "--letter-n",
            "1",
            "--phoneme-n",
            "1",
            "--g2p-n",
            "1",
            "--calibration-n",
            str(AUDIT_STRATUM_CALIBRATION_N),
            "--ensemble-n",
            "1",
            "--repeat-n",
            "1",
        ]
    )
    return command


def _warming_session_lock_path(isolated_ledger: Path) -> Path:
    """Outer lock that admits only one initializer-plus-recorder session."""
    return isolated_ledger.with_name(f".{isolated_ledger.name}.warm-session.lock")


def _run_recorder_with_isolated_ledger(
    command: list[str],
    *,
    ledger: Path = LEDGER,
    isolated_ledger: Path = ISOLATED_LEDGER,
    project_root: Path = PROJECT_ROOT,
    environment: dict[str, str] | None = None,
    cache_path: Path | None = None,
) -> tuple[subprocess.CompletedProcess[bytes], bytes, bytes | None]:
    """Initialize and run while excluding every competing warmer or writer.

    There are two distinct critical sections:

    * the outer session lock stays held until the recorder exits, so a second
      warmer refuses before it can replace the first run's isolated ledger;
    * initialization itself takes the same per-directory claims-writer lock the
      child recorder will take.  This catches an older warmer or a directly
      launched recorder that is already writing ``warming_claims.json``.

    The writer lock is released immediately after atomic initialization so the
    child can acquire it normally.  Holding it across ``subprocess.run`` would
    make the parent reject its own child.  The outer lock closes that hand-off
    gap for other warmers.
    """
    from _artifact_io import atomic_write_bytes, read_authenticated_regular_file

    from ailm.ledger_lock import LOCK_FILENAME, single_writer_lock

    # ``Path.resolve`` follows a pre-existing leaf symlink. Atomic publication
    # safely replaces that leaf, so resolving it *before* publication would give
    # the child the old external target while the parent initializes the local
    # path. Use a lexical absolute path: absolute for ``ClaimsLedger`` isolation,
    # but deliberately never dereferenced.
    isolated_target = Path(os.path.abspath(isolated_ledger))
    isolated_target.parent.mkdir(parents=True, exist_ok=True)
    session_lock = _warming_session_lock_path(isolated_target)
    writer_lock = isolated_target.parent / LOCK_FILENAME
    child_environment = dict(os.environ if environment is None else environment)
    child_environment["AILM_CLAIMS_PATH"] = str(isolated_target)
    warm_cache_target = _cache_path() if cache_path is None else Path(os.path.abspath(cache_path))
    child_environment["AILM_CALL_CACHE_PATH"] = str(warm_cache_target)

    try:
        with single_writer_lock(session_lock):
            with single_writer_lock(writer_lock):
                recorded = _read_validated_ledger(
                    ledger,
                    description="source claims ledger for isolated warming",
                )
                atomic_write_bytes(isolated_target, recorded)
            result = subprocess.run(
                command,
                cwd=str(project_root),
                env=child_environment,
                check=False,
            )
            # A directly launched recorder knows only the claims-writer lock,
            # not this warmer's outer session lock. Reacquire its lock before
            # reading the result so such a writer can neither replace the file
            # in the child-to-snapshot handoff nor be mistaken for our child.
            with single_writer_lock(writer_lock):
                try:
                    measured = read_authenticated_regular_file(
                        isolated_target,
                        description="completed isolated warming ledger",
                    ).content
                    _validated_ledger_snapshot(
                        measured,
                        description=f"completed isolated warming ledger at {isolated_target}",
                    )
                except FileNotFoundError:
                    measured = None
            # Capture before releasing the outer session lock. A second warmer
            # may legitimately reinitialize the shared scratch ledger as soon
            # as this context exits; reporting from the pathname afterward
            # would let that later run rewrite what this run appears to have
            # measured.
            return result, recorded, measured
    except (FileNotFoundError, OSError, TypeError, ValueError, SystemExit) as error:
        raise SystemExit(
            "warming session refused during isolated-ledger initialization, recorder "
            f"execution, or completed-ledger snapshot: {error}"
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="run the preflight checks and stop")
    mode.add_argument(
        "--audit-strata-only",
        action="store_true",
        help=(
            "populate a small, real sample in every model/request-kind cache-audit "
            "stratum without claiming to reproduce the full benchmark"
        ),
    )
    args = parser.parse_args()

    status = preflight()
    if status or args.check:
        return status

    print("\nWarming. This resumes: only the calls not already stored are measured.")
    command = _recorder_command(audit_strata_only=args.audit_strata_only)
    if args.audit_strata_only:
        print(
            "Audit-strata mode: this measures a small real sample in every "
            "registered model/request-kind cell. It does not reproduce or replace "
            "the full benchmark ledger.",
            flush=True,
        )
    result, recorded_ledger, measured_ledger = _run_recorder_with_isolated_ledger(command)
    trusted = _warm_inventory()
    print(f"\nStore now holds {trusted:,} validated v2 calls.")
    if result.returncode != 0:
        print(
            f"The recorder exited {result.returncode}. The store keeps every call it "
            f"already wrote, so re-running resumes rather than restarting.",
            file=sys.stderr,
        )
        return result.returncode

    if args.audit_strata_only:
        print(
            "\nAudit strata are populated in the call store. The intentionally "
            "reduced isolated-ledger measurement is not compared with or promoted "
            "to data/claims.json. Run the ordinary command for full reproduction."
        )
    else:
        report_ledger_delta(recorded_ledger, measured_ledger)
    next_steps()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
