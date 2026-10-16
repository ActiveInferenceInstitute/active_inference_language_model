"""Network-recorder replay provenance, including planted stale inputs.

The publication replay excludes recorders that require a live model server.
These tests prove that the complementary receipt is an actual boundary: a
valid current-schema call set passes, while stale source, wrong configuration,
wrong model, missing calls, and unsupported cache schemas all fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import ailm.replay_provenance as provenance
from ailm.call_cache import CacheCorruptionError, CacheRequest, CallCache
from ailm.replay_provenance import (
    BINDING_SCHEMA,
    NetworkClaimBindingError,
    binding_path,
    discover_network_recorders,
    maximum_prompt_eval_tokens,
    network_recorder_configuration,
    source_manifest,
    using_publication_ledger,
    validate_all_network_claim_bindings,
    validate_network_claim_binding,
    validate_network_measurement_inputs,
    write_network_claim_binding,
)

ENDPOINT = "http://127.0.0.1:11434"
PROTOCOL = "test-protocol-v1"
MODEL = "probe:1b"
IDENTITY = "sha256:" + "a" * 64
RECORDER = "scripts/record_probe_claims.py"


def _configuration(
    *,
    positions: int = 8,
    model: str = MODEL,
    endpoint: str = ENDPOINT,
    protocol: str = PROTOCOL,
) -> dict[str, object]:
    return network_recorder_configuration(
        endpoint=endpoint,
        call_protocol=protocol,
        model_tags=[model],
        parameters={"positions": positions, "seed": 7},
    )


def _project(tmp_path: Path) -> tuple[Path, CallCache, CacheRequest]:
    root = tmp_path / "project"
    (root / "src" / "ailm").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "vendor").mkdir()
    (root / "data").mkdir()
    (root / "src" / "ailm" / "probe.py").write_text(
        '"""Production probe helper."""\nVALUE = 1\n', encoding="utf-8"
    )
    (root / "scripts" / "_artifact_io.py").write_text(
        '"""Publication writer helper."""\nVALUE = 1\n', encoding="utf-8"
    )
    (root / RECORDER).write_text(
        "from _artifact_io import VALUE\n"
        "from ailm.benchmarks import ask_ollama\n"
        "SOURCE = 'scripts/record_probe_claims.py'\n",
        encoding="utf-8",
    )
    (root / "scripts" / "record_offline_claims.py").write_text(
        "from ailm.benchmarks import subsample_positions\n", encoding="utf-8"
    )
    (root / "vendor" / "fixture.json").write_text('{"value":1}\n', encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    ledger = {
        "_schema": "ailm-claims-v2",
        "_description": "fixture",
        "claims": {
            "probe_accuracy": {
                "value": 0.75,
                "run_id": "probe-run",
                "source": RECORDER,
                "status": "descriptive",
            },
            "offline_count": {
                "value": 2,
                "run_id": "offline-run",
                "source": "scripts/record_offline_claims.py",
                "status": "descriptive",
            },
        },
    }
    (root / "data" / "claims.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    cache_path = root / "cache" / "bench_calls.jsonl"
    cache_path.parent.mkdir()
    store = CallCache(cache_path)
    request = CacheRequest(
        payload={"model": MODEL, "prompt": "question", "stream": False},
        endpoint=ENDPOINT,
        protocol=PROTOCOL,
        identity=IDENTITY,
    )
    store.put(request, {"response": "answer"})
    return root, store, request


def _write_valid(root: Path, store: CallCache, request: CacheRequest) -> Path:
    return write_network_claim_binding(
        project_root=root,
        recorder=RECORDER,
        expected_source_manifest=source_manifest(root, RECORDER),
        configuration=_configuration(),
        store=store,
        cache_keys={request.key},
    )


def test_provider_receipt_requires_exact_cache_payload_and_survives_environment_removal(
    tmp_path: Path, monkeypatch
) -> None:
    root, store, request = _project(tmp_path)
    route = {
        "only": ["pinned-provider"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 1.0, "completion": 1.0, "request": 0.0},
    }
    pinned = CacheRequest(
        payload={**request.payload, "provider": route},
        endpoint=ENDPOINT,
        protocol=PROTOCOL,
        identity=IDENTITY,
    )
    store.put(pinned, {"response": "answer"})
    configuration = _configuration()
    configuration["parameters"]["openrouter_provider"] = route
    path = write_network_claim_binding(
        project_root=root,
        recorder=RECORDER,
        expected_source_manifest=source_manifest(root, RECORDER),
        configuration=configuration,
        store=store,
        cache_keys={pinned.key},
    )
    monkeypatch.delenv("AILM_OPENROUTER_BUDGET_FILE", raising=False)
    monkeypatch.delenv("AILM_OPENROUTER_BUDGET_LIMIT_USD", raising=False)
    assert provenance.declared_openrouter_provider(root, RECORDER) == route
    assert (
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=configuration,
            cache_path=store.path,
        )["configuration"]
        == configuration
    )
    for changed, keys in (
        (_configuration(), {pinned.key}),
        (configuration, {request.key}),
        (configuration, {request.key, pinned.key}),
    ):
        with pytest.raises(NetworkClaimBindingError, match="provider constraints"):
            validate_network_measurement_inputs(configuration=changed, store=store, cache_keys=keys)
    payload = json.loads(path.read_text())
    payload["configuration"]["parameters"]["openrouter_provider"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(NetworkClaimBindingError, match="provider declaration"):
        provenance.declared_openrouter_provider(root, RECORDER)


def test_historical_receipt_provider_is_absent_not_inferred_from_environment(
    tmp_path: Path, monkeypatch
) -> None:
    root, store, request = _project(tmp_path)
    assert provenance.declared_openrouter_provider(root, RECORDER) is None
    _write_valid(root, store, request)
    monkeypatch.setenv("AILM_OPENROUTER_BUDGET_FILE", str(tmp_path / "unrelated.json"))
    monkeypatch.setenv("AILM_OPENROUTER_BUDGET_LIMIT_USD", "1")
    assert provenance.declared_openrouter_provider(root, RECORDER) is None


def test_measurement_preflight_normalizes_finite_evidence_without_writing(tmp_path: Path) -> None:
    """Receipt inputs are validated before a recorder is allowed to replace claims."""
    root, store, request = _project(tmp_path)
    ledger_path = root / "data" / "claims.json"
    before = ledger_path.read_bytes()
    configuration, measurement, cache = validate_network_measurement_inputs(
        configuration=_configuration(),
        store=store,
        cache_keys={request.key},
        measurement={"z": [2, 1], "a": {"finite": 0.25}},
    )
    assert configuration == _configuration()
    assert measurement == {"a": {"finite": 0.25}, "z": [2, 1]}
    assert list(measurement) == ["a", "z"]
    assert cache["keys"] == [request.key]
    assert ledger_path.read_bytes() == before


def test_prompt_token_maximum_is_read_from_exact_cached_replies(tmp_path: Path) -> None:
    """The context guard uses server token counts, not prompt character length."""

    store = CallCache(tmp_path / "prompt-tokens.jsonl")
    requests = [
        CacheRequest(
            payload={"model": MODEL, "prompt": prompt},
            endpoint=ENDPOINT,
            protocol=PROTOCOL,
            identity=IDENTITY,
        )
        for prompt in ("short", "longer")
    ]
    store.put(requests[0], {"response": "a", "prompt_eval_count": 3})
    store.put(requests[1], {"response": "b", "prompt_eval_count": 9})

    assert maximum_prompt_eval_tokens(store, {request.key for request in requests}) == 9


@pytest.mark.parametrize(
    "reply",
    (
        {"response": "a"},
        {"response": "a", "prompt_eval_count": True},
        {"response": "a", "prompt_eval_count": 1.5},
        {"response": "a", "prompt_eval_count": 0},
        {"response": "a", "prompt_eval_count": -1},
    ),
)
def test_prompt_token_maximum_refuses_invalid_server_counts(
    tmp_path: Path, reply: dict[str, object]
) -> None:
    store = CallCache(tmp_path / "invalid-prompt-tokens.jsonl")
    request = CacheRequest(
        payload={"model": MODEL, "prompt": "question"},
        endpoint=ENDPOINT,
        protocol=PROTOCOL,
        identity=IDENTITY,
    )
    store.put(request, reply)

    with pytest.raises(NetworkClaimBindingError, match="invalid prompt_eval_count"):
        maximum_prompt_eval_tokens(store, {request.key})


def test_prompt_token_maximum_refuses_empty_or_missing_key_sets(tmp_path: Path) -> None:
    store = CallCache(tmp_path / "missing-prompt-tokens.jsonl")
    with pytest.raises(NetworkClaimBindingError, match="requires call keys"):
        maximum_prompt_eval_tokens(store, set())
    with pytest.raises(NetworkClaimBindingError, match="missing cache keys"):
        maximum_prompt_eval_tokens(store, {"0" * 64})


@pytest.mark.parametrize(
    "measurement",
    (
        ["not", "an", "object"],
        {"nan": float("nan")},
        {"infinite": float("inf")},
        {"unsupported": object()},
    ),
)
def test_measurement_preflight_rejects_nonobject_or_noncanonical_evidence(
    tmp_path: Path, measurement: object
) -> None:
    root, store, request = _project(tmp_path)
    ledger_path = root / "data" / "claims.json"
    before = ledger_path.read_bytes()
    with pytest.raises(NetworkClaimBindingError, match="measurement|finite canonical JSON"):
        validate_network_measurement_inputs(
            configuration=_configuration(),
            store=store,
            cache_keys={request.key},
            measurement=measurement,  # type: ignore[arg-type]
        )
    assert ledger_path.read_bytes() == before


def _rewrite_authenticated(path: Path, mutate) -> None:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutate(receipt)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    encoded = json.dumps(
        body, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )


def test_current_binding_is_deterministic_and_validates_complete_namespace(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    path = _write_valid(root, store, request)
    first = path.read_bytes()
    validated = validate_network_claim_binding(
        project_root=root,
        recorder=RECORDER,
        expected_configuration=_configuration(),
    )
    _write_valid(root, store, request)

    assert path.read_bytes() == first
    assert validated["schema"] == BINDING_SCHEMA
    assert validated["claims"]["labels"] == ["probe_accuracy"]
    assert validated["cache"]["keys"] == [request.key]
    assert validated["cache"]["models"] == [MODEL]
    assert validated["cache"]["identities"] == [IDENTITY]


def test_isolated_ledger_cannot_overwrite_publication_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, store, request = _project(tmp_path)
    isolated = tmp_path / "isolated-claims.json"
    isolated.write_text((root / "data" / "claims.json").read_text(encoding="utf-8"))
    monkeypatch.setenv("AILM_CLAIMS_PATH", str(isolated))

    assert using_publication_ledger(root) is False
    with pytest.raises(NetworkClaimBindingError, match="isolated AILM_CLAIMS_PATH"):
        _write_valid(root, store, request)
    assert not binding_path(root, RECORDER).exists()


def test_missing_malformed_and_nonobject_receipts_fail_closed(tmp_path: Path) -> None:
    root, _store, _request = _project(tmp_path)
    path = binding_path(root, RECORDER)
    with pytest.raises(NetworkClaimBindingError, match="cannot inspect"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )

    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff")
    with pytest.raises(NetworkClaimBindingError, match="invalid network receipt JSON"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(NetworkClaimBindingError, match="root must be an object"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )


def test_recorder_path_and_parse_controls_fail_closed(tmp_path: Path) -> None:
    root, _store, _request = _project(tmp_path)
    with pytest.raises(NetworkClaimBindingError, match="existing file beneath"):
        binding_path(root, "scripts/record_missing_claims.py")
    wrong_name = root / "scripts" / "helper.py"
    wrong_name.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(NetworkClaimBindingError, match="does not name a claim recorder"):
        binding_path(root, wrong_name)

    linked = root / "scripts" / "record_linked_claims.py"
    linked.symlink_to(root / RECORDER)
    with pytest.raises(NetworkClaimBindingError, match="not a real regular file"):
        binding_path(root, linked)

    broken = root / "scripts" / "record_broken_claims.py"
    broken.write_text("from ailm.benchmarks import (\n", encoding="utf-8")
    with pytest.raises(NetworkClaimBindingError, match="cannot parse recorder"):
        discover_network_recorders(root)


def test_source_manifest_is_sorted_and_changes_with_code_or_vendored_input(tmp_path: Path) -> None:
    root, _store, _request = _project(tmp_path)
    first = source_manifest(root, RECORDER)
    paths = [row["path"] for row in first["files"]]
    assert paths == sorted(paths)
    assert {
        RECORDER,
        "scripts/_artifact_io.py",
        "src/ailm/probe.py",
        "vendor/fixture.json",
        "pyproject.toml",
        "uv.lock",
    } <= set(paths)

    (root / "vendor" / "fixture.json").write_text('{"value":2}\n', encoding="utf-8")
    assert source_manifest(root, RECORDER)["manifest_sha256"] != first["manifest_sha256"]

    helper_bound = source_manifest(root, RECORDER)
    (root / "scripts" / "_artifact_io.py").write_text(
        '"""Changed publication writer helper."""\nVALUE = 2\n', encoding="utf-8"
    )
    assert source_manifest(root, RECORDER)["manifest_sha256"] != helper_bound["manifest_sha256"]


def test_writer_refuses_source_drift_after_measurement_snapshot(tmp_path: Path) -> None:
    """A final receipt may bind only the source snapshot that produced the run."""

    root, store, request = _project(tmp_path)
    measured_source = source_manifest(root, RECORDER)
    (root / "src" / "ailm" / "probe.py").write_text(
        '"""Changed after the measurement."""\nVALUE = 2\n',
        encoding="utf-8",
    )

    with pytest.raises(NetworkClaimBindingError, match="changed after measurement began"):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=measured_source,
            configuration=_configuration(),
            store=store,
            cache_keys={request.key},
        )

    assert not binding_path(root, RECORDER).exists()


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        ("src/ailm/probe.py", '"""Changed helper."""\nVALUE = 2\n'),
        ("scripts/_artifact_io.py", '"""Changed writer."""\nVALUE = 2\n'),
        ("vendor/fixture.json", '{"value":2}\n'),
        ("pyproject.toml", "[project]\nname='changed'\n"),
    ],
)
def test_stale_source_or_input_receipt_fails_closed(
    tmp_path: Path, relative: str, replacement: str
) -> None:
    root, store, request = _project(tmp_path)
    _write_valid(root, store, request)
    (root / relative).write_text(replacement, encoding="utf-8")

    with pytest.raises(NetworkClaimBindingError, match="source/input manifest is stale"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )


def test_wrong_or_stale_configuration_receipt_fails_closed(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    _write_valid(root, store, request)

    with pytest.raises(NetworkClaimBindingError, match="configuration is stale"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(positions=9),
        )


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        (_configuration(endpoint="http://127.0.0.1:9999"), "cache endpoint"),
        (_configuration(protocol="other-protocol"), "cache protocol"),
    ],
)
def test_cache_endpoint_and_protocol_must_match_configuration(
    tmp_path: Path, configuration: dict[str, object], message: str
) -> None:
    root, store, request = _project(tmp_path)
    with pytest.raises(NetworkClaimBindingError, match=message):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=source_manifest(root, RECORDER),
            configuration=configuration,
            store=store,
            cache_keys={request.key},
        )


def test_explicit_cache_purpose_protocols_are_bound_as_one_measurement(
    tmp_path: Path,
) -> None:
    """Two identical server payloads can retain distinct trial observations."""

    root, store, primary = _project(tmp_path)
    repeat_protocol = f"{PROTOCOL}/repeat-trial"
    repeat = CacheRequest(
        payload=primary.payload_snapshot(),
        endpoint=ENDPOINT,
        protocol=repeat_protocol,
        identity=IDENTITY,
    )
    store.put(repeat, {"response": "second observation"})
    configuration = _configuration()
    configuration["parameters"]["cache_protocols"] = sorted((PROTOCOL, repeat_protocol))
    path = write_network_claim_binding(
        project_root=root,
        recorder=RECORDER,
        expected_source_manifest=source_manifest(root, RECORDER),
        configuration=configuration,
        store=store,
        cache_keys={primary.key, repeat.key},
    )
    assert path.is_file()
    assert validate_network_claim_binding(
        project_root=root,
        recorder=RECORDER,
        expected_configuration=configuration,
    )["cache"]["protocols"] == sorted((PROTOCOL, repeat_protocol))


@pytest.mark.parametrize(
    "cache_protocols",
    (
        [],
        [PROTOCOL, PROTOCOL],
        ["repeat-only"],
        [PROTOCOL, ""],
    ),
)
def test_invalid_cache_purpose_protocol_declarations_fail_closed(
    tmp_path: Path, cache_protocols: list[str]
) -> None:
    root, store, request = _project(tmp_path)
    configuration = _configuration()
    configuration["parameters"]["cache_protocols"] = cache_protocols
    with pytest.raises(NetworkClaimBindingError, match="cache_protocols"):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=source_manifest(root, RECORDER),
            configuration=configuration,
            store=store,
            cache_keys={request.key},
        )


def test_non_digest_or_multiple_model_identities_fail_closed(tmp_path: Path) -> None:
    root, _store, _request = _project(tmp_path)
    cache_path = root / "cache" / "identity-calls.jsonl"
    store = CallCache(cache_path)
    invalid = CacheRequest(
        payload={"model": MODEL, "prompt": "one"},
        endpoint=ENDPOINT,
        protocol=PROTOCOL,
        identity="mutable-tag",
    )
    store.put(invalid, {"response": "one"})
    with pytest.raises(NetworkClaimBindingError, match="immutable SHA-256 identity"):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=source_manifest(root, RECORDER),
            configuration=_configuration(),
            store=store,
            cache_keys={invalid.key},
        )

    second = CacheRequest(
        payload={"model": MODEL, "prompt": "two"},
        endpoint=ENDPOINT,
        protocol=PROTOCOL,
        identity="sha256:" + "b" * 64,
    )
    first = CacheRequest(
        payload={"model": MODEL, "prompt": "three"},
        endpoint=ENDPOINT,
        protocol=PROTOCOL,
        identity=IDENTITY,
    )
    store = CallCache(root / "cache" / "changed-model.jsonl")
    store.put(first, {"response": "one"})
    store.put(second, {"response": "two"})
    with pytest.raises(NetworkClaimBindingError, match="immutable SHA-256 identity"):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=source_manifest(root, RECORDER),
            configuration=_configuration(),
            store=store,
            cache_keys={first.key, second.key},
        )


def test_wrong_model_configuration_cannot_create_a_plausible_receipt(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)

    with pytest.raises(NetworkClaimBindingError, match="cache models.*configured models"):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=source_manifest(root, RECORDER),
            configuration=_configuration(model="other:1b"),
            store=store,
            cache_keys={request.key},
        )


def test_empty_cache_key_set_cannot_bind_claims(tmp_path: Path) -> None:
    root, store, _request = _project(tmp_path)
    with pytest.raises(NetworkClaimBindingError, match="at least one validated"):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=source_manifest(root, RECORDER),
            configuration=_configuration(),
            store=store,
            cache_keys=set(),
        )


def test_missing_or_replaced_current_cache_artifact_fails_closed(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    _write_valid(root, store, request)

    replacement_path = root / "cache" / "replacement.jsonl"
    replacement = CallCache(replacement_path)
    other = CacheRequest(
        payload={"model": MODEL, "prompt": "different", "stream": False},
        endpoint=ENDPOINT,
        protocol=PROTOCOL,
        identity=IDENTITY,
    )
    replacement.put(other, {"response": "other"})

    with pytest.raises(NetworkClaimBindingError, match="validated call-store keys are missing"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
            cache_path=replacement_path,
        )


def test_unsupported_v1_cache_is_rejected_before_binding_validation(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    _write_valid(root, store, request)
    unsupported_path = root / "cache" / "unsupported-v1.jsonl"
    unsupported_path.write_text(
        json.dumps(
            {
                "schema": "ailm-call-cache-v1",
                "key": "b" * 64,
                "payload": {"model": MODEL, "prompt": "question"},
                "value": {"response": "answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CacheCorruptionError, match="unsupported schema 'ailm-call-cache-v1'"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
            cache_path=unsupported_path,
        )


def test_changed_claim_value_or_run_id_invalidates_the_receipt(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    _write_valid(root, store, request)
    ledger_path = root / "data" / "claims.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["claims"]["probe_accuracy"]["value"] = 0.5
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(NetworkClaimBindingError, match="complete ledger namespace"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda ledger: ledger.update({"_schema": "ailm-claims-v1"}), "ailm-claims-v2"),
        (lambda ledger: ledger.update({"claims": []}), "ailm-claims-v2"),
        (
            lambda ledger: ledger["claims"].update({"probe_accuracy": "not-an-object"}),
            "is not an object",
        ),
        (
            lambda ledger: ledger["claims"]["probe_accuracy"].update({"extra": True}),
            "unsupported fields",
        ),
        (
            lambda ledger: ledger["claims"]["probe_accuracy"].update({"run_id": 3}),
            "invalid provenance",
        ),
        (
            lambda ledger: ledger["claims"]["probe_accuracy"].update(
                {"source": "scripts/record_other_claims.py"}
            ),
            "owns no ledger claims",
        ),
    ],
)
def test_malformed_or_unowned_ledger_namespace_fails_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    root, store, request = _project(tmp_path)
    ledger_path = root / "data" / "claims.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    mutation(ledger)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(NetworkClaimBindingError, match=message):
        write_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_source_manifest=source_manifest(root, RECORDER),
            configuration=_configuration(),
            store=store,
            cache_keys={request.key},
        )


def test_status_assignment_does_not_invalidate_measurement_binding(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    _write_valid(root, store, request)
    ledger_path = root / "data" / "claims.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["claims"]["probe_accuracy"]["status"] = "exploratory"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    validated = validate_network_claim_binding(
        project_root=root,
        recorder=RECORDER,
        expected_configuration=_configuration(),
    )
    assert validated["claims"]["entries"]["probe_accuracy"]["value"] == 0.75


def test_discovery_and_complete_set_validation_are_default_deny(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    assert discover_network_recorders(root) == (RECORDER,)
    _write_valid(root, store, request)

    validated = validate_all_network_claim_bindings(
        project_root=root,
        expected_configurations={RECORDER: _configuration()},
    )
    assert set(validated) == {RECORDER}

    extra = binding_path(root, RECORDER).parent / "retired.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NetworkClaimBindingError, match="stale extras"):
        validate_all_network_claim_bindings(
            project_root=root,
            expected_configurations={RECORDER: _configuration()},
        )


@pytest.mark.parametrize(
    ("source", "detected"),
    (
        ("from ailm.benchmarks import ask_ollama as ask\n", True),
        ("import ailm.benchmarks as bench\nbench.ask_ollama('x')\n", True),
        (
            "from ailm.benchmarks import ollama as client\nclient.ask_ollama('x')\n",
            True,
        ),
        ("from ailm.benchmarks.ollama import ask_ollama as ask\n", True),
        (
            "import ailm.benchmarks.ollama\nailm.benchmarks.ollama.ask_ollama('x')\n",
            True,
        ),
        (
            "import ailm.benchmarks as bench\nask = getattr(bench, 'ask_ollama')\nask('x')\n",
            True,
        ),
        (
            "import ailm.benchmarks as bench\nbench.subsample_positions([1, 2], 1)\n",
            False,
        ),
    ),
)
def test_discovery_resolves_alias_and_submodule_import_forms(
    tmp_path: Path, source: str, detected: bool
) -> None:
    """Planted real AST forms cannot bypass or over-widen receipt discovery."""
    root, _store, _request = _project(tmp_path)
    (root / RECORDER).write_text(source, encoding="utf-8")
    assert (RECORDER in discover_network_recorders(root)) is detected


def test_no_discovered_network_recorder_is_not_a_vacuous_pass(tmp_path: Path) -> None:
    root, _store, _request = _project(tmp_path)
    (root / RECORDER).write_text(
        "from ailm.benchmarks import subsample_positions\n", encoding="utf-8"
    )
    with pytest.raises(NetworkClaimBindingError, match="no network recorders"):
        validate_all_network_claim_bindings(
            project_root=root,
            expected_configurations={},
        )


def test_missing_receipt_and_missing_configuration_coverage_fail_closed(tmp_path: Path) -> None:
    root, _store, _request = _project(tmp_path)
    with pytest.raises(NetworkClaimBindingError, match="receipt set is incomplete"):
        validate_all_network_claim_bindings(
            project_root=root,
            expected_configurations={RECORDER: _configuration()},
        )
    with pytest.raises(NetworkClaimBindingError, match="configuration coverage mismatch"):
        validate_all_network_claim_bindings(
            project_root=root,
            expected_configurations={},
        )


def test_receipt_digest_duplicate_key_and_symlink_controls_fire(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    path = _write_valid(root, store, request)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["claims"]["count"] = 999
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(NetworkClaimBindingError, match="receipt digest mismatch"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )

    path.write_text('{"schema":"one","schema":"two"}\n', encoding="utf-8")
    with pytest.raises(NetworkClaimBindingError, match="duplicate JSON key"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )

    path.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    path.symlink_to(outside)
    with pytest.raises(NetworkClaimBindingError, match="not a real regular file"):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda receipt: receipt.update({"extra": True}), "unexpected fields"),
        (lambda receipt: receipt.update({"schema": "future"}), "unsupported.*schema"),
        (lambda receipt: receipt.update({"configuration_sha256": "bad"}), "invalid digest"),
    ],
)
def test_receipt_envelope_shape_controls_fire(mutate, message: str) -> None:
    payload = {
        "schema": BINDING_SCHEMA,
        "recorder": RECORDER,
        "source": {},
        "configuration": {},
        "configuration_sha256": "a" * 64,
        "claims": {},
        "cache": {},
        "measurement": None,
        "receipt_sha256": "b" * 64,
    }
    mutate(payload)
    with pytest.raises(NetworkClaimBindingError, match=message):
        provenance._validate_receipt_shape(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt: receipt.update({"recorder": "scripts/record_other_claims.py"}),
            "wrong recorder",
        ),
        (
            lambda receipt: receipt.update({"configuration_sha256": "c" * 64}),
            "configuration digest",
        ),
        (lambda receipt: receipt.update({"cache": []}), "invalid cache binding"),
        (
            lambda receipt: receipt["cache"].update({"keys": ["not-a-digest"]}),
            "invalid call key",
        ),
        (
            lambda receipt: receipt["cache"].update(
                {"record_count": receipt["cache"]["record_count"] + 1}
            ),
            "cache records are stale",
        ),
    ],
)
def test_authenticated_but_semantically_wrong_receipts_fail_closed(
    tmp_path: Path, mutate, message: str
) -> None:
    root, store, request = _project(tmp_path)
    path = _write_valid(root, store, request)
    _rewrite_authenticated(path, mutate)
    with pytest.raises(NetworkClaimBindingError, match=message):
        validate_network_claim_binding(
            project_root=root,
            recorder=RECORDER,
            expected_configuration=_configuration(),
        )


def test_symlinked_manifest_input_and_binding_parent_fail_closed(tmp_path: Path) -> None:
    root, store, request = _project(tmp_path)
    vendor = root / "vendor" / "fixture.json"
    outside_vendor = tmp_path / "outside-vendor.json"
    outside_vendor.write_text("{}\n", encoding="utf-8")
    vendor.unlink()
    vendor.symlink_to(outside_vendor)
    with pytest.raises(NetworkClaimBindingError, match="authenticate manifest input"):
        source_manifest(root, RECORDER)

    vendor.unlink()
    vendor.write_text("{}\n", encoding="utf-8")
    binding_parent = binding_path(root, RECORDER).parent
    outside_parent = tmp_path / "outside-bindings"
    outside_parent.mkdir()
    binding_parent.symlink_to(outside_parent, target_is_directory=True)
    with pytest.raises(NetworkClaimBindingError, match="parent must be a real directory"):
        _write_valid(root, store, request)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"endpoint": "", "call_protocol": PROTOCOL, "model_tags": [MODEL]},
        {"endpoint": ENDPOINT, "call_protocol": "", "model_tags": [MODEL]},
        {"endpoint": ENDPOINT, "call_protocol": PROTOCOL, "model_tags": []},
        {"endpoint": ENDPOINT, "call_protocol": PROTOCOL, "model_tags": [MODEL, MODEL]},
        {"endpoint": ENDPOINT, "call_protocol": PROTOCOL, "model_tags": [MODEL], "parameters": []},
        {
            "endpoint": ENDPOINT,
            "call_protocol": PROTOCOL,
            "model_tags": [MODEL],
            "parameters": {"bad": float("nan")},
        },
    ],
)
def test_invalid_configuration_controls_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(NetworkClaimBindingError):
        call_kwargs = dict(kwargs)
        parameters = call_kwargs.pop("parameters", {})
        network_recorder_configuration(  # type: ignore[arg-type]
            parameters=parameters,
            **call_kwargs,
        )


@pytest.mark.parametrize(
    "configuration",
    [
        None,
        {},
        {**_configuration(), "extra": True},
        {**_configuration(), "schema": "future"},
        {**_configuration(), "model_tags": "not-a-list"},
    ],
)
def test_internal_configuration_validation_is_fail_closed(configuration: object) -> None:
    with pytest.raises(NetworkClaimBindingError):
        provenance._validate_configuration(configuration)


def test_relative_claims_override_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _store, _request = _project(tmp_path)
    monkeypatch.setenv("AILM_CLAIMS_PATH", "relative/claims.json")
    with pytest.raises(NetworkClaimBindingError, match="must be absolute"):
        using_publication_ledger(root)
