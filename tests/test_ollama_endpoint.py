"""Project endpoint selection exercised in fresh processes and real loopback HTTP."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = "AILM_OLLAMA_ENDPOINT"


def _run(code: str, endpoint: str | None, *args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop(ENV, None)
    if endpoint is not None:
        environment[ENV] = endpoint
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_selected_endpoint_drives_real_request_and_cache_identity(tmp_path, local_http_server):
    local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "a" * 64})
    local_http_server.enqueue_json("/api/generate", {"response": '{"answer":"q"}'})
    code = """
import json, sys
from pathlib import Path
from ailm.benchmarks.constants import OLLAMA_URL
assert OLLAMA_URL == sys.argv[1], (OLLAMA_URL, sys.argv[1])
from ailm.benchmarks.ollama import ask_ollama, enable_call_cache
cache = enable_call_cache(Path(sys.argv[2]))
assert ask_ollama('fixture:latest', 'prompt', 'letter') == '{"answer":"q"}'
keys = {entry[0] for entry in cache.entries()}
print(json.dumps(cache.provenance_binding(keys)))
"""
    result = _run(
        code, local_http_server.base_url, local_http_server.base_url, str(tmp_path / "calls.jsonl")
    )
    assert result.returncode == 0, result.stdout + result.stderr
    binding = json.loads(result.stdout)
    assert binding["endpoints"] == [local_http_server.base_url]
    [sent] = local_http_server.requests_for("/api/generate")
    assert json.loads(sent.body)["options"]["num_ctx"] == 8192
    assert json.loads(sent.body)["options"]["use_mmap"] is False


def test_release_child_environment_preserves_selected_endpoint():
    code = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('release_endpoint_control', Path('scripts/verify_release.py'))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module._base_release_environment().get('AILM_OLLAMA_ENDPOINT') == sys.argv[1]
print(json.dumps(module._base_release_environment()['AILM_OLLAMA_ENDPOINT']))
"""
    result = _run(code, "http://127.0.0.1:15434", "http://127.0.0.1:15434")
    assert result.returncode == 0, result.stdout + result.stderr


def test_absent_override_retains_the_shipped_default():
    result = _run("from ailm.benchmarks.constants import OLLAMA_URL; print(OLLAMA_URL)", None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "http://localhost:11434"


@pytest.mark.parametrize("endpoint", [None, "http://127.0.0.1:15434"])
def test_warming_preflight_uses_the_transport_endpoint(endpoint):
    result = _run(
        "from scripts.warm_call_cache import OLLAMA_ENDPOINT; print(OLLAMA_ENDPOINT)",
        endpoint,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == (endpoint or "http://localhost:11434")


def test_warming_preflight_checks_selected_server_before_planning(local_http_server):
    local_http_server.enqueue_json("/api/tags", {"models": [{"name": "unregistered:latest"}]})
    result = _run(
        "from scripts.warm_call_cache import preflight; raise SystemExit(preflight())",
        local_http_server.base_url,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(local_http_server.requests_for("/api/tags")) == 1
    assert "the server does not have every declared probe" in result.stderr
    assert not local_http_server.requests_for("/api/show")
    assert not local_http_server.requests_for("/api/generate")


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        " ",
        "https://127.0.0.1:15434",
        "http://198.51.100.1:15434",
        "http://example.com:15434",
        "http://user@localhost:15434",
        "http://localhost:15434/api",
        "http://localhost:15434?other=1",
        "http://localhost:15434#other",
        "http://localhost:99999",
    ],
)
def test_invalid_override_refuses_import_before_any_http(endpoint, local_http_server):
    result = _run(
        'from ailm.benchmarks.ollama import ask_ollama; ask_ollama("fixture", "prompt", "letter")',
        endpoint,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert not local_http_server.requests


def test_endpoint_is_canonicalized_and_frozen_at_process_import():
    result = _run(
        """
import os
from ailm.benchmarks.constants import OLLAMA_URL
assert OLLAMA_URL == 'http://localhost:15434'
os.environ['AILM_OLLAMA_ENDPOINT'] = 'http://localhost:11434'
from ailm.benchmarks.ollama import OLLAMA_URL as transport_endpoint
assert transport_endpoint == OLLAMA_URL
""",
        " HTTP://LOCALHOST:15434/ ",
    )
    assert result.returncode == 0, result.stderr


def test_selected_endpoint_binds_producer_drift_and_warm_configuration():
    result = _run(
        """
import sys
sys.path.insert(0, 'scripts')
import warm_call_cache
import record_logprob_drift_claims as drift
from ailm.benchmarks.constants import OLLAMA_URL
assert warm_call_cache._publication_configuration()['endpoint'] == OLLAMA_URL
assert drift.publication_replay_configuration()['endpoint'] == OLLAMA_URL
""",
        "http://127.0.0.1:15434",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_selected_endpoint_receipt_replays_and_wrong_endpoint_refuses(tmp_path):
    result = _run(
        """
import importlib.util, sys
from pathlib import Path
from ailm.replay_provenance import NetworkClaimBindingError
spec=importlib.util.spec_from_file_location('endpoint_semantic_fixture', 'tests/test_external_benchmark_provenance.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
recorder, receipt, store = module._fixture(Path(sys.argv[1]))
assert receipt['configuration']['endpoint'] == 'http://127.0.0.1:15434'
recorder.validate_publication_measurement(receipt, store)
receipt['configuration']['endpoint'] = 'http://localhost:11434'
try:
    recorder.validate_publication_measurement(receipt, store)
except NetworkClaimBindingError as error:
    assert 'endpoint differs' in str(error)
else:
    raise AssertionError('wrong endpoint receipt accepted')
""",
        "http://127.0.0.1:15434",
        str(tmp_path / "semantic.jsonl"),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:15434", ""])
def test_controlled_release_subprocess_uses_or_refuses_exact_override(endpoint):
    result = _run(
        """
import importlib.util, json, subprocess, sys, tempfile
from pathlib import Path
spec = importlib.util.spec_from_file_location('release_endpoint_child', Path('scripts/verify_release.py'))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='endpoint-control-', dir=Path('.venv')) as directory:
    target = Path(directory) / 'observe.py'
    target.write_text('from ailm.benchmarks.constants import OLLAMA_URL; print(OLLAMA_URL)\\n')
    child = subprocess.run(module._controlled_python_argv('script', str(target.resolve().relative_to(Path.cwd()))), env=module._base_release_environment(), text=True, capture_output=True, timeout=90)
    print(json.dumps({'returncode': child.returncode, 'stdout': child.stdout, 'stderr': child.stderr}))
""",
        endpoint,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    child = json.loads(result.stdout)
    if endpoint:
        assert child["returncode"] == 0, child
        assert child["stdout"].strip() == endpoint
    else:
        assert child["returncode"] != 0
        assert "loopback endpoint must be a non-empty string" in child["stderr"]
