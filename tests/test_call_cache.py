"""The call store's guarantees, each stated as something that can fail.

The store exists so a twenty-hour run resumes per call rather than per model.
That is only worth having if a replayed answer is the answer the same question
would have received, so most of what is checked here is the key: what it
separates, what it collapses, and what it refuses to key at all.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ailm import benchmarks
from ailm.benchmarks.constants import repeat_trial_protocols
from ailm.call_cache import (
    CACHE_SCHEMA,
    CacheCorruptionError,
    CacheRequest,
    CacheStats,
    CallCache,
    cache_key,
    default_cache_path,
    model_digest,
)

PAYLOAD = {
    "model": "gemma3:4b",
    "prompt": "the quick brown fo",
    "options": {
        "temperature": 0,
        "seed": 20260811,
        "num_predict": 32,
        "use_mmap": False,
        "num_ctx": 8192,
    },
}
REPLY = {"response": '{"answer":"x"}', "logprobs": [{"token": "x"}]}
REPLY_LOGPROBS = {
    "response": '{"answer":"x"}',
    "logprobs": [
        {
            "token": "x",
            "top_logprobs": [
                {"token": "x", "logprob": -0.2},
                {"token": "y", "logprob": -2.0},
            ],
        }
    ],
}
OLLAMA_REPLY = {
    "model": "gemma3:4b",
    "created_at": "2026-08-24T12:00:00Z",
    "response": '{"answer":"x"}',
    "done": True,
    "done_reason": "stop",
    "context": [11, 22, 33],
    "total_duration": 1_000_000,
    "load_duration": 100_000,
    "prompt_eval_count": 7,
    "prompt_eval_duration": 200_000,
    "eval_count": 3,
    "eval_duration": 700_000,
    "logprobs": REPLY_LOGPROBS["logprobs"],
}

#: A payload in the OTHER declared verify stratum. `_VERIFY_STRATA` partitions on
#: whether the *payload* requested logprobs, and the recorder now refuses to audit
#: a store that cannot populate every declared stratum. A one-entry fixture
#: therefore no longer stands in for a real store: it would be asking the audit to
#: report coverage it does not have, which is the defect that guard exists for.
PAYLOAD_LOGPROBS = {**PAYLOAD, "prompt": "a different prefi", "logprobs": True}
PAYLOAD_QWEN = {**PAYLOAD, "model": "qwen2.5:3b", "prompt": "qwen prefix"}
PAYLOAD_QWEN_LOGPROBS = {
    **PAYLOAD_QWEN,
    "prompt": "qwen calibration prefix",
    "logprobs": True,
}
PAYLOAD_SMOLLM = {
    **PAYLOAD,
    "model": "smollm:1.7b-base-v0.2-q4_K_M",
    "prompt": "smollm prefix",
}
PAYLOAD_SMOLLM_LOGPROBS = {
    **PAYLOAD_SMOLLM,
    "prompt": "smollm calibration prefix",
    "logprobs": True,
}
GEMMA_IDENTITY = "sha256:" + "a" * 64
QWEN_IDENTITY = "sha256:" + "b" * 64
SMOLLM_IDENTITY = "sha256:" + "c" * 64


def _registered_test_identities() -> dict[str, str]:
    return {
        "gemma3:4b": GEMMA_IDENTITY,
        "qwen2.5:3b": QWEN_IDENTITY,
        "smollm:1.7b-base-v0.2-q4_K_M": SMOLLM_IDENTITY,
    }


def _cache(tmp_path: Path, **kwargs) -> CallCache:
    return CallCache(path=tmp_path / "calls.jsonl", **kwargs)


def _request(
    payload: dict = PAYLOAD,
    *,
    endpoint: str = "http://localhost:11434",
    protocol: str = "external-benchmark-v4",
    identity: str = GEMMA_IDENTITY,
) -> CacheRequest:
    return CacheRequest(
        payload=payload,
        endpoint=endpoint,
        protocol=protocol,
        identity=identity,
    )


def _fully_stratified_cache(
    tmp_path: Path, *, endpoint: str = "http://localhost:11434"
) -> CallCache:
    store = CallCache(path=tmp_path / "calls.jsonl")
    for payload, reply in (
        (PAYLOAD, REPLY),
        (PAYLOAD_LOGPROBS, REPLY_LOGPROBS),
        (PAYLOAD_QWEN, REPLY),
        (PAYLOAD_QWEN_LOGPROBS, REPLY_LOGPROBS),
        (PAYLOAD_SMOLLM, REPLY),
        (PAYLOAD_SMOLLM_LOGPROBS, REPLY_LOGPROBS),
    ):
        model = payload.get("model")
        identity = (
            QWEN_IDENTITY
            if model == "qwen2.5:3b"
            else SMOLLM_IDENTITY
            if model == "smollm:1.7b-base-v0.2-q4_K_M"
            else GEMMA_IDENTITY
        )
        store.put(_request(payload, identity=identity, endpoint=endpoint), reply)
    return store


def _race_same_request(
    path: Path, tmp_path: Path, replies: tuple[dict, dict]
) -> list[subprocess.CompletedProcess[str]]:
    """Start two real processes after both have loaded the same empty store."""

    program = """
import json
import sys
import time
from pathlib import Path
from ailm.call_cache import CacheCorruptionError, CacheRequest, CallCache

path, ready, start = map(Path, sys.argv[1:4])
reply = json.loads(sys.argv[4])
cache = CallCache(path)
ready.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 10
while not start.exists():
    if time.monotonic() >= deadline:
        raise SystemExit(24)
    time.sleep(0.005)
request = CacheRequest(
    payload={"model": "m", "prompt": "one shared prompt"},
    endpoint="http://127.0.0.1:11434",
    protocol="same-key-race",
    identity="sha256:same-key-race",
)
try:
    cache.put(request, reply)
except CacheCorruptionError:
    raise SystemExit(23)
"""
    start = tmp_path / "start"
    running: list[subprocess.Popen[str]] = []
    ready_paths: list[Path] = []
    for index, reply in enumerate(replies):
        ready = tmp_path / f"ready-{index}"
        ready_paths.append(ready)
        running.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    program,
                    str(path),
                    str(ready),
                    str(start),
                    json.dumps(reply),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    deadline = time.monotonic() + 10
    while not all(ready.exists() for ready in ready_paths):
        if time.monotonic() >= deadline:
            for process in running:
                process.terminate()
            pytest.fail("same-key subprocesses did not reach the synchronized start")
        time.sleep(0.005)
    start.write_text("start", encoding="utf-8")
    completed = []
    for process in running:
        stdout, stderr = process.communicate(timeout=120)
        completed.append(
            subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
        )
    return completed


def test_call_store_refuses_symlinked_and_special_cache_paths(tmp_path: Path) -> None:
    """NEGATIVE CONTROLS: replay and append never follow an external path."""

    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(b"external sentinel\n")
    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(outside)
    with pytest.raises(CacheCorruptionError, match="regular file"):
        CallCache(linked)
    assert outside.read_bytes() == b"external sentinel\n"

    late = tmp_path / "late.jsonl"
    store = CallCache(late)
    late.symlink_to(outside)
    with pytest.raises(CacheCorruptionError, match="regular file"):
        store.put(_request(), REPLY)
    assert outside.read_bytes() == b"external sentinel\n"

    fifo = tmp_path / "special.jsonl"
    os.mkfifo(fifo)
    with pytest.raises(CacheCorruptionError, match="not a regular file"):
        CallCache(fifo)

    outside_directory = tmp_path / "outside-cache"
    outside_directory.mkdir()
    linked_directory = tmp_path / "linked-cache"
    linked_directory.symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(CacheCorruptionError, match="parent must be a real directory"):
        CallCache(linked_directory / "calls.jsonl")


def _enqueue_matching_strata(local_http_server, *, cycles: int = 1) -> None:
    """Queue replies in the recorder's declared stratum traversal order."""
    for _ in range(cycles):
        local_http_server.enqueue_json("/api/generate", REPLY_LOGPROBS)
        local_http_server.enqueue_json("/api/generate", REPLY)
        local_http_server.enqueue_json("/api/generate", REPLY_LOGPROBS)
        local_http_server.enqueue_json("/api/generate", REPLY)
        local_http_server.enqueue_json("/api/generate", REPLY_LOGPROBS)
        local_http_server.enqueue_json("/api/generate", REPLY)


def _enqueue_current_model_identities(local_http_server) -> None:
    local_http_server.enqueue_json("/api/show", {"digest": GEMMA_IDENTITY})
    local_http_server.enqueue_json("/api/show", {"digest": QWEN_IDENTITY})
    local_http_server.enqueue_json("/api/show", {"digest": SMOLLM_IDENTITY})


class TestKey:
    """What the key separates is what the store is allowed to reuse."""

    def _key(self, **overrides) -> str:
        base = {
            "payload": PAYLOAD,
            "endpoint": "http://localhost:11434",
            "protocol": "external-benchmark-v4",
            "identity": "sha256:aaa",
        }
        base.update(overrides)
        return cache_key(**base)

    def test_identical_questions_share_a_key(self) -> None:
        assert self._key() == self._key()

    def test_key_order_in_the_payload_does_not_split_a_key(self) -> None:
        """Two encodings of one request must not be measured twice.

        A payload built by ``{**a, "logprobs": True}`` and one built by literal
        order are the same JSON object and the same question. Without canonical
        encoding they would key differently, and the census would pay twice for
        every position while the store looked like it was working.
        """
        reordered = {k: PAYLOAD[k] for k in reversed(list(PAYLOAD))}
        assert self._key(payload=reordered) == self._key()

    @pytest.mark.parametrize(
        "field,value",
        [
            ("identity", "sha256:bbb"),
            ("protocol", "external-benchmark-v3"),
            ("endpoint", "http://elsewhere:11434"),
        ],
    )
    def test_every_named_input_separates_the_key(self, field: str, value: str) -> None:
        """Each key component must be load-bearing, or it is decoration.

        A component that does not change the key when it changes is not
        protecting anything, and the store would answer a question from a
        different model, protocol or server without saying so.
        """
        assert self._key(**{field: value}) != self._key()

    def test_a_changed_prompt_separates_the_key(self) -> None:
        other = {**PAYLOAD, "prompt": "the quick brown f"}
        assert self._key(payload=other) != self._key()

    def test_a_changed_seed_separates_the_key(self) -> None:
        options = {**PAYLOAD["options"], "seed": 1}
        assert self._key(payload={**PAYLOAD, "options": options}) != self._key()


class TestStore:
    def test_a_stored_reply_is_returned_verbatim(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        request = _request()
        cache.put(request, REPLY)
        assert cache.get(request) == REPLY

    def test_a_reply_survives_the_process_that_wrote_it(self, tmp_path: Path) -> None:
        """Resume is the whole point, and resume means a new process."""
        first = _cache(tmp_path)
        request = _request()
        first.put(request, REPLY)
        assert _cache(tmp_path).get(request) == REPLY

    def test_a_torn_final_line_costs_one_record_and_not_the_store(self, tmp_path: Path) -> None:
        """A kill mid-append must not make the earlier hours unreadable.

        This is the positive control for the crash-safety claim. Truncating the
        last line is exactly what an interrupted append leaves behind, and the
        store has to read every complete record that precedes it.
        """
        cache = _cache(tmp_path)
        requests = []
        for index in range(5):
            request = _request({**PAYLOAD, "prompt": f"prompt {index}"})
            requests.append(request)
            cache.put(request, {"response": str(index)})
        path = tmp_path / "calls.jsonl"
        text = path.read_text()
        path.write_text(text[: text.rindex("\n{") + 12])

        reloaded = _cache(tmp_path)
        assert len(reloaded) == 4
        assert reloaded.get(requests[0]) == {"response": "0"}
        assert reloaded.get(requests[4]) is None

    def test_a_store_written_under_another_schema_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "calls.jsonl"
        path.write_text(
            json.dumps({"schema": "something-else", "key": "k", "value": REPLY}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(CacheCorruptionError, match="unsupported schema"):
            _cache(tmp_path)

    def test_a_disabled_store_neither_answers_nor_writes(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path, enabled=False)
        cache.put(_request(), REPLY)
        assert cache.get(_request()) is None
        assert not (tmp_path / "calls.jsonl").exists()

    def test_an_unidentified_call_is_counted_and_not_stored(self, tmp_path: Path) -> None:
        """A ``None`` key means the model's identity could not be read.

        Storing it under any substitute key — the tag, a constant — would let a
        re-pulled model be answered from the old build's measurements. Not
        storing it is the safe direction, and counting it is what keeps the
        omission visible in the receipt instead of looking like a call that
        never happened.
        """
        cache = _cache(tmp_path)
        cache.put(None, REPLY)
        assert len(cache) == 0
        assert cache.stats.unidentified == 1
        assert cache.get(None) is None

    def test_a_divergent_reply_for_one_request_is_refused(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        request = _request()
        cache.put(request, REPLY)
        with pytest.raises(CacheCorruptionError, match="divergent reply"):
            cache.put(request, {"response": "different"})
        assert cache.get(request) == REPLY
        assert (tmp_path / "calls.jsonl").read_text().count(CACHE_SCHEMA) == 1

    def test_only_enumerated_ollama_envelope_volatility_is_duplicate_equivalent(
        self, tmp_path: Path
    ) -> None:
        cache = _cache(tmp_path)
        request = _request({**PAYLOAD, "prompt": "real Ollama envelope"})
        first = json.loads(json.dumps(OLLAMA_REPLY))
        later = {
            **json.loads(json.dumps(first)),
            "created_at": "2026-08-24T12:00:01Z",
            "total_duration": 1_100_000,
            "load_duration": 110_000,
            "prompt_eval_duration": 210_000,
            "eval_duration": 780_000,
        }
        cache.put(request, first)
        stored_line = cache.path.read_bytes()
        cache.put(request, later)

        assert cache.path.read_bytes() == stored_line
        assert cache.stats.stored == 1
        assert cache.get(request) == first, "the first complete envelope was not retained"

    def test_equivalent_duplicate_lines_retain_the_first_full_snapshot(
        self, tmp_path: Path
    ) -> None:
        request = _request({**PAYLOAD, "prompt": "two complete duplicate lines"})
        first = json.loads(json.dumps(OLLAMA_REPLY))
        later = {
            **json.loads(json.dumps(first)),
            "created_at": "2026-08-24T12:00:09Z",
            "total_duration": 9_000_000,
        }
        cache = _cache(tmp_path)
        cache.put(request, first)
        first_line = cache.path.read_bytes()
        other_path = tmp_path / "equivalent-line.jsonl"
        other = CallCache(other_path)
        other.put(request, later)
        cache.path.write_bytes(first_line + other_path.read_bytes())

        loaded = _cache(tmp_path)
        assert len(loaded) == 1
        assert loaded.get(request) == first

    @pytest.mark.parametrize(
        "field,changed",
        [
            ("response", '{"answer":"different"}'),
            ("context", [11, 22, 44]),
            ("prompt_eval_count", 8),
            ("eval_count", 4),
            ("logprobs", [{"token": "different", "logprob": -9.0}]),
            ("unreviewed_duration", 1),
        ],
    )
    def test_consumed_and_unlisted_reply_changes_remain_divergent(
        self, tmp_path: Path, field: str, changed: object
    ) -> None:
        cache = _cache(tmp_path)
        request = _request({**PAYLOAD, "prompt": f"semantic field {field}"})
        first = json.loads(json.dumps(OLLAMA_REPLY))
        cache.put(request, first)
        with pytest.raises(CacheCorruptionError, match="divergent reply"):
            cache.put(request, {**first, field: changed})
        assert cache.get(request) == first

    def test_entries_carry_the_payload_that_produced_them(self, tmp_path: Path) -> None:
        """The audit replays recorded payloads, so a store without them is unauditable."""
        cache = _cache(tmp_path)
        request = _request()
        cache.put(request, REPLY)
        [(key, payload, value)] = _cache(tmp_path).entries()
        assert (key, payload, value) == (request.key, PAYLOAD, REPLY)

    def test_put_get_and_entries_never_expose_mutable_cache_aliases(self, tmp_path: Path) -> None:
        """NEGATIVE CONTROL: caller mutation cannot forge an authenticated reply."""

        cache = _cache(tmp_path)
        caller_payload = json.loads(json.dumps(PAYLOAD))
        caller_reply = json.loads(json.dumps(REPLY_LOGPROBS))
        request = _request(caller_payload)
        original_key = request.key
        cache.put(request, caller_reply)
        digest = cache.records_digest()
        disk = cache.path.read_bytes()

        caller_payload["options"]["seed"] = -1
        caller_reply["response"] = "POST_PUT_FORGED"
        caller_reply["logprobs"][0]["top_logprobs"][0]["token"] = "FORGED"
        request.payload["prompt"] = "MUTATED_EXPOSED_REQUEST"
        assert request.key == original_key

        returned = cache.get(request)
        assert returned == REPLY_LOGPROBS
        assert returned is not None
        returned["response"] = "POST_GET_FORGED"
        returned["logprobs"][0]["top_logprobs"][0]["token"] = "FORGED_AGAIN"

        [(entry_key, entry_payload, entry_value)] = cache.entries()
        entry_payload["options"]["seed"] = -2
        entry_value["logprobs"][0]["top_logprobs"][0]["token"] = "ENTRY_FORGED"
        assert entry_key == original_key
        assert cache.get(request) == REPLY_LOGPROBS
        assert cache.records_digest() == digest
        assert cache.path.read_bytes() == disk

    def test_access_log_can_start_a_new_measurement_window_without_changing_store(
        self, tmp_path: Path
    ) -> None:
        """A precondition hit must not hide the same key from measurement provenance."""

        cache = _cache(tmp_path)
        request = _request()
        cache.put(request, REPLY)
        assert cache.accessed_keys == {request.key}
        records = cache.records_digest()
        stats = cache.stats.as_dict()

        cache.clear_access_log()
        assert cache.accessed_keys == frozenset()
        assert cache.records_digest() == records
        assert cache.stats.as_dict() == stats

        assert cache.get(request) == REPLY
        assert cache.accessed_keys == {request.key}


class TestLoadSkipsWhatItCannotTrust:
    """Every malformed line costs itself and nothing else."""

    def test_blank_lines_between_records_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        """A store edited by hand, or concatenated from two, can carry blank
        lines. Each is nothing, not a torn record, and reading past it must not
        cost the records around it."""
        cache = _cache(tmp_path)
        first = _request({**PAYLOAD, "prompt": "first"})
        second = _request({**PAYLOAD, "prompt": "second"})
        cache.put(first, {"response": "0"})
        cache.put(second, {"response": "1"})
        path = tmp_path / "calls.jsonl"
        lines = path.read_text().splitlines()
        path.write_text(lines[0] + "\n\n\n" + lines[1] + "\n", encoding="utf-8")

        reloaded = _cache(tmp_path)
        assert len(reloaded) == 2
        assert reloaded.get(first) == {"response": "0"}
        assert reloaded.get(second) == {"response": "1"}

    def test_a_record_with_the_wrong_value_shape_is_refused(self, tmp_path: Path) -> None:
        """Invalid complete records invalidate the store rather than disappearing."""
        path = tmp_path / "calls.jsonl"
        cache = CallCache(path)
        cache.put(_request(), REPLY)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["value"] = "not-a-dict"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(CacheCorruptionError, match="payload and value must be objects"):
            _cache(tmp_path)

    def test_malformed_interior_json_is_fatal(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.put(_request(), REPLY)
        good_line = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
        (tmp_path / "calls.jsonl").write_text(
            good_line + "{malformed interior\n" + good_line,
            encoding="utf-8",
        )
        with pytest.raises(CacheCorruptionError, match="malformed interior JSON"):
            _cache(tmp_path)

    def test_a_terminated_malformed_final_line_is_not_a_torn_append(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.put(_request(), REPLY)
        with (tmp_path / "calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{malformed but terminated\n")
        with pytest.raises(CacheCorruptionError, match="malformed interior JSON"):
            _cache(tmp_path)

    @pytest.mark.parametrize("field", ["payload", "endpoint", "protocol", "identity"])
    def test_changed_key_material_is_detected_on_load(self, tmp_path: Path, field: str) -> None:
        cache = _cache(tmp_path)
        cache.put(_request(), REPLY)
        path = tmp_path / "calls.jsonl"
        record = json.loads(path.read_text(encoding="utf-8"))
        if field == "payload":
            record[field] = {**record[field], "prompt": "forged"}
        else:
            record[field] = f"forged-{field}"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(CacheCorruptionError, match="key does not match"):
            _cache(tmp_path)

    def test_changed_reply_is_detected_on_load(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        cache.put(_request(), REPLY)
        path = tmp_path / "calls.jsonl"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["value"] = {"response": "forged"}
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(CacheCorruptionError, match="value digest"):
            _cache(tmp_path)

    def test_a_torn_final_fragment_is_removed_before_the_next_append(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        first = _request({**PAYLOAD, "prompt": "first"})
        second = _request({**PAYLOAD, "prompt": "second"})
        cache.put(first, REPLY)
        path = tmp_path / "calls.jsonl"
        with path.open("ab") as handle:
            handle.write(b'{"schema":"ailm-call')

        recovered = _cache(tmp_path)
        assert recovered.get(first) == REPLY
        recovered.put(second, {"response": "second"})
        final = _cache(tmp_path)
        assert final.get(first) == REPLY
        assert final.get(second) == {"response": "second"}


class TestIncrementalReconciliation:
    """The append fast path is linear in new bytes, without weakening validation."""

    def test_one_writer_does_not_reparse_historical_bytes_on_each_put(self, tmp_path: Path) -> None:
        """Positive and negative controls for the non-quadratic append path."""

        seed = _cache(tmp_path)
        seed.put(_request({**PAYLOAD, "prompt": "seed"}), {"response": "seed"})
        cache = _cache(tmp_path)
        initial_size = cache.path.stat().st_size
        baseline = cache.reconciliation_stats
        assert baseline == {"full": 1, "tail": 0, "bytes": initial_size}

        for index in range(40):
            cache.put(
                _request({**PAYLOAD, "prompt": f"fast-{index}"}),
                {"response": str(index), "padding": "x" * 256},
            )

        assert cache.path.stat().st_size > initial_size
        assert cache.reconciliation_stats == baseline

        # NEGATIVE CONTROL: the counters are live. A fresh process-equivalent
        # instance must actually read the now-larger file once.
        reloaded = _cache(tmp_path)
        assert reloaded.reconciliation_stats == {
            "full": 1,
            "tail": 0,
            "bytes": reloaded.path.stat().st_size,
        }
        assert reloaded.reconciliation_stats["bytes"] > baseline["bytes"]

    def test_external_append_is_validated_from_the_prior_eof(self, tmp_path: Path) -> None:
        first = _request({**PAYLOAD, "prompt": "first"})
        external = _request({**PAYLOAD, "prompt": "external"})
        final = _request({**PAYLOAD, "prompt": "final"})
        writer = _cache(tmp_path)
        writer.put(first, {"response": "first"})
        observer = _cache(tmp_path)
        before = observer.reconciliation_stats
        before_external = observer.path.stat().st_size

        writer.put(external, {"response": "external"})
        after_external = observer.path.stat().st_size
        observer.put(final, {"response": "final"})

        assert observer.reconciliation_stats == {
            "full": before["full"],
            "tail": before["tail"] + 1,
            "bytes": before["bytes"] + after_external - before_external,
        }
        assert observer.get(first) == {"response": "first"}
        assert observer.get(external) == {"response": "external"}
        assert observer.get(final) == {"response": "final"}

    def test_store_created_after_an_empty_instance_is_fully_loaded(self, tmp_path: Path) -> None:
        observer = _cache(tmp_path)
        external = _request({**PAYLOAD, "prompt": "created externally"})
        writer = _cache(tmp_path)
        writer.put(external, {"response": "external"})

        final = _request({**PAYLOAD, "prompt": "observer append"})
        observer.put(final, {"response": "final"})
        assert observer.reconciliation_stats["full"] == 1
        assert observer.get(external) == {"response": "external"}
        assert observer.get(final) == {"response": "final"}

    def test_replacement_reloads_the_new_inode_before_append(self, tmp_path: Path) -> None:
        old = _request({**PAYLOAD, "prompt": "old inode"})
        replacement = _request({**PAYLOAD, "prompt": "replacement inode"})
        appended = _request({**PAYLOAD, "prompt": "after replacement"})
        cache = _cache(tmp_path)
        cache.put(old, {"response": "old"})
        baseline = cache.reconciliation_stats

        replacement_path = tmp_path / "replacement.jsonl"
        replacement_cache = CallCache(replacement_path)
        replacement_cache.put(replacement, {"response": "replacement"})
        os.replace(replacement_path, cache.path)
        cache.put(appended, {"response": "appended"})

        assert cache.reconciliation_stats["full"] == baseline["full"] + 1
        assert cache.get(old) is None
        assert cache.get(replacement) == {"response": "replacement"}
        assert cache.get(appended) == {"response": "appended"}

    def test_truncation_reloads_the_surviving_prefix_before_append(self, tmp_path: Path) -> None:
        requests = [_request({**PAYLOAD, "prompt": f"truncate-{index}"}) for index in range(3)]
        cache = _cache(tmp_path)
        for index, request in enumerate(requests):
            cache.put(request, {"response": str(index)})
        first_line = cache.path.read_bytes().splitlines(keepends=True)[0]
        with cache.path.open("r+b") as handle:
            handle.truncate(len(first_line))

        after = _request({**PAYLOAD, "prompt": "after truncate"})
        cache.put(after, {"response": "after"})
        assert cache.reconciliation_stats["full"] == 1
        assert cache.get(requests[0]) == {"response": "0"}
        assert cache.get(requests[1]) is None
        assert cache.get(requests[2]) is None
        assert cache.get(after) == {"response": "after"}

    def test_same_size_external_mutation_forces_full_validation(self, tmp_path: Path) -> None:
        request = _request({**PAYLOAD, "prompt": "mutated in place"})
        cache = _cache(tmp_path)
        cache.put(request, {"response": "original"})
        original = cache.path.read_bytes()
        forged = original.replace(b'"response":"original"', b'"response":"forged!!"')
        assert len(forged) == len(original)
        cache.path.write_bytes(forged)

        with pytest.raises(CacheCorruptionError, match="value digest"):
            cache.put(_request({**PAYLOAD, "prompt": "must not append"}), REPLY)
        assert cache.path.read_bytes() == forged

    def test_external_torn_tail_is_recovered_by_the_next_locked_append(
        self, tmp_path: Path
    ) -> None:
        first = _request({**PAYLOAD, "prompt": "before torn tail"})
        second = _request({**PAYLOAD, "prompt": "after torn tail"})
        cache = _cache(tmp_path)
        cache.put(first, {"response": "first"})
        with cache.path.open("ab") as handle:
            handle.write(b'{"schema":"ailm-call')

        cache.put(second, {"response": "second"})
        assert cache.reconciliation_stats["tail"] == 1
        assert cache.get(first) == {"response": "first"}
        assert cache.get(second) == {"response": "second"}
        assert b'{"schema":"ailm-call' not in cache.path.read_bytes()
        assert len(_cache(tmp_path)) == 2

    def test_valid_unterminated_record_is_separated_before_append(self, tmp_path: Path) -> None:
        first = _request({**PAYLOAD, "prompt": "valid but unterminated"})
        second = _request({**PAYLOAD, "prompt": "after separator"})
        writer = _cache(tmp_path)
        writer.put(first, {"response": "first"})
        writer.path.write_bytes(writer.path.read_bytes().removesuffix(b"\n"))

        recovered = _cache(tmp_path)
        assert recovered.get(first) == {"response": "first"}
        recovered.put(second, {"response": "second"})
        assert recovered.path.read_bytes().endswith(b"\n")
        assert len(recovered.path.read_bytes().splitlines()) == 2
        assert _cache(tmp_path).get(second) == {"response": "second"}

    def test_growth_after_unterminated_record_takes_one_full_validation(
        self, tmp_path: Path
    ) -> None:
        first = _request({**PAYLOAD, "prompt": "unterminated before growth"})
        external = _request({**PAYLOAD, "prompt": "external after unterminated"})
        final = _request({**PAYLOAD, "prompt": "final after full reload"})
        writer = _cache(tmp_path)
        writer.put(first, {"response": "first"})
        writer.path.write_bytes(writer.path.read_bytes().removesuffix(b"\n"))
        observer = _cache(tmp_path)
        baseline = observer.reconciliation_stats

        other_path = tmp_path / "external-line.jsonl"
        other = CallCache(other_path)
        other.put(external, {"response": "external"})
        with observer.path.open("ab") as handle:
            handle.write(b"\n" + other_path.read_bytes())
        observer.put(final, {"response": "final"})

        assert observer.reconciliation_stats["full"] == baseline["full"] + 1
        assert observer.reconciliation_stats["tail"] == 0
        assert observer.get(first) == {"response": "first"}
        assert observer.get(external) == {"response": "external"}
        assert observer.get(final) == {"response": "final"}

    def test_external_terminated_corruption_is_not_treated_as_a_torn_tail(
        self, tmp_path: Path
    ) -> None:
        cache = _cache(tmp_path)
        cache.put(_request({**PAYLOAD, "prompt": "before corruption"}), REPLY)
        corrupted = cache.path.read_bytes() + b"{malformed interior\n"
        cache.path.write_bytes(corrupted)

        with pytest.raises(CacheCorruptionError, match="malformed interior JSON"):
            cache.put(_request({**PAYLOAD, "prompt": "must fail"}), REPLY)
        assert cache.path.read_bytes() == corrupted

    def test_external_divergent_duplicate_is_rejected_during_tail_reconciliation(
        self, tmp_path: Path
    ) -> None:
        shared = _request({**PAYLOAD, "prompt": "duplicate key"})
        cache = _cache(tmp_path)
        cache.put(shared, {"response": "first"})

        other_path = tmp_path / "divergent.jsonl"
        other = CallCache(other_path)
        other.put(shared, {"response": "meaningfully different"})
        with cache.path.open("ab") as handle:
            handle.write(other_path.read_bytes())

        with pytest.raises(CacheCorruptionError, match="divergent duplicate"):
            cache.put(_request({**PAYLOAD, "prompt": "must not append"}), REPLY)

    def test_exact_duplicate_put_is_idempotent(self, tmp_path: Path) -> None:
        cache = _cache(tmp_path)
        request = _request({**PAYLOAD, "prompt": "same call twice"})
        cache.put(request, {"response": "same"})
        first_bytes = cache.path.read_bytes()
        cache.put(request, {"response": "same"})
        assert cache.path.read_bytes() == first_bytes
        assert cache.stats.stored == 1


class TestReplayAndPublicationReconciliation:
    """No observable cache state may outrun validation of its backing file."""

    @pytest.mark.parametrize(
        "boundary",
        ["get", "entries", "records_digest", "provenance_binding", "receipt"],
    )
    def test_every_trust_boundary_rejects_hit_only_same_size_mutation(
        self, tmp_path: Path, boundary: str
    ) -> None:
        cache = _cache(tmp_path)
        request = _request({**PAYLOAD, "prompt": f"mutation before {boundary}"})
        cache.put(request, {"response": "original"})
        original = cache.path.read_bytes()
        forged = original.replace(b'"response":"original"', b'"response":"forged!!"')
        assert len(forged) == len(original)
        cache.path.write_bytes(forged)

        with pytest.raises(CacheCorruptionError, match="value digest"):
            if boundary == "get":
                cache.get(request)
            elif boundary == "entries":
                cache.entries()
            elif boundary == "records_digest":
                cache.records_digest()
            elif boundary == "provenance_binding":
                cache.provenance_binding({request.key})
            else:
                cache.receipt()

    def test_hit_only_truncation_cannot_replay_a_removed_record(self, tmp_path: Path) -> None:
        first = _request({**PAYLOAD, "prompt": "survives truncation"})
        removed = _request({**PAYLOAD, "prompt": "removed by truncation"})
        cache = _cache(tmp_path)
        cache.put(first, {"response": "first"})
        cache.put(removed, {"response": "removed"})
        first_line = cache.path.read_bytes().splitlines(keepends=True)[0]
        with cache.path.open("r+b") as handle:
            handle.truncate(len(first_line))

        assert cache.get(removed) is None
        assert cache.get(first) == {"response": "first"}
        assert cache.reconciliation_stats["full"] == 1

    def test_hit_only_replacement_publishes_only_the_new_inode(self, tmp_path: Path) -> None:
        old = _request({**PAYLOAD, "prompt": "old replay inode"})
        new = _request({**PAYLOAD, "prompt": "new replay inode"})
        cache = _cache(tmp_path)
        cache.put(old, {"response": "old"})

        replacement_path = tmp_path / "read-replacement.jsonl"
        replacement = CallCache(replacement_path)
        replacement.put(new, {"response": "new"})
        os.replace(replacement_path, cache.path)

        [(key, payload, value)] = cache.entries()
        assert (key, payload, value) == (new.key, new.payload_snapshot(), {"response": "new"})
        assert cache.get(old) is None

    def test_disappeared_validated_store_fails_closed_on_replay(self, tmp_path: Path) -> None:
        request = _request({**PAYLOAD, "prompt": "store disappears"})
        cache = _cache(tmp_path)
        cache.put(request, REPLY)
        os.replace(cache.path, tmp_path / "moved-store.jsonl")
        with pytest.raises(CacheCorruptionError, match="disappeared"):
            cache.get(request)

    def test_prefix_rewrite_plus_valid_growth_is_caught_before_replay(self, tmp_path: Path) -> None:
        """NEGATIVE CONTROL: a valid tail cannot bless an unseen forged prefix."""

        first = _request({**PAYLOAD, "prompt": "prefix record"})
        cache = _cache(tmp_path)
        cache.put(first, {"response": "first"})
        original_prefix = cache.path.read_bytes()
        forged_prefix = original_prefix.replace(b'"response":"first"', b'"response":"other"')
        assert len(forged_prefix) == len(original_prefix)

        external_path = tmp_path / "valid-growth.jsonl"
        external = CallCache(external_path)
        external_request = _request({**PAYLOAD, "prompt": "valid external tail"})
        external.put(external_request, {"response": "external"})
        cache.path.write_bytes(forged_prefix + external_path.read_bytes())

        # A writer may reconcile just the structurally valid tail and keep the
        # append path fast, but that state remains explicitly provisional.
        final = _request({**PAYLOAD, "prompt": "append before replay"})
        cache.put(final, {"response": "final"})
        assert cache.reconciliation_stats["tail"] == 1
        with pytest.raises(CacheCorruptionError, match="value digest"):
            cache.get(first)

    def test_direct_replay_of_growth_validates_prefix_and_tail_together(
        self, tmp_path: Path
    ) -> None:
        first = _request({**PAYLOAD, "prompt": "direct replay prefix"})
        cache = _cache(tmp_path)
        cache.put(first, {"response": "first"})
        prefix = cache.path.read_bytes()
        forged_prefix = prefix.replace(b'"response":"first"', b'"response":"other"')
        assert len(forged_prefix) == len(prefix)

        tail_path = tmp_path / "direct-valid-tail.jsonl"
        tail = CallCache(tail_path)
        tail.put(
            _request({**PAYLOAD, "prompt": "direct valid tail"}),
            {"response": "tail"},
        )
        cache.path.write_bytes(forged_prefix + tail_path.read_bytes())

        with pytest.raises(CacheCorruptionError, match="value digest"):
            cache.get(first)
        assert cache.reconciliation_stats["tail"] == 0

    def test_unchanged_replays_do_not_repeat_full_validation(self, tmp_path: Path) -> None:
        request = _request({**PAYLOAD, "prompt": "many unchanged hits"})
        cache = _cache(tmp_path)
        cache.put(request, REPLY)
        baseline = cache.reconciliation_stats
        for _ in range(40):
            assert cache.get(request) == REPLY
        assert cache.reconciliation_stats == baseline


class TestSchemaBoundary:
    def test_v1_row_is_rejected_as_an_unsupported_schema(self, tmp_path: Path) -> None:
        """NEGATIVE CONTROL: a plausible old row invalidates the whole store."""

        current_path = tmp_path / "current.jsonl"
        current = CallCache(current_path)
        current.put(_request(), REPLY)
        assert CallCache(current_path).get(_request()) == REPLY

        unsupported_path = tmp_path / "unsupported-v1.jsonl"
        unsupported_path.write_text(
            json.dumps(
                {
                    "schema": "ailm-call-cache-v1",
                    "key": "0" * 64,
                    "payload": PAYLOAD,
                    "value": {"response": "FORGED"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(
            CacheCorruptionError,
            match=r"unsupported schema 'ailm-call-cache-v1'; expected 'ailm-call-cache-v2'",
        ):
            CallCache(unsupported_path)


@pytest.mark.timeout(240)
def test_two_processes_append_without_lost_or_interleaved_records(tmp_path: Path) -> None:
    """The OS lock, not only an in-process mutex, serializes JSONL appends."""
    path = tmp_path / "calls.jsonl"
    program = """
import sys
from pathlib import Path
from ailm.call_cache import CacheRequest, CallCache

path = Path(sys.argv[1])
offset = int(sys.argv[2])
cache = CallCache(path)
for index in range(offset, offset + 25):
    request = CacheRequest(
        payload={"model": "m", "prompt": f"prompt-{index}"},
        endpoint="http://127.0.0.1:11434",
        protocol="process-lock-test",
        identity="sha256:process-lock",
    )
    cache.put(request, {"response": str(index)})
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program, str(path), str(offset)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for offset in (0, 25)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, f"stdout:\n{stdout}\nstderr:\n{stderr}"

    cache = CallCache(path)
    assert len(cache) == 50
    prompts = {payload["prompt"] for _, payload, _ in cache.entries()}
    assert prompts == {f"prompt-{index}" for index in range(50)}


@pytest.mark.timeout(240)
def test_concurrent_identical_semantic_replies_collapse_to_one_record(tmp_path: Path) -> None:
    """Key order differs, but canonical JSON semantics and reply digests agree."""

    path = tmp_path / "calls.jsonl"
    first = {"response": "same", "metadata": {"a": 1, "b": 2}}
    second = {"metadata": {"b": 2, "a": 1}, "response": "same"}
    results = _race_same_request(path, tmp_path, (first, second))
    assert [result.returncode for result in results] == [0, 0], [
        (result.stdout, result.stderr) for result in results
    ]
    assert len(path.read_bytes().splitlines()) == 1
    assert len(CallCache(path)) == 1


@pytest.mark.timeout(240)
def test_concurrent_ollama_timing_variants_collapse_to_one_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    first = json.loads(json.dumps(OLLAMA_REPLY))
    later = {
        **json.loads(json.dumps(first)),
        "created_at": "2026-08-24T12:00:02Z",
        "total_duration": 1_500_000,
        "load_duration": 150_000,
        "prompt_eval_duration": 250_000,
        "eval_duration": 1_100_000,
    }
    results = _race_same_request(path, tmp_path, (first, later))
    assert [result.returncode for result in results] == [0, 0], [
        (result.stdout, result.stderr) for result in results
    ]
    assert len(path.read_bytes().splitlines()) == 1
    stored = CallCache(path).entries()[0][2]
    assert stored in (first, later), "the winning complete snapshot was not retained"


@pytest.mark.timeout(240)
def test_concurrent_meaningfully_divergent_replies_never_collapse(tmp_path: Path) -> None:
    """One exact response wins; the other process must surface divergence."""

    path = tmp_path / "calls.jsonl"
    results = _race_same_request(
        path,
        tmp_path,
        ({"response": "first"}, {"response": "different"}),
    )
    assert sorted(result.returncode for result in results) == [0, 23], [
        (result.stdout, result.stderr) for result in results
    ]
    assert len(path.read_bytes().splitlines()) == 1
    stored = CallCache(path).entries()[0][2]
    assert stored in ({"response": "first"}, {"response": "different"})


class TestModelDigest:
    """Content-bound identity resolution on a real loopback server.

    The HTTP client is the external boundary here; everything on our side of it
    — path priority, the content-bound manifest fallback, and the refusal to
    guess from metadata — is exercised for real.
    """

    def _digest(self, server, body: dict) -> str | None:
        server.enqueue_json("/api/show", body)
        result = model_digest("gemma3:4b", server.base_url)
        request = server.requests_for("/api/show")[-1]
        assert request.method == "POST"
        assert json.loads(request.body) == {"model": "gemma3:4b"}
        return result

    def test_a_reported_digest_is_returned_from_either_declared_path(
        self, local_http_server
    ) -> None:
        top = "a" * 64
        nested = "B" * 64
        assert self._digest(local_http_server, {"digest": f"sha256:{top}"}) == f"sha256:{top}"
        assert (
            self._digest(local_http_server, {"details": {"digest": nested}})
            == f"sha256:{nested.lower()}"
        )

    def test_noncryptographic_digest_labels_are_not_accepted_as_identity(
        self, local_http_server
    ) -> None:
        assert self._digest(local_http_server, {"digest": "sha256:not-a-digest"}) is None
        assert (
            self._digest(
                local_http_server,
                {"details": {"digest": "Q4_K_M"}},
            )
            is None
        )
        assert (
            self._digest(
                local_http_server,
                {"model_info": {"general.file_type": "Q4_K_M"}},
            )
            is None
        )

    def test_a_missing_digest_hashes_a_content_bound_manifest_not_the_tag(
        self, local_http_server
    ) -> None:
        """The manifest changes with the build and the tag does not, so hashing
        the manifest is still identity. The defect this guards: two different
        builds under one tag sharing a key."""
        first_blob = "a" * 64
        second_blob = "b" * 64
        same = self._digest(
            local_http_server, {"modelfile": f"FROM /models/blobs/sha256-{first_blob}"}
        )
        again = self._digest(
            local_http_server, {"modelfile": f"FROM /models/blobs/sha256-{first_blob}"}
        )
        other = self._digest(
            local_http_server, {"modelfile": f"FROM /models/blobs/sha256-{second_blob}"}
        )
        assert same is not None and same == again
        assert other != same, "two different builds hashed to one identity"
        assert "gemma3" not in same, "the tag leaked into the identity"

    def test_same_file_type_different_builds_never_share_an_identity(
        self, local_http_server
    ) -> None:
        """Reproduce the collision: quantization format is not model content."""
        first = self._digest(
            local_http_server,
            {
                "modelfile": f"FROM sha256:{'a' * 64}",
                "model_info": {"general.file_type": "Q4_K_M"},
            },
        )
        second = self._digest(
            local_http_server,
            {
                "modelfile": f"FROM sha256:{'b' * 64}",
                "model_info": {"general.file_type": "Q4_K_M"},
            },
        )
        assert first is not None and second is not None
        assert first != second

    @pytest.mark.parametrize(
        "body",
        [
            {"license": "same license", "tensors": [{"name": "changed"}]},
            {"template": "same template", "capabilities": ["completion"]},
            {"model_info": {"general.file_type": "Q4_K_M", "layers": 26}},
            {"modelfile": "FROM mutable-tag:latest", "details": {"family": "qwen"}},
        ],
    )
    def test_metadata_without_a_content_digest_cannot_become_identity(
        self, local_http_server, body: dict
    ) -> None:
        """NEGATIVE CONTROL: changing unbound tensor metadata cannot collide."""
        assert self._digest(local_http_server, body) is None

    def test_a_body_naming_nothing_returns_none_not_a_guess(self, local_http_server) -> None:
        """An empty answer must propagate as "do not cache". Any substitute —
        the tag, a constant — would let a re-pulled model replay the old
        build's measurements."""
        assert self._digest(local_http_server, {}) is None
        assert self._digest(local_http_server, {"modelfile": None, "details": None}) is None

    def test_an_unreachable_server_returns_none(self, local_http_server) -> None:
        local_http_server.enqueue_disconnect("/api/show")
        assert model_digest("gemma3:4b", local_http_server.base_url) is None

    def test_a_body_that_is_not_json_returns_none(self, local_http_server) -> None:
        local_http_server.enqueue_bytes("/api/show", b"<html>not json</html>")
        assert model_digest("gemma3:4b", local_http_server.base_url) is None


class TestReceipt:
    def test_hits_and_misses_are_reported_separately(self, tmp_path: Path) -> None:
        """A run that replayed everything and one that measured everything
        produce identical claims. A receipt with only a total cannot tell them
        apart, which is the whole reason it is written."""
        cache = _cache(tmp_path)
        request = _request()
        cache.put(request, REPLY)
        cache.get(request)
        cache.get(_request({**PAYLOAD, "prompt": "absent"}))
        cache.stats.misses += 1
        receipt = cache.receipt()
        assert set(receipt) == {
            "schema",
            "path",
            "enabled",
            "records",
            "records_sha256",
            "hits",
            "misses",
            "stored",
            "unidentified",
            "fresh_fraction",
        }
        assert receipt["hits"] == 1
        assert receipt["misses"] == 1
        assert receipt["fresh_fraction"] == pytest.approx(0.5)

    def test_fresh_fraction_of_an_untouched_process_is_zero_not_undefined(self) -> None:
        assert CacheStats().fresh_fraction == 0.0


def test_the_store_is_an_input_and_not_a_recorded_claim(tmp_path: Path) -> None:
    """It belongs in neither ``data/`` nor ``output/``, for different reasons.

    Not ``data/``: a cache beside ``claims.json`` reads as part of the record.
    It is neither measured nor asserted, only how a run avoided re-asking a
    question.

    Not ``output/`` either, which is where it used to live and where this test
    used to require it. ``output/`` promises to be regenerable and to hold no
    source of truth; this store cannot be rebuilt without a live model server
    and some hours of calls, and the release profile's coverage floor depends on
    it — falling from about 93% to 91.87% on a cold store with no failing test
    naming the cause. Clearing the derived tree to prove the project rebuilds
    destroyed it once. An artifact that cannot be regenerated is an input
    whatever directory it sits in, so it moved to ``cache/``.

    NEGATIVE CONTROL: it must still be under the project root and named as a
    store, so this cannot pass on a path that has simply gone missing.
    """
    path = default_cache_path(tmp_path)
    assert path.parent.name == "cache"
    assert "output" not in path.parts
    assert "data" not in path.parts
    assert path.is_relative_to(tmp_path) and path.suffix == ".jsonl"


class TestSeam:
    """The store is wired at the one function every model call passes through."""

    def test_every_ask_helper_routes_through_the_cached_poster(self) -> None:
        """Three helpers differ in what they read from a reply, not in how they
        get one. If any obtained its own reply, that arm would silently opt out
        of resumability and a census would restart it on every attempt."""
        # inspect.getsource, not a hardcoded module path: the structural
        # requirement is about this function, and reading it by name survives
        # the module being split or moved. The path form broke the moment
        # `benchmarks.py` became `benchmarks/`.
        helpers = (
            benchmarks.ask_ollama,
            benchmarks.ask_ollama_answer_candidates,
            benchmarks.ask_ollama_distribution,
        )
        names = {helper.__name__ for helper in helpers}
        for helper in helpers:
            body = inspect.getsource(helper)
            assert "post_generate_uncached(" not in body, (
                f"{helper.__name__} bypasses the cached poster"
            )
            # Either it posts through the cached poster itself, or it delegates
            # to a sibling that does. `ask_ollama_distribution` takes the second
            # route; what none of them may do is obtain a reply of its own.
            delegates = any(f"{other}(" in body for other in names - {helper.__name__})
            assert "_post_generate(" in body or delegates, (
                f"{helper.__name__} neither posts through the cached poster nor "
                f"delegates to a helper that does; that arm cannot resume"
            )

    def test_caching_is_off_unless_a_caller_asks_for_it(self) -> None:
        """A cache that switches itself on can answer a question nobody knew was
        being answered. Tests, the oracle, and ad-hoc scripts must keep the
        previous behavior with no opt-out to remember."""
        from ailm import benchmarks

        assert benchmarks.call_cache() is None

    def test_environment_override_selects_one_absolute_isolated_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tests may isolate a recorder without changing its command line.

        POSITIVE: the absolute environment path is the path actually opened.
        NEGATIVE: an explicit path takes precedence, so inherited process state
        cannot redirect a caller that already selected an isolated file.
        """
        isolated = tmp_path / "environment" / "calls.jsonl"
        explicit = tmp_path / "explicit" / "calls.jsonl"
        monkeypatch.setenv(benchmarks.CALL_CACHE_PATH_ENV, str(isolated))
        benchmarks.enable_call_cache(enabled=False)
        try:
            store = benchmarks.enable_call_cache()
            assert store is not None and store.path == isolated
            benchmarks.enable_call_cache(enabled=False)

            store = benchmarks.enable_call_cache(explicit)
            assert store is not None and store.path == explicit
        finally:
            benchmarks.enable_call_cache(enabled=False)

    @pytest.mark.parametrize("source", ["environment", "argument"])
    def test_relative_cache_override_is_refused(
        self, source: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative cache path changes identity with the caller's cwd."""
        benchmarks.enable_call_cache(enabled=False)
        monkeypatch.delenv(benchmarks.CALL_CACHE_PATH_ENV, raising=False)
        try:
            if source == "environment":
                monkeypatch.setenv(benchmarks.CALL_CACHE_PATH_ENV, "cache/test-calls.jsonl")
                with pytest.raises(ValueError, match="absolute path"):
                    benchmarks.enable_call_cache()
            else:
                with pytest.raises(ValueError, match="absolute path"):
                    benchmarks.enable_call_cache(Path("cache/test-calls.jsonl"))
        finally:
            benchmarks.enable_call_cache(enabled=False)

    def test_publication_mode_refuses_an_isolated_override_before_opening_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A receipt validator and recorder must not silently read two stores."""
        isolated = tmp_path / "publication-mismatch" / "calls.jsonl"
        monkeypatch.setenv(benchmarks.CALL_CACHE_PATH_ENV, str(isolated))
        benchmarks.enable_call_cache(enabled=False)
        with pytest.raises(ValueError, match="canonical repository store"):
            benchmarks.enable_call_cache(require_default_path=True)
        assert not isolated.exists(), "publication refusal happened after opening the override"
        assert benchmarks.call_cache() is None

    @pytest.mark.parametrize(
        ("script_name", "late_refusal"),
        [
            ("record_external_benchmark_claims.py", "Ollama/model unavailable"),
            ("record_logprob_drift_claims.py", "no stored calls requested logprobs"),
        ],
    )
    @pytest.mark.timeout(600)
    def test_network_publication_recorders_refuse_cache_override_before_measurement(
        self, tmp_path: Path, script_name: str, late_refusal: str
    ) -> None:
        """The canonical CLI consumes the strict path mode, not just the API.

        The real subprocess targets the tracked ledger but snapshots it first.
        A noncanonical cache path must fail before creating that path or reaching
        the recorder's later live/input refusal.
        """
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / script_name
        ledger = project_root / "data" / "claims.json"
        ledger_before = ledger.read_bytes()
        isolated = tmp_path / "external-publication" / "calls.jsonl"
        environment = os.environ.copy()
        environment.pop("AILM_CLAIMS_PATH", None)
        environment[benchmarks.CALL_CACHE_PATH_ENV] = str(isolated)

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0
        assert "publication call-cache path must be the canonical repository store" in combined
        assert late_refusal not in combined, "the recorder reached its later preflight first"
        assert not isolated.exists()
        assert ledger.read_bytes() == ledger_before


class TestVerifyConventions:
    """The audit returns differently as a precondition than as a standalone run.

    Both readings are defensible and they conflict, which is why they are
    separated by an argument rather than settled by one rule. A cold store has
    nothing to disagree with: as a precondition that must not block a multi-hour
    run before it has measured anything, and as an audit somebody explicitly
    asked for it must not report success on a comparison that never happened.
    Server unreachability is diagnostically distinct from disagreement but
    blocks both modes because neither may call an unmade comparison verified.
    """

    @staticmethod
    def _recorder():
        import importlib.util

        path = (
            Path(__file__).resolve().parents[1] / "scripts" / "record_external_benchmark_claims.py"
        )
        spec = importlib.util.spec_from_file_location("_bench_recorder_verify", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_cold_store_passes_as_a_precondition_and_fails_as_an_audit(
        self, tmp_path: Path
    ) -> None:
        recorder = self._recorder()
        empty = CallCache(path=tmp_path / "calls.jsonl")
        assert recorder._verify_cache(5, store=empty, precondition=True) == 0
        assert recorder._verify_cache(5, store=empty, precondition=False) == 1

    @pytest.mark.parametrize("missing_option", ["use_mmap", "num_ctx"])
    def test_legacy_loader_only_store_is_cold_but_cannot_pass_a_strict_audit(
        self, tmp_path: Path, local_http_server, capsys, missing_option
    ) -> None:
        recorder = self._recorder()
        store = _cache(tmp_path)
        legacy = json.loads(json.dumps(PAYLOAD))
        legacy["options"].pop(missing_option)
        store.put(_request(legacy), REPLY)
        before = store.path.read_bytes()
        assert (
            recorder._verify_cache(
                5,
                store=store,
                precondition=True,
                allow_missing_strata=True,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == recorder.CACHE_VERIFY_OK
        )
        assert "not an audit pass" in capsys.readouterr().out
        for precondition in (False, True):
            with pytest.raises(SystemExit, match="current loader configuration has no records"):
                recorder._verify_cache(
                    5,
                    store=store,
                    precondition=precondition,
                    endpoint=local_http_server.base_url,
                    expected_identities=_registered_test_identities(),
                )
        assert not local_http_server.requests
        assert store.path.read_bytes() == before

    @pytest.mark.parametrize("missing_option", ["use_mmap", "num_ctx"])
    def test_legacy_loader_rows_are_preserved_but_never_replayed(
        self, tmp_path: Path, local_http_server, missing_option
    ) -> None:
        recorder = self._recorder()
        store = _fully_stratified_cache(tmp_path, endpoint=local_http_server.base_url)
        initial = list(store.entries())
        for _key, payload, reply in initial:
            legacy = json.loads(json.dumps(payload))
            legacy["options"].pop(missing_option)
            store.put(
                _request(legacy, identity=_registered_test_identities()[legacy["model"]]), reply
            )
        before = store.path.read_bytes()
        _enqueue_matching_strata(local_http_server)
        _enqueue_current_model_identities(local_http_server)
        assert (
            recorder._verify_cache(
                1,
                store=store,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == recorder.CACHE_VERIFY_OK
        )
        sent = local_http_server.requests_for("/api/generate")
        assert len(sent) == len(recorder._VERIFY_STRATA)
        assert all(json.loads(request.body)["options"].get("use_mmap") is False for request in sent)
        assert all(json.loads(request.body)["options"].get("num_ctx") == 8192 for request in sent)
        assert len(store.entries()) == len(initial) * 2
        assert store.path.read_bytes() == before

    @pytest.mark.timeout(600)
    def test_partial_model_measurement_is_refused_before_any_server_call(self) -> None:
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "record_external_benchmark_claims.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), "gemma3"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "partial model measurements" in result.stderr

    def test_every_model_request_kind_cell_passes_a_strict_audit(
        self, tmp_path: Path, local_http_server, capsys
    ) -> None:
        """Positive control: a real replay from every declared cell succeeds."""
        recorder = self._recorder()
        store = _fully_stratified_cache(tmp_path, endpoint=local_http_server.base_url)
        _enqueue_matching_strata(local_http_server)
        _enqueue_current_model_identities(local_http_server)
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=False,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == 0
        )
        assert "across 6 of 6 declared stratum(s)" in capsys.readouterr().out

    def test_an_unreachable_server_is_not_reported_as_a_cache_mismatch(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """A flaky endpoint must not read as a falsified key.

        The two failures call for opposite responses: a disagreeing store must be
        deleted and re-measured, an unreachable server must be waited on. Merging
        them sends the next reader hunting a bug in the cache that is not there.
        """
        recorder = self._recorder()
        store = _fully_stratified_cache(tmp_path, endpoint=local_http_server.base_url)

        for _ in range(12):
            local_http_server.enqueue_disconnect("/api/generate")
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=True,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == recorder.CACHE_VERIFY_UNREACHABLE
        )
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=False,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == recorder.CACHE_VERIFY_UNREACHABLE
        )

    def test_one_unreachable_cell_cannot_be_hidden_by_three_real_comparisons(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """Negative control: partial reachability is an incomplete audit."""
        recorder = self._recorder()
        store = _fully_stratified_cache(tmp_path, endpoint=local_http_server.base_url)
        local_http_server.enqueue_json("/api/generate", REPLY_LOGPROBS)
        local_http_server.enqueue_json("/api/generate", REPLY)
        local_http_server.enqueue_disconnect("/api/generate")
        local_http_server.enqueue_json("/api/generate", REPLY)
        local_http_server.enqueue_json("/api/generate", REPLY_LOGPROBS)
        local_http_server.enqueue_json("/api/generate", REPLY)
        local_http_server.enqueue_json("/api/generate", REPLY_LOGPROBS)
        local_http_server.enqueue_json("/api/generate", REPLY)
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=False,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == recorder.CACHE_VERIFY_UNREACHABLE
        )

    def test_a_disagreeing_store_fails_in_both_readings(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """The one case with no ambiguity: the server answered, and differently."""
        recorder = self._recorder()
        store = _fully_stratified_cache(tmp_path, endpoint=local_http_server.base_url)
        local_http_server.enqueue_json("/api/generate", {"response": "something else"}, repeat=12)
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=True,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == 1
        )
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=False,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )
            == 1
        )

    def test_a_measurement_may_bootstrap_missing_strata_but_an_audit_may_not(
        self, tmp_path: Path, local_http_server, capsys
    ) -> None:
        """Bootstrap is two-phase: existing replies are checked, then all arms
        are required before any claim can be written.

        POSITIVE CONTROL: a matching populated stratum lets measurement start.
        NEGATIVE CONTROLS: a standalone audit still refuses partial coverage,
        and bootstrap still rejects a populated stratum that changed.
        """
        recorder = self._recorder()
        store = CallCache(path=tmp_path / "calls.jsonl")
        store.put(_request(endpoint=local_http_server.base_url), REPLY)

        with pytest.raises(SystemExit, match="contributed zero entries"):
            recorder._verify_cache(
                1,
                store=store,
                precondition=False,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )

        local_http_server.enqueue_json("/api/generate", REPLY)
        _enqueue_current_model_identities(local_http_server)
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=True,
                endpoint=local_http_server.base_url,
                allow_missing_strata=True,
                expected_identities=_registered_test_identities(),
            )
            == 0
        )
        assert "across 1 of 6 declared stratum(s)" in capsys.readouterr().out

        local_http_server.enqueue_json("/api/generate", {"response": "changed"})
        assert (
            recorder._verify_cache(
                1,
                store=store,
                precondition=True,
                endpoint=local_http_server.base_url,
                allow_missing_strata=True,
                expected_identities=_registered_test_identities(),
            )
            == 1
        )

    def test_live_identity_change_refuses_an_otherwise_matching_audit(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """Negative control: answer equality cannot hide a repointed model tag."""

        recorder = self._recorder()
        store = _fully_stratified_cache(tmp_path, endpoint=local_http_server.base_url)
        _enqueue_matching_strata(local_http_server)
        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "d" * 64})
        local_http_server.enqueue_json("/api/show", {"digest": QWEN_IDENTITY})
        local_http_server.enqueue_json("/api/show", {"digest": SMOLLM_IDENTITY})
        with pytest.raises(SystemExit, match="identity changed during replay"):
            recorder._verify_cache(
                1,
                store=store,
                precondition=False,
                endpoint=local_http_server.base_url,
                expected_identities=_registered_test_identities(),
            )

    def test_the_recorder_requires_all_strata_again_after_measurement(self) -> None:
        """Removing the strict post-check must fail before a long run can write."""
        recorder = self._recorder()
        body = inspect.getsource(recorder._measure_and_record)
        first = body.index("allow_missing_strata=True")
        measured = body.index("benchmark_model(")
        strict = body.rindex("_verify_cache(")
        ledger = body.index("ledger.replace_claims(")
        assert first < measured < strict < ledger

    def test_poisoned_sidecar_has_no_path_into_claim_results(self, tmp_path: Path) -> None:
        """A forged result file is inert even when the call-store audit is unrelated.

        NEGATIVE CONTROL: the planted v1-shaped sidecar used to be returned by
        ``_load_sidecar`` on settings equality alone. The reader no longer
        exists, and the sole assignment into one model's result is the direct
        benchmark call. POSITIVE CONTROL: a direct result can still be emitted
        as a bound, explicitly non-reusable receipt.
        """
        recorder = self._recorder()
        assert not hasattr(recorder, "_load_sidecar")
        body = inspect.getsource(recorder._measure_and_record)
        assert "results[name] = benchmark_model(" in body
        assert "reusing sidecar" not in body
        assert "read_text(" not in body

        sidecar = tmp_path / "gemma3.json"
        sidecar.write_text(
            json.dumps(
                {
                    "settings": {"protocol": recorder.PROTOCOL_VERSION},
                    "result": {"fabricated": True, "score": 100},
                }
            ),
            encoding="utf-8",
        )
        settings = {
            "protocol": recorder.PROTOCOL_VERSION,
            "endpoint": recorder.OLLAMA_URL.rstrip("/"),
            "letter_n": 1,
            "phoneme_n": 1,
            "g2p_n": 1,
            "calibration_n": 1,
            "ensemble_n": 1,
            "repeat_n": 1,
            "seed": recorder.DEFAULT_SEED,
        }
        result = {
            "model": "gemma3:4b",
            "seed": recorder.DEFAULT_SEED,
            "protocol_version": recorder.PROTOCOL_VERSION,
            "letter_pool_total": 1,
            "letter_pool_unique": 1,
            "letter_positions_scored": 1,
            "letter": {},
            "letter_alphabetic": {},
            "prompt_sensitivity": {},
            "letter_non_alphabetic": {},
            "phoneme": {},
            "g2p": {},
            "context": {},
            "calibration": {},
            "ensemble": {"n": 0},
            "repeat": {"n": 0},
        }
        source = {"files": [], "manifest_sha256": recorder._json_digest([])}
        cache = {
            "enabled": False,
            "measurement": "live",
            "keys": [],
            "keys_sha256": recorder._json_digest([]),
        }
        written = recorder._write_sidecar(
            sidecar,
            logical_name="gemma3",
            model_tag="gemma3:4b",
            model_identity="sha256:" + "c" * 64,
            settings=settings,
            source=source,
            cache=cache,
            result=result,
        )
        assert written["reuse_policy"] == "never"
        assert written["result"] == result
        assert recorder._read_sidecar_receipt_for_audit(sidecar) == written
        assert not list(tmp_path.glob(".gemma3.json.*.tmp"))

    def test_sidecar_receipt_tampering_and_wrong_identity_fail_closed(self, tmp_path: Path) -> None:
        recorder = self._recorder()
        store = CallCache(path=tmp_path / "calls.jsonl")
        run_scope = "20260905T000000000003Z"
        run_protocols = recorder._cache_protocols(run_scope)
        requests = [
            _request(
                endpoint=recorder.OLLAMA_URL,
                protocol=protocol,
                identity="sha256:" + "a" * 64,
            )
            for protocol in run_protocols
        ]
        for request in requests:
            store.put(request, REPLY)
        keys = {request.key for request in requests}
        request = requests[0]
        binding = recorder._cache_binding(
            store,
            keys,
            model_tag="gemma3:4b",
            model_identity=request.identity,
            cache_protocols=run_protocols,
        )
        assert binding["schema"] == CACHE_SCHEMA
        assert binding["keys"] == sorted(keys)
        assert binding["records_sha256"] == store.records_digest(keys)

        with pytest.raises(ValueError, match="model identity"):
            recorder._cache_binding(
                store,
                keys,
                model_tag="gemma3:4b",
                model_identity="sha256:different",
                cache_protocols=run_protocols,
            )

        other_endpoint = (
            "http://localhost:11434"
            if recorder.OLLAMA_URL != "http://localhost:11434"
            else "http://127.0.0.1:15434"
        )
        wrong_origin = CallCache(path=tmp_path / "wrong-origin.jsonl")
        wrong_keys = set()
        for protocol in run_protocols:
            wrong_request = _request(
                endpoint=other_endpoint, protocol=protocol, identity=request.identity
            )
            wrong_origin.put(wrong_request, REPLY)
            wrong_keys.add(wrong_request.key)
        with pytest.raises(ValueError, match="benchmark endpoint"):
            recorder._cache_binding(
                wrong_origin,
                wrong_keys,
                model_tag="gemma3:4b",
                model_identity=request.identity,
                cache_protocols=run_protocols,
            )

        # A complete receipt's result and envelope digests are both live gates.
        settings = {
            "protocol": recorder.PROTOCOL_VERSION,
            "endpoint": recorder.OLLAMA_URL.rstrip("/"),
            "seed": recorder.DEFAULT_SEED,
        }
        result = {
            "model": "gemma3:4b",
            "seed": recorder.DEFAULT_SEED,
            "protocol_version": recorder.PROTOCOL_VERSION,
            "letter_pool_total": 0,
            "letter_pool_unique": 0,
            "letter_positions_scored": 0,
            "letter": {},
            "letter_alphabetic": {},
            "prompt_sensitivity": {},
            "letter_non_alphabetic": {},
            "phoneme": {},
            "g2p": {},
            "context": {},
            "calibration": {},
            "ensemble": {},
            "repeat": {},
        }
        path = tmp_path / "receipt.json"
        receipt = recorder._write_sidecar(
            path,
            logical_name="gemma3",
            model_tag="gemma3:4b",
            model_identity=request.identity,
            settings=settings,
            source={"files": [], "manifest_sha256": recorder._json_digest([])},
            cache=binding,
            result=result,
        )
        receipt["result"]["letter_pool_total"] = 1
        path.write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(ValueError, match="result digest mismatch"):
            recorder._read_sidecar_receipt_for_audit(path)

        result["letter"]["score"] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            recorder._write_sidecar(
                path,
                logical_name="gemma3",
                model_tag="gemma3:4b",
                model_identity=request.identity,
                settings=settings,
                source={"files": [], "manifest_sha256": recorder._json_digest([])},
                cache=binding,
                result=result,
            )

    def test_distribution_comparison_cannot_pass_without_shared_evidence(self) -> None:
        """The distribution arm must compare values, not merely see two lists.

        POSITIVE CONTROL: matching answer-slot probabilities pass.
        NEGATIVE CONTROLS: a missing list, disjoint candidate sets, and drift over
        the declared tolerance all fail. Ordinary replies with no requested
        distribution remain comparable through their exact response string.
        """
        recorder = self._recorder()
        assert recorder._distributions_agree(REPLY, REPLY)
        assert recorder._distributions_agree(REPLY_LOGPROBS, REPLY_LOGPROBS)

        disjoint = {
            **REPLY_LOGPROBS,
            "logprobs": [{"top_logprobs": [{"token": "z", "logprob": -0.2}]}],
        }
        drifted = {
            **REPLY_LOGPROBS,
            "logprobs": [
                {
                    "top_logprobs": [
                        {
                            "token": "x",
                            "logprob": -0.2 - recorder.LOGPROB_TOLERANCE - 0.01,
                        }
                    ]
                }
            ],
        }
        assert not recorder._distributions_agree(REPLY_LOGPROBS, REPLY)
        assert not recorder._distributions_agree(REPLY_LOGPROBS, disjoint)
        assert not recorder._distributions_agree(REPLY_LOGPROBS, drifted)


class TestPositionAlignedReplay:
    @staticmethod
    def _captured_pair():
        path = Path(__file__).parent / "fixtures" / "ollama_position_replay.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_real_tail_membership_change_never_compares_different_positions(self) -> None:
        pair = self._captured_pair()
        recorder = TestVerifyConventions._recorder()
        stored, fresh = pair["stored"], pair["fresh"]
        assert recorder._comparable(stored) == recorder._comparable(fresh)
        assert len(stored["logprobs"]) > 1
        # This candidate remains earlier but drops from the final position.
        assert any(c["token"] == " <" for c in stored["logprobs"][-1]["top_logprobs"])
        assert all(c["token"] != " <" for c in fresh["logprobs"][-1]["top_logprobs"])
        assert recorder._distributions_agree(stored, fresh)
        assert recorder._distributions_agree(fresh, stored)

    @pytest.mark.parametrize(
        "mutation",
        [
            "none",
            "answer_drift",
            "envelope_drift",
            "text",
            "missing_row",
            "reordered_rows",
            "token",
            "missing_token",
            "empty_candidates",
            "both_empty_position",
            "disjoint",
            "nan",
            "infinity",
            "duplicate",
            "missing_logprob",
            "bad_candidates",
            "bad_row",
            "bad_rows",
            "both_absent",
            "both_missing_tokens",
            "both_empty",
            "hidden_answer_drift",
        ],
    )
    def test_actual_http_replay_accepts_only_aligned_complete_evidence(
        self, tmp_path: Path, local_http_server, mutation: str
    ) -> None:
        pair = self._captured_pair()
        stored, fresh = pair["stored"], pair["fresh"]
        rows = fresh["logprobs"]
        if mutation in {"answer_drift", "envelope_drift"}:
            index = 4 if mutation == "answer_drift" else 0
            candidate = rows[index]["top_logprobs"][0]
            original = next(
                c
                for c in stored["logprobs"][index]["top_logprobs"]
                if c["token"] == candidate["token"]
            )
            candidate["logprob"] = original["logprob"] - 1.0
        elif mutation == "hidden_answer_drift":
            fresh = json.loads(json.dumps(stored))
            candidate = next(c for c in fresh["logprobs"][4]["top_logprobs"] if c["token"] == " <")
            candidate["logprob"] -= 1.0
        elif mutation == "both_missing_tokens":
            for body in (stored, fresh):
                for entry in body["logprobs"]:
                    entry.pop("token")
        elif mutation in {"both_absent", "both_empty"}:
            for body in (stored, fresh):
                if mutation == "both_absent":
                    body.pop("logprobs")
                else:
                    body["logprobs"] = []
        elif mutation == "text":
            fresh["response"] = '{"answer":"b"}'
        elif mutation == "missing_row":
            rows.pop()
        elif mutation == "reordered_rows":
            rows[0], rows[1] = rows[1], rows[0]
        elif mutation == "token":
            rows[0]["token"] = "different"
        elif mutation == "missing_token":
            rows[0].pop("token")
        elif mutation == "empty_candidates":
            rows[0]["top_logprobs"] = []
        elif mutation == "both_empty_position":
            rows[0]["top_logprobs"] = []
            stored["logprobs"][0]["top_logprobs"] = []
        elif mutation == "disjoint":
            rows[4]["top_logprobs"] = [{"token": "unshared", "logprob": -1.0}]
        elif mutation in {"nan", "infinity"}:
            rows[0]["top_logprobs"][0]["logprob"] = float(mutation.replace("infinity", "inf"))
        elif mutation == "duplicate":
            rows[0]["top_logprobs"].append(dict(rows[0]["top_logprobs"][0]))
        elif mutation == "missing_logprob":
            rows[0]["top_logprobs"][0].pop("logprob")
        elif mutation == "bad_candidates":
            rows[0]["top_logprobs"] = "invalid"
        elif mutation == "bad_row":
            rows[0] = "invalid"
        elif mutation == "bad_rows":
            fresh["logprobs"] = "invalid"
        store = _cache(tmp_path)
        store.put(
            _request(
                pair["payload"], identity=SMOLLM_IDENTITY, endpoint=local_http_server.base_url
            ),
            stored,
        )
        before = store.path.read_bytes()
        local_http_server.enqueue_json("/api/generate", fresh)
        _enqueue_current_model_identities(local_http_server)
        recorder = TestVerifyConventions._recorder()
        result = recorder._verify_cache(
            1,
            store=store,
            precondition=True,
            allow_missing_strata=True,
            endpoint=local_http_server.base_url,
            expected_identities=_registered_test_identities(),
        )
        assert result == (
            recorder.CACHE_VERIFY_OK if mutation == "none" else recorder.CACHE_VERIFY_MISMATCH
        )
        assert store.path.read_bytes() == before
        assert len(local_http_server.requests_for("/api/generate")) == 1

    @pytest.mark.parametrize(
        "field,value",
        [
            ("logprob", 10**400),
            ("logprob", True),
            ("logprob", "-1"),
            ("token", None),
            ("token", 3),
        ],
    )
    def test_malformed_candidate_values_refuse_without_exceptions(self, field, value) -> None:
        pair = self._captured_pair()
        pair["fresh"]["logprobs"][0]["top_logprobs"][0][field] = value
        assert not TestVerifyConventions._recorder()._distributions_agree(
            pair["stored"], pair["fresh"], require_distributions=True
        )

    @pytest.mark.parametrize("text", ['{"answer":"invalid-symbol"}', "not JSON"])
    def test_identical_invalid_answers_remain_scoreable(self, text: str) -> None:
        recorder = TestVerifyConventions._recorder()
        body = {
            "response": text,
            "logprobs": [
                {
                    "token": text,
                    "top_logprobs": [
                        {"token": text, "logprob": -0.2},
                        {"token": "alternative", "logprob": -2.0},
                    ],
                }
            ],
        }
        assert recorder._comparable(body) == recorder._comparable(body)
        assert recorder._distributions_agree(body, body)
        assert recorder._distributions_agree({"response": text}, {"response": text})


class TestTheCachedPosterRoundTrip:
    """The seam end to end through a real server: identity, key, store, replay.

    `TestSeam` above proves every helper routes through `_post_generate`; these
    prove `_post_generate` itself does what the receipt claims, with the HTTP
    client as the only substituted piece.
    """

    @staticmethod
    def _restore_identity(module):
        snapshot = dict(module._MODEL_IDENTITY)
        module._MODEL_IDENTITY.clear()
        return snapshot

    def test_an_identical_question_is_measured_once_and_replayed_after(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """POSITIVE: one server call for two identical asks, and the digest is
        resolved once per process rather than once per position.

        NEGATIVE CONTROL: a different prompt reaches the server again, so this
        cannot pass on a store that replays everything.
        """
        from ailm.benchmarks import ollama as ollama_module

        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "d" * 64})
        local_http_server.enqueue_json("/api/generate", {"response": "measured"}, repeat=2)
        endpoint = local_http_server.base_url
        snapshot = self._restore_identity(ollama_module)
        try:
            store = benchmarks.enable_call_cache(tmp_path / "calls.jsonl", enabled=True)
            assert store is not None and benchmarks.call_cache() is store

            first = benchmarks.ask_ollama("gemma3:4b", "prefix", "letter", endpoint=endpoint)
            second = benchmarks.ask_ollama("gemma3:4b", "prefix", "letter", endpoint=endpoint)
            assert first == second == "measured"
            assert len(local_http_server.requests_for("/api/generate")) == 1, (
                "the identical question was measured twice"
            )
            assert len(local_http_server.requests_for("/api/show")) == 1, (
                "the digest was re-resolved for a known model"
            )
            assert (store.stats.hits, store.stats.misses, store.stats.stored) == (1, 1, 1)

            benchmarks.ask_ollama("gemma3:4b", "another prefix", "letter", endpoint=endpoint)
            assert len(local_http_server.requests_for("/api/generate")) == 2, (
                "a different question was answered from the store"
            )
        finally:
            benchmarks.enable_call_cache(enabled=False)
            ollama_module._MODEL_IDENTITY.clear()
            ollama_module._MODEL_IDENTITY.update(snapshot)

    def test_protocol_identity_separates_the_same_payload(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """The protocol is key material, not unrecorded caller context.

        The two first calls are real misses with different replies. Repeating
        each protocol must recover its own reply without a third server call.
        """
        from ailm.benchmarks import ollama as ollama_module

        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "e" * 64})
        local_http_server.enqueue_json("/api/generate", {"response": "protocol-a"})
        local_http_server.enqueue_json("/api/generate", {"response": "protocol-b"})
        endpoint = local_http_server.base_url
        snapshot = self._restore_identity(ollama_module)
        try:
            store = benchmarks.enable_call_cache(tmp_path / "calls.jsonl")
            assert store is not None
            first = benchmarks.ask_ollama(
                "gemma3:4b",
                "same prefix",
                "letter",
                endpoint=endpoint,
                cache_protocol="synthetic-zero-v2",
            )
            second = benchmarks.ask_ollama(
                "gemma3:4b",
                "same prefix",
                "letter",
                endpoint=endpoint,
                cache_protocol="synthetic-incontext-v2",
            )
            assert (first, second) == ("protocol-a", "protocol-b")
            assert (
                benchmarks.ask_ollama(
                    "gemma3:4b",
                    "same prefix",
                    "letter",
                    endpoint=endpoint,
                    cache_protocol="synthetic-zero-v2",
                )
                == "protocol-a"
            )
            assert (
                benchmarks.ask_ollama(
                    "gemma3:4b",
                    "same prefix",
                    "letter",
                    endpoint=endpoint,
                    cache_protocol="synthetic-incontext-v2",
                )
                == "protocol-b"
            )
            assert len(local_http_server.requests_for("/api/generate")) == 2
            assert len(store) == 2
            assert (store.stats.hits, store.stats.misses, store.stats.stored) == (2, 2, 2)
        finally:
            benchmarks.enable_call_cache(enabled=False)
            ollama_module._MODEL_IDENTITY.clear()
            ollama_module._MODEL_IDENTITY.update(snapshot)

    def test_cache_read_false_reaches_the_server_and_retains_duplicate_guard(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """Canonical measurement bypasses replay without permitting drift.

        Two semantically identical replies must make two HTTP requests and one
        durable record. A later ordinary call replays that record. This proves
        bypass by call count while preserving the duplicate-equivalence check.
        """
        from ailm.benchmarks import ollama as ollama_module

        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "f" * 64})
        local_http_server.enqueue_json(
            "/api/generate", {"response": "measured", "total_duration": 10}
        )
        local_http_server.enqueue_json(
            "/api/generate", {"response": "measured", "total_duration": 20}
        )
        endpoint = local_http_server.base_url
        snapshot = self._restore_identity(ollama_module)
        try:
            store = benchmarks.enable_call_cache(tmp_path / "calls.jsonl")
            assert store is not None
            for _ in range(2):
                assert (
                    benchmarks.ask_ollama(
                        "gemma3:4b",
                        "same prefix",
                        "letter",
                        endpoint=endpoint,
                        cache_protocol="publication-v2",
                        cache_read=False,
                    )
                    == "measured"
                )
            assert len(local_http_server.requests_for("/api/generate")) == 2
            assert len(store) == 1
            assert (store.stats.hits, store.stats.misses, store.stats.stored) == (0, 2, 1)

            assert (
                benchmarks.ask_ollama(
                    "gemma3:4b",
                    "same prefix",
                    "letter",
                    endpoint=endpoint,
                    cache_protocol="publication-v2",
                )
                == "measured"
            )
            assert len(local_http_server.requests_for("/api/generate")) == 2
            assert store.stats.hits == 1
        finally:
            benchmarks.enable_call_cache(enabled=False)
            ollama_module._MODEL_IDENTITY.clear()
            ollama_module._MODEL_IDENTITY.update(snapshot)

    def test_cache_read_false_rejects_a_divergent_live_duplicate(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """NEGATIVE CONTROL: bypass cannot turn one key into two answers."""
        from ailm.benchmarks import ollama as ollama_module

        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "a" * 64})
        local_http_server.enqueue_json("/api/generate", {"response": "first"})
        local_http_server.enqueue_json("/api/generate", {"response": "second"})
        endpoint = local_http_server.base_url
        snapshot = self._restore_identity(ollama_module)
        try:
            benchmarks.enable_call_cache(tmp_path / "calls.jsonl")
            common = {
                "endpoint": endpoint,
                "cache_protocol": "publication-v2",
                "cache_read": False,
            }
            assert benchmarks.ask_ollama("gemma3:4b", "prefix", "letter", **common) == "first"
            with pytest.raises(CacheCorruptionError, match="divergent reply"):
                benchmarks.ask_ollama("gemma3:4b", "prefix", "letter", **common)
            assert len(local_http_server.requests_for("/api/generate")) == 2
        finally:
            benchmarks.enable_call_cache(enabled=False)
            ollama_module._MODEL_IDENTITY.clear()
            ollama_module._MODEL_IDENTITY.update(snapshot)

    def test_an_unresolvable_identity_measures_every_time_and_stores_nothing(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """An unknown model identity degrades to no caching, not to caching
        under a guess — and the failed lookup is remembered, so a census does
        not retry `/api/show` for every one of its positions.
        """
        from ailm.benchmarks import ollama as ollama_module

        local_http_server.enqueue_disconnect("/api/show")
        local_http_server.enqueue_json("/api/generate", {"response": "measured"}, repeat=2)
        endpoint = local_http_server.base_url
        snapshot = self._restore_identity(ollama_module)
        try:
            store = benchmarks.enable_call_cache(tmp_path / "calls.jsonl", enabled=True)
            assert store is not None

            benchmarks.ask_ollama("nameless:1b", "prefix", "letter", endpoint=endpoint)
            benchmarks.ask_ollama("nameless:1b", "prefix", "letter", endpoint=endpoint)
            assert len(local_http_server.requests_for("/api/generate")) == 2, (
                "an unidentified call was replayed"
            )
            assert len(local_http_server.requests_for("/api/show")) == 1, (
                "the failed identity lookup was retried per call"
            )
            assert len(store) == 0, "a reply was stored under an unresolved identity"
            assert store.stats.unidentified == 2
            assert store.stats.hits == 0
        finally:
            benchmarks.enable_call_cache(enabled=False)
            ollama_module._MODEL_IDENTITY.clear()
            ollama_module._MODEL_IDENTITY.update(snapshot)


class TestTheDeterminismProbeCannotBeSatisfiedByTheCache:
    """The measurement that must reach the network, asserted mechanically.

    `repeat_agreement` asks the server the same question twice and reports how
    often it answers the same way. The two identical server payloads are stored
    under distinct cache-purpose protocols, so neither trial can satisfy the
    other while both remain available to semantic receipt replay.

    A measurement whose value is forced by the mechanism it measures is worse
    than no measurement: it reads as a passing check.
    """

    def test_two_identical_requests_reach_the_server_twice(
        self, tmp_path, local_http_server
    ) -> None:
        """The positive control for the whole probe: count the calls."""
        from ailm import benchmarks

        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "a" * 64})
        local_http_server.enqueue_json("/api/generate", {"response": "x"}, repeat=2)
        trials = repeat_trial_protocols("20260905T000000000004Z")
        store = benchmarks.enable_call_cache(tmp_path / "calls.jsonl", enabled=True)
        try:
            benchmarks.repeat_agreement(
                "m",
                [("the quick brown fo", "x")],
                repeat_trial_protocols=trials,
                endpoint=local_http_server.base_url,
            )
        finally:
            benchmarks.enable_call_cache(enabled=False)
        calls = len(local_http_server.requests_for("/api/generate"))
        assert calls == 2, (
            f"repeat_agreement made {calls} server call(s) for one position with the "
            f"store enabled. It must make two: if the second is answered from the store, "
            f"agreement is 1.0 by construction and the probe cannot fail."
        )
        assert store is not None
        assert len(store) == 2
        assert store.provenance_binding(store.accessed_keys)["protocols"] == sorted(trials)

    def test_the_probe_reports_disagreement_when_the_server_disagrees(
        self, tmp_path, local_http_server
    ) -> None:
        """The other half: a probe that always reaches the network is still
        worthless if it cannot report a mismatch. A cached second leg would make
        this impossible to observe at all."""
        from ailm import benchmarks

        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "a" * 64})
        local_http_server.enqueue_json("/api/generate", {"response": "first"})
        local_http_server.enqueue_json("/api/generate", {"response": "second"})
        benchmarks.enable_call_cache(tmp_path / "calls.jsonl", enabled=True)
        try:
            result = benchmarks.repeat_agreement(
                "m",
                [("prefix", "x")],
                repeat_trial_protocols=repeat_trial_protocols("20260905T000000000005Z"),
                endpoint=local_http_server.base_url,
            )
        finally:
            benchmarks.enable_call_cache(enabled=False)
        assert result["agreement"] == 0.0, (
            "the probe reported agreement on two different replies, so it is not "
            "comparing what it claims to compare"
        )

    def test_the_probe_forces_two_fresh_purpose_bound_writes(self) -> None:
        """The structural control prevents a future cache-read regression."""
        body = inspect.getsource(benchmarks.repeat_agreement)
        assert "_post_generate(" in body
        assert "cache_read=False" in body
        assert "trial_protocols" in body
        # No default: a caller that does not name its run cannot make the call
        # at all, which is what keeps two runs off one key.
        signature = inspect.signature(benchmarks.repeat_agreement)
        parameter = signature.parameters["repeat_trial_protocols"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


_REPEAT_RUN_PROGRAM = """
import json
import sys
from pathlib import Path

from ailm import benchmarks
from ailm.benchmarks.constants import repeat_trial_protocols

store_path, endpoint, scope = sys.argv[1:4]
benchmarks.enable_call_cache(Path(store_path), enabled=True)
result = benchmarks.repeat_agreement(
    "m",
    [("the quick brown fo", "x")],
    repeat_trial_protocols=repeat_trial_protocols(scope),
    endpoint=endpoint,
)
print(json.dumps(result))
"""


def _canonical_repeat_run(
    store_path: Path, endpoint: str, scope: str
) -> subprocess.CompletedProcess[str]:
    """One whole run of the reliability probe, in its own process."""

    return subprocess.run(
        [sys.executable, "-c", _REPEAT_RUN_PROGRAM, str(store_path), endpoint, scope],
        capture_output=True,
        text=True,
        timeout=480,
        check=False,
    )


class TestASecondRunAppendsInsteadOfCollidingWithTheFirst:
    """The defect this scoping exists to remove, exercised end to end.

    Both repeat-trial calls are deliberately live, so a second canonical run
    re-asks the server the same two questions. Before the tags carried a run,
    the second run's replies landed on the first run's keys; when the server had
    since reloaded the model or served another workload it answered differently,
    and the store refused the divergent record with the recorder mid-flight.
    That made every publication receipt unre-bindable after any source edit.

    The store's refusal is not the defect and is not relaxed here. What changed
    is that a second run no longer asks it to overwrite anything.
    """

    @pytest.mark.timeout(960)
    def test_a_divergent_second_run_appends_under_its_own_tags(
        self, tmp_path: Path, local_http_server
    ) -> None:
        first_scope = "20260905T000000000020Z"
        second_scope = "20260905T000000000021Z"
        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "e" * 64}, repeat=2)
        local_http_server.enqueue_json("/api/generate", {"response": "first"}, repeat=2)
        # The server's answer changes between runs, which is exactly the case the
        # old shared key turned into a hard stop.
        local_http_server.enqueue_json("/api/generate", {"response": "second"}, repeat=2)
        store_path = tmp_path / "repeat-runs.jsonl"

        first = _canonical_repeat_run(store_path, local_http_server.base_url, first_scope)
        assert first.returncode == 0, first.stderr
        second = _canonical_repeat_run(store_path, local_http_server.base_url, second_scope)
        assert second.returncode == 0, (
            "a second canonical run was refused by the store; run scoping is not in "
            f"effect. stderr:\n{second.stderr}"
        )

        # Both runs measured agreement on their own two replies, not on each
        # other's: each reports 1.0 because each saw one answer twice.
        assert json.loads(first.stdout)["agreement"] == 1.0
        assert json.loads(second.stdout)["agreement"] == 1.0

        store = CallCache(store_path)
        assert len(store) == 4, "the second run did not append its own two records"
        keys = {key for key, _payload, _reply in store.entries()}
        assert store.provenance_binding(keys)["protocols"] == sorted(
            (*repeat_trial_protocols(first_scope), *repeat_trial_protocols(second_scope))
        )
        replies = sorted(str(reply["response"]) for _key, _payload, reply in store.entries())
        assert replies == ["first", "first", "second", "second"]

    @pytest.mark.timeout(960)
    def test_the_store_still_refuses_a_divergent_reply_on_one_runs_own_key(
        self, tmp_path: Path, local_http_server
    ) -> None:
        """The guard the fix must not have weakened, asserted on the same path."""

        scope = "20260905T000000000022Z"
        local_http_server.enqueue_json("/api/show", {"digest": "sha256:" + "e" * 64})
        local_http_server.enqueue_json("/api/generate", {"response": "first"}, repeat=2)
        store_path = tmp_path / "same-run.jsonl"
        run = _canonical_repeat_run(store_path, local_http_server.base_url, scope)
        assert run.returncode == 0, run.stderr

        store = CallCache(store_path)
        key, payload, _reply = next(iter(store.entries()))
        binding = store.provenance_binding({key})
        request = CacheRequest(
            payload=payload,
            endpoint=binding["endpoints"][0],
            protocol=binding["protocols"][0],
            identity=binding["identities"][0],
        )
        with pytest.raises(CacheCorruptionError, match="divergent reply for existing key"):
            store.put(request, {"response": "something else"})


class TestServerCacheAccountingIsNotAnAnswerDifference:
    """A repeated question must be answerable twice without being refused.

    Ollama reports ``prompt_eval_cached_count``: how many prompt tokens it
    served from its own KV cache rather than re-evaluating. It is zero the first
    time a prompt is seen and positive while the server still holds that prefix,
    so two identical requests moments apart differ in it as a matter of course.
    Two arms of this project ask the same question twice on purpose — the
    stipulated probes repeat short prompts across positions, and the reliability
    trials repeat one prompt deliberately — so treating that count as part of
    the answer refused the second observation of a question the protocol meant
    to ask twice.
    """

    def test_two_records_differing_only_in_server_cache_accounting_are_one(
        self, tmp_path: Path
    ) -> None:
        store = CallCache(tmp_path / "accounting.jsonl")
        request = _request(protocol="server-cache-accounting", identity="sha256:" + "a" * 64)
        cold = {"response": "x", "prompt_eval_count": 64, "prompt_eval_cached_count": 0}
        warm = {"response": "x", "prompt_eval_count": 64, "prompt_eval_cached_count": 64}
        store.put(request, cold)
        store.put(request, warm)
        assert len(store) == 1, "one question asked twice became two records"
        # The first full observation is retained, volatile fields included.
        _key, _payload, reply = next(iter(store.entries()))
        assert reply == cold

    def test_a_different_answer_under_the_same_accounting_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        """POSITIVE CONTROL: the guard the allowlist must not have widened away."""

        store = CallCache(tmp_path / "answers.jsonl")
        request = _request(protocol="server-cache-accounting", identity="sha256:" + "a" * 64)
        store.put(request, {"response": "x", "prompt_eval_cached_count": 0})
        with pytest.raises(CacheCorruptionError, match="divergent reply for existing key"):
            store.put(request, {"response": "y", "prompt_eval_cached_count": 0})

    def test_the_allowlist_is_enumerated_rather_than_matched_by_suffix(self) -> None:
        """A new server field stays semantic until someone has reviewed it.

        The value of this allowlist is that adding to it is a decision. A suffix
        rule would admit every future ``*_count`` or ``*_duration`` field without
        anyone asking whether it can vary with the answer.
        """

        from ailm import call_cache as module

        body = inspect.getsource(module)
        declaration = body[body.index("_VOLATILE_OLLAMA_REPLY_FIELDS = frozenset(") :]
        declaration = declaration[: declaration.index(")")]
        assert "endswith" not in declaration and "startswith" not in declaration
        assert "eval_count" not in module._VOLATILE_OLLAMA_REPLY_FIELDS, (
            "the token counts the measurement reads must stay semantic"
        )
        assert "response" not in module._VOLATILE_OLLAMA_REPLY_FIELDS
        assert "logprobs" not in module._VOLATILE_OLLAMA_REPLY_FIELDS
        assert "context" not in module._VOLATILE_OLLAMA_REPLY_FIELDS


def test_other_endpoint_rows_are_cold_and_cannot_satisfy_strict_audit(
    tmp_path, local_http_server, capsys
):
    recorder = TestVerifyConventions._recorder()
    store = _fully_stratified_cache(tmp_path)
    before = store.path.read_bytes()
    assert (
        recorder._verify_cache(
            6,
            store=store,
            endpoint=local_http_server.base_url,
            precondition=True,
            allow_missing_strata=True,
            expected_identities=_registered_test_identities(),
        )
        == recorder.CACHE_VERIFY_OK
    )
    assert "not an audit pass" in capsys.readouterr().out
    for precondition in (False, True):
        with pytest.raises(SystemExit, match="current loader configuration has no records"):
            recorder._verify_cache(
                6,
                store=store,
                endpoint=local_http_server.base_url,
                precondition=precondition,
                expected_identities=_registered_test_identities(),
            )
    assert not local_http_server.requests
    assert store.path.read_bytes() == before


def test_old_endpoint_growth_cannot_change_current_audit_population_or_sample(
    tmp_path, local_http_server
):
    from ailm.benchmarks.ollama import current_ollama_cache_entries

    recorder = TestVerifyConventions._recorder()
    store = _fully_stratified_cache(tmp_path, endpoint=local_http_server.base_url)
    entries = current_ollama_cache_entries(store, endpoint=local_http_server.base_url)
    expected_keys = {key for key, _payload, _body in entries}
    expected_digest = store.records_digest(expected_keys)
    _enqueue_matching_strata(local_http_server)
    _enqueue_current_model_identities(local_http_server)
    assert (
        recorder._verify_cache(
            6,
            store=store,
            endpoint=local_http_server.base_url,
            expected_identities=_registered_test_identities(),
        )
        == recorder.CACHE_VERIFY_OK
    )
    first_requests = [request.body for request in local_http_server.requests_for("/api/generate")]
    for _key, payload, reply in entries:
        store.put(
            _request(
                payload,
                endpoint="http://localhost:11434",
                identity=_registered_test_identities()[payload["model"]],
            ),
            reply,
        )
    # Stable real files must not be reparsed once per candidate during mixed-origin filtering.
    store.entries()
    before = store.reconciliation_stats.copy()
    selected = current_ollama_cache_entries(store, endpoint=local_http_server.base_url)
    assert {entry[0] for entry in selected} == expected_keys
    assert store.records_digest(expected_keys) == expected_digest
    assert store.reconciliation_stats["full"] == before["full"]
    assert store.reconciliation_stats["bytes"] == before["bytes"]
    _enqueue_matching_strata(local_http_server)
    _enqueue_current_model_identities(local_http_server)
    assert (
        recorder._verify_cache(
            6,
            store=store,
            endpoint=local_http_server.base_url,
            expected_identities=_registered_test_identities(),
        )
        == recorder.CACHE_VERIFY_OK
    )
    all_requests = [request.body for request in local_http_server.requests_for("/api/generate")]
    assert all_requests[len(first_requests) :] == first_requests
    assert len(store.entries()) == 2 * len(entries)
