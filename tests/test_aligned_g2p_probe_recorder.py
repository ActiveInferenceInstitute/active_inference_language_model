"""Contracts for the P2 zero-shot probe-arm recorder (protocol §3.4).

Every test runs real code against real artifacts and real recorded responses
— no mocks and no live server. The recorded responses below were captured
from the three registered probes (one uncached ``/api/generate`` call each,
same payload shape as a scored call, bypassing the call store) and are
replayed through the recorder's shipped parser/scorer exactly as the scored
measurement replays them. The network transport itself is exercised only by
the real recorder run.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path

import pytest

from ailm.benchmarks import aligned_g2p as ag
from ailm.benchmarks.synthetic_protocol import (
    SYNTHETIC_ALIGNED_G2P_PROTOCOL,
    SYNTHETIC_INCONTEXT_PROTOCOL,
    SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL,
    SYNTHETIC_ZERO_SHOT_PROTOCOL,
)
from ailm.claims import ClaimBatch, load_ledger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _recorder():
    return importlib.import_module("scripts.record_aligned_g2p_probe_claims")


# ── The recorded-response fixture ────────────────────────────────────────────
#
# One real response per registered probe, recorded 2026-09-15 against the
# artifact's first test spelling ``depete`` (gold UW B K B B B) with the exact
# scored payload shape (temperature 0, DEFAULT_SEED, num_predict 32, num_ctx
# 8192, the g2p response schema), issued uncached so no store entry was
# written. The three replies happen to be contract-compliant and wrong —
# which is exactly the descriptive zero-shot regime the arm reports.


def _frozen_artifact():
    payload = ag.load_aligned_g2p()
    return ag.artifact_pairs(payload)


@pytest.fixture(scope="module")
def artifact():
    return _frozen_artifact()


GEMMA3_RECORDED_RESPONSE = '{"answer":["D","IH","EH","T","IH","EH"]}'
QWEN25_RECORDED_RESPONSE = '{"answer":["D","AY","P","EY","T","EY"]}'
SMOLLM_RECORDED_RESPONSE = '{ "answer":["D","EH","P","T","EH"]}'


def test_recorded_fixture_replies_parse_under_the_shipped_parser():
    """The captured bytes replay through parse_g2p_answer as recorded."""

    from ailm.benchmarks import parse_g2p_answer

    parsed, bad = parse_g2p_answer(GEMMA3_RECORDED_RESPONSE)
    assert parsed == ["D", "IH", "EH", "T", "IH", "EH"] and not bad
    parsed, bad = parse_g2p_answer(QWEN25_RECORDED_RESPONSE)
    assert parsed == ["D", "AY", "P", "EY", "T", "EY"] and not bad
    # The base model's leading whitespace inside the JSON envelope is the
    # insignificant formatting the parser tolerates; the answer is 5 phones.
    parsed, bad = parse_g2p_answer(SMOLLM_RECORDED_RESPONSE)
    assert parsed == ["D", "EH", "P", "T", "EH"] and not bad


def test_replay_scoring_matches_hand_arithmetic():
    """The recorded replies plus edge replies score exactly as computed."""

    recorder = _recorder()
    _correspondence, _train, test = _frozen_artifact()
    words = [word for word, _gold in test[:5]]
    golds = [gold for _word, gold in test[:5]]
    # Replies 1-3 are the recorded fixture; reply 4 is the exact gold of its
    # word (the constructive positive control); reply 5 is a non-JSON reply.
    replies = [
        GEMMA3_RECORDED_RESPONSE,
        QWEN25_RECORDED_RESPONSE,
        SMOLLM_RECORDED_RESPONSE,
        json.dumps({"answer": golds[3]}),
        "not json at all",
    ]
    scored = recorder.score_probe_replies(words, golds, replies)
    assert scored["words_scored"] == 5
    # Only the constructive control is exact.
    assert scored["exact_accuracy"] == pytest.approx(1 / 5)
    # All three recorded replies miss every position of their word; the gold
    # reply contributes 6/6; the non-JSON reply contributes 0/6.
    assert scored["position_accuracy"] == pytest.approx(6 / 30)
    # Compliance is parser acceptance, independent of correctness: the four
    # JSON replies comply, the free-text reply does not.
    assert scored["contract_compliance"] == pytest.approx(4 / 5)


def test_replay_scoring_treats_short_answers_positionally():
    """A compliant 4-token answer earns 4 positions and is never exact."""

    recorder = _recorder()
    _correspondence, _train, test = _frozen_artifact()
    _word, gold = test[0]
    scored = recorder.score_probe_replies(
        [test[0][0]],
        [gold],
        [json.dumps({"answer": gold[:4]})],
    )
    assert scored["exact_accuracy"] == 0.0
    assert scored["position_accuracy"] == pytest.approx(4 / 6)
    assert scored["contract_compliance"] == 1.0


# ── The pre-registered gates ─────────────────────────────────────────────────


def test_the_contract_floor_is_the_inventory_uniform_error_rate():
    recorder = _recorder()
    floor = recorder.contract_compliance_floor(
        len(SYNTHETIC_ALIGNED_G2P_PROTOCOL.response_inventory)
    )
    assert floor == pytest.approx(1 - 1 / 39)
    # The derivation is inventory arithmetic only: no word count, no
    # population factor, nothing carried over from another arm.
    assert recorder.contract_compliance_floor(39) == pytest.approx(floor)
    with pytest.raises(ValueError):
        recorder.contract_compliance_floor(0)


def test_contract_floor_gate_refuses_below_and_admits_at_the_floor():
    recorder = _recorder()
    floor = recorder.contract_compliance_floor(39)
    with pytest.raises(SystemExit, match="response-contract floor"):
        recorder.assert_contract_floor(0.9, floor, "gemma3", 60)
    # Strictly-below refusal: a run at the floor is still a measurement.
    recorder.assert_contract_floor(floor, floor, "gemma3", 60)
    recorder.assert_contract_floor(1.0, floor, "gemma3", 60)


def test_gold_isolation_holds_on_the_frozen_artifact_payload_set(artifact):
    recorder = _recorder()
    _correspondence, _train, test = artifact
    words = [word for word, _gold in test]
    payloads = ag.probe_request_payloads(words)
    # POSITIVE CONTROL: the real payload set passes its own scan.
    recorder.assert_gold_isolation(payloads, test)


@pytest.mark.parametrize("victim,leaker", [(0, 0), (0, 1)])
def test_gold_isolation_gate_refuses_a_planted_gold_string(artifact, victim, leaker):
    """Both self-leaks and cross-payload leaks refuse the run."""

    recorder = _recorder()
    _correspondence, _train, test = artifact
    words = [word for word, _gold in test]
    payloads = ag.probe_request_payloads(words)
    planted = copy.deepcopy(payloads)
    gold = " ".join(test[victim][1])
    planted[leaker]["prompt"] = planted[leaker]["prompt"] + " " + gold
    with pytest.raises(SystemExit, match="L2 gold isolation"):
        recorder.assert_gold_isolation(planted, test)


def test_split_disjointness_holds_on_the_frozen_artifact(artifact):
    recorder = _recorder()
    _correspondence, train, test = artifact
    recorder.assert_split_disjointness(train, test)


def test_split_disjointness_gate_refuses_planted_corruptions(artifact):
    recorder = _recorder()
    _correspondence, train, test = artifact
    collided = (copy.deepcopy(train), copy.deepcopy(test))
    collided[0].append(collided[1][0])
    with pytest.raises(SystemExit, match="collision"):
        recorder.assert_split_disjointness(*collided)
    duplicated = (copy.deepcopy(train), copy.deepcopy(test))
    duplicated[1].append(duplicated[1][0])
    with pytest.raises(SystemExit, match="duplicate"):
        recorder.assert_split_disjointness(*duplicated)
    misaligned = (copy.deepcopy(train), copy.deepcopy(test))
    word, gold = misaligned[1][0]
    misaligned[1][0] = (word, gold[:-1])
    with pytest.raises(SystemExit, match="1:1 aligned"):
        recorder.assert_split_disjointness(*misaligned)


def _seeded_ledger(tmp_path: Path, digest: str | None):
    ledger = load_ledger(tmp_path)
    if digest is not None:
        ledger.replace_claims(
            batches=[
                ClaimBatch(
                    values={
                        "bench_synthg2p_comparator_artifact_sha256": digest,
                    },
                    run_id="aligned-g2p-comparator-v1",
                    source="scripts/record_aligned_g2p_comparator_claims.py",
                )
            ]
        )
    return ledger


def test_artifact_binding_gate_admits_the_recorded_digest(tmp_path, artifact):
    import hashlib

    recorder = _recorder()
    artifact_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "vendor" / ag.ALIGNED_G2P_ARTIFACT_NAME).read_bytes()
    ).hexdigest()
    recorder.assert_artifact_binding(_seeded_ledger(tmp_path, artifact_sha256), artifact_sha256)


def test_artifact_binding_gate_refuses_missing_and_mismatched_digests(tmp_path):
    recorder = _recorder()
    with pytest.raises(SystemExit, match="has not been"):
        recorder.assert_artifact_binding(_seeded_ledger(tmp_path, None), "deadbeef")
    with pytest.raises(SystemExit, match="digest mismatch"):
        recorder.assert_artifact_binding(_seeded_ledger(tmp_path, "deadbeef"), "cafebabe")


# ── The claim namespace ──────────────────────────────────────────────────────


def _identities():
    # Stand-in identity strings: the digest value is recorded verbatim per
    # probe; its provenance is the transport's resolved_model_identity, which
    # only the live run exercises.
    return {
        "gemma3": "identity-gemma3",
        "qwen25": "identity-qwen25",
        "smollm": "identity-smollm",
    }


def _scored_fixture(recorder):
    _correspondence, _train, test = _frozen_artifact()
    words = [word for word, _gold in test[:5]]
    golds = [gold for _word, gold in test[:5]]
    replies = [
        GEMMA3_RECORDED_RESPONSE,
        QWEN25_RECORDED_RESPONSE,
        SMOLLM_RECORDED_RESPONSE,
        json.dumps({"answer": golds[3]}),
        "not json at all",
    ]
    return {
        key: recorder.score_probe_replies(words, golds, replies)
        for key in ("gemma3", "qwen25", "smollm")
    }


def test_claim_namespace_is_complete_and_pre_registered(tmp_path):
    import hashlib

    recorder = _recorder()
    artifact_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "vendor" / ag.ALIGNED_G2P_ARTIFACT_NAME).read_bytes()
    ).hexdigest()
    values = recorder.claim_values(
        _scored_fixture(recorder),
        _identities(),
        artifact_sha256=artifact_sha256,
        floor=recorder.contract_compliance_floor(39),
        test_word_count=5,
        test_position_count=30,
    )
    assert set(values) <= recorder.all_claim_labels()
    assert not any("contract_refused" in label for label in values)
    assert all(label.startswith("bench_synthg2p_probe_") for label in values)
    assert values["bench_synthg2p_probe_artifact_sha256"] == artifact_sha256
    assert values["bench_synthg2p_probe_contract_floor"] == pytest.approx(1 - 1 / 39)
    for key in ("gemma3", "qwen25", "smollm"):
        assert values[f"bench_synthg2p_probe_{key}_words_scored"] == 5
        assert values[f"bench_synthg2p_probe_{key}_exact_accuracy"] == pytest.approx(0.2)
        assert values[f"bench_synthg2p_probe_{key}_position_accuracy"] == pytest.approx(0.2)
        assert values[f"bench_synthg2p_probe_{key}_contract_compliance"] == pytest.approx(0.8)
        assert values[f"bench_synthg2p_probe_{key}_model_digest"] == _identities()[key]


def test_happy_path_records_exactly_the_recorder_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    recorder = _recorder()
    artifact_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "vendor" / ag.ALIGNED_G2P_ARTIFACT_NAME).read_bytes()
    ).hexdigest()
    values = recorder.claim_values(
        _scored_fixture(recorder),
        _identities(),
        artifact_sha256=artifact_sha256,
        floor=recorder.contract_compliance_floor(39),
        test_word_count=5,
        test_position_count=30,
    )
    monkeypatch.setenv("AILM_CLAIMS_PATH", str(tmp_path / "claims.json"))
    recorder.record(values)
    ledger = load_ledger(PROJECT_ROOT)
    owned = ledger.labels_owned_by(recorder.SOURCE)
    assert set(owned) == set(values)
    for label in values:
        entry = ledger.get(label)
        assert entry["run_id"] == recorder.RUN_ID
        assert entry["source"] == recorder.SOURCE
        assert entry["value"] == values[label]
    # A second run with a shrunken namespace retires its stale labels.
    shrunken = {label: value for label, value in values.items() if "qwen25" not in label}
    recorder.record(shrunken)
    ledger = load_ledger(PROJECT_ROOT)
    assert set(ledger.labels_owned_by(recorder.SOURCE)) == set(shrunken)


# ── The frozen probe-arm execution contract ──────────────────────────────────


def test_the_probe_contract_is_a_distinct_registry_member():
    protocol = SYNTHETIC_ALIGNED_G2P_PROTOCOL
    assert protocol.probe_run_id == "synthetic_aligned_g2p_probe_v1"
    assert protocol.probe_source == "scripts/record_aligned_g2p_probe_claims.py"
    assert protocol.probe_response_tokens == 32
    assert protocol.probe_context_window > protocol.probe_response_tokens
    assert protocol.probe_warmup_policy == "one-uncached-warmup-per-probe-v1"
    assert protocol.probe_cache_protocol.endswith(":probe-arms-v1")
    assert protocol.probe_cache_protocol not in {
        SYNTHETIC_ZERO_SHOT_PROTOCOL.cache_protocol,
        SYNTHETIC_INCONTEXT_PROTOCOL.cache_protocol,
        SYNTHETIC_PHONE_ZERO_SHOT_PROTOCOL.cache_protocol,
        protocol.cache_protocol,
    }


def test_publication_run_requires_the_canonical_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint gate fires before anything else, including the network."""

    recorder = _recorder()
    monkeypatch.delenv("AILM_CLAIMS_PATH", raising=False)
    args = argparse.Namespace(endpoint="http://127.0.0.1:9999", vendor_dir=None)
    with pytest.raises(SystemExit, match="canonical endpoint"):
        recorder.measure(args)


# ── The publication binding surface ─────────────────────────────────────────


def _replay_identity(tag: str) -> str:
    """A well-formed stand-in identity for one model tag."""

    import hashlib

    return f"sha256:{hashlib.sha256(tag.encode('utf-8')).hexdigest()}"


def _replay_reply(index: int, gold: list[str]) -> str:
    """Cycle the recorded fixture with one constructive exact gold reply."""

    cycle = (
        GEMMA3_RECORDED_RESPONSE,
        QWEN25_RECORDED_RESPONSE,
        SMOLLM_RECORDED_RESPONSE,
        json.dumps({"answer": gold}),
    )
    return cycle[index % len(cycle)]


def _replay_fixture(tmp_path: Path, recorder):
    """A real call-store fixture: the exact scored request set, replay replies."""

    from ailm.benchmarks import OLLAMA_URL
    from ailm.benchmarks.aligned_g2p import probe_request_payloads
    from ailm.benchmarks.constants import DEFAULT_SEED
    from ailm.benchmarks.ollama import _generate_payload
    from ailm.call_cache import CacheRequest, CallCache
    from ailm.replay_provenance import generate_run_scope

    store = CallCache(tmp_path / "store.jsonl")
    scope = generate_run_scope()
    protocol = recorder._run_cache_protocol(scope)
    _correspondence, _train, test = _frozen_artifact()
    words = [word for word, _gold in test]
    golds = [gold for _word, gold in test]
    payloads = probe_request_payloads(words, response_tokens=recorder.RESPONSE_TOKENS)
    identities = {tag: _replay_identity(tag) for tag in recorder.PROBE_MODEL_TAGS.values()}
    keys: set[str] = set()
    for tag in sorted(recorder.PROBE_MODEL_TAGS.values()):
        for index, row in enumerate(payloads):
            request_payload = _generate_payload(
                tag,
                row["prompt"],
                "g2p",
                recorder.RESPONSE_TOKENS,
                DEFAULT_SEED,
                recorder.NUM_CTX,
            )
            request = CacheRequest(
                payload=request_payload,
                endpoint=OLLAMA_URL,
                protocol=protocol,
                identity=identities[tag],
            )
            store.put(request, {"response": _replay_reply(index, golds[index])})
            keys.add(request.key)
    scored = {
        key: recorder.score_probe_replies(
            words, golds, [_replay_reply(index, golds[index]) for index in range(len(words))]
        )
        for key in recorder.PROBE_MODEL_TAGS
    }
    values = recorder.claim_values(
        scored,
        {key: identities[tag] for key, tag in recorder.PROBE_MODEL_TAGS.items()},
        artifact_sha256=recorder._artifact_digest(),
        floor=recorder.contract_compliance_floor(
            len(SYNTHETIC_ALIGNED_G2P_PROTOCOL.response_inventory)
        ),
        test_word_count=len(words),
        test_position_count=sum(len(gold) for gold in golds),
    )
    measurement = recorder._measurement_evidence(
        scored,
        {key: identities[tag] for key, tag in recorder.PROBE_MODEL_TAGS.items()},
        cache_hits=0,
        cache_misses=len(keys),
        unique_cache_keys=len(keys),
        test_word_count=len(words),
        test_position_count=sum(len(gold) for gold in golds),
    )
    receipt = {
        "cache": {
            "keys": sorted(keys),
            "record_count": len(keys),
            "model_identities": [
                {"model": tag, "identity": identities[tag]} for tag in sorted(identities)
            ],
        },
        "configuration": {
            "endpoint": OLLAMA_URL,
            "call_protocol": protocol,
            "parameters": {
                "run_scope": scope,
                "cache_protocols": [protocol],
            },
        },
        "measurement": measurement,
        "claims": {"entries": {label: {"value": value} for label, value in values.items()}},
    }
    return store, receipt, values


def test_publication_replay_configuration_is_valid():
    from ailm.benchmarks import OLLAMA_URL
    from ailm.replay_provenance import generate_run_scope

    recorder = _recorder()
    configuration = recorder.publication_replay_configuration(run_scope=generate_run_scope())
    assert configuration["endpoint"] == OLLAMA_URL
    assert configuration["model_tags"] == sorted(recorder.PROBE_MODEL_TAGS.values())
    assert configuration["parameters"]["artifact"] == (f"vendor/{ag.ALIGNED_G2P_ARTIFACT_NAME}")
    protocol = configuration["call_protocol"]
    assert protocol in configuration["parameters"]["cache_protocols"]
    assert protocol.endswith("/" + configuration["parameters"]["run_scope"])


def test_semantic_replay_rederives_the_receipt_claims(tmp_path: Path) -> None:
    """The full 60-word scored request set replays back to the same claims."""

    recorder = _recorder()
    store, receipt, values = _replay_fixture(tmp_path, recorder)
    derived = recorder.validate_publication_measurement(receipt, store)
    assert derived == values


def test_semantic_replay_refuses_a_tampered_claim(tmp_path: Path) -> None:
    import copy as _copy

    from ailm.replay_provenance import NetworkClaimBindingError

    recorder = _recorder()
    store, receipt, _values = _replay_fixture(tmp_path, recorder)
    tampered = _copy.deepcopy(receipt)
    label = "bench_synthg2p_probe_gemma3_exact_accuracy"
    tampered["claims"]["entries"][label]["value"] += 0.25
    with pytest.raises(NetworkClaimBindingError, match="do not rederive"):
        recorder.validate_publication_measurement(tampered, store)


def test_semantic_replay_refuses_a_tampered_measurement(tmp_path: Path) -> None:
    import copy as _copy

    from ailm.replay_provenance import NetworkClaimBindingError

    recorder = _recorder()
    store, receipt, _values = _replay_fixture(tmp_path, recorder)
    tampered = _copy.deepcopy(receipt)
    tampered["measurement"]["arms"]["smollm"]["position_accuracy"] = 0.5
    with pytest.raises(NetworkClaimBindingError, match="disagrees"):
        recorder.validate_publication_measurement(tampered, store)
