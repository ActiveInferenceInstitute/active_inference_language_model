#!/usr/bin/env python3
"""Record the P2 zero-shot probe arms' claims (protocol §3.4, probe-arm bullet).

What is measured
----------------
For every registered served probe (``ailm.benchmarks.constants.SERVED_PROBES``:
gemma3, qwen25, smollm), this recorder scores ALL test-spelling positions of
the frozen aligned spelling-to-pronunciation artifact
(``vendor/synthetic_aligned_g2p_counts.json``) zero-shot:

- the request payloads are exactly ``probe_request_payloads`` — the L1-pure
  frozen prompt builder applied to each test word, under the strict JSON
  response schema (an array of canonical ARPAbet phones) with temperature 0,
  the shared fixed seed, and the declared response-token budget;
- each probe gets one deliberately uncached warm-up generation on a prompt
  outside the scored set (the first train word — L3 disjointness guarantees
  that), then every scored call goes through the shipped content-addressed
  call store, keyed by payload + endpoint + cache protocol + the model's
  resolved immutable identity;
- replies are parsed with the shipped strict JSON array parser
  (``parse_g2p_answer``) and scored as exact-word matches and per-position
  accuracy against the scorer-only gold, plus the response-contract
  compliance rate.

The demonstration sweep and its A4 control are out of scope here: zero-shot
only, per the recorder grant.

Pre-registered refusal gates (SystemExit before any claim write)
----------------------------------------------------------------
1. **Artifact binding.** The vendored artifact's sha256 must equal the
   ``bench_synthg2p_comparator_artifact_sha256`` the offline comparator
   recorder recorded: the probe arms score exactly the artifact the
   comparator arm was recorded on (§3.4: comparator arm recorded offline
   first). A missing label means the comparator run has not happened; a
   mismatch means the artifact moved in between.
2. **L3 disjointness.** The recorder re-asserts the §3.1 set assertions over
   the loaded splits — no duplicate word within a split, no cross-split
   collision, and every word 1:1 aligned at the declared length — before any
   call.
3. **L2 gold isolation.** Before any call, every request payload of the
   scored set is serialized and scanned for every test gold string; any hit
   is a leak and refuses the run.
4. **Response-contract compliance floor.** The floor is derived offline from
   the artifact's own declared inventory, with no factor carried over from
   any other arm (the §3.3 A1-derivation discipline). The strict parser's
   inventory is the canonical 39-phone set; the maximally uninformative
   responder consistent with the artifact's uniform-floor arithmetic is the
   uniform draw over that inventory, whose per-position error rate is
   same rate has response behavior exactly as uninformative as a uniform
   responder's task behavior — the arm cannot support any pronunciation
   claim. The refusal is per-probe: an arm below the floor records a single
   descriptive ``bench_synthg2p_probe_{tag}_contract_refused`` boolean claim
   (no accuracy claims for that arm) in the same atomic ClaimBatch, while
   the remaining compliant arms record their full measurement. On the
   frozen 60-word test population this admits at most one non-compliant
   word per probe.

Claims and provenance
---------------------
Namespace ``bench_synthg2p_probe_{gemma3,qwen25,smollm}_*`` (exact accuracy,
position accuracy, contract compliance, words scored, model digest) plus the
shared artifact digest, the pre-registered contract floor, and the scored
population counts; a probe below the floor records only its
``bench_synthg2p_probe_{tag}_contract_refused`` boolean instead. Labels are
replaced wholesale under the ledger's single-writer lock, the comparator
recorder's discipline. Per §3.5 the probe arms are reported descriptively;
no gate compares a probe to the comparator.
single-writer lock, the comparator recorder's discipline. Per §3.5 the probe
arms are reported descriptively; no gate compares a probe to the comparator.
On a publication run the recorder writes the standard network claim binding
receipt and exposes the executable semantic replay
(``validate_publication_measurement``) the binding verifier runs against the
cached replies — the ``scripts/record_synthetic_benchmark_claims.py``
discipline, extended to this namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Collection, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.benchmarks import (  # noqa: E402  (path setup precedes imports)
    OLLAMA_URL,
    ask_ollama,
    enable_call_cache,
    ollama_available,
    parse_g2p_answer,
    resolved_model_identity,
    warm_ollama_model,
)
from ailm.benchmarks.aligned_g2p import (  # noqa: E402  (path setup precedes imports)
    ALIGNED_G2P_ARTIFACT_NAME,
    artifact_pairs,
    load_aligned_g2p,
    probe_prompt,
    probe_request_payloads,
)
from ailm.benchmarks.constants import DEFAULT_SEED, PROBE_MODEL_TAGS  # noqa: E402
from ailm.benchmarks.ollama import _generate_payload  # noqa: E402
from ailm.benchmarks.synthetic_protocol import (  # noqa: E402
    SYNTHETIC_ALIGNED_G2P_PROTOCOL,
)
from ailm.call_cache import CacheRequest, CallCache, model_digest  # noqa: E402
from ailm.claims import ClaimBatch, ClaimsLedger, load_ledger  # noqa: E402
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.replay_provenance import (  # noqa: E402
    NetworkClaimBindingError,
    generate_run_scope,
    network_recorder_configuration,
    scoped_cache_protocol,
    source_manifest,
    using_publication_ledger,
    validate_network_measurement_inputs,
    validate_run_scope,
    write_network_claim_binding,
)

PROTOCOL = SYNTHETIC_ALIGNED_G2P_PROTOCOL
RUN_ID = PROTOCOL.probe_run_id
SOURCE = PROTOCOL.probe_source
RESPONSE_TOKENS = PROTOCOL.probe_response_tokens
NUM_CTX = PROTOCOL.probe_context_window

#: The offline comparator arm this run binds to (§3.4 execution order). The
#: binding label is the comparator recorder's own artifact digest claim.
COMPARATOR_SOURCE = "scripts/record_aligned_g2p_comparator_claims.py"
COMPARATOR_DIGEST_LABEL = "bench_synthg2p_comparator_artifact_sha256"

#: The shared (probe-independent) labels of the recorder's namespace. The
#: per-probe suffixes are generated over the registered probe keys; a probe
#: added to SERVED_PROBES extends the namespace by construction.
SHARED_LABELS = (
    "bench_synthg2p_probe_artifact_sha256",
    "bench_synthg2p_probe_contract_floor",
    "bench_synthg2p_probe_test_word_count",
    "bench_synthg2p_probe_test_position_count",
)

PER_PROBE_STEMS = (
    "exact_accuracy",
    "position_accuracy",
    "contract_compliance",
    "words_scored",
    "model_digest",
)

#: The boolean per-probe refusal stem: a probe whose contract compliance
#: falls below the pre-registered floor records exactly this claim — and no
#: accuracy claims — in the same atomic ClaimBatch.
REFUSED_STEM = "contract_refused"


def _artifact_digest(vendor_dir: Path | None = None) -> str:
    """The sha256 of the vendored artifact this run binds to."""

    artifact_path = (
        vendor_dir / ALIGNED_G2P_ARTIFACT_NAME
        if vendor_dir is not None
        else REPO_ROOT / "vendor" / ALIGNED_G2P_ARTIFACT_NAME
    )
    return hashlib.sha256(artifact_path.read_bytes()).hexdigest()


def _run_cache_protocol(run_scope: str) -> str:
    """This run's cache namespace: the protocol identity scoped by the run.

    The unscoped ``probe_cache_protocol`` remains the protocol identity the
    ledger records and a reader cites; only the cache key carries the run,
    so a re-run writes its own keys instead of colliding with the first.
    """

    return scoped_cache_protocol(PROTOCOL.probe_cache_protocol, run_scope)


def contract_compliance_floor(inventory_size: int) -> float:
    """The pre-registered response-contract floor, derived offline.

    ``1 - 1/inventory_size``: the uniform draw over the declared response
    inventory's per-position error rate, applied to the contract itself (the
    derivation in the module docstring). No factor is carried over from any
    other arm; the constant comes from this artifact's own inventory at
    finalization time, before any probe call.
    """

    if inventory_size <= 0:
        raise ValueError("inventory size must be positive")
    return 1.0 - 1.0 / inventory_size


def assert_artifact_binding(ledger: ClaimsLedger, artifact_sha256: str) -> None:
    """Gate 1: the probe run scores the comparator run's exact artifact."""

    recorded = ledger.get(COMPARATOR_DIGEST_LABEL)
    if recorded is None:
        raise SystemExit(
            "aligned-g2p probe refused: the offline comparator arm has not been "
            f"recorded yet (no {COMPARATOR_DIGEST_LABEL} from {COMPARATOR_SOURCE}); "
            "the probe arms bind to that run's artifact. No claims written."
        )
    if recorded.get("value") != artifact_sha256:
        raise SystemExit(
            "aligned-g2p probe refused: artifact digest mismatch — the vendored "
            f"artifact is {artifact_sha256} but the recorded comparator arm was "
            f"measured on {recorded.get('value')}. No claims written."
        )


def assert_split_disjointness(
    train: list[tuple[str, list[str]]], test: list[tuple[str, list[str]]]
) -> None:
    """Gate 2 (L3): the §3.1 set assertions, re-asserted before any call."""

    for name, split in (("train", train), ("test", test)):
        words = [word for word, _gold in split]
        if len(words) != len(set(words)):
            raise SystemExit(
                f"aligned-g2p probe refused: duplicate word inside the {name} "
                "split. No claims written."
            )
    overlap = sorted({word for word, _ in train} & {word for word, _ in test})
    if overlap:
        raise SystemExit(
            f"aligned-g2p probe refused: cross-split word collision {overlap}; "
            "the L3 train/eval disjointness assertion fails. No claims written."
        )
    for name, split in (("train", train), ("test", test)):
        misaligned = [
            word
            for word, gold in split
            if len(word) != PROTOCOL.word_length or len(gold) != PROTOCOL.word_length
        ]
        if misaligned:
            raise SystemExit(
                f"aligned-g2p probe refused: {name} split is not 1:1 aligned at "
                f"the declared length {PROTOCOL.word_length}: {misaligned[:3]}. "
                "No claims written."
            )


def assert_gold_isolation(
    payloads: list[dict[str, object]], test: list[tuple[str, list[str]]]
) -> None:
    """Gate 3 (L2): no gold string anywhere in any request payload.

    The payloads are exactly what the scored calls will serialize, so the
    scan runs against the real request material before the first call. The
    scan is symmetric: every gold string against every payload.
    """

    golds = [" ".join(gold) for _word, gold in test]
    for index, payload in enumerate(payloads):
        blob = json.dumps(payload, sort_keys=True)
        for gold in golds:
            if gold in blob:
                raise SystemExit(
                    f"aligned-g2p probe refused (L2 gold isolation): test gold "
                    f"{gold!r} appears in request payload {index}. No claims "
                    "written."
                )


def assert_contract_floor(
    compliance: float, floor: float, probe_key: str, words_scored: int
) -> None:
    """Gate 4: the response-contract compliance floor, pre-registered above.

    Refuses strictly below the floor; at the floor the arm is exactly as
    informative as the uniform responder is at the task and is still a
    (degenerate) measurement.
    """

    if compliance < floor:
        raise SystemExit(
            f"aligned-g2p probe refused (response-contract floor): probe "
            f"{probe_key} complied with the strict JSON array contract on "
            f"{compliance:.4f} of {words_scored} scored words, below the "
            f"pre-registered floor {floor:.6f} = 1 - 1/"
            f"{len(PROTOCOL.response_inventory)} (the uniform draw's per-position "
            "error rate over the artifact's own 39-phone inventory). The arm "
            "cannot support a pronunciation claim. No claims written."
        )


def score_probe_replies(
    words: list[str], golds: list[list[str]], replies: list[str]
) -> dict[str, object]:
    """Score one probe's replies through the shipped strict parser.

    The replayable core of the measurement: pure, no network, no store.
    Exact match is whole-list equality against the scorer-only gold; the
    per-position denominator is every gold position, and a reply shorter
    than its word's gold simply leaves the unmatched gold positions not
    correct (zip stops at the shorter side). Contract compliance is the
    strict parser's acceptance, independent of correctness.
    """

    if not (len(words) == len(golds) == len(replies)):
        raise ValueError("words, golds, and replies must be equal length")
    exact_flags: list[bool] = []
    compliant_flags: list[bool] = []
    position_correct = 0
    position_total = 0
    for gold, reply in zip(golds, replies, strict=True):
        parsed, bad = parse_g2p_answer(reply)
        exact_flags.append(parsed == gold)
        compliant_flags.append(not bad)
        position_correct += sum(p == g for p, g in zip(parsed, gold, strict=False))
        position_total += len(gold)
    total = len(words)
    return {
        "exact_accuracy": sum(exact_flags) / total if total else 0.0,
        "position_accuracy": position_correct / position_total if position_total else 0.0,
        "contract_compliance": sum(compliant_flags) / total if total else 0.0,
        "words_scored": total,
    }


def claim_values(
    scored: dict[str, dict[str, object]],
    identities: dict[str, str],
    *,
    refused: Collection[str] = (),
    artifact_sha256: str,
    floor: float,
    test_word_count: int,
    test_position_count: int,
) -> dict[str, object]:
    """Map the per-probe results onto the recorder's claim-label namespace.

    Probes in ``refused`` fell below the pre-registered contract floor: each
    records exactly one ``bench_synthg2p_probe_{key}_contract_refused``
    boolean claim and no accuracy claims.
    """

    values: dict[str, object] = {
        "bench_synthg2p_probe_artifact_sha256": artifact_sha256,
        "bench_synthg2p_probe_contract_floor": float(floor),
        "bench_synthg2p_probe_test_word_count": test_word_count,
        "bench_synthg2p_probe_test_position_count": test_position_count,
    }
    for key in sorted(PROBE_MODEL_TAGS):
        if key in refused:
            values[f"bench_synthg2p_probe_{key}_{REFUSED_STEM}"] = True
            continue
        results = scored[key]
        values[f"bench_synthg2p_probe_{key}_exact_accuracy"] = round(
            float(results["exact_accuracy"]), 8
        )
        values[f"bench_synthg2p_probe_{key}_position_accuracy"] = round(
            float(results["position_accuracy"]), 8
        )
        values[f"bench_synthg2p_probe_{key}_contract_compliance"] = round(
            float(results["contract_compliance"]), 8
        )
        values[f"bench_synthg2p_probe_{key}_words_scored"] = results["words_scored"]
        values[f"bench_synthg2p_probe_{key}_model_digest"] = identities[key]
    return values


def all_claim_labels() -> frozenset[str]:
    """The recorder's complete namespace, per-probe suffixes expanded."""

    per_probe = {
        f"bench_synthg2p_probe_{key}_{stem}" for key in PROBE_MODEL_TAGS for stem in PER_PROBE_STEMS
    } | {f"bench_synthg2p_probe_{key}_{REFUSED_STEM}" for key in PROBE_MODEL_TAGS}
    return frozenset(per_probe | set(SHARED_LABELS))


def publication_replay_configuration(*, run_scope: str) -> dict[str, object]:
    """Canonical configuration the receipt binds this recorder's run to.

    Only ``run_scope`` is free; everything else is fixed by this module's
    code. The binding verifier reads the scope from the receipt under audit
    and requires exactly the configuration that scope implies.
    """

    return network_recorder_configuration(
        endpoint=OLLAMA_URL,
        call_protocol=_run_cache_protocol(run_scope),
        model_tags=list(PROBE_MODEL_TAGS.values()),
        parameters={
            "run_scope": validate_run_scope(run_scope),
            "cache_protocols": [_run_cache_protocol(run_scope)],
            "artifact": f"vendor/{ALIGNED_G2P_ARTIFACT_NAME}",
            "response_tokens": RESPONSE_TOKENS,
            "context_window": NUM_CTX,
            "seed": DEFAULT_SEED,
            "probes": sorted(PROBE_MODEL_TAGS.values()),
            "test_words": PROTOCOL.test_words,
            "warmup_policy": PROTOCOL.probe_warmup_policy,
        },
    )


def _measurement_evidence(
    scored: dict[str, dict[str, object]],
    identities: dict[str, str],
    *,
    refused: Collection[str] = (),
    cache_hits: int,
    cache_misses: int,
    unique_cache_keys: int,
    test_word_count: int,
    test_position_count: int,
) -> dict[str, object]:
    """The measurement object bound into the receipt and re-checked at replay.

    A probe below the pre-registered contract floor carries a descriptive
    ``contract_refused`` arm naming its observed compliance instead of the
    accuracy fields the compliant arms carry.
    """

    arms: dict[str, object] = {}
    for key in sorted(PROBE_MODEL_TAGS):
        arm: dict[str, object] = {"model": PROBE_MODEL_TAGS[key]}
        if key in refused:
            arm["status"] = REFUSED_STEM
        else:
            arm["words_scored"] = scored[key]["words_scored"]
            arm["exact_accuracy"] = scored[key]["exact_accuracy"]
            arm["position_accuracy"] = scored[key]["position_accuracy"]
        arm["contract_compliance"] = scored[key]["contract_compliance"]
        arms[key] = arm
    return {
        "arms": arms,
        "model_digests": {key: identities[key] for key in sorted(identities)},
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "unique_cache_keys": unique_cache_keys,
        "test_word_count": test_word_count,
        "test_position_count": test_position_count,
    }


def validate_publication_measurement(
    receipt: Mapping[str, object], store: CallCache
) -> dict[str, object]:
    """Re-derive every claim from the exact cached replies (semantic replay).

    The binding verifier runs this against the receipt the publication run
    wrote. The full request set is reconstructed offline from the frozen
    artifact and the module constants, each cached record must carry exactly
    the reconstructed payload under the receipt's binding, and the replies
    are re-parsed and re-scored; any disagreement with the receipt's claim
    projection refuses the binding.
    """

    cache = receipt.get("cache")
    configuration = receipt.get("configuration")
    measurement = receipt.get("measurement")
    if (
        not isinstance(cache, dict)
        or not isinstance(configuration, dict)
        or not isinstance(measurement, dict)
    ):
        raise NetworkClaimBindingError("aligned-g2p probe receipt envelope is invalid")
    endpoint = configuration.get("endpoint")
    protocol = configuration.get("call_protocol")
    if not isinstance(endpoint, str) or not isinstance(protocol, str):
        raise NetworkClaimBindingError("aligned-g2p probe receipt identity is invalid")
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict):
        raise NetworkClaimBindingError("aligned-g2p probe receipt parameters are invalid")
    declared_scope = validate_run_scope(parameters.get("run_scope"))
    if parameters.get("cache_protocols") != [_run_cache_protocol(declared_scope)]:
        raise NetworkClaimBindingError(
            "aligned-g2p probe receipt cache namespaces do not match its declared run scope"
        )
    if protocol != _run_cache_protocol(declared_scope):
        raise NetworkClaimBindingError(
            "aligned-g2p probe receipt call protocol does not match its declared run scope"
        )
    identity_rows = cache.get("model_identities")
    if not isinstance(identity_rows, list):
        raise NetworkClaimBindingError("aligned-g2p probe receipt identities are invalid")
    identities = {
        row["model"]: row["identity"]
        for row in identity_rows
        if isinstance(row, dict)
        and isinstance(row.get("model"), str)
        and isinstance(row.get("identity"), str)
    }
    if set(identities) != set(PROBE_MODEL_TAGS.values()):
        raise NetworkClaimBindingError("aligned-g2p probe receipt model identities are incomplete")
    cache_keys = cache.get("keys")
    if not isinstance(cache_keys, list):
        raise NetworkClaimBindingError("aligned-g2p probe receipt cache keys are invalid")
    selected = set(cache_keys)
    cached = {key: (payload, reply) for key, payload, reply in store.entries() if key in selected}
    if set(cached) != selected:
        raise NetworkClaimBindingError(
            "aligned-g2p probe semantic replay is missing cached replies"
        )

    artifact_sha256 = _artifact_digest()
    artifact = load_aligned_g2p()
    _correspondence, _train, test = artifact_pairs(artifact)
    words = [word for word, _gold in test]
    golds = [gold for _word, gold in test]
    payloads = probe_request_payloads(words, response_tokens=RESPONSE_TOKENS)

    floor = contract_compliance_floor(len(PROTOCOL.response_inventory))
    refused: set[str] = set()
    scored: dict[str, dict[str, object]] = {}
    for key, tag in sorted(PROBE_MODEL_TAGS.items()):
        replies: list[str] = []
        for row in payloads:
            request_payload = _generate_payload(
                tag, row["prompt"], "g2p", RESPONSE_TOKENS, DEFAULT_SEED, NUM_CTX
            )
            request = CacheRequest(
                payload=request_payload,
                endpoint=endpoint,
                protocol=protocol,
                identity=identities[tag],
            )
            if request.key not in selected:
                raise NetworkClaimBindingError(
                    "aligned-g2p probe semantic replay produced keys outside the "
                    "receipt's cache binding"
                )
            record = cached.get(request.key)
            if record is None or record[0] != request_payload:
                raise NetworkClaimBindingError(
                    f"aligned-g2p probe semantic replay lacks the exact {tag} "
                    f"request for {row['prompt'][-20:]!r}"
                )
            replies.append(str(record[1].get("response", "")))
        scored[key] = score_probe_replies(words, golds, replies)
        if float(scored[key]["contract_compliance"]) < floor:
            refused.add(key)
        arm = measurement.get("arms", {}).get(key)
        if not isinstance(arm, dict):
            raise NetworkClaimBindingError(
                f"aligned-g2p probe receipt measurement is missing the arm for {key}"
            )
        if key in refused:
            if (
                arm.get("status") != REFUSED_STEM
                or arm.get("contract_compliance") != scored[key]["contract_compliance"]
            ):
                raise NetworkClaimBindingError(
                    f"aligned-g2p probe receipt measurement disagrees with the "
                    f"cached replies for {key}"
                )
        elif any(
            arm.get(field) != scored[key][field]
            for field in (
                "words_scored",
                "exact_accuracy",
                "position_accuracy",
                "contract_compliance",
            )
        ):
            raise NetworkClaimBindingError(
                f"aligned-g2p probe receipt measurement disagrees with the cached replies for {key}"
            )
        if arm.get("model") != tag:
            raise NetworkClaimBindingError(
                f"aligned-g2p probe receipt names the wrong model for {key}"
            )
    if measurement.get("unique_cache_keys") != len(selected):
        raise NetworkClaimBindingError(
            "aligned-g2p probe receipt cache-key count is not the scored set"
        )

    values = claim_values(
        scored,
        {key: identities[tag] for key, tag in PROBE_MODEL_TAGS.items()},
        refused=refused,
        artifact_sha256=artifact_sha256,
        floor=floor,
        test_word_count=len(words),
        test_position_count=sum(len(gold) for gold in golds),
    )
    claims = receipt.get("claims")
    entries = claims.get("entries") if isinstance(claims, dict) else None
    if not isinstance(entries, dict):
        raise NetworkClaimBindingError("aligned-g2p probe receipt claims are invalid")
    recorded = {
        label: entry.get("value")
        for label, entry in entries.items()
        if isinstance(label, str)
        and label.startswith("bench_synthg2p_probe_")
        and isinstance(entry, dict)
    }
    mismatches = {
        label: (recorded.get(label), expected)
        for label, expected in values.items()
        if recorded.get(label) != expected
    }
    missing_or_extra = sorted(set(recorded) ^ set(values))
    if mismatches or missing_or_extra:
        raise NetworkClaimBindingError(
            "aligned-g2p probe receipt claims do not rederive from cached replies: "
            f"mismatches={mismatches}, missing_or_extra={missing_or_extra}"
        )
    return values


def record(values: dict[str, object]) -> None:
    """Wholesale namespace replacement under the single-writer lock."""

    ledger = load_ledger(REPO_ROOT)
    stale = [label for label in ledger.labels_owned_by(SOURCE) if label not in values]
    ledger.replace_claims(
        remove=stale,
        batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE)],
    )


def measure(args: argparse.Namespace, *, vendor_dir: Path | None = None) -> dict[str, object]:
    """Run the gates, score every registered probe, and write the claims.

    Every gate refusal here fires before the ledger write, so a refused run
    writes nothing (the comparator recorder's ordering discipline). The
    response-contract floor is the one per-probe exception: a probe below
    the floor is recorded as a single ``contract_refused`` claim (no
    accuracy claims) in the same atomic ClaimBatch as the compliant arms.
    On a publication run the standard network claim binding receipt is
    written after the atomic ledger commit.
    """
    publication_run = using_publication_ledger(REPO_ROOT)
    if publication_run and args.endpoint.rstrip("/") != OLLAMA_URL.rstrip("/"):
        raise SystemExit(
            "aligned-g2p probe refused: the publication ledger requires the "
            f"canonical endpoint {OLLAMA_URL!r}, got {args.endpoint!r}; "
            "no claims written"
        )
    run_scope = generate_run_scope()
    print(f"aligned-g2p probe run scope: {run_scope}", flush=True)

    unavailable = sorted(
        tag
        for tag in PROBE_MODEL_TAGS.values()
        if not ollama_available(tag, endpoint=args.endpoint)
    )
    if unavailable:
        raise SystemExit(
            f"aligned-g2p probe refused: Ollama/model unavailable: {unavailable}; "
            "a partial probe set cannot be recorded. No claims written."
        )
    identities: dict[str, str] = {}
    for key, tag in sorted(PROBE_MODEL_TAGS.items()):
        identity = resolved_model_identity(tag, endpoint=args.endpoint)
        if identity is None:
            raise SystemExit(
                "aligned-g2p probe refused: immutable identity unavailable for "
                f"{tag}; no claims written"
            )
        identities[key] = identity
    if len(set(identities.values())) != len(identities):
        raise SystemExit(
            "aligned-g2p probe refused: registered model tags resolved to one "
            "immutable identity; no claims written"
        )
    initial_source = source_manifest(REPO_ROOT, SOURCE)

    # ── Offline gates, before any call ───────────────────────────────────────
    artifact_sha256 = _artifact_digest(vendor_dir)
    payload = load_aligned_g2p(vendor_dir)
    _correspondence, train, test = artifact_pairs(payload)
    assert_split_disjointness(train, test)
    assert_artifact_binding(load_ledger(REPO_ROOT), artifact_sha256)

    words = [word for word, _gold in test]
    golds = [gold for _word, gold in test]
    payloads = probe_request_payloads(words, response_tokens=RESPONSE_TOKENS)
    if len(payloads) != len(words) or any(
        row["prompt"] != probe_prompt(word) for row, word in zip(payloads, words, strict=True)
    ):
        raise SystemExit(
            "aligned-g2p probe refused: the scored payload set is not exactly "
            "the L1-pure prompt builder over the test spellings. No claims written."
        )
    assert_gold_isolation(payloads, test)

    floor = contract_compliance_floor(len(PROTOCOL.response_inventory))
    maximum_prompt_characters = max(len(row["prompt"]) for row in payloads)
    if maximum_prompt_characters + RESPONSE_TOKENS >= NUM_CTX:
        raise SystemExit(
            "aligned-g2p probe refused: the declared context window does not "
            "cover the longest scored prompt plus the response budget; "
            "no claims written"
        )
    print(
        f"gates live: artifact {artifact_sha256[:12]}…, {len(words)} test words, "
        f"floor {floor:.6f}; scoring {len(PROBE_MODEL_TAGS)} probes",
        flush=True,
    )

    store = enable_call_cache(require_default_path=publication_run)
    if store is None:  # enabled=True above; defensive against future API drift
        raise SystemExit("aligned-g2p probe refused: call store could not be enabled")

    # ── The scored measurement ───────────────────────────────────────────────
    cache_protocol = _run_cache_protocol(run_scope)
    scored: dict[str, dict[str, object]] = {}
    refused: set[str] = set()
    measured_hits = 0
    measured_misses = 0
    measurement_keys: set[str] = set()
    for key, tag in sorted(PROBE_MODEL_TAGS.items()):
        # One deliberately uncached warm-up per probe, on a prompt outside the
        # scored set (L3 disjointness puts the first train word there).
        warm_ollama_model(
            tag,
            probe_prompt(train[0][0]),
            "g2p",
            RESPONSE_TOKENS,
            endpoint=args.endpoint,
            num_ctx=NUM_CTX,
        )
        store.clear_access_log()
        hits_before = store.stats.hits
        misses_before = store.stats.misses
        replies = [
            ask_ollama(
                tag,
                row["prompt"],
                "g2p",
                RESPONSE_TOKENS,
                seed=DEFAULT_SEED,
                endpoint=args.endpoint,
                cache_protocol=cache_protocol,
                cache_read=not publication_run,
                num_ctx=NUM_CTX,
            )
            for row in payloads
        ]
        hits = store.stats.hits - hits_before
        misses = store.stats.misses - misses_before
        if hits + misses != len(payloads):
            raise SystemExit(
                "aligned-g2p probe refused: scored cache accounting does not "
                f"match the schedule for {tag}: hits={hits}, misses={misses}, "
                f"expected={len(payloads)}. No claims written."
            )
        if publication_run and hits:
            raise SystemExit(
                "aligned-g2p probe refused: canonical run provenance requires "
                "fresh scored calls, but "
                f"{tag} replayed {hits} cached request(s); use an isolated "
                "ledger for resumable validation. No claims written."
            )
        measurement_keys.update(store.accessed_keys)
        measured_hits += hits
        measured_misses += misses
        if model_digest(tag, args.endpoint) != identities[key]:
            raise SystemExit(
                "aligned-g2p probe refused: model identity changed while "
                f"measuring {tag}; no claims written"
            )
        if source_manifest(REPO_ROOT, SOURCE) != initial_source:
            raise SystemExit(
                "aligned-g2p probe refused: source or vendored input changed "
                f"during measurement of {tag}; no claims written"
            )
        results = score_probe_replies(words, golds, replies)
        scored[key] = results
        compliance = float(results["contract_compliance"])
        if compliance < floor:
            refused.add(key)
            print(
                f"{tag} ({key}): contract refused — compliance {compliance:.4f} "
                f"below the floor {floor:.6f}; recording "
                f"bench_synthg2p_probe_{key}_{REFUSED_STEM} only",
                flush=True,
            )
            continue
        print(
            f"{tag} ({key}): exact {results['exact_accuracy']:.4f}, position "
            f"{results['position_accuracy']:.4f}, compliance "
            f"{results['contract_compliance']:.4f} over {len(words)} words",
            flush=True,
        )

    if source_manifest(REPO_ROOT, SOURCE) != initial_source:
        raise SystemExit(
            "aligned-g2p probe refused: source or vendored input changed before "
            "ledger publication; no claims written"
        )
    if not measurement_keys:
        raise SystemExit(
            "aligned-g2p probe refused: scored measurement produced no trusted "
            "cache keys; no claims written"
        )
    values = claim_values(
        scored,
        identities,
        refused=refused,
        artifact_sha256=artifact_sha256,
        floor=floor,
        test_word_count=len(words),
        test_position_count=sum(len(gold) for gold in golds),
    )
    measurement = _measurement_evidence(
        scored,
        refused=refused,
        identities=identities,
        cache_hits=measured_hits,
        cache_misses=measured_misses,
        unique_cache_keys=len(measurement_keys),
        test_word_count=len(words),
        test_position_count=sum(len(gold) for gold in golds),
    )
    if publication_run:
        # Pre-publication validation: a bad configuration, missing key, or
        # identity drift is a refusal before the ledger is touched.
        validate_network_measurement_inputs(
            configuration=publication_replay_configuration(run_scope=run_scope),
            store=store,
            cache_keys=measurement_keys,
            measurement=measurement,
        )
    record(values)
    print(f"recorded {len(values)} bench_synthg2p_probe claims (run_id={RUN_ID}):")
    for label in sorted(values):
        print(f"  {label} = {values[label]}")
    if refused:
        print(
            "contract-refused probes (recorded as contract_refused only): "
            + ", ".join(sorted(refused))
        )
    if publication_run:
        receipt_path = write_network_claim_binding(
            project_root=REPO_ROOT,
            recorder=SOURCE,
            expected_source_manifest=initial_source,
            configuration=publication_replay_configuration(run_scope=run_scope),
            store=store,
            cache_keys=measurement_keys,
            measurement=measurement,
        )
        print(f"network claim binding: {receipt_path.relative_to(REPO_ROOT)}")
    else:
        print("isolated ledger run: publication network claim binding left unchanged")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record the P2 zero-shot probe arms' bench_synthg2p_probe claims (protocol §3.4)."
        )
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=None,
        help=(
            "directory holding "
            f"{ALIGNED_G2P_ARTIFACT_NAME} (default: the canonical vendored artifact)"
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=OLLAMA_URL,
        help="loopback Ollama endpoint (non-default endpoints are isolated-validation only)",
    )
    args = parser.parse_args(argv)
    with single_writer_lock(ledger_lock_path(REPO_ROOT)):
        measure(args, vendor_dir=args.vendor_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
