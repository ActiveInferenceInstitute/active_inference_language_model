#!/usr/bin/env python3
"""Generate a fitted-probe fitting dataset and its verifiable receipt.

Thin offline data-preparation entry point for the fitted-probe arm. Three
datasets, all deterministic functions of their declared inputs:

- ``letter``  — one supervised target per position of the declared normalized
  stream at the scored prefix window, one SFT line per supervised target in
  the exact scored prompt format (``build_prompt("letter", context)``, gold =
  the stream's next character), the positional diagnostic carve (seam gap,
  contiguous tail, optional diagnostic stride cap), and the experimental
  native comparator's refit counts over the fit region.
- ``phoneme`` — next-phone instances over the vendored CMUdict train split,
  prefix from the word's start, per-pair reset (the scored task's contract).
- ``g2p``     — one strict-1:1 SFT instance per vendored train pair.

Outputs land under ``--output-dir`` (default ``cache/fitted_probe_data/``):
``<dataset>-instances.jsonl`` plus an immutable receipt whose file digests
the script re-verifies before exiting. Rerunning with the same inputs onto an
existing receipt is refused (exit 2) — receipts are write-once. Small payloads
(g2p, phoneme, the letter refit counts) are built in memory and the receipt is
validated before their final byte is written; the letter instance files are
written streaming with digests computed as lines flush, so a failed run leaves
no orphan outputs and a full-corpus pass never materializes its millions of
instances in memory.

Full-corpus letter preparation (``--stream-source nltk-brown
--allow-full-corpus``) fetches the pinned NLTK ``brown.zip`` through the
authenticated path in ``src/ailm/corpus/brown.py``, rebuilds the normalized
full-corpus stream with the canonical writer's own recorded normalization
(``scripts/generate_counts.py``), reconciles the streaming census against the
vendored artifact's recorded counts, and writes the dataset plus its
write-once receipt. No training and no model calls happen here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ailm.benchmarks.constants import LETTER_WINDOW_SHORT
from ailm.benchmarks.fitted.census import StreamCensus
from ailm.benchmarks.fitted.data import (
    FitDiagnosticSplit,
    build_g2p_instances,
    build_phoneme_stream_instances,
    canonicalize_pairs,
    carve_diagnostic_tail,
    g2p_train_test_pairs,
    letter_sft_record,
    letter_stream_from_raw,
    reconcile_contexts,
)
from ailm.benchmarks.fitted.identity import sha256_digest
from ailm.benchmarks.fitted.receipt import DataReceipt, verify_receipt, write_receipt

DIAGNOSTIC_SIZE = 40_000
_CENSUS_CHUNK = 256
_FLUSH_LINES = 8192
_BARE_HEX = re.compile(r"^[0-9a-f]{64}$")


def _instances_blob(records: list[dict[str, object]]) -> bytes:
    """The exact bytes a dataset file will contain, built before any write."""
    lines = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        print(f"refusing to overwrite existing outputs: {existing}", file=sys.stderr)
        raise SystemExit(2)


def _cmudict_binding(counts: dict) -> dict[str, str]:
    """Bind the vendored artifact's bare-hex digest in receipt form."""
    digest = counts["source_sha256"]
    if not isinstance(digest, str) or not _BARE_HEX.match(digest):
        print(f"vendored cmudict artifact has malformed source_sha256: {digest!r}", file=sys.stderr)
        raise SystemExit(1)
    return {"cmudict_counts": "sha256:" + digest}


def _load_cmudict_pairs() -> tuple[dict, list[tuple[str, list[str]]]]:
    from ailm.levels.phoneme import load_phoneme_counts

    counts = load_phoneme_counts()
    pairs = [(word, phones_text.split()) for word, phones_text in counts["g2p_pairs"]]
    return counts, pairs


def _finish(
    receipt: DataReceipt, blobs: list[tuple[Path, bytes]], output_dir: Path, name: str
) -> int:
    """Write payloads only after the receipt validated, then self-verify."""
    for path, blob in blobs:
        path.write_bytes(blob)
    receipt_path = write_receipt(output_dir / f"{name}-receipt.json", receipt)
    report = verify_receipt(receipt_path, data_dir=output_dir)
    print(json.dumps({"receipt": str(receipt_path), **report}, sort_keys=True))
    return 0


def _sft_line(context: str, target: str) -> str:
    """One dataset line: the exact scored prompt plus the stream's next character."""
    return json.dumps(letter_sft_record(context, target), sort_keys=True, separators=(",", ":"))


def _default_brown_stream_builder() -> str:
    """The canonical writer's own authenticated pipeline, reused verbatim.

    ``generate_counts.load_brown_text`` reads the pinned ``brown.zip`` through
    ``src/ailm/corpus/brown.py`` and applies exactly the normalization string
    recorded in ``vendor/brown_counts.json`` — the same code path that
    produced the vendored artifact, so the rebuilt stream cannot drift from it.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generate_counts

    return generate_counts.load_brown_text()


def prepare_brown_stream(
    *, stream_builder: Callable[[], str] | None = None
) -> tuple[str, str, dict[str, str], dict[str, int]] | None:
    """Authenticate the pinned Brown source and rebuild the declared stream.

    Returns ``(stream, normalization, source bindings, recorded counts)``, or
    ``None`` after printing a refusal when the authenticated fetch fails. The
    archive is never fetched from the network here: the operator supplies it
    (``AILM_BROWN_SOURCE_ZIP``, an NLTK data root, or the builder's own
    resolution) and every byte is digest-verified against the recorded pin
    before use.

    ``stream_builder`` is this module's stream-source seam — callers (and the
    offline tests) inject the source as a parameter; the module is never
    monkeypatched.
    """
    from ailm.corpus.brown import BROWN_SOURCE_ENV, validate_brown_artifact_provenance
    from ailm.resources import resolve_vendored_artifact

    try:
        artifact_path = resolve_vendored_artifact("brown_counts.json")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        validate_brown_artifact_provenance(artifact)
    except (OSError, TypeError, ValueError) as exc:
        print(f"authenticated Brown provenance unavailable: {exc}", file=sys.stderr)
        return None
    provenance = artifact["provenance"]
    try:
        stream = (stream_builder or _default_brown_stream_builder)()
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"authenticated Brown fetch failed: {exc}. Fetch "
            f"{provenance['raw_source']['source_url']} and pass --source-zip PATH "
            f"or set {BROWN_SOURCE_ENV}.",
            file=sys.stderr,
        )
        return None
    bindings = {
        "brown_raw_source": "sha256:" + str(provenance["raw_source"]["source_sha256"]),
        "brown_counts_artifact": sha256_digest(artifact_path.read_bytes()),
    }
    return stream, str(provenance["normalization"]), bindings, dict(artifact["counts"])


def _stream_letter_dataset(
    stream: str,
    *,
    window: int,
    split: FitDiagnosticSplit,
    instances_path: Path,
    diagnostic_path: Path,
) -> tuple[dict[str, str], bytes]:
    """One cursor pass over the stream, writing both instance files streaming.

    The fit file receives one SFT line per fit position, the diagnostic file
    one per carved diagnostic position, and the same pass accumulates the
    comparator's refit counts over the fit region. Lines are flushed and
    hashed in bounded buffers, so the full-corpus pass never materializes its
    millions of per-position records. Returns the file digests for the
    receipt and the (small) refit-counts blob.
    """
    fit_hi = split.fit_positions[1]
    diagnostic_positions = set(split.diagnostic_positions)
    refit: dict[tuple[str, str], int] = {}
    instances_digest = hashlib.sha256()
    diagnostic_digest = hashlib.sha256()
    fit_lines: list[str] = []
    diagnostic_lines: list[str] = []
    n_fit = 0
    n_diagnostic = 0

    def flush(lines, digest, handle) -> None:
        if not lines:
            return
        chunk = "".join(line + "\n" for line in lines)
        handle.write(chunk)
        digest.update(chunk.encode("utf-8"))
        lines.clear()

    try:
        with (
            instances_path.open("w", encoding="utf-8") as instances_handle,
            diagnostic_path.open("w", encoding="utf-8") as diagnostic_handle,
        ):
            for i in range(1, len(stream)):
                target = stream[i]
                context = stream[max(0, i - window) : i]
                if i <= fit_hi:
                    fit_lines.append(_sft_line(context, target))
                    pair = (stream[i - 1], target)
                    refit[pair] = refit.get(pair, 0) + 1
                    n_fit += 1
                    if len(fit_lines) >= _FLUSH_LINES:
                        flush(fit_lines, instances_digest, instances_handle)
                elif i in diagnostic_positions:
                    diagnostic_lines.append(_sft_line(context, target))
                    n_diagnostic += 1
                    if len(diagnostic_lines) >= _FLUSH_LINES:
                        flush(diagnostic_lines, diagnostic_digest, diagnostic_handle)
            flush(fit_lines, instances_digest, instances_handle)
            flush(diagnostic_lines, diagnostic_digest, diagnostic_handle)
    except BaseException:
        instances_path.unlink(missing_ok=True)
        diagnostic_path.unlink(missing_ok=True)
        raise
    if n_fit != fit_hi or n_diagnostic != len(diagnostic_positions):
        instances_path.unlink(missing_ok=True)
        diagnostic_path.unlink(missing_ok=True)
        print(
            "streaming instance counts do not reconcile against the declared carve: "
            f"fit {n_fit} != {fit_hi}, diagnostic {n_diagnostic} != "
            f"{len(diagnostic_positions)}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    refit_blob = (json.dumps(canonicalize_pairs(refit), sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    digests = {
        instances_path.name: "sha256:" + instances_digest.hexdigest(),
        diagnostic_path.name: "sha256:" + diagnostic_digest.hexdigest(),
    }
    return digests, refit_blob


def construct_letter_dataset(
    stream: str,
    *,
    normalization: str,
    source_bindings: dict[str, str],
    artifact_counts: dict[str, int] | None,
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    """Census, seam carve, streaming write, and write-once receipt for one stream.

    The single construction path for both the authenticated full-corpus
    stream and a caller-supplied stream (the tests' injection seam): the
    streaming exact census, the declared diagnostic carve, the streaming
    instance writer, and the same receipt discipline. ``artifact_counts``,
    when given (the full-corpus path), must equal the streaming census
    exactly before anything is written.
    """
    instances_path = output_dir / "letter-instances.jsonl"
    diagnostic_path = output_dir / "letter-diagnostic-instances.jsonl"
    refit_path = output_dir / "letter-refit-bigram-counts.json"

    census = StreamCensus(window=args.window)
    for start in range(0, len(stream), _CENSUS_CHUNK):
        census.feed(stream[start : start + _CENSUS_CHUNK])
    totals = census.reconcile()
    if artifact_counts is not None and canonicalize_pairs(census.bigram_counts) != artifact_counts:
        print(
            "streaming census does not reconcile with the declared artifact's recorded "
            "counts; refusing to write under a stream that is not the declared one",
            file=sys.stderr,
        )
        return 1

    split = carve_diagnostic_tail(
        totals["supervised_targets"],
        window=args.window,
        gap=args.gap,
        diagnostic_size=min(args.diagnostic_size, totals["supervised_targets"] // 4),
        diagnostic_stride_cap=args.diagnostic_stride_cap or None,
    )
    separation = dict(split.counts())
    if args.diagnostic_stride_cap:
        tail_size = totals["supervised_targets"] - split.gap_positions[0] - args.gap + 1
        separation["diagnostic_stride_skipped"] = tail_size - len(split.diagnostic_positions)

    try:
        file_digests, refit_blob = _stream_letter_dataset(
            stream,
            window=args.window,
            split=split,
            instances_path=instances_path,
            diagnostic_path=diagnostic_path,
        )
        receipt = DataReceipt(
            dataset="letter",
            run_scope=args.run_scope,
            policy={
                "window": args.window,
                "gap": args.gap,
                "epochs": args.epochs,
                "seed": args.seed,
                "diagnostic_stride_cap": args.diagnostic_stride_cap or 0,
            },
            source_bindings={
                "declared_stream": sha256_digest(stream.encode("utf-8")),
                "normalization": sha256_digest(normalization.encode("utf-8")),
                **source_bindings,
            },
            separation=separation,
            counts={**totals, "context_occurrences_total": totals["supervised_targets"]},
            files={**file_digests, refit_path.name: sha256_digest(refit_blob)},
            generator="scripts/generate_fitted_probe_data.py",
        )
    except BaseException:
        for path in (instances_path, diagnostic_path, refit_path):
            path.unlink(missing_ok=True)
        raise
    refit_path.write_bytes(refit_blob)
    receipt_path = write_receipt(output_dir / "letter-receipt.json", receipt)
    report = verify_receipt(receipt_path, data_dir=output_dir)
    print(json.dumps({"receipt": str(receipt_path), **report}, sort_keys=True))
    return 0


def generate_letter(args: argparse.Namespace, output_dir: Path) -> int:
    instances_path = output_dir / "letter-instances.jsonl"
    diagnostic_path = output_dir / "letter-diagnostic-instances.jsonl"
    refit_path = output_dir / "letter-refit-bigram-counts.json"
    _refuse_existing([instances_path, diagnostic_path, refit_path])

    if args.stream_source == "nltk-brown":
        if not args.allow_full_corpus:
            print(
                "full-corpus letter preparation is gated: pass --allow-full-corpus "
                "under its own scheduled handoff",
                file=sys.stderr,
            )
            return 2
        prepared = prepare_brown_stream()
        if prepared is None:
            return 2
        stream, normalization, source_bindings, artifact_counts = prepared
        return construct_letter_dataset(
            stream,
            normalization=normalization,
            source_bindings=source_bindings,
            artifact_counts=artifact_counts,
            args=args,
            output_dir=output_dir,
        )

    if not args.stream_file:
        print("letter dataset needs --stream-file or --stream-source", file=sys.stderr)
        return 2
    stream_text = Path(args.stream_file).read_text(encoding="utf-8")
    if args.raw_stream:
        stream = letter_stream_from_raw(stream_text)
        normalization = "ailm.levels.letter.normalize_text"
    else:
        stream = stream_text
        normalization = "identity (pre-normalized fixture input)"
    return construct_letter_dataset(
        stream,
        normalization=normalization,
        source_bindings={},
        artifact_counts=None,
        args=args,
        output_dir=output_dir,
    )


def generate_g2p(args: argparse.Namespace, output_dir: Path) -> int:
    instances_path = output_dir / "g2p-instances.jsonl"
    _refuse_existing([instances_path])
    counts, pairs = _load_cmudict_pairs()
    train_pairs, _test_pairs = g2p_train_test_pairs(
        [(word, " ".join(phones)) for word, phones in pairs], test_every=counts["test_every"]
    )
    instances = build_g2p_instances(train_pairs)
    blob = _instances_blob(instances)
    reconciliation = reconcile_contexts(instances, key="gold")
    receipt = DataReceipt(
        dataset="g2p",
        run_scope=args.run_scope,
        policy={
            "test_every": counts["test_every"],
            "epochs": args.epochs,
            "seed": args.seed,
        },
        source_bindings=_cmudict_binding(counts),
        separation={
            "fit_positions": len(instances),
            "gap_skipped": 0,
            "diagnostic_positions": 0,
        },
        counts={
            "supervised_targets": len(instances),
            "bigram_tokens": len(instances),
            "distinct_contexts": reconciliation["distinct_contexts"],
        },
        files={instances_path.name: sha256_digest(blob)},
        generator="scripts/generate_fitted_probe_data.py",
    )
    return _finish(receipt, [(instances_path, blob)], output_dir, "g2p")


def generate_phoneme(args: argparse.Namespace, output_dir: Path) -> int:
    instances_path = output_dir / "phoneme-instances.jsonl"
    _refuse_existing([instances_path])
    counts, pairs = _load_cmudict_pairs()
    train_pairs, _test_pairs = g2p_train_test_pairs(
        [(word, " ".join(phones)) for word, phones in pairs], test_every=counts["test_every"]
    )
    instances = build_phoneme_stream_instances(train_pairs)
    blob = _instances_blob(instances)
    reconciliation = reconcile_contexts(instances, key="prefix")
    receipt = DataReceipt(
        dataset="phoneme",
        run_scope=args.run_scope,
        policy={
            "test_every": counts["test_every"],
            "epochs": args.epochs,
            "seed": args.seed,
            "window": 0,
        },
        source_bindings=_cmudict_binding(counts),
        separation={
            "fit_positions": len(instances),
            "gap_skipped": 0,
            "diagnostic_positions": 0,
        },
        counts={
            "supervised_targets": len(instances),
            "bigram_tokens": len(instances),
            "distinct_contexts": reconciliation["distinct_contexts"],
        },
        files={instances_path.name: sha256_digest(blob)},
        generator="scripts/generate_fitted_probe_data.py",
    )
    return _finish(receipt, [(instances_path, blob)], output_dir, "phoneme")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=("letter", "phoneme", "g2p"))
    parser.add_argument("--output-dir", default="cache/fitted_probe_data")
    parser.add_argument("--run-scope", required=True, help="declared run scope; no clock reads")
    parser.add_argument("--stream-file", help="letter only: path to the declared stream text")
    parser.add_argument(
        "--raw-stream",
        action="store_true",
        help="letter only: normalize --stream-file with the recorded normalizer",
    )
    parser.add_argument(
        "--stream-source",
        choices=("nltk-brown",),
        help="letter only: full-corpus source; gated behind --allow-full-corpus",
    )
    parser.add_argument("--allow-full-corpus", action="store_true")
    parser.add_argument("--window", type=int, default=LETTER_WINDOW_SHORT)
    parser.add_argument("--gap", type=int, default=LETTER_WINDOW_SHORT)
    parser.add_argument("--diagnostic-size", type=int, default=DIAGNOSTIC_SIZE)
    parser.add_argument("--diagnostic-stride-cap", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be >= 1")
    if args.gap < 1:
        parser.error("gap must be >= 1")
    if args.gap < args.window:
        parser.error(
            f"seam invariant refused: declared gap {args.gap} is below the "
            f"conditioning window {args.window}; the seam requires gap >= window "
            "(protocol §2.2) so no diagnostic conditioning window reaches across "
            "it into the fit region. Relaxing this requires a protocol-text "
            "change, not a narrower gap."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "letter":
        return generate_letter(args, output_dir)
    if args.dataset == "phoneme":
        return generate_phoneme(args, output_dir)
    return generate_g2p(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
