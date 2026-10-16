# /// script
# requires-python = ">=3.11"
# ///
# Run: .venv/bin/python scripts/record_connected_gum_claims.py
"""Record the complete locally registered, previously unexamined GUM population.

Default execution always replays the full frozen live-agent comparison. The
validation-only mode authenticates inputs and population without scoring or
writing claims. There is no limit, substituted-corpus or precomputed-score mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _artifact_io import atomic_write_bytes  # noqa: E402

from ailm.claims import ClaimBatch, ClaimsLedger, load_ledger  # noqa: E402
from ailm.corpus.connected_gum import (  # noqa: E402
    GumPopulation,
    GumProtocolError,
    load_frozen_population,
    validate_registration,
)
from ailm.ledger_lock import ledger_lock_path, single_writer_lock  # noqa: E402
from ailm.levels.connected_builder import load_connected_model  # noqa: E402
from ailm.levels.connected_evaluation import (  # noqa: E402
    CONNECTED_STRENGTH,
    DocumentPair,
    document_uncertainty,
    evaluate_documents,
)
from ailm.verification.connected_provenance import (  # noqa: E402
    ExecutionBindingError,
    canonical_execution_binding,
)

SOURCE = "scripts/record_connected_gum_claims.py"
RUN_ID = "connected_gum_unexamined_v1"


def execution_binding(root: Path) -> dict[str, str]:
    """Delegate the canonical inventory shared with publication readers."""
    try:
        return canonical_execution_binding(root)
    except ExecutionBindingError as error:
        raise GumProtocolError(str(error)) from error


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def publish_measurement(
    pairs: tuple[DocumentPair, ...],
    binding: dict[str, str],
    ledger: ClaimsLedger,
    report_path: Path,
    population: GumPopulation,
) -> dict[str, int | float | str | bool]:
    """Publish only measured paired summaries; callers own authentication and the writer lock."""
    if tuple(pair.document_id for pair in pairs) != population.document_ids:
        raise GumProtocolError(
            "measured document identities differ from the authenticated population"
        )
    uncertainty = document_uncertainty([pair.letter_gain for pair in pairs])
    documents = len(pairs)
    tokens = sum(pair.integrated.normalized_tokens for pair in pairs)
    oov = sum(pair.integrated.oov_tokens for pair in pairs)
    source_hash = hashlib.sha256(_json_bytes(binding)).hexdigest()
    report = {
        "schema": "ailm-connected-gum-report-v1",
        "scope": "locally prospectively specified population; not external preregistration or global exact inference",
        "protocol_sha256": population.protocol_sha256,
        "input_manifest_sha256": population.input_manifest_sha256,
        "corpus_sha256": population.corpus_sha256,
        "excluded_documents": list(population.excluded_documents),
        "execution_binding": binding,
        "execution_sha256": source_hash,
        "runtime_sha256": binding["src/ailm/levels/connected.py"],
        "versions": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("inferactively-pymdp", "jax", "numpy")
            },
        },
        "uncertainty": dataclasses.asdict(uncertainty),
        "documents": [dataclasses.asdict(pair) for pair in pairs],
        "full_stream_word_loss_status": "infinite_oov" if oov else "finite",
    }
    report_bytes = _json_bytes(report)
    values: dict[str, int | float | str | bool] = {
        "connected_gum_documents": documents,
        "connected_gum_strength": CONNECTED_STRENGTH,
        "connected_gum_excluded_documents": len(population.excluded_documents),
        "connected_gum_source_sentences": sum(pair.source_sentences for pair in pairs),
        "connected_gum_source_characters": sum(pair.source_characters for pair in pairs),
        "connected_gum_normalized_tokens": tokens,
        "connected_gum_oov_tokens": oov,
        "connected_gum_in_vocab_tokens": tokens - oov,
        "connected_gum_vocab_coverage_pct": 100.0 * (tokens - oov) / tokens,
        "connected_gum_normalized_letters": sum(
            pair.integrated.normalized_letters for pair in pairs
        ),
        "connected_gum_scored_characters": sum(pair.integrated.scored_characters for pair in pairs),
        "connected_gum_document_integrated_log_loss": float(
            np.mean([p.integrated.letter_log_loss for p in pairs])
        ),
        "connected_gum_document_disconnected_log_loss": float(
            np.mean([p.disconnected.letter_log_loss for p in pairs])
        ),
        "connected_gum_document_mean_gain": uncertainty.document_mean_gain,
        "connected_gum_document_ci_low": uncertainty.ci_low,
        "connected_gum_document_ci_high": uncertainty.ci_high,
        "connected_gum_outcome": uncertainty.outcome,
        "connected_gum_bootstrap_seed": uncertainty.seed,
        "connected_gum_bootstrap_resamples": uncertainty.resamples,
        "connected_gum_bootstrap_confidence_pct": uncertainty.confidence * 100,
        "connected_gum_word_loss_is_infinite": bool(oov),
        "connected_gum_corpus_sha256": population.corpus_sha256,
        "connected_gum_protocol_sha256": population.protocol_sha256,
        "connected_gum_execution_sha256": source_hash,
        "connected_gum_runtime_sha256": binding["src/ailm/levels/connected.py"],
        "connected_gum_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }
    atomic_write_bytes(report_path, report_bytes)
    ledger.replace_claims(
        remove=[label for label in ledger.labels_owned_by(SOURCE) if label not in values],
        batches=[ClaimBatch(values=values, run_id=RUN_ID, source=SOURCE, status="descriptive")],
    )
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    population, selected = load_frozen_population(PROJECT_ROOT)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated_without_scoring",
                    "documents": len(selected),
                    "excluded": list(population.excluded_documents),
                    "corpus_sha256": population.corpus_sha256,
                }
            )
        )
        return 0
    before = execution_binding(PROJECT_ROOT)
    model = load_connected_model()
    pairs = []
    for pair in evaluate_documents(model, selected):
        pairs.append(pair)
        print(
            json.dumps(
                {
                    "completed_document": pair.document_id,
                    "completed": len(pairs),
                    "total": len(selected),
                    "gain": pair.letter_gain,
                }
            ),
            flush=True,
        )
    if execution_binding(PROJECT_ROOT) != before:
        raise GumProtocolError("implementation or model inputs changed during evaluation")
    validate_registration(PROJECT_ROOT)
    override = os.environ.get("AILM_CLAIMS_PATH")
    report = args.report or (
        Path(override).parent / "connected_gum_report.json"
        if override
        else PROJECT_ROOT / "output" / "reports" / "connected_gum.json"
    )
    values = publish_measurement(
        tuple(pairs), before, load_ledger(PROJECT_ROOT), report, population
    )
    print(json.dumps({"report": str(report), "claims": values}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    with single_writer_lock(ledger_lock_path(PROJECT_ROOT)):
        raise SystemExit(main())
