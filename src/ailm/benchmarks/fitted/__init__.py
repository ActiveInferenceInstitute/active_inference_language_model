"""Fitted-probe data/identity foundation for the external reference probes.

The separate experimental surface for the fitted-probe arm: identity and
lineage digests (:mod:`.identity`), streaming exact-census counting and the
stride-d aliasing refutation (:mod:`.census`), deterministic dataset builders
for the declared letter/phoneme/G2P sources (:mod:`.data`), and the receipt
and checkpoint contracts a later isolated trainer consumes (:mod:`.receipt`).

Nothing in this package enters ``ailm.benchmarks.constants.SERVED_PROBES``,
touches any historical bench namespace, downloads a model, or trains. All
modules are importable without the numerical stack; dataset builders import
the level helpers they reuse lazily, at call time.
"""

from ailm.benchmarks.fitted.census import (
    StreamCensus,
    bigram_relative_frequencies,
    stride_aliasing_demo,
)
from ailm.benchmarks.fitted.identity import (
    FITTED_PROBES,
    LINEAGE_STAGES,
    TASKS,
    FittedIdentityError,
    FittedProbeSpec,
    build_lineage_chain,
    sha256_digest,
    sha256_text,
    validate_fitted_probes,
)
from ailm.benchmarks.fitted.receipt import (
    RECEIPT_PROTOCOL,
    CheckpointSpec,
    DataReceipt,
    DataReceiptError,
    load_receipt,
    verify_receipt,
    write_receipt,
)

__all__ = [
    "FITTED_PROBES",
    "LINEAGE_STAGES",
    "RECEIPT_PROTOCOL",
    "TASKS",
    "CheckpointSpec",
    "DataReceipt",
    "DataReceiptError",
    "FittedIdentityError",
    "FittedProbeSpec",
    "StreamCensus",
    "bigram_relative_frequencies",
    "build_lineage_chain",
    "load_receipt",
    "sha256_digest",
    "sha256_text",
    "stride_aliasing_demo",
    "validate_fitted_probes",
    "verify_receipt",
    "write_receipt",
]
