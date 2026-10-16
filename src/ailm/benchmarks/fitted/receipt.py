"""Receipt and checkpoint contracts for the fitted-probe data foundation.

A data receipt binds one generated dataset to its exact counts, declared
policy, source bindings, file digests, and separation arithmetic — written
once, immutable, and independently verifiable by recomputing every file
digest. A checkpoint spec is the contract a later isolated trainer consumes
to bind an adapter run to a verified data receipt. Nothing here trains,
downloads, or reads the network; all digests are computed locally.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from ailm.benchmarks.fitted.identity import DIGEST_PATTERN, TASKS, sha256_digest

RECEIPT_PROTOCOL = "fitted-probe-data-receipt-v1"


class DataReceiptError(ValueError):
    """Raised for any malformed, tampered, or unverifiable receipt."""


def _require_digest(value: object, what: str) -> str:
    if not isinstance(value, str) or not DIGEST_PATTERN.match(value):
        raise DataReceiptError(f"{what} must be 'sha256:<64 hex>', got {value!r}")
    return value


@dataclass(frozen=True)
class DataReceipt:
    """Immutable record of one generated fitting dataset.

    ``counts.bigram_tokens == counts.supervised_targets`` and
    ``separation.fit + separation.gap_skipped + separation.diagnostic ==
    counts.supervised_targets`` are enforced: a receipt whose arithmetic does
    not reconcile is refused, and no receipt covers a thinned census.
    """

    dataset: str
    run_scope: str
    policy: Dict[str, int]
    source_bindings: Dict[str, str]
    separation: Dict[str, int]
    counts: Dict[str, int]
    files: Dict[str, str]
    generator: str
    protocol: str = RECEIPT_PROTOCOL

    def __post_init__(self) -> None:
        if self.dataset not in TASKS:
            raise DataReceiptError(f"dataset must be one of {TASKS}, got {self.dataset!r}")
        if self.protocol != RECEIPT_PROTOCOL:
            raise DataReceiptError(f"protocol must be {RECEIPT_PROTOCOL!r}, got {self.protocol!r}")
        if not isinstance(self.run_scope, str) or not self.run_scope.strip():
            raise DataReceiptError("run_scope must be a nonempty string")
        for name, digest in self.source_bindings.items():
            _require_digest(digest, f"source binding {name!r}")
        for name, digest in self.files.items():
            _require_digest(digest, f"file binding {name!r}")
        for group in (self.policy, self.separation, self.counts):
            for key, value in group.items():
                if not isinstance(key, str) or not key:
                    raise DataReceiptError("count/policy keys must be nonempty strings")
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise DataReceiptError(f"{key} must be a nonnegative integer, got {value!r}")
        if (
            "window" in self.policy
            and "gap" in self.policy
            and self.policy["gap"] < self.policy["window"]
        ):
            raise DataReceiptError(
                f"seam invariant refused: receipt policy records gap {self.policy['gap']} "
                f"below the conditioning window {self.policy['window']}; the seam requires "
                "gap >= window so no diagnostic conditioning window reaches across it "
                "into the fit region. A receipt recording a narrower gap is fail-closed."
            )
        if self.counts.get("bigram_tokens") != self.counts.get("supervised_targets"):
            raise DataReceiptError(
                "receipt arithmetic refuses to reconcile: bigram_tokens "
                f"{self.counts.get('bigram_tokens')} != supervised_targets "
                f"{self.counts.get('supervised_targets')}"
            )
        separated = (
            self.separation.get("fit_positions", 0)
            + self.separation.get("gap_skipped", 0)
            + self.separation.get("diagnostic_positions", 0)
            + self.separation.get("diagnostic_stride_skipped", 0)
        )
        if separated != self.counts.get("supervised_targets"):
            raise DataReceiptError(
                f"separation {separated} does not reconcile with supervised_targets "
                f"{self.counts.get('supervised_targets')}"
            )
        if not isinstance(self.generator, str) or not self.generator.strip():
            raise DataReceiptError("generator must name its producing path")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "run_scope": self.run_scope,
            "policy": dict(sorted(self.policy.items())),
            "source_bindings": dict(sorted(self.source_bindings.items())),
            "separation": dict(sorted(self.separation.items())),
            "counts": dict(sorted(self.counts.items())),
            "files": dict(sorted(self.files.items())),
            "generator": self.generator,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataReceipt":
        required = {
            "dataset",
            "protocol",
            "run_scope",
            "policy",
            "source_bindings",
            "separation",
            "counts",
            "files",
            "generator",
        }
        missing = sorted(required - set(data))
        if missing:
            raise DataReceiptError(f"receipt is missing fields: {missing}")
        unknown = sorted(set(data) - required)
        if unknown:
            raise DataReceiptError(f"receipt has unknown fields: {unknown}")
        return cls(
            dataset=data["dataset"],
            protocol=data["protocol"],
            run_scope=data["run_scope"],
            policy=dict(data["policy"]),
            source_bindings=dict(data["source_bindings"]),
            separation=dict(data["separation"]),
            counts=dict(data["counts"]),
            files=dict(data["files"]),
            generator=data["generator"],
        )


def write_receipt(path: Path, receipt: DataReceipt) -> Path:
    """Write the receipt atomically, refusing to overwrite an existing one."""
    destination = Path(path)
    if destination.exists():
        raise DataReceiptError(f"refusing to overwrite existing receipt {destination}")
    payload = json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def load_receipt(path: Path) -> DataReceipt:
    """Load and validate a receipt from disk."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataReceiptError(f"receipt at {path} is not valid JSON: {error}") from error
    return DataReceipt.from_dict(data)


def verify_receipt(path: Path, *, data_dir: Path) -> Dict[str, Any]:
    """Recompute every bound file digest and reconcile against the receipt.

    Returns the verified totals. Any missing file, undigestable binding, or
    digest mismatch raises :class:`DataReceiptError` — verification is real
    recomputation, never a re-read of recorded claims.
    """
    receipt = load_receipt(path)
    if not receipt.files:
        raise DataReceiptError(f"receipt at {path} binds no files")
    for name, digest in receipt.files.items():
        bound = Path(data_dir) / name
        if not bound.is_file():
            raise DataReceiptError(f"receipt binds missing file {name!r} under {data_dir}")
        actual = sha256_digest(bound.read_bytes())
        if actual != digest:
            raise DataReceiptError(
                f"file {name!r} digest mismatch: receipt {digest} != actual {actual}"
            )
    return {
        "dataset": receipt.dataset,
        "run_scope": receipt.run_scope,
        "verified_files": len(receipt.files),
        "supervised_targets": receipt.counts["supervised_targets"],
        "separation": dict(receipt.separation),
    }


@dataclass(frozen=True)
class CheckpointSpec:
    """The contract a later isolated trainer binds its adapter run to.

    Fields only — no training logic. ``data_receipt_digest`` pins the exact
    verified dataset; ``adapter_path`` is where learned weights land; the
    step/epoch counters let a resumed run prove where it continues from.
    """

    dataset: str
    data_receipt_digest: str
    adapter_path: str
    optimizer_step: int
    epochs_completed: int
    trainer: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.dataset not in TASKS:
            raise DataReceiptError(f"dataset must be one of {TASKS}, got {self.dataset!r}")
        _require_digest(self.data_receipt_digest, "data_receipt_digest")
        if not isinstance(self.adapter_path, str) or not self.adapter_path.strip():
            raise DataReceiptError("adapter_path must be a nonempty path")
        for name in ("optimizer_step", "epochs_completed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DataReceiptError(f"{name} must be a nonnegative integer, got {value!r}")
        if not isinstance(self.trainer, str) or not self.trainer.strip():
            raise DataReceiptError("trainer must name its owning module or command")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "data_receipt_digest": self.data_receipt_digest,
            "adapter_path": self.adapter_path,
            "optimizer_step": self.optimizer_step,
            "epochs_completed": self.epochs_completed,
            "trainer": self.trainer,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointSpec":
        required = {
            "dataset",
            "data_receipt_digest",
            "adapter_path",
            "optimizer_step",
            "epochs_completed",
            "trainer",
        }
        missing = sorted(required - set(data))
        if missing:
            raise DataReceiptError(f"checkpoint is missing fields: {missing}")
        return cls(
            dataset=data["dataset"],
            data_receipt_digest=data["data_receipt_digest"],
            adapter_path=data["adapter_path"],
            optimizer_step=data["optimizer_step"],
            epochs_completed=data["epochs_completed"],
            trainer=data["trainer"],
            notes=str(data.get("notes", "")),
        )
