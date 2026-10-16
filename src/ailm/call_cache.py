"""Integrity-checked, content-addressed storage for deterministic model calls.

One cache entry is useful only when it is inseparable from the exact request
that produced it. Each current record therefore carries the payload, endpoint,
protocol, immutable model identity, their recomputed key, and a digest of the
reply. A mismatched field, divergent duplicate, or malformed interior record
invalidates the store instead of becoming a plausible-looking answer.

The sole recoverable parse failure is a genuinely torn final append: the final
line is both unterminated and invalid JSON. It is discarded at the next locked
write. Invalid JSON anywhere else is corruption, not crash recovery.

Each process retains a cursor for the exact inode and size it has validated.
An unchanged file needs no reread; clean growth is reconciled from the prior
line boundary under the exclusive file lock. Inode replacement, truncation,
same-size mutation, or growth after an unterminated line forces a complete
validation before another reply can be written.

No portable filesystem API can prove that a non-cooperative writer appended
without also rewriting unseen bytes of the same inode. Tail reconciliation is
therefore provisional: it preserves the cooperative append fast path, but the
next replay, enumeration, digest, provenance binding, or receipt rereads and
validates the complete store. An unchanged, fully validated file stays O(1) at
those boundaries.

Trust boundary: request keys and reply digests are unkeyed SHA-256 checks. They
provide integrity and self-consistency under cooperative trusted-local writers;
they do not authenticate the server or resist a local writer that can replace a
record and recompute its digest. Immutable model identities and the release
profile's live re-query detect a different class of drift.

Only the current schema is accepted. A row from any other schema invalidates the
store before replay, enumeration, or publication; unsupported rows cannot coexist
with trusted calls behind a filtered index.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from ailm.loopback_http import LoopbackTransportError, loopback_json_request

__all__ = [
    "CACHE_SCHEMA",
    "CacheCorruptionError",
    "CacheRequest",
    "CallCache",
    "CacheStats",
    "cache_key",
    "default_cache_path",
    "model_digest",
]

CACHE_SCHEMA = "ailm-call-cache-v2"
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
#: Reply fields that describe *when and how* the server answered rather than
#: *what it answered*, and may therefore differ between two records of one
#: request without the records disagreeing.
#:
#: Reviewed individually, never by suffix. ``prompt_eval_cached_count`` is the
#: number of prompt tokens Ollama served from its own KV cache instead of
#: re-evaluating. It is 0 the first time a prompt is seen and positive when the
#: server still holds that prefix, so two identical requests issued moments
#: apart differ in it as a matter of course. It is a statement about the
#: server's cache, not about the model's output, and nothing in this project
#: reads it. Excluding it is what lets a probe legitimately ask the same
#: question twice — the stipulated arms repeat short prompts across positions,
#: and the reliability trials repeat one prompt deliberately — without the
#: second observation being refused as divergent.
#:
#: Keep the reviewing habit: a new Ollama field is semantic until someone has
#: shown it cannot vary with the answer.
_VOLATILE_OLLAMA_REPLY_FIELDS = frozenset(
    {
        "created_at",
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "prompt_eval_cached_count",
        "eval_duration",
    }
)


class CacheCorruptionError(ValueError):
    """The call store contains data that cannot be trusted or recovered."""


def default_cache_path(project_root: Path) -> Path:
    """Return the untracked, non-derived call-store path."""
    return project_root / "cache" / "bench_calls.jsonl"


def _canonical_json(value: Any) -> str:
    """A strict canonical JSON encoding used for every current digest."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_snapshot(value: Any) -> Any:
    """Return an alias-free value containing only canonical JSON types."""

    return json.loads(_canonical_json(value))


def _semantic_reply_digest(value: dict[str, Any]) -> str:
    """Digest a reply excluding only documented top-level timing metadata.

    Counts, context, log probabilities, response text, model, completion state,
    and every unrecognized field remain semantic. The allowlist is deliberately
    not suffix- or recursively based: a newly introduced ``*_duration`` field
    must be reviewed before concurrent replies may differ in it.
    """

    stable = {key: item for key, item in value.items() if key not in _VOLATILE_OLLAMA_REPLY_FIELDS}
    return _content_digest(stable)


def _current_records_equivalent(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Compare one request's records while retaining the first full snapshot."""

    structural_fields = (
        "schema",
        "key",
        "endpoint",
        "protocol",
        "identity",
        "payload",
    )
    if any(first[field] != second[field] for field in structural_fields):
        return False
    return hmac.compare_digest(
        _semantic_reply_digest(first["value"]),
        _semantic_reply_digest(second["value"]),
    )


def cache_key(
    *,
    payload: dict[str, Any],
    endpoint: str,
    protocol: str,
    identity: str,
) -> str:
    """Digest every request input that can change the server's answer."""
    material = {
        "schema": CACHE_SCHEMA,
        "endpoint": endpoint.rstrip("/"),
        "protocol": protocol,
        "identity": identity,
        "payload": payload,
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheRequest:
    """The complete material required to address and validate one call."""

    payload: dict[str, Any]
    endpoint: str
    protocol: str
    identity: str
    _canonical_payload: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise TypeError("cache payload must be a JSON object")
        if not self.endpoint or not self.protocol or not self.identity:
            raise ValueError("cache endpoint, protocol, and model identity must be non-empty")
        object.__setattr__(self, "endpoint", self.endpoint.rstrip("/"))
        canonical_payload = _canonical_json(self.payload)
        object.__setattr__(self, "payload", json.loads(canonical_payload))
        object.__setattr__(self, "_canonical_payload", canonical_payload)

    def payload_snapshot(self) -> dict[str, Any]:
        """Return the immutable-at-construction request payload as a fresh object."""

        payload = json.loads(self._canonical_payload)
        if not isinstance(payload, dict):  # defensive: __post_init__ enforces this
            raise TypeError("cache payload snapshot is not a JSON object")
        return payload

    @property
    def key(self) -> str:
        """Identity-bound cache key for this request payload."""
        return cache_key(
            payload=self.payload_snapshot(),
            endpoint=self.endpoint,
            protocol=self.protocol,
            identity=self.identity,
        )


@dataclass
class CacheStats:
    """Counts that distinguish live measurement from disk replay."""

    hits: int = 0
    misses: int = 0
    stored: int = 0
    unidentified: int = 0

    def as_dict(self) -> dict[str, int]:
        """Plain-dict view of the counters (hits/misses/stored/unidentified)."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stored": self.stored,
            "unidentified": self.unidentified,
        }

    @property
    def fresh_fraction(self) -> float:
        """Fraction of lookups that were cache misses (fresh computations)."""
        total = self.hits + self.misses
        return self.misses / total if total else 0.0


_MODELFILE_CONTENT_DIGEST_RE = re.compile(r"(?mi)^\s*FROM\s+\S*sha256[:-]([0-9a-f]{64})(?:\s|$)")


def _modelfile_content_digest(value: object) -> str | None:
    """Extract a resolved content blob from an Ollama Modelfile ``FROM`` line."""
    if not isinstance(value, str):
        return None
    match = _MODELFILE_CONTENT_DIGEST_RE.search(value)
    return f"sha256:{match.group(1).lower()}" if match is not None else None


def model_digest(model: str, endpoint: str, *, timeout: float = 10.0) -> str | None:
    """Return a content-bound model build identity from ``/api/show``."""
    try:
        body = loopback_json_request(
            endpoint, "/api/show", payload={"model": model}, timeout=timeout
        )
    except (OSError, ValueError, LoopbackTransportError):
        return None
    if not isinstance(body, dict):
        return None
    for path in (("digest",), ("details", "digest")):
        value: Any = body
        for step in path:
            value = value.get(step) if isinstance(value, dict) else None
        if not isinstance(value, str):
            continue
        match = re.fullmatch(r"(?:sha256:)?([0-9a-fA-F]{64})", value)
        if match is not None:
            return f"sha256:{match.group(1).lower()}"

    # The Modelfile's FROM line resolves to a content-bearing weights-blob
    # digest: when the server omits a top-level digest, the weights blob is
    # the model identity. The former stable-manifest fallback — a hash of the
    # server's presentation fields — is retired: that presentation moved twice
    # under an unchanged blob (decision 126 and its successor epoch), the
    # second time across server restarts on the same serving binary, so the
    # manifest hash was restart-unstable by construction. A response without a
    # server digest or a content-bearing FROM blob degrades to no caching: a
    # metadata-only answer cannot tell a re-pulled build from the measured one,
    # and any substitute would let the wrong build replay old measurements.
    content_blob = _modelfile_content_digest(body.get("modelfile"))
    if content_blob is not None:
        return content_blob
    return None


@dataclass
class _StoreState:
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    torn_offset: int | None = None
    needs_separator: bool = False
    parsed_offset: int = 0
    line_count: int = 0


@dataclass(frozen=True, slots=True)
class _FileStamp:
    """Identity and mutation metadata for the bytes represented in memory."""

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, file_stat: os.stat_result) -> _FileStamp:
        """Stamp a file from one ``os.stat`` result."""
        return cls(
            device=file_stat.st_dev,
            inode=file_stat.st_ino,
            size=file_stat.st_size,
            modified_ns=file_stat.st_mtime_ns,
            changed_ns=file_stat.st_ctime_ns,
        )

    @property
    def identity(self) -> tuple[int, int]:
        """(device, inode) pair identifying the file across path renames."""
        return self.device, self.inode


def _fail(path: Path, line_number: int, message: str) -> CacheCorruptionError:
    return CacheCorruptionError(f"{path}: line {line_number}: {message}")


def _validate_current_record(path: Path, line_number: int, record: Any) -> dict[str, Any]:
    required = {
        "schema",
        "key",
        "endpoint",
        "protocol",
        "identity",
        "payload",
        "value",
        "value_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise _fail(path, line_number, f"v2 record fields must be exactly {sorted(required)}")
    for name in ("key", "endpoint", "protocol", "identity", "value_sha256"):
        if not isinstance(record[name], str) or not record[name]:
            raise _fail(path, line_number, f"{name} must be a non-empty string")
    if _HEX_DIGEST.fullmatch(record["key"]) is None:
        raise _fail(path, line_number, "key is not a SHA-256 digest")
    if _HEX_DIGEST.fullmatch(record["value_sha256"]) is None:
        raise _fail(path, line_number, "value_sha256 is not a SHA-256 digest")
    if not isinstance(record["payload"], dict) or not isinstance(record["value"], dict):
        raise _fail(path, line_number, "payload and value must be objects")
    request = CacheRequest(
        payload=record["payload"],
        endpoint=record["endpoint"],
        protocol=record["protocol"],
        identity=record["identity"],
    )
    if not hmac.compare_digest(record["key"], request.key):
        raise _fail(path, line_number, "key does not match its recorded request material")
    if not hmac.compare_digest(record["value_sha256"], _content_digest(record["value"])):
        raise _fail(path, line_number, "value digest does not match the recorded reply")
    return record


def _parse_store(
    path: Path,
    data: bytes,
    *,
    prior: _StoreState | None = None,
    byte_offset: int = 0,
    line_offset: int = 0,
) -> _StoreState:
    """Validate a whole store or a tail beginning at a proven line boundary.

    Tail reconciliation copies the already-validated indexes, then parses only
    bytes appended after ``byte_offset``.  Copying a few dictionary references
    keeps a failed tail validation from partially mutating the live index while
    avoiding the dominant cost: decoding and hashing every historical reply.
    """

    state = _StoreState(
        records={} if prior is None else prior.records.copy(),
        parsed_offset=byte_offset,
        line_count=line_offset,
    )
    offset = byte_offset
    lines = data.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        line_number = line_offset + index + 1
        is_last = index == len(lines) - 1
        terminated = raw_line.endswith((b"\n", b"\r"))
        content = raw_line.strip()
        if not content:
            offset += len(raw_line)
            continue
        try:
            record = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if is_last and not terminated:
                state.torn_offset = offset
                state.parsed_offset = offset
                state.line_count = line_number - 1
                return state
            raise _fail(path, line_number, f"malformed interior JSON: {error}") from error
        if not isinstance(record, dict):
            raise _fail(path, line_number, "record is not an object")
        schema = record.get("schema")
        if schema != CACHE_SCHEMA:
            raise _fail(
                path,
                line_number,
                f"unsupported schema {schema!r}; expected {CACHE_SCHEMA!r}",
            )
        validated = _validate_current_record(path, line_number, record)
        key = validated["key"]
        previous = state.records.get(key)
        if previous is not None:
            if not _current_records_equivalent(previous, validated):
                raise _fail(path, line_number, f"divergent duplicate for key {key}")
            # Retain the complete first observation, including its volatile
            # envelope fields, as the canonical replay/provenance snapshot.
        else:
            state.records[key] = validated
        offset += len(raw_line)
    state.parsed_offset = offset
    state.line_count = line_offset + len(lines)
    state.needs_separator = bool(data) and not data.endswith((b"\n", b"\r"))
    return state


def _record_for(request: CacheRequest, value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("cached server reply must be a JSON object")
    reply = _json_snapshot(value)
    return {
        "schema": CACHE_SCHEMA,
        "key": request.key,
        "endpoint": request.endpoint,
        "protocol": request.protocol,
        "identity": request.identity,
        "payload": request.payload_snapshot(),
        "value": reply,
        "value_sha256": _content_digest(reply),
    }


def _open_regular_store(path: Path, flags: int, *, missing_ok: bool = False) -> int | None:
    """Open one leaf beneath a pinned, non-symlink directory descriptor.

    ``Path.open`` follows symbolic links and can block forever on a FIFO.  The
    call store is a release input, so both reads and appends instead use
    ``O_NOFOLLOW``/``O_NONBLOCK`` and validate the opened descriptor before any
    byte is consumed or written.  Opening relative to the directory descriptor
    also closes the check/open race on the immediate parent path.
    """

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    try:
        parent_fd = os.open(path.parent, os.O_RDONLY | directory | nofollow | cloexec)
    except OSError as error:
        raise CacheCorruptionError(
            f"{path}: cache parent must be a real directory, not a symlink or special path: {error}"
        ) from error
    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise CacheCorruptionError(f"{path}: cache parent is not a directory")
        try:
            secure_flags = nofollow | cloexec | nonblock
            if flags & os.O_CREAT:
                # macOS can transiently report ENOENT when two processes both
                # issue openat(O_CREAT | O_NOFOLLOW) for the same absent leaf.
                # Make creation explicit and exclusive instead: one process
                # creates the regular leaf and every loser reopens that exact
                # name without following a link.  This is also easier to audit
                # than relying on platform-specific O_CREAT race semantics.
                existing_flags = flags & ~(os.O_CREAT | os.O_EXCL)
                try:
                    descriptor = os.open(
                        path.name,
                        existing_flags | secure_flags,
                        dir_fd=parent_fd,
                    )
                except FileNotFoundError:
                    try:
                        descriptor = os.open(
                            path.name,
                            flags | os.O_EXCL | secure_flags,
                            0o600,
                            dir_fd=parent_fd,
                        )
                    except FileExistsError:
                        descriptor = os.open(
                            path.name,
                            existing_flags | secure_flags,
                            dir_fd=parent_fd,
                        )
            else:
                descriptor = os.open(
                    path.name,
                    flags | secure_flags,
                    dir_fd=parent_fd,
                )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        except OSError as error:
            raise CacheCorruptionError(
                f"{path}: cache entry must be a real regular file, not a symlink or "
                f"special path: {error}"
            ) from error
    finally:
        os.close(parent_fd)

    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise CacheCorruptionError(f"{path}: cache entry is not a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@dataclass
class CallCache:
    """Append-only call store with strict replay and cross-process locking."""

    path: Path
    enabled: bool = True
    stats: CacheStats = field(default_factory=CacheStats)
    _records: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _accessed_keys: set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _file_stamp: _FileStamp | None = field(default=None, init=False, repr=False)
    _torn_offset: int | None = field(default=None, init=False, repr=False)
    _needs_separator: bool = field(default=False, init=False, repr=False)
    _parsed_offset: int = field(default=0, init=False, repr=False)
    _line_count: int = field(default=0, init=False, repr=False)
    _full_reconciliations: int = field(default=0, init=False, repr=False)
    _tail_reconciliations: int = field(default=0, init=False, repr=False)
    _reconciled_bytes: int = field(default=0, init=False, repr=False)
    _requires_full_validation: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = _open_regular_store(self.path, os.O_RDONLY, missing_ok=True)
        if descriptor is None:
            return
        with os.fdopen(descriptor, "rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                self._verify_named_handle(handle)
                self._load_full_locked(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _state(self) -> _StoreState:
        """Return the validated in-memory index plus its parse cursor."""

        return _StoreState(
            records=self._records,
            torn_offset=self._torn_offset,
            needs_separator=self._needs_separator,
            parsed_offset=self._parsed_offset,
            line_count=self._line_count,
        )

    def _install_state(self, state: _StoreState, stamp: _FileStamp) -> None:
        self._records = state.records
        self._torn_offset = state.torn_offset
        self._needs_separator = state.needs_separator
        self._parsed_offset = state.parsed_offset
        self._line_count = state.line_count
        self._file_stamp = stamp

    def _verify_named_handle(self, handle: BinaryIO) -> None:
        """Refuse a descriptor whose directory entry was replaced after open."""

        opened = os.fstat(handle.fileno())
        try:
            named = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise CacheCorruptionError(
                f"{self.path}: cache path disappeared or changed after it was opened: {error}"
            ) from error
        if not stat.S_ISREG(named.st_mode):
            raise CacheCorruptionError(f"{self.path}: cache entry is not a regular file")
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise CacheCorruptionError(f"{self.path}: cache path was replaced while it was open")

    def _load_full_locked(self, handle: BinaryIO) -> _StoreState:
        """Reload the complete file after identity, truncation, or mutation."""

        before = _FileStamp.from_stat(os.fstat(handle.fileno()))
        handle.seek(0)
        data = handle.read()
        after = _FileStamp.from_stat(os.fstat(handle.fileno()))
        if before != after or len(data) != after.size:
            raise CacheCorruptionError(
                f"{self.path}: cache changed while a locked full reconciliation was reading it"
            )
        state = _parse_store(self.path, data)
        self._verify_named_handle(handle)
        self._full_reconciliations += 1
        self._reconciled_bytes += len(data)
        self._install_state(state, after)
        self._requires_full_validation = False
        return state

    def _reconcile_locked(
        self, handle: BinaryIO, *, require_full_validation: bool = False
    ) -> _StoreState:
        """Reconcile disk state, optionally proving the complete prefix anew."""

        self._verify_named_handle(handle)
        observed = _FileStamp.from_stat(os.fstat(handle.fileno()))
        previous = self._file_stamp
        if previous is None:
            if observed.size == 0:
                state = self._state()
                self._install_state(state, observed)
                return state
            return self._load_full_locked(handle)

        if require_full_validation and self._requires_full_validation:
            return self._load_full_locked(handle)

        if observed.identity != previous.identity or observed.size < previous.size:
            return self._load_full_locked(handle)

        if observed.size == previous.size:
            if observed == previous:
                return self._state()
            # A write which preserves length is not an append.  Revalidate every
            # byte because an earlier validated record may have been edited.
            return self._load_full_locked(handle)

        # Growth can be reconciled from the prior EOF only when that EOF was a
        # complete line boundary.  An old torn or unterminated final line may
        # have been completed, replaced, or made into malformed interior data,
        # so it deliberately takes the full-validation path once.
        if self._torn_offset is not None or self._needs_separator:
            return self._load_full_locked(handle)

        if require_full_validation:
            # Size growth alone cannot prove a non-cooperative writer left the
            # already-validated prefix untouched. Replay/publication boundaries
            # therefore validate the complete grown file once.
            return self._load_full_locked(handle)

        handle.seek(previous.size)
        tail = handle.read(observed.size - previous.size)
        after = _FileStamp.from_stat(os.fstat(handle.fileno()))
        if observed != after or len(tail) != observed.size - previous.size:
            raise CacheCorruptionError(
                f"{self.path}: cache changed while a locked tail reconciliation was reading it"
            )
        state = _parse_store(
            self.path,
            tail,
            prior=self._state(),
            byte_offset=previous.size,
            line_offset=self._line_count,
        )
        self._tail_reconciliations += 1
        self._reconciled_bytes += len(tail)
        self._install_state(state, after)
        self._requires_full_validation = True
        return state

    @property
    def reconciliation_stats(self) -> dict[str, int]:
        """Return read counters that make the incremental contract testable."""

        return {
            "full": self._full_reconciliations,
            "tail": self._tail_reconciliations,
            "bytes": self._reconciled_bytes,
        }

    def _refresh_locked(self, *, require_full_validation: bool) -> None:
        """Refresh observable state while the caller holds ``self._lock``."""

        if not self.enabled:
            return
        descriptor = _open_regular_store(self.path, os.O_RDONLY, missing_ok=True)
        if descriptor is None:
            if self._file_stamp is None:
                return
            raise CacheCorruptionError(
                f"{self.path}: validated cache disappeared before replay or publication"
            )
        with os.fdopen(descriptor, "rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                self._reconcile_locked(handle, require_full_validation=require_full_validation)
                self._verify_named_handle(handle)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @property
    def accessed_keys(self) -> frozenset[str]:
        """Snapshot of every cache key read under this session, lock-guarded."""
        with self._lock:
            return frozenset(self._accessed_keys)

    def clear_access_log(self) -> None:
        """Start a new provenance window without changing stored replies.

        Cache verification and scientific measurement can run in the same
        process.  A monotone access set cannot distinguish them, and set
        subtraction also loses a measurement call when the precondition happened
        to sample the same key first.  Clearing only this in-memory log after the
        precondition makes the next snapshot exactly the calls the measurement
        consumed; no cache record or statistic is altered.
        """

        with self._lock:
            self._accessed_keys.clear()

    def entries(self) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        """Return only fully validated current records for live remeasurement."""
        with self._lock:
            self._refresh_locked(require_full_validation=True)
            return [
                (key, _json_snapshot(record["payload"]), _json_snapshot(record["value"]))
                for key, record in self._records.items()
            ]

    def _records_digest_locked(self, keys: set[str] | frozenset[str] | None = None) -> str:
        selected = sorted(self._records if keys is None else keys)
        missing = [key for key in selected if key not in self._records]
        if missing:
            raise KeyError(f"validated call-store keys are missing: {missing[:3]}")
        return _content_digest([self._records[key] for key in selected])

    def records_digest(self, keys: set[str] | frozenset[str] | None = None) -> str:
        """Digest validated records consumed by a result sidecar."""
        with self._lock:
            self._refresh_locked(require_full_validation=True)
            return self._records_digest_locked(keys)

    def provenance_binding(self, keys: set[str] | frozenset[str]) -> dict[str, Any]:
        """Bind a derived receipt to exact validated records and key material."""
        with self._lock:
            self._refresh_locked(require_full_validation=True)
            selected = sorted(keys)
            missing = [key for key in selected if key not in self._records]
            if missing:
                raise KeyError(f"validated call-store keys are missing: {missing[:3]}")
            records = [self._records[key] for key in selected]
            return {
                "schema": CACHE_SCHEMA,
                "keys": selected,
                "keys_sha256": _content_digest(selected),
                "records_sha256": _content_digest(records),
                "endpoints": sorted({record["endpoint"] for record in records}),
                "protocols": sorted({record["protocol"] for record in records}),
                "identities": sorted({record["identity"] for record in records}),
                "models": sorted({str(record["payload"].get("model", "")) for record in records}),
                "model_identities": [
                    {"model": model, "identity": identity}
                    for model, identity in sorted(
                        {
                            (str(record["payload"].get("model", "")), record["identity"])
                            for record in records
                        }
                    )
                ],
            }

    def _append_records(self, records: list[dict[str, Any]]) -> int:
        """Append non-conflicting records while holding thread and OS locks."""
        if not records:
            return 0
        appended = 0
        with self._lock:
            descriptor = _open_regular_store(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT)
            if descriptor is None:  # pragma: no cover - creation cannot return missing
                raise CacheCorruptionError(f"{self.path}: cache creation unexpectedly failed")
            with os.fdopen(descriptor, "a+b") as handle:
                appended = self._append_locked(handle, records)
        return appended

    def _append_locked(self, handle: BinaryIO, records: list[dict[str, Any]]) -> int:
        """Validate and append while ``self._lock`` and the file lock are held."""

        appended = 0
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            state = self._reconcile_locked(handle)
            if state.torn_offset is not None:
                handle.truncate(state.torn_offset)
                handle.flush()
                os.fsync(handle.fileno())
                state.parsed_offset = state.torn_offset
                state.torn_offset = None
                state.needs_separator = False
            pending: dict[str, dict[str, Any]] = {}
            for record in records:
                key = record["key"]
                existing = state.records.get(key, pending.get(key))
                if existing is not None:
                    if not _current_records_equivalent(existing, record):
                        raise CacheCorruptionError(
                            f"{self.path}: divergent reply for existing key {key}"
                        )
                    continue
                pending[key] = record
            handle.seek(0, os.SEEK_END)
            if state.needs_separator:
                handle.write(b"\n")
            for record in pending.values():
                handle.write((_canonical_json(record) + "\n").encode("utf-8"))
            appended = len(pending)
            if pending or state.needs_separator:
                handle.flush()
                os.fsync(handle.fileno())
            # The live fast-path state can share these dictionaries. Install
            # new entries only after their complete lines are durable, so an I/O
            # failure cannot make this process replay a reply absent from disk.
            state.records.update(pending)
            state.line_count += appended
            state.needs_separator = False
            final_stamp = _FileStamp.from_stat(os.fstat(handle.fileno()))
            state.parsed_offset = final_stamp.size
            self._verify_named_handle(handle)
            self._install_state(state, final_stamp)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        self.stats.stored += appended
        return appended

    def get(self, request: CacheRequest | None) -> dict[str, Any] | None:
        """Return an exact reply from the validated current-schema store."""
        if not self.enabled or request is None:
            return None
        with self._lock:
            self._refresh_locked(require_full_validation=True)
            record = self._records.get(request.key)
            if record is None:
                return None
            self.stats.hits += 1
            self._accessed_keys.add(request.key)
            value = _json_snapshot(record["value"])
            if not isinstance(value, dict):  # validated stores only carry object replies
                raise CacheCorruptionError(f"validated reply for {request.key} is not an object")
            return value

    def put(self, request: CacheRequest | None, value: dict[str, Any]) -> None:
        """Durably append a reply; unidentified models are never cached."""
        if request is None:
            self.stats.unidentified += 1
            return
        if not self.enabled:
            return
        record = _record_for(request, value)
        self._append_records([record])
        with self._lock:
            self._accessed_keys.add(request.key)

    def __len__(self) -> int:
        with self._lock:
            self._refresh_locked(require_full_validation=True)
            return len(self._records)

    def receipt(self) -> dict[str, Any]:
        """Full cache receipt (schema, path, records, counts) after revalidation."""
        with self._lock:
            self._refresh_locked(require_full_validation=True)
            return {
                "schema": CACHE_SCHEMA,
                "path": str(self.path),
                "enabled": self.enabled,
                "records": len(self._records),
                "records_sha256": self._records_digest_locked(),
                **self.stats.as_dict(),
                "fresh_fraction": self.stats.fresh_fraction,
            }
