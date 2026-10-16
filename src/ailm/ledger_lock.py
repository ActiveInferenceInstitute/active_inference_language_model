"""Single-writer mutual exclusion for the claims ledger.

``ClaimsLedger`` reads the whole ledger at construction and writes the whole
ledger at the end, so two overlapping recorder runs are a lost update: the second
to finish silently discards everything the first wrote, with no error anywhere.
The ledger has always assumed a single writer and the assumption was enforced
only by convention, which does not survive concurrent sessions.

The lock lives here rather than inside one recorder so every writer can share one
implementation instead of each growing its own subtly different copy.
"""

from __future__ import annotations

import contextlib
import errno
import os
import time
from collections.abc import Iterator
from pathlib import Path

__all__ = ["LOCK_FILENAME", "holder_is_alive", "ledger_lock_path", "single_writer_lock"]

#: Lock file name, kept beside the ledger it guards so a checkout carries its own.
LOCK_FILENAME = ".claims-recorder.lock"

#: How many times to re-read a lock whose contents cannot be parsed before
#: concluding it is genuinely corrupt rather than mid-write. Only reachable for a
#: lock left by an older implementation; the current one is never observably empty.
_UNREADABLE_RETRIES = 3

#: Seconds between those re-reads. Without a delay all three land inside the same
#: microsecond, which cannot outlast the mid-write the retry exists to wait out —
#: the loop would report a corrupt lock for a lock that was merely being written.
_UNREADABLE_RETRY_DELAY_S = 0.05


def ledger_lock_path(project_root: Path) -> Path:
    """Path of the lock guarding the ledger this process would actually write.

    Keyed to the resolved ledger, not to the project root. ``AILM_CLAIMS_PATH``
    redirects recorders at an isolated ledger — the replay test and the synthetic
    recorder's usage block both do it — and a lock derived from the project root
    would put every one of those runs in contention over a lock file in the
    tracked tree, guarding a ledger none of them touches.

    The override resolution mirrors :func:`ailm.claims.load_ledger`, which is the
    authority; a non-absolute override is ignored here because that function
    rejects it moments later with a better message.
    """
    override = os.environ.get("AILM_CLAIMS_PATH", "").strip()
    if override and Path(override).is_absolute():
        return Path(override).parent / LOCK_FILENAME
    return Path(project_root) / "data" / LOCK_FILENAME


def holder_is_alive(pid: int) -> bool:
    """Whether ``pid`` still exists, so a crashed run cannot block forever.

    Signal 0 performs the permission and existence checks without delivering
    anything. ``EPERM`` means the process exists under another user, which still
    counts as alive; only ``ESRCH`` means it is gone.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def _read_holder(path: Path) -> int | None:
    """The pid recorded in ``path``, or ``None`` if it cannot be read."""
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _claim(path: Path) -> bool:
    """Atomically create ``path`` already containing our pid.

    Creating the file and then writing the pid is two steps, and between them the
    lock is observably empty. A second process reading it then finds no pid,
    concludes the lock is stale, removes it and takes its own — so both processes
    hold the lock and the mutual exclusion this file exists to provide is gone.

    ``os.link`` makes it one step: the file appears with its contents already in
    place, and the link fails if the name is taken. That is the property the
    create-then-write sequence lacked.
    """
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(scratch, "w", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
        try:
            os.link(scratch, path)
        except FileExistsError:
            return False
        return True
    finally:
        with contextlib.suppress(OSError):
            scratch.unlink()


@contextlib.contextmanager
def single_writer_lock(path: Path) -> Iterator[None]:
    """Hold the ledger's single-writer lock, or refuse to start.

    Self-healing: a lock whose recorded holder no longer exists is reclaimed, so
    a crashed or killed run cannot wedge the repository. A lock that turned a
    race into a permanent block would be the worse failure of the two.

    Release is ownership-checked. Unlinking unconditionally would let a process
    that had already lost its lock delete the lock of whoever holds it now,
    cascading one stolen lock into a third concurrent writer.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    unreadable_seen = 0
    while not _claim(path):
        holder = _read_holder(path)
        if holder is None:
            # Mid-write under an older implementation, or a truncated file.
            # Treat as held until it has stayed unreadable across several looks:
            # assuming "unreadable means abandoned" is exactly how two writers
            # end up holding one lock.
            unreadable_seen += 1
            if unreadable_seen < _UNREADABLE_RETRIES:
                time.sleep(_UNREADABLE_RETRY_DELAY_S)
                continue
            raise SystemExit(
                f"recorder refused: {path} exists but records no readable pid. "
                f"Remove it by hand once you have confirmed no recorder is running."
            )
        if holder_is_alive(holder):
            raise SystemExit(
                f"recorder refused: another recorder run (pid {holder}) holds {path}. "
                f"The ledger is single-writer — two overlapping runs lose one of the "
                f"two results. Wait for it to finish, or remove the lock if you know "
                f"that process is gone."
            )
        # Stale: the recorded holder is gone. Reclaim only that exact lock.
        with contextlib.suppress(OSError):
            if _read_holder(path) == holder:
                path.unlink()
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            if _read_holder(path) == os.getpid():
                path.unlink()
