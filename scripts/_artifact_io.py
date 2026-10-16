"""Atomic serialization shared by deterministic artifact generators."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, NamedTuple


class AuthenticatedFileError(ValueError):
    """A provenance input could not be authenticated as ordinary file bytes."""


class AuthenticatedBytes(NamedTuple):
    """Bytes and digest derived from one authenticated open-file description."""

    content: bytes
    sha256: str


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate member names."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    """Reject the JavaScript NaN/Infinity extensions accepted by ``json``."""
    raise ValueError(f"non-finite JSON constant {value}")


@contextmanager
def open_authenticated_regular_file(path: Path, *, description: str) -> Iterator[BinaryIO]:
    """Open one ordinary file without following its final symlink component.

    ``O_NONBLOCK`` is security-significant here. Opening a FIFO read-only can
    otherwise wait forever for a writer *before* ``fstat`` gets a chance to
    reject it. The type decision and every later read use the same descriptor,
    so replacing ``path`` after the open cannot substitute different bytes.
    """
    required = ("O_NOFOLLOW", "O_NONBLOCK")
    missing = [name for name in required if not hasattr(os, name)]
    if missing:
        raise AuthenticatedFileError("authenticated file reads require " + ", ".join(missing))
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise AuthenticatedFileError(
            f"{path}: cannot open {description} as a regular non-symlink file: {error}"
        ) from error

    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise AuthenticatedFileError(
                f"{path}: cannot inspect {description}: {error}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise AuthenticatedFileError(f"{path}: {description} must be a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def read_authenticated_regular_file(
    path: Path,
    *,
    description: str,
    max_bytes: int | None = None,
) -> AuthenticatedBytes:
    """Read and hash bytes from one descriptor-authenticated ordinary file."""
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative or None")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        with open_authenticated_regular_file(path, description=description) as handle:
            while chunk := handle.read(1024 * 1024):
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise AuthenticatedFileError(
                        f"{path}: {description} exceeds the {max_bytes}-byte limit"
                    )
                digest.update(chunk)
                chunks.append(chunk)
    except OSError as error:
        raise AuthenticatedFileError(f"{path}: cannot read {description}: {error}") from error
    return AuthenticatedBytes(b"".join(chunks), digest.hexdigest())


def sha256_authenticated_regular_file(path: Path, *, description: str) -> str:
    """Hash one descriptor-authenticated ordinary file without buffering it."""
    digest = hashlib.sha256()
    try:
        with open_authenticated_regular_file(path, description=description) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AuthenticatedFileError(f"{path}: cannot hash {description}: {error}") from error
    return digest.hexdigest()


def read_strict_json_regular_file(
    path: Path,
    *,
    description: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> tuple[object, str]:
    """Decode strict JSON from the exact regular-file bytes that are hashed."""
    authenticated = read_authenticated_regular_file(
        path,
        description=description,
        max_bytes=max_bytes,
    )
    try:
        text = authenticated.content.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AuthenticatedFileError(f"{path}: {description} is unreadable: {error}") from error
    return payload, authenticated.sha256


def _publication_mode(parent_descriptor: int, name: str) -> int:
    """Preserve an existing regular file's mode, otherwise publish as ``0644``."""
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return 0o644
    return stat.S_IMODE(metadata.st_mode) if stat.S_ISREG(metadata.st_mode) else 0o644


def _directory_open_flags() -> int:
    """Return the flags required for descriptor-anchored directory traversal."""
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    missing = [name for name in required if not hasattr(os, name)]
    if missing:
        raise NotImplementedError("secure atomic publication requires " + ", ".join(missing))
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _symlink_refusal(path: Path, traversed: Path) -> NotADirectoryError:
    """The error for a symlinked component: an ``OSError`` that names the link.

    ``O_NOFOLLOW`` already refuses the link, but the kernel's refusal surfaces as
    a bare ``NotADirectoryError('tmp')`` that says neither which component nor
    what to do about it.  The link's name and the fix travel in the message.
    """
    return NotADirectoryError(
        errno.ENOTDIR,
        f"atomic publication refuses symlinked path component {traversed.name!r} "
        f"({traversed}) while publishing {path}: pass os.path.realpath() of the "
        f"destination (macOS /tmp and /var are symlinks)",
    )


def _is_symlink(name: str, parent_descriptor: int) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode)


def _refuse_symlinked_component(path: Path, traversed: Path, parent_descriptor: int) -> None:
    """Raise the named refusal before any open or ``mkdir`` touches the component."""
    if _is_symlink(traversed.name, parent_descriptor):
        raise _symlink_refusal(path, traversed)


def _open_directory_component(path: Path, traversed: Path, parent_descriptor: int) -> int:
    """Open one component without following, naming a link the kernel refused.

    A component that appeared between the pre-check and this open still trips
    ``O_NOFOLLOW``; that refusal (``ENOTDIR`` on macOS, ``ELOOP`` on Linux) is
    re-raised with the same named message.  A regular file in the way is
    reported as what it is rather than as a link.
    """
    try:
        return os.open(traversed.name, _directory_open_flags(), dir_fd=parent_descriptor)
    except OSError as error:
        if error.errno not in (errno.ENOTDIR, errno.ELOOP):
            raise
        if _is_symlink(traversed.name, parent_descriptor):
            raise _symlink_refusal(path, traversed) from error
        raise NotADirectoryError(
            errno.ENOTDIR,
            f"atomic publication path component {traversed.name!r} ({traversed}) of "
            f"{path} is not a directory",
        ) from error


def _open_or_create_directory(path: Path) -> int:
    """Open ``path`` without a pathname gap, creating missing components.

    Traversal starts from an already-open filesystem root (absolute paths) or
    current directory (relative paths).  Every lookup and ``mkdir`` is then
    relative to the preceding directory descriptor.  An attacker can rename
    an opened ancestor, but cannot redirect subsequent creation into a
    replacement pathname or symlink.

    The returned descriptor belongs to the caller.
    """
    expanded = path.expanduser()
    parts = expanded.parts
    if expanded.is_absolute():
        descriptor = os.open(expanded.anchor, _directory_open_flags())
        parts = parts[1:]
        traversed = Path(expanded.anchor)
    else:
        descriptor = os.open(".", _directory_open_flags())
        traversed = Path()

    try:
        for component in parts:
            if component in ("", "."):
                continue
            if component == "..":
                raise ValueError("atomic publication paths must not contain parent traversal")
            traversed = traversed / component
            _refuse_symlinked_component(expanded, traversed, descriptor)
            try:
                child_descriptor = _open_directory_component(expanded, traversed, descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    # A concurrent creator won.  The non-following open below
                    # still verifies that it created a real directory.
                    pass
                child_descriptor = _open_directory_component(expanded, traversed, descriptor)
            os.close(descriptor)
            descriptor = child_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_parent(path: Path) -> int:
    """Create and open the publication directory through anchored descriptors."""
    if not path.name:
        raise ValueError("atomic publication requires a file name")
    return _open_or_create_directory(path.parent)


def _open_unique_scratch(parent_descriptor: int, target_name: str) -> tuple[int, str]:
    """Reserve a non-following scratch name relative to an anchored directory."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _attempt in range(128):
        name = f".{target_name}.{secrets.token_hex(16)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=parent_descriptor), name
        except FileExistsError:
            continue
    raise FileExistsError(f"could not reserve a unique scratch beside {target_name!r}")


def atomic_write_bytes(path: Path, content: bytes) -> bytes:
    """Fsync and atomically publish bytes relative to one anchored parent FD.

    The parent is opened once with ``O_DIRECTORY|O_NOFOLLOW``. Scratch creation,
    replacement, cleanup, and directory synchronization are all relative to that
    descriptor, so swapping a pathname component after the open cannot redirect
    publication or substitute a same-named scratch file.
    """
    if not isinstance(content, bytes):
        raise TypeError("atomic_write_bytes content must be bytes")
    parent_descriptor = _open_parent(path)
    descriptor = -1
    scratch_name: str | None = None
    try:
        descriptor, scratch_name = _open_unique_scratch(parent_descriptor, path.name)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            os.fchmod(handle.fileno(), _publication_mode(parent_descriptor, path.name))
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            scratch_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        scratch_name = None
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if scratch_name is not None:
            try:
                os.unlink(scratch_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)
    return content


def atomic_write_text(path: Path, content: str) -> bytes:
    """Encode UTF-8 text once and publish it through :func:`atomic_write_bytes`."""
    if not isinstance(content, str):
        raise TypeError("atomic_write_text content must be str")
    return atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
) -> bytes:
    """Serialize once, fsync a unique sibling temporary, then publish atomically."""
    content = (
        json.dumps(
            payload,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
            indent=indent,
            separators=separators,
        )
        + "\n"
    ).encode("utf-8")
    return atomic_write_bytes(path, content)
