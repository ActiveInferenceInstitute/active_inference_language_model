"""The dashboard capture receipt: what binds a browser raster to the sources it shows.

The manuscript's dashboard figure is an authentic screenshot, not a drawing of
the interface. What makes it evidence rather than decoration is the receipt
beside it under ``data/dashboard_capture/``: the raster's byte and decoded-pixel
digests, and a digest of every asset the server handed the browser while the
capture was taken. :func:`ailm.figures.dashboard_observation.dashboard_capture_pixels`
refuses the raster when any of those bindings fails, which is why editing the
dashboard's static tree requires a new capture rather than an edited receipt.

This module owns the receipt's shape and the served-asset table, with the
standard library only, so the capture tool can assemble a receipt from the
environment that drives the browser while the figure generator validates it
from the project's own environment. Neither side keeps a private copy of the
other's rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

RECEIPT_SCHEMA: Final = "ailm-dashboard-capture-v1"
RECEIPT_NAME: Final = "receipt.json"
IMAGE_NAME: Final = "analysis.png"
CAPTURE_DIRECTORY: Final = Path("data") / "dashboard_capture"

SCOPE_STATEMENT: Final = (
    "Warm interactive demonstration; input analysis is separate from L3-to-L1g "
    "generated reply; speech disabled; no cold-start, speech-device, "
    "semantic-conversation or new scientific measurement claim."
)
TRANSFORM_STATEMENT: Final = (
    "Native screenshot scaling only; no fabricated UI, crop or image painting"
)

_STATIC: Final = "src/ailm/dashboard/static/"
_SCRIPT_MODULES: Final = (
    "matrices",
    "config",
    "speech",
    "main",
    "stt",
    "state",
    "status",
    "chat",
    "inference",
)

SERVED_ROUTES: Final[Mapping[str, str]] = {
    "/": _STATIC + "index.html",
    "/dashboard.css": _STATIC + "dashboard.css",
    "/explorer/": "output/web/ailm.html",
    **{f"/js/{name}.js": f"{_STATIC}js/{name}.js" for name in _SCRIPT_MODULES},
}
"""Every route the page loads, and the repository file the server serves at it."""

CAPTURE_SOURCE_PATHS: Final[frozenset[str]] = frozenset(SERVED_ROUTES.values())
"""
The exact surface authenticated by the capture receipt; changing that surface
requires recapture rather than accepting a shortened or enlarged receipt.
"""


class CaptureError(ValueError):
    """A receipt could not be assembled from what the server and browser produced."""


def sha256_hex(content: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``content``."""
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class ServedAsset:
    """One asset as the browser received it, bound to the file the server read."""

    route: str
    source: str
    bytes: int
    sha256: str
    http_status: int
    exact_prior_main_blob: bool


@dataclass(frozen=True)
class ImageIdentity:
    """The raster's file and decoded-pixel identities."""

    path: str
    sha256: str
    rgba_sha256: str
    width: int
    height: int


def served_asset(
    root: Path,
    route: str,
    body: bytes,
    http_status: int,
    *,
    prior_blob: bytes | None,
) -> ServedAsset:
    """Bind one served response to its source file.

    The body the browser received must be byte-identical to the file on disk:
    the receipt names the file, and the figure generator verifies the file, so
    a server that transformed the bytes would leave the receipt describing
    something the capture did not show.

    Args:
        root: Repository root.
        route: The request path, which must be in :data:`SERVED_ROUTES`.
        body: The response body as received.
        http_status: The response status; anything but 200 is refused.
        prior_blob: The same path's content at the receipt's source commit, or
            None when that commit does not track the path.
    """
    try:
        source = SERVED_ROUTES[route]
    except KeyError as error:
        raise CaptureError(f"route {route!r} is not part of the captured surface") from error
    if http_status != 200:
        raise CaptureError(f"{route} answered {http_status}, not 200")
    on_disk = (root / source).read_bytes()
    if body != on_disk:
        raise CaptureError(f"{route} served bytes that differ from {source}")
    return ServedAsset(
        route=route,
        source=source,
        bytes=len(body),
        sha256=sha256_hex(body),
        http_status=http_status,
        exact_prior_main_blob=prior_blob is not None and prior_blob == body,
    )


def image_identity(png: bytes, rgba: bytes, width: int, height: int) -> ImageIdentity:
    """Record the raster's file digest and its decoded RGBA digest.

    ``rgba`` is the row-major, 8-bit RGBA pixel buffer of the decoded image, the
    same bytes :mod:`numpy` yields for ``np.array(image.convert("RGBA"))``, so a
    re-encoded PNG with identical pixels keeps the same ``rgba_sha256`` while a
    repainted one does not.
    """
    if len(rgba) != width * height * 4:
        raise CaptureError(f"decoded pixel buffer has {len(rgba)} bytes, not {width}x{height}x4")
    return ImageIdentity(
        path=IMAGE_NAME,
        sha256=sha256_hex(png),
        rgba_sha256=sha256_hex(rgba),
        width=width,
        height=height,
    )


def git_blob(root: Path, commit: str, path: str) -> bytes | None:
    """Return ``path``'s content at ``commit``, or None when the commit lacks it."""
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{path}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def git_head(root: Path) -> str:
    """Return the checkout's HEAD commit."""
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "-q", "HEAD"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CaptureError(f"{root} has no HEAD commit to bind the capture to")
    return completed.stdout.strip()


def build_receipt(
    *,
    captured_at: str,
    source_commit: str,
    url: str,
    input_text: str,
    reply: str,
    viewport: tuple[int, int],
    image: ImageIdentity,
    served_assets: Sequence[ServedAsset],
    initial_sha256: str,
    responsive_sha256: str,
    browser: Mapping[str, Any],
    procedure: str,
) -> dict[str, Any]:
    """Assemble the receipt the figure generator validates.

    Every route in :data:`SERVED_ROUTES` must appear exactly once: a receipt
    that lists fewer assets would leave part of the shown page unbound, and one
    that lists more would claim a binding the figure generator cannot check.
    """
    sources = [asset.source for asset in served_assets]
    if sorted(sources) != sorted(CAPTURE_SOURCE_PATHS):
        missing = sorted(CAPTURE_SOURCE_PATHS - set(sources))
        extra = sorted(set(sources) - CAPTURE_SOURCE_PATHS)
        duplicate = sorted({name for name in sources if sources.count(name) > 1})
        raise CaptureError(
            f"served assets do not cover the captured surface exactly: "
            f"missing={missing} extra={extra} duplicate={duplicate}"
        )
    width, height = viewport
    if width <= 0 or height <= 0:
        raise CaptureError(f"viewport must be positive, not {width}x{height}")
    return {
        "schema": RECEIPT_SCHEMA,
        "captured_at": captured_at,
        "source_commit": source_commit,
        "url": url,
        "input": input_text,
        "reply": reply,
        "scope": SCOPE_STATEMENT,
        "viewport": {"width": width, "height": height},
        "image": asdict(image),
        "served_assets": [asdict(asset) for asset in served_assets],
        "capture_receipts": {
            "initial_sha256": initial_sha256,
            "responsive_sha256": responsive_sha256,
        },
        "transform": TRANSFORM_STATEMENT,
        "procedure": procedure,
        "browser": dict(browser),
    }


def write_capture(directory: Path, png: bytes, receipt: Mapping[str, Any]) -> None:
    """Publish the raster and receipt together, each through an atomic replace.

    The raster lands first so a reader that sees the new receipt never sees the
    old image beside it; an interrupted write leaves the previous complete pair
    or a new raster with the previous receipt, which the validator refuses.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _replace(directory / IMAGE_NAME, png)
    _replace(
        directory / RECEIPT_NAME,
        (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def _replace(target: Path, content: bytes) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, target)
