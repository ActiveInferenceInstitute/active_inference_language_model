#!/usr/bin/env python3
"""Capture the manuscript's dashboard figure: a real browser raster and its receipt.

The figure under ``data/dashboard_capture/`` is an authentic screenshot of the
local dashboard after one chat turn, bound by receipt to the raster's bytes and
to every asset the server handed the browser. Any change to the dashboard's
static tree makes the retained receipt refuse, so this tool records a new pair
instead of anyone editing the old one.

It runs under an interpreter that has Playwright and Pillow — a scratch
environment, never the project's locked ``.venv`` — and launches the dashboard
with the project's own interpreter, which is the one that serves the page in
use. The browser is the locally installed Chrome, driven headless, so no
browser download is involved::

    python3 -m venv /tmp/ailm-capture && /tmp/ailm-capture/bin/pip install playwright pillow
    /tmp/ailm-capture/bin/python -I scripts/capture_dashboard.py

Afterwards regenerate the figures so the framing PNG and provenance follow::

    MPLBACKEND=Agg .venv/bin/python scripts/generate_figures.py

The receipt records the interpreter-independent facts (served bytes, raster
digests, viewport, browser build). Rasters are not byte-reproducible across
browser builds and font stacks; the binding is to this capture, and the
procedure is what a reader can repeat.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ailm.dashboard.capture import (  # noqa: E402
    CAPTURE_DIRECTORY,
    SERVED_ROUTES,
    CaptureError,
    ServedAsset,
    build_receipt,
    git_blob,
    git_head,
    image_identity,
    served_asset,
    sha256_hex,
    write_capture,
)

PROCEDURE = "scripts/capture_dashboard.py"
DEFAULT_INPUT = "The ant followed the trail."
DEFAULT_PROJECT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--input", default=DEFAULT_INPUT, help="the one chat turn to send")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--project-python",
        type=Path,
        default=DEFAULT_PROJECT_PYTHON,
        help="interpreter that serves the dashboard and validates the result",
    )
    parser.add_argument("--viewport", default="900x1050", help="CSS pixels, WIDTHxHEIGHT")
    parser.add_argument("--scale", type=float, default=2.0, help="device scale factor")
    parser.add_argument(
        "--channel", default="chrome", help="Playwright browser channel (installed Chrome)"
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds for the reply")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / CAPTURE_DIRECTORY, help="receipt home"
    )
    return parser.parse_args(argv)


def parse_viewport(text: str) -> tuple[int, int]:
    """Parse ``WIDTHxHEIGHT`` in CSS pixels."""
    try:
        width, height = (int(part) for part in text.lower().split("x", 1))
    except ValueError as error:
        raise CaptureError(f"viewport must be WIDTHxHEIGHT, not {text!r}") from error
    return width, height


def wait_for_server(url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 loopback
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() > deadline:
            raise CaptureError(f"dashboard did not answer at {url} within {seconds:.0f}s")
        time.sleep(0.25)


def fetch_served_assets(base: str, source_commit: str) -> list[ServedAsset]:
    assets = []
    for route in SERVED_ROUTES:
        request = urllib.request.Request(base.rstrip("/") + route)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                status, body = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, body = error.code, b""
        assets.append(
            served_asset(
                REPO_ROOT,
                route,
                body,
                status,
                prior_blob=git_blob(REPO_ROOT, source_commit, SERVED_ROUTES[route]),
            )
        )
    return assets


def capture(args: argparse.Namespace, url: str) -> tuple[bytes, bytes, str, dict[str, object]]:
    """Drive one chat turn and return (initial png, final png, reply, browser facts)."""
    from playwright.sync_api import sync_playwright

    width, height = parse_viewport(args.viewport)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=args.scale,
                reduced_motion="reduce",
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            # Speech stays off: the capture claims nothing about speech devices,
            # and a headless page has no synthesizer to drain anyway.
            page.uncheck("#chat-speak-replies")
            initial = page.screenshot(type="png")
            page.fill("#chat-input", args.input)
            page.click("#chat-send-btn")
            page.wait_for_selector(".chat-turn.model .chat-bubble", timeout=args.timeout * 1000)
            page.wait_for_function(
                "() => document.getElementById('generated-echo').textContent.trim() !== '...'",
                timeout=args.timeout * 1000,
            )
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(500)
            reply = page.inner_text(".chat-turn.model .chat-bubble").strip()
            final = page.screenshot(type="png")
            facts: dict[str, object] = {
                "engine": "chromium",
                "channel": args.channel,
                "version": browser.version,
                "playwright": _playwright_version(),
                "headless": True,
                "device_scale_factor": args.scale,
                "reduced_motion": "reduce",
            }
        finally:
            browser.close()
    return initial, final, reply, facts


def _playwright_version() -> str:
    from importlib.metadata import version

    return version("playwright")


def decode(png: bytes) -> tuple[bytes, int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(png)) as image:
        rgba = image.convert("RGBA")
        return rgba.tobytes(), rgba.width, rgba.height


def validate_with_project(python: Path) -> None:
    """Have the figure generator's own validator accept the new pair."""
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "from ailm.figures.dashboard_observation import dashboard_capture_pixels as f; "
            "p = f(); print(f'validated {p.shape[1]}x{p.shape[0]} capture')",
        ],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise CaptureError("the project's validator refused the capture it just received")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.project_python.is_file():
        print(f"project interpreter missing: {args.project_python}", file=sys.stderr)
        return 2
    url = f"http://{args.host}:{args.port}/"
    source_commit = git_head(REPO_ROOT)
    server = subprocess.Popen(
        [str(args.project_python), "-m", "ailm.dashboard", str(args.port), args.host],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(url, 60)
        assets = fetch_served_assets(url, source_commit)
        initial, final, reply, browser = capture(args, url)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
    rgba, width, height = decode(final)
    receipt = build_receipt(
        captured_at=dt.datetime.now(dt.UTC).isoformat(),
        source_commit=source_commit,
        url=url,
        input_text=args.input,
        reply=reply,
        viewport=parse_viewport(args.viewport),
        image=image_identity(final, rgba, width, height),
        served_assets=assets,
        initial_sha256=sha256_hex(initial),
        responsive_sha256=sha256_hex(final),
        browser=browser,
        procedure=PROCEDURE,
    )
    write_capture(args.output_dir, final, receipt)
    print(f"wrote {args.output_dir / 'analysis.png'} ({width}x{height}) and receipt.json")
    print(f"reply: {reply}")
    unchanged = sum(asset.exact_prior_main_blob for asset in assets)
    print(f"served assets: {len(assets)}, {unchanged} identical to {source_commit[:12]}")
    if args.output_dir == REPO_ROOT / CAPTURE_DIRECTORY:
        validate_with_project(args.project_python)
    return 0


if __name__ == "__main__":
    sys.exit(main())
