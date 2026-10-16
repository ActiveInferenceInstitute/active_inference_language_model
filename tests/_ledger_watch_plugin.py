"""Attribute every rewrite of the real ledger to the test that caused it.

Watches mtime and content hash around each test, which catches an in-process
write and a subprocess write alike — the in-process spy tried earlier could only
see the former, which is why it came back clean.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

LEDGER = pathlib.Path(__file__).resolve().parents[1] / "data" / "claims.json"
FINDINGS: list[tuple[str, str, bool]] = []


def _state() -> tuple[float, str]:
    stat = LEDGER.stat()
    digest = hashlib.sha256(LEDGER.read_bytes()).hexdigest()[:16]
    return stat.st_mtime, digest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    before = _state()
    yield
    after = _state()
    if after[0] != before[0]:
        FINDINGS.append((item.nodeid, f"{before[1]} -> {after[1]}", before[1] != after[1]))


def pytest_sessionfinish(session, exitstatus):
    print("\n\n=== ledger rewrites, attributed ===")
    if not FINDINGS:
        print("  none")
        return
    for nodeid, transition, changed in FINDINGS:
        flag = "CONTENT CHANGED" if changed else "same content"
        print(f"  {nodeid}\n      {transition}  [{flag}]")
