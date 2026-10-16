"""Guard: no ``zip(seq, seq[1:], strict=True)`` anywhere in the tree.

A self-offset zip pairs a sequence with its own tail, so its arguments differ
in length by exactly one *by construction*; ``strict=True`` on that idiom
raises ``ValueError`` on every non-empty input. The 2026-08-04 hardening pass
added ``strict=True`` mechanically (for ruff B905) to twelve such sites, which
passed every focused test and broke every full-corpus bigram path. The correct
idiom is ``itertools.pairwise``.

AST-based, not lexical: a match requires a ``zip`` call whose second argument
is ``<name>[1:]`` of the same ``<name>`` as the first argument, with a
``strict=True`` keyword.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src", "scripts", "tests")


def _is_self_offset_strict_zip(call: ast.Call) -> bool:
    if not (isinstance(call.func, ast.Name) and call.func.id == "zip"):
        return False
    if len(call.args) < 2 or not isinstance(call.args[0], ast.Name):
        return False
    second = call.args[1]
    if not (
        isinstance(second, ast.Subscript)
        and isinstance(second.value, ast.Name)
        and second.value.id == call.args[0].id
        and isinstance(second.slice, ast.Slice)
        and second.slice.upper is None
        and isinstance(second.slice.lower, ast.Constant)
        and second.slice.lower.value == 1
    ):
        return False
    return any(
        kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def _is_tail_subscript_of(node: ast.expr, name: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
        and isinstance(node.slice, ast.Slice)
        and node.slice.upper is None
        and isinstance(node.slice.lower, ast.Constant)
        and node.slice.lower.value == 1
    )


def _is_nested_offset_strict_zip(call: ast.Call) -> bool:
    """``zip(zip(A, B, ...), zip(A[1:], B[1:], ...), strict=True)``.

    The outer zip pairs an n-iterator with an (n-1)-iterator, so it is the
    same defect as the flat form; ``pairwise(zip(A, B, strict=True))`` is the
    correct spelling.
    """
    if not (isinstance(call.func, ast.Name) and call.func.id == "zip"):
        return False
    if not any(
        kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    ):
        return False
    if len(call.args) != 2:
        return False
    first, second = call.args
    if not all(
        isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "zip"
        for arg in (first, second)
    ):
        return False
    assert isinstance(first, ast.Call) and isinstance(second, ast.Call)
    if len(first.args) != len(second.args):
        return False
    names = [a.id for a in first.args if isinstance(a, ast.Name)]
    if len(names) != len(first.args):
        return False
    return all(
        _is_tail_subscript_of(tail_arg, name)
        for tail_arg, name in zip(second.args, names, strict=True)
    )


def _offending_sites(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (_is_self_offset_strict_zip(node) or _is_nested_offset_strict_zip(node))
    ]


def test_detector_flags_the_shipped_defect() -> None:
    """Positive control: the exact code that shipped must be flagged."""
    shipped = ast.parse("for a, b in zip(phones, phones[1:], strict=True):\n    pass\n")
    assert _offending_sites(shipped) == [1]
    fixed = ast.parse("for a, b in pairwise(phones):\n    pass\n")
    assert _offending_sites(fixed) == []
    nested = ast.parse(
        "for p, q in zip(zip(a, b, strict=True), zip(a[1:], b[1:], strict=True), strict=True):\n"
        "    pass\n"
    )
    assert _offending_sites(nested) == [1]
    nested_fixed = ast.parse("for p, q in pairwise(zip(a, b, strict=True)):\n    pass\n")
    assert _offending_sites(nested_fixed) == []


def test_no_self_offset_strict_zip_in_tree() -> None:
    scanned = 0
    offenders: list[str] = []
    for directory in SCAN_DIRS:
        for path in sorted((PROJECT_ROOT / directory).rglob("*.py")):
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.relative_to(PROJECT_ROOT)}:{lineno}" for lineno in _offending_sites(tree)
            )
    assert scanned > 50, f"scan set implausibly small ({scanned} files) — gate would be vacuous"
    assert offenders == [], (
        "self-offset zip with strict=True raises on every non-empty input; "
        f"use itertools.pairwise instead: {offenders}"
    )
