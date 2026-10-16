"""Offline newcomer demo: two things the shipped code already does, made visible.

Reachable as ``python -m ailm`` or the ``ailm-demo`` console script. Runs two
read-only demonstrations, both against data already committed to the
repository — no network access, no re-measurement, no ledger writes:

1. **The gate-verdict table**, read straight from the committed
   ``output/reports/gate_audit.json`` (produced by ``scripts/audit_gates.py``
   from :mod:`ailm.gates` and the claims ledger). This module does not
   recompute a single verdict; it only loads and prints the report, plus one
   spot cross-check of a baseline value against the raw claim in
   ``data/claims.json`` via :func:`ailm.claims.load_ledger`.
2. **A live Stage 2B G2P decode.** One word is run letter-by-letter through a
   freshly built ``pymdp`` agent from :mod:`ailm.levels.phoneme` (the same
   ``build_g2p_*`` functions :func:`ailm.levels.phoneme.decode_phonemes`
   uses), printing the full posterior over the top phone candidates at every
   position instead of only the argmax. This *is* real inference — the first
   call in a fresh environment pays JAX's one-time import/compilation cost and
   can take a minute or two with no console output in between; it is fast on
   every call after that.

No number in this module is typed as a literal: gate values come from the
committed report, phone/letter symbols and probabilities come from
:mod:`ailm.levels.phoneme` and :mod:`ailm.levels.letter` at run time.

This demo needs a source checkout (``output/`` and ``data/`` are intentionally
not embedded in the installed wheel — the same policy
:mod:`ailm.resources` documents for raw corpus downloads). Run it with the
project's own interpreter from the repository root::

    .venv/bin/python -m ailm
    .venv/bin/python -m ailm science --top-k 5
    ailm-demo science
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

#: Where the committed gate-audit report lives, relative to the project root.
_GATE_AUDIT_RELATIVE_PATH = Path("output") / "reports" / "gate_audit.json"

#: Default word for the G2P decode when none is given on the command line.
_DEFAULT_WORD = "science"


def _find_project_root() -> Path | None:
    """Locate the repository root from this file, mirroring ``ailm.resources``.

    Returns ``None`` when this module is running from an installed
    distribution with no adjacent source checkout — ``output/`` and ``data/``
    are not wheel resources, so the gate-verdict table has nothing to read.
    """
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").is_file() and (
        candidate / "src" / "ailm" / "demo.py"
    ).is_file():
        return candidate
    return None


def load_gate_audit(project_root: Path | None = None) -> list[dict[str, Any]]:
    """Load the committed per-stage gate-verdict rows.

    Args:
        project_root: Repository root. ``None`` resolves it from this file's
            location.

    Returns:
        The list of gate rows exactly as ``scripts/audit_gates.py`` wrote
        them (see ``tests/test_gate_audit_export.py`` for the schema).

    Raises:
        FileNotFoundError: If no source checkout is available, or the report
            has not been generated in this checkout.
        ValueError: If the file does not contain a non-empty JSON list.
    """
    root = project_root if project_root is not None else _find_project_root()
    if root is None:
        raise FileNotFoundError(
            "the gate-verdict table reads output/reports/gate_audit.json, which "
            "ships in a source checkout but not in an installed wheel (the same "
            "policy ailm.resources documents for raw corpus downloads). Run this "
            "demo from a source checkout with `.venv/bin/python -m ailm`."
        )
    path = root / _GATE_AUDIT_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate it with "
            f"`.venv/bin/python scripts/audit_gates.py --full` — that recomputes "
            f"every stage from data/claims.json and the shipped models, which is "
            f"why this fast demo reads the already-committed report instead of "
            f"calling it directly."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} did not contain a non-empty JSON list of gate rows")
    return rows


def _format_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "" if value is None else str(value)


def format_gate_table(rows: list[dict[str, Any]]) -> str:
    """Render gate rows as a fixed-width text table.

    Args:
        rows: Rows from :func:`load_gate_audit`.

    Returns:
        A multi-line string, one header, one rule, one row per gate.
    """
    columns: tuple[tuple[str, str], ...] = (
        ("gate_id", "gate"),
        ("stage", "stage"),
        ("model", "model"),
        ("baseline", "baseline"),
        ("ceiling", "ceiling"),
        ("ceiling_type", "ceiling type"),
        ("verdict", "verdict"),
        ("outcome", "outcome"),
    )
    rendered = [{header: _format_number(row.get(key)) for key, header in columns} for row in rows]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rendered))
        for _key, header in columns
    }
    header_line = "  ".join(header.ljust(widths[header]) for _key, header in columns)
    rule = "  ".join("-" * widths[header] for _key, header in columns)
    body = [
        "  ".join(row[header].ljust(widths[header]) for _key, header in columns) for row in rendered
    ]
    return "\n".join([header_line, rule, *body])


def _cross_check_against_ledger(rows: list[dict[str, Any]], root: Path) -> str | None:
    """Spot-check one gate-audit baseline against its raw ``data/claims.json`` entry.

    This is the one place the demo reaches into :mod:`ailm.claims` directly,
    to show the printed table traces back to the ledger rather than being a
    second, disconnected number. It reports agreement; it never recomputes.
    """
    try:
        from ailm.claims import load_ledger
    except ImportError:
        return None

    row = next((r for r in rows if r.get("gate_id") == "stage1_letter"), None)
    if row is None or not isinstance(row.get("baseline"), int | float):
        return None

    ledger = load_ledger(root)
    entry = ledger.get("stage1_unigram_baseline")
    if entry is None or not isinstance(entry.get("value"), int | float):
        return None

    audit_value = float(row["baseline"])
    ledger_value = float(entry["value"])
    agrees = math.isclose(audit_value, ledger_value, rel_tol=1e-3)
    return (
        f"cross-check: stage1_letter baseline {audit_value:.4f} from "
        f"output/reports/gate_audit.json vs stage1_unigram_baseline "
        f"{ledger_value:.4f} from data/claims.json "
        f"({'agree' if agrees else 'DISAGREE — investigate before trusting either'})"
    )


def decode_word_with_posteriors(
    word: str,
    *,
    vendor_dir: Path | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Decode a word's letters to phones, keeping the full posterior per position.

    Builds a real Stage 2B G2P ``pymdp`` agent from the vendored CMUdict
    artifact — the identical construction
    :func:`ailm.levels.phoneme.decode_phonemes` uses — but records the top
    ``k`` phone probabilities at each filtering step instead of only the
    argmax phone.

    Args:
        word: Any word; normalized and filtered to ALPHABET symbols exactly as
            :func:`ailm.levels.phoneme.decode_phonemes` does. Need not be a
            dictionary word — the decoder generalizes from letter/phone
            bigram statistics, it is not a lookup table.
        vendor_dir: Passed through to
            :func:`ailm.levels.phoneme.load_phoneme_counts`. ``None`` resolves
            the shipped vendor artifact (wheel resource or source checkout).
        top_k: Phone candidates to keep per position. Keyword-only.

    Returns:
        One dict per retained letter: ``position``, ``letter``, and
        ``top_phones`` (a list of ``(phone, probability)`` pairs, most likely
        first).

    Raises:
        ValueError: If ``word`` has no ALPHABET-recognized letters, or
            ``top_k`` is not a positive integer.
        FileNotFoundError: If the vendored CMUdict artifact is unavailable.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

    # ailm.levels.letter is cheap; ailm.levels.phoneme pulls in pymdp/JAX and
    # pays their cold-import cost (roughly a minute in a fresh environment —
    # see the module docstring). Validating the word first means a bad word
    # fails fast instead of paying that cost only to raise anyway.
    from ailm.levels.letter import ALPHA_TO_IDX, normalize_text

    letters = [c for c in normalize_text(word) if c in ALPHA_TO_IDX]
    if not letters:
        raise ValueError(f"{word!r} has no ALPHABET-recognized letters to decode")

    import jax.numpy as jnp

    from ailm.levels.phoneme import (
        IDX_TO_PHONE,
        PHONES,
        build_g2p_agent,
        build_g2p_emission,
        build_g2p_start,
        build_g2p_transition,
        g2p_split,
        load_phoneme_counts,
    )

    counts = load_phoneme_counts(vendor_dir)
    train_pairs, _test_pairs = g2p_split(counts)
    emission = build_g2p_emission(train_pairs)
    transition = build_g2p_transition(train_pairs)
    start = build_g2p_start(train_pairs)
    agent = build_g2p_agent(emission, transition, start)

    empirical_prior = [jnp.asarray(agent.D[0])]
    action = jnp.zeros((1, 1), dtype=jnp.int32)
    n_phones = len(PHONES)
    k = min(top_k, n_phones)
    positions: list[dict[str, Any]] = []

    for position, letter in enumerate(letters):
        obs_idx = ALPHA_TO_IDX[letter]
        qs = agent.infer_states([jnp.array([obs_idx])], empirical_prior)
        posterior = jnp.asarray(qs[0]).ravel()
        ranked = sorted(range(n_phones), key=lambda i: float(posterior[i]), reverse=True)[:k]
        top_phones = [(IDX_TO_PHONE[i], float(posterior[i])) for i in ranked]
        positions.append({"position": position, "letter": letter, "top_phones": top_phones})
        empirical_prior = agent.update_empirical_prior(action, qs)

    return positions


def format_decode_table(word: str, positions: list[dict[str, Any]]) -> str:
    """Render :func:`decode_word_with_posteriors` output as text."""
    lines = [
        f"Stage 2B G2P filtering decode of {word!r} "
        f"({len(positions)} letter position(s), top phone candidates per position):"
    ]
    for entry in positions:
        candidates = ", ".join(f"{phone} {prob:.3f}" for phone, prob in entry["top_phones"])
        lines.append(f"  [{entry['position']:>2}] {entry['letter']!r:>4} -> {candidates}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run both demonstrations and print their output.

    Args:
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 2 if either step fails (missing report, bad input).
    """
    parser = argparse.ArgumentParser(
        prog="ailm",
        description=(
            "Offline AILM newcomer demo: the committed gate-verdict table, then a "
            "live Stage 2B G2P decode of one word."
        ),
    )
    parser.add_argument(
        "word",
        nargs="?",
        default=_DEFAULT_WORD,
        help=f"word to decode letters -> phones (default: {_DEFAULT_WORD!r})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="phone candidates to print per position (default: 3)",
    )
    args = parser.parse_args(argv)

    root = _find_project_root()

    print("=== AILM gate-verdict table (output/reports/gate_audit.json) ===")
    try:
        rows = load_gate_audit(root)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(format_gate_table(rows))
    if root is not None:
        note = _cross_check_against_ledger(rows, root)
        if note is not None:
            print(note)
    print()

    print(f"=== Stage 2B G2P decode of {args.word!r} (live pymdp inference) ===")
    print("(first run in a fresh environment compiles JAX; expect up to a minute or")
    print(" two of silence, fast on every call after that)")
    try:
        positions = decode_word_with_posteriors(args.word, top_k=args.top_k)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(format_decode_table(args.word, positions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
