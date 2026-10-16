"""Deterministic offline dataset generation for the fitted-probe fitting sources.

Builds the declared letter/phoneme/G2P fitting instances with their exact
census counts, the positional diagnostic carve, and the experimental native
comparator's refit counts. All builders are pure functions of their declared
inputs; nothing here downloads, trains, or touches the network.

Contracts implemented (see ``docs/extension_protocols.md`` §2.2):

- **Streaming exact census** — one supervised target per declared position;
  the instance list preserves positional order so the diagnostic carve is
  well-defined. Any instance shuffle happens at the trainer under the
  declared seed, never here.
- **Diagnostic separation** — a contiguous tail block of nonoverlapping
  target positions behind a declared seam gap; gap positions belong to
  neither set. The diagnostic slice may carry a declared stride cap (a
  measurement subsample of the diagnostic slice only; the fit set is never
  thinned).
- **Experimental comparator refit** — the letter comparator used in
  diagnostic contrasts counts bigrams whose two positions both lie in the
  fit region, never the full-corpus counts.
- **Phoneme windows tied to the scored task** — the scored English phoneme
  task conditions on the whole prefix from the word's start
  (``corpora.py::phoneme_positions``), so the phoneme instances do too.
- **Strict 1:1 for G2P** — the declared sources contain only pairs whose
  letter count equals their phone count; a mismatch raises rather than skips.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Dict, List, Tuple

DATA_PROTOCOL = "fitted-probe-data-v1"


def letter_stream_from_raw(raw_text: str) -> str:
    """Normalize raw corpus text with the letter level's recorded normalizer.

    Lazy import: the letter level's module pulls the numerical stack, and this
    package must stay importable without it. The normalizer itself is the
    same recorded function that produced ``vendor/brown_counts.json``.
    """
    from ailm.levels.letter import normalize_text

    return normalize_text(raw_text)


def build_letter_instances(stream: str, *, window: int) -> List[Dict[str, object]]:
    """One supervised target per stream position, in positional order.

    Returns records ``{"position_index", "target", "context_window"}`` where
    ``position_index`` is the index of the target character in ``stream`` and
    ``context_window`` is the up-to-``window``-character prefix before it.
    The first target is the stream's second character (declared boundary).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if len(stream) < 2:
        raise ValueError(f"stream needs at least 2 characters, got {len(stream)}")
    return [
        {
            "position_index": i,
            "target": stream[i],
            "context_window": stream[max(0, i - window) : i],
        }
        for i in range(1, len(stream))
    ]


def letter_sft_record(context_window: str, target: str) -> Dict[str, object]:
    """The §2.2 SFT record: the exact scored-arm prompt, completions-style.

    ``prompt`` is byte-identical to ``build_prompt("letter", context_window)``
    — the fixed instruction every scored letter position receives — and
    ``completion`` is the stream's next character. Lazy import: the prompts
    module pulls the letter level's numerical stack, and this package must
    stay importable without it.
    """
    from ailm.benchmarks.prompts import build_prompt

    return {"prompt": build_prompt("letter", context_window), "completion": target}


@dataclass(frozen=True)
class FitDiagnosticSplit:
    """Positional fit/diagnostic separation over census target positions.

    Positions are 1-based target indices into the stream. ``fit_positions``
    and ``diagnostic_positions`` are disjoint ranges/lists; ``gap_positions``
    belongs to neither set and is recorded, never silently dropped.
    """

    fit_positions: Tuple[int, int]
    gap_positions: Tuple[int, int]
    diagnostic_positions: Tuple[int, ...]

    def __post_init__(self) -> None:
        fit_lo, fit_hi = self.fit_positions
        gap_lo, gap_hi = self.gap_positions
        if not (1 <= fit_lo <= fit_hi < gap_lo <= gap_hi):
            raise ValueError("fit and gap ranges must be ordered and disjoint")
        if not self.diagnostic_positions:
            raise ValueError("diagnostic_positions must be nonempty")
        if self.diagnostic_positions[0] <= gap_hi:
            raise ValueError("diagnostic positions must start after the gap")
        if len(set(self.diagnostic_positions)) != len(self.diagnostic_positions):
            raise ValueError("diagnostic positions must be unique")

    def counts(self) -> Dict[str, int]:
        fit_size = self.fit_positions[1] - self.fit_positions[0] + 1
        gap_size = self.gap_positions[1] - self.gap_positions[0] + 1
        return {
            "fit_positions": fit_size,
            "gap_skipped": gap_size,
            "diagnostic_positions": len(self.diagnostic_positions),
        }


def carve_diagnostic_tail(
    n_targets: int,
    *,
    window: int,
    gap: int,
    diagnostic_size: int,
    diagnostic_stride_cap: int | None = None,
) -> FitDiagnosticSplit:
    """Carve a contiguous diagnostic tail behind a declared seam gap.

    The fit region is positions ``1 .. n - gap - diagnostic_size``; the gap is
    skipped in full; the diagnostic is the tail. With ``diagnostic_stride_cap``
    the diagnostic keeps every ``cap``-th tail position (a declared measurement
    subsample of the diagnostic slice only — the fit region is never thinned).

    Seam invariant (protocol §2.2, fail-closed): ``gap >= window``. A gap
    smaller than the conditioning window would let every diagnostic instance's
    context reach back across the seam into fit-region characters, which is
    exactly the leak the declared seam exists to prevent. A narrower gap is a
    protocol-text change, never a silent pass.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if gap < window:
        raise ValueError(
            f"seam invariant refused: declared gap {gap} is below the conditioning "
            f"window {window}; the seam requires gap >= window so no diagnostic "
            "conditioning window reaches across it into the fit region. Relaxing "
            "this requires a protocol-text change, not a narrower gap."
        )
    if gap < 1:
        raise ValueError(f"gap must be >= 1, got {gap}")
    if diagnostic_size < 1:
        raise ValueError(f"diagnostic_size must be >= 1, got {diagnostic_size}")
    if n_targets < gap + diagnostic_size + 1:
        raise ValueError(
            f"n_targets {n_targets} too small for gap {gap} + diagnostic {diagnostic_size}"
        )
    fit_hi = n_targets - gap - diagnostic_size
    diagnostic_source = range(fit_hi + gap + 1, n_targets + 1)
    if diagnostic_stride_cap is not None:
        if diagnostic_stride_cap < 1:
            raise ValueError(f"diagnostic_stride_cap must be >= 1, got {diagnostic_stride_cap}")
        diagnostic_positions = tuple(diagnostic_source)[::diagnostic_stride_cap]
    else:
        diagnostic_positions = tuple(diagnostic_source)
    return FitDiagnosticSplit(
        fit_positions=(1, fit_hi),
        gap_positions=(fit_hi + 1, fit_hi + gap),
        diagnostic_positions=diagnostic_positions,
    )


def refit_letter_bigram_counts(
    stream: str, fit_positions: Tuple[int, int]
) -> Dict[Tuple[str, str], int]:
    """The experimental native comparator's exact fit material.

    Counts every bigram whose target position lies in the fit region and whose
    predecessor is the stream character before it — the same add-k consumption
    the shipped comparator applies (``build_letter_transition``), but fitted on
    exactly the fit positions rather than the full stream. For a contiguous fit
    range ``1..k`` this is the bigrams ending at positions ``1..k``: the first
    target's bigram starts at stream position ``0``, which is stream content,
    not a target, and stays inside the fit span.
    """
    lo, hi = fit_positions
    if lo < 1 or hi > len(stream) - 1:
        raise ValueError(f"fit range {fit_positions} outside targets 1..{len(stream) - 1}")
    counts: Dict[Tuple[str, str], int] = {}
    for i in range(lo, hi + 1):
        pair = (stream[i - 1], stream[i])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def g2p_train_test_pairs(
    pairs: Sequence[Tuple[str, str]], *, test_every: int
) -> tuple[list[Tuple[str, List[str]]], list[Tuple[str, List[str]]]]:
    """Split G2P pairs exactly as the shipped artifact defines the split.

    Reuses ``ailm.levels.phoneme.g2p_split`` unchanged: index ``i`` is TEST iff
    ``i % test_every == 0``. The pairs are ``(word, space-joined phones)``
    strings exactly as stored in the vendored artifact.
    """
    from ailm.levels.phoneme import g2p_split

    counts = {"g2p_pairs": list(pairs), "test_every": test_every}
    return g2p_split(counts)


def build_g2p_instances(
    train_pairs: Sequence[Tuple[str, List[str]]],
) -> List[Dict[str, object]]:
    """One SFT instance per train pair, strict 1:1, in artifact order."""
    instances: List[Dict[str, object]] = []
    for index, (word, phones) in enumerate(train_pairs):
        if len(word) != len(phones):
            raise ValueError(
                f"pair {index} ({word!r}) is not 1:1: {len(word)} letters vs "
                f"{len(phones)} phones; the declared sources contain only 1:1 pairs"
            )
        instances.append({"pair_index": index, "word": word, "gold": list(phones)})
    return instances


def build_phoneme_stream_instances(
    train_pairs: Sequence[Tuple[str, List[str]]],
) -> List[Dict[str, object]]:
    """Next-phone instances over the train pronunciations, per-pair reset.

    Mirrors the scored phoneme task exactly: the prefix is the whole phone
    sequence from the word's start, space-joined (``phones[:i]``), and the
    target is ``phones[i]`` — the same construction
    ``corpora.py::phoneme_positions`` scores.
    """
    instances: List[Dict[str, object]] = []
    for pair_index, (_word, phones) in enumerate(train_pairs):
        for position in range(1, len(phones)):
            instances.append(
                {
                    "pair_index": pair_index,
                    "position": position,
                    "prefix": " ".join(phones[:position]),
                    "target": phones[position],
                }
            )
    return instances


def count_contexts(instances: Sequence[Dict[str, object]], *, key: str) -> Counter[str]:
    """Per-occurrence context accounting over one instance list."""
    return Counter(str(instance[key]) for instance in instances)


def reconcile_contexts(instances: Sequence[Dict[str, object]], *, key: str) -> Dict[str, int]:
    """Return exact totals, asserting context occurrences equal target count."""
    occurrences = count_contexts(instances, key=key)
    total = sum(occurrences.values())
    if total != len(instances):
        raise AssertionError(
            f"context occurrences {total} != instances {len(instances)} under key {key!r}"
        )
    return {
        "supervised_targets": len(instances),
        "distinct_contexts": len(occurrences),
        "context_occurrences_total": total,
    }


def synthetic_letter_stream(seed: int, n: int) -> str:
    """A deterministic synthetic normalized-letter stream for fixtures.

    Not English, not a corpus draw: a seeded alternation over a small symbol
    set, for boundary and exact-count tests only. Never a fit source for a
    recorded run.
    """
    import random

    rng = random.Random(seed)
    return "".join(rng.choice("ab c") for _ in range(n))


def canonicalize_pairs(counts: Dict[Tuple[str, str], int]) -> Dict[str, int]:
    """Serialize bigram counts with the vendored artifacts' ``"a,b"`` keys."""
    return {f"{previous},{current}": count for (previous, current), count in sorted(counts.items())}


def normalized_stream_from_file(path: str, *, raw: bool) -> Tuple[str, Callable[[str], str]]:
    """Read a stream file, optionally normalizing it with the recorded normalizer.

    Returns the stream and the normalizer actually applied (identity when the
    file is already normalized), so the receipt can name the exact transform.
    """
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if raw:
        return letter_stream_from_raw(text), letter_stream_from_raw
    return text, lambda value: value


def iter_jsonl(records: Iterator[Dict[str, object]]):
    """Serialize records to JSONL lines (compact, sorted keys)."""
    import json

    for record in records:
        yield json.dumps(record, sort_keys=True, separators=(",", ":"))
