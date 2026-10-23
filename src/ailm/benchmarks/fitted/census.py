"""Streaming exact-census counting for the fitted-probe fitting objective.

The fitted-probe protocol keeps a full-census fitting objective: one
supervised target per declared stream position at stride one, computed by a
streaming pass so the material is never materialized twice or subsampled.
Two exposure quantities are tracked separately and both are exact:

- **supervised targets** — one per census position; equals the comparator's
  count-fit exposure at stride one;
- **conditioning contexts** — the window each target conditions on; repeated
  contexts each count once per occurrence (per-occurrence accounting), and
  the occurrence multiset always reconciles to the target total.

This module also carries the checkable refutation of stride-d subsampling:
:strfunc:`stride_aliasing_demo` builds the alternating-stream counterexample
showing that a strided "sample" does not preserve bigram relative
frequencies. Striding is sampling; the census stands.
"""

from __future__ import annotations

from collections import Counter


class StreamCensus:
    """Streaming exact census over one character stream.

    Feed chunks with :meth:`feed`; the census crosses chunk boundaries
    correctly (a carry of ``window`` characters bridges feeds) and yields the
    same counts as one single feed of the concatenated stream.

    Boundary contract: position ``0`` has no predecessor and is never a
    target; the first supervised target is the stream's second character,
    whose context window is truncated to the one preceding character. The
    stream's final character is always a target (of the pass) and never a
    supervised target itself.
    """

    def __init__(self, *, window: int) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._carry = ""
        self._position = 0
        self.supervised_targets = 0
        self.bigram_counts: Counter[tuple[str, str]] = Counter()
        self.context_occurrences: Counter[str] = Counter()

    def feed(self, chunk: str) -> None:
        """Consume one chunk of the stream, crossing chunk boundaries."""
        if not chunk:
            return
        text = self._carry + chunk
        start = len(self._carry)
        for i in range(max(start, 1), len(text)):
            target = text[i]
            context = text[max(0, i - self._window) : i]
            self.supervised_targets += 1
            self.bigram_counts[(text[i - 1], target)] += 1
            self.context_occurrences[context] += 1
        self._position += len(chunk)
        self._carry = text[-(self._window + 1) :] if len(text) > self._window else text

    def reconcile(self) -> dict[str, int]:
        """Return the exact totals and assert both reconciliations hold."""
        bigram_tokens = sum(self.bigram_counts.values())
        context_total = sum(self.context_occurrences.values())
        if bigram_tokens != self.supervised_targets:
            raise AssertionError(
                f"bigram tokens {bigram_tokens} != supervised targets {self.supervised_targets}"
            )
        if context_total != self.supervised_targets:
            raise AssertionError(
                f"context occurrences {context_total} != supervised targets {self.supervised_targets}"
            )
        return {
            "supervised_targets": self.supervised_targets,
            "bigram_tokens": bigram_tokens,
            "distinct_bigrams": len(self.bigram_counts),
            "distinct_contexts": len(self.context_occurrences),
        }


def bigram_relative_frequencies(
    text: str, *, stride: int | None = None
) -> dict[tuple[str, str], float]:
    """Bigram relative frequencies over the census or over a stride-d sample.

    With ``stride=None`` this is the exact census: every consecutive pair.
    With ``stride=d`` this is the stride-d *sample*: only pairs whose target
    position is a multiple of ``d``. The two coincide only by accident —
    see :func:`stride_aliasing_demo` for the counterexample.
    """
    if stride is None:
        pairs: list[tuple[str, str]] = list(zip(text, text[1:], strict=False))
    else:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        pairs = [(text[i - 1], text[i]) for i in range(stride, len(text), stride)]
    total = len(pairs)
    if total == 0:
        return {}
    counts = Counter(pairs)
    return {pair: count / total for pair, count in counts.items()}


def stride_aliasing_demo(
    n: int = 16,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """The alternating-stream aliasing counterexample.

    On ``"abab…"`` the census carries both ``(a, b)`` and ``(b, a)`` at equal
    frequency, while the stride-2 sample sees only one of them: half the
    bigram inventory is gone. This is the checkable form of the claim that
    stride-d subsampling neither preserves bigram relative frequencies nor is
    automatically unbiased.
    """
    if n < 4 or n % 2:
        raise ValueError(f"n must be an even number >= 4, got {n}")
    text = "ab" * (n // 2)
    return (
        bigram_relative_frequencies(text),
        bigram_relative_frequencies(text, stride=2),
    )
