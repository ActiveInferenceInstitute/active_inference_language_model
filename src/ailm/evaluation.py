"""Versioned, immutable evaluation fixtures shared by production and tests.

The Stage 1 population is a scientific protocol input, not a test convenience.
Keeping its prose in a test module previously made recorders import test code,
while a second production copy could drift independently.  This module owns the
single canonical byte string and authenticates it at import time.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass


class EvaluationFixtureError(ValueError):
    """Raised when a protocol fixture no longer matches its pinned identity."""


@dataclass(frozen=True)
class EvaluationFixture:
    """One immutable, self-identifying evaluation population."""

    protocol: str
    text: str
    sha256: str
    size_bytes: int
    repetitions: int
    source: str
    license: str
    source_references: tuple[str, ...]

    def validate(self) -> None:
        """Fail closed if content or provenance differs from the protocol pin."""
        if not self.protocol or not self.source or not self.license:
            raise EvaluationFixtureError("fixture protocol, source, and license must be non-empty")
        if self.repetitions <= 0:
            raise EvaluationFixtureError("fixture repetitions must be a positive integer")
        if not self.source_references or any(not item for item in self.source_references):
            raise EvaluationFixtureError("fixture source references must be non-empty")
        encoded = self.text.encode("utf-8")
        if len(encoded) != self.size_bytes:
            raise EvaluationFixtureError(
                f"fixture has {len(encoded)} bytes, expected pinned extent {self.size_bytes}"
            )
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != self.sha256:
            raise EvaluationFixtureError(
                f"fixture sha256 {digest} does not match pinned identity {self.sha256}"
            )

    def metadata(self) -> dict[str, object]:
        """Return the complete serializable identity used by docs and renderers."""
        self.validate()
        return {
            "protocol": self.protocol,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "repetitions": self.repetitions,
            "source": self.source,
            "license": self.license,
            "source_references": list(self.source_references),
        }


_STAGE1_BASE_TEXT = (
    "the cat sat on the mat and the rat ran from the flat "
    "she sells sea shells by the sea shore "
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood "
    "the quick brown fox jumps over the lazy dog "
    "peter piper picked a peck of pickled peppers "
    "mary had a little lamb its fleece was white as snow "
    "humpty dumpty sat on a wall humpty dumpty had a great fall "
    "all the king-s horses and all the king-s men "
    "couldn-t put humpty together again "
)

STAGE1_EVALUATION = EvaluationFixture(
    protocol="ailm-stage1-letter-evaluation-v1",
    text=_STAGE1_BASE_TEXT * 3,
    sha256="513f73d02c2220185daf0a30865bed07a9af2167b40dcf5bb4341617efc6c20f",
    size_bytes=1323,
    repetitions=3,
    source=(
        "AILM-curated normalized composite of an AILM-authored rhyming sentence, "
        "traditional English tongue twisters and nursery verse, and a pangram"
    ),
    license=(
        "CC BY-SA 4.0 for the AILM selection and normalization; cited source "
        "phrases are public domain in the United States"
    ),
    source_references=(
        "https://blogs.loc.gov/folklife/2017/07/she-sells-seashells-and-mary-anning-metafolklore-with-a-twist/",
        "https://www.loc.gov/resource/sn84020645/1906-04-15/ed-1/?sp=20",
        "https://www.gutenberg.org/ebooks/66541",
        "https://www.gutenberg.org/ebooks/25027",
        "https://www.gutenberg.org/ebooks/67302",
        "https://www.gutenberg.org/ebooks/24623",
    ),
)

# Authentication occurs before any consumer can score or record the fixture.
STAGE1_EVALUATION.validate()
STAGE1_EVALUATION_TEXT = STAGE1_EVALUATION.text


def evaluation_population_bound(text: str) -> dict[str, float]:
    """Return the exact all-position order-1 floor for one letter sequence.

    Transition cross-entropy is empirical conditional entropy plus a
    non-negative conditional KL term. The production scorer also predicts the
    first character from a fixed uniform prior, so its lower bound is

    ``(log(|alphabet|) + (T - 1) * conditional_entropy) / T``.

    The import is local so merely loading fixture metadata does not initialize
    pymdp/JAX; scoring uses the production alphabet and normalization rather
    than a second copy here.
    """
    from ailm.levels.letter import N_SYMBOLS, normalize_text

    normalized = normalize_text(text)
    if len(normalized) < 2:
        raise ValueError(f"passage has {len(normalized)} normalized characters, need >= 2")
    pairs = list(zip(normalized, normalized[1:], strict=False))
    joint = Counter(pairs)
    previous = Counter(first for first, _ in pairs)
    transitions = len(pairs)
    conditional_entropy = -sum(
        (count / transitions) * math.log((count / transitions) / (previous[first] / transitions))
        for (first, _), count in joint.items()
    )
    positions = len(normalized)
    initial_nll = math.log(N_SYMBOLS)
    all_position_entropy = (initial_nll + transitions * conditional_entropy) / positions
    return {
        "conditional_entropy_nats": conditional_entropy,
        "all_position_entropy_nats": all_position_entropy,
        "initial_nll_nats": initial_nll,
        "max_gain_nats": math.log(N_SYMBOLS) - all_position_entropy,
        "perplexity": math.exp(all_position_entropy),
        "positions": float(positions),
        "transitions": float(transitions),
    }


__all__ = [
    "EvaluationFixture",
    "EvaluationFixtureError",
    "STAGE1_EVALUATION",
    "STAGE1_EVALUATION_TEXT",
    "evaluation_population_bound",
]
