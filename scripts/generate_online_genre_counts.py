#!/usr/bin/env python3
"""Generate the vendored entry-28 online-learning drift artifact from Brown.

Writes ``vendor/brown_online_drift.json``: the letter streams for the two
Brown genres that form the **non-stationary** learning regime of
``docs/decisions.md`` entry 28 — government (pre-drift distribution) and
romance (post-drift distribution). Each genre contributes a *learn* stream and
a held-out *evaluate* stream, split by sentence-index parity so the two are
disjoint, and each is normalized to the Stage 1 letter alphabet.

Why a genre drift, and why these two
------------------------------------
Entry 28 needs a *principled, injected drift*: a second distribution whose
letter-bigram transition statistics genuinely differ from the first, so that
an online agent that keeps learning has something real to track. Brown's
genre categories provide exactly this — different genres use different letter
sequences — and the split is fully offline and deterministic. Government vs
romance is chosen because their letter-bigram total-variation distance is
large (~0.20), which makes the drift measurable (P1) rather than asserted.

What each stream is
-------------------
The ``splits`` object has four entries:

- ``pre_learn`` / ``post_learn`` — the training streams of the first (pre-)
  and second (post-) distributions. The online agent filters ``pre_learn``,
  then continues ``infer_parameters`` over ``post_learn`` (the drift).
- ``pre_hold`` / ``post_hold`` — held-out streams, disjoint from the learn
  streams by sentence parity. ``pre_hold`` scores P1's identity-before-the-
  drift and P3's identity check; ``post_hold`` scores P2 (online beats frozen)
  and P3's re-fit control.

Streams are normalized letter-character strings (the same ALPHABET indexing
``ailm.levels.letter`` uses), bounded to a fixed number of characters so the
runtime stays tractable; the actual bounds are recorded in provenance.

Determinism
-----------
Sentence-index parity split, fixed genre order, sorted keys, no RNG, no
timestamp: the artifact regenerates byte for byte.

Usage
-----
    python scripts/generate_online_genre_counts.py [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.corpus.brown import brown_source_metadata, load_brown_corpus  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "brown_online_drift.json"

#: Pre-drift (first) distribution's genre.
PRE_GENRE = "government"

#: Post-drift (second) distribution's genre.
POST_GENRE = "romance"

#: Max characters kept per learn stream.
LEARN_LIMIT = 40000

#: Max characters kept per hold stream.
HOLD_LIMIT = 15000


def _normalize_letters(text: str) -> str:
    from ailm.levels.letter import normalize_text

    normalized = normalize_text(text)
    return "".join(c for c in normalized if c in _ALPHABET_SET)


from ailm.levels.letter import ALPHABET  # noqa: E402

_ALPHABET_SET = frozenset(ALPHABET)


def _genre_streams(brown, genre: str, learn_limit: int, hold_limit: int, test_every: int = 2):
    learn_sents: list[str] = []
    hold_sents: list[str] = []
    for index, sentence in enumerate(brown.sents(categories=genre)):
        target = hold_sents if index % test_every == 0 else learn_sents
        target.append(" ".join(sentence))
    learn_text = _normalize_letters(" ".join(learn_sents))[:learn_limit]
    hold_text = _normalize_letters(" ".join(hold_sents))[:hold_limit]
    return learn_text, hold_text, len(learn_sents), len(hold_sents)


def build_artifact(
    learn_limit: int = LEARN_LIMIT,
    hold_limit: int = HOLD_LIMIT,
    source_zip: Path | str | None = None,
) -> dict:
    """Assemble the vendored entry-28 drift artifact from Brown genres."""
    for name, value in (("learn_limit", learn_limit), ("hold_limit", hold_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    brown = load_brown_corpus(source_zip)
    pre_learn, pre_hold, pre_learn_sents, pre_hold_sents = _genre_streams(
        brown, PRE_GENRE, learn_limit, hold_limit
    )
    post_learn, post_hold, post_learn_sents, post_hold_sents = _genre_streams(
        brown, POST_GENRE, learn_limit, hold_limit
    )
    return {
        "schema": "ailm-online-drift-v2",
        "provenance": {
            "corpus": "NLTK Brown",
            "raw_source": brown_source_metadata(),
            "pre_genre": PRE_GENRE,
            "post_genre": POST_GENRE,
            "drift": (
                "letter-bigram distribution difference between the two Brown "
                "genres; measured (P1) not asserted"
            ),
            "alphabet": sorted(ALPHABET),
            "split": "sentence-index parity (even->hold, odd->learn) per genre",
            "learn_limit": learn_limit,
            "hold_limit": hold_limit,
            "pre_learn_sentences": pre_learn_sents,
            "pre_hold_sentences": pre_hold_sents,
            "post_learn_sentences": post_learn_sents,
            "post_hold_sentences": post_hold_sents,
        },
        "splits": {
            "pre_learn": pre_learn,
            "pre_hold": pre_hold,
            "post_learn": post_learn,
            "post_hold": post_hold,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--learn-limit", type=int, default=LEARN_LIMIT)
    parser.add_argument("--hold-limit", type=int, default=HOLD_LIMIT)
    parser.add_argument("--source-zip", type=Path, default=None, help="exact pinned NLTK brown.zip")
    args = parser.parse_args(argv)

    artifact = build_artifact(args.learn_limit, args.hold_limit, args.source_zip)
    atomic_write_json(args.output, artifact, ensure_ascii=True, sort_keys=False, indent=1)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    for name in ("pre_learn", "pre_hold", "post_learn", "post_hold"):
        print(f"  {name:10s} {len(artifact['splits'][name]):>8,} chars")
    print(f"  sha256   {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
