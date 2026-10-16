#!/usr/bin/env python3
"""Generate vendor/cmudict_counts.json — the Stage 2 phoneme corpus artifact.

Derives everything Stage 2 needs from CMUdict and vendors it so tests never
touch the network. Output is deterministic: no RNG, no dict-iteration-order
dependence (all keys sorted), no timestamps.

Two datasets are produced:

1. **Phone sequence counts** — unigram and bigram counts over the primary
   pronunciation of every base word, stress digits stripped. Feeds the phoneme
   bigram model (the direct analogue of Stage 1's letter model).

2. **G2P length-matched pairs** — the subset of words whose letter count equals
   their phone count, so the 1:1 alignment is unambiguous and no alignment is
   fabricated. Roughly a quarter of CMUdict. The train/test split is by sorted
   index (every 5th item to test), so it is reproducible from the artifact alone
   with no seed to record.

Run:  .venv/bin/python scripts/generate_phoneme_counts.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.corpus.cmudict import (  # noqa: E402
    CANONICAL_ARPABET,
    CMUDICT_SOURCE_METADATA,
    PinnedCMUDictSource,
    _strip_stress,
    load_pinned_cmudict_source,
    verify_pinned_cmudict_bytes,
)
from ailm.levels.letter import ALPHA_TO_IDX  # noqa: E402

SCHEMA = "ailm-cmudict-counts-v1"

#: Every Nth sorted length-matched pair goes to the held-out test set.
TEST_EVERY = 5

#: Shortest length-matched word accepted. Single-letter words carry no bigram.
MIN_WORD_LEN = 2


def build_payload(
    source: PinnedCMUDictSource,
) -> dict:
    """Derive the full Stage 2 artifact from one authenticated snapshot.

    Args:
        source: Raw bytes, production parse, and deliberately naive control,
            all derived from one pinned in-memory source snapshot.

    Returns:
        JSON-serialisable dict with the ailm-cmudict-counts-v1 schema.

    Raises:
        ValueError: If the production parser and raw comment audit disagree.
    """
    verify_pinned_cmudict_bytes(source.content)
    cmudict = source.dictionary
    comment_audit = source.comment_audit

    # Primary pronunciation only, stress stripped. Sorted for determinism.
    prons: dict[str, list[str]] = {
        word: [_strip_stress(p) for p in pronunciations[0]]
        for word, pronunciations in sorted(cmudict.items())
        if pronunciations
    }

    # ── 1. Phone sequence counts ─────────────────────────────────────────────
    uni: Counter[str] = Counter()
    bi: Counter[tuple[str, str]] = Counter()
    # Word-initial phone counts. With a per-word prior reset, position 0 of every
    # word is scored under the start prior D rather than under a transition, so
    # these counts plus the bigram counts are *exact sufficient statistics* for the
    # full-corpus perplexity:
    #
    #   logL = SUM_p initials[p]*log D[p] + SUM_(a,b) bigrams[a,b]*log T[b|a]
    #   n    = SUM initials + SUM bigrams
    #
    # That means Part A can be evaluated over all 126k base words in closed form
    # without vendoring 126k sequences. It matters because the only sequence list
    # in this artifact is the length-matched G2P subset, which is biased toward
    # short, transparent-orthography words and measurably *flatters* the model
    # (15.29 vs 16.74 perplexity). Evaluating Part A on that subset would repeat,
    # in a new place, exactly the subsample-bias error documented in
    # ailm.levels.phoneme._iter_pairs.
    initials: Counter[str] = Counter()
    for seq in prons.values():
        uni.update(seq)
        bi.update(pairwise(seq))
        if seq:
            initials[seq[0]] += 1

    unknown = set(uni) - CANONICAL_ARPABET
    if unknown:
        raise ValueError(
            f"Non-canonical phones in CMUdict after stress stripping: "
            f"{sorted(unknown)}. Refusing to vendor a contaminated inventory."
        )
    clean_inventory = frozenset(uni)
    if clean_inventory != comment_audit.canonical_inventory:
        raise ValueError(
            "CMUdict payload and comment-stripping audit produced different canonical inventories"
        )
    if comment_audit.naive_inventory - comment_audit.leaked_comment_tokens != clean_inventory:
        raise ValueError(
            "The naive inventory is not explained exactly by canonical phones plus "
            "inline-comment tokens"
        )

    # ── 2. G2P length-matched pairs ──────────────────────────────────────────
    # Only words whose letters map 1:1 onto phones. Anything else would need an
    # alignment model, and a guessed alignment is fabricated training data.
    matched: list[tuple[str, list[str]]] = []
    for word, seq in prons.items():
        letters = [c for c in word.lower() if c in ALPHA_TO_IDX]
        if len(letters) == len(seq) and len(letters) >= MIN_WORD_LEN:
            matched.append(("".join(letters), seq))
    matched.sort()  # explicit: the split depends on this order

    payload = {
        "_schema": SCHEMA,
        "_description": (
            "Stage 2 phoneme corpus derived from CMUdict. Phone counts use the "
            "primary pronunciation of every base word with stress digits "
            "stripped. phone_initials holds word-initial phone counts; together "
            "with phone_bigrams these are exact sufficient statistics for the "
            "full-corpus per-word-reset perplexity, so Part A need not be "
            "evaluated on the biased length-matched subset. G2P pairs are "
            "restricted to words whose letter count equals their phone count so "
            "the 1:1 alignment is unambiguous; every "
            f"{TEST_EVERY}th sorted pair is held out for test."
        ),
        "phones": sorted(uni),
        "test_every": TEST_EVERY,
        "min_word_len": MIN_WORD_LEN,
        "source_base_words": len(prons),
        "source_pronunciation_entries": sum(len(value) for value in cmudict.values()),
        "naive_phoneme_count": len(comment_audit.naive_inventory),
        "comment_phantom_count": len(comment_audit.leaked_comment_tokens),
        "comment_phantom_tokens": sorted(comment_audit.leaked_comment_tokens),
        "phone_unigrams": {p: c for p, c in sorted(uni.items())},
        "phone_bigrams": {f"{a},{b}": c for (a, b), c in sorted(bi.items())},
        "phone_initials": {p: c for p, c in sorted(initials.items())},
        "g2p_pairs": [[w, " ".join(seq)] for w, seq in matched],
    }
    payload.update(CMUDICT_SOURCE_METADATA)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "vendor" / "cmudict_counts.json",
        help="Where to write the artifact.",
    )
    parser.add_argument(
        "--cmudict",
        type=Path,
        default=None,
        help=(
            "Path to the pinned cmudict.dict revision. Defaults to "
            "vendor/cmudict.dict; any other bytes are rejected."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CMUdict if the file is absent.",
    )
    args = parser.parse_args()

    source_path = args.cmudict or (_PROJECT_ROOT / "vendor" / "cmudict.dict")
    source = load_pinned_cmudict_source(source_path, download=args.download)
    payload = build_payload(source)

    # sort_keys keeps the byte output stable across runs and Python versions.
    # No indent: g2p_pairs alone is ~32k entries, and pretty-printing it inflates
    # the tracked artifact roughly 5x for no reviewability gain (nobody reads a
    # 32k-entry array by eye — scripts/ and the tests are the readable surface).
    atomic_write_json(
        args.output,
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    n_test = len(payload["g2p_pairs"]) // TEST_EVERY
    print(
        f"source_base_words={payload['source_base_words']} "
        f"phones={len(payload['phones'])} "
        f"unigrams={len(payload['phone_unigrams'])} "
        f"bigrams={len(payload['phone_bigrams'])}"
    )
    print(
        f"g2p_pairs={len(payload['g2p_pairs'])} "
        f"(~{len(payload['g2p_pairs']) - n_test} train / ~{n_test} test)"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
