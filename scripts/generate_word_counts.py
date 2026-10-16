#!/usr/bin/env python3
"""Generate the vendored Stage 3 word artifact from the NLTK Brown corpus.

Writes ``vendor/brown_word_counts.json``, the sufficient statistics for the word
level: training word-type frequencies, the held-out token stream, and the
provenance needed to reproduce both.

What the split is, and why it is by document rather than by type
----------------------------------------------------------------
The word level is a **closed-vocabulary** model: it assigns probability only to
words in its lexicon. Splitting held-out data by word *type* would therefore put
every evaluation token outside the model's support and drive the perplexity to
infinity — the split would be measuring the closed-vocabulary assumption, not
the model. The split is by authenticated Brown source document instead. Within
each Brown genre, a revision-bound SHA-256 ordering selects the whole-document
held-out count nearest ``1 - BROWN_TRAIN_DOCUMENT_FRACTION``. Evaluation is then restricted to
tokens whose type is in vocabulary. No source document contributes to both arms,
and the genre-blocked corpus order cannot collapse evaluation onto its last genres.

The fraction of held-out tokens that survive that restriction is the honest
scoping number for the whole stage, so it is recorded rather than mentioned:
``claim:stage3_coverage_pct``. Every Stage 3 result describes that sublanguage.

Normalization
-------------
Tokens are lowercased and kept only if every character is an ASCII letter. That
drops punctuation, numerals and hyphenated forms, which keeps the observation
alphabet at 26 letters plus a single end-of-word symbol. It is a *different*
alphabet from Level 1's 30 running-text symbols, so Stage 3 perplexities are not
comparable with ``claim:stage1_letter_perplexity`` and the two must never be
placed in one table.

Determinism
-----------
The corpus reader verifies the exact pinned ``brown.zip`` bytes before reading,
and the artifact embeds that immutable source identity plus an ordered,
length-framed digest of all Brown file IDs. The split is a deterministic function
of the immutable revision, file ID, and category—not token content or model
output—and every frequency table is explicitly sorted by descending count and
then alphabetically. Re-running
therefore reproduces the artifact byte for byte, which
``tests/test_provenance_checksums.py`` enforces whenever the pinned archive is
available locally.

Usage
-----
    python scripts/generate_word_counts.py [--source-zip PATH] [--output PATH] [--vocab-size N]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.corpus.brown import (  # noqa: E402
    BROWN_TRAIN_DOCUMENT_FRACTION,
    BROWN_WORD_ARTIFACT_SCHEMA,
    brown_document_identity_sha256,
    brown_partition_input_sha256,
    brown_source_metadata,
    build_brown_document_partition,
)

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "brown_word_counts.json"

#: Observation alphabet: 26 lowercase letters, index 0-25, plus END at 26.
LETTERS = "abcdefghijklmnopqrstuvwxyz"

#: Default bounded vocabulary. See docs/decisions.md entry 13 for the measured
#: trade-off between coverage and the word factor's dense transition.
DEFAULT_VOCAB_SIZE = 2000

#: Words longer than this are dropped from the vocabulary rather than inflating
#: the position factor for a handful of types. The drop count is recorded.
MAX_WORD_LEN = 20


def _parse_custom_lexicon(path: Path) -> tuple[dict[str, int], int, str]:
    """Parse a custom lexicon and retain its validation/provenance metadata."""
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8: {exc}") from exc

    letters = frozenset(LETTERS)
    freq: dict[str, int] = {}
    skipped = 0
    dropped_for_length: set[str] = set()
    saw_frequency = False
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        word = parts[0].lower()
        if not word or not letters.issuperset(word):
            skipped += 1
            continue
        if len(word) > MAX_WORD_LEN:
            skipped += 1
            dropped_for_length.add(word)
            continue
        if len(parts) > 2:
            raise ValueError(
                f"{path}:{line_number}: expected 'word[<TAB>positive-frequency]', got {line!r}"
            )
        if len(parts) == 2:
            try:
                count = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: frequency must be a positive integer, got {parts[1]!r}"
                ) from exc
            if count < 1:
                raise ValueError(
                    f"{path}:{line_number}: frequency must be a positive integer, got {parts[1]!r}"
                )
            saw_frequency = True
        else:
            count = 1
        freq[word] = freq.get(word, 0) + count

    if not freq:
        sys.exit(
            f"{path} yielded no usable words. Entries must be alphabetic, at most "
            f"{MAX_WORD_LEN} characters, one per line, optionally followed by a "
            f"frequency."
        )
    if not saw_frequency:
        print(
            f"WARNING: {path} carries no frequency column, so the word prior will "
            f"be UNIFORM over {len(freq):,} words. A uniform word prior is one of "
            f"the mis-wirings docs/decisions.md entry 14 records as beating the "
            f"accuracy bar while being wrong. Supply frequencies if you have them.",
            file=sys.stderr,
        )
    if skipped:
        print(f"  skipped {skipped:,} unusable entries", file=sys.stderr)
    ordered = dict(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))
    return ordered, len(dropped_for_length), hashlib.sha256(raw_bytes).hexdigest()


def load_custom_lexicon(path: Path) -> dict[str, int]:
    """Read a user-supplied lexicon: one ``word[<TAB>frequency]`` per line.

    A frequency column is optional; without it every word gets a count of 1,
    which makes the word prior uniform over the lexicon. That is a *legitimate*
    choice for a domain lexicon with no corpus behind it, but it is not a
    neutral one: decision 14 records that a uniform word prior still beats the
    letter-bigram bar while discarding all frequency information, so a lexicon
    supplied without frequencies produces a model whose prior is one of the
    documented mis-wirings. The warning is emitted rather than buried.

    Args:
        path: Lexicon file. Lines are lowercased; anything containing a
            character outside ``LETTERS`` is skipped, because the observation
            alphabet is 26 letters plus one end symbol.

    Returns:
        ``word -> frequency``, in descending frequency then alphabetical order.

    Raises:
        ValueError: If a usable entry has a malformed, non-positive, or extra
            frequency field. A declared frequency is evidence, so it must never
            be silently replaced by the uniform-prior default.
        SystemExit: If the file yields no usable words at all.
    """
    frequencies, _dropped_for_length, _digest = _parse_custom_lexicon(path)
    return frequencies


def build_artifact(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    lexicon: Path | None = None,
    source_zip: Path | str | None = None,
) -> dict:
    """Assemble the vendored artifact from the Brown token stream.

    Args:
        vocab_size: Bound on the vocabulary. ``0`` means *no bound* — every
            length-eligible training type is admitted. The closed-form scorer
            handles any size; the pymdp agent is separately bounded by
            ``MAX_AGENT_VOCAB_BYTES``.
        lexicon: Optional custom word list replacing the frequency-ranked Brown
            vocabulary. The *evaluation* stream still comes from held-out Brown,
            restricted to whatever the lexicon covers — so a custom lexicon is
            scored on real text rather than on itself.
        source_zip: Optional path to the exact pinned NLTK ``brown.zip``. When
            absent, the canonical resolver checks its environment setting and
            NLTK data roots for byte-identical source bytes.
    """
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 0:
        raise ValueError(
            f"vocab_size must be a non-negative integer (zero means unbounded), got {vocab_size!r}"
        )

    partition = build_brown_document_partition(BROWN_TRAIN_DOCUMENT_FRACTION, source_zip)
    train = partition.train_tokens
    held = partition.heldout_tokens

    train_freq = Counter(train)
    eligible = [w for w in train_freq if len(w) <= MAX_WORD_LEN]
    dropped_long = len(train_freq) - len(eligible)

    lexicon_digest: str | None = None
    if lexicon is not None:
        custom, dropped_long, lexicon_digest = _parse_custom_lexicon(lexicon)
        # The lexicon supplies the word set AND its frequencies; Brown supplies
        # only the held-out text it is scored on.
        train_freq = Counter(custom)
        eligible = list(custom)
        ranked = list(custom)
    else:
        ranked = [
            w
            for w, _ in sorted(
                ((w, train_freq[w]) for w in eligible),
                key=lambda kv: (-kv[1], kv[0]),
            )
        ]
    vocab = ranked if vocab_size <= 0 else ranked[:vocab_size]
    vset = frozenset(vocab)
    eval_tokens: list[str] = []
    eval_document_indices: list[int] = []
    for word, document_index in zip(held, partition.heldout_token_document_indices, strict=True):
        if word in vset:
            eval_tokens.append(word)
            eval_document_indices.append(document_index)

    return {
        "schema": BROWN_WORD_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "NLTK Brown",
            "raw_source": brown_source_metadata(),
            "normalization": "lowercased; tokens of ASCII letters only",
            "target_train_document_fraction": BROWN_TRAIN_DOCUMENT_FRACTION,
            "train_document_fraction": len(partition.train_indices) / len(partition.document_ids),
            "train_token_fraction": len(train) / (len(train) + len(held)),
            "split": (
                "category-stratified revision-bound SHA-256 document split; "
                "not by word type (closed vocabulary)"
            ),
            "vocab_size": len(vocab),
            "vocab_bound": vocab_size or "unbounded",
            "lexicon_source": f"custom:{lexicon.name}" if lexicon else "brown-frequency-ranked",
            **(
                {
                    "lexicon_sha256": lexicon_digest,
                    "lexicon_frequency_total": sum(train_freq.values()),
                }
                if lexicon_digest is not None
                else {}
            ),
            "max_word_len": MAX_WORD_LEN,
            "types_dropped_for_length": dropped_long,
            "total_alphabetic_tokens": len(train) + len(held),
            "train_tokens": len(train),
            "held_tokens": len(held),
            "train_types": len(train_freq),
            "documents": len(partition.document_ids),
            "train_documents": len(partition.train_indices),
            "heldout_documents": len(partition.heldout_indices),
            "categories": len(set(partition.categories)),
            "document_identity_sha256": brown_document_identity_sha256(partition.document_ids),
            "document_partition_input_sha256": brown_partition_input_sha256(
                partition.document_ids, partition.categories
            ),
        },
        "documents": {
            "ids": list(partition.document_ids),
            "categories": list(partition.categories),
            "alphabetic_token_counts": list(partition.alphabetic_token_counts),
            "split_indices": {
                "train": list(partition.train_indices),
                "heldout": list(partition.heldout_indices),
            },
            "eval_token_document_indices": eval_document_indices,
        },
        # Frequencies of the bounded vocabulary, in descending frequency order.
        "word_freq": {w: train_freq[w] for w in vocab},
        # Every training type's frequency is needed for the letter-bigram bar,
        # which must be trained on the same data as the model but is NOT
        # restricted to the vocabulary — a bar handicapped to the model's
        # lexicon would be a crippled comparator (decision 2).
        "bar_train_freq": dict(sorted(train_freq.items(), key=lambda kv: (-kv[1], kv[0]))),
        # The held-out evaluation stream, restricted to in-vocabulary types.
        "eval_tokens": eval_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=None,
        help="path to the exact pinned NLTK brown.zip archive",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Vocabulary bound; 0 admits all length-eligible training types.",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help=(
            "Custom word list, one 'word[<TAB>frequency]' per line, replacing "
            "the frequency-ranked Brown vocabulary. Evaluation still uses "
            "held-out Brown text restricted to what the lexicon covers."
        ),
    )
    args = parser.parse_args(argv)

    artifact = build_artifact(args.vocab_size, args.lexicon, args.source_zip)
    payload = atomic_write_json(
        args.output,
        artifact,
        ensure_ascii=True,
        sort_keys=False,
        indent=1,
    )

    digest = hashlib.sha256(payload).hexdigest()
    prov = artifact["provenance"]
    coverage = 100.0 * len(artifact["eval_tokens"]) / prov["held_tokens"]
    try:
        shown = args.output.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"Wrote {shown}")
    print(f"  vocabulary        {len(artifact['word_freq']):,} types")
    print(f"  train tokens      {prov['train_tokens']:,}")
    print(f"  held-out tokens   {prov['held_tokens']:,}")
    print(f"  in-vocabulary     {len(artifact['eval_tokens']):,} ({coverage:.2f}%)")
    print(f"  sha256            {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
