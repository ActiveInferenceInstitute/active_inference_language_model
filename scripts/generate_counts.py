#!/usr/bin/env python3
"""Generate character-bigram counts over the AILM 30-symbol alphabet.

Writes ``vendor/brown_counts.json`` as a versioned artifact containing a
``counts`` mapping ``{"<a>,<b>": count}`` and the immutable raw-source identity
that produced it.  ``<a>`` and ``<b>`` are single symbols drawn from ALPHABET.

Corpus selection
----------------
1. Canonical: the exact NLTK ``brown.zip`` revision, byte extent, and digest
   pinned in ``ailm.corpus.brown``.  Extracted or moving-source data is refused.
2. Noncanonical diagnostic: ``FALLBACK_TEXT`` below -- an assembled block of public-domain
   English prose used *only* as a source of character-transition statistics.
   It is not a verbatim quotation source and must not be cited as one.  Its
   schema is rejected by the production loader, and the CLI refuses to publish
   it under the canonical artifact filename.

Normalization (applied identically to both corpora)
---------------------------------------------------
* Tokens are joined with single spaces, then spaces immediately preceding
  ``.`` or ``'`` are removed. This undoes NLTK's tokenizer splitting so that
  ``evidence .`` becomes ``evidence.`` and ``boys '`` becomes ``boys'``,
  yielding realiztic end-of-word transitions.
* Text is lowercased.
* Any character outside ALPHABET becomes a space.
* Runs of consecutive spaces collapse to one, and the text is stripped.
  Without collapsing, tokenizer artifacts (dropped quote/comma tokens) would
  make ``(space, space)`` the single most frequent bigram by a wide margin and
  swamp the letter statistics.

Usage
-----
    .venv/bin/python scripts/generate_counts.py [--source-zip PATH] [--output PATH]
    .venv/bin/python scripts/generate_counts.py --force-fallback \
        --output /path/to/noncanonical-brown-diagnostic.json
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from _artifact_io import atomic_write_json  # noqa: E402

from ailm.corpus.brown import (  # noqa: E402
    BROWN_BIGRAM_ARTIFACT_SCHEMA,
    BROWN_NONCANONICAL_ARTIFACT_SCHEMA,
    brown_source_metadata,
    load_brown_corpus,
)

# 26 letters + space + period + apostrophe + hyphen == 30 symbols.
ALPHABET: list[str] = sorted("abcdefghijklmnopqrstuvwxyz ." + "'" + "-")
ALPHA_SET: frozenset[str] = frozenset(ALPHABET)

# Explicit raises: this script writes a vendored artifact, and `python -O` would
# strip `assert`, letting a malformed alphabet produce a bad counts file silently.
if len(ALPHABET) != 30:
    raise AssertionError(f"expected 30 symbols, got {len(ALPHABET)}")
if len(ALPHA_SET) != 30:
    raise AssertionError("ALPHABET contains duplicates")

DEFAULT_OUTPUT = REPO_ROOT / "vendor" / "brown_counts.json"

_SPACE_BEFORE_PUNCT = re.compile(r"\s+(?=[.'])")
_SPACE_RUN = re.compile(r" {2,}")

# Assembled public-domain English prose. Statistics source only.
FALLBACK_TEXT = """
It is a truth universally acknowledged, that a single man in possession of a
good fortune, must be in want of a wife. However little known the feelings or
views of such a man may be on his first entering a neighborhood, this truth
is so well fixed in the minds of the surrounding families, that he is
considered the rightful property of some one or other of their daughters.
My dear Mr. Bennet, said his lady to him one day, have you heard that
Netherfield Park is let at last? Mr. Bennet replied that he had not. But it
is, returned she, for Mrs. Long has just been here, and she told me all about
it. Mr. Bennet made no answer. Do you not want to know who has taken it? cried
his wife impatiently. You want to tell me, and I have no objection to hearing
it. This was invitation enough.

Alice was beginning to get very tired of sitting by her sister on the bank,
and of having nothing to do. Once or twice she had peeped into the book her
sister was reading, but it had no pictures or conversations in it, and what is
the use of a book, thought Alice, without pictures or conversations? So she
was considering in her own mind, as well as she could, for the hot day made
her feel very sleepy and stupid, whether the pleasure of making a daisy chain
would be worth the trouble of getting up and picking the daisies, when
suddenly a white rabbit with pink eyes ran close by her. There was nothing so
very remarkable in that, nor did Alice think it so very much out of the way to
hear the rabbit say to itself, oh dear, oh dear, I shall be too late. But when
the rabbit actually took a watch out of its waistcoat pocket and looked at it
and then hurried on, Alice started to her feet, for it flashed across her mind
that she had never before seen a rabbit with either a waistcoat pocket or a
watch to take out of it, and burning with curiosity she ran across the field
after it, and was just in time to see it pop down a large rabbit hole under the
hedge. In another moment down went Alice after it, never once considering how
in the world she was to get out again.

Call me Ishmael. Some years ago, never mind how long precisely, having little
or no money in my purse, and nothing particular to interest me on shore, I
thought I would sail about a little and see the watery part of the world. It is
a way I have of driving off the spleen and regulating the circulation. Whenever
I find myself growing grim about the mouth, whenever it is a damp, drizzly
November in my soul, whenever I find myself involuntarily pausing before coffin
warehouses, then I account it high time to get to sea as soon as I can. There
now is your insular city of the Manhattoes, belted round by wharves as Indian
isles by coral reefs, commerce surrounds it with her surf. Right and left, the
streets take you waterward. Almost all men in their degree, some time or other,
cherish very nearly the same feelings towards the ocean with me.

Four score and seven years ago our fathers brought forth on this continent a
new nation, conceived in liberty, and dedicated to the proposition that all men
are created equal. Now we are engaged in a great civil war, testing whether
that nation, or any nation so conceived and so dedicated, can long endure. We
are met on a great battlefield of that war. We have come to dedicate a portion
of that field as a final resting place for those who here gave their lives that
that nation might live. It is altogether fitting and proper that we should do
this. But in a larger sense we cannot dedicate, we cannot consecrate, we cannot
hallow this ground. The brave men, living and dead, who struggled here have
consecrated it far above our poor power to add or detract. The world will little
note nor long remember what we say here, but it can never forget what they did
here. It is for us the living rather to be dedicated here to the unfinished
work which they who fought here have thus far so nobly advanced.

We hold these truths to be self-evident, that all men are created equal, that
they are endowed with certain unalienable rights, that among these are life,
liberty, and the pursuit of happiness. That to secure these rights, governments
are instituted among men, deriving their just powers from the consent of the
governed. That whenever any form of government becomes destructive of these
ends, it is the right of the people to alter or to abolish it, and to institute
new government, laying its foundation on such principles and organizing its
powers in such form, as to them shall seem most likely to effect their safety
and happiness. Prudence, indeed, will dictate that governments long established
should not be changed for light and transient causes.

To Sherlock Holmes she is always the woman. I have seldom heard him mention her
under any other name. In his eyes she eclipses and predominates the whole of her
sex. It was not that he felt any emotion akin to love for Irene Adler. All
emotions, and that one particularly, were abhorrent to his cold, precise, but
admirably balanced mind. He was, I take it, the most perfect reasoning and
observing machine that the world has seen, but as a lover he would have placed
himself in a false position. He never spoke of the softer passions, save with a
gibe and a sneer. They were admirable things for the observer, excellent for
drawing the veil from men's motives and actions. But for the trained reasoner to
admit such intrusions into his own delicate and finely adjusted temperament was
to introduce a distracting factor which might throw a doubt upon all his mental
results. Grit in a sensitive instrument, or a crack in one of his own high-power
lenses, would not be more disturbing than a strong emotion in a nature such as
his.

A crow, having stolen a piece of cheese from a cottage window, flew up into a
high tree to eat it. A fox, who saw her, longed to possess the cheese himself,
and by a wily stratagem succeeded. How handsome is the crow, he exclaimed, in
the beauty of her shape and in the fairness of her complexion. If only her voice
were equal to her looks, she ought undoubtedly to be queen of the birds. The
crow, quite overcome by flattery, opened her mouth to show that she could sing,
and dropped the cheese. The fox snatched it up and said, my good friend, you
have a voice, but what you want is judgment. A hungry fox, seeing some bunches
of grapes hanging from a vine that was trained over a tall branch, tried every
means he could think of to reach them, but wearied himself in vain, for he could
not get at them. At last he turned away, hiding his disappointment, and saying,
the grapes are sour and not ripe as I thought.

The north wind and the sun disputed which of them was the stronger, and they
agreed that the one who first stripped a traveller of his clothing should be
accounted the more powerful. The north wind began, and blew with all his might,
but the keener his blasts the closer the traveller wrapped his cloak around him,
until at last, resigning all hope of victory, the wind called upon the sun to
see what he could do. The sun suddenly shone out with all his warmth. The
traveller no sooner felt the genial rays than he took off one garment after
another, and at last, fairly overcome with heat, undressed and bathed in a
stream that lay in his path. Persuasion is better than force, and gentle means
will often accomplish what harshness cannot.

In the beginning of the study of any natural science, the student must learn to
observe carefully and to record what has been observed without regard to what he
expects to find. The habit of accurate observation is not easily acquired, and
it is more easily lost than gained. Measurement, comparison, and the patient
repetition of an experiment under varied conditions are the ordinary means by
which a doubtful question is brought to a decision. A single result, however
striking, proves little. It is the agreement of many independent results that
gives confidence, and it is the honest report of results that do not agree which
gives a science its power of self-correction over the course of many years.

The ant, as everyone knows, is a creature of industry, and the anthill is a
commonwealth in which no individual labours for herself alone. Watch a column of
foragers on a summer afternoon and you will see that the road is kept open by the
passage of many feet, that the burden of one is taken up by another when she
tires, and that news of a fallen crumb travels back along the line faster than
any single traveller could carry it. There is no overseer in the nest, and yet
the work is apportioned. The young remain within and tend the brood, the older
sisters venture out and take the risk of the open ground. When the season turns,
the whole society alters its habits together, without any word of command being
given, as though the colony were one animal with a thousand bodies. It is this
that makes the study of such creatures so instructive to those who would
understand the ordering of larger and prouder communities.

A man's mind may be compared to a lantern carried through a dark wood. The light
does not create the trees, but without it he would walk into them. What he sees
depends partly on the wood and partly on the lantern, and it is idle to ask which
of the two contributes more, since neither alone would give him a path. So it is
with our knowledge of the world. We do not receive impressions as a blank sheet
receives ink. We come already furnished with expectations, and we correct them
slowly, at the cost of some pain, whenever the world refuses to answer to them.
The wise man is not the one who has no expectations, for such a man could learn
nothing, but the one who holds them loosely and gives them up without bitterness
when the evidence is against him. To be surprised often, and to be willing to be
surprised, is the beginning of understanding.

Consider how a letter follows a letter and a word follows a word. No speaker
chooses each sound afresh out of the whole store of sounds available to him. The
tongue runs in channels worn by long use, so that after one letter a few others
are likely and the rest are nearly impossible. The child who is learning to read
discovers this before he can explain it, guessing the end of a word from its
beginning, and being right more often than chance would allow. What he has
acquired is not a rule that he could state, but a habit of expectation, built up
out of thousands of small encounters with print and speech. In this he is doing
in little what every student of nature does in large, gathering from many
particular instances a sense of what is probable, and using that sense to see
further than his eyes alone would carry him.
"""


def normalize(text: str) -> str:
    """Lowercase, restrict to ALPHABET, and collapse whitespace runs."""
    text = _SPACE_BEFORE_PUNCT.sub("", text)
    text = text.lower()
    text = "".join(ch if ch in ALPHA_SET else " " for ch in text)
    return _SPACE_RUN.sub(" ", text).strip()


def load_brown_text(source_zip: Path | str | None = None) -> str:
    """Return normalized text from the verified, pinned Brown archive."""
    brown = load_brown_corpus(source_zip)
    sentences = brown.sents()
    print(f"Brown corpus: {len(sentences)} sentences")
    return normalize(" ".join(" ".join(sent) for sent in sentences))


def count_bigrams(text: str) -> Counter[tuple[str, str]]:
    """Count adjacent character pairs in already-normalized text."""
    counts: Counter[tuple[str, str]] = Counter()
    for a, b in pairwise(text):
        counts[(a, b)] += 1
    return counts


def serialize(counts: Counter[tuple[str, str]]) -> dict[str, int]:
    """Render counts as {"<a>,<b>": count}, sorted by ALPHABET order."""
    order = {sym: i for i, sym in enumerate(ALPHABET)}
    for a, b in counts:
        if a not in ALPHA_SET or b not in ALPHA_SET:
            raise ValueError(f"bigram outside alphabet: {a!r},{b!r}")
    return {
        f"{a},{b}": counts[(a, b)]
        for a, b in sorted(counts, key=lambda pair: (order[pair[0]], order[pair[1]]))
    }


def _canonical_payload(text: str, counts: Counter[tuple[str, str]]) -> dict:
    """Build the loadable artifact around measured counts and pinned identity."""
    serialized = serialize(counts)
    return {
        "schema": BROWN_BIGRAM_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "NLTK Brown",
            "canonical": True,
            "raw_source": brown_source_metadata(),
            "normalization": (
                "tokenized sentences space-joined; space-before-period/apostrophe "
                "repair; lowercase; out-of-alphabet to space; space-run collapse; strip"
            ),
            "alphabet": ALPHABET,
            "normalized_characters": len(text),
            "distinct_bigrams": len(serialized),
            "bigram_tokens": sum(serialized.values()),
        },
        "counts": serialized,
    }


def _fallback_payload(text: str, counts: Counter[tuple[str, str]]) -> dict:
    """Build an explicitly noncanonical diagnostic artifact."""
    serialized = serialize(counts)
    return {
        "schema": BROWN_NONCANONICAL_ARTIFACT_SCHEMA,
        "provenance": {
            "corpus": "embedded public-domain diagnostic prose",
            "canonical": False,
            "source_kind": "embedded-fallback",
            "normalization": "same character normalization as the canonical writer",
            "alphabet": ALPHABET,
            "normalized_characters": len(text),
            "distinct_bigrams": len(serialized),
            "bigram_tokens": sum(serialized.values()),
        },
        "counts": serialized,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--source-zip",
        type=Path,
        default=None,
        help="path to the exact pinned NLTK brown.zip archive",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="write an explicitly noncanonical diagnostic from embedded prose",
    )
    args = parser.parse_args(argv)

    output = args.output or DEFAULT_OUTPUT
    if args.force_fallback:
        if args.output is None or output.name == DEFAULT_OUTPUT.name:
            print(
                "REFUSING TO WRITE: fallback data is noncanonical and may not be "
                "published as brown_counts.json. Supply --output with a diagnostic "
                "filename such as noncanonical-brown-diagnostic.json.",
                file=sys.stderr,
            )
            return 2
        text = normalize(FALLBACK_TEXT)
        payload = _fallback_payload(text, count_bigrams(text))
        print(f"Fallback text: {len(text)} normalized chars")
        source = "noncanonical-fallback"
    else:
        try:
            text = load_brown_text(args.source_zip)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"REFUSING TO WRITE: {exc}", file=sys.stderr)
            return 1
        payload = _canonical_payload(text, count_bigrams(text))
        source = "pinned-brown"

    if len(text) < 2:
        print("corpus too short to produce bigrams", file=sys.stderr)
        return 1

    atomic_write_json(output, payload, indent=2)

    counts_payload = payload["counts"]
    total = sum(counts_payload.values())
    print(
        f"source={source} chars={len(text)} "
        f"distinct_bigrams={len(counts_payload)}/900 total={total}"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
