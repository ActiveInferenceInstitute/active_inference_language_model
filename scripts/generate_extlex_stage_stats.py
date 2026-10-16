#!/usr/bin/env python3
"""Stage and characterize the external WikiPron lexicon candidate (extlex lane).

Reads a pinned WikiPron eng_latn_us_broad.tsv snapshot from staging plus the
vendored CMUdict revision, and reports (deterministically, to JSON):

- entry / unique-grapheme counts and the raw IPA segment inventory;
- comment-stripped CMUdict comparison per decision 79's registered rule
  (candidate delta = TSV words absent from CMUdict), overlap, case/multiword notes;
- transcription caveat quantification: every TSV entry is IPA, so ALL entries
  require normalization before comparability with ARPAbet-based artifacts; entries
  are further split by whether their IPA contains only characters with plausible
  single-symbol ARPAbet correspondences.

No scoring, no claims.json writes, no vendor/ modification. Run:
  .venv/bin/python scripts/generate_extlex_stage_stats.py \
      --tsv <staging>/eng_latn_us_broad.tsv --output <path>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ailm.corpus.cmudict import load_cmudict  # noqa: E402

# Single-character IPA -> plausible ARPAbet correspondences (stress stripped).
# Two-character sequences (diphthongs, affricates) are handled separately below.
_IPA_TO_ARPABET_PLAUSIBLE: dict[str, str] = {
    "a": "AA",
    "\u0251": "AA",
    "\u00e6": "AE",
    "\u028c": "AH",
    "\u0254": "AO",
    "b": "B",
    "d": "D",
    "\u00f0": "DH",
    "\u0259": "AH",
    "\u025b": "EH",
    "\u025c": "ER",
    "e": "EY",
    "\u026a": "IH",
    "i": "IY",
    "\u0261": "G",
    "g": "G",
    "\u014b": "NG",
    "o": "OW",
    "p": "P",
    "\u0279": "R",
    "s": "S",
    "\u0283": "SH",
    "t": "T",
    "\u03b8": "TH",
    "\u028a": "UH",
    "u": "UW",
    "v": "V",
    "w": "W",
    "j": "Y",
    "z": "Z",
    "\u0292": "ZH",
    "h": "HH",
    "f": "F",
    "m": "M",
    "n": "N",
    "l": "L",
    "k": "K",
    "r": "R",
}
_TWO_CHAR_IPA = {"a\u028a": None, "\u0254\u026a": None, "t\u0283": "CH", "d\u0292": "JH"}
_SINGLE_CHARS = set(_IPA_TO_ARPABET_PLAUSIBLE)
_MARKS = {"\u02c8", "\u02cc", "\u02d0", "\u0306", ".", "(", ")", "-", "'", "/", "+"}


def parse_tsv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0] or not fields[1]:
            continue
        pairs.append((fields[0], fields[1]))
    return pairs


def classify_ipa(ipa: str) -> tuple[list[str], bool]:
    """Return (core IPA segments, whether all map via plausible single/two-char table)."""
    decomposed = unicodedata.normalize("NFD", ipa.strip("/"))
    core = [
        ch
        for ch in decomposed
        if ch != "\u0361" and unicodedata.category(ch) != "Mn" and ch not in _MARKS
    ]
    normalized = unicodedata.normalize("NFC", "".join(core))
    # greedy two-char then single-char matching
    i = 0
    mapped_ok = True
    while i < len(normalized):
        if normalized[i : i + 2] in _TWO_CHAR_IPA:
            i += 2
        elif normalized[i] in _SINGLE_CHARS:
            i += 1
        else:
            mapped_ok = False
            break
    return core, mapped_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--cmudict", type=Path, default=_PROJECT_ROOT / "vendor" / "cmudict.dict")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    content = args.tsv.read_bytes()
    digest = hashlib.sha256(content).hexdigest()

    pairs = parse_tsv(args.tsv)
    words_casefolded = [w.upper() for w, _ in pairs]

    segments: Counter[str] = Counter()
    normalizable_chars_only = 0
    for _, ipa in pairs:
        core, ok = classify_ipa(ipa)
        segments.update(core)
        normalizable_chars_only += ok

    cmudict = load_cmudict(args.cmudict, download=False)
    cmu_words = set(cmudict)

    present = sum(1 for w in words_casefolded if w in cmu_words)

    multiword = [w for w, _ in pairs if " " in w]
    hyphenated = [w for w, _ in pairs if "-" in w]
    mixed_case = [w for w, _ in pairs if w != w.lower()]

    payload = {
        "schema": "ailm-extlex-stage-stats-v1",
        "source": {
            "url": (
                "https://raw.githubusercontent.com/CUNY-CL/wikipron/"
                "d282e848a211ea31cfd730f0ced8bc8cdab9e83d/"
                "data/scrape/tsv/eng_latn_us_broad.tsv"
            ),
            "revision": "d282e848a211ea31cfd730f0ced8bc8cdab9e83d",
            "sha256": digest,
            "size_bytes": len(content),
        },
        "tsv": {
            "entry_count": len(pairs),
            "unique_grapheme_forms": len({w for w, _ in pairs}),
            "unique_grapheme_forms_lowercase": len(set(words_casefolded)),
            "multiword_entries": len(multiword),
            "hyphenated_entries": len(hyphenated),
            "mixed_case_entries": len(mixed_case),
            "ipa_segment_inventory": dict(sorted(segments.items())),
            "ipa_segment_type_count": len(segments),
        },
        "normalization_caveat": {
            "entries_requiring_ipa_to_arpabet_normalization": len(pairs),
            "fraction_of_all_entries": 1.0,
            "entries_with_plausible_direct_map_only": normalizable_chars_only,
            "entries_needing_residue_handling": len(pairs) - normalizable_chars_only,
        },
        "cmudict_comparison": {
            "cmudict_base_words": len(cmu_words),
            "overlap_count": present,
            "delta_absent_from_cmudict": len(pairs) - present,
            "rule": (
                "decision 79: candidate delta = entries whose words are absent "
                "from the pinned CMUdict revision (case-folded comparison)"
            ),
        },
    }

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
