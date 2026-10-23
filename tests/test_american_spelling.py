"""House style: American English, in prose and in identifiers alike.

Corpus data is exempt and must stay exempt. ``vendor/*.json`` holds Brown and
CMUdict, whose tokens include British spellings because that is what the text
says; rewriting them would corrupt the measurement rather than fix the style.
The distinction this file enforces is between *our* words and *the corpus's*.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Forms that are unambiguously British. Deliberately excludes ``analysis`` and
#: ``analyses`` (the noun is spelled the same in American English) and
#: ``towards`` (acceptable American usage), because a style guard that flags
#: correct text trains people to ignore it.
#: The ``-ise`` family, spelled out by ending rather than with ``\w*``. A
#: trailing wildcard is what makes a style guard cry wolf: ``optimis\w*`` flags
#: "optimistic" and ``realis\w*`` flags "realism", both correct American
#: English, and a rule that flags correct text trains people to ignore it.
_ISE_STEMS = (
    "normal",
    "marginal",
    "parameter",
    "memo",
    "random",
    "maxim",
    "minim",
    "summar",
    "recogn",
    "character",
    "initial",
    "general",
    "special",
    "real",
    "organ",
    "optim",
    "penal",
    "stabil",
    "factor",
    "discret",
    "quant",
)
_ISE_ENDINGS = ("ise", "ised", "ises", "ising", "isation", "isations")

#: Forms that are unambiguously British. Deliberately excludes ``analysis`` and
#: ``analyses`` (the noun is spelled the same in American English) and
#: ``towards`` (acceptable American usage).
BRITISH = tuple(f"{stem}{ending}" for stem in _ISE_STEMS for ending in _ISE_ENDINGS) + (
    r"behaviour\w*",
    r"colour\w*",
    r"favour\w*",
    r"neighbour\w*",
    r"labelling",
    r"labelled",
    r"modelling",
    r"modelled",
    r"catalogue",
    r"centre",
    r"centres",
    r"defence",
    r"licence",
    r"whilst",
    r"amongst",
    r"artefact\w*",
    # ``programme`` and ``programmes`` are British; ``programmed`` and
    # ``programmer`` are correct American words that share the prefix, so the
    # lookahead excludes exactly those two continuations.
    r"programmes?(?!r|d)",
    # The one entry where a British form is a prefix of a correct American word:
    # ``analyses`` is the ordinary noun plural. Anchored against a following
    # ``s`` so it flags the verb and leaves the noun alone. The cost is that the
    # British third-person ``analyses`` goes unflagged, which is the right trade
    # — a guard that fires on correct text gets ignored.
    r"analyse(?!s)",
)
#: No ``\b`` on either side. A word boundary cannot match between a letter and
#: an underscore, so ``\bnormalise\b`` misses ``normalise_text``,
#: ``summarise_paired_effect`` and ``stat_stage2_favoured_ratio`` — every
#: identifier this sweep renamed, which is most of what the sweep touched. A
#: guard against British spellings that cannot see snake_case is a guard against
#: British spellings in prose only, and it would have reported the tree clean
#: while every renamed function still carried the old spelling. Every entry in
#: BRITISH is a complete British form, so matching it as a substring cannot fire
#: on a correct American word.
PATTERN = re.compile("(" + "|".join(BRITISH) + ")", re.IGNORECASE)

#: English words that genuinely end in ``-ise`` and are not British variants.
#: This is the exception list a GENERAL rule needs. The stem list above can only
#: catch the British forms someone thought to enumerate: ``operationalisation``
#: sat in two manuscript sections and one decision record for as long as they
#: existed, because nobody had added ``operational`` as a stem. A rule that
#: matches the whole ``-ise`` family and subtracts the legitimate words catches
#: the next one too.
NOT_BRITISH_ISE = frozenset(
    [
        "advertise",
        "anywise",
        "arise",
        "chastise",
        "circumcise",
        "clockwise",
        "compromise",
        "concise",
        "denise",
        "exercise",
        "supervise",
        "demise",
        "despise",
        "devise",
        "disguise",
        "excise",
        "exercise",
        "expertise",
        "franchise",
        "guise",
        "improvise",
        "incise",
        "likewise",
        "merchandise",
        "noise",
        "otherwise",
        "paradise",
        "poise",
        "praise",
        "precise",
        "premise",
        "promise",
        "raise",
        "reprise",
        "revise",
        "rise",
        "supervise",
        "surmise",
        "surprise",
        "televise",
        "treatise",
        "wise",
        "comprise",
        "comprises",
        "comprised",
        "comprising",
        "imprecise",
    ]
)

#: Any ``-ise`` family word, to be filtered against the list above.
_ISE_FAMILY = re.compile(r"\b(\w{3,}?is(?:e|es|ed|ing|ation|ations))\b", re.IGNORECASE)


def _unlisted_ise_words(text: str) -> set[str]:
    """``-ise`` words that are neither legitimate nor already covered.

    Returns the offenders a stem-based list would miss.
    """
    found = set()
    for match in _ISE_FAMILY.findall(text):
        word = match.lower()
        if word.endswith("wise"):
            continue  # familywise, pairwise, pointwise: a suffix, not a variant
        # The family is <stem>ise / ises / ised / ising / isation. Removing the
        # ending leaves "...is", so the base word is that plus "e" — not that
        # minus "is", which turned "exercised" into "exerci" and flagged it.
        trimmed = re.sub(r"(es|ed|ing|ation|ations)$", "", word)
        base = f"{trimmed}e" if trimmed.endswith("is") else f"{trimmed}ise"
        if word in NOT_BRITISH_ISE or base in NOT_BRITISH_ISE:
            continue
        found.add(match)
    return found


SEARCHED_SUFFIXES = {".py", ".md", ".lean", ".toml"}
#: ``refs/`` holds vendored copies of other people's repositories, whose prose
#: is theirs to spell. Sweeping it would rewrite text this project does not own
#: and cannot regenerate.
EXEMPT_PARTS = {
    ".venv",
    "__pycache__",
    "output",
    "vendor",
    ".git",
    "handoff",
    "refs",
    # Untracked working space: lane reports quote reviewer text verbatim and
    # carry scratch copies of the tree taken before a sweep. Neither is a
    # publication surface, and rewriting a quotation to satisfy a style gate
    # would misquote it.
    "scratchpad",
}
#: Individual files exempted by relative path. Dated fleet-lane reports and
#: review snapshots live under ``handoff/reports/`` (an ``EXEMPT_PARTS`` entry),
#: so nothing at the root needs a by-name exemption today; the set stays so a
#: future exemption is a reviewable one-line change rather than a new mechanism.
EXEMPT_FILES: set[str] = set()

#: Places a British form appears *because it is being discussed*. Each entry is
#: a file whose text names the spelling it is replacing; listing the file rather
#: than suppressing the whole rule keeps the exemption auditable.
#: Files whose PROSE names a British spelling on purpose. Kept as short as
#: possible: code spans are already stripped, so an entry here means the file
#: discusses the spelling in running text.
DISCUSSES_THE_SPELLING = {"tests/test_american_spelling.py"}

#: Fixed names of external standards and tool-owned boilerplate that contain a
#: British form and cannot be renamed by this project. Each entry is a file plus
#: the exact text removed before matching; any other
#: British form in the same file still fails, so the exemption stays auditable.
TECHNICAL_TOKENS: dict[str, tuple[str, ...]] = {
    # W3C ARIA attribute; renaming it would break the accessibility checks.
    "tests/test_dashboard.py": ("aria-labelledby",),
    # Fixed sentence in the external Graft writeIndex template.
    "graft/INDEX.md": ("Small markdown nodes summarising this repo.",),
}


def _prose_only(text: str) -> str:
    """The text with inline code spans and fenced blocks removed.

    A document that *describes* a rename has to name the old identifier, and
    ``docs/decisions.md`` does exactly that in the entries recording this sweep.
    Exempting the whole file would stop checking its prose too, which is most of
    it. Code spans are where a quoted identifier belongs, so they are dropped and
    everything else is still scanned — an old spelling in a sentence still fails,
    the same spelling in backticks does not.
    """
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def _searched_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """Prune exempt directories before walking, preserving the declared scope."""
    found: list[Path] = []

    def fail_on_walk_error(error: OSError) -> None:
        raise error

    for directory, directories, filenames in os.walk(root, onerror=fail_on_walk_error):
        directories[:] = sorted(name for name in directories if name not in EXEMPT_PARTS)
        for name in sorted(filenames):
            path = Path(directory) / name
            if name not in EXEMPT_PARTS and path.suffix in SEARCHED_SUFFIXES and path.is_file():
                found.append(path)
    return sorted(found)


def test_the_file_walk_preserves_in_scope_files_and_excludes_only_declared_trees(tmp_path):
    included = {
        "README.md",
        "src/reader.py",
        "formal/Proof.lean",
        ".settings/style.toml",
        "graft/INDEX.md",
        "graft/src/reader.md",
        "docs/manuscript/section.md",
    }
    excluded = {"docs/diagram.png", "src/output/copy.py"}
    excluded.update(f"{name}/nested/copy.md" for name in EXEMPT_PARTS)
    for relative in included | excluded:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    assert {path.relative_to(tmp_path).as_posix() for path in _searched_files(tmp_path)} == included


def _spelling_offenders(root: Path = PROJECT_ROOT) -> dict[str, list[str]]:
    """Scan real files with the same path and exact-text exceptions as the gate."""
    offenders: dict[str, list[str]] = {}
    for path in _searched_files(root):
        relative = str(path.relative_to(root))
        if relative in DISCUSSES_THE_SPELLING or relative in EXEMPT_FILES:
            continue
        prose = _prose_only(path.read_text(encoding="utf-8"))
        for token in TECHNICAL_TOKENS.get(relative, ()):
            prose = prose.replace(token, "")
        found = set(PATTERN.findall(prose))
        # The general -ise rule runs on the manuscript only. It is the surface
        # where the convention is publication-visible, and a repository-wide
        # sweep would touch Lean proofs, test identifiers and API names, which
        # is a larger and riskier change than a spelling gate should force.
        # `docs/research_queue.md` tracks the wider sweep.
        if relative.startswith("docs/manuscript/"):
            found |= _unlisted_ise_words(prose)
        if found:
            offenders[relative] = sorted(found)[:6]
    return offenders


@pytest.mark.timeout(300)
def test_no_british_spelling_outside_the_corpus() -> None:
    offenders = _spelling_offenders()
    assert offenders == {}, f"British spellings outside the corpus: {offenders}"


def test_the_scan_actually_reaches_the_tree() -> None:
    """NON-VACUITY: an empty file list would make the check above pass forever."""
    files = _searched_files()
    assert len(files) > 100, f"only {len(files)} files searched; the walk is broken"
    assert any(p.suffix == ".md" for p in files)
    assert any(p.suffix == ".py" for p in files)
    assert any(p.suffix == ".lean" for p in files)


def test_the_pattern_catches_what_it_claims_to() -> None:
    """NON-VACUITY for the pattern, including the forms deliberately allowed."""
    for british in (
        "normalisation",
        "behaviour",
        "favoured",
        "modelling",
        "artefact",
        "whilst",
        "programme",
        "programmes",
    ):
        assert PATTERN.search(british), f"{british} should be flagged"
    # The case the boundary version missed entirely: identifiers, which is what
    # the sweep actually renamed.
    for identifier in (
        "normalise_text",
        "_summarise_paired_effect",
        "stat_stage2_favoured_ratio",
        "MARGINALISE_OVER_LETTERS",
        "initialisation_prior",
    ):
        assert PATTERN.search(identifier), f"{identifier} should be flagged inside an identifier"
    for american in (
        "normalization",
        "behavior",
        "favored",
        "modeling",
        "artifact",
        "while",
        "programmed",
        "programmer",
    ):
        assert not PATTERN.search(american), f"{american} must not be flagged"
    for allowed in ("analysis", "analyses", "towards", "centered"):
        assert not PATTERN.search(allowed), f"{allowed} is acceptable and must not be flagged"


def test_corpus_data_is_exempt_and_still_contains_british_words() -> None:
    """The exemption must be load-bearing, not decorative.

    If the vendored corpora stopped containing British spellings, the exemption
    would be protecting nothing and its removal should be considered. More
    usefully: this fails loudly if someone ever runs the style sweep over
    ``vendor/``, which would silently rewrite the measurements.
    """
    counts = PROJECT_ROOT / "vendor" / "brown_word_counts.json"
    if not counts.exists():
        return
    assert PATTERN.search(counts.read_text(encoding="utf-8")), (
        "the vendored Brown counts no longer contain any British spelling, which "
        "suggests the corpus was rewritten by a style sweep"
    )


def test_the_general_ise_rule_catches_what_the_stem_list_missed() -> None:
    """POSITIVE CONTROL for the rule added after a reviewer found what it misses.

    The stem-based list can only catch British forms someone enumerated.
    `operationalisation` sat in two manuscript sections and a decision record for
    as long as they existed, because nobody had added `operational` as a stem.

    NEGATIVE CONTROL: the legitimate `-ise` words must NOT fire, or the rule
    would be unusable and get suppressed. `exercised` in particular reduces to
    `exercis`, and an incorrect reduction turns it into `exerci` and flags it —
    which is exactly the bug this control caught while the rule was being
    written.
    """
    caught = _unlisted_ise_words(
        "The operationalisation memorised a regularisation and materialises harmonising."
    )
    assert caught == {
        "operationalisation",
        "memorised",
        "regularisation",
        "materialises",
        "harmonising",
    }

    legitimate = (
        "The exercised supervised analysis raises promises, revised and revising, "
        "a familywise pairwise pointwise surprise otherwise likewise."
    )
    assert _unlisted_ise_words(legitimate) == set(), (
        f"the rule fires on legitimate words: {_unlisted_ise_words(legitimate)}"
    )


@pytest.mark.parametrize(
    ("relative", "text", "expected"),
    [
        ("graft/INDEX.md", "Small markdown nodes summarising this repo.", []),
        (
            "graft/INDEX.md",
            "Small markdown nodes summarising this repo.\nWe are summarising results.",
            ["summarising"],
        ),
        (
            "graft/INDEX.md",
            "Small markdown nodes summarising this repo.\nThe colour changes.",
            ["colour"],
        ),
        ("graft/INDEX.md", "Small markdown nodes summarising this project.", ["summarising"]),
        ("README.md", "Small markdown nodes summarising this repo.", ["summarising"]),
        ("graft/src/reader.md", "Small markdown nodes summarising this repo.", ["summarising"]),
        (
            "docs/manuscript/section.md",
            "Small markdown nodes summarising this repo.",
            ["summarising"],
        ),
        ("tests/test_dashboard.py", "aria-labelledby", []),
        ("tests/test_dashboard.py", "aria-labelledby and colour", ["colour"]),
        ("README.md", "aria-labelledby", ["labelled"]),
    ],
)
def test_exact_external_text_exceptions_are_path_and_text_scoped(
    tmp_path, relative, text, expected
):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    assert _searched_files(tmp_path) == [path]
    assert _spelling_offenders(tmp_path) == ({relative: expected} if expected else {})


def test_corpus_exemption_does_not_hide_authored_or_graft_prose(tmp_path):
    for relative in ("vendor/corpus.md", "graft/INDEX.md", "graft/src/reader.md", "src/reader.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("The colour changes.\n", encoding="utf-8")
    assert _spelling_offenders(tmp_path) == {
        "graft/INDEX.md": ["colour"],
        "graft/src/reader.md": ["colour"],
        "src/reader.py": ["colour"],
    }
