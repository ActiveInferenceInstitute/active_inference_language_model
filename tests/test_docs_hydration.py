"""Documentation numbers come from the ledger, in both directions.

`tests/test_claims_discipline.py` checks one direction: every `claim:<label>` in
prose resolves. That is necessary and not sufficient. It says nothing about the
numbers a document types *instead* of tagging, and those are the ones that
drift — decision 13's vocabulary table disagreed with all four of the node
counts it summarized, and with two of the four coverage figures recorded in the
same rows, while every tag in the file resolved perfectly.

So these tests close the other direction. A distinctive numeral in prose that
exactly equals a recorded claim is a copy of that claim, and copies go stale.
"""

from __future__ import annotations

import importlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from ailm.benchmarks.constants import SERVED_PROBES
from ailm.channel_registry import (
    CROSS_LEVEL_CHANNELS,
    ChannelOutcome,
    ChannelSourceKind,
    channel_by_key,
)
from ailm.docs_hydration import (
    COINCIDENTAL_MATCHES,
    LITERAL_EXEMPTIONS,
    find_absorbed_tags,
    find_hardcoded_values,
    hydrate,
    scan_repository,
    strip_code,
)
from ailm.manuscript_variables import (
    LEVEL_QUALIFYING_VERDICT_FLAGS,
    ManuscriptVariableError,
    _quantified_clause,
    canonical_stage5_passes,
    classify_cross_level_channels,
    generate_manuscript_variables,
    qualified_verdict_flags,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _renderer_module():
    """Load the real renderer so publication primitives can be race-tested."""
    return importlib.import_module("scripts.z_render_docs")


@pytest.fixture(scope="module")
def claims() -> dict:
    return json.loads((REPO_ROOT / "data" / "claims.json").read_text(encoding="utf-8"))["claims"]


# ── The detector ─────────────────────────────────────────────────────────────


def test_no_document_types_a_number_the_ledger_already_holds(claims: dict):
    """The check that would have caught decision 13.

    Every hit is a numeral a reader will believe and nothing will update.
    """
    report = scan_repository(REPO_ROOT, claims)
    assert not report, "documents carrying hardcoded ledger values: " + "; ".join(
        f"{path}:{hit.line} {hit.raw} == claim:{hit.label}"
        for path, hits in report.items()
        for hit in hits
    )


def test_the_detector_fires_on_an_injected_hardcode(claims: dict):
    """POSITIVE CONTROL. A green result above must mean the scan works.

    Without this, deleting the body of `find_hardcoded_values` would make the
    repository check pass, which is the failure mode this project's testing
    doctrine names first.
    """
    label, entry = next(
        (label, entry)
        for label, entry in claims.items()
        if isinstance(entry.get("value"), float) and abs(entry["value"]) > 1
    )
    injected = f"The measured value was {entry['value']:.10g} on the held-out split."
    hits = find_hardcoded_values(injected, claims, path="fake.md")
    assert hits, f"the scanner missed an exact copy of claim:{label}"
    assert hits[0].label in claims


def test_the_detector_ignores_a_coincidental_small_value(claims: dict):
    """NEGATIVE CONTROL. A ledger of 1,241 claims collides with ordinary prose.

    A detector that flagged "two hidden factors" because some claim equals 2
    would be suppressed wholesale, and a suppressed check constrains nothing.
    """
    assert not find_hardcoded_values("There are 2 hidden factors and 8 case roles.", claims)
    assert not find_hardcoded_values("Roughly 0.13% of the ceiling.", claims)


def test_the_detector_ignores_code(claims: dict):
    """A numeral inside code is a literal the code needs, not a claim in prose."""
    label, entry = next(
        (label, entry)
        for label, entry in claims.items()
        if isinstance(entry.get("value"), float) and abs(entry["value"]) > 1
    )
    literal = f"{entry['value']:.10g}"
    assert not find_hardcoded_values(f"```\nx = {literal}\n```", claims)
    assert not find_hardcoded_values(f"Set it with `--alpha {literal}`.", claims)
    # NEGATIVE CONTROL: the same numeral in prose IS reported, so the exclusion
    # is about code and not about the number.
    assert find_hardcoded_values(f"It measured {literal} on the split.", claims)


def test_stripping_code_preserves_line_numbers():
    """A report is what someone edits from, so its line numbers must be true."""
    text = "prose\n```\ncode\nmore\n```\nafter `inline` end\n"
    stripped = strip_code(text)
    assert stripped.count("\n") == text.count("\n")
    assert stripped.splitlines()[5].startswith("after")


# ── The exemptions ───────────────────────────────────────────────────────────


def test_every_exemption_states_a_reason():
    """A suppression with no reason is how a check stops meaning anything."""
    for path, reason in LITERAL_EXEMPTIONS.items():
        assert len(reason.split()) >= 12, f"{path}'s exemption is not an explanation"
    for key, reason in COINCIDENTAL_MATCHES.items():
        assert len(reason.split()) >= 6, f"{key} is exempted without saying why"


def test_every_exemption_still_applies_to_a_real_file():
    """An exemption for a file that moved is dead weight that hides the next one."""
    for path in LITERAL_EXEMPTIONS:
        assert (REPO_ROOT / path).exists(), f"exemption names a missing file: {path}"
    for path, literal, anchor in COINCIDENTAL_MATCHES:
        source = REPO_ROOT / path
        assert source.exists(), f"exemption names a missing file: {path}"
        # Checked against the STRIPPED view, because that is what the scanner
        # matches against. An anchor that only exists inside a code span would
        # pass a raw-text check here and never fire in find_hardcoded_values.
        lines = strip_code(source.read_text(encoding="utf-8")).splitlines()
        matching = [line for line in lines if anchor in line and literal in line]
        # Exactly one, not at least one. An anchor matching two lines would exempt
        # both, and the second is by definition unreviewed -- the same silent
        # over-suppression the old line-number key risked, arriving a different way.
        assert len(matching) == 1, (
            f"exemption ({path}, {literal!r}, {anchor!r}) matches {len(matching)} lines, "
            f"not one. Zero means the sentence was rewritten and the coincidence needs "
            f"a fresh judgment; more than one means the anchor is not distinctive and "
            f"is suppressing a numeral nobody reviewed."
        )


# ── Hydration ────────────────────────────────────────────────────────────────


def test_hydration_substitutes_every_tag(claims: dict):
    label = "stage3_trie_nodes_v2000"
    assert label in claims
    rendered = hydrate(f"The trie has claim:{label} nodes.", claims)
    assert "claim:" not in rendered
    assert str(claims[label]["value"]) in rendered


def test_hydration_refuses_an_unresolvable_tag(claims: dict):
    """An unresolved tag rendered into a document reads as a fabricated result.

    NEGATIVE CONTROL: with `strict=False` the same input passes through, so the
    refusal is a decision rather than an inability to parse.
    """
    with pytest.raises(KeyError):
        hydrate("value claim:no_such_label_anywhere here", claims)
    assert "claim:no_such_label_anywhere" in hydrate(
        "value claim:no_such_label_anywhere here", claims, strict=False
    )


def test_no_tag_swallows_the_word_after_it(claims: dict):
    """`claim:label-word` parses as one label and names no claim.

    Three of these were introduced in one mechanical conversion, all of the form
    `2,000-word`. The hydrator refuses them, but only at render time and only
    for the tree it renders, so the class is checked directly here.
    """
    for source in sorted((REPO_ROOT / "docs").rglob("*.md")):
        absorbed = find_absorbed_tags(source.read_text(encoding="utf-8"), claims)
        assert not absorbed, f"{source.relative_to(REPO_ROOT)}: {absorbed}"

    # POSITIVE CONTROL: the detector must fire on the shape it exists for.
    label = next(iter(claims))
    assert find_absorbed_tags(f"the claim:{label}-word rung", claims) == [f"{label}-word"]


@pytest.mark.timeout(240)
def test_document_hydration_needs_no_third_party_runtime():
    """A real isolated interpreter must hydrate without site packages.

    This fails if formatting imports manuscript generation, which initializes
    the benchmark/model stack before an ordinary documentation CLI can start.
    """
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from ailm.docs_hydration import hydrate
assert hydrate('Recorded: claim:alpha', {'alpha': {'value': 12345}}) == 'Recorded: 12,345'
try:
    hydrate('claim:unrecorded', {})
except KeyError as error:
    assert 'has no ledger entry' in str(error)
else:
    raise AssertionError('an unrecorded claim was accepted')
assert not {'jax', 'numpy', 'pymdp', 'scipy', 'yaml'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(REPO_ROOT / "src")],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_the_hydrated_tree_can_be_generated_and_checked_from_a_clean_checkout(tmp_path):
    """A fresh checkout needs no ignored output from an earlier local run.

    The test generates a disposable mirror, checks it through the renderer's
    own ``--check`` operation, then plants drift and proves the check fails.
    Each real CLI invocation excludes site packages: the renderer is stdlib-only,
    so repeated environment initialization is unrelated to this publication check.
    """
    output_dir = tmp_path / "hydrated-docs"
    command = [
        sys.executable,
        "-I",
        "-S",
        str(REPO_ROOT / "scripts" / "z_render_docs.py"),
        "--output-dir",
        str(output_dir),
    ]
    generated = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, (
        f"hydrated documentation generation failed:\n{generated.stderr}"
    )
    assert output_dir.is_dir()
    assert not output_dir.is_symlink(), "renderer published a symlink instead of a real directory"

    checked = subprocess.run(
        [*command, "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, f"freshly hydrated documentation is stale:\n{checked.stderr}"

    # An unchanged write must not publish another full retained generation.
    identity = output_dir.stat().st_ino
    retained = set(tmp_path.iterdir())
    repeated = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert repeated.returncode == 0, repeated.stderr
    assert output_dir.stat().st_ino == identity
    assert set(tmp_path.iterdir()) == retained

    planted_stale = output_dir / "README.md"
    planted_stale.write_text(
        planted_stale.read_text(encoding="utf-8") + "\nplanted stale content\n",
        encoding="utf-8",
    )
    rejected = subprocess.run(
        [*command, "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 1, (
        "renderer accepted a planted stale hydrated document; its check cannot fail"
    )
    assert "README.md" in rejected.stderr

    repaired = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert repaired.returncode == 0, repaired.stderr
    assert output_dir.stat().st_ino != identity, "changed content did not publish a new generation"
    assert "planted stale content" not in planted_stale.read_text(encoding="utf-8")

    overlap = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(REPO_ROOT / "scripts" / "z_render_docs.py"),
            "--output-dir",
            str(REPO_ROOT / "docs"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert overlap.returncode == 2, "renderer would overwrite authored documentation"
    assert "overlaps authored documentation" in overlap.stderr


def test_renderer_refuses_descendant_symlink_escape_without_partial_publication(tmp_path):
    output_dir = tmp_path / "hydrated-docs"
    output_dir.mkdir()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    sentinel = escaped / "README.md"
    sentinel.write_text("outside sentinel\n", encoding="utf-8")
    (output_dir / "docs").symlink_to(escaped, target_is_directory=True)
    prior = output_dir / "AGENTS.md"
    prior.write_text("prior generation\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "z_render_docs.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "outside sentinel\n"
    assert prior.read_text(encoding="utf-8") == "prior generation\n"


def test_renderer_refuses_an_unmanaged_publication_symlink(tmp_path):
    output_dir = tmp_path / "hydrated-docs"
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    sentinel = escaped / "README.md"
    sentinel.write_text("outside sentinel\n", encoding="utf-8")
    output_dir.symlink_to(escaped.name, target_is_directory=True)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "z_render_docs.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unmanaged symlink" in result.stderr
    assert output_dir.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "outside sentinel\n"


def test_renderer_refuses_invalid_existing_tree_before_replacing_any_file(tmp_path):
    output_dir = tmp_path / "hydrated-docs"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "z_render_docs.py"),
        "--output-dir",
        str(output_dir),
    ]
    generated = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stderr
    prior = (output_dir / "AGENTS.md").read_bytes()
    blocked = output_dir / "docs" / "decisions.md"
    saved = tmp_path / "saved-decisions.md"
    blocked.replace(saved)
    blocked.mkdir()

    rejected = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)

    assert rejected.returncode != 0
    assert "unexpected directory" in rejected.stderr
    assert (output_dir / "AGENTS.md").read_bytes() == prior
    assert blocked.is_dir()


@pytest.mark.parametrize(
    "relative", ["docs/manuscript/retired-section-control.md", "docs/retired-control/old.md"]
)
def test_renderer_check_names_retired_sources_without_removing_them(tmp_path, relative):
    """An obsolete rendered section is stale output, with an actionable path."""
    output_dir = tmp_path / "hydrated-docs"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "z_render_docs.py"),
        "--output-dir",
        str(output_dir),
    ]
    generated = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stderr
    retired = output_dir / relative
    retired.parent.mkdir(parents=True, exist_ok=True)
    retired.write_text("prior published section\n", encoding="utf-8")
    prior = (output_dir / "README.md").read_bytes()
    checked = subprocess.run(
        [*command, "--check"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert checked.returncode == 1, checked.stderr
    assert "no current source" in checked.stderr
    assert retired.relative_to(output_dir).as_posix() in checked.stderr
    assert retired.read_text(encoding="utf-8") == "prior published section\n"
    assert (output_dir / "README.md").read_bytes() == prior
    refused_write = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert refused_write.returncode == 2, refused_write.stderr
    assert "no current source" in refused_write.stderr
    assert retired.read_text(encoding="utf-8") == "prior published section\n"
    assert (output_dir / "README.md").read_bytes() == prior


def test_real_directory_publication_is_reader_atomic_when_exchange_is_available(tmp_path):
    """Native exchange gives readers a complete old or new real directory."""
    renderer = _renderer_module()
    target = tmp_path / "hydrated-docs"
    expected = {Path("README.md")}

    def generation(index: int) -> Path:
        path = tmp_path / f".hydrated-docs.generation.{index:04d}"
        path.mkdir()
        (path / "README.md").write_text(f"generation-{index % 2}\n", encoding="utf-8")
        return path

    initial = renderer._publish_tree(generation(0), target, expected)
    assert initial.committed and initial.reader_atomic
    assert target.is_dir() and not target.is_symlink()

    stop = threading.Event()
    observations: list[str] = []
    failures: list[BaseException] = []

    def observe() -> None:
        while not stop.is_set():
            try:
                observations.append((target / "README.md").read_text(encoding="utf-8"))
            except BaseException as error:
                failures.append(error)

    observer = threading.Thread(target=observe)
    observer.start()
    results = []
    for index in range(1, 201):
        result = renderer._publish_tree(generation(index), target, expected)
        assert result.committed
        results.append(result)
        time.sleep(0)
    stop.set()
    observer.join(timeout=10)

    assert not observer.is_alive()
    assert observations, "concurrent observer never read the public path"
    if all(result.reader_atomic for result in results):
        assert failures == []
    else:
        assert all(isinstance(error, FileNotFoundError) for error in failures)
    assert set(observations) <= {"generation-0\n", "generation-1\n"}
    assert target.is_dir() and not target.is_symlink()
    assert not any(path.is_symlink() for path in tmp_path.rglob("*"))

    # POSITIVE CONTROL: the same observer detects the historical two-rename
    # gap, so an empty failure list above is evidence rather than scheduler luck.
    gap_target = tmp_path / "gap-docs"
    gap_target.mkdir()
    (gap_target / "README.md").write_text("before\n", encoding="utf-8")
    gap_backup = tmp_path / "gap-backup"
    gap_replacement = tmp_path / "gap-replacement"
    gap_replacement.mkdir()
    (gap_replacement / "README.md").write_text("after\n", encoding="utf-8")
    gap_seen = threading.Event()
    gap_stop = threading.Event()

    def observe_gap() -> None:
        while not gap_stop.is_set():
            try:
                (gap_target / "README.md").read_bytes()
            except FileNotFoundError:
                gap_seen.set()

    gap_observer = threading.Thread(target=observe_gap)
    gap_observer.start()
    gap_target.rename(gap_backup)
    time.sleep(0.02)
    gap_replacement.rename(gap_target)
    gap_stop.set()
    gap_observer.join(timeout=10)
    assert gap_seen.is_set(), "observer failed to detect a planted two-rename gap"


def test_any_publication_symlink_is_refused_in_write_and_check_modes(tmp_path):
    """A prior managed-link shape has no compatibility path into publication."""
    output_dir = tmp_path / "hydrated-docs"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "z_render_docs.py"),
        "--output-dir",
        str(output_dir),
    ]
    generated = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stderr

    unsupported_generation = tmp_path / ".hydrated-docs.generation.unsupported"
    output_dir.rename(unsupported_generation)
    output_dir.symlink_to(unsupported_generation.name, target_is_directory=True)
    before = output_dir.readlink()

    for arguments in (command, [*command, "--check"]):
        rejected = subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert rejected.returncode == 2
        assert "must be a real directory" in rejected.stderr
        assert output_dir.is_symlink() and output_dir.readlink() == before


# ── The manuscript ───────────────────────────────────────────────────────────


def _manuscript_measured_decimals(text: str) -> tuple[str, list[str]]:
    """Classify complete DOI link destinations, retaining labels and adjacent prose."""
    import re

    text = strip_code(text)
    text = re.sub(r"\{\{[A-Z0-9_]+\}\}", " ", text)
    text = re.sub(r"\$[^$\n]*\$", " ", text)
    text = re.sub(
        r"(?<![!\\])(\[(?:[^\]\\\n]|\\\\)+\]\()"
        r"https://doi\.org/10\.[0-9]{4,9}/[A-Za-z0-9][-._;/:A-Za-z0-9]*(\))",
        r"\1DOI_IDENTIFIER\2",
        text,
    )
    text = re.sub(
        r"\b(?:Supplement|Section|Figure|Table|Appendix|v|version)\s*\d+\.\d+",
        " ",
        text,
        flags=re.I,
    )
    decimal = re.compile(r"(?<![\w.$])(\d+\.\d{2,})(?![\w.])")
    return text, decimal.findall(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("[release DOI](https://doi.org/10.5281/zenodo.22409419)", []),
        ("Bare 10.5281 is not an identifier link.", ["10.5281"]),
        ("[DOI](https://doi.org/10.5281/zenodo.22409419) gain 1.23", ["1.23"]),
        ("[gain 1.23](https://doi.org/10.5281/zenodo.22409419)", ["1.23"]),
        ("[bad DOI](https://doi.org/10.5281/)", ["10.5281"]),
        ("[bad resolver](https://doi.org.example/10.5281/zenodo.22409419)", ["10.5281"]),
        ("[other URL](https://example.org/10.5281/zenodo.22409419)", ["10.5281"]),
        (r"\[escaped label](https://doi.org/10.5281/zenodo.22409419)", ["10.5281"]),
        (r"[label\](https://doi.org/10.5281/zenodo.22409419)", ["10.5281"]),
        (r"[label\\](https://doi.org/10.5281/zenodo.22409419)", []),
        (r"[label\\\](https://doi.org/10.5281/zenodo.22409419)", ["10.5281"]),
        (r"[label\\\\](https://doi.org/10.5281/zenodo.22409419)", []),
        ("https://doi.org/10.5281/zenodo.22409419", ["10.5281"]),
        ("Measured gain 1.23", ["1.23"]),
    ),
)
def test_manuscript_decimal_classifier_preserves_measurements(text, expected):
    _, found = _manuscript_measured_decimals(text)
    assert found == expected


def test_the_manuscript_types_no_measured_numeral_at_all():
    """The manuscript's rule is stricter than the documentation's.

    Documentation may carry a literal that no ledger entry holds — a corpus
    year, a section count, an illustrative figure. The manuscript may not carry
    a *measured* one at all: every quantity it reports is a `{{TOKEN}}` resolved
    at render time, so a numeral in its prose is either a token that failed to
    resolve or a number nothing will ever update.

    The check is for decimals rather than for ledger membership, because in the
    manuscript the defect is typing the number, not typing a number that happens
    to match. Structural counts stay legal: they are integers the code owns.

    NEGATIVE CONTROL: the scan must see actual prose, or a mistake in the
    stripping would make this pass on an empty string.
    """
    manuscript = REPO_ROOT / "docs" / "manuscript"

    scanned_chars = 0
    offenders: dict[str, list[str]] = {}
    for source in sorted(manuscript.glob("*.md")):
        if source.name in ("AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"):
            continue  # instructions to authors, not the paper
        # A locator is not a measurement: "CEREBRUM v1.2, Supplement 2.12"
        # names a place in a cited document, and no ledger entry could update
        # it. The exclusion is by the word in front of the numeral, so it
        # cannot be widened accidentally into "any decimal we would rather not
        # tokenize".
        text, found = _manuscript_measured_decimals(source.read_text(encoding="utf-8"))
        scanned_chars += len(text)
        if found:
            offenders[source.name] = sorted(set(found))

    assert scanned_chars > 100_000, (
        f"only {scanned_chars} characters scanned; the stripping has eaten the paper"
    )
    assert not offenders, f"manuscript prose types measured values: {offenders}"


# ── Spelled-out architecture counts ──────────────────────────────────────────


def test_the_papers_spelled_counts_match_the_code():
    """Architecture counts are injected from registries, including word forms."""
    from ailm.cases import CaseRole
    from ailm.levels.narrative import N_PHASES
    from ailm.parameters import GNN_SPEC_FILES

    variables = generate_manuscript_variables(REPO_ROOT)
    level_ids = {int(name[1]) for name in GNN_SPEC_FILES}
    assert int(variables["BUILT_LEVEL_COUNT"]) == len(level_ids)
    assert variables["BUILT_LEVEL_COUNT_WORD"] == "six"
    assert int(variables["STAGE0_CASE_ROLES"]) == len(CaseRole)
    assert int(variables["NARRATIVE_PHASE_COUNT"]) == N_PHASES

    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "docs" / "manuscript").glob("[0-9]*.md"))
    )
    assert "{{BUILT_LEVEL_COUNT}}" in corpus
    assert "{{BUILT_LEVEL_COUNT_WORD}}" in corpus
    assert "six levels" not in corpus.casefold()


# ── Claims the paper states in words, checked against the numbers ────────────


def test_the_channel_sign_sentence_matches_the_recorded_margins(claims: dict):
    """The prose tallies the typed intervention registry, not factorial cells."""
    expected = {channel.key: channel.expected_outcome.value for channel in CROSS_LEVEL_CHANNELS}
    assert classify_cross_level_channels(claims) == expected
    variables = generate_manuscript_variables(REPO_ROOT)
    outcome_counts = {
        outcome: sum(value == outcome for value in expected.values())
        for outcome in (
            ChannelOutcome.HELPS.value,
            ChannelOutcome.HARMS.value,
            ChannelOutcome.PRIOR_ONLY.value,
            ChannelOutcome.MIXED_DIRECTION.value,
        )
    }
    assert variables["CHANNEL_COUNT"] == str(len(expected))
    assert variables["CHANNEL_HELPING_COUNT"] == str(outcome_counts[ChannelOutcome.HELPS.value])
    assert variables["CHANNEL_HARMING_COUNT"] == str(outcome_counts[ChannelOutcome.HARMS.value])
    assert variables["CHANNEL_PRIOR_ONLY_COUNT"] == str(
        outcome_counts[ChannelOutcome.PRIOR_ONLY.value]
    )
    assert variables["CHANNEL_MIXED_DIRECTION_COUNT"] == str(
        outcome_counts[ChannelOutcome.MIXED_DIRECTION.value]
    )
    source_counts = {
        kind: sum(channel.source_kind is kind for channel in CROSS_LEVEL_CHANNELS)
        for kind in ChannelSourceKind
    }
    assert variables["MODEL_OUTPUT_INTERVENTION_COUNT"] == str(
        source_counts[ChannelSourceKind.MODEL_OUTPUT]
    )
    assert variables["PRIVILEGED_ORACLE_INTERVENTION_COUNT"] == str(
        source_counts[ChannelSourceKind.PRIVILEGED_ORACLE]
    )
    results = (REPO_ROOT / "docs" / "manuscript" / "04_results.md").read_text(encoding="utf-8")
    for token in (
        "{{CHANNEL_COUNT_WORD}}",
        "{{CHANNEL_OUTCOME_SUMMARY}}",
        "{{CHANNEL_SOURCE_SUMMARY}}",
    ):
        assert token in results

    # POSITIVE CONTROLS: each seam reads its own evidence family.  Flipping one
    # relationship changes that seam alone instead of leaving a generic mixed-
    # sign check green.
    mutations = {
        "generation": ("generation_tv_improvement", -1),
        "lexical": ("lexical_best_margin_pts", -1),
        "letter_coupling": ("coupling_margin_pts", 1),
        "narrative_prior": ("stage6_opening_interrogative_shift", 0),
        "clause_word": ("clause_word_confirms_harm", 1),
    }
    for seam, (label, value) in mutations.items():
        changed = deepcopy(claims)
        changed[label]["value"] = value
        outcome = classify_cross_level_channels(changed)
        assert outcome[seam] == "changed"
        assert {name: state for name, state in outcome.items() if name != seam} == {
            name: state for name, state in expected.items() if name != seam
        }

    # The category is determined by both estimands.  Plant the retired
    # mixed-direction pattern under a still-helpful aggregate and require the
    # classifier to distinguish it from the current concordant result.
    changed = deepcopy(claims)
    changed["clause_word_best_improved_tokens"]["value"] = 1
    changed["clause_word_best_degraded_tokens"]["value"] = 2
    assert classify_cross_level_channels(changed)["clause_word"] == "mixed-direction"


def test_channel_registry_rejects_duplicate_identity_and_evidence_rules():
    with pytest.raises(ValueError, match="duplicate keys"):
        channel_by_key((*CROSS_LEVEL_CHANNELS, CROSS_LEVEL_CHANNELS[0]))
    duplicate_rule = replace(
        CROSS_LEVEL_CHANNELS[1],
        key="different_key",
        evidence_rule=CROSS_LEVEL_CHANNELS[0].evidence_rule,
    )
    with pytest.raises(ValueError, match="duplicate evidence rules"):
        channel_by_key((*CROSS_LEVEL_CHANNELS, duplicate_rule))


def test_clause_word_registry_cannot_misattribute_the_oracle_to_l5():
    channels = channel_by_key()
    clause_word = channels["clause_word"]
    assert clause_word.sender == "punctuation-type oracle"
    assert clause_word.source_kind is ChannelSourceKind.PRIVILEGED_ORACLE
    assert sum(
        channel.source_kind is ChannelSourceKind.MODEL_OUTPUT for channel in channels.values()
    ) == len(channels) - sum(
        channel.source_kind is ChannelSourceKind.PRIVILEGED_ORACLE for channel in channels.values()
    )

    false_sender = replace(clause_word, sender="L5")
    with pytest.raises(ValueError, match="cannot claim a model level"):
        channel_by_key(
            tuple(
                false_sender if channel.key == "clause_word" else channel
                for channel in CROSS_LEVEL_CHANNELS
            )
        )

    false_source_kind = replace(
        clause_word,
        source_kind=ChannelSourceKind.MODEL_OUTPUT,
    )
    with pytest.raises(ValueError, match="punctuation-defined reference type"):
        channel_by_key(
            tuple(
                false_source_kind if channel.key == "clause_word" else channel
                for channel in CROSS_LEVEL_CHANNELS
            )
        )


@pytest.mark.parametrize(
    ("count", "expected"),
    (
        (0, "none are mixed-direction"),
        (1, "one is mixed-direction"),
        (2, "two are mixed-direction"),
    ),
)
def test_generated_count_clauses_agree_at_zero_one_and_two(count: int, expected: str):
    """Positive controls for the grammar branch used by outcome/source tokens."""
    assert (
        _quantified_clause(
            count,
            singular_verb="is",
            plural_verb="are",
            complement="mixed-direction",
        )
        == expected
    )


def test_generated_count_clause_rejects_a_negative_count():
    with pytest.raises(ManuscriptVariableError, match="cannot be negative"):
        _quantified_clause(
            -1,
            singular_verb="is",
            plural_verb="are",
            complement="mixed-direction",
        )


def test_the_conclusions_qualified_verdict_count_matches_the_ledger(claims: dict):
    """Qualifications count LEVELS, and since decision 114 that is Stage 6 alone.

    Both guards stay evaluated. The clause guard now protects the retained
    case-only ablation rather than a level's verdict, so it leaves the count
    while staying in the dict; a version of this test that summed the whole dict
    would have kept subtracting a level that clears its own bar.
    """
    conclusion = (REPO_ROOT / "docs" / "manuscript" / "08_conclusion.md").read_text(
        encoding="utf-8"
    )
    flags = qualified_verdict_flags(claims)
    assert flags == {
        "stage5_nonpassing_unresolved": True,
        "stage6_primary_nonconfirmatory": True,
    }
    assert LEVEL_QUALIFYING_VERDICT_FLAGS == ("stage6_primary_nonconfirmatory",)
    assert set(LEVEL_QUALIFYING_VERDICT_FLAGS) <= set(flags), (
        "a level-qualifying flag names a guard that qualified_verdict_flags does "
        "not compute, so the count would silently read a missing key"
    )
    variables = generate_manuscript_variables(REPO_ROOT)
    assert variables["QUALIFIED_VERDICT_COUNT"] == str(
        sum(flags[key] for key in LEVEL_QUALIFYING_VERDICT_FLAGS)
    )
    assert "{{QUALIFIED_VERDICT_COUNT_WORD}}" in conclusion
    assert "{{BUILT_LEVEL_COUNT_WORD}}" in conclusion


def test_the_canonical_level_5_is_checked_before_it_is_counted_as_passing(claims: dict):
    """The passing-level count asserts the promotion rather than assuming it.

    Decision 114 promoted the widened-observation clause arm on the strength of a
    registered gate that could have failed. Its falsifier is the mixture losing
    its bar, either document-resampled gain interval coming to span zero, or the
    detectability control going dead. Each is planted here, because a fail-closed
    check nobody has seen fail is a check nobody has tested.
    """
    assert canonical_stage5_passes(claims)

    for label, value in (
        # the mixture stops beating its refitted bar
        ("stage5w_mixture_perplexity", claims["stage5w_typefree_perplexity"]["value"] + 0.01),
        # a gain interval comes to span zero, on either scale
        ("stage5w_oracle_gain_sentence_ci_low", -0.001),
        ("stage5w_oracle_gain_position_ci_low", -0.001),
        # the detectability control stops firing
        ("stage5w_control_classification_pct", 90.0),
    ):
        broken = deepcopy(claims)
        broken[label]["value"] = value
        assert not canonical_stage5_passes(broken), (
            f"the canonical Level 5 check survives a broken {label}; it would "
            "let the abstract count a level that no longer clears its bar"
        )

    stage5_changed = deepcopy(claims)
    stage5_changed["stage5_mixture_perplexity"]["value"] = (
        stage5_changed["stage5_typefree_perplexity"]["value"] - 0.01
    )
    assert qualified_verdict_flags(stage5_changed)["stage5_nonpassing_unresolved"] is False

    stage5_dead_control = deepcopy(claims)
    stage5_dead_control["stage5_control_classification_pct"]["value"] = 90.0
    assert qualified_verdict_flags(stage5_dead_control)["stage5_nonpassing_unresolved"] is False

    stage5_resolved_gain = deepcopy(claims)
    stage5_resolved_gain["stage5_oracle_gain_sentence_ci_low"]["value"] = 0.001
    assert qualified_verdict_flags(stage5_resolved_gain)["stage5_nonpassing_unresolved"] is False

    stage6_changed = deepcopy(claims)
    stage6_changed["cluster_stage6_logloss_ci_low"]["value"] = 0.001
    assert qualified_verdict_flags(stage6_changed)["stage6_primary_nonconfirmatory"] is False

    stage6_significant = deepcopy(claims)
    stage6_significant["cluster_stage6_logloss_p"]["value"] = 0.049
    assert qualified_verdict_flags(stage6_significant)["stage6_primary_nonconfirmatory"] is False

    stage6_negative = deepcopy(claims)
    stage6_negative["stage6_logloss_improvement_nats"]["value"] = -0.001
    assert qualified_verdict_flags(stage6_negative)["stage6_primary_nonconfirmatory"] is False


def test_the_benchmark_interval_kind_agrees_with_its_confidence():
    """The supplement's "exact interval" coverage must be the scorer's.

    The coverage is a token now, but the token and the scorer are two places,
    and a Wilson interval built on a hard-coded z is a third. If
    `DIFFERENCE_CONFIDENCE` ever moves, `z = 1.96` does not follow it, and the
    supplement would print a coverage the intervals do not have.
    """
    import inspect
    from math import isclose

    from scipy.stats import norm

    from ailm.benchmarks import scoring
    from ailm.statistics import DIFFERENCE_CONFIDENCE

    paired = inspect.signature(scoring.paired_difference_interval)
    assert paired.parameters["confidence"].default == DIFFERENCE_CONFIDENCE, (
        "the paired interval's default coverage no longer matches "
        "DIFFERENCE_CONFIDENCE, which the supplement now prints"
    )

    wilson_z = inspect.signature(scoring.wilson_interval).parameters["z"].default
    expected_z = norm.ppf(1.0 - (1.0 - DIFFERENCE_CONFIDENCE) / 2.0)
    assert isclose(wilson_z, expected_z, abs_tol=0.005), (
        f"the Wilson interval uses z={wilson_z} while the declared coverage "
        f"implies z={expected_z:.4f}; the two describe different intervals"
    )


def test_the_context_window_table_rungs_match_the_recorded_grid(claims: dict):
    """The k rows are generated from the recorded grid, never typed in source."""
    import re

    grid = [int(k) for k in str(claims["bench_ctx_k_grid"]["value"]).split(",")]
    supplement = (
        REPO_ROOT / "docs" / "manuscript" / "98_supplement_external_benchmarks.md"
    ).read_text(encoding="utf-8")

    assert "{{BENCH_CTX_TABLES}}" in supplement
    assert not re.search(r"^\| \d+ \| \{\{BENCH_CTX_TOP1_K", supplement, re.M)

    generated = generate_manuscript_variables(REPO_ROOT)["BENCH_CTX_TABLES"]
    printed = [int(match.group(1)) for match in re.finditer(r"^\| (\d+) \|", generated, re.M)]
    assert printed == grid * len(SERVED_PROBES), (
        f"the generated tables print rungs {printed} while the run swept {grid} "
        f"for each of {len(SERVED_PROBES)} probes"
    )


def _assert_phone_context_interpretation(claims: dict, reading: str) -> None:
    """Independently bind the hydrated cohort outcomes to this measured run."""
    floor = claims["bench_synthphone_uniform_baseline"]["value"]
    threshold = claims["bench_synthphone_context_family_alpha"]["value"]
    expected = []
    for probe in SERVED_PROBES:
        short = claims[f"bench_synthphone_short_top1_{probe.key}"]["value"]
        assert short <= floor, f"{probe.key}: context floor outcome changed"
        p_value = claims[f"bench_synthphone_context_paired_p_{probe.key}"]["value"]
        outcome = (
            "clears the corrected threshold"
            if p_value < threshold
            else "does not clear the corrected threshold"
        )
        expected.append(f"{probe.key} run: {outcome}")
    assert reading == "; ".join(expected) + ".", "context prose disagrees with recorded contrasts"


def test_phone_context_interpretation_matches_the_ledger(claims: dict):
    """The floor reading stays pinned; paired outcomes come from validated tokens."""
    supplement = " ".join(
        (REPO_ROOT / "docs/manuscript/98_supplement_external_benchmarks.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert (
        "at or below the uniform floor of one in "
        "{{BENCH_SYNTHPHONE_RESPONSE_INVENTORY_SIZE}}, the recorded finding that "
        "decision 108 pre-registered"
    ) in supplement
    assert "{{BENCH_SYNTHPHONE_CONTEXT_RESULT}}" in supplement
    assert "is itself not run-stable at these cell sizes" in supplement
    for retired in (
        "lowers the point estimates for Gemma and SmolLM but raises Qwen's",
        "lowers the point estimate for every registered probe",
        "{{BENCH_MODEL_SMOLLM_DISPLAY}} alone",
    ):
        assert retired not in supplement, f"retired direction sentence present: {retired!r}"
    reading = generate_manuscript_variables(REPO_ROOT)["BENCH_SYNTHPHONE_CONTEXT_RESULT"]
    _assert_phone_context_interpretation(claims, reading)


@pytest.mark.parametrize("probe", [probe.key for probe in SERVED_PROBES])
@pytest.mark.parametrize("changed_reading", ["floor", "threshold"])
def test_phone_context_interpretation_refuses_changed_results(claims, probe, changed_reading):
    """Changing a measured relation without rehydrating its reading must fail."""
    reading = generate_manuscript_variables(REPO_ROOT)["BENCH_SYNTHPHONE_CONTEXT_RESULT"]
    changed = deepcopy(claims)
    if changed_reading == "floor":
        changed[f"bench_synthphone_short_top1_{probe}"]["value"] = (
            claims["bench_synthphone_uniform_baseline"]["value"] * 2
        )
    else:
        alpha = changed["bench_synthphone_context_family_alpha"]["value"]
        label = f"bench_synthphone_context_paired_p_{probe}"
        changed[label]["value"] = alpha * 2 if claims[label]["value"] < alpha else alpha / 2
    with pytest.raises(AssertionError, match="context"):
        _assert_phone_context_interpretation(changed, reading)


def test_phone_context_interpretation_refuses_wrong_prose(claims, tmp_path):
    reading = generate_manuscript_variables(REPO_ROOT)["BENCH_SYNTHPHONE_CONTEXT_RESULT"]
    path = tmp_path / "context-reading.md"
    path.write_text(reading, encoding="utf-8")
    _assert_phone_context_interpretation(claims, path.read_text(encoding="utf-8"))
    for probe in SERVED_PROBES:
        prefix = f"{probe.key} run: "
        good = next(part for part in reading.rstrip(".").split("; ") if part.startswith(prefix))
        bad = prefix + (
            "clears the corrected threshold"
            if "does not clear" in good
            else "does not clear the corrected threshold"
        )
        path.write_text(reading.replace(good, bad), encoding="utf-8")
        with pytest.raises(AssertionError, match="context prose disagrees"):
            _assert_phone_context_interpretation(claims, path.read_text(encoding="utf-8"))


def test_no_null_is_written_as_an_absence():
    """A null bounds an effect; it does not prove one is zero.

    The manuscript said "the effect does not exist" of the Stage 5 result. A
    synthetic detectability control shows the machinery recovers a large planted
    signal; it cannot show that a smaller real effect is absent. The project's
    own doctrine is that nulls are reported as bounds, and this sentence was the
    one place that slipped.

    **The scan is multiline on purpose.** The manuscript is hard-wrapped, and
    that phrase sat across a line break as "and the effect does not\\nexist". A
    line-anchored grep cannot see it, which is why it survived three separate
    searches that reported it absent. Every prose check in this file that
    matters must join lines before matching.

    NEGATIVE CONTROL: the scan must find the hedged forms that are legitimate,
    or it would pass by being unable to see anything at all.
    """
    import re

    manuscript = REPO_ROOT / "docs" / "manuscript"
    joined = {
        path.name: " ".join(path.read_text(encoding="utf-8").split())
        for path in sorted(manuscript.glob("*.md"))
    }
    assert joined, "no manuscript files found"

    absence = re.compile(
        r"(?:the\s+)?effect\s+does\s+not\s+exist"
        r"|there\s+is\s+no\s+effect\b"
        r"|proves?\s+(?:the\s+)?(?:effect|null)\s+is\s+zero"
        r"|establishes?\s+(?:the\s+)?absence\s+of\s+(?:the\s+)?effect",
        re.I,
    )
    offenders = {
        name: absence.findall(text) for name, text in joined.items() if absence.search(text)
    }
    assert not offenders, (
        f"a null written as an absence rather than a bound: {offenders}. A "
        f"detectability control shows what the machinery can recover, not that a "
        f"smaller effect is missing."
    )

    # NEGATIVE CONTROL: the multiline scan can see across wraps at all.
    probe = "and the effect does not\nexist in this corpus"
    assert absence.search(" ".join(probe.split())), (
        "the scan cannot see a phrase wrapped across a line break, which is the "
        "exact failure it was written to fix"
    )


# ── Retained-generation retention ────────────────────────────────────────────


def _prune_fixture(tmp_path: Path, generations: int) -> tuple[Path, Path]:
    """A live publication plus publisher-shaped retained siblings beside it."""
    root = tmp_path / "output"
    root.mkdir()
    target = root / "docs"
    target.mkdir()
    (target / "README.md").write_text("live publication\n", encoding="utf-8")
    oldest = time.time() - 86400 * 30
    for index in range(generations):
        path = root / f".docs.generation.{secrets.token_hex(16)}"
        path.mkdir()
        (path / "README.md").write_text(f"retained generation {index}\n", encoding="utf-8")
        stamp = oldest + index * 3600
        os.utime(path, (stamp, stamp))
    return root, target


def test_prune_plan_marks_the_bound_excess_and_the_dry_run_changes_nothing(tmp_path):
    """The plan is the reviewable inventory; a dry run deletes nothing."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=5)

    plan = renderer.plan_retired_generation_cleanup(target, keep=2, min_age_seconds=0.0)
    prunes = [entry for entry in plan.entries if entry.action == "prune"]
    keeps = [entry for entry in plan.entries if entry.action == "keep" and entry.name != "docs"]
    publication = [entry for entry in plan.entries if entry.name == "docs"]
    refusals = [entry for entry in plan.entries if entry.action == "refuse"]
    assert (len(prunes), len(keeps), len(refusals)) == (3, 2, 0)
    assert len(publication) == 1 and publication[0].action == "keep", (
        "the inventory stopped reporting the protected publication"
    )
    assert "never a prune candidate" in publication[0].reason
    assert [entry.mtime for entry in prunes] == sorted(entry.mtime for entry in prunes), (
        "the inventory is not ordered oldest first"
    )
    assert {entry.reason for entry in prunes} == {"beyond the retention bound"}
    assert {entry.reason for entry in keeps} == {"within the retention bound"}

    dry = renderer.apply_retired_generation_cleanup(plan, dry_run=True)
    assert dry.deleted == () and dry.failed == ()
    assert len(list(root.iterdir())) == 6, "the dry run moved or removed an entry"
    assert (target / "README.md").read_text(encoding="utf-8") == "live publication\n"


def test_prune_apply_deletes_exactly_the_bound_excess_and_keeps_the_publication(tmp_path):
    """POSITIVE CONTROL: only publisher-shaped siblings beyond --keep disappear."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=5)
    live_before = (target / "README.md").read_bytes()
    live_identity = (target.stat().st_dev, target.stat().st_ino)

    plan = renderer.plan_retired_generation_cleanup(target, keep=2, min_age_seconds=0.0)
    result = renderer.apply_retired_generation_cleanup(plan, dry_run=False)

    assert len(result.deleted) == 3 and result.failed == ()
    assert len(list(root.iterdir())) == 3, "the live tree or a bound-kept sibling vanished"
    assert (target.stat().st_dev, target.stat().st_ino) == live_identity
    assert (target / "README.md").read_bytes() == live_before
    survivors = {
        (path / "README.md").read_text(encoding="utf-8")
        for path in root.iterdir()
        if path.name != "docs"
    }
    assert survivors == {"retained generation 3\n", "retained generation 4\n"}, (
        "the newest generations were not the ones kept"
    )


def test_prune_refuses_a_symlink_in_the_managed_namespace(tmp_path):
    """POSITIVE CONTROL: a link in the namespace is never followed or removed."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=2)
    outside = tmp_path / "outside-tree"
    outside.mkdir()
    sentinel = outside / "README.md"
    sentinel.write_text("outside sentinel\n", encoding="utf-8")
    link = root / f".docs.generation.{'a' * 32}"
    link.symlink_to(Path("../outside-tree"), target_is_directory=True)

    plan = renderer.plan_retired_generation_cleanup(target, keep=0, min_age_seconds=0.0)
    entries = {entry.name: entry for entry in plan.entries}
    assert entries[link.name].action == "refuse"
    assert "symlink" in entries[link.name].reason

    result = renderer.apply_retired_generation_cleanup(plan, dry_run=False)
    assert link.name not in {name for name, _ in result.failed}
    assert link.is_symlink() and link.readlink() == Path("../outside-tree")
    assert sentinel.read_text(encoding="utf-8") == "outside sentinel\n"
    assert len(result.deleted) == 2, "a refused entry blocked pruning real generations"


def test_prune_refuses_unmanaged_namespace_lookalikes(tmp_path):
    """A file or a non-token name wearing the prefix is reported, never touched."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=1)
    impostor_file = root / f".docs.retired.{'b' * 32}"
    impostor_file.write_text("not a generation\n", encoding="utf-8")
    loose_name = root / ".docs.generation.notes"
    loose_name.mkdir()

    plan = renderer.plan_retired_generation_cleanup(target, keep=0, min_age_seconds=0.0)
    entries = {entry.name: entry for entry in plan.entries}
    assert entries[impostor_file.name].action == "refuse"
    assert entries[loose_name.name].action == "refuse"
    assert "publisher generation" in entries[loose_name.name].reason

    renderer.apply_retired_generation_cleanup(plan, dry_run=False)
    assert impostor_file.read_text(encoding="utf-8") == "not a generation\n"
    assert loose_name.is_dir()


def test_prune_grace_period_spares_a_concurrent_render_staging_tree(tmp_path):
    """A fresh sibling is presumed in-flight staging and is kept at any bound."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=1)
    staging = root / f".docs.generation.{'c' * 32}"
    staging.mkdir()
    (staging / "README.md").write_text("staging\n", encoding="utf-8")

    protected = renderer.plan_retired_generation_cleanup(target, keep=0)
    entries = {entry.name: entry for entry in protected.entries}
    assert entries[staging.name].action == "keep"
    assert "grace" in entries[staging.name].reason

    exposed = renderer.plan_retired_generation_cleanup(target, keep=0, min_age_seconds=0.0)
    entries = {entry.name: entry for entry in exposed.entries}
    assert entries[staging.name].action == "prune"
    result = renderer.apply_retired_generation_cleanup(exposed, dry_run=False)
    assert staging.name in result.deleted and not staging.exists()


def test_prune_never_names_the_active_publication_or_the_namespace_root(tmp_path):
    """POSITIVE CONTROL: the live tree is never a candidate; a dotted target refuses."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=2)

    plan = renderer.plan_retired_generation_cleanup(target, keep=0, min_age_seconds=0.0)
    publication = [entry for entry in plan.entries if entry.name == "docs"]
    assert len(publication) == 1 and publication[0].action == "keep"
    assert "never a prune candidate" in publication[0].reason
    renderer.apply_retired_generation_cleanup(plan, dry_run=False)
    assert target.is_dir() and not target.is_symlink()
    assert (target / "README.md").read_text(encoding="utf-8") == "live publication\n"

    dotted = root / f".docs.generation.{'d' * 32}"
    dotted.mkdir()
    with pytest.raises(ValueError, match="namespace"):
        renderer.plan_retired_generation_cleanup(dotted, keep=0, min_age_seconds=0.0)
    assert dotted.is_dir()


def test_prune_reports_a_real_deletion_failure_and_keeps_the_rest(tmp_path):
    """An undeletable directory fails loudly without masking the run or the rest."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=3)
    stuck = root / f".docs.generation.{'e' * 32}"
    stuck.mkdir()
    (stuck / "README.md").write_text("stuck generation\n", encoding="utf-8")
    os.chmod(stuck, 0o555)
    try:
        plan = renderer.plan_retired_generation_cleanup(target, keep=0, min_age_seconds=0.0)
        result = renderer.apply_retired_generation_cleanup(plan, dry_run=False)
        assert len(result.deleted) == 3, "one failure masked the other deletions"
        assert [name for name, _ in result.failed] == [stuck.name]
        assert "deletion failed" in dict(result.failed)[stuck.name]
        assert (stuck / "README.md").read_text(encoding="utf-8") == "stuck generation\n"
    finally:
        os.chmod(stuck, 0o755)


def test_prune_cleans_history_when_no_publication_exists(tmp_path):
    """Retained generations are prunable after the publication tree is gone."""
    renderer = _renderer_module()
    root = tmp_path / "output"
    root.mkdir()
    orphan = root / f".docs.generation.{secrets.token_hex(16)}"
    orphan.mkdir()
    (orphan / "README.md").write_text("orphan\n", encoding="utf-8")
    os.utime(orphan, (time.time() - 86400, time.time() - 86400))

    plan = renderer.plan_retired_generation_cleanup(root / "docs", keep=0, min_age_seconds=0.0)
    result = renderer.apply_retired_generation_cleanup(plan, dry_run=False)
    assert result.deleted == (orphan.name,)
    assert not orphan.exists()


def test_prune_cli_routes_flags_and_rejects_orphans(tmp_path, capsys):
    """The dry run prints the protected publication; apply prunes; bad flags exit 2."""
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=4)

    status = renderer.main(
        [
            "--prune-retired",
            "--keep",
            "1",
            "--min-age",
            "0",
            "--dry-run",
            "--output-dir",
            str(target),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert "dry-run" in captured.out
    inventory_rows = [line.split() for line in captured.out.splitlines() if line.startswith("  ")]
    assert any(row[1] == "docs" for row in inventory_rows), "the inventory omitted the publication"
    assert len(list(root.iterdir())) == 5, "a dry run deleted a retained generation"

    status = renderer.main(
        ["--prune-retired", "--keep", "1", "--min-age", "0", "--output-dir", str(target)]
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    survivors = {path.name for path in root.iterdir()}
    assert len(survivors) == 2 and "docs" in survivors, (
        "the bound did not hold through the command line"
    )

    status = renderer.main(["--prune-retired", "--keep", "-1", "--output-dir", str(target)])
    captured = capsys.readouterr()
    assert status == 2
    assert "negative" in captured.err
    assert len(list(root.iterdir())) == 2, "a refused bound still pruned"

    with pytest.raises(SystemExit) as orphan:
        renderer.main(["--keep", "2"])
    assert orphan.value.code == 2
    with pytest.raises(SystemExit) as conflict:
        renderer.main(["--prune-retired", "--check"])
    assert conflict.value.code == 2


@pytest.mark.parametrize("age", ["nan", "inf", "-inf", "1e999"])
def test_prune_cli_refuses_nonfinite_age_without_touching_any_generation(tmp_path, capsys, age):
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=2)
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    status = renderer.main(
        ["--prune-retired", "--keep", "0", f"--min-age={age}", "--output-dir", str(target)]
    )
    assert status == 2
    assert "must be finite" in capsys.readouterr().err
    assert len(list(root.iterdir())) == 3
    assert before == {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("now", [float("nan"), float("inf"), -float("inf")])
def test_prune_plan_refuses_a_nonfinite_clock(tmp_path, now):
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=1)
    with pytest.raises(ValueError, match="clock must be finite"):
        renderer.plan_retired_generation_cleanup(target, keep=0, now=now)
    assert len(list(root.iterdir())) == 2


@pytest.mark.parametrize("change", ["vanished", "replaced", "refreshed"])
def test_prune_refuses_a_plan_whose_candidate_changed(tmp_path, change):
    renderer = _renderer_module()
    root, target = _prune_fixture(tmp_path, generations=1)
    plan = renderer.plan_retired_generation_cleanup(target, keep=0, min_age_seconds=0.0)
    candidate = next(entry for entry in plan.entries if entry.action == "prune")
    original = (candidate.path / "README.md").read_bytes()
    if change in {"vanished", "replaced"}:
        saved = root / "saved-generation"
        candidate.path.rename(saved)
        if change == "replaced":
            candidate.path.mkdir()
            (candidate.path / "README.md").write_text("replacement sentinel\n", encoding="utf-8")
    else:
        os.utime(candidate.path, ns=(candidate.mtime_ns + 1, candidate.mtime_ns + 1))
    result = renderer.apply_retired_generation_cleanup(plan)
    assert result.deleted == ()
    assert len(result.failed) == 1
    assert result.failed[0][0] == candidate.name
    assert ("vanished" if change == "vanished" else "changed") in result.failed[0][1]
    assert (target / "README.md").read_text(encoding="utf-8") == "live publication\n"
    if change in {"vanished", "replaced"}:
        assert (saved / "README.md").read_bytes() == original
    if change == "replaced":
        assert (candidate.path / "README.md").read_text(
            encoding="utf-8"
        ) == "replacement sentinel\n"
    if change == "refreshed":
        assert (candidate.path / "README.md").read_bytes() == original
